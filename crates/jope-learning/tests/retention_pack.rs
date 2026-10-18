//! S3-RET job (b): write-time tags, `jope.s3-pack/v1` packs and the `gp_distilled/v1` table.
//! Fixtures only; nothing here talks to S3.

use jope_learning::canon::sha256_hex;
use jope_learning::retention::distilled::{self, DistilledRow, Outcome, SplitRole};
use jope_learning::retention::pack::{read_entry, verify, PackWriter, SealedPack, MIN_PACK_BYTES};
use jope_learning::retention::parquet::{self, Cell, ColumnSpec, ColumnType, Table};
use jope_learning::retention::tags::{ObjectKind, Supersession, Tier, WriteTags};
use std::collections::BTreeSet;

fn tags() -> WriteTags {
    WriteTags::new(
        Tier::T2,
        "run-20261013-a",
        "gp-teacher-v6",
        false,
        None,
        ObjectKind::Pack,
    )
    .unwrap()
}

fn record(i: usize) -> Vec<u8> {
    // ~10 KB JSON-ish bodies with repetition, like GP attempt receipts.
    let line = format!("{{\"attempt\":{i},\"program\":\"(pipe X (standardize) (ridge alpha={}))\",\"score\":0.{i:04}}}\n", i % 17);
    line.repeat(10_000 / line.len() + 1).into_bytes()
}

#[test]
fn write_tags_enforce_retention_invariants() {
    let t = tags();
    let pairs: Vec<(&str, &str)> = t
        .pairs()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("tier", "T2"),
            ("run_id", "run-20261013-a"),
            ("family", "gp-teacher-v6"),
            ("qualified", "false"),
            ("superseded_by", "none"),
            ("kind", "pack")
        ]
    );
    assert_eq!(
        t.header_value(),
        "tier=T2&run_id=run-20261013-a&family=gp-teacher-v6&qualified=false&superseded_by=none&kind=pack"
    );
    // qualified results, splits and evidence are T0.
    assert!(WriteTags::new(Tier::T2, "r", "f", true, None, ObjectKind::Result).is_err());
    assert!(WriteTags::new(Tier::T1, "r", "f", false, None, ObjectKind::Split).is_err());
    assert!(WriteTags::new(Tier::T3, "r", "f", false, None, ObjectKind::Evidence).is_err());
    assert!(WriteTags::new(Tier::T0, "r", "f", true, None, ObjectKind::Result).is_ok());
    // Only a qualified champion supersedes.
    let unq = Supersession {
        champion_run_id: "champ".into(),
        champion_qualified: false,
    };
    let q = Supersession {
        champion_run_id: "champ".into(),
        champion_qualified: true,
    };
    assert!(WriteTags::new(Tier::T3, "r", "f", false, Some(&unq), ObjectKind::Result).is_err());
    let ok = WriteTags::new(Tier::T3, "r", "f", false, Some(&q), ObjectKind::Result).unwrap();
    assert!(ok
        .pairs()
        .contains(&("superseded_by".into(), "champ".into())));
    // S3 tag charset.
    assert!(WriteTags::new(Tier::T2, "run#1", "f", false, None, ObjectKind::Result).is_err());
}

fn write_packs(n: usize, target: u64) -> (Vec<SealedPack>, Vec<Vec<u8>>) {
    let mut w = PackWriter::fixture("gp-teacher-v6", "run-20261013-a", tags(), target).unwrap();
    let mut packs = Vec::new();
    let mut bodies = Vec::new();
    for i in 0..n {
        let body = record(i);
        let key = format!(
            "experiments/gp-teacher-v6/attempts/ctx{:02}/{i:05}.json",
            i % 7
        );
        if let Some(p) = w.add(&key, Some(&format!("v{i}")), &body).unwrap() {
            packs.push(p);
        }
        bodies.push(body);
    }
    if let Some(p) = w.finish() {
        packs.push(p);
    }
    (packs, bodies)
}

#[test]
fn packs_round_trip_and_verify() {
    let (packs, bodies) = write_packs(300, 1_200_000);
    assert!(
        packs.len() >= 2,
        "300 x 10 KB at a 1.2 MB target seals several packs"
    );
    let mut seen = 0usize;
    for (seq, p) in packs.iter().enumerate() {
        let v = verify(&p.bytes, &p.index).unwrap();
        assert_eq!(v.records, p.index.entries.len() as u64);
        assert!(p.index.fixture);
        assert_eq!(p.index.sequence as usize, seq);
        assert_eq!(p.index.pack_sha256, sha256_hex(&p.bytes));
        assert!(p
            .index
            .pack_key()
            .ends_with(&format!("{seq:05}-{}.zst", p.index.pack_sha256)));
        assert!(
            p.index.pack_bytes < p.index.decoded_bytes / 4,
            "repetitive bodies compress"
        );
        assert!(p.index.tags.contains(&("kind".into(), "pack".into())));
        for (i, e) in p.index.entries.iter().enumerate() {
            let body = read_entry(&p.bytes, &p.index, i).unwrap();
            assert_eq!(body, bodies[seen], "entry {} round-trips", e.key);
            assert_eq!(e.version_id.as_deref(), Some(format!("v{seen}").as_str()));
            seen += 1;
        }
        // Decoding every frame in order yields the records back to back (`zstd -d` view).
        let mut all = Vec::new();
        for c in &p.index.chunks {
            let frame = &p.bytes[c.offset as usize..(c.offset + c.length) as usize];
            all.extend(jope_records::zstd_frame::decompress(frame, c.decoded as usize).unwrap());
        }
        let expected: Vec<u8> = p
            .index
            .entries
            .iter()
            .flat_map(|e| {
                bodies[e.key[e.key.len() - 10..e.key.len() - 5]
                    .parse::<usize>()
                    .unwrap()]
                .clone()
            })
            .collect();
        assert_eq!(all, expected);
    }
    assert_eq!(seen, 300);
    // Optional external cross-check: dump the first pack and its index for `zstd -d`.
    if let Some(dir) = std::env::var_os("JOPE_RETENTION_PACK_SAMPLE") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::write(dir.join("pack-sample.zst"), &packs[0].bytes).unwrap();
        std::fs::write(
            dir.join("pack-sample.index.json"),
            serde_json::to_vec_pretty(&packs[0].index).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn pack_tampering_and_misuse_are_refused() {
    let (packs, _) = write_packs(40, 10_000_000);
    let p = &packs[0];
    let mut bad = p.bytes.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xff;
    assert!(verify(&bad, &p.index).is_err(), "flipped pack byte");
    let mut idx = p.index.clone();
    idx.entries[3].sha256 = "0".repeat(64);
    // The pack sha still matches, so only the entry check can catch it.
    assert!(verify(&p.bytes, &idx)
        .unwrap_err()
        .to_string()
        .contains("sha256 mismatch"));
    let mut idx = p.index.clone();
    idx.entries[1].key = idx.entries[0].key.clone();
    assert!(verify(&p.bytes, &idx)
        .unwrap_err()
        .to_string()
        .contains("duplicate key"));
    let mut idx = p.index.clone();
    idx.decoded_bytes += 1;
    assert!(verify(&p.bytes, &idx).is_err());
    // Production writers refuse small targets; keys are unique per run.
    assert!(PackWriter::new("f", "r", tags(), MIN_PACK_BYTES - 1).is_err());
    assert!(PackWriter::new("f", "r", tags(), MIN_PACK_BYTES).is_ok());
    assert!(PackWriter::fixture("bad/family", "r", tags(), 10).is_err());
    let mut w = PackWriter::fixture("f", "r", tags(), 1 << 20).unwrap();
    w.add("k", None, b"a").unwrap();
    assert!(w.add("k", None, b"b").is_err());
}

fn known() -> BTreeSet<String> {
    [
        "cv_geomean_rmse_ratio",
        "best_rust_rmse",
        "validation_rmse",
        "l6",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn row(prog: char, ctx: char, score: Option<f64>, outcome: Outcome) -> DistilledRow {
    DistilledRow {
        program_sha: prog.to_string().repeat(64),
        program: Some("(pipe X (standardize) (ridge alpha=1))".into()),
        dataset: "000c7ae796cb2672".into(),
        context_sha256: Some(ctx.to_string().repeat(64)),
        fold: Some(0),
        split_role: SplitRole::Train,
        outcome,
        score,
        score_name: score.map(|_| "cv_geomean_rmse_ratio".into()),
        rank_ctx: score.map(|_| 1),
        champion_id: None,
        superseded_by: None,
        seed_packet: None,
        policylm_ckpt: None,
        campaign: Some("regression-research-2k-per-fold-hwm-v1-20261001".into()),
        run_id: "run-20261013-a".into(),
        family: "gp-teacher-v6".into(),
        qualified: false,
        l6_delta: None,
        source_bucket: "veox-jopedime-learning-use2".into(),
        source_key: format!("experiments/gp-teacher-v6/attempts/{prog}{ctx}.json"),
        source_version_id: Some("v1".into()),
        source_sha256: "e".repeat(64),
        pack_key: Some("packs/v1/gp-teacher-v6/run-20261013-a/00000-x.zst".into()),
        pack_entry: Some(0),
    }
}

#[test]
fn distilled_table_round_trips_and_reconciles() {
    let mut l6 = row('c', 'b', Some(0.8961), Outcome::Ok);
    l6.score_name = Some("l6".into());
    l6.l6_delta = Some(0.8961 - 0.897327);
    l6.qualified = true;
    l6.split_role = SplitRole::Val263;
    let mut failed = row('d', 'a', None, Outcome::Timeout);
    failed.program = None;
    failed.rank_ctx = None;
    let rows = vec![
        row('b', 'a', Some(0.61), Outcome::Ok),
        row('a', 'a', Some(0.59), Outcome::Ok),
        l6,
        failed,
    ];
    let bytes = distilled::write(
        &rows,
        &known(),
        &[("source_manifest".into(), "fixture".into())],
    )
    .unwrap();
    let back = distilled::read(&bytes, &known()).unwrap();
    let mut expected = rows.clone();
    expected.sort_by(|x, y| x.program_sha.cmp(&y.program_sha));
    assert_eq!(back, expected, "rows come back sorted by identity");
    assert_eq!(
        distilled::write(
            &rows,
            &known(),
            &[("source_manifest".into(), "fixture".into())]
        )
        .unwrap(),
        bytes,
        "deterministic bytes"
    );
    let keys: BTreeSet<String> = rows.iter().map(|r| r.source_key.clone()).collect();
    distilled::reconcile(&back, &keys).unwrap();
    let mut more = keys.clone();
    more.insert("experiments/gp-teacher-v6/attempts/zz.json".into());
    assert!(distilled::reconcile(&back, &more)
        .unwrap_err()
        .to_string()
        .contains("missing"));
}

#[test]
fn distilled_rows_that_would_corrupt_meaning_are_refused() {
    let k = known();
    let mut r = row('a', 'a', Some(0.5), Outcome::Ok);
    r.score_name = Some("l7_mystery".into());
    assert!(
        distilled::validate(&r, &k).is_err(),
        "unknown score identity"
    );
    let mut r = row('a', 'a', Some(0.5), Outcome::Ok);
    r.qualified = true;
    assert!(
        distilled::validate(&r, &k).is_err(),
        "qualified without a paired L6 eval"
    );
    let r = row('a', 'a', Some(0.5), Outcome::Failed);
    assert!(
        distilled::validate(&r, &k).is_err(),
        "scored row with a failure outcome"
    );
    let mut r = row('a', 'a', Some(f64::NAN), Outcome::Ok);
    r.score = Some(f64::NAN);
    assert!(distilled::validate(&r, &k).is_err(), "non-finite score");
    let mut r = row('a', 'a', None, Outcome::Failed);
    r.superseded_by = Some("champ".into());
    assert!(
        distilled::validate(&r, &k).is_err(),
        "superseded_by without champion_id"
    );
    let dup = vec![
        row('a', 'a', Some(0.5), Outcome::Ok),
        row('a', 'a', Some(0.5), Outcome::Ok),
    ];
    assert!(distilled::write(&dup, &k, &[])
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
}

#[test]
fn parquet_subset_round_trips_every_type_and_null() {
    let columns = vec![
        ColumnSpec {
            name: "s".into(),
            ty: ColumnType::Utf8,
            required: true,
        },
        ColumnSpec {
            name: "os".into(),
            ty: ColumnType::Utf8,
            required: false,
        },
        ColumnSpec {
            name: "i".into(),
            ty: ColumnType::Int64,
            required: false,
        },
        ColumnSpec {
            name: "d".into(),
            ty: ColumnType::Double,
            required: false,
        },
        ColumnSpec {
            name: "b".into(),
            ty: ColumnType::Boolean,
            required: true,
        },
    ];
    let rows: Vec<Vec<Cell>> = (0..37)
        .map(|n| {
            vec![
                Cell::Str(format!("row-{n}")),
                if n % 3 == 0 {
                    Cell::Null
                } else {
                    Cell::Str("x".repeat(n))
                },
                if n % 4 == 0 {
                    Cell::Null
                } else {
                    Cell::I64(n as i64 * -7)
                },
                if n % 5 == 0 {
                    Cell::Null
                } else {
                    Cell::F64(n as f64 / 3.0)
                },
                Cell::Bool(n % 2 == 0),
            ]
        })
        .collect();
    let table = Table {
        columns,
        rows,
        metadata: vec![("k".into(), "v".into())],
    };
    for zstd in [false, true] {
        let bytes = parquet::write(&table, zstd, "test").unwrap();
        assert_eq!(parquet::read(&bytes).unwrap(), table);
    }
    let empty = Table {
        rows: vec![],
        ..table.clone()
    };
    assert_eq!(
        parquet::read(&parquet::write(&empty, true, "test").unwrap()).unwrap(),
        empty
    );
    let mut bad = parquet::write(&table, true, "test").unwrap();
    bad[0] = b'X';
    assert!(parquet::read(&bad).is_err());
    let mut null_in_required = table.clone();
    null_in_required.rows[0][0] = Cell::Null;
    assert!(parquet::write(&null_in_required, true, "test").is_err());
    // Optional external cross-check: write a sample for an independent reader (pyarrow).
    if let Some(out) = std::env::var_os("JOPE_RETENTION_PARQUET_SAMPLE") {
        std::fs::write(&out, parquet::write(&table, true, "test").unwrap()).unwrap();
    }
}

#[test]
fn selfplay_observations_distill_to_parquet_and_bridge_into_frozen_pools() {
    use jope_learning::pools::bootstrap::Verdict;
    use jope_learning::pools::evaluate::{evaluate, Config, Inputs};
    use jope_learning::pools::pool::{freeze, FreezeRule, Reference};
    use jope_learning::pools::selectors::Features;
    use jope_learning::retention::distilled::{
        distill_selfplay_observations, distilled_rows_to_measurement_rows, score_direction,
        ScoreDirection, SelfPlayDistillObservation, SelfPlayDistillSpec,
    };

    let scores = known();
    assert_eq!(
        score_direction("best_rust_rmse").unwrap(),
        ScoreDirection::LowerIsBetter
    );
    assert!(score_direction("unregistered_metric").is_err());

    let mut all_distilled = Vec::new();
    let mut cf_rows = std::collections::BTreeMap::new();
    let mut pf_rows = std::collections::BTreeMap::new();
    for c in 0..32usize {
        let parent = format!("dataset-parent-{}", c / 2);
        let mut ctx_sha = sha256_hex(format!("selfplay-ctx-{c}").as_bytes());
        let z = if (c / 2) % 2 == 0 { 1.0 } else { -1.0 };
        ctx_sha.replace_range(..2, if z > 0.0 { "ff" } else { "00" });
        cf_rows.insert(
            ctx_sha.clone(),
            Features::context_id_vector(&ctx_sha, 1).unwrap(),
        );
        let spec = SelfPlayDistillSpec {
            dataset: parent,
            context_sha256: ctx_sha.clone(),
            fold: Some((c % 2) as i64),
            split_role: SplitRole::Train,
            score_name: "best_rust_rmse".into(),
            policylm_ckpt: None,
            campaign: Some("selfplay-flywheel-phase3".into()),
            run_id: format!("run-sp-{c}"),
            family: "clover-selfplay".into(),
            source_bucket: "veox-jopedime-learning-use2".into(),
        };
        // Disjoint program cohort per parent so cross-parent ID memorization (ProgramPrior / RetrievalKnn)
        // has zero overlap across folds and context-blind RidgeProgram averages out across z=+1/-1,
        // while ContextualSurrogateQHat generalizes via z * phi(program) on negated best_rust_rmse.
        let cohort = c / 2;
        let mut obs = Vec::new();
        let mut best_expected_sha = String::new();
        let mut best_expected_rmse = f64::INFINITY;
        for p in 0..11usize {
            let expr = if p % 2 == 0 {
                format!(
                    "(pipe X (standardize) (ridge alpha={}))",
                    cohort * 100 + p + 1
                )
            } else {
                format!(
                    "(pipe X ( winsorize ) (lasso alpha={}))",
                    cohort * 100 + p + 1
                )
            };
            let prog_sha = sha256_hex(expr.as_bytes());
            let failed = p == 10;
            let phi = if p % 2 == 0 { 1.0 } else { -1.0 };
            pf_rows.insert(prog_sha.clone(), vec![phi]);
            // Lower RMSE is better: when z * phi > 0, RMSE is ~0.20 (best); when z * phi < 0, RMSE is ~0.80 (worst).
            let rmse = (!failed).then(|| {
                let val = 0.50 - 0.30 * (z * phi) + 0.001 * (p as f64);
                if val < best_expected_rmse {
                    best_expected_rmse = val;
                    best_expected_sha = prog_sha.clone();
                }
                val
            });
            obs.push(SelfPlayDistillObservation {
                program_sha: prog_sha,
                program: Some(expr),
                outcome: if failed { Outcome::Failed } else { Outcome::Ok },
                score: rmse,
                seed_packet: Some(if p < 2 {
                    "selfplay_archive_r1".into()
                } else {
                    "ordinary_gp".into()
                }),
                source_key: format!("experiments/clover-selfplay/ctx-{c}/eval-{p}.json"),
                source_version_id: Some("v1".into()),
                source_sha256: sha256_hex(format!("raw-{c}-{p}").as_bytes()),
            });
        }
        let batch = distill_selfplay_observations(&spec, &obs, &scores).unwrap();
        // Verify rank_ctx = 1..=10 on the 10 completed evaluations and None on the failed one.
        let ranks: BTreeSet<i64> = batch.iter().filter_map(|r| r.rank_ctx).collect();
        assert_eq!(ranks, (1..=10).collect());
        // Verify rank 1 and champion_id belong to the LOWEST-RMSE program (not highest RMSE).
        let rank1 = batch.iter().find(|r| r.rank_ctx == Some(1)).unwrap();
        assert_eq!(rank1.program_sha, best_expected_sha);
        assert!((rank1.score.unwrap() - best_expected_rmse).abs() < 1e-12);
        assert!(batch.iter().filter(|r| r.outcome == Outcome::Ok).all(|r| {
            r.champion_id.as_deref() == Some(best_expected_sha.as_str())
                && !r.qualified
                && r.superseded_by.is_none()
        }));
        all_distilled.extend(batch);
    }

    // Round-trip through gp_distilled/v1 Parquet.
    let parquet_bytes = distilled::write(&all_distilled, &scores, &[]).unwrap();
    let reloaded = distilled::read(&parquet_bytes, &scores).unwrap();
    assert_eq!(reloaded.len(), all_distilled.len());

    // Convert into MeasurementRows and verify lower-is-better RMSE is negated into higher-is-better q.
    let protocol_sha = sha256_hex(b"selfplay-inner-cut-protocol-v1");
    let m_rows =
        distilled_rows_to_measurement_rows(&reloaded, "best_rust_rmse", &protocol_sha, &scores)
            .unwrap();
    for (d_row, m_row) in reloaded.iter().zip(&m_rows) {
        if let Some(rmse) = d_row.score {
            assert_eq!(m_row.q, Some(-rmse));
        } else {
            assert_eq!(m_row.q, None);
        }
    }

    let freeze_rule = FreezeRule {
        min_pool: 10,
        max_pool: None,
        salt: "distilled-bridge".into(),
        min_distinct_measured: 2,
        source_grade: "gp_distilled_v1".into(),
    };
    let (pools, manifest) = freeze(&m_rows, &freeze_rule, &sha256_hex(&parquet_bytes)).unwrap();
    assert_eq!(pools.len(), 32);

    let cf = Features {
        dim: 1,
        rows: cf_rows,
    };
    let pf = Features {
        dim: 1,
        rows: pf_rows,
    };
    let eval_cfg = Config {
        reference: Reference::PoolFloor,
        budget: 2,
        folds: 5,
        fold_salt: "distilled-folds".into(),
        prior_weight: 2.0,
        knn_k: 1,
        ridge_lambda: 1.0,
        bootstrap_draws: 500,
        bootstrap_seed: 11,
        alpha: 0.05,
    };
    let inputs = Inputs {
        context_features: Some(&cf),
        program_features: Some(&pf),
        external: vec![],
        unavailable: vec![],
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &eval_cfg).unwrap();
    assert_eq!(receipt.ranking_learnability, "ESTABLISHED_SHADOW");
    let cmp = receipt
        .comparisons
        .iter()
        .find(|c| {
            c.candidate == "contextual_surrogate_qhat" && c.baseline == receipt.strongest_baseline
        })
        .unwrap();
    assert_eq!(cmp.realized_gain.verdict, Verdict::Greater);
    assert_eq!(
        cmp.pairwise_accuracy.as_ref().unwrap().verdict,
        Verdict::Greater
    );

    // Verify l6 and mixed-fold rows are refused by distilled_rows_to_measurement_rows.
    assert!(
        distilled_rows_to_measurement_rows(&reloaded, "l6", &protocol_sha, &scores).is_err(),
        "l6 must be refused for shadow training pool bridging"
    );
    let mut mixed_fold = reloaded.clone();
    mixed_fold[0].fold = Some(99);
    assert!(distilled_rows_to_measurement_rows(
        &mixed_fold,
        "best_rust_rmse",
        &protocol_sha,
        &scores
    )
    .is_err());

    // Verify held-out Val263 rows are preserved as "val" and refused by freeze.
    let mut leaked = reloaded.clone();
    leaked[0].split_role = SplitRole::Val263;
    let leaked_m =
        distilled_rows_to_measurement_rows(&leaked, "best_rust_rmse", &protocol_sha, &scores)
            .unwrap();
    assert!(freeze(&leaked_m, &freeze_rule, &sha256_hex(&parquet_bytes)).is_err());

    // Verify distill_gp_results_index distills S3 context-hwm records and reconciles every S3 key.
    let fake_index = serde_json::json!({
        "schema": "jopedime.gp-results-index/v1",
        "runs": [
            {
                "bucket": "veox-jopedime-learning-use2",
                "key": "experiments/gp-policy-dl/v2-trainonly/gp-hwm/ds1/f0/ctx1/obj1.json",
                "version_id": "v1",
                "object_sha256": "11".repeat(32),
                "program_sha": "22".repeat(32),
                "program": "(pipe X (standardize) (ridge alpha=1))",
                "dataset": "ds1",
                "context_sha256": "33".repeat(32),
                "fold": 0,
                "score": 0.58,
                "score_name": "cv_geomean_rmse_ratio",
                "qualified": false,
                "grain": "context-hwm",
                "eval": "jope.s1-two-fold/v1",
                "family": "gp-policy-dl"
            },
            {
                "bucket": "veox-jopedime-learning-use2",
                "key": "experiments/gp-evolution/v1/discoveries/CLOSE.json",
                "object_sha256": "44".repeat(32),
                "program_sha": null,
                "dataset": null,
                "score": null,
                "score_name": null,
                "qualified": false,
                "grain": "campaign-summary",
                "family": "gp-evolution"
            },
            {
                "bucket": "local-disk",
                "key": "local://xbabe3/row.json",
                "object_sha256": null,
                "program_sha": null,
                "dataset": "ds1",
                "context_sha256": null,
                "score": 0.42,
                "score_name": "validation_rmse",
                "qualified": false,
                "grain": "local-summary",
                "family": "xbabe3-scratch"
            }
        ]
    });
    let (idx_rows, idx_keys, stats) =
        distilled::distill_gp_results_index_with_stats(&fake_index, SplitRole::Train, &scores)
            .unwrap();
    assert_eq!(stats.total_index_runs, 3);
    assert_eq!(stats.distilled_rows, 1);
    assert_eq!(stats.skipped_unscored_runs, 1);
    assert_eq!(stats.skipped_incomplete_provenance_runs, 1);
    assert_eq!(stats.reconciled_s3_keys, 1);
    assert_eq!(idx_rows.len(), 1);
    assert_eq!(idx_keys.len(), 1);
    let idx_parquet = distilled::write(&idx_rows, &scores, &[]).unwrap();
    let idx_readback = distilled::read(&idx_parquet, &scores).unwrap();
    distilled::reconcile(&idx_readback, &idx_keys).unwrap();
}
