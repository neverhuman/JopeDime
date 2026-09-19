//! Genebank (a 20-row SQLite built here), a lineage-A snapshot (manifest + streamed rows) and
//! SOTArena results (two track versions) through the binary.

mod common;

use common::jl;
use jope_learning::canon::sha256_hex;
use jope_learning::sources::{genebank, snapshot, sotarena};
use jope_learning::state::{PointerKey, State};
use serde_json::json;
use std::fs;

fn build_genebank(path: &std::path::Path) {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(
        "CREATE TABLE best_scores(task TEXT, dataset_hash TEXT, score_tier TEXT, metric TEXT, \
         score REAL, semantic_sha256 TEXT, canonical_genome TEXT, achieved_at_ms INTEGER, \
         receipt_ref TEXT, PRIMARY KEY(task,dataset_hash,score_tier));",
    )
    .unwrap();
    let mut n = 0;
    for i in 0..10u32 {
        for (tier, metric, task) in [
            ("cv_robust", "oof_r2", "regression"),
            (
                "test_exact",
                if i % 2 == 0 { "r2" } else { "auc" },
                if i % 2 == 0 { "regression" } else { "binary" },
            ),
        ] {
            n += 1;
            let ds = format!("{:016x}", 0x0259_e0b6_af04_3f58u64 + u64::from(i));
            let sem = if n == 3 {
                String::new()
            } else {
                sha256_hex(format!("sem{n}").as_bytes())
            };
            conn.execute(
                "INSERT INTO best_scores VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                rusqlite::params![
                    task,
                    ds,
                    tier,
                    metric,
                    0.5 + f64::from(n) / 100.0,
                    sem,
                    format!("(pipe X (ridge alpha={n}))"),
                    1_784_970_730_000i64 + i64::from(n),
                    ""
                ],
            )
            .unwrap();
        }
    }
    assert_eq!(n, 20);
}

#[test]
fn genebank_rows_become_unbound_records() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("genebank.db");
    build_genebank(&db);
    let state_dir = tmp.path().join("state");
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
        ],
    )
    .ok()
    .json();
    assert_eq!(
        (rep["candidates"].as_u64(), rep["accepted"].as_u64()),
        (Some(20), Some(20))
    );
    assert_eq!(rep["scan"]["rows"], 20);
    assert_eq!(rep["pointers_written"], 20);
    let state = State::at(&state_dir);
    let pointers = state.pointers().unwrap();
    assert_eq!(pointers.len(), 20);
    for (key, r) in &pointers {
        assert_eq!(r.evaluator.sha256, "unbound");
        assert_eq!(r.evaluator.version, genebank::EVALUATOR_VERSION);
        assert!(!r.replayable && !r.benchmark_valid && !r.sealed);
        assert_eq!(r.improvement_number, 0);
        assert_eq!(r.identity.protocol, "genebank-gen1");
        assert!(r.ts_unix >= 1_784_970_730);
        assert!(
            key.cut_id == genebank::cut_id("cv_robust").unwrap()
                || key.cut_id == genebank::cut_id("test_exact").unwrap()
        );
        // The blob is the canonical row; its hash is the record hash.
        let blob = state
            .read_blob(&r.provenance.record_sha256)
            .unwrap()
            .unwrap();
        let row: serde_json::Value = serde_json::from_slice(&blob).unwrap();
        assert_eq!(row["canonical_genome"], r.genome.dsl);
    }
    assert_eq!(
        pointers
            .iter()
            .filter(|(_, r)| r.genome.semantic_sha256.is_none())
            .count(),
        1
    );
    // A dataset shows its cv_robust and test_exact marks; best first by direction.
    let shown = jl(&state_dir, &["hwm", "show", "0259e0b6af043f58"])
        .ok()
        .json();
    assert_eq!(shown.as_array().unwrap().len(), 2);
    assert!(
        shown[0]["record"]["score"]["value"].as_f64()
            >= shown[1]["record"]["score"]["value"].as_f64()
    );
    // Idempotent: every row is a tie the second time.
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
            "--show-rejects",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["accepted"], 0);
    assert_eq!(rep["rejected"], 20);
    assert!(rep["rejects"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["reason"] == "tie"));
    // An improved score for one (task, dataset, tier) moves exactly that pointer and chains it.
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE best_scores SET score = 0.99, achieved_at_ms = achieved_at_ms + 5000 WHERE dataset_hash = '0259e0b6af043f58' AND score_tier = 'test_exact'",
            [],
        )
        .unwrap();
    }
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
        ],
    )
    .ok()
    .json();
    assert_eq!(
        (
            rep["accepted"].as_u64(),
            rep["rejected"].as_u64(),
            rep["pointers_written"].as_u64()
        ),
        (Some(1), Some(19), Some(1))
    );
    let key = PointerKey {
        task: "regression".into(),
        dataset_hash: "0259e0b6af043f58".into(),
        cut_id: genebank::cut_id("test_exact").unwrap(),
    };
    let cur = state.read_pointer(&key).unwrap().unwrap();
    assert_eq!((cur.score.value, cur.improvement_number), (0.99, 1));
    assert!(cur.prev_event_sha256.is_some());
    let v = jl(&state_dir, &["hwm", "verify"]).ok().json();
    assert_eq!(v["breaks"].as_array().unwrap().len(), 0);
    assert_eq!(v["events"], 60);
    // Unbound records are excluded from a freeze unless asked for.
    let out = tmp.path().join("freeze");
    let f = jl(
        &state_dir,
        &["hwm", "freeze", "--out", out.to_str().unwrap()],
    )
    .ok()
    .json();
    assert_eq!(
        (f["included"].as_u64(), f["excluded_unbound"].as_u64()),
        (Some(0), Some(20))
    );
    let f = jl(
        &state_dir,
        &[
            "hwm",
            "freeze",
            "--out",
            out.to_str().unwrap(),
            "--include-unbound",
        ],
    )
    .ok()
    .json();
    assert_eq!(f["included"], 20);
}

fn corpus_row(
    dataset: &str,
    task: &str,
    metric: &str,
    score: f64,
    tier: &str,
    valid: bool,
    ts: u64,
) -> String {
    json!({
        "dataset_id": dataset, "source": "internal_test_exact_selfplay",
        "embedding": [0.1, 0.2, 0.3], "target_embedding": [0.0],
        "genome": format!("(pipe X (ridge alpha={score}))"), "canonical_genome": format!("(pipe X (ridge alpha={score}))"),
        "valid": valid, "metric": metric, "score": score, "cost_ms": 12.5, "engine": "featmath",
        "metadata": {"task_type": task, "dataset_hash": dataset, "split": "train", "score_tier": tier,
            "selection_bias": "test-exact-reused", "benchmark_valid": false, "generation": "strict-wave-000003",
            "semantic_plan_hash": sha256_hex(format!("{dataset}{score}").as_bytes()),
            "receipt_sha256": sha256_hex(b"r"), "label_created_at_ms": ts}
    })
    .to_string()
        + "\n"
}

#[test]
fn snapshot_maxima_are_bound_to_the_manifest_hashes() {
    let tmp = tempfile::tempdir().unwrap();
    let train = [
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.5,
            "test_exact",
            true,
            1_785_000_000_000,
        ),
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.7,
            "test_exact",
            true,
            1_785_000_001_000,
        ),
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.9,
            "test_exact",
            false,
            0,
        ), // invalid
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.95,
            "cv_robust",
            true,
            0,
        ), // wrong tier
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "auc",
            0.6,
            "test_exact",
            true,
            0,
        ),
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "logloss",
            0.1,
            "test_exact",
            true,
            0,
        ), // wrong metric
    ]
    .concat();
    let val = [
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.7,
            "test_exact",
            true,
            1_785_000_002_000,
        ), // tie keeps train
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "auc",
            0.8,
            "test_exact",
            true,
            1_785_000_003_000,
        ),
    ]
    .concat();
    let train_p = tmp.path().join("corpus.train.jsonl");
    let val_p = tmp.path().join("corpus.val.jsonl");
    fs::write(&train_p, &train).unwrap();
    fs::write(&val_p, &val).unwrap();
    let manifest = json!({
        "schema": "JopeTestOnlyTrainingCorpus.v1", "source_label_watermark": 4754860,
        "counts": {"regression/train": 4, "binary/train": 2, "regression/validation": 1, "binary/validation": 1},
        "strict_filter": {"binary": {"metric": "auc", "score_tier": "test_exact"}, "regression": {"metric": "r2", "score_tier": "test_exact"}, "selection_bias": "test-exact-reused"},
        "train_sha256": sha256_hex(train.as_bytes()), "validation_sha256": sha256_hex(val.as_bytes()),
    });
    let m_p = tmp.path().join("corpus-manifest.json");
    fs::write(&m_p, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let state_dir = tmp.path().join("state");
    let rows_t = format!("train={}", train_p.display());
    let rows_v = format!("validation={}", val_p.display());
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "snapshot",
            m_p.to_str().unwrap(),
            "--rows",
            &rows_t,
            "--rows",
            &rows_v,
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["scan"]["counts_match_manifest"], true);
    assert_eq!(rep["scan"]["splits"]["train"]["sha256_matches"], true);
    assert_eq!(rep["scan"]["splits"]["train"]["rows"], 6);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_invalid"], 1);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_tier"], 1);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_metric"], 1);
    assert_eq!(rep["scan"]["maxima"], 2);
    assert_eq!(rep["accepted"], 2);
    let state = State::at(&state_dir);
    let cut = snapshot::cut_id("test-exact-reused").unwrap();
    let a = state
        .read_pointer(&PointerKey {
            task: "regression".into(),
            dataset_hash: "aaaaaaaaaaaaaaaa".into(),
            cut_id: cut.clone(),
        })
        .unwrap()
        .unwrap();
    assert_eq!(a.score.value, 0.7);
    assert_eq!(a.provenance.run_id, "strict-wave-000003/train");
    assert_eq!(a.provenance.source, "snapshot-4754860");
    assert_eq!(a.evaluator.sha256, "unbound");
    assert_eq!(a.ts_unix, 1_785_000_001);
    let b = state
        .read_pointer(&PointerKey {
            task: "binary".into(),
            dataset_hash: "bbbbbbbbbbbbbbbb".into(),
            cut_id: cut,
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        (b.score.value, b.provenance.run_id.as_str()),
        (0.8, "strict-wave-000003/validation")
    );
    // A stream that does not hash to the manifest is refused outright.
    fs::write(&val_p, val.replace("0.8", "0.81")).unwrap();
    let r = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "snapshot",
            m_p.to_str().unwrap(),
            "--rows",
            &rows_v,
        ],
    );
    assert_eq!(r.status, 2);
    assert!(r.stderr.contains("manifest binds"), "{}", r.stderr);
    assert_eq!(jl(&state_dir, &["hwm", "verify"]).ok().json()["events"], 2);
}

const HEADER: &str = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n";

fn score_line(track: &str, ds: &str, method: &str, value: f64) -> String {
    format!(
        "{track},{ds},{method},public-development,rmse,{value},ok,True,reproducible-run-report,2026-06-30T00:40:20+00:00,{},results/external-references/golden42/{ds}/{method}.json,Exact Golden42 report; method identity is the report filename.\n",
        sha256_hex(format!("{track}{ds}{method}").as_bytes())
    )
}

#[test]
fn sotarena_rows_append_verify_and_query() {
    let tmp = tempfile::tempdir().unwrap();
    let results = tmp.path().join("results");
    let v1 = HEADER.to_string()
        + &score_line(
            "tabular-regression-v1",
            "019de5d76a718163",
            "ada_boost",
            0.0733,
        )
        + &score_line(
            "tabular-regression-v1",
            "019de5d76a718163",
            "jope-dime",
            0.0611,
        )
        + &score_line("tabular-regression-v1", "0259e0b6af043f58", "ridge", 0.5);
    let v2 = HEADER.to_string()
        + &score_line(
            "tabular-regression-v2",
            "019de5d76a718163",
            "ada_boost",
            0.0733,
        )
        + &score_line(
            "tabular-regression-v2",
            "019de5d76a718163",
            "jope-dime",
            0.0598,
        )
        + &score_line("tabular-regression-v2", "ZTS_small", "ada_boost", 0.2142);
    fs::create_dir_all(results.join("tabular-regression-v1")).unwrap();
    fs::create_dir_all(results.join("tabular-regression-v2")).unwrap();
    fs::create_dir_all(results.join("external-references")).unwrap();
    fs::write(results.join("tabular-regression-v1/scores.csv"), &v1).unwrap();
    fs::write(results.join("tabular-regression-v2/scores.csv"), &v2).unwrap();
    assert_eq!(sotarena::track_versions(&results).unwrap().len(), 2);
    let state_dir = tmp.path().join("state");
    let rep = jl(
        &state_dir,
        &[
            "scores",
            "append",
            "--source",
            "sotarena",
            results.to_str().unwrap(),
            "--git-sha",
            "e5db30a07c093661c8f6940d3e22192511f8388b",
        ],
    )
    .ok()
    .json();
    let rep = rep.as_array().unwrap();
    assert_eq!(rep.len(), 2);
    assert_eq!(rep[0]["rows"], 3);
    assert_eq!(rep[1]["rows"], 3);
    assert_eq!(
        rep[1]["batch"]["prev_batch_sha256"],
        rep[0]["batch"]["sha256"]
    );
    // The published csv is kept byte-identical with its hash.
    let pub_v2 =
        state_dir.join("scores/sotarena/tabular-regression-v2/tabular-regression-v2/scores.csv");
    assert_eq!(fs::read_to_string(&pub_v2).unwrap(), v2);
    assert!(fs::read_to_string(pub_v2.with_extension("csv.sha256"))
        .unwrap()
        .starts_with(&sha256_hex(v2.as_bytes())));
    let v = jl(&state_dir, &["scores", "verify-ledger"]).ok().json();
    assert_eq!(
        (v["rows"].as_u64(), v["ledger"]["batches"].as_u64()),
        (Some(6), Some(2))
    );
    // Re-append: every row is a duplicate, no new batch.
    let rep = jl(
        &state_dir,
        &[
            "scores",
            "append",
            "--source",
            "sotarena",
            results.to_str().unwrap(),
            "--git-sha",
            "e5db30a07c093661c8f6940d3e22192511f8388b",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep[0]["rows"], 0);
    assert_eq!(rep[0]["duplicates_skipped"], 3);
    assert!(rep[0]["batch"].is_null());
    let best = jl(&state_dir, &["scores", "query", "--best-per-dataset"])
        .ok()
        .json();
    let best = best.as_array().unwrap();
    assert_eq!(best.len(), 3);
    let top = best
        .iter()
        .find(|b| b["dataset_hash"] == "019de5d76a718163")
        .unwrap();
    assert_eq!(
        (
            top["method_id"].as_str(),
            top["value"].as_f64(),
            top["version"].as_str()
        ),
        (
            Some("jope-dime"),
            Some(0.0598),
            Some("tabular-regression-v2")
        )
    );
    assert_eq!(
        (top["rows"].as_u64(), top["methods"].as_u64()),
        (Some(4), Some(2))
    );
    let best = jl(
        &state_dir,
        &[
            "scores",
            "query",
            "--best-per-dataset",
            "--version",
            "tabular-regression-v1",
        ],
    )
    .ok()
    .json();
    assert_eq!(
        best.as_array()
            .unwrap()
            .iter()
            .find(|b| b["dataset_hash"] == "019de5d76a718163")
            .unwrap()["value"],
        0.0611
    );
    // Tampering with a value breaks the row hash and the batch seal.
    let ledger = jope_learning::ledger::Ledger::open(&state_dir, "scores");
    let (_, first) = ledger.batches().unwrap().into_iter().next().unwrap();
    fs::write(
        &first,
        fs::read_to_string(&first)
            .unwrap()
            .replace("0.0611", "0.0001"),
    )
    .unwrap();
    let r = jl(&state_dir, &["scores", "verify-ledger"]);
    assert_eq!(r.status, 1);
    let v = r.json();
    assert!(!v["invalid_rows"].as_array().unwrap().is_empty());
    assert!(!v["ledger"]["breaks"].as_array().unwrap().is_empty());
}
