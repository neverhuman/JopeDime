use super::*;

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
