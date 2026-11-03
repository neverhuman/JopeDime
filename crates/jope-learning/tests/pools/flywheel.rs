use super::*;

pub(crate) fn flywheel_isolates_outer_folds_enforces_kl_freeze_and_ingests_s3_prior_body() {
    use jope_learning::clover::expansion::MeasuredTiltPolicyConfig;
    use jope_learning::pools::evaluate::fold_of;
    use jope_learning::pools::flywheel::{
        build_fold_s3_priors, known_scores, run_distilled_flywheel_with_s3_prior, FlywheelConfig,
    };
    use jope_learning::retention::distilled::{
        self, distill_selfplay_observations, DistilledRow, Outcome, SelfPlayDistillObservation,
        SelfPlayDistillSpec, SplitRole,
    };

    let mut rows = Vec::new();
    let mut cf_rows = std::collections::BTreeMap::new();
    let mut pf_rows = std::collections::BTreeMap::new();
    for c in 0..20usize {
        let parent = format!("iso-parent-{}", c / 2);
        let mut ctx_sha = h(&format!("iso-ctx-{c}"));
        let z0 = if (c / 2) % 2 == 0 { 1.0 } else { -1.0 };
        ctx_sha.replace_range(..2, if z0 > 0.0 { "ff" } else { "00" });
        cf_rows.insert(
            ctx_sha.clone(),
            Features::context_id_vector(&ctx_sha, 16).unwrap(),
        );
        let cohort = c / 2;
        for p in 0..10usize {
            let prog_sha = h(&format!("iso-prog-{cohort}-{p}"));
            let phi0 = if p % 2 == 0 { 1.4 } else { -1.4 };
            let mut vec_p = vec![0.0; 16];
            vec_p[0] = phi0;
            pf_rows.insert(prog_sha.clone(), vec_p);
            let q = 0.50 + 0.25 * (z0 * phi0) + 0.01 * u(&format!("iso-n-{c}-{p}"));
            rows.push(MeasurementRow {
                context_id: ctx_sha.clone(),
                parent_id: parent.clone(),
                role: "train".into(),
                program_sha256: prog_sha,
                status: Status::Complete,
                q: Some(q),
                incumbent_q: Some(0.20),
                protocol_sha256: h(PROTOCOL),
                tokens: Some(vec![1, 3 + (p % 6) as u32, 2]),
            });
        }
    }
    let (pools, manifest) = freeze(&rows, &rule(), &h("iso-input")).unwrap();
    let cf = Features {
        dim: 16,
        rows: cf_rows,
    };
    let pf = Features {
        dim: 16,
        rows: pf_rows,
    };

    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let scores_set = known_scores();

    // Synthesize S3 gp_distilled/v1 rows: one per known context plus one unknown dataset without a known context.
    let mut s3_rows = Vec::new();
    for (idx, pool) in pools.iter().enumerate() {
        let spec = SelfPlayDistillSpec {
            dataset: pool.parent_id.clone(),
            context_sha256: pool.context_id.clone(),
            fold: Some(0),
            split_role: SplitRole::Train,
            score_name: "cv_geomean_rmse_ratio".into(),
            policylm_ckpt: None,
            campaign: Some("s3-prior-fixture".into()),
            run_id: format!("s3-run-{idx}"),
            family: "gp-results-index".into(),
            source_bucket: "jope-data".into(),
        };
        let obs = vec![
            SelfPlayDistillObservation {
                program_sha: pool.members[0].program_sha256.clone(),
                program: Some("x0 x1 +".into()),
                outcome: Outcome::Ok,
                score: Some(0.20),
                seed_packet: None,
                source_key: format!("results/{idx}-a.json"),
                source_version_id: None,
                source_sha256: h(&format!("s3-src-{idx}-a")),
            },
            SelfPlayDistillObservation {
                program_sha: pool.members[1].program_sha256.clone(),
                program: Some("x0 x1 -".into()),
                outcome: Outcome::Ok,
                score: Some(0.80),
                seed_packet: None,
                source_key: format!("results/{idx}-b.json"),
                source_version_id: None,
                source_sha256: h(&format!("s3-src-{idx}-b")),
            },
        ];
        s3_rows.extend(distill_selfplay_observations(&spec, &obs, &scores_set).unwrap());
    }
    let unknown_spec = SelfPlayDistillSpec {
        dataset: "unmapped_external_family".into(),
        context_sha256: h("unmapped-ctx"),
        fold: Some(0),
        split_role: SplitRole::Train,
        score_name: "cv_geomean_rmse_ratio".into(),
        policylm_ckpt: None,
        campaign: Some("s3-prior-fixture".into()),
        run_id: "s3-unknown".into(),
        family: "gp-results-index".into(),
        source_bucket: "jope-data".into(),
    };
    let unknown_obs = vec![SelfPlayDistillObservation {
        program_sha: pools[0].members[0].program_sha256.clone(),
        program: Some("x0".into()),
        outcome: Outcome::Ok,
        score: Some(0.50),
        seed_packet: None,
        source_key: "results/unmapped.json".into(),
        source_version_id: None,
        source_sha256: h("s3-src-unmapped"),
    }];
    s3_rows
        .extend(distill_selfplay_observations(&unknown_spec, &unknown_obs, &scores_set).unwrap());

    let fw_cfg = FlywheelConfig {
        eval: Config {
            reference: Reference::PoolFloor,
            budget: 1,
            folds: 5,
            fold_salt: "iso-folds".into(),
            prior_weight: 2.0,
            knn_k: 4,
            ridge_lambda: 1.0,
            bootstrap_draws: 200,
            bootstrap_seed: 19,
            alpha: 0.05,
        },
        eval_budgets: vec![1, 4],
        rounds: 3,
        initial_budget_per_context: 2,
        step_budget_per_context: 2,
        proposal_pool_k: 10,
        score_name: "shadow_pool_q".into(),
        tilt: MeasuredTiltPolicyConfig {
            temperature_tau: 0.25,
            advantage_clip: 2.0,
            ref_new_support_mass: 0.05,
            policy_new_support_cap: 0.12,
            step_size_eta: 0.20,
        },
        max_conditional_base_kl: 0.20,
        max_new_support_mass: 0.15,
    };

    // Verify fold isolation and continuous score-aware differentiation of S3 priors:
    // held-out fold 0 parents and unknown parents are excluded from fold 0 prior, and lower RMSE ratio (0.20)
    // yields a higher prior target (0.80) than higher RMSE ratio (0.80 -> 0.20).
    let (fold_priors, fold_counts) = build_fold_s3_priors(&pools, &fw_cfg.eval, &s3_rows).unwrap();
    assert_eq!(fold_priors.len(), 5);
    assert!(fold_counts.iter().all(|&c| c > 0));
    let fold0_heldout_progs: std::collections::BTreeSet<String> = pools
        .iter()
        .filter(|p| fold_of("iso-folds", &p.parent_id, 5) == 0)
        .flat_map(|p| p.members.iter().map(|m| m.program_sha256.clone()))
        .collect();
    for prog in &fold0_heldout_progs {
        assert!(
            !fold_priors[0].tot_stats.contains_key(prog),
            "held-out fold 0 program leaked into fold 0 S3 prior"
        );
    }
    let train_pool_for_fold0 = pools
        .iter()
        .find(|p| fold_of("iso-folds", &p.parent_id, 5) != 0)
        .unwrap();
    let (s_better, c_better) =
        fold_priors[0].tot_stats[&train_pool_for_fold0.members[0].program_sha256];
    let (s_worse, c_worse) =
        fold_priors[0].tot_stats[&train_pool_for_fold0.members[1].program_sha256];
    assert!(
        (s_better / c_better as f64) > (s_worse / c_worse as f64) + 0.4,
        "expected continuous S3 score quality to rank rmse_ratio=0.20 above rmse_ratio=0.80"
    );
    let mut non_ratio_row = s3_rows[0].clone();
    non_ratio_row.context_sha256 = Some(train_pool_for_fold0.context_id.clone());
    non_ratio_row.program_sha = train_pool_for_fold0.members[0].program_sha256.clone();
    non_ratio_row.score_name = Some("validation_rmse".into());
    non_ratio_row.score = Some(3.0);
    non_ratio_row.rank_ctx = Some(2);
    let mut unknown_dir_row = non_ratio_row.clone();
    unknown_dir_row.program_sha = train_pool_for_fold0.members[1].program_sha256.clone();
    unknown_dir_row.score_name = Some("unregistered_metric".into());
    unknown_dir_row.score = Some(3.0);
    unknown_dir_row.rank_ctx = Some(2);
    let (edge_priors, _) =
        build_fold_s3_priors(&pools, &fw_cfg.eval, &[non_ratio_row, unknown_dir_row]).unwrap();
    assert!(
        (edge_priors[0].tot_stats[&train_pool_for_fold0.members[0].program_sha256].0 - 0.25).abs()
            < 1e-12,
        "expected unbounded LowerIsBetter validation_rmse=3.0 to map to 1/(1+3)=0.25 without collapsing to 0"
    );
    assert!(
        (edge_priors[0].tot_stats[&train_pool_for_fold0.members[1].program_sha256].0 - 0.5).abs()
            < 1e-12,
        "expected unknown score_name to fall back to 1/rank_ctx=0.5"
    );

    // Perturb all outcomes in held-out fold 0 (`fold_of == 0`) and verify outer fold 0's
    // training acquisition rows (`run_id == "flywheel-r{r:02}-fold0"`) are 100% identical across all rounds.
    let mut perturbed_pools = pools.clone();
    for pool in &mut perturbed_pools {
        if fold_of("iso-folds", &pool.parent_id, 5) == 0 {
            for member in &mut pool.members {
                if let Some(q) = member.q.as_mut() {
                    *q = 10.0 - *q;
                }
            }
        }
    }
    let mut perturbed_manifest = manifest.clone();
    super::validation::rehash(&perturbed_pools, &mut perturbed_manifest);

    let shards_a = temp_a.path().join("shards");
    let shards_b = temp_b.path().join("shards");
    let receipt_a = run_distilled_flywheel_with_s3_prior(
        &pools,
        &manifest,
        &cf,
        Some(&pf),
        &[],
        Some(&s3_rows),
        &fw_cfg,
        Some(&shards_a),
    )
    .unwrap();
    let _receipt_b = run_distilled_flywheel_with_s3_prior(
        &perturbed_pools,
        &perturbed_manifest,
        &cf,
        Some(&pf),
        &[],
        Some(&s3_rows),
        &fw_cfg,
        Some(&shards_b),
    )
    .unwrap();
    assert!(receipt_a.s3_prior_rows_by_fold.iter().all(|&c| c > 0));
    for r in 0..3 {
        let prefix = format!("flywheel-r{r:02}-fold0");
        let bytes_a = std::fs::read(shards_a.join(format!("round-{r:02}.parquet"))).unwrap();
        let bytes_b = std::fs::read(shards_b.join(format!("round-{r:02}.parquet"))).unwrap();
        let rows_a: Vec<DistilledRow> = distilled::read(&bytes_a, &scores_set)
            .unwrap()
            .into_iter()
            .filter(|row| row.run_id == prefix)
            .collect();
        let rows_b: Vec<DistilledRow> = distilled::read(&bytes_b, &scores_set)
            .unwrap()
            .into_iter()
            .filter(|row| row.run_id == prefix)
            .collect();
        assert!(!rows_a.is_empty());
        assert_eq!(
            rows_a, rows_b,
            "held-out fold 0 outcome perturbation leaked into fold 0 training acquisition at round {r}"
        );
    }

    // Verify KL guard enforcement: setting max_new_support_mass below ref_new_support_mass (0.04 < 0.05)
    // triggers !kl.admissible in rounds >= 1 and actively freezes context acquisition (`kl_frozen_contexts > 0`).
    let mut strict_kl_cfg = fw_cfg.clone();
    strict_kl_cfg.max_new_support_mass = 0.04;
    let kl_receipt = run_distilled_flywheel_with_s3_prior(
        &pools,
        &manifest,
        &cf,
        Some(&pf),
        &[],
        None,
        &strict_kl_cfg,
        None,
    )
    .unwrap();
    assert!(!kl_receipt.all_rounds_kl_admissible);
    assert!(kl_receipt.rounds[1].kl_frozen_contexts > 0);
}
