use super::*;

pub(crate) fn high_dim_surrogate_and_distilled_flywheel_establish_multi_round_lift_body() {
    use jope_learning::clover::expansion::MeasuredTiltPolicyConfig;
    use jope_learning::pools::flywheel::{
        run_distilled_flywheel, FlywheelConfig, FLYWHEEL_RECEIPT_SCHEMA,
    };
    use jope_learning::pools::selectors::{ContextualSurrogateQHat, MAX_SURROGATE_DESIGN_DIM};

    // Verify 2,051-d Kumo context features x 292-d token_bag features compress within MAX_SURROGATE_DESIGN_DIM.
    assert!(ContextualSurrogateQHat::design_dim(2051, 292).unwrap() > MAX_SURROGATE_DESIGN_DIM);
    let eff = ContextualSurrogateQHat::effective_design_dim(2051, 292).unwrap();
    assert!(eff <= MAX_SURROGATE_DESIGN_DIM);

    // Construct 40 contexts across 20 parents with 2,051-d context vectors and 16 candidate programs per context.
    // Only the top-variance context coordinate (coord 0) and program coordinate (coord 0) carry the true
    // interaction signal `z_0 * phi_0`, plus a secondary interaction `z_1 * phi_1` discovered once more programs
    // are acquired across flywheel rounds.
    let mut rows = Vec::new();
    let mut cf_rows = std::collections::BTreeMap::new();
    let mut pf_rows = std::collections::BTreeMap::new();
    for c in 0..40usize {
        let parent = format!("fw-parent-{}", c / 2);
        let mut ctx_sha = "80".repeat(32);
        ctx_sha.replace_range(62..64, &format!("{c:02x}"));
        let cohort = c / 2;
        let z0 = if cohort % 2 == 0 { 1.0 } else { -1.0 };
        let z1 = if (cohort / 2) % 2 == 0 { 1.0 } else { -1.0 };
        let z2 = if (cohort / 4) % 2 == 0 { 1.0 } else { -1.0 };
        let z3 = if (c % 2) == 0 { 1.0 } else { -1.0 };
        for (j, z) in [z0, z1, z2, z3].iter().enumerate() {
            ctx_sha.replace_range(2 * j..2 * j + 2, if *z > 0.0 { "ff" } else { "00" });
        }
        cf_rows.insert(
            ctx_sha.clone(),
            Features::context_id_vector(&ctx_sha, 2051).unwrap(),
        );
        for p in 0..16usize {
            let prog_sha = h(&format!("fw-prog-{cohort}-{p}"));
            let phi0 = if (p & 1) == 0 { 1.5 } else { -1.5 };
            let phi1 = if ((p >> 1) & 1) == 0 { 1.4 } else { -1.4 };
            let phi2 = if ((p >> 2) & 1) == 0 { 1.3 } else { -1.3 };
            let phi3 = if ((p >> 3) & 1) == 0 { 1.2 } else { -1.2 };
            let mut vec_p = vec![0.0; 64];
            vec_p[0] = phi0;
            vec_p[1] = phi1;
            vec_p[2] = phi2;
            vec_p[3] = phi3;
            pf_rows.insert(prog_sha.clone(), vec_p);
            let linear =
                0.14 * (z0 * phi0) + 0.12 * (z1 * phi1) + 0.10 * (z2 * phi2) + 0.08 * (z3 * phi3);
            let needle_bonus =
                if z0 * phi0 > 0.0 && z1 * phi1 > 0.0 && z2 * phi2 > 0.0 && z3 * phi3 > 0.0 {
                    0.35
                } else {
                    0.0
                };
            let q = 0.50 + linear + needle_bonus + 0.005 * u(&format!("fw-n-{c}-{p}"));
            rows.push(MeasurementRow {
                context_id: ctx_sha.clone(),
                parent_id: parent.clone(),
                role: "train".into(),
                program_sha256: prog_sha,
                status: Status::Complete,
                q: Some(q),
                incumbent_q: Some(0.20),
                protocol_sha256: h(PROTOCOL),
                tokens: Some(vec![1, 3 + (p % 8) as u32, 2]),
            });
        }
    }
    let (pools, manifest) = freeze(&rows, &rule(), &h("fw-input")).unwrap();
    let cf = Features {
        dim: 2051,
        rows: cf_rows,
    };
    let pf = Features {
        dim: 64,
        rows: pf_rows,
    };

    let temp = tempfile::tempdir().unwrap();
    let parquet_dir = temp.path().join("shards");
    let fw_cfg = FlywheelConfig {
        eval: Config {
            reference: Reference::PoolFloor,
            budget: 1,
            folds: 5,
            fold_salt: "fw-folds".into(),
            prior_weight: 2.0,
            knn_k: 4,
            ridge_lambda: 1.0,
            bootstrap_draws: 400,
            bootstrap_seed: 19,
            alpha: 0.05,
        },
        eval_budgets: vec![1, 4, 8],
        rounds: 4,
        initial_budget_per_context: 2,
        step_budget_per_context: 2,
        proposal_pool_k: 16,
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

    let receipt = run_distilled_flywheel(
        &pools,
        &manifest,
        &cf,
        Some(&pf),
        &[],
        &fw_cfg,
        Some(&parquet_dir),
    )
    .unwrap();

    if let Some(path) = std::env::var_os("I03_EVIDENCE_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("high-dimensional-flywheel.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
    }
    // An unbound cut-0 vector refuses before any shard directory is created.
    let mut unbound = cf.clone();
    unbound.rows.values_mut().next().unwrap()[0] = 0.75;
    let rejected = temp.path().join("unbound-shards");
    let error = run_distilled_flywheel(
        &pools,
        &manifest,
        &unbound,
        Some(&pf),
        &[],
        &fw_cfg,
        Some(&rejected),
    )
    .err()
    .unwrap();
    assert!(format!("{error:#}").contains("provenance"));
    assert!(!rejected.exists());
    assert_eq!(receipt.schema, FLYWHEEL_RECEIPT_SCHEMA);
    assert!(receipt.shadow_only && !receipt.qualified_claim && !receipt.l6_claim);
    assert_eq!(receipt.rounds.len(), 4);
    assert!(receipt.monotonic_train_hwm_gain);
    assert!(receipt.all_rounds_kl_admissible);
    assert!(receipt.active_discovery_established);
    assert!(receipt.surrogate_beats_strongest_baseline);
    assert_eq!(
        receipt.final_round_receipt.ranking_learnability,
        "ESTABLISHED_SHADOW"
    );
    assert_eq!(
        serde_json::to_value(&receipt.final_round_receipt.summaries["contextual_surrogate_qhat"])
            .unwrap(),
        serde_json::to_value(&receipt.rounds[3].heldout_by_budget[&1]).unwrap()
    );
    assert!(receipt.active_vs_passive_discovery_lift.mean >= 0.0);
    for r in 0..4 {
        let shard = parquet_dir.join(format!("round-{r:02}.parquet"));
        assert!(shard.exists(), "missing parquet shard {}", shard.display());
        assert_eq!(receipt.rounds[r].kl_frozen_contexts, 0);
    }
    assert!(
        receipt.rounds[3].heldout_by_budget[&1].mean_realized_gain
            >= receipt.rounds[0].heldout_by_budget[&1].mean_realized_gain
    );
}
