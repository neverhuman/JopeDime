use super::*;

#[test]
fn real_equal_attempt_scratch_measures_policy_seeds_and_never_qualifies() {
    let (roster, cut, mut policy, retrieval) = fixture();
    if let Some(path) = std::env::var_os("CLOVER_GB06_POLICY_BATCH") {
        policy = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    }
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("scratch");
    let budget = ScratchBudget {
        evaluations_per_arm: 16,
        seconds_per_arm: 30,
        rng_seed: 20260907,
    };
    let role_bytes = roles();
    let request = ScratchRequest {
        roster: &roster,
        role_manifest: &role_bytes,
        context_sha256: &roster.contexts[0].context_sha256,
        cut: &cut,
        policy: &policy,
        retrieval: &retrieval,
        budget: &budget,
    };
    let receipt = compare_scratch(request, &output).unwrap();
    assert_eq!(receipt["arms"].as_array().unwrap().len(), 3);
    assert_eq!(receipt["equal_attempts"], true);
    assert_eq!(receipt["arms"][1]["seed_consumption"][0]["consumed"], true);
    assert_eq!(receipt["arms"][2]["seed_consumption"][0]["consumed"], true);
    assert!(receipt["arms"]
        .as_array()
        .unwrap()
        .iter()
        .all(|arm| arm["operator_counts"].get("mutate").is_some()
            || arm["operator_counts"].get("crossover").is_some()));
    assert_eq!(receipt["qualified_rewards"], 0);
    assert_eq!(receipt["h1_established"], false);
    assert_eq!(receipt["equal_resource_comparison"], false);
    assert!(compare_scratch(request, &output).is_err());
    if let Some(path) = std::env::var_os("CLOVER_GB06_TEST_RECEIPT") {
        std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    }
}

#[test]
fn multi_round_selfplay_ratchet_maintains_monotonic_q_and_kl_guard() {
    // Construct a nonlinear regression cut where feature interactions / stats improve over plain ridge.
    let cut = Cut {
        index: 0,
        train_x: (0..24)
            .map(|i| {
                let u = (i as f64 - 12.0) * 0.25;
                vec![u, u * u + 0.5 * u.sin(), (i % 5) as f64 - 2.0]
            })
            .collect(),
        train_y: (0..24)
            .map(|i| {
                let u = (i as f64 - 12.0) * 0.25;
                2.0 + 1.5 * u + 0.8 * (u * u) - 0.3 * ((i % 5) as f64 - 2.0).abs()
            })
            .collect(),
        hold_x: (24..32)
            .map(|i| {
                let u = (i as f64 - 20.0) * 0.25;
                vec![u, u * u + 0.5 * u.sin(), (i % 5) as f64 - 2.0]
            })
            .collect(),
        hold_y: (24..32)
            .map(|i| {
                let u = (i as f64 - 20.0) * 0.25;
                2.0 + 1.5 * u + 0.8 * (u * u) - 0.3 * ((i % 5) as f64 - 2.0).abs()
            })
            .collect(),
        hold_indices: (24..32).collect(),
    };
    let context = DiscoveryContext {
        context_sha256: pin("nonlinear-ratchet-context"),
        parent_family: "synthetic-linear-parent".into(),
        role: "train".into(),
        dataset_sha256: pin("nonlinear-ratchet-data"),
        cut: 0,
        training_sha256: rows_digest(&cut.train_x, &cut.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut.hold_x, &cut.hold_y).unwrap(),
        preprocessing_sha256: pin("fixture-preprocessing"),
        encoder_input_sha256: jope_pipeline::sha256_hex(&vec![0; 2051 * 8]),
    };
    let (_, _, mut policy, mut retrieval) = fixture();
    policy.context_sha256 = context.context_sha256.clone();
    policy.encoder_input_sha256 = context.encoder_input_sha256.clone();
    retrieval.seeds[0].source_context_sha256 = context.context_sha256.clone();
    let roster = TrainingRoster {
        schema: "clover.discovery-roster/v1".into(),
        role_manifest_sha256: jope_pipeline::sha256_hex(&roles()),
        contexts: vec![context.clone()],
    };

    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("ratchet");
    let round_budgets = vec![
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261001,
        },
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261002,
        },
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261003,
        },
    ];
    let role_bytes = roles();

    // 1. Up-front bound validation fails before any round directory is created.
    let bad_output = temp.path().join("ratchet_bad_bounds");
    let bad_req = SelfPlayRatchetRequest {
        roster: &roster,
        role_manifest: &role_bytes,
        context_sha256: &context.context_sha256,
        cut: &cut,
        initial_policy: &policy,
        initial_retrieval: &retrieval,
        round_budgets: &round_budgets,
        seed_cap_b: 4,
        max_conditional_base_kl: -0.01,
        max_new_support_mass: 0.15,
        round_leaf_log_prob_fn: None,
        measured_tilt_policy: None,
    };
    assert!(run_selfplay_ratchet(bad_req, &bad_output).is_err());
    assert!(!bad_output.exists());

    let bad_mass_req = SelfPlayRatchetRequest {
        max_conditional_base_kl: 0.05,
        max_new_support_mass: 1.2,
        ..bad_req
    };
    assert!(run_selfplay_ratchet(bad_mass_req, &bad_output).is_err());
    assert!(!bad_output.exists());

    // 2. Evaluated policy KL guard on actual runtime leaf identities (admissible across all 3 rounds,
    // including exact 0.15 new-mass boundary at round 2 when archive leaves exist).
    let admissible_eval = |round_idx: usize,
                           base_leaves: &[String],
                           archive_leaves: &[String]|
     -> (Vec<f64>, Vec<f64>) {
        let n_base = base_leaves.len();
        let n_new = archive_leaves.len();
        assert!(n_base >= 2);
        let ref_base_prob = 1.0 / (n_base as f64);
        // Reference base distribution is uniform; policy base distribution has mild tilt
        // (+0.05 on leaf 0 relative to uniform -> KL_base ≈ 0.005 <= 0.05).
        let p0 = ref_base_prob + 0.05;
        let pol_base_probs: Vec<f64> = (0..n_base)
            .map(|i| {
                if i == 0 {
                    p0
                } else {
                    (1.0 - p0) / ((n_base - 1) as f64)
                }
            })
            .collect();

        if n_new == 0 {
            let pol_lp = pol_base_probs.iter().map(|p| p.ln()).collect();
            let ref_lp = vec![ref_base_prob.ln(); n_base];
            (pol_lp, ref_lp)
        } else {
            let p_new = 0.05 * ((round_idx + 1) as f64);
            let p_base = 1.0 - p_new;
            let q_new = 0.01_f64;
            let q_base = 1.0 - q_new;
            let mut pol_lp = Vec::with_capacity(n_base + n_new);
            let mut ref_lp = Vec::with_capacity(n_base + n_new);
            for &pb in &pol_base_probs {
                pol_lp.push((p_base * pb).ln());
                ref_lp.push((q_base * ref_base_prob).ln());
            }
            for _ in 0..n_new {
                pol_lp.push((p_new / (n_new as f64)).ln());
                ref_lp.push((q_new / (n_new as f64)).ln());
            }
            (pol_lp, ref_lp)
        }
    };

    let req = SelfPlayRatchetRequest {
        roster: &roster,
        role_manifest: &role_bytes,
        context_sha256: &context.context_sha256,
        cut: &cut,
        initial_policy: &policy,
        initial_retrieval: &retrieval,
        round_budgets: &round_budgets,
        seed_cap_b: 4,
        max_conditional_base_kl: 0.05,
        max_new_support_mass: 0.15,
        round_leaf_log_prob_fn: Some(&admissible_eval),
        measured_tilt_policy: None,
    };
    let receipt = run_selfplay_ratchet(req, &output).unwrap();
    assert_eq!(receipt["schema"], "clover.selfplay-ratchet/v1");
    assert_eq!(receipt["monotonic_non_decreasing_best_q"], true);
    assert_eq!(receipt["matched_or_beat_ordinary_in_final_round"], true);
    assert_eq!(receipt["kl_guard_evaluated"], true);
    assert_eq!(receipt["all_rounds_kl_admissible"], true);
    assert_eq!(receipt["any_round_frozen_by_kl_guard"], false);
    assert_eq!(receipt["equal_resource_comparison"], false);
    let rounds = receipt["rounds"].as_array().unwrap();
    assert_eq!(rounds.len(), 3);
    assert_eq!(rounds[0]["equal_resource_across_rounds"], true);
    assert_eq!(rounds[1]["equal_resource_across_rounds"], false);
    for r in rounds {
        assert_eq!(r["bootstrap_anchor_source"], "measured_bootstrap");
        assert_eq!(r["kl_source"], "caller_evaluated_leaf_log_probs");
        assert!(r["kl_malformed_reason"].is_null());
        let base_len = r["base_leaf_genomes"].as_array().unwrap().len();
        let arch_len = r["archive_leaf_genomes"].as_array().unwrap().len();
        assert_eq!(
            base_len + arch_len,
            r["active_support_leaves"].as_u64().unwrap() as usize
        );
        assert!(r["decomposed_kl"]["conditional_base_kl"].as_f64().unwrap() > 1e-4);
        assert_eq!(r["decomposed_kl"]["admissible"], true);
        assert_eq!(r["seed_update_frozen_by_kl_guard"], false);
        let round_rec: SelfPlayRoundReceipt = serde_json::from_value(r.clone()).unwrap();
        assert!(!round_rec.ordinary_champion_genome.is_empty());
        assert_eq!(
            canonical_genome(&round_rec.ordinary_champion_genome).unwrap(),
            round_rec.ordinary_champion_genome
        );
        assert!(!round_rec.ratchet_champion_genome.is_empty());
        assert_eq!(
            canonical_genome(&round_rec.ratchet_champion_genome).unwrap(),
            round_rec.ratchet_champion_genome
        );
    }
    let q0 = rounds[0]["ratchet_best_q"].as_f64().unwrap();
    let q1 = rounds[1]["ratchet_best_q"].as_f64().unwrap();
    let q2 = rounds[2]["ratchet_best_q"].as_f64().unwrap();
    assert!(q1 + 1e-12 >= q0 && q2 + 1e-12 >= q1);

    // 3. Unconditional test: policy base-drift in Round 1 exceeds max_conditional_base_kl,
    // recording an inadmissible round and freezing Round 2 seeds to Round 1's seeds.
    let inadmissible_eval = |round_idx: usize,
                             base_leaves: &[String],
                             archive_leaves: &[String]|
     -> (Vec<f64>, Vec<f64>) {
        if round_idx != 1 {
            return admissible_eval(round_idx, base_leaves, archive_leaves);
        }
        let n_base = base_leaves.len();
        let n_new = archive_leaves.len();
        // Concentrate 0.98 of base mass on leaf 0 vs uniform reference -> KL_base > 0.50 > 0.05.
        let pol_base_probs: Vec<f64> = (0..n_base)
            .map(|i| {
                if i == 0 {
                    0.98
                } else {
                    0.02 / ((n_base - 1) as f64)
                }
            })
            .collect();
        let ref_base_prob = 1.0 / (n_base as f64);
        if n_new == 0 {
            (
                pol_base_probs.iter().map(|p| p.ln()).collect(),
                vec![ref_base_prob.ln(); n_base],
            )
        } else {
            let p_new = 0.40_f64;
            let p_base = 1.0 - p_new;
            let q_new = 0.01_f64;
            let q_base = 1.0 - q_new;
            let mut pol_lp = Vec::with_capacity(n_base + n_new);
            let mut ref_lp = Vec::with_capacity(n_base + n_new);
            for &pb in &pol_base_probs {
                pol_lp.push((p_base * pb).ln());
                ref_lp.push((q_base * ref_base_prob).ln());
            }
            for _ in 0..n_new {
                pol_lp.push((p_new / (n_new as f64)).ln());
                ref_lp.push((q_new / (n_new as f64)).ln());
            }
            (pol_lp, ref_lp)
        }
    };
    let output_over = temp.path().join("ratchet_over_kl");
    let req_over = SelfPlayRatchetRequest {
        round_leaf_log_prob_fn: Some(&inadmissible_eval),
        ..req
    };
    let receipt_over = run_selfplay_ratchet(req_over, &output_over).unwrap();
    assert_eq!(receipt_over["kl_guard_evaluated"], true);
    assert_eq!(receipt_over["all_rounds_kl_admissible"], false);
    assert_eq!(receipt_over["any_round_frozen_by_kl_guard"], true);
    assert_eq!(
        receipt_over["rounds"][1]["decomposed_kl"]["admissible"],
        false
    );
    assert_eq!(
        receipt_over["rounds"][1]["seed_update_frozen_by_kl_guard"],
        true
    );
    assert_eq!(
        receipt_over["rounds"][2]["seeded_genomes"],
        receipt_over["rounds"][1]["seeded_genomes"]
    );

    // 4. Unconditional test: malformed callback response (both wrong length and non-finite NaN/Inf)
    // is treated as an inadmissible round (freezing seeds and recording kl_malformed_reason) rather than aborting with Err.
    let malformed_eval = |round_idx: usize,
                          base_leaves: &[String],
                          archive_leaves: &[String]|
     -> (Vec<f64>, Vec<f64>) {
        let total = base_leaves.len() + archive_leaves.len();
        if round_idx == 1 {
            (Vec::new(), Vec::new())
        } else if round_idx == 2 {
            (vec![f64::NAN; total], vec![-1.0; total])
        } else {
            admissible_eval(round_idx, base_leaves, archive_leaves)
        }
    };
    let output_malformed = temp.path().join("ratchet_malformed_kl");
    let req_malformed = SelfPlayRatchetRequest {
        round_budgets: &round_budgets,
        round_leaf_log_prob_fn: Some(&malformed_eval),
        ..req
    };
    let receipt_malformed = run_selfplay_ratchet(req_malformed, &output_malformed).unwrap();
    assert_eq!(receipt_malformed["kl_guard_evaluated"], true);
    assert_eq!(receipt_malformed["all_rounds_kl_admissible"], false);
    assert_eq!(receipt_malformed["any_round_frozen_by_kl_guard"], true);
    assert_eq!(
        receipt_malformed["rounds"][1]["kl_source"],
        "caller_evaluated_leaf_log_probs_malformed"
    );
    assert!(receipt_malformed["rounds"][1]["kl_malformed_reason"]
        .as_str()
        .unwrap()
        .contains("expected"));
    assert!(receipt_malformed["rounds"][1]["decomposed_kl"].is_null());
    assert_eq!(
        receipt_malformed["rounds"][1]["seed_update_frozen_by_kl_guard"],
        true
    );
    assert_eq!(
        receipt_malformed["rounds"][2]["kl_source"],
        "caller_evaluated_leaf_log_probs_malformed"
    );
    assert!(receipt_malformed["rounds"][2]["kl_malformed_reason"]
        .as_str()
        .unwrap()
        .contains("finite leaf log-probabilities required"));
    assert_eq!(
        receipt_malformed["rounds"][2]["seed_update_frozen_by_kl_guard"],
        true
    );

    // 5. Unmeasured default path (round_leaf_log_prob_fn: None, measured_tilt_policy: None) makes no KL admissibility claims.
    let output_unmeasured = temp.path().join("ratchet_unmeasured");
    let req_unmeasured = SelfPlayRatchetRequest {
        round_budgets: &round_budgets[..2],
        round_leaf_log_prob_fn: None,
        measured_tilt_policy: None,
        ..req
    };
    let receipt_unmeasured = run_selfplay_ratchet(req_unmeasured, &output_unmeasured).unwrap();
    assert_eq!(receipt_unmeasured["kl_guard_evaluated"], false);
    assert!(receipt_unmeasured["all_rounds_kl_admissible"].is_null());
    assert!(receipt_unmeasured["any_round_frozen_by_kl_guard"].is_null());
    assert_eq!(
        receipt_unmeasured["rounds"][0]["kl_source"],
        "unmeasured_no_policy_log_probs"
    );
    assert!(receipt_unmeasured["rounds"][0]["decomposed_kl"].is_null());
    assert!(receipt_unmeasured["rounds"][0]["seed_update_frozen_by_kl_guard"].is_null());
}
