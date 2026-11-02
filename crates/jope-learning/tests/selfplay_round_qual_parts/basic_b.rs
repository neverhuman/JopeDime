use super::*;

#[test]
fn downgrades_unequal_attempts_unequal_resources_and_insufficient_rounds() {
    let (identity, plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();
    ratchets[0]["equal_resource_comparison"] = json!(false);
    ratchets[1]["rounds"][1]["equal_attempts_in_round"] = json!(false);
    ratchets[2]["rounds"]
        .as_array_mut()
        .expect("rounds")
        .truncate(1);

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_unequal_resource_contexts, 1);
    assert_eq!(rep.n_unequal_attempt_rounds, 1);
    assert_eq!(rep.n_insufficient_rounds_contexts, 1);
    assert!(!rep.qualified_claim);
    assert!(rep
        .disqualification_reasons
        .contains(&"unequal_resource_comparison".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"unequal_attempts_in_round".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"insufficient_rounds".to_owned()));
}

#[test]
fn b5_kl_guard_fails_closed_on_malformed_unknown_or_missing_even_when_allow_frozen() {
    let (identity, mut plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();

    // 1. Valid measured KL freeze is disqualified when allow_kl_frozen_rounds is false,
    // and permitted (without kl_guard_unmeasured) when allow_kl_frozen_rounds is true.
    ratchets[0] = make_ratchet(
        &identity,
        "ctx1",
        "g0_ctx1",
        "g1_ctx1",
        true,
        true,
        "measured_tilt_leaf_policy",
        true,
    );
    let rep_frozen = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep_frozen.n_unmeasured_kl_contexts, 0);
    assert_eq!(rep_frozen.n_kl_frozen_contexts, 1);
    assert!(rep_frozen
        .disqualification_reasons
        .contains(&"kl_guard_frozen_round".to_owned()));

    plan.allow_kl_frozen_rounds = true;
    let rep_allowed = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep_allowed.n_unmeasured_kl_contexts, 0);
    assert!(rep_allowed.caller_asserted_diagnostic_checks_passed);

    // 2. Producer's "measured_tilt_leaf_policy_malformed" (or unknown kl_source, or missing decomposed_kl,
    // or missing top-level kl_guard_evaluated) MUST fail closed as unmeasured even when allow_kl_frozen_rounds is true!
    for bad_kl_source in [
        "measured_tilt_leaf_policy_malformed",
        "caller_evaluated_leaf_log_probs_malformed",
        "unmeasured_no_policy_log_probs",
        "custom_unknown_source",
    ] {
        let mut bad_ratchets = ratchets.clone();
        bad_ratchets[0]["rounds"][1]["kl_source"] = json!(bad_kl_source);
        let rep = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &bad_ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("report");
        assert_eq!(
            rep.n_unmeasured_kl_contexts, 1,
            "kl_source={bad_kl_source} must be counted as unmeasured"
        );
        assert!(rep
            .disqualification_reasons
            .contains(&"kl_guard_unmeasured".to_owned()));
        assert!(!rep.caller_asserted_diagnostic_checks_passed);
    }

    // Missing top-level kl_guard_evaluated must also fail closed.
    let mut missing_top_kl = ratchets.clone();
    missing_top_kl[0]
        .as_object_mut()
        .unwrap()
        .remove("kl_guard_evaluated");
    let rep_missing_top = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &missing_top_kl,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep_missing_top.n_unmeasured_kl_contexts, 1);

    // 3. Probability mass > 1.0, non-unit support mass sum (base + new != 1.0), or broken
    // total_kl chain-rule identity must fail closed as unmeasured KL.
    for bad_decomposed in [
        json!({
            "conditional_base_kl": 0.04,
            "conditional_new_kl": 0.01,
            "base_support_mass": 2.0,
            "new_support_mass": 2.0,
            "ref_new_support_mass": 2.0,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": 0.102,
            "admissible": true
        }),
        json!({
            "conditional_base_kl": 0.04,
            "conditional_new_kl": 0.01,
            "base_support_mass": 0.50,
            "new_support_mass": 0.20,
            "ref_new_support_mass": 0.08,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": 0.024,
            "admissible": true
        }),
        json!({
            "conditional_base_kl": 0.04,
            "conditional_new_kl": 0.01,
            "base_support_mass": 0.90,
            "new_support_mass": 0.10,
            "ref_new_support_mass": 0.08,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": 0.999,
            "admissible": true
        }),
    ] {
        let mut bad_mass_ratchets = ratchets.clone();
        bad_mass_ratchets[0]["rounds"][0]["decomposed_kl"] = bad_decomposed;
        let rep_bad_mass = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &bad_mass_ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("report");
        assert_eq!(
            rep_bad_mass.n_unmeasured_kl_contexts, 1,
            "invalid DecomposedSupportKl probability mass or chain-rule total_kl must be unmeasured"
        );
        assert!(rep_bad_mass
            .disqualification_reasons
            .contains(&"kl_guard_unmeasured".to_owned()));
        assert!(!rep_bad_mass.caller_asserted_diagnostic_checks_passed);
    }

    // 4. Forged admissible=true / unfrozen when conditional_base_kl > max_conditional_base_kl (0.35 > 0.20)
    // or new_support_mass > max_new_support_mass (0.20 > 0.15) must fail closed as unmeasured KL.
    for forged_admissible in [
        json!({
            "conditional_base_kl": 0.35,
            "conditional_new_kl": 0.01,
            "base_support_mass": 0.90,
            "new_support_mass": 0.10,
            "ref_new_support_mass": 0.08,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": 0.002 + 0.90 * 0.35 + 0.10 * 0.01,
            "admissible": true
        }),
        json!({
            "conditional_base_kl": 0.04,
            "conditional_new_kl": 0.01,
            "base_support_mass": 0.80,
            "new_support_mass": 0.20,
            "ref_new_support_mass": 0.08,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": 0.002 + 0.80 * 0.04 + 0.20 * 0.01,
            "admissible": true
        }),
    ] {
        let mut forged_ratchets = ratchets.clone();
        forged_ratchets[0]["rounds"][0]["decomposed_kl"] = forged_admissible;
        forged_ratchets[0]["rounds"][0]["seed_update_frozen_by_kl_guard"] = json!(false);
        let rep_forged = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &forged_ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("report");
        assert_eq!(
            rep_forged.n_unmeasured_kl_contexts, 1,
            "forged admissible=true exceeding KL or new-support mass caps must be unmeasured"
        );
        assert!(rep_forged
            .disqualification_reasons
            .contains(&"kl_guard_unmeasured".to_owned()));
        assert!(!rep_forged.caller_asserted_diagnostic_checks_passed);
    }
}

#[test]
fn m3_requires_at_least_two_parent_clusters_and_checks_practical_margin() {
    let (identity, mut plan, ratchets, scores, parity, mut parent_map) = fixture_bundle();

    // Plan with min_parent_clusters = 1 is refused.
    let mut one_cluster_plan = plan.clone();
    one_cluster_plan.min_parent_clusters = 1;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &one_cluster_plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // Collapse all contexts into 1 observed parent cluster: interval must NOT be computed (None).
    for v in parent_map.values_mut() {
        *v = "single_parent".to_owned();
    }
    let rep_single = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep_single.n_observed_parent_clusters, 1);
    assert!(rep_single.paired_log_reduction.is_none());
    assert!(rep_single
        .disqualification_reasons
        .contains(&"insufficient_parent_clusters".to_owned()));
    assert!(rep_single
        .disqualification_reasons
        .contains(&"paired_interval_not_computed".to_owned()));

    // Restore 3 parent clusters, but require an impossibly large practical margin (0.80).
    let (_, _, _, _, _, restored_parents) = fixture_bundle();
    plan.min_parent_clusters = 5;
    plan.practical_margin_log_ratio = 0.80;
    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &restored_parents,
    })
    .expect("report");
    assert!(!rep.margin_cleared);
    assert!(rep
        .disqualification_reasons
        .contains(&"insufficient_parent_clusters".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"margin_not_cleared".to_owned()));
}

#[test]
fn b3_b4_m1_m2_m4_m5_safety_guards() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();

    // B3: alpha > 0.025 is refused while Holm-Bonferroni is unimplemented.
    let mut high_alpha = plan.clone();
    high_alpha.alpha = 0.05;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &high_alpha,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // B4: Endpoint::RatchetVsOrdinaryFinalRound is refused as unsupported.
    let mut ord_plan = plan.clone();
    ord_plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &ord_plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // M1: Inconsistent dataset fingerprint between reference and candidate scores for ctx1 is refused.
    let mut mismatched_fp_scores = scores.clone();
    mismatched_fp_scores[1].dataset.test_sha256 = "d".repeat(64);
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &mismatched_fp_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // M1: Cross-dataset alias collision (dataset.id of one record equals dataset_hash of another) is refused.
    let mut alias_scores = scores.clone();
    alias_scores[2].dataset.dataset_hash = "ctx1".to_owned();
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &alias_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // M2: Reordered rounds ([round 1, round 0]) or wrong cumulative_prior_rounds is refused.
    let mut reordered_ratchets = ratchets.clone();
    reordered_ratchets[0]["rounds"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &reordered_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    let mut bad_cum_prior = ratchets.clone();
    bad_cum_prior[0]["rounds"][1]["cumulative_prior_rounds"] = json!(0);
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &bad_cum_prior,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // M4: Duplicate keys in JSON and TSV parent_map and extra TSV columns are refused.
    assert!(parse_parent_map(br#"{"ctx1": "p1", "ctx1": "p2"}"#).is_err());
    assert!(parse_parent_map(b"ctx1\tp1\nctx1\tp2\n").is_err());
    assert!(parse_parent_map(b"ctx1\tp1\textra_col\n").is_err());
    assert!(parse_parent_map(b"ctx1\tp1\nctx2\tp2\n").is_ok());

    // M5: Missing expected context (present in parent_of_context, dropped from ratchet_outputs)
    // and unreferenced test_score / parity records are counted and disqualified.
    let mut extra_parents = parent_map.clone();
    extra_parents.insert("ctx5_missing".to_owned(), "parent_d".to_owned());
    let mut extra_scores = scores.clone();
    extra_scores.push(make_score(
        "ctx1",
        "unreported_arm_genome",
        0.99,
        Vec::new(),
    ));
    let mut extra_parity = parity.clone();
    extra_parity.push(ParityRecord {
        dataset_id: "ctx1".to_owned(),
        genome: "unreported_arm_genome".to_owned(),
        passed: true,
        failure_reason: None,
    });
    let rep_m5 = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &extra_scores,
        parity: &extra_parity,
        parent_of_context: &extra_parents,
    })
    .expect("report");
    assert_eq!(rep_m5.n_missing_expected_contexts, 1);
    assert_eq!(rep_m5.n_unreferenced_test_scores, 1);
    assert_eq!(rep_m5.n_unreferenced_parity_records, 1);
    assert!(!rep_m5.caller_asserted_diagnostic_checks_passed);
    assert!(rep_m5
        .disqualification_reasons
        .contains(&"missing_expected_contexts".to_owned()));
    assert!(rep_m5
        .disqualification_reasons
        .contains(&"unreferenced_test_score_records".to_owned()));
    assert!(rep_m5
        .disqualification_reasons
        .contains(&"unreferenced_parity_records".to_owned()));
}
