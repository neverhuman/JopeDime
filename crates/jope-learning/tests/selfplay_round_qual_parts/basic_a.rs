use super::*;

#[test]
fn refuses_malformed_inputs_and_identity_mismatch() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();

    // 1. Identity SHA mismatch
    let mut bad_plan = plan.clone();
    bad_plan.experiment_identity_sha256 = "0".repeat(64);
    assert!(qualify_rounds(RoundQualInputs {
        plan: &bad_plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 2. Non-hex prereg_sha256
    let mut bad_prereg = plan.clone();
    bad_prereg.prereg_sha256 = "not-a-hex-digest".to_owned();
    assert!(qualify_rounds(RoundQualInputs {
        plan: &bad_prereg,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 3. Too few bootstrap resamples (< 1000)
    let mut low_resamples = plan.clone();
    low_resamples.bootstrap_resamples = 500;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &low_resamples,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 4. Negative practical margin
    let mut neg_margin = plan.clone();
    neg_margin.practical_margin_log_ratio = -0.01;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &neg_margin,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 5. Non-positive test_rmse (0.0)
    let mut zero_scores = scores.clone();
    zero_scores[0].test_rmse = 0.0;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &zero_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 6. Context missing from parent_of_context
    let mut incomplete_parents = parent_map.clone();
    incomplete_parents.remove("ctx2");
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &incomplete_parents,
    })
    .is_err());

    // 7. Duplicate (dataset, genome) test score
    let mut dup_scores = scores.clone();
    dup_scores.push(scores[0].clone());
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &dup_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 8. Ratchet roster_sha256 mismatch against identity.data_roster_sha256
    let mut bad_roster_ratchets = ratchets.clone();
    bad_roster_ratchets[0]["roster_sha256"] = json!("f".repeat(64));
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &bad_roster_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 9. Missing policy_optimizer_updates on ratchet output
    let mut missing_updates = ratchets.clone();
    missing_updates[0]
        .as_object_mut()
        .unwrap()
        .remove("policy_optimizer_updates");
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &missing_updates,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());
}

#[test]
fn downgrades_scratch_and_reused_development_sources() {
    let (identity, mut plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();
    plan.source_grade = SourceGrade::ReusedDevelopment;
    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(!rep.qualified_claim);
    assert!(!rep.caller_asserted_diagnostic_checks_passed);
    assert_eq!(rep.provenance_binding_status, PROVENANCE_BINDING_STATUS);
    assert_eq!(rep.multiplicity_adjustment, MULTIPLICITY_ADJUSTMENT_STATUS);
    assert!(rep
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));

    // Even when plan.source_grade is Qualified, a ratchet output marked "scratch_only" is flagged.
    plan.source_grade = SourceGrade::Qualified;
    ratchets[0]["qualification"] = json!("scratch_only");
    let rep2 = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(rep2
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));
}

#[test]
fn b1_sequential_same_registry_looks_and_violations() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();

    // 1. Legitimate same-registry sequential pair (ref earlier_looks=[], cand earlier_looks=[ref_sha])
    // passes all caller-asserted diagnostic checks.
    let clean_rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("clean report");
    assert_eq!(clean_rep.n_earlier_looks_violations, 0);
    assert_eq!(clean_rep.n_registry_mismatch_violations, 0);
    assert!(clean_rep.caller_asserted_diagnostic_checks_passed);
    assert!(!clean_rep.qualified_claim);

    // 2. Registry reset / separate registry evasion: both reference and candidate have earlier_looks=[]
    // for two distinct genomes on ctx1. Must be disqualified!
    let mut reset_scores = scores.clone();
    reset_scores[1].earlier_looks = Some(Vec::new());
    let reset_rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &reset_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("reset report");
    assert_eq!(reset_rep.n_earlier_looks_violations, 1);
    assert!(!reset_rep.caller_asserted_diagnostic_checks_passed);
    assert!(reset_rep
        .disqualification_reasons
        .contains(&"test_split_earlier_looks_sequence_violation".to_owned()));

    // 3. Omitted earlier_looks field in JSON (deserializes as None in QualTestScore, whereas
    // TestScore's serde(default) previously turned it into []). Must be disqualified!
    let mut raw_score_json = serde_json::to_value(&scores[1]).unwrap();
    raw_score_json
        .as_object_mut()
        .unwrap()
        .remove("earlier_looks");
    let decoded_omitted: QualTestScore = serde_json::from_value(raw_score_json).unwrap();
    assert!(decoded_omitted.earlier_looks.is_none());
    let mut omitted_scores = scores.clone();
    omitted_scores[1] = decoded_omitted;
    let omitted_rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &omitted_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("omitted report");
    assert_eq!(omitted_rep.n_earlier_looks_violations, 1);
    assert!(!omitted_rep.caller_asserted_diagnostic_checks_passed);

    // 4. Extra adaptive look in candidate earlier_looks and feeds_selection: true
    let mut extra_scores = scores.clone();
    extra_scores[1]
        .earlier_looks
        .as_mut()
        .unwrap()
        .push("extra_prior_genome_sha".to_owned());
    extra_scores[2].feeds_selection = true;
    extra_scores[3].registry = "different-registry".to_owned();
    let extra_rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &extra_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("extra report");
    assert_eq!(extra_rep.n_earlier_looks_violations, 2);
    assert_eq!(extra_rep.n_feeds_selection_violations, 1);
    assert_eq!(extra_rep.n_registry_mismatch_violations, 1);
    assert!(extra_rep
        .disqualification_reasons
        .contains(&"test_score_registry_mismatch".to_owned()));
    assert!(extra_rep
        .disqualification_reasons
        .contains(&"test_split_earlier_looks_sequence_violation".to_owned()));
    assert!(extra_rep
        .disqualification_reasons
        .contains(&"test_score_feeds_selection".to_owned()));
}

#[test]
fn downgrades_missing_or_failed_parity_without_dropping_r5_failures() {
    let (identity, plan, ratchets, scores, mut parity, parent_map) = fixture_bundle();
    // Mark ctx1's candidate parity as failed.
    parity[1].passed = false;
    parity[1].failure_reason = Some("cpp17 status 8 width mismatch".to_owned());

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_failed_parity, 1);
    assert!(!rep.contexts[0].included_in_paired_interval);
    assert!(!rep.qualified_claim);
    assert!(!rep.caller_asserted_diagnostic_checks_passed);
    assert!(rep
        .disqualification_reasons
        .contains(&"parity_failure".to_owned()));

    // Remove ctx2's parity record and ctx2's scores so no unreferenced record remains, testing missing_parity_record.
    parity.retain(|p| !(p.dataset_id == "ctx2" && p.genome == "g1_ctx2"));
    let rep_missing = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep_missing.n_missing_parity, 1);
    assert!(rep_missing
        .disqualification_reasons
        .contains(&"missing_parity_record".to_owned()));
}
