//! Integration tests for [`jope_learning::clover::round_qual`] and `jope_pools qualify-rounds`.

use std::collections::BTreeMap;
use std::process::Command;

use jope_learning::canon::{canonical_sha256, sha256_hex};
use jope_learning::clover::round_qual::{
    parse_parent_map, qualify_rounds, qualify_rounds_with_prereg, Endpoint, ParityRecord,
    QualTestScore, RoundQualInputs, RoundQualPlan, SelfPlayPrereg, SourceGrade, UnpinnedOr,
    MULTIPLICITY_ADJUSTMENT_STATUS, PREREG_CALLER_ASSERTED_REASON, PREREG_PINNED_BINDING_STATUS,
    PREREG_SCHEMA, PREREG_SECONDARY_STATUS, PREREG_UNPINNED_BINDING_STATUS,
    PROVENANCE_BINDING_STATUS, RATCHET_SCHEMA, ROUND_QUAL_SCHEMA,
};
use jope_learning::clover::{Budget, ExperimentIdentity, InitialState, SourceIdentity, SCHEMA};
use jope_learning::pools::bootstrap::Verdict;
use jope_pipeline::test_score::{ScoredDataset, TEST_SCORE_SCHEMA_ID};
use serde_json::{json, Value};

fn fixture_identity() -> ExperimentIdentity {
    let hex64 = |b: u8| std::iter::repeat_n(b as char, 64).collect::<String>();
    let mut guards = BTreeMap::new();
    guards.insert("quantile".to_owned(), hex64(b'1'));
    guards.insert("label_visibility".to_owned(), hex64(b'2'));
    guards.insert("hold_recovery".to_owned(), hex64(b'3'));
    ExperimentIdentity {
        schema: SCHEMA.to_owned(),
        source: SourceIdentity {
            commit: "525be65b38540eb64b84c9bf36f7da6812812f2f".to_owned(),
            closure_sha256: hex64(b'a'),
        },
        initial_state: InitialState::Checkpoint {
            parameters_sha256: hex64(b'b'),
            optimizer_sha256: hex64(b'c'),
            rng_state_sha256: hex64(b'd'),
        },
        evaluator_sha256: hex64(b'e'),
        parity_contract_sha256: hex64(b'f'),
        support_sha256: hex64(b'4'),
        objective_sha256: hex64(b'5'),
        data_roster_sha256: hex64(b'6'),
        seed: 20_261_016,
        budget: Budget {
            attempts: 64,
            wall_seconds: 120,
            cpu_seconds: 240,
            gpu_seconds: 0,
            memory_bytes: 1_073_741_824,
            daily_bytes: 1_000_000,
            daily_puts: 1_000,
        },
        guards,
    }
}

fn fixture_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
    RoundQualPlan {
        experiment_identity_sha256: identity.sha256().expect("valid identity"),
        prereg_sha256: "8".repeat(64),
        source_grade: SourceGrade::Qualified,
        pinned_test_score_registry: "fixture-registry".to_owned(),
        min_parent_clusters: 3,
        practical_margin_log_ratio: 0.05,
        bootstrap_resamples: 1000,
        bootstrap_seed: 20_261_016,
        alpha: 0.025,
        endpoint: Endpoint::FinalVsRound0Incumbent,
        allow_kl_frozen_rounds: false,
        max_conditional_base_kl: 0.20,
        max_new_support_mass: 0.15,
        expected_contexts: Vec::new(),
    }
}

fn fixture_decomposed_kl(admissible: bool) -> Value {
    let conditional_base_kl = if admissible { 0.04 } else { 0.35 };
    let conditional_new_kl = 0.01;
    let base_support_mass = 0.90;
    let new_support_mass = 0.10;
    let ref_new_support_mass = 0.08;
    let bernoulli_allocation_kl = 0.002;
    let total_kl = bernoulli_allocation_kl
        + base_support_mass * conditional_base_kl
        + new_support_mass * conditional_new_kl;
    json!({
        "conditional_base_kl": conditional_base_kl,
        "conditional_new_kl": conditional_new_kl,
        "base_support_mass": base_support_mass,
        "new_support_mass": new_support_mass,
        "ref_new_support_mass": ref_new_support_mass,
        "bernoulli_allocation_kl": bernoulli_allocation_kl,
        "total_kl": total_kl,
        "admissible": admissible
    })
}

#[allow(clippy::too_many_arguments)]
fn make_ratchet(
    identity: &ExperimentIdentity,
    ctx: &str,
    g_round0: &str,
    g_final: &str,
    equal_resource: bool,
    equal_attempts: bool,
    kl_source: &str,
    kl_frozen: bool,
) -> Value {
    let round1_decomposed = if matches!(
        kl_source,
        "measured_tilt_leaf_policy" | "caller_evaluated_leaf_log_probs"
    ) {
        fixture_decomposed_kl(!kl_frozen)
    } else {
        Value::Null
    };
    json!({
        "schema": RATCHET_SCHEMA,
        "qualification": "qualified",
        "roster_sha256": identity.data_roster_sha256,
        "context_sha256": ctx,
        "equal_resource_comparison": equal_resource,
        "kl_guard_evaluated": kl_source != "unmeasured_no_policy_log_probs",
        "any_round_frozen_by_kl_guard": kl_frozen,
        "policy_optimizer_updates": 0,
        "rounds": [
            {
                "round": 0,
                "cumulative_prior_rounds": 0,
                "ordinary_best_rmse": 1.00,
                "ordinary_best_q": 0.50,
                "ratchet_best_rmse": 1.00,
                "ratchet_best_q": 0.50,
                "surplus_q_vs_ordinary": 0.00,
                "ratchet_champion_genome": g_round0,
                "equal_attempts_in_round": equal_attempts,
                "kl_source": "measured_tilt_leaf_policy",
                "kl_malformed_reason": Value::Null,
                "decomposed_kl": fixture_decomposed_kl(true),
                "seed_update_frozen_by_kl_guard": false
            },
            {
                "round": 1,
                "cumulative_prior_rounds": 1,
                "ordinary_best_rmse": 0.96,
                "ordinary_best_q": 0.51,
                "ratchet_best_rmse": 0.72,
                "ratchet_best_q": 0.58,
                "surplus_q_vs_ordinary": 0.07,
                "ratchet_champion_genome": g_final,
                "equal_attempts_in_round": equal_attempts,
                "kl_source": kl_source,
                "kl_malformed_reason": Value::Null,
                "decomposed_kl": round1_decomposed,
                "seed_update_frozen_by_kl_guard": kl_frozen
            }
        ]
    })
}

fn make_score(
    ctx: &str,
    genome: &str,
    test_rmse: f64,
    earlier_looks: Vec<String>,
) -> QualTestScore {
    QualTestScore {
        schema: TEST_SCORE_SCHEMA_ID.to_owned(),
        tool_version: "0.3.0".to_owned(),
        created_at: "2026-10-16T17:30:00Z".to_owned(),
        split: "test".to_owned(),
        receipt_sha256: sha256_hex(format!("receipt:{ctx}:{genome}").as_bytes()),
        dataset: ScoredDataset {
            id: ctx.to_owned(),
            dataset_hash: ctx.to_owned(),
            train_sha256: "b".repeat(64),
            test_sha256: "c".repeat(64),
            train_rows: 100,
            test_rows: 50,
        },
        genome: genome.to_owned(),
        program_sha256: sha256_hex(genome.as_bytes()),
        test_rmse,
        test_r2: 0.85,
        validation_rmse: test_rmse * 0.98,
        registry: "fixture-registry".to_owned(),
        earlier_looks: Some(earlier_looks),
        feeds_selection: false,
        seconds: 0.1,
    }
}

type FixtureBundle = (
    ExperimentIdentity,
    RoundQualPlan,
    Vec<Value>,
    Vec<QualTestScore>,
    Vec<ParityRecord>,
    BTreeMap<String, String>,
);

fn fixture_bundle() -> FixtureBundle {
    let identity = fixture_identity();
    let plan = fixture_plan(&identity);
    let mut parent_map = BTreeMap::new();
    let mut ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();

    for (ctx, parent, ref_rmse, cand_rmse) in [
        ("ctx1", "parent_a", 1.00, 0.72),
        ("ctx2", "parent_b", 1.10, 0.78),
        ("ctx3", "parent_c", 0.90, 0.64),
        ("ctx4", "parent_c", 1.20, 0.84),
    ] {
        parent_map.insert(ctx.to_owned(), parent.to_owned());
        let g0 = format!("g0_{ctx}");
        let g1 = format!("g1_{ctx}");
        let ref_score = make_score(ctx, &g0, ref_rmse, Vec::new());
        let cand_score = make_score(ctx, &g1, cand_rmse, vec![ref_score.program_sha256.clone()]);
        ratchets.push(make_ratchet(
            &identity,
            ctx,
            &g0,
            &g1,
            true,
            true,
            "measured_tilt_leaf_policy",
            false,
        ));
        scores.push(ref_score);
        scores.push(cand_score);
        for g in [&g0, &g1] {
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: (*g).clone(),
                passed: true,
                failure_reason: None,
            });
        }
    }

    (identity, plan, ratchets, scores, parity, parent_map)
}

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

#[test]
fn cli_qualify_rounds_create_only_and_canonical_digest() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();
    let direct = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("direct report");
    assert_eq!(direct.verdict, Verdict::Greater);
    assert!(direct.margin_cleared);

    let dir = tempfile::tempdir().expect("tempdir");
    let plan_path = dir.path().join("plan.json");
    let id_path = dir.path().join("identity.json");
    let ratchets_path = dir.path().join("ratchets.json");
    let scores_path = dir.path().join("scores.json");
    let parity_path = dir.path().join("parity.json");
    let parents_path = dir.path().join("parents.json");
    let out_path = dir.path().join("report.json");

    std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    std::fs::write(&id_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    std::fs::write(&ratchets_path, serde_json::to_vec(&ratchets).unwrap()).unwrap();
    std::fs::write(&scores_path, serde_json::to_vec(&scores).unwrap()).unwrap();
    std::fs::write(&parity_path, serde_json::to_vec(&parity).unwrap()).unwrap();
    std::fs::write(&parents_path, serde_json::to_vec(&parent_map).unwrap()).unwrap();

    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let first = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run jope_pools qualify-rounds");
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let stdout = String::from_utf8_lossy(&first.stdout);
    let expected_sha = direct.canonical_sha256().unwrap();
    assert!(stdout.contains(&expected_sha), "stdout={stdout}");

    let written: Value = serde_json::from_slice(&std::fs::read(&out_path).unwrap()).unwrap();
    assert_eq!(written["schema"], ROUND_QUAL_SCHEMA);

    // Second invocation against the same --out path must fail closed (create-only).
    let second = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("re-run jope_pools qualify-rounds");
    assert!(!second.status.success());
}

#[test]
fn exp_next_02_secondary_endpoint_and_prereg_binding_verification() {
    let identity = fixture_identity();
    let mut plan = fixture_plan(&identity);
    plan.bootstrap_resamples = 2000;
    plan.bootstrap_seed = 20_261_016;
    plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;

    let prereg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/internal/clover/selfplay-round1-prereg.v1.json");
    let committed_prereg_bytes =
        std::fs::read(&prereg_path).expect("read committed selfplay-round1-prereg.v1.json");
    let parsed_committed: SelfPlayPrereg =
        serde_json::from_slice(&committed_prereg_bytes).expect("parse SelfPlayPrereg");
    assert_eq!(parsed_committed.schema, PREREG_SCHEMA);
    assert!(parsed_committed.has_unpinned_fields());
    assert!(!parsed_committed.execution_grant.authorized);

    plan.prereg_sha256 = sha256_hex(&committed_prereg_bytes);

    let mut parent_map = BTreeMap::new();
    let mut ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();

    for (ctx, parent, ord_rmse, cand_rmse) in [
        ("ctx1", "parent_a", 1.00, 0.72),
        ("ctx2", "parent_b", 1.10, 0.78),
        ("ctx3", "parent_c", 0.90, 0.64),
        ("ctx4", "parent_c", 1.20, 0.84),
    ] {
        parent_map.insert(ctx.to_owned(), parent.to_owned());
        let g_r0 = format!("g_r0_{ctx}");
        let g_ord_final = format!("g_ord_final_{ctx}");
        let g_rat_final = format!("g_rat_final_{ctx}");

        let ref_score = make_score(ctx, &g_ord_final, ord_rmse, Vec::new());
        let cand_score = make_score(
            ctx,
            &g_rat_final,
            cand_rmse,
            vec![ref_score.program_sha256.clone()],
        );
        scores.push(ref_score);
        scores.push(cand_score);
        for g in [&g_ord_final, &g_rat_final] {
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: (*g).clone(),
                passed: true,
                failure_reason: None,
            });
        }

        let rounds: Vec<Value> = (0..4)
            .map(|r_idx| {
                let is_last = r_idx == 3;
                let rat_g = if is_last { &g_rat_final } else { &g_r0 };
                let ord_g = if is_last { &g_ord_final } else { &g_r0 };
                json!({
                    "round": r_idx,
                    "cumulative_prior_rounds": r_idx,
                    "ordinary_best_rmse": 1.00 - 0.01 * (r_idx as f64),
                    "ordinary_best_q": 0.50 + 0.005 * (r_idx as f64),
                    "ordinary_champion_genome": ord_g,
                    "ordinary_search_sha256": sha256_hex(format!("ord_s:{ctx}:{r_idx}").as_bytes()),
                    "ordinary_observations_sha256": sha256_hex(format!("ord_o:{ctx}:{r_idx}").as_bytes()),
                    "ratchet_best_rmse": 1.00 - 0.08 * (r_idx as f64),
                    "ratchet_best_q": 0.50 + 0.03 * (r_idx as f64),
                    "ratchet_champion_genome": rat_g,
                    "ratchet_search_sha256": sha256_hex(format!("rat_s:{ctx}:{r_idx}").as_bytes()),
                    "ratchet_observations_sha256": sha256_hex(format!("rat_o:{ctx}:{r_idx}").as_bytes()),
                    "surplus_q_vs_ordinary": 0.025 * (r_idx as f64),
                    "equal_attempts_in_round": true,
                    "kl_source": "measured_tilt_leaf_policy",
                    "kl_malformed_reason": Value::Null,
                    "decomposed_kl": fixture_decomposed_kl(true),
                    "seed_update_frozen_by_kl_guard": false
                })
            })
            .collect();

        ratchets.push(json!({
            "schema": RATCHET_SCHEMA,
            "qualification": "qualified",
            "roster_sha256": identity.data_roster_sha256,
            "context_sha256": ctx,
            "seed_cap_b": 8,
            "max_conditional_base_kl": 0.20,
            "max_new_support_mass": 0.15,
            "measured_tilt_policy": parsed_committed.arms_and_rounds.measured_tilt_policy,
            "equal_resource_comparison": true,
            "kl_guard_evaluated": true,
            "any_round_frozen_by_kl_guard": false,
            "policy_optimizer_updates": 0,
            "rounds": rounds
        }));
    }

    assert_eq!(
        parsed_committed.endpoints.secondary_status,
        PREREG_SECONDARY_STATUS
    );
    assert_eq!(
        parsed_committed.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert_eq!(
        parsed_committed
            .arms_and_rounds
            .kl_guard
            .allowlisted_measured_sources,
        vec!["measured_tilt_leaf_policy".to_owned()]
    );

    // 1. Verify Endpoint::RatchetVsOrdinaryFinalRound with committed UNPINNED prereg packet.
    let rep_unpinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&committed_prereg_bytes),
    )
    .expect("valid report with committed unpinned prereg");

    assert_eq!(rep_unpinned.endpoint, Endpoint::RatchetVsOrdinaryFinalRound);
    assert_eq!(
        rep_unpinned.contexts[0].candidate_genome.as_deref(),
        Some("g_rat_final_ctx1")
    );
    assert_eq!(
        rep_unpinned.contexts[0].reference_genome.as_deref(),
        Some("g_ord_final_ctx1")
    );
    assert_eq!(rep_unpinned.verdict, Verdict::Greater);
    assert!(rep_unpinned.margin_cleared);
    assert!(rep_unpinned.caller_asserted_diagnostic_checks_passed);
    assert_eq!(
        rep_unpinned.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert!(!rep_unpinned.qualified_claim);
    assert!(!rep_unpinned.promotion);
    assert!(!rep_unpinned.l6_claim);
    assert_eq!(
        rep_unpinned.disqualification_reasons,
        vec![
            "prereg_unpinned_or_no_execution_grant".to_owned(),
            PREREG_CALLER_ASSERTED_REASON.to_owned(),
            "qualified_execution_mode_disabled_offline_verifier_only".to_owned(),
            "no_adopted_sealed_family_boundary".to_owned(),
        ]
    );

    // 2. Verify CLI `jope_pools qualify-rounds --prereg <PATH>` produces the exact same canonical digest
    // and includes provenance_binding_status in stdout.
    let dir = tempfile::tempdir().expect("tempdir");
    let plan_file = dir.path().join("plan.json");
    let id_file = dir.path().join("id.json");
    let ratchets_file = dir.path().join("ratchets.json");
    let scores_file = dir.path().join("scores.json");
    let parity_file = dir.path().join("parity.json");
    let parents_file = dir.path().join("parents.json");
    let out_file = dir.path().join("report_prereg.json");
    std::fs::write(&plan_file, serde_json::to_vec(&plan).unwrap()).unwrap();
    std::fs::write(&id_file, serde_json::to_vec(&identity).unwrap()).unwrap();
    std::fs::write(&ratchets_file, serde_json::to_vec(&ratchets).unwrap()).unwrap();
    std::fs::write(&scores_file, serde_json::to_vec(&scores).unwrap()).unwrap();
    std::fs::write(&parity_file, serde_json::to_vec(&parity).unwrap()).unwrap();
    std::fs::write(&parents_file, serde_json::to_vec(&parent_map).unwrap()).unwrap();

    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let cli_out = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_file.to_str().unwrap(),
            "--identity",
            id_file.to_str().unwrap(),
            "--ratchets",
            ratchets_file.to_str().unwrap(),
            "--test-scores",
            scores_file.to_str().unwrap(),
            "--parity",
            parity_file.to_str().unwrap(),
            "--parent-map",
            parents_file.to_str().unwrap(),
            "--prereg",
            prereg_path.to_str().unwrap(),
            "--out",
            out_file.to_str().unwrap(),
        ])
        .output()
        .expect("run jope_pools qualify-rounds --prereg");
    assert!(
        cli_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&cli_out.stderr)
    );
    let stdout = String::from_utf8_lossy(&cli_out.stdout);
    assert!(stdout.contains(&rep_unpinned.canonical_sha256().unwrap()));
    assert!(stdout.contains(PREREG_UNPINNED_BINDING_STATUS));

    // 3. Construct a fully pinned SelfPlayPrereg packet and verify provenance_binding_status transitions
    // to PREREG_PINNED_BINDING_STATUS while qualified_claim/promotion/l6_claim stay false with
    // both caller-asserted and offline-verifier-only disqualification reasons.
    let mut pinned_prereg = parsed_committed.clone();
    pinned_prereg.status = "PRE_REGISTERED_PINNED_SYNTHETIC_FIXTURE".to_owned();
    pinned_prereg.provenance_binding_status = PREREG_PINNED_BINDING_STATUS.to_owned();
    pinned_prereg.experiment_identity_sha256 =
        UnpinnedOr::Pinned(plan.experiment_identity_sha256.clone());
    pinned_prereg.roster_and_families.sealed_roster_sha256 =
        UnpinnedOr::Pinned(identity.data_roster_sha256.clone());
    pinned_prereg.roster_and_families.parent_cluster_map_sha256 =
        UnpinnedOr::Pinned(canonical_sha256(&parent_map).unwrap());
    pinned_prereg.roster_and_families.g23_sealed_boundary_ruling =
        UnpinnedOr::Pinned("g23-synthetic-ruling-fixture".to_owned());
    pinned_prereg.sample_size_and_margin.min_parent_clusters =
        UnpinnedOr::Pinned(plan.min_parent_clusters);
    pinned_prereg
        .sample_size_and_margin
        .practical_margin_log_ratio = UnpinnedOr::Pinned(plan.practical_margin_log_ratio);
    pinned_prereg
        .sample_size_and_margin
        .c158_g_judge_binding_receipt = UnpinnedOr::Pinned("c158-synthetic-receipt".to_owned());
    pinned_prereg
        .held_out_and_parity_rules
        .pinned_test_score_registry = UnpinnedOr::Pinned(plan.pinned_test_score_registry.clone());
    pinned_prereg.execution_grant.authorized = true;
    pinned_prereg.execution_grant.owner_admission_locator =
        UnpinnedOr::Pinned("pending-review-locator-2026".to_owned());
    pinned_prereg.execution_grant.hosts = UnpinnedOr::Pinned(vec!["xbabe2".to_owned()]);
    pinned_prereg.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(50);
    pinned_prereg.execution_grant.deadline_utc =
        UnpinnedOr::Pinned("2026-10-18T00:00:00Z".to_owned());
    assert!(!pinned_prereg.has_unpinned_fields());

    let pinned_bytes = serde_json::to_vec_pretty(&pinned_prereg).unwrap();
    let mut pinned_plan = plan.clone();
    pinned_plan.prereg_sha256 = sha256_hex(&pinned_bytes);

    let rep_pinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &pinned_plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .expect("valid report with pinned prereg");
    assert_eq!(
        rep_pinned.provenance_binding_status,
        PREREG_PINNED_BINDING_STATUS
    );
    assert!(rep_pinned.caller_asserted_diagnostic_checks_passed);
    assert!(!rep_pinned.qualified_claim);
    assert!(!rep_pinned.promotion);
    assert!(!rep_pinned.l6_claim);
    assert_eq!(
        rep_pinned.disqualification_reasons,
        vec![
            PREREG_CALLER_ASSERTED_REASON.to_owned(),
            "qualified_execution_mode_disabled_offline_verifier_only".to_owned(),
            "no_adopted_sealed_family_boundary".to_owned(),
        ]
    );

    // Also verify canonical SHA-256 binding when plan.prereg_sha256 is canonical_sha256(&pinned_prereg)
    // and the file is pretty-printed JSON.
    let mut canonical_pinned_plan = pinned_plan.clone();
    canonical_pinned_plan.prereg_sha256 = canonical_sha256(&pinned_prereg).unwrap();
    assert_ne!(
        canonical_pinned_plan.prereg_sha256,
        pinned_plan.prereg_sha256
    );
    let rep_canonical_pinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &canonical_pinned_plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .expect("valid report with canonical prereg_sha256");
    assert_eq!(
        rep_canonical_pinned.prereg_sha256,
        canonical_pinned_plan.prereg_sha256
    );
    assert_eq!(
        rep_canonical_pinned.provenance_binding_status,
        PREREG_PINNED_BINDING_STATUS
    );
    assert_eq!(
        rep_canonical_pinned.disqualification_reasons,
        rep_pinned.disqualification_reasons
    );

    // 4. Table-driven fail-closed negative matrix on mutated SelfPlayPrereg packets (with recomputed SHA-256):
    let assert_prereg_rejected = |label: &str, mutated: &SelfPlayPrereg| {
        let bytes = serde_json::to_vec(mutated).unwrap();
        let mut p = pinned_plan.clone();
        p.prereg_sha256 = sha256_hex(&bytes);
        let res = qualify_rounds_with_prereg(
            RoundQualInputs {
                plan: &p,
                identity: &identity,
                ratchet_outputs: &ratchets,
                test_scores: &scores,
                parity: &parity,
                parent_of_context: &parent_map,
            },
            Some(&bytes),
        );
        assert!(
            res.is_err(),
            "expected prereg mutation `{label}` to be rejected, got Ok"
        );
    };

    type PreregMutator = (&'static str, Box<dyn Fn(&mut SelfPlayPrereg)>);
    let prereg_mutations: Vec<PreregMutator> = vec![
        ("schema", Box::new(|p| p.schema = "wrong".into())),
        (
            "qualification_report_schema",
            Box::new(|p| p.qualification_report_schema = "wrong".into()),
        ),
        (
            "source_commit_at_draft_len",
            Box::new(|p| p.source_commit_at_draft = "not-40-hex".into()),
        ),
        (
            "endpoints.primary",
            Box::new(|p| p.endpoints.primary = Endpoint::RatchetVsOrdinaryFinalRound),
        ),
        (
            "endpoints.secondary",
            Box::new(|p| p.endpoints.secondary = Endpoint::FinalVsRound0Incumbent),
        ),
        (
            "endpoints.secondary_status",
            Box::new(|p| p.endpoints.secondary_status = "WRONG_SECONDARY_STATUS".into()),
        ),
        (
            "endpoints.alpha_primary_strict",
            Box::new(|p| p.endpoints.alpha_primary_strict = 0.05),
        ),
        (
            "endpoints.cli_single_endpoint_alpha_cap",
            Box::new(|p| p.endpoints.cli_single_endpoint_alpha_cap = 0.01),
        ),
        (
            "endpoints.f10_tie_tolerance_log_ratio",
            Box::new(|p| p.endpoints.f10_tie_tolerance_log_ratio = 1e-6),
        ),
        (
            "roster.unit_of_inference",
            Box::new(|p| p.roster_and_families.unit_of_inference = "dataset".into()),
        ),
        (
            "roster.roster_role",
            Box::new(|p| p.roster_and_families.roster_role = "test".into()),
        ),
        (
            "roster.sealed_roster_sha256",
            Box::new(|p| {
                p.roster_and_families.sealed_roster_sha256 = UnpinnedOr::Pinned("0".repeat(64));
            }),
        ),
        (
            "roster.parent_cluster_map_sha256",
            Box::new(|p| {
                p.roster_and_families.parent_cluster_map_sha256 =
                    UnpinnedOr::Pinned("0".repeat(64));
            }),
        ),
        (
            "roster.g23_empty",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling = UnpinnedOr::Pinned("   ".into());
            }),
        ),
        (
            "roster.g23_sentinel_unpinned_ws",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling =
                    UnpinnedOr::Pinned(" unpinned ".into());
            }),
        ),
        (
            "roster.g23_sentinel_tbd",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling = UnpinnedOr::Pinned("TBD".into());
            }),
        ),
        (
            "sample.bootstrap_resamples",
            Box::new(|p| p.sample_size_and_margin.bootstrap_resamples = 999),
        ),
        (
            "sample.bootstrap_seed",
            Box::new(|p| p.sample_size_and_margin.bootstrap_seed = 42),
        ),
        (
            "sample.bootstrap_method",
            Box::new(|p| p.sample_size_and_margin.bootstrap_method = "percentile".into()),
        ),
        (
            "sample.min_parent_clusters",
            Box::new(|p| p.sample_size_and_margin.min_parent_clusters = UnpinnedOr::Pinned(99)),
        ),
        (
            "sample.practical_margin_log_ratio",
            Box::new(|p| {
                p.sample_size_and_margin.practical_margin_log_ratio = UnpinnedOr::Pinned(0.99);
            }),
        ),
        (
            "sample.c158_sentinel_todo",
            Box::new(|p| {
                p.sample_size_and_margin.c158_g_judge_binding_receipt =
                    UnpinnedOr::Pinned(" todo ".into());
            }),
        ),
        (
            "schedule.rounds",
            Box::new(|p| p.arms_and_rounds.rounds = 2),
        ),
        (
            "schedule.seed_cap_b",
            Box::new(|p| p.arms_and_rounds.seed_cap_b = 4),
        ),
        (
            "schedule.equal_attempts",
            Box::new(|p| p.arms_and_rounds.equal_attempts_per_round_required = false),
        ),
        (
            "schedule.equal_resource",
            Box::new(|p| p.arms_and_rounds.equal_resource_comparison_required = false),
        ),
        (
            "schedule.tilt.temperature_tau",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.temperature_tau = 0.99),
        ),
        (
            "schedule.tilt.advantage_clip",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.advantage_clip = 9.0),
        ),
        (
            "schedule.tilt.ref_new_support_mass",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.ref_new_support_mass = 0.9),
        ),
        (
            "schedule.tilt.policy_new_support_cap",
            Box::new(|p| {
                p.arms_and_rounds
                    .measured_tilt_policy
                    .policy_new_support_cap = 0.9
            }),
        ),
        (
            "schedule.tilt.step_size_eta",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.step_size_eta = 0.9),
        ),
        (
            "kl.max_conditional_base_kl",
            Box::new(|p| p.arms_and_rounds.kl_guard.max_conditional_base_kl = 0.50),
        ),
        (
            "kl.max_new_support_mass",
            Box::new(|p| p.arms_and_rounds.kl_guard.max_new_support_mass = 0.50),
        ),
        (
            "kl.allow_kl_frozen_rounds",
            Box::new(|p| p.arms_and_rounds.kl_guard.allow_kl_frozen_rounds = true),
        ),
        (
            "kl.allowlisted_both_sources_rejected_in_prereg",
            Box::new(|p| {
                p.arms_and_rounds.kl_guard.allowlisted_measured_sources = vec![
                    "measured_tilt_leaf_policy".into(),
                    "caller_evaluated_leaf_log_probs".into(),
                ];
            }),
        ),
        (
            "kl.allowlisted_caller_source_only_rejected_in_prereg",
            Box::new(|p| {
                p.arms_and_rounds.kl_guard.allowlisted_measured_sources =
                    vec!["caller_evaluated_leaf_log_probs".into()];
            }),
        ),
        (
            "held_out.test_score_schema",
            Box::new(|p| p.held_out_and_parity_rules.test_score_schema = "wrong".into()),
        ),
        (
            "held_out.pinned_test_score_registry",
            Box::new(|p| {
                p.held_out_and_parity_rules.pinned_test_score_registry =
                    UnpinnedOr::Pinned("wrong".into());
            }),
        ),
        (
            "held_out.sequential_same_registry_looks.reject_omitted",
            Box::new(|p| {
                p.held_out_and_parity_rules
                    .sequential_same_registry_looks
                    .reject_omitted_earlier_looks_field = false;
            }),
        ),
        (
            "held_out.require_feeds_selection_false",
            Box::new(|p| p.held_out_and_parity_rules.require_feeds_selection_false = false),
        ),
        (
            "held_out.r5_parity_failure_disposition",
            Box::new(|p| {
                p.held_out_and_parity_rules.r5_parity_failure_disposition = "ignore".into()
            }),
        ),
        (
            "held_out.reject_unreferenced_score_or_parity_records",
            Box::new(|p| {
                p.held_out_and_parity_rules
                    .reject_unreferenced_score_or_parity_records = false;
            }),
        ),
        (
            "experiment_identity_sha256",
            Box::new(|p| p.experiment_identity_sha256 = UnpinnedOr::Pinned("0".repeat(64))),
        ),
        (
            "grant.authorized_true_with_unpinned_field",
            Box::new(|p| p.roster_and_families.sealed_roster_sha256 = UnpinnedOr::Unpinned),
        ),
        (
            "grant.owner_admission_locator_sentinel_pending",
            Box::new(|p| {
                p.execution_grant.owner_admission_locator = UnpinnedOr::Pinned(" PENDING ".into());
            }),
        ),
        (
            "grant.hosts_empty",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec![])),
        ),
        (
            "grant.hosts_sentinel_unpinned",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec!["UNPINNED".into()])),
        ),
        (
            "grant.hosts_sentinel_tbd",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec![" tbd ".into()])),
        ),
        (
            "grant.cpu_ceiling_0",
            Box::new(|p| p.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(0)),
        ),
        (
            "grant.cpu_ceiling_101",
            Box::new(|p| p.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(101)),
        ),
        (
            "grant.memory_floor_0",
            Box::new(|p| p.execution_grant.memory_floor_free_percent = 0),
        ),
        (
            "grant.memory_floor_101",
            Box::new(|p| p.execution_grant.memory_floor_free_percent = 101),
        ),
        (
            "grant.deadline_sentinel_unpinned",
            Box::new(|p| {
                p.execution_grant.deadline_utc = UnpinnedOr::Pinned("unpinned".into());
            }),
        ),
        (
            "pinned.status_contains_unpinned",
            Box::new(|p| p.status = "PRE_REGISTERED_UNPINNED_CONTRADICTION".into()),
        ),
        (
            "pinned.provenance_binding_status_wrong",
            Box::new(|p| p.provenance_binding_status = PREREG_UNPINNED_BINDING_STATUS.into()),
        ),
    ];

    for (label, mutate) in &prereg_mutations {
        let mut candidate = pinned_prereg.clone();
        mutate(&mut candidate);
        assert_prereg_rejected(label, &candidate);
    }

    // Also test unpinned consistency violations (authorized=false with wrong provenance_binding_status or status):
    let mut bad_unpinned_binding = parsed_committed.clone();
    bad_unpinned_binding.provenance_binding_status = PREREG_PINNED_BINDING_STATUS.to_owned();
    assert_prereg_rejected("unpinned.wrong_binding_status", &bad_unpinned_binding);

    let mut bad_unpinned_status = parsed_committed.clone();
    bad_unpinned_status.status = "PRE_REGISTERED_PINNED_NO_UNPINNED_TOKEN".to_owned();
    assert_prereg_rejected("unpinned.status_missing_unpinned", &bad_unpinned_status);

    // Digest mismatch against plan.prereg_sha256:
    assert!(qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .is_err());

    // 5. Duplicate-key JSON smuggling rejection (both orders, raw and canonical SHA-256):
    let base_pinned_str = String::from_utf8(pinned_bytes.clone()).unwrap();
    let dup_false_then_true = base_pinned_str.replacen(
        "\"authorized\": true",
        "\"authorized\": false,\n    \"authorized\": true",
        1,
    );
    let dup_true_then_false = base_pinned_str.replacen(
        "\"authorized\": true",
        "\"authorized\": true,\n    \"authorized\": false",
        1,
    );
    for (dup_label, dup_str) in [
        ("dup_false_then_true", &dup_false_then_true),
        ("dup_true_then_false", &dup_true_then_false),
    ] {
        let dup_bytes = dup_str.as_bytes();
        for sha in [
            sha256_hex(dup_bytes),
            canonical_sha256(&pinned_prereg).unwrap(),
            canonical_sha256(&parsed_committed).unwrap(),
        ] {
            let mut dup_plan = pinned_plan.clone();
            dup_plan.prereg_sha256 = sha;
            assert!(
                qualify_rounds_with_prereg(
                    RoundQualInputs {
                        plan: &dup_plan,
                        identity: &identity,
                        ratchet_outputs: &ratchets,
                        test_scores: &scores,
                        parity: &parity,
                        parent_of_context: &parent_map,
                    },
                    Some(dup_bytes),
                )
                .is_err(),
                "expected duplicate-key payload `{dup_label}` to be rejected"
            );
        }
    }

    // 6. Ratchet provenance cross-check failures when --prereg is supplied:
    type RatchetMutator = (&'static str, Box<dyn Fn(&mut Vec<Value>)>);
    let ratchet_mutations: Vec<RatchetMutator> = vec![
        (
            "seed_cap_b_mismatch",
            Box::new(|r| r[0]["seed_cap_b"] = json!(4)),
        ),
        (
            "max_conditional_base_kl_missing",
            Box::new(|r| {
                r[0].as_object_mut()
                    .unwrap()
                    .remove("max_conditional_base_kl");
            }),
        ),
        (
            "max_conditional_base_kl_mismatch",
            Box::new(|r| r[0]["max_conditional_base_kl"] = json!(0.19)),
        ),
        (
            "max_new_support_mass_missing",
            Box::new(|r| {
                r[0].as_object_mut().unwrap().remove("max_new_support_mass");
            }),
        ),
        (
            "max_new_support_mass_mismatch",
            Box::new(|r| r[0]["max_new_support_mass"] = json!(0.14)),
        ),
        (
            "measured_tilt_policy_missing",
            Box::new(|r| {
                r[0].as_object_mut().unwrap().remove("measured_tilt_policy");
            }),
        ),
        (
            "measured_tilt_policy_tau_mismatch",
            Box::new(|r| r[0]["measured_tilt_policy"]["temperature_tau"] = json!(0.50)),
        ),
        (
            "rounds_len_mismatch",
            Box::new(|r| {
                r[0]["rounds"].as_array_mut().unwrap().pop();
            }),
        ),
        (
            "round_kl_source_caller_evaluated_rejected_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][1]["kl_source"] = json!("caller_evaluated_leaf_log_probs");
            }),
        ),
        (
            "ordinary_champion_genome_missing_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("ordinary_champion_genome");
            }),
        ),
        (
            "ordinary_champion_genome_sentinel_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][0]["ordinary_champion_genome"] = json!(" UNPINNED ");
            }),
        ),
        (
            "ordinary_search_sha256_invalid",
            Box::new(|r| {
                r[0]["rounds"][2]["ordinary_search_sha256"] = json!("not-64-hex");
            }),
        ),
        (
            "ratchet_observations_sha256_invalid",
            Box::new(|r| {
                r[0]["rounds"][2]["ratchet_observations_sha256"] = json!("not-64-hex");
            }),
        ),
    ];

    for (label, mutate) in &ratchet_mutations {
        let mut bad_ratchets = ratchets.clone();
        mutate(&mut bad_ratchets);
        assert!(
            qualify_rounds_with_prereg(
                RoundQualInputs {
                    plan: &pinned_plan,
                    identity: &identity,
                    ratchet_outputs: &bad_ratchets,
                    test_scores: &scores,
                    parity: &parity,
                    parent_of_context: &parent_map,
                },
                Some(&pinned_bytes),
            )
            .is_err(),
            "expected ratchet provenance mutation `{label}` to be rejected under --prereg"
        );
    }

    // 7. Verify R3 A-M3 & R2 P3-3 backward compatibility on legacy no-`--prereg` calls:
    // (a) `kl_source = "caller_evaluated_leaf_log_probs"` remains accepted by `qualify_rounds` (no `--prereg`).
    let mut caller_kl_ratchets = ratchets.clone();
    caller_kl_ratchets[0]["rounds"][1]["kl_source"] = json!("caller_evaluated_leaf_log_probs");
    let rep_caller_kl_no_prereg = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &caller_kl_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("legacy qualify_rounds without --prereg still accepts caller_evaluated_leaf_log_probs");
    assert_eq!(rep_caller_kl_no_prereg.n_unmeasured_kl_contexts, 0);

    // (b) Legacy receipt omitting `ordinary_champion_genome`:
    //     - deserializes cleanly into `SelfPlayRoundReceipt` (`ordinary_champion_genome == ""`),
    //     - is rejected on `Endpoint::RatchetVsOrdinaryFinalRound` when missing on final round,
    //     - is accepted on `Endpoint::FinalVsRound0Incumbent` without `--prereg`.
    let mut legacy_no_ord_ratchets = ratchets.clone();
    for r in &mut legacy_no_ord_ratchets {
        for round_obj in r["rounds"].as_array_mut().unwrap() {
            round_obj
                .as_object_mut()
                .unwrap()
                .remove("ordinary_champion_genome");
        }
    }
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &legacy_no_ord_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    let mut primary_plan = plan.clone();
    primary_plan.endpoint = Endpoint::FinalVsRound0Incumbent;
    let mut primary_scores = Vec::new();
    let mut primary_parity = Vec::new();
    for ctx in ["ctx1", "ctx2", "ctx3", "ctx4"] {
        let g_r0 = format!("g_r0_{ctx}");
        let g_rat_final = format!("g_rat_final_{ctx}");
        let ref_s = make_score(ctx, &g_r0, 1.00, Vec::new());
        let cand_s = make_score(ctx, &g_rat_final, 0.72, vec![ref_s.program_sha256.clone()]);
        primary_scores.push(ref_s);
        primary_scores.push(cand_s);
        for g in [&g_r0, &g_rat_final] {
            primary_parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: (*g).clone(),
                passed: true,
                failure_reason: None,
            });
        }
    }
    let rep_legacy_primary = qualify_rounds(RoundQualInputs {
        plan: &primary_plan,
        identity: &identity,
        ratchet_outputs: &legacy_no_ord_ratchets,
        test_scores: &primary_scores,
        parity: &primary_parity,
        parent_of_context: &parent_map,
    })
    .expect("legacy receipt without ordinary_champion_genome succeeds on FinalVsRound0Incumbent without --prereg");
    assert_eq!(
        rep_legacy_primary.endpoint,
        Endpoint::FinalVsRound0Incumbent
    );

    // 8. CLI negative cases (`--prereg` missing file, hash mismatch, duplicate JSON key):
    let missing_prereg_out = dir.path().join("report_missing_prereg.json");
    let cli_missing = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_file.to_str().unwrap(),
            "--identity",
            id_file.to_str().unwrap(),
            "--ratchets",
            ratchets_file.to_str().unwrap(),
            "--test-scores",
            scores_file.to_str().unwrap(),
            "--parity",
            parity_file.to_str().unwrap(),
            "--parent-map",
            parents_file.to_str().unwrap(),
            "--prereg",
            dir.path().join("does_not_exist.json").to_str().unwrap(),
            "--out",
            missing_prereg_out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli with missing prereg");
    assert!(!cli_missing.status.success());

    let dup_prereg_file = dir.path().join("dup_prereg.json");
    std::fs::write(&dup_prereg_file, dup_false_then_true.as_bytes()).unwrap();
    let dup_out_file = dir.path().join("report_dup_prereg.json");
    let cli_dup = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_file.to_str().unwrap(),
            "--identity",
            id_file.to_str().unwrap(),
            "--ratchets",
            ratchets_file.to_str().unwrap(),
            "--test-scores",
            scores_file.to_str().unwrap(),
            "--parity",
            parity_file.to_str().unwrap(),
            "--parent-map",
            parents_file.to_str().unwrap(),
            "--prereg",
            dup_prereg_file.to_str().unwrap(),
            "--out",
            dup_out_file.to_str().unwrap(),
        ])
        .output()
        .expect("run cli with duplicate-key prereg");
    assert!(!cli_dup.status.success());
}

#[test]
fn exp_next_02_producer_selfplay_ratchet_to_round_qual_bridge() {
    use jope_learning::clover::expansion::{
        canonical_genome, rows_digest, run_selfplay_ratchet, DiscoveryContext,
        MeasuredTiltPolicyConfig, PolicyProposal, PolicySeeds, PolicySnapshot, RetrievalSeed,
        RetrievalSeeds, ScratchBudget, SelfPlayRatchetRequest, SelfPlayRoundReceipt,
        TrainingRoster,
    };
    use jope_pipeline::prepare::Cut;

    let role_bytes = serde_json::to_vec(&json!({
        "schema": "clover.discovery-family-roles/v1",
        "families": {
            "parent_family_a": "train",
            "parent_family_b": "train"
        }
    }))
    .unwrap();
    let role_sha = sha256_hex(&role_bytes);

    let make_cut = |offset: f64| Cut {
        index: 0,
        train_x: (0..20)
            .map(|i| {
                let u = (i as f64 - 10.0) * 0.25 + offset;
                vec![u, u * u + 0.3 * u.sin()]
            })
            .collect(),
        train_y: (0..20)
            .map(|i| {
                let u = (i as f64 - 10.0) * 0.25 + offset;
                1.5 + 1.2 * u + 0.6 * u * u
            })
            .collect(),
        hold_x: (20..26)
            .map(|i| {
                let u = (i as f64 - 16.0) * 0.25 + offset;
                vec![u, u * u + 0.3 * u.sin()]
            })
            .collect(),
        hold_y: (20..26)
            .map(|i| {
                let u = (i as f64 - 16.0) * 0.25 + offset;
                1.5 + 1.2 * u + 0.6 * u * u
            })
            .collect(),
        hold_indices: (20..26).collect(),
    };

    let cut_a = make_cut(0.0);
    let cut_b = make_cut(0.5);
    let ctx_a = DiscoveryContext {
        context_sha256: sha256_hex(b"bridge-ctx-a"),
        parent_family: "parent_family_a".into(),
        role: "train".into(),
        dataset_sha256: sha256_hex(b"bridge-data-a"),
        cut: 0,
        training_sha256: rows_digest(&cut_a.train_x, &cut_a.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut_a.hold_x, &cut_a.hold_y).unwrap(),
        preprocessing_sha256: sha256_hex(b"bridge-prep"),
        encoder_input_sha256: sha256_hex(&vec![0u8; 2051 * 8]),
    };
    let ctx_b = DiscoveryContext {
        context_sha256: sha256_hex(b"bridge-ctx-b"),
        parent_family: "parent_family_b".into(),
        role: "train".into(),
        dataset_sha256: sha256_hex(b"bridge-data-b"),
        cut: 0,
        training_sha256: rows_digest(&cut_b.train_x, &cut_b.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut_b.hold_x, &cut_b.hold_y).unwrap(),
        preprocessing_sha256: sha256_hex(b"bridge-prep"),
        encoder_input_sha256: sha256_hex(&vec![0u8; 2051 * 8]),
    };
    let roster = TrainingRoster {
        schema: "clover.discovery-roster/v1".into(),
        role_manifest_sha256: role_sha,
        contexts: vec![ctx_a.clone(), ctx_b.clone()],
    };
    let roster_digest = roster.digest().unwrap();

    let seed_g1 = canonical_genome("(pipe X (standardize append=false) (ridge))").unwrap();
    let seed_g2 = canonical_genome("(pipe X (row_stats append=true) (ridge))").unwrap();
    let budgets = vec![
        ScratchBudget {
            evaluations_per_arm: 12,
            seconds_per_arm: 30,
            rng_seed: 20261011,
        },
        ScratchBudget {
            evaluations_per_arm: 12,
            seconds_per_arm: 30,
            rng_seed: 20261012,
        },
    ];
    let tilt_cfg = MeasuredTiltPolicyConfig {
        temperature_tau: 0.25,
        advantage_clip: 2.0,
        ref_new_support_mass: 0.05,
        policy_new_support_cap: 0.12,
        step_size_eta: 0.20,
    };

    let temp = tempfile::tempdir().unwrap();
    let mut produced_ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();
    let mut parent_map = BTreeMap::new();

    for (idx, (ctx, cut)) in [(&ctx_a, &cut_a), (&ctx_b, &cut_b)].into_iter().enumerate() {
        parent_map.insert(ctx.context_sha256.clone(), ctx.parent_family.clone());
        let policy = PolicySeeds {
            schema: "clover.policy-seeds/v1".into(),
            snapshot: PolicySnapshot {
                model_class: "PolicyLM".into(),
                context_width: 2051,
                parameter_sha256: sha256_hex(b"bridge-params"),
                checkpoint_sha256: sha256_hex(b"bridge-ckpt"),
                code_sha256: sha256_hex(b"bridge-code"),
                tokenizer_sha256: sha256_hex(b"bridge-tok"),
                support_sha256: sha256_hex(b"bridge-sup"),
                normalizer_sha256: sha256_hex(b"bridge-norm"),
            },
            context_sha256: ctx.context_sha256.clone(),
            encoder_input_sha256: ctx.encoder_input_sha256.clone(),
            sampling_protocol: "trie_children_temperature_v1".into(),
            batch_protocol_sha256: sha256_hex(b"bridge-batch"),
            temperature: 1.0,
            proposals: vec![PolicyProposal {
                program_sha256: sha256_hex(seed_g1.as_bytes()),
                genome: seed_g1.clone(),
                draw_seed: 1,
                behavior_logprob: -2.0,
            }],
            surrogate_funnel: None,
        };
        let retrieval = RetrievalSeeds {
            seeds: vec![RetrievalSeed {
                genome: seed_g2.clone(),
                source_context_sha256: ctx.context_sha256.clone(),
                measurement_receipt_sha256: sha256_hex(b"bridge-ret-rec"),
            }],
        };
        let out_dir = temp.path().join(format!("ratchet_{idx}"));
        let receipt = run_selfplay_ratchet(
            SelfPlayRatchetRequest {
                roster: &roster,
                role_manifest: &role_bytes,
                context_sha256: &ctx.context_sha256,
                cut,
                initial_policy: &policy,
                initial_retrieval: &retrieval,
                round_budgets: &budgets,
                seed_cap_b: 4,
                max_conditional_base_kl: 0.20,
                max_new_support_mass: 0.15,
                round_leaf_log_prob_fn: None,
                measured_tilt_policy: Some(tilt_cfg),
            },
            &out_dir,
        )
        .unwrap();

        assert_eq!(receipt["max_conditional_base_kl"], json!(0.20));
        assert_eq!(receipt["max_new_support_mass"], json!(0.15));
        assert_eq!(
            receipt["measured_tilt_policy"],
            serde_json::to_value(tilt_cfg).unwrap()
        );

        let round_recs: Vec<SelfPlayRoundReceipt> =
            serde_json::from_value(receipt["rounds"].clone()).unwrap();
        assert_eq!(round_recs.len(), 2);
        let final_round = round_recs.last().unwrap();
        assert!(!final_round.ordinary_champion_genome.is_empty());
        assert!(!final_round.ratchet_champion_genome.is_empty());

        let ref_g = &final_round.ordinary_champion_genome;
        let cand_g = &final_round.ratchet_champion_genome;
        if ref_g == cand_g {
            scores.push(make_score(&ctx.context_sha256, ref_g, 0.80, Vec::new()));
            parity.push(ParityRecord {
                dataset_id: ctx.context_sha256.clone(),
                genome: ref_g.clone(),
                passed: true,
                failure_reason: None,
            });
        } else {
            let ref_s = make_score(&ctx.context_sha256, ref_g, 0.95, Vec::new());
            let cand_s = make_score(
                &ctx.context_sha256,
                cand_g,
                0.70,
                vec![ref_s.program_sha256.clone()],
            );
            scores.push(ref_s);
            scores.push(cand_s);
            for g in [ref_g, cand_g] {
                parity.push(ParityRecord {
                    dataset_id: ctx.context_sha256.clone(),
                    genome: g.clone(),
                    passed: true,
                    failure_reason: None,
                });
            }
        }
        produced_ratchets.push(receipt);
    }

    let mut identity = fixture_identity();
    identity.data_roster_sha256 = roster_digest;
    let mut plan = fixture_plan(&identity);
    plan.min_parent_clusters = 2;
    plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;

    let report = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &produced_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("qualify real run_selfplay_ratchet outputs on RatchetVsOrdinaryFinalRound");

    assert_eq!(report.n_ratchet_contexts, 2);
    assert_eq!(report.n_complete_paired_contexts, 2);
    assert_eq!(report.n_observed_parent_clusters, 2);
    assert_eq!(report.n_unmeasured_kl_contexts, 0);
    assert!(!report.qualified_claim);
    assert!(!report.promotion);
    assert!(!report.l6_claim);
    assert!(report
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));
    assert!(report
        .disqualification_reasons
        .contains(&"unequal_resource_comparison".to_owned()));

    // Also verify end-to-end producer -> qualify_rounds_with_prereg binding using the real
    // `run_selfplay_ratchet` receipts (including their emitted `measured_tilt_policy`,
    // `max_conditional_base_kl`, and `max_new_support_mass` contract fields).
    let prereg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/internal/clover/selfplay-round1-prereg.v1.json");
    let mut bridge_prereg: SelfPlayPrereg =
        serde_json::from_slice(&std::fs::read(&prereg_path).unwrap()).unwrap();
    bridge_prereg.arms_and_rounds.rounds = 2;
    bridge_prereg.arms_and_rounds.seed_cap_b = 4;
    bridge_prereg.sample_size_and_margin.bootstrap_resamples = plan.bootstrap_resamples;
    bridge_prereg.sample_size_and_margin.bootstrap_seed = plan.bootstrap_seed;
    let bridge_prereg_bytes = serde_json::to_vec(&bridge_prereg).unwrap();
    plan.prereg_sha256 = sha256_hex(&bridge_prereg_bytes);

    let report_with_prereg = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &produced_ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&bridge_prereg_bytes),
    )
    .expect("qualify real run_selfplay_ratchet outputs with --prereg binding");
    assert_eq!(
        report_with_prereg.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert!(!report_with_prereg.qualified_claim);
    assert!(!report_with_prereg.promotion);
    assert!(!report_with_prereg.l6_claim);
    assert!(report_with_prereg
        .disqualification_reasons
        .contains(&PREREG_CALLER_ASSERTED_REASON.to_owned()));
    assert!(report_with_prereg
        .disqualification_reasons
        .contains(&"qualified_execution_mode_disabled_offline_verifier_only".to_owned()));
}
