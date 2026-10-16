//! Integration tests for [`jope_learning::clover::round_qual`] and `jope_pools qualify-rounds`.

use std::collections::BTreeMap;
use std::process::Command;

use jope_learning::canon::sha256_hex;
use jope_learning::clover::round_qual::{
    parse_parent_map, qualify_rounds, Endpoint, ParityRecord, QualTestScore, RoundQualInputs,
    RoundQualPlan, SourceGrade, MULTIPLICITY_ADJUSTMENT_STATUS, PROVENANCE_BINDING_STATUS,
    RATCHET_SCHEMA, ROUND_QUAL_SCHEMA,
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
