use super::*;
use crate::clover::{Budget, InitialState, SourceIdentity, SCHEMA};
use serde_json::json;

fn sample_identity() -> ExperimentIdentity {
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

fn sample_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
    RoundQualPlan {
        experiment_identity_sha256: identity.sha256().expect("valid identity"),
        prereg_sha256: "9".repeat(64),
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

fn sample_decomposed_kl(admissible: bool) -> Value {
    let conditional_base_kl = if admissible { 0.05 } else { 0.35 };
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

fn sample_ratchet_output(
    identity: &ExperimentIdentity,
    context_id: &str,
    round0_genome: &str,
    final_genome: &str,
    equal_resource: bool,
    equal_attempts: bool,
    kl_frozen: bool,
) -> Value {
    json!({
        "schema": RATCHET_SCHEMA,
        "qualification": "qualified",
        "roster_sha256": identity.data_roster_sha256,
        "context_sha256": context_id,
        "equal_resource_comparison": equal_resource,
        "kl_guard_evaluated": true,
        "any_round_frozen_by_kl_guard": kl_frozen,
        "policy_optimizer_updates": 0,
        "rounds": [
            {
                "round": 0,
                "cumulative_prior_rounds": 0,
                "ordinary_best_rmse": 1.0,
                "ordinary_best_q": 0.5,
                "ratchet_best_rmse": 1.0,
                "ratchet_best_q": 0.5,
                "surplus_q_vs_ordinary": 0.0,
                "ratchet_champion_genome": round0_genome,
                "equal_attempts_in_round": equal_attempts,
                "kl_source": "measured_tilt_leaf_policy",
                "kl_malformed_reason": Value::Null,
                "decomposed_kl": sample_decomposed_kl(true),
                "seed_update_frozen_by_kl_guard": false
            },
            {
                "round": 1,
                "cumulative_prior_rounds": 1,
                "ordinary_best_rmse": 0.95,
                "ordinary_best_q": 0.51,
                "ratchet_best_rmse": 0.70,
                "ratchet_best_q": 0.58,
                "surplus_q_vs_ordinary": 0.07,
                "ratchet_champion_genome": final_genome,
                "equal_attempts_in_round": equal_attempts,
                "kl_source": "measured_tilt_leaf_policy",
                "kl_malformed_reason": Value::Null,
                "decomposed_kl": sample_decomposed_kl(!kl_frozen),
                "seed_update_frozen_by_kl_guard": kl_frozen
            }
        ]
    })
}

fn sample_test_score(
    context_id: &str,
    genome: &str,
    test_rmse: f64,
    earlier_looks: Vec<String>,
) -> QualTestScore {
    QualTestScore {
        schema: TEST_SCORE_SCHEMA_ID.to_owned(),
        tool_version: "0.3.0".to_owned(),
        created_at: "2026-10-16T17:30:00Z".to_owned(),
        split: "test".to_owned(),
        receipt_sha256: "a".repeat(64),
        dataset: ScoredDataset {
            id: context_id.to_owned(),
            dataset_hash: context_id.to_owned(),
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

#[test]
fn clean_synthetic_fixture_passes_caller_asserted_diagnostics_only() {
    let identity = sample_identity();
    let plan = sample_plan(&identity);
    let mut parent_map = BTreeMap::new();
    let mut ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();

    for (ctx, parent, ref_rmse, cand_rmse) in [
        ("ctx1", "parent_a", 1.00, 0.75),
        ("ctx2", "parent_b", 1.10, 0.80),
        ("ctx3", "parent_c", 0.90, 0.65),
        ("ctx4", "parent_c", 1.20, 0.85),
    ] {
        parent_map.insert(ctx.to_owned(), parent.to_owned());
        let g0 = format!("g0_{ctx}");
        let g1 = format!("g1_{ctx}");
        let ref_score = sample_test_score(ctx, &g0, ref_rmse, Vec::new());
        let cand_score =
            sample_test_score(ctx, &g1, cand_rmse, vec![ref_score.program_sha256.clone()]);
        ratchets.push(sample_ratchet_output(
            &identity, ctx, &g0, &g1, true, true, false,
        ));
        scores.push(ref_score);
        scores.push(cand_score);
        parity.push(ParityRecord {
            dataset_id: ctx.to_owned(),
            genome: g0,
            passed: true,
            failure_reason: None,
        });
        parity.push(ParityRecord {
            dataset_id: ctx.to_owned(),
            genome: g1,
            passed: true,
            failure_reason: None,
        });
    }

    let report = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("valid qualification report");

    assert_eq!(report.schema, ROUND_QUAL_SCHEMA);
    assert_eq!(report.n_ratchet_contexts, 4);
    assert_eq!(report.n_complete_paired_contexts, 4);
    assert_eq!(report.n_observed_parent_clusters, 3);
    assert_eq!(report.verdict, Verdict::Greater);
    assert!(report.margin_cleared);
    assert!(report.caller_asserted_diagnostic_checks_passed);
    assert!(!report.sealed_family_boundary_adopted);
    assert!(!report.qualified_claim);
    assert!(!report.promotion);
    assert!(!report.l6_claim);
    assert_eq!(
        report.disqualification_reasons,
        vec![
            "provenance_and_prereg_binding_unimplemented".to_owned(),
            "no_adopted_sealed_family_boundary".to_owned(),
        ]
    );
    assert_eq!(report.canonical_sha256().expect("sha256").len(), 64);
}

#[test]
fn f10_tie_tolerance_and_extreme_positive_rmse_log_difference() {
    let identity = sample_identity();
    let mut plan = sample_plan(&identity);
    plan.min_parent_clusters = 2;
    let mut parent_map = BTreeMap::new();
    parent_map.insert("ctx1".to_owned(), "p1".to_owned());
    parent_map.insert("ctx2".to_owned(), "p2".to_owned());

    let ratchets = vec![
        sample_ratchet_output(&identity, "ctx1", "g0_1", "g1_1", true, true, false),
        sample_ratchet_output(&identity, "ctx2", "g0_2", "g1_2", true, true, false),
    ];
    let s1_ref = sample_test_score("ctx1", "g0_1", 1.0, Vec::new());
    let s1_cand = sample_test_score(
        "ctx1",
        "g1_1",
        1.0 * (1.0 - 5e-13),
        vec![s1_ref.program_sha256.clone()],
    );
    // Extreme positive RMSEs where (c / r) would underflow/overflow if computed via division,
    // whereas ln(c) - ln(r) stays finite.
    let s2_ref = sample_test_score("ctx2", "g0_2", 1e200, Vec::new());
    let s2_cand = sample_test_score("ctx2", "g1_2", 1e-200, vec![s2_ref.program_sha256.clone()]);
    let scores = vec![s1_ref, s1_cand, s2_ref, s2_cand];
    let parity = vec![
        ParityRecord {
            dataset_id: "ctx1".to_owned(),
            genome: "g0_1".to_owned(),
            passed: true,
            failure_reason: None,
        },
        ParityRecord {
            dataset_id: "ctx1".to_owned(),
            genome: "g1_1".to_owned(),
            passed: true,
            failure_reason: None,
        },
        ParityRecord {
            dataset_id: "ctx2".to_owned(),
            genome: "g0_2".to_owned(),
            passed: true,
            failure_reason: None,
        },
        ParityRecord {
            dataset_id: "ctx2".to_owned(),
            genome: "g1_2".to_owned(),
            passed: true,
            failure_reason: None,
        },
    ];

    let report = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");

    assert_eq!(report.contexts[0].tie_within_f10_tolerance, Some(true));
    assert_eq!(report.contexts[0].paired_log_ratio, Some(0.0));
    assert_eq!(report.contexts[0].paired_log_reduction, Some(0.0));
    assert!(report.contexts[1]
        .paired_log_reduction
        .expect("finite reduction")
        .is_finite());
    assert!(report.contexts[1].paired_log_reduction.unwrap() > 900.0);
}
