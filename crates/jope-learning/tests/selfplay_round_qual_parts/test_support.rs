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

pub(crate) fn fixture_identity() -> ExperimentIdentity {
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

pub(crate) fn fixture_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
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

pub(crate) fn fixture_decomposed_kl(admissible: bool) -> Value {
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
pub(crate) fn make_ratchet(
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

pub(crate) fn make_score(
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

pub(crate) type FixtureBundle = (
    ExperimentIdentity,
    RoundQualPlan,
    Vec<Value>,
    Vec<QualTestScore>,
    Vec<ParityRecord>,
    BTreeMap<String, String>,
);

pub(crate) fn fixture_bundle() -> FixtureBundle {
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

#[path = "test_support/prereg.rs"]
mod prereg;
pub(crate) use prereg::{prereg_fixture, write_cli_inputs, CliInputs, PreregFixture};
