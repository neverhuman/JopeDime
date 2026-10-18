//! Synthetic contract goldens for the proposed I41 policy-discovery manifest.
//!
//! These tests pin the written protocol and schema's non-authorizing invariants. They are not
//! a production manifest validator, evidence verifier, scientific qualification, or adapter to
//! `qualify::promotion::decide`.

use serde_json::{json, Value};

const SCHEMA: &str = include_str!("../../../schemas/jope.policy-arena.v2.schema.json");

fn schema() -> Value {
    serde_json::from_str(SCHEMA).expect("the checked-in schema must be valid JSON")
}

fn manifest() -> Value {
    json!({
        "schema": "jope.policy-arena/v2",
        "arena_id": "synthetic-i41-example",
        "status": "proposal_unadopted",
        "execution_authorized": false,
        "authority": {
            "adoption_status": "held",
            "issuer_status": "unknown",
            "adopted_adr": null,
            "roster_status": "proposed"
        },
        "estimand": {
            "id": "policy_discovery_linear_budget_q_auc_difference_v1",
            "primary": "equal_family_mean_paired_q_auc_difference",
            "metric": {
                "id": "synthetic-q-score",
                "version": "1",
                "units": "q-units",
                "higher_is_better": true,
                "definition_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "clock": {
                "id": "cumulative_cpu_seconds",
                "units": "cpu_seconds",
                "horizon": 120.0,
                "normalization": "divide_by_common_horizon_clamp_0_1",
                "integration": "trapezoidal_include_baseline_at_zero_and_horizon"
            },
            "aggregation": "equal_context_seed_within_family_then_equal_family",
            "direction": "positive_challenger_minus_incumbent",
            "practical_margin": 0.01,
            "nonprimary_diagnostics": ["terminal_regularized_log_mse"]
        },
        "comparison": {
            "accepted_parent": artifact("parent"),
            "incumbent_actor": artifact("incumbent"),
            "challenger_actor": artifact("challenger"),
            "candidate_selection": "single_candidate_frozen_before_outer_access",
            "program_delta": contrast("program"),
            "critic_allocation": contrast("critic"),
            "actor_policy_delta": contrast("actor"),
            "fresh_feedback_contrast": contrast("fresh-feedback")
        },
        "controls": {
            "strongest_simple_baseline": true,
            "shuffled_context": true,
            "matched_update_no_fresh_feedback": true,
            "matching_fields": ["update_count", "compute", "task_context_support", "initialization", "ancestry", "resource_clock"]
        },
        "roster": {
            "role": "proposed_outer_confirmation",
            "family_ids": ["family-a", "family-b"],
            "minimum_families": 2,
            "contexts_by_family": {"family-a": ["ctx-1"], "family-b": ["ctx-2"]},
            "seeds_by_family": {"family-a": [1, 2], "family-b": [3]},
            "outer_holdout": true,
            "exposure_audit_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "missingness": {
            "expected_cell_dispositions": ["observed", "failed", "missing"],
            "failed_or_missing_rule": "retain_all_expected_cells_and_mark_primary_inconclusive",
            "retain_denominator": true,
            "imputation": "none",
            "family_deletion": "none"
        },
        "multiplicity": {
            "primary_endpoint_count": 1,
            "primary_contrast_count": 1,
            "efficacy_looks": 1,
            "selection_method": "untouched_outer_family_holdout",
            "candidate_budget_hyperparameter_selection": "development families only; freeze one candidate and budget before outer access",
            "alpha_total": 0.05,
            "alpha_plan": "one final look; no interim efficacy looks",
            "outer_roster_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "outer_reuse": "prohibited_after_confirmation"
        },
        "uncertainty": {
            "family_level_method": "paired family-level percentile interval, implementation pending adoption",
            "resamples": 10000,
            "seed": 41,
            "confidence_level": 0.95,
            "minimum_detectable_effect": 0.02,
            "power_target": 0.8,
            "status": "proposed_unvalidated"
        },
        "evidence_bindings": {
            "parent_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "candidate_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "source_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "build_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "evaluator_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "resource_ledger_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "analysis_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "raw_results_sha256": "5555555555555555555555555555555555555555555555555555555555555555"
        }
    })
}

fn artifact(id: &str) -> Value {
    json!({
        "id": id,
        "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "build_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "initialization_id": "init-1",
        "ancestry_id": "ancestry-1"
    })
}

fn contrast(id: &str) -> Value {
    json!({
        "control_id": id,
        "status": "required",
        "matching_rule_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    })
}

fn equal_family_mean(family_effects: &[Vec<f64>]) -> f64 {
    let means: Vec<f64> = family_effects
        .iter()
        .map(|cells| cells.iter().sum::<f64>() / cells.len() as f64)
        .collect();
    means.iter().sum::<f64>() / means.len() as f64
}

fn proposal_is_non_authorizing(value: &Value) -> bool {
    value["schema"] == "jope.policy-arena/v2"
        && value["status"] == "proposal_unadopted"
        && value["execution_authorized"] == false
        && value["authority"]["adoption_status"] == "held"
}

#[test]
fn schema_is_closed_json_and_has_non_authorizing_constants() {
    let parsed = schema();
    assert_eq!(
        parsed["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(parsed["additionalProperties"], false);
    assert_eq!(
        parsed["properties"]["status"]["const"],
        "proposal_unadopted"
    );
    assert_eq!(parsed["properties"]["execution_authorized"]["const"], false);
}

#[test]
fn synthetic_manifest_binds_one_linear_clock_primary() {
    let proposal = manifest();
    assert!(proposal_is_non_authorizing(&proposal));
    assert_eq!(proposal["estimand"]["clock"]["units"], "cpu_seconds");
    assert_eq!(
        proposal["estimand"]["direction"],
        "positive_challenger_minus_incumbent"
    );
    assert_eq!(proposal["multiplicity"]["primary_endpoint_count"], 1);
    assert_eq!(proposal["multiplicity"]["efficacy_looks"], 1);
    println!(
        "I41_SCHEMA_INSTANCE={}",
        serde_json::to_string(&proposal).unwrap()
    );
}

#[test]
fn unequal_family_and_seed_counts_keep_equal_family_weight() {
    let estimate = equal_family_mean(&[vec![0.2, 0.2, 0.2, 0.2], vec![0.8]]);
    assert!((estimate - 0.5).abs() < 1e-12);
    let pooled = (0.2_f64 * 4.0 + 0.8) / 5.0;
    assert!((pooled - 0.32).abs() < 1e-12);
    assert_ne!(estimate, pooled);
}

#[test]
fn context_and_seed_cells_are_averaged_before_family_weighting() {
    let estimate = equal_family_mean(&[vec![0.0, 0.2, 0.4], vec![0.9]]);
    assert!((estimate - 0.55).abs() < 1e-12);
}

#[test]
fn missing_primary_cell_makes_result_inconclusive_without_shrinking_roster() {
    let proposal = manifest();
    assert_eq!(
        proposal["roster"]["family_ids"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        proposal["missingness"]["failed_or_missing_rule"],
        "retain_all_expected_cells_and_mark_primary_inconclusive"
    );
    assert_eq!(proposal["missingness"]["retain_denominator"], true);
}

#[test]
fn failed_primary_cell_is_not_imputed_as_zero() {
    let proposal = manifest();
    assert_eq!(
        proposal["missingness"]["expected_cell_dispositions"],
        json!(["observed", "failed", "missing"])
    );
    assert_eq!(proposal["missingness"]["imputation"], "none");
}

#[test]
fn family_deletion_is_forbidden_for_incomplete_contexts() {
    assert_eq!(manifest()["missingness"]["family_deletion"], "none");
}

#[test]
fn candidate_budget_selection_is_confined_to_development_or_holdout() {
    let proposal = manifest();
    assert_eq!(
        proposal["multiplicity"]["selection_method"],
        "untouched_outer_family_holdout"
    );
    assert!(
        proposal["multiplicity"]["candidate_budget_hyperparameter_selection"]
            .as_str()
            .unwrap()
            .contains("development families only")
    );
    assert_eq!(proposal["roster"]["outer_holdout"], true);
}

#[test]
fn repeated_confirmatory_use_of_outer_families_is_prohibited() {
    assert_eq!(
        manifest()["multiplicity"]["outer_reuse"],
        "prohibited_after_confirmation"
    );
}

#[test]
fn no_fresh_feedback_contrast_requires_matching_updates() {
    let proposal = manifest();
    assert_eq!(
        proposal["controls"]["matched_update_no_fresh_feedback"],
        true
    );
    assert!(proposal["controls"]["matching_fields"]
        .as_array()
        .unwrap()
        .contains(&json!("update_count")));
    assert_eq!(
        proposal["comparison"]["fresh_feedback_contrast"]["status"],
        "required"
    );
}

#[test]
fn program_critic_actor_and_feedback_contrasts_remain_distinct() {
    let comparison = manifest()["comparison"].clone();
    for key in [
        "program_delta",
        "critic_allocation",
        "actor_policy_delta",
        "fresh_feedback_contrast",
    ] {
        assert!(comparison.get(key).is_some(), "missing contrast {key}");
    }
    assert_ne!(
        comparison["program_delta"]["control_id"],
        comparison["actor_policy_delta"]["control_id"]
    );
}

#[test]
fn absent_or_unverified_authority_never_becomes_adopted_by_schema() {
    let mut proposal = manifest();
    assert!(proposal_is_non_authorizing(&proposal));
    proposal["execution_authorized"] = json!(true);
    assert!(!proposal_is_non_authorizing(&proposal));
    assert_eq!(
        schema()["properties"]["execution_authorized"]["const"],
        false
    );
}

#[test]
fn existing_f10_is_not_relabelled_as_discovery_q_auc() {
    let proposal = manifest();
    assert_eq!(
        proposal["estimand"]["id"],
        "policy_discovery_linear_budget_q_auc_difference_v1"
    );
    assert_ne!(proposal["estimand"]["id"], "f10");
    assert_eq!(proposal["authority"]["adoption_status"], "held");
}

#[test]
fn scientific_power_inputs_are_explicit_but_unvalidated() {
    let uncertainty = manifest()["uncertainty"].clone();
    assert!(uncertainty["minimum_detectable_effect"].as_f64().unwrap() > 0.0);
    assert!(uncertainty["power_target"].as_f64().unwrap() > 0.0);
    assert_eq!(uncertainty["status"], "proposed_unvalidated");
}
