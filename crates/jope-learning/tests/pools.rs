//! Synthetic fixtures for freeze → evaluate. No real data, no network, no GPU.

use jope_learning::canon::sha256_hex;
use jope_learning::pools::bootstrap::Verdict;
use jope_learning::pools::evaluate::{evaluate, fold_of, Config, Inputs, Unavailable, UNIFORM};
use jope_learning::pools::pool::{
    export_views, freeze, verify, Exclusion, FreezeRule, Manifest, MeasurementRow, Pool, PoolView,
    Reference, Status,
};
use jope_learning::pools::selectors::{
    Class, External, ExternalDocument, ExternalScore, Features, Selector, EXTERNAL_SCHEMA,
};

fn h(label: &str) -> String {
    sha256_hex(label.as_bytes())
}

/// Deterministic pseudo-random value in [0, 1) from a label.
fn u(label: &str) -> f64 {
    let d = h(label);
    u64::from_str_radix(&d[..12], 16).unwrap() as f64 / (1u64 << 48) as f64
}

const PROTOCOL: &str = "protocol";

/// `contexts` contexts in `parents` parents, `programs` programs each. A program's true quality is
/// a program effect plus a context-specific effect, so a context-blind prior is informative but
/// not perfect. Every fifth member failed.
fn synthetic_rows(parents: usize, contexts: usize, programs: usize) -> Vec<MeasurementRow> {
    let mut rows = Vec::new();
    for c in 0..contexts {
        let parent = format!("parent-{}", c % parents);
        let context = h(&format!("context-{c}"));
        for p in 0..programs {
            let program = h(&format!("program-{p}"));
            let failed = (c + p) % 5 == 0;
            let q = 0.6 * u(&format!("effect-{p}")) + 0.4 * u(&format!("noise-{c}-{p}"));
            rows.push(MeasurementRow {
                context_id: context.clone(),
                parent_id: parent.clone(),
                role: "train".into(),
                program_sha256: program,
                status: if failed {
                    Status::ProgramFailed
                } else {
                    Status::Complete
                },
                q: (!failed).then_some(q),
                incumbent_q: Some(0.55),
                protocol_sha256: h(PROTOCOL),
                tokens: Some(vec![1, 3 + (p % 7) as u32, 10 + (p % 3) as u32, 2]),
            });
        }
    }
    rows
}

fn rule() -> FreezeRule {
    FreezeRule {
        min_pool: 9,
        max_pool: None,
        salt: "test".into(),
        min_distinct_measured: 2,
        source_grade: "synthetic_fixture".into(),
    }
}

fn cfg() -> Config {
    Config {
        reference: Reference::Incumbent,
        budget: 8,
        folds: 4,
        fold_salt: "folds".into(),
        prior_weight: 2.0,
        knn_k: 4,
        ridge_lambda: 1.0,
        bootstrap_draws: 400,
        bootstrap_seed: 7,
        alpha: 0.05,
    }
}

fn external(
    name: &str,
    class: Class,
    pools: &[Pool],
    score: impl Fn(&Pool, usize) -> f64,
) -> External {
    let scores = pools
        .iter()
        .flat_map(|p| {
            (0..p.members.len()).map(|i| ExternalScore {
                context_id: p.context_id.clone(),
                program_sha256: p.members[i].program_sha256.clone(),
                score: score(p, i),
            })
        })
        .collect();
    External::from_document(&ExternalDocument {
        schema: EXTERNAL_SCHEMA.into(),
        name: name.into(),
        class,
        rule: format!("{name}/test"),
        provenance: serde_json::json!({"fixture": true}),
        scores,
    })
    .unwrap()
}

#[test]
fn freeze_is_deterministic_and_verifiable() {
    selection::freeze_is_deterministic_and_verifiable_body();
}

#[test]
fn tampered_pool_fails_verification() {
    selection::tampered_pool_fails_verification_body();
}

#[test]
fn held_out_roles_are_refused() {
    selection::held_out_roles_are_refused_body();
}

#[test]
fn unknown_and_conflicting_contexts_are_excluded_not_imputed() {
    selection::unknown_and_conflicting_contexts_are_excluded_not_imputed_body();
}

#[test]
fn cap_is_salted_and_selector_independent() {
    selection::cap_is_salted_and_selector_independent_body();
}

#[test]
fn oracle_candidate_is_established_and_constant_candidate_is_not() {
    selection::oracle_candidate_is_established_and_constant_candidate_is_not_body();
}

#[test]
fn beating_uniform_alone_does_not_establish() {
    selection::beating_uniform_alone_does_not_establish_body();
}

#[test]
fn folds_keep_whole_parents_together_and_every_selector_scores_every_context() {
    surrogate::folds_keep_whole_parents_together_and_every_selector_scores_every_context_body();
}

#[test]
fn contextual_surrogate_qhat_establishes_h2_on_interaction_pools() {
    surrogate::contextual_surrogate_qhat_establishes_h2_on_interaction_pools_body();
}

#[test]
fn missing_external_score_is_an_error() {
    surrogate::missing_external_score_is_an_error_body();
}

#[test]
fn pool_floor_reference_needs_no_incumbent_and_derives_token_features() {
    surrogate::pool_floor_reference_needs_no_incumbent_and_derives_token_features_body();
}

#[test]
fn exported_views_carry_no_outcomes() {
    surrogate::exported_views_carry_no_outcomes_body();
}

#[test]
fn reloaded_valid_pools_pass_and_rehashed_invalid_pools_are_refused() {
    validation::reloaded_valid_pools_pass_and_rehashed_invalid_pools_are_refused_body();
}

#[test]
fn high_dim_surrogate_and_distilled_flywheel_establish_multi_round_lift() {
    lift::high_dim_surrogate_and_distilled_flywheel_establish_multi_round_lift_body();
}

#[test]
fn flywheel_isolates_outer_folds_enforces_kl_freeze_and_ingests_s3_prior() {
    flywheel::flywheel_isolates_outer_folds_enforces_kl_freeze_and_ingests_s3_prior_body();
}

mod flywheel;
mod lift;
mod selection;
mod surrogate;
mod validation;
