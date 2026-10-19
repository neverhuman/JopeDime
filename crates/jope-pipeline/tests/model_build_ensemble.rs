//! Winner-ensemble admission through the public surface (SC-DEC-184M C06, SC-DEC-184R JA-F1).
//!
//! The native bindings cover one regression head, so a multi-member ensemble cannot be fitted
//! through the public API. Multi-member weights, the selection-only rule and the outer-label
//! poison test are covered by the inline tests in `src/model_build/ensemble.rs`, which reach the
//! private core. These tests cover what a caller can reach: a genuine single-member record, and
//! the budget refusal.
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::model_build::automl::expected_plan;
use jope_pipeline::model_build::oof::AdmittedBundle;
use jope_pipeline::model_build::{
    admit_winner_ensemble, refit_members, run_bakeoff, BudgetCounters, EnsembleDisposition,
    EvaluationBudget,
};
use model_build_support::*;

/// The admitted ridge member and a genuine outer refit of it.
fn admitted_and_refit(
    u: &Union,
    id: &jope_pipeline::model_build::ExecutionIdentity,
) -> (
    Vec<AdmittedBundle>,
    jope_pipeline::model_build::RefitOutcome,
) {
    let fitter = split_fitter(Task::Regression, u, id);
    let plan = plan(Task::Regression, u);
    let bakeoff = run_bakeoff(
        &plan,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10),
    )
    .unwrap();
    let expected = expected_plan(&plan);
    let admitted = bakeoff
        .verified
        .into_iter()
        .map(|v| AdmittedBundle::admit(v.bundle, &expected, &v.records).unwrap())
        .collect();
    let refit = refit_members(
        &fitter,
        id,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &report_ids(u),
        &u.report_labels,
        &mut EvaluationBudget::new(10),
    )
    .unwrap();
    (admitted, refit)
}

#[test]
fn a_single_admitted_member_is_not_an_ensemble() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, refit) = admitted_and_refit(&u, &id);
    let ensemble = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &refit.report,
        &refit.record,
        &mut EvaluationBudget::new(10),
    )
    .unwrap();
    assert!(matches!(
        ensemble.disposition,
        EnsembleDisposition::NotAttempted { .. }
    ));
    assert_eq!(ensemble.report_loss_basis, "outer_refit_predictions");
}

#[test]
fn an_exhausted_ensemble_budget_refuses_the_attempt() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, refit) = admitted_and_refit(&u, &id);
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &refit.report,
        &refit.record,
        &mut EvaluationBudget::with_limits(BudgetCounters {
            ensemble_attempts: 0,
            ..BudgetCounters::UNLIMITED
        }),
    );
    assert!(result.is_err());
}
