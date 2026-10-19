//! AutoML request over the native heads (SC-DEC-184M C05/C06).
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_score::Predictions;
use jope_pipeline::model_build::native::{data_digest, NativeFitter};
use jope_pipeline::model_build::selection::NotStarted;
use jope_pipeline::model_build::{
    refit_members, run_automl, BakeoffPlan, BudgetCounters, CandidateFitter, CandidateStatus,
    EvaluationBudget, ExecutionIdentity,
};
use model_build_support::*;

fn run(
    task: Task,
    u: &Union,
    ids: &[&str],
    budget: &mut EvaluationBudget,
) -> jope_pipeline::model_build::AutomlOutput {
    let id = identity(u);
    let fitter = split_fitter(task, u, &id);
    run_automl(
        &plan(task, u),
        &roster_of(task, ids),
        &fitter,
        &id,
        &report_ids(u),
        &u.report_labels,
        budget,
    )
    .unwrap()
}

#[test]
fn regression_request_admits_the_bound_head_and_refuses_the_unbound_one() {
    let u = regression_union();
    let mut budget = EvaluationBudget::new(20);
    let out = run(
        Task::Regression,
        &u,
        &["rust.ridge", "rust.gpr"],
        &mut budget,
    );
    assert_eq!(out.receipt.selected.as_deref(), Some("rust.ridge"));
    let gpr = out
        .receipt
        .candidates
        .iter()
        .find(|c| c.candidate_id == "rust.gpr")
        .unwrap();
    assert!(matches!(
        gpr.status,
        CandidateStatus::FoldFailed { fold: 0, .. }
    ));
    assert_eq!(gpr.physical_starts, Some(0));
    // Two candidates, six fold fits (five for ridge, one refused for gpr), one refit, one ensemble.
    assert_eq!(
        budget.counters(),
        BudgetCounters {
            candidate_evaluations: 2,
            fold_fits: 6,
            outer_refits: 1,
            ensemble_attempts: 1,
        }
    );
    let persisted = out.persisted.expect("the bound head is admitted");
    assert_eq!(
        persisted.member_states.keys().collect::<Vec<_>>(),
        vec!["rust.ridge"]
    );
    let report = out.report.expect("report predictions accompany the record");
    assert_eq!(
        report.per_member.keys().collect::<Vec<_>>(),
        vec!["rust.ridge"]
    );
    assert_eq!(report.ids, report_ids(&u));
    assert!(persisted.ensemble.report_loss.is_finite());
}

#[test]
fn binary_request_produces_probabilities_over_the_frozen_classes() {
    let u = binary_union();
    let mut budget = EvaluationBudget::new(20);
    let out = run(
        Task::Binary,
        &u,
        &["rust.logit_ridge", "rust.tab_boost"],
        &mut budget,
    );
    assert_eq!(out.receipt.selected.as_deref(), Some("rust.logit_ridge"));
    let report = out.report.unwrap();
    match &report.per_member["rust.logit_ridge"] {
        Predictions::Probabilities { classes, values } => {
            assert_eq!(classes, &vec!["no".to_string(), "yes".to_string()]);
            assert_eq!(values.len(), REPORT);
            assert!(values
                .iter()
                .all(|row| (row[0] + row[1] - 1.0).abs() < 1e-9));
        }
        other => panic!("expected probabilities, got {other:?}"),
    }
}

#[test]
fn a_roster_with_no_bound_head_persists_nothing() {
    let u = regression_union();
    let mut budget = EvaluationBudget::new(20);
    let out = run(Task::Regression, &u, &["rust.gpr"], &mut budget);
    assert_eq!(out.receipt.selected, None);
    assert!(out.persisted.is_none());
    assert!(out.report.is_none());
}

#[test]
fn a_plan_whose_rows_differ_from_the_fitter_rows_is_refused_before_any_fit() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let mut plan: BakeoffPlan = plan(Task::Regression, &u);
    plan.row_ids[0] = "row-other".into();
    let result = run_automl(
        &plan,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &id,
        &report_ids(&u),
        &u.report_labels,
        &mut EvaluationBudget::new(20),
    );
    assert!(result.is_err());
}

#[test]
fn training_on_a_report_row_is_refused_before_any_fit_starts() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let spec = roster_of(Task::Regression, &["rust.ridge"]).remove(0);
    let mut rows = selection_rows();
    rows.push(SELECTION);
    let err = fitter.fit_state(&spec, &rows).unwrap_err();
    assert!(err.downcast_ref::<NotStarted>().is_some(), "{err:#}");
}

#[test]
fn an_out_of_range_predict_row_is_a_typed_refusal() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let spec = roster_of(Task::Regression, &["rust.ridge"]).remove(0);
    let err = fitter
        .fit_predict(&spec, &selection_rows(), &[SELECTION + REPORT + 5])
        .unwrap_err();
    assert!(err.downcast_ref::<NotStarted>().is_some(), "{err:#}");
}

#[test]
fn a_split_fitter_refuses_report_labels_that_do_not_reproduce_the_identity() {
    let u = regression_union();
    let id = identity(&u);
    let mut labels = u.report_labels.clone();
    labels[0] = "999".into();
    let result = NativeFitter::new_split(
        Task::Regression,
        vec![],
        u.data.clone(),
        SELECTION,
        &u.selection_labels,
        &labels,
        &id,
    );
    assert!(result.is_err());
}

#[test]
fn a_split_fitter_refuses_union_bytes_that_do_not_reproduce_the_identity() {
    let u = regression_union();
    let id = identity(&u);
    let mut altered = u.data.clone();
    altered.features[SELECTION][0] += 1e-9;
    assert_ne!(data_digest(&altered), id.data_sha256);
    let result = NativeFitter::new_split(
        Task::Regression,
        vec![],
        altered,
        SELECTION,
        &u.selection_labels,
        &u.report_labels,
        &id,
    );
    assert!(result.is_err());
}

#[test]
fn the_outer_refit_refuses_report_ids_the_fitter_was_not_bound_to() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let members = roster_of(Task::Regression, &["rust.ridge"]);
    let mut wrong = report_ids(&u);
    wrong.reverse();
    let result = refit_members(
        &fitter,
        &id,
        &members,
        &wrong,
        &u.report_labels,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn the_outer_refit_is_charged_to_the_budget_and_refuses_when_exhausted() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let members = roster_of(Task::Regression, &["rust.ridge"]);
    let result = refit_members(
        &fitter,
        &id,
        &members,
        &report_ids(&u),
        &u.report_labels,
        &mut EvaluationBudget::with_limits(BudgetCounters {
            outer_refits: 0,
            ..BudgetCounters::UNLIMITED
        }),
    );
    assert!(result.is_err());
}

#[test]
fn the_identity_is_the_one_the_fitter_was_bound_to() {
    let u = regression_union();
    let id: ExecutionIdentity = identity(&u);
    assert_eq!(id.request_sha256, REQUEST);
    assert_eq!(id.protocol, jope_pipeline::model_build::identity::PROTOCOL);
}
