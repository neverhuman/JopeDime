//! Winner-ensemble rules over admitted bundles (SC-DEC-184M C06).
//!
//! The native bindings cover one regression head, so a multi-member ensemble cannot be fitted
//! here. These tests use synthetic bundles with known out-of-fold predictions. They exercise the
//! rules that do not depend on which head produced a bundle: weights from the selection side
//! only, identity refusals, admission against verified fold records, and the budget.
//! Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_score::Predictions;
use jope_pipeline::model_build::automl::{expected_plan, fold_records};
use jope_pipeline::model_build::oof::{AdmittedBundle, OofPredictionBundle};
use jope_pipeline::model_build::{
    admit_winner_ensemble, BudgetCounters, EnsembleDisposition, EvaluationBudget,
    ExecutionIdentity, FoldStatus, MemberRefit, OuterRefitRecord, ReportPredictions,
    WinnerEnsemble, APPROVED_FOLD_COUNT, OOF_SCHEMA,
};
use model_build_support::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn selection_targets(u: &Union) -> Vec<f64> {
    u.data.targets[..SELECTION].to_vec()
}

/// A bundle over the selection rows with the given predictions, in plan row order.
fn bundle(
    id: &str,
    predictions: Vec<f64>,
    request: &str,
    plan_folds: &[u32],
) -> OofPredictionBundle {
    let u = regression_union();
    OofPredictionBundle {
        schema: OOF_SCHEMA.into(),
        task: Task::Regression,
        candidate_id: id.into(),
        request_sha256: request.into(),
        fitted_artifact_sha256: (0..APPROVED_FOLD_COUNT)
            .map(|f| sha(&format!("{id}:{f}")))
            .collect(),
        row_ids: u.data.row_ids[..SELECTION].to_vec(),
        folds: plan_folds.to_vec(),
        group_ids: vec![],
        classes: vec![],
        predictions: Predictions::Values(predictions),
        failed_folds: vec![],
    }
}

fn plan_folds() -> Vec<u32> {
    (0..SELECTION)
        .map(|i| (i % APPROVED_FOLD_COUNT) as u32)
        .collect()
}

/// Admits a bundle against its own plan and verified fold records.
fn admit(bundle: OofPredictionBundle, u: &Union) -> AdmittedBundle {
    let mut plan = plan(Task::Regression, u);
    plan.request_sha256 = bundle.request_sha256.clone();
    let expected = expected_plan(&plan);
    let records = fold_records(&bundle);
    AdmittedBundle::admit(bundle, &expected, &records).unwrap()
}

/// Two members whose errors cancel: `a` is `y + 0.5 s` and `b` is `y - 0.5 s` for alternating
/// signs `s`. Their equal blend is exact, so a blend beats either alone.
fn complementary_pair(u: &Union) -> (Vec<AdmittedBundle>, Vec<f64>) {
    let y = selection_targets(u);
    let s: Vec<f64> = (0..SELECTION)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let a: Vec<f64> = y.iter().zip(&s).map(|(v, e)| v + 0.5 * e).collect();
    let b: Vec<f64> = y.iter().zip(&s).map(|(v, e)| v - 0.5 * e).collect();
    let admitted = vec![
        admit(bundle("a", a, REQUEST, &plan_folds()), u),
        admit(bundle("b", b.clone(), REQUEST, &plan_folds()), u),
    ];
    (admitted, b)
}

fn report_for(u: &Union, members: &[(&str, Vec<f64>)]) -> ReportPredictions {
    ReportPredictions {
        ids: report_ids(u),
        labels: u.report_labels.clone(),
        per_member: members
            .iter()
            .map(|(id, values)| (id.to_string(), Predictions::Values(values.clone())))
            .collect::<BTreeMap<_, _>>(),
    }
}

fn report_targets(u: &Union) -> Vec<f64> {
    u.data.targets[SELECTION..].to_vec()
}

/// The outer refit record a real refit would produce for these members: each member hashed to a
/// placeholder state, trained on the selection side and predicting the report side.
fn refit_for(
    identity: &ExecutionIdentity,
    u: &Union,
    report: &ReportPredictions,
) -> OuterRefitRecord {
    let members = report
        .per_member
        .keys()
        .map(|id| {
            (
                id.clone(),
                MemberRefit {
                    state_sha256: sha(id),
                    train_rows: SELECTION,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    OuterRefitRecord::new(identity, &u.data.row_ids[..SELECTION], &report.ids, members)
}

fn build(
    admitted: &[AdmittedBundle],
    u: &Union,
    identity: &ExecutionIdentity,
    report: &ReportPredictions,
    budget_cap: u64,
) -> anyhow::Result<WinnerEnsemble> {
    admit_winner_ensemble(
        admitted,
        &u.selection_labels,
        identity,
        report,
        &refit_for(identity, u, report),
        &mut EvaluationBudget::with_limits(BudgetCounters {
            ensemble_attempts: budget_cap,
            ..BudgetCounters::UNLIMITED
        }),
    )
}

#[test]
fn weights_are_chosen_from_the_selection_side_and_ignore_the_report_side() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let report = report_for(
        &u,
        &[
            ("a", report_targets(&u).iter().map(|v| v + 0.5).collect()),
            ("b", report_targets(&u).iter().map(|v| v - 0.5).collect()),
        ],
    );
    // Same selection side, a report side that is far off: the weights must not move.
    let shifted = report_for(
        &u,
        &[
            ("a", report_targets(&u).iter().map(|v| v + 1000.0).collect()),
            ("b", report_targets(&u).iter().map(|v| v - 1000.0).collect()),
        ],
    );
    let first = build(&admitted, &u, &id, &report, 10).unwrap();
    let second = build(&admitted, &u, &id, &shifted, 10).unwrap();
    assert_eq!(first.counts, second.counts);
    assert_eq!(
        first.selection_loss.to_bits(),
        second.selection_loss.to_bits()
    );
    assert_eq!(first.best_single, second.best_single);
    assert_ne!(first.report_loss, second.report_loss);
    assert_eq!(first.report_loss_basis, "outer_refit_predictions");
}

#[test]
fn report_labels_under_a_different_identity_do_not_change_the_weights() {
    let u = regression_union();
    let (admitted, _) = complementary_pair(&u);
    let id = identity(&u);
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    let relabelled: Vec<String> = u.report_labels.iter().map(|_| "999".into()).collect();
    let other_id = identity_with_report_labels(&u, &relabelled);
    let other = ReportPredictions {
        labels: relabelled,
        ..report.clone()
    };
    let first = build(&admitted, &u, &id, &report, 10).unwrap();
    let second = build(&admitted, &u, &other_id, &other, 10).unwrap();
    assert_eq!(first.counts, second.counts);
    assert_eq!(
        first.selection_loss.to_bits(),
        second.selection_loss.to_bits()
    );
}

#[test]
fn a_blend_is_chosen_when_it_beats_the_best_single_member_on_the_selection_side() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    let ensemble = build(&admitted, &u, &id, &report, 10).unwrap();
    assert!(matches!(ensemble.disposition, EnsembleDisposition::Blend));
    assert!(ensemble.counts.values().all(|c| *c > 0));
    assert!(
        ensemble.selection_loss < 1e-12,
        "{}",
        ensemble.selection_loss
    );
}

#[test]
fn a_blend_that_does_not_beat_the_best_member_falls_back_to_it() {
    let u = regression_union();
    let id = identity(&u);
    let y = selection_targets(&u);
    let noisy: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.3 * ((i % 3) as f64 - 1.0))
        .collect();
    let admitted = vec![
        admit(bundle("a", y.clone(), REQUEST, &plan_folds()), &u),
        admit(bundle("b", noisy, REQUEST, &plan_folds()), &u),
    ];
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    let ensemble = build(&admitted, &u, &id, &report, 10).unwrap();
    assert!(matches!(
        ensemble.disposition,
        EnsembleDisposition::BestSingleFallback { .. }
    ));
    assert_eq!(ensemble.best_single.as_deref(), Some("a"));
    assert_eq!(ensemble.counts["a"], 1);
    assert_eq!(ensemble.counts["b"], 0);
}

#[test]
fn a_single_member_is_not_an_ensemble() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = vec![admit(
        bundle("a", selection_targets(&u), REQUEST, &plan_folds()),
        &u,
    )];
    let report = report_for(&u, &[("a", report_targets(&u))]);
    let ensemble = build(&admitted, &u, &id, &report, 10).unwrap();
    assert!(matches!(
        ensemble.disposition,
        EnsembleDisposition::NotAttempted { .. }
    ));
}

#[test]
fn a_member_from_a_different_request_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let other = "3333333333333333333333333333333333333333333333333333333333333333";
    let (mut admitted, _) = complementary_pair(&u);
    admitted[1] = admit(bundle("b", selection_targets(&u), other, &plan_folds()), &u);
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    assert!(build(&admitted, &u, &id, &report, 10).is_err());
}

#[test]
fn a_member_with_a_different_group_identity_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let folds = plan_folds();
    let groups: Vec<String> = folds.iter().map(|f| format!("g{f}")).collect();
    let mut grouped = bundle("a", selection_targets(&u), REQUEST, &folds);
    grouped.group_ids = groups.clone();
    let mut plan_with_groups = plan(Task::Regression, &u);
    plan_with_groups.group_ids = groups;
    let expected = expected_plan(&plan_with_groups);
    let records = fold_records(&grouped);
    let admitted = vec![AdmittedBundle::admit(grouped, &expected, &records).unwrap()];
    let report = report_for(&u, &[("a", report_targets(&u))]);
    assert!(build(&admitted, &u, &id, &report, 10).is_err());
}

#[test]
fn selection_labels_that_do_not_reproduce_the_cohort_are_refused() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    let mut labels = u.selection_labels.clone();
    labels[0] = "999".into();
    let result = admit_winner_ensemble(
        &admitted,
        &labels,
        &id,
        &report,
        &refit_for(&id, &u, &report),
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn report_labels_that_do_not_reproduce_the_report_identity_are_refused() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let mut report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    report.labels[0] = "999".into();
    assert!(build(&admitted, &u, &id, &report, 10).is_err());
}

#[test]
fn report_predictions_must_cover_exactly_the_members() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let report = report_for(&u, &[("a", report_targets(&u))]);
    assert!(build(&admitted, &u, &id, &report, 10).is_err());
}

#[test]
fn an_exhausted_budget_refuses_the_ensemble_attempt() {
    let u = regression_union();
    let id = identity(&u);
    let (admitted, _) = complementary_pair(&u);
    let report = report_for(&u, &[("a", report_targets(&u)), ("b", report_targets(&u))]);
    assert!(build(&admitted, &u, &id, &report, 0).is_err());
}

#[test]
fn a_fold_that_did_not_verify_cannot_be_admitted() {
    let u = regression_union();
    let b = bundle("a", selection_targets(&u), REQUEST, &plan_folds());
    let plan = plan(Task::Regression, &u);
    let expected = expected_plan(&plan);
    let mut records = fold_records(&b);
    records[0].status = FoldStatus::Failed;
    assert!(AdmittedBundle::admit(b, &expected, &records).is_err());
}

#[test]
fn a_fold_record_that_fitted_on_the_wrong_rows_cannot_be_admitted() {
    let u = regression_union();
    let b = bundle("a", selection_targets(&u), REQUEST, &plan_folds());
    let plan = plan(Task::Regression, &u);
    let expected = expected_plan(&plan);
    let mut records = fold_records(&b);
    // Fold 0 scored its own rows, so a record that fitted on them is leakage.
    let leaked = records[0].scoring_row_ids[0].clone();
    records[0].fitting_row_ids.push(leaked);
    assert!(AdmittedBundle::admit(b, &expected, &records).is_err());
}
