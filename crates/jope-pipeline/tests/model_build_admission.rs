//! Admission, binding and budget refusals for the model-build path (SC-DEC-184R JA-F1, JA-F2,
//! JA-F3, C06).
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::model_build::automl::{expected_plan, fold_records};
use jope_pipeline::model_build::native::{NativeData, NativeFitter};
use jope_pipeline::model_build::oof::AdmittedBundle;
use jope_pipeline::model_build::{
    admit_winner_ensemble, artifact, refit_members, run_automl, run_bakeoff, BudgetCounters,
    CandidateStatus, EvaluationBudget, ExecutionIdentity, MemberRefit, OuterRefitRecord,
    ReportPredictions,
};
use model_build_support::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The admitted members of a regression bake-off over the native ridge head.
fn admitted_ridge(u: &Union, id: &ExecutionIdentity) -> Vec<AdmittedBundle> {
    let fitter = split_fitter(Task::Regression, u, id);
    let plan = plan(Task::Regression, u);
    let out = run_bakeoff(
        &plan,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10),
    )
    .unwrap();
    let expected = expected_plan(&plan);
    out.bundles
        .into_iter()
        .map(|b| {
            let records = fold_records(&b);
            AdmittedBundle::admit(b, &expected, &records).unwrap()
        })
        .collect()
}

/// A real outer refit of the ridge head, with its report predictions and record.
fn refit_ridge(u: &Union, id: &ExecutionIdentity) -> jope_pipeline::model_build::RefitOutcome {
    let fitter = split_fitter(Task::Regression, u, id);
    refit_members(
        &fitter,
        id,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &report_ids(u),
        &u.report_labels,
        &mut EvaluationBudget::new(10),
    )
    .unwrap()
}

#[test]
fn a_row_shared_by_the_selection_and_report_sides_is_refused() {
    let u = regression_union();
    let sel = u.data.row_ids[..SELECTION].to_vec();
    let mut rep = report_ids(&u);
    rep[0] = sel[2].clone();
    let result = ExecutionIdentity::new(
        REQUEST,
        &plan(Task::Regression, &u).digest(),
        &jope_pipeline::model_build::native::data_digest(&u.data),
        &sel,
        &u.selection_labels,
        &[],
        &rep,
        &u.report_labels,
    );
    assert!(result.is_err());
}

#[test]
fn a_repeated_report_row_is_refused_at_identity_admission() {
    let u = regression_union();
    let sel = u.data.row_ids[..SELECTION].to_vec();
    let mut rep = report_ids(&u);
    rep[1] = rep[0].clone();
    let result = ExecutionIdentity::new(
        REQUEST,
        &plan(Task::Regression, &u).digest(),
        &jope_pipeline::model_build::native::data_digest(&u.data),
        &sel,
        &u.selection_labels,
        &[],
        &rep,
        &u.report_labels,
    );
    assert!(result.is_err());
}

#[test]
fn report_ids_and_labels_of_different_lengths_are_refused_at_ensemble_admission() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    let mut report: ReportPredictions = outcome.report.clone();
    report.labels.pop();
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &report,
        &outcome.record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn an_outer_refit_trained_on_other_selection_rows_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    let record = OuterRefitRecord::new(
        &id,
        &u.data.row_ids[..SELECTION - 1],
        &report_ids(&u),
        outcome.record.members.clone(),
    );
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &outcome.report,
        &record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn an_outer_refit_that_predicted_other_report_rows_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    let mut shuffled = report_ids(&u);
    shuffled.reverse();
    let record = OuterRefitRecord::new(
        &id,
        &u.data.row_ids[..SELECTION],
        &shuffled,
        outcome.record.members.clone(),
    );
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &outcome.report,
        &record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn an_outer_refit_under_another_plan_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    let other_plan = "4".repeat(64);
    let other_identity = ExecutionIdentity::new(
        REQUEST,
        &other_plan,
        &jope_pipeline::model_build::native::data_digest(&u.data),
        &u.data.row_ids[..SELECTION],
        &u.selection_labels,
        &[],
        &report_ids(&u),
        &u.report_labels,
    )
    .unwrap();
    let record = OuterRefitRecord::new(
        &other_identity,
        &u.data.row_ids[..SELECTION],
        &report_ids(&u),
        outcome.record.members.clone(),
    );
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &outcome.report,
        &record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn an_outer_refit_that_does_not_cover_the_report_members_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    let record = OuterRefitRecord::new(
        &id,
        &u.data.row_ids[..SELECTION],
        &report_ids(&u),
        BTreeMap::<String, MemberRefit>::new(),
    );
    let result = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &outcome.report,
        &record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn a_member_whose_rows_differ_from_the_identity_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let outcome = refit_ridge(&u, &id);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let out = run_bakeoff(
        &plan(Task::Regression, &u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10),
    )
    .unwrap();
    // The member and its plan both name a different first row. Admission against that plan
    // succeeds, so only the shared identity can refuse the member.
    let mut altered_plan = plan(Task::Regression, &u);
    altered_plan.row_ids[0] = "row-elsewhere".into();
    let mut bundle = out.bundles[0].clone();
    bundle.row_ids[0] = "row-elsewhere".into();
    let records = fold_records(&bundle);
    let member = AdmittedBundle::admit(bundle, &expected_plan(&altered_plan), &records).unwrap();
    let result = admit_winner_ensemble(
        &[member],
        &u.selection_labels,
        &id,
        &outcome.report,
        &outcome.record,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn a_row_permutation_is_refused_by_the_native_fitter() {
    let u = regression_union();
    let id = identity(&u);
    let mut swapped: NativeData = u.data.clone();
    swapped.row_ids.swap(0, 1);
    swapped.features.swap(0, 1);
    swapped.targets.swap(0, 1);
    let result = NativeFitter::new_split(
        Task::Regression,
        vec![],
        swapped,
        SELECTION,
        &u.selection_labels,
        &u.report_labels,
        &id,
    );
    assert!(result.is_err());
}

#[test]
fn a_changed_selection_label_is_refused_by_the_native_fitter() {
    let u = regression_union();
    let id = identity(&u);
    let mut labels = u.selection_labels.clone();
    labels[4] = "12345".into();
    let result = NativeFitter::new_split(
        Task::Regression,
        vec![],
        u.data.clone(),
        SELECTION,
        &labels,
        &u.report_labels,
        &id,
    );
    assert!(result.is_err());
}

#[test]
fn swapped_class_order_is_refused_by_the_native_fitter() {
    let u = binary_union();
    let id = identity(&u);
    let swapped_classes = vec!["yes".to_string(), "no".to_string()];
    let result = NativeFitter::new_split(
        Task::Binary,
        swapped_classes,
        u.data.clone(),
        SELECTION,
        &u.selection_labels,
        &u.report_labels,
        &id,
    );
    assert!(result.is_err());
}

#[test]
fn a_fitter_bound_to_another_plan_is_refused_by_the_bake_off() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let mut altered = plan(Task::Regression, &u);
    altered.labels[0] = "999".into();
    let result = run_bakeoff(
        &altered,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn the_fold_fit_budget_refuses_the_fold_that_would_exceed_it() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let out = run_bakeoff(
        &plan(Task::Regression, &u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::with_limits(BudgetCounters {
            fold_fits: 3,
            ..BudgetCounters::UNLIMITED
        }),
    )
    .unwrap();
    match &out.receipt.candidates[0].status {
        CandidateStatus::FoldFailed { fold, reason } => {
            assert_eq!(*fold, 3);
            assert!(reason.contains("request budget"), "{reason}");
        }
        other => panic!("expected a budget refusal at fold 3, got {other:?}"),
    }
    assert!(out.bundles.is_empty());
}

#[test]
fn the_outer_refit_budget_refuses_before_any_refit_starts() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let mut budget = EvaluationBudget::with_limits(BudgetCounters {
        outer_refits: 0,
        ..BudgetCounters::UNLIMITED
    });
    let result = refit_members(
        &fitter,
        &id,
        &roster_of(Task::Regression, &["rust.ridge"]),
        &report_ids(&u),
        &u.report_labels,
        &mut budget,
    );
    assert!(result.is_err());
    assert_eq!(budget.counters().outer_refits, 0);
}

#[test]
fn a_state_that_does_not_hash_to_its_refit_record_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let outcome = refit_ridge(&u, &id);
    let mut states = outcome.member_states.clone();
    let state = states.get_mut("rust.ridge").unwrap();
    state.push(b' ');
    assert!(outcome.record.verify_states(&states).is_err());
    assert!(outcome.record.verify_states(&outcome.member_states).is_ok());
}

#[test]
fn a_new_artifact_is_written_once_and_never_overwritten() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let out = run_automl(
        &plan(Task::Regression, &u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &id,
        &report_ids(&u),
        &u.report_labels,
        &mut EvaluationBudget::new(20),
    )
    .unwrap();
    let persisted = out.persisted.unwrap();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("model-build-artifacts");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("write-once.json");
    let _ = std::fs::remove_file(&path);
    artifact::write_new(&path, &persisted).unwrap();
    assert_eq!(artifact::read(&path).unwrap(), persisted);
    assert!(artifact::write_new(&path, &persisted).is_err());
}
