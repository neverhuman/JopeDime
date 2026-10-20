//! Admission, binding, budget and route refusals for the model-build path (SC-DEC-184R JA-F1,
//! JA-F2, JA-F3, SC-DEC-184T).
//!
//! The outer-refit and fold records are sealed: their fields are private, and a record read from JSON
//! carries no provenance, so admission refuses it. The tests that edit a serialised record
//! therefore reach the provenance refusal, not the digest checks behind it. The digest refusals are
//! reached by the genuine-record path that alters report predictions. Records that come back from disk
//! reload through `artifact::verify_reload`, which does not grant admission.
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_score::Predictions;
use jope_pipeline::model_build::automl::expected_plan;
use jope_pipeline::model_build::config::{verify_checkpoint, CheckpointClaim, ReleaseClosure};
use jope_pipeline::model_build::executor::{admit_route, RouteAdmission, RouteBoundary};
use jope_pipeline::model_build::identity::{digest_ids, AdmittedIdentity};
use jope_pipeline::model_build::native::{data_digest, NativeData, NativeFitter};
use jope_pipeline::model_build::oof::AdmittedBundle;
use jope_pipeline::model_build::{
    admit_winner_ensemble, artifact, refit_members, run_automl, run_bakeoff, BudgetCounters,
    CandidateStatus, EvaluationBudget, ExecutionIdentity, OuterRefitRecord, RefitOutcome,
    ReportPredictions, Route, VerifiedBundle,
};
use jope_pipeline::model_build::{AutomlOutput, FoldRecord};
use jope_pipeline::pathway::ThinkingLevel;
use model_build_support::*;
use std::path::PathBuf;
use std::time::Duration;

const SOURCE: &str = "9c9202aa39357bc64758c0550ad509e7541c4bb4";

/// The bake-off over the native ridge head, with each bundle's own verified fold records.
fn ridge_bakeoff(u: &Union, id: &ExecutionIdentity) -> Vec<VerifiedBundle> {
    let fitter = split_fitter(Task::Regression, u, id);
    run_bakeoff(
        &plan(Task::Regression, u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10),
    )
    .unwrap()
    .verified
}

/// The admitted members of the ridge bake-off, admitted from their verified records.
fn admitted_ridge(u: &Union, id: &ExecutionIdentity) -> Vec<AdmittedBundle> {
    let expected = expected_plan(&plan(Task::Regression, u));
    ridge_bakeoff(u, id)
        .into_iter()
        .map(|v| AdmittedBundle::admit(v.bundle, &expected, &v.records).unwrap())
        .collect()
}

/// A real outer refit of the ridge head, with its report predictions and record.
fn refit_ridge(u: &Union, id: &ExecutionIdentity) -> RefitOutcome {
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

/// The record with its serialised form edited, then deserialised.
fn forged(
    record: &OuterRefitRecord,
    edit: impl FnOnce(&mut serde_json::Value),
) -> OuterRefitRecord {
    let mut value = serde_json::to_value(record).unwrap();
    edit(&mut value);
    serde_json::from_value(value).unwrap()
}

/// The identity over the regression union, with the given groups on each side.
fn identity_over(
    u: &Union,
    groups: &[String],
    report_groups: &[String],
) -> anyhow::Result<ExecutionIdentity> {
    let (selection_ids, report) = u.data.row_ids.split_at(SELECTION);
    ExecutionIdentity::new(
        REQUEST,
        &plan(Task::Regression, u).digest(),
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        groups,
        report,
        &u.report_labels,
        report_groups,
    )
}

#[test]
fn a_row_shared_by_the_selection_and_report_sides_is_refused() {
    let u = regression_union();
    let mut report = report_ids(&u);
    report[0] = u.data.row_ids[2].clone();
    let (selection_ids, _) = u.data.row_ids.split_at(SELECTION);
    let result = ExecutionIdentity::new(
        REQUEST,
        &plan(Task::Regression, &u).digest(),
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        &[],
        &report,
        &u.report_labels,
        &[],
    );
    assert!(result.is_err());
}

#[test]
fn a_repeated_report_row_is_refused_at_identity_admission() {
    let u = regression_union();
    let mut report = report_ids(&u);
    report[1] = report[0].clone();
    let (selection_ids, _) = u.data.row_ids.split_at(SELECTION);
    let result = ExecutionIdentity::new(
        REQUEST,
        &plan(Task::Regression, &u).digest(),
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        &[],
        &report,
        &u.report_labels,
        &[],
    );
    assert!(result.is_err());
}

#[test]
fn a_group_shared_by_the_selection_and_report_sides_is_refused() {
    let u = regression_union();
    let groups: Vec<String> = (0..SELECTION).map(|i| format!("g{}", i % 5)).collect();
    let report_groups: Vec<String> = (0..REPORT).map(|i| format!("g{}", i % 5)).collect();
    assert!(identity_over(&u, &groups, &report_groups).is_err());
}

#[test]
fn grouping_that_covers_only_one_side_is_refused() {
    let u = regression_union();
    let groups: Vec<String> = (0..SELECTION).map(|i| format!("g{}", i % 5)).collect();
    assert!(identity_over(&u, &groups, &[]).is_err());
    let disjoint: Vec<String> = (0..REPORT).map(|i| format!("r{i}")).collect();
    assert!(identity_over(&u, &[], &disjoint).is_err());
}

#[test]
fn disjoint_groups_on_both_sides_are_admitted() {
    let u = regression_union();
    let groups: Vec<String> = (0..SELECTION).map(|i| format!("g{}", i % 5)).collect();
    let report_groups: Vec<String> = (0..REPORT).map(|i| format!("r{}", i % 2)).collect();
    assert!(identity_over(&u, &groups, &report_groups).is_ok());
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
        &outcome.verified,
        &mut EvaluationBudget::new(10),
    );
    assert!(result.is_err());
}

#[test]
fn a_genuine_refit_record_with_altered_report_predictions_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let admitted = admitted_ridge(&u, &id);
    let outcome = refit_ridge(&u, &id);
    // The record is genuine; the predictions it is paired with are not the ones it produced.
    let mut altered: ReportPredictions = outcome.report.clone();
    if let Some(Predictions::Values(values)) = altered.per_member.get_mut("rust.ridge") {
        values[0] += 1.0;
    }
    let refused = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &altered,
        &outcome.verified,
        &mut EvaluationBudget::new(10),
    );
    assert!(refused.is_err());
    let genuine = admit_winner_ensemble(
        &admitted,
        &u.selection_labels,
        &id,
        &outcome.report,
        &outcome.verified,
        &mut EvaluationBudget::new(10),
    );
    assert!(genuine.is_ok());
}

/// A bundle whose rows are renamed, admitted against a plan with the same renamed rows, is still
/// refused: the genuine verified fold records name the original rows, and a `VerifiedFold` cannot
/// be rebuilt with new rows from outside the crate.
#[test]
fn a_member_whose_rows_differ_from_its_verified_records_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let mut altered_plan = plan(Task::Regression, &u);
    altered_plan.row_ids[0] = "row-elsewhere".into();
    let mut verified = ridge_bakeoff(&u, &id).remove(0);
    verified.bundle.row_ids[0] = "row-elsewhere".into();
    let result = AdmittedBundle::admit(
        verified.bundle,
        &expected_plan(&altered_plan),
        &verified.records,
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
    assert!(out.verified.is_empty());
}

#[test]
fn an_expired_wall_clock_budget_refuses_before_any_candidate_starts() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let out = run_bakeoff(
        &plan(Task::Regression, &u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &mut EvaluationBudget::new(10).with_wall_limit(Duration::ZERO),
    )
    .unwrap();
    assert!(matches!(
        out.receipt.candidates[0].status,
        CandidateStatus::BudgetLimited
    ));
    assert_eq!(out.receipt.candidates[0].fold_attempts, 0);
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
fn the_production_driver_writes_the_artifact_once_and_never_overwrites_it() {
    let u = regression_union();
    let id = identity(&u);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("model-build-artifacts");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("write-once.json");
    let _ = std::fs::remove_file(&path);
    let run = |output: Option<&std::path::Path>| {
        let fitter = split_fitter(Task::Regression, &u, &id);
        run_automl(
            &plan(Task::Regression, &u),
            &roster_of(Task::Regression, &["rust.ridge"]),
            &fitter,
            &id,
            &report_ids(&u),
            &u.report_labels,
            &mut EvaluationBudget::new(20),
            output,
        )
    };
    let out = run(Some(&path)).unwrap();
    assert_eq!(artifact::read(&path).unwrap(), out.persisted.unwrap());
    assert!(
        run(Some(&path)).is_err(),
        "an existing artifact must not be overwritten"
    );
}

#[test]
fn a_low_thinking_level_cannot_run_the_neural_population() {
    let u = regression_union();
    let id = identity(&u);
    let checkpoint = verified_checkpoint();
    let mut budget = EvaluationBudget::new(1000);
    let result = admit_route(RouteBoundary {
        route: Route::FastNeuralGp,
        level: ThinkingLevel::Low,
        checkpoint: &checkpoint,
        identity: &admitted_identity(&u, &id),
        budget: &mut budget,
    });
    assert!(result.is_err());
    assert_eq!(budget.counters().neural_gp_work_units, 0);
    let _ = u;
}

#[test]
fn an_admitted_route_charges_the_neural_population_and_stays_pending_without_a_runtime() {
    let u = regression_union();
    let id = identity(&u);
    let checkpoint = verified_checkpoint();
    let mut budget = EvaluationBudget::with_limits(BudgetCounters {
        neural_gp_work_units: 80,
        ..BudgetCounters::UNLIMITED
    });
    let admission = admit_route(RouteBoundary {
        route: Route::CombinedRosterWinnerEnsemble,
        level: ThinkingLevel::Medium,
        checkpoint: &checkpoint,
        identity: &admitted_identity(&u, &id),
        budget: &mut budget,
    })
    .unwrap();
    assert_eq!(budget.counters().neural_gp_work_units, 80);
    match admission {
        RouteAdmission::Pending { missing, .. } => {
            assert!(missing.contains(&"loaded neural checkpoint runtime"));
        }
        other => panic!("a route without a loaded runtime must stay pending, got {other:?}"),
    }
}

/// A checkpoint that matches a closure. It is an identity match only; no weights are loaded.
fn verified_checkpoint() -> jope_pipeline::model_build::config::VerifiedNeuralCheckpoint {
    let claim = CheckpointClaim {
        release_id: "native-policy-lm-282-r1".into(),
        source_commit: SOURCE.into(),
        checkpoint_sha256: "a".repeat(64),
        runtime_sha256: "b".repeat(64),
    };
    let closure = ReleaseClosure {
        release_id: claim.release_id.clone(),
        source_commit: claim.source_commit.clone(),
        checkpoint_sha256: claim.checkpoint_sha256.clone(),
        runtime_sha256: claim.runtime_sha256.clone(),
    };
    verify_checkpoint(&claim, Some(&closure), SOURCE).unwrap()
}

/// The selection-side fitter for a plan, built only through the plan-derived factory.
fn fitter_for(
    u: &Union,
    plan: &jope_pipeline::model_build::BakeoffPlan,
    data: NativeData,
    id: &ExecutionIdentity,
) -> anyhow::Result<NativeFitter> {
    NativeFitter::for_plan(plan, data, SELECTION, &u.report_labels, id)
}

#[test]
fn a_row_permutation_is_refused_by_the_plan_factory() {
    let u = regression_union();
    let id = identity(&u);
    let mut swapped: NativeData = u.data.clone();
    swapped.row_ids.swap(0, 1);
    swapped.features.swap(0, 1);
    swapped.targets.swap(0, 1);
    assert!(fitter_for(&u, &plan(Task::Regression, &u), swapped, &id).is_err());
}

#[test]
fn a_changed_selection_label_is_refused_by_the_plan_factory() {
    let u = regression_union();
    let id = identity(&u);
    let mut altered = plan(Task::Regression, &u);
    altered.labels[4] = "12345".into();
    assert!(fitter_for(&u, &altered, u.data.clone(), &id).is_err());
}

#[test]
fn swapped_class_order_is_refused_by_the_plan_factory() {
    let u = binary_union();
    let id = identity(&u);
    let mut swapped = plan(Task::Binary, &u);
    swapped.classes = vec!["yes".to_string(), "no".to_string()];
    assert!(fitter_for(&u, &swapped, u.data.clone(), &id).is_err());
}

/// JA-N1: the identity carries plan B's digest, while the data and labels are plan A's. Construction
/// must refuse it, because the digest was not derived from the data it would fit.
#[test]
fn a_plan_digest_from_another_plan_is_refused_at_construction() {
    let u = regression_union();
    let plan_a = plan(Task::Regression, &u);
    let mut plan_b = plan_a.clone();
    plan_b.folds = plan_b.folds.iter().map(|f| (f + 1) % 5).collect();
    let (selection_ids, report_ids) = u.data.row_ids.split_at(SELECTION);
    let identity_b = ExecutionIdentity::new(
        REQUEST,
        &plan_b.digest(),
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        &[],
        report_ids,
        &u.report_labels,
        &[],
    )
    .unwrap();
    assert!(fitter_for(&u, &plan_a, u.data.clone(), &identity_b).is_err());
}

/// JA-N1: the plan's labels disagree with the targets they claim to describe.
#[test]
fn plan_labels_that_differ_from_the_targets_are_refused_at_construction() {
    let u = regression_union();
    let mut plan_b = plan(Task::Regression, &u);
    plan_b.labels[0] = "999".into();
    let (selection_ids, report_ids) = u.data.row_ids.split_at(SELECTION);
    let identity_b = ExecutionIdentity::new(
        REQUEST,
        &plan_b.digest(),
        &data_digest(&u.data),
        selection_ids,
        &plan_b.labels,
        &[],
        report_ids,
        &u.report_labels,
        &[],
    )
    .unwrap();
    assert!(fitter_for(&u, &plan_b, u.data.clone(), &identity_b).is_err());
}

/// JA-N2: every short or ragged array is a typed refusal, not a slice panic.
#[test]
fn a_short_target_array_is_a_typed_refusal() {
    let u = regression_union();
    let id = identity(&u);
    let mut data = u.data.clone();
    data.targets.pop();
    assert!(fitter_for(&u, &plan(Task::Regression, &u), data, &id).is_err());
}

#[test]
fn a_short_feature_array_is_a_typed_refusal() {
    let u = regression_union();
    let id = identity(&u);
    let mut data = u.data.clone();
    data.features.pop();
    assert!(fitter_for(&u, &plan(Task::Regression, &u), data, &id).is_err());
}

#[test]
fn a_ragged_feature_row_is_a_typed_refusal() {
    let u = regression_union();
    let id = identity(&u);
    let mut data = u.data.clone();
    data.features[3].push(1.0);
    assert!(fitter_for(&u, &plan(Task::Regression, &u), data, &id).is_err());
}

#[test]
fn split_boundaries_at_zero_and_at_the_full_length_are_refused() {
    let u = regression_union();
    let id = identity(&u);
    let p = plan(Task::Regression, &u);
    let at_zero = NativeFitter::for_plan(&p, u.data.clone(), 0, &u.report_labels, &id);
    assert!(at_zero.is_err());
    let at_full = NativeFitter::for_plan(
        &p,
        u.data.clone(),
        u.data.row_ids.len(),
        &u.report_labels,
        &id,
    );
    assert!(at_full.is_err());
}

/// SC-DEC-184V item 3: each of the four independent (automl, ensemble) combinations is admitted
/// through the boundary, maps to its own route and stage list, and stays pending on the neural
/// stage. Low thinking refuses all four.
#[test]
fn all_four_automl_ensemble_combinations_pass_the_boundary() {
    use jope_pipeline::model_build::executor::{admit_combination, RouteAdmission, Stage};
    use jope_pipeline::model_build::{route_of, EvaluationBudget as Budget};
    let u = regression_union();
    let id = identity(&u);
    let checkpoint = verified_checkpoint();
    let cases = [
        (false, false, vec![Stage::NeuralToGp]),
        (true, false, vec![Stage::NeuralToGp, Stage::NativeBakeoff]),
        (false, true, vec![Stage::NeuralToGp, Stage::WinnerEnsemble]),
        (
            true,
            true,
            vec![
                Stage::NeuralToGp,
                Stage::NativeBakeoff,
                Stage::WinnerEnsemble,
            ],
        ),
    ];
    let mut routes = Vec::new();
    for (automl, ensemble, expected) in cases {
        routes.push(route_of(automl, ensemble));
        let admission = admit_combination(
            automl,
            ensemble,
            ThinkingLevel::Medium,
            &checkpoint,
            &admitted_identity(&u, &id),
            &mut Budget::with_limits(BudgetCounters {
                neural_gp_work_units: 80,
                ..BudgetCounters::UNLIMITED
            }),
        )
        .unwrap();
        match admission {
            RouteAdmission::Pending { stages, .. } => assert_eq!(stages, expected),
            other => panic!("({automl}, {ensemble}) must stay pending, got {other:?}"),
        }
        assert!(admit_combination(
            automl,
            ensemble,
            ThinkingLevel::Low,
            &checkpoint,
            &admitted_identity(&u, &id),
            &mut Budget::new(1000),
        )
        .is_err());
    }
    routes.sort_by_key(|r| format!("{r:?}"));
    routes.dedup();
    assert_eq!(
        routes.len(),
        4,
        "the four combinations must map to four routes"
    );
}

/// The identity checked against the regression plan and data, as the route boundary requires.
fn admitted_identity(u: &Union, id: &ExecutionIdentity) -> AdmittedIdentity {
    AdmittedIdentity::admit(
        id.clone(),
        &plan(Task::Regression, u),
        &u.data,
        SELECTION,
        &u.report_labels,
    )
    .unwrap()
}

/// C05 physical starts: every fold start is measured from the fit that ran, and a second start on
/// the same fold is refused.
#[test]
fn a_fold_start_is_measured_once_and_a_repeat_start_is_refused() {
    use jope_pipeline::model_build::{CandidateFitter, FitEvidence};
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let spec = roster_of(Task::Regression, &["rust.ridge"]).remove(0);
    let train: Vec<usize> = (0..SELECTION).filter(|i| i % 5 != 0).collect();
    let predict: Vec<usize> = (0..SELECTION).filter(|i| i % 5 == 0).collect();
    let first = fitter.fit_predict(&spec, &train, &predict).unwrap();
    assert!(matches!(
        first.evidence,
        FitEvidence::Verified {
            physical_starts: 1,
            ..
        }
    ));
    assert!(fitter.fit_predict(&spec, &train, &predict).is_err());
}

/// The genuine run over the regression union, without an output path.
fn genuine_run(u: &Union, id: &ExecutionIdentity, fitter: &NativeFitter) -> AutomlOutput {
    run_automl(
        &plan(Task::Regression, u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        fitter,
        id,
        &report_ids(u),
        &u.report_labels,
        &mut EvaluationBudget::new(20),
        None,
    )
    .unwrap()
}

/// Restores each member from its state bytes and predicts the report rows.
fn restorer(fitter: &NativeFitter) -> impl Fn(&str, &[u8]) -> anyhow::Result<Predictions> + '_ {
    move |_id, bytes| {
        let model = NativeFitter::restore_state(bytes)?;
        Ok(Predictions::Values(
            fitter.predict_rows(&model, &report_rows())?,
        ))
    }
}

/// Reloads the genuine persisted ensemble after editing its refit record's serialised form. The
/// reload runs the content checks, so each forgery must be refused by them. Returns the refusal.
fn reload_after_edit(
    u: &Union,
    id: &ExecutionIdentity,
    edit: impl FnOnce(&mut serde_json::Value),
) -> String {
    let fitter = split_fitter(Task::Regression, u, id);
    let out = genuine_run(u, id, &fitter);
    let mut persisted = out.persisted.clone().unwrap();
    persisted.refit = forged(&persisted.refit, edit);
    let report = out.report.clone().unwrap();
    let result = artifact::verify_reload(
        &persisted,
        id,
        &admitted_ridge(u, id),
        &u.selection_labels,
        &report,
        &mut EvaluationBudget::new(20),
        &restorer(&fitter),
    );
    let refusal = result.expect_err("a forged refit record must not reload");
    refusal.to_string()
}

#[test]
fn an_outer_refit_trained_on_other_selection_rows_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let message = reload_after_edit(&u, &id, |v| {
        v["train_rows_sha256"] = digest_ids(&u.data.row_ids[..SELECTION - 1]).into();
    });
    assert!(message.contains("trained on rows"), "{message}");
}

#[test]
fn an_outer_refit_that_predicted_other_report_rows_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let mut shuffled = report_ids(&u);
    shuffled.reverse();
    let message = reload_after_edit(&u, &id, |v| {
        v["predict_rows_sha256"] = digest_ids(&shuffled).into();
    });
    assert!(message.contains("predicted rows"), "{message}");
}

#[test]
fn an_outer_refit_under_another_plan_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let message = reload_after_edit(&u, &id, |v| {
        v["plan_sha256"] = "4".repeat(64).into();
    });
    assert!(message.contains("different plan"), "{message}");
}

#[test]
fn an_outer_refit_that_does_not_cover_the_report_members_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let message = reload_after_edit(&u, &id, |v| {
        v["members"] = serde_json::json!({});
    });
    assert!(message.contains("cover"), "{message}");
}

#[test]
fn a_state_that_does_not_hash_to_its_refit_record_is_refused() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let persisted = genuine_run(&u, &id, &fitter).persisted.unwrap();
    let mut states = persisted.member_states.clone();
    states.get_mut("rust.ridge").unwrap().push(b' ');
    assert!(persisted.refit.verify_states(&states).is_err());
    assert!(persisted
        .refit
        .verify_states(&persisted.member_states)
        .is_ok());
}

/// A serialised outer-refit record reloads, and its content is unchanged. Admission takes a
/// `VerifiedRefit`, which only `refit_members` produces, so a deserialised record has no path to
/// admission. That is a compile-time guarantee, checked by the `compile_fail` examples on
/// `VerifiedRefit`.
#[test]
fn a_serialised_outer_refit_record_reloads_without_gaining_admission() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let out = genuine_run(&u, &id, &fitter);
    let mut persisted = out.persisted.clone().unwrap();
    let json = serde_json::to_string(&persisted.refit).unwrap();
    let decoded: OuterRefitRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, persisted.refit, "the round trip keeps the content");
    persisted.refit = decoded;
    artifact::verify_reload(
        &persisted,
        &id,
        &admitted_ridge(&u, &id),
        &u.selection_labels,
        out.report.as_ref().unwrap(),
        &mut EvaluationBudget::new(20),
        &restorer(&fitter),
    )
    .unwrap();
}

/// A serialised fold record keeps its content. Bundle admission takes a `VerifiedFold`, which only
/// a verified fit produces, so a deserialised record has no path to admission. The
/// `compile_fail` examples on `VerifiedFold` check that at compile time.
#[test]
fn a_serialised_fold_record_keeps_its_content_and_has_no_admission_path() {
    let u = regression_union();
    let id = identity(&u);
    let expected = expected_plan(&plan(Task::Regression, &u));
    let verified = ridge_bakeoff(&u, &id).remove(0);
    let decoded: Vec<FoldRecord> = verified
        .records
        .iter()
        .map(|v| serde_json::from_str(&serde_json::to_string(v.record()).unwrap()).unwrap())
        .collect();
    let original: Vec<FoldRecord> = verified
        .records
        .iter()
        .map(|v| v.record().clone())
        .collect();
    assert_eq!(decoded, original, "the round trip keeps the content");
    assert!(AdmittedBundle::admit(verified.bundle.clone(), &expected, &verified.records).is_ok());
}

/// A record read back from disk reloads and verifies against its persisted states.
#[test]
fn a_record_read_back_from_disk_reloads() {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("model-build-reload");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("reload.json");
    let _ = std::fs::remove_file(&path);
    let out = run_automl(
        &plan(Task::Regression, &u),
        &roster_of(Task::Regression, &["rust.ridge"]),
        &fitter,
        &id,
        &report_ids(&u),
        &u.report_labels,
        &mut EvaluationBudget::new(20),
        Some(&path),
    )
    .unwrap();
    let read_back = artifact::read(&path).unwrap();
    assert_eq!(read_back, out.persisted.clone().unwrap());
    artifact::verify_reload(
        &read_back,
        &id,
        &admitted_ridge(&u, &id),
        &u.selection_labels,
        out.report.as_ref().unwrap(),
        &mut EvaluationBudget::new(20),
        &restorer(&fitter),
    )
    .unwrap();
}

/// A fold record edited in serialised form is refused by the bundle it claims to belong to. The
/// edit is made to the JSON, the record is read back, and validation runs against the bundle.
#[test]
fn a_tampered_serialised_fold_record_is_refused_by_validation() {
    let u = regression_union();
    let id = identity(&u);
    let expected = expected_plan(&plan(Task::Regression, &u));
    let verified = ridge_bakeoff(&u, &id).remove(0);
    let edits: Vec<fn(&mut serde_json::Value)> = vec![
        // A row that was scored moves into the fitting side: the fold would train on its own rows.
        |v| {
            let scored = v["scoring_row_ids"][0].clone();
            v["fitting_row_ids"].as_array_mut().unwrap().push(scored);
        },
        // The recorded artifact no longer matches the bundle's artifact for that fold.
        |v| v["fitted_artifact_sha256"] = "f".repeat(64).into(),
        // The record claims a different request.
        |v| v["request_sha256"] = "e".repeat(64).into(),
        // The record claims a status that is not verified.
        |v| v["status"] = serde_json::json!("unknown"),
    ];
    for (index, edit) in edits.into_iter().enumerate() {
        let tampered: Vec<FoldRecord> = verified
            .records
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut value = serde_json::to_value(v.record()).unwrap();
                if i == 0 {
                    edit(&mut value);
                }
                serde_json::from_value(value).unwrap()
            })
            .collect();
        assert!(
            verified
                .bundle
                .validate_against(&expected, &tampered)
                .is_err(),
            "tampering {index} was not refused"
        );
    }
    assert!(verified
        .bundle
        .validate_against(
            &expected,
            &verified
                .records
                .iter()
                .map(|v| v.record().clone())
                .collect::<Vec<_>>()
        )
        .is_ok());
    // The genuine VerifiedFold records reach admission. The deserialised FoldRecords above never
    // became VerifiedFold values, so they had no route to AdmittedBundle::admit.
    assert!(AdmittedBundle::admit(verified.bundle.clone(), &expected, &verified.records).is_ok());
}
