//! AutoML driver for the native heads (SC-DEC-184M C05/C06).
//!
//! One request runs in this order, and each step consumes only what the previous steps admitted:
//!
//! 1. the five-fold bake-off on the selection side (`run_bakeoff`);
//! 2. admission of every `Scored` bundle against fold records derived from the bake-off's own
//!    verified evidence (`AdmittedBundle::admit`);
//! 3. the outer refit of the admitted members on the whole selection side, which supplies the
//!    report-side predictions (`refit_members`);
//! 4. the winner ensemble (`build_winner_ensemble`). Its weights come from selection-side
//!    out-of-fold predictions only; the report predictions it receives are used for the report
//!    loss and nothing else;
//! 5. the persisted record: the ensemble plus the fitted state bytes of every member.
//!
//! The refit reads no selection-side weights, and the weights read no report predictions, so the
//! order between steps 3 and 4 does not let the report side influence the choice. When nothing is
//! admitted there is no ensemble, and the output says so rather than substituting a fallback.
use super::artifact::PersistedEnsemble;
use super::candidates::CandidateSpec;
use super::ensemble::{admit_winner_ensemble, ReportPredictions};
use super::identity::ExecutionIdentity;
use super::native::NativeFitter;
use super::oof::{AdmittedBundle, ExpectedPlan, FoldRecord, FoldStatus, OofPredictionBundle};
use super::refit::refit_members;
use super::selection::{run_bakeoff, BakeoffPlan, BakeoffReceipt, EvaluationBudget};
use anyhow::{ensure, Result};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct AutomlOutput {
    pub receipt: BakeoffReceipt,
    /// `None` when no candidate was admitted, so no ensemble exists to persist.
    pub persisted: Option<PersistedEnsemble>,
    /// The report-side predictions the ensemble was scored against. `None` with `persisted`.
    pub report: Option<ReportPredictions>,
}

/// The admitted plan that a bake-off bundle must reproduce.
pub fn expected_plan(plan: &BakeoffPlan) -> ExpectedPlan {
    ExpectedPlan {
        request_sha256: plan.request_sha256.clone(),
        row_ids: plan.row_ids.clone(),
        folds: plan.folds.clone(),
        group_ids: plan.group_ids.clone(),
        classes: plan.classes.clone(),
    }
}

/// Fold records for a `Scored` bundle. Each fold fitted on every row outside it and scored its
/// own rows. They are derived from the bake-off's status, which `run_bakeoff` grants only when
/// every fold carried verified fit evidence.
pub fn fold_records(bundle: &OofPredictionBundle) -> Vec<FoldRecord> {
    let folds: Vec<u32> = bundle
        .folds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    folds
        .iter()
        .enumerate()
        .map(|(index, &fold)| FoldRecord {
            fold,
            request_sha256: bundle.request_sha256.clone(),
            fitted_artifact_sha256: bundle.fitted_artifact_sha256[index].clone(),
            fitting_row_ids: bundle
                .row_ids
                .iter()
                .zip(&bundle.folds)
                .filter(|(_, f)| **f != fold)
                .map(|(id, _)| id.clone())
                .collect(),
            scoring_row_ids: bundle
                .row_ids
                .iter()
                .zip(&bundle.folds)
                .filter(|(_, f)| **f == fold)
                .map(|(id, _)| id.clone())
                .collect(),
            status: FoldStatus::Verified,
        })
        .collect()
}

/// Runs the whole AutoML request for the native heads.
///
/// `fitter` must be bound to the union of the plan's selection rows, in plan order, followed by
/// the report rows, with `identity` describing both sides. `report_ids` and `report_labels` are
/// that report side.
pub fn run_automl(
    plan: &BakeoffPlan,
    roster: &[CandidateSpec],
    fitter: &NativeFitter,
    identity: &ExecutionIdentity,
    report_ids: &[String],
    report_labels: &[String],
    budget: &mut EvaluationBudget,
) -> Result<AutomlOutput> {
    ensure!(
        identity.plan_sha256 == plan.digest(),
        "the identity was not built for this plan"
    );
    ensure!(
        fitter.selection_len() == plan.row_ids.len()
            && fitter.row_ids()[..plan.row_ids.len()] == plan.row_ids[..],
        "the fitter's selection rows are not the plan's rows"
    );
    let outcome = run_bakeoff(plan, roster, fitter, budget)?;
    let expected = expected_plan(plan);
    let admitted: Vec<AdmittedBundle> = outcome
        .bundles
        .iter()
        .map(|bundle| AdmittedBundle::admit(bundle.clone(), &expected, &fold_records(bundle)))
        .collect::<Result<_>>()?;
    if admitted.is_empty() {
        return Ok(AutomlOutput {
            receipt: outcome.receipt,
            persisted: None,
            report: None,
        });
    }
    let members: Vec<CandidateSpec> = roster
        .iter()
        .filter(|spec| admitted.iter().any(|m| m.bundle().candidate_id == spec.id))
        .cloned()
        .collect();
    let refit = refit_members(
        fitter,
        identity,
        &members,
        report_ids,
        report_labels,
        budget,
    )?;
    let ensemble = admit_winner_ensemble(
        &admitted,
        &plan.labels,
        identity,
        &refit.report,
        &refit.record,
        budget,
    )?;
    Ok(AutomlOutput {
        receipt: outcome.receipt,
        persisted: Some(PersistedEnsemble {
            ensemble,
            refit: refit.record,
            member_states: refit.member_states,
        }),
        report: Some(refit.report),
    })
}
