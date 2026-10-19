//! AutoML driver for the native heads (SC-DEC-184M C05/C06; SC-DEC-184R JA-F1, JA-F3).
//!
//! One request runs in this order, and each step consumes only what the previous steps admitted:
//!
//! 1. the five-fold bake-off on the selection side (`run_bakeoff`), which returns each `Scored`
//!    bundle together with the fold records its own verified fits produced;
//! 2. admission of those bundles against the expected plan (`AdmittedBundle::admit`);
//! 3. the outer refit of the admitted members on the whole selection side, which supplies the
//!    report-side predictions and an `OuterRefitRecord` (`refit_members`);
//! 4. the winner ensemble (`admit_winner_ensemble`). It refuses unless the refit record agrees
//!    with the identity, the report side and the report predictions, and its weights come from
//!    selection-side out-of-fold predictions only;
//! 5. the persisted record, written with `artifact::write_new` when an output path is given.
//!
//! The refit reads no selection-side weights, and the weights read no report predictions. When
//! nothing is admitted there is no ensemble, and the output says so rather than substituting one.
use super::artifact::{self, PersistedEnsemble};
use super::candidates::CandidateSpec;
use super::ensemble::{admit_winner_ensemble, ReportPredictions};
use super::identity::ExecutionIdentity;
use super::native::NativeFitter;
use super::oof::{AdmittedBundle, ExpectedPlan};
use super::refit::refit_members;
use super::selection::{run_bakeoff, BakeoffPlan, BakeoffReceipt, EvaluationBudget};
use anyhow::{ensure, Result};
use std::path::Path;

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

/// Runs the whole AutoML request for the native heads.
///
/// `fitter` must be bound to the union of the plan's selection rows, in plan order, followed by
/// the report rows, with `identity` describing both sides. `report_ids` and `report_labels` are
/// that report side. When `output` is given, a produced record is written there as a new file.
#[allow(clippy::too_many_arguments)]
pub fn run_automl(
    plan: &BakeoffPlan,
    roster: &[CandidateSpec],
    fitter: &NativeFitter,
    identity: &ExecutionIdentity,
    report_ids: &[String],
    report_labels: &[String],
    budget: &mut EvaluationBudget,
    output: Option<&Path>,
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
        .verified
        .iter()
        .map(|verified| {
            AdmittedBundle::admit(verified.bundle.clone(), &expected, &verified.records)
        })
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
        &refit.verified,
        budget,
    )?;
    let persisted = PersistedEnsemble {
        ensemble,
        refit: refit.verified.record().clone(),
        member_states: refit.member_states,
    };
    if let Some(path) = output {
        artifact::write_new(path, &persisted)?;
    }
    Ok(AutomlOutput {
        receipt: outcome.receipt,
        persisted: Some(persisted),
        report: Some(refit.report),
    })
}
