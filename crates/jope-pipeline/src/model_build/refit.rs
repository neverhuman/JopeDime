//! Outer refit of the admitted members (SC-DEC-184M JA-R1, C05).
//!
//! Each member is refitted once on the whole selection side and predicts the report side. Those
//! report predictions are the only source of report losses. The selection side never sees them,
//! and the ensemble weights are chosen before this module runs.
//!
//! Every refit is charged to the request budget before it starts. The fitted state bytes are
//! returned alongside the predictions, so the persisted ensemble can be reloaded in another
//! process and checked against the same predictions (see `artifact::verify_reload`).
use super::candidates::CandidateSpec;
use super::ensemble::ReportPredictions;
use super::native::{to_predictions, NativeFitter};
use super::selection::EvaluationBudget;
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct RefitOutcome {
    pub report: ReportPredictions,
    /// Fitted state bytes per member, exactly as the fitter produced them.
    pub member_states: BTreeMap<String, Vec<u8>>,
}

/// Refits every member on the selection side of `fitter` and predicts its report side.
///
/// `report_ids` and `report_labels` must be the report rows the fitter was bound to; a mismatch
/// is refused before any fit starts.
pub fn refit_members(
    fitter: &NativeFitter,
    members: &[CandidateSpec],
    report_ids: &[String],
    report_labels: &[String],
    budget: &mut EvaluationBudget,
) -> Result<RefitOutcome> {
    ensure!(!members.is_empty(), "no members to refit");
    ensure!(
        report_ids.len() == report_labels.len(),
        "report ids and labels are not aligned"
    );
    let row_ids = fitter.row_ids();
    let selection_len = fitter.selection_len();
    ensure!(
        row_ids[selection_len..] == *report_ids,
        "report ids differ from the rows the fitter was bound to"
    );
    let selection_rows: Vec<usize> = (0..selection_len).collect();
    let report_rows: Vec<usize> = (selection_len..row_ids.len()).collect();

    let mut per_member = BTreeMap::new();
    let mut member_states = BTreeMap::new();
    for spec in members {
        ensure!(
            !per_member.contains_key(spec.id),
            "duplicate member {} in the refit",
            spec.id
        );
        budget
            .charge(1)
            .map_err(|e| anyhow::anyhow!("refit of {} over the request budget: {e:?}", spec.id))?;
        let (model, state) = fitter.fit_state(spec, &selection_rows)?;
        let raw = fitter.predict_rows(&model, &report_rows)?;
        per_member.insert(
            spec.id.to_owned(),
            to_predictions(fitter.task(), fitter.classes(), raw)?,
        );
        member_states.insert(spec.id.to_owned(), state);
    }
    Ok(RefitOutcome {
        report: ReportPredictions {
            ids: report_ids.to_vec(),
            labels: report_labels.to_vec(),
            per_member,
        },
        member_states,
    })
}
