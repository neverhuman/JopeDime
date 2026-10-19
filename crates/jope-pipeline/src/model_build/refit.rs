//! Outer refit of the admitted members (SC-DEC-184M JA-R1, C05; SC-DEC-184R JA-F1, JA-F3).
//!
//! Each member is refitted once on the whole selection side and predicts the report side. Those
//! report predictions are the only source of report losses. The selection side never sees them,
//! and the ensemble weights are chosen before this module runs.
//!
//! The refit produces an `OuterRefitRecord`: the request, the plan, the selection rows each refit
//! trained on, the report rows each refit predicted, and the sha256 of every member's fitted
//! state. The ensemble accepts report losses only together with this record, and the record is
//! checked against the identity and, at reload, against the persisted state bytes.
use super::candidates::CandidateSpec;
use super::ensemble::ReportPredictions;
use super::identity::{digest_ids, ExecutionIdentity};
use super::native::{to_predictions, NativeFitter};
use super::selection::{BudgetKind, EvaluationBudget};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const REFIT_RECORD_SCHEMA: &str = "jope.outer-refit-record/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberRefit {
    /// sha256 of the member's fitted state bytes.
    pub state_sha256: String,
    /// Selection-side rows the member was refitted on.
    pub train_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterRefitRecord {
    pub schema: String,
    pub request_sha256: String,
    pub plan_sha256: String,
    /// Digest of the selection-side row identifiers every refit trained on.
    pub train_rows_sha256: String,
    /// Digest of the report-side row identifiers every refit predicted.
    pub predict_rows_sha256: String,
    pub members: BTreeMap<String, MemberRefit>,
}

impl OuterRefitRecord {
    /// Builds the record from the identity and the refits that were actually performed. Callers
    /// in production obtain it from `refit_members`. Tests may build one to exercise refusals.
    pub fn new(
        identity: &ExecutionIdentity,
        train_rows: &[String],
        predict_rows: &[String],
        members: BTreeMap<String, MemberRefit>,
    ) -> Self {
        Self {
            schema: REFIT_RECORD_SCHEMA.into(),
            request_sha256: identity.request_sha256.clone(),
            plan_sha256: identity.plan_sha256.clone(),
            train_rows_sha256: digest_ids(train_rows),
            predict_rows_sha256: digest_ids(predict_rows),
            members,
        }
    }

    /// Refuses unless the states cover exactly the recorded members and each state hashes to the
    /// digest the refit recorded.
    pub fn verify_states(&self, states: &BTreeMap<String, Vec<u8>>) -> Result<()> {
        ensure!(
            self.members.keys().eq(states.keys()),
            "persisted states do not cover exactly the refitted members"
        );
        for (id, bytes) in states {
            let digest = format!("{:x}", Sha256::digest(bytes));
            ensure!(
                self.members[id].state_sha256 == digest,
                "fitted state for {id} does not match its refit record"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RefitOutcome {
    pub report: ReportPredictions,
    /// Fitted state bytes per member, exactly as the fitter produced them.
    pub member_states: BTreeMap<String, Vec<u8>>,
    pub record: OuterRefitRecord,
}

/// Refits every member on the selection side of `fitter` and predicts its report side.
///
/// The fitter must be bound to `identity`'s plan, and its rows must be the rows `identity`
/// describes: the selection rows first, then the report rows. Each refit is charged to the
/// budget before it starts.
pub fn refit_members(
    fitter: &NativeFitter,
    identity: &ExecutionIdentity,
    members: &[CandidateSpec],
    report_ids: &[String],
    report_labels: &[String],
    budget: &mut EvaluationBudget,
) -> Result<RefitOutcome> {
    use super::selection::CandidateFitter;
    ensure!(!members.is_empty(), "no members to refit");
    ensure!(
        fitter.bound_plan_sha256() == Some(identity.plan_sha256.as_str()),
        "the fitter is not bound to the identity's plan"
    );
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
    ensure!(
        digest_ids(&row_ids[..selection_len]) == identity.row_sha256,
        "selection rows differ from the identity"
    );
    let selection_rows: Vec<usize> = (0..selection_len).collect();
    let report_rows: Vec<usize> = (selection_len..row_ids.len()).collect();

    let mut per_member = BTreeMap::new();
    let mut member_states = BTreeMap::new();
    let mut records = BTreeMap::new();
    for spec in members {
        ensure!(
            !per_member.contains_key(spec.id),
            "duplicate member {} in the refit",
            spec.id
        );
        budget
            .charge(BudgetKind::OuterRefit, 1)
            .map_err(|e| anyhow::anyhow!("refit of {} over the request budget: {e:?}", spec.id))?;
        let (model, state) = fitter.fit_state(spec, &selection_rows)?;
        let raw = fitter.predict_rows(&model, &report_rows)?;
        per_member.insert(
            spec.id.to_owned(),
            to_predictions(fitter.task(), fitter.classes(), raw)?,
        );
        records.insert(
            spec.id.to_owned(),
            MemberRefit {
                state_sha256: format!("{:x}", Sha256::digest(&state)),
                train_rows: selection_rows.len(),
            },
        );
        member_states.insert(spec.id.to_owned(), state);
    }
    let record = OuterRefitRecord::new(identity, &row_ids[..selection_len], report_ids, records);
    Ok(RefitOutcome {
        report: ReportPredictions {
            ids: report_ids.to_vec(),
            labels: report_labels.to_vec(),
            per_member,
        },
        member_states,
        record,
    })
}
