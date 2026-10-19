//! Outer refit of the admitted members (SC-DEC-184M JA-R1, C05; SC-DEC-184R JA-F1, JA-F3).
//!
//! Each member is refitted once on the whole selection side and predicts the report side. Those
//! report predictions are the only source of report losses. The selection side never sees them,
//! and the ensemble weights are chosen before this module runs.
//!
//! The refit produces an `OuterRefitRecord`: the request, the plan, the selection rows each refit
//! trained on, the report rows each refit predicted, and per member the sha256 of its fitted
//! state and of its exact report prediction bits. The record's fields are private to the
//! `model_build` module and its only constructor is `refit_members`. Admission compares the
//! record with the identity, the report predictions and, at reload, the persisted state bytes.
//! A record that was not produced by a refit cannot match the report predictions bit for bit.
use super::candidates::CandidateSpec;
use super::ensemble::ReportPredictions;
use super::identity::{digest_ids, ExecutionIdentity};
use super::native::{to_predictions, NativeFitter};
use super::selection::{BudgetKind, EvaluationBudget};
use crate::evaluation_score::Predictions;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const REFIT_RECORD_SCHEMA: &str = "jope.outer-refit-record/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberRefit {
    /// sha256 of the member's fitted state bytes.
    pub(super) state_sha256: String,
    /// Selection-side rows the member was refitted on.
    pub(super) train_rows: usize,
    /// sha256 of the member's exact report-side prediction bits.
    pub(super) report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterRefitRecord {
    pub(super) schema: String,
    pub(super) request_sha256: String,
    pub(super) plan_sha256: String,
    /// Digest of the selection-side row identifiers every refit trained on.
    pub(super) train_rows_sha256: String,
    /// Digest of the report-side row identifiers every refit predicted.
    pub(super) predict_rows_sha256: String,
    pub(super) members: BTreeMap<String, MemberRefit>,
}

/// Digest of a prediction set's exact bits, including its class order.
pub(super) fn predictions_digest(predictions: &Predictions) -> String {
    let mut hasher = Sha256::new();
    match predictions {
        Predictions::Values(values) => {
            hasher.update(b"values");
            for value in values {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        Predictions::Probabilities { classes, values } => {
            hasher.update(b"probabilities");
            for class in classes {
                hasher.update(class.as_bytes());
                hasher.update([0u8]);
            }
            for row in values {
                for value in row {
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

impl OuterRefitRecord {
    /// Only `refit_members` calls this. The record is a statement about fits that ran, and the
    /// digests it carries are checked against the report predictions and the states.
    pub(super) fn new(
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
            .check_wall_clock()
            .and_then(|()| budget.charge(BudgetKind::OuterRefit, 1))
            .map_err(|e| anyhow::anyhow!("refit of {} over the request budget: {e:?}", spec.id))?;
        let (model, state) = fitter.fit_state(spec, &selection_rows)?;
        let raw = fitter.predict_rows(&model, &report_rows)?;
        let predicted = to_predictions(fitter.task(), fitter.classes(), raw)?;
        let report_sha256 = predictions_digest(&predicted);
        per_member.insert(spec.id.to_owned(), predicted);
        records.insert(
            spec.id.to_owned(),
            MemberRefit {
                state_sha256: format!("{:x}", Sha256::digest(&state)),
                train_rows: selection_rows.len(),
                report_sha256,
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
