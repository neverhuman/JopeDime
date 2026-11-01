//! Inventory of exact bytes exported from the committed private outbox.
use super::{capture::digest, RecordIdentity, ResearchRound, Rung0Measurement};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordCounts {
    pub rounds: u64,
    pub proposals: u64,
    pub admissions: u64,
    pub dispatches: u64,
    pub outcomes: u64,
    pub successes: u64,
    pub failures: u64,
    pub incomplete_rounds: u64,
}

/// An integrity/readback receipt, not corpus admission or evaluator qualification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordInventory {
    pub schema: String,
    pub identity: RecordIdentity,
    pub evaluate_all_proposals: bool,
    pub records_sha256: String,
    pub counts: RecordCounts,
}

impl RecordInventory {
    pub(super) fn committed(records: &[ResearchRound], bytes: &[u8]) -> Result<Self> {
        let first = records.first().context("empty committed research stream")?;
        let mut counts = RecordCounts::default();
        for record in records {
            ensure!(
                record.identity == first.identity
                    && record.evaluate_all_proposals == first.evaluate_all_proposals,
                "committed record inventory binding changed"
            );
            counts.rounds += 1;
            counts.proposals += record.proposals.len() as u64;
            counts.admissions += record.policy.admitted.len() as u64;
            counts.dispatches += record.dispatched.len() as u64;
            counts.outcomes += record.outcomes.len() as u64;
            counts.incomplete_rounds += u64::from(!record.capture_complete);
            for outcome in &record.outcomes {
                match outcome.measurement {
                    Rung0Measurement::Success { .. } => counts.successes += 1,
                    Rung0Measurement::Failed { .. } => counts.failures += 1,
                }
            }
        }
        ensure!(
            counts.dispatches == counts.outcomes
                && counts.outcomes == counts.successes + counts.failures,
            "committed research inventory differs from actual outcomes"
        );
        Ok(Self {
            schema: "jope.research.inventory/v1".into(),
            identity: first.identity.clone(),
            evaluate_all_proposals: first.evaluate_all_proposals,
            records_sha256: digest(bytes),
            counts,
        })
    }
}
