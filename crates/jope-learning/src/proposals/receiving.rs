//! Pure adapter between persisted I26 delivery state and a coordinator-supplied release hook.

use super::outbox::{Lease, ProposalOutbox, TerminalKind};
use anyhow::{Context, Result};

/// Outcome returned by a pipeline-owned fenced release implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// The `Released` case of `ReleaseOutcome`.
    Released {
        /// Identifier for the associated physical attempt.
        physical_attempt_id: String,
        /// Physical cost units recorded for this release outcome.
        physical_cost_units: u64,
    },
    /// The `Refused` case of `ReleaseOutcome`.
    Refused {
        /// SHA-256 digest identifying the associated reason.
        reason_sha256: String,
    },
    /// The `Failed` case of `ReleaseOutcome`.
    Failed {
        /// SHA-256 digest of the receipt that authenticates this result.
        receipt_sha256: String,
        /// Physical cost units recorded for this release outcome.
        physical_cost_units: u64,
        /// Retryable recorded for this release outcome.
        retryable: bool,
    },
    /// The `Unknown` case of `ReleaseOutcome`.
    Unknown,
}

/// The integrator supplies this trait at `ExecutionPool::run`; it owns actual fit authority.
pub trait FencedRelease {
    /// Performs the authorize and release operation and returns an error if it cannot be completed.
    fn authorize_and_release(&mut self, request_key: &str, lease: &Lease)
        -> Result<ReleaseOutcome>;
}

/// Persist the start before invoking the authorized release hook, then persist its exact terminal
/// identity. A lost response is UNKNOWN and is never retried by this adapter.
pub fn dispatch(
    outbox: &mut ProposalOutbox,
    lease: &Lease,
    now_ms: u64,
    release: &mut impl FencedRelease,
) -> Result<ReleaseOutcome> {
    outbox.start(lease, now_ms)?;
    let outcome = match release.authorize_and_release(&lease.request_key, lease) {
        Ok(outcome) => outcome,
        Err(error) => {
            outbox.mark_unknown(lease)?;
            return Err(error).with_context(|| {
                "release hook failed after durable start; request outcome is unknown"
            });
        }
    };
    match &outcome {
        ReleaseOutcome::Released {
            physical_attempt_id,
            physical_cost_units,
        } => {
            outbox.finish(
                lease,
                TerminalKind::Released {
                    physical_attempt_id: physical_attempt_id.clone(),
                },
                *physical_cost_units,
            )?;
        }
        ReleaseOutcome::Refused { reason_sha256 } => {
            outbox.finish(
                lease,
                TerminalKind::Refused {
                    reason_sha256: reason_sha256.clone(),
                },
                0,
            )?;
        }
        ReleaseOutcome::Failed {
            receipt_sha256,
            physical_cost_units,
            retryable,
        } => {
            outbox.finish(
                lease,
                TerminalKind::Failed {
                    receipt_sha256: receipt_sha256.clone(),
                    retryable: *retryable,
                },
                *physical_cost_units,
            )?;
        }
        ReleaseOutcome::Unknown => outbox.mark_unknown(lease)?,
    }
    Ok(outcome)
}
