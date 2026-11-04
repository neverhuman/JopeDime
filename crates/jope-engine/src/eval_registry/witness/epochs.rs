//! Root-reviewed transitions in the original witness journal. No ledger reset.
use super::*;
const TRANSITION_DOMAIN: &str = "parallel.witness_epoch_transition.v1";
const TRANSITION_ACK_DOMAIN: &str = "parallel.witness_epoch_transition_ack.v1";
pub const MAX_EPOCH_TRANSITIONS: usize = 64;
mod store_ops;
mod transition;

pub use store_ops::{serve_stdio_epoch_transition, validate_native_grant_request_for_epoch};
#[cfg(test)]
use transition::verify_epoch_chain_checked;
pub(super) fn validate_effective_shape(policy: &WitnessPolicy) -> Result<()> {
    transition::validate_effective_shape(policy)
}
pub(super) fn validate_transition(
    base: &WitnessPolicy,
    previous: &WitnessPolicy,
    request: &SignedWitnessEpochTransition,
    when: u64,
) -> Result<()> {
    transition::validate_transition(base, previous, request, when)
}
pub use transition::{
    verify_epoch_chain, EpochTransitionMode, HistoricalBudgetAccounting,
    SignedWitnessEpochTransition, SignedWitnessEpochTransitionAck, WitnessEpochProof,
    WitnessEpochTransitionAckBody, WitnessEpochTransitionBody,
};

#[cfg(test)]
mod tests;
