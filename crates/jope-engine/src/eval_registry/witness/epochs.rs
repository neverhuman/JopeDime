//! Root-reviewed transitions in the original witness journal. No ledger reset.
use super::*;
const TRANSITION_DOMAIN: &str = "parallel.witness_epoch_transition.v1";
const TRANSITION_ACK_DOMAIN: &str = "parallel.witness_epoch_transition_ack.v1";
pub const MAX_EPOCH_TRANSITIONS: usize = 64;
mod store_ops;
mod transition;

pub use store_ops::{serve_stdio_epoch_transition, validate_native_grant_request_for_epoch};
pub(super) use transition::{
    validate_effective_shape, validate_transition, verify_epoch_chain_checked,
};
pub use transition::{
    verify_epoch_chain, EpochTransitionMode, HistoricalBudgetAccounting,
    SignedWitnessEpochTransition, SignedWitnessEpochTransitionAck, WitnessEpochProof,
    WitnessEpochTransitionAckBody, WitnessEpochTransitionBody, MAX_EPOCH_TRANSITIONS,
};

#[cfg(test)]
mod tests;
