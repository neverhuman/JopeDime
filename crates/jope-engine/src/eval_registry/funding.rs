use std::collections::BTreeSet;

use anyhow::{ensure, Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ProcessIdentity;

mod ledger;
mod types;
mod validation;

pub use ledger::{
    claim_participant, consume_start, event, participant_id, preflight, register_authorization,
    reserve, validate_witness_event, verify_event,
};
pub use types::{
    digest, ExecutionClaim, FundingAllocation, FundingAuthorization, FundingClock, FundingEvent,
    FundingReceipt, FundingTrust, ResourceCaps, SignedAuthorization, SignedReservation, TaskDemand,
    TaskReservation, VerifiedReservation, MAX_APPROVED_TASKS, MAX_AUTHORIZATION_BYTES, SQL_SCHEMA,
};
use types::{hash_valid, signed_bytes, text_valid, verify_signature, AUTH_TAG, RESERVATION_TAG};
pub use validation::{sign_reservation, verify_authorization, verify_reservation};

#[cfg(test)]
mod tests;
