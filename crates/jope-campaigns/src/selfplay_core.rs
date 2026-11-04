//! Fast, deterministic self-play policy primitives.
//!
//! This module intentionally contains no service, database, or process I/O.
//! Callers own persistence and orchestration; these functions own the policy
//! formulas and fail closed on malformed numeric inputs.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const SELFPLAY_SELECTION_BIAS: &str = "test-exact-reused";
pub const RAMP_MID_SLOTS: i64 = 30;
pub const RAMP_FINAL_SLOTS: i64 = 40;
pub const RAMP_MID_DURATION_MS: i64 = 10 * 60_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl Display for PolicyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PolicyError {}

pub type PolicyResult<T> = Result<T, PolicyError>;

fn error(message: impl Into<String>) -> PolicyError {
    PolicyError(message.into())
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(sha256_bytes(bytes))
}

fn validate_finite(value: f64, label: &str) -> PolicyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(error(format!("{label} must be finite")))
    }
}

mod capacity;
mod dataset;
mod queue;
mod ranking;
#[cfg(test)]
mod tests;

pub use capacity::*;
pub use dataset::*;
pub use queue::*;
pub use ranking::*;

use queue::family_order_digest;
