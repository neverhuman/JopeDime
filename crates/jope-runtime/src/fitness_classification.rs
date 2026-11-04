//! Binary probability objectives (`jope.classification-fitness/v1`, spec §5.10).
//!
//! Probability columns and labels use the explicit, ordered class names in the specification.
//! `class_order[1]` is the positive class. Logloss uses natural logarithms without clipping;
//! Brier is the mean/sum of `(p_positive - is_positive)^2`, without a factor of two.
//! Invalid specifications fail before evaluation. Invalid inputs or undefined values return
//! [`FitnessError::Undefined`]; the caller must fail that candidate, never assign a placeholder.
//! Calibration, event publication and discovery-loop integration belong to their owning tasks.

pub use crate::fitness::Aggregation;
use crate::fitness::{Direction, Score};
use serde::{Deserialize, Serialize};

pub const FITNESS_VERSION: &str = "jope.classification-fitness/v1";
pub const MAX_SPEC_BYTES: usize = 32 * 1024;
pub const DEFAULT_CLASSIFICATION_THRESHOLD: f64 = 0.5;
/// Absolute tolerance for a probability pair's sum. Inputs are never renormalized.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

pub fn schema() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../schemas/jope.classification-fitness.v1.schema.json"
    ))
    .expect("committed classification fitness schema")
}

#[derive(Debug, thiserror::Error)]
pub enum FitnessError {
    #[error("invalid classification fitness specification: {0}")]
    InvalidSpecification(String),
    #[error("FITNESS_UNDEFINED: {0}")]
    Undefined(String),
}

type Result<T> = std::result::Result<T, FitnessError>;

mod metrics;
mod spec;

pub use spec::*;

#[cfg(test)]
mod tests;
