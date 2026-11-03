//! Read-only value and packing review plans. This module returns data in memory and has no
//! storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.

use super::policy::Policy;
use crate::canon::sha256_hex;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Schema identifier for value policy v2 values.
pub const VALUE_POLICY_V2_SCHEMA: &str = "jopedime.retention-value-policy/v2";
/// Schema identifier for retention plan v2 values.
pub const RETENTION_PLAN_V2_SCHEMA: &str = "jope.retention-plan/v2";
/// Configured upper or lower bound for max inventory age secs.
pub const MAX_INVENTORY_AGE_SECS: u64 = 24 * 60 * 60;

mod plan_build;
mod replacement;
mod types;

use replacement::{replacement_for, replacement_gap, sum_bytes, validate_digest};
pub use types::{
    Candidate, ConsumerClosure, LearningSignals, PhysicalObject, PlanItem, Recommendation,
    ReplacementProof, RetentionPlanV2, RetentionPolicyV2, RetentionValue, ValueWeights,
};
