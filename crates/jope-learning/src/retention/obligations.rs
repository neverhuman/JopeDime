//! Offline model of exact-version dependencies and retention promises.
//!
//! This module returns review data only. It cannot publish, move, or remove stored objects.

use crate::canon::canonical_sha256;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for obligation plan values.
pub const OBLIGATION_PLAN_SCHEMA: &str = "jope.retention-plan/v2";

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

mod plan;
mod promises;
mod references;

pub use plan::{
    Candidate, Disposition, LifecycleAssessment, ObligationPlanV2, ObligationRow,
    RetentionObligationDisposition, TagAssessment,
};
pub use promises::{PinGeneration, RecoveryPromise, RetiredVersion};
pub use references::{
    PromiseKind, ProvenanceRef, RuntimeRestoreProof, StrongPhysicalRef, VersionId,
};
