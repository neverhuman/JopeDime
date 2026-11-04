//! Fail-closed recovery authorities for the Phase 5.1 cutover.
//!
//! The types in this module are deliberately independent of the active
//! runtime pointer.  In particular, a rollback capsule binds an unchanged
//! runtime bundle and a separately content-addressed unit overlay.

use crate::encoder_v2_identity::ArtifactIdentity;
use crate::paths;
use crate::selfplay_data::canonical_json_sha256;
use crate::selfplay_runtime::sha256_bytes;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path};

pub const V1_QUARANTINE_SCHEMA: &str = "JopeV1EncodingQuarantine.v1";
pub const ROLLBACK_CAPSULE_SCHEMA: &str = "JopeRollbackCapsule.v1";
pub const ROLLBACK_CAPSULE_V2_SCHEMA: &str = "JopeRollbackCapsule.v2";
pub const ROLLBACK_BASELINE_DERIVATION_SCHEMA: &str = "JopeRollbackBaselineDerivation.v1";
pub const ROLLBACK_CHALLENGE_SCHEMA: &str = "JopeRollbackChallenge.v1";
pub const ROLLBACK_NODE_PROBE_SCHEMA: &str = "JopeRollbackNodeProbe.v2";
pub const ROLLBACK_PATH_AUDIT_SCHEMA: &str = "JopeRollbackPathAudit.v2";
pub const ROLLBACK_INSTALL_SIMULATION_SCHEMA: &str = "JopeRollbackInstallSimulation.v2";
pub const ROLLBACK_EVALUATOR_RETENTION_SCHEMA: &str = "JopeRollbackEvaluatorRetention.v2";
pub const ROLLBACK_STATE_COMPATIBILITY_SCHEMA: &str = "JopeRollbackStateCompatibility.v2";
pub const RECOVERY_CONTINUITY_OBSERVATION_SCHEMA: &str = "JopeRecoveryContinuityObservation.v1";
pub const ROLLBACK_RUNTIME_BUNDLE_SHA256: &str =
    "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4";
jope_paths::once_str!(
    pub fn immutable_flywheel_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    pub fn runtime_current_root_str => paths::runtime_bundle_current()
);
pub const REQUIRED_NODES: [&str; 5] = ["xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"];
pub const CONTINUITY_SERVICES: [&str; 7] = [
    "freezer",
    "freshness",
    "ingestion",
    "liveness",
    "queue",
    "reporting",
    "supply",
];

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn require_absolute_normalized(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(path.is_absolute(), "{field} is not absolute");
    ensure!(
        path.components()
            .all(|part| !matches!(part, Component::ParentDir | Component::CurDir)),
        "{field} is not normalized"
    );
    Ok(())
}

fn require_relative_normalized(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(!path.is_absolute(), "{field} is absolute");
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "{field} is not a normalized relative path"
    );
    Ok(())
}

mod continuity;
pub use continuity::*;

mod quarantine;
pub use quarantine::*;

mod rollback_v1;
pub use rollback_v1::*;

mod rollback_v2;
pub use rollback_v2::*;

#[cfg(test)]
#[path = "recovery_contracts/rollback_v2_tests.rs"]
mod rollback_v2_tests;
