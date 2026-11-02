//! Opt-in target binding. Hashes prove retained bytes, not execution or authority.
use crate::transpile::EmitOptions;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

pub const SCHEMA: &str = "jope.battle-artifact/v3";
pub const ENVELOPE: &str = "jope.canonical-json/v1";
const FORMULA: &str = "sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)";

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

mod canonical;
mod identity;
mod manifest;
mod read;
mod records;
mod targets;

pub(super) use canonical::{canonical, check_digest, domain_hash, safe_relative};
pub use canonical::{ArtifactDomain, Domain};
pub use identity::{BuildExecution, BuildIdentity, FittedIdentity, OptionsIdentity};
pub use manifest::ArtifactManifestV3;
pub use read::{
    read_artifact, read_manifest, verify_directory, ExpectedArtifactV3, TargetEmission,
};
pub use records::{FileBinding, TargetRecord, TargetState};
pub use targets::Target;
