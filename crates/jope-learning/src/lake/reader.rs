//! Hash- and version-pinned reads from immutable source manifests.

use super::roles::{RoleAuthorizer, RoleFilter, RowRole};
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MANIFEST_SCHEMA: &str = "jope.lake-manifest/v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceObject {
    pub key: String,
    pub version_id: String,
    pub sha256: String,
    pub role: RowRole,
    pub exposure_ancestry_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum Outcome {
    Ok { value: f64 },
    Failed { reason: String },
    Timeout { reason: String },
    Unknown,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifest {
    pub schema: String,
    pub evaluator: String,
    pub profile: String,
    pub source_objects: Vec<SourceObject>,
}

impl SourceManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == MANIFEST_SCHEMA,
            "unsupported lake manifest schema"
        );
        ensure!(
            !self.evaluator.is_empty() && !self.profile.is_empty(),
            "missing evaluator/profile"
        );
        let mut identities = BTreeSet::new();
        for object in &self.source_objects {
            ensure!(
                safe_key(&object.key) && !object.version_id.is_empty(),
                "missing source locator"
            );
            ensure!(
                is_sha256(&object.sha256) && is_sha256(&object.exposure_ancestry_sha256),
                "invalid source digest"
            );
            ensure!(
                identities.insert((&object.key, &object.version_id)),
                "duplicate source object identity"
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(hex::encode(Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ExclusionReason {
    Missing,
    Corrupt,
    VersionMismatch,
    Denied,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExcludedRow {
    pub index: usize,
    pub reason: ExclusionReason,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutcomeRow {
    pub role: RowRole,
    pub outcome: Outcome,
    pub source_key: String,
    pub source_version_id: String,
    pub source_sha256: String,
    pub exposure_ancestry_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutcomeView {
    pub source_manifest_sha256: String,
    pub evaluator: String,
    pub profile: String,
    pub requested: usize,
    pub observed: usize,
    pub rows: Vec<OutcomeRow>,
    pub excluded: Vec<ExcludedRow>,
    pub complete: bool,
}

/// Exact versioned-object interface. Production adapters must bind these results to immutable
/// version IDs and use the existing Store/Mirror transport; tests use an in-memory fake.
pub trait VersionedReader {
    fn get_versioned(&self, key: &str, version_id: &str) -> Result<Option<VersionedBytes>>;
}

#[derive(Clone, Debug)]
pub struct VersionedBytes {
    pub version_id: String,
    pub bytes: Vec<u8>,
}

pub fn read_view(
    manifest: &SourceManifest,
    expected_manifest_sha256: &str,
    principal: &impl RoleAuthorizer,
    filter: RoleFilter,
    reader: &impl VersionedReader,
) -> Result<OutcomeView> {
    let manifest_digest = manifest.digest()?;
    ensure!(
        manifest_digest == expected_manifest_sha256,
        "source manifest digest mismatch"
    );
    let mut rows = Vec::new();
    let mut excluded = Vec::new();
    for (index, object) in manifest.source_objects.iter().enumerate() {
        if filter.authorize(principal, object.role).is_err() {
            excluded.push(ExcludedRow {
                index,
                reason: ExclusionReason::Denied,
            });
            continue;
        }
        match reader.get_versioned(&object.key, &object.version_id)? {
            None => excluded.push(ExcludedRow {
                index,
                reason: ExclusionReason::Missing,
            }),
            Some(found) if found.version_id != object.version_id => excluded.push(ExcludedRow {
                index,
                reason: ExclusionReason::VersionMismatch,
            }),
            Some(found) if hex::encode(Sha256::digest(&found.bytes)) != object.sha256 => excluded
                .push(ExcludedRow {
                    index,
                    reason: ExclusionReason::Corrupt,
                }),
            Some(found) => match serde_json::from_slice::<Outcome>(&found.bytes) {
                Ok(outcome) => rows.push(OutcomeRow {
                    role: object.role,
                    outcome,
                    source_key: object.key.clone(),
                    source_version_id: object.version_id.clone(),
                    source_sha256: object.sha256.clone(),
                    exposure_ancestry_sha256: object.exposure_ancestry_sha256.clone(),
                }),
                Err(_) => excluded.push(ExcludedRow {
                    index,
                    reason: ExclusionReason::Corrupt,
                }),
            },
        }
    }
    let requested = manifest.source_objects.len();
    let observed = rows.len();
    Ok(OutcomeView {
        source_manifest_sha256: manifest_digest,
        evaluator: manifest.evaluator.clone(),
        profile: manifest.profile.clone(),
        requested,
        observed,
        complete: requested == observed && excluded.is_empty(),
        rows,
        excluded,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn safe_key(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.contains('\\')
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

pub fn require_complete(view: &OutcomeView) -> Result<()> {
    if !view.complete {
        bail!(
            "lake view is incomplete; requested={}, observed={}, excluded={}",
            view.requested,
            view.observed,
            view.excluded.len()
        );
    }
    Ok(())
}
