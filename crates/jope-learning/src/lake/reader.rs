//! Hash- and version-pinned reads from immutable source manifests.

use super::roles::{RoleAuthorizer, RoleFilter, RowRole};
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Schema identifier for manifest values.
pub const MANIFEST_SCHEMA: &str = "jope.lake-manifest/v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `SourceObject` is a record for the source object contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct SourceObject {
    /// Key recorded for this source object.
    pub key: String,
    /// Identifier for the associated version.
    pub version_id: String,
    /// Sha-256 recorded for this source object.
    pub sha256: String,
    /// Data or authority role assigned to this value.
    pub role: RowRole,
    /// SHA-256 digest identifying the associated exposure ancestry.
    pub exposure_ancestry_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
/// `Outcome` is an enum for the outcome contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub enum Outcome {
    /// The `Ok` case of `Outcome`.
    Ok {
        /// Measured value carried by this outcome.
        value: f64,
    },
    /// The `Failed` case of `Outcome`.
    Failed {
        /// Explanation accompanying this outcome.
        reason: String,
    },
    /// The `Timeout` case of `Outcome`.
    Timeout {
        /// Explanation accompanying this outcome.
        reason: String,
    },
    /// The `Unknown` case of `Outcome`.
    Unknown,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `SourceManifest` is a record for the source manifest contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct SourceManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Evaluator recorded for this source manifest.
    pub evaluator: String,
    /// Profile recorded for this source manifest.
    pub profile: String,
    /// Collection of source objects values carried by this source manifest.
    pub source_objects: Vec<SourceObject>,
}

impl SourceManifest {
    /// Validates this value against its invariants and returns an error when validation fails.
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

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(hex::encode(Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// `ExclusionReason` is an enum for the exclusion reason contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub enum ExclusionReason {
    /// The `Missing` case of `ExclusionReason`.
    Missing,
    /// The `Corrupt` case of `ExclusionReason`.
    Corrupt,
    /// The `VersionMismatch` case of `ExclusionReason`.
    VersionMismatch,
    /// The `Denied` case of `ExclusionReason`.
    Denied,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// `ExcludedRow` is a record for the excluded row contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct ExcludedRow {
    /// Index recorded for this excluded row.
    pub index: usize,
    /// Explanation associated with the recorded disposition.
    pub reason: ExclusionReason,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `OutcomeRow` is a record for the outcome row contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct OutcomeRow {
    /// Data or authority role assigned to this value.
    pub role: RowRole,
    /// Outcome recorded for this outcome row.
    pub outcome: Outcome,
    /// Source key recorded for this outcome row.
    pub source_key: String,
    /// Identifier for the associated source version.
    pub source_version_id: String,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// SHA-256 digest identifying the associated exposure ancestry.
    pub exposure_ancestry_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `OutcomeView` is a record for the outcome view contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct OutcomeView {
    /// SHA-256 digest identifying the associated source manifest.
    pub source_manifest_sha256: String,
    /// Evaluator recorded for this outcome view.
    pub evaluator: String,
    /// Profile recorded for this outcome view.
    pub profile: String,
    /// Requested recorded for this outcome view.
    pub requested: usize,
    /// Observed recorded for this outcome view.
    pub observed: usize,
    /// Number of rows represented by this value.
    pub rows: Vec<OutcomeRow>,
    /// Collection of excluded values carried by this outcome view.
    pub excluded: Vec<ExcludedRow>,
    /// Complete recorded for this outcome view.
    pub complete: bool,
}

/// Exact versioned-object interface. Production adapters must bind these results to immutable
/// version IDs and use the existing Store/Mirror transport; tests use an in-memory fake.
pub trait VersionedReader {
    /// Looks up versioned and returns the matching value when present.
    fn get_versioned(&self, key: &str, version_id: &str) -> Result<Option<VersionedBytes>>;
}

#[derive(Clone, Debug)]
/// `VersionedBytes` is a record for the versioned bytes contract in this module: Hash- and version-pinned reads from immutable source manifests.
pub struct VersionedBytes {
    /// Identifier for the associated version.
    pub version_id: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: Vec<u8>,
}

/// Reads view from the configured source.
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

/// Performs the require complete operation and returns an error if it cannot be completed.
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
