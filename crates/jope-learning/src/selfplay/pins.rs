//! Compatibility identity for self-play snapshots.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A provider's physical object version. `version_id` is opaque and is never inferred from a
/// path, alias, ETag, or filename.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalVersion {
    /// Resource recorded for this physical version.
    pub resource: String,
    /// Identifier for the associated version.
    pub version_id: String,
}

/// Exact immutable identities required to combine a candidate with its parent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityPins {
    /// SHA-256 digest identifying the associated vocabulary.
    pub vocabulary_sha256: String,
    /// SHA-256 digest identifying the associated support.
    pub support_sha256: String,
    /// SHA-256 digest identifying the associated critic.
    pub critic_sha256: String,
    /// SHA-256 digest identifying the associated replay.
    pub replay_sha256: String,
    /// SHA-256 digest identifying the associated library.
    pub library_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated lineage.
    pub lineage_sha256: String,
    /// SHA-256 digest identifying the associated training intent.
    pub training_intent_sha256: String,
    /// SHA-256 digest identifying the associated witness source.
    pub witness_source_sha256: String,
    /// SHA-256 digest identifying the associated authority source.
    pub authority_source_sha256: String,
    /// Optimizer epoch recorded for this compatibility pins.
    pub optimizer_epoch: u64,
    /// Collection of physical versions values carried by this compatibility pins.
    pub physical_versions: Vec<PhysicalVersion>,
}

impl CompatibilityPins {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self, retired: &RetiredVersions) -> Result<()> {
        for (name, digest) in [
            ("vocabulary", &self.vocabulary_sha256),
            ("support", &self.support_sha256),
            ("critic", &self.critic_sha256),
            ("replay", &self.replay_sha256),
            ("library", &self.library_sha256),
            ("checkpoint", &self.checkpoint_sha256),
            ("lineage", &self.lineage_sha256),
            ("training intent", &self.training_intent_sha256),
            ("witness source", &self.witness_source_sha256),
            ("authority source", &self.authority_source_sha256),
        ] {
            ensure!(is_sha256(digest), "{name} pin must be a lowercase sha256");
        }
        let mut seen = BTreeSet::new();
        let mut previous: Option<&PhysicalVersion> = None;
        for version in &self.physical_versions {
            ensure!(
                !version.resource.trim().is_empty(),
                "physical version resource is empty"
            );
            ensure!(
                !version.version_id.trim().is_empty(),
                "physical VersionId is empty"
            );
            ensure!(
                seen.insert((version.resource.clone(), version.version_id.clone())),
                "duplicate physical VersionId for {}",
                version.resource
            );
            if let Some(prior) = previous {
                ensure!(
                    prior < version,
                    "physical versions must be sorted by resource and VersionId"
                );
            }
            ensure!(
                !retired.contains(&version.resource, &version.version_id),
                "physical VersionId {} for {} is retired",
                version.version_id,
                version.resource
            );
            previous = Some(version);
        }
        Ok(())
    }

    /// Performs the ensure compatible operation and returns an error if it cannot be completed.
    pub fn ensure_compatible(&self, parent: &Self) -> Result<()> {
        ensure!(
            self == parent,
            "candidate compatibility pins differ from the parent (vocabulary/support/critic/replay/library/checkpoint/lineage/training-intent/witness/authority/epoch/version mismatch)"
        );
        Ok(())
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        #[derive(Serialize)]
        struct Payload<'a> {
            schema: &'static str,
            pins: &'a CompatibilityPins,
        }
        let bytes = crate::canon::canonical_json(&Payload {
            schema: "jope.learning-snapshot-pins/v1",
            pins: self,
        })?;
        let frame = jope_records::wire_contract::StorageEnvelopeAdapter::seal(&bytes, None, None)?;
        Ok(frame.own_id().sha256_hex().to_string())
    }
}

/// Explicit tombstones for immutable physical versions. No alias-based lookup is accepted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetiredVersions(BTreeSet<(String, String)>);

impl RetiredVersions {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(entries: impl IntoIterator<Item = (String, String)>) -> Self {
        Self(entries.into_iter().collect())
    }

    /// Performs the contains operation.
    pub fn contains(&self, resource: &str, version_id: &str) -> bool {
        self.0
            .contains(&(resource.to_owned(), version_id.to_owned()))
    }
}

/// Returns whether SHA-256.
pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
