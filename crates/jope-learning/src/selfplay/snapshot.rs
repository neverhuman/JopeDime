//! Acyclic, complete snapshot candidate and its content identity.

use crate::canon::is_hex64;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::pins::{CompatibilityPins, RetiredVersions};

/// Schema identifier for snapshot values.
pub const SNAPSHOT_SCHEMA: &str = "jope.learning-snapshot/v1";

/// Hash of one immutable object in the execution-to-capsule chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    /// Sha-256 recorded for this artifact ref.
    pub sha256: String,
    /// SHA-256 digest identifying the associated parent.
    pub parent_sha256: Option<String>,
}

/// Ordered references: execution -> raw result -> payload -> assessment -> capsule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateArtifacts {
    /// Execution recorded for this candidate artifacts.
    pub execution: ArtifactRef,
    /// Raw result recorded for this candidate artifacts.
    pub raw_result: ArtifactRef,
    /// Payload recorded for this candidate artifacts.
    pub payload: ArtifactRef,
    /// Assessment recorded for this candidate artifacts.
    pub assessment: ArtifactRef,
    /// Capsule recorded for this candidate artifacts.
    pub capsule: ArtifactRef,
}

impl CandidateArtifacts {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for (name, item, expected_parent) in [
            ("execution", &self.execution, None),
            ("raw_result", &self.raw_result, Some(&self.execution.sha256)),
            ("payload", &self.payload, Some(&self.raw_result.sha256)),
            ("assessment", &self.assessment, Some(&self.payload.sha256)),
            ("capsule", &self.capsule, Some(&self.assessment.sha256)),
        ] {
            ensure!(is_hex64(&item.sha256), "{name} must name a sha256 artifact");
            ensure!(
                seen.insert(&item.sha256),
                "artifact chain contains a repeated digest"
            );
            if let Some(parent) = expected_parent {
                ensure!(
                    item.parent_sha256.as_ref() == Some(parent),
                    "{name} must follow its predecessor in execution -> raw result -> payload -> assessment -> capsule order"
                );
            } else {
                ensure!(
                    item.parent_sha256.is_none(),
                    "execution cannot refer to a later artifact"
                );
            }
        }
        Ok(())
    }
}

/// Immutable candidate payload. The digest excludes authority state and filenames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated candidate.
    pub candidate_id: String,
    /// Expected generation recorded for this snapshot.
    pub expected_generation: u64,
    /// Optional previous root value; `None` means it is not available.
    pub previous_root: Option<String>,
    /// Provenance recorded for this snapshot.
    pub provenance: SnapshotProvenance,
    /// Pins recorded for this snapshot.
    pub pins: CompatibilityPins,
    /// Artifacts recorded for this snapshot.
    pub artifacts: CandidateArtifacts,
}

/// I27 caller-supplied provenance. The verifier must authenticate the referenced input-root
/// receipt; validating digest shape alone is not an authority check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotProvenance {
    /// Source commit recorded for this snapshot provenance.
    pub source_commit: String,
    /// SHA-256 digest identifying the associated command identity.
    pub command_identity_sha256: String,
    /// SHA-256 digest identifying the associated input root receipt.
    pub input_root_receipt_sha256: String,
}

impl SnapshotProvenance {
    fn validate(&self) -> Result<()> {
        ensure!(
            (self.source_commit.len() == 40 || self.source_commit.len() == 64)
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "source commit must be a lowercase Git object id"
        );
        ensure!(
            is_hex64(&self.command_identity_sha256),
            "command identity must be a sha256"
        );
        ensure!(
            is_hex64(&self.input_root_receipt_sha256),
            "input-root receipt must be a sha256"
        );
        Ok(())
    }
}

impl Snapshot {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self, retired: &RetiredVersions) -> Result<()> {
        ensure!(
            self.schema == SNAPSHOT_SCHEMA,
            "unsupported snapshot schema"
        );
        ensure!(
            !self.candidate_id.trim().is_empty(),
            "candidate id is empty"
        );
        ensure!(
            (self.expected_generation == 0) == self.previous_root.is_none(),
            "generation zero must have no previous root and later generations must pin one"
        );
        if let Some(root) = &self.previous_root {
            ensure!(is_hex64(root), "previous root must be a sha256");
        }
        self.provenance.validate()?;
        self.pins.validate(retired)?;
        self.artifacts.validate()
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        let payload = crate::canon::canonical_json(self)?;
        let frame =
            jope_records::wire_contract::StorageEnvelopeAdapter::seal(&payload, None, None)?;
        Ok(frame.own_id().sha256_hex().to_string())
    }

    /// Encodes this value in its canonical serialized representation.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let payload = crate::canon::canonical_json(self)?;
        let frame =
            jope_records::wire_contract::StorageEnvelopeAdapter::seal(&payload, None, None)?;
        Ok(frame.to_canonical_bytes())
    }

    /// Decodes the canonical representation and rejects malformed or noncanonical input.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let frame = jope_records::wire_contract::StorageEnvelopeAdapter::open(bytes)?;
        ensure!(
            frame.schema_id() == SNAPSHOT_SCHEMA,
            "unexpected snapshot payload schema"
        );
        let value: Self = serde_json::from_slice(frame.payload().as_bytes())?;
        ensure!(
            value.encode()? == bytes,
            "snapshot envelope is not canonical"
        );
        Ok(value)
    }
}

/// A root pointer is a compare-and-swap fact. Candidate filenames are deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRoot {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Generation recorded for this snapshot root.
    pub generation: u64,
    /// SHA-256 digest identifying the associated root.
    pub root_sha256: String,
    /// Optional previous root value; `None` means it is not available.
    pub previous_root: Option<String>,
    /// SHA-256 digest identifying the associated pins.
    pub pins_sha256: String,
}

impl SnapshotRoot {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.learning-authority/v1",
            "unsupported authority schema"
        );
        ensure!(is_hex64(&self.root_sha256), "root digest must be a sha256");
        ensure!(is_hex64(&self.pins_sha256), "pin digest must be a sha256");
        ensure!(
            self.generation > 0,
            "committed root generation must be positive"
        );
        ensure!(
            (self.generation == 1) == self.previous_root.is_none(),
            "generation one must have no previous root and later generations must pin one"
        );
        if let Some(parent) = &self.previous_root {
            ensure!(is_hex64(parent), "previous root must be a sha256");
        }
        Ok(())
    }

    /// Encodes this value in its canonical serialized representation.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let payload = crate::canon::canonical_json(self)?;
        let frame =
            jope_records::wire_contract::StorageEnvelopeAdapter::seal(&payload, None, None)?;
        Ok(frame.to_canonical_bytes())
    }

    /// Decodes the canonical representation and rejects malformed or noncanonical input.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let frame = jope_records::wire_contract::StorageEnvelopeAdapter::open(bytes)?;
        ensure!(
            frame.schema_id() == "jope.learning-authority/v1",
            "unexpected authority schema"
        );
        let value: Self = serde_json::from_slice(frame.payload().as_bytes())?;
        value.validate()?;
        ensure!(
            value.encode()? == bytes,
            "authority envelope is not canonical"
        );
        Ok(value)
    }

    /// Performs the verify snapshot operation and returns an error if it cannot be completed.
    pub fn verify_snapshot(&self, snapshot: &Snapshot, retired: &RetiredVersions) -> Result<()> {
        self.validate()?;
        snapshot.validate(retired)?;
        ensure!(
            self.root_sha256 == snapshot.digest()?,
            "root digest does not bind snapshot"
        );
        ensure!(
            self.pins_sha256 == snapshot.pins.digest()?,
            "root digest does not bind compatibility pins"
        );
        ensure!(
            self.previous_root == snapshot.previous_root,
            "root parent differs from snapshot parent"
        );
        ensure!(
            snapshot.expected_generation.checked_add(1) == Some(self.generation),
            "root generation differs from snapshot expectation"
        );
        Ok(())
    }
}

/// Observable lifecycle states. This worker adapter only creates LocalCommitted or
/// RemotePending. ResearchParent and AcceptedServing require distinct downstream evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotRole {
    /// The `ResearchParent` case of `SnapshotRole`.
    ResearchParent,
    /// The `AcceptedServing` case of `SnapshotRole`.
    AcceptedServing,
}

/// Candidate result states this adapter may persist. Serving authority has no transition here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    /// The `LocalCommitted` case of `CandidateState`.
    LocalCommitted,
    /// The `RemotePending` case of `CandidateState`.
    RemotePending,
    /// The `RemotePublished` case of `CandidateState`.
    RemotePublished,
    /// The `Lost` case of `CandidateState`.
    Lost,
}

/// Result authority is explicit and is never encoded in a filename.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateOutcome {
    /// The `LocalCommitted` case of `CandidateOutcome`.
    LocalCommitted {
        /// Root recorded for this candidate outcome.
        root: SnapshotRoot,
    },
    /// The `RemotePending` case of `CandidateOutcome`.
    RemotePending {
        /// Root recorded for this candidate outcome.
        root: SnapshotRoot,
        /// Explanation associated with the recorded disposition.
        reason: String,
    },
    /// The `RemotePublished` case of `CandidateOutcome`.
    RemotePublished {
        /// Root recorded for this candidate outcome.
        root: SnapshotRoot,
    },
    /// The `Lost` case of `CandidateOutcome`.
    Lost {
        /// SHA-256 digest identifying the associated candidate.
        candidate_sha256: String,
        /// Winner recorded for this candidate outcome.
        winner: SnapshotRoot,
    },
}
