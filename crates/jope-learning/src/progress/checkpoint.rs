//! Proof-bound checkpoints for deterministic derived progress reconstruction.
//!
//! These types carry identities and references only. They do not certify optimizer mutation,
//! parameter changes, or training authority. A committed update is counted only when the caller's
//! existing owner verifies the referenced source receipts.

use crate::canon::{canonical_json, is_hex64, sha256_hex};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema identifier for checkpoint values.
pub const CHECKPOINT_SCHEMA: &str = "jope.progress.segment-checkpoint/v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ProofReference` is a record for the proof reference contract in this module: Proof-bound checkpoints for deterministic derived progress reconstruction.
pub struct ProofReference {
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Key recorded for this proof reference.
    pub key: String,
    /// Sha-256 recorded for this proof reference.
    pub sha256: String,
}

impl ProofReference {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.kind.is_empty() && self.kind.len() <= 96,
            "invalid proof kind"
        );
        ensure!(
            !self.key.is_empty() && self.key.len() <= 1024,
            "invalid proof key"
        );
        ensure!(is_hex64(&self.sha256), "invalid proof digest");
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `CommittedUpdate` is a record for the committed update contract in this module: Proof-bound checkpoints for deterministic derived progress reconstruction.
pub struct CommittedUpdate {
    /// Update key recorded for this committed update.
    pub update_key: String,
    /// Source commit recorded for this committed update.
    pub source_commit: String,
    /// SHA-256 digest identifying the associated source root.
    pub source_root_sha256: String,
    /// Checkpoint recorded for this committed update.
    pub checkpoint: ProofReference,
    /// Intent recorded for this committed update.
    pub intent: ProofReference,
    /// Occurrence recorded for this committed update.
    pub occurrence: ProofReference,
    /// Rank recorded for this committed update.
    pub rank: u64,
    /// Attempt recorded for this committed update.
    pub attempt: u64,
    /// Epoch recorded for this committed update.
    pub epoch: u64,
}

impl CommittedUpdate {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.update_key.is_empty() && self.update_key.len() <= 512,
            "invalid update identity"
        );
        ensure!(
            !self.source_commit.is_empty() && self.source_commit.len() <= 128,
            "invalid source commit"
        );
        ensure!(
            is_hex64(&self.source_root_sha256),
            "invalid source root digest"
        );
        self.checkpoint.validate()?;
        self.intent.validate()?;
        self.occurrence.validate()?;
        ensure!(
            self.rank > 0 && self.attempt > 0,
            "committed update rank and attempt must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "state", content = "receipt")]
/// `CheckpointEvidence` is an enum for the checkpoint evidence contract in this module: Proof-bound checkpoints for deterministic derived progress reconstruction.
pub enum CheckpointEvidence {
    /// Boxed: a committed update is ~3x the size of every other variant, and
    /// evidence lists are held whole while a segment is rebuilt. Serde encodes
    /// `Box<T>` exactly as `T`, so the receipt bytes are unchanged.
    Committed(Box<CommittedUpdate>),
    /// The `Proposed` case of `CheckpointEvidence`.
    Proposed {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Computed` case of `CheckpointEvidence`.
    Computed {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Applied` case of `CheckpointEvidence`.
    Applied {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Uncommitted` case of `CheckpointEvidence`.
    Uncommitted {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Failed` case of `CheckpointEvidence`.
    Failed {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Unknown` case of `CheckpointEvidence`.
    Unknown {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Inconclusive` case of `CheckpointEvidence`.
    Inconclusive {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Divergent` case of `CheckpointEvidence`.
    Divergent {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `Quarantined` case of `CheckpointEvidence`.
    Quarantined {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `LocalOnly` case of `CheckpointEvidence`.
    LocalOnly {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
    /// The `RemotePending` case of `CheckpointEvidence`.
    RemotePending {
        /// Key recorded for this checkpoint evidence.
        key: String,
        /// Provenance source for this value.
        source: ProofReference,
    },
}

impl CheckpointEvidence {
    fn identity(&self) -> &str {
        match self {
            Self::Committed(c) => &c.update_key,
            Self::Proposed { key, .. }
            | Self::Computed { key, .. }
            | Self::Applied { key, .. }
            | Self::Uncommitted { key, .. }
            | Self::Failed { key, .. }
            | Self::Unknown { key, .. }
            | Self::Inconclusive { key, .. }
            | Self::Divergent { key, .. }
            | Self::Quarantined { key, .. }
            | Self::LocalOnly { key, .. }
            | Self::RemotePending { key, .. } => key,
        }
    }

    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.identity().is_empty() && self.identity().len() <= 512,
            "invalid checkpoint evidence identity"
        );
        match self {
            Self::Committed(update) => update.validate(),
            Self::Proposed { source, .. }
            | Self::Computed { source, .. }
            | Self::Applied { source, .. }
            | Self::Uncommitted { source, .. }
            | Self::Failed { source, .. }
            | Self::Unknown { source, .. }
            | Self::Inconclusive { source, .. }
            | Self::Divergent { source, .. }
            | Self::Quarantined { source, .. }
            | Self::LocalOnly { source, .. }
            | Self::RemotePending { source, .. } => source.validate(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `SegmentCheckpoint` is a record for the segment checkpoint contract in this module: Proof-bound checkpoints for deterministic derived progress reconstruction.
pub struct SegmentCheckpoint {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Generation recorded for this segment checkpoint.
    pub generation: u64,
    /// Source commit recorded for this segment checkpoint.
    pub source_commit: String,
    /// SHA-256 digest identifying the associated source root.
    pub source_root_sha256: String,
    /// Collection of evidence values carried by this segment checkpoint.
    pub evidence: Vec<CheckpointEvidence>,
    /// Digest recorded for this segment checkpoint.
    pub digest: String,
}

impl SegmentCheckpoint {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        generation: u64,
        source_commit: String,
        source_root_sha256: String,
        evidence: Vec<CheckpointEvidence>,
    ) -> Result<Self> {
        ensure!(
            !source_commit.is_empty() && source_commit.len() <= 128,
            "invalid checkpoint source commit"
        );
        ensure!(
            is_hex64(&source_root_sha256),
            "invalid checkpoint source root digest"
        );
        let mut seen = BTreeMap::new();
        for item in &evidence {
            item.validate()?;
            let bytes = canonical_json(item)?;
            if let Some(previous) = seen.insert(item.identity().to_owned(), sha256_hex(&bytes)) {
                ensure!(
                    previous == sha256_hex(&bytes),
                    "checkpoint key is reused for conflicting source evidence"
                );
            }
        }
        let mut checkpoint = Self {
            schema: CHECKPOINT_SCHEMA.into(),
            generation,
            source_commit,
            source_root_sha256,
            evidence,
            digest: String::new(),
        };
        checkpoint.digest = checkpoint.body_digest()?;
        Ok(checkpoint)
    }

    /// Verifies the bound evidence and returns an error when verification fails.
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema == CHECKPOINT_SCHEMA,
            "unsupported checkpoint schema"
        );
        ensure!(
            self.digest == self.body_digest()?,
            "checkpoint digest mismatch"
        );
        Ok(())
    }

    fn body_digest(&self) -> Result<String> {
        #[derive(Serialize)]
        struct Body<'a> {
            schema: &'a str,
            generation: u64,
            source_commit: &'a str,
            source_root_sha256: &'a str,
            evidence: &'a [CheckpointEvidence],
        }
        Ok(sha256_hex(&canonical_json(&Body {
            schema: &self.schema,
            generation: self.generation,
            source_commit: &self.source_commit,
            source_root_sha256: &self.source_root_sha256,
            evidence: &self.evidence,
        })?))
    }
}

/// Coordinator/owner-supplied check against actual committed source receipts. False means held.
pub trait CommittedUpdateVerifier {
    /// Performs the verify committed update operation and returns an error if it cannot be completed.
    fn verify_committed_update(&self, update: &CommittedUpdate) -> Result<bool>;
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RebuiltProgress` is a record for the rebuilt progress contract in this module: Proof-bound checkpoints for deterministic derived progress reconstruction.
pub struct RebuiltProgress {
    /// Committed updates recorded for this rebuilt progress.
    pub committed_updates: u64,
    /// Proposed recorded for this rebuilt progress.
    pub proposed: u64,
    /// Computed recorded for this rebuilt progress.
    pub computed: u64,
    /// Applied recorded for this rebuilt progress.
    pub applied: u64,
    /// Uncommitted recorded for this rebuilt progress.
    pub uncommitted: u64,
    /// Failed recorded for this rebuilt progress.
    pub failed: u64,
    /// Unknown recorded for this rebuilt progress.
    pub unknown: u64,
    /// Inconclusive recorded for this rebuilt progress.
    pub inconclusive: u64,
    /// Divergent recorded for this rebuilt progress.
    pub divergent: u64,
    /// Quarantined recorded for this rebuilt progress.
    pub quarantined: u64,
    /// Local only recorded for this rebuilt progress.
    pub local_only: u64,
    /// Remote pending recorded for this rebuilt progress.
    pub remote_pending: u64,
    /// Held committed recorded for this rebuilt progress.
    pub held_committed: u64,
}

/// Deterministic reconstruction from retained proof records. Exact duplicate deliveries have no
/// effect; conflicting reuse of an identity is an error. Unsupported committed receipts stay held.
pub fn rebuild(
    checkpoints: &[SegmentCheckpoint],
    verifier: &impl CommittedUpdateVerifier,
) -> Result<RebuiltProgress> {
    let mut unique = BTreeMap::<String, CheckpointEvidence>::new();
    for checkpoint in checkpoints {
        checkpoint.verify()?;
        for evidence in &checkpoint.evidence {
            evidence.validate()?;
            if let Some(old) = unique.get(evidence.identity()) {
                ensure!(
                    old == evidence,
                    "progress evidence key conflicts across checkpoints"
                );
            } else {
                unique.insert(evidence.identity().to_owned(), evidence.clone());
            }
        }
    }
    let mut result = RebuiltProgress::default();
    for evidence in unique.values() {
        match evidence {
            CheckpointEvidence::Committed(update) => {
                if verifier.verify_committed_update(update)? {
                    result.committed_updates += 1;
                } else {
                    result.held_committed += 1;
                }
            }
            CheckpointEvidence::Proposed { .. } => result.proposed += 1,
            CheckpointEvidence::Computed { .. } => result.computed += 1,
            CheckpointEvidence::Applied { .. } => result.applied += 1,
            CheckpointEvidence::Uncommitted { .. } => result.uncommitted += 1,
            CheckpointEvidence::Failed { .. } => result.failed += 1,
            CheckpointEvidence::Unknown { .. } => result.unknown += 1,
            CheckpointEvidence::Inconclusive { .. } => result.inconclusive += 1,
            CheckpointEvidence::Divergent { .. } => result.divergent += 1,
            CheckpointEvidence::Quarantined { .. } => result.quarantined += 1,
            CheckpointEvidence::LocalOnly { .. } => result.local_only += 1,
            CheckpointEvidence::RemotePending { .. } => result.remote_pending += 1,
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    struct Verify(bool);
    impl CommittedUpdateVerifier for Verify {
        fn verify_committed_update(&self, _: &CommittedUpdate) -> Result<bool> {
            Ok(self.0)
        }
    }

    fn reference(key: &str) -> ProofReference {
        ProofReference {
            kind: "fixture".into(),
            key: key.into(),
            sha256: "a".repeat(64),
        }
    }

    fn update() -> CommittedUpdate {
        CommittedUpdate {
            update_key: "update/step-1".into(),
            source_commit: "commit-1".into(),
            source_root_sha256: "b".repeat(64),
            checkpoint: reference("checkpoint/1"),
            intent: reference("intent/1"),
            occurrence: reference("occurrence/1"),
            rank: 1,
            attempt: 1,
            epoch: 1,
        }
    }

    #[test]
    fn duplicate_checkpoint_and_ack_identity_do_not_inflate_verified_updates() -> Result<()> {
        let committed = CheckpointEvidence::Committed(Box::new(update()));
        let unknown = CheckpointEvidence::Unknown {
            key: "update/unknown".into(),
            source: reference("unknown/1"),
        };
        let a = SegmentCheckpoint::new(
            1,
            "commit-1".into(),
            "c".repeat(64),
            vec![committed.clone(), unknown],
        )?;
        let b = SegmentCheckpoint::new(2, "commit-2".into(), "d".repeat(64), vec![committed])?;
        let rebuilt = rebuild(&[a, b], &Verify(true))?;
        assert_eq!(rebuilt.committed_updates, 1);
        assert_eq!(rebuilt.unknown, 1);
        Ok(())
    }

    #[test]
    fn unsupported_or_conflicting_commit_proof_is_held_or_refused() -> Result<()> {
        let evidence = CheckpointEvidence::Committed(Box::new(update()));
        let a =
            SegmentCheckpoint::new(1, "commit-1".into(), "c".repeat(64), vec![evidence.clone()])?;
        assert_eq!(
            rebuild(std::slice::from_ref(&a), &Verify(false))?.held_committed,
            1
        );
        let mut different = update();
        different.occurrence = reference("occurrence/other");
        let b = SegmentCheckpoint::new(
            2,
            "commit-2".into(),
            "d".repeat(64),
            vec![CheckpointEvidence::Committed(Box::new(different))],
        )?;
        assert!(rebuild(&[a, b], &Verify(true)).is_err());
        Ok(())
    }
}
