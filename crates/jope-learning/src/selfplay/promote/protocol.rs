use super::operations::{is_digest, require_digest};
use super::*;

/// Schema identifier for publication values.
pub const PUBLICATION_SCHEMA: &str = "jope.accepted-root-publication/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `PointerKind` is an enum for the pointer kind contract in this module: Accepted-root publication protocol.
pub enum PointerKind {
    /// The `ResearchParent` case of `PointerKind`.
    ResearchParent,
    /// The `Serving` case of `PointerKind`.
    Serving,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `GrantPurpose` is an enum for the grant purpose contract in this module: Accepted-root publication protocol.
pub enum GrantPurpose {
    /// The `ResearchParentAdoption` case of `GrantPurpose`.
    ResearchParentAdoption,
    /// The `ServingPromotion` case of `GrantPurpose`.
    ServingPromotion,
    /// The `Rollback` case of `GrantPurpose`.
    Rollback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `Assessment` is an enum for the assessment contract in this module: Accepted-root publication protocol.
pub enum Assessment {
    /// The `Qualified` case of `Assessment`.
    Qualified,
    /// The `Negative` case of `Assessment`.
    Negative,
    /// The `Inconclusive` case of `Assessment`.
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PublisherGrant` is a record for the publisher grant contract in this module: Accepted-root publication protocol.
pub struct PublisherGrant {
    /// Digest of the signature-verified grant supplied by the authority adapter.
    pub verified_grant_sha256: String,
    /// Purpose recorded for this publisher grant.
    pub purpose: GrantPurpose,
    /// Expected generation recorded for this publisher grant.
    pub expected_generation: u64,
    /// SHA-256 digest identifying the associated expected parent.
    pub expected_parent_sha256: Option<String>,
    /// Expires at ms recorded for this publisher grant.
    pub expires_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ClosureProof` is a record for the closure proof contract in this module: Accepted-root publication protocol.
pub struct ClosureProof {
    /// SHA-256 digest identifying the associated closure.
    pub closure_sha256: String,
    /// Required objects recorded for this closure proof.
    pub required_objects: BTreeSet<String>,
    /// Remote objects recorded for this closure proof.
    pub remote_objects: BTreeSet<String>,
    /// Tombstoned/retired strong references may not be used to satisfy closure.
    pub retired_strong_refs: BTreeSet<String>,
}

impl ClosureProof {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        require_digest(&self.closure_sha256, "closure digest")?;
        ensure!(
            !self.required_objects.is_empty(),
            "compatible closure has no required objects"
        );
        ensure!(
            self.required_objects.iter().all(|id| is_digest(id)),
            "required closure object pin is invalid"
        );
        ensure!(
            self.remote_objects.iter().all(|id| is_digest(id)),
            "remote closure object pin is invalid"
        );
        ensure!(
            self.retired_strong_refs.iter().all(|id| is_digest(id)),
            "retired strong reference pin is invalid"
        );
        ensure!(
            self.required_objects.is_subset(&self.remote_objects),
            "remote closure is missing a required object"
        );
        ensure!(
            self.required_objects.is_disjoint(&self.retired_strong_refs),
            "required closure contains a retired strong reference"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RollbackRoot` is a record for the rollback root contract in this module: Accepted-root publication protocol.
pub struct RollbackRoot {
    /// Generation recorded for this rollback root.
    pub generation: u64,
    /// SHA-256 digest identifying the associated root.
    pub root_sha256: String,
    /// SHA-256 digest identifying the associated closure.
    pub closure_sha256: String,
    /// Required objects recorded for this rollback root.
    pub required_objects: BTreeSet<String>,
}

impl RollbackRoot {
    fn validate_before(&self, next_generation: u64) -> Result<()> {
        require_digest(&self.root_sha256, "rollback root")?;
        require_digest(&self.closure_sha256, "rollback closure")?;
        ensure!(
            self.generation < next_generation,
            "rollback root must precede the next generation"
        );
        ensure!(
            self.required_objects.iter().all(|id| is_digest(id)),
            "rollback object pin is invalid"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RootPointer` is a record for the root pointer contract in this module: Accepted-root publication protocol.
pub struct RootPointer {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Monotone pointer revision. Rollback advances this revision while restoring an older root.
    pub generation: u64,
    /// SHA-256 digest identifying the associated root.
    pub root_sha256: String,
    /// SHA-256 digest identifying the associated parent root.
    pub parent_root_sha256: Option<String>,
    /// SHA-256 digest identifying the associated decision.
    pub decision_sha256: String,
    /// SHA-256 digest identifying the associated closure.
    pub closure_sha256: String,
    /// Rollback recorded for this root pointer.
    pub rollback: RollbackRoot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `VersionedPointer` is a record for the versioned pointer contract in this module: Accepted-root publication protocol.
pub struct VersionedPointer {
    /// State pointer associated with this record.
    pub pointer: RootPointer,
    /// Store version recorded for this versioned pointer.
    pub store_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `CheckpointCommit` is a record for the checkpoint commit contract in this module: Accepted-root publication protocol.
pub struct CheckpointCommit {
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated candidate evidence.
    pub candidate_evidence_sha256: String,
    /// SHA-256 digest identifying the associated closure.
    pub closure_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ResearchParentDecision` is a record for the research parent decision contract in this module: Accepted-root publication protocol.
pub struct ResearchParentDecision {
    /// Expected generation recorded for this research parent decision.
    pub expected_generation: u64,
    /// SHA-256 digest identifying the associated expected parent.
    pub expected_parent_sha256: Option<String>,
    /// SHA-256 digest identifying the associated candidate root.
    pub candidate_root_sha256: String,
    /// SHA-256 digest identifying the associated candidate evidence.
    pub candidate_evidence_sha256: String,
    /// Assessment recorded for this research parent decision.
    pub assessment: Assessment,
    /// SHA-256 digest identifying the associated compatible closure.
    pub compatible_closure_sha256: String,
    /// Closure recorded for this research parent decision.
    pub closure: ClosureProof,
    /// Rollback recorded for this research parent decision.
    pub rollback: RollbackRoot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ServingPromotionDecision` is a record for the serving promotion decision contract in this module: Accepted-root publication protocol.
pub struct ServingPromotionDecision {
    /// Expected generation recorded for this serving promotion decision.
    pub expected_generation: u64,
    /// SHA-256 digest identifying the associated expected parent.
    pub expected_parent_sha256: Option<String>,
    /// SHA-256 digest identifying the associated expected research parent.
    pub expected_research_parent_sha256: String,
    /// SHA-256 digest identifying the associated candidate root.
    pub candidate_root_sha256: String,
    /// SHA-256 digest identifying the associated candidate evidence.
    pub candidate_evidence_sha256: String,
    /// Independent serving qualification receipt; absent means no promotion authority.
    pub qualification_sha256: Option<String>,
    /// Assessment recorded for this serving promotion decision.
    pub assessment: Assessment,
    /// SHA-256 digest identifying the associated compatible closure.
    pub compatible_closure_sha256: String,
    /// Closure recorded for this serving promotion decision.
    pub closure: ClosureProof,
    /// Rollback recorded for this serving promotion decision.
    pub rollback: RollbackRoot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
/// `PublicationEvent` is an enum for the publication event contract in this module: Accepted-root publication protocol.
pub enum PublicationEvent {
    /// The `CheckpointCommitted` case of `PublicationEvent`.
    CheckpointCommitted {
        /// Checkpoint recorded for this publication event.
        checkpoint: CheckpointCommit,
    },
    /// The `ResearchParentAssessed` case of `PublicationEvent`.
    ResearchParentAssessed {
        /// SHA-256 digest identifying the associated decision.
        decision_sha256: String,
        /// Decision recorded for this publication event.
        decision: ResearchParentDecision,
    },
    /// The `ServingAssessed` case of `PublicationEvent`.
    ServingAssessed {
        /// SHA-256 digest identifying the associated decision.
        decision_sha256: String,
        /// Decision recorded for this publication event.
        decision: ServingPromotionDecision,
    },
    /// The `RootRolledBack` case of `PublicationEvent`.
    RootRolledBack {
        /// SHA-256 digest identifying the associated decision.
        decision_sha256: String,
        /// Record kind or category represented by this value.
        kind: PointerKind,
        /// Previous root recorded for this publication event.
        previous_root: String,
        /// Rollback recorded for this publication event.
        rollback: RollbackRoot,
    },
}

/// The adapter exposes no deletion operation: immutable events/artifacts survive every pointer
/// update. `compare_exchange` must be backed by a real conditional write in a production adapter.
pub trait PublicationStore {
    /// Reads pointer from the configured source.
    fn read_pointer(&self, kind: PointerKind) -> Result<Option<VersionedPointer>>;
    /// Performs the append event operation and returns an error if it cannot be completed.
    fn append_event(&self, event: PublicationEvent) -> Result<()>;
    /// Performs the compare exchange operation and returns an error if it cannot be completed.
    fn compare_exchange(
        &self,
        kind: PointerKind,
        expected_store_version: Option<&str>,
        next: RootPointer,
    ) -> Result<bool>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `AdoptionResult` is an enum for the adoption result contract in this module: Accepted-root publication protocol.
pub enum AdoptionResult {
    /// The `Advanced` case of `AdoptionResult`.
    Advanced(RootPointer),
    /// The `Retained` case of `AdoptionResult`.
    Retained {
        /// Optional current value; `None` means it is not available.
        current: Option<RootPointer>,
        /// SHA-256 digest identifying the associated evidence.
        evidence_sha256: String,
        /// Assessment recorded for this adoption result.
        assessment: Assessment,
    },
}
