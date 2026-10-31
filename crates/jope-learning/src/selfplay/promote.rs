//! Accepted-root publication protocol.
//!
//! This module is a storage-agnostic policy boundary. Callers verify signed grants and build
//! current remote-closure proofs; the store adapter must provide atomic compare-and-swap and
//! append-only event storage. Checkpoint commitment, research-parent adoption, serving promotion
//! and rollback are distinct operations.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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

/// Performs the record checkpoint commit operation and returns an error if it cannot be completed.
pub fn record_checkpoint_commit(
    store: &impl PublicationStore,
    commit: CheckpointCommit,
) -> Result<()> {
    require_digest(&commit.checkpoint_sha256, "checkpoint")?;
    require_digest(&commit.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&commit.closure_sha256, "checkpoint closure")?;
    store.append_event(PublicationEvent::CheckpointCommitted { checkpoint: commit })
}

/// Performs the adopt research parent operation and returns an error if it cannot be completed.
pub fn adopt_research_parent(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    decision: ResearchParentDecision,
    now_ms: u64,
) -> Result<AdoptionResult> {
    validate_research_decision(&decision)?;
    let decision_sha256 = crate::canon::canonical_sha256(&decision)?;
    // The assessment and evidence pin are durably recorded before any attempted pointer move.
    store.append_event(PublicationEvent::ResearchParentAssessed {
        decision_sha256: decision_sha256.clone(),
        decision: decision.clone(),
    })?;
    validate_grant(
        grant,
        GrantPurpose::ResearchParentAdoption,
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
        now_ms,
    )?;
    let observed = store.read_pointer(PointerKind::ResearchParent)?;
    ensure_expected_parent(
        observed.as_ref(),
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
    )?;
    if decision.assessment != Assessment::Qualified {
        return Ok(AdoptionResult::Retained {
            current: observed.map(|v| v.pointer),
            evidence_sha256: decision.candidate_evidence_sha256,
            assessment: decision.assessment,
        });
    }
    validate_closure(&decision.closure, &decision.compatible_closure_sha256)?;
    ensure!(
        decision
            .rollback
            .required_objects
            .is_subset(&decision.closure.required_objects),
        "compatible closure omits rollback pins"
    );
    ensure!(
        decision.expected_parent_sha256.as_deref() == Some(decision.rollback.root_sha256.as_str()),
        "rollback root is not the exact current parent"
    );
    let next_generation = decision
        .expected_generation
        .checked_add(1)
        .context("research-parent generation exhausted")?;
    decision.rollback.validate_before(next_generation)?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: decision.candidate_root_sha256,
        parent_root_sha256: decision.expected_parent_sha256,
        decision_sha256,
        closure_sha256: decision.compatible_closure_sha256,
        rollback: decision.rollback,
    };
    compare_and_advance(store, PointerKind::ResearchParent, observed, next)
}

/// Performs the promote serving operation and returns an error if it cannot be completed.
pub fn promote_serving(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    decision: ServingPromotionDecision,
    now_ms: u64,
) -> Result<AdoptionResult> {
    validate_serving_decision(&decision)?;
    let decision_sha256 = crate::canon::canonical_sha256(&decision)?;
    store.append_event(PublicationEvent::ServingAssessed {
        decision_sha256: decision_sha256.clone(),
        decision: decision.clone(),
    })?;
    validate_grant(
        grant,
        GrantPurpose::ServingPromotion,
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
        now_ms,
    )?;
    let observed = store.read_pointer(PointerKind::Serving)?;
    ensure_expected_parent(
        observed.as_ref(),
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
    )?;
    let research = store.read_pointer(PointerKind::ResearchParent)?;
    ensure!(
        research
            .as_ref()
            .is_some_and(|r| r.pointer.root_sha256 == decision.expected_research_parent_sha256),
        "research parent is not the independently adopted root"
    );
    if decision.assessment != Assessment::Qualified {
        return Ok(AdoptionResult::Retained {
            current: observed.map(|v| v.pointer),
            evidence_sha256: decision.candidate_evidence_sha256,
            assessment: decision.assessment,
        });
    }
    ensure!(
        decision.qualification_sha256.is_some(),
        "serving promotion needs independent qualification evidence"
    );
    validate_closure(&decision.closure, &decision.compatible_closure_sha256)?;
    ensure!(
        decision
            .rollback
            .required_objects
            .is_subset(&decision.closure.required_objects),
        "compatible closure omits rollback pins"
    );
    ensure!(
        decision.expected_parent_sha256.as_deref() == Some(decision.rollback.root_sha256.as_str()),
        "rollback root is not the exact current parent"
    );
    let next_generation = decision
        .expected_generation
        .checked_add(1)
        .context("serving generation exhausted")?;
    decision.rollback.validate_before(next_generation)?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: decision.candidate_root_sha256,
        parent_root_sha256: decision.expected_parent_sha256,
        decision_sha256,
        closure_sha256: decision.compatible_closure_sha256,
        rollback: decision.rollback,
    };
    compare_and_advance(store, PointerKind::Serving, observed, next)
}

/// Performs the rollback root operation and returns an error if it cannot be completed.
pub fn rollback_root(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    kind: PointerKind,
    rollback: RollbackRoot,
    closure: ClosureProof,
    decision_evidence_sha256: String,
    now_ms: u64,
) -> Result<RootPointer> {
    require_digest(&decision_evidence_sha256, "rollback decision evidence")?;
    let observed = store.read_pointer(kind)?;
    let current = observed
        .as_ref()
        .context("cannot roll back an absent root")?;
    validate_grant(
        grant,
        GrantPurpose::Rollback,
        current.pointer.generation,
        Some(&current.pointer.root_sha256),
        now_ms,
    )?;
    rollback.validate_before(current.pointer.generation)?;
    validate_closure(&closure, &rollback.closure_sha256)?;
    ensure!(
        rollback
            .required_objects
            .is_subset(&closure.required_objects),
        "rollback proof omits retained rollback pins"
    );
    let decision_sha256 = crate::canon::canonical_sha256(&(
        &kind,
        &current.pointer.root_sha256,
        &rollback,
        &decision_evidence_sha256,
    ))?;
    store.append_event(PublicationEvent::RootRolledBack {
        decision_sha256: decision_sha256.clone(),
        kind,
        previous_root: current.pointer.root_sha256.clone(),
        rollback: rollback.clone(),
    })?;
    let next_generation = current
        .pointer
        .generation
        .checked_add(1)
        .context("pointer generation exhausted")?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: rollback.root_sha256.clone(),
        parent_root_sha256: Some(current.pointer.root_sha256.clone()),
        decision_sha256,
        closure_sha256: rollback.closure_sha256.clone(),
        rollback,
    };
    match store.compare_exchange(kind, Some(&current.store_version), next.clone())? {
        true => Ok(next),
        false => bail!("rollback CAS lost; accepted root remains unchanged"),
    }
}

fn validate_research_decision(d: &ResearchParentDecision) -> Result<()> {
    require_digest(&d.candidate_root_sha256, "candidate root")?;
    require_digest(&d.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&d.compatible_closure_sha256, "compatible closure")?;
    if let Some(parent) = &d.expected_parent_sha256 {
        require_digest(parent, "expected parent")?;
    }
    Ok(())
}

fn validate_serving_decision(d: &ServingPromotionDecision) -> Result<()> {
    require_digest(&d.candidate_root_sha256, "candidate root")?;
    require_digest(&d.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&d.expected_research_parent_sha256, "research parent")?;
    require_digest(&d.compatible_closure_sha256, "compatible closure")?;
    if let Some(parent) = &d.expected_parent_sha256 {
        require_digest(parent, "expected serving parent")?;
    }
    if let Some(qualification) = &d.qualification_sha256 {
        require_digest(qualification, "serving qualification")?;
    }
    Ok(())
}

fn validate_closure(closure: &ClosureProof, expected_sha256: &str) -> Result<()> {
    closure.validate()?;
    ensure!(
        closure.closure_sha256 == expected_sha256,
        "compatible closure snapshot changed"
    );
    Ok(())
}

fn validate_grant(
    grant: &PublisherGrant,
    purpose: GrantPurpose,
    generation: u64,
    parent: Option<&str>,
    now_ms: u64,
) -> Result<()> {
    require_digest(&grant.verified_grant_sha256, "verified grant")?;
    ensure!(grant.purpose == purpose, "grant purpose mismatch");
    ensure!(
        grant.expires_at_ms > now_ms,
        "publisher grant is stale/expired"
    );
    ensure!(
        grant.expected_generation == generation
            && grant.expected_parent_sha256.as_deref() == parent,
        "grant is not bound to expected parent/generation"
    );
    Ok(())
}

fn ensure_expected_parent(
    observed: Option<&VersionedPointer>,
    generation: u64,
    parent: Option<&str>,
) -> Result<()> {
    match observed {
        Some(value) => ensure!(
            value.pointer.generation == generation
                && Some(value.pointer.root_sha256.as_str()) == parent,
            "accepted root changed; stale parent/generation"
        ),
        None => ensure!(
            generation == 0 && parent.is_none(),
            "expected root is absent or stale"
        ),
    }
    Ok(())
}

fn compare_and_advance(
    store: &impl PublicationStore,
    kind: PointerKind,
    observed: Option<VersionedPointer>,
    next: RootPointer,
) -> Result<AdoptionResult> {
    let expected_version = observed.as_ref().map(|v| v.store_version.as_str());
    match store.compare_exchange(kind, expected_version, next.clone())? {
        true => Ok(AdoptionResult::Advanced(next)),
        false => bail!("publisher CAS lost; accepted root remains unchanged"),
    }
}

fn require_digest(value: &str, name: &str) -> Result<()> {
    ensure!(is_digest(value), "{name} must be a lowercase SHA256 digest");
    Ok(())
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
