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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
/// `VersionId` is an enum for the version ID contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum VersionId {
    /// The `Explicit` case of `VersionId`.
    Explicit(String),
    /// The `Null` case of `VersionId`.
    Null,
    /// The `Unknown` case of `VersionId`.
    Unknown,
}

/// Identity of one physical stored version. Content identity never substitutes for version identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct StrongPhysicalRef {
    /// Account recorded for this strong physical ref.
    pub account: String,
    /// Bucket recorded for this strong physical ref.
    pub bucket: String,
    /// Key recorded for this strong physical ref.
    pub key: String,
    /// Identifier for the associated version.
    pub version_id: VersionId,
    /// Sha-256 recorded for this strong physical ref.
    pub sha256: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
}

impl StrongPhysicalRef {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.account.is_empty() && !self.bucket.is_empty() && !self.key.is_empty(),
            "physical reference needs account, bucket, and key"
        );
        ensure!(
            self.account.len() == 12 && self.account.bytes().all(|b| b.is_ascii_digit()),
            "physical reference account must be a twelve-digit identity"
        );
        if let VersionId::Explicit(id) = &self.version_id {
            ensure!(!id.is_empty(), "explicit version id is empty");
        }
        ensure!(
            valid_digest(&self.sha256),
            "invalid physical content digest"
        );
        Ok(())
    }

    fn exact_version(&self) -> bool {
        matches!(self.version_id, VersionId::Explicit(_))
    }
}

/// Historical identity metadata. By itself, it does not keep physical bytes pinned.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceRef {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identity recorded for this provenance ref.
    pub identity: String,
    /// Sha-256 recorded for this provenance ref.
    pub sha256: String,
    /// Description recorded for this provenance ref.
    pub description: String,
}

impl ProvenanceRef {
    fn validate(&self) -> Result<()> {
        ensure!(!self.schema.is_empty(), "provenance schema is missing");
        ensure!(!self.identity.is_empty(), "provenance identity is missing");
        ensure!(valid_digest(&self.sha256), "invalid provenance digest");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
/// `PromiseKind` is an enum for the promise kind contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum PromiseKind {
    /// The `ExactResume` case of `PromiseKind`.
    ExactResume,
    /// The `Rollback` case of `PromiseKind`.
    Rollback,
    /// The `ExactRetraining` case of `PromiseKind`.
    ExactRetraining,
    /// The `ClaimReproduction` case of `PromiseKind`.
    ClaimReproduction,
}

impl PromiseKind {
    fn restore_capability(self) -> &'static str {
        match self {
            Self::ExactResume => "resume",
            Self::Rollback => "rollback",
            Self::ExactRetraining => "retrain",
            Self::ClaimReproduction => "reproduce_claim",
        }
    }
}

/// Runtime evidence for replacing one physical dependency while fulfilling the same promise.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRestoreProof {
    /// Provenance source for this value.
    pub source: StrongPhysicalRef,
    /// Replacement recorded for this runtime restore proof.
    pub replacement: StrongPhysicalRef,
    /// Capability recorded for this runtime restore proof.
    pub capability: String,
    /// Runtime tested recorded for this runtime restore proof.
    pub runtime_tested: bool,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Pin generation recorded for this runtime restore proof.
    pub pin_generation: String,
    /// Readers migrated recorded for this runtime restore proof.
    pub readers_migrated: bool,
}

impl RuntimeRestoreProof {
    fn validates_for(
        &self,
        kind: PromiseKind,
        source: &StrongPhysicalRef,
        generation: &str,
    ) -> bool {
        self.source == *source
            && self.replacement.exact_version()
            && self.capability == kind.restore_capability()
            && self.runtime_tested
            && self.readers_migrated
            && self.pin_generation == generation
            && valid_digest(&self.receipt_sha256)
            && valid_digest(&self.replacement.sha256)
    }
}

/// An active promise protects the complete declared dependency set until a tested migration or
/// an explicit generation-bound release is present.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPromise {
    /// Stable identifier for this value.
    pub id: String,
    /// Record kind or category represented by this value.
    pub kind: PromiseKind,
    /// Identifier for the associated root.
    pub root_id: String,
    /// Active recorded for this recovery promise.
    pub active: bool,
    /// Collection of dependencies values carried by this recovery promise.
    pub dependencies: Vec<StrongPhysicalRef>,
    /// Collection of restore proofs values carried by this recovery promise.
    pub restore_proofs: Vec<RuntimeRestoreProof>,
    /// Optional released by transition value; `None` means it is not available.
    pub released_by_transition: Option<String>,
    /// Optional released at generation value; `None` means it is not available.
    pub released_at_generation: Option<String>,
}

impl RecoveryPromise {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.is_empty() && !self.root_id.is_empty(),
            "promise identity is missing"
        );
        ensure!(
            !self.dependencies.is_empty(),
            "promise dependency closure is empty"
        );
        for dep in &self.dependencies {
            dep.validate()?;
        }
        for proof in &self.restore_proofs {
            proof.source.validate()?;
            proof.replacement.validate()?;
            ensure!(
                valid_digest(&proof.receipt_sha256),
                "invalid restore receipt digest"
            );
        }
        if !self.active {
            ensure!(
                self.released_by_transition
                    .as_deref()
                    .is_some_and(|s| !s.is_empty())
                    && self
                        .released_at_generation
                        .as_deref()
                        .is_some_and(|s| !s.is_empty()),
                "inactive promise requires a recorded root transition"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `RetiredVersion` is a record for the retired version contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct RetiredVersion {
    /// Object recorded for this retired version.
    pub object: StrongPhysicalRef,
    /// Retired at pin generation recorded for this retired version.
    pub retired_at_pin_generation: String,
}

/// Atomic view of root and pin authorities used to fence stale publishers.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PinGeneration {
    /// Root generation recorded for this pin generation.
    pub root_generation: String,
    /// Pin generation recorded for this pin generation.
    pub pin_generation: String,
    /// Observed root generation recorded for this pin generation.
    pub observed_root_generation: String,
    /// Observed pin generation recorded for this pin generation.
    pub observed_pin_generation: String,
    /// Closure complete recorded for this pin generation.
    pub closure_complete: bool,
    /// Registered publishers recorded for this pin generation.
    pub registered_publishers: BTreeSet<String>,
    /// Observed publishers recorded for this pin generation.
    pub observed_publishers: BTreeSet<String>,
    /// Collection of unresolved values carried by this pin generation.
    pub unresolved: Vec<String>,
    /// Mapping of strong dependencies values carried by this pin generation.
    pub strong_dependencies: BTreeMap<StrongPhysicalRef, BTreeSet<String>>,
    /// Mapping of provenance values carried by this pin generation.
    pub provenance: BTreeMap<StrongPhysicalRef, BTreeSet<ProvenanceRef>>,
    /// Collection of promises values carried by this pin generation.
    pub promises: Vec<RecoveryPromise>,
    /// Mapping of retired versions values carried by this pin generation.
    pub retired_versions: BTreeMap<StrongPhysicalRef, RetiredVersion>,
}

impl PinGeneration {
    fn gaps(&self) -> Vec<String> {
        let mut gaps = self.unresolved.clone();
        if self.root_generation.is_empty() || self.pin_generation.is_empty() {
            gaps.push("root or pin generation identity is unknown".into());
        }
        if self.observed_root_generation != self.root_generation
            || self.observed_pin_generation != self.pin_generation
        {
            gaps.push("observed root/pin generation changed".into());
        }
        if !self.closure_complete {
            gaps.push("authoritative root closure is incomplete".into());
        }
        for publisher in self
            .registered_publishers
            .difference(&self.observed_publishers)
        {
            gaps.push(format!(
                "registered publisher is not quiesced or observed: {publisher}"
            ));
        }
        gaps.sort();
        gaps.dedup();
        gaps
    }

    /// Performs the validate publish operation and returns an error if it cannot be completed.
    pub fn validate_publish(
        &self,
        publisher_id: &str,
        publisher_pin_generation: &str,
        refs: &[StrongPhysicalRef],
    ) -> Result<()> {
        ensure!(
            self.gaps().is_empty(),
            "cannot validate publisher against an incomplete pin generation"
        );
        ensure!(
            publisher_pin_generation == self.pin_generation,
            "stale publisher pin generation"
        );
        ensure!(
            self.registered_publishers.contains(publisher_id)
                && self.observed_publishers.contains(publisher_id),
            "publisher is not registered and observed in this pin generation"
        );
        for reference in refs {
            reference.validate()?;
            ensure!(
                !self.retired_versions.contains_key(reference),
                "publisher cannot restore a retired physical version"
            );
        }
        Ok(())
    }

    fn promise_pins(&self) -> Result<BTreeMap<StrongPhysicalRef, BTreeSet<String>>> {
        let mut pinned = BTreeMap::<StrongPhysicalRef, BTreeSet<String>>::new();
        for promise in &self.promises {
            promise.validate()?;
            if !promise.active {
                ensure!(
                    promise.released_at_generation.as_deref()
                        == Some(self.root_generation.as_str()),
                    "released promise does not name the current root generation"
                );
                continue;
            }
            let mut replaced = BTreeSet::new();
            for dependency in &promise.dependencies {
                if let Some(proof) = promise
                    .restore_proofs
                    .iter()
                    .find(|p| p.validates_for(promise.kind, dependency, &self.pin_generation))
                {
                    replaced.insert(dependency.clone());
                    pinned
                        .entry(proof.replacement.clone())
                        .or_default()
                        .insert(promise.id.clone());
                }
            }
            for dependency in promise
                .dependencies
                .iter()
                .filter(|dep| !replaced.contains(*dep))
            {
                pinned
                    .entry(dependency.clone())
                    .or_default()
                    .insert(promise.id.clone());
            }
        }
        Ok(pinned)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
/// `TagAssessment` is an enum for the tag assessment contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum TagAssessment {
    /// The `Verified` case of `TagAssessment`.
    Verified {
        /// Required tags present recorded for this outcome.
        required_tags_present: bool,
    },
    /// The `Unknown` case of `TagAssessment`.
    Unknown {
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `LifecycleAssessment` is a record for the lifecycle assessment contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct LifecycleAssessment {
    /// SHA-256 digest identifying the associated configuration.
    pub configuration_sha256: Option<String>,
    /// Complete recorded for this lifecycle assessment.
    pub complete: bool,
    /// Collection of matching current expiration rules values carried by this lifecycle assessment.
    pub matching_current_expiration_rules: Vec<String>,
    /// Collection of matching noncurrent expiration rules values carried by this lifecycle assessment.
    pub matching_noncurrent_expiration_rules: Vec<String>,
    /// Collection of unsupported fields values carried by this lifecycle assessment.
    pub unsupported_fields: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `Candidate` is a record for the candidate contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct Candidate {
    /// Object recorded for this candidate.
    pub object: StrongPhysicalRef,
    /// Tags recorded for this candidate.
    pub tags: TagAssessment,
    /// Lifecycle recorded for this candidate.
    pub lifecycle: LifecycleAssessment,
    /// Collection of unknowns values carried by this candidate.
    pub unknowns: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// `Disposition` is an enum for the disposition contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum Disposition {
    /// The `Keep` case of `Disposition`.
    Keep,
    /// The `Hold` case of `Disposition`.
    Hold,
    /// The `ReviewAfterRootTransition` case of `Disposition`.
    ReviewAfterRootTransition,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ObligationRow` is a record for the obligation row contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct ObligationRow {
    /// Object recorded for this obligation row.
    pub object: StrongPhysicalRef,
    /// Disposition recorded for this obligation row.
    pub disposition: Disposition,
    /// Collection of strong roots values carried by this obligation row.
    pub strong_roots: Vec<String>,
    /// Collection of promises values carried by this obligation row.
    pub promises: Vec<String>,
    /// Collection of provenance values carried by this obligation row.
    pub provenance: Vec<ProvenanceRef>,
    /// Collection of unresolved values carried by this obligation row.
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ObligationPlanV2` is a record for the obligation plan v2 contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct ObligationPlanV2 {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Root generation recorded for this obligation plan v2.
    pub root_generation: String,
    /// Pin generation recorded for this obligation plan v2.
    pub pin_generation: String,
    /// Number of rows represented by this value.
    pub rows: Vec<ObligationRow>,
    /// SHA-256 digest identifying the associated plan.
    pub plan_sha256: String,
}

impl ObligationPlanV2 {
    /// Evaluate supplied exact-version candidates against one pinned authority snapshot.
    pub fn build(snapshot: &PinGeneration, mut candidates: Vec<Candidate>) -> Result<Self> {
        let promise_pins = snapshot.promise_pins()?;
        for (object, owners) in &snapshot.strong_dependencies {
            object.validate()?;
            ensure!(!owners.is_empty(), "strong dependency has no owning root");
        }
        for (object, references) in &snapshot.provenance {
            object.validate()?;
            for reference in references {
                reference.validate()?;
            }
        }
        for (object, retired) in &snapshot.retired_versions {
            object.validate()?;
            ensure!(
                &retired.object == object && object.exact_version(),
                "retired registry key must identify one exact physical version"
            );
            ensure!(
                !retired.retired_at_pin_generation.is_empty(),
                "retired registry entry has no fencing generation"
            );
        }
        candidates.sort_by(|a, b| a.object.cmp(&b.object));
        let mut seen = BTreeSet::new();
        let generation_gaps = snapshot.gaps();
        let mut rows = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            candidate.object.validate()?;
            ensure!(
                seen.insert(candidate.object.clone()),
                "duplicate candidate physical version"
            );
            let mut unresolved = generation_gaps.clone();
            unresolved.extend(
                candidate
                    .unknowns
                    .iter()
                    .map(|u| format!("unknown configuration: {u}")),
            );
            if !candidate.object.exact_version() {
                unresolved.push("physical version is null or unknown".into());
            }
            match &candidate.tags {
                TagAssessment::Verified {
                    required_tags_present: false,
                } => {
                    unresolved.push(
                        "required object tags are missing; tag state cannot establish safety"
                            .into(),
                    );
                }
                TagAssessment::Unknown { reason } => {
                    unresolved.push(format!("tag configuration is unknown: {reason}"));
                }
                TagAssessment::Verified {
                    required_tags_present: true,
                } => {}
            }
            if !candidate.lifecycle.complete {
                unresolved.push("lifecycle configuration is incomplete".into());
            }
            if candidate
                .lifecycle
                .configuration_sha256
                .as_deref()
                .is_none_or(|s| !valid_digest(s))
            {
                unresolved.push("lifecycle configuration identity is unknown".into());
            }
            unresolved.extend(
                candidate
                    .lifecycle
                    .unsupported_fields
                    .iter()
                    .map(|f| format!("unsupported lifecycle field: {f}")),
            );
            unresolved.extend(
                candidate
                    .lifecycle
                    .matching_current_expiration_rules
                    .iter()
                    .map(|r| format!("matching current-version lifecycle expiration: {r}")),
            );
            unresolved.extend(
                candidate
                    .lifecycle
                    .matching_noncurrent_expiration_rules
                    .iter()
                    .map(|r| format!("matching noncurrent-version lifecycle expiration: {r}")),
            );
            let mut strong_roots = snapshot
                .strong_dependencies
                .get(&candidate.object)
                .cloned()
                .unwrap_or_default();
            let promises = promise_pins
                .get(&candidate.object)
                .cloned()
                .unwrap_or_default();
            for promise in snapshot
                .promises
                .iter()
                .filter(|p| promises.contains(&p.id))
            {
                strong_roots.insert(promise.root_id.clone());
            }
            let retired = snapshot.retired_versions.contains_key(&candidate.object);
            if retired {
                unresolved.push("candidate is already retired in the pinned authority".into());
            }
            unresolved.sort();
            unresolved.dedup();
            let disposition = if !unresolved.is_empty() {
                Disposition::Hold
            } else if !strong_roots.is_empty() || !promises.is_empty() {
                Disposition::Keep
            } else {
                Disposition::ReviewAfterRootTransition
            };
            rows.push(ObligationRow {
                provenance: snapshot
                    .provenance
                    .get(&candidate.object)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                object: candidate.object,
                disposition,
                strong_roots: strong_roots.into_iter().collect(),
                promises: promises.into_iter().collect(),
                unresolved,
            });
        }
        let mut plan = Self {
            schema: OBLIGATION_PLAN_SCHEMA.into(),
            root_generation: snapshot.root_generation.clone(),
            pin_generation: snapshot.pin_generation.clone(),
            rows,
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = canonical_sha256(&plan)?;
        Ok(plan)
    }
}
