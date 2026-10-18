//! Offline model of exact-version dependencies and retention promises.
//!
//! This module returns review data only. It cannot publish, move, or remove stored objects.

use crate::canon::canonical_sha256;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const OBLIGATION_PLAN_SCHEMA: &str = "jope.retention-plan/v2";

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
pub enum VersionId {
    Explicit(String),
    Null,
    Unknown,
}

/// Identity of one physical stored version. Content identity never substitutes for version identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct StrongPhysicalRef {
    pub account: String,
    pub bucket: String,
    pub key: String,
    pub version_id: VersionId,
    pub sha256: String,
    pub bytes: u64,
}

impl StrongPhysicalRef {
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
    pub schema: String,
    pub identity: String,
    pub sha256: String,
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
pub enum PromiseKind {
    ExactResume,
    Rollback,
    ExactRetraining,
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
    pub source: StrongPhysicalRef,
    pub replacement: StrongPhysicalRef,
    pub capability: String,
    pub runtime_tested: bool,
    pub receipt_sha256: String,
    pub pin_generation: String,
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
    pub id: String,
    pub kind: PromiseKind,
    pub root_id: String,
    pub active: bool,
    pub dependencies: Vec<StrongPhysicalRef>,
    pub restore_proofs: Vec<RuntimeRestoreProof>,
    pub released_by_transition: Option<String>,
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
pub struct RetiredVersion {
    pub object: StrongPhysicalRef,
    pub retired_at_pin_generation: String,
}

/// Atomic view of root and pin authorities used to fence stale publishers.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PinGeneration {
    pub root_generation: String,
    pub pin_generation: String,
    pub observed_root_generation: String,
    pub observed_pin_generation: String,
    pub closure_complete: bool,
    pub registered_publishers: BTreeSet<String>,
    pub observed_publishers: BTreeSet<String>,
    pub unresolved: Vec<String>,
    pub strong_dependencies: BTreeMap<StrongPhysicalRef, BTreeSet<String>>,
    pub provenance: BTreeMap<StrongPhysicalRef, BTreeSet<ProvenanceRef>>,
    pub promises: Vec<RecoveryPromise>,
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
pub enum TagAssessment {
    Verified { required_tags_present: bool },
    Unknown { reason: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LifecycleAssessment {
    pub configuration_sha256: Option<String>,
    pub complete: bool,
    pub matching_current_expiration_rules: Vec<String>,
    pub matching_noncurrent_expiration_rules: Vec<String>,
    pub unsupported_fields: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub object: StrongPhysicalRef,
    pub tags: TagAssessment,
    pub lifecycle: LifecycleAssessment,
    pub unknowns: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Keep,
    Hold,
    ReviewAfterRootTransition,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObligationRow {
    pub object: StrongPhysicalRef,
    pub disposition: Disposition,
    pub strong_roots: Vec<String>,
    pub promises: Vec<String>,
    pub provenance: Vec<ProvenanceRef>,
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObligationPlanV2 {
    pub schema: String,
    pub root_generation: String,
    pub pin_generation: String,
    pub rows: Vec<ObligationRow>,
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
