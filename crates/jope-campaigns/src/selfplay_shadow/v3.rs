use super::{
    build_causal_evidence_ledger_v2, CausalEvaluationReceiptV2, CausalEvidenceLedgerV2,
    Sha256Digest, Task, TaskCausalEvidenceV2, CAUSAL_EVIDENCE_EPOCH_SCHEMA,
    CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA, CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA,
    DATASET_PROFILE_V1_SCHEMA, DATASET_PROVENANCE_V2_SCHEMA, FLYWHEEL_PROGRESS_V2_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, PACKED_CORPUS_V3_SCHEMA, SEALED_PANEL_CONSUMPTION_V1_SCHEMA,
    SEALED_PANEL_POOL_V2_SCHEMA, TASK_PROMOTION_EVALUATION_V3_SCHEMA,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const EVIDENCE_SLOTS_PER_CONTEXT: usize = 8;
pub const PREFLIGHT_COHORTS_REQUIRED: usize = 10;
pub const PREFLIGHT_CONTEXTS_PER_COHORT: usize = 64;
pub const PREFLIGHT_MIN_VALIDITY: f64 = 0.995;
pub const PROMOTION_MIN_VALIDITY: f64 = 0.99;
pub const PROMOTION_MIN_CONTEXTS: usize = 128;
pub const PROMOTION_MAX_CONTEXTS: usize = 512;
pub const PROMOTION_CONTEXT_STEP: usize = 64;
pub const PROMOTION_BOOTSTRAP_RESAMPLES: usize = 10_000;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePreflightCohortV1 {
    pub sequence: usize,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub contexts: usize,
    pub requested_slots: usize,
    pub valid_slots: usize,
    pub completed: bool,
}

impl EvidencePreflightCohortV1 {
    pub fn validity(&self) -> f64 {
        if self.requested_slots == 0 {
            0.0
        } else {
            self.valid_slots as f64 / self.requested_slots as f64
        }
    }

    pub fn validate_for(
        &self,
        task: Task,
        accepted_parent: &Sha256Digest,
        sequence: usize,
    ) -> Result<()> {
        ensure!(
            self.sequence == sequence,
            "preflight cohorts are not consecutive"
        );
        ensure!(self.task == task, "preflight task drift");
        ensure!(
            &self.accepted_parent_checkpoint_sha256 == accepted_parent,
            "preflight accepted-parent drift"
        );
        ensure!(
            self.contexts == PREFLIGHT_CONTEXTS_PER_COHORT
                && self.requested_slots
                    == PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
            "preflight cohort must request eight slots for 64 contexts"
        );
        ensure!(
            self.completed
                && self.valid_slots <= self.requested_slots
                && self.validity() >= PREFLIGHT_MIN_VALIDITY,
            "preflight cohort does not meet 99.5% requested-slot validity"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEpochContextV1 {
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub requested_wave: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceEpochV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub start_wave: u64,
    pub runtime_policy_sha256: Sha256Digest,
    pub proposer_policy_sha256: Sha256Digest,
    pub evaluator_policy_sha256: Sha256Digest,
    pub catalog_eligibility_sha256: Sha256Digest,
    pub requested_contexts: Vec<EvidenceEpochContextV1>,
    pub slots_per_context: usize,
    pub preflight_cohorts: Vec<EvidencePreflightCohortV1>,
    pub opened_at_ms: u128,
}

impl CausalEvidenceEpochV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_EPOCH_SCHEMA,
            "causal evidence epoch schema is not v1"
        );
        ensure!(
            self.start_wave >= 191,
            "evidence epoch starts before causal waves"
        );
        ensure!(
            self.slots_per_context == EVIDENCE_SLOTS_PER_CONTEXT,
            "evidence epoch must request eight slots per context"
        );
        ensure!(
            !self.requested_contexts.is_empty(),
            "evidence epoch has no requested contexts"
        );
        let identities = self
            .requested_contexts
            .iter()
            .map(|context| context.context_identity.as_str())
            .collect::<BTreeSet<_>>();
        let digests = self
            .requested_contexts
            .iter()
            .map(|context| &context.context_sha256)
            .collect::<BTreeSet<_>>();
        ensure!(
            identities.len() == self.requested_contexts.len()
                && digests.len() == self.requested_contexts.len()
                && self.requested_contexts.iter().all(|context| {
                    !context.context_identity.trim().is_empty()
                        && context.requested_wave >= self.start_wave
                }),
            "evidence epoch contexts are empty, duplicated, or pre-epoch"
        );
        ensure!(
            self.preflight_cohorts.len() == PREFLIGHT_COHORTS_REQUIRED,
            "evidence epoch requires ten preflight cohorts"
        );
        for (sequence, cohort) in self.preflight_cohorts.iter().enumerate() {
            cohort.validate_for(self.task, &self.accepted_parent_checkpoint_sha256, sequence)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }

    pub fn requested_slots(&self) -> usize {
        self.requested_contexts.len() * self.slots_per_context
    }
}

pub fn ensure_evidence_epoch_immutable(
    existing: &CausalEvidenceEpochV1,
    proposed: &CausalEvidenceEpochV1,
) -> Result<()> {
    ensure!(
        existing.identity()? == proposed.identity()?,
        "an opened evidence epoch cannot be reset or mutated"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionEpochEvidenceV1 {
    pub epoch_sha256: Sha256Digest,
    pub evidence: TaskCausalEvidenceV2,
    pub policy_pins_match: bool,
    pub logged_slots: usize,
    pub permanent_missing_slots: usize,
    pub direct_count: usize,
    pub comparator_count: usize,
    pub production_gate_open: bool,
    #[serde(default)]
    pub gate_reasons: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceLedgerV3 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub authoritative: bool,
    pub all_history: CausalEvidenceLedgerV2,
    pub promotion_epoch: BTreeMap<Task, PromotionEpochEvidenceV1>,
}

impl CausalEvidenceLedgerV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA,
            "causal evidence ledger schema is not v3"
        );
        self.all_history.validate()?;
        for (task, row) in &self.promotion_epoch {
            ensure!(
                row.direct_count == row.comparator_count
                    && row.direct_count == row.evidence.complete_pairs,
                "{task} promotion epoch lane counts differ"
            );
            if row.production_gate_open {
                ensure!(
                    self.authoritative
                        && row.policy_pins_match
                        && row.logged_slots == row.evidence.requested_slots
                        && row.permanent_missing_slots == 0
                        && row.evidence.complete_pairs >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.evidence.proposal_validity >= PROMOTION_MIN_VALIDITY
                        && row.evidence.zero_cross_context_backfill
                        && row.evidence.completed_causal_waves
                        && row.evidence.valid_transitive_lineage,
                    "{task} v3 gate opened without clean epoch evidence"
                );
            }
        }
        Ok(())
    }
}

pub fn build_causal_evidence_ledger_v3(
    all_history: CausalEvidenceLedgerV2,
    epochs: &BTreeMap<Task, CausalEvidenceEpochV1>,
    receipts: &[CausalEvaluationReceiptV2],
    policy_pins_match: &BTreeMap<Task, bool>,
    logged_slots: &BTreeMap<Task, usize>,
    permanent_missing_slots: &BTreeMap<Task, usize>,
    generated_at_ms: u128,
) -> Result<CausalEvidenceLedgerV3> {
    all_history.validate()?;
    let mut promotion_epoch = BTreeMap::new();
    for (task, epoch) in epochs {
        epoch.validate()?;
        ensure!(epoch.task == *task, "evidence epoch task-key drift");
        let contexts = epoch
            .requested_contexts
            .iter()
            .map(|context| context.context_identity.as_str())
            .collect::<BTreeSet<_>>();
        let epoch_receipts = receipts
            .iter()
            .filter(|receipt| {
                receipt.task == *task
                    && receipt.wave >= epoch.start_wave
                    && receipt.accepted_parent_checkpoint_sha256
                        == epoch.accepted_parent_checkpoint_sha256
                    && contexts.contains(receipt.context_identity.as_str())
                    && receipt.pair_index < epoch.slots_per_context
            })
            .cloned()
            .collect::<Vec<_>>();
        let requested = BTreeMap::from([(*task, epoch.requested_slots())]);
        let parents = BTreeMap::from([(*task, epoch.accepted_parent_checkpoint_sha256.clone())]);
        let evidence =
            build_causal_evidence_ledger_v2(&parents, &requested, epoch_receipts, generated_at_ms)?
                .tasks
                .get(task)
                .cloned()
                .context("v3 epoch builder lost task evidence")?;
        let pins_match = policy_pins_match.get(task).copied().unwrap_or(false);
        let slots_logged = logged_slots.get(task).copied().unwrap_or_default();
        let permanent_missing = permanent_missing_slots
            .get(task)
            .copied()
            .unwrap_or_default();
        let mut gate_reasons = evidence.gate_reasons.clone();
        if !pins_match {
            gate_reasons.push("epoch policy pins do not match wave receipts".to_owned());
        }
        if slots_logged != epoch.requested_slots() {
            gate_reasons.push(format!(
                "slot log {slots_logged}/{}",
                epoch.requested_slots()
            ));
        }
        if permanent_missing > 0 {
            gate_reasons.push(format!("permanent missing epoch slots {permanent_missing}"));
        }
        let gate_open = gate_reasons.is_empty();
        promotion_epoch.insert(
            *task,
            PromotionEpochEvidenceV1 {
                epoch_sha256: epoch.identity()?,
                direct_count: evidence.complete_pairs,
                comparator_count: evidence.complete_pairs,
                evidence,
                policy_pins_match: pins_match,
                logged_slots: slots_logged,
                permanent_missing_slots: permanent_missing,
                production_gate_open: gate_open,
                gate_reasons,
            },
        );
    }
    let ledger = CausalEvidenceLedgerV3 {
        schema: CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA.to_owned(),
        generated_at_ms,
        authoritative: true,
        all_history,
        promotion_epoch,
    };
    ledger.validate()?;
    Ok(ledger)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalRejectionReasonV3 {
    Invalid,
    Duplicate,
    Unavailable,
    Compilation,
    Ancestry,
    Diversity,
}

pub fn all_rejection_counts() -> BTreeMap<ProposalRejectionReasonV3, usize> {
    [
        ProposalRejectionReasonV3::Invalid,
        ProposalRejectionReasonV3::Duplicate,
        ProposalRejectionReasonV3::Unavailable,
        ProposalRejectionReasonV3::Compilation,
        ProposalRejectionReasonV3::Ancestry,
        ProposalRejectionReasonV3::Diversity,
    ]
    .into_iter()
    .map(|reason| (reason, 0))
    .collect()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalSlotStatusV3 {
    Ready,
    PendingRetry,
    PermanentFailure,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalSlotReceiptV3 {
    pub slot: usize,
    pub status: ProposalSlotStatusV3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_sha256: Option<Sha256Digest>,
    #[serde(default)]
    pub rejections: BTreeMap<ProposalRejectionReasonV3, usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProposalKeyV3 {
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub seed: u64,
    pub decoding_policy_sha256: Sha256Digest,
    pub reservoir_size: usize,
}

impl ContextProposalKeyV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.reservoir_size, 64 | 128 | 256),
            "proposal reservoir must be 64, 128, or 256"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }

    pub fn next_reservoir_size(&self) -> Option<usize> {
        match self.reservoir_size {
            64 => Some(128),
            128 => Some(256),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CachedContextProposalV3 {
    pub slot: usize,
    pub candidate_sha256: Sha256Digest,
    pub canonical_genome: String,
    pub decoder: String,
    pub decoder_seed: u64,
    pub tokens: Vec<u32>,
    pub head_type: String,
    pub reward_mean: f64,
    pub reward_uncertainty: f64,
    pub reward_lcb: f64,
    pub instability_probability: f64,
    pub selection_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_risk_reason: Option<String>,
}

impl CachedContextProposalV3 {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.canonical_genome.trim().is_empty()
                && !self.decoder.trim().is_empty()
                && !self.head_type.trim().is_empty()
                && !self.tokens.is_empty(),
            "cached proposal payload is incomplete"
        );
        ensure!(
            [
                self.reward_mean,
                self.reward_uncertainty,
                self.reward_lcb,
                self.instability_probability,
                self.selection_score,
            ]
            .into_iter()
            .all(f64::is_finite)
                && self.reward_uncertainty >= 0.0
                && (0.0..=1.0).contains(&self.instability_probability),
            "cached proposal scores are invalid"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProposalArtifactV3 {
    pub schema: String,
    pub key: ContextProposalKeyV3,
    pub generated_at_ms: u128,
    pub decoded_candidates: usize,
    pub rejection_counts: BTreeMap<ProposalRejectionReasonV3, usize>,
    pub slots: Vec<ProposalSlotReceiptV3>,
    #[serde(default)]
    pub proposals: Vec<CachedContextProposalV3>,
}

impl ContextProposalArtifactV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA,
            "context proposal artifact schema is not v3"
        );
        self.key.validate()?;
        ensure!(
            self.decoded_candidates <= self.key.reservoir_size,
            "decoded candidates exceed proposal reservoir"
        );
        ensure!(
            [
                ProposalRejectionReasonV3::Invalid,
                ProposalRejectionReasonV3::Duplicate,
                ProposalRejectionReasonV3::Unavailable,
                ProposalRejectionReasonV3::Compilation,
                ProposalRejectionReasonV3::Ancestry,
                ProposalRejectionReasonV3::Diversity,
            ]
            .into_iter()
            .all(|reason| self.rejection_counts.contains_key(&reason)),
            "proposal artifact must report every rejection category"
        );
        ensure!(
            self.slots.len() == EVIDENCE_SLOTS_PER_CONTEXT
                && self
                    .slots
                    .iter()
                    .enumerate()
                    .all(|(slot, receipt)| receipt.slot == slot),
            "proposal artifact must log eight dense slots before decoding"
        );
        let mut candidates = BTreeSet::new();
        for receipt in &self.slots {
            match receipt.status {
                ProposalSlotStatusV3::Ready => {
                    ensure!(
                        receipt
                            .candidate_sha256
                            .as_ref()
                            .is_some_and(|candidate| candidates.insert(candidate)),
                        "ready proposal slot is missing or duplicates a candidate"
                    );
                }
                ProposalSlotStatusV3::PendingRetry => {
                    ensure!(
                        self.key.reservoir_size < 256
                            && receipt.candidate_sha256.is_none()
                            && !receipt.rejections.is_empty(),
                        "a retrying proposal slot must have a pre-256 rejection receipt"
                    );
                }
                ProposalSlotStatusV3::PermanentFailure => {
                    ensure!(
                        self.key.reservoir_size == 256
                            && receipt.candidate_sha256.is_none()
                            && !receipt.rejections.is_empty(),
                        "a missing slot becomes permanent only after reservoir 256"
                    );
                }
            }
        }
        ensure!(
            self.proposals.len() == self.ready_slots(),
            "ready proposal slots and cached payloads differ"
        );
        for (slot, proposal) in self.proposals.iter().enumerate() {
            proposal.validate()?;
            ensure!(
                proposal.slot == slot
                    && self.slots[slot].status == ProposalSlotStatusV3::Ready
                    && self.slots[slot].candidate_sha256.as_ref()
                        == Some(&proposal.candidate_sha256),
                "cached proposal payload does not match its slot receipt"
            );
        }
        Ok(())
    }

    pub fn ready_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.status == ProposalSlotStatusV3::Ready)
            .count()
    }
}

pub fn context_proposal_cache_path(
    cache_root: &Path,
    key: &ContextProposalKeyV3,
) -> Result<PathBuf> {
    Ok(cache_root
        .join(key.task.as_str())
        .join(key.accepted_checkpoint_sha256.as_str())
        .join(key.context_sha256.as_str())
        .join(key.seed.to_string())
        .join(key.decoding_policy_sha256.as_str())
        .join(format!("{}.v3.json", key.reservoir_size)))
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetOriginV2 {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseDecisionV2 {
    AutomaticAllow,
    ManualApproval,
    Quarantine,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticFamilyKeyV2 {
    pub generator_name: String,
    pub generator_version: String,
    pub scenario: String,
    pub target_family: String,
    pub feature_family_set: BTreeSet<String>,
    pub shift_pathology_class: String,
}

impl SyntheticFamilyKeyV2 {
    pub fn exact_family_sha256(&self) -> Result<Sha256Digest> {
        ensure!(
            [
                self.generator_name.as_str(),
                self.generator_version.as_str(),
                self.scenario.as_str(),
                self.target_family.as_str(),
                self.shift_pathology_class.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty())
                && !self.feature_family_set.is_empty(),
            "synthetic family key has an empty stable component"
        );
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProvenanceV2 {
    pub schema: String,
    pub provider: String,
    pub provider_version: String,
    pub upstream_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_spdx: Option<String>,
    #[serde(default)]
    pub manual_license_approved: bool,
    pub raw_sha256: BTreeMap<String, Sha256Digest>,
    pub origin: DatasetOriginV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_superfamily: Option<String>,
    pub exact_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
}

impl DatasetProvenanceV2 {
    pub fn license_decision(&self) -> LicenseDecisionV2 {
        match self
            .license_spdx
            .as_deref()
            .map(|value| value.to_ascii_uppercase())
        {
            Some(value) if matches!(value.as_str(), "CC0-1.0" | "CC-BY-4.0" | "CC-BY-4.0-ONLY") => {
                LicenseDecisionV2::AutomaticAllow
            }
            Some(_) if self.manual_license_approved => LicenseDecisionV2::ManualApproval,
            _ => LicenseDecisionV2::Quarantine,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DATASET_PROVENANCE_V2_SCHEMA,
            "dataset provenance schema is not v2"
        );
        ensure!(
            [
                self.provider.as_str(),
                self.provider_version.as_str(),
                self.upstream_root.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty())
                && !self.raw_sha256.is_empty(),
            "dataset provenance is incomplete"
        );
        ensure!(
            self.origin == DatasetOriginV2::Synthetic || self.generator_superfamily.is_none(),
            "real datasets cannot claim a generator superfamily"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSizeSliceV1 {
    Small,
    Medium,
    Large,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingnessSliceV1 {
    None,
    Low,
    High,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetShapeSliceV1 {
    Balanced,
    Imbalanced,
    LightTailed,
    HeavyTailed,
    Multimodal,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProfileV1 {
    pub schema: String,
    pub rows: usize,
    pub encoded_features: usize,
    pub missing_fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minority_majority_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric_target_distinct_finite: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finite_target_fraction: Option<f64>,
    pub size_slice: DatasetSizeSliceV1,
    pub missingness_slice: MissingnessSliceV1,
    pub target_shape_slice: TargetShapeSliceV1,
    pub embedding_novelty: f64,
    #[serde(default)]
    pub sensitive_or_pii_like: bool,
    #[serde(default)]
    pub ambiguous_target: bool,
    #[serde(default)]
    pub sparse_text: bool,
    #[serde(default)]
    pub grouped_or_time_series: bool,
    #[serde(default)]
    pub duplicate_or_leakage_prone: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetEligibilityV1 {
    pub eligible: bool,
    #[serde(default)]
    pub quarantine_reasons: Vec<String>,
}

impl DatasetProfileV1 {
    pub fn eligibility(
        &self,
        task: Task,
        provenance: &DatasetProvenanceV2,
    ) -> Result<DatasetEligibilityV1> {
        provenance.validate()?;
        ensure!(
            self.schema == DATASET_PROFILE_V1_SCHEMA,
            "dataset profile schema is not v1"
        );
        ensure!(
            self.missing_fraction.is_finite()
                && (0.0..=1.0).contains(&self.missing_fraction)
                && self.embedding_novelty.is_finite()
                && self
                    .minority_majority_ratio
                    .is_none_or(|value| value.is_finite())
                && self
                    .finite_target_fraction
                    .is_none_or(|value| value.is_finite()),
            "dataset profile contains a non-finite value"
        );
        let mut reasons = Vec::new();
        if provenance.license_decision() == LicenseDecisionV2::Quarantine {
            reasons.push("license is absent or not approved".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("row count is outside 500..100000".to_owned());
        }
        if self.encoded_features > 5_000 {
            reasons.push("encoded feature count exceeds 5000".to_owned());
        }
        for (condition, reason) in [
            (self.sensitive_or_pii_like, "sensitive or PII-like"),
            (self.ambiguous_target, "ambiguous target"),
            (self.sparse_text, "sparse text"),
            (self.grouped_or_time_series, "grouped or time-series"),
            (
                self.duplicate_or_leakage_prone,
                "duplicate or leakage-prone",
            ),
        ] {
            if condition {
                reasons.push(reason.to_owned());
            }
        }
        match task {
            Task::Binary => {
                if self.class_count != Some(2) {
                    reasons.push("binary task does not have exactly two classes".to_owned());
                }
                if self
                    .minority_majority_ratio
                    .is_none_or(|ratio| ratio < 0.05)
                {
                    reasons.push("minority/majority ratio is below 0.05".to_owned());
                }
            }
            Task::Regression => {
                if self
                    .numeric_target_distinct_finite
                    .is_none_or(|count| count < 50)
                {
                    reasons.push("regression target has fewer than 50 finite values".to_owned());
                }
                if self
                    .finite_target_fraction
                    .is_none_or(|fraction| fraction < 0.99)
                {
                    reasons.push("finite regression targets are below 99%".to_owned());
                }
            }
            // The OpenML acquisition lane only sources the two legacy tasks;
            // quarantine anything else fail-closed until per-task eligibility
            // rules exist for the new lanes.
            Task::Timeseries | Task::Multiclass => {
                reasons.push(format!(
                    "{task} datasets are not eligible for the OpenML acquisition lane"
                ));
            }
        }
        Ok(DatasetEligibilityV1 {
            eligible: reasons.is_empty(),
            quarantine_reasons: reasons,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenMlDatasetMetadataV1 {
    pub dataset_id: u64,
    pub version: u64,
    pub name: String,
    #[serde(alias = "licence")]
    pub license: Option<String>,
    pub default_target_attribute: Option<String>,
    pub original_data_url: Option<String>,
    pub download_url: String,
    pub rows: usize,
    pub features: usize,
    pub missing_values: usize,
}

impl OpenMlDatasetMetadataV1 {
    pub fn deterministic_key(&self) -> (u64, u64, &str) {
        (self.dataset_id, self.version, self.name.as_str())
    }

    pub fn discovery_quarantine_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.version == 0 {
            reasons.push("OpenML dataset version is not immutable".to_owned());
        }
        if self.license.as_deref().is_none_or(str::is_empty) {
            reasons.push("OpenML license metadata is absent".to_owned());
        }
        if self
            .default_target_attribute
            .as_deref()
            .is_none_or(str::is_empty)
        {
            reasons.push("OpenML default target is absent".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("OpenML row count is outside initial eligibility".to_owned());
        }
        reasons
    }
}

pub fn deterministic_openml_discovery(
    mut datasets: Vec<OpenMlDatasetMetadataV1>,
) -> Vec<OpenMlDatasetMetadataV1> {
    datasets.sort_by(|left, right| left.deterministic_key().cmp(&right.deterministic_key()));
    datasets.dedup_by(|left, right| {
        left.dataset_id == right.dataset_id && left.version == right.version
    });
    datasets
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UciDatasetMetadataV1 {
    pub persistent_id: String,
    pub version: String,
    pub name: String,
    pub license: Option<String>,
    pub dataset_url: String,
    pub target_columns: Vec<String>,
    pub rows: usize,
    pub features: usize,
}

impl UciDatasetMetadataV1 {
    pub fn discovery_quarantine_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.persistent_id.trim().is_empty() || self.version.trim().is_empty() {
            reasons.push("UCI dataset lacks an immutable identity/version".to_owned());
        }
        if self.license.as_deref().is_none_or(str::is_empty) {
            reasons.push("UCI per-dataset license is absent".to_owned());
        }
        if self.target_columns.len() != 1 {
            reasons.push("UCI target is absent or ambiguous".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("UCI row count is outside initial eligibility".to_owned());
        }
        reasons
    }
}

pub fn deterministic_uci_discovery(
    mut datasets: Vec<UciDatasetMetadataV1>,
) -> Vec<UciDatasetMetadataV1> {
    datasets.sort_by(|left, right| {
        (&left.persistent_id, &left.version, &left.name).cmp(&(
            &right.persistent_id,
            &right.version,
            &right.name,
        ))
    });
    datasets.dedup_by(|left, right| {
        left.persistent_id == right.persistent_id && left.version == right.version
    });
    datasets
}

pub fn cross_provider_duplicate_cluster(
    raw_sha256: &BTreeMap<String, Sha256Digest>,
) -> Result<Sha256Digest> {
    ensure!(
        !raw_sha256.is_empty(),
        "duplicate clustering requires raw content hashes"
    );
    let mut digests = raw_sha256.values().collect::<Vec<_>>();
    digests.sort();
    digests.dedup();
    Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&digests)?))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurationCandidateV1 {
    pub dataset_sha256: Sha256Digest,
    pub dimensions: BTreeMap<String, String>,
    pub embedding_novelty: f64,
}

fn curation_distance(left: &CurationCandidateV1, right: &CurationCandidateV1) -> f64 {
    let dimensions = left
        .dimensions
        .keys()
        .chain(right.dimensions.keys())
        .collect::<BTreeSet<_>>();
    let categorical = dimensions
        .into_iter()
        .filter(|dimension| left.dimensions.get(*dimension) != right.dimensions.get(*dimension))
        .count() as f64;
    categorical + (left.embedding_novelty - right.embedding_novelty).abs()
}

pub fn deterministic_max_min_selection(
    candidates: &[CurationCandidateV1],
    count: usize,
) -> Result<Vec<Sha256Digest>> {
    ensure!(
        count <= candidates.len()
            && candidates
                .iter()
                .all(|candidate| candidate.embedding_novelty.is_finite()),
        "invalid deterministic max-min request"
    );
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut remaining = candidates.iter().collect::<Vec<_>>();
    remaining.sort_by(|left, right| left.dataset_sha256.cmp(&right.dataset_sha256));
    let mut selected = vec![remaining.remove(0)];
    while selected.len() < count {
        let (index, _) = remaining
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let distance = selected
                    .iter()
                    .map(|known| curation_distance(candidate, known))
                    .fold(f64::INFINITY, f64::min);
                (index, distance)
            })
            .max_by(
                |(left_index, left_distance), (right_index, right_distance)| {
                    left_distance.total_cmp(right_distance).then_with(|| {
                        remaining[*right_index]
                            .dataset_sha256
                            .cmp(&remaining[*left_index].dataset_sha256)
                    })
                },
            )
            .context("max-min selection exhausted candidates")?;
        selected.push(remaining.remove(index));
    }
    Ok(selected
        .into_iter()
        .map(|candidate| candidate.dataset_sha256.clone())
        .collect())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelPoolKindV2 {
    Training,
    Development,
    Promotion,
    Audit,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelCandidateV2 {
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelEntryV2 {
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
    pub pool: PanelPoolKindV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion_panel: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelPoolV2 {
    pub schema: String,
    pub entries: Vec<SealedPanelEntryV2>,
    pub created_at_ms: u128,
}

impl SealedPanelPoolV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_POOL_V2_SCHEMA,
            "sealed panel pool schema is not v2"
        );
        ensure!(!self.entries.is_empty(), "sealed panel pool is empty");
        let mut contexts = BTreeSet::new();
        let mut family_pool = BTreeMap::new();
        let mut cluster_pool = BTreeMap::new();
        let mut panel_counts: BTreeMap<(Task, &str), (usize, usize)> = BTreeMap::new();
        for entry in &self.entries {
            ensure!(
                !entry.context_identity.trim().is_empty() && contexts.insert(&entry.context_sha256),
                "sealed panel repeats or empties a context"
            );
            if let Some(existing) = family_pool.insert(&entry.source_family_sha256, entry.pool) {
                ensure!(
                    existing == entry.pool,
                    "a source family crosses sealed panel pools"
                );
            }
            if let Some(existing) =
                cluster_pool.insert(&entry.near_duplicate_cluster_sha256, entry.pool)
            {
                ensure!(
                    existing == entry.pool,
                    "a near-duplicate cluster crosses sealed panel pools"
                );
            }
            match entry.pool {
                PanelPoolKindV2::Promotion => {
                    let panel = entry
                        .promotion_panel
                        .as_deref()
                        .context("promotion entry lacks a panel identity")?;
                    let counts = panel_counts.entry((entry.task, panel)).or_default();
                    counts.0 += 1;
                    counts.1 += usize::from(entry.origin == DatasetOriginV2::Real);
                }
                _ => ensure!(
                    entry.promotion_panel.is_none(),
                    "non-promotion entry claims a promotion panel"
                ),
            }
        }
        let total = self.entries.len() as f64;
        let fraction = |pool| {
            self.entries
                .iter()
                .filter(|entry| entry.pool == pool)
                .count() as f64
                / total
        };
        ensure!(
            (fraction(PanelPoolKindV2::Training) - 0.70).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Development) - 0.10).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Promotion) - 0.15).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Audit) - 0.05).abs() <= 0.01,
            "sealed panel allocation is not 70/10/15/5"
        );
        for ((task, panel), (count, real)) in panel_counts {
            ensure!(
                real * 5 >= count * 4,
                "{task} promotion panel {panel} is less than 80% real-world"
            );
        }
        Ok(())
    }

    pub fn promotion_panel(&self, task: Task, panel: &str) -> Vec<&SealedPanelEntryV2> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.task == task
                    && entry.pool == PanelPoolKindV2::Promotion
                    && entry.promotion_panel.as_deref() == Some(panel)
            })
            .collect()
    }
}

#[derive(Debug)]
struct PanelComponent {
    indexes: Vec<usize>,
    key: Sha256Digest,
    real: usize,
}

fn panel_find(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = panel_find(parents, parents[index]);
    }
    parents[index]
}

fn panel_union(parents: &mut [usize], left: usize, right: usize) {
    let left = panel_find(parents, left);
    let right = panel_find(parents, right);
    if left != right {
        parents[right] = left;
    }
}

pub fn build_sealed_panel_pool_v2(
    candidates: Vec<SealedPanelCandidateV2>,
    created_at_ms: u128,
) -> Result<SealedPanelPoolV2> {
    ensure!(
        candidates.len() >= 5_120,
        "sealed panel builder needs at least 5120 eligible contexts"
    );
    let mut contexts = BTreeSet::new();
    ensure!(
        candidates.iter().all(|candidate| {
            !candidate.context_identity.trim().is_empty()
                && contexts.insert(&candidate.context_sha256)
        }),
        "sealed panel candidates repeat or empty a context"
    );
    let mut parents = (0..candidates.len()).collect::<Vec<_>>();
    let mut family_owner = BTreeMap::new();
    let mut cluster_owner = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if let Some(owner) = family_owner.insert(&candidate.source_family_sha256, index) {
            panel_union(&mut parents, owner, index);
        }
        if let Some(owner) = cluster_owner.insert(&candidate.near_duplicate_cluster_sha256, index) {
            panel_union(&mut parents, owner, index);
        }
    }
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..candidates.len() {
        let parent = panel_find(&mut parents, index);
        grouped.entry(parent).or_default().push(index);
    }
    let mut components = grouped
        .into_values()
        .map(|mut indexes| {
            indexes.sort_by(|left, right| {
                candidates[*left]
                    .context_sha256
                    .cmp(&candidates[*right].context_sha256)
            });
            let key = candidates[indexes[0]].context_sha256.clone();
            let real = indexes
                .iter()
                .filter(|index| candidates[**index].origin == DatasetOriginV2::Real)
                .count();
            PanelComponent { indexes, key, real }
        })
        .collect::<Vec<_>>();
    let total = candidates.len();
    let promotion_target = total * 15 / 100;
    let development_target = total * 10 / 100;
    let audit_target = total * 5 / 100;
    components.sort_by(|left, right| {
        (right.real * left.indexes.len())
            .cmp(&(left.real * right.indexes.len()))
            .then_with(|| left.key.cmp(&right.key))
    });
    let mut assignment = vec![PanelPoolKindV2::Training; candidates.len()];
    let mut promotion_count = 0;
    let mut remaining = Vec::new();
    for component in components {
        if promotion_count < promotion_target {
            for index in &component.indexes {
                assignment[*index] = PanelPoolKindV2::Promotion;
            }
            promotion_count += component.indexes.len();
        } else {
            remaining.push(component);
        }
    }
    remaining.sort_by(|left, right| left.key.cmp(&right.key));
    let mut development_count = 0;
    let mut audit_count = 0;
    for component in remaining {
        let pool = if development_count < development_target {
            development_count += component.indexes.len();
            PanelPoolKindV2::Development
        } else if audit_count < audit_target {
            audit_count += component.indexes.len();
            PanelPoolKindV2::Audit
        } else {
            PanelPoolKindV2::Training
        };
        for index in component.indexes {
            assignment[index] = pool;
        }
    }
    let mut panel_by_index = BTreeMap::<usize, String>::new();
    for task in [Task::Binary, Task::Regression] {
        let mut promotion = candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                candidate.task == task && assignment[*index] == PanelPoolKindV2::Promotion
            })
            .map(|(index, candidate)| (index, candidate))
            .collect::<Vec<_>>();
        promotion.sort_by(|(_, left), (_, right)| {
            usize::from(right.origin == DatasetOriginV2::Real)
                .cmp(&usize::from(left.origin == DatasetOriginV2::Real))
                .then_with(|| left.context_sha256.cmp(&right.context_sha256))
        });
        ensure!(
            promotion.len() >= 3 * PROMOTION_MIN_CONTEXTS,
            "{task} promotion pool cannot supply three independent panels"
        );
        let panel_count = 3.max(promotion.len().div_ceil(PROMOTION_MAX_CONTEXTS));
        ensure!(
            promotion.len() >= panel_count * PROMOTION_MIN_CONTEXTS,
            "{task} promotion panels cannot all reach 128 contexts"
        );
        for (offset, (index, _)) in promotion.into_iter().enumerate() {
            let panel = offset % panel_count;
            panel_by_index.insert(index, format!("{}-promotion-{panel:03}", task.as_str()));
        }
    }
    let entries = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| SealedPanelEntryV2 {
            task: candidate.task,
            context_identity: candidate.context_identity,
            context_sha256: candidate.context_sha256,
            source_family_sha256: candidate.source_family_sha256,
            near_duplicate_cluster_sha256: candidate.near_duplicate_cluster_sha256,
            origin: candidate.origin,
            pool: assignment[index],
            promotion_panel: panel_by_index.remove(&index),
        })
        .collect();
    let pool = SealedPanelPoolV2 {
        schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
        entries,
        created_at_ms,
    };
    pool.validate()?;
    for task in [Task::Binary, Task::Regression] {
        let panels = pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
            .filter_map(|entry| entry.promotion_panel.as_deref())
            .collect::<BTreeSet<_>>();
        ensure!(
            panels.len() >= 3
                && panels
                    .iter()
                    .all(|panel| pool.promotion_panel(task, panel).len() >= PROMOTION_MIN_CONTEXTS),
            "{task} sealed pool lacks three usable promotion panels"
        );
    }
    Ok(pool)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelConsumptionV1 {
    pub schema: String,
    pub task: Task,
    pub panel: String,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub run_index: usize,
    pub seed: u64,
    pub context_sha256: Vec<Sha256Digest>,
    pub consumed_at_ms: u128,
}

impl SealedPanelConsumptionV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_CONSUMPTION_V1_SCHEMA,
            "sealed panel consumption schema is not v1"
        );
        ensure!(self.run_index < 3, "promotion run index must be 0..3");
        ensure!(
            (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&self.context_sha256.len())
                && (self.context_sha256.len() - PROMOTION_MIN_CONTEXTS) % PROMOTION_CONTEXT_STEP
                    == 0
                && self.context_sha256.iter().collect::<BTreeSet<_>>().len()
                    == self.context_sha256.len(),
            "promotion consumption must contain 128..512 unique contexts in steps of 64"
        );
        Ok(())
    }
}

pub fn validate_panel_consumption(
    pool: &SealedPanelPoolV2,
    receipt: &SealedPanelConsumptionV1,
    prior: &[SealedPanelConsumptionV1],
) -> Result<()> {
    pool.validate()?;
    receipt.validate()?;
    ensure!(
        prior.iter().all(|row| row.panel != receipt.panel),
        "a promotion panel is single-use and was already consumed"
    );
    let eligible = pool
        .promotion_panel(receipt.task, &receipt.panel)
        .into_iter()
        .map(|entry| &entry.context_sha256)
        .collect::<BTreeSet<_>>();
    ensure!(
        receipt
            .context_sha256
            .iter()
            .all(|context| eligible.contains(context)),
        "consumption includes a context outside its sealed promotion panel"
    );
    let prior_contexts = prior
        .iter()
        .flat_map(|row| row.context_sha256.iter())
        .collect::<BTreeSet<_>>();
    ensure!(
        receipt
            .context_sha256
            .iter()
            .all(|context| !prior_contexts.contains(context)),
        "promotion runs are not context-disjoint"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionObservationV3 {
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub candidate_score: f64,
    pub accepted_score: f64,
    pub gp_score: f64,
    pub candidate_conditional_advantage: f64,
    pub accepted_conditional_advantage: f64,
    pub valid: bool,
    pub replay_verified: bool,
    pub candidate_failed: bool,
    pub accepted_failed: bool,
    pub candidate_latency_ms: u64,
    pub accepted_latency_ms: u64,
    pub strict_win: bool,
    pub equal_budget_win: bool,
    pub slices: BTreeMap<String, String>,
}

impl PromotionObservationV3 {
    fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.candidate_score,
                self.accepted_score,
                self.gp_score,
                self.candidate_conditional_advantage,
                self.accepted_conditional_advantage,
            ]
            .into_iter()
            .all(f64::is_finite),
            "promotion observation contains a non-finite score"
        );
        for dimension in ["origin", "size", "missingness", "imbalance", "target_shape"] {
            ensure!(
                self.slices
                    .get(dimension)
                    .is_some_and(|value| !value.trim().is_empty()),
                "promotion observation lacks predefined {dimension} slice"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusteredConfidenceBoundsV3 {
    pub resamples: usize,
    pub seed: u64,
    pub candidate_vs_accepted_lower: f64,
    pub candidate_vs_gp_lower: f64,
    pub conditional_advantage_lower: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionMetricsV3 {
    pub contexts: usize,
    pub confidence: ClusteredConfidenceBoundsV3,
    pub maximum_slice_regression: f64,
    pub validity: f64,
    pub replay_verification: f64,
    pub failure_rate_increase: f64,
    pub latency_ratio: f64,
    pub strict_wins: usize,
    pub equal_budget_wins: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionEvaluationV3 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub panel_consumption_sha256: Sha256Digest,
    pub run_index: usize,
    pub metrics: TaskPromotionMetricsV3,
    pub accepted: bool,
    #[serde(default)]
    pub rejection_reasons: Vec<String>,
    pub evaluated_at_ms: u128,
}

impl TaskPromotionEvaluationV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_PROMOTION_EVALUATION_V3_SCHEMA,
            "task promotion evaluation schema is not v3"
        );
        ensure!(self.run_index < 3, "promotion run index must be 0..3");
        ensure!(
            (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&self.metrics.contexts)
                && (self.metrics.contexts - PROMOTION_MIN_CONTEXTS) % PROMOTION_CONTEXT_STEP == 0,
            "promotion evaluation context count is outside the sequential panel schedule"
        );
        ensure!(
            self.metrics.confidence.resamples == PROMOTION_BOOTSTRAP_RESAMPLES,
            "promotion evaluation must use exactly 10000 clustered resamples"
        );
        ensure!(
            [
                self.metrics.confidence.candidate_vs_accepted_lower,
                self.metrics.confidence.candidate_vs_gp_lower,
                self.metrics.confidence.conditional_advantage_lower,
                self.metrics.maximum_slice_regression,
                self.metrics.validity,
                self.metrics.replay_verification,
                self.metrics.failure_rate_increase,
                self.metrics.latency_ratio,
            ]
            .into_iter()
            .all(f64::is_finite),
            "promotion evaluation contains a non-finite metric"
        );
        ensure!(
            self.accepted == self.rejection_reasons.is_empty(),
            "promotion evaluation decision/reason mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct BootstrapFamily {
    candidate_accepted_sum: f64,
    candidate_gp_sum: f64,
    conditional_sum: f64,
    count: usize,
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn lower_confidence_bound(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    // A one-sided 95% deterministic percentile bound.
    values[(values.len() * 5 / 100).min(values.len() - 1)]
}

pub fn clustered_confidence_bounds_v3(
    observations: &[PromotionObservationV3],
    seed: u64,
) -> Result<ClusteredConfidenceBoundsV3> {
    ensure!(
        !observations.is_empty(),
        "clustered confidence requires observations"
    );
    let mut grouped: BTreeMap<&Sha256Digest, BootstrapFamily> = BTreeMap::new();
    for row in observations {
        row.validate()?;
        let family = grouped
            .entry(&row.source_family_sha256)
            .or_insert(BootstrapFamily {
                candidate_accepted_sum: 0.0,
                candidate_gp_sum: 0.0,
                conditional_sum: 0.0,
                count: 0,
            });
        family.candidate_accepted_sum += row.candidate_score - row.accepted_score;
        family.candidate_gp_sum += row.candidate_score - row.gp_score;
        family.conditional_sum +=
            row.candidate_conditional_advantage - row.accepted_conditional_advantage;
        family.count += 1;
    }
    let families = grouped.values().copied().collect::<Vec<_>>();
    ensure!(
        families.len() >= 2,
        "clustered confidence requires at least two source families"
    );
    let mut random = SplitMix64::new(seed);
    let mut accepted = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    let mut gp = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    let mut conditional = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    for _ in 0..PROMOTION_BOOTSTRAP_RESAMPLES {
        let mut accepted_sum = 0.0;
        let mut gp_sum = 0.0;
        let mut conditional_sum = 0.0;
        let mut count = 0;
        for _ in 0..families.len() {
            let family = families[random.next() as usize % families.len()];
            accepted_sum += family.candidate_accepted_sum;
            gp_sum += family.candidate_gp_sum;
            conditional_sum += family.conditional_sum;
            count += family.count;
        }
        accepted.push(accepted_sum / count as f64);
        gp.push(gp_sum / count as f64);
        conditional.push(conditional_sum / count as f64);
    }
    Ok(ClusteredConfidenceBoundsV3 {
        resamples: PROMOTION_BOOTSTRAP_RESAMPLES,
        seed,
        candidate_vs_accepted_lower: lower_confidence_bound(accepted),
        candidate_vs_gp_lower: lower_confidence_bound(gp),
        conditional_advantage_lower: lower_confidence_bound(conditional),
    })
}

fn maximum_slice_regression(observations: &[PromotionObservationV3]) -> f64 {
    let mut slices: BTreeMap<(&str, &str), (f64, usize)> = BTreeMap::new();
    for row in observations {
        for (dimension, value) in &row.slices {
            let summary = slices
                .entry((dimension.as_str(), value.as_str()))
                .or_default();
            summary.0 += row.candidate_score - row.accepted_score;
            summary.1 += 1;
        }
    }
    slices
        .values()
        .map(|(sum, count)| -(sum / *count as f64))
        .fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_task_promotion_v3(
    task: Task,
    accepted_parent_checkpoint_sha256: Sha256Digest,
    candidate_checkpoint_sha256: Sha256Digest,
    panel_consumption_sha256: Sha256Digest,
    run_index: usize,
    observations: &[PromotionObservationV3],
    bootstrap_seed: u64,
    evaluated_at_ms: u128,
) -> Result<TaskPromotionEvaluationV3> {
    ensure!(run_index < 3, "promotion run index must be 0..3");
    ensure!(
        (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&observations.len())
            && (observations.len() - PROMOTION_MIN_CONTEXTS) % PROMOTION_CONTEXT_STEP == 0,
        "promotion run must evaluate 128..512 contexts in steps of 64"
    );
    let contexts = observations
        .iter()
        .map(|row| &row.context_sha256)
        .collect::<BTreeSet<_>>();
    ensure!(
        contexts.len() == observations.len(),
        "promotion observations repeat a context"
    );
    let confidence = clustered_confidence_bounds_v3(observations, bootstrap_seed)?;
    let count = observations.len() as f64;
    let validity = observations.iter().filter(|row| row.valid).count() as f64 / count;
    let replay = observations
        .iter()
        .filter(|row| row.replay_verified)
        .count() as f64
        / count;
    let candidate_failures = observations
        .iter()
        .filter(|row| row.candidate_failed)
        .count() as f64
        / count;
    let accepted_failures = observations
        .iter()
        .filter(|row| row.accepted_failed)
        .count() as f64
        / count;
    let candidate_latency = observations
        .iter()
        .map(|row| row.candidate_latency_ms as u128)
        .sum::<u128>() as f64
        / count;
    let accepted_latency = observations
        .iter()
        .map(|row| row.accepted_latency_ms as u128)
        .sum::<u128>() as f64
        / count;
    let latency_ratio = if accepted_latency == 0.0 {
        f64::INFINITY
    } else {
        candidate_latency / accepted_latency
    };
    let metrics = TaskPromotionMetricsV3 {
        contexts: observations.len(),
        confidence,
        maximum_slice_regression: maximum_slice_regression(observations),
        validity,
        replay_verification: replay,
        failure_rate_increase: candidate_failures - accepted_failures,
        latency_ratio,
        strict_wins: observations.iter().filter(|row| row.strict_win).count(),
        equal_budget_wins: observations
            .iter()
            .filter(|row| row.equal_budget_win)
            .count(),
    };
    let mut rejection_reasons = Vec::new();
    for (condition, reason) in [
        (
            metrics.confidence.candidate_vs_accepted_lower >= 0.002,
            "candidate-versus-accepted lower bound < 0.002",
        ),
        (
            metrics.confidence.candidate_vs_gp_lower > 0.0,
            "candidate-versus-GP lower bound <= 0",
        ),
        (
            metrics.confidence.conditional_advantage_lower >= 0.005,
            "conditional-advantage lower bound < 0.005",
        ),
        (
            metrics.maximum_slice_regression <= 0.005,
            "a predefined slice regresses by more than 0.005",
        ),
        (metrics.validity >= 0.99, "validity < 0.99"),
        (
            (metrics.replay_verification - 1.0).abs() < f64::EPSILON,
            "replay verification is not 100%",
        ),
        (
            metrics.failure_rate_increase <= 0.0025,
            "failure-rate increase > 0.0025",
        ),
        (metrics.latency_ratio <= 1.25, "latency ratio > 1.25"),
        (metrics.strict_wins > 0, "no strict win"),
        (metrics.equal_budget_wins > 0, "no equal-budget win"),
    ] {
        if !condition {
            rejection_reasons.push(reason.to_owned());
        }
    }
    let evaluation = TaskPromotionEvaluationV3 {
        schema: TASK_PROMOTION_EVALUATION_V3_SCHEMA.to_owned(),
        task,
        accepted_parent_checkpoint_sha256,
        candidate_checkpoint_sha256,
        panel_consumption_sha256,
        run_index,
        metrics,
        accepted: rejection_reasons.is_empty(),
        rejection_reasons,
        evaluated_at_ms,
    };
    evaluation.validate()?;
    Ok(evaluation)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PackedLabelShardV3 {
    pub first_receipt: u64,
    pub last_receipt: u64,
    pub rows: usize,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from_corpus_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedColumnV3 {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub dtype: String,
    pub rows: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PackedCorpusV3 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_row_index: PackedColumnV3,
    pub task_column: PackedColumnV3,
    pub split_column: PackedColumnV3,
    pub exact_pair_sidecar: PackedColumnV3,
    pub label_shards: Vec<PackedLabelShardV3>,
    pub created_at_ms: u128,
}

impl PackedCorpusV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PACKED_CORPUS_V3_SCHEMA,
            "packed corpus schema is not v3"
        );
        ensure!(self.watermark > 0, "packed corpus watermark is zero");
        ensure!(
            !self.label_shards.is_empty(),
            "packed corpus has no content-addressed label shards"
        );
        let rows = self.selected_row_index.rows;
        ensure!(
            self.task_column.rows == rows
                && self.split_column.rows == rows
                && self
                    .label_shards
                    .iter()
                    .map(|shard| shard.rows)
                    .sum::<usize>()
                    == rows,
            "packed corpus selected index, columns, and shard rows differ"
        );
        ensure!(
            self.task_column.dtype == "u1" && self.split_column.dtype == "u1",
            "packed task/split columns must be memory-mappable u1 arrays"
        );
        ensure!(
            [
                &self.selected_row_index.path,
                &self.task_column.path,
                &self.split_column.path,
                &self.exact_pair_sidecar.path,
            ]
            .into_iter()
            .all(|path| path.file_name().and_then(|name| name.to_str())
                != Some("observations.jsonl")),
            "PackedCorpus.v3 must not copy observations.jsonl"
        );
        let mut previous = 0;
        let mut hashes = BTreeSet::new();
        for shard in &self.label_shards {
            ensure!(
                shard.rows > 0
                    && shard.first_receipt <= shard.last_receipt
                    && shard.first_receipt > previous
                    && shard.last_receipt <= self.watermark
                    && hashes.insert(&shard.sha256),
                "packed corpus shards overlap, repeat, or exceed the frozen watermark"
            );
            previous = shard.last_receipt;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationEvidenceV1 {
    pub error_correlation_lower_95: f64,
    pub interval_coverage: f64,
    pub target_coverage: f64,
    pub native_head_accuracy: f64,
    pub frequency_baseline_accuracy: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyRankingSourceV1 {
    Learned,
    EmpiricalEnsembleDispersion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankingFallbackV1 {
    pub uncertainty: UncertaintyRankingSourceV1,
    pub native_head_enabled: bool,
}

pub fn ranking_fallback(evidence: &CalibrationEvidenceV1) -> Result<RankingFallbackV1> {
    ensure!(
        [
            evidence.error_correlation_lower_95,
            evidence.interval_coverage,
            evidence.target_coverage,
            evidence.native_head_accuracy,
            evidence.frequency_baseline_accuracy,
        ]
        .into_iter()
        .all(f64::is_finite),
        "calibration evidence contains a non-finite metric"
    );
    let uncertainty = if evidence.error_correlation_lower_95 >= 0.0
        && (evidence.interval_coverage - evidence.target_coverage).abs() <= 0.05
    {
        UncertaintyRankingSourceV1::Learned
    } else {
        UncertaintyRankingSourceV1::EmpiricalEnsembleDispersion
    };
    Ok(RankingFallbackV1 {
        uncertainty,
        native_head_enabled: evidence.native_head_accuracy
            >= evidence.frequency_baseline_accuracy + 0.05,
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedGenerationV2 {
    pub task: Task,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub child_checkpoint_sha256: Sha256Digest,
    pub independent_external_passes: Vec<Sha256Digest>,
    pub accepted_at_ms: u128,
}

impl AcceptedGenerationV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.parent_checkpoint_sha256 != self.child_checkpoint_sha256,
            "accepted generation child equals its parent"
        );
        ensure!(
            self.independent_external_passes.len() == 3
                && self
                    .independent_external_passes
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == 3,
            "accepted generation requires three independent external passes"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauDiagnosisV1 {
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    pub rejected_candidates: Vec<Sha256Digest>,
    pub proposal_validity: f64,
    pub gp_disadvantage: f64,
    pub calibration_summary: String,
    pub data_slice_summary: String,
    pub latency_summary: String,
    pub replay_summary: String,
    pub diagnosed_at_ms: u128,
}

impl PlateauDiagnosisV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.rejected_candidates.len() >= 3
                && self
                    .rejected_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.rejected_candidates.len(),
            "plateau diagnosis requires three distinct rejected shadows"
        );
        ensure!(
            self.proposal_validity.is_finite() && self.gp_disadvantage.is_finite(),
            "plateau diagnosis contains a non-finite metric"
        );
        ensure!(
            [
                self.calibration_summary.as_str(),
                self.data_slice_summary.as_str(),
                self.latency_summary.as_str(),
                self.replay_summary.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty()),
            "plateau diagnosis is incomplete"
        );
        Ok(())
    }
}

pub const PLATEAU_DIAGNOSIS_V2_SCHEMA: &str = "PlateauDiagnosis.v2";
pub const RECIPE_CHANGE_DIRECTIVE_V1_SCHEMA: &str = "RecipeChangeDirective.v1";

/// One rejected shadow counted toward a recipe-scoped plateau, with the
/// recipe identity its trainer intent resolves to (None = legacy receipt
/// without a resolvable recipe; counted conservatively unless an operator
/// RecipeChangeDirective bridges it).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauCountedCandidateV1 {
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub recipe_identity_sha256: Option<String>,
}

/// Recipe-identity-scoped plateau diagnosis. The v1 pause keyed only on
/// (>=3 rejected shadows, unchanged parent) — a hard deadlock, because no
/// new shadow can train while paused, so the parent can never change. v2
/// counts a rejected shadow only when its training recipe identity matches
/// the recipe the lane would train with RIGHT NOW: a genuine recipe change
/// re-schedules training exactly as the v1 status string always promised
/// ("identical shadow training is paused"). v1 documents are preserved and
/// still written; this receipt is additive.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauDiagnosisV2 {
    pub schema: String,
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    /// Identity of the recipe the paused lane would use now; None when it
    /// could not be computed (scoping then stays conservative: everything
    /// counts, exactly the v1 behavior).
    pub current_recipe_identity_sha256: Option<String>,
    pub counted_candidates: Vec<PlateauCountedCandidateV1>,
    pub recipe_change_directive_present: bool,
    pub proposal_validity: f64,
    pub gp_disadvantage: f64,
    pub calibration_summary: String,
    pub data_slice_summary: String,
    pub latency_summary: String,
    pub replay_summary: String,
    pub diagnosed_at_ms: u128,
}

impl PlateauDiagnosisV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PLATEAU_DIAGNOSIS_V2_SCHEMA,
            "plateau diagnosis v2 schema mismatch"
        );
        ensure!(
            self.counted_candidates.len() >= 3
                && self
                    .counted_candidates
                    .iter()
                    .map(|candidate| &candidate.candidate_checkpoint_sha256)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.counted_candidates.len(),
            "plateau diagnosis v2 requires three distinct counted candidates"
        );
        ensure!(
            self.proposal_validity.is_finite() && self.gp_disadvantage.is_finite(),
            "plateau diagnosis v2 contains a non-finite metric"
        );
        Ok(())
    }
}

/// Operator receipt attesting that the training recipe HAS changed in a way
/// receipts cannot prove (the six legacy rejects predate recipe stamping in
/// TrainerIntent.v3). Its only effect: rejected shadows whose recipe
/// identity is UNRESOLVABLE stop counting toward the named parent's
/// plateau. Rejects that resolve to the currently-scheduled recipe identity
/// still pause — ground truth beats attestation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeChangeDirectiveV1 {
    pub schema: String,
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    /// Path or digest of the plateau diagnosis this directive answers.
    pub answers_diagnosis: String,
    /// The concrete, enumerated recipe changes the operator is attesting to.
    pub enumerated_changes: Vec<String>,
    pub created_at_ms: u128,
}

impl RecipeChangeDirectiveV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RECIPE_CHANGE_DIRECTIVE_V1_SCHEMA,
            "recipe change directive schema mismatch"
        );
        ensure!(
            !self.answers_diagnosis.trim().is_empty(),
            "recipe change directive must name the diagnosis it answers"
        );
        ensure!(
            !self.enumerated_changes.is_empty()
                && self
                    .enumerated_changes
                    .iter()
                    .all(|change| !change.trim().is_empty()),
            "recipe change directive requires enumerated concrete changes"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputProgressV2 {
    pub queue_depth: usize,
    pub ready_reserve: usize,
    pub dynamic_target: usize,
    pub p95_context_to_ready_ms: u64,
    pub freeze_to_publish_ms: u64,
    pub corpus_lag_labels: u64,
    pub supply_target_fraction: u16,
    pub disk_used_percent: u8,
    pub expansion_paused: bool,
}

impl ThroughputProgressV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dynamic_target == 4_096
                && self.supply_target_fraction <= 1_000
                && self.disk_used_percent <= 100,
            "flywheel throughput progress is outside its declared bounds"
        );
        ensure!(
            self.disk_used_percent < 85 || self.expansion_paused,
            "catalog expansion must stop at 85% disk usage"
        );
        Ok(())
    }

    pub fn next_expansion_paused(&self, disk_used_percent: u8) -> bool {
        if self.expansion_paused {
            disk_used_percent >= 80
        } else {
            disk_used_percent >= 85
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelProgressV2 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub accepted_generations: Vec<AcceptedGenerationV2>,
    pub causal_evidence_ledger_sha256: Sha256Digest,
    pub external_evaluation_sha256: BTreeMap<Task, Vec<Sha256Digest>>,
    pub throughput: ThroughputProgressV2,
    pub catalog_coverage: BTreeMap<Task, f64>,
    pub real_data_share: BTreeMap<Task, f64>,
    pub plateau: BTreeMap<Task, PlateauDiagnosisV1>,
    pub self_improving: bool,
    pub recursive_improvement_proven: bool,
}

impl FlywheelProgressV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == FLYWHEEL_PROGRESS_V2_SCHEMA,
            "flywheel progress schema is not v2"
        );
        self.throughput.validate()?;
        for generation in &self.accepted_generations {
            generation.validate()?;
        }
        for diagnosis in self.plateau.values() {
            diagnosis.validate()?;
        }
        ensure!(
            self.catalog_coverage
                .values()
                .chain(self.real_data_share.values())
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "flywheel breadth fraction is outside [0,1]"
        );
        let recursive = self.accepted_generations.iter().any(|child| {
            self.accepted_generations.iter().any(|parent| {
                child.task == parent.task
                    && child.parent_checkpoint_sha256 == parent.child_checkpoint_sha256
            })
        });
        ensure!(
            self.self_improving == !self.accepted_generations.is_empty()
                && self.recursive_improvement_proven == recursive,
            "flywheel improvement claim does not match its accepted generation chain"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn preflight(parent: &Sha256Digest) -> Vec<EvidencePreflightCohortV1> {
        (0..PREFLIGHT_COHORTS_REQUIRED)
            .map(|sequence| EvidencePreflightCohortV1 {
                sequence,
                task: Task::Binary,
                accepted_parent_checkpoint_sha256: parent.clone(),
                contexts: PREFLIGHT_CONTEXTS_PER_COHORT,
                requested_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                valid_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                completed: true,
            })
            .collect()
    }

    fn epoch(parent: &Sha256Digest) -> CausalEvidenceEpochV1 {
        CausalEvidenceEpochV1 {
            schema: CAUSAL_EVIDENCE_EPOCH_SCHEMA.to_owned(),
            task: Task::Binary,
            accepted_parent_checkpoint_sha256: parent.clone(),
            start_wave: 300,
            runtime_policy_sha256: digest("runtime"),
            proposer_policy_sha256: digest("proposer"),
            evaluator_policy_sha256: digest("evaluator"),
            catalog_eligibility_sha256: digest("eligibility"),
            requested_contexts: vec![EvidenceEpochContextV1 {
                context_identity: "context".to_owned(),
                context_sha256: digest("context"),
                requested_wave: 300,
            }],
            slots_per_context: EVIDENCE_SLOTS_PER_CONTEXT,
            preflight_cohorts: preflight(parent),
            opened_at_ms: 1,
        }
    }

    #[test]
    fn evidence_epoch_requires_preflight_and_is_non_resettable() {
        let parent = digest("parent");
        let original = epoch(&parent);
        original.validate().unwrap();
        let mut changed = original.clone();
        changed.start_wave += 1;
        assert!(ensure_evidence_epoch_immutable(&original, &changed).is_err());
        let mut failed_preflight = original.clone();
        failed_preflight.preflight_cohorts[9].valid_slots = 500;
        assert!(failed_preflight.validate().is_err());
    }

    #[test]
    fn proposal_cache_identity_survives_catalog_append_and_escalates() {
        let key = ContextProposalKeyV3 {
            task: Task::Regression,
            accepted_checkpoint_sha256: digest("parent"),
            context_sha256: digest("context"),
            seed: 7,
            decoding_policy_sha256: digest("decode"),
            reservoir_size: 64,
        };
        let first = context_proposal_cache_path(Path::new("/cache/catalog-a"), &key).unwrap();
        let second = context_proposal_cache_path(Path::new("/cache/catalog-a"), &key).unwrap();
        assert_eq!(first, second);
        assert_eq!(key.next_reservoir_size(), Some(128));

        let proposals = (0..EVIDENCE_SLOTS_PER_CONTEXT)
            .map(|slot| CachedContextProposalV3 {
                slot,
                candidate_sha256: digest(&format!("candidate-{slot}")),
                canonical_genome: format!("genome-{slot}"),
                decoder: "nucleus".to_owned(),
                decoder_seed: 7,
                tokens: vec![1, 2, 3],
                head_type: "ridge".to_owned(),
                reward_mean: 1.0,
                reward_uncertainty: 0.1,
                reward_lcb: 0.95,
                instability_probability: 0.1,
                selection_score: 0.85,
                candidate_risk_reason: None,
            })
            .collect::<Vec<_>>();
        let slots = (0..EVIDENCE_SLOTS_PER_CONTEXT)
            .map(|slot| ProposalSlotReceiptV3 {
                slot,
                status: ProposalSlotStatusV3::Ready,
                candidate_sha256: Some(digest(&format!("candidate-{slot}"))),
                rejections: BTreeMap::new(),
            })
            .collect();
        let artifact = ContextProposalArtifactV3 {
            schema: CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA.to_owned(),
            key,
            generated_at_ms: 1,
            decoded_candidates: 64,
            rejection_counts: all_rejection_counts(),
            slots,
            proposals,
        };
        artifact.validate().unwrap();
    }

    #[test]
    fn a_missing_slot_is_permanent_only_after_reservoir_256() {
        let mut artifact = ContextProposalArtifactV3 {
            schema: CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA.to_owned(),
            key: ContextProposalKeyV3 {
                task: Task::Binary,
                accepted_checkpoint_sha256: digest("parent"),
                context_sha256: digest("context"),
                seed: 7,
                decoding_policy_sha256: digest("decode"),
                reservoir_size: 128,
            },
            generated_at_ms: 1,
            decoded_candidates: 128,
            rejection_counts: all_rejection_counts(),
            slots: (0..EVIDENCE_SLOTS_PER_CONTEXT)
                .map(|slot| ProposalSlotReceiptV3 {
                    slot,
                    status: ProposalSlotStatusV3::Ready,
                    candidate_sha256: Some(digest(&format!("candidate-{slot}"))),
                    rejections: BTreeMap::new(),
                })
                .collect(),
            proposals: (0..EVIDENCE_SLOTS_PER_CONTEXT)
                .map(|slot| CachedContextProposalV3 {
                    slot,
                    candidate_sha256: digest(&format!("candidate-{slot}")),
                    canonical_genome: format!("genome-{slot}"),
                    decoder: "nucleus".to_owned(),
                    decoder_seed: 7,
                    tokens: vec![1, 2, 3],
                    head_type: "ridge".to_owned(),
                    reward_mean: 1.0,
                    reward_uncertainty: 0.1,
                    reward_lcb: 0.95,
                    instability_probability: 0.1,
                    selection_score: 0.85,
                    candidate_risk_reason: None,
                })
                .collect(),
        };
        artifact.slots[7] = ProposalSlotReceiptV3 {
            slot: 7,
            status: ProposalSlotStatusV3::PermanentFailure,
            candidate_sha256: None,
            rejections: BTreeMap::from([(ProposalRejectionReasonV3::Invalid, 256)]),
        };
        assert!(artifact.validate().is_err());
        artifact.key.reservoir_size = 256;
        artifact.proposals.pop();
        artifact.validate().unwrap();
    }

    fn provenance(license: Option<&str>) -> DatasetProvenanceV2 {
        DatasetProvenanceV2 {
            schema: DATASET_PROVENANCE_V2_SCHEMA.to_owned(),
            provider: "OpenML".to_owned(),
            provider_version: "1".to_owned(),
            upstream_root: "openml:1".to_owned(),
            license_spdx: license.map(str::to_owned),
            manual_license_approved: false,
            raw_sha256: BTreeMap::from([("data".to_owned(), digest("raw"))]),
            origin: DatasetOriginV2::Real,
            generator_superfamily: None,
            exact_family_sha256: digest("family"),
            near_duplicate_cluster_sha256: digest("cluster"),
        }
    }

    fn binary_profile() -> DatasetProfileV1 {
        DatasetProfileV1 {
            schema: DATASET_PROFILE_V1_SCHEMA.to_owned(),
            rows: 1_000,
            encoded_features: 20,
            missing_fraction: 0.0,
            class_count: Some(2),
            minority_majority_ratio: Some(0.2),
            numeric_target_distinct_finite: None,
            finite_target_fraction: None,
            size_slice: DatasetSizeSliceV1::Small,
            missingness_slice: MissingnessSliceV1::None,
            target_shape_slice: TargetShapeSliceV1::Balanced,
            embedding_novelty: 1.0,
            sensitive_or_pii_like: false,
            ambiguous_target: false,
            sparse_text: false,
            grouped_or_time_series: false,
            duplicate_or_leakage_prone: false,
        }
    }

    #[test]
    fn license_and_task_eligibility_fail_closed() {
        assert!(
            binary_profile()
                .eligibility(Task::Binary, &provenance(Some("CC-BY-4.0")))
                .unwrap()
                .eligible
        );
        assert!(
            !binary_profile()
                .eligibility(Task::Binary, &provenance(None))
                .unwrap()
                .eligible
        );
    }

    #[test]
    fn public_discovery_is_versioned_and_cross_provider_duplicates_cluster() {
        let openml = deterministic_openml_discovery(vec![
            OpenMlDatasetMetadataV1 {
                dataset_id: 2,
                version: 1,
                name: "later".to_owned(),
                license: Some("CC-BY-4.0".to_owned()),
                default_target_attribute: Some("target".to_owned()),
                original_data_url: None,
                download_url: "https://example.test/2".to_owned(),
                rows: 1_000,
                features: 10,
                missing_values: 0,
            },
            OpenMlDatasetMetadataV1 {
                dataset_id: 1,
                version: 1,
                name: "first".to_owned(),
                license: Some("CC0-1.0".to_owned()),
                default_target_attribute: Some("target".to_owned()),
                original_data_url: None,
                download_url: "https://example.test/1".to_owned(),
                rows: 1_000,
                features: 10,
                missing_values: 0,
            },
        ]);
        assert_eq!(openml[0].dataset_id, 1);
        let raw = BTreeMap::from([("data".to_owned(), digest("same bytes"))]);
        assert_eq!(
            cross_provider_duplicate_cluster(&raw).unwrap(),
            cross_provider_duplicate_cluster(&raw).unwrap()
        );
    }

    #[test]
    fn synthetic_family_identity_has_only_stable_components() {
        let family = SyntheticFamilyKeyV2 {
            generator_name: "make_classification".to_owned(),
            generator_version: "2".to_owned(),
            scenario: "covariate_shift".to_owned(),
            target_family: "binary".to_owned(),
            feature_family_set: BTreeSet::from(["numeric".to_owned()]),
            shift_pathology_class: "missing_at_random".to_owned(),
        };
        assert_eq!(
            family.exact_family_sha256().unwrap(),
            family.clone().exact_family_sha256().unwrap()
        );
    }

    fn panel_pool() -> SealedPanelPoolV2 {
        let entries = (0..1_000)
            .map(|index| {
                let pool = match index {
                    0..=699 => PanelPoolKindV2::Training,
                    700..=799 => PanelPoolKindV2::Development,
                    800..=949 => PanelPoolKindV2::Promotion,
                    _ => PanelPoolKindV2::Audit,
                };
                SealedPanelEntryV2 {
                    task: Task::Binary,
                    context_identity: format!("context-{index}"),
                    context_sha256: digest(&format!("context-{index}")),
                    source_family_sha256: digest(&format!("family-{index}")),
                    near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
                    origin: DatasetOriginV2::Real,
                    pool,
                    promotion_panel: (pool == PanelPoolKindV2::Promotion)
                        .then(|| "panel-a".to_owned()),
                }
            })
            .collect();
        SealedPanelPoolV2 {
            schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
            entries,
            created_at_ms: 1,
        }
    }

    #[test]
    fn sealed_panels_are_family_disjoint_and_single_use() {
        let pool = panel_pool();
        pool.validate().unwrap();
        let contexts = pool
            .promotion_panel(Task::Binary, "panel-a")
            .into_iter()
            .take(PROMOTION_MIN_CONTEXTS)
            .map(|entry| entry.context_sha256.clone())
            .collect();
        let receipt = SealedPanelConsumptionV1 {
            schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
            task: Task::Binary,
            panel: "panel-a".to_owned(),
            candidate_checkpoint_sha256: digest("candidate"),
            run_index: 0,
            seed: 9,
            context_sha256: contexts,
            consumed_at_ms: 1,
        };
        validate_panel_consumption(&pool, &receipt, &[]).unwrap();
        assert!(validate_panel_consumption(&pool, &receipt, &[receipt.clone()]).is_err());
    }

    #[test]
    fn sealed_panel_builder_creates_three_real_disjoint_panels_per_task() {
        let candidates = (0..5_120)
            .map(|index| SealedPanelCandidateV2 {
                task: if index % 2 == 0 {
                    Task::Binary
                } else {
                    Task::Regression
                },
                context_identity: format!("context-{index}"),
                context_sha256: Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                source_family_sha256: Sha256Digest::parse(format!("{:064x}", index + 10_000))
                    .unwrap(),
                near_duplicate_cluster_sha256: Sha256Digest::parse(format!(
                    "{:064x}",
                    index + 20_000
                ))
                .unwrap(),
                origin: if index < 1_024 {
                    DatasetOriginV2::Real
                } else {
                    DatasetOriginV2::Synthetic
                },
            })
            .collect();
        let pool = build_sealed_panel_pool_v2(candidates, 1).unwrap();
        pool.validate().unwrap();
        for task in [Task::Binary, Task::Regression] {
            let panels = pool
                .entries
                .iter()
                .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
                .filter_map(|entry| entry.promotion_panel.as_deref())
                .collect::<BTreeSet<_>>();
            assert_eq!(panels.len(), 3);
            assert!(panels
                .iter()
                .all(|panel| pool.promotion_panel(task, panel).len() == 128));
        }
    }

    fn promotion_observations(lift: f64) -> Vec<PromotionObservationV3> {
        (0..PROMOTION_MIN_CONTEXTS)
            .map(|index| PromotionObservationV3 {
                context_sha256: digest(&format!("context-{index}")),
                source_family_sha256: digest(&format!("family-{index}")),
                candidate_score: 1.0 + lift,
                accepted_score: 1.0,
                gp_score: 1.0,
                candidate_conditional_advantage: 0.02,
                accepted_conditional_advantage: 0.01,
                valid: true,
                replay_verified: true,
                candidate_failed: false,
                accepted_failed: false,
                candidate_latency_ms: 100,
                accepted_latency_ms: 100,
                strict_win: index == 0,
                equal_budget_win: index == 1,
                slices: BTreeMap::from([
                    ("origin".to_owned(), "real".to_owned()),
                    ("size".to_owned(), "small".to_owned()),
                    ("missingness".to_owned(), "none".to_owned()),
                    ("imbalance".to_owned(), "balanced".to_owned()),
                    ("target_shape".to_owned(), "balanced".to_owned()),
                ]),
            })
            .collect()
    }

    #[test]
    fn clustered_promotion_gates_require_external_and_gp_lift() {
        let accepted = evaluate_task_promotion_v3(
            Task::Binary,
            digest("parent"),
            digest("candidate"),
            digest("panel"),
            0,
            &promotion_observations(0.01),
            11,
            1,
        )
        .unwrap();
        assert!(accepted.accepted);
        let rejected = evaluate_task_promotion_v3(
            Task::Binary,
            digest("parent"),
            digest("candidate"),
            digest("panel"),
            0,
            &promotion_observations(0.0),
            11,
            1,
        )
        .unwrap();
        assert!(!rejected.accepted);
    }

    #[test]
    fn packed_v3_rejects_observations_copy_and_calibration_falls_back() {
        let column = |name: &str, rows| PackedColumnV3 {
            path: PathBuf::from(name),
            sha256: digest(name),
            dtype: "u1".to_owned(),
            rows,
        };
        let mut corpus = PackedCorpusV3 {
            schema: PACKED_CORPUS_V3_SCHEMA.to_owned(),
            watermark: 10,
            catalog_binding_sha256: digest("catalog"),
            selected_row_index: column("selected.u64", 10),
            task_column: column("tasks.u1", 10),
            split_column: column("splits.u1", 10),
            exact_pair_sidecar: column("pairs.jsonl", 0),
            label_shards: vec![PackedLabelShardV3 {
                first_receipt: 1,
                last_receipt: 10,
                rows: 10,
                path: PathBuf::from("labels/one.shard"),
                sha256: digest("shard"),
                reused_from_corpus_sha256: None,
            }],
            created_at_ms: 1,
        };
        corpus.validate().unwrap();
        corpus.selected_row_index.path = PathBuf::from("observations.jsonl");
        assert!(corpus.validate().is_err());

        let fallback = ranking_fallback(&CalibrationEvidenceV1 {
            error_correlation_lower_95: -0.1,
            interval_coverage: 0.8,
            target_coverage: 0.95,
            native_head_accuracy: 0.51,
            frequency_baseline_accuracy: 0.50,
        })
        .unwrap();
        assert_eq!(
            fallback.uncertainty,
            UncertaintyRankingSourceV1::EmpiricalEnsembleDispersion
        );
        assert!(!fallback.native_head_enabled);
    }

    #[test]
    fn recursive_claim_requires_a_child_of_an_accepted_child() {
        let first = AcceptedGenerationV2 {
            task: Task::Binary,
            parent_checkpoint_sha256: digest("root"),
            child_checkpoint_sha256: digest("child-1"),
            independent_external_passes: vec![digest("pass-1"), digest("pass-2"), digest("pass-3")],
            accepted_at_ms: 1,
        };
        let second = AcceptedGenerationV2 {
            task: Task::Binary,
            parent_checkpoint_sha256: first.child_checkpoint_sha256.clone(),
            child_checkpoint_sha256: digest("child-2"),
            independent_external_passes: vec![digest("pass-4"), digest("pass-5"), digest("pass-6")],
            accepted_at_ms: 2,
        };
        let progress = FlywheelProgressV2 {
            schema: FLYWHEEL_PROGRESS_V2_SCHEMA.to_owned(),
            generated_at_ms: 3,
            accepted_generations: vec![first, second],
            causal_evidence_ledger_sha256: digest("ledger"),
            external_evaluation_sha256: BTreeMap::new(),
            throughput: ThroughputProgressV2 {
                queue_depth: 4_096,
                ready_reserve: 4_096,
                dynamic_target: 4_096,
                p95_context_to_ready_ms: 1,
                freeze_to_publish_ms: 1,
                corpus_lag_labels: 0,
                supply_target_fraction: 1_000,
                disk_used_percent: 50,
                expansion_paused: false,
            },
            catalog_coverage: BTreeMap::from([(Task::Binary, 0.5)]),
            real_data_share: BTreeMap::from([(Task::Binary, 0.4)]),
            plateau: BTreeMap::new(),
            self_improving: true,
            recursive_improvement_proven: true,
        };
        progress.validate().unwrap();
    }
}
