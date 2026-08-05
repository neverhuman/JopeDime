use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

pub const ACCEPTED_BY_TASK_SCHEMA: &str = "AcceptedByTask.v2";
pub const EVALUATION_KEY_SCHEMA: &str = "EvaluationKey.v2";
pub const EVALUATION_RESULT_SCHEMA: &str = "EvaluationResult.v2";
pub const CACHE_RECEIPT_SCHEMA: &str = "EvaluationCacheReceipt.v2";
pub const SHADOW_WAVE_SCHEMA: &str = "SelfPlayWave.v2";
pub const RUNTIME_BUNDLE_SCHEMA: &str = "RuntimeBundle.v3";
pub const COHORT_PROPOSAL_POINTER_SCHEMA: &str = "CohortProposalPointer.v1";
pub const WAVE_SUMMARY_SCHEMA: &str = "WaveSummary.v1";
pub const ENCODING_CHUNK_SCHEMA: &str = "EncodingChunk.v2";
pub const ENCODING_ADOPTION_RECEIPT_SCHEMA: &str = "EncodingAdoptionReceipt.v2";
pub const SUPPLY_STATE_SCHEMA: &str = "CohortSupplyState.v1";
pub const SUPPLY_STATE_V2_SCHEMA: &str = "CohortSupplyState.v2";
pub const RUNTIME_PINNED_INTENT_SCHEMA: &str = "RuntimePinnedIntent.v1";
pub const CATALOG_BINDING_SCHEMA: &str = "CatalogBinding.v1";
pub const CORPUS_FREEZE_STATE_SCHEMA: &str = "CorpusFreezeState.v2";
pub const PACKED_CORPUS_SCHEMA: &str = "PackedCorpus.v2";
pub const GPU_LANE_ADMISSION_SCHEMA: &str = "GpuLaneAdmission.v1";
pub const CAUSAL_EVIDENCE_LEDGER_SCHEMA: &str = "CausalEvidenceLedger.v1";
pub const CAUSAL_EVALUATION_RECEIPT_SCHEMA: &str = "CausalEvaluationReceipt.v2";
pub const CAUSAL_EVIDENCE_LEDGER_V2_SCHEMA: &str = "CausalEvidenceLedger.v2";
pub const CAUSAL_EVIDENCE_EPOCH_SCHEMA: &str = "CausalEvidenceEpoch.v1";
pub const CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA: &str = "CausalEvidenceLedger.v3";
pub const CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA: &str = "ContextProposalArtifact.v3";
pub const DATASET_PROVENANCE_V2_SCHEMA: &str = "DatasetProvenance.v2";
pub const DATASET_PROFILE_V1_SCHEMA: &str = "DatasetProfile.v1";
pub const SEALED_PANEL_POOL_V2_SCHEMA: &str = "SealedPanelPool.v2";
pub const SEALED_PANEL_CANDIDATE_SET_V2_SCHEMA: &str = "SealedPanelCandidateSet.v2";
pub const SEALED_PANEL_CONSUMPTION_V1_SCHEMA: &str = "SealedPanelConsumption.v1";
pub const SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA: &str = "SealedPanelContextFacts.v1";
pub const TASK_PROMOTION_EVALUATION_INTENT_V3_SCHEMA: &str = "TaskPromotionEvaluationIntent.v3";
pub const TASK_PROMOTION_OBSERVATION_SET_V3_SCHEMA: &str = "TaskPromotionObservationSet.v3";
pub const TASK_PROMOTION_EVALUATION_V3_SCHEMA: &str = "TaskPromotionEvaluation.v3";
pub const PACKED_CORPUS_V3_SCHEMA: &str = "PackedCorpus.v3";
pub const FLYWHEEL_PROGRESS_V2_SCHEMA: &str = "FlywheelProgress.v2";
pub const CANDIDATE_PROVENANCE_SCHEMA: &str = "CandidateProvenance.v2";
pub const CANARY_INTENT_SCHEMA: &str = "CanaryIntent.v2";
pub const TASK_CANARY_ACCEPTANCE_V2_SCHEMA: &str = "TaskCanaryAcceptance.v2";
pub const TRAINER_INTENT_SCHEMA: &str = "TrainerIntent.v2";
pub const TRAINER_INTENT_V3_SCHEMA: &str = "TrainerIntent.v3";
pub const SHADOW_CHECKPOINT_SCHEMA: &str = "ShadowCheckpoint.v1";
pub const SHADOW_BATTLE_SUMMARY_SCHEMA: &str = "ShadowBattleSummary.v1";
pub const FLYWHEEL_LIVENESS_SCHEMA: &str = "FlywheelLiveness.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Binary,
    Regression,
    Timeseries,
    Multiclass,
}

impl Task {
    /// Every wire task, legacy tasks first (order is load-bearing for the
    /// `find`-style scans that historically probed `[Regression, Binary]`).
    pub const ALL: [Task; 4] = [
        Task::Regression,
        Task::Binary,
        Task::Timeseries,
        Task::Multiclass,
    ];

    /// The 8-slot alternation cycle (D6): R,B,R,B,TS,R,B,MC. With only
    /// regression/binary eligible the starvation-skip reproduces the legacy
    /// strict R,B alternation exactly.
    pub const ROTATION_CYCLE: [Task; 8] = [
        Task::Regression,
        Task::Binary,
        Task::Regression,
        Task::Binary,
        Task::Timeseries,
        Task::Regression,
        Task::Binary,
        Task::Multiclass,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Regression => "regression",
            Self::Timeseries => "timeseries",
            Self::Multiclass => "multiclass",
        }
    }

    /// On-disk corpus directory for this task (D1): the multiclass corpus
    /// keeps its legacy `multi_label` directory name; every other task's
    /// directory matches its wire name.
    pub fn corpus_dir_name(self) -> &'static str {
        crate::task::TaskKind::from(self).corpus_dir_name()
    }

    /// Starting at slot `position` of [`Task::ROTATION_CYCLE`], return the
    /// first eligible task scanning forward (wrapping), plus the slot index
    /// immediately after the chosen slot. Ineligible tasks are skipped
    /// (starvation-skip). If NO task is eligible the slot at `position` is
    /// returned unconditionally, so the successor can never deadlock.
    pub fn rotation_successor(position: usize, eligible: impl Fn(Task) -> bool) -> (Task, usize) {
        let length = Self::ROTATION_CYCLE.len();
        for offset in 0..length {
            let slot = (position + offset) % length;
            let task = Self::ROTATION_CYCLE[slot];
            if eligible(task) {
                return (task, (slot + 1) % length);
            }
        }
        let slot = position % length;
        (Self::ROTATION_CYCLE[slot], (slot + 1) % length)
    }
}

impl From<crate::task::TaskKind> for Task {
    fn from(kind: crate::task::TaskKind) -> Self {
        match kind {
            crate::task::TaskKind::Regression => Self::Regression,
            crate::task::TaskKind::Binary => Self::Binary,
            crate::task::TaskKind::Timeseries => Self::Timeseries,
            crate::task::TaskKind::Multiclass => Self::Multiclass,
        }
    }
}

impl From<Task> for crate::task::TaskKind {
    fn from(task: Task) -> Self {
        match task {
            Task::Regression => Self::Regression,
            Task::Binary => Self::Binary,
            Task::Timeseries => Self::Timeseries,
            Task::Multiclass => Self::Multiclass,
        }
    }
}

impl Display for Task {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Task {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "binary" => Ok(Self::Binary),
            "regression" => Ok(Self::Regression),
            "timeseries" => Ok(Self::Timeseries),
            "multiclass" => Ok(Self::Multiclass),
            _ => bail!("task must be regression, binary, timeseries, or multiclass"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        ensure!(
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "digest must be 64 lowercase hexadecimal characters"
        );
        Ok(Self(value))
    }

    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(hex::encode(Sha256::digest(bytes)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for Sha256Digest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortProposalTaskScope {
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub context_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortProposalPointerV1 {
    pub schema: String,
    pub task_scopes: Vec<CohortProposalTaskScope>,
    pub cohort_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub decoder_configuration_sha256: Sha256Digest,
    pub seed: u64,
    pub proposal_sha256: Sha256Digest,
    pub proposal_job_id: String,
}

impl CohortProposalPointerV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == COHORT_PROPOSAL_POINTER_SCHEMA,
            "cohort proposal pointer schema is not v1"
        );
        ensure!(
            !self.proposal_job_id.trim().is_empty(),
            "proposal job identity is empty"
        );
        ensure!(
            matches!(self.task_scopes.len(), 1 | 2),
            "a proposal pointer must scope one or two task cohorts"
        );
        let mut tasks = self
            .task_scopes
            .iter()
            .map(|scope| scope.task)
            .collect::<Vec<_>>();
        tasks.sort_unstable();
        tasks.dedup();
        ensure!(
            tasks.len() == self.task_scopes.len(),
            "proposal pointer contains a duplicate task scope"
        );
        ensure!(
            self.task_scopes
                .iter()
                .all(|scope| scope.context_count == super::DATASETS_PER_COHORT),
            "each proposal task scope must contain exactly 64 contexts"
        );
        if self.task_scopes.len() == 2 {
            ensure!(
                self.task_scopes[0].accepted_checkpoint_sha256
                    == self.task_scopes[1].accepted_checkpoint_sha256,
                "task cohorts may share a proposal job only while their accepted checkpoints match"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_scope(
        &self,
        task: Task,
        accepted_checkpoint_sha256: &Sha256Digest,
        cohort_sha256: &Sha256Digest,
        catalog_sha256: &Sha256Digest,
        decoder_configuration_sha256: &Sha256Digest,
        seed: u64,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            &self.cohort_sha256 == cohort_sha256
                && &self.catalog_sha256 == catalog_sha256
                && &self.decoder_configuration_sha256 == decoder_configuration_sha256
                && self.seed == seed,
            "proposal pointer immutable cohort scope mismatch"
        );
        let scope = self
            .task_scopes
            .iter()
            .find(|scope| scope.task == task)
            .context("proposal pointer does not contain the requested task")?;
        ensure!(
            &scope.accepted_checkpoint_sha256 == accepted_checkpoint_sha256,
            "proposal pointer accepted checkpoint mismatch"
        );
        Ok(())
    }

    pub fn id(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheDispositionCounts {
    pub novel: usize,
    pub exact_cache_hit: usize,
    pub active_or_reserved: usize,
    pub cross_protocol_blocked: usize,
    pub requalification: usize,
    pub evaluated: usize,
}

impl CacheDispositionCounts {
    pub fn record(&mut self, disposition: CacheDisposition) {
        match disposition {
            CacheDisposition::Novel => self.novel += 1,
            CacheDisposition::ExactCacheHit => self.exact_cache_hit += 1,
            CacheDisposition::ActiveOrReserved => self.active_or_reserved += 1,
            CacheDisposition::CrossProtocolBlocked => self.cross_protocol_blocked += 1,
            CacheDisposition::Requalification => self.requalification += 1,
            CacheDisposition::Evaluated => self.evaluated += 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextCompositionSummary {
    pub dataset_content_sha256: Sha256Digest,
    pub nominal_d: usize,
    pub actual_d: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEligibility {
    pub reward_and_stability: bool,
    pub matched_dl_efficiency: bool,
    pub promotion: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaveSummaryV1 {
    pub schema: String,
    pub wave_id: String,
    pub task: Task,
    pub nominal_d: usize,
    pub actual_d_total: usize,
    pub native_direct_count: usize,
    pub matched_gp_count: usize,
    pub context_composition: Vec<ContextCompositionSummary>,
    pub cache_dispositions: CacheDispositionCounts,
    pub zero_cross_context_backfill: bool,
    pub recovery_accounting_only: bool,
    pub evidence_eligibility: EvidenceEligibility,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingDatasetIdentity {
    pub dataset_id: String,
    pub content_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingChunkV2 {
    pub schema: String,
    pub chunk_sha256: Sha256Digest,
    pub encoder_identity_sha256: Sha256Digest,
    pub attempt_number: u32,
    pub datasets: Vec<EncodingDatasetIdentity>,
    pub preemptible: bool,
    pub reserved_cpu_cores: usize,
    pub required_node: String,
    pub project: String,
}

impl EncodingChunkV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ENCODING_CHUNK_SCHEMA,
            "encoding chunk schema is not v2"
        );
        ensure!(
            !self.datasets.is_empty() && self.datasets.len() <= super::ENCODING_CHUNK_SIZE,
            "encoding chunk must contain 1..=256 datasets"
        );
        ensure!(self.attempt_number > 0, "encoding attempt number is zero");
        ensure!(
            self.preemptible
                && self.reserved_cpu_cores == 4
                && self.required_node == "xbabe3"
                && self.project == "jope-selfplay-breadth-encode",
            "encoding chunk resource reservation drift"
        );
        let unique = self
            .datasets
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            unique.len() == self.datasets.len(),
            "encoding chunk contains duplicate dataset identities"
        );
        let identity = self
            .datasets
            .iter()
            .map(|dataset| format!("{}:{}", dataset.dataset_id, dataset.content_sha256.as_str()))
            .collect::<Vec<_>>()
            .join("|");
        ensure!(
            self.chunk_sha256
                == Sha256Digest::of_bytes(
                    format!(
                        "breadth-encode-v2|{}|{identity}",
                        self.encoder_identity_sha256
                    )
                    .as_bytes()
                ),
            "encoding chunk content identity mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EncodingAdoptionDisposition {
    Succeeded {
        encoded_train_sha256: Sha256Digest,
        encoded_test_sha256: Sha256Digest,
    },
    Failed {
        error_class: String,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingAdoptionReceiptV2 {
    pub schema: String,
    pub dataset: EncodingDatasetIdentity,
    pub attempt_number: u32,
    pub job_identity: String,
    pub encoder_identity_sha256: Sha256Digest,
    pub disposition: EncodingAdoptionDisposition,
    pub recorded_at_ms: u64,
}

impl EncodingAdoptionReceiptV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ENCODING_ADOPTION_RECEIPT_SCHEMA,
            "encoding adoption receipt schema is not v2"
        );
        ensure!(
            !self.dataset.dataset_id.trim().is_empty(),
            "encoding receipt dataset id is empty"
        );
        ensure!(self.attempt_number > 0, "encoding attempt number is zero");
        ensure!(
            !self.job_identity.trim().is_empty(),
            "encoding job identity is empty"
        );
        ensure!(
            self.recorded_at_ms > 0,
            "encoding receipt timestamp is absent"
        );
        if let EncodingAdoptionDisposition::Failed { error_class } = &self.disposition {
            ensure!(
                !error_class.trim().is_empty(),
                "failed encoding receipt requires an error class"
            );
        }
        Ok(())
    }

    pub fn succeeded(&self) -> bool {
        matches!(
            self.disposition,
            EncodingAdoptionDisposition::Succeeded { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    DlDirect,
    ShadowDlDirect,
    GpComparator,
    GpExploration,
    CachedReference,
    Requalification,
}

impl Origin {
    pub fn is_gp(self) -> bool {
        matches!(self, Self::GpComparator | Self::GpExploration)
    }

    pub fn is_new_action(self) -> bool {
        !matches!(self, Self::CachedReference)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DlAncestryKind {
    #[default]
    None,
    AcceptedDl,
    ShadowDl,
    Mixed,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Parentage {
    #[serde(default)]
    pub parent_semantic_sha256: Vec<Sha256Digest>,
    #[serde(default)]
    pub has_dl_parent_or_ancestor: bool,
    #[serde(default)]
    pub dl_ancestry_kind: DlAncestryKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSignals {
    pub conditional_reward: f64,
    pub stability: f64,
    pub incumbent_lift_probability: f64,
    pub diversity: f64,
    /// Recorded for calibration, but intentionally receives zero ranking
    /// weight until a family-held calibration proves monotonic utility.
    pub raw_uncertainty: f64,
}

impl SelectionSignals {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.conditional_reward,
                self.stability,
                self.incumbent_lift_probability,
                self.diversity,
                self.raw_uncertainty,
            ]
            .into_iter()
            .all(f64::is_finite),
            "selection signals must be finite"
        );
        ensure!(
            (0.0..=1.0).contains(&self.incumbent_lift_probability),
            "incumbent lift probability must be in [0, 1]"
        );
        Ok(())
    }

    pub fn action_score(&self) -> f64 {
        // raw_uncertainty is deliberately absent.
        self.conditional_reward
            + 0.5 * self.stability
            + 2.0 * self.incumbent_lift_probability
            + 0.25 * self.diversity
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetContext {
    pub task: Task,
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub source_family: String,
    pub context_sha256: Sha256Digest,
}

impl DatasetContext {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.dataset_id.trim().is_empty(), "dataset id is empty");
        ensure!(
            !self.source_family.trim().is_empty(),
            "source family is empty"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCandidate {
    pub task: Task,
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub canonical_program: String,
    pub semantic_sha256: Sha256Digest,
    pub source_family: String,
    pub origin: Origin,
    pub parentage: Parentage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_log_probability: Option<f64>,
    pub proposal_rank: u32,
    pub seed: u64,
    pub signals: SelectionSignals,
}

impl PolicyCandidate {
    pub fn validate_for(
        &self,
        context: &DatasetContext,
        accepted_parent: &Sha256Digest,
    ) -> Result<()> {
        context.validate()?;
        self.signals.validate()?;
        ensure!(self.task == context.task, "candidate task/context mismatch");
        ensure!(
            self.dataset_id == context.dataset_id
                && self.dataset_content_sha256 == context.dataset_content_sha256
                && self.context_sha256 == context.context_sha256,
            "cross-context candidate backfill is forbidden"
        );
        ensure!(
            !self.canonical_program.trim().is_empty(),
            "canonical program is empty"
        );
        ensure!(
            Sha256Digest::of_bytes(self.canonical_program.as_bytes()) == self.semantic_sha256,
            "semantic digest does not match authoritative canonical bytes"
        );
        ensure!(
            !self.source_family.trim().is_empty(),
            "candidate source family is empty"
        );
        if let Some(value) = self.decoder_log_probability {
            ensure!(value.is_finite(), "decoder log probability must be finite");
        }
        match self.origin {
            Origin::DlDirect => {
                ensure!(
                    self.producer_checkpoint_sha256.as_ref() == Some(accepted_parent),
                    "dl_direct must be produced by the task's accepted parent"
                );
                ensure!(
                    self.behavior_checkpoint_sha256.as_ref() == Some(accepted_parent),
                    "dl_direct behavior checkpoint must equal its accepted parent"
                );
                ensure!(
                    self.parentage.parent_semantic_sha256.is_empty()
                        && !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "dl_direct must be a native checkpoint action, not an offspring"
                );
            }
            Origin::ShadowDlDirect => {
                ensure!(
                    self.producer_checkpoint_sha256.is_some()
                        && self.behavior_checkpoint_sha256 == self.producer_checkpoint_sha256,
                    "shadow_dl_direct requires one explicit shadow producer/behavior checkpoint"
                );
                ensure!(
                    self.parentage.parent_semantic_sha256.is_empty()
                        && !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "shadow_dl_direct must be a native shadow checkpoint action"
                );
                ensure!(
                    self.producer_checkpoint_sha256.as_ref() != Some(accepted_parent),
                    "shadow_dl_direct cannot masquerade as accepted-DL evidence"
                );
            }
            Origin::GpComparator | Origin::GpExploration => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "GP candidates cannot carry a producer checkpoint"
                );
                ensure!(
                    !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "GP candidates cannot have a DL parent or ancestor"
                );
            }
            Origin::CachedReference => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "cached references are not checkpoint actions"
                );
            }
            Origin::Requalification => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "protocol requalification is not a direct policy action"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationKeyV2 {
    pub schema: String,
    pub task: Task,
    pub dataset_content_sha256: Sha256Digest,
    pub semantic_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
}

impl EvaluationKeyV2 {
    pub fn new(
        task: Task,
        dataset_content_sha256: Sha256Digest,
        semantic_sha256: Sha256Digest,
        evaluator_sha256: Sha256Digest,
        protocol_sha256: Sha256Digest,
    ) -> Self {
        Self {
            schema: EVALUATION_KEY_SCHEMA.to_owned(),
            task,
            dataset_content_sha256,
            semantic_sha256,
            evaluator_sha256,
            protocol_sha256,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATION_KEY_SCHEMA,
            "evaluation key schema is not v2"
        );
        Ok(())
    }

    pub fn id(&self) -> Sha256Digest {
        let bytes = serde_json::to_vec(self).expect("evaluation key serialization cannot fail");
        Sha256Digest::of_bytes(&bytes)
    }

    pub fn same_semantic_pair(&self, other: &Self) -> bool {
        self.task == other.task
            && self.dataset_content_sha256 == other.dataset_content_sha256
            && self.semantic_sha256 == other.semantic_sha256
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDisposition {
    Novel,
    ExactCacheHit,
    ActiveOrReserved,
    CrossProtocolBlocked,
    Requalification,
    Evaluated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCacheReceiptV2 {
    pub schema: String,
    pub evaluation_key: EvaluationKeyV2,
    pub disposition: CacheDisposition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_result_sha256: Option<Sha256Digest>,
    pub counts_as_action: bool,
    pub counts_as_evaluation: bool,
    pub counts_as_statistical_evidence: bool,
}

impl EvaluationCacheReceiptV2 {
    pub fn exact_hit(key: EvaluationKeyV2, result: Sha256Digest) -> Self {
        Self {
            schema: CACHE_RECEIPT_SCHEMA.to_owned(),
            evaluation_key: key,
            disposition: CacheDisposition::ExactCacheHit,
            cached_result_sha256: Some(result),
            counts_as_action: false,
            counts_as_evaluation: false,
            counts_as_statistical_evidence: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CACHE_RECEIPT_SCHEMA,
            "cache receipt schema is not v2"
        );
        self.evaluation_key.validate()?;
        if self.disposition == CacheDisposition::ExactCacheHit {
            ensure!(
                self.cached_result_sha256.is_some()
                    && !self.counts_as_action
                    && !self.counts_as_evaluation
                    && !self.counts_as_statistical_evidence,
                "identical protocol hits cannot count as new evidence"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationOutcome {
    pub valid: bool,
    pub strict_win: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_class: Option<String>,
}

impl EvaluationOutcome {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.reward.is_none_or(f64::is_finite) && self.stability.is_none_or(f64::is_finite),
            "evaluation values must be finite"
        );
        ensure!(
            self.valid || self.failure_class.is_some(),
            "invalid outcomes require a failure class"
        );
        ensure!(
            !self.strict_win || self.valid,
            "an invalid outcome cannot be a strict win"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedTrainingMetadata {
    pub origin: Origin,
    pub parentage: Parentage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_log_probability: Option<f64>,
    pub proposal_rank: u32,
    pub seed: u64,
    pub context_sha256: Sha256Digest,
    pub cache_disposition: CacheDisposition,
}

impl From<&PolicyCandidate> for PackedTrainingMetadata {
    fn from(candidate: &PolicyCandidate) -> Self {
        Self {
            origin: candidate.origin,
            parentage: candidate.parentage.clone(),
            producer_checkpoint_sha256: candidate.producer_checkpoint_sha256.clone(),
            behavior_checkpoint_sha256: candidate.behavior_checkpoint_sha256.clone(),
            decoder_log_probability: candidate.decoder_log_probability,
            proposal_rank: candidate.proposal_rank,
            seed: candidate.seed,
            context_sha256: candidate.context_sha256.clone(),
            cache_disposition: CacheDisposition::Evaluated,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResultV2 {
    pub schema: String,
    pub evaluation_key: EvaluationKeyV2,
    pub result_sha256: Sha256Digest,
    pub metadata: PackedTrainingMetadata,
    pub outcome: EvaluationOutcome,
    pub authoritative: bool,
    pub completed_at_ms: u64,
}

impl EvaluationResultV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATION_RESULT_SCHEMA,
            "evaluation result schema is not v2"
        );
        self.evaluation_key.validate()?;
        self.outcome.validate()?;
        ensure!(self.completed_at_ms > 0, "completion timestamp is absent");
        ensure!(
            matches!(
                self.metadata.cache_disposition,
                CacheDisposition::Evaluated | CacheDisposition::Requalification
            ),
            "persisted outcomes must be evaluated or explicitly requalified"
        );
        match self.metadata.origin {
            Origin::DlDirect => ensure!(
                self.metadata.producer_checkpoint_sha256.is_some()
                    && self.metadata.parentage.parent_semantic_sha256.is_empty()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "dl_direct result attribution is inconsistent"
            ),
            Origin::ShadowDlDirect => ensure!(
                self.metadata.producer_checkpoint_sha256.is_some()
                    && self.metadata.behavior_checkpoint_sha256
                        == self.metadata.producer_checkpoint_sha256
                    && self.metadata.parentage.parent_semantic_sha256.is_empty()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "shadow_dl_direct result attribution is inconsistent"
            ),
            Origin::GpComparator | Origin::GpExploration => ensure!(
                self.metadata.producer_checkpoint_sha256.is_none()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "GP result attribution is inconsistent"
            ),
            Origin::CachedReference => bail!("cached references cannot create EvaluationResult.v2"),
            Origin::Requalification => ensure!(
                self.metadata.cache_disposition == CacheDisposition::Requalification,
                "requalification outcomes must remain in the requalification lane"
            ),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    #[test]
    fn task_wire_names_and_conversions_are_stable() {
        for task in Task::ALL {
            assert_eq!(
                serde_json::to_value(task).unwrap(),
                serde_json::Value::String(task.as_str().to_owned())
            );
            assert_eq!(task.as_str().parse::<Task>().unwrap(), task);
            assert_eq!(Task::from(crate::task::TaskKind::from(task)), task);
        }
        assert_eq!(Task::Multiclass.corpus_dir_name(), "multi_label");
        assert_eq!(Task::Timeseries.corpus_dir_name(), "timeseries_reduced_v2");
        assert!("multi_label".parse::<Task>().is_err());
    }

    #[test]
    fn rotation_successor_with_legacy_eligibility_reproduces_strict_alternation() {
        let legacy = |task: Task| matches!(task, Task::Regression | Task::Binary);
        let mut position = 0;
        let mut picks = Vec::new();
        for _ in 0..8 {
            let (task, next) = Task::rotation_successor(position, legacy);
            picks.push(task);
            position = next;
        }
        // Exactly the pre-widening R,B alternation, for eight successive calls.
        assert_eq!(
            picks,
            vec![
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
            ]
        );
    }

    #[test]
    fn rotation_successor_visits_every_eligible_task_and_never_deadlocks() {
        let mut position = 0;
        let mut picks = Vec::new();
        for _ in 0..8 {
            let (task, next) = Task::rotation_successor(position, |_| true);
            picks.push(task);
            position = next;
        }
        assert_eq!(picks, Task::ROTATION_CYCLE.to_vec());

        // Starvation-skip: with multiclass still ineligible the cycle covers
        // the other three tasks and returns to the top.
        let no_multiclass = |task: Task| task != Task::Multiclass;
        let mut position = 4;
        let mut picks = Vec::new();
        for _ in 0..4 {
            let (task, next) = Task::rotation_successor(position, no_multiclass);
            picks.push(task);
            position = next;
        }
        assert_eq!(
            picks,
            vec![
                Task::Timeseries,
                Task::Regression,
                Task::Binary,
                Task::Regression
            ]
        );

        // No eligible task at all: the current slot is returned unconditionally.
        let (task, next) = Task::rotation_successor(4, |_| false);
        assert_eq!(task, Task::Timeseries);
        assert_eq!(next, 5);
    }

    #[test]
    fn evaluation_key_is_protocol_exact() {
        let base = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
        );
        let changed = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('e'),
        );
        assert_ne!(base.id(), changed.id());
        assert!(base.same_semantic_pair(&changed));
    }

    #[test]
    fn cohort_proposal_pointer_scope_is_immutable_and_task_exact() {
        let pointer = CohortProposalPointerV1 {
            schema: COHORT_PROPOSAL_POINTER_SCHEMA.into(),
            task_scopes: vec![CohortProposalTaskScope {
                task: Task::Binary,
                accepted_checkpoint_sha256: digest('a'),
                context_count: 64,
            }],
            cohort_sha256: digest('b'),
            catalog_sha256: digest('c'),
            decoder_configuration_sha256: digest('d'),
            seed: 7,
            proposal_sha256: digest('e'),
            proposal_job_id: "cohort-proposer-binary-a".into(),
        };
        pointer
            .validate_scope(
                Task::Binary,
                &digest('a'),
                &digest('b'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .unwrap();
        assert!(pointer
            .validate_scope(
                Task::Regression,
                &digest('a'),
                &digest('b'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .is_err());
        assert!(pointer
            .validate_scope(
                Task::Binary,
                &digest('a'),
                &digest('f'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .is_err());
        assert_eq!(pointer.id().unwrap(), pointer.id().unwrap());
    }

    #[test]
    fn shared_proposal_job_requires_a_shared_checkpoint() {
        let pointer = CohortProposalPointerV1 {
            schema: COHORT_PROPOSAL_POINTER_SCHEMA.into(),
            task_scopes: vec![
                CohortProposalTaskScope {
                    task: Task::Binary,
                    accepted_checkpoint_sha256: digest('a'),
                    context_count: 64,
                },
                CohortProposalTaskScope {
                    task: Task::Regression,
                    accepted_checkpoint_sha256: digest('b'),
                    context_count: 64,
                },
            ],
            cohort_sha256: digest('c'),
            catalog_sha256: digest('d'),
            decoder_configuration_sha256: digest('e'),
            seed: 9,
            proposal_sha256: digest('f'),
            proposal_job_id: "invalid-shared-job".into(),
        };
        assert!(pointer.validate().is_err());
    }

    #[test]
    fn uncertainty_has_zero_action_weight() {
        let mut signals = SelectionSignals {
            conditional_reward: 1.0,
            stability: 0.5,
            incumbent_lift_probability: 0.75,
            diversity: 0.2,
            raw_uncertainty: -100.0,
        };
        let score = signals.action_score();
        signals.raw_uncertainty = 100.0;
        assert_eq!(signals.action_score(), score);
    }

    #[test]
    fn cache_hits_are_not_new_evidence() {
        let receipt = EvaluationCacheReceiptV2::exact_hit(
            EvaluationKeyV2::new(
                Task::Regression,
                digest('a'),
                digest('b'),
                digest('c'),
                digest('d'),
            ),
            digest('e'),
        );
        receipt.validate().unwrap();
        assert!(!receipt.counts_as_action);
        assert!(!receipt.counts_as_evaluation);
        assert!(!receipt.counts_as_statistical_evidence);
    }

    #[test]
    fn origin_attribution_rejects_gp_descendants_of_dl() {
        let context = DatasetContext {
            task: Task::Binary,
            dataset_id: "dataset".into(),
            dataset_content_sha256: digest('a'),
            source_family: "family".into(),
            context_sha256: digest('b'),
        };
        let program = "(pipe X (ridge))".to_owned();
        let candidate = PolicyCandidate {
            task: Task::Binary,
            dataset_id: "dataset".into(),
            dataset_content_sha256: digest('a'),
            context_sha256: digest('b'),
            semantic_sha256: Sha256Digest::of_bytes(program.as_bytes()),
            canonical_program: program,
            source_family: "family".into(),
            origin: Origin::GpComparator,
            parentage: Parentage {
                parent_semantic_sha256: vec![digest('c')],
                has_dl_parent_or_ancestor: true,
                dl_ancestry_kind: DlAncestryKind::AcceptedDl,
            },
            producer_checkpoint_sha256: None,
            behavior_checkpoint_sha256: None,
            decoder_log_probability: None,
            proposal_rank: 0,
            seed: 1,
            signals: SelectionSignals {
                conditional_reward: 0.1,
                stability: 0.1,
                incumbent_lift_probability: 0.5,
                diversity: 0.1,
                raw_uncertainty: 0.1,
            },
        };
        assert!(candidate.validate_for(&context, &digest('d')).is_err());
    }
}
