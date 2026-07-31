use super::{
    DlAncestryKind, Sha256Digest, Task, CAUSAL_EVIDENCE_LEDGER_SCHEMA, GPU_LANE_ADMISSION_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, SHADOW_BATTLE_SUMMARY_SCHEMA, SHADOW_CHECKPOINT_SCHEMA,
    TRAINER_INTENT_SCHEMA, TRAINER_INTENT_V3_SCHEMA,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const GPU_COMPANION_CORES: usize = 4;
pub const GPU_IDLE_ALARM_MS: u128 = 5 * 60 * 1_000;
pub const GPU_ADMISSION_RETRY_MS: u128 = 2 * 60 * 1_000;
pub const SHADOW_CANARY_CONTEXTS: usize = 16;
pub const SHADOW_PROPOSALS_PER_CONTEXT: usize = 8;
pub const SHADOW_COMPARATORS_PER_CONTEXT: usize = 8;
pub const REQUIRED_CONSECUTIVE_CANARY_ACCEPTANCES: usize = 3;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLaneRole {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
}

impl GpuLaneRole {
    pub fn target_node(self) -> &'static str {
        match self {
            Self::CohortProposer => "xbabe1",
            Self::AlternatingTrainer => "xbabe2",
            Self::BreadthEncoder => "xbabe3",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CohortProposer => "cohort_proposer",
            Self::AlternatingTrainer => "alternating_trainer",
            Self::BreadthEncoder => "breadth_encoder",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuResourceReservation {
    pub gpus: usize,
    pub companion_cpu_cores: usize,
}

impl Default for GpuResourceReservation {
    fn default() -> Self {
        Self {
            gpus: 1,
            companion_cpu_cores: GPU_COMPANION_CORES,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuAdmissionStage {
    Requested,
    QueuePaused,
    HoldingEvaluators,
    WaitingForCapacity,
    Submitted,
    ReleasingEvaluators,
    Complete,
    DependencyFailed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryDisposition {
    ContinueRequest,
    ResumeHolding,
    ResumeCapacityWait,
    ConfirmSubmissionThenRelease,
    ResumeExactRelease,
    NothingToRecover,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuLaneAdmissionV1 {
    pub schema: String,
    pub role: GpuLaneRole,
    pub target_node: String,
    pub resource_reservation: GpuResourceReservation,
    pub immutable_spec_path: PathBuf,
    pub immutable_spec_sha256: Sha256Digest,
    pub immutable_job_identity: String,
    #[serde(default)]
    pub held_evaluator_ids: Vec<String>,
    #[serde(default)]
    pub released_evaluator_ids: Vec<String>,
    pub stage: GpuAdmissionStage,
    pub requested_at_ms: u128,
    pub updated_at_ms: u128,
    pub eligible_idle_since_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_capacity_backoff: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency_failure: Option<String>,
    pub recovery_disposition: RecoveryDisposition,
}

impl GpuLaneAdmissionV1 {
    pub fn new(
        role: GpuLaneRole,
        immutable_spec_path: PathBuf,
        immutable_spec_sha256: Sha256Digest,
        immutable_job_identity: String,
        now_ms: u128,
    ) -> Result<Self> {
        ensure!(
            !immutable_job_identity.trim().is_empty(),
            "GPU job identity is empty"
        );
        let value = Self {
            schema: GPU_LANE_ADMISSION_SCHEMA.to_owned(),
            role,
            target_node: role.target_node().to_owned(),
            resource_reservation: GpuResourceReservation::default(),
            immutable_spec_path,
            immutable_spec_sha256,
            immutable_job_identity,
            held_evaluator_ids: Vec::new(),
            released_evaluator_ids: Vec::new(),
            stage: GpuAdmissionStage::Requested,
            requested_at_ms: now_ms,
            updated_at_ms: now_ms,
            eligible_idle_since_ms: now_ms,
            last_capacity_backoff: None,
            dependency_failure: None,
            recovery_disposition: RecoveryDisposition::ContinueRequest,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == GPU_LANE_ADMISSION_SCHEMA,
            "GPU lane admission schema is not v1"
        );
        ensure!(
            self.target_node == self.role.target_node(),
            "GPU role/target-node reservation drift"
        );
        ensure!(
            self.resource_reservation == GpuResourceReservation::default(),
            "GPU lane must reserve one GPU and four companion cores"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "GPU job identity is empty"
        );
        let held = self.held_evaluator_ids.iter().collect::<BTreeSet<_>>();
        let released = self.released_evaluator_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            held.len() == self.held_evaluator_ids.len(),
            "held evaluator identity is duplicated"
        );
        ensure!(
            released.len() == self.released_evaluator_ids.len(),
            "released evaluator identity is duplicated"
        );
        ensure!(
            released.is_subset(&held),
            "an evaluator was released without an exact recorded hold"
        );
        if self.stage == GpuAdmissionStage::Complete {
            ensure!(
                held == released,
                "completed admission has asymmetric hold/release identities"
            );
        }
        Ok(())
    }

    pub fn record_held(&mut self, evaluator_id: String, now_ms: u128) -> Result<bool> {
        ensure!(
            matches!(
                self.stage,
                GpuAdmissionStage::QueuePaused
                    | GpuAdmissionStage::HoldingEvaluators
                    | GpuAdmissionStage::WaitingForCapacity
            ),
            "evaluator hold is not legal in the current admission stage"
        );
        if self.held_evaluator_ids.contains(&evaluator_id) {
            return Ok(false);
        }
        self.stage = GpuAdmissionStage::HoldingEvaluators;
        self.held_evaluator_ids.push(evaluator_id);
        self.held_evaluator_ids.sort();
        self.updated_at_ms = now_ms;
        self.recovery_disposition = RecoveryDisposition::ResumeHolding;
        self.validate()?;
        Ok(true)
    }

    pub fn record_released(&mut self, evaluator_id: &str, now_ms: u128) -> Result<bool> {
        ensure!(
            self.held_evaluator_ids.iter().any(|id| id == evaluator_id),
            "release identity was never held"
        );
        if self
            .released_evaluator_ids
            .iter()
            .any(|id| id == evaluator_id)
        {
            return Ok(false);
        }
        self.stage = GpuAdmissionStage::ReleasingEvaluators;
        self.released_evaluator_ids.push(evaluator_id.to_owned());
        self.released_evaluator_ids.sort();
        self.updated_at_ms = now_ms;
        self.recovery_disposition = RecoveryDisposition::ResumeExactRelease;
        self.validate()?;
        Ok(true)
    }

    pub fn unreleased_evaluator_ids(&self) -> Vec<&str> {
        let released = self
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.held_evaluator_ids
            .iter()
            .map(String::as_str)
            .filter(|id| !released.contains(id))
            .collect()
    }

    pub fn recover(&mut self) {
        self.recovery_disposition = match self.stage {
            GpuAdmissionStage::Requested => RecoveryDisposition::ContinueRequest,
            GpuAdmissionStage::QueuePaused | GpuAdmissionStage::HoldingEvaluators => {
                RecoveryDisposition::ResumeHolding
            }
            GpuAdmissionStage::WaitingForCapacity => RecoveryDisposition::ResumeCapacityWait,
            GpuAdmissionStage::Submitted => RecoveryDisposition::ConfirmSubmissionThenRelease,
            GpuAdmissionStage::ReleasingEvaluators => RecoveryDisposition::ResumeExactRelease,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed => {
                RecoveryDisposition::NothingToRecover
            }
        };
    }

    pub fn idle_alarm_due(&self, now_ms: u128) -> bool {
        !matches!(
            self.stage,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
        ) && now_ms.saturating_sub(self.eligible_idle_since_ms) > GPU_IDLE_ALARM_MS
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    NativeDlDirect,
    MatchedGpComparator,
    ShadowDlDirect,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceKey {
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub wave: u64,
    pub manifest_sha256: Sha256Digest,
    pub candidate_semantic_sha256: Sha256Digest,
    pub receipt_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceObservation {
    pub key: CausalEvidenceKey,
    pub context_identity: String,
    pub kind: EvidenceKind,
    pub successfully_ingested: bool,
    pub valid_proposal: bool,
    pub zero_cross_context_backfill: bool,
    pub completed_causal_parent_wave: bool,
    pub valid_original_lineage: bool,
    pub dl_ancestry_kind: DlAncestryKind,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceExclusions {
    pub wave_189: usize,
    pub shadow: usize,
    pub failed: usize,
    pub duplicate: usize,
    pub wrong_parent: usize,
    pub dl_ancestry: usize,
    pub cross_context_backfill: usize,
    pub incomplete_parent_wave: usize,
    pub invalid_original_lineage: usize,
    pub unmatched_partial_manifest: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCausalEvidence {
    pub accepted_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub completed_native_direct: usize,
    pub completed_matched_gp: usize,
    pub valid_native_direct: usize,
    pub native_direct_attempts: usize,
    pub proposal_validity: f64,
    pub zero_cross_context_backfill: bool,
    pub completed_causal_parent_wave: bool,
    pub valid_original_lineage: bool,
    pub eligible_waves: BTreeSet<u64>,
    pub exclusions: EvidenceExclusions,
    pub production_gate_open: bool,
    #[serde(default)]
    pub gate_reasons: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceLedgerV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub tasks: BTreeMap<Task, TaskCausalEvidence>,
    pub records: Vec<CausalEvidenceObservation>,
}

impl CausalEvidenceLedgerV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_LEDGER_SCHEMA,
            "causal evidence ledger schema is not v1"
        );
        for (task, row) in &self.tasks {
            ensure!(
                row.completed_matched_gp <= row.completed_native_direct,
                "{task} matched-GP count exceeds direct evidence"
            );
            ensure!(
                row.proposal_validity.is_finite() && (0.0..=1.0).contains(&row.proposal_validity),
                "{task} proposal validity is outside [0,1]"
            );
            if row.production_gate_open {
                ensure!(
                    row.completed_native_direct >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.completed_matched_gp >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.completed_matched_gp == row.completed_native_direct
                        && row.proposal_validity >= 0.99
                        && row.zero_cross_context_backfill
                        && row.completed_causal_parent_wave
                        && row.valid_original_lineage,
                    "{task} production gate opened without strict causal evidence"
                );
            }
        }
        Ok(())
    }
}

pub fn build_causal_evidence_ledger(
    accepted_parents: &BTreeMap<Task, Sha256Digest>,
    observations: Vec<CausalEvidenceObservation>,
    generated_at_ms: u128,
) -> Result<CausalEvidenceLedgerV1> {
    let mut tasks = [Task::Binary, Task::Regression]
        .into_iter()
        .map(|task| {
            (
                task,
                TaskCausalEvidence {
                    accepted_parent_checkpoint_sha256: accepted_parents.get(&task).cloned(),
                    zero_cross_context_backfill: true,
                    valid_original_lineage: true,
                    ..TaskCausalEvidence::default()
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut unique = BTreeSet::new();
    let mut eligible_direct = Vec::new();
    let mut comparators: BTreeMap<(Task, u64, String), Vec<CausalEvidenceObservation>> =
        BTreeMap::new();

    for observation in observations {
        let row = tasks
            .get_mut(&observation.key.task)
            .expect("both task rows are initialized");
        if observation.kind == EvidenceKind::NativeDlDirect {
            row.native_direct_attempts += 1;
            if observation.valid_proposal {
                row.valid_native_direct += 1;
            }
        }
        if observation.key.wave == 189 {
            row.exclusions.wave_189 += 1;
            continue;
        }
        if observation.kind == EvidenceKind::ShadowDlDirect {
            row.exclusions.shadow += 1;
            continue;
        }
        if observation.kind == EvidenceKind::Failed || !observation.successfully_ingested {
            row.exclusions.failed += 1;
            continue;
        }
        if !unique.insert(observation.key.clone()) {
            row.exclusions.duplicate += 1;
            continue;
        }
        if accepted_parents.get(&observation.key.task)
            != Some(&observation.key.accepted_parent_checkpoint_sha256)
        {
            row.exclusions.wrong_parent += 1;
            continue;
        }
        if observation.dl_ancestry_kind != DlAncestryKind::None {
            row.exclusions.dl_ancestry += 1;
            continue;
        }
        if !observation.zero_cross_context_backfill {
            row.zero_cross_context_backfill = false;
            row.exclusions.cross_context_backfill += 1;
            continue;
        }
        if !observation.completed_causal_parent_wave {
            row.exclusions.incomplete_parent_wave += 1;
            continue;
        }
        if !observation.valid_original_lineage {
            row.valid_original_lineage = false;
            row.exclusions.invalid_original_lineage += 1;
            continue;
        }
        row.completed_causal_parent_wave = true;
        row.eligible_waves.insert(observation.key.wave);
        match observation.kind {
            EvidenceKind::NativeDlDirect => eligible_direct.push(observation),
            EvidenceKind::MatchedGpComparator => comparators
                .entry((
                    observation.key.task,
                    observation.key.wave,
                    observation.context_identity.clone(),
                ))
                .or_default()
                .push(observation),
            EvidenceKind::ShadowDlDirect | EvidenceKind::Failed => unreachable!(),
        }
    }

    for values in comparators.values_mut() {
        values.sort_by(|left, right| left.key.cmp(&right.key));
    }
    eligible_direct.sort_by(|left, right| left.key.cmp(&right.key));
    let mut retained = Vec::new();
    for direct in eligible_direct {
        let row = tasks
            .get_mut(&direct.key.task)
            .expect("both task rows are initialized");
        row.completed_native_direct += 1;
        let group = (
            direct.key.task,
            direct.key.wave,
            direct.context_identity.clone(),
        );
        if let Some(pool) = comparators.get_mut(&group) {
            if let Some(comparator) = pool.pop() {
                row.completed_matched_gp += 1;
                retained.push(comparator);
            } else {
                row.exclusions.unmatched_partial_manifest += 1;
            }
        } else {
            row.exclusions.unmatched_partial_manifest += 1;
        }
        retained.push(direct);
    }
    for row in tasks.values_mut() {
        row.proposal_validity = if row.native_direct_attempts == 0 {
            0.0
        } else {
            row.valid_native_direct as f64 / row.native_direct_attempts as f64
        };
        if row.completed_native_direct < MIN_PARENT_DIRECT_OUTCOMES {
            row.gate_reasons.push(format!(
                "native direct evidence {}/{}",
                row.completed_native_direct, MIN_PARENT_DIRECT_OUTCOMES
            ));
        }
        if row.completed_matched_gp < MIN_PARENT_DIRECT_OUTCOMES {
            row.gate_reasons.push(format!(
                "matched GP evidence {}/{}",
                row.completed_matched_gp, MIN_PARENT_DIRECT_OUTCOMES
            ));
        }
        if row.completed_matched_gp != row.completed_native_direct {
            row.gate_reasons.push(format!(
                "matched GP evidence must equal native direct evidence ({}/{})",
                row.completed_matched_gp, row.completed_native_direct
            ));
        }
        if row.proposal_validity < 0.99 {
            row.gate_reasons.push(format!(
                "proposal validity {:.4} < 0.99",
                row.proposal_validity
            ));
        }
        if !row.zero_cross_context_backfill {
            row.gate_reasons
                .push("cross-context backfill is nonzero".to_owned());
        }
        if !row.completed_causal_parent_wave {
            row.gate_reasons
                .push("no completed causal parent wave".to_owned());
        }
        if !row.valid_original_lineage {
            row.gate_reasons
                .push("original accepted lineage is invalid".to_owned());
        }
        row.production_gate_open = row.gate_reasons.is_empty();
    }
    let ledger = CausalEvidenceLedgerV1 {
        schema: CAUSAL_EVIDENCE_LEDGER_SCHEMA.to_owned(),
        generated_at_ms,
        tasks,
        records: retained,
    };
    ledger.validate()?;
    Ok(ledger)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainerMode {
    Shadow,
    Production,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerIntentV2 {
    pub schema: String,
    pub task: Task,
    pub mode: TrainerMode,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub warm_start_checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub seed: u64,
    pub immutable_job_identity: String,
    pub immutable_spec_sha256: Sha256Digest,
    pub created_at_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_bundle_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_bundle_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_binding_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_binding_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<serde_json::Value>,
}

impl TrainerIntentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.schema.as_str(),
                TRAINER_INTENT_SCHEMA | TRAINER_INTENT_V3_SCHEMA
            ),
            "trainer intent schema is not v2 or v3"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "trainer job identity is empty"
        );
        match self.mode {
            TrainerMode::Production => {
                ensure!(
                    self.shadow_parent_checkpoint_sha256.is_none()
                        && self.warm_start_checkpoint_sha256
                            == self.accepted_parent_checkpoint_sha256,
                    "production training must warm-start only from the task accepted pointer"
                );
            }
            TrainerMode::Shadow => {
                let expected = self
                    .shadow_parent_checkpoint_sha256
                    .as_ref()
                    .unwrap_or(&self.accepted_parent_checkpoint_sha256);
                ensure!(
                    &self.warm_start_checkpoint_sha256 == expected,
                    "shadow warm-start is outside its accepted-parent epoch"
                );
            }
        }
        if self.schema == TRAINER_INTENT_V3_SCHEMA {
            ensure!(
                self.runtime_bundle_path.is_some()
                    && self.runtime_bundle_sha256.is_some()
                    && self.catalog_binding_path.is_some()
                    && self.catalog_binding_sha256.is_some()
                    && self.recipe.is_some(),
                "trainer intent v3 lacks runtime, catalog, or recipe pins"
            );
            let recipe = self.recipe.as_ref().expect("checked");
            let broad = recipe
                .get("broad_replay_fraction")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            let paired = recipe
                .get("exact_causal_pair_fraction")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            ensure!(
                broad.is_finite() && paired.is_finite() && (broad + paired - 1.0).abs() < 1e-9,
                "trainer intent v3 recipe fractions must sum to one"
            );
            if paired > 0.0 {
                ensure!(
                    self.mode == TrainerMode::Shadow
                        && (broad - 0.75).abs() < 1e-9
                        && (paired - 0.25).abs() < 1e-9,
                    "causal-pair training is shadow-only 75/25"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowCheckpointV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub model_hashes: BTreeMap<String, Sha256Digest>,
    pub trainer_intent_sha256: Sha256Digest,
    pub created_at_ms: u128,
    pub promotion_eligible: bool,
}

impl ShadowCheckpointV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SHADOW_CHECKPOINT_SCHEMA,
            "shadow checkpoint schema is not v1"
        );
        ensure!(
            !self.promotion_eligible,
            "shadow checkpoints are explicitly promotion-ineligible"
        );
        ensure!(
            self.checkpoint_sha256 != self.accepted_parent_checkpoint_sha256,
            "shadow checkpoint equals its accepted parent"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryContextKind {
    Frontier,
    Hard,
    Anchor,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowBattleMetrics {
    pub paired_validity: f64,
    pub paired_novelty: f64,
    pub paired_win_rate: f64,
    pub paired_stability: f64,
    pub paired_efficiency: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowBattleSummaryV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub shadow_checkpoint_sha256: Sha256Digest,
    pub contexts: Vec<CanaryContextKind>,
    pub shadow_proposals_per_context: usize,
    pub matched_gp_comparators_per_context: usize,
    pub metrics: ShadowBattleMetrics,
    pub gp_seed_eligible_winners: Vec<Sha256Digest>,
    pub promotion_eligible: bool,
    pub created_at_ms: u128,
}

impl ShadowBattleSummaryV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SHADOW_BATTLE_SUMMARY_SCHEMA,
            "shadow battle summary schema is not v1"
        );
        ensure!(
            !self.promotion_eligible,
            "shadow battle evidence cannot satisfy promotion"
        );
        ensure!(
            self.contexts.len() == SHADOW_CANARY_CONTEXTS
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Frontier)
                    .count()
                    == 12
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Hard)
                    .count()
                    == 2
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Anchor)
                    .count()
                    == 2,
            "shadow canary must contain 12 frontier, 2 hard, and 2 anchor contexts"
        );
        ensure!(
            self.shadow_proposals_per_context == SHADOW_PROPOSALS_PER_CONTEXT
                && self.matched_gp_comparators_per_context == SHADOW_COMPARATORS_PER_CONTEXT,
            "shadow canary must use eight shadow proposals and eight matched GP comparators"
        );
        ensure!(
            [
                self.metrics.paired_validity,
                self.metrics.paired_novelty,
                self.metrics.paired_win_rate,
                self.metrics.paired_stability,
                self.metrics.paired_efficiency,
            ]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
            "shadow paired metric is outside [0,1]"
        );
        Ok(())
    }
}

pub fn canonical_shadow_canary_contexts() -> Vec<CanaryContextKind> {
    let mut contexts = vec![CanaryContextKind::Frontier; 12];
    contexts.extend([CanaryContextKind::Hard; 2]);
    contexts.extend([CanaryContextKind::Anchor; 2]);
    contexts
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowAlternationState {
    #[serde(default)]
    pub next_task: Option<Task>,
    #[serde(default)]
    pub accepted_parent_by_task: BTreeMap<Task, Sha256Digest>,
    #[serde(default)]
    pub last_shadow_by_task: BTreeMap<Task, Sha256Digest>,
}

impl ShadowAlternationState {
    pub fn next(&mut self, accepted: &BTreeMap<Task, Sha256Digest>) -> Task {
        for task in [Task::Regression, Task::Binary] {
            if self.accepted_parent_by_task.get(&task) != accepted.get(&task) {
                self.last_shadow_by_task.remove(&task);
            }
        }
        self.accepted_parent_by_task = accepted.clone();
        let task = self.next_task.unwrap_or(Task::Regression);
        self.next_task = Some(task.other());
        task
    }

    pub fn warm_start(&self, task: Task) -> Option<&Sha256Digest> {
        self.last_shadow_by_task
            .get(&task)
            .or_else(|| self.accepted_parent_by_task.get(&task))
    }
}

pub fn production_task(ledger: &CausalEvidenceLedgerV1) -> Option<Task> {
    [Task::Regression, Task::Binary].into_iter().find(|task| {
        ledger
            .tasks
            .get(task)
            .is_some_and(|row| row.production_gate_open)
    })
}

pub fn can_admit_dl_origin_seed(
    total_after_insert: usize,
    accepted_dl_origin_after_insert: usize,
    shadow_dl_origin_after_insert: usize,
) -> bool {
    accepted_dl_origin_after_insert.saturating_add(shadow_dl_origin_after_insert)
        <= total_after_insert / 3
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCanaryStreaks {
    #[serde(default)]
    pub streaks: BTreeMap<Task, usize>,
}

impl TaskCanaryStreaks {
    pub fn record(&mut self, task: Task, accepted: bool) -> bool {
        let streak = self.streaks.entry(task).or_default();
        *streak = if accepted {
            streak.saturating_add(1)
        } else {
            0
        };
        *streak >= REQUIRED_CONSECUTIVE_CANARY_ACCEPTANCES
    }

    pub fn reset_after_promotion(&mut self, task: Task) {
        self.streaks.insert(task, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    fn observation(
        task: Task,
        parent: char,
        semantic: char,
        receipt: char,
        kind: EvidenceKind,
    ) -> CausalEvidenceObservation {
        CausalEvidenceObservation {
            key: CausalEvidenceKey {
                task,
                accepted_parent_checkpoint_sha256: digest(parent),
                wave: 190,
                manifest_sha256: digest('a'),
                candidate_semantic_sha256: digest(semantic),
                receipt_sha256: digest(receipt),
            },
            context_identity: "context".to_owned(),
            kind,
            successfully_ingested: true,
            valid_proposal: true,
            zero_cross_context_backfill: true,
            completed_causal_parent_wave: true,
            valid_original_lineage: true,
            dl_ancestry_kind: DlAncestryKind::None,
        }
    }

    #[test]
    fn admission_recovers_exact_hold_release_symmetry() {
        let mut admission = GpuLaneAdmissionV1::new(
            GpuLaneRole::AlternatingTrainer,
            "/tmp/spec".into(),
            digest('a'),
            "trainer".to_owned(),
            1,
        )
        .unwrap();
        admission.stage = GpuAdmissionStage::QueuePaused;
        admission.record_held("eval-1".to_owned(), 2).unwrap();
        admission.record_held("eval-2".to_owned(), 3).unwrap();
        admission.stage = GpuAdmissionStage::Submitted;
        admission.recover();
        assert_eq!(
            admission.recovery_disposition,
            RecoveryDisposition::ConfirmSubmissionThenRelease
        );
        admission.record_released("eval-1", 4).unwrap();
        admission.recover();
        assert_eq!(
            admission.recovery_disposition,
            RecoveryDisposition::ResumeExactRelease
        );
        assert_eq!(admission.unreleased_evaluator_ids(), vec!["eval-2"]);
        assert!(admission.record_released("never-held", 5).is_err());
        admission.record_released("eval-2", 6).unwrap();
        admission.stage = GpuAdmissionStage::Complete;
        admission.validate().unwrap();
    }

    #[test]
    fn lane_resources_are_node_specific_and_fixed() {
        for (role, node) in [
            (GpuLaneRole::CohortProposer, "xbabe1"),
            (GpuLaneRole::AlternatingTrainer, "xbabe2"),
            (GpuLaneRole::BreadthEncoder, "xbabe3"),
        ] {
            let row =
                GpuLaneAdmissionV1::new(role, "/tmp/spec".into(), digest('a'), "job".into(), 1)
                    .unwrap();
            assert_eq!(row.target_node, node);
            assert_eq!(row.resource_reservation.gpus, 1);
            assert_eq!(row.resource_reservation.companion_cpu_cores, 4);
        }
    }

    #[test]
    fn evidence_deduplicates_and_excludes_shadow_and_wave_189() {
        let accepted =
            BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
        let direct = observation(Task::Binary, 'b', 'd', 'e', EvidenceKind::NativeDlDirect);
        let comparator = observation(
            Task::Binary,
            'b',
            'f',
            'a',
            EvidenceKind::MatchedGpComparator,
        );
        let mut wave_189 = direct.clone();
        wave_189.key.wave = 189;
        wave_189.key.candidate_semantic_sha256 = digest('1');
        let mut shadow = direct.clone();
        shadow.kind = EvidenceKind::ShadowDlDirect;
        shadow.key.candidate_semantic_sha256 = digest('2');
        let ledger = build_causal_evidence_ledger(
            &accepted,
            vec![direct.clone(), direct, comparator, wave_189, shadow],
            1,
        )
        .unwrap();
        let binary = &ledger.tasks[&Task::Binary];
        assert_eq!(binary.completed_native_direct, 1);
        assert_eq!(binary.completed_matched_gp, 1);
        assert_eq!(binary.exclusions.duplicate, 1);
        assert_eq!(binary.exclusions.wave_189, 1);
        assert_eq!(binary.exclusions.shadow, 1);
    }

    #[test]
    fn task_parent_selection_and_shadow_lineage_reset() {
        let first = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
        let mut state = ShadowAlternationState::default();
        assert_eq!(state.next(&first), Task::Regression);
        state
            .last_shadow_by_task
            .insert(Task::Regression, digest('d'));
        assert_eq!(state.next(&first), Task::Binary);
        assert_eq!(state.warm_start(Task::Regression), Some(&digest('d')));
        let changed =
            BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('e'))]);
        assert_eq!(state.next(&changed), Task::Regression);
        assert_eq!(state.warm_start(Task::Regression), Some(&digest('e')));
    }

    #[test]
    fn production_intent_cannot_use_shadow_lineage() {
        let intent = TrainerIntentV2 {
            schema: TRAINER_INTENT_SCHEMA.to_owned(),
            task: Task::Regression,
            mode: TrainerMode::Production,
            accepted_parent_checkpoint_sha256: digest('a'),
            shadow_parent_checkpoint_sha256: Some(digest('d')),
            warm_start_checkpoint_sha256: digest('d'),
            corpus_identity_sha256: digest('c'),
            corpus_watermark: 1,
            seed: 1,
            immutable_job_identity: "job".to_owned(),
            immutable_spec_sha256: digest('d'),
            created_at_ms: 1,
            runtime_bundle_path: None,
            runtime_bundle_sha256: None,
            catalog_binding_path: None,
            catalog_binding_sha256: None,
            recipe: None,
        };
        assert!(intent.validate().is_err());
    }

    #[test]
    fn shadow_canary_shape_and_dl_origin_cap_are_strict() {
        let summary = ShadowBattleSummaryV1 {
            schema: SHADOW_BATTLE_SUMMARY_SCHEMA.to_owned(),
            task: Task::Regression,
            accepted_parent_checkpoint_sha256: digest('a'),
            shadow_checkpoint_sha256: digest('f'),
            contexts: canonical_shadow_canary_contexts(),
            shadow_proposals_per_context: 8,
            matched_gp_comparators_per_context: 8,
            metrics: ShadowBattleMetrics {
                paired_validity: 1.0,
                paired_novelty: 0.5,
                paired_win_rate: 0.5,
                paired_stability: 0.5,
                paired_efficiency: 0.5,
            },
            gp_seed_eligible_winners: Vec::new(),
            promotion_eligible: false,
            created_at_ms: 1,
        };
        summary.validate().unwrap();
        assert!(can_admit_dl_origin_seed(6, 1, 1));
        assert!(!can_admit_dl_origin_seed(6, 2, 1));
    }

    #[test]
    fn promotion_requires_three_independent_task_acceptances() {
        let mut streaks = TaskCanaryStreaks::default();
        assert!(!streaks.record(Task::Binary, true));
        assert!(!streaks.record(Task::Binary, true));
        assert!(streaks.record(Task::Binary, true));
        assert_eq!(streaks.streaks.get(&Task::Regression), None);
        streaks.reset_after_promotion(Task::Binary);
        assert_eq!(streaks.streaks[&Task::Binary], 0);
    }
}
