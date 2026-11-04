pub use crate::task::Task;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt::{Display, Formatter};

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

mod candidate;
mod cohort;
mod digest;
mod encoding;
mod evaluation;
mod summary;
#[cfg(test)]
mod tests;

pub use candidate::{
    DatasetContext, DlAncestryKind, Origin, Parentage, PolicyCandidate, SelectionSignals,
    SelfplayShadowOrigin,
};
pub use cohort::{CohortProposalPointerV1, CohortProposalTaskScope};
pub use digest::Sha256Digest;
pub use encoding::{
    EncodingAdoptionDisposition, EncodingAdoptionReceiptV2, EncodingChunkV2,
    EncodingDatasetIdentity,
};
pub use evaluation::{
    CacheDisposition, EvaluationCacheReceiptV2, EvaluationKeyV2, EvaluationOutcome,
    EvaluationResultV2, PackedTrainingMetadata,
};
pub use summary::{
    CacheDispositionCounts, ContextCompositionSummary, EvidenceEligibility, WaveSummaryV1,
};
