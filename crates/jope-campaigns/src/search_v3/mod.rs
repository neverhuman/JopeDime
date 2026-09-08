//! Search V3 scientific contracts and deterministic evidence selection.
//!
//! This module is additive. The legacy exact-CV implementation remains a
//! compatibility oracle and is intentionally not called or modified here.

mod checkpoint;
mod contracts;
mod evidence;
mod folds;
mod hybrid;
mod identity;
mod qualification;

pub use checkpoint::{CheckpointStoreV2, VerificationMode, VerificationReportV2};
pub use contracts::{
    CandidateProducerV1, ConditioningPolicyV1, FutureExportRequirementsV1, RegisteredFitnessV1,
    SearchBudgetV1, SearchCandidateIdentityV1, SearchCandidatePlanV1, SearchRunContractV3,
    SearchTaskV1,
};
pub use evidence::{
    build_generation_state, score_prediction, CandidateEvidenceV1, ConfirmedArchiveV1,
    EvaluationRecordV2, FitnessVectorV1, GenerationStateV2, PredictionObjectV1,
    PromotionScheduleV1, ScoreKeyV2, ScoringTapeV2, SelectionFreezeV2, TypedMetricValueV1,
};
pub use folds::{
    build_tabular_fold_plan, build_timeseries_fold_plan, FoldMembershipV2, FoldPlanV2,
    TabularFoldInputV2, TabularRowV2, TimeseriesFoldInputV2, TimeseriesOriginV2, TimeseriesRowV2,
};
pub use hybrid::{HybridContributionReceiptV1, HybridMatchedRunV1};
pub use identity::{canonical_json_bytes, canonical_sha256, validate_sha256};
pub use qualification::{qualify_against_exhaustive, SearchQualificationReportV1};
