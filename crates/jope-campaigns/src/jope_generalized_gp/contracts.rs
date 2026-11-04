use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use feat_math::genome::GenomeIr;
use serde::{Deserialize, Serialize};

use crate::search_v3::{canonical_sha256, validate_sha256, PredictionObjectV1};

pub const REGRESSION_SHADOW_PREFIX: &str = "e0aef7d74afd";
pub const BINARY_SHADOW_PREFIX: &str = "13b1c6bbe3e7";
pub const DEVELOPMENT_REGRESSION_DATASETS: u64 = 17_263;
pub const DEVELOPMENT_BINARY_DATASETS: u64 = 5_030;
pub const VALIDATION_REGRESSION_DATASETS: u64 = 1_928;
pub const VALIDATION_BINARY_DATASETS: u64 = 569;

pub(crate) fn schema(actual: &str, expected: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("schema must be {expected}, found {actual}"))
    }
}

pub(crate) fn finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

pub(crate) fn nonempty(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}

mod catalog_freeze;
mod contract_pilot;
mod evaluation;
mod genome;

pub use catalog_freeze::{
    CampaignArmV1, CampaignFreezeV1, CatalogCountsV1, CatalogDatasetV1, CatalogFreezeV1,
    CatalogPartitionV1, CheckpointFreezeV1, DatasetOriginV1, EvaluatorFreezeV1, EvolutionBudgetV1,
    FrozenArtifactV1, GpTaskV1, ModelShadowFreezeV1, PilotDatasetFreezeV1,
};
pub use contract_pilot::{
    artifact_exists, ensure_selection_before_test, validate_prediction_binding, LaunchGateV1,
    NoImprovementV1, PilotArmSummaryV1, PilotComparisonV1, PilotDatasetResultV1, PilotDecisionV1,
    SealedTestResultV1, TranchePlanV1, TrancheV1,
};
pub use evaluation::{
    sample_sd, AnchorBaselineV1, AnchorSkillV1, ArchiveCheckpointV1, ArchiveMemberV1,
    CandidateSplitEvaluationV1, Cv7CandidateEvidenceV1, Cv7PlanV1, Cv7SelectionFreezeV1,
    GenerationCheckpointV1, PredictionCacheKeyV1, RankedCandidateV1, RotatingSplitPlanV1,
    RowSplitV1, RuntimeObservationV1, SplitRowV1,
};
pub use genome::{
    raw_sha256, task_head_valid, GenerationZeroV1, GenomeAncestryV1, GenomeOriginV1, GenomeV1,
    GlobalGeneObservationV1, GlobalGeneRankV1, InferenceLedgerEntryV1, InferenceLedgerV1,
    PopulationMemberV1, ProposalFreezeV1,
};
