use super::*;

pub(super) const MICRO_WAVE_KIND: &str = "deterministic_128_dataset_micro_wave";
pub(super) const WAVE_SCHEMA: &str = "JopeRealtimeSelfplayWave.v1";
pub(super) const DATASETS_PER_WAVE: usize = 128;
pub(super) const CANDIDATES_PER_DATASET: usize = 256;
pub(super) const COHORT_PROPOSAL_SEED: u64 = 20_260_726;
pub(super) const COHORT_NATIVE_D: usize = 8;
pub(super) const SUPPLY_STATE_V1_SCHEMA: &str = "CohortSupplyState.v1";
pub(super) const SUPPLY_STATE_V2_SCHEMA: &str = "CohortSupplyState.v2";
pub(super) const RUNTIME_PINNED_INTENT_SCHEMA: &str = "RuntimePinnedIntent.v1";
pub(super) const COHORT_CONTEXT_MIGRATION_SCHEMA: &str = "CohortContextMigrationReceipt.v1";
pub(super) const COHORT_CONTEXT_EXCLUSION_SCHEMA: &str = "CohortContextExclusionReceipt.v1";
pub(super) const ERROR_BACKOFF_MAX_SECONDS: u64 = 120;
pub(super) const MIN_USEFUL_WORK: usize = 6_144;
pub(super) const CANDIDATE_EVALUATIONS_PER_WAVE: usize = DATASETS_PER_WAVE * CANDIDATES_PER_DATASET;
pub(super) const MIN_CANDIDATE_EVALUATION_WORK: usize = 8 * CANDIDATE_EVALUATIONS_PER_WAVE;
pub(super) const MIN_CANDIDATE_RESERVE: usize = 2 * CANDIDATE_EVALUATIONS_PER_WAVE;
pub(super) const MIN_COMPLETE_RESERVE_WAVES: usize = 2;
