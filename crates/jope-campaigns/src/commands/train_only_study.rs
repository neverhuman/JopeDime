//! Checkpoint-assisted, ten-generation Golden42 regression study.
//!
//! The module is deliberately isolated from the published campaigns.  Phase
//! one never constructs a test path or reads baseline metrics; phase two can
//! only start from independently validated, hash-linked selection receipts.

use crate::data::{read_regression_csv, RegressionData};
use anyhow::{bail, Context, Result};
use feat_math::dsl::ParamValue;
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};
use feat_math::model::fit_with_stage_cache;
use feat_math::stage_cache::{StageCache, StageCacheCtx, DEFAULT_BYTE_CAP};
use feat_math::{fit, GenomeIr};
use jope_dime_regressor::paths;
use rayon::prelude::*;
use rayon::ThreadPool;
use rusqlite::{Connection, OpenFlags};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub(crate) const STUDY_SEED: u64 = 20_260_729;
pub(crate) const GENERATIONS: usize = 10;
pub(crate) const POPULATION: usize = 60;
pub(crate) const DL_QUOTA: usize = 30;
pub(crate) const GLOBAL_QUOTA: usize = 30;
pub(crate) const ELITES: usize = 3;
pub(crate) const PARENT_POOL: usize = 30;
pub(crate) const TOURNAMENT: usize = 3;
pub(crate) const MAX_TRANSFORMS: usize = 12;
pub(crate) const KERNEL_VERSION: &str = "legacy-v2.checkpoint-assisted-study.v2";
pub(crate) const STUDY_DESIGN: &str = "checkpoint_assisted_transfer";
const CONTEXT_FLOATS: usize = 1536 + 512;
const CONTEXT_BYTES: usize = CONTEXT_FLOATS * 4;
const DEFAULT_STUDY_ROOT: &str = "runs/train-only-studies";
jope_paths::once_str!(
    fn default_sotarena_root_str => paths::sotarena_reporter_root()
);
jope_paths::once_str!(
    fn default_checkpoint_pointer_str => paths::flywheel_root().join("training/accepted-by-task.json")
);
jope_paths::once_str!(
    fn default_checkpoint_provenance_str => paths::flywheel_training_root().join("round1/rotation-inputs.v1.json")
);
jope_paths::once_str!(
    fn default_catalog_pointer_str => paths::catalog_pointer()
);
jope_paths::once_str!(
    fn default_observation_db_str => paths::corpus_db()
);
const DEFAULT_PROPOSAL_GENERATOR: &str = "scripts/test_exact_dl_generate.py";
const MIN_RANKABLE: usize = 30;
const EXPECTED_ELIGIBLE: usize = 37;
const FINAL_CV_TOP_K: usize = 10;
const FINAL_CV_FOLDS: usize = 5;

const FROZEN_DATASETS: &[&str] = &[
    "ee56a1015e344c11",
    "019de5d76a718163",
    "5f91a00bd3b66742",
    "f8536dfeef471c2f",
    "7af85a04de0c7018",
    "aebf5d861a8365b4",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "bd49e740a79bcb6c",
    "b5beff4a53fa12c0",
];

const GOLDEN42: &[&str] = &[
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[derive(Clone, Debug)]
pub(crate) struct PrepareArgs {
    pub run_id: String,
    pub study_root: PathBuf,
    pub checkpoint_pointer: PathBuf,
    pub checkpoint_provenance: PathBuf,
    pub catalog_pointer: PathBuf,
    pub observation_db: PathBuf,
    pub dl_proposals: Option<PathBuf>,
    pub proposal_generator: PathBuf,
    pub python: PathBuf,
    pub proposal_device: String,
    pub sotarena_root: PathBuf,
    pub workers: usize,
    pub smoke: bool,
    pub proposal_manifest_only: bool,
}

impl PrepareArgs {
    pub(crate) fn production(run_id: String, study_root: Option<PathBuf>) -> Self {
        Self {
            run_id,
            study_root: study_root.unwrap_or_else(|| PathBuf::from(DEFAULT_STUDY_ROOT)),
            checkpoint_pointer: PathBuf::from(default_checkpoint_pointer_str()),
            checkpoint_provenance: PathBuf::from(default_checkpoint_provenance_str()),
            catalog_pointer: PathBuf::from(default_catalog_pointer_str()),
            observation_db: PathBuf::from(default_observation_db_str()),
            dl_proposals: None,
            proposal_generator: PathBuf::from(DEFAULT_PROPOSAL_GENERATOR),
            python: PathBuf::from("python3"),
            proposal_device: "cuda".to_string(),
            sotarena_root: PathBuf::from(default_sotarena_root_str()),
            workers: 8,
            smoke: false,
            proposal_manifest_only: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointProvenance {
    accepted_pointer_path: PathBuf,
    accepted_pointer_sha256: String,
    checkpoint_path: PathBuf,
    checkpoint_sha256: String,
    model_version: String,
    training_corpus_manifest_path: PathBuf,
    training_corpus_manifest_sha256: String,
    training_corpus_train_path: PathBuf,
    training_corpus_train_sha256: String,
    training_corpus_validation_path: PathBuf,
    training_corpus_validation_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CatalogPointer {
    catalog: PathBuf,
    catalog_sha256: String,
    data: PathBuf,
    data_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Catalog {
    dataset_dimensions: usize,
    target_dimensions: usize,
    record_stride_bytes: usize,
    entries: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct CatalogEntry {
    dataset_hash: String,
    task: String,
    train_csv: PathBuf,
    train_sha256: String,
    source_family: String,
    context_sha256: String,
    catalog_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CorpusExclusions {
    schema: String,
    run_id: String,
    checkpoint_sha256: String,
    training_corpus_train_sha256: String,
    training_corpus_validation_sha256: String,
    dataset_ids: Vec<String>,
    source_families: Vec<String>,
    target_identities: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointOverlapDisclosure {
    schema: String,
    checkpoint_sha256: String,
    exact_golden42_dataset_ids: Vec<String>,
    golden42_source_families: Vec<String>,
    golden42_target_identities: Vec<String>,
    selected_exact_dataset_ids: Vec<String>,
    selected_source_families: Vec<String>,
    selected_target_identities: Vec<String>,
    checkpoint_was_trained_on_golden42_identities: bool,
    zero_shot_valid: bool,
    disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SotarenaReportHashes {
    tabpfn_v3_path: PathBuf,
    tabpfn_v3_sha256: String,
    tabfm_path: PathBuf,
    tabfm_sha256: String,
    tabfm_ensemble_path: PathBuf,
    tabfm_ensemble_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EligibilityRecord {
    dataset_hash: String,
    source_family: Option<String>,
    target_identity: Option<String>,
    rows: Option<usize>,
    features: Option<usize>,
    train_sha256: Option<String>,
    context_sha256: Option<String>,
    checkpoint_exact_overlap: bool,
    checkpoint_source_family_overlap: bool,
    checkpoint_target_identity_overlap: bool,
    baseline_reports: Option<SotarenaReportHashes>,
    selection_key: String,
    eligible: bool,
    reasons: Vec<String>,
    selected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EncodingSlice {
    blob_path: PathBuf,
    blob_sha256: String,
    offset_bytes: usize,
    length_bytes: usize,
    context_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetStudyEntry {
    dataset_hash: String,
    source_family: String,
    target_identity: String,
    train_path: PathBuf,
    train_sha256: String,
    rows: usize,
    features: usize,
    encoding: EncodingSlice,
    split_manifest_path: PathBuf,
    split_manifest_sha256: String,
    generation_zero_path: PathBuf,
    generation_zero_sha256: String,
    transfer_ranking_path: PathBuf,
    transfer_ranking_sha256: String,
    baseline_reports: SotarenaReportHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyStudyManifest {
    schema: String,
    study_design: String,
    zero_shot_valid: bool,
    checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    campaign_scope: String,
    run_id: String,
    seed: u64,
    generations: usize,
    population: usize,
    elites: usize,
    parent_pool: usize,
    tournament_size: usize,
    mutation_probability: f64,
    crossover_probability: f64,
    max_transforms: usize,
    workers: usize,
    kernel_version: String,
    sotarena_root: PathBuf,
    sotarena_registry_path: PathBuf,
    sotarena_registry_sha256: String,
    golden42_ids: Vec<String>,
    eligibility: Vec<EligibilityRecord>,
    selected_dataset_ids: Vec<String>,
    checkpoint: CheckpointProvenance,
    corpus_exclusions_path: PathBuf,
    corpus_exclusions_sha256: String,
    catalog_pointer_path: PathBuf,
    catalog_pointer_sha256: String,
    source_catalog_sha256: String,
    filtered_context_blob_sha256: String,
    observation_db_path: PathBuf,
    observation_db_sha256: String,
    dl_proposals_path: PathBuf,
    dl_proposals_sha256: String,
    proposal_dataset_manifest_path: PathBuf,
    proposal_dataset_manifest_sha256: String,
    datasets: Vec<DatasetStudyEntry>,
    firewall: PhaseOneFirewall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PhaseOneFirewall {
    train_only: bool,
    contains_test_paths: bool,
    contains_test_hashes: bool,
    contains_baseline_scores: bool,
    test_paths_constructed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlySplitManifest {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    seed: u64,
    rows: usize,
    splits: Vec<SplitReceipt>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SplitReceipt {
    generation: usize,
    seed: u64,
    train_indices: Vec<usize>,
    validation_indices: Vec<usize>,
    validation_target_sha256: String,
    split_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Ancestry {
    operation: String,
    parents: Vec<String>,
    roots: BTreeSet<String>,
    retry_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GenomeMember {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    source: String,
    source_evidence: Value,
    ancestry: Ancestry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyGenerationZero {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    encoding_sha256: String,
    checkpoint_sha256: String,
    dl_proposals_sha256: String,
    transfer_ranking_sha256: String,
    members: Vec<GenomeMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProposalRow {
    task: String,
    dataset_hash: String,
    canonical_genome: String,
    semantic_sha256: String,
    model_checkpoint_sha256: String,
    #[serde(default)]
    context_sha256: Option<String>,
    #[serde(default)]
    proposal_score: Option<f64>,
    #[serde(default)]
    selection_score: Option<f64>,
    #[serde(default)]
    reward_lcb: Option<f64>,
    #[serde(default)]
    instability_probability: Option<f64>,
    #[serde(default)]
    slot: usize,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GlobalRankedGenome {
    rank: usize,
    canonical_genome: String,
    semantic_sha256: String,
    mean_percentile: f64,
    worst_percentile: f64,
    strict_win_count: usize,
    dataset_count: usize,
    source_dataset_ids: Vec<String>,
    source_families: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GlobalRankingReceipt {
    schema: String,
    run_id: String,
    target_dataset_id: String,
    target_source_family: String,
    observation_db_sha256: String,
    source_catalog_sha256: String,
    excluded_dataset_ids: Vec<String>,
    excluded_source_family: String,
    direct_dl_sources_excluded: bool,
    ranking_policy: String,
    genomes: Vec<GlobalRankedGenome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyFitnessObservation {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    encoding_sha256: String,
    checkpoint_sha256: String,
    canonical_genome: String,
    semantic_sha256: String,
    split_generation: usize,
    split_sha256: String,
    kernel_version: String,
    cache_key: String,
    r2: Option<f64>,
    rmse: Option<f64>,
    mae: Option<f64>,
    fit_seconds: f64,
    predict_seconds: f64,
    failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ObservationLink {
    split_generation: usize,
    split_sha256: String,
    receipt_path: PathBuf,
    receipt_sha256: String,
    cache_key: String,
    cache_hit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScoredMember {
    member: GenomeMember,
    observations: Vec<ObservationLink>,
    split_r2: Vec<f64>,
    robust_fitness: Option<f64>,
    worst_split_r2: Option<f64>,
    failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyGenerationCheckpoint {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    encoding_sha256: String,
    checkpoint_sha256: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    split_manifest_sha256: String,
    population: Vec<ScoredMember>,
    ranked_semantic_sha256: Vec<String>,
    champion_semantic_sha256: String,
    rankable_members: usize,
    cache_hits: usize,
    cache_misses: usize,
    invalid_members: usize,
    next_population: Vec<GenomeMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlySelectionComplete {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    encoding_sha256: String,
    checkpoint_sha256: String,
    split_manifest_sha256: String,
    generation_checkpoint_sha256: Vec<String>,
    generation_champions: Vec<String>,
    final_champion_semantic_sha256: String,
    final_champion_genome: String,
    cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChampionFixedGrid {
    generation: usize,
    semantic_sha256: String,
    canonical_genome: String,
    split_r2: Vec<f64>,
    robust_fitness: f64,
    worst_split_r2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyChampionValidation {
    schema: String,
    run_id: String,
    dataset_hash: String,
    selection_complete_sha256: String,
    split_manifest_sha256: String,
    champions: Vec<ChampionFixedGrid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyCohortSelectionComplete {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    dataset_ids: Vec<String>,
    selection_complete_sha256: BTreeMap<String, String>,
    champion_validation_sha256: BTreeMap<String, String>,
    cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyValidationReceipt {
    schema: String,
    study_design: String,
    zero_shot_valid: bool,
    checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    run_id: String,
    manifest_sha256: String,
    cohort_selection_complete_sha256: String,
    dataset_selection_sha256: BTreeMap<String, String>,
    dataset_validation_sha256: BTreeMap<String, String>,
    benchmark_valid: bool,
    checkpoint_disclosure_audit_passed: bool,
    transfer_exclusion_audit_passed: bool,
    firewall_audit_passed: bool,
    population_audit_passed: bool,
    paired_history_audit_passed: bool,
    cache_audit_passed: bool,
    validated_dataset_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyDatasetValidationReceipt {
    schema: String,
    study_design: String,
    run_id: String,
    dataset_hash: String,
    manifest_sha256: String,
    cohort_selection_complete_sha256: String,
    selection_complete_sha256: String,
    champion_validation_sha256: String,
    kernel_version: String,
    metric_replay_passed: bool,
    population_audit_passed: bool,
    paired_history_audit_passed: bool,
    cache_audit_passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BaselineEntry {
    dataset_hash: String,
    prior_aspfm_report_path: PathBuf,
    prior_aspfm_report_sha256: String,
    prior_aspfm_rmse: f64,
    tabpfn_v3_report_path: PathBuf,
    tabpfn_v3_report_sha256: String,
    tabpfn_v3_rmse: f64,
    tabfm_report_path: PathBuf,
    tabfm_report_sha256: String,
    tabfm_rmse: Option<f64>,
    tabfm_ensemble_report_path: PathBuf,
    tabfm_ensemble_report_sha256: String,
    tabfm_ensemble_rmse: f64,
    stronger_tabfm_algorithm: String,
    stronger_tabfm_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BaselineManifest {
    schema: String,
    run_id: String,
    entries: Vec<BaselineEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GenerationTestScore {
    generation: usize,
    semantic_sha256: String,
    canonical_genome: String,
    rmse: f64,
    r2: f64,
    mae: f64,
    fit_seconds: f64,
    predict_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FinalCvCandidate {
    original_rank: usize,
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
    fold_r2: Vec<f64>,
    fold_rmse: Vec<f64>,
    robust_fitness: Option<f64>,
    worst_fold_r2: Option<f64>,
    failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyFinalCvRerank {
    schema: String,
    run_id: String,
    dataset_hash: String,
    validation_receipt_sha256: String,
    selection_complete_sha256: String,
    final_generation_checkpoint_sha256: String,
    train_sha256: String,
    kernel_version: String,
    top_k: usize,
    folds: usize,
    split_manifest: TrainOnlySplitManifest,
    candidates: Vec<FinalCvCandidate>,
    selected_semantic_sha256: String,
    selected_canonical_genome: String,
    selected_robust_fitness: f64,
    selected_worst_fold_r2: f64,
    selected_original_rank: usize,
    selection_frozen_before_test_access: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyPosthocScore {
    schema: String,
    run_id: String,
    dataset_hash: String,
    validation_receipt_sha256: String,
    selection_complete_sha256: String,
    generation_checkpoint_sha256: Vec<String>,
    baseline_manifest_sha256: String,
    train_sha256: String,
    encoding_sha256: String,
    checkpoint_sha256: String,
    test_sha256: String,
    kernel_version: String,
    scores: Vec<GenerationTestScore>,
    final_cv_rerank_sha256: String,
    final_cv_rerank_score: GenerationTestScore,
    prior_aspfm_rmse: f64,
    tabpfn_v3_rmse: f64,
    tabfm_rmse: Option<f64>,
    tabfm_ensemble_rmse: f64,
    stronger_tabfm_algorithm: String,
    stronger_tabfm_rmse: f64,
    beats_tabpfn_v3: bool,
    beats_stronger_tabfm: bool,
    beats_both: bool,
    beats_prior_aspfm: bool,
    cv_rerank_beats_tabpfn_v3: bool,
    cv_rerank_beats_stronger_tabfm: bool,
    cv_rerank_beats_both: bool,
    cv_rerank_beats_prior_aspfm: bool,
    test_trajectory_post_selection: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyPosthocIndex {
    schema: String,
    run_id: String,
    validation_receipt_sha256: String,
    baseline_manifest_sha256: String,
    dataset_ids: Vec<String>,
    posthoc_score_sha256: BTreeMap<String, String>,
    cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetReport {
    dataset_hash: String,
    generation_zero_validation_fitness: f64,
    generation_nine_validation_fitness: f64,
    validation_fitness_change: f64,
    generation_zero_rmse: f64,
    generation_nine_rmse: f64,
    relative_rmse_improvement: f64,
    prior_aspfm_rmse: f64,
    tabpfn_v3_rmse: f64,
    tabfm_rmse: Option<f64>,
    tabfm_ensemble_rmse: f64,
    stronger_tabfm_algorithm: String,
    stronger_tabfm_rmse: f64,
    best_baseline_algorithm: String,
    best_baseline_rmse: f64,
    generation_nine_relative_gap_to_best_baseline: f64,
    cv_rerank_relative_gap_to_best_baseline: f64,
    beats_tabpfn_v3: bool,
    beats_stronger_tabfm: bool,
    beats_both: bool,
    beats_prior_aspfm: bool,
    cv_rerank_semantic_sha256: String,
    cv_rerank_rmse: f64,
    cv_rerank_beats_tabpfn_v3: bool,
    cv_rerank_beats_stronger_tabfm: bool,
    cv_rerank_beats_both: bool,
    cv_rerank_beats_prior_aspfm: bool,
    final_ancestry: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EloRating {
    system: String,
    rating: f64,
    wins: usize,
    losses: usize,
    ties: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyStudyReport {
    schema: String,
    study_design: String,
    zero_shot_valid: bool,
    checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    run_id: String,
    validation_receipt_sha256: String,
    baseline_manifest_sha256: String,
    posthoc_score_sha256: BTreeMap<String, String>,
    status: String,
    wins_vs_tabpfn_v3: usize,
    wins_vs_stronger_tabfm: usize,
    wins_vs_both: usize,
    wins_vs_prior_aspfm: usize,
    cv_rerank_wins_vs_tabpfn_v3: usize,
    cv_rerank_wins_vs_stronger_tabfm: usize,
    cv_rerank_wins_vs_both: usize,
    cv_rerank_wins_vs_prior_aspfm: usize,
    median_relative_rmse_improvement: f64,
    bootstrap_interval_95: [f64; 2],
    paired_sign_test_p_value: f64,
    elo_methodology: String,
    elo_ratings: Vec<EloRating>,
    test_trajectories_post_selection: bool,
    datasets: Vec<DatasetReport>,
}

mod eligibility;
mod evaluation;
mod fitness;
mod prepare;
mod provenance;
mod report;
mod run;
mod scoring;
mod splits;
mod validation;
pub(super) use eligibility::{
    build_eligibility, build_transfer_rankings, copy_context_slices, generation_zero,
    load_dl_proposals, proposal_score, EligibleDatasets,
};
pub(super) use evaluation::{
    evaluate_member_split, offspring_seed, refill_population, score_population, slice_data,
    tournament,
};
pub(crate) use fitness::paired_robust_fitness;
pub(super) use fitness::{
    canonical_json_bytes, canonicalize_json, expression_column_indices, genome_compatible,
    genome_complexity, immutable_write, metrics, rank_scored, read_receipt, receipt_digest,
    registry_source_family, selection_key, semantic_identity, sha256_bytes, sha256_file,
    sotarena_metadata_path, study_dir, target_identity, validate_sha256, write_receipt,
};
pub(crate) use prepare::prepare;
pub(super) use prepare::{
    cache_key, validate_execution_inputs, validate_manifest_contract, validate_prepared_inputs,
};
pub(super) use provenance::{
    absolute_path, checkpoint_exclusions, completed_baseline_report, corpus_dataset_ids,
    extract_json_string_field, load_catalog, resolve_checkpoint_provenance, resolve_pointer_path,
    sha256_f64, sotarena_report_hashes, validate_train_data,
};
pub(crate) use report::report;
pub(super) use report::{
    bootstrap_interval, cohort_elo, evolution_svg, median, paired_sign_test, panel_points,
    panel_points_scaled, report_dataset,
};
pub(crate) use run::run;
pub(super) use run::{
    champion_validation_path, checkpoint_path, dataset_validation_path, run_dataset, selection_path,
};
pub(crate) use scoring::score;
pub(super) use scoring::{
    baseline_metric, load_or_create_baselines, load_or_create_final_cv_reranks,
    optional_baseline_metric, prior_aspfm_metric, score_frozen_champion, stronger_tabfm,
};
pub(super) use splits::{build_final_cv_splits, build_splits, rank_final_cv};
pub(crate) use validation::validate;
pub(super) use validation::{
    read_dataset_validation_shard, require_validated_score_gate, validate_dataset_run,
    validate_dataset_shard, validate_generation_zero, validate_observation_link,
};

#[cfg(test)]
mod tests;
