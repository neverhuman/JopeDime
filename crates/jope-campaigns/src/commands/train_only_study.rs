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

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sorted = Map::new();
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&canonicalize_json(value.clone()))?)
}

fn receipt_digest<T: Serialize>(receipt: &T) -> Result<String> {
    let value = serde_json::to_value(receipt)?;
    Ok(sha256_bytes(&canonical_json_bytes(&value)?))
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        let present = fs::read(path)?;
        if present == bytes {
            return Ok(());
        }
        bail!(
            "immutable artifact already exists with different bytes: {}",
            path.display()
        );
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    value
        .as_object_mut()
        .context("receipt must serialize as an object")?
        .insert("receipt_sha256".to_string(), Value::String(digest.clone()));
    let mut bytes = canonical_json_bytes(&value)?;
    bytes.push(b'\n');
    immutable_write(path, &bytes)?;
    Ok(digest)
}

fn read_receipt<T: DeserializeOwned>(path: &Path) -> Result<(T, String)> {
    let mut value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read receipt {}", path.display()))?,
    )?;
    let claimed = value
        .as_object_mut()
        .and_then(|object| object.remove("receipt_sha256"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .context("receipt lacks receipt_sha256")?;
    let actual = sha256_bytes(&canonical_json_bytes(&value)?);
    if claimed != actual {
        bail!("receipt digest mismatch: {}", path.display());
    }
    Ok((serde_json::from_value(value)?, actual))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} is not a SHA-256");
    }
    Ok(())
}

fn study_dir(study_root: &Path, run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || run_id == "."
        || run_id == ".."
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("run id must contain only ASCII letters, digits, '-' or '_'");
    }
    Ok(study_root.join(run_id))
}

fn semantic_identity(canonical_genome: &str) -> String {
    sha256_bytes(canonical_genome.as_bytes())
}

fn genome_complexity(genome: &GenomeIr) -> usize {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.len(),
        _ => 0,
    }
}

fn genome_compatible(canonical_genome: &str, features: usize) -> bool {
    let Ok(genome) = GenomeIr::parse(canonical_genome) else {
        return false;
    };
    let GenomeIr::Pipeline(pipeline) = &genome else {
        return false;
    };
    if genome.canonical() != canonical_genome
        || pipeline.stages.is_empty()
        || pipeline.stages.len().saturating_sub(1) > MAX_TRANSFORMS
    {
        return false;
    }
    let head = &pipeline.stages.last().expect("nonempty").op;
    if matches!(head.as_str(), "logit_ridge" | "logistic") {
        return false;
    }
    // Expressions can address raw columns as cN, colN, or col(N).  Reject a
    // reference outside the dataset's raw schema.  Row summaries and `x` are
    // feature-count independent.
    for stage in &pipeline.stages {
        let Some(ParamValue::Str(expression)) = stage.params.get("expr") else {
            continue;
        };
        if expression_column_indices(expression)
            .into_iter()
            .any(|index| index >= features)
        {
            return false;
        }
    }
    true
}

fn expression_column_indices(expression: &str) -> Vec<usize> {
    let bytes = expression.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let start = if bytes[index] == b'c' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 1)
        } else if bytes[index..].starts_with(b"col(")
            && bytes.get(index + 4).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 4)
        } else if bytes[index..].starts_with(b"col")
            && bytes.get(index + 3).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 3)
        } else {
            None
        };
        if let Some(start) = start {
            let mut end = start;
            while bytes.get(end).is_some_and(u8::is_ascii_digit) {
                end += 1;
            }
            if let Ok(value) = expression[start..end].parse() {
                result.push(value);
            }
            index = end;
        } else {
            index += 1;
        }
    }
    result
}

fn sotarena_metadata_path(sotarena_root: &Path, dataset_hash: &str) -> PathBuf {
    let legacy = sotarena_root.join(dataset_hash).join("meta.json");
    if legacy.is_file() {
        legacy
    } else {
        sotarena_root
            .join("splits/tabular-regression-v1")
            .join(dataset_hash)
            .join("metadata.json")
    }
}

fn target_identity(sotarena_root: &Path, dataset_hash: &str) -> Result<String> {
    let meta_path = sotarena_metadata_path(sotarena_root, dataset_hash);
    let value: Value = serde_json::from_slice(
        &fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?,
    )?;
    let field = |key: &str| {
        value
            .get(key)
            .or_else(|| value.pointer(&format!("/acquisition/{key}")))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let material = json!({
        "source_platform": field("source_platform"),
        "source_dataset_id": field("source_dataset_id"),
        "source_dataset_name": field("source_dataset_name"),
        "source_target_column_original": field("source_target_column_original"),
        "target_column": field("target_column"),
        "target_names": value.get("target_names").cloned().unwrap_or(Value::Null),
    });
    let canonical = canonical_json_bytes(&material)?;
    if canonical == b"{\"source_dataset_id\":\"\",\"source_dataset_name\":\"\",\"source_platform\":\"\",\"source_target_column_original\":\"\",\"target_column\":\"\",\"target_names\":null}" {
        Ok(format!("dataset:{dataset_hash}"))
    } else {
        Ok(format!("target:{}", sha256_bytes(&canonical)))
    }
}

fn registry_source_family(sotarena_root: &Path, dataset_hash: &str) -> Result<Option<String>> {
    let meta_path = sotarena_metadata_path(sotarena_root, dataset_hash);
    if !meta_path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&meta_path)?)?;
    Ok(value
        .get("source_family")
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .pointer("/acquisition/source_dataset_name")
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string))
}

fn selection_key(dataset_hash: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"checkpoint-assisted-transfer-v2\0");
    hash.update(STUDY_SEED.to_string().as_bytes());
    hash.update(b"\0");
    hash.update(dataset_hash.as_bytes());
    hex::encode(hash.finalize())
}

fn metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
    if truth.is_empty()
        || truth.len() != predictions.len()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("metrics require equal, non-empty, finite inputs");
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let mut squared_error = 0.0;
    let mut absolute_error = 0.0;
    let mut total_sum_squares = 0.0;
    for (&actual, &predicted) in truth.iter().zip(predictions) {
        squared_error += (actual - predicted).powi(2);
        absolute_error += (actual - predicted).abs();
        total_sum_squares += (actual - mean).powi(2);
    }
    let rmse = (squared_error / n).sqrt();
    let r2 = if total_sum_squares > 1e-24 {
        1.0 - squared_error / total_sum_squares
    } else {
        bail!("metrics target variance is zero");
    };
    let mae = absolute_error / n;
    if [rmse, r2, mae].iter().any(|value| !value.is_finite()) {
        bail!("metrics are non-finite");
    }
    Ok((rmse, r2, mae))
}

pub(crate) fn paired_robust_fitness(split_r2: &[f64]) -> Result<f64> {
    if split_r2.is_empty() || split_r2.iter().any(|value| !value.is_finite()) {
        bail!("paired robust fitness needs finite split scores");
    }
    let clipped = split_r2
        .iter()
        .map(|value| value.max(-1.0))
        .collect::<Vec<_>>();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let population_sd = (clipped
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / clipped.len() as f64)
        .sqrt();
    Ok(mean - 0.25 * population_sd.min(1.0))
}

fn rank_scored(left: &ScoredMember, right: &ScoredMember) -> Ordering {
    match (left.robust_fitness, right.robust_fitness) {
        (Some(left_fitness), Some(right_fitness)) => right_fitness
            .total_cmp(&left_fitness)
            .then_with(|| {
                right
                    .worst_split_r2
                    .unwrap()
                    .total_cmp(&left.worst_split_r2.unwrap())
            })
            .then_with(|| left.member.complexity.cmp(&right.member.complexity))
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .complexity
            .cmp(&right.member.complexity)
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
    }
}

fn build_splits(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
    generations: usize,
) -> Result<TrainOnlySplitManifest> {
    if targets.len() < 100 || targets.iter().any(|value| !value.is_finite()) {
        bail!("split construction needs at least 100 finite targets");
    }
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let sorted_len = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        let decile = (rank * 10 / sorted_len).min(9);
        deciles[decile].push(row);
    }
    let validation_total = targets.len().div_ceil(5);
    let mut receipts = Vec::with_capacity(generations);
    let mut seen_hashes = HashSet::new();
    for generation in 0..generations {
        let split_seed = {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-split-seed-v2\0");
            hash.update(STUDY_SEED.to_le_bytes());
            hash.update(dataset_hash.as_bytes());
            hash.update((generation as u64).to_le_bytes());
            u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
        };
        let mut ordered = Vec::with_capacity(10);
        let mut base_total = 0usize;
        for (decile, rows) in deciles.iter().enumerate() {
            let mut rows = rows.clone();
            rows.sort_by_key(|row| {
                let mut hash = Sha256::new();
                hash.update(b"checkpoint-assisted-split-order-v2\0");
                hash.update(split_seed.to_le_bytes());
                hash.update((*row as u64).to_le_bytes());
                hex::encode(hash.finalize())
            });
            let base = rows.len() / 5;
            let remainder = rows.len() % 5;
            base_total += base;
            ordered.push((decile, rows, base, remainder));
        }
        let mut extra = validation_total - base_total;
        let mut remainder_order = (0..10).collect::<Vec<_>>();
        remainder_order.sort_by(|left, right| {
            ordered[*right]
                .3
                .cmp(&ordered[*left].3)
                .then_with(|| left.cmp(right))
        });
        let mut quotas = ordered.iter().map(|entry| entry.2).collect::<Vec<_>>();
        for decile in remainder_order {
            if extra == 0 {
                break;
            }
            if quotas[decile] < ordered[decile].1.len() {
                quotas[decile] += 1;
                extra -= 1;
            }
        }
        if extra != 0 {
            bail!("could not allocate exact 20 percent validation split");
        }
        let mut validation_indices = Vec::with_capacity(validation_total);
        for (decile, rows, _, _) in &ordered {
            validation_indices.extend_from_slice(&rows[..quotas[*decile]]);
        }
        validation_indices.sort_unstable();
        let validation_set = validation_indices.iter().copied().collect::<HashSet<_>>();
        let train_indices = (0..targets.len())
            .filter(|row| !validation_set.contains(row))
            .collect::<Vec<_>>();
        if validation_indices.len() != validation_total
            || train_indices.len() + validation_indices.len() != targets.len()
            || train_indices.iter().any(|row| validation_set.contains(row))
        {
            bail!("split allocation overlap or size error");
        }
        let validation_targets = validation_indices
            .iter()
            .map(|row| targets[*row])
            .collect::<Vec<_>>();
        let mean = validation_targets.iter().sum::<f64>() / validation_targets.len() as f64;
        if validation_targets
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            bail!("generation {generation} validation target variance is zero");
        }
        let validation_target_sha256 = sha256_f64(&validation_targets);
        let split_value = json!({
            "dataset_hash": dataset_hash,
            "generation": generation,
            "seed": split_seed,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "validation_target_sha256": validation_target_sha256,
        });
        let split_sha256 = sha256_bytes(&canonical_json_bytes(&split_value)?);
        if !seen_hashes.insert(split_sha256.clone()) {
            bail!("validation split hashes are not distinct");
        }
        receipts.push(SplitReceipt {
            generation,
            seed: split_seed,
            train_indices: split_value["train_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect(),
            validation_indices: split_value["validation_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect(),
            validation_target_sha256,
            split_sha256,
        });
    }
    Ok(TrainOnlySplitManifest {
        schema: "TrainOnlySplitManifest.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        seed: STUDY_SEED,
        rows: targets.len(),
        splits: receipts,
    })
}

fn build_final_cv_splits(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
) -> Result<TrainOnlySplitManifest> {
    if targets.len() < 100 || targets.iter().any(|value| !value.is_finite()) {
        bail!("final CV construction needs at least 100 finite targets");
    }
    let cv_seed = {
        let mut hash = Sha256::new();
        hash.update(b"checkpoint-assisted-final-top10-cv-seed-v2\0");
        hash.update(STUDY_SEED.to_le_bytes());
        hash.update(dataset_hash.as_bytes());
        u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
    };
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let sorted_len = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        deciles[(rank * 10 / sorted_len).min(9)].push(row);
    }
    let mut validation_by_fold = vec![Vec::new(); FINAL_CV_FOLDS];
    for (decile, rows) in deciles.iter_mut().enumerate() {
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-final-top10-cv-order-v2\0");
            hash.update(cv_seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hex::encode(hash.finalize())
        });
        for (position, row) in rows.iter().enumerate() {
            validation_by_fold[(position + decile) % FINAL_CV_FOLDS].push(*row);
        }
    }
    let mut receipts = Vec::with_capacity(FINAL_CV_FOLDS);
    let mut seen_rows = HashSet::new();
    let mut seen_hashes = HashSet::new();
    for (fold, validation_indices) in validation_by_fold.iter_mut().enumerate() {
        validation_indices.sort_unstable();
        if validation_indices.is_empty()
            || validation_indices.iter().any(|row| !seen_rows.insert(*row))
        {
            bail!("final CV validation folds overlap or are empty");
        }
        let validation_set = validation_indices.iter().copied().collect::<HashSet<_>>();
        let train_indices = (0..targets.len())
            .filter(|row| !validation_set.contains(row))
            .collect::<Vec<_>>();
        let validation_targets = validation_indices
            .iter()
            .map(|row| targets[*row])
            .collect::<Vec<_>>();
        let mean = validation_targets.iter().sum::<f64>() / validation_targets.len() as f64;
        if validation_targets
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            bail!("final CV fold {fold} target variance is zero");
        }
        let split_seed = {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-final-top10-cv-fold-v2\0");
            hash.update(cv_seed.to_le_bytes());
            hash.update((fold as u64).to_le_bytes());
            u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
        };
        let validation_target_sha256 = sha256_f64(&validation_targets);
        let split_value = json!({
            "purpose": "final_top10_cv_rerank",
            "dataset_hash": dataset_hash,
            "fold": fold,
            "seed": split_seed,
            "train_indices": &train_indices,
            "validation_indices": &validation_indices,
            "validation_target_sha256": validation_target_sha256,
        });
        let split_sha256 = sha256_bytes(&canonical_json_bytes(&split_value)?);
        if !seen_hashes.insert(split_sha256.clone()) {
            bail!("final CV split hashes are not distinct");
        }
        receipts.push(SplitReceipt {
            generation: fold,
            seed: split_seed,
            train_indices,
            validation_indices: validation_indices.clone(),
            validation_target_sha256,
            split_sha256,
        });
    }
    if seen_rows.len() != targets.len() {
        bail!("final CV folds do not cover every training row exactly once");
    }
    Ok(TrainOnlySplitManifest {
        schema: "TrainOnlyFinalCvSplitManifest.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        seed: cv_seed,
        rows: targets.len(),
        splits: receipts,
    })
}

fn rank_final_cv(left: &FinalCvCandidate, right: &FinalCvCandidate) -> Ordering {
    match (left.robust_fitness, right.robust_fitness) {
        (Some(left_fitness), Some(right_fitness)) => right_fitness
            .total_cmp(&left_fitness)
            .then_with(|| {
                right
                    .worst_fold_r2
                    .unwrap()
                    .total_cmp(&left.worst_fold_r2.unwrap())
            })
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .complexity
            .cmp(&right.complexity)
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256)),
    }
}

fn sha256_f64(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hex::encode(hash.finalize())
}

fn resolve_pointer_path(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_path_buf()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn resolve_checkpoint_provenance(args: &PrepareArgs) -> Result<CheckpointProvenance> {
    let pointer_bytes = fs::read(&args.checkpoint_pointer).with_context(|| {
        format!(
            "read accepted regression checkpoint pointer {}",
            args.checkpoint_pointer.display()
        )
    })?;
    let pointer: Value = serde_json::from_slice(&pointer_bytes)?;
    if pointer.get("schema").and_then(Value::as_str) != Some("JopeSelfplayAcceptedByTask.v1") {
        bail!("accepted checkpoint pointer has unsupported schema");
    }
    let regression = pointer
        .pointer("/tasks/regression")
        .context("accepted checkpoint pointer lacks regression task")?;
    let checkpoint_path = PathBuf::from(
        regression
            .get("checkpoint")
            .and_then(Value::as_str)
            .context("regression checkpoint path is absent")?,
    );
    let checkpoint_sha256 = regression
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .context("regression checkpoint hash is absent")?
        .to_string();
    validate_sha256(&checkpoint_sha256, "accepted regression checkpoint")?;
    if sha256_file(&checkpoint_path)? != checkpoint_sha256 {
        bail!("accepted regression checkpoint pointer content drift");
    }

    let provenance_bytes = fs::read(&args.checkpoint_provenance).with_context(|| {
        format!(
            "read checkpoint provenance {}",
            args.checkpoint_provenance.display()
        )
    })?;
    let provenance: Value = serde_json::from_slice(&provenance_bytes)?;
    if provenance.get("checkpoint_sha").and_then(Value::as_str) != Some(&checkpoint_sha256) {
        bail!("checkpoint provenance does not describe the accepted regression checkpoint");
    }
    let manifest_path = PathBuf::from(
        provenance
            .get("corpus_manifest")
            .and_then(Value::as_str)
            .context("checkpoint provenance lacks corpus manifest")?,
    );
    let corpus_manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let outputs = corpus_manifest
        .get("outputs")
        .and_then(Value::as_object)
        .context("checkpoint corpus manifest lacks outputs")?;
    let manifest_dir = manifest_path
        .parent()
        .context("corpus manifest has no parent")?;
    let train_path = manifest_dir.join("corpus.train.jsonl");
    let validation_path = manifest_dir.join("corpus.val.jsonl");
    let train_sha256 = outputs
        .get("corpus.train.jsonl")
        .and_then(Value::as_str)
        .context("checkpoint corpus manifest lacks train hash")?
        .to_string();
    let validation_sha256 = outputs
        .get("corpus.val.jsonl")
        .and_then(Value::as_str)
        .context("checkpoint corpus manifest lacks validation hash")?
        .to_string();
    if sha256_file(&train_path)? != train_sha256
        || sha256_file(&validation_path)? != validation_sha256
    {
        bail!("checkpoint training-corpus content drift");
    }
    let model_version = provenance
        .pointer("/checkpoint_config/model_version")
        .and_then(Value::as_str)
        .context("checkpoint provenance lacks model version")?
        .to_string();
    Ok(CheckpointProvenance {
        accepted_pointer_path: absolute_path(&args.checkpoint_pointer)?,
        accepted_pointer_sha256: sha256_bytes(&pointer_bytes),
        checkpoint_path,
        checkpoint_sha256,
        model_version,
        training_corpus_manifest_path: manifest_path.clone(),
        training_corpus_manifest_sha256: sha256_file(&manifest_path)?,
        training_corpus_train_path: train_path,
        training_corpus_train_sha256: train_sha256,
        training_corpus_validation_path: validation_path,
        training_corpus_validation_sha256: validation_sha256,
    })
}

fn extract_json_string_field(line: &[u8], field: &[u8]) -> Option<String> {
    let mut start = 0usize;
    while start + field.len() <= line.len() {
        let relative = line[start..]
            .windows(field.len())
            .position(|window| window == field)?;
        let mut index = start + relative + field.len();
        while line.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if line.get(index) != Some(&b':') {
            start = index;
            continue;
        }
        index += 1;
        while line.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if line.get(index) != Some(&b'"') {
            start = index;
            continue;
        }
        index += 1;
        let end = line[index..].iter().position(|byte| *byte == b'"')? + index;
        return std::str::from_utf8(&line[index..end])
            .ok()
            .map(str::to_string);
    }
    None
}

fn corpus_dataset_ids(paths: &[&Path]) -> Result<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    for path in paths {
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            let dataset = extract_json_string_field(&line, b"\"dataset_id\"")
                .context("checkpoint corpus row lacks dataset_id")?;
            result.insert(dataset);
        }
    }
    Ok(result)
}

fn load_catalog(pointer_path: &Path) -> Result<(CatalogPointer, Catalog)> {
    let pointer_bytes = fs::read(pointer_path)?;
    let pointer: CatalogPointer = serde_json::from_slice(&pointer_bytes)?;
    validate_sha256(&pointer.catalog_sha256, "catalog")?;
    validate_sha256(&pointer.data_sha256, "catalog data")?;
    let catalog_path = resolve_pointer_path(pointer_path, &pointer.catalog);
    let data_path = resolve_pointer_path(pointer_path, &pointer.data);
    if sha256_file(&catalog_path)? != pointer.catalog_sha256
        || sha256_file(&data_path)? != pointer.data_sha256
    {
        bail!("context catalog pointer content drift");
    }
    let catalog: Catalog = serde_json::from_slice(&fs::read(catalog_path)?)?;
    if catalog.dataset_dimensions != 1536
        || catalog.target_dimensions != 512
        || catalog.record_stride_bytes != CONTEXT_BYTES
    {
        bail!("context catalog is not the required 1536+512 encoding");
    }
    Ok((
        CatalogPointer {
            catalog: pointer.catalog,
            catalog_sha256: pointer.catalog_sha256,
            data: data_path,
            data_sha256: pointer.data_sha256,
        },
        catalog,
    ))
}

fn checkpoint_exclusions(
    run_id: &str,
    registry: &Path,
    checkpoint: &CheckpointProvenance,
    catalog: &Catalog,
) -> Result<CorpusExclusions> {
    let ids = corpus_dataset_ids(&[
        &checkpoint.training_corpus_train_path,
        &checkpoint.training_corpus_validation_path,
    ])?;
    let families_by_id = catalog
        .entries
        .iter()
        .map(|entry| (entry.dataset_hash.as_str(), entry.source_family.as_str()))
        .collect::<HashMap<_, _>>();
    let mut source_families = BTreeSet::new();
    let mut target_identities = BTreeSet::new();
    for dataset in &ids {
        if let Some(family) = families_by_id.get(dataset.as_str()) {
            source_families.insert((*family).to_string());
        } else if let Some(family) = registry_source_family(registry, dataset)? {
            source_families.insert(family);
        }
        let meta_path = sotarena_metadata_path(registry, dataset);
        if meta_path.is_file() {
            target_identities.insert(target_identity(registry, dataset)?);
        } else {
            target_identities.insert(format!("dataset:{dataset}"));
        }
    }
    Ok(CorpusExclusions {
        schema: "TrainOnlyCheckpointCorpusDisclosure.v2".to_string(),
        run_id: run_id.to_string(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        training_corpus_train_sha256: checkpoint.training_corpus_train_sha256.clone(),
        training_corpus_validation_sha256: checkpoint.training_corpus_validation_sha256.clone(),
        dataset_ids: ids.into_iter().collect(),
        source_families: source_families.into_iter().collect(),
        target_identities: target_identities.into_iter().collect(),
    })
}

fn validate_train_data(data: &RegressionData) -> Vec<String> {
    let mut reasons = Vec::new();
    if data.y.len() < 100 {
        reasons.push("fewer_than_100_training_rows".to_string());
    }
    if data.x.len() != data.y.len()
        || data
            .x
            .iter()
            .any(|row| row.len() != data.features || row.iter().any(|value| !value.is_finite()))
        || data.y.iter().any(|value| !value.is_finite())
    {
        reasons.push("nonfinite_or_ragged_numeric_training_data".to_string());
    }
    if !data.y.is_empty() {
        let mean = data.y.iter().sum::<f64>() / data.y.len() as f64;
        if data
            .y
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            reasons.push("zero_target_variance".to_string());
        }
    }
    reasons
}

fn completed_baseline_report(path: &Path, dataset_hash: &str, algorithm: &str) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("read baseline report {}", path.display()))?;
    let report: Value = serde_json::from_slice(&bytes)?;
    let status = report.get("status").and_then(Value::as_str);
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
        || !matches!(status, Some("completed" | "failed"))
    {
        bail!(
            "baseline report identity/status mismatch for {dataset_hash}: {}",
            path.display()
        );
    }
    Ok(sha256_bytes(&bytes))
}

fn sotarena_report_hashes(
    sotarena_root: &Path,
    dataset_hash: &str,
) -> Result<SotarenaReportHashes> {
    let root = sotarena_root
        .join("results/external-references/golden42")
        .join(dataset_hash);
    let tabpfn_v3_path = root.join("tabpfn_v3_regressor.json");
    let tabfm_path = root.join("tabfm_regressor.json");
    let tabfm_ensemble_path = root.join("tabfm_ensemble_regressor.json");
    Ok(SotarenaReportHashes {
        tabpfn_v3_sha256: completed_baseline_report(
            &tabpfn_v3_path,
            dataset_hash,
            "tabpfn_v3_regressor",
        )?,
        tabpfn_v3_path: absolute_path(&tabpfn_v3_path)?,
        tabfm_sha256: completed_baseline_report(&tabfm_path, dataset_hash, "tabfm_regressor")?,
        tabfm_path: absolute_path(&tabfm_path)?,
        tabfm_ensemble_sha256: completed_baseline_report(
            &tabfm_ensemble_path,
            dataset_hash,
            "tabfm_ensemble_regressor",
        )?,
        tabfm_ensemble_path: absolute_path(&tabfm_ensemble_path)?,
    })
}

fn build_eligibility(
    run_id: &str,
    sotarena_root: &Path,
    catalog: &Catalog,
    exclusions: &CorpusExclusions,
    generations: usize,
    selected_ids: &[&str],
    expected_eligible: Option<usize>,
) -> Result<(
    Vec<EligibilityRecord>,
    Vec<(CatalogEntry, RegressionData, String, SotarenaReportHashes)>,
)> {
    let entries = catalog
        .entries
        .iter()
        .filter(|entry| entry.task == "regression")
        .map(|entry| (entry.dataset_hash.as_str(), entry))
        .collect::<HashMap<_, _>>();
    let overlapping_ids = exclusions
        .dataset_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let overlapping_families = exclusions
        .source_families
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let overlapping_targets = exclusions
        .target_identities
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut records = Vec::with_capacity(GOLDEN42.len());
    let mut valid = Vec::new();
    for dataset in GOLDEN42 {
        let mut reasons = Vec::new();
        let mut source_family = None;
        let mut identity = None;
        let mut rows = None;
        let mut features = None;
        let mut train_sha256 = None;
        let mut context_sha256 = None;
        let mut checkpoint_exact_overlap = false;
        let mut checkpoint_source_family_overlap = false;
        let mut checkpoint_target_identity_overlap = false;
        let mut baseline_reports = None;
        if let Some(entry) = entries.get(dataset) {
            source_family = Some(entry.source_family.clone());
            context_sha256 = Some(entry.context_sha256.clone());
            let target = target_identity(sotarena_root, dataset)?;
            identity = Some(target.clone());
            checkpoint_exact_overlap = overlapping_ids.contains(dataset);
            checkpoint_source_family_overlap =
                overlapping_families.contains(entry.source_family.as_str());
            checkpoint_target_identity_overlap = overlapping_targets.contains(target.as_str());
            let train_path = sotarena_root
                .join("data/tabular-regression-v1")
                .join(dataset)
                .join("train.csv");
            match sha256_file(&train_path) {
                Ok(actual) if actual == entry.train_sha256 => {
                    train_sha256 = Some(actual);
                    match read_regression_csv(&train_path) {
                        Ok(data) => {
                            rows = Some(data.y.len());
                            features = Some(data.features);
                            reasons.extend(validate_train_data(&data));
                            if build_splits(
                                run_id,
                                dataset,
                                &entry.train_sha256,
                                &data.y,
                                generations,
                            )
                            .is_err()
                            {
                                reasons.push("invalid_or_nondistinct_splits".to_string());
                            }
                            match sotarena_report_hashes(sotarena_root, dataset) {
                                Ok(hashes) => baseline_reports = Some(hashes),
                                Err(_) => reasons
                                    .push("completed_required_baseline_reports_absent".to_string()),
                            }
                            if reasons.is_empty() {
                                let mut frozen_entry = (*entry).clone();
                                frozen_entry.train_csv = train_path;
                                valid.push((
                                    frozen_entry,
                                    data,
                                    target,
                                    baseline_reports.clone().unwrap(),
                                ));
                            }
                        }
                        Err(_) => reasons.push("training_csv_parse_failure".to_string()),
                    }
                }
                Ok(_) => reasons.push("training_csv_hash_drift".to_string()),
                Err(_) => reasons.push("training_csv_missing".to_string()),
            }
        } else {
            reasons.push("verified_encoding_absent".to_string());
        }
        reasons.sort();
        reasons.dedup();
        records.push(EligibilityRecord {
            dataset_hash: (*dataset).to_string(),
            source_family,
            target_identity: identity,
            rows,
            features,
            train_sha256,
            context_sha256,
            checkpoint_exact_overlap,
            checkpoint_source_family_overlap,
            checkpoint_target_identity_overlap,
            baseline_reports,
            selection_key: selection_key(dataset),
            eligible: reasons.is_empty(),
            reasons,
            selected: false,
        });
    }
    if expected_eligible.is_some_and(|expected| valid.len() != expected) {
        bail!(
            "Golden42 eligibility drift: expected {}, found {}",
            expected_eligible.unwrap(),
            valid.len()
        );
    }
    let by_id = valid
        .into_iter()
        .map(|row| (row.0.dataset_hash.clone(), row))
        .collect::<BTreeMap<_, _>>();
    let selected = selected_ids
        .iter()
        .map(|dataset| {
            by_id
                .get(*dataset)
                .cloned()
                .with_context(|| format!("frozen dataset is no longer eligible: {dataset}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let selected_ids = selected
        .iter()
        .map(|(entry, _, _, _)| entry.dataset_hash.as_str())
        .collect::<HashSet<_>>();
    for record in &mut records {
        record.selected = selected_ids.contains(record.dataset_hash.as_str());
    }
    Ok((records, selected))
}

fn copy_context_slices(
    source_blob: &Path,
    output_blob: &Path,
    selected: &[(CatalogEntry, RegressionData, String, SotarenaReportHashes)],
) -> Result<(String, BTreeMap<String, EncodingSlice>)> {
    let mut source = File::open(source_blob)?;
    let mut bytes = Vec::with_capacity(selected.len() * CONTEXT_BYTES);
    let mut slices = BTreeMap::new();
    for (slot, (entry, _, _, _)) in selected.iter().enumerate() {
        let source_offset = entry
            .catalog_index
            .checked_mul(CONTEXT_BYTES)
            .context("context offset overflow")?;
        source.seek(SeekFrom::Start(source_offset as u64))?;
        let mut slice = vec![0_u8; CONTEXT_BYTES];
        source.read_exact(&mut slice)?;
        if sha256_bytes(&slice) != entry.context_sha256 {
            bail!("encoding slice drift for {}", entry.dataset_hash);
        }
        let output_offset = slot * CONTEXT_BYTES;
        bytes.extend_from_slice(&slice);
        slices.insert(
            entry.dataset_hash.clone(),
            EncodingSlice {
                blob_path: absolute_path(output_blob)?,
                blob_sha256: String::new(),
                offset_bytes: output_offset,
                length_bytes: CONTEXT_BYTES,
                context_sha256: entry.context_sha256.clone(),
            },
        );
    }
    let digest = sha256_bytes(&bytes);
    immutable_write(output_blob, &bytes)?;
    for slice in slices.values_mut() {
        slice.blob_sha256 = digest.clone();
    }
    Ok((digest, slices))
}

fn proposal_score(row: &ProposalRow) -> Result<f64> {
    let score = row
        .selection_score
        .or(row.proposal_score)
        .or(row.reward_lcb)
        .context("DL proposal lacks reward-LCB/stability ranking evidence")?;
    if !score.is_finite()
        || row
            .instability_probability
            .is_some_and(|value| !value.is_finite())
    {
        bail!("DL proposal has non-finite ranking evidence");
    }
    Ok(score)
}

fn load_dl_proposals(
    path: &Path,
    dataset_hash: &str,
    checkpoint_sha256: &str,
    context_sha256: &str,
    features: usize,
) -> Result<Vec<ProposalRow>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let row: ProposalRow = serde_json::from_str(&line?)?;
        if row.task != "regression" || row.dataset_hash != dataset_hash {
            continue;
        }
        if row.model_checkpoint_sha256 != checkpoint_sha256
            || row.context_sha256.as_deref() != Some(context_sha256)
            || semantic_identity(&row.canonical_genome) != row.semantic_sha256
            || !genome_compatible(&row.canonical_genome, features)
        {
            continue;
        }
        proposal_score(&row)?;
        rows.push(row);
    }
    rows.sort_by(|left, right| {
        proposal_score(right)
            .unwrap()
            .total_cmp(&proposal_score(left).unwrap())
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let mut semantics = HashSet::new();
    rows.retain(|row| semantics.insert(row.semantic_sha256.clone()));
    if rows.len() < 64 {
        bail!(
            "{dataset_hash} has {} valid unique checkpoint proposals; needs a 64-proposal reservoir",
            rows.len()
        );
    }
    rows.truncate(64);
    Ok(rows)
}

fn build_transfer_rankings(
    run_id: &str,
    observation_db: &Path,
    observation_db_sha256: &str,
    catalog: &Catalog,
    catalog_sha256: &str,
    target_entries: &[CatalogEntry],
) -> Result<BTreeMap<String, GlobalRankingReceipt>> {
    let targets = target_entries.to_vec();
    let family_by_dataset = catalog
        .entries
        .iter()
        .map(|entry| (entry.dataset_hash.clone(), entry.source_family.clone()))
        .collect::<HashMap<_, _>>();
    let connection = Connection::open_with_flags(
        observation_db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.execute_batch(
        "CREATE TEMP TABLE dataset_family(
            dataset_hash TEXT PRIMARY KEY,
            source_family TEXT NOT NULL
        ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection
            .prepare("INSERT INTO dataset_family(dataset_hash, source_family) VALUES (?1, ?2)")?;
        for entry in &catalog.entries {
            insert.execute([&entry.dataset_hash, &entry.source_family])?;
        }
    }
    let mut sql = String::from(
        r#"
        WITH clean AS (
          SELECT o.dataset_hash, o.semantic_sha256, o.canonical_genome, o.score,
                 coalesce(f.source_family, 'dataset:' || o.dataset_hash) AS source_family
          FROM observations o
          LEFT JOIN dataset_family f ON f.dataset_hash = o.dataset_hash
          WHERE o.task = 'regression'
            AND lower(candidate_source) NOT LIKE '%proposal%'
            AND lower(candidate_source) NOT LIKE '%model_sampled%'
            AND lower(candidate_source) NOT LIKE '%checkpoint_dl%'
            AND lower(origin_class) != 'dl_origin'
        ),
        positioned AS (
          SELECT dataset_hash, source_family, semantic_sha256, canonical_genome, score,
                 rank() OVER (PARTITION BY dataset_hash ORDER BY score ASC) AS asc_rank,
                 count(*) OVER (PARTITION BY dataset_hash) AS dataset_n,
                 max(score) OVER (PARTITION BY dataset_hash) AS dataset_max
          FROM clean
        ),
        with_tops AS (
          SELECT *, sum(CASE WHEN score = dataset_max THEN 1 ELSE 0 END)
                    OVER (PARTITION BY dataset_hash) AS top_count
          FROM positioned
        )
        SELECT semantic_sha256, canonical_genome, group_concat(dataset_hash)
        "#,
    );
    for target in &targets {
        let dataset = target.dataset_hash.replace('\'', "''");
        let family = target.source_family.replace('\'', "''");
        let include = format!("dataset_hash != '{dataset}' AND source_family != '{family}'");
        write!(
            sql,
            ", avg(CASE WHEN {include} THEN CASE WHEN dataset_n = 1 THEN 1.0 \
             ELSE CAST(asc_rank - 1 AS REAL) / CAST(dataset_n - 1 AS REAL) END END), \
             min(CASE WHEN {include} THEN CASE WHEN dataset_n = 1 THEN 1.0 \
             ELSE CAST(asc_rank - 1 AS REAL) / CAST(dataset_n - 1 AS REAL) END END), \
             sum(CASE WHEN {include} AND score = dataset_max AND top_count = 1 THEN 1 ELSE 0 END), \
             count(CASE WHEN {include} THEN 1 END)"
        )?;
    }
    sql.push_str(" FROM with_tops GROUP BY semantic_sha256, canonical_genome");

    let mut ranked = targets
        .iter()
        .map(|target| {
            (
                target.dataset_hash.clone(),
                Vec::<GlobalRankedGenome>::new(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut statement = connection.prepare(&sql)?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let semantic_sha256: String = row.get(0)?;
        let canonical_genome: String = row.get(1)?;
        let source_ids = row
            .get::<_, String>(2)?
            .split(',')
            .map(str::to_string)
            .collect::<Vec<_>>();
        for (target_index, target) in targets.iter().enumerate() {
            let base = 3 + target_index * 4;
            let mean_percentile: Option<f64> = row.get(base)?;
            let worst_percentile: Option<f64> = row.get(base + 1)?;
            let strict_win_count = row.get::<_, i64>(base + 2)? as usize;
            let dataset_count = row.get::<_, i64>(base + 3)? as usize;
            let (Some(mean_percentile), Some(worst_percentile)) =
                (mean_percentile, worst_percentile)
            else {
                continue;
            };
            let mut included_ids = source_ids
                .iter()
                .filter(|dataset| {
                    dataset.as_str() != target.dataset_hash
                        && family_by_dataset
                            .get(dataset.as_str())
                            .map_or(true, |family| family != &target.source_family)
                })
                .cloned()
                .collect::<Vec<_>>();
            included_ids.sort();
            included_ids.dedup();
            let source_families = included_ids
                .iter()
                .map(|dataset| {
                    family_by_dataset
                        .get(dataset)
                        .cloned()
                        .unwrap_or_else(|| format!("dataset:{dataset}"))
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            ranked
                .get_mut(&target.dataset_hash)
                .unwrap()
                .push(GlobalRankedGenome {
                    rank: 0,
                    canonical_genome: canonical_genome.clone(),
                    semantic_sha256: semantic_sha256.clone(),
                    mean_percentile,
                    worst_percentile,
                    strict_win_count,
                    dataset_count,
                    source_dataset_ids: included_ids,
                    source_families,
                });
        }
    }

    let mut receipts = BTreeMap::new();
    for target in targets {
        let mut genomes = ranked.remove(&target.dataset_hash).unwrap();
        genomes.sort_by(|left, right| {
            right
                .mean_percentile
                .total_cmp(&left.mean_percentile)
                .then_with(|| right.worst_percentile.total_cmp(&left.worst_percentile))
                .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
                .then_with(|| right.dataset_count.cmp(&left.dataset_count))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        genomes.truncate(50_000);
        for (rank, genome) in genomes.iter_mut().enumerate() {
            genome.rank = rank;
        }
        if genomes.len() < GLOBAL_QUOTA {
            bail!(
                "{} has fewer than {GLOBAL_QUOTA} target-clean transfer genomes",
                target.dataset_hash
            );
        }
        let excluded_dataset_ids = catalog
            .entries
            .iter()
            .filter(|entry| {
                entry.dataset_hash == target.dataset_hash
                    || entry.source_family == target.source_family
            })
            .map(|entry| entry.dataset_hash.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        receipts.insert(
            target.dataset_hash.clone(),
            GlobalRankingReceipt {
                schema: "TrainOnlyTransferGenomeRanking.v2".to_string(),
                run_id: run_id.to_string(),
                target_dataset_id: target.dataset_hash,
                target_source_family: target.source_family.clone(),
                observation_db_sha256: observation_db_sha256.to_string(),
                source_catalog_sha256: catalog_sha256.to_string(),
                excluded_dataset_ids,
                excluded_source_family: target.source_family,
                direct_dl_sources_excluded: true,
                ranking_policy:
                    "target and matching source family excluded; mean within-dataset percentile desc; worst percentile desc; strict win count desc; dataset count desc; semantic SHA-256 asc"
                        .to_string(),
                genomes,
            },
        );
    }
    Ok(receipts)
}

fn generation_zero(
    run_id: &str,
    entry: &CatalogEntry,
    data: &RegressionData,
    encoding: &EncodingSlice,
    checkpoint: &CheckpointProvenance,
    proposals_path: &Path,
    proposals_sha256: &str,
    ranking: &GlobalRankingReceipt,
    ranking_sha256: &str,
) -> Result<TrainOnlyGenerationZero> {
    let proposals = load_dl_proposals(
        proposals_path,
        &entry.dataset_hash,
        &checkpoint.checkpoint_sha256,
        &entry.context_sha256,
        data.features,
    )?;
    let mut members = Vec::with_capacity(POPULATION);
    let mut used = HashSet::new();
    for proposal in proposals.into_iter().take(DL_QUOTA) {
        if !used.insert(proposal.semantic_sha256.clone()) {
            continue;
        }
        let score = proposal_score(&proposal)?;
        members.push(GenomeMember {
            slot: members.len(),
            complexity: genome_complexity(&GenomeIr::parse(&proposal.canonical_genome)?),
            canonical_genome: proposal.canonical_genome,
            semantic_sha256: proposal.semantic_sha256,
            source: "latest_checkpoint_dl".to_string(),
            source_evidence: json!({
                "reservoir_slot": proposal.slot,
                "ranking_score": score,
                "reward_lcb": proposal.reward_lcb,
                "instability_probability": proposal.instability_probability,
            }),
            ancestry: Ancestry {
                operation: "generation_zero_dl".to_string(),
                parents: Vec::new(),
                roots: BTreeSet::from(["dl".to_string()]),
                retry_count: 0,
            },
        });
    }
    if members.len() != DL_QUOTA {
        bail!("{} cannot fill the exact DL quota", entry.dataset_hash);
    }
    for ranked in &ranking.genomes {
        if members.len() == POPULATION {
            break;
        }
        if used.contains(&ranked.semantic_sha256)
            || semantic_identity(&ranked.canonical_genome) != ranked.semantic_sha256
            || !genome_compatible(&ranked.canonical_genome, data.features)
        {
            continue;
        }
        used.insert(ranked.semantic_sha256.clone());
        members.push(GenomeMember {
            slot: members.len(),
            complexity: genome_complexity(&GenomeIr::parse(&ranked.canonical_genome)?),
            canonical_genome: ranked.canonical_genome.clone(),
            semantic_sha256: ranked.semantic_sha256.clone(),
            source: "target_clean_cross_dataset_transfer".to_string(),
            source_evidence: serde_json::to_value(ranked)?,
            ancestry: Ancestry {
                operation: "generation_zero_transfer".to_string(),
                parents: Vec::new(),
                roots: BTreeSet::from(["transfer".to_string()]),
                retry_count: 0,
            },
        });
    }
    if members.len() != POPULATION {
        bail!(
            "{} cannot fill the exact target-clean transfer quota after compatibility and deduplication",
            entry.dataset_hash
        );
    }
    Ok(TrainOnlyGenerationZero {
        schema: "TrainOnlyGenerationZero.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: entry.dataset_hash.clone(),
        train_sha256: entry.train_sha256.clone(),
        encoding_sha256: encoding.context_sha256.clone(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        dl_proposals_sha256: proposals_sha256.to_string(),
        transfer_ranking_sha256: ranking_sha256.to_string(),
        members,
    })
}

pub(crate) fn prepare(_registry: &Path, args: PrepareArgs) -> Result<PathBuf> {
    if args.workers == 0 {
        bail!("--workers must be at least one");
    }
    let generations = if args.smoke { 2 } else { GENERATIONS };
    let expected_datasets = if args.smoke { 1 } else { FROZEN_DATASETS.len() };
    let root = study_dir(&args.study_root, &args.run_id)?;
    fs::create_dir_all(root.join("inputs"))?;
    fs::create_dir_all(root.join("datasets"))?;
    let manifest_path = root.join("manifest.v2.json");
    if manifest_path.exists() {
        let (manifest, _): (TrainOnlyStudyManifest, _) = read_receipt(&manifest_path)?;
        if manifest.run_id != args.run_id {
            bail!("existing study manifest run id drift");
        }
        return Ok(manifest_path);
    }

    let mut checkpoint = resolve_checkpoint_provenance(&args)?;
    let pointer_snapshot_path = root.join("inputs/accepted-regression-pointer.snapshot.json");
    immutable_write(&pointer_snapshot_path, &fs::read(&args.checkpoint_pointer)?)?;
    checkpoint.accepted_pointer_path = absolute_path(&pointer_snapshot_path)?;
    let catalog_pointer_snapshot_path = root.join("inputs/context-catalog-pointer.snapshot.json");
    immutable_write(
        &catalog_pointer_snapshot_path,
        &fs::read(&args.catalog_pointer)?,
    )?;
    let (catalog_pointer, catalog) = load_catalog(&args.catalog_pointer)?;
    let exclusions =
        checkpoint_exclusions(&args.run_id, &args.sotarena_root, &checkpoint, &catalog)?;
    let exclusions_path = root.join("inputs/checkpoint-corpus-disclosure.v2.json");
    let exclusions_sha256 = write_receipt(&exclusions_path, &exclusions)?;

    let frozen_ids = if args.smoke {
        &FROZEN_DATASETS[..1]
    } else {
        FROZEN_DATASETS
    };
    let (mut eligibility, selected) = build_eligibility(
        &args.run_id,
        &args.sotarena_root,
        &catalog,
        &exclusions,
        generations,
        frozen_ids,
        Some(EXPECTED_ELIGIBLE),
    )?;
    if selected.len() != expected_datasets {
        let failure_path = root.join("preparation-failure.v2.json");
        let failure = json!({
            "schema": "TrainOnlyPreparationFailure.v2",
            "run_id": args.run_id,
            "seed": STUDY_SEED,
            "reason": "frozen_checkpoint_assisted_cohort_unavailable",
            "selected_count": selected.len(),
            "eligibility": eligibility,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "checkpoint_disclosure_sha256": exclusions_sha256,
            "no_fallback": true,
        });
        write_receipt(&failure_path, &failure)?;
        bail!(
            "only {} frozen Golden42 datasets pass the v2 study gates; preparation failed closed ({})",
            selected.len(),
            failure_path.display()
        );
    }

    let context_blob_path = root.join("inputs/contexts-1536+512.f32.bin");
    let (context_blob_sha256, encodings) =
        copy_context_slices(&catalog_pointer.data, &context_blob_path, &selected)?;
    let proposal_manifest_path = root.join("inputs/proposal-datasets.v1.json");
    let proposal_manifest = json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "run_id": args.run_id,
        "catalog_sha256": catalog_pointer.catalog_sha256,
        "entries": selected.iter().map(|(entry, _, _, _)| json!({
            "task": "regression",
            "dataset_hash": entry.dataset_hash,
            "context_sha256": entry.context_sha256,
        })).collect::<Vec<_>>(),
    });
    let proposal_manifest_sha256 = write_receipt(&proposal_manifest_path, &proposal_manifest)?;
    if args.proposal_manifest_only {
        return Ok(proposal_manifest_path);
    }

    let proposals_path = if let Some(path) = args.dl_proposals {
        let output = root.join("inputs/dl-proposals.jsonl");
        immutable_write(&output, &fs::read(&path)?)?;
        output
    } else {
        let output = root.join("inputs/dl-proposals.jsonl");
        let status = Command::new(&args.python)
            .arg(&args.proposal_generator)
            .arg("--catalog-pointer")
            .arg(&args.catalog_pointer)
            .arg("--checkpoint")
            .arg(&checkpoint.checkpoint_path)
            .arg("--dataset-manifest")
            .arg(&proposal_manifest_path)
            .arg("--task")
            .arg("regression")
            .arg("--proposals-per-dataset")
            .arg("64")
            .arg("--seed")
            .arg(STUDY_SEED.to_string())
            .arg("--device")
            .arg(&args.proposal_device)
            .arg("--output")
            .arg(&output)
            .status()
            .context("launch exact-manifest DL proposal generator")?;
        if !status.success() {
            bail!("exact-manifest DL proposal generator failed");
        }
        output
    };
    let proposals_path = absolute_path(&proposals_path)?;
    let proposals_sha256 = sha256_file(&proposals_path)?;
    for (entry, data, _, _) in &selected {
        load_dl_proposals(
            &proposals_path,
            &entry.dataset_hash,
            &checkpoint.checkpoint_sha256,
            &entry.context_sha256,
            data.features,
        )?;
    }

    let observation_db_sha256 = sha256_file(&args.observation_db)?;
    let target_entries = selected
        .iter()
        .map(|(entry, _, _, _)| entry.clone())
        .collect::<Vec<_>>();
    let rankings = build_transfer_rankings(
        &args.run_id,
        &args.observation_db,
        &observation_db_sha256,
        &catalog,
        &catalog_pointer.catalog_sha256,
        &target_entries,
    )?;

    let mut datasets = Vec::with_capacity(expected_datasets);
    for (entry, data, identity, baseline_reports) in &selected {
        let dataset_root = root.join("datasets").join(&entry.dataset_hash);
        fs::create_dir_all(&dataset_root)?;
        let splits = build_splits(
            &args.run_id,
            &entry.dataset_hash,
            &entry.train_sha256,
            &data.y,
            generations,
        )?;
        let split_path = dataset_root.join("splits.v2.json");
        let split_sha256 = write_receipt(&split_path, &splits)?;
        let encoding = encodings
            .get(&entry.dataset_hash)
            .context("filtered encoding slice missing")?
            .clone();
        let ranking = rankings
            .get(&entry.dataset_hash)
            .context("target-specific transfer ranking missing")?;
        let ranking_path = dataset_root.join("transfer-ranking.v2.json");
        let ranking_sha256 = write_receipt(&ranking_path, ranking)?;
        let generation_zero = generation_zero(
            &args.run_id,
            entry,
            data,
            &encoding,
            &checkpoint,
            &proposals_path,
            &proposals_sha256,
            ranking,
            &ranking_sha256,
        )?;
        let generation_zero_path = dataset_root.join("generation-zero.v2.json");
        let generation_zero_sha256 = write_receipt(&generation_zero_path, &generation_zero)?;
        datasets.push(DatasetStudyEntry {
            dataset_hash: entry.dataset_hash.clone(),
            source_family: entry.source_family.clone(),
            target_identity: identity.clone(),
            train_path: absolute_path(&entry.train_csv)?,
            train_sha256: entry.train_sha256.clone(),
            rows: data.y.len(),
            features: data.features,
            encoding,
            split_manifest_path: absolute_path(&split_path)?,
            split_manifest_sha256: split_sha256,
            generation_zero_path: absolute_path(&generation_zero_path)?,
            generation_zero_sha256,
            transfer_ranking_path: absolute_path(&ranking_path)?,
            transfer_ranking_sha256: ranking_sha256,
            baseline_reports: baseline_reports.clone(),
        });
    }
    datasets.sort_by_key(|entry| {
        selected
            .iter()
            .position(|(selected, _, _, _)| selected.dataset_hash == entry.dataset_hash)
            .unwrap()
    });
    let selected_dataset_ids = datasets
        .iter()
        .map(|entry| entry.dataset_hash.clone())
        .collect::<Vec<_>>();
    for record in &mut eligibility {
        record.selected = selected_dataset_ids.contains(&record.dataset_hash);
    }
    let exact_golden42_dataset_ids = exclusions
        .dataset_ids
        .iter()
        .filter(|dataset| GOLDEN42.contains(&dataset.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let golden42_source_families = exclusions
        .source_families
        .iter()
        .filter(|family| {
            catalog.entries.iter().any(|entry| {
                GOLDEN42.contains(&entry.dataset_hash.as_str()) && entry.source_family == **family
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let golden42_target_identities = eligibility
        .iter()
        .filter(|record| record.checkpoint_target_identity_overlap)
        .filter_map(|record| record.target_identity.clone())
        .collect::<Vec<_>>();
    let checkpoint_overlap_disclosure = CheckpointOverlapDisclosure {
        schema: "CheckpointOverlapDisclosure.v2".to_string(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        selected_exact_dataset_ids: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_exact_overlap)
            .map(|record| record.dataset_hash.clone())
            .collect(),
        selected_source_families: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_source_family_overlap)
            .filter_map(|record| record.source_family.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        selected_target_identities: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_target_identity_overlap)
            .filter_map(|record| record.target_identity.clone())
            .collect(),
        checkpoint_was_trained_on_golden42_identities: !exact_golden42_dataset_ids.is_empty()
            || !golden42_source_families.is_empty()
            || !golden42_target_identities.is_empty(),
        exact_golden42_dataset_ids,
        golden42_source_families,
        golden42_target_identities,
        zero_shot_valid: false,
        disclosure: "The accepted checkpoint was trained with Golden42 identities. This is an exploratory checkpoint-assisted transfer study, not a zero-shot generalization benchmark.".to_string(),
    };
    let sotarena_registry_path = args.sotarena_root.join("registry/datasets.json");
    let manifest = TrainOnlyStudyManifest {
        schema: "TrainOnlyStudyManifest.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure,
        campaign_scope: if args.smoke { "smoke" } else { "full" }.to_string(),
        run_id: args.run_id,
        seed: STUDY_SEED,
        generations,
        population: POPULATION,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament_size: TOURNAMENT,
        mutation_probability: 0.75,
        crossover_probability: 0.25,
        max_transforms: MAX_TRANSFORMS,
        workers: args.workers,
        kernel_version: KERNEL_VERSION.to_string(),
        sotarena_root: absolute_path(&args.sotarena_root)?,
        sotarena_registry_sha256: sha256_file(&sotarena_registry_path)?,
        sotarena_registry_path: absolute_path(&sotarena_registry_path)?,
        golden42_ids: GOLDEN42.iter().map(|value| (*value).to_string()).collect(),
        eligibility,
        selected_dataset_ids,
        checkpoint,
        corpus_exclusions_path: absolute_path(&exclusions_path)?,
        corpus_exclusions_sha256: exclusions_sha256,
        catalog_pointer_path: absolute_path(&catalog_pointer_snapshot_path)?,
        catalog_pointer_sha256: sha256_file(&catalog_pointer_snapshot_path)?,
        source_catalog_sha256: catalog_pointer.catalog_sha256,
        filtered_context_blob_sha256: context_blob_sha256,
        observation_db_path: absolute_path(&args.observation_db)?,
        observation_db_sha256,
        dl_proposals_path: proposals_path,
        dl_proposals_sha256: proposals_sha256,
        proposal_dataset_manifest_path: absolute_path(&proposal_manifest_path)?,
        proposal_dataset_manifest_sha256: proposal_manifest_sha256,
        datasets,
        firewall: PhaseOneFirewall {
            train_only: true,
            contains_test_paths: false,
            contains_test_hashes: false,
            contains_baseline_scores: false,
            test_paths_constructed: false,
        },
    };
    let serialized = String::from_utf8(canonical_json_bytes(&serde_json::to_value(&manifest)?)?)?;
    for forbidden in ["/test.csv", "\"test_sha256\"", "\"baseline_scores\""] {
        if serialized.to_ascii_lowercase().contains(forbidden) {
            bail!("sanitized phase-one manifest contains forbidden material: {forbidden}");
        }
    }
    write_receipt(&manifest_path, &manifest)?;
    Ok(manifest_path)
}

fn validate_manifest_contract(manifest_path: &Path) -> Result<(TrainOnlyStudyManifest, String)> {
    let (manifest, digest): (TrainOnlyStudyManifest, String) = read_receipt(manifest_path)?;
    let (expected_generations, expected_ids): (usize, &[&str]) =
        if manifest.campaign_scope == "smoke" {
            (2, &FROZEN_DATASETS[..1])
        } else if manifest.campaign_scope == "full" {
            (GENERATIONS, FROZEN_DATASETS)
        } else {
            bail!("study manifest has an unsupported campaign scope");
        };
    if manifest.schema != "TrainOnlyStudyManifest.v2"
        || manifest.study_design != STUDY_DESIGN
        || manifest.zero_shot_valid
        || manifest.checkpoint_overlap_disclosure.zero_shot_valid
        || !manifest
            .checkpoint_overlap_disclosure
            .checkpoint_was_trained_on_golden42_identities
        || manifest.seed != STUDY_SEED
        || manifest.generations != expected_generations
        || manifest.population != POPULATION
        || manifest.elites != ELITES
        || manifest.parent_pool != PARENT_POOL
        || manifest.tournament_size != TOURNAMENT
        || manifest.mutation_probability != 0.75
        || manifest.crossover_probability != 0.25
        || manifest.max_transforms != MAX_TRANSFORMS
        || manifest.workers == 0
        || manifest.kernel_version != KERNEL_VERSION
        || manifest.datasets.len() != expected_ids.len()
        || manifest.selected_dataset_ids
            != expected_ids
                .iter()
                .map(|dataset| (*dataset).to_string())
                .collect::<Vec<_>>()
        || !manifest.firewall.train_only
        || manifest.firewall.contains_test_paths
        || manifest.firewall.contains_test_hashes
        || manifest.firewall.contains_baseline_scores
        || manifest.firewall.test_paths_constructed
    {
        bail!("study manifest violates the fixed train-only contract");
    }
    let unique = manifest.selected_dataset_ids.iter().collect::<HashSet<_>>();
    if unique.len() != expected_ids.len()
        || manifest
            .datasets
            .iter()
            .map(|entry| &entry.dataset_hash)
            .collect::<Vec<_>>()
            != manifest.selected_dataset_ids.iter().collect::<Vec<_>>()
    {
        bail!("study manifest dataset cohort is not ten ordered unique identities");
    }
    let bytes = fs::read(manifest_path)?;
    let lower = String::from_utf8_lossy(&bytes).to_ascii_lowercase();
    for forbidden in ["/test.csv", "\"test_sha256\"", "\"baseline_scores\""] {
        if lower.contains(forbidden) {
            bail!("phase-one manifest firewall violation: {forbidden}");
        }
    }
    Ok((manifest, digest))
}

fn validate_prepared_inputs(manifest: &TrainOnlyStudyManifest) -> Result<()> {
    if sha256_file(&manifest.checkpoint.accepted_pointer_path)?
        != manifest.checkpoint.accepted_pointer_sha256
        || sha256_file(&manifest.checkpoint.checkpoint_path)?
            != manifest.checkpoint.checkpoint_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_manifest_path)?
            != manifest.checkpoint.training_corpus_manifest_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_train_path)?
            != manifest.checkpoint.training_corpus_train_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_validation_path)?
            != manifest.checkpoint.training_corpus_validation_sha256
        || sha256_file(&manifest.catalog_pointer_path)? != manifest.catalog_pointer_sha256
        || sha256_file(&manifest.sotarena_registry_path)? != manifest.sotarena_registry_sha256
        || sha256_file(&manifest.dl_proposals_path)? != manifest.dl_proposals_sha256
    {
        bail!("prepared input provenance drift");
    }
    // The authoritative observation store is append-active. Preparation seals
    // its content hash into every immutable per-target ranking; validation
    // audits those receipts below instead of requiring the live SQLite path to
    // remain byte-identical after the study was frozen.
    validate_sha256(
        &manifest.observation_db_sha256,
        "frozen observation database",
    )?;
    let (_, exclusion_digest): (CorpusExclusions, _) =
        read_receipt(&manifest.corpus_exclusions_path)?;
    let (_, proposal_manifest_digest): (Value, _) =
        read_receipt(&manifest.proposal_dataset_manifest_path)?;
    if exclusion_digest != manifest.corpus_exclusions_sha256
        || proposal_manifest_digest != manifest.proposal_dataset_manifest_sha256
    {
        bail!("prepared disclosure or proposal-manifest receipt drift");
    }
    for entry in &manifest.datasets {
        if sha256_file(&entry.train_path)? != entry.train_sha256
            || sha256_file(&entry.encoding.blob_path)? != entry.encoding.blob_sha256
            || entry.encoding.blob_sha256 != manifest.filtered_context_blob_sha256
        {
            bail!("prepared dataset input drift for {}", entry.dataset_hash);
        }
        let (_, split_digest): (TrainOnlySplitManifest, _) =
            read_receipt(&entry.split_manifest_path)?;
        let (_, generation_zero_digest): (TrainOnlyGenerationZero, _) =
            read_receipt(&entry.generation_zero_path)?;
        let (ranking, ranking_digest): (GlobalRankingReceipt, _) =
            read_receipt(&entry.transfer_ranking_path)?;
        if split_digest != entry.split_manifest_sha256
            || generation_zero_digest != entry.generation_zero_sha256
            || ranking_digest != entry.transfer_ranking_sha256
            || ranking.target_dataset_id != entry.dataset_hash
            || ranking.target_source_family != entry.source_family
            || sha256_file(&entry.baseline_reports.tabpfn_v3_path)?
                != entry.baseline_reports.tabpfn_v3_sha256
            || sha256_file(&entry.baseline_reports.tabfm_path)?
                != entry.baseline_reports.tabfm_sha256
            || sha256_file(&entry.baseline_reports.tabfm_ensemble_path)?
                != entry.baseline_reports.tabfm_ensemble_sha256
        {
            bail!("prepared dataset receipt drift for {}", entry.dataset_hash);
        }
    }
    Ok(())
}

fn validate_execution_inputs(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
) -> Result<()> {
    if sha256_file(&dataset.train_path)? != dataset.train_sha256
        || sha256_file(&dataset.encoding.blob_path)? != dataset.encoding.blob_sha256
        || sha256_file(&manifest.dl_proposals_path)? != manifest.dl_proposals_sha256
    {
        bail!("sharded execution input drift for {}", dataset.dataset_hash);
    }
    let (_, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    let (_, generation_zero_digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    let (_, transfer_digest): (GlobalRankingReceipt, _) =
        read_receipt(&dataset.transfer_ranking_path)?;
    if split_digest != dataset.split_manifest_sha256
        || generation_zero_digest != dataset.generation_zero_sha256
        || transfer_digest != dataset.transfer_ranking_sha256
    {
        bail!(
            "sharded execution receipt drift for {}",
            dataset.dataset_hash
        );
    }
    Ok(())
}

fn cache_key(train_sha256: &str, semantic_sha256: &str, split_sha256: &str) -> Result<String> {
    receipt_digest(&json!({
        "train_sha256": train_sha256,
        "semantic_sha256": semantic_sha256,
        "split_sha256": split_sha256,
        "kernel_version": KERNEL_VERSION,
    }))
}

fn slice_data(data: &RegressionData, indices: &[usize]) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut x = Vec::with_capacity(indices.len());
    let mut y = Vec::with_capacity(indices.len());
    for &index in indices {
        x.push(
            data.x
                .get(index)
                .with_context(|| format!("row {index} outside training data"))?
                .clone(),
        );
        y.push(data.y[index]);
    }
    Ok((x, y))
}

struct StudySplitSlices {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
    validation_x: Vec<Vec<f64>>,
    validation_y: Vec<f64>,
}

struct StudyEvalContext {
    context_tag: String,
    folds: usize,
    slices: BTreeMap<String, StudySplitSlices>,
    stage_cache: StageCache,
}

impl StudyEvalContext {
    fn new(
        data: &RegressionData,
        dataset: &DatasetStudyEntry,
        splits: &TrainOnlySplitManifest,
    ) -> Result<Self> {
        let mut cached = BTreeMap::new();
        for split in &splits.splits {
            let (train_x, train_y) = slice_data(data, &split.train_indices)?;
            let (validation_x, validation_y) = slice_data(data, &split.validation_indices)?;
            cached.insert(
                split.split_sha256.clone(),
                StudySplitSlices {
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                },
            );
        }
        Ok(Self {
            context_tag: format!(
                "{}|{}|rows={}|features={}",
                dataset.dataset_hash,
                dataset.train_sha256,
                data.y.len(),
                data.features
            ),
            folds: splits.splits.len(),
            slices: cached,
            stage_cache: StageCache::new(DEFAULT_BYTE_CAP),
        })
    }

    fn begin_generation(&self, generation: usize) {
        self.stage_cache.begin_generation(generation as u64);
    }

    fn sweep(&self) {
        self.stage_cache.sweep();
    }

    fn evaluate(
        &self,
        genome: &GenomeIr,
        split: &SplitReceipt,
    ) -> Result<(f64, f64, f64, f64, f64)> {
        let slices = self
            .slices
            .get(&split.split_sha256)
            .context("split slice cache miss")?;
        let context = StageCacheCtx {
            context_tag: self.context_tag.clone(),
            folds: self.folds,
            fold: split.generation,
        };
        let fit_started = Instant::now();
        let model = fit_with_stage_cache(
            genome,
            &slices.train_x,
            &slices.train_y,
            &self.stage_cache,
            &context,
        )?;
        let fit_seconds = fit_started.elapsed().as_secs_f64();
        let predict_started = Instant::now();
        let predictions = model.predict(&slices.validation_x);
        let predict_seconds = predict_started.elapsed().as_secs_f64();
        let (rmse, r2, mae) = metrics(&slices.validation_y, &predictions)?;
        Ok((rmse, r2, mae, fit_seconds, predict_seconds))
    }
}

fn evaluate_member_split(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    member: &GenomeMember,
    split: &SplitReceipt,
) -> Result<(TrainOnlyFitnessObservation, ObservationLink)> {
    let key = cache_key(
        &dataset.train_sha256,
        &member.semantic_sha256,
        &split.split_sha256,
    )?;
    let path = root
        .join("evaluations")
        .join(&dataset.dataset_hash)
        .join(format!("{key}.v1.json"));
    if path.exists() {
        let (observation, digest): (TrainOnlyFitnessObservation, _) = read_receipt(&path)?;
        if observation.schema != "TrainOnlyFitnessObservation.v2"
            || observation.run_id != manifest.run_id
            || observation.dataset_hash != dataset.dataset_hash
            || observation.train_sha256 != dataset.train_sha256
            || observation.encoding_sha256 != dataset.encoding.context_sha256
            || observation.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
            || observation.semantic_sha256 != member.semantic_sha256
            || observation.split_generation != split.generation
            || observation.split_sha256 != split.split_sha256
            || observation.kernel_version != KERNEL_VERSION
            || observation.cache_key != key
        {
            bail!("evaluation cache identity drift: {}", path.display());
        }
        return Ok((
            observation,
            ObservationLink {
                split_generation: split.generation,
                split_sha256: split.split_sha256.clone(),
                receipt_path: absolute_path(&path)?,
                receipt_sha256: digest,
                cache_key: key,
                cache_hit: true,
            },
        ));
    }

    let mut observation = TrainOnlyFitnessObservation {
        schema: "TrainOnlyFitnessObservation.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        encoding_sha256: dataset.encoding.context_sha256.clone(),
        checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        semantic_sha256: member.semantic_sha256.clone(),
        split_generation: split.generation,
        split_sha256: split.split_sha256.clone(),
        kernel_version: KERNEL_VERSION.to_string(),
        cache_key: key.clone(),
        r2: None,
        rmse: None,
        mae: None,
        fit_seconds: 0.0,
        predict_seconds: 0.0,
        failure_reason: None,
    };
    let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        if semantic_identity(&member.canonical_genome) != member.semantic_sha256
            || !genome_compatible(&member.canonical_genome, dataset.features)
        {
            bail!("canonical, semantic, or feature compatibility failure");
        }
        let genome = GenomeIr::parse(&member.canonical_genome)?;
        evaluator.evaluate(&genome, split)
    }));
    match evaluated {
        Ok(Ok((rmse, r2, mae, fit_seconds, predict_seconds))) => {
            observation.rmse = Some(rmse);
            observation.r2 = Some(r2);
            observation.mae = Some(mae);
            observation.fit_seconds = fit_seconds;
            observation.predict_seconds = predict_seconds;
        }
        Ok(Err(error)) => observation.failure_reason = Some(format!("{error:#}")),
        Err(payload) => {
            observation.failure_reason = Some(
                payload
                    .downcast_ref::<&str>()
                    .map(|value| format!("panic: {value}"))
                    .or_else(|| {
                        payload
                            .downcast_ref::<String>()
                            .map(|value| format!("panic: {value}"))
                    })
                    .unwrap_or_else(|| "panic: non-string payload".to_string()),
            )
        }
    }
    let digest = write_receipt(&path, &observation)?;
    Ok((
        observation,
        ObservationLink {
            split_generation: split.generation,
            split_sha256: split.split_sha256.clone(),
            receipt_path: absolute_path(&path)?,
            receipt_sha256: digest,
            cache_key: key,
            cache_hit: false,
        },
    ))
}

fn score_population(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    splits: &TrainOnlySplitManifest,
    generation: usize,
    population: &[GenomeMember],
    pool: &ThreadPool,
) -> Result<Vec<ScoredMember>> {
    if population.len() != POPULATION {
        bail!("generation {generation} population is not exactly {POPULATION}");
    }
    let mut semantics = HashSet::new();
    if population
        .iter()
        .any(|member| !semantics.insert(member.semantic_sha256.as_str()))
    {
        bail!("generation {generation} population repeats canonical semantics");
    }
    let mut scored = pool.install(|| {
        population
            .par_iter()
            .map(|member| -> Result<ScoredMember> {
                let mut observations = Vec::with_capacity(generation + 1);
                let mut r2 = Vec::with_capacity(generation + 1);
                let mut failure = None;
                for split in &splits.splits[..=generation] {
                    let (observation, mut link) =
                        evaluate_member_split(root, manifest, dataset, evaluator, member, split)?;
                    // Ledger cache accounting is logical rather than dependent on
                    // whether a prior interrupted attempt happened to materialize the
                    // same immutable observation before the generation barrier.
                    link.cache_hit =
                        member.ancestry.operation == "elite" && split.generation < generation;
                    observations.push(link);
                    if let Some(score) = observation.r2 {
                        r2.push(score);
                    } else {
                        failure = observation.failure_reason.clone();
                    }
                }
                let robust_fitness = if failure.is_none() && r2.len() == generation + 1 {
                    Some(paired_robust_fitness(&r2)?)
                } else {
                    None
                };
                let worst_split_r2 = robust_fitness.map(|_| {
                    r2.iter()
                        .copied()
                        .min_by(f64::total_cmp)
                        .expect("nonempty paired history")
                });
                Ok(ScoredMember {
                    member: member.clone(),
                    observations,
                    split_r2: r2,
                    robust_fitness,
                    worst_split_r2,
                    failure_reason: failure,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    scored.sort_by(rank_scored);
    Ok(scored)
}

#[derive(Clone, Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn offspring_seed(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let material =
        format!("train-only-offspring-v1\0{run_id}\0{dataset_hash}\0{generation}\0{slot}\0{retry}");
    let digest = Sha256::digest(material.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn tournament<'a>(rng: &mut DeterministicRng, ranked: &'a [ScoredMember]) -> &'a ScoredMember {
    let mut winner = (rng.next_u64() as usize) % PARENT_POOL;
    for _ in 1..TOURNAMENT {
        winner = winner.min((rng.next_u64() as usize) % PARENT_POOL);
    }
    &ranked[winner]
}

fn refill_population(
    run_id: &str,
    dataset_hash: &str,
    features: usize,
    generation: usize,
    ranked: &[ScoredMember],
    campaign_history: &mut HashSet<String>,
) -> Result<Vec<GenomeMember>> {
    if ranked
        .iter()
        .filter(|member| member.robust_fitness.is_some())
        .count()
        < PARENT_POOL
    {
        bail!("cannot refill from fewer than {PARENT_POOL} rankable parents");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut generation_semantics = HashSet::new();
    for elite in ranked.iter().take(ELITES) {
        generation_semantics.insert(elite.member.semantic_sha256.clone());
        let mut member = elite.member.clone();
        member.slot = next.len();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![member.semantic_sha256.clone()],
            roots: member.ancestry.roots.clone(),
            retry_count: 0,
        };
        next.push(member);
    }
    for slot in ELITES..POPULATION {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(offspring_seed(
                run_id,
                dataset_hash,
                generation,
                slot,
                retry,
            ));
            let crossover = rng.next_u64() % 4 == 0;
            let parent_a = tournament(&mut rng, ranked);
            let (operation, parents, roots, child) = if crossover {
                let parent_b = tournament(&mut rng, ranked);
                let Ok(child) = crossover_canonical_pipelines(
                    &parent_a.member.canonical_genome,
                    &parent_b.member.canonical_genome,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                ) else {
                    continue;
                };
                let mut roots = parent_a.member.ancestry.roots.clone();
                roots.extend(parent_b.member.ancestry.roots.iter().cloned());
                (
                    "crossover".to_string(),
                    vec![
                        parent_a.member.semantic_sha256.clone(),
                        parent_b.member.semantic_sha256.clone(),
                    ],
                    roots,
                    child,
                )
            } else {
                let Ok(child) = mutate_canonical_pipeline_v2(
                    &parent_a.member.canonical_genome,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                    true,
                ) else {
                    continue;
                };
                (
                    "mutation".to_string(),
                    vec![parent_a.member.semantic_sha256.clone()],
                    parent_a.member.ancestry.roots.clone(),
                    child,
                )
            };
            let Ok(genome) = GenomeIr::parse(&child) else {
                continue;
            };
            let canonical = genome.canonical();
            let semantic = semantic_identity(&canonical);
            if !matches!(genome, GenomeIr::Pipeline(_))
                || !genome_compatible(&canonical, features)
                || !generation_semantics.insert(semantic.clone())
            {
                continue;
            }
            if !campaign_history.insert(semantic.clone()) {
                generation_semantics.remove(&semantic);
                continue;
            }
            accepted = Some(GenomeMember {
                slot,
                canonical_genome: canonical,
                semantic_sha256: semantic,
                complexity: genome_complexity(&genome),
                source: format!("evolution_{operation}"),
                source_evidence: json!({
                    "generation": generation,
                    "slot": slot,
                    "operator_version": "canonical-v2",
                }),
                ancestry: Ancestry {
                    operation,
                    parents,
                    roots,
                    retry_count: retry,
                },
            });
            break;
        }
        next.push(accepted.with_context(|| {
            format!(
                "could not generate a campaign-unique child for generation {generation} slot {slot}"
            )
        })?);
    }
    Ok(next)
}

fn checkpoint_path(root: &Path, dataset: &str, generation: usize) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("generations")
        .join(format!("generation-{generation:03}.v2.json"))
}

fn selection_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("selection-complete.v2.json")
}

fn champion_validation_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("champion-fixed-grid.v2.json")
}

fn dataset_validation_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("validation-success.v2.json")
}

fn run_dataset(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    resume: bool,
    pool: &ThreadPool,
) -> Result<(String, String)> {
    let (splits, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    let (generation_zero, generation_zero_digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    if split_digest != dataset.split_manifest_sha256
        || generation_zero_digest != dataset.generation_zero_sha256
        || splits.splits.len() != manifest.generations
        || generation_zero.members.len() != POPULATION
    {
        bail!("{} preparation receipts are invalid", dataset.dataset_hash);
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let evaluator = StudyEvalContext::new(&data, dataset, &splits)?;
    let mut population = generation_zero.members;
    let mut previous = None;
    let mut checkpoint_hashes = Vec::with_capacity(manifest.generations);
    let mut champions = Vec::with_capacity(manifest.generations);
    let mut campaign_history = population
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    if campaign_history.len() != POPULATION {
        bail!("generation zero repeats semantics");
    }

    for generation in 0..manifest.generations {
        evaluator.begin_generation(generation);
        let path = checkpoint_path(root, &dataset.dataset_hash, generation);
        if path.exists() {
            if !resume {
                bail!("generation checkpoint exists; pass --resume");
            }
            let (checkpoint, digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(&path)?;
            if checkpoint.generation != generation
                || checkpoint.previous_checkpoint_sha256 != previous
                || checkpoint.population.len() != POPULATION
            {
                bail!("resume checkpoint chain drift at generation {generation}");
            }
            champions.push(checkpoint.champion_semantic_sha256.clone());
            checkpoint_hashes.push(digest.clone());
            previous = Some(digest);
            if generation + 1 < manifest.generations {
                population = checkpoint.next_population.clone();
                for member in population.iter().skip(ELITES) {
                    if !campaign_history.insert(member.semantic_sha256.clone()) {
                        bail!("resume campaign history repeats a descendant");
                    }
                }
            }
            continue;
        }
        let scored = score_population(
            root,
            manifest,
            dataset,
            &evaluator,
            &splits,
            generation,
            &population,
            pool,
        )?;
        let rankable = scored
            .iter()
            .filter(|member| member.robust_fitness.is_some())
            .count();
        if rankable < MIN_RANKABLE {
            bail!(
                "{} generation {} has only {} rankable members",
                dataset.dataset_hash,
                generation,
                rankable
            );
        }
        let champion = scored[0].member.semantic_sha256.clone();
        let next_population = if generation + 1 < manifest.generations {
            refill_population(
                &manifest.run_id,
                &dataset.dataset_hash,
                dataset.features,
                generation + 1,
                &scored,
                &mut campaign_history,
            )?
        } else {
            Vec::new()
        };
        let cache_hits = scored
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| link.cache_hit)
            .count();
        let cache_misses = scored
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| !link.cache_hit)
            .count();
        let checkpoint = TrainOnlyGenerationCheckpoint {
            schema: "TrainOnlyGenerationCheckpoint.v2".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            train_sha256: dataset.train_sha256.clone(),
            encoding_sha256: dataset.encoding.context_sha256.clone(),
            checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
            generation,
            previous_checkpoint_sha256: previous.clone(),
            split_manifest_sha256: split_digest.clone(),
            ranked_semantic_sha256: scored
                .iter()
                .map(|member| member.member.semantic_sha256.clone())
                .collect(),
            champion_semantic_sha256: champion.clone(),
            rankable_members: rankable,
            cache_hits,
            cache_misses,
            invalid_members: POPULATION - rankable,
            population: scored,
            next_population: next_population.clone(),
        };
        let digest = write_receipt(&path, &checkpoint)?;
        champions.push(champion);
        checkpoint_hashes.push(digest.clone());
        previous = Some(digest);
        population = next_population;
        evaluator.sweep();
    }
    let final_checkpoint_path =
        checkpoint_path(root, &dataset.dataset_hash, manifest.generations - 1);
    let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) =
        read_receipt(&final_checkpoint_path)?;
    let final_member = &final_checkpoint.population[0].member;
    let cumulative_sha256 = receipt_digest(&json!({
        "generation_checkpoint_sha256": checkpoint_hashes,
        "generation_champions": champions,
        "split_manifest_sha256": split_digest,
        "generation_zero_sha256": generation_zero_digest,
    }))?;
    let selection = TrainOnlySelectionComplete {
        schema: "TrainOnlySelectionComplete.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        encoding_sha256: dataset.encoding.context_sha256.clone(),
        checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
        split_manifest_sha256: split_digest.clone(),
        generation_checkpoint_sha256: checkpoint_hashes,
        generation_champions: champions.clone(),
        final_champion_semantic_sha256: final_member.semantic_sha256.clone(),
        final_champion_genome: final_member.canonical_genome.clone(),
        cumulative_sha256,
    };
    let selection_path = selection_path(root, &dataset.dataset_hash);
    let selection_digest = write_receipt(&selection_path, &selection)?;

    let mut fixed_grid = Vec::with_capacity(manifest.generations);
    evaluator.begin_generation(manifest.generations);
    for (generation, semantic) in champions.iter().enumerate() {
        let (checkpoint, _): (TrainOnlyGenerationCheckpoint, _) =
            read_receipt(&checkpoint_path(root, &dataset.dataset_hash, generation))?;
        let member = checkpoint
            .population
            .iter()
            .find(|member| member.member.semantic_sha256 == *semantic)
            .context("frozen generation champion is absent")?
            .member
            .clone();
        let mut split_r2 = Vec::with_capacity(manifest.generations);
        for split in &splits.splits {
            let (observation, _) =
                evaluate_member_split(root, manifest, dataset, &evaluator, &member, split)?;
            split_r2.push(observation.r2.with_context(|| {
                format!("post-hoc champion failed: {:?}", observation.failure_reason)
            })?);
        }
        fixed_grid.push(ChampionFixedGrid {
            generation,
            semantic_sha256: member.semantic_sha256,
            canonical_genome: member.canonical_genome,
            robust_fitness: paired_robust_fitness(&split_r2)?,
            worst_split_r2: split_r2.iter().copied().min_by(f64::total_cmp).unwrap(),
            split_r2,
        });
    }
    let champion_validation = TrainOnlyChampionValidation {
        schema: "TrainOnlyChampionValidation.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        selection_complete_sha256: selection_digest.clone(),
        split_manifest_sha256: split_digest,
        champions: fixed_grid,
    };
    let validation_digest = write_receipt(
        &champion_validation_path(root, &dataset.dataset_hash),
        &champion_validation,
    )?;
    evaluator.sweep();
    Ok((selection_digest, validation_digest))
}

pub(crate) fn run(
    manifest_path: &Path,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let (manifest, manifest_digest) = validate_manifest_contract(manifest_path)?;
    if selected_dataset.is_none() {
        validate_prepared_inputs(&manifest)?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build study evaluation worker pool")?;
    let root = manifest_path
        .parent()
        .context("study manifest has no parent")?;
    let mut selection_hashes = BTreeMap::new();
    let mut validation_hashes = BTreeMap::new();
    let datasets = manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|wanted| dataset.dataset_hash == wanted))
        .collect::<Vec<_>>();
    if datasets.is_empty() {
        bail!("requested dataset is not in the frozen study manifest");
    }
    for dataset in datasets {
        if selected_dataset.is_some() {
            validate_execution_inputs(&manifest, dataset)?;
        }
        match run_dataset(root, &manifest, dataset, resume, &pool) {
            Ok((selection, validation)) => {
                selection_hashes.insert(dataset.dataset_hash.clone(), selection);
                validation_hashes.insert(dataset.dataset_hash.clone(), validation);
            }
            Err(error) => {
                let failure_path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join("failure.v2.json");
                let failure = json!({
                    "schema": "TrainOnlyDatasetFailure.v2",
                    "run_id": manifest.run_id,
                    "dataset_hash": dataset.dataset_hash,
                    "error": format!("{error:#}"),
                    "population_unchanged": true,
                    "dataset_not_replaced": true,
                });
                write_receipt(&failure_path, &failure)?;
                return Err(error);
            }
        }
    }
    if let Some(dataset) = selected_dataset {
        return Ok(selection_path(root, dataset));
    }
    let cumulative_sha256 = receipt_digest(&json!({
        "manifest_sha256": manifest_digest,
        "selection_complete_sha256": selection_hashes,
        "champion_validation_sha256": validation_hashes,
    }))?;
    let cohort = TrainOnlyCohortSelectionComplete {
        schema: "TrainOnlyCohortSelectionComplete.v2".to_string(),
        run_id: manifest.run_id.clone(),
        manifest_sha256: manifest_digest,
        dataset_ids: manifest.selected_dataset_ids,
        selection_complete_sha256: selection_hashes,
        champion_validation_sha256: validation_hashes,
        cumulative_sha256,
    };
    let path = root.join("selection-complete.v2.json");
    write_receipt(&path, &cohort)?;
    Ok(path)
}

fn validate_generation_zero(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    ranking: &GlobalRankingReceipt,
) -> Result<TrainOnlyGenerationZero> {
    let (generation_zero, digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    if digest != dataset.generation_zero_sha256
        || generation_zero.schema != "TrainOnlyGenerationZero.v2"
        || generation_zero.run_id != manifest.run_id
        || generation_zero.dataset_hash != dataset.dataset_hash
        || generation_zero.train_sha256 != dataset.train_sha256
        || generation_zero.encoding_sha256 != dataset.encoding.context_sha256
        || generation_zero.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || generation_zero.dl_proposals_sha256 != manifest.dl_proposals_sha256
        || generation_zero.transfer_ranking_sha256 != dataset.transfer_ranking_sha256
        || generation_zero.members.len() != POPULATION
    {
        bail!("generation-zero receipt drift for {}", dataset.dataset_hash);
    }
    let mut semantics = HashSet::new();
    let rank_by_semantic = ranking
        .genomes
        .iter()
        .map(|genome| (genome.semantic_sha256.as_str(), genome.rank))
        .collect::<HashMap<_, _>>();
    let mut previous_transfer_rank = None;
    for (slot, member) in generation_zero.members.iter().enumerate() {
        if member.slot != slot
            || semantic_identity(&member.canonical_genome) != member.semantic_sha256
            || !genome_compatible(&member.canonical_genome, dataset.features)
            || member.complexity != genome_complexity(&GenomeIr::parse(&member.canonical_genome)?)
            || !semantics.insert(member.semantic_sha256.as_str())
        {
            bail!("generation-zero canonical identity drift");
        }
        if slot < DL_QUOTA {
            if member.source != "latest_checkpoint_dl"
                || member.ancestry.roots != BTreeSet::from(["dl".to_string()])
            {
                bail!("generation-zero DL quota/source drift");
            }
        } else {
            if member.source != "target_clean_cross_dataset_transfer"
                || member.ancestry.roots != BTreeSet::from(["transfer".to_string()])
            {
                bail!("generation-zero transfer quota/source drift");
            }
            let rank = *rank_by_semantic
                .get(member.semantic_sha256.as_str())
                .context("transfer seed is absent from target-clean ranking")?;
            if previous_transfer_rank.is_some_and(|previous| rank <= previous) {
                bail!("generation-zero transfer seeds do not walk the target-clean ranking");
            }
            previous_transfer_rank = Some(rank);
        }
    }
    Ok(generation_zero)
}

fn validate_observation_link(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    member: &GenomeMember,
    split: &SplitReceipt,
    link: &ObservationLink,
) -> Result<TrainOnlyFitnessObservation> {
    let (observation, digest): (TrainOnlyFitnessObservation, _) = read_receipt(&link.receipt_path)?;
    let expected_key = cache_key(
        &dataset.train_sha256,
        &member.semantic_sha256,
        &split.split_sha256,
    )?;
    if digest != link.receipt_sha256
        || link.split_generation != split.generation
        || link.split_sha256 != split.split_sha256
        || link.cache_key != expected_key
        || observation.schema != "TrainOnlyFitnessObservation.v2"
        || observation.run_id != manifest.run_id
        || observation.dataset_hash != dataset.dataset_hash
        || observation.train_sha256 != dataset.train_sha256
        || observation.encoding_sha256 != dataset.encoding.context_sha256
        || observation.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || observation.canonical_genome != member.canonical_genome
        || observation.semantic_sha256 != member.semantic_sha256
        || observation.split_generation != split.generation
        || observation.split_sha256 != split.split_sha256
        || observation.kernel_version != KERNEL_VERSION
        || observation.cache_key != expected_key
    {
        bail!("fitness observation identity drift");
    }
    if observation.failure_reason.is_none()
        && (observation.r2.is_none() || observation.rmse.is_none() || observation.mae.is_none())
    {
        bail!("successful fitness observation lacks metrics");
    }
    if !link.cache_hit {
        let genome = GenomeIr::parse(&member.canonical_genome)?;
        let replay = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate(&genome, split)
        }));
        match replay {
            Ok(Ok((rmse, r2, mae, _, _)))
                if observation.failure_reason.is_none()
                    && observation.rmse == Some(rmse)
                    && observation.r2 == Some(r2)
                    && observation.mae == Some(mae) => {}
            Ok(Err(_)) | Err(_) if observation.failure_reason.is_some() => {}
            _ => bail!("fitness observation metric replay drift"),
        }
    }
    Ok(observation)
}

fn validate_dataset_run(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    ranking: &GlobalRankingReceipt,
    pool: &ThreadPool,
) -> Result<String> {
    let data = read_regression_csv(&dataset.train_path)?;
    let rebuilt_splits = build_splits(
        &manifest.run_id,
        &dataset.dataset_hash,
        &dataset.train_sha256,
        &data.y,
        manifest.generations,
    )?;
    let (splits, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    if split_digest != dataset.split_manifest_sha256
        || receipt_digest(&rebuilt_splits)? != split_digest
    {
        bail!("split replay drift for {}", dataset.dataset_hash);
    }
    let generation_zero = validate_generation_zero(manifest, dataset, ranking)?;
    let evaluator = StudyEvalContext::new(&data, dataset, &splits)?;
    let mut history = generation_zero
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut previous = None;
    let mut checkpoint_hashes = Vec::new();
    let mut champions = Vec::new();
    for generation in 0..manifest.generations {
        evaluator.begin_generation(generation);
        let path = checkpoint_path(root, &dataset.dataset_hash, generation);
        let (checkpoint, digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(&path)?;
        if checkpoint.schema != "TrainOnlyGenerationCheckpoint.v2"
            || checkpoint.run_id != manifest.run_id
            || checkpoint.dataset_hash != dataset.dataset_hash
            || checkpoint.train_sha256 != dataset.train_sha256
            || checkpoint.encoding_sha256 != dataset.encoding.context_sha256
            || checkpoint.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.split_manifest_sha256 != split_digest
            || checkpoint.population.len() != POPULATION
            || checkpoint.rankable_members < MIN_RANKABLE
        {
            bail!("generation checkpoint contract drift at generation {generation}");
        }
        let mut semantics = HashSet::new();
        for scored in &checkpoint.population {
            if !semantics.insert(scored.member.semantic_sha256.as_str())
                || semantic_identity(&scored.member.canonical_genome)
                    != scored.member.semantic_sha256
                || scored.observations.len() != generation + 1
                || (generation > 0
                    && (scored.member.slot < ELITES)
                        != (scored.member.ancestry.operation == "elite"))
            {
                bail!("population identity or paired-history drift");
            }
        }
        let checked = pool.install(|| {
            checkpoint
                .population
                .par_iter()
                .map(|scored| -> Result<(usize, usize)> {
                    let mut r2 = Vec::new();
                    let mut cache_hits = 0usize;
                    let mut cache_misses = 0usize;
                    for (split, link) in splits.splits[..=generation]
                        .iter()
                        .zip(&scored.observations)
                    {
                        let observation = validate_observation_link(
                            manifest,
                            dataset,
                            &evaluator,
                            &scored.member,
                            split,
                            link,
                        )?;
                        if generation == 0 && link.cache_hit {
                            bail!("generation zero unexpectedly reused an evaluation");
                        }
                        if generation > 0 {
                            let expected_hit = scored.member.ancestry.operation == "elite"
                                && split.generation < generation;
                            if link.cache_hit != expected_hit {
                                bail!("elite/new-descendant cache accounting drift");
                            }
                        }
                        if link.cache_hit {
                            cache_hits += 1;
                        } else {
                            cache_misses += 1;
                        }
                        if let Some(score) = observation.r2 {
                            r2.push(score);
                        }
                    }
                    let expected_fitness = if r2.len() == generation + 1 {
                        Some(paired_robust_fitness(&r2)?)
                    } else {
                        None
                    };
                    let expected_worst = expected_fitness.map(|_| {
                        r2.iter()
                            .copied()
                            .min_by(f64::total_cmp)
                            .expect("nonempty history")
                    });
                    if scored.split_r2 != r2
                        || scored.robust_fitness != expected_fitness
                        || scored.worst_split_r2 != expected_worst
                    {
                        bail!("paired robust fitness replay drift");
                    }
                    Ok((cache_hits, cache_misses))
                })
                .collect::<Vec<_>>()
        });
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;
        for row in checked {
            let (member_hits, member_misses) = row?;
            cache_hits += member_hits;
            cache_misses += member_misses;
        }
        let mut replay_ranked = checkpoint.population.clone();
        replay_ranked.sort_by(rank_scored);
        let replay_semantics = replay_ranked
            .iter()
            .map(|member| member.member.semantic_sha256.clone())
            .collect::<Vec<_>>();
        if replay_semantics != checkpoint.ranked_semantic_sha256
            || checkpoint.champion_semantic_sha256 != replay_semantics[0]
            || checkpoint.cache_hits != cache_hits
            || checkpoint.cache_misses != cache_misses
            || checkpoint.invalid_members != POPULATION - checkpoint.rankable_members
            || checkpoint.rankable_members
                != checkpoint
                    .population
                    .iter()
                    .filter(|member| member.robust_fitness.is_some())
                    .count()
        {
            bail!("ranking, winner, or cache summary replay drift");
        }
        if generation + 1 < manifest.generations {
            let mut replay_history = history.clone();
            let expected_next = refill_population(
                &manifest.run_id,
                &dataset.dataset_hash,
                dataset.features,
                generation + 1,
                &replay_ranked,
                &mut replay_history,
            )?;
            if checkpoint.next_population != expected_next
                || checkpoint.next_population[..ELITES]
                    .iter()
                    .map(|member| member.semantic_sha256.as_str())
                    .collect::<Vec<_>>()
                    != replay_ranked[..ELITES]
                        .iter()
                        .map(|member| member.member.semantic_sha256.as_str())
                        .collect::<Vec<_>>()
            {
                bail!("top-three survival or deterministic offspring replay drift");
            }
            history = replay_history;
        } else if !checkpoint.next_population.is_empty() {
            bail!("generation nine must not have a next population");
        }
        champions.push(checkpoint.champion_semantic_sha256.clone());
        checkpoint_hashes.push(digest.clone());
        previous = Some(digest);
        evaluator.sweep();
    }

    let selection_path = selection_path(root, &dataset.dataset_hash);
    let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path)?;
    let final_checkpoint: TrainOnlyGenerationCheckpoint =
        read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
            root,
            &dataset.dataset_hash,
            manifest.generations - 1,
        ))?
        .0;
    if selection.schema != "TrainOnlySelectionComplete.v2"
        || selection.run_id != manifest.run_id
        || selection.dataset_hash != dataset.dataset_hash
        || selection.train_sha256 != dataset.train_sha256
        || selection.encoding_sha256 != dataset.encoding.context_sha256
        || selection.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || selection.split_manifest_sha256 != split_digest
        || selection.generation_checkpoint_sha256 != checkpoint_hashes
        || selection.generation_champions != champions
        || selection.final_champion_semantic_sha256
            != final_checkpoint.population[0].member.semantic_sha256
        || selection.final_champion_genome != final_checkpoint.population[0].member.canonical_genome
    {
        bail!("selection-complete replay drift");
    }
    let expected_cumulative = receipt_digest(&json!({
        "generation_checkpoint_sha256": checkpoint_hashes,
        "generation_champions": champions,
        "split_manifest_sha256": split_digest,
        "generation_zero_sha256": dataset.generation_zero_sha256,
    }))?;
    if selection.cumulative_sha256 != expected_cumulative {
        bail!("selection cumulative digest drift");
    }
    let (fixed, _): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if fixed.schema != "TrainOnlyChampionValidation.v2"
        || fixed.run_id != manifest.run_id
        || fixed.dataset_hash != dataset.dataset_hash
        || fixed.selection_complete_sha256 != selection_digest
        || fixed.split_manifest_sha256 != split_digest
        || fixed.champions.len() != manifest.generations
    {
        bail!("champion fixed-grid receipt drift");
    }
    evaluator.begin_generation(manifest.generations);
    let fixed_checks = pool.install(|| {
        fixed
            .champions
            .par_iter()
            .map(|champion| -> Result<()> {
                let genome = GenomeIr::parse(&champion.canonical_genome)?;
                let replay_r2 = splits
                    .splits
                    .iter()
                    .map(|split| {
                        evaluator
                            .evaluate(&genome, split)
                            .map(|(_, r2, _, _, _)| r2)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if champion.generation >= manifest.generations
                    || champion.semantic_sha256
                        != selection.generation_champions[champion.generation]
                    || semantic_identity(&champion.canonical_genome) != champion.semantic_sha256
                    || champion.split_r2.len() != manifest.generations
                    || champion.split_r2 != replay_r2
                    || champion.robust_fitness != paired_robust_fitness(&champion.split_r2)?
                    || champion.worst_split_r2
                        != champion
                            .split_r2
                            .iter()
                            .copied()
                            .min_by(f64::total_cmp)
                            .unwrap()
                {
                    bail!("champion fixed-grid score replay drift");
                }
                Ok(())
            })
            .collect::<Vec<_>>()
    });
    for check in fixed_checks {
        check?;
    }
    evaluator.sweep();
    Ok(selection_digest)
}

fn validate_dataset_shard(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    manifest_digest: &str,
    cohort: &TrainOnlyCohortSelectionComplete,
    cohort_digest: &str,
    dataset: &DatasetStudyEntry,
    pool: &ThreadPool,
) -> Result<(String, String)> {
    let (ranking, _): (GlobalRankingReceipt, _) = read_receipt(&dataset.transfer_ranking_path)?;
    let selection_digest = validate_dataset_run(root, manifest, dataset, &ranking, pool)?;
    if cohort.selection_complete_sha256.get(&dataset.dataset_hash) != Some(&selection_digest) {
        bail!("cohort dataset selection digest drift");
    }
    let (_, champion_digest): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&champion_digest) {
        bail!("cohort champion-validation digest drift");
    }
    let receipt = TrainOnlyDatasetValidationReceipt {
        schema: "TrainOnlyDatasetValidationReceipt.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        manifest_sha256: manifest_digest.to_string(),
        cohort_selection_complete_sha256: cohort_digest.to_string(),
        selection_complete_sha256: selection_digest.clone(),
        champion_validation_sha256: champion_digest,
        kernel_version: KERNEL_VERSION.to_string(),
        metric_replay_passed: true,
        population_audit_passed: true,
        paired_history_audit_passed: true,
        cache_audit_passed: true,
    };
    let digest = write_receipt(
        &dataset_validation_path(root, &dataset.dataset_hash),
        &receipt,
    )?;
    Ok((selection_digest, digest))
}

fn read_dataset_validation_shard(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    manifest_digest: &str,
    cohort: &TrainOnlyCohortSelectionComplete,
    cohort_digest: &str,
    dataset: &DatasetStudyEntry,
) -> Result<(String, String)> {
    let (receipt, digest): (TrainOnlyDatasetValidationReceipt, _) =
        read_receipt(&dataset_validation_path(root, &dataset.dataset_hash))?;
    let (_, selection_digest): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path(root, &dataset.dataset_hash))?;
    let (_, champion_digest): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if receipt.schema != "TrainOnlyDatasetValidationReceipt.v2"
        || receipt.study_design != STUDY_DESIGN
        || receipt.run_id != manifest.run_id
        || receipt.dataset_hash != dataset.dataset_hash
        || receipt.manifest_sha256 != manifest_digest
        || receipt.cohort_selection_complete_sha256 != cohort_digest
        || receipt.selection_complete_sha256 != selection_digest
        || receipt.champion_validation_sha256 != champion_digest
        || receipt.kernel_version != KERNEL_VERSION
        || !receipt.metric_replay_passed
        || !receipt.population_audit_passed
        || !receipt.paired_history_audit_passed
        || !receipt.cache_audit_passed
        || cohort.selection_complete_sha256.get(&dataset.dataset_hash) != Some(&selection_digest)
        || cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&champion_digest)
    {
        bail!(
            "dataset validation shard drift for {}",
            dataset.dataset_hash
        );
    }
    Ok((selection_digest, digest))
}

pub(crate) fn validate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_execution_inputs(&manifest, dataset)?;
    } else {
        validate_prepared_inputs(&manifest)?;
    }
    let (exclusions, exclusion_digest): (CorpusExclusions, _) =
        read_receipt(&manifest.corpus_exclusions_path)?;
    if exclusion_digest != manifest.corpus_exclusions_sha256
        || manifest.checkpoint_overlap_disclosure.checkpoint_sha256
            != manifest.checkpoint.checkpoint_sha256
        || manifest.checkpoint_overlap_disclosure.zero_shot_valid
        || !manifest
            .checkpoint_overlap_disclosure
            .checkpoint_was_trained_on_golden42_identities
    {
        bail!("checkpoint-overlap disclosure audit drift");
    }
    let overlapping_ids = exclusions.dataset_ids.iter().collect::<HashSet<_>>();
    let overlapping_families = exclusions.source_families.iter().collect::<HashSet<_>>();
    let overlapping_targets = exclusions.target_identities.iter().collect::<HashSet<_>>();
    for dataset in manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|wanted| dataset.dataset_hash == wanted))
    {
        let record = manifest
            .eligibility
            .iter()
            .find(|record| record.dataset_hash == dataset.dataset_hash)
            .context("selected dataset lacks eligibility record")?;
        if !record.eligible || !record.selected || !record.reasons.is_empty() {
            bail!("selected dataset eligibility audit failed");
        }
        if record.checkpoint_exact_overlap != overlapping_ids.contains(&dataset.dataset_hash)
            || record.checkpoint_source_family_overlap
                != overlapping_families.contains(&dataset.source_family)
            || record.checkpoint_target_identity_overlap
                != overlapping_targets.contains(&dataset.target_identity)
        {
            bail!("selected dataset checkpoint-overlap disclosure drift");
        }
        let (ranking, ranking_digest): (GlobalRankingReceipt, _) =
            read_receipt(&dataset.transfer_ranking_path)?;
        if ranking_digest != dataset.transfer_ranking_sha256
            || ranking.schema != "TrainOnlyTransferGenomeRanking.v2"
            || ranking.run_id != manifest.run_id
            || ranking.target_dataset_id != dataset.dataset_hash
            || ranking.target_source_family != dataset.source_family
            || ranking.observation_db_sha256 != manifest.observation_db_sha256
            || ranking.source_catalog_sha256 != manifest.source_catalog_sha256
            || ranking.excluded_source_family != dataset.source_family
            || !ranking.excluded_dataset_ids.contains(&dataset.dataset_hash)
            || !ranking.direct_dl_sources_excluded
        {
            bail!("target transfer-ranking identity drift");
        }
        for (rank, genome) in ranking.genomes.iter().enumerate() {
            if genome.rank != rank
                || genome
                    .source_dataset_ids
                    .iter()
                    .any(|source| ranking.excluded_dataset_ids.contains(source))
                || genome.source_families.contains(&dataset.source_family)
            {
                bail!("target/source-family transfer exclusion drift");
            }
        }
        let mut blob = File::open(&dataset.encoding.blob_path)?;
        blob.seek(SeekFrom::Start(dataset.encoding.offset_bytes as u64))?;
        let mut bytes = vec![0_u8; dataset.encoding.length_bytes];
        blob.read_exact(&mut bytes)?;
        if sha256_bytes(&bytes) != dataset.encoding.context_sha256 {
            bail!("filtered context slice audit failed");
        }
    }
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if cohort.schema != "TrainOnlyCohortSelectionComplete.v2"
        || cohort.run_id != manifest.run_id
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
        || cohort.selection_complete_sha256.len() != manifest.datasets.len()
        || cohort.champion_validation_sha256.len() != manifest.datasets.len()
    {
        bail!("cohort selection-complete receipt drift");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build study validation worker pool")?;
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_dataset_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
            &pool,
        )?;
        return Ok(dataset_validation_path(&root, wanted));
    }
    let mut selections = BTreeMap::new();
    let mut dataset_validations = BTreeMap::new();
    for dataset in &manifest.datasets {
        let path = dataset_validation_path(&root, &dataset.dataset_hash);
        let (selection_digest, validation_digest) = if path.exists() {
            read_dataset_validation_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
            )?
        } else {
            validate_dataset_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
                &pool,
            )?
        };
        selections.insert(dataset.dataset_hash.clone(), selection_digest);
        dataset_validations.insert(dataset.dataset_hash.clone(), validation_digest);
    }
    let expected_cohort_cumulative = receipt_digest(&json!({
        "manifest_sha256": manifest_digest,
        "selection_complete_sha256": cohort.selection_complete_sha256,
        "champion_validation_sha256": cohort.champion_validation_sha256,
    }))?;
    if cohort.cumulative_sha256 != expected_cohort_cumulative {
        bail!("cohort cumulative digest drift");
    }
    let receipt = TrainOnlyValidationReceipt {
        schema: "TrainOnlyValidationReceipt.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure: manifest.checkpoint_overlap_disclosure.clone(),
        run_id: manifest.run_id.clone(),
        manifest_sha256: manifest_digest,
        cohort_selection_complete_sha256: cohort_digest,
        dataset_selection_sha256: selections,
        dataset_validation_sha256: dataset_validations,
        benchmark_valid: true,
        checkpoint_disclosure_audit_passed: true,
        transfer_exclusion_audit_passed: true,
        firewall_audit_passed: true,
        population_audit_passed: true,
        paired_history_audit_passed: true,
        cache_audit_passed: true,
        validated_dataset_count: manifest.datasets.len(),
    };
    let path = root.join("validation-success.v2.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}

fn require_validated_score_gate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<(
    PathBuf,
    TrainOnlyStudyManifest,
    String,
    TrainOnlyValidationReceipt,
    String,
)> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let validation_path = root.join("validation-success.v2.json");
    if !validation_path.exists() {
        bail!(
            "post-hoc scoring requires a successful independent validation receipt: {}",
            validation_path.display()
        );
    }

    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    validate_prepared_inputs(&manifest)?;
    let (validation, validation_digest): (TrainOnlyValidationReceipt, _) =
        read_receipt(&validation_path)?;
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if validation.schema != "TrainOnlyValidationReceipt.v2"
        || validation.study_design != STUDY_DESIGN
        || validation.zero_shot_valid
        || validation.run_id != run_id
        || validation.manifest_sha256 != manifest_digest
        || validation.cohort_selection_complete_sha256 != cohort_digest
        || !validation.benchmark_valid
        || !validation.checkpoint_disclosure_audit_passed
        || !validation.transfer_exclusion_audit_passed
        || !validation.firewall_audit_passed
        || !validation.population_audit_passed
        || !validation.paired_history_audit_passed
        || !validation.cache_audit_passed
        || validation.validated_dataset_count != manifest.datasets.len()
        || validation.dataset_selection_sha256.len() != manifest.datasets.len()
        || validation.dataset_validation_sha256.len() != manifest.datasets.len()
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
    {
        bail!("validation receipt does not authorize post-hoc scoring");
    }
    for dataset in &manifest.datasets {
        let (shard_selection_digest, shard_digest) = read_dataset_validation_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
        )?;
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
            || cohort.selection_complete_sha256.get(&dataset.dataset_hash)
                != Some(&selection_digest)
            || shard_selection_digest != selection_digest
            || validation
                .dataset_validation_sha256
                .get(&dataset.dataset_hash)
                != Some(&shard_digest)
            || selection.generation_checkpoint_sha256.len() != manifest.generations
        {
            bail!(
                "validated selection digest drift for {}",
                dataset.dataset_hash
            );
        }
        let (_, fixed_digest): (TrainOnlyChampionValidation, _) =
            read_receipt(&champion_validation_path(&root, &dataset.dataset_hash))?;
        if cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&fixed_digest) {
            bail!(
                "validated champion-grid digest drift for {}",
                dataset.dataset_hash
            );
        }
        for (generation, expected_digest) in
            selection.generation_checkpoint_sha256.iter().enumerate()
        {
            let (_, digest): (TrainOnlyGenerationCheckpoint, _) =
                read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
            if &digest != expected_digest {
                bail!(
                    "validated generation checkpoint digest drift for {} generation {}",
                    dataset.dataset_hash,
                    generation
                );
            }
        }
    }
    Ok((
        root,
        manifest,
        manifest_digest,
        validation,
        validation_digest,
    ))
}

fn baseline_metric(path: &Path, dataset_hash: &str, algorithm: &str) -> Result<f64> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read baseline {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
        || report.get("status").and_then(Value::as_str) != Some("completed")
    {
        bail!("baseline identity/status mismatch: {}", path.display());
    }
    let rmse = report
        .pointer("/metrics/rmse")
        .and_then(Value::as_f64)
        .context("baseline report lacks numeric metrics.rmse")?;
    if !rmse.is_finite() || rmse < 0.0 {
        bail!("baseline RMSE is invalid: {}", path.display());
    }
    Ok(rmse)
}

fn optional_baseline_metric(
    path: &Path,
    dataset_hash: &str,
    algorithm: &str,
) -> Result<Option<f64>> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read baseline {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
    {
        bail!("baseline identity mismatch: {}", path.display());
    }
    match report.get("status").and_then(Value::as_str) {
        Some("completed") => baseline_metric(path, dataset_hash, algorithm).map(Some),
        Some("failed") => Ok(None),
        _ => bail!("baseline status mismatch: {}", path.display()),
    }
}

fn prior_aspfm_metric(path: &Path, dataset_hash: &str) -> Result<f64> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read prior ASPFM report {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("schema").and_then(Value::as_str) != Some("JopeDimeDatasetReport.v3")
        || report.get("status").and_then(Value::as_str) != Some("completed")
        || report.get("benchmark_valid").and_then(Value::as_bool) != Some(true)
        || report.pointer("/algorithm/name").and_then(Value::as_str) != Some("jope-dime")
        || report.get("protocol").and_then(Value::as_str) != Some("aspfm-generated")
        || report
            .pointer("/generation_contract/algorithm_input")
            .and_then(Value::as_str)
            != Some("dataset-training-data")
        || report
            .pointer("/generation_contract/algorithm_output")
            .and_then(Value::as_str)
            != Some("task-specific-executable-algorithm")
        || report
            .pointer("/generation_contract/test_targets_used_for_generation")
            .and_then(Value::as_bool)
            != Some(false)
    {
        bail!(
            "prior ASPFM report identity/status mismatch: {}",
            path.display()
        );
    }
    let rmse = report
        .pointer("/metrics/rmse")
        .and_then(Value::as_f64)
        .context("prior ASPFM report lacks numeric metrics.rmse")?;
    if !rmse.is_finite() || rmse < 0.0 {
        bail!("prior ASPFM RMSE is invalid: {}", path.display());
    }
    Ok(rmse)
}

fn stronger_tabfm(tabfm: Option<f64>, ensemble: f64) -> (&'static str, f64) {
    if tabfm.is_some_and(|value| value <= ensemble) {
        ("tabfm_regressor", tabfm.unwrap())
    } else {
        ("tabfm_ensemble_regressor", ensemble)
    }
}

fn load_or_create_baselines(
    root: &Path,
    run_id: &str,
    datasets: &[DatasetStudyEntry],
) -> Result<(BaselineManifest, String)> {
    let path = root.join("baselines.v2.json");
    if path.exists() {
        let (manifest, digest): (BaselineManifest, _) = read_receipt(&path)?;
        if manifest.schema != "TrainOnlyBaselineManifest.v2"
            || manifest.run_id != run_id
            || manifest
                .entries
                .iter()
                .map(|entry| &entry.dataset_hash)
                .collect::<Vec<_>>()
                != datasets
                    .iter()
                    .map(|entry| &entry.dataset_hash)
                    .collect::<Vec<_>>()
        {
            bail!("pinned baseline manifest identity drift");
        }
        for entry in &manifest.entries {
            if sha256_file(&entry.prior_aspfm_report_path)? != entry.prior_aspfm_report_sha256
                || prior_aspfm_metric(&entry.prior_aspfm_report_path, &entry.dataset_hash)?
                    != entry.prior_aspfm_rmse
                || sha256_file(&entry.tabpfn_v3_report_path)? != entry.tabpfn_v3_report_sha256
                || sha256_file(&entry.tabfm_report_path)? != entry.tabfm_report_sha256
                || sha256_file(&entry.tabfm_ensemble_report_path)?
                    != entry.tabfm_ensemble_report_sha256
                || baseline_metric(
                    &entry.tabpfn_v3_report_path,
                    &entry.dataset_hash,
                    "tabpfn_v3_regressor",
                )? != entry.tabpfn_v3_rmse
                || optional_baseline_metric(
                    &entry.tabfm_report_path,
                    &entry.dataset_hash,
                    "tabfm_regressor",
                )? != entry.tabfm_rmse
                || baseline_metric(
                    &entry.tabfm_ensemble_report_path,
                    &entry.dataset_hash,
                    "tabfm_ensemble_regressor",
                )? != entry.tabfm_ensemble_rmse
                || (
                    entry.stronger_tabfm_algorithm.as_str(),
                    entry.stronger_tabfm_rmse,
                ) != stronger_tabfm(entry.tabfm_rmse, entry.tabfm_ensemble_rmse)
            {
                bail!("pinned baseline content drift for {}", entry.dataset_hash);
            }
        }
        return Ok((manifest, digest));
    }

    let mut entries = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let dataset_hash = &dataset.dataset_hash;
        let tabpfn = &dataset.baseline_reports.tabpfn_v3_path;
        let tabfm = &dataset.baseline_reports.tabfm_path;
        let tabfm_ensemble = &dataset.baseline_reports.tabfm_ensemble_path;
        let prior_aspfm = tabpfn
            .parent()
            .context("TabPFN baseline report has no dataset directory")?
            .join("jope-dime-regressor.json");
        let tabfm_rmse = optional_baseline_metric(tabfm, dataset_hash, "tabfm_regressor")?;
        let tabfm_ensemble_rmse =
            baseline_metric(tabfm_ensemble, dataset_hash, "tabfm_ensemble_regressor")?;
        let (stronger_tabfm_algorithm, stronger_tabfm_rmse) =
            stronger_tabfm(tabfm_rmse, tabfm_ensemble_rmse);
        entries.push(BaselineEntry {
            dataset_hash: dataset_hash.clone(),
            prior_aspfm_report_sha256: sha256_file(&prior_aspfm)?,
            prior_aspfm_rmse: prior_aspfm_metric(&prior_aspfm, dataset_hash)?,
            prior_aspfm_report_path: absolute_path(&prior_aspfm)?,
            tabpfn_v3_report_sha256: sha256_file(tabpfn)?,
            tabpfn_v3_rmse: baseline_metric(&tabpfn, dataset_hash, "tabpfn_v3_regressor")?,
            tabpfn_v3_report_path: absolute_path(&tabpfn)?,
            tabfm_report_sha256: sha256_file(tabfm)?,
            tabfm_rmse,
            tabfm_report_path: absolute_path(&tabfm)?,
            tabfm_ensemble_report_sha256: sha256_file(tabfm_ensemble)?,
            tabfm_ensemble_rmse,
            tabfm_ensemble_report_path: absolute_path(tabfm_ensemble)?,
            stronger_tabfm_algorithm: stronger_tabfm_algorithm.to_string(),
            stronger_tabfm_rmse,
        });
    }
    let manifest = BaselineManifest {
        schema: "TrainOnlyBaselineManifest.v2".to_string(),
        run_id: run_id.to_string(),
        entries,
    };
    let digest = write_receipt(&path, &manifest)?;
    Ok((manifest, digest))
}

fn score_frozen_champion(
    member: &GenomeMember,
    train: &RegressionData,
    test: &RegressionData,
    features: usize,
    generation: usize,
) -> Result<GenerationTestScore> {
    if train.features != features
        || test.features != features
        || semantic_identity(&member.canonical_genome) != member.semantic_sha256
        || !genome_compatible(&member.canonical_genome, features)
    {
        bail!("frozen champion data/genome compatibility failure");
    }
    let genome = GenomeIr::parse(&member.canonical_genome)?;
    let started = Instant::now();
    let model = fit(&genome, &train.x, &train.y)?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let predictions = model.predict(&test.x);
    let predict_seconds = started.elapsed().as_secs_f64();
    let (rmse, r2, mae) = metrics(&test.y, &predictions)?;
    Ok(GenerationTestScore {
        generation,
        semantic_sha256: member.semantic_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        rmse,
        r2,
        mae,
        fit_seconds,
        predict_seconds,
    })
}

fn load_or_create_final_cv_reranks(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    validation: &TrainOnlyValidationReceipt,
    validation_digest: &str,
    workers: usize,
) -> Result<BTreeMap<String, (TrainOnlyFinalCvRerank, String)>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("final-cv-{index}"))
        .build()
        .context("build final CV worker pool")?;
    let mut reranks = BTreeMap::new();
    for dataset in &manifest.datasets {
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
        {
            bail!("final CV selection is not covered by the validation receipt");
        }
        let final_generation = manifest.generations - 1;
        let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
            &checkpoint_path(root, &dataset.dataset_hash, final_generation),
        )?;
        if selection.generation_checkpoint_sha256[final_generation] != checkpoint_digest
            || selection.generation_champions[final_generation]
                != checkpoint.champion_semantic_sha256
            || checkpoint.ranked_semantic_sha256.len() < FINAL_CV_TOP_K
        {
            bail!("final CV source ranking drift for {}", dataset.dataset_hash);
        }
        let train = read_regression_csv(&dataset.train_path)?;
        let split_manifest = build_final_cv_splits(
            &manifest.run_id,
            &dataset.dataset_hash,
            &dataset.train_sha256,
            &train.y,
        )?;
        let evaluator = StudyEvalContext::new(&train, dataset, &split_manifest)?;
        evaluator.begin_generation(0);
        let source_candidates = checkpoint.ranked_semantic_sha256[..FINAL_CV_TOP_K]
            .iter()
            .enumerate()
            .map(|(original_rank, semantic)| {
                let scored = checkpoint
                    .population
                    .iter()
                    .find(|candidate| candidate.member.semantic_sha256 == *semantic)
                    .with_context(|| {
                        format!(
                            "final top-10 member {} absent for {}",
                            semantic, dataset.dataset_hash
                        )
                    })?;
                Ok((original_rank, scored.member.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidates = pool.install(|| {
            source_candidates
                .par_iter()
                .map(|(original_rank, member)| {
                    let evaluated =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                            if semantic_identity(&member.canonical_genome) != member.semantic_sha256
                                || !genome_compatible(&member.canonical_genome, dataset.features)
                            {
                                bail!("final CV genome compatibility failure");
                            }
                            let genome = GenomeIr::parse(&member.canonical_genome)?;
                            let mut fold_r2 = Vec::with_capacity(FINAL_CV_FOLDS);
                            let mut fold_rmse = Vec::with_capacity(FINAL_CV_FOLDS);
                            for split in &split_manifest.splits {
                                let (rmse, r2, _, _, _) = evaluator.evaluate(&genome, split)?;
                                fold_r2.push(r2);
                                fold_rmse.push(rmse);
                            }
                            Ok((fold_r2, fold_rmse))
                        }));
                    let (fold_r2, fold_rmse, robust_fitness, worst_fold_r2, failure_reason) =
                        match evaluated {
                            Ok(Ok((fold_r2, fold_rmse))) => {
                                let fitness = paired_robust_fitness(&fold_r2)?;
                                let worst = fold_r2
                                    .iter()
                                    .copied()
                                    .min_by(f64::total_cmp)
                                    .context("final CV score list is empty")?;
                                (fold_r2, fold_rmse, Some(fitness), Some(worst), None)
                            }
                            Ok(Err(error)) => (
                                Vec::new(),
                                Vec::new(),
                                None,
                                None,
                                Some(format!("{error:#}")),
                            ),
                            Err(payload) => (
                                Vec::new(),
                                Vec::new(),
                                None,
                                None,
                                Some(
                                    payload
                                        .downcast_ref::<&str>()
                                        .map(|value| format!("panic: {value}"))
                                        .or_else(|| {
                                            payload
                                                .downcast_ref::<String>()
                                                .map(|value| format!("panic: {value}"))
                                        })
                                        .unwrap_or_else(|| "panic: non-string payload".to_string()),
                                ),
                            ),
                        };
                    Ok(FinalCvCandidate {
                        original_rank: *original_rank,
                        semantic_sha256: member.semantic_sha256.clone(),
                        canonical_genome: member.canonical_genome.clone(),
                        complexity: member.complexity,
                        fold_r2,
                        fold_rmse,
                        robust_fitness,
                        worst_fold_r2,
                        failure_reason,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        evaluator.sweep();
        candidates.sort_by(rank_final_cv);
        let selected = candidates
            .first()
            .filter(|candidate| candidate.robust_fitness.is_some())
            .context("none of the final top-10 candidates completed CV")?;
        let selected_semantic_sha256 = selected.semantic_sha256.clone();
        let selected_canonical_genome = selected.canonical_genome.clone();
        let selected_robust_fitness = selected.robust_fitness.unwrap();
        let selected_worst_fold_r2 = selected.worst_fold_r2.unwrap();
        let selected_original_rank = selected.original_rank;
        let receipt = TrainOnlyFinalCvRerank {
            schema: "TrainOnlyFinalCvRerank.v2".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_receipt_sha256: validation_digest.to_string(),
            selection_complete_sha256: selection_digest,
            final_generation_checkpoint_sha256: checkpoint_digest,
            train_sha256: dataset.train_sha256.clone(),
            kernel_version: KERNEL_VERSION.to_string(),
            top_k: FINAL_CV_TOP_K,
            folds: FINAL_CV_FOLDS,
            split_manifest,
            candidates,
            selected_semantic_sha256,
            selected_canonical_genome,
            selected_robust_fitness,
            selected_worst_fold_r2,
            selected_original_rank,
            selection_frozen_before_test_access: true,
        };
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("final-top10-cv-rerank.v2.json");
        let digest = write_receipt(&path, &receipt)?;
        reranks.insert(dataset.dataset_hash.clone(), (receipt, digest));
    }
    Ok(reranks)
}

pub(crate) fn score(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    // This is the only gate above any construction of a test or baseline path.
    let (root, manifest, _, validation, validation_digest) =
        require_validated_score_gate(study_root, run_id, sotarena_root, workers)?;
    // Freeze the independent train-only top-10 CV choice for every dataset
    // before baseline metrics or test paths are constructed.
    let final_cv_reranks = load_or_create_final_cv_reranks(
        &root,
        &manifest,
        &validation,
        &validation_digest,
        workers,
    )?;
    let (baselines, baseline_digest) = load_or_create_baselines(&root, run_id, &manifest.datasets)?;

    let mut posthoc_hashes = BTreeMap::new();
    for dataset in &manifest.datasets {
        let (cv_rerank, cv_rerank_digest) = final_cv_reranks
            .get(&dataset.dataset_hash)
            .context("frozen final CV rerank missing")?;
        let baseline = baselines
            .entries
            .iter()
            .find(|entry| entry.dataset_hash == dataset.dataset_hash)
            .context("pinned baseline entry missing")?;
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
        {
            bail!("selection is not covered by the validation receipt");
        }
        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        let output_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("posthoc-score.v2.json");
        if output_path.exists() {
            let (existing, digest): (TrainOnlyPosthocScore, _) = read_receipt(&output_path)?;
            if existing.schema != "TrainOnlyPosthocScore.v2"
                || existing.run_id != run_id
                || existing.dataset_hash != dataset.dataset_hash
                || existing.validation_receipt_sha256 != validation_digest
                || existing.selection_complete_sha256 != selection_digest
                || existing.generation_checkpoint_sha256 != selection.generation_checkpoint_sha256
                || existing.baseline_manifest_sha256 != baseline_digest
                || existing.train_sha256 != dataset.train_sha256
                || existing.encoding_sha256 != dataset.encoding.context_sha256
                || existing.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
                || existing.test_sha256 != test_sha256
                || existing.kernel_version != KERNEL_VERSION
                || existing.scores.len() != manifest.generations
                || existing.final_cv_rerank_sha256 != *cv_rerank_digest
                || existing.final_cv_rerank_score.semantic_sha256
                    != cv_rerank.selected_semantic_sha256
                || existing.prior_aspfm_rmse != baseline.prior_aspfm_rmse
                || existing.tabpfn_v3_rmse != baseline.tabpfn_v3_rmse
                || existing.tabfm_rmse != baseline.tabfm_rmse
                || existing.tabfm_ensemble_rmse != baseline.tabfm_ensemble_rmse
                || existing.stronger_tabfm_algorithm != baseline.stronger_tabfm_algorithm
                || existing.stronger_tabfm_rmse != baseline.stronger_tabfm_rmse
                || !existing.test_trajectory_post_selection
            {
                bail!("existing post-hoc score identity drift");
            }
            for (generation, recorded) in existing.scores.iter().enumerate() {
                let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) =
                    read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
                if checkpoint_digest != selection.generation_checkpoint_sha256[generation]
                    || recorded.generation != generation
                    || recorded.semantic_sha256 != selection.generation_champions[generation]
                {
                    bail!("existing post-hoc generation identity drift");
                }
                let member = checkpoint
                    .population
                    .iter()
                    .find(|member| member.member.semantic_sha256 == recorded.semantic_sha256)
                    .context("existing post-hoc champion is absent")?;
                let replay = score_frozen_champion(
                    &member.member,
                    &train,
                    &test,
                    dataset.features,
                    generation,
                )?;
                if recorded.canonical_genome != replay.canonical_genome
                    || recorded.rmse != replay.rmse
                    || recorded.r2 != replay.r2
                    || recorded.mae != replay.mae
                {
                    bail!("existing post-hoc metric replay drift");
                }
            }
            let final_rmse = existing.scores[manifest.generations - 1].rmse;
            let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
                &checkpoint_path(&root, &dataset.dataset_hash, manifest.generations - 1),
            )?;
            let cv_member = final_checkpoint
                .population
                .iter()
                .find(|candidate| {
                    candidate.member.semantic_sha256 == cv_rerank.selected_semantic_sha256
                })
                .context("existing final CV selection is absent from final population")?;
            let cv_replay = score_frozen_champion(
                &cv_member.member,
                &train,
                &test,
                dataset.features,
                manifest.generations - 1,
            )?;
            if existing.final_cv_rerank_score.canonical_genome != cv_replay.canonical_genome
                || existing.final_cv_rerank_score.rmse != cv_replay.rmse
                || existing.final_cv_rerank_score.r2 != cv_replay.r2
                || existing.final_cv_rerank_score.mae != cv_replay.mae
            {
                bail!("existing final CV test metric replay drift");
            }
            let cv_rmse = existing.final_cv_rerank_score.rmse;
            if existing.beats_tabpfn_v3 != (final_rmse < baseline.tabpfn_v3_rmse)
                || existing.beats_stronger_tabfm != (final_rmse < baseline.stronger_tabfm_rmse)
                || existing.beats_prior_aspfm != (final_rmse < baseline.prior_aspfm_rmse)
                || existing.beats_both
                    != (final_rmse < baseline.tabpfn_v3_rmse
                        && final_rmse < baseline.stronger_tabfm_rmse)
                || existing.cv_rerank_beats_tabpfn_v3 != (cv_rmse < baseline.tabpfn_v3_rmse)
                || existing.cv_rerank_beats_stronger_tabfm
                    != (cv_rmse < baseline.stronger_tabfm_rmse)
                || existing.cv_rerank_beats_prior_aspfm != (cv_rmse < baseline.prior_aspfm_rmse)
                || existing.cv_rerank_beats_both
                    != (cv_rmse < baseline.tabpfn_v3_rmse && cv_rmse < baseline.stronger_tabfm_rmse)
            {
                bail!("existing post-hoc strict-win comparison drift");
            }
            posthoc_hashes.insert(dataset.dataset_hash.clone(), digest);
            continue;
        }

        let mut scores = Vec::with_capacity(manifest.generations);
        for generation in 0..manifest.generations {
            let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) =
                read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
            if selection.generation_checkpoint_sha256[generation] != checkpoint_digest
                || selection.generation_champions[generation] != checkpoint.champion_semantic_sha256
            {
                bail!("frozen generation champion ledger drift");
            }
            let champion = checkpoint
                .population
                .iter()
                .find(|scored| scored.member.semantic_sha256 == checkpoint.champion_semantic_sha256)
                .context("frozen generation champion absent from checkpoint")?;
            scores.push(score_frozen_champion(
                &champion.member,
                &train,
                &test,
                dataset.features,
                generation,
            )?);
        }
        let final_score = scores.last().context("post-hoc score list is empty")?;
        let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
            &checkpoint_path(&root, &dataset.dataset_hash, manifest.generations - 1),
        )?;
        let cv_member = final_checkpoint
            .population
            .iter()
            .find(|candidate| {
                candidate.member.semantic_sha256 == cv_rerank.selected_semantic_sha256
            })
            .context("final CV selection is absent from final population")?;
        let final_cv_rerank_score = score_frozen_champion(
            &cv_member.member,
            &train,
            &test,
            dataset.features,
            manifest.generations - 1,
        )?;
        let receipt = TrainOnlyPosthocScore {
            schema: "TrainOnlyPosthocScore.v2".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_receipt_sha256: validation_digest.clone(),
            selection_complete_sha256: selection_digest,
            generation_checkpoint_sha256: selection.generation_checkpoint_sha256,
            baseline_manifest_sha256: baseline_digest.clone(),
            train_sha256: dataset.train_sha256.clone(),
            encoding_sha256: dataset.encoding.context_sha256.clone(),
            checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
            test_sha256,
            kernel_version: KERNEL_VERSION.to_string(),
            final_cv_rerank_sha256: cv_rerank_digest.clone(),
            cv_rerank_beats_prior_aspfm: final_cv_rerank_score.rmse < baseline.prior_aspfm_rmse,
            cv_rerank_beats_tabpfn_v3: final_cv_rerank_score.rmse < baseline.tabpfn_v3_rmse,
            cv_rerank_beats_stronger_tabfm: final_cv_rerank_score.rmse
                < baseline.stronger_tabfm_rmse,
            cv_rerank_beats_both: final_cv_rerank_score.rmse < baseline.tabpfn_v3_rmse
                && final_cv_rerank_score.rmse < baseline.stronger_tabfm_rmse,
            final_cv_rerank_score,
            prior_aspfm_rmse: baseline.prior_aspfm_rmse,
            tabpfn_v3_rmse: baseline.tabpfn_v3_rmse,
            tabfm_rmse: baseline.tabfm_rmse,
            tabfm_ensemble_rmse: baseline.tabfm_ensemble_rmse,
            stronger_tabfm_algorithm: baseline.stronger_tabfm_algorithm.clone(),
            stronger_tabfm_rmse: baseline.stronger_tabfm_rmse,
            beats_prior_aspfm: final_score.rmse < baseline.prior_aspfm_rmse,
            beats_tabpfn_v3: final_score.rmse < baseline.tabpfn_v3_rmse,
            beats_stronger_tabfm: final_score.rmse < baseline.stronger_tabfm_rmse,
            beats_both: final_score.rmse < baseline.tabpfn_v3_rmse
                && final_score.rmse < baseline.stronger_tabfm_rmse,
            scores,
            test_trajectory_post_selection: true,
        };
        let digest = write_receipt(&output_path, &receipt)?;
        posthoc_hashes.insert(dataset.dataset_hash.clone(), digest);
    }
    let cumulative_sha256 = receipt_digest(&json!({
        "validation_receipt_sha256": validation_digest,
        "baseline_manifest_sha256": baseline_digest,
        "posthoc_score_sha256": posthoc_hashes,
    }))?;
    let index = TrainOnlyPosthocIndex {
        schema: "TrainOnlyPosthocIndex.v2".to_string(),
        run_id: run_id.to_string(),
        validation_receipt_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        dataset_ids: manifest.selected_dataset_ids,
        posthoc_score_sha256: posthoc_hashes,
        cumulative_sha256,
    };
    let path = root.join("posthoc-scores.v2.json");
    write_receipt(&path, &index)?;
    Ok(path)
}

fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("median requires finite values");
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn bootstrap_interval(values: &[f64]) -> Result<[f64; 2]> {
    if values.len() != 10 {
        bail!("study bootstrap is pinned to ten datasets");
    }
    let mut rng = DeterministicRng::new(STUDY_SEED ^ 0x6273_7472_6170);
    let mut estimates = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let sample = (0..values.len())
            .map(|_| values[(rng.next_u64() as usize) % values.len()])
            .collect::<Vec<_>>();
        estimates.push(median(&sample)?);
    }
    estimates.sort_by(f64::total_cmp);
    Ok([estimates[249], estimates[9749]])
}

fn paired_sign_test(values: &[f64]) -> f64 {
    let positive = values.iter().filter(|value| **value > 0.0).count();
    let negative = values.iter().filter(|value| **value < 0.0).count();
    let n = positive + negative;
    if n == 0 {
        return 1.0;
    }
    let tail = positive.min(negative);
    let mut coefficient = 1.0;
    let mut probability = 1.0;
    for k in 1..=tail {
        coefficient *= (n + 1 - k) as f64 / k as f64;
        probability += coefficient;
    }
    (2.0 * probability / 2_f64.powi(n as i32)).min(1.0)
}

fn cohort_elo(datasets: &[DatasetReport]) -> Result<Vec<EloRating>> {
    if datasets.is_empty() {
        bail!("Elo requires at least one dataset");
    }
    const SYSTEMS: [&str; 5] = [
        "ours_gen9",
        "ours_top10_5fold_cv",
        "prior_aspfm",
        "tabfm_ensemble",
        "tabpfn_v3",
    ];
    const INITIAL: f64 = 1500.0;
    const K: f64 = 32.0;
    let mut ratings = [INITIAL; SYSTEMS.len()];
    let mut wins = [0usize; SYSTEMS.len()];
    let mut losses = [0usize; SYSTEMS.len()];
    let mut ties = [0usize; SYSTEMS.len()];
    for dataset in datasets {
        let rmse = [
            dataset.generation_nine_rmse,
            dataset.cv_rerank_rmse,
            dataset.prior_aspfm_rmse,
            dataset.tabfm_ensemble_rmse,
            dataset.tabpfn_v3_rmse,
        ];
        if rmse.iter().any(|value| !value.is_finite() || *value < 0.0) {
            bail!("Elo received an invalid RMSE");
        }
        let before = ratings;
        let mut delta = [0.0; SYSTEMS.len()];
        for left in 0..SYSTEMS.len() {
            for right in (left + 1)..SYSTEMS.len() {
                let outcome = match rmse[left].total_cmp(&rmse[right]) {
                    Ordering::Less => {
                        wins[left] += 1;
                        losses[right] += 1;
                        1.0
                    }
                    Ordering::Greater => {
                        losses[left] += 1;
                        wins[right] += 1;
                        0.0
                    }
                    Ordering::Equal => {
                        ties[left] += 1;
                        ties[right] += 1;
                        0.5
                    }
                };
                let expected = 1.0 / (1.0 + 10_f64.powf((before[right] - before[left]) / 400.0));
                let change = K * (outcome - expected);
                delta[left] += change;
                delta[right] -= change;
            }
        }
        for index in 0..SYSTEMS.len() {
            ratings[index] += delta[index];
        }
    }
    let mut result = SYSTEMS
        .iter()
        .enumerate()
        .map(|(index, system)| EloRating {
            system: (*system).to_string(),
            rating: ratings[index],
            wins: wins[index],
            losses: losses[index],
            ties: ties[index],
        })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| {
        right
            .rating
            .total_cmp(&left.rating)
            .then_with(|| left.system.cmp(&right.system))
    });
    Ok(result)
}

fn panel_points(values: &[f64], left: f64, top: f64, width: f64, height: f64) -> String {
    panel_points_scaled(values, values, left, top, width, height)
}

fn panel_points_scaled(
    values: &[f64],
    scale: &[f64],
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) -> String {
    let minimum = scale.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = scale.iter().copied().max_by(f64::total_cmp).unwrap();
    let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
    let low = minimum - padding;
    let high = maximum + padding;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = left + width * index as f64 / (values.len() - 1).max(1) as f64;
            let y = top + height * (high - value) / (high - low);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn evolution_svg(
    title: &str,
    validation: &[f64],
    test: &[f64],
    tabpfn: f64,
    tabfm: Option<f64>,
    tabfm_ensemble: f64,
) -> String {
    let mut rmse_scale = test.to_vec();
    rmse_scale.extend([tabpfn, tabfm_ensemble]);
    if let Some(tabfm) = tabfm {
        rmse_scale.push(tabfm);
    }
    let validation_points = panel_points(validation, 80.0, 70.0, 820.0, 210.0);
    let test_points = panel_points_scaled(test, &rmse_scale, 80.0, 370.0, 820.0, 210.0);
    let baseline_y = |value: f64| {
        let minimum = rmse_scale.iter().copied().min_by(f64::total_cmp).unwrap();
        let maximum = rmse_scale.iter().copied().max_by(f64::total_cmp).unwrap();
        let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
        370.0 + 210.0 * (maximum + padding - value) / (maximum + padding - (minimum - padding))
    };
    let tabfm_line = tabfm.map_or_else(
        || {
            "<text x=\"905\" y=\"350\" font-family=\"sans-serif\" font-size=\"12\">TabFM unavailable</text>"
                .to_string()
        },
        |value| {
            let y = baseline_y(value);
            format!("<line x1=\"80\" y1=\"{y:.2}\" x2=\"900\" y2=\"{y:.2}\" stroke=\"#d97706\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM</text>")
        },
    );
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"650\" viewBox=\"0 0 960 650\">\n\
<rect width=\"960\" height=\"650\" fill=\"white\"/>\n\
<text x=\"480\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"20\">{title}</text>\n\
<text x=\"20\" y=\"175\" transform=\"rotate(-90 20 175)\" font-family=\"sans-serif\" font-size=\"14\">10-split robust validation fitness</text>\n\
<line x1=\"80\" y1=\"280\" x2=\"900\" y2=\"280\" stroke=\"#999\"/><polyline points=\"{validation_points}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\"/>\n\
<text x=\"20\" y=\"490\" transform=\"rotate(-90 20 490)\" font-family=\"sans-serif\" font-size=\"14\">post-hoc test RMSE</text>\n\
<line x1=\"80\" y1=\"580\" x2=\"900\" y2=\"580\" stroke=\"#999\"/><polyline points=\"{test_points}\" fill=\"none\" stroke=\"#059669\" stroke-width=\"3\"/>\n\
<line x1=\"80\" y1=\"{pfn_y:.2}\" x2=\"900\" y2=\"{pfn_y:.2}\" stroke=\"#dc2626\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{pfn_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabPFN v3</text>\n\
{tabfm_line}\n\
<line x1=\"80\" y1=\"{fm_ensemble_y:.2}\" x2=\"900\" y2=\"{fm_ensemble_y:.2}\" stroke=\"#7c3aed\" stroke-dasharray=\"4 4\"/><text x=\"905\" y=\"{fm_ensemble_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM ensemble</text>\n\
<text x=\"480\" y=\"625\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">generation 0–9</text>\n</svg>\n",
        pfn_y = baseline_y(tabpfn),
        fm_ensemble_y = baseline_y(tabfm_ensemble),
    )
}

fn report_dataset(
    root: &Path,
    dataset: &DatasetStudyEntry,
    posthoc: &TrainOnlyPosthocScore,
) -> Result<(DatasetReport, Vec<f64>)> {
    let (fixed, _): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    let (selection, _): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path(root, &dataset.dataset_hash))?;
    let final_checkpoint: TrainOnlyGenerationCheckpoint =
        read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
            root,
            &dataset.dataset_hash,
            posthoc.scores.len() - 1,
        ))?
        .0;
    let final_member = final_checkpoint
        .population
        .iter()
        .find(|member| member.member.semantic_sha256 == selection.final_champion_semantic_sha256)
        .context("final champion missing from generation-nine checkpoint")?;
    let ancestry = if final_member.member.ancestry.roots == BTreeSet::from(["dl".to_string()]) {
        "dl"
    } else if final_member.member.ancestry.roots == BTreeSet::from(["transfer".to_string()]) {
        "transfer"
    } else {
        "mixed"
    };
    let test_rmse = posthoc
        .scores
        .iter()
        .map(|score| score.rmse)
        .collect::<Vec<_>>();
    let validation = fixed
        .champions
        .iter()
        .map(|champion| champion.robust_fitness)
        .collect::<Vec<_>>();
    if test_rmse.is_empty() || test_rmse.len() != validation.len() {
        bail!("dataset report needs matching validation and test generations");
    }
    let final_generation = test_rmse.len() - 1;
    let relative = (test_rmse[0] - test_rmse[final_generation]) / test_rmse[0];
    if !relative.is_finite() {
        bail!("generation-zero relative improvement is undefined");
    }
    let mut best_baseline_algorithm = "prior_aspfm".to_string();
    let mut best_baseline_rmse = posthoc.prior_aspfm_rmse;
    if posthoc.tabpfn_v3_rmse < best_baseline_rmse {
        best_baseline_algorithm = "tabpfn_v3".to_string();
        best_baseline_rmse = posthoc.tabpfn_v3_rmse;
    }
    if posthoc.stronger_tabfm_rmse < best_baseline_rmse {
        best_baseline_algorithm = posthoc.stronger_tabfm_algorithm.clone();
        best_baseline_rmse = posthoc.stronger_tabfm_rmse;
    }
    let generation_nine_relative_gap_to_best_baseline =
        (test_rmse[final_generation] - best_baseline_rmse) / best_baseline_rmse;
    let cv_rerank_relative_gap_to_best_baseline =
        (posthoc.final_cv_rerank_score.rmse - best_baseline_rmse) / best_baseline_rmse;

    let report_root = root.join("reports").join(&dataset.dataset_hash);
    fs::create_dir_all(&report_root)?;
    immutable_write(
        &report_root.join("evolution.svg"),
        evolution_svg(
            &format!("{} checkpoint-assisted evolution", dataset.dataset_hash),
            &validation,
            &test_rmse,
            posthoc.tabpfn_v3_rmse,
            posthoc.tabfm_rmse,
            posthoc.tabfm_ensemble_rmse,
        )
        .as_bytes(),
    )?;

    let mut trends = String::from(
        "generation,validation_robust_fitness,test_rmse,unique_genomes,dl_rooted,transfer_rooted,mixed_rooted,invalid_rate,cache_hit_rate,fit_seconds,predict_seconds\n",
    );
    for generation in 0..test_rmse.len() {
        let checkpoint: TrainOnlyGenerationCheckpoint =
            read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
                root,
                &dataset.dataset_hash,
                generation,
            ))?
            .0;
        let mut dl = 0;
        let mut transfer = 0;
        let mut mixed = 0;
        for member in &checkpoint.population {
            if member.member.ancestry.roots == BTreeSet::from(["dl".to_string()]) {
                dl += 1;
            } else if member.member.ancestry.roots == BTreeSet::from(["transfer".to_string()]) {
                transfer += 1;
            } else {
                mixed += 1;
            }
        }
        let total_observations = checkpoint.cache_hits + checkpoint.cache_misses;
        let (fit_seconds, predict_seconds) = checkpoint
            .population
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| !link.cache_hit)
            .try_fold((0.0, 0.0), |(fit_sum, predict_sum), link| {
                let (observation, _): (TrainOnlyFitnessObservation, _) =
                    read_receipt(&link.receipt_path)?;
                Ok::<_, anyhow::Error>((
                    fit_sum + observation.fit_seconds,
                    predict_sum + observation.predict_seconds,
                ))
            })?;
        writeln!(
            trends,
            "{generation},{},{},{},{dl},{transfer},{mixed},{},{},{fit_seconds},{predict_seconds}",
            validation[generation],
            test_rmse[generation],
            checkpoint
                .population
                .iter()
                .map(|member| &member.member.semantic_sha256)
                .collect::<HashSet<_>>()
                .len(),
            checkpoint.invalid_members as f64 / POPULATION as f64,
            if total_observations == 0 {
                0.0
            } else {
                checkpoint.cache_hits as f64 / total_observations as f64
            },
        )?;
    }
    immutable_write(
        &report_root.join("generation-trends.csv"),
        trends.as_bytes(),
    )?;
    let tabfm_display = posthoc.tabfm_rmse.map_or_else(
        || "unavailable (source report failed)".to_string(),
        |value| format!("{value:.12}"),
    );
    let markdown = format!(
        "# Dataset {}\n\n\
Final ancestry: `{ancestry}`.\n\n\
Generation 0 test RMSE: `{:.12}`. Final-generation test RMSE: `{:.12}` ({:+.2}% relative improvement).\n\n\
Prior ASPFM RMSE: `{:.12}`. TabPFN v3 RMSE: `{:.12}`. TabFM RMSE: `{}`. TabFM ensemble RMSE: `{:.12}`. The stronger TabFM comparison is `{}` at `{:.12}`. Strict final wins: prior ASPFM `{}`, TabPFN `{}`, stronger TabFM `{}`, both fixed baselines `{}`.\n\n\
The independently frozen top-10/five-fold CV rerank selected `{}` and scored test RMSE `{:.12}`. Strict CV-rerank wins: prior ASPFM `{}`, TabPFN `{}`, stronger TabFM `{}`, both fixed baselines `{}`.\n\n\
This is an exploratory checkpoint-assisted transfer study. The checkpoint has seen Golden42 identities, so these results are not zero-shot generalization evidence.\n\n\
The entire test trajectory was generated post-hoc after selection and independent validation. The top-10 CV choice was also sealed before any test path or baseline metric was opened.\n",
        dataset.dataset_hash,
        test_rmse[0],
        test_rmse[final_generation],
        relative * 100.0,
        posthoc.prior_aspfm_rmse,
        posthoc.tabpfn_v3_rmse,
        tabfm_display,
        posthoc.tabfm_ensemble_rmse,
        posthoc.stronger_tabfm_algorithm,
        posthoc.stronger_tabfm_rmse,
        posthoc.beats_prior_aspfm,
        posthoc.beats_tabpfn_v3,
        posthoc.beats_stronger_tabfm,
        posthoc.beats_both,
        posthoc.final_cv_rerank_score.semantic_sha256,
        posthoc.final_cv_rerank_score.rmse,
        posthoc.cv_rerank_beats_prior_aspfm,
        posthoc.cv_rerank_beats_tabpfn_v3,
        posthoc.cv_rerank_beats_stronger_tabfm,
        posthoc.cv_rerank_beats_both,
    );
    immutable_write(&report_root.join("report.md"), markdown.as_bytes())?;
    Ok((
        DatasetReport {
            dataset_hash: dataset.dataset_hash.clone(),
            generation_zero_validation_fitness: validation[0],
            generation_nine_validation_fitness: validation[final_generation],
            validation_fitness_change: validation[final_generation] - validation[0],
            generation_zero_rmse: test_rmse[0],
            generation_nine_rmse: test_rmse[final_generation],
            relative_rmse_improvement: relative,
            prior_aspfm_rmse: posthoc.prior_aspfm_rmse,
            tabpfn_v3_rmse: posthoc.tabpfn_v3_rmse,
            tabfm_rmse: posthoc.tabfm_rmse,
            tabfm_ensemble_rmse: posthoc.tabfm_ensemble_rmse,
            stronger_tabfm_algorithm: posthoc.stronger_tabfm_algorithm.clone(),
            stronger_tabfm_rmse: posthoc.stronger_tabfm_rmse,
            best_baseline_algorithm,
            best_baseline_rmse,
            generation_nine_relative_gap_to_best_baseline,
            cv_rerank_relative_gap_to_best_baseline,
            beats_tabpfn_v3: posthoc.beats_tabpfn_v3,
            beats_stronger_tabfm: posthoc.beats_stronger_tabfm,
            beats_both: posthoc.beats_both,
            beats_prior_aspfm: posthoc.beats_prior_aspfm,
            cv_rerank_semantic_sha256: posthoc.final_cv_rerank_score.semantic_sha256.clone(),
            cv_rerank_rmse: posthoc.final_cv_rerank_score.rmse,
            cv_rerank_beats_tabpfn_v3: posthoc.cv_rerank_beats_tabpfn_v3,
            cv_rerank_beats_stronger_tabfm: posthoc.cv_rerank_beats_stronger_tabfm,
            cv_rerank_beats_both: posthoc.cv_rerank_beats_both,
            cv_rerank_beats_prior_aspfm: posthoc.cv_rerank_beats_prior_aspfm,
            final_ancestry: ancestry.to_string(),
        },
        validation,
    ))
}

pub(crate) fn report(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    let (root, manifest, _, _, validation_digest) =
        require_validated_score_gate(study_root, run_id, sotarena_root, workers)?;
    let (index, index_digest): (TrainOnlyPosthocIndex, _) =
        read_receipt(&root.join("posthoc-scores.v2.json"))?;
    let (baselines, baseline_digest): (BaselineManifest, _) =
        read_receipt(&root.join("baselines.v2.json"))?;
    if index.schema != "TrainOnlyPosthocIndex.v2"
        || index.run_id != run_id
        || index.validation_receipt_sha256 != validation_digest
        || index.baseline_manifest_sha256 != baseline_digest
        || index.dataset_ids != manifest.selected_dataset_ids
        || index.posthoc_score_sha256.len() != manifest.datasets.len()
        || baselines.entries.len() != manifest.datasets.len()
    {
        bail!("post-hoc index does not cover the validated cohort");
    }
    let expected_index_cumulative = receipt_digest(&json!({
        "validation_receipt_sha256": index.validation_receipt_sha256,
        "baseline_manifest_sha256": index.baseline_manifest_sha256,
        "posthoc_score_sha256": index.posthoc_score_sha256,
    }))?;
    if expected_index_cumulative != index.cumulative_sha256 {
        bail!("post-hoc index cumulative hash drift");
    }

    let mut dataset_reports = Vec::with_capacity(manifest.datasets.len());
    let mut validations = Vec::with_capacity(manifest.datasets.len());
    let mut tests = Vec::with_capacity(manifest.datasets.len());
    for dataset in &manifest.datasets {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("posthoc-score.v2.json");
        let (posthoc, digest): (TrainOnlyPosthocScore, _) = read_receipt(&path)?;
        let (cv_rerank, cv_rerank_digest): (TrainOnlyFinalCvRerank, _) = read_receipt(
            &root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("final-top10-cv-rerank.v2.json"),
        )?;
        if index.posthoc_score_sha256.get(&dataset.dataset_hash) != Some(&digest)
            || posthoc.schema != "TrainOnlyPosthocScore.v2"
            || posthoc.run_id != run_id
            || posthoc.dataset_hash != dataset.dataset_hash
            || posthoc.validation_receipt_sha256 != validation_digest
            || posthoc.baseline_manifest_sha256 != baseline_digest
            || posthoc.scores.len() != manifest.generations
            || posthoc.final_cv_rerank_sha256 != cv_rerank_digest
            || cv_rerank.schema != "TrainOnlyFinalCvRerank.v2"
            || cv_rerank.dataset_hash != dataset.dataset_hash
            || cv_rerank.validation_receipt_sha256 != validation_digest
            || cv_rerank.selected_semantic_sha256 != posthoc.final_cv_rerank_score.semantic_sha256
            || !cv_rerank.selection_frozen_before_test_access
            || !posthoc.test_trajectory_post_selection
        {
            bail!("post-hoc dataset receipt drift");
        }
        let (dataset_report, validation) = report_dataset(&root, dataset, &posthoc)?;
        validations.push(validation);
        tests.push(
            posthoc
                .scores
                .iter()
                .map(|score| score.rmse)
                .collect::<Vec<_>>(),
        );
        dataset_reports.push(dataset_report);
    }
    let relative = dataset_reports
        .iter()
        .map(|dataset| dataset.relative_rmse_improvement)
        .collect::<Vec<_>>();
    let wins_vs_tabpfn_v3 = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_tabpfn_v3)
        .count();
    let wins_vs_stronger_tabfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_stronger_tabfm)
        .count();
    let wins_vs_both = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_both)
        .count();
    let wins_vs_prior_aspfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_prior_aspfm)
        .count();
    let cv_rerank_wins_vs_tabpfn_v3 = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_tabpfn_v3)
        .count();
    let cv_rerank_wins_vs_stronger_tabfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_stronger_tabfm)
        .count();
    let cv_rerank_wins_vs_both = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_both)
        .count();
    let cv_rerank_wins_vs_prior_aspfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_prior_aspfm)
        .count();
    let status = if wins_vs_both >= 5 {
        "checkpoint_assisted_claim_supported"
    } else {
        "checkpoint_assisted_claim_not_supported"
    };
    let elo_ratings = cohort_elo(&dataset_reports)?;
    let receipt = TrainOnlyStudyReport {
        schema: "TrainOnlyStudyReport.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure: manifest.checkpoint_overlap_disclosure.clone(),
        run_id: run_id.to_string(),
        validation_receipt_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        posthoc_score_sha256: index.posthoc_score_sha256.clone(),
        status: status.to_string(),
        wins_vs_tabpfn_v3,
        wins_vs_stronger_tabfm,
        wins_vs_both,
        wins_vs_prior_aspfm,
        cv_rerank_wins_vs_tabpfn_v3,
        cv_rerank_wins_vs_stronger_tabfm,
        cv_rerank_wins_vs_both,
        cv_rerank_wins_vs_prior_aspfm,
        median_relative_rmse_improvement: median(&relative)?,
        bootstrap_interval_95: bootstrap_interval(&relative)?,
        paired_sign_test_p_value: paired_sign_test(&relative),
        elo_methodology: "Five systems start at 1500 with K=32. Each frozen dataset is one simultaneous round-robin round; lower RMSE wins, exact equality ties, and all pair deltas use pre-round ratings. Dataset order is the frozen manifest order. The comparison set is our generation-9 champion, our train-only top-10/five-fold-CV choice, prior ASPFM (`jope-dime`), TabFM ensemble, and TabPFN v3.".to_string(),
        elo_ratings,
        test_trajectories_post_selection: true,
        datasets: dataset_reports,
    };
    let report_root = root.join("reports");
    fs::create_dir_all(&report_root)?;
    let report_path = report_root.join("study-report.v2.json");
    let report_digest = write_receipt(&report_path, &receipt)?;

    let cohort_validation = (0..manifest.generations)
        .map(|generation| {
            median(
                &validations
                    .iter()
                    .map(|values| values[generation])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let cohort_test = (0..manifest.generations)
        .map(|generation| {
            median(
                &tests
                    .iter()
                    .map(|values| values[generation])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let median_tabpfn = median(
        &receipt
            .datasets
            .iter()
            .map(|dataset| dataset.tabpfn_v3_rmse)
            .collect::<Vec<_>>(),
    )?;
    let median_tabfm = median(
        &receipt
            .datasets
            .iter()
            .filter_map(|dataset| dataset.tabfm_rmse)
            .collect::<Vec<_>>(),
    )?;
    let median_tabfm_ensemble = median(
        &receipt
            .datasets
            .iter()
            .map(|dataset| dataset.tabfm_ensemble_rmse)
            .collect::<Vec<_>>(),
    )?;
    immutable_write(
        &report_root.join("cohort-evolution.svg"),
        evolution_svg(
            "Golden42 checkpoint-assisted cohort medians",
            &cohort_validation,
            &cohort_test,
            median_tabpfn,
            Some(median_tabfm),
            median_tabfm_ensemble,
        )
        .as_bytes(),
    )?;
    let mut table = String::from(
        "| Dataset | Validation fitness 0→9 | Gen 0 RMSE | Gen 9 RMSE | Top-10 CV RMSE | Best baseline | Gen 9 gap | CV gap | Gen 9 win | CV win | Ancestry |\n|---|---:|---:|---:|---:|---|---:|---:|:---:|:---:|---|\n",
    );
    let mut baseline_table = String::from(
        "| Dataset | Prior ASPFM | TabPFN v3 | TabFM | TabFM ensemble | Stronger TabFM |\n|---|---:|---:|---:|---:|---|\n",
    );
    for dataset in &receipt.datasets {
        let tabfm_display = dataset
            .tabfm_rmse
            .map_or_else(|| "unavailable".to_string(), |value| format!("{value:.8}"));
        writeln!(
            table,
            "| {} | {:.4}→{:.4} ({:+.4}) | {:.8} | {:.8} | {:.8} | {} ({:.8}) | {:+.2}% | {:+.2}% | {} | {} | {} |",
            dataset.dataset_hash,
            dataset.generation_zero_validation_fitness,
            dataset.generation_nine_validation_fitness,
            dataset.validation_fitness_change,
            dataset.generation_zero_rmse,
            dataset.generation_nine_rmse,
            dataset.cv_rerank_rmse,
            dataset.best_baseline_algorithm,
            dataset.best_baseline_rmse,
            dataset.generation_nine_relative_gap_to_best_baseline * 100.0,
            dataset.cv_rerank_relative_gap_to_best_baseline * 100.0,
            dataset.beats_both,
            dataset.cv_rerank_beats_both,
            dataset.final_ancestry,
        )?;
        writeln!(
            baseline_table,
            "| {} | {:.8} | {:.8} | {} | {:.8} | {} ({:.8}) |",
            dataset.dataset_hash,
            dataset.prior_aspfm_rmse,
            dataset.tabpfn_v3_rmse,
            tabfm_display,
            dataset.tabfm_ensemble_rmse,
            dataset.stronger_tabfm_algorithm,
            dataset.stronger_tabfm_rmse,
        )?;
    }
    let mut elo_table =
        String::from("| Rank | System | Elo | W | L | T |\n|---:|---|---:|---:|---:|---:|\n");
    for (rank, rating) in receipt.elo_ratings.iter().enumerate() {
        writeln!(
            elo_table,
            "| {} | {} | {:.1} | {} | {} | {} |",
            rank + 1,
            rating.system,
            rating.rating,
            rating.wins,
            rating.losses,
            rating.ties,
        )?;
    }
    let elo_leader = receipt.elo_ratings.first().context("Elo table is empty")?;
    let gen9_elo_rank = receipt
        .elo_ratings
        .iter()
        .position(|rating| rating.system == "ours_gen9")
        .context("generation-nine system is absent from Elo")?
        + 1;
    let cv_elo_rank = receipt
        .elo_ratings
        .iter()
        .position(|rating| rating.system == "ours_top10_5fold_cv")
        .context("CV-rerank system is absent from Elo")?
        + 1;
    let markdown = format!(
        "# Checkpoint-Assisted 10-Dataset Regression Evolution Study\n\n\
Status: **{}**.\n\n\
The generation-9 champions won strictly against prior ASPFM on {}/10 datasets, TabPFN v3 on {}/10, the per-dataset stronger TabFM variant on {}/10, and both fixed baselines on {}/10. The fixed success threshold was at least 5/10 wins against both fixed baselines.\n\n\
The secondary final-top-10/five-fold-CV strategy won strictly against prior ASPFM on {}/10 datasets, TabPFN v3 on {}/10, the stronger TabFM variant on {}/10, and both fixed baselines on {}/10. This secondary result does not replace the fixed primary criterion.\n\n\
Median generation-0-to-9 relative RMSE improvement: {:+.2}% (deterministic 95% bootstrap interval {:+.2}% to {:+.2}%; paired sign-test p={:.6}).\n\n\
This experiment is exploratory: the accepted checkpoint has seen Golden42 identities. `zero_shot_valid` is `false`, and this report makes no zero-shot generalization claim.\n\n\
All selection receipts were sealed and independently validated before any test path was opened or baseline metric was used. The top-10 CV rerank was then computed from training data and sealed before those paths or metrics were opened. Test trajectories are post-hoc and never influenced either selection strategy.\n\n\
## Dataset outcomes\n\n\
{}\n\n\
Negative gaps mean our RMSE is lower than the best baseline for that dataset.\n\n\
## Baseline detail\n\n\
{}\n\n\
## Cohort Elo\n\n\
{}\n\n\
The descriptive five-system Elo leader is `{}` at `{:.1}`. Our generation-9 system ranks `{}` and our top-10/five-fold-CV system ranks `{}` on this frozen ten-dataset cohort. This is a within-cohort pace check; its numeric ratings are not directly comparable to SOTArena's official 42-dataset Elo scale.\n\n\
{}\n\n\
Report receipt: `{}`. Post-hoc index receipt: `{}`.\n",
        receipt.status,
        receipt.wins_vs_prior_aspfm,
        receipt.wins_vs_tabpfn_v3,
        receipt.wins_vs_stronger_tabfm,
        receipt.wins_vs_both,
        receipt.cv_rerank_wins_vs_prior_aspfm,
        receipt.cv_rerank_wins_vs_tabpfn_v3,
        receipt.cv_rerank_wins_vs_stronger_tabfm,
        receipt.cv_rerank_wins_vs_both,
        receipt.median_relative_rmse_improvement * 100.0,
        receipt.bootstrap_interval_95[0] * 100.0,
        receipt.bootstrap_interval_95[1] * 100.0,
        receipt.paired_sign_test_p_value,
        table,
        baseline_table,
        elo_table,
        elo_leader.system,
        elo_leader.rating,
        gen9_elo_rank,
        cv_elo_rank,
        receipt.elo_methodology,
        report_digest,
        index_digest,
    );
    immutable_write(&report_root.join("study-report.md"), markdown.as_bytes())?;
    Ok(report_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn canonical(source: &str) -> String {
        GenomeIr::parse(source).unwrap().canonical()
    }

    fn member(slot: usize, source: &str, complexity: usize) -> GenomeMember {
        let canonical_genome = canonical(source);
        GenomeMember {
            slot,
            semantic_sha256: semantic_identity(&canonical_genome),
            canonical_genome,
            complexity,
            source: "fixture".to_string(),
            source_evidence: Value::Null,
            ancestry: Ancestry {
                operation: "fixture".to_string(),
                parents: Vec::new(),
                roots: BTreeSet::from(["dl".to_string()]),
                retry_count: 0,
            },
        }
    }

    fn scored(member: GenomeMember, fitness: f64, worst: f64) -> ScoredMember {
        ScoredMember {
            member,
            observations: Vec::new(),
            split_r2: vec![fitness],
            robust_fitness: Some(fitness),
            worst_split_r2: Some(worst),
            failure_reason: None,
        }
    }

    #[test]
    fn target_decile_splits_are_exact_distinct_and_deterministic() {
        let targets = (0..103)
            .map(|index| (index as f64).sin() + index as f64 / 10.0)
            .collect::<Vec<_>>();
        let first = build_splits("fixture", "dataset", "train", &targets, GENERATIONS).unwrap();
        let second = build_splits("fixture", "dataset", "train", &targets, GENERATIONS).unwrap();
        assert_eq!(
            receipt_digest(&first).unwrap(),
            receipt_digest(&second).unwrap()
        );
        assert_eq!(first.splits.len(), GENERATIONS);
        let mut hashes = HashSet::new();
        let mut seeds = HashSet::new();
        for split in first.splits {
            assert_eq!(split.validation_indices.len(), targets.len().div_ceil(5));
            assert_eq!(
                split.train_indices.len() + split.validation_indices.len(),
                targets.len()
            );
            let train = split.train_indices.into_iter().collect::<HashSet<_>>();
            assert!(split
                .validation_indices
                .iter()
                .all(|index| !train.contains(index)));
            assert!(hashes.insert(split.split_sha256));
            assert!(seeds.insert(split.seed));
        }
    }

    #[test]
    fn final_top10_cv_folds_are_deterministic_disjoint_and_exhaustive() {
        let targets = (0..103)
            .map(|index| (index as f64).cos() + index as f64 / 7.0)
            .collect::<Vec<_>>();
        let first = build_final_cv_splits("fixture", "dataset", "train", &targets).unwrap();
        let second = build_final_cv_splits("fixture", "dataset", "train", &targets).unwrap();
        assert_eq!(
            receipt_digest(&first).unwrap(),
            receipt_digest(&second).unwrap()
        );
        assert_eq!(first.splits.len(), FINAL_CV_FOLDS);
        let mut validation_rows = HashSet::new();
        let mut hashes = HashSet::new();
        let mut seeds = HashSet::new();
        for split in first.splits {
            assert_eq!(
                split.train_indices.len() + split.validation_indices.len(),
                targets.len()
            );
            let train = split.train_indices.into_iter().collect::<HashSet<_>>();
            for row in split.validation_indices {
                assert!(!train.contains(&row));
                assert!(validation_rows.insert(row));
            }
            assert!(hashes.insert(split.split_sha256));
            assert!(seeds.insert(split.seed));
        }
        assert_eq!(validation_rows.len(), targets.len());
    }

    #[test]
    #[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
    fn production_sotarena_registry_freezes_exact_37_to_requested_10() {
        let qualified_root = std::env::var_os("JOPE_TEST_SOTARENA_ROOT").map(PathBuf::from);
        let sotarena_root = qualified_root
            .as_deref()
            .unwrap_or_else(|| Path::new(default_sotarena_root_str()));
        assert!(
            Path::new(default_catalog_pointer_str()).is_file(),
            "unavailable: production context catalog pointer {} (provision <JOPE_STATE>/legacy/test_exact_gen0)",
            default_catalog_pointer_str()
        );
        let (_, catalog) = load_catalog(Path::new(default_catalog_pointer_str()))
            .expect("production context catalog");
        let disclosure = CorpusExclusions {
            schema: "fixture".to_string(),
            run_id: "fixture".to_string(),
            checkpoint_sha256: "a".repeat(64),
            training_corpus_train_sha256: "b".repeat(64),
            training_corpus_validation_sha256: "c".repeat(64),
            dataset_ids: Vec::new(),
            source_families: Vec::new(),
            target_identities: Vec::new(),
        };
        let (eligibility, selected) = build_eligibility(
            "fixture",
            sotarena_root,
            &catalog,
            &disclosure,
            GENERATIONS,
            FROZEN_DATASETS,
            Some(EXPECTED_ELIGIBLE),
        )
        .expect("production Golden42 eligibility");
        assert_eq!(
            eligibility.iter().filter(|row| row.eligible).count(),
            EXPECTED_ELIGIBLE
        );
        assert_eq!(
            selected
                .iter()
                .map(|(entry, _, _, _)| entry.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            FROZEN_DATASETS
        );
        assert!(selected.iter().all(|(_, _, _, reports)| {
            reports.tabpfn_v3_path.is_file()
                && reports.tabfm_path.is_file()
                && reports.tabfm_ensemble_path.is_file()
        }));
    }

    #[test]
    fn paired_fitness_clips_and_uses_population_sd() {
        let scores = [-2.0, 0.0, 1.0];
        let clipped = [-1.0_f64, 0.0, 1.0];
        let mean = 0.0;
        let sd = (clipped
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 3.0)
            .sqrt();
        assert_eq!(paired_robust_fitness(&scores).unwrap(), mean - 0.25 * sd);
        assert!(paired_robust_fitness(&[]).is_err());
    }

    #[test]
    fn ranking_ties_use_worst_complexity_then_semantic_hash() {
        let a = scored(member(0, "(pipe X (ridge l2=0.1))", 4), 0.5, 0.1);
        let b = scored(member(1, "(pipe X (ridge l2=0.2))", 3), 0.5, 0.2);
        let c = scored(member(2, "(pipe X (ridge l2=0.3))", 2), 0.5, 0.2);
        let mut values = vec![a, b, c.clone()];
        values.sort_by(rank_scored);
        assert_eq!(values[0].member.semantic_sha256, c.member.semantic_sha256);

        let mut same = vec![
            scored(member(0, "(pipe X (ridge l2=0.4))", 2), 0.5, 0.2),
            scored(member(1, "(pipe X (ridge l2=0.5))", 2), 0.5, 0.2),
        ];
        same.sort_by(rank_scored);
        assert!(same[0].member.semantic_sha256 < same[1].member.semantic_sha256);
    }

    #[test]
    fn cache_identity_includes_split_and_kernel() {
        let first = cache_key("train", "semantic", "split-a").unwrap();
        let second = cache_key("train", "semantic", "split-b").unwrap();
        assert_ne!(first, second);
        assert_eq!(first, cache_key("train", "semantic", "split-a").unwrap());
    }

    #[test]
    fn split_and_stage_caches_are_metric_bit_identical_to_plain_fit() {
        let data = RegressionData {
            x: (0..100)
                .map(|index| vec![index as f64 / 10.0, (index as f64).sin()])
                .collect(),
            y: (0..100)
                .map(|index| 0.5 * index as f64 + (index as f64).cos())
                .collect(),
            features: 2,
        };
        let splits = build_splits("fixture", "dataset", "train", &data.y, GENERATIONS).unwrap();
        let dataset = DatasetStudyEntry {
            dataset_hash: "dataset".to_string(),
            source_family: "fixture".to_string(),
            target_identity: "fixture".to_string(),
            train_path: PathBuf::new(),
            train_sha256: "a".repeat(64),
            rows: 100,
            features: 2,
            encoding: EncodingSlice {
                blob_path: PathBuf::new(),
                blob_sha256: "b".repeat(64),
                offset_bytes: 0,
                length_bytes: CONTEXT_BYTES,
                context_sha256: "c".repeat(64),
            },
            split_manifest_path: PathBuf::new(),
            split_manifest_sha256: "d".repeat(64),
            generation_zero_path: PathBuf::new(),
            generation_zero_sha256: "e".repeat(64),
            transfer_ranking_path: PathBuf::new(),
            transfer_ranking_sha256: "f".repeat(64),
            baseline_reports: SotarenaReportHashes {
                tabpfn_v3_path: PathBuf::new(),
                tabpfn_v3_sha256: "1".repeat(64),
                tabfm_path: PathBuf::new(),
                tabfm_sha256: "2".repeat(64),
                tabfm_ensemble_path: PathBuf::new(),
                tabfm_ensemble_sha256: "3".repeat(64),
            },
        };
        let evaluator = StudyEvalContext::new(&data, &dataset, &splits).unwrap();
        evaluator.begin_generation(0);
        let genome = GenomeIr::parse("(pipe X (standardize append=true) (ridge l2=0.25))").unwrap();
        let cached_first = evaluator.evaluate(&genome, &splits.splits[0]).unwrap();
        let cached_second = evaluator.evaluate(&genome, &splits.splits[0]).unwrap();
        let (train_x, train_y) = slice_data(&data, &splits.splits[0].train_indices).unwrap();
        let (validation_x, validation_y) =
            slice_data(&data, &splits.splits[0].validation_indices).unwrap();
        let predictions = fit(&genome, &train_x, &train_y)
            .unwrap()
            .predict(&validation_x);
        let plain = metrics(&validation_y, &predictions).unwrap();
        assert_eq!((cached_first.0, cached_first.1, cached_first.2), plain);
        assert_eq!((cached_second.0, cached_second.1, cached_second.2), plain);
        assert!(evaluator.stage_cache.stats().0 > 0);

        let sources = [
            "(pipe X (standardize append=true) (ridge l2=0.25))",
            "(pipe X (standardize append=true) (ridge l2=0.75))",
            "(pipe X (ridge l2=0.5))",
        ];
        let mut cached_rank = Vec::new();
        let mut uncached_rank = Vec::new();
        for (slot, source) in sources.iter().enumerate() {
            let genome = GenomeIr::parse(source).unwrap();
            let cached_r2 = splits.splits[..2]
                .iter()
                .map(|split| evaluator.evaluate(&genome, split).unwrap().1)
                .collect::<Vec<_>>();
            let uncached_r2 = splits.splits[..2]
                .iter()
                .map(|split| {
                    let (train_x, train_y) = slice_data(&data, &split.train_indices).unwrap();
                    let (validation_x, validation_y) =
                        slice_data(&data, &split.validation_indices).unwrap();
                    let predictions = fit(&genome, &train_x, &train_y)
                        .unwrap()
                        .predict(&validation_x);
                    metrics(&validation_y, &predictions).unwrap().1
                })
                .collect::<Vec<_>>();
            assert_eq!(cached_r2, uncached_r2);
            let make_scored = |scores: Vec<f64>| ScoredMember {
                member: member(slot, source, genome_complexity(&genome)),
                observations: Vec::new(),
                robust_fitness: Some(paired_robust_fitness(&scores).unwrap()),
                worst_split_r2: scores.iter().copied().min_by(f64::total_cmp),
                split_r2: scores,
                failure_reason: None,
            };
            cached_rank.push(make_scored(cached_r2));
            uncached_rank.push(make_scored(uncached_r2));
        }
        cached_rank.sort_by(rank_scored);
        uncached_rank.sort_by(rank_scored);
        assert_eq!(
            cached_rank
                .iter()
                .map(|row| &row.member.semantic_sha256)
                .collect::<Vec<_>>(),
            uncached_rank
                .iter()
                .map(|row| &row.member.semantic_sha256)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sealed_receipts_detect_all_required_tamper_classes() {
        let directory = tempdir().unwrap();
        for artifact in [
            "checkpoint",
            "proposal",
            "transfer-seed",
            "split",
            "observation",
            "generation-checkpoint",
            "baseline-report",
        ] {
            let path = directory.path().join(format!("{artifact}.json"));
            write_receipt(
                &path,
                &json!({"schema": "Fixture.v2", "artifact": artifact, "value": 1}),
            )
            .unwrap();
            let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            value["value"] = json!(2);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(
                read_receipt::<Value>(&path).is_err(),
                "{artifact} tampering was not rejected"
            );
        }
    }

    #[test]
    fn checkpoint_overlap_is_disclosed_without_zero_shot_rejection() {
        let directory = tempdir().unwrap();
        let registry = directory.path().join("registry");
        let benchmark_id = GOLDEN42[0];
        let benchmark_root = registry.join(benchmark_id);
        fs::create_dir_all(&benchmark_root).unwrap();
        let meta = json!({
            "source_family": "same-family",
            "source_platform": "fixture",
            "source_dataset_id": "same-source",
            "source_dataset_name": "same-name",
            "source_target_column_original": "y",
            "target_column": "target",
            "target_names": ["target"],
        });
        fs::write(
            benchmark_root.join("meta.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
        let train = (0..100)
            .map(|index| format!("{},{}\n", index as f64, index as f64 / 10.0))
            .collect::<String>();
        fs::write(benchmark_root.join("train.csv"), train).unwrap();
        let train_hash = sha256_file(&benchmark_root.join("train.csv")).unwrap();
        let context_hash = "a".repeat(64);
        let catalog = Catalog {
            dataset_dimensions: 1536,
            target_dimensions: 512,
            record_stride_bytes: CONTEXT_BYTES,
            entries: vec![CatalogEntry {
                dataset_hash: benchmark_id.to_string(),
                task: "regression".to_string(),
                train_csv: benchmark_root.join("train.csv"),
                train_sha256: train_hash,
                source_family: "same-family".to_string(),
                context_sha256: context_hash,
                catalog_index: 0,
            }],
        };
        let other_id = "checkpoint-source";
        let other_root = registry.join(other_id);
        fs::create_dir_all(&other_root).unwrap();
        fs::write(
            other_root.join("meta.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
        let corpus_train = directory.path().join("train.jsonl");
        let corpus_validation = directory.path().join("validation.jsonl");
        fs::write(
            &corpus_train,
            format!("{{\"dataset_id\":\"{other_id}\"}}\n{{\"dataset_id\":\"{benchmark_id}\"}}\n"),
        )
        .unwrap();
        fs::write(&corpus_validation, "").unwrap();
        let checkpoint = CheckpointProvenance {
            accepted_pointer_path: PathBuf::new(),
            accepted_pointer_sha256: String::new(),
            checkpoint_path: PathBuf::new(),
            checkpoint_sha256: "b".repeat(64),
            model_version: "fixture".to_string(),
            training_corpus_manifest_path: PathBuf::new(),
            training_corpus_manifest_sha256: String::new(),
            training_corpus_train_path: corpus_train,
            training_corpus_train_sha256: "c".repeat(64),
            training_corpus_validation_path: corpus_validation,
            training_corpus_validation_sha256: "d".repeat(64),
        };
        let exclusions =
            checkpoint_exclusions("fixture", &registry, &checkpoint, &catalog).unwrap();
        assert!(exclusions
            .source_families
            .contains(&"same-family".to_string()));
        assert!(exclusions
            .target_identities
            .contains(&target_identity(&registry, benchmark_id).unwrap()));
        let (records, selected) = build_eligibility(
            "fixture",
            &registry,
            &catalog,
            &exclusions,
            GENERATIONS,
            &[],
            None,
        )
        .unwrap();
        let record = records
            .iter()
            .find(|record| record.dataset_hash == benchmark_id)
            .unwrap();
        assert!(record.checkpoint_source_family_overlap);
        assert!(record.checkpoint_exact_overlap);
        assert!(record.checkpoint_target_identity_overlap);
        assert!(!record
            .reasons
            .iter()
            .any(|reason| reason.starts_with("checkpoint_corpus_")));
        assert!(selected.is_empty());
    }

    fn proposal_row(dataset: &str, checkpoint: &str, context: &str, index: usize) -> ProposalRow {
        let canonical_genome = canonical(&format!("(pipe X (ridge l2={}))", index as f64 + 0.125));
        ProposalRow {
            task: "regression".to_string(),
            dataset_hash: dataset.to_string(),
            semantic_sha256: semantic_identity(&canonical_genome),
            canonical_genome,
            model_checkpoint_sha256: checkpoint.to_string(),
            context_sha256: Some(context.to_string()),
            proposal_score: Some(100.0 - index as f64),
            selection_score: None,
            reward_lcb: Some(100.0 - index as f64),
            instability_probability: Some(0.0),
            slot: index,
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn generation_zero_preserves_quotas_deduplicates_and_fails_closed() {
        let directory = tempdir().unwrap();
        let proposals_path = directory.path().join("proposals.jsonl");
        let dataset_hash = "fixture";
        let checkpoint_hash = "1".repeat(64);
        let context_hash = "2".repeat(64);
        let proposals = (0..64)
            .map(|index| proposal_row(dataset_hash, &checkpoint_hash, &context_hash, index))
            .collect::<Vec<_>>();
        let proposal_bytes = proposals
            .iter()
            .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
            .collect::<String>();
        fs::write(&proposals_path, proposal_bytes).unwrap();
        let proposal_hash = sha256_file(&proposals_path).unwrap();
        let mut global = Vec::new();
        global.push(GlobalRankedGenome {
            rank: 0,
            canonical_genome: proposals[0].canonical_genome.clone(),
            semantic_sha256: proposals[0].semantic_sha256.clone(),
            mean_percentile: 1.0,
            worst_percentile: 1.0,
            strict_win_count: 1,
            dataset_count: 1,
            source_dataset_ids: vec!["source".to_string()],
            source_families: vec!["source-family".to_string()],
        });
        for index in 0..30 {
            let canonical_genome = canonical(&format!(
                "(pipe X (elasticnet alpha={}))",
                index as f64 + 0.75
            ));
            global.push(GlobalRankedGenome {
                rank: index + 1,
                semantic_sha256: semantic_identity(&canonical_genome),
                canonical_genome,
                mean_percentile: 1.0 - index as f64 / 100.0,
                worst_percentile: 0.5,
                strict_win_count: 0,
                dataset_count: 1,
                source_dataset_ids: vec!["source".to_string()],
                source_families: vec!["source-family".to_string()],
            });
        }
        let ranking = GlobalRankingReceipt {
            schema: "TrainOnlyTransferGenomeRanking.v2".to_string(),
            run_id: "fixture".to_string(),
            target_dataset_id: dataset_hash.to_string(),
            target_source_family: "fixture".to_string(),
            observation_db_sha256: "3".repeat(64),
            source_catalog_sha256: "4".repeat(64),
            excluded_dataset_ids: Vec::new(),
            excluded_source_family: "fixture".to_string(),
            direct_dl_sources_excluded: true,
            ranking_policy: "fixture".to_string(),
            genomes: global,
        };
        let entry = CatalogEntry {
            dataset_hash: dataset_hash.to_string(),
            task: "regression".to_string(),
            train_csv: PathBuf::new(),
            train_sha256: "5".repeat(64),
            source_family: "fixture".to_string(),
            context_sha256: context_hash.clone(),
            catalog_index: 0,
        };
        let data = RegressionData {
            x: vec![vec![0.0]; 100],
            y: (0..100).map(|value| value as f64).collect(),
            features: 1,
        };
        let encoding = EncodingSlice {
            blob_path: PathBuf::new(),
            blob_sha256: "6".repeat(64),
            offset_bytes: 0,
            length_bytes: CONTEXT_BYTES,
            context_sha256: context_hash,
        };
        let checkpoint = CheckpointProvenance {
            accepted_pointer_path: PathBuf::new(),
            accepted_pointer_sha256: String::new(),
            checkpoint_path: PathBuf::new(),
            checkpoint_sha256: checkpoint_hash,
            model_version: "fixture".to_string(),
            training_corpus_manifest_path: PathBuf::new(),
            training_corpus_manifest_sha256: String::new(),
            training_corpus_train_path: PathBuf::new(),
            training_corpus_train_sha256: String::new(),
            training_corpus_validation_path: PathBuf::new(),
            training_corpus_validation_sha256: String::new(),
        };
        let generation = generation_zero(
            "fixture",
            &entry,
            &data,
            &encoding,
            &checkpoint,
            &proposals_path,
            &proposal_hash,
            &ranking,
            "7",
        )
        .unwrap();
        assert_eq!(generation.members.len(), POPULATION);
        assert!(generation.members[..DL_QUOTA]
            .iter()
            .all(|member| member.source == "latest_checkpoint_dl"));
        assert!(generation.members[DL_QUOTA..]
            .iter()
            .all(|member| member.source == "target_clean_cross_dataset_transfer"));
        assert_eq!(
            generation
                .members
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<HashSet<_>>()
                .len(),
            POPULATION
        );

        let mut underfilled = ranking;
        underfilled.genomes.truncate(30);
        assert!(generation_zero(
            "fixture",
            &entry,
            &data,
            &encoding,
            &checkpoint,
            &proposals_path,
            &proposal_hash,
            &underfilled,
            "7",
        )
        .is_err());
    }

    #[test]
    fn transfer_ranking_excludes_target_family_and_direct_dl_sources() {
        let directory = tempdir().unwrap();
        let database = directory.path().join("observations.sqlite");
        let connection = Connection::open(&database).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE observations(
                    task TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    canonical_genome TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    score REAL NOT NULL,
                    origin_class TEXT NOT NULL,
                    PRIMARY KEY(task,dataset_hash,semantic_sha256)
                ) WITHOUT ROWID;",
            )
            .unwrap();
        let entries = ["target", "same-family-source", "clean-source"]
            .into_iter()
            .enumerate()
            .map(|(index, dataset)| CatalogEntry {
                dataset_hash: dataset.to_string(),
                task: "regression".to_string(),
                train_csv: PathBuf::new(),
                train_sha256: "a".repeat(64),
                source_family: if index < 2 { "family-a" } else { "family-b" }.to_string(),
                context_sha256: "b".repeat(64),
                catalog_index: index,
            })
            .collect::<Vec<_>>();
        for index in 0..35 {
            let genome = canonical(&format!(
                "(pipe X (elasticnet alpha={} l1_ratio=0.5))",
                index as f64 + 0.75
            ));
            for dataset in ["target", "same-family-source", "clean-source"] {
                connection
                    .execute(
                        "INSERT INTO observations VALUES('regression',?1,?2,?3,'transfer',?4,'gp_origin')",
                        rusqlite::params![dataset, semantic_identity(&genome), genome, index as f64],
                    )
                    .unwrap();
            }
        }
        let dl_genome = canonical("(pipe X (ridge l2=99))");
        connection
            .execute(
                "INSERT INTO observations VALUES('regression','clean-source',?1,?2,'checkpoint_dl',999.0,'dl_origin')",
                rusqlite::params![semantic_identity(&dl_genome), dl_genome],
            )
            .unwrap();
        drop(connection);
        let catalog = Catalog {
            dataset_dimensions: 1536,
            target_dimensions: 512,
            record_stride_bytes: CONTEXT_BYTES,
            entries: entries.clone(),
        };
        let rankings = build_transfer_rankings(
            "fixture",
            &database,
            &sha256_file(&database).unwrap(),
            &catalog,
            &"c".repeat(64),
            &entries[..1],
        )
        .unwrap();
        let ranking = &rankings["target"];
        assert!(ranking.genomes.len() >= GLOBAL_QUOTA);
        assert!(ranking.genomes.iter().all(|genome| {
            genome.source_dataset_ids == vec!["clean-source".to_string()]
                && genome.source_families == vec!["family-b".to_string()]
                && genome.semantic_sha256 != semantic_identity(&dl_genome)
        }));
    }

    #[test]
    fn refill_is_deterministic_preserves_three_elites_and_history_uniqueness() {
        let mut ranked = (0..POPULATION)
            .map(|index| {
                scored(
                    member(
                        index,
                        &format!("(pipe X (ridge l2={}))", index as f64 + 0.25),
                        1,
                    ),
                    1.0 - index as f64 / 100.0,
                    0.5 - index as f64 / 1000.0,
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(rank_scored);
        let initial = ranked
            .iter()
            .map(|member| member.member.semantic_sha256.clone())
            .collect::<HashSet<_>>();
        let mut first_history = initial.clone();
        let mut second_history = initial;
        let first =
            refill_population("fixture", "dataset", 4, 1, &ranked, &mut first_history).unwrap();
        let second =
            refill_population("fixture", "dataset", 4, 1, &ranked, &mut second_history).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), POPULATION);
        assert_eq!(
            first[..ELITES]
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<Vec<_>>(),
            ranked[..ELITES]
                .iter()
                .map(|member| &member.member.semantic_sha256)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            first
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<HashSet<_>>()
                .len(),
            POPULATION
        );
        assert_eq!(first_history, second_history);
        assert_eq!(first_history.len(), POPULATION + (POPULATION - ELITES));
    }

    #[test]
    #[ignore = "legacy v1 strict-firewall fixture; v2 production registry is covered separately"]
    fn preparation_completes_without_test_or_baseline_files() {
        let directory = tempdir().unwrap();
        let registry = directory.path().join("registry");
        fs::create_dir_all(&registry).unwrap();
        let input_root = directory.path().join("inputs");
        fs::create_dir_all(&input_root).unwrap();

        let checkpoint_path = input_root.join("checkpoint.model");
        fs::write(&checkpoint_path, b"fixture checkpoint").unwrap();
        let checkpoint_sha256 = sha256_file(&checkpoint_path).unwrap();
        let accepted_pointer = input_root.join("accepted.json");
        fs::write(
            &accepted_pointer,
            serde_json::to_vec(&json!({
                "schema": "JopeSelfplayAcceptedByTask.v1",
                "tasks": {
                    "regression": {
                        "checkpoint": checkpoint_path,
                        "checkpoint_sha256": checkpoint_sha256,
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let corpus_train = input_root.join("corpus.train.jsonl");
        let corpus_validation = input_root.join("corpus.val.jsonl");
        fs::write(&corpus_train, "").unwrap();
        fs::write(&corpus_validation, "").unwrap();
        let corpus_manifest = input_root.join("corpus-manifest.json");
        fs::write(
            &corpus_manifest,
            serde_json::to_vec(&json!({
                "outputs": {
                    "corpus.train.jsonl": sha256_file(&corpus_train).unwrap(),
                    "corpus.val.jsonl": sha256_file(&corpus_validation).unwrap(),
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let provenance = input_root.join("provenance.json");
        fs::write(
            &provenance,
            serde_json::to_vec(&json!({
                "checkpoint_sha": checkpoint_sha256,
                "corpus_manifest": corpus_manifest,
                "checkpoint_config": {"model_version": "fixture-v1"},
            }))
            .unwrap(),
        )
        .unwrap();

        let context_slice = vec![0_u8; CONTEXT_BYTES];
        let context_sha256 = sha256_bytes(&context_slice);
        let mut context_blob = Vec::new();
        let mut catalog_entries = Vec::new();
        for (catalog_index, dataset_hash) in GOLDEN42[..10].iter().enumerate() {
            let dataset_root = registry.join(dataset_hash);
            fs::create_dir_all(&dataset_root).unwrap();
            let train = (0..100)
                .map(|row| {
                    format!(
                        "{},{},{}\n",
                        row as f64 / 100.0,
                        (row as f64).sin(),
                        row as f64 / 10.0 + (row as f64).cos()
                    )
                })
                .collect::<String>();
            let train_path = dataset_root.join("train.csv");
            fs::write(&train_path, train).unwrap();
            fs::write(
                dataset_root.join("meta.json"),
                serde_json::to_vec(&json!({
                    "source_family": format!("fixture-family-{catalog_index}"),
                    "source_platform": "fixture",
                    "source_dataset_id": format!("fixture-source-{catalog_index}"),
                    "source_dataset_name": format!("fixture-name-{catalog_index}"),
                    "source_target_column_original": "y",
                    "target_column": "target",
                    "target_names": ["target"],
                }))
                .unwrap(),
            )
            .unwrap();
            catalog_entries.push(json!({
                "dataset_hash": dataset_hash,
                "task": "regression",
                "train_csv": train_path,
                "train_sha256": sha256_file(&train_path).unwrap(),
                "source_family": format!("fixture-family-{catalog_index}"),
                "context_sha256": context_sha256,
                "catalog_index": catalog_index,
            }));
            context_blob.extend_from_slice(&context_slice);
        }
        let context_data = input_root.join("contexts.bin");
        fs::write(&context_data, &context_blob).unwrap();
        let catalog_path = input_root.join("catalog.json");
        fs::write(
            &catalog_path,
            serde_json::to_vec(&json!({
                "dataset_dimensions": 1536,
                "target_dimensions": 512,
                "record_stride_bytes": CONTEXT_BYTES,
                "entries": catalog_entries,
            }))
            .unwrap(),
        )
        .unwrap();
        let catalog_pointer = input_root.join("catalog-pointer.json");
        fs::write(
            &catalog_pointer,
            serde_json::to_vec(&json!({
                "catalog": catalog_path,
                "catalog_sha256": sha256_file(&catalog_path).unwrap(),
                "data": context_data,
                "data_sha256": sha256_file(&context_data).unwrap(),
            }))
            .unwrap(),
        )
        .unwrap();

        let proposals_path = input_root.join("proposals.jsonl");
        let mut proposal_lines = String::new();
        for dataset_hash in &GOLDEN42[..10] {
            for index in 0..64 {
                let row = proposal_row(dataset_hash, &checkpoint_sha256, &context_sha256, index);
                writeln!(proposal_lines, "{}", serde_json::to_string(&row).unwrap()).unwrap();
            }
        }
        fs::write(&proposals_path, proposal_lines).unwrap();

        let observation_db = input_root.join("observations.sqlite");
        let connection = Connection::open(&observation_db).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE observations(
                    task TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    canonical_genome TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    score REAL NOT NULL,
                    origin_class TEXT NOT NULL
                );",
            )
            .unwrap();
        for index in 0..35 {
            let genome = canonical(&format!(
                "(pipe X (elasticnet alpha={} l1_ratio=0.5))",
                index as f64 + 0.75
            ));
            connection
                .execute(
                    "INSERT INTO observations VALUES('regression','external',?1,?2,'global',?3,'gp_origin')",
                    rusqlite::params![
                        semantic_identity(&genome),
                        genome,
                        index as f64
                    ],
                )
                .unwrap();
        }
        drop(connection);

        let study_root = directory.path().join("studies");
        let args = PrepareArgs {
            run_id: "fixture-study".to_string(),
            study_root: study_root.clone(),
            checkpoint_pointer: accepted_pointer,
            checkpoint_provenance: provenance,
            catalog_pointer,
            observation_db,
            dl_proposals: Some(proposals_path),
            proposal_generator: PathBuf::new(),
            python: PathBuf::new(),
            proposal_device: "cpu".to_string(),
            sotarena_root: registry.clone(),
            workers: 1,
            smoke: false,
            proposal_manifest_only: false,
        };
        let manifest_path = prepare(&registry, args.clone()).unwrap();
        let first_bytes = fs::read(&manifest_path).unwrap();
        let (manifest, _): (TrainOnlyStudyManifest, _) = read_receipt(&manifest_path).unwrap();
        assert_eq!(manifest.datasets.len(), 10);
        assert!(manifest.firewall.train_only);
        let lower = String::from_utf8_lossy(&first_bytes).to_ascii_lowercase();
        assert!(!lower.contains("/test.csv"));
        assert!(!lower.contains("tabpfn"));
        assert!(!lower.contains("tabfm"));
        assert!(registry.read_dir().unwrap().all(|entry| !entry
            .unwrap()
            .path()
            .join("test.csv")
            .exists()));

        let repeated = prepare(&registry, args).unwrap();
        assert_eq!(manifest_path, repeated);
        assert_eq!(first_bytes, fs::read(repeated).unwrap());
    }

    #[test]
    fn score_refuses_before_validation_receipt_exists() {
        let directory = tempdir().unwrap();
        let error = score(directory.path(), "fixture", directory.path(), 1).unwrap_err();
        assert!(format!("{error:#}").contains("requires a successful independent validation"));
        assert!(!directory.path().join("fixture/test.csv").exists());
    }
}
