use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CheckpointProvenance {
    pub(super) accepted_pointer_path: PathBuf,
    pub(super) accepted_pointer_sha256: String,
    pub(super) checkpoint_path: PathBuf,
    pub(super) checkpoint_sha256: String,
    pub(super) model_version: String,
    pub(super) training_corpus_manifest_path: PathBuf,
    pub(super) training_corpus_manifest_sha256: String,
    pub(super) training_corpus_train_path: PathBuf,
    pub(super) training_corpus_train_sha256: String,
    pub(super) training_corpus_validation_path: PathBuf,
    pub(super) training_corpus_validation_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CatalogPointer {
    pub(super) catalog: PathBuf,
    pub(super) catalog_sha256: String,
    pub(super) data: PathBuf,
    pub(super) data_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Catalog {
    pub(super) dataset_dimensions: usize,
    pub(super) target_dimensions: usize,
    pub(super) record_stride_bytes: usize,
    pub(super) entries: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct CatalogEntry {
    pub(super) dataset_hash: String,
    pub(super) task: String,
    pub(super) train_csv: PathBuf,
    pub(super) train_sha256: String,
    pub(super) source_family: String,
    pub(super) context_sha256: String,
    pub(super) catalog_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CorpusExclusions {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) checkpoint_sha256: String,
    pub(super) training_corpus_train_sha256: String,
    pub(super) training_corpus_validation_sha256: String,
    pub(super) dataset_ids: Vec<String>,
    pub(super) source_families: Vec<String>,
    pub(super) target_identities: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CheckpointOverlapDisclosure {
    pub(super) schema: String,
    pub(super) checkpoint_sha256: String,
    pub(super) exact_golden42_dataset_ids: Vec<String>,
    pub(super) golden42_source_families: Vec<String>,
    pub(super) golden42_target_identities: Vec<String>,
    pub(super) selected_exact_dataset_ids: Vec<String>,
    pub(super) selected_source_families: Vec<String>,
    pub(super) selected_target_identities: Vec<String>,
    pub(super) checkpoint_was_trained_on_golden42_identities: bool,
    pub(super) zero_shot_valid: bool,
    pub(super) disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SotarenaReportHashes {
    pub(super) tabpfn_v3_path: PathBuf,
    pub(super) tabpfn_v3_sha256: String,
    pub(super) tabfm_path: PathBuf,
    pub(super) tabfm_sha256: String,
    pub(super) tabfm_ensemble_path: PathBuf,
    pub(super) tabfm_ensemble_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct EligibilityRecord {
    pub(super) dataset_hash: String,
    pub(super) source_family: Option<String>,
    pub(super) target_identity: Option<String>,
    pub(super) rows: Option<usize>,
    pub(super) features: Option<usize>,
    pub(super) train_sha256: Option<String>,
    pub(super) context_sha256: Option<String>,
    pub(super) checkpoint_exact_overlap: bool,
    pub(super) checkpoint_source_family_overlap: bool,
    pub(super) checkpoint_target_identity_overlap: bool,
    pub(super) baseline_reports: Option<SotarenaReportHashes>,
    pub(super) selection_key: String,
    pub(super) eligible: bool,
    pub(super) reasons: Vec<String>,
    pub(super) selected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct EncodingSlice {
    pub(super) blob_path: PathBuf,
    pub(super) blob_sha256: String,
    pub(super) offset_bytes: usize,
    pub(super) length_bytes: usize,
    pub(super) context_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct DatasetStudyEntry {
    pub(super) dataset_hash: String,
    pub(super) source_family: String,
    pub(super) target_identity: String,
    pub(super) train_path: PathBuf,
    pub(super) train_sha256: String,
    pub(super) rows: usize,
    pub(super) features: usize,
    pub(super) encoding: EncodingSlice,
    pub(super) split_manifest_path: PathBuf,
    pub(super) split_manifest_sha256: String,
    pub(super) generation_zero_path: PathBuf,
    pub(super) generation_zero_sha256: String,
    pub(super) transfer_ranking_path: PathBuf,
    pub(super) transfer_ranking_sha256: String,
    pub(super) baseline_reports: SotarenaReportHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyStudyManifest {
    pub(super) schema: String,
    pub(super) study_design: String,
    pub(super) zero_shot_valid: bool,
    pub(super) checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    pub(super) campaign_scope: String,
    pub(super) run_id: String,
    pub(super) seed: u64,
    pub(super) generations: usize,
    pub(super) population: usize,
    pub(super) elites: usize,
    pub(super) parent_pool: usize,
    pub(super) tournament_size: usize,
    pub(super) mutation_probability: f64,
    pub(super) crossover_probability: f64,
    pub(super) max_transforms: usize,
    pub(super) workers: usize,
    pub(super) kernel_version: String,
    pub(super) sotarena_root: PathBuf,
    pub(super) sotarena_registry_path: PathBuf,
    pub(super) sotarena_registry_sha256: String,
    pub(super) golden42_ids: Vec<String>,
    pub(super) eligibility: Vec<EligibilityRecord>,
    pub(super) selected_dataset_ids: Vec<String>,
    pub(super) checkpoint: CheckpointProvenance,
    pub(super) corpus_exclusions_path: PathBuf,
    pub(super) corpus_exclusions_sha256: String,
    pub(super) catalog_pointer_path: PathBuf,
    pub(super) catalog_pointer_sha256: String,
    pub(super) source_catalog_sha256: String,
    pub(super) filtered_context_blob_sha256: String,
    pub(super) observation_db_path: PathBuf,
    pub(super) observation_db_sha256: String,
    pub(super) dl_proposals_path: PathBuf,
    pub(super) dl_proposals_sha256: String,
    pub(super) proposal_dataset_manifest_path: PathBuf,
    pub(super) proposal_dataset_manifest_sha256: String,
    pub(super) datasets: Vec<DatasetStudyEntry>,
    pub(super) firewall: PhaseOneFirewall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PhaseOneFirewall {
    pub(super) train_only: bool,
    pub(super) contains_test_paths: bool,
    pub(super) contains_test_hashes: bool,
    pub(super) contains_baseline_scores: bool,
    pub(super) test_paths_constructed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlySplitManifest {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) train_sha256: String,
    pub(super) seed: u64,
    pub(super) rows: usize,
    pub(super) splits: Vec<SplitReceipt>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SplitReceipt {
    pub(super) generation: usize,
    pub(super) seed: u64,
    pub(super) train_indices: Vec<usize>,
    pub(super) validation_indices: Vec<usize>,
    pub(super) validation_target_sha256: String,
    pub(super) split_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Ancestry {
    pub(super) operation: String,
    pub(super) parents: Vec<String>,
    pub(super) roots: BTreeSet<String>,
    pub(super) retry_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct GenomeMember {
    pub(super) slot: usize,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) complexity: usize,
    pub(super) source: String,
    pub(super) source_evidence: Value,
    pub(super) ancestry: Ancestry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyGenerationZero {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) train_sha256: String,
    pub(super) encoding_sha256: String,
    pub(super) checkpoint_sha256: String,
    pub(super) dl_proposals_sha256: String,
    pub(super) transfer_ranking_sha256: String,
    pub(super) members: Vec<GenomeMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ProposalRow {
    pub(super) task: String,
    pub(super) dataset_hash: String,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) model_checkpoint_sha256: String,
    #[serde(default)]
    pub(super) context_sha256: Option<String>,
    #[serde(default)]
    pub(super) proposal_score: Option<f64>,
    #[serde(default)]
    pub(super) selection_score: Option<f64>,
    #[serde(default)]
    pub(super) reward_lcb: Option<f64>,
    #[serde(default)]
    pub(super) instability_probability: Option<f64>,
    #[serde(default)]
    pub(super) slot: usize,
    #[serde(flatten)]
    pub(super) extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GlobalRankedGenome {
    pub(super) rank: usize,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) mean_percentile: f64,
    pub(super) worst_percentile: f64,
    pub(super) strict_win_count: usize,
    pub(super) dataset_count: usize,
    pub(super) source_dataset_ids: Vec<String>,
    pub(super) source_families: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GlobalRankingReceipt {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) target_dataset_id: String,
    pub(super) target_source_family: String,
    pub(super) observation_db_sha256: String,
    pub(super) source_catalog_sha256: String,
    pub(super) excluded_dataset_ids: Vec<String>,
    pub(super) excluded_source_family: String,
    pub(super) direct_dl_sources_excluded: bool,
    pub(super) ranking_policy: String,
    pub(super) genomes: Vec<GlobalRankedGenome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyFitnessObservation {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) train_sha256: String,
    pub(super) encoding_sha256: String,
    pub(super) checkpoint_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) split_generation: usize,
    pub(super) split_sha256: String,
    pub(super) kernel_version: String,
    pub(super) cache_key: String,
    pub(super) r2: Option<f64>,
    pub(super) rmse: Option<f64>,
    pub(super) mae: Option<f64>,
    pub(super) fit_seconds: f64,
    pub(super) predict_seconds: f64,
    pub(super) failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ObservationLink {
    pub(super) split_generation: usize,
    pub(super) split_sha256: String,
    pub(super) receipt_path: PathBuf,
    pub(super) receipt_sha256: String,
    pub(super) cache_key: String,
    pub(super) cache_hit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ScoredMember {
    pub(super) member: GenomeMember,
    pub(super) observations: Vec<ObservationLink>,
    pub(super) split_r2: Vec<f64>,
    pub(super) robust_fitness: Option<f64>,
    pub(super) worst_split_r2: Option<f64>,
    pub(super) failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyGenerationCheckpoint {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) train_sha256: String,
    pub(super) encoding_sha256: String,
    pub(super) checkpoint_sha256: String,
    pub(super) generation: usize,
    pub(super) previous_checkpoint_sha256: Option<String>,
    pub(super) split_manifest_sha256: String,
    pub(super) population: Vec<ScoredMember>,
    pub(super) ranked_semantic_sha256: Vec<String>,
    pub(super) champion_semantic_sha256: String,
    pub(super) rankable_members: usize,
    pub(super) cache_hits: usize,
    pub(super) cache_misses: usize,
    pub(super) invalid_members: usize,
    pub(super) next_population: Vec<GenomeMember>,
}
