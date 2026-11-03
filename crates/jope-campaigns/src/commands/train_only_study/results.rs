use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlySelectionComplete {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) train_sha256: String,
    pub(super) encoding_sha256: String,
    pub(super) checkpoint_sha256: String,
    pub(super) split_manifest_sha256: String,
    pub(super) generation_checkpoint_sha256: Vec<String>,
    pub(super) generation_champions: Vec<String>,
    pub(super) final_champion_semantic_sha256: String,
    pub(super) final_champion_genome: String,
    pub(super) cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ChampionFixedGrid {
    pub(super) generation: usize,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) split_r2: Vec<f64>,
    pub(super) robust_fitness: f64,
    pub(super) worst_split_r2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyChampionValidation {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) selection_complete_sha256: String,
    pub(super) split_manifest_sha256: String,
    pub(super) champions: Vec<ChampionFixedGrid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyCohortSelectionComplete {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) manifest_sha256: String,
    pub(super) dataset_ids: Vec<String>,
    pub(super) selection_complete_sha256: BTreeMap<String, String>,
    pub(super) champion_validation_sha256: BTreeMap<String, String>,
    pub(super) cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyValidationReceipt {
    pub(super) schema: String,
    pub(super) study_design: String,
    pub(super) zero_shot_valid: bool,
    pub(super) checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    pub(super) run_id: String,
    pub(super) manifest_sha256: String,
    pub(super) cohort_selection_complete_sha256: String,
    pub(super) dataset_selection_sha256: BTreeMap<String, String>,
    pub(super) dataset_validation_sha256: BTreeMap<String, String>,
    pub(super) benchmark_valid: bool,
    pub(super) checkpoint_disclosure_audit_passed: bool,
    pub(super) transfer_exclusion_audit_passed: bool,
    pub(super) firewall_audit_passed: bool,
    pub(super) population_audit_passed: bool,
    pub(super) paired_history_audit_passed: bool,
    pub(super) cache_audit_passed: bool,
    pub(super) validated_dataset_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyDatasetValidationReceipt {
    pub(super) schema: String,
    pub(super) study_design: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) manifest_sha256: String,
    pub(super) cohort_selection_complete_sha256: String,
    pub(super) selection_complete_sha256: String,
    pub(super) champion_validation_sha256: String,
    pub(super) kernel_version: String,
    pub(super) metric_replay_passed: bool,
    pub(super) population_audit_passed: bool,
    pub(super) paired_history_audit_passed: bool,
    pub(super) cache_audit_passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BaselineEntry {
    pub(super) dataset_hash: String,
    pub(super) prior_aspfm_report_path: PathBuf,
    pub(super) prior_aspfm_report_sha256: String,
    pub(super) prior_aspfm_rmse: f64,
    pub(super) tabpfn_v3_report_path: PathBuf,
    pub(super) tabpfn_v3_report_sha256: String,
    pub(super) tabpfn_v3_rmse: f64,
    pub(super) tabfm_report_path: PathBuf,
    pub(super) tabfm_report_sha256: String,
    pub(super) tabfm_rmse: Option<f64>,
    pub(super) tabfm_ensemble_report_path: PathBuf,
    pub(super) tabfm_ensemble_report_sha256: String,
    pub(super) tabfm_ensemble_rmse: f64,
    pub(super) stronger_tabfm_algorithm: String,
    pub(super) stronger_tabfm_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BaselineManifest {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) entries: Vec<BaselineEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GenerationTestScore {
    pub(super) generation: usize,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) rmse: f64,
    pub(super) r2: f64,
    pub(super) mae: f64,
    pub(super) fit_seconds: f64,
    pub(super) predict_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FinalCvCandidate {
    pub(super) original_rank: usize,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
    pub(super) fold_r2: Vec<f64>,
    pub(super) fold_rmse: Vec<f64>,
    pub(super) robust_fitness: Option<f64>,
    pub(super) worst_fold_r2: Option<f64>,
    pub(super) failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyFinalCvRerank {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) validation_receipt_sha256: String,
    pub(super) selection_complete_sha256: String,
    pub(super) final_generation_checkpoint_sha256: String,
    pub(super) train_sha256: String,
    pub(super) kernel_version: String,
    pub(super) top_k: usize,
    pub(super) folds: usize,
    pub(super) split_manifest: TrainOnlySplitManifest,
    pub(super) candidates: Vec<FinalCvCandidate>,
    pub(super) selected_semantic_sha256: String,
    pub(super) selected_canonical_genome: String,
    pub(super) selected_robust_fitness: f64,
    pub(super) selected_worst_fold_r2: f64,
    pub(super) selected_original_rank: usize,
    pub(super) selection_frozen_before_test_access: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TrainOnlyPosthocScore {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) validation_receipt_sha256: String,
    pub(super) selection_complete_sha256: String,
    pub(super) generation_checkpoint_sha256: Vec<String>,
    pub(super) baseline_manifest_sha256: String,
    pub(super) train_sha256: String,
    pub(super) encoding_sha256: String,
    pub(super) checkpoint_sha256: String,
    pub(super) test_sha256: String,
    pub(super) kernel_version: String,
    pub(super) scores: Vec<GenerationTestScore>,
    pub(super) final_cv_rerank_sha256: String,
    pub(super) final_cv_rerank_score: GenerationTestScore,
    pub(super) prior_aspfm_rmse: f64,
    pub(super) tabpfn_v3_rmse: f64,
    pub(super) tabfm_rmse: Option<f64>,
    pub(super) tabfm_ensemble_rmse: f64,
    pub(super) stronger_tabfm_algorithm: String,
    pub(super) stronger_tabfm_rmse: f64,
    pub(super) beats_tabpfn_v3: bool,
    pub(super) beats_stronger_tabfm: bool,
    pub(super) beats_both: bool,
    pub(super) beats_prior_aspfm: bool,
    pub(super) cv_rerank_beats_tabpfn_v3: bool,
    pub(super) cv_rerank_beats_stronger_tabfm: bool,
    pub(super) cv_rerank_beats_both: bool,
    pub(super) cv_rerank_beats_prior_aspfm: bool,
    pub(super) test_trajectory_post_selection: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyPosthocIndex {
    pub(super) schema: String,
    pub(super) run_id: String,
    pub(super) validation_receipt_sha256: String,
    pub(super) baseline_manifest_sha256: String,
    pub(super) dataset_ids: Vec<String>,
    pub(super) posthoc_score_sha256: BTreeMap<String, String>,
    pub(super) cumulative_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DatasetReport {
    pub(super) dataset_hash: String,
    pub(super) generation_zero_validation_fitness: f64,
    pub(super) generation_nine_validation_fitness: f64,
    pub(super) validation_fitness_change: f64,
    pub(super) generation_zero_rmse: f64,
    pub(super) generation_nine_rmse: f64,
    pub(super) relative_rmse_improvement: f64,
    pub(super) prior_aspfm_rmse: f64,
    pub(super) tabpfn_v3_rmse: f64,
    pub(super) tabfm_rmse: Option<f64>,
    pub(super) tabfm_ensemble_rmse: f64,
    pub(super) stronger_tabfm_algorithm: String,
    pub(super) stronger_tabfm_rmse: f64,
    pub(super) best_baseline_algorithm: String,
    pub(super) best_baseline_rmse: f64,
    pub(super) generation_nine_relative_gap_to_best_baseline: f64,
    pub(super) cv_rerank_relative_gap_to_best_baseline: f64,
    pub(super) beats_tabpfn_v3: bool,
    pub(super) beats_stronger_tabfm: bool,
    pub(super) beats_both: bool,
    pub(super) beats_prior_aspfm: bool,
    pub(super) cv_rerank_semantic_sha256: String,
    pub(super) cv_rerank_rmse: f64,
    pub(super) cv_rerank_beats_tabpfn_v3: bool,
    pub(super) cv_rerank_beats_stronger_tabfm: bool,
    pub(super) cv_rerank_beats_both: bool,
    pub(super) cv_rerank_beats_prior_aspfm: bool,
    pub(super) final_ancestry: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct EloRating {
    pub(super) system: String,
    pub(super) rating: f64,
    pub(super) wins: usize,
    pub(super) losses: usize,
    pub(super) ties: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainOnlyStudyReport {
    pub(super) schema: String,
    pub(super) study_design: String,
    pub(super) zero_shot_valid: bool,
    pub(super) checkpoint_overlap_disclosure: CheckpointOverlapDisclosure,
    pub(super) run_id: String,
    pub(super) validation_receipt_sha256: String,
    pub(super) baseline_manifest_sha256: String,
    pub(super) posthoc_score_sha256: BTreeMap<String, String>,
    pub(super) status: String,
    pub(super) wins_vs_tabpfn_v3: usize,
    pub(super) wins_vs_stronger_tabfm: usize,
    pub(super) wins_vs_both: usize,
    pub(super) wins_vs_prior_aspfm: usize,
    pub(super) cv_rerank_wins_vs_tabpfn_v3: usize,
    pub(super) cv_rerank_wins_vs_stronger_tabfm: usize,
    pub(super) cv_rerank_wins_vs_both: usize,
    pub(super) cv_rerank_wins_vs_prior_aspfm: usize,
    pub(super) median_relative_rmse_improvement: f64,
    pub(super) bootstrap_interval_95: [f64; 2],
    pub(super) paired_sign_test_p_value: f64,
    pub(super) elo_methodology: String,
    pub(super) elo_ratings: Vec<EloRating>,
    pub(super) test_trajectories_post_selection: bool,
    pub(super) datasets: Vec<DatasetReport>,
}
