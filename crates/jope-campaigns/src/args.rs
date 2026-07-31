use std::path::PathBuf;

// Compatibility surface retained only for the vendored predecessor's legacy
// single-dataset entrypoint. The jope-dime binary never constructs this type.
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub(crate) struct JainRegressorArgs {
    pub fitness: String,
    pub root: PathBuf,
    pub exact_cv_dir: Option<PathBuf>,
    pub dataset: Vec<String>,
    pub hashes_file: Option<PathBuf>,
    pub limit: Option<usize>,
    pub overwrite: bool,
    pub skip_existing: bool,
    pub compound_existing: bool,
    pub targets_file: Option<PathBuf>,
    pub time_budget_secs: f64,
    pub gpu_kernel_configs: usize,
    pub gpu_screen: bool,
    pub gpu_cache_dir: Option<PathBuf>,
    pub gpr_proposal_receipt: Option<PathBuf>,
    pub rung0_exact_surrogate_dedup: bool,
    pub rung0_trace_dir: Option<PathBuf>,
    pub candidate_block_dir: Option<PathBuf>,
    pub candidate_block_only: bool,
    pub calib: bool,
    pub hard_ten_priors: bool,
    pub lab_hurdle2_train_only: bool,
    pub dry_run: bool,
    pub status_jsonl: Option<PathBuf>,
    pub gens: usize,
    pub pop: usize,
    pub folds: usize,
    pub confirm_folds: usize,
    pub exact_initial_population: bool,
    pub seed_genomes_file: Option<PathBuf>,
    pub skip_holdout_metrics: bool,
    pub no_predictions: bool,
    pub initial_population_sha256: Option<String>,
    pub threads: usize,
    pub seed: u64,
    pub exact_cv_phase2: bool,
}
