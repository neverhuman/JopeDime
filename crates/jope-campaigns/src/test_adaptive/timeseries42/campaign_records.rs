#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CampaignManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_campaign_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_start_generation: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_terminal_generation: Option<usize>,
    registry_root: String,
    output_root: String,
    release_binary_sha256: String,
    corpus_eval_sha256: String,
    scq_binary: String,
    scq_binary_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coverage_binding: Option<CoverageBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seed_populations: Vec<SeedPopulationBinding>,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    config: CampaignConfig,
    datasets: Vec<DatasetSpec>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SeedPopulationBinding {
    dataset_hash: String,
    seed_bank_path: String,
    seed_bank_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CoverageBinding {
    multiclass_manifest_sha256: String,
    multiclass_final_receipt_sha256: String,
    timeseries_manifest_sha256: String,
    timeseries_final_receipt_sha256: String,
    timeseries_baseline_receipt_sha256: String,
    comparator_root: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedDataset {
    dataset_hash: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedNode {
    node: String,
    hostname: String,
    bundle_root: String,
    runner_path: String,
    runner_sha256: String,
    corpus_eval_path: String,
    corpus_eval_sha256: String,
    datasets: Vec<StagedDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagingReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    release_binary_sha256: String,
    corpus_eval_sha256: String,
    nodes: Vec<StagedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LabelAuthority {
    score_tier: String,
    metric: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    selection_bias: Option<String>,
    generation: String,
    frozen_seed_sha256: String,
    model_checkpoint_sha256: String,
    hall_of_fame_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ShardCandidate {
    semantic_sha256: String,
    canonical_genome: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CorpusShardManifest {
    schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    campaign_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    release_binary_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparator_lock_sha256: Option<String>,
    task: String,
    dataset_hash: String,
    train_csv: String,
    test_csv: String,
    train_sha256: String,
    test_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_epoch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    catalog_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    protocol_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    generated_at_ms: Option<u128>,
    label_authority: LabelAuthority,
    meta_json: String,
    meta_sha256: String,
    horizon: usize,
    seasonality: usize,
    frequency: String,
    target_is_final_column: bool,
    known_future_covariates: Vec<String>,
    past_only_covariates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reduction: Option<CorpusReductionSpec>,
    candidates: Vec<ShardCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CorpusReductionSpec {
    version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ShardRecord {
    generation: usize,
    shard: usize,
    node: String,
    attempt: usize,
    job_id: String,
    corpus_manifest_file: String,
    corpus_manifest_sha256: String,
    scq_spec_file: String,
    scq_spec_sha256: String,
    receipt_file: String,
    receipt_sha256: String,
    valid_candidates: usize,
    total_candidates: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    campaign_sha256: String,
    release_binary_sha256: String,
    dataset_hash: String,
    generation: usize,
    members: Vec<Member>,
    shards: Vec<ShardRecord>,
    best_semantic_sha256: Option<String>,
    best_mase: Option<f64>,
    valid_candidates: usize,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorkerReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    dataset_hash: String,
    manifest_sha256: String,
    receipt_sha256: String,
    done_sha256: String,
    rows: usize,
    valid_rows: usize,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WaveReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    campaign_sha256: String,
    dataset_hash: String,
    generation: usize,
    nodes: Vec<String>,
    job_ids: Vec<String>,
    corpus_manifest_sha256: Vec<String>,
    scq_spec_sha256: Vec<String>,
    terminal_rows: usize,
    submitted_at_unix_ms: u128,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug)]
struct PreparedShard {
    generation: usize,
    shard: usize,
    node: String,
    attempt: usize,
    job_id: String,
    remote_manifest_sha256: String,
    local_manifest_path: PathBuf,
    spec_path: PathBuf,
    spec_text: String,
    remote_output: String,
    receipt_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct JobReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    job_id: String,
    node: String,
    attempt: usize,
    state: String,
    exit_code: i64,
    corpus_manifest_sha256: String,
    scq_spec_sha256: String,
    scq_show_sha256: String,
    worker_receipt_sha256: String,
    receipt_sha256: String,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Deserialize)]
struct EvalReceipt {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    canonical_genome: String,
    metric: String,
    score: Option<f64>,
    secondary_metric: String,
    secondary_score: Option<f64>,
    valid: bool,
    failure: Option<String>,
    runtime_seconds: f64,
    predictions_sha256: Option<String>,
    rmse: Option<f64>,
    protocol_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TimeSeriesScore {
    mase: f64,
    rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LogicalSlot {
    generation: usize,
    slot: usize,
    member: Member,
    score: Option<TimeSeriesScore>,
    valid: bool,
    failure: Option<String>,
    receipt_file: String,
    receipt_sha256: String,
    prediction_sha256: Option<String>,
    runtime_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetResult {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_strict_winner: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_best_jope_mase: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_evidence_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_receipt_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_prediction_sha256: Option<String>,
    campaign_sha256: String,
    release_binary_sha256: String,
    dataset_hash: String,
    status: String,
    official_leader_model: String,
    official_leader_mase: f64,
    generation_zero_best_mase: Option<f64>,
    best_jope_mase: Option<f64>,
    best_jope_rmse: Option<f64>,
    mase_delta_vs_official: Option<f64>,
    strict_winner: bool,
    best_beats_official: bool,
    best_semantic_sha256: Option<String>,
    best_canonical_genome: Option<String>,
    best_generation: Option<usize>,
    best_slot: Option<usize>,
    seed_source_family: Option<String>,
    logical_evaluations: usize,
    valid_evaluations: usize,
    completed_generations: usize,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TerminalSummary {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inherited_base_wins: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    improved_base_wins: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_strict_wins: Option<usize>,
    dataset_count: usize,
    wins: usize,
    losses: usize,
    ties: usize,
    exhausted: usize,
    logical_evaluations: usize,
    valid_evaluations: usize,
    datasets: Vec<DatasetResult>,
    sotarena_scan_output: Option<String>,
    completed_at_unix_ms: u128,
}
