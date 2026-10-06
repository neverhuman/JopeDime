use super::*;

const FINAL_METHOD_ID: &str = "jope-dime-train-only-redteam-v1";
const GEN9_VARIANT: &str = "existing-v2-generation-9-control";
const G20_RAW_VARIANT: &str = "raw-generation-19-champion";
const G20_OOF_VARIANT: &str = "generation-0-19-train-only-oof-selection";
const SMART_SINGLE_VARIANT: &str = "smart-v3-single-pipeline";
const SMART_NO_JITTER_VARIANT: &str = "smart-v3-no-jitter-counterfactual";
const SMART_ORIGINAL_SEED_VARIANT: &str = "smart-v3-original-decoder-seed-only-counterfactual";
const FINAL_VARIANT: &str = "smart-v3-final";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetValidationShard {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    owner_node: String,
    replay: execution::DatasetValidationReplay,
    #[serde(default)]
    scq_execution: Option<ScqExecutionProvenance>,
    baseline_paths_accessed: bool,
    test_paths_accessed: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GlobalValidationReceipt {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    g20_arm_sha256: String,
    smart_arm_sha256: String,
    selector_cohort_sha256: String,
    dataset_validation_sha256: BTreeMap<String, String>,
    #[serde(default)]
    scq_execution: BTreeMap<String, ScqExecutionProvenance>,
    every_selection_receipt_validated: bool,
    baseline_paths_accessed: bool,
    test_paths_accessed: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BaselineRow {
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
    oracle_aspfm_rmse: f64,
    prior_train_only_source_sha256: String,
    tabpfn_v3_source_sha256: String,
    stronger_tabfm_source_sha256: String,
    oracle_source_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FrozenTestScore {
    variant: String,
    semantic_sha256: Vec<String>,
    weights: Vec<f64>,
    rmse: f64,
    predictions_sha256: String,
    full_data_refit_parity: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetScoreReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    validation_sha256: String,
    selector_sha256: String,
    train_sha256: String,
    test_sha256: String,
    baseline_manifest_sha256: String,
    oracle_index_sha256: String,
    variants: Vec<FrozenTestScore>,
    baselines: BaselineRow,
    selection_frozen_before_test_access: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScoreIndex {
    schema: String,
    run_id: String,
    validation_sha256: String,
    baseline_manifest_sha256: String,
    oracle_index_sha256: String,
    dataset_score_sha256: BTreeMap<String, String>,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WinCounts {
    tabpfn_v3: usize,
    stronger_tabfm: usize,
    both_tabpfn_and_stronger_tabfm: usize,
    prior_train_only: usize,
    generation_9_control: usize,
    oracle_aspfm_envelope: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReportDataset {
    dataset_hash: String,
    final_rmse: f64,
    generation_9_control_rmse: f64,
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_rmse: f64,
    oracle_aspfm_rmse: f64,
    final_variant: String,
    ensemble_accepted: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StudyReport {
    schema: String,
    run_id: String,
    validation_sha256: String,
    score_index_sha256: String,
    fair_ladder: Vec<EloRating>,
    oracle_reference_ladder: Vec<EloRating>,
    wins: WinCounts,
    median_final_rmse: f64,
    median_generation_9_control_rmse: f64,
    development_strategy_improved: bool,
    oracle_comparison_determines_success: bool,
    authoritative_method_labels: BTreeMap<String, String>,
    datasets: Vec<ReportDataset>,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SmartV3TestRow {
    dataset_hash: String,
    final_variant: String,
    assigned_scq_node: String,
    selector_sha256: String,
    train_sha256: String,
    test_sha256: String,
    test_rmse: f64,
    test_mae: f64,
    test_r2: f64,
    previous_train_only_rmse: f64,
    relative_rmse_change: f64,
    tabpfn_v3_rmse: f64,
    relative_rmse_change_tabpfn_v3: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
    relative_rmse_change_stronger_tabfm: f64,
    prior_comparison: String,
    tabpfn_v3_comparison: String,
    stronger_tabfm_comparison: String,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3Aggregate {
    wins: usize,
    ties: usize,
    losses: usize,
    median_relative_rmse_change: f64,
    geometric_mean_rmse_ratio: f64,
    minimum_wins_required: usize,
    smart_v3_improved: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3BaselineAggregate {
    wins: usize,
    ties: usize,
    losses: usize,
    median_relative_rmse_change: f64,
    geometric_mean_rmse_ratio: f64,
    minimum_wins_required: usize,
    smart_v3_beats_baseline: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3AggregateComparisons {
    previous_train_only: SmartV3BaselineAggregate,
    tabpfn_v3: SmartV3BaselineAggregate,
    stronger_tabfm: SmartV3BaselineAggregate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SmartV3TestResults {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    baseline_manifest_sha256: String,
    selector_count: usize,
    test_result_rows: Vec<SmartV3TestRow>,
    aggregate: SmartV3Aggregate,
    baseline_comparisons: SmartV3AggregateComparisons,
    selection_frozen_before_test_access: bool,
    test_paths_opened_only_by_this_command: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

struct SmartComparisonBaselines {
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
}

fn arm_receipt_path(root: &Path, arm: &str) -> PathBuf {
    root.join(format!("{arm}-complete.v1.json"))
}

fn selector_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("selector-complete.v1.json")
}

fn validation_shard_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("validation-shard.v1.json")
}

fn validate_arm_receipt(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    manifest_digest: &str,
    arm: &str,
    dataset_receipts: &BTreeMap<String, String>,
) -> Result<String> {
    let path = arm_receipt_path(root, arm);
    let (receipt, digest): (Value, _) = read_receipt(&path)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("TrainOnlyRedteamArmComplete.v1")
        || receipt.get("run_id").and_then(Value::as_str) != Some(&manifest.run_id)
        || receipt.get("arm").and_then(Value::as_str) != Some(arm)
        || receipt.get("manifest_sha256").and_then(Value::as_str) != Some(manifest_digest)
        || receipt.get("development_only").and_then(Value::as_bool) != Some(true)
        || receipt.get("confirmatory_valid").and_then(Value::as_bool) != Some(false)
        || receipt
            .get("scq_execution")
            .and_then(Value::as_object)
            .is_none()
        || receipt
            .get("dataset_receipt_sha256")
            .cloned()
            .map(serde_json::from_value::<BTreeMap<String, String>>)
            .transpose()?
            .as_ref()
            != Some(dataset_receipts)
    {
        bail!("{arm} cohort receipt drift");
    }
    Ok(digest)
}

fn validate_selector_cohort(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    manifest_digest: &str,
    selector_sha256: &BTreeMap<String, String>,
) -> Result<String> {
    let (receipt, digest): (Value, _) = read_receipt(&root.join("selector-complete.v1.json"))?;
    if receipt.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyRedteamSelectorCohortComplete.v1")
        || receipt.get("run_id").and_then(Value::as_str) != Some(&manifest.run_id)
        || receipt.get("manifest_sha256").and_then(Value::as_str) != Some(manifest_digest)
        || receipt
            .get("selector_sha256")
            .cloned()
            .map(serde_json::from_value::<BTreeMap<String, String>>)
            .transpose()?
            .as_ref()
            != Some(selector_sha256)
        || receipt
            .get("selection_frozen_before_test_access")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("development_only").and_then(Value::as_bool) != Some(true)
        || receipt.get("confirmatory_valid").and_then(Value::as_bool) != Some(false)
        || receipt
            .get("scq_execution")
            .and_then(Value::as_object)
            .is_none()
    {
        bail!("selector cohort receipt drift");
    }
    Ok(digest)
}

pub(super) fn validate_run(
    study_root: &Path,
    run_id: &str,
    _sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    let root = run_root(study_root, run_id)?;
    let manifest_path = root.join("manifest.redteam.v1.json");
    let (manifest, manifest_digest) = execution::validate_manifest(&manifest_path)?;
    if let Some(selected) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == selected)
            .context("selected dataset is absent from the red-team manifest")?;
        let replay = execution::validate_dataset_selection(&root, &manifest, dataset, workers)?;
        let expected_owner = manifest
            .config
            .scq
            .evaluation_ownership
            .get(selected)
            .context("validation ownership is absent")?
            .clone();
        let receipt = DatasetValidationShard {
            schema: "TrainOnlyRedteamDatasetValidation.v1".to_string(),
            run_id: run_id.to_string(),
            manifest_sha256: manifest_digest,
            owner_node: expected_owner,
            scq_execution: replay.scq_execution.clone(),
            replay,
            baseline_paths_accessed: false,
            test_paths_accessed: false,
            development_only: true,
            confirmatory_valid: false,
        };
        let path = validation_shard_path(&root, selected);
        write_receipt(&path, &receipt)?;
        return Ok(path);
    }

    let mut shards = BTreeMap::new();
    let mut g20_datasets = BTreeMap::new();
    let mut smart_datasets = BTreeMap::new();
    let mut selectors = BTreeMap::new();
    let mut scq_execution = BTreeMap::new();
    for dataset in &manifest.datasets {
        let path = validation_shard_path(&root, &dataset.dataset_hash);
        if !path.exists() {
            bail!(
                "independent validation shard is missing for {}; run it on {} first",
                dataset.dataset_hash,
                manifest.config.scq.evaluation_ownership[&dataset.dataset_hash]
            );
        }
        let (shard, digest): (DatasetValidationShard, _) = read_receipt(&path)?;
        if shard.schema != "TrainOnlyRedteamDatasetValidation.v1"
            || shard.run_id != run_id
            || shard.manifest_sha256 != manifest_digest
            || shard.owner_node != manifest.config.scq.evaluation_ownership[&dataset.dataset_hash]
            || shard.replay.dataset_hash != dataset.dataset_hash
            || !shard.replay.deterministic_generation_replay
            || !shard.replay.uncached_metric_parity
            || shard.scq_execution != shard.replay.scq_execution
            || !shard.replay.development_only
            || shard.replay.confirmatory_valid
            || shard.baseline_paths_accessed
            || shard.test_paths_accessed
            || !shard.development_only
            || shard.confirmatory_valid
        {
            bail!("independent validation shard drift");
        }
        g20_datasets.insert(
            dataset.dataset_hash.clone(),
            shard.replay.g20_complete_sha256.clone(),
        );
        smart_datasets.insert(
            dataset.dataset_hash.clone(),
            shard.replay.smart_complete_sha256.clone(),
        );
        selectors.insert(
            dataset.dataset_hash.clone(),
            shard.replay.selector_sha256.clone(),
        );
        let execution = shard
            .scq_execution
            .clone()
            .or_else(|| shard.replay.scq_execution.clone())
            .context("SCQ execution provenance is missing")?;
        if execution.scq_job_id.is_empty()
            || execution.scheduler_reported_node.is_empty()
            || execution.release_binary_sha256.is_empty()
            || execution.source_bundle_sha256.is_empty()
            || execution.manifest_sha256.is_empty()
            || execution.proposal_bundle_sha256.is_empty()
            || execution.fold_bank_sha256.is_empty()
            || execution.parent_input_sha256.is_empty()
            || execution.output_package_sha256.is_empty()
            || execution.output_package_sha256 == "not-captured"
            || execution.node_marker_probe.is_empty()
            || execution.expected_node
                != manifest.config.scq.evaluation_ownership[&dataset.dataset_hash]
            || execution.scheduler_reported_node != execution.expected_node
            || execution.workers != manifest.config.scq.evaluation_cpus
            || execution.ram_gb != manifest.config.scq.evaluation_ram_gb
            || !execution.non_preemptible
            || execution.max_runtime_hours != manifest.config.scq.max_runtime_hours
            || !execution.imported_without_overwrite
        {
            bail!("SCQ execution provenance does not match the sealed contract");
        }
        scq_execution.insert(dataset.dataset_hash.clone(), execution);
        shards.insert(dataset.dataset_hash.clone(), digest);
    }
    let g20_arm_sha256 = validate_arm_receipt(
        &root,
        &manifest,
        &manifest_digest,
        "g20-extension",
        &g20_datasets,
    )?;
    let smart_arm_sha256 = validate_arm_receipt(
        &root,
        &manifest,
        &manifest_digest,
        "smart-v3",
        &smart_datasets,
    )?;
    let selector_cohort_sha256 =
        validate_selector_cohort(&root, &manifest, &manifest_digest, &selectors)?;
    let receipt = GlobalValidationReceipt {
        schema: "TrainOnlyRedteamGlobalValidation.v1".to_string(),
        run_id: run_id.to_string(),
        manifest_sha256: manifest_digest,
        g20_arm_sha256,
        smart_arm_sha256,
        selector_cohort_sha256,
        dataset_validation_sha256: shards,
        scq_execution,
        every_selection_receipt_validated: true,
        baseline_paths_accessed: false,
        test_paths_accessed: false,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let path = root.join("validation-success.v1.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}

fn require_global_validation(
    study_root: &Path,
    run_id: &str,
) -> Result<(
    PathBuf,
    TrainOnlyRedteamManifest,
    String,
    GlobalValidationReceipt,
    String,
)> {
    let root = run_root(study_root, run_id)?;
    let (manifest, manifest_digest) =
        execution::validate_manifest(&root.join("manifest.redteam.v1.json"))?;
    let path = root.join("validation-success.v1.json");
    let (validation, validation_digest): (GlobalValidationReceipt, _) = read_receipt(&path)?;
    if validation.schema != "TrainOnlyRedteamGlobalValidation.v1"
        || validation.run_id != run_id
        || validation.manifest_sha256 != manifest_digest
        || validation.dataset_validation_sha256.len() != manifest.datasets.len()
        || !validation.every_selection_receipt_validated
        || validation.baseline_paths_accessed
        || validation.test_paths_accessed
        || !validation.development_only
        || validation.confirmatory_valid
        || validation.test_reuse_disclosure != DISCLOSURE
        || validation.scq_execution.len() != manifest.datasets.len()
    {
        bail!("global validation gate drift");
    }
    for dataset in &manifest.datasets {
        let (_, digest): (DatasetValidationShard, _) =
            read_receipt(&validation_shard_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_validation_sha256
            .get(&dataset.dataset_hash)
            != Some(&digest)
        {
            bail!("validation shard is not covered by the global gate");
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

fn finite_rmse(value: &Value, pointer: &str) -> Result<f64> {
    let value = value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing RMSE at {pointer}"))?;
    if !value.is_finite() || value < 0.0 {
        bail!("invalid RMSE at {pointer}");
    }
    Ok(value)
}

fn sum_squared_error(truth: &[f64], predictions: &[f64]) -> Result<f64> {
    if truth.len() != predictions.len()
        || truth.is_empty()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("score vectors are invalid");
    }
    Ok(truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum())
}

fn baseline_row(
    entry: &Value,
    dataset_hash: &str,
    oracle: &Value,
    oracle_source_sha256: String,
) -> Result<BaselineRow> {
    if entry.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash) {
        bail!("baseline dataset identity drift");
    }
    let stronger_method = value_string(entry, "/stronger_tabfm_algorithm")?;
    let stronger_path = if stronger_method == "tabfm_regressor" {
        "/tabfm_report_path"
    } else if stronger_method == "tabfm_ensemble_regressor" {
        "/tabfm_ensemble_report_path"
    } else {
        bail!("stronger TabFM identity drift");
    };
    let source = |path_pointer: &str, hash_pointer: &str| -> Result<String> {
        let path =
            execution::relocated_input_path(&PathBuf::from(value_string(entry, path_pointer)?));
        let expected = value_string(entry, hash_pointer)?;
        if sha256_file(&path)? != expected {
            bail!("authoritative baseline source drift");
        }
        Ok(expected)
    };
    if oracle.get("schema").and_then(Value::as_str)
        != Some("jope-dime.combined-regression-dataset-index.v1")
        || oracle.get("dataset").and_then(Value::as_str) != Some(dataset_hash)
        || oracle.get("cohort").and_then(Value::as_str) != Some("golden42")
    {
        bail!("ASPFM oracle source identity drift");
    }
    Ok(BaselineRow {
        prior_train_only_rmse: finite_rmse(entry, "/prior_aspfm_rmse")?,
        tabpfn_v3_rmse: finite_rmse(entry, "/tabpfn_v3_rmse")?,
        stronger_tabfm_method_id: stronger_method,
        stronger_tabfm_rmse: finite_rmse(entry, "/stronger_tabfm_rmse")?,
        oracle_aspfm_rmse: finite_rmse(oracle, "/best_rmse")?,
        prior_train_only_source_sha256: source(
            "/prior_aspfm_report_path",
            "/prior_aspfm_report_sha256",
        )?,
        tabpfn_v3_source_sha256: source("/tabpfn_v3_report_path", "/tabpfn_v3_report_sha256")?,
        stronger_tabfm_source_sha256: if stronger_path == "/tabfm_report_path" {
            source("/tabfm_report_path", "/tabfm_report_sha256")?
        } else {
            source(
                "/tabfm_ensemble_report_path",
                "/tabfm_ensemble_report_sha256",
            )?
        },
        oracle_source_sha256,
    })
}

fn predict_variant(
    variant: &execution::FrozenVariant,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<Vec<f64>> {
    if variant.candidates.is_empty()
        || variant.candidates.len() != variant.weights.len()
        || variant
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (variant.weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        bail!("frozen full-data variant violates the convex contract");
    }
    let mut predictions = vec![0.0; test.y.len()];
    for (candidate, weight) in variant.candidates.iter().zip(&variant.weights) {
        let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
        let first = fit(&parsed, &train.x, &train.y)?.predict(&test.x)?;
        let replay = fit(&parsed, &train.x, &train.y)?.predict(&test.x)?;
        if first != replay {
            bail!("full-data refit parity failure");
        }
        for (prediction, value) in predictions.iter_mut().zip(first) {
            *prediction += weight * value;
        }
    }
    Ok(predictions)
}

fn score_variant(
    name: &str,
    variant: &execution::FrozenVariant,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<FrozenTestScore> {
    let predictions = predict_variant(variant, train, test)?;
    let rmse = (sum_squared_error(&test.y, &predictions)? / test.y.len() as f64).sqrt();
    Ok(FrozenTestScore {
        variant: name.to_string(),
        semantic_sha256: variant
            .candidates
            .iter()
            .map(|candidate| candidate.semantic_sha256.clone())
            .collect(),
        weights: variant.weights.clone(),
        rmse,
        predictions_sha256: receipt_digest(&predictions)?,
        full_data_refit_parity: true,
        development_only: true,
        confirmatory_valid: false,
    })
}

fn frozen_variants(
    dataset: &DatasetInput,
    g20: &execution::G20Complete,
    selector: &execution::SelectorComplete,
) -> Result<Vec<(&'static str, execution::FrozenVariant)>> {
    let (parent_checkpoint, _): (Value, _) = read_receipt(&dataset.parent_generation_paths[9])?;
    let parent_champion = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .and_then(|population| population.first())
        .context("parent generation-nine champion is absent")?;
    let gen9 = execution::candidate_from_parent(parent_champion)?;
    let g20_oof = g20
        .top_archive_search_oof
        .first()
        .context("G20 OOF archive is empty")?
        .candidate
        .clone();
    let single = |name: &str, candidate: execution::Candidate| execution::FrozenVariant {
        name: name.to_string(),
        candidates: vec![candidate],
        weights: vec![1.0],
    };
    Ok(vec![
        (GEN9_VARIANT, single(GEN9_VARIANT, gen9)),
        (
            G20_RAW_VARIANT,
            single(G20_RAW_VARIANT, g20.raw_generation_19.clone()),
        ),
        (G20_OOF_VARIANT, single(G20_OOF_VARIANT, g20_oof)),
        (
            SMART_SINGLE_VARIANT,
            single(SMART_SINGLE_VARIANT, selector.smart_single.clone()),
        ),
        (
            SMART_NO_JITTER_VARIANT,
            single(SMART_NO_JITTER_VARIANT, selector.no_jitter.clone()),
        ),
        (
            SMART_ORIGINAL_SEED_VARIANT,
            single(
                SMART_ORIGINAL_SEED_VARIANT,
                selector.original_decoder_seed_only.clone(),
            ),
        ),
        (FINAL_VARIANT, selector.final_variant.clone()),
    ])
}

pub(super) fn score_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    // This is the only gate above any construction of a test or baseline path.
    let (root, manifest, _, validation, validation_digest) =
        require_global_validation(study_root, run_id)?;

    let baseline_path = manifest
        .parent
        .manifest_path
        .parent()
        .context("parent manifest has no run root")?
        .join("baselines.v2.json");
    let (baseline_manifest, baseline_digest): (Value, _) = read_receipt(&baseline_path)?;
    if baseline_manifest.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyBaselineManifest.v2")
        || baseline_manifest
            .get("entries")
            .and_then(Value::as_array)
            .is_none_or(|entries| entries.len() != manifest.datasets.len())
    {
        bail!("authoritative baseline manifest drift");
    }
    let oracle_index_path =
        sotarena_root.join("results/tabular-regression-v1/jope-dime-evolution/index.v1.json");
    let oracle_index_sha256 = sha256_file(&oracle_index_path)?;
    let oracle_index: Value = serde_json::from_slice(&fs::read(&oracle_index_path)?)?;
    if oracle_index.get("schema").and_then(Value::as_str)
        != Some("jope-dime.combined-regression-index.v1")
        || oracle_index
            .get("evaluation_records")
            .and_then(Value::as_u64)
            != Some(63_108)
        || oracle_index.get("indexed_datasets").and_then(Value::as_u64) != Some(42)
    {
        bail!("authoritative ASPFM oracle index drift");
    }

    let mut score_sha256 = BTreeMap::new();
    for dataset in &manifest.datasets {
        let selector_path = selector_path(&root, &dataset.dataset_hash);
        let (selector, selector_digest): (execution::SelectorComplete, _) =
            read_receipt(&selector_path)?;
        let shard_path = validation_shard_path(&root, &dataset.dataset_hash);
        let (shard, shard_digest): (DatasetValidationShard, _) = read_receipt(&shard_path)?;
        if validation
            .dataset_validation_sha256
            .get(&dataset.dataset_hash)
            != Some(&shard_digest)
            || shard.replay.selector_sha256 != selector_digest
            || !selector.selection_frozen_before_test_access
        {
            bail!("selector is not covered by the global validation gate");
        }
        let g20_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20-extension-complete.v1.json");
        let (g20, g20_digest): (execution::G20Complete, _) = read_receipt(&g20_path)?;
        if g20_digest != shard.replay.g20_complete_sha256 {
            bail!("G20 receipt is not covered by the validation gate");
        }
        let baseline = baseline_manifest
            .get("entries")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .find(|entry| {
                entry.get("dataset_hash").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("authoritative baseline row is absent")?;
        let oracle = oracle_index
            .get("datasets")
            .and_then(Value::as_array)
            .context("ASPFM oracle index lacks datasets")?
            .iter()
            .find(|entry| {
                entry.get("dataset").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("ASPFM oracle row is absent")?;
        let oracle_source_sha256 = receipt_digest(oracle)?;
        let baselines = baseline_row(
            baseline,
            &dataset.dataset_hash,
            oracle,
            oracle_source_sha256,
        )?;

        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.features != dataset.features || test.features != dataset.features {
            bail!("train/test feature shape drift");
        }
        let variants = frozen_variants(dataset, &g20, &selector)?
            .into_iter()
            .map(|(name, variant)| score_variant(name, &variant, &train, &test))
            .collect::<Result<Vec<_>>>()?;
        if variants.len() != 7
            || variants.last().map(|variant| variant.variant.as_str()) != Some(FINAL_VARIANT)
        {
            bail!("controlled ablation ledger drift");
        }
        let receipt = DatasetScoreReceipt {
            schema: "TrainOnlyRedteamDatasetScore.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_sha256: validation_digest.clone(),
            selector_sha256: selector_digest,
            train_sha256: dataset.train_sha256.clone(),
            test_sha256,
            baseline_manifest_sha256: baseline_digest.clone(),
            oracle_index_sha256: oracle_index_sha256.clone(),
            variants,
            baselines,
            selection_frozen_before_test_access: true,
            development_only: true,
            confirmatory_valid: false,
            test_reuse_disclosure: DISCLOSURE.to_string(),
        };
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("scores.v1.json");
        let digest = write_receipt(&path, &receipt)?;
        score_sha256.insert(dataset.dataset_hash.clone(), digest);
    }
    let receipt = ScoreIndex {
        schema: "TrainOnlyRedteamScoreIndex.v1".to_string(),
        run_id: run_id.to_string(),
        validation_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        oracle_index_sha256,
        dataset_score_sha256: score_sha256,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let path = root.join("scores.v1.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}

fn smart_comparison_baselines(
    entry: &Value,
    dataset_hash: &str,
) -> Result<SmartComparisonBaselines> {
    if entry.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash) {
        bail!("baseline dataset identity drift");
    }
    let stronger_tabfm_method_id = value_string(entry, "/stronger_tabfm_algorithm")?;
    let stronger_source = match stronger_tabfm_method_id.as_str() {
        "tabfm_regressor" => ("/tabfm_report_path", "/tabfm_report_sha256"),
        "tabfm_ensemble_regressor" => (
            "/tabfm_ensemble_report_path",
            "/tabfm_ensemble_report_sha256",
        ),
        _ => bail!("stronger TabFM identity drift"),
    };
    for (path_pointer, hash_pointer) in [
        ("/prior_aspfm_report_path", "/prior_aspfm_report_sha256"),
        ("/tabpfn_v3_report_path", "/tabpfn_v3_report_sha256"),
        stronger_source,
    ] {
        let path =
            execution::relocated_input_path(&PathBuf::from(value_string(entry, path_pointer)?));
        let expected = value_string(entry, hash_pointer)?;
        if sha256_file(&path)? != expected {
            bail!("authoritative baseline source drift");
        }
    }
    Ok(SmartComparisonBaselines {
        prior_train_only_rmse: finite_rmse(entry, "/prior_aspfm_rmse")?,
        tabpfn_v3_rmse: finite_rmse(entry, "/tabpfn_v3_rmse")?,
        stronger_tabfm_method_id,
        stronger_tabfm_rmse: finite_rmse(entry, "/stronger_tabfm_rmse")?,
    })
}

fn regression_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
    let squared_error = sum_squared_error(truth, predictions)?;
    let mae = truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).abs())
        .sum::<f64>()
        / truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let total_squared_error = truth
        .iter()
        .map(|truth| (truth - mean).powi(2))
        .sum::<f64>();
    if total_squared_error <= 0.0 {
        bail!("R2 requires a nonconstant test target");
    }
    Ok((
        (squared_error / truth.len() as f64).sqrt(),
        mae,
        1.0 - squared_error / total_squared_error,
    ))
}

fn rmse_comparison(smart: f64, baseline: f64, tie_relative_tolerance: f64) -> &'static str {
    let tolerance = tie_relative_tolerance * smart.abs().max(baseline.abs());
    if (smart - baseline).abs() <= tolerance {
        "tie"
    } else if smart < baseline {
        "win"
    } else {
        "loss"
    }
}

fn smart_v3_aggregate(
    comparisons: &[(f64, f64)],
    tie_relative_tolerance: f64,
    minimum_wins_required: usize,
) -> Result<SmartV3Aggregate> {
    let aggregate =
        smart_v3_baseline_aggregate(comparisons, tie_relative_tolerance, minimum_wins_required)?;
    Ok(SmartV3Aggregate {
        wins: aggregate.wins,
        ties: aggregate.ties,
        losses: aggregate.losses,
        median_relative_rmse_change: aggregate.median_relative_rmse_change,
        geometric_mean_rmse_ratio: aggregate.geometric_mean_rmse_ratio,
        minimum_wins_required: aggregate.minimum_wins_required,
        smart_v3_improved: aggregate.smart_v3_beats_baseline,
    })
}

fn smart_v3_baseline_aggregate(
    comparisons: &[(f64, f64)],
    tie_relative_tolerance: f64,
    minimum_wins_required: usize,
) -> Result<SmartV3BaselineAggregate> {
    if comparisons.is_empty() || !tie_relative_tolerance.is_finite() || tie_relative_tolerance < 0.0
    {
        bail!("Smart-v3 aggregate inputs are invalid");
    }
    let mut wins = 0;
    let mut ties = 0;
    let mut losses = 0;
    let mut changes = Vec::with_capacity(comparisons.len());
    let mut log_ratios = Vec::with_capacity(comparisons.len());
    for &(smart, prior) in comparisons {
        if !smart.is_finite() || !prior.is_finite() || smart <= 0.0 || prior <= 0.0 {
            bail!("Smart-v3 RMSE ratios require positive finite values");
        }
        let tolerance = tie_relative_tolerance * smart.abs().max(prior.abs());
        if (smart - prior).abs() <= tolerance {
            ties += 1;
        } else if smart < prior {
            wins += 1;
        } else {
            losses += 1;
        }
        changes.push((smart - prior) / prior);
        log_ratios.push((smart / prior).ln());
    }
    let geometric_mean_rmse_ratio =
        (log_ratios.iter().sum::<f64>() / log_ratios.len() as f64).exp();
    Ok(SmartV3BaselineAggregate {
        wins,
        ties,
        losses,
        median_relative_rmse_change: median(changes)?,
        geometric_mean_rmse_ratio,
        minimum_wins_required,
        smart_v3_beats_baseline: wins >= minimum_wins_required && geometric_mean_rmse_ratio < 1.0,
    })
}

fn smart_selector_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("smart-v3-selector-complete.v1.json")
}

fn validate_smart_selectors(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
) -> Result<Vec<(execution::SmartV3SelectorComplete, String, String)>> {
    let selector_files = fs::read_dir(root.join("datasets"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join("smart-v3-selector-complete.v1.json"))
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    if selector_files.len() != DATASET_COUNT || manifest.datasets.len() != DATASET_COUNT {
        bail!("Smart-v3 scoring requires exactly ten frozen selector receipts");
    }
    let mut selectors = Vec::with_capacity(DATASET_COUNT);
    let mut nodes = BTreeSet::new();
    for dataset in &manifest.datasets {
        let path = smart_selector_path(root, &dataset.dataset_hash);
        let (selector, selector_digest): (execution::SmartV3SelectorComplete, _) =
            read_receipt(&path)?;
        let smart_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart-v3-complete.v1.json");
        let (smart, smart_digest): (execution::SmartComplete, _) = read_receipt(&smart_path)?;
        if selector.schema != "TrainOnlyRedteamSmartV3SelectorComplete.v1"
            || selector.run_id != manifest.run_id
            || selector.dataset_hash != dataset.dataset_hash
            || selector.smart_complete_sha256 != smart_digest
            || selector.candidates.is_empty()
            || !selector.selection_frozen_before_test_access
            || !selector.development_only
            || selector.confirmatory_valid
            || smart.schema != "TrainOnlyRedteamSmartComplete.v1"
            || smart.run_id != manifest.run_id
            || smart.dataset_hash != dataset.dataset_hash
            || !smart.development_only
            || smart.confirmatory_valid
        {
            bail!("Smart-v3 selector receipt drift");
        }
        let execution = selector
            .scq_execution
            .as_ref()
            .context("Smart-v3 selector lacks SCQ execution provenance")?;
        let expected_node = &manifest.config.scq.evaluation_ownership[&dataset.dataset_hash];
        if &execution.expected_node != expected_node
            || execution.scheduler_reported_node != execution.expected_node
            || execution.scq_job_id.is_empty()
            || execution.workers != manifest.config.scq.evaluation_cpus
            || execution.cpus_reserved != manifest.config.scq.evaluation_cpus
            || execution.ram_gb != manifest.config.scq.evaluation_ram_gb
            || !execution.non_preemptible
            || execution.max_runtime_hours != manifest.config.scq.max_runtime_hours
        {
            bail!("Smart-v3 selector SCQ provenance drift");
        }
        let assigned_node = execution.expected_node.clone();
        nodes.insert(assigned_node.clone());
        selectors.push((selector, selector_digest, assigned_node));
    }
    let expected_nodes = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if nodes != expected_nodes {
        bail!("Smart-v3 selectors do not cover all four assigned SCQ nodes");
    }
    Ok(selectors)
}

pub(super) fn score_smart_v3_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    let root = run_root(study_root, run_id)?;
    let (manifest, manifest_digest) =
        execution::validate_manifest(&root.join("manifest.redteam.v1.json"))?;

    // Freeze and validate all ten selectors before constructing any baseline
    // or test path. Evolution jobs therefore have no code path to either.
    let selectors = validate_smart_selectors(&root, &manifest)?;

    let baseline_path = manifest
        .parent
        .manifest_path
        .parent()
        .context("parent manifest has no run root")?
        .join("baselines.v2.json");
    let (baseline_manifest, baseline_digest): (Value, _) = read_receipt(&baseline_path)?;
    let baseline_entries = baseline_manifest
        .get("entries")
        .and_then(Value::as_array)
        .context("authoritative baseline manifest lacks entries")?;
    if baseline_manifest.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyBaselineManifest.v2")
        || baseline_entries.len() != DATASET_COUNT
    {
        bail!("authoritative baseline manifest drift");
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .ok();
    let selector_by_dataset = selectors
        .into_iter()
        .map(|(selector, digest, node)| (selector.dataset_hash.clone(), (selector, digest, node)))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::with_capacity(DATASET_COUNT);
    let mut prior_comparisons = Vec::with_capacity(DATASET_COUNT);
    let mut tabpfn_comparisons = Vec::with_capacity(DATASET_COUNT);
    let mut stronger_tabfm_comparisons = Vec::with_capacity(DATASET_COUNT);
    for dataset in &manifest.datasets {
        let (selector, selector_digest, assigned_scq_node) =
            &selector_by_dataset[&dataset.dataset_hash];
        let baseline = baseline_entries
            .iter()
            .find(|entry| {
                entry.get("dataset_hash").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("authoritative baseline row is absent")?;
        let baselines = smart_comparison_baselines(baseline, &dataset.dataset_hash)?;
        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.features != dataset.features || test.features != dataset.features {
            bail!("train/test feature shape drift");
        }
        let predictions = predict_variant(&selector.final_variant, &train, &test)?;
        let (test_rmse, test_mae, test_r2) = regression_metrics(&test.y, &predictions)?;
        if baselines.prior_train_only_rmse <= 0.0
            || baselines.tabpfn_v3_rmse <= 0.0
            || baselines.stronger_tabfm_rmse <= 0.0
        {
            bail!("Smart-v3 comparisons require positive baseline RMSEs");
        }
        let relative_rmse_change =
            (test_rmse - baselines.prior_train_only_rmse) / baselines.prior_train_only_rmse;
        let prior_comparison = rmse_comparison(
            test_rmse,
            baselines.prior_train_only_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        let relative_rmse_change_tabpfn_v3 =
            (test_rmse - baselines.tabpfn_v3_rmse) / baselines.tabpfn_v3_rmse;
        let tabpfn_v3_comparison = rmse_comparison(
            test_rmse,
            baselines.tabpfn_v3_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        let relative_rmse_change_stronger_tabfm =
            (test_rmse - baselines.stronger_tabfm_rmse) / baselines.stronger_tabfm_rmse;
        let stronger_tabfm_comparison = rmse_comparison(
            test_rmse,
            baselines.stronger_tabfm_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        prior_comparisons.push((test_rmse, baselines.prior_train_only_rmse));
        tabpfn_comparisons.push((test_rmse, baselines.tabpfn_v3_rmse));
        stronger_tabfm_comparisons.push((test_rmse, baselines.stronger_tabfm_rmse));
        rows.push(SmartV3TestRow {
            dataset_hash: dataset.dataset_hash.clone(),
            final_variant: selector.final_variant.name.clone(),
            assigned_scq_node: assigned_scq_node.clone(),
            selector_sha256: selector_digest.clone(),
            train_sha256: dataset.train_sha256.clone(),
            test_sha256,
            test_rmse,
            test_mae,
            test_r2,
            previous_train_only_rmse: baselines.prior_train_only_rmse,
            relative_rmse_change,
            tabpfn_v3_rmse: baselines.tabpfn_v3_rmse,
            relative_rmse_change_tabpfn_v3,
            stronger_tabfm_method_id: baselines.stronger_tabfm_method_id,
            stronger_tabfm_rmse: baselines.stronger_tabfm_rmse,
            relative_rmse_change_stronger_tabfm,
            prior_comparison: prior_comparison.to_string(),
            tabpfn_v3_comparison: tabpfn_v3_comparison.to_string(),
            stronger_tabfm_comparison: stronger_tabfm_comparison.to_string(),
            development_only: true,
            confirmatory_valid: false,
        });
    }
    if rows.len() != DATASET_COUNT {
        bail!("Smart-v3 scoring did not produce exactly ten test-result rows");
    }
    let aggregate = smart_v3_aggregate(
        &prior_comparisons,
        manifest.config.scoring.tie_relative_tolerance,
        manifest.config.scoring.success_wins_required,
    )?;
    let baseline_comparisons = SmartV3AggregateComparisons {
        previous_train_only: smart_v3_baseline_aggregate(
            &prior_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
        tabpfn_v3: smart_v3_baseline_aggregate(
            &tabpfn_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
        stronger_tabfm: smart_v3_baseline_aggregate(
            &stronger_tabfm_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
    };
    let results = SmartV3TestResults {
        schema: "TrainOnlyRedteamSmartV3TestResults.v1".to_string(),
        run_id: run_id.to_string(),
        manifest_sha256: manifest_digest,
        baseline_manifest_sha256: baseline_digest,
        selector_count: selector_by_dataset.len(),
        test_result_rows: rows,
        aggregate,
        baseline_comparisons,
        selection_frozen_before_test_access: true,
        test_paths_opened_only_by_this_command: true,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let json_path = root.join("smart-v3-test-results.json");
    write_receipt(&json_path, &results)?;

    let mut csv = String::from(
        "dataset_hash,final_variant,assigned_scq_node,test_rmse,test_mae,test_r2,previous_train_only_rmse,relative_rmse_change,tabpfn_v3_rmse,relative_rmse_change_tabpfn_v3,stronger_tabfm_method_id,stronger_tabfm_rmse,relative_rmse_change_stronger_tabfm,prior_comparison,tabpfn_v3_comparison,stronger_tabfm_comparison,development_only,confirmatory_valid\n",
    );
    for row in &results.test_result_rows {
        csv.push_str(&format!(
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{},{},{},true,false\n",
            row.dataset_hash,
            row.final_variant,
            row.assigned_scq_node,
            row.test_rmse,
            row.test_mae,
            row.test_r2,
            row.previous_train_only_rmse,
            row.relative_rmse_change,
            row.tabpfn_v3_rmse,
            row.relative_rmse_change_tabpfn_v3,
            row.stronger_tabfm_method_id,
            row.stronger_tabfm_rmse,
            row.relative_rmse_change_stronger_tabfm,
            row.prior_comparison,
            row.tabpfn_v3_comparison,
            row.stronger_tabfm_comparison,
        ));
    }
    immutable_write(&root.join("smart-v3-test-results.csv"), csv.as_bytes())?;

    let mut markdown = format!(
        "# Smart-v3 CPU development results\n\n\
         `development_only=true`; `confirmatory_valid=false`.\n\n\
         {}\n\n\
         Smart-v3 improved: `{}`. Acceptance requires at least {}/10 wins over the previous train-only method and a geometric-mean RMSE ratio below 1.0.\n\n\
         Wins/ties/losses: `{}/{}/{}`. Median relative RMSE change: `{:.6}%`. Geometric-mean Smart-v3/prior RMSE ratio: `{:.12}`.\n\n\
         ## Independent baseline comparisons\n\n\
         | Baseline | Strict wins | Ties | Losses | Median relative RMSE change | Geometric-mean Smart-v3/baseline RMSE | Smart-v3 beats baseline |\n\
         |---|---:|---:|---:|---:|---:|---|\n\
         | Previous train-only Jope | {} | {} | {} | {:.6}% | {:.12} | {} |\n\
         | TabPFN v3 | {} | {} | {} | {:.6}% | {:.12} | {} |\n\
         | Stronger per-dataset TabFM | {} | {} | {} | {:.6}% | {:.12} | {} |\n\n\
         | Dataset | SCQ node | Final variant | RMSE | MAE | R² | Previous train-only RMSE | Relative change | TabPFN v3 RMSE | Relative change | Stronger TabFM RMSE | Relative change | Result / TabPFN / TabFM |\n\
         |---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        DISCLOSURE,
        results.aggregate.smart_v3_improved,
        results.aggregate.minimum_wins_required,
        results.aggregate.wins,
        results.aggregate.ties,
        results.aggregate.losses,
        100.0 * results.aggregate.median_relative_rmse_change,
        results.aggregate.geometric_mean_rmse_ratio,
        results.baseline_comparisons.previous_train_only.wins,
        results.baseline_comparisons.previous_train_only.ties,
        results.baseline_comparisons.previous_train_only.losses,
        100.0 * results.baseline_comparisons.previous_train_only.median_relative_rmse_change,
        results.baseline_comparisons.previous_train_only.geometric_mean_rmse_ratio,
        results.baseline_comparisons.previous_train_only.smart_v3_beats_baseline,
        results.baseline_comparisons.tabpfn_v3.wins,
        results.baseline_comparisons.tabpfn_v3.ties,
        results.baseline_comparisons.tabpfn_v3.losses,
        100.0 * results.baseline_comparisons.tabpfn_v3.median_relative_rmse_change,
        results.baseline_comparisons.tabpfn_v3.geometric_mean_rmse_ratio,
        results.baseline_comparisons.tabpfn_v3.smart_v3_beats_baseline,
        results.baseline_comparisons.stronger_tabfm.wins,
        results.baseline_comparisons.stronger_tabfm.ties,
        results.baseline_comparisons.stronger_tabfm.losses,
        100.0 * results.baseline_comparisons.stronger_tabfm.median_relative_rmse_change,
        results.baseline_comparisons.stronger_tabfm.geometric_mean_rmse_ratio,
        results.baseline_comparisons.stronger_tabfm.smart_v3_beats_baseline,
    );
    for row in &results.test_result_rows {
        markdown.push_str(&format!(
            "| {} | {} | {} | {:.12} | {:.12} | {:.12} | {:.12} | {:.6}% | {:.12} | {:.6}% | {:.12} ({}) | {:.6}% | {}/{}/{} |\n",
            row.dataset_hash,
            row.assigned_scq_node,
            row.final_variant,
            row.test_rmse,
            row.test_mae,
            row.test_r2,
            row.previous_train_only_rmse,
            100.0 * row.relative_rmse_change,
            row.tabpfn_v3_rmse,
            100.0 * row.relative_rmse_change_tabpfn_v3,
            row.stronger_tabfm_rmse,
            row.stronger_tabfm_method_id,
            100.0 * row.relative_rmse_change_stronger_tabfm,
            row.prior_comparison,
            row.tabpfn_v3_comparison,
            row.stronger_tabfm_comparison,
        ));
    }
    immutable_write(&root.join("smart-v3-test-results.md"), markdown.as_bytes())?;
    Ok(json_path)
}

fn score_for<'a>(receipt: &'a DatasetScoreReceipt, variant: &str) -> Result<&'a FrozenTestScore> {
    receipt
        .variants
        .iter()
        .find(|score| score.variant == variant)
        .with_context(|| format!("score receipt lacks variant {variant}"))
}

fn median(mut values: Vec<f64>) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("median needs finite values");
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

pub(super) fn report_run(
    study_root: &Path,
    run_id: &str,
    _sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    let (root, manifest, _, _, validation_digest) = require_global_validation(study_root, run_id)?;
    let (index, index_digest): (ScoreIndex, _) = read_receipt(&root.join("scores.v1.json"))?;
    if index.schema != "TrainOnlyRedteamScoreIndex.v1"
        || index.run_id != run_id
        || index.validation_sha256 != validation_digest
        || index.dataset_score_sha256.len() != manifest.datasets.len()
        || !index.development_only
        || index.confirmatory_valid
        || index.test_reuse_disclosure != DISCLOSURE
    {
        bail!("score index drift");
    }
    let mut scores = Vec::new();
    for dataset in &manifest.datasets {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("scores.v1.json");
        let (receipt, digest): (DatasetScoreReceipt, _) = read_receipt(&path)?;
        if index.dataset_score_sha256.get(&dataset.dataset_hash) != Some(&digest)
            || receipt.schema != "TrainOnlyRedteamDatasetScore.v1"
            || receipt.run_id != run_id
            || receipt.dataset_hash != dataset.dataset_hash
            || receipt.validation_sha256 != validation_digest
            || receipt.baseline_manifest_sha256 != index.baseline_manifest_sha256
            || receipt.oracle_index_sha256 != index.oracle_index_sha256
            || receipt.variants.len() != 7
            || !receipt.selection_frozen_before_test_access
            || !receipt.development_only
            || receipt.confirmatory_valid
            || receipt.test_reuse_disclosure != DISCLOSURE
        {
            bail!("dataset score receipt drift");
        }
        scores.push(receipt);
    }

    let mut fair_scores = BTreeMap::new();
    let mut oracle_scores = BTreeMap::new();
    let mut wins = WinCounts {
        tabpfn_v3: 0,
        stronger_tabfm: 0,
        both_tabpfn_and_stronger_tabfm: 0,
        prior_train_only: 0,
        generation_9_control: 0,
        oracle_aspfm_envelope: 0,
    };
    let mut datasets = Vec::new();
    let mut final_values = Vec::new();
    let mut gen9_values = Vec::new();
    for receipt in &scores {
        let final_score = score_for(receipt, FINAL_VARIANT)?;
        let gen9 = score_for(receipt, GEN9_VARIANT)?;
        wins.tabpfn_v3 += usize::from(final_score.rmse < receipt.baselines.tabpfn_v3_rmse);
        wins.stronger_tabfm +=
            usize::from(final_score.rmse < receipt.baselines.stronger_tabfm_rmse);
        wins.both_tabpfn_and_stronger_tabfm += usize::from(
            final_score.rmse < receipt.baselines.tabpfn_v3_rmse
                && final_score.rmse < receipt.baselines.stronger_tabfm_rmse,
        );
        wins.prior_train_only +=
            usize::from(final_score.rmse < receipt.baselines.prior_train_only_rmse);
        wins.generation_9_control += usize::from(final_score.rmse < gen9.rmse);
        wins.oracle_aspfm_envelope +=
            usize::from(final_score.rmse < receipt.baselines.oracle_aspfm_rmse);
        let fair = BTreeMap::from([
            (FINAL_METHOD_ID.to_string(), final_score.rmse),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .prior_train_only
                    .method_id
                    .clone(),
                receipt.baselines.prior_train_only_rmse,
            ),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .tabpfn_v3
                    .method_id
                    .clone(),
                receipt.baselines.tabpfn_v3_rmse,
            ),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .tabfm
                    .method_id
                    .clone(),
                receipt.baselines.stronger_tabfm_rmse,
            ),
        ]);
        let mut oracle = fair.clone();
        oracle.insert(
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .method_id
                .clone(),
            receipt.baselines.oracle_aspfm_rmse,
        );
        fair_scores.insert(receipt.dataset_hash.clone(), fair);
        oracle_scores.insert(receipt.dataset_hash.clone(), oracle);
        final_values.push(final_score.rmse);
        gen9_values.push(gen9.rmse);
        let (selector, _): (execution::SelectorComplete, _) =
            read_receipt(&selector_path(&root, &receipt.dataset_hash))?;
        datasets.push(ReportDataset {
            dataset_hash: receipt.dataset_hash.clone(),
            final_rmse: final_score.rmse,
            generation_9_control_rmse: gen9.rmse,
            prior_train_only_rmse: receipt.baselines.prior_train_only_rmse,
            tabpfn_v3_rmse: receipt.baselines.tabpfn_v3_rmse,
            stronger_tabfm_rmse: receipt.baselines.stronger_tabfm_rmse,
            oracle_aspfm_rmse: receipt.baselines.oracle_aspfm_rmse,
            final_variant: selector.final_variant.name,
            ensemble_accepted: selector.ensemble_accepted,
            development_only: true,
            confirmatory_valid: false,
        });
    }
    let fair_ladder = bradley_terry_elo(&fair_scores, &manifest.config.scoring)?;
    let oracle_reference_ladder = bradley_terry_elo(&oracle_scores, &manifest.config.scoring)?;
    let median_final_rmse = median(final_values)?;
    let median_generation_9_control_rmse = median(gen9_values)?;
    let development_strategy_improved = wins.both_tabpfn_and_stronger_tabfm
        >= manifest.config.scoring.success_wins_required
        && fair_ladder
            .first()
            .is_some_and(|rating| rating.method_id == FINAL_METHOD_ID)
        && median_final_rmse < median_generation_9_control_rmse;
    let labels = BTreeMap::from([
        (
            FINAL_METHOD_ID.to_string(),
            "train-only red-team final method".to_string(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .prior_train_only
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .prior_train_only
                .label
                .clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .tabpfn_v3
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .tabpfn_v3
                .label
                .clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .tabfm
                .method_id
                .clone(),
            manifest.config.authoritative_baselines.tabfm.label.clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .label
                .clone(),
        ),
    ]);
    let report = StudyReport {
        schema: "TrainOnlyRedteamReport.v1".to_string(),
        run_id: run_id.to_string(),
        validation_sha256: validation_digest,
        score_index_sha256: index_digest,
        fair_ladder,
        oracle_reference_ladder,
        wins,
        median_final_rmse,
        median_generation_9_control_rmse,
        development_strategy_improved,
        oracle_comparison_determines_success: false,
        authoritative_method_labels: labels,
        datasets,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let json_path = root.join("report.v1.json");
    write_receipt(&json_path, &report)?;

    let mut markdown = format!(
        "# Train-only red-team development study\n\n\
         `development_only=true`; `confirmatory_valid=false`.\n\n\
         {}\n\n\
         Development strategy improved: `{}`. The oracle-reference comparison is descriptive and does not determine success.\n\n\
         Strict wins — TabPFN v3: {}/10; stronger per-dataset TabFM: {}/10; both on the same dataset: {}/10; prior train-only Jope Dime Regressor: {}/10; current Gen9 control: {}/10; ASPFM oracle envelope: {}/10.\n\n\
         Median final RMSE: `{:.12}`; median current Gen9 control RMSE: `{:.12}`.\n\n\
         ## Fair Bradley–Terry Elo ladder\n\n\
         | Rank | Method | Elo |\n|---:|---|---:|\n",
        DISCLOSURE,
        report.development_strategy_improved,
        report.wins.tabpfn_v3,
        report.wins.stronger_tabfm,
        report.wins.both_tabpfn_and_stronger_tabfm,
        report.wins.prior_train_only,
        report.wins.generation_9_control,
        report.wins.oracle_aspfm_envelope,
        report.median_final_rmse,
        report.median_generation_9_control_rmse,
    );
    for (rank, rating) in report.fair_ladder.iter().enumerate() {
        markdown.push_str(&format!(
            "| {} | {} | {:.6} |\n",
            rank + 1,
            report.authoritative_method_labels[&rating.method_id],
            rating.rating
        ));
    }
    markdown.push_str(
        "\n## Oracle-reference Bradley–Terry Elo ladder\n\n\
         The `jope-dime` row below is the best-test ASPFM envelope over 63,108 Golden42 candidate evaluations; it is not the prior train-only Jope Dime Regressor and cannot determine study success.\n\n\
         | Rank | Method | Elo |\n|---:|---|---:|\n",
    );
    for (rank, rating) in report.oracle_reference_ladder.iter().enumerate() {
        markdown.push_str(&format!(
            "| {} | {} | {:.6} |\n",
            rank + 1,
            report.authoritative_method_labels[&rating.method_id],
            rating.rating
        ));
    }
    markdown.push_str(
        "\n## Per-dataset results\n\n\
         Every row is `development_only=true`, `confirmatory_valid=false`.\n\n\
         | Dataset | Final | Gen9 control | Prior train-only | TabPFN v3 | Stronger TabFM | ASPFM oracle | Final selector |\n\
         |---|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for dataset in &report.datasets {
        markdown.push_str(&format!(
            "| {} | {:.12} | {:.12} | {:.12} | {:.12} | {:.12} | {:.12} | {} |\n",
            dataset.dataset_hash,
            dataset.final_rmse,
            dataset.generation_9_control_rmse,
            dataset.prior_train_only_rmse,
            dataset.tabpfn_v3_rmse,
            dataset.stronger_tabfm_rmse,
            dataset.oracle_aspfm_rmse,
            dataset.final_variant,
        ));
    }
    let markdown_path = root.join("report.v1.md");
    immutable_write(&markdown_path, markdown.as_bytes())?;
    Ok(markdown_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics_match_fixed_fixture() {
        let (rmse, mae, r2) = regression_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]).unwrap();
        assert!((rmse - (1.0_f64 / 3.0).sqrt()).abs() < 1e-15);
        assert!((mae - 1.0 / 3.0).abs() < 1e-15);
        assert!((r2 - 0.5).abs() < 1e-15);
    }

    #[test]
    fn smart_v3_acceptance_uses_wins_and_geometric_mean() {
        let comparisons = [
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (1.0, 1.0),
            (1.05, 1.0),
            (1.05, 1.0),
            (1.05, 1.0),
        ];
        let aggregate = smart_v3_aggregate(&comparisons, 1e-12, 6).unwrap();
        let expected_geometric_mean = (0.9_f64.powi(6) * 1.05_f64.powi(3)).powf(0.1);
        assert_eq!(
            (aggregate.wins, aggregate.ties, aggregate.losses),
            (6, 1, 3)
        );
        assert!((aggregate.median_relative_rmse_change + 0.1).abs() < 1e-15);
        assert!((aggregate.geometric_mean_rmse_ratio - expected_geometric_mean).abs() < 1e-15);
        assert!(aggregate.smart_v3_improved);
    }

    #[test]
    fn geometric_mean_alone_cannot_pass_acceptance() {
        let comparisons = [
            (0.5, 1.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (1.01, 1.0),
            (1.01, 1.0),
            (1.01, 1.0),
            (1.01, 1.0),
            (1.01, 1.0),
        ];
        let aggregate = smart_v3_aggregate(&comparisons, 1e-12, 6).unwrap();
        assert_eq!(aggregate.wins, 5);
        assert!(aggregate.geometric_mean_rmse_ratio < 1.0);
        assert!(!aggregate.smart_v3_improved);
    }

    #[test]
    fn each_baseline_gets_independent_strict_aggregate_metrics() {
        let comparisons = [
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (0.9, 1.0),
            (1.0, 1.0),
            (1.1, 1.0),
            (1.1, 1.0),
            (1.1, 1.0),
        ];
        let aggregate = smart_v3_baseline_aggregate(&comparisons, 1e-12, 6).unwrap();
        assert_eq!(
            (aggregate.wins, aggregate.ties, aggregate.losses),
            (6, 1, 3)
        );
        assert!(aggregate.smart_v3_beats_baseline);
        assert!((aggregate.median_relative_rmse_change + 0.1).abs() < 1e-15);
        let expected = (0.9_f64.powi(6) * 1.1_f64.powi(3)).powf(0.1);
        assert!((aggregate.geometric_mean_rmse_ratio - expected).abs() < 1e-15);
    }
}
