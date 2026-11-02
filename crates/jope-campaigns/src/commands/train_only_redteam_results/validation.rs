use super::*;

fn arm_receipt_path(root: &Path, arm: &str) -> PathBuf {
    root.join(format!("{arm}-complete.v1.json"))
}

pub(super) fn selector_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("selector-complete.v1.json")
}

pub(super) fn validation_shard_path(root: &Path, dataset_hash: &str) -> PathBuf {
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
