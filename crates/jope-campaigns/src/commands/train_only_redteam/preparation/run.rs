use super::*;

pub(crate) fn prepare(args: PrepareArgs) -> Result<PathBuf> {
    if args.workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = run_root(&args.study_root, &args.run_id)?;
    fs::create_dir_all(root.join("inputs"))?;
    fs::create_dir_all(root.join("datasets"))?;
    let manifest_path = root.join("manifest.redteam.v1.json");
    if manifest_path.exists() {
        let (manifest, _): (TrainOnlyRedteamManifest, _) = read_receipt(&manifest_path)?;
        manifest.config.validate()?;
        return Ok(manifest_path);
    }
    let (parent, parent_digest): (Value, _) = read_receipt(&args.parent_run)?;
    if parent.get("schema").and_then(Value::as_str) != Some(PARENT_MANIFEST_SCHEMA)
        || parent.get("campaign_scope").and_then(Value::as_str) != Some("full")
        || parent
            .get("selected_dataset_ids")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(DATASET_COUNT)
    {
        bail!("parent v2 manifest is not the sealed full ten-dataset study");
    }
    let parent_run_id = value_string(&parent, "/run_id")?;
    let config = TrainOnlyRedteamConfig::sealed(&args.run_id, &parent_run_id, args.smoke);
    config.validate()?;
    let config_path = root.join("config.v1.json");
    let config_digest = write_receipt(&config_path, &config)?;

    let validation_path = args
        .parent_run
        .parent()
        .context("parent manifest has no run directory")?
        .join("validation-success.v2.json");
    let (parent_validation, parent_validation_digest): (Value, _) = read_receipt(&validation_path)?;
    if parent_validation.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyValidationReceipt.v2")
        || parent_validation
            .get("benchmark_valid")
            .and_then(Value::as_bool)
            != Some(true)
        || parent_validation
            .get("validated_dataset_count")
            .and_then(Value::as_u64)
            != Some(DATASET_COUNT as u64)
    {
        bail!("parent validation receipt is not complete");
    }

    let scaling = compute_scaling_statistic(&parent)?;
    let scaling_path = root.join("inputs/embedding-sd.v1.json");
    let scaling_digest = write_receipt(&scaling_path, &scaling)?;
    let proposal_dataset_manifest_path = root.join("inputs/proposal-datasets.v1.json");
    proposal_manifest_from_parent(&parent, &proposal_dataset_manifest_path, &args.run_id)?;

    if args.proposals.is_none() {
        let checkpoint_path = PathBuf::from(value_string(&parent, "/checkpoint/checkpoint_path")?);
        let checkpoint_sha256 = value_string(&parent, "/checkpoint/checkpoint_sha256")?;
        if sha256_file(&checkpoint_path)? != checkpoint_sha256 {
            bail!("parent checkpoint bytes drift");
        }
        let request = ProposalRequest {
            schema: "TrainOnlyRedteamProposalRequest.v1".to_string(),
            run_id: args.run_id,
            parent_manifest_sha256: parent_digest,
            checkpoint_path: absolute(&checkpoint_path)?,
            checkpoint_sha256,
            catalog_pointer_path: PathBuf::from(value_string(&parent, "/catalog_pointer_path")?),
            proposal_dataset_manifest_path: absolute(&proposal_dataset_manifest_path)?,
            scaling_statistic_path: absolute(&scaling_path)?,
            scaling_statistic_sha256: scaling_digest,
            exact_decoder_seeds: config.proposal.exact_decoder_seeds,
            jitter_decoder_seeds: config.proposal.jitter_decoder_seeds,
            perturbation_seeds: config.proposal.perturbation_seeds,
            proposals_per_invocation: 64,
            expected_combined_path: absolute(&root.join("inputs/proposals.v1.jsonl"))?,
            development_only: true,
            confirmatory_valid: false,
        };
        let path = root.join("proposal-request.v1.json");
        write_receipt(&path, &request)?;
        return Ok(path);
    }
    let proposals = parse_proposal_bundle(
        args.proposals.as_deref().unwrap(),
        &config,
        &scaling_digest,
        &parent,
    )?;

    let parent_datasets = parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks dataset entries")?;
    let mut by_dataset = parent_datasets
        .iter()
        .map(|entry| Ok((value_string(entry, "/dataset_hash")?, entry)))
        .collect::<Result<HashMap<_, _>>>()?;
    let parent_root = args
        .parent_run
        .parent()
        .context("parent manifest has no run directory")?;
    let mut datasets = Vec::with_capacity(config.dataset_ids.len());
    let training_registry = absolute(&args.sotarena_root.join("data/tabular-regression-v1"))?;
    for dataset_hash in &config.dataset_ids {
        let entry = by_dataset
            .remove(dataset_hash)
            .with_context(|| format!("parent lacks {dataset_hash}"))?;
        let train_path = PathBuf::from(value_string(entry, "/train_path")?);
        let expected_train_path = training_registry.join(dataset_hash).join("train.csv");
        if absolute(&train_path)? != expected_train_path {
            bail!("parent training path is outside the authoritative registry");
        }
        let train_sha256 = value_string(entry, "/train_sha256")?;
        if sha256_file(&train_path)? != train_sha256 {
            bail!("parent training bytes drift for {dataset_hash}");
        }
        let data = read_regression_csv(&train_path)?;
        let bank = build_fold_bank(&args.run_id, dataset_hash, &train_sha256, &data.y, &config)?;
        let dataset_root = root.join("datasets").join(dataset_hash);
        fs::create_dir_all(&dataset_root)?;
        let fold_bank_path = dataset_root.join("fold-banks.v1.json");
        let fold_bank_digest = write_receipt(&fold_bank_path, &bank)?;
        let transfer_ranking_path = PathBuf::from(value_string(entry, "/transfer_ranking_path")?);
        let transfer_ranking_sha256 = value_string(entry, "/transfer_ranking_sha256")?;
        let mut parent_generation_paths = Vec::with_capacity(10);
        let mut parent_generation_sha256 = Vec::with_capacity(10);
        for generation in 0..10 {
            let path = parent_root
                .join("datasets")
                .join(dataset_hash)
                .join("generations")
                .join(format!("generation-{generation:03}.v2.json"));
            let (_, digest): (Value, _) = read_receipt(&path)?;
            parent_generation_paths.push(absolute(&path)?);
            parent_generation_sha256.push(digest);
        }
        datasets.push(DatasetInput {
            dataset_hash: dataset_hash.clone(),
            train_path: absolute(&train_path)?,
            train_sha256,
            rows: data.y.len(),
            features: data.features,
            context_sha256: value_string(entry, "/encoding/context_sha256")?,
            transfer_ranking_path: absolute(&transfer_ranking_path)?,
            transfer_ranking_sha256,
            parent_generation_paths,
            parent_generation_sha256,
            fold_bank_path: absolute(&fold_bank_path)?,
            fold_bank_sha256: fold_bank_digest,
        });
    }
    let manifest = TrainOnlyRedteamManifest {
        schema: "TrainOnlyRedteamManifest.v1".to_string(),
        run_id: args.run_id,
        config,
        config_sha256: config_digest,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
        parent: ParentLink {
            run_id: parent_run_id,
            manifest_path: absolute(&args.parent_run)?,
            manifest_sha256: parent_digest,
            validation_path: absolute(&validation_path)?,
            validation_sha256: parent_validation_digest,
        },
        proposal_bundle: proposals,
        scaling_statistic_path: absolute(&scaling_path)?,
        scaling_statistic_sha256: scaling_digest,
        datasets,
        firewall: SelectionFirewall {
            selection_workers_receive_training_only: true,
            contains_test_paths: false,
            contains_baseline_paths: false,
            test_paths_constructed: false,
            baseline_paths_constructed: false,
        },
    };
    let serialized = String::from_utf8(canonical_json_bytes(&serde_json::to_value(&manifest)?)?)?
        .to_ascii_lowercase();
    for forbidden in [
        "/test.csv",
        "\"test_sha256\"",
        "\"baseline_path\"",
        "\"metrics/rmse\"",
    ] {
        if serialized.contains(forbidden) {
            bail!("selection manifest contains forbidden material: {forbidden}");
        }
    }
    write_receipt(&manifest_path, &manifest)?;
    Ok(manifest_path)
}
