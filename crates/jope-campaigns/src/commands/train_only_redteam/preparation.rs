use super::*;

pub(super) fn build_repeat(
    purpose: &str,
    repeat: usize,
    seed: u64,
    targets: &[f64],
) -> Result<FoldRepeat> {
    if targets.len() < FOLDS || targets.iter().any(|value| !value.is_finite()) {
        bail!("fold bank needs at least five finite targets");
    }
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut strata = vec![Vec::new(); 10];
    let row_count = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        strata[(rank * 10 / row_count).min(9)].push(row);
    }
    let mut validation_indices = vec![Vec::new(); FOLDS];
    for (stratum, rows) in strata.iter_mut().enumerate() {
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"train-only-redteam-fold-order-v1\0");
            hash.update(seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hash.finalize().to_vec()
        });
        for (position, row) in rows.iter().enumerate() {
            validation_indices[(position + stratum) % FOLDS].push(*row);
        }
    }
    let mut coverage = vec![0_u8; targets.len()];
    for fold in &mut validation_indices {
        fold.sort_unstable();
        if fold.is_empty() {
            bail!("fold bank contains an empty fold");
        }
        for row in fold {
            coverage[*row] = coverage[*row].saturating_add(1);
        }
    }
    if coverage.iter().any(|count| *count != 1) {
        bail!("every row must appear exactly once as validation per repeat");
    }
    let repeat_sha256 = receipt_digest(&json!({
        "purpose": purpose,
        "repeat": repeat,
        "seed": seed,
        "validation_indices": validation_indices,
    }))?;
    Ok(FoldRepeat {
        purpose: purpose.to_string(),
        repeat,
        seed,
        validation_indices,
        repeat_sha256,
    })
}

pub(super) fn build_fold_bank(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
    config: &TrainOnlyRedteamConfig,
) -> Result<FoldBankReceipt> {
    let make = |purpose: &str, seeds: &[u64]| {
        seeds
            .iter()
            .enumerate()
            .map(|(repeat, seed)| build_repeat(purpose, repeat, *seed, targets))
            .collect::<Result<Vec<_>>>()
    };
    let bank = FoldBankReceipt {
        schema: "TrainOnlyRedteamFoldBank.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        rows: targets.len(),
        folds: FOLDS,
        search: make("search", &config.fold_banks.search_seeds)?,
        selector: make("selector", &config.fold_banks.selector_seeds)?,
        audit: make("audit", &config.fold_banks.audit_seeds)?,
    };
    validate_fold_bank(&bank, config)?;
    Ok(bank)
}

pub(super) fn validate_fold_bank(
    bank: &FoldBankReceipt,
    config: &TrainOnlyRedteamConfig,
) -> Result<()> {
    if bank.schema != "TrainOnlyRedteamFoldBank.v1"
        || bank.folds != FOLDS
        || bank.search.len() != 2
        || bank.selector.len() != 2
        || bank.audit.len() != 1
    {
        bail!("fold-bank shape drift");
    }
    let expected = [
        ("search", config.fold_banks.search_seeds.as_slice()),
        ("selector", config.fold_banks.selector_seeds.as_slice()),
        ("audit", config.fold_banks.audit_seeds.as_slice()),
    ];
    for (purpose, seeds) in expected {
        let repeats = match purpose {
            "search" => &bank.search,
            "selector" => &bank.selector,
            _ => &bank.audit,
        };
        for (repeat, expected_seed) in repeats.iter().zip(seeds) {
            if repeat.purpose != purpose || repeat.seed != *expected_seed {
                bail!("fold-bank purpose or seed drift");
            }
            let rebuilt = receipt_digest(&json!({
                "purpose": repeat.purpose,
                "repeat": repeat.repeat,
                "seed": repeat.seed,
                "validation_indices": repeat.validation_indices,
            }))?;
            if rebuilt != repeat.repeat_sha256 {
                bail!("fold-bank repeat digest drift");
            }
            let mut coverage = vec![0_u8; bank.rows];
            for fold in &repeat.validation_indices {
                for row in fold {
                    if *row >= bank.rows {
                        bail!("fold-bank row is out of bounds");
                    }
                    coverage[*row] = coverage[*row].saturating_add(1);
                }
            }
            if coverage.iter().any(|count| *count != 1) {
                bail!("fold-bank coverage drift");
            }
        }
    }
    Ok(())
}

pub(super) fn compute_scaling_statistic(parent: &Value) -> Result<ScalingStatistic> {
    let pointer_path = PathBuf::from(value_string(parent, "/catalog_pointer_path")?);
    let pointer: Value = serde_json::from_slice(&fs::read(&pointer_path)?)?;
    let catalog_path = PathBuf::from(value_string(&pointer, "/catalog")?);
    let data_path = PathBuf::from(value_string(&pointer, "/data")?);
    let catalog_sha256 = value_string(&pointer, "/catalog_sha256")?;
    if sha256_file(&catalog_path)? != catalog_sha256
        || sha256_file(&data_path)? != value_string(&pointer, "/data_sha256")?
    {
        bail!("parent context catalog pointer drift");
    }
    let catalog: Value = serde_json::from_slice(&fs::read(catalog_path)?)?;
    let entries = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("context catalog lacks entries")?;
    let dimensions = value_usize(&catalog, "/dataset_dimensions")?;
    let target_dimensions = value_usize(&catalog, "/target_dimensions")?;
    if dimensions != 1536 || target_dimensions != 512 {
        bail!("context catalog dimensions drift");
    }
    let stride_floats = dimensions + target_dimensions;
    let blob = fs::read(data_path)?;
    let regression = entries
        .iter()
        .filter(|entry| {
            entry.get("task").and_then(Value::as_str) == Some("regression")
                && entry.get("split").and_then(Value::as_str) == Some("train")
        })
        .collect::<Vec<_>>();
    if regression.len() < 2 {
        bail!("scaling statistic needs at least two regression contexts");
    }
    let mut sums = vec![0.0; dimensions];
    let mut squared_sums = vec![0.0; dimensions];
    for entry in &regression {
        let index = value_usize(entry, "/catalog_index")?;
        let offset = index
            .checked_mul(stride_floats * 4)
            .context("context offset overflow")?;
        if offset + stride_floats * 4 > blob.len() {
            bail!("context slice outside frozen blob");
        }
        for coordinate in 0..dimensions {
            let start = offset + coordinate * 4;
            let value = f32::from_le_bytes(blob[start..start + 4].try_into().unwrap()) as f64;
            if !value.is_finite() {
                bail!("non-finite training-context coordinate");
            }
            sums[coordinate] += value;
            squared_sums[coordinate] += value * value;
        }
    }
    let count = regression.len() as f64;
    let population_sd = sums
        .iter()
        .zip(squared_sums)
        .map(|(sum, squared_sum)| {
            let variance = (squared_sum / count - (sum / count).powi(2)).max(0.0);
            variance.sqrt()
        })
        .collect();
    Ok(ScalingStatistic {
        schema: "TrainOnlyEmbeddingScalingStatistic.v1".to_string(),
        source_catalog_sha256: catalog_sha256,
        contexts: regression.len(),
        dimensions,
        population_sd,
    })
}

pub(super) fn proposal_manifest_from_parent(
    parent: &Value,
    path: &Path,
    run_id: &str,
) -> Result<String> {
    let entries = parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?;
    let catalog_sha256 = value_string(parent, "/source_catalog_sha256")?;
    let receipt = json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "run_id": run_id,
        "catalog_sha256": catalog_sha256,
        "entries": entries.iter().map(|entry| json!({
            "task": "regression",
            "dataset_hash": entry.get("dataset_hash").cloned().unwrap_or(Value::Null),
            "context_sha256": entry.pointer("/encoding/context_sha256").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    write_receipt(path, &receipt)
}

pub(super) fn parse_proposal_bundle(
    path: &Path,
    config: &TrainOnlyRedteamConfig,
    scaling_sha256: &str,
    parent: &Value,
) -> Result<ProposalBundleLink> {
    let mut expected_contexts = HashMap::new();
    for entry in parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?
    {
        let dataset = value_string(entry, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            continue;
        }
        let context_sha256 = value_string(entry, "/encoding/context_sha256")?;
        let blob_path = PathBuf::from(value_string(entry, "/encoding/blob_path")?);
        let offset = entry
            .pointer("/encoding/offset_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks offset_bytes")? as usize;
        let length = entry
            .pointer("/encoding/length_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks length_bytes")? as usize;
        if length != (1536 + 512) * std::mem::size_of::<f32>() {
            bail!("parent context has an invalid byte length");
        }
        let blob = fs::read(&blob_path)?;
        let context = blob
            .get(offset..offset + length)
            .context("parent context is outside its frozen blob")?;
        if sha256_bytes(context) != context_sha256 {
            bail!("parent context bytes drift");
        }
        let target_summary_sha256 = sha256_bytes(&context[1536 * std::mem::size_of::<f32>()..]);
        expected_contexts.insert(dataset, (context_sha256, target_summary_sha256));
    }
    if expected_contexts.len() != config.dataset_ids.len() {
        bail!("parent context provenance does not cover the sealed cohort");
    }
    let file = File::open(path)?;
    let mut exact: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    let mut jittered: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let dataset = value_string(&row, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            bail!("proposal bundle contains an unfrozen dataset");
        }
        let (expected_context_sha256, expected_target_summary_sha256) = expected_contexts
            .get(&dataset)
            .context("proposal dataset lacks frozen context provenance")?;
        if row.get("task").and_then(Value::as_str) != Some("regression")
            || row.get("target_summary_unchanged").and_then(Value::as_bool) != Some(true)
            || row.get("context_sha256") != row.get("exact_context_sha256")
            || row.get("exact_context_sha256").and_then(Value::as_str)
                != Some(expected_context_sha256.as_str())
            || row.get("target_summary_sha256").and_then(Value::as_str)
                != Some(expected_target_summary_sha256.as_str())
        {
            bail!("proposal row violates exact-context/target-summary provenance");
        }
        let origin = value_string(&row, "/proposal_origin")?;
        let decoder_seed = row
            .get("invocation_seed")
            .and_then(Value::as_u64)
            .context("proposal lacks invocation_seed")?;
        match origin.as_str() {
            "exact_context" => {
                if !config.proposal.exact_decoder_seeds.contains(&decoder_seed)
                    || row
                        .get("perturbation_seed")
                        .is_some_and(|value| !value.is_null())
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.0)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        != Some(expected_context_sha256.as_str())
                {
                    bail!("exact proposal seed or perturbation metadata drift");
                }
                *exact
                    .entry(dataset)
                    .or_default()
                    .entry(decoder_seed)
                    .or_default() += 1;
            }
            "jittered_context" => {
                let perturbation_seed = row
                    .get("perturbation_seed")
                    .and_then(Value::as_u64)
                    .context("jittered proposal lacks perturbation seed")?;
                let perturbation_index = config
                    .proposal
                    .perturbation_seeds
                    .iter()
                    .position(|seed| *seed == perturbation_seed);
                if perturbation_index.is_none()
                    || config.proposal.jitter_decoder_seeds[perturbation_index.unwrap()]
                        != decoder_seed
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.05)
                    || row
                        .get("embedding_scaling_statistic_sha256")
                        .and_then(Value::as_str)
                        != Some(scaling_sha256)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        == Some(expected_context_sha256.as_str())
                {
                    bail!("jittered proposal provenance drift");
                }
                *jittered
                    .entry(dataset)
                    .or_default()
                    .entry(perturbation_seed)
                    .or_default() += 1;
            }
            _ => bail!("unsupported proposal origin"),
        }
    }
    for dataset in &config.dataset_ids {
        let exact_seeds = exact
            .get(dataset)
            .with_context(|| format!("no exact proposals for {dataset}"))?;
        for seed in &config.proposal.exact_decoder_seeds {
            if exact_seeds.get(seed) != Some(&config.proposal.exact_per_seed) {
                bail!("exact proposal quota drift for {dataset} seed {seed}");
            }
        }
        let jitter_seeds = jittered
            .get(dataset)
            .with_context(|| format!("no jittered proposals for {dataset}"))?;
        for seed in &config.proposal.perturbation_seeds {
            if jitter_seeds.get(seed) != Some(&config.proposal.jitter_per_perturbation) {
                bail!("jittered proposal quota drift for {dataset} seed {seed}");
            }
        }
    }
    Ok(ProposalBundleLink {
        path: absolute(path)?,
        sha256: sha256_file(path)?,
        exact_rows_per_dataset: config.proposal.exact_decoder_seeds.len()
            * config.proposal.exact_per_seed,
        jittered_rows_per_dataset: config.proposal.perturbation_seeds.len()
            * config.proposal.jitter_per_perturbation,
        scaling_statistic_sha256: scaling_sha256.to_string(),
    })
}

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
