use super::*;

pub(crate) fn bind_frozen_comparator(
    mut binding: DatasetBinding,
    comparator_root: &Path,
    gate: &CoverageGate,
) -> Result<Option<DatasetBinding>> {
    let dataset_hash = binding.dataset_hash.as_str();
    let comparator_path = comparator_root
        .join("multiclass")
        .join(dataset_hash)
        .join("comparator.json");
    if !comparator_path.is_file() {
        return Ok(None);
    }
    let comparator = read_value(&comparator_path)?;
    if comparator.get("schema").and_then(Value::as_str) != Some(COMPARATOR_SCHEMA)
        || comparator.get("task").and_then(Value::as_str) != Some("multiclass")
        || comparator.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || comparator.get("run_id").and_then(Value::as_str) != Some(RUN_ID)
        || comparator.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || comparator.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || comparator
            .get("coverage_campaign_id")
            .and_then(Value::as_str)
            != Some("mc-union-runnable272-v5")
        || comparator
            .get("coverage_manifest_sha256")
            .and_then(Value::as_str)
            != Some(MC_COVERAGE_MANIFEST_SHA256)
        || comparator
            .get("coverage_final_receipt_sha256")
            .and_then(Value::as_str)
            != Some(gate.multiclass_receipt_sha256.as_str())
        || comparator
            .get("accepted_report_count")
            .and_then(Value::as_u64)
            != Some(43)
        || comparator
            .get("accepted_reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(43)
    {
        bail!("{dataset_hash} comparator identity drift");
    }
    let protocol_sha256 = required_hash(&comparator, "/protocol_sha256")?;
    let current_evaluator_sha256 = sha256_file(Path::new(corpus_eval_str()))?;
    if comparator
        .pointer("/protocol/protocol_sha256")
        .and_then(Value::as_str)
        != Some(protocol_sha256.as_str())
        || comparator
            .pointer("/protocol/evaluator/sha256")
            .and_then(Value::as_str)
            != Some(current_evaluator_sha256.as_str())
        || comparator
            .pointer("/protocol/minimum_generations")
            .and_then(Value::as_u64)
            != Some(MIN_GENERATIONS as u64)
        || comparator
            .pointer("/protocol/maximum_generations")
            .and_then(Value::as_u64)
            != Some(MAX_GENERATIONS as u64)
        || comparator
            .pointer("/protocol/population")
            .and_then(Value::as_u64)
            != Some(POPULATION as u64)
    {
        bail!("{dataset_hash} comparator protocol drift");
    }
    let mut protocol = comparator
        .get("protocol")
        .cloned()
        .context("comparator lacks protocol")?;
    protocol
        .as_object_mut()
        .context("comparator protocol must be an object")?
        .remove("protocol_sha256");
    if sha256_bytes(&serde_json::to_vec(&protocol)?) != protocol_sha256 {
        bail!("{dataset_hash} comparator protocol digest drift");
    }
    let embedded = required_hash(&comparator, "/lock_sha256")?;
    let mut unsigned = comparator.clone();
    unsigned
        .as_object_mut()
        .context("comparator must be an object")?
        .remove("lock_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("{dataset_hash} comparator digest drift");
    }
    let locked_files = comparator
        .get("dataset_file_hashes")
        .and_then(Value::as_object)
        .context("comparator lacks dataset_file_hashes")?;
    if locked_files.get("train.csv").and_then(Value::as_str) != Some(binding.train_sha256.as_str())
        || locked_files.get("test.csv").and_then(Value::as_str)
            != Some(binding.test_sha256.as_str())
        || locked_files.get("meta.json").and_then(Value::as_str)
            != Some(binding.meta_sha256.as_str())
    {
        bail!("{dataset_hash} comparator input binding drift");
    }
    if sha256_file(Path::new(&binding.train_path))? != binding.train_sha256
        || sha256_file(Path::new(&binding.test_path))? != binding.test_sha256
        || sha256_file(Path::new(&binding.meta_path))? != binding.meta_sha256
    {
        bail!("{dataset_hash} frozen dataset input drift");
    }
    let winner = comparator
        .get("winner")
        .and_then(Value::as_object)
        .context("comparator lacks winner")?;
    let strongest = Incumbent {
        method: winner
            .get("model")
            .and_then(Value::as_str)
            .context("comparator winner lacks model")?
            .to_string(),
        macro_f1: winner
            .get("primary")
            .and_then(Value::as_f64)
            .context("comparator winner lacks primary score")?,
        accuracy: winner
            .get("secondary")
            .and_then(Value::as_f64)
            .context("comparator winner lacks secondary score")?,
        report_path: winner
            .get("report_path")
            .and_then(Value::as_str)
            .context("comparator winner lacks report path")?
            .to_string(),
        report_sha256: required_hash(&comparator, "/winner/report_sha256")?,
    };
    binding.fast_baselines_complete = true;
    binding.completed_incumbent_count = comparator
        .get("accepted_reports")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    binding.comparator_sha256 = embedded;
    binding.strongest_available_incumbent = Some(strongest);
    Ok(Some(binding))
}

pub(crate) fn build_plan(
    run_id: &str,
    corpus_path: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<Plan> {
    validate_run_id(run_id)?;
    if run_id != RUN_ID {
        bail!("multiclass campaign run-id is frozen as {RUN_ID}");
    }
    let gate = coverage_gate(comparator_root)?;
    let corpus = validate_source_manifest(corpus_path, CORPUS_SCHEMA)?;
    let corpus_sha = required_hash(&corpus, "/manifest_sha256")?;
    if corpus_path != Path::new(mc_coverage_manifest_str())
        || corpus_sha != MC_COVERAGE_MANIFEST_SHA256
    {
        bail!("multiclass campaign must use the pinned v5 coverage manifest");
    }
    let rows = corpus
        .get("datasets")
        .and_then(Value::as_array)
        .context("corpus manifest lacks datasets")?;
    let execution_order = rows
        .iter()
        .map(|row| {
            row.get("dataset_hash")
                .and_then(Value::as_str)
                .map(str::to_string)
                .context("corpus dataset lacks dataset_hash")
        })
        .collect::<Result<Vec<_>>>()?;
    if execution_order.len() != 272 || execution_order.iter().collect::<BTreeSet<_>>().len() != 272
    {
        bail!("multiclass corpus must contain exactly 272 unique dataset hashes");
    }
    let entries = dataset_entries(&corpus)?;
    let seeds = seed_population(seed_catalog)?;
    let mut datasets = Vec::new();
    let mut comparator_hashes = Vec::new();
    for (order, dataset_hash) in execution_order.iter().enumerate() {
        let entry = entries
            .get(dataset_hash)
            .with_context(|| format!("corpus manifest lacks {dataset_hash}"))?;
        let root = Path::new(data_root_str()).join(dataset_hash);
        let train_path = root.join("train.csv");
        let test_path = root.join("test.csv");
        let meta_path = root.join("meta.json");
        let train_sha = required_hash(entry, "/files/train.csv/sha256")?;
        let test_sha = required_hash(entry, "/files/test.csv/sha256")?;
        let meta_sha = required_hash(entry, "/files/meta.json/sha256")?;
        if sha256_file(&train_path)? != train_sha
            || sha256_file(&test_path)? != test_sha
            || sha256_file(&meta_path)? != meta_sha
        {
            bail!("{dataset_hash} corpus input digest drift");
        }
        let class_count = entry.get("n_classes").and_then(Value::as_u64).unwrap_or(0) as usize;
        if class_count < 3 {
            bail!("{dataset_hash} has fewer than three classes");
        }
        let n_features = entry
            .get("n_features")
            .and_then(Value::as_u64)
            .context("corpus dataset lacks n_features")? as usize;
        let base = DatasetBinding {
            order,
            wave: "all-272".to_string(),
            dataset_hash: dataset_hash.clone(),
            train_path: train_path.display().to_string(),
            train_sha256: train_sha,
            test_path: test_path.display().to_string(),
            test_sha256: test_sha,
            meta_path: meta_path.display().to_string(),
            meta_sha256: meta_sha,
            class_count,
            n_features,
            fast_baselines_complete: false,
            completed_incumbent_count: 0,
            comparator_sha256: String::new(),
            strongest_available_incumbent: None,
        };
        let binding = bind_frozen_comparator(base, comparator_root, &gate)?
            .with_context(|| format!("missing post-coverage comparator for {dataset_hash}"))?;
        comparator_hashes.push((dataset_hash.clone(), binding.comparator_sha256.clone()));
        datasets.push(binding);
    }
    let seed_population_sha256 = sha256_bytes(&serde_json::to_vec(&seeds)?);
    if comparator_hashes.len() != 272 {
        bail!("multiclass comparator binding set is incomplete");
    }
    let comparator_set_sha256 = gate.comparator_set_sha256.clone();
    Ok(Plan {
        schema: PLAN_SCHEMA.to_string(),
        run_id: run_id.to_string(),
        test_adaptive: true,
        selection_bias: "test-selected".to_string(),
        benchmark_valid: false,
        official_combined_rank_eligible: false,
        project: PROJECT.to_string(),
        // New plans must match the live multiclass Priority-10 policy.  The
        // immutable-plan loader still reads older plans as-is; this only
        // changes the value emitted when a new plan is generated.
        project_weight: 10,
        project_max_gpus: 0,
        scq_binary: scq_str().to_string(),
        scq_binary_sha256: sha256_file(Path::new(scq_str()))?,
        release_binary_sha256: current_binary_sha256()?,
        corpus_eval: corpus_eval_str().to_string(),
        corpus_eval_sha256: sha256_file(Path::new(corpus_eval_str()))?,
        corpus_manifest: corpus_path.display().to_string(),
        corpus_manifest_sha256: corpus_sha,
        multiclass_coverage_final_receipt_sha256: gate.multiclass_receipt_sha256,
        timeseries_coverage_manifest_sha256: TS_COVERAGE_MANIFEST_SHA256.to_string(),
        timeseries_coverage_final_receipt_sha256: gate.timeseries_receipt_sha256,
        comparator_root: comparator_root.display().to_string(),
        comparator_set_path: gate.comparator_set_path,
        comparator_set_sha256,
        seed_catalog: format!(
            "{} + {}",
            seed_catalog.display(),
            seed_catalog
                .with_file_name("generation-zero-binary.txt")
                .display()
        ),
        seed_catalog_sha256: seed_sources_sha256(seed_catalog)?,
        seed_population_sha256,
        priority_count: PRIORITY_DATASETS,
        execution_order,
        fast_baselines: FAST_BASELINES
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        config: Config::fixed(),
        seeds,
        datasets,
    })
}
