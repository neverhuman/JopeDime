use super::*;

pub(super) fn canonical_manifest_sha256(mut value: Value) -> Result<String> {
    value
        .as_object_mut()
        .context("manifest must be an object")?
        .remove("manifest_sha256");
    Ok(sha256_bytes(&serde_json::to_vec(&value)?))
}

pub(super) fn required_hash(value: &Value, pointer: &str) -> Result<String> {
    let hash = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing {pointer}"))?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{pointer} is not a SHA-256");
    }
    Ok(hash.to_string())
}

pub(super) fn read_value(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

pub(super) fn validate_source_manifest(path: &Path, schema: &str) -> Result<Value> {
    let value = read_value(path)?;
    if value.get("schema_version").and_then(Value::as_str) != Some(schema) {
        bail!("{} has the wrong schema", path.display());
    }
    let embedded = required_hash(&value, "/manifest_sha256")?;
    if canonical_manifest_sha256(value.clone())? != embedded {
        bail!("{} manifest SHA-256 mismatch", path.display());
    }
    Ok(value)
}

pub(super) fn validate_coverage_receipt(
    path: &Path,
    campaign_id: &str,
    manifest_sha256: &str,
    task: &str,
    expected: usize,
) -> Result<String> {
    let receipt = read_value(path)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("coverage.final-receipt.v1")
        || receipt.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
        || receipt
            .get("campaign_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || receipt.get("task").and_then(Value::as_str) != Some(task)
        || receipt.get("expected").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("valid").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("missing").and_then(Value::as_u64) != Some(0)
        || receipt.get("invalid").and_then(Value::as_u64) != Some(0)
        || receipt.get("complete").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(expected)
    {
        bail!("invalid or incomplete coverage receipt {}", path.display());
    }
    let embedded = required_hash(&receipt, "/receipt_sha256")?;
    let mut unsigned = receipt;
    unsigned
        .as_object_mut()
        .context("coverage receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("coverage receipt digest drift at {}", path.display());
    }
    validate_receipt_report_files(&unsigned, expected, true)?;
    Ok(embedded)
}

pub(super) fn coverage_gate(comparator_root: &Path) -> Result<CoverageGate> {
    let mc_manifest =
        validate_source_manifest(Path::new(mc_coverage_manifest_str()), CORPUS_SCHEMA)?;
    if required_hash(&mc_manifest, "/manifest_sha256")? != MC_COVERAGE_MANIFEST_SHA256 {
        bail!("multiclass successor manifest pin drift");
    }
    let ts_manifest = read_value(Path::new(ts_coverage_manifest_str()))?;
    if ts_manifest.get("schema_version").and_then(Value::as_str)
        != Some("timeseries42-forecast-campaign-v10")
        || required_hash(&ts_manifest, "/manifest_sha256")? != TS_COVERAGE_MANIFEST_SHA256
        || canonical_manifest_sha256(ts_manifest)? != TS_COVERAGE_MANIFEST_SHA256
    {
        bail!("timeseries successor manifest pin drift");
    }
    let multiclass_receipt_sha256 = validate_coverage_receipt(
        Path::new(mc_coverage_receipt_str()),
        "mc-union-runnable272-v5",
        MC_COVERAGE_MANIFEST_SHA256,
        "multiclass",
        11_696,
    )?;
    let timeseries_receipt_sha256 = validate_coverage_receipt(
        Path::new(ts_coverage_receipt_str()),
        "timeseries42-forecast3-v10",
        TS_COVERAGE_MANIFEST_SHA256,
        "timeseries",
        126,
    )?;
    let baseline = read_value(Path::new(ts_baseline_receipt_str()))?;
    if baseline.get("schema").and_then(Value::as_str)
        != Some("timeseries42.baseline-all21-final-receipt.v1")
        || baseline.get("expected").and_then(Value::as_u64) != Some(882)
        || baseline.get("completed_numeric").and_then(Value::as_u64) != Some(877)
        || baseline.get("skipped_ineligible").and_then(Value::as_u64) != Some(5)
        || baseline.get("missing").and_then(Value::as_u64) != Some(0)
        || baseline.get("invalid").and_then(Value::as_u64) != Some(0)
        || baseline.get("errored").and_then(Value::as_u64) != Some(0)
        || baseline.get("complete").and_then(Value::as_bool) != Some(true)
        || baseline
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(882)
    {
        bail!("invalid exact 877+5 time-series baseline receipt");
    }
    let baseline_receipt_sha256 = required_hash(&baseline, "/receipt_sha256")?;
    let mut unsigned_baseline = baseline;
    unsigned_baseline
        .as_object_mut()
        .context("baseline receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned_baseline)?) != baseline_receipt_sha256 {
        bail!("time-series baseline receipt digest drift");
    }
    validate_receipt_report_files(&unsigned_baseline, 882, false)?;
    let comparator_set_path = comparator_root.join(COMPARATOR_SET_FILE);
    let comparator_set = read_value(&comparator_set_path)?;
    if comparator_set.get("schema").and_then(Value::as_str) != Some(COMPARATOR_SET_SCHEMA)
        || comparator_set.get("root").and_then(Value::as_str)
            != Some(comparator_root.to_string_lossy().as_ref())
        || comparator_set
            .get("multiclass_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(multiclass_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_baseline_receipt_sha256")
            .and_then(Value::as_str)
            != Some(baseline_receipt_sha256.as_str())
        || comparator_set.get("lock_count").and_then(Value::as_u64) != Some(314)
    {
        bail!(
            "invalid post-coverage comparator set {}",
            comparator_set_path.display()
        );
    }
    let locks = comparator_set
        .get("locks")
        .and_then(Value::as_array)
        .context("comparator set lacks locks")?;
    let identities = locks
        .iter()
        .filter_map(|row| {
            Some((
                row.get("task")?.as_str()?,
                row.get("dataset_hash")?.as_str()?,
            ))
        })
        .collect::<BTreeSet<_>>();
    if locks.len() != 314
        || identities.len() != 314
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("multiclass"))
            .count()
            != 272
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("timeseries"))
            .count()
            != 42
    {
        bail!("post-coverage comparator set membership drift");
    }
    let embedded = required_hash(&comparator_set, "/comparator_set_sha256")?;
    let mut unsigned = comparator_set;
    unsigned
        .as_object_mut()
        .context("comparator set must be an object")?
        .remove("comparator_set_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("post-coverage comparator set digest drift");
    }
    Ok(CoverageGate {
        multiclass_receipt_sha256,
        timeseries_receipt_sha256,
        comparator_set_path: comparator_set_path.display().to_string(),
        comparator_set_sha256: embedded,
    })
}

pub(super) fn rewrite_seed(source: &str, slot: usize) -> Result<SeedMember> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass generation-zero seeds must be pipelines");
    };
    if pipeline.stages.iter().any(|stage| stage.op == "target") {
        bail!("unsafe target transform in generation-zero seed");
    }
    let head = pipeline
        .stages
        .pop()
        .context("generation-zero seed has no terminal head")?;
    if !feat_math::heads::is_head_op(&head.op) {
        bail!("unknown terminal head in generation-zero seed");
    }
    let mut replacement = Stage::new(MULTICLASS_HEADS[slot % MULTICLASS_HEADS.len()]);
    let l2_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
    match replacement.op.as_str() {
        "multiclass_ridge" => {
            replacement.params.insert(
                "class_weight".to_string(),
                ParamValue::Str("balanced".to_string()),
            );
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_ovr_logistic" => {
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_diagonal_lda" => {
            replacement.params.insert(
                "shrinkage".to_string(),
                ParamValue::Float([0.0, 0.01, 0.1, 0.5][slot % 4]),
            );
        }
        "multiclass_nearest_centroid" => {}
        _ => unreachable!("frozen multiclass head family"),
    }
    replacement.params.insert(
        "feature_budget".to_string(),
        ParamValue::Int(
            FEATURE_BUDGETS[(slot / MULTICLASS_HEADS.len()) % FEATURE_BUDGETS.len()] as i64,
        ),
    );
    pipeline.stages.push(replacement);
    let canonical = GenomeIr::Pipeline(pipeline).canonical();
    let genome = GenomeIr::parse(&canonical)?;
    let schema = DataSchema {
        columns: (0..8)
            .map(|feature| ColumnSpec::numeric(format!("f{feature}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    validate_v3_genome(&genome, &TaskSpec::multiclass(Objective::MacroF1)?, &schema)?;
    Ok(SeedMember {
        slot,
        semantic_sha256: sha256_bytes(canonical.as_bytes()),
        canonical_genome: canonical,
        source:
            "non-test-adaptive generation-zero transformation body; native multiclass head rewrite"
                .to_string(),
    })
}

pub(super) fn seed_population(path: &Path) -> Result<Vec<SeedMember>> {
    let mut seen = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut members = Vec::new();
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut sources = Vec::new();
    for source_path in [path, fallback.as_path()] {
        let content = fs::read_to_string(source_path)?;
        sources.extend(
            content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string),
        );
    }
    for rewrite_variant in 0..24 {
        for (source_index, line) in sources.iter().enumerate() {
            let Ok(mut member) = rewrite_seed(line, source_index + rewrite_variant) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_head_family(&member.canonical_genome)?;
            let count = family_counts.entry(family).or_default();
            if *count >= POPULATION / MULTICLASS_HEADS.len() {
                continue;
            }
            *count += 1;
            member.slot = members.len();
            members.push(member);
            if members.len() == POPULATION {
                break;
            }
        }
        if members.len() == POPULATION {
            break;
        }
    }
    if members.len() != POPULATION {
        bail!(
            "multiclass generation-zero requires exactly {POPULATION} members, found {}",
            members.len()
        );
    }
    Ok(members)
}

pub(super) fn multiclass_head_family(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member must be a pipeline");
    };
    let head = pipeline
        .stages
        .last()
        .context("multiclass member has no head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass member has incompatible head {}", head.op);
    }
    Ok(head.op.clone())
}

pub(super) fn seed_sources_sha256(path: &Path) -> Result<String> {
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut bytes = fs::read(path)?;
    bytes.push(0);
    bytes.extend_from_slice(&fs::read(fallback)?);
    Ok(sha256_bytes(&bytes))
}

pub(super) fn dataset_entries(trusted: &Value) -> Result<BTreeMap<String, Value>> {
    let rows = trusted
        .get("datasets")
        .and_then(Value::as_array)
        .context("trusted manifest lacks datasets")?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("dataset_hash")?.as_str()?.to_string(), row.clone())))
        .collect())
}

pub(super) fn bind_frozen_comparator(
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

pub(super) fn build_plan(
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

pub(super) fn run_root(campaign_root: &Path, run_id: &str) -> PathBuf {
    campaign_root.join(run_id)
}

pub(super) fn plan(
    run_id: &str,
    campaign_root: &Path,
    corpus_manifest: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<()> {
    let root = run_root(campaign_root, run_id);
    let path = root.join("plan.v1.json");
    let plan = if path.exists() {
        let existing = read_plan(campaign_root, run_id)?;
        if existing.corpus_manifest != corpus_manifest.display().to_string()
            || existing.comparator_root != comparator_root.display().to_string()
            || existing.seed_catalog_sha256 != seed_sources_sha256(seed_catalog)?
            || existing.scq_binary != scq_str()
        {
            bail!(
                "immutable multiclass plan input drift at {}",
                path.display()
            );
        }
        existing
    } else {
        let plan = build_plan(run_id, corpus_manifest, comparator_root, seed_catalog)?;
        let bytes = canonical_json(&plan)?;
        immutable_write(&path, &bytes)?;
        plan
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "plan": path,
            "datasets": plan.datasets.len(),
            "priority": PRIORITY_DATASETS,
            "tail": plan.datasets.len() - PRIORITY_DATASETS,
            "fast_baseline_released": plan.datasets.iter().filter(|row| row.fast_baselines_complete).count(),
            "blocked_on_fast_baselines": plan.datasets.iter().filter(|row| !row.fast_baselines_complete).count(),
        }))?
    );
    Ok(())
}

pub(super) fn read_plan(campaign_root: &Path, run_id: &str) -> Result<Plan> {
    let path = run_root(campaign_root, run_id).join("plan.v1.json");
    let plan: Plan = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("read {}", path.display()))?;
    let gate = coverage_gate(Path::new(&plan.comparator_root))?;
    if plan.schema != PLAN_SCHEMA
        || plan.run_id != RUN_ID
        || !plan.test_adaptive
        || plan.selection_bias != "test-selected"
        || plan.benchmark_valid
        || plan.official_combined_rank_eligible
        || plan.config != Config::fixed()
        || plan.datasets.len() != 272
        || plan.seeds.len() != POPULATION
        || plan.corpus_eval != corpus_eval_str()
        || plan.release_binary_sha256 != current_binary_sha256()?
        || plan.scq_binary != scq_str()
        || sha256_file(Path::new(&plan.scq_binary))? != plan.scq_binary_sha256
        || plan.corpus_manifest != mc_coverage_manifest_str()
        || plan.corpus_manifest_sha256 != MC_COVERAGE_MANIFEST_SHA256
        || plan.multiclass_coverage_final_receipt_sha256 != gate.multiclass_receipt_sha256
        || plan.timeseries_coverage_manifest_sha256 != TS_COVERAGE_MANIFEST_SHA256
        || plan.timeseries_coverage_final_receipt_sha256 != gate.timeseries_receipt_sha256
        || plan.comparator_set_path != gate.comparator_set_path
        || plan.comparator_set_sha256 != gate.comparator_set_sha256
        || plan.seed_catalog_sha256
            != seed_sources_sha256(Path::new(
                &plan.seed_catalog.split(" + ").next().unwrap_or(""),
            ))?
        || plan.seed_population_sha256 != sha256_bytes(&serde_json::to_vec(&plan.seeds)?)
        || sha256_file(Path::new(&plan.corpus_eval))? != plan.corpus_eval_sha256
    {
        bail!("multiclass plan identity or policy drift");
    }
    Ok(plan)
}

pub(super) fn ready_dataset_bindings(plan: &Plan, root: &Path) -> Result<Vec<DatasetBinding>> {
    let comparator_root = Path::new(&plan.comparator_root);
    let gate = coverage_gate(comparator_root)?;
    let mut ready = Vec::new();
    for planned in &plan.datasets {
        let Some(binding) = bind_frozen_comparator(planned.clone(), comparator_root, &gate)? else {
            if planned.fast_baselines_complete {
                bail!(
                    "{} planned comparator lock disappeared",
                    planned.dataset_hash
                );
            }
            continue;
        };
        if binding != *planned {
            bail!("{} planned comparator drift", planned.dataset_hash);
        }
        let path = root
            .join("datasets")
            .join(&binding.dataset_hash)
            .join("comparator-binding.v1.json");
        let bytes = canonical_json(&binding)?;
        if path.is_file() {
            if fs::read(&path)? != bytes {
                bail!(
                    "immutable multiclass comparator binding collision at {}",
                    path.display()
                );
            }
        } else {
            immutable_write(&path, &bytes)?;
        }
        ready.push(binding);
    }
    Ok(ready)
}
