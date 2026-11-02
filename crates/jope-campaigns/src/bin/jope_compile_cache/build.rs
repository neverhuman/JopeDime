use super::*;

pub(super) fn run_build(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    ensure!(
        !qualified.is_empty(),
        "no candidates qualify for compilation"
    );
    let replay_path = args
        .replay_receipt
        .as_deref()
        .context("--replay-receipt is required with --build")?;
    let replay: ReplayReceipt = serde_json::from_value(read_json(replay_path)?)?;
    let candidate_map = candidate_map(request);
    let semantics = qualified
        .iter()
        .map(|row| {
            let candidate = candidate_map
                .get(&(row.dataset_hash.as_str(), row.semantic_sha256.as_str()))
                .expect("request was validated");
            (
                row.semantic_sha256.clone(),
                candidate.canonical_genome.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    validate_replay_receipt(&replay, request, request_sha256, panel, &semantics)?;
    let replay_receipt_sha256 = sha256_file(replay_path)?;
    let idempotency_key = args
        .idempotency_key
        .as_deref()
        .context("--idempotency-key is required with --build")?;
    ensure!(
        !idempotency_key.trim().is_empty(),
        "idempotency key is empty"
    );

    let datasets = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let runtime_root = cache_root(args).join("shared-runtime");
    let runtime = FittedCppRuntime::new(&runtime_root)?;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut entries = Vec::new();
    for finalist in qualified {
        let dataset = datasets
            .get(finalist.dataset_hash.as_str())
            .context("finalist dataset disappeared")?;
        let candidate = candidate_map
            .get(&(
                finalist.dataset_hash.as_str(),
                finalist.semantic_sha256.as_str(),
            ))
            .context("finalist body disappeared")?;
        let key = compiled_cache_key(
            &request.task,
            &dataset.train_sha256,
            &candidate.semantic_sha256,
            &dataset.protocol_sha256,
            &request.compiler_bundle_sha256,
            &request.target_triple,
        )
        .map_err(anyhow::Error::msg)?;
        let entry = entry_root(args, &key);
        let hit = if entry.join("artifact-manifest.json").is_file() {
            validate_entry(&entry, request, dataset, candidate, &key)?;
            true
        } else {
            build_entry(
                &entry,
                request,
                dataset,
                candidate,
                finalist,
                &key,
                &replay_receipt_sha256,
                &runtime,
            )?;
            false
        };
        hits += usize::from(hit);
        misses += usize::from(!hit);
        let operation = write_operation(
            args,
            request_sha256,
            idempotency_key,
            dataset,
            candidate,
            &key,
            hit,
        )?;
        entries.push(json!({
            "cache_key": key,
            "dataset_hash": dataset.dataset_hash,
            "semantic_sha256": candidate.semantic_sha256,
            "cache_hit": hit,
            "artifact_manifest": entry.join("artifact-manifest.json"),
            "operation_receipt": operation,
        }));
    }
    Ok(json!({
        "schema": "JopeCompileEconomyBuild.v1",
        "mode": "build",
        "request_sha256": request_sha256,
        "replay_receipt": replay_path,
        "replay_receipt_sha256": replay_receipt_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "entries": entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits as f64 / (hits + misses) as f64,
        "cache": cache_status(&cache_root(args))?,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_entry(
    entry: &Path,
    request: &Request,
    dataset: &Dataset,
    candidate: &Candidate,
    finalist: &Qualified,
    key: &str,
    replay_receipt_sha256: &str,
    runtime: &FittedCppRuntime,
) -> Result<()> {
    verify_dataset_files(dataset)?;
    let train = read_numeric_csv(&dataset.train_csv)?;
    let test = read_numeric_csv(&dataset.test_csv)?;
    ensure!(
        train.features == test.features,
        "train/test feature widths differ"
    );
    validate_targets(&request.task, &train.y, "train")?;
    validate_targets(&request.task, &test.y, "test")?;
    let genome = GenomeIr::parse(&candidate.canonical_genome).context("parse finalist genome")?;
    ensure!(
        genome.canonical() == candidate.canonical_genome,
        "finalist genome is not canonical"
    );
    let train_table = raw_table(train.features, &train.x)?;
    let test_table = raw_table(test.features, &test.x)?;
    let model = if request.task == "regression" {
        fit_legacy_numeric_bound(&genome, &train.x, &train.y)?
    } else {
        let task = TaskSpec::binary(Objective::RocAuc)?;
        fit_table(
            &genome,
            &task,
            &numeric_schema(train.features),
            &train_table,
            &train.y,
        )?
    };
    supports_fitted_cpp17(&model)?;
    let codec = encode_fitted_cpp17(&model)?;
    validate_fitted_cpp_codec(&codec.bytes)?;
    ensure!(
        sha256_bytes(&codec.bytes) == codec.sha256,
        "codec identity drift"
    );
    let interpreted = model.predict_table(&test_table)?;
    let encoded_test = EncodedTable::from_bound_model(&model, &test_table)?;
    let compiled_model = runtime.load(&codec.bytes)?;
    let compiled = compiled_model.predict(&encoded_test)?;
    let interpreted_score = score(&request.task, &test.y, &interpreted)?;
    let compiled_score = score(&request.task, &test.y, &compiled)?;
    let parity = exact_parity(&interpreted, &compiled, interpreted_score, compiled_score);
    ensure!(
        parity.qualified,
        "compiled/interpreted prediction or score bit parity failed"
    );
    let interpreted_prediction_sha256 = prediction_sha256(&interpreted);
    let compiled_prediction_sha256 = prediction_sha256(&compiled);
    ensure!(
        interpreted_prediction_sha256 == compiled_prediction_sha256,
        "prediction hashes differ despite parity"
    );

    let parent = entry.parent().context("cache entry has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{key}.tmp-{}", std::process::id()));
    if temporary.exists() {
        bail!("temporary cache entry exists: {}", temporary.display());
    }
    fs::create_dir(&temporary)?;
    let result = (|| -> Result<()> {
        let artifact = temporary.join("fitted-model.codec");
        atomic_bytes(&artifact, &codec.bytes)?;
        let manifest = json!({
            "schema": ARTIFACT_SCHEMA,
            "cache_key": key,
            "task": request.task,
            "dataset_hash": dataset.dataset_hash,
            "dataset_train_sha256": dataset.train_sha256,
            "dataset_test_sha256": dataset.test_sha256,
            "semantic_sha256": candidate.semantic_sha256,
            "evaluator_protocol_sha256": dataset.protocol_sha256,
            "compiler_bundle_sha256": request.compiler_bundle_sha256,
            "target_triple": request.target_triple,
            "checkpoint_sha256": request.checkpoint_sha256,
            "artifact": "fitted-model.codec",
            "artifact_sha256": codec.sha256,
            "score": finalist.score,
            "within_dataset_rank": finalist.rank,
            "strict_incumbent_beat": finalist.strict_incumbent_beat,
            "top_two": finalist.top_two,
            "replay_receipt_sha256": replay_receipt_sha256,
            "prediction_bit_parity": parity.prediction_bit_parity,
            "score_bit_parity": parity.score_bit_parity,
            "qualified": parity.qualified,
            "interpreted_score": interpreted_score,
            "compiled_score": compiled_score,
            "interpreted_predictions_sha256": interpreted_prediction_sha256,
            "compiled_predictions_sha256": compiled_prediction_sha256,
            "fitted_state_scope": {
                "task": request.task,
                "dataset_train_sha256": dataset.train_sha256,
                "semantic_sha256": candidate.semantic_sha256,
            },
            "created_at_ms": now_ms(),
        });
        atomic_json(&temporary.join("artifact-manifest.json"), &manifest)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    match fs::rename(&temporary, entry) {
        Ok(()) => {}
        Err(_) if entry.join("artifact-manifest.json").is_file() => {
            fs::remove_dir_all(&temporary)?;
            validate_entry(entry, request, dataset, candidate, key)?;
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error.into());
        }
    }
    validate_entry(entry, request, dataset, candidate, key)
}

pub(super) fn validate_entry(
    entry: &Path,
    request: &Request,
    dataset: &Dataset,
    candidate: &Candidate,
    key: &str,
) -> Result<()> {
    let manifest_path = entry.join("artifact-manifest.json");
    let manifest = read_json(&manifest_path)?;
    ensure!(
        manifest["schema"] == ARTIFACT_SCHEMA,
        "artifact schema drift"
    );
    ensure!(manifest["cache_key"] == key, "artifact cache key drift");
    ensure!(
        manifest["task"] == request.task
            && manifest["dataset_hash"] == dataset.dataset_hash
            && manifest["dataset_train_sha256"] == dataset.train_sha256
            && manifest["semantic_sha256"] == candidate.semantic_sha256
            && manifest["evaluator_protocol_sha256"] == dataset.protocol_sha256
            && manifest["compiler_bundle_sha256"] == request.compiler_bundle_sha256
            && manifest["target_triple"] == request.target_triple,
        "artifact fitted identity drift"
    );
    ensure!(
        manifest["prediction_bit_parity"] == true
            && manifest["score_bit_parity"] == true
            && manifest["qualified"] == true,
        "artifact lacks exact parity qualification"
    );
    let relative = manifest["artifact"]
        .as_str()
        .context("artifact path is missing")?;
    ensure!(
        relative == "fitted-model.codec",
        "unexpected artifact filename"
    );
    let artifact = entry.join(relative);
    let expected = manifest["artifact_sha256"]
        .as_str()
        .context("artifact hash is missing")?;
    ensure!(sha256_file(&artifact)? == expected, "artifact hash drift");
    validate_fitted_cpp_codec(&fs::read(artifact)?)?;
    Ok(())
}
