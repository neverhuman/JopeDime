use super::*;

pub(super) fn run_replay(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    validate_native_elf(&args.evaluator, "evaluator")?;
    ensure!(!qualified.is_empty(), "no candidates qualify for replay");
    let candidate_map = candidate_map(request);
    let mut semantic_rows = BTreeMap::<String, String>::new();
    for row in qualified {
        let candidate = candidate_map
            .get(&(row.dataset_hash.as_str(), row.semantic_sha256.as_str()))
            .context("qualified candidate body disappeared")?;
        semantic_rows
            .entry(row.semantic_sha256.clone())
            .or_insert_with(|| candidate.canonical_genome.clone());
    }

    let replay_root = cache_root(args).join("replays").join(request_sha256);
    let final_receipt = replay_root.join("replay-receipt.json");
    if final_receipt.is_file() {
        let existing: ReplayReceipt = serde_json::from_value(read_json(&final_receipt)?)?;
        validate_replay_receipt(&existing, request, request_sha256, panel, &semantic_rows)?;
        return Ok(json!({
            "mode": "replay",
            "idempotent": true,
            "receipt": final_receipt,
            "receipt_sha256": sha256_file(&final_receipt)?,
            "complete": existing.complete,
            "observed_receipts": existing.observed_receipts,
            "valid_receipts": existing.valid_receipts,
        }));
    }
    fs::create_dir_all(&replay_root)?;
    let manifest_root = replay_root.join("manifests");
    let output_root = replay_root.join("outputs");
    fs::create_dir_all(&manifest_root)?;
    fs::create_dir_all(&output_root)?;
    let evaluator_sha256 = sha256_file(&args.evaluator)?;
    let candidates = semantic_rows
        .iter()
        .map(|(semantic, genome)| json!({"semantic_sha256": semantic, "canonical_genome": genome}))
        .collect::<Vec<_>>();
    let checkpoint_epoch = format!("checkpoint-{}", request.checkpoint_sha256);
    let generated_at_ms = now_ms();
    let metric = if request.task == "binary" {
        "auc"
    } else {
        "r2"
    };
    let mut receipt_files = Vec::new();
    let mut observed = 0usize;
    let mut valid = 0usize;
    for dataset in panel {
        verify_dataset_files(dataset)?;
        let manifest = json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "task": request.task,
            "dataset_hash": dataset.dataset_hash,
            "train_csv": dataset.train_csv,
            "test_csv": dataset.test_csv,
            "train_sha256": dataset.train_sha256,
            "test_sha256": dataset.test_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_sha256": request.checkpoint_sha256,
            "catalog_sha256": request.catalog_sha256,
            "protocol_sha256": dataset.protocol_sha256,
            "generated_at_ms": generated_at_ms,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": metric,
                "test_adaptive": true,
                "benchmark_valid": false,
                "selection_bias": "test-exact-reused",
                "generation": request.generation,
                "frozen_seed_sha256": request.frozen_seed_sha256,
                "model_checkpoint_sha256": request.checkpoint_sha256,
                "hall_of_fame_sha256": request.hall_of_fame_sha256,
            },
            "candidates": candidates,
        });
        let manifest_path = manifest_root.join(format!("{}.json", dataset.dataset_hash));
        atomic_json(&manifest_path, &manifest)?;
        let output_dir = output_root.join(&dataset.dataset_hash);
        let output = Command::new(&args.evaluator)
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--threads")
            .arg(args.replay_threads.to_string())
            .arg("--timeout-seconds")
            .arg(args.timeout_seconds.to_string())
            .output()
            .with_context(|| format!("execute native evaluator {}", args.evaluator.display()))?;
        if !output.status.success() {
            bail!(
                "native replay failed for {}: {}",
                dataset.dataset_hash,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let receipt_path = output_dir.join(format!("{}.receipts.jsonl", dataset.dataset_hash));
        let (rows, valid_rows) = validate_replay_jsonl(
            &receipt_path,
            request,
            dataset,
            semantic_rows.keys().map(String::as_str).collect(),
        )?;
        observed += rows;
        valid += valid_rows;
        receipt_files.push(ReplayFile {
            dataset_hash: dataset.dataset_hash.clone(),
            path: receipt_path.display().to_string(),
            sha256: sha256_file(&receipt_path)?,
            receipts: rows,
            valid: valid_rows,
        });
    }
    let expected = panel
        .len()
        .checked_mul(semantic_rows.len())
        .context("replay receipt count overflow")?;
    let receipt = ReplayReceipt {
        schema: REPLAY_SCHEMA.to_string(),
        request_sha256: request_sha256.to_owned(),
        task: request.task.clone(),
        checkpoint_sha256: request.checkpoint_sha256.clone(),
        evaluator: args.evaluator.display().to_string(),
        evaluator_sha256,
        panel_size: panel.len(),
        panel_dataset_hashes: panel.iter().map(|row| row.dataset_hash.clone()).collect(),
        semantic_sha256s: semantic_rows.keys().cloned().collect(),
        receipt_files,
        expected_receipts: expected,
        observed_receipts: observed,
        valid_receipts: valid,
        complete: observed == expected,
        created_at_ms: now_ms(),
    };
    ensure!(receipt.complete, "native replay receipt set is incomplete");
    atomic_json(&final_receipt, &serde_json::to_value(&receipt)?)?;
    Ok(json!({
        "mode": "replay",
        "idempotent": false,
        "receipt": final_receipt,
        "receipt_sha256": sha256_file(&final_receipt)?,
        "complete": receipt.complete,
        "expected_receipts": expected,
        "observed_receipts": observed,
        "valid_receipts": valid,
    }))
}

pub(super) fn validate_native_elf(path: &Path, label: &str) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("{label} {} is unavailable", path.display()))?;
    ensure!(metadata.is_file(), "{label} is not a regular file");
    ensure!(
        metadata.permissions().mode() & 0o111 != 0,
        "{label} is not executable"
    );
    let mut bytes = [0u8; 4];
    File::open(path)?.read_exact(&mut bytes)?;
    ensure!(
        bytes == *b"\x7fELF",
        "{label} must be a native ELF; Python and scripts are forbidden"
    );
    Ok(())
}

fn validate_replay_jsonl(
    path: &Path,
    request: &Request,
    dataset: &Dataset,
    expected: BTreeSet<&str>,
) -> Result<(usize, usize)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read replay receipts {}", path.display()))?;
    let mut observed = BTreeSet::new();
    let mut valid = 0usize;
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let row: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
        ensure!(
            row["schema"] == "JopeFlywheelCorpusEvalReceipt.v1",
            "unexpected replay receipt schema"
        );
        ensure!(row["task"] == request.task, "replay task drift");
        ensure!(
            row["dataset_hash"] == dataset.dataset_hash,
            "replay dataset drift"
        );
        ensure!(
            row["train_sha256"] == dataset.train_sha256
                && row["test_sha256"] == dataset.test_sha256,
            "replay dataset content drift"
        );
        ensure!(
            row["protocol_sha256"] == dataset.protocol_sha256,
            "replay protocol drift"
        );
        ensure!(
            row["model_checkpoint_sha256"] == request.checkpoint_sha256,
            "replay checkpoint drift"
        );
        ensure!(
            row["test_adaptive"] == true && row["benchmark_valid"] == false,
            "replay lost adaptive/benchmark-invalid disclosure"
        );
        let semantic = row["semantic_sha256"]
            .as_str()
            .context("replay semantic is missing")?;
        ensure!(
            expected.contains(semantic),
            "replay contains an unexpected semantic"
        );
        ensure!(
            observed.insert(semantic.to_owned()),
            "duplicate replay semantic receipt"
        );
        valid += usize::from(row["valid"] == true);
    }
    ensure!(
        observed.len() == expected.len(),
        "replay receipt count differs from requested semantics"
    );
    Ok((observed.len(), valid))
}

pub(super) fn validate_replay_receipt(
    receipt: &ReplayReceipt,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    semantics: &BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        receipt.schema == REPLAY_SCHEMA,
        "replay receipt schema drift"
    );
    ensure!(
        receipt.request_sha256 == request_sha256
            && receipt.task == request.task
            && receipt.checkpoint_sha256 == request.checkpoint_sha256,
        "replay receipt provenance drift"
    );
    ensure!(
        receipt.complete
            && receipt.panel_size == PANEL_SIZE
            && receipt.panel_dataset_hashes
                == panel
                    .iter()
                    .map(|row| row.dataset_hash.clone())
                    .collect::<Vec<_>>()
            && receipt.semantic_sha256s == semantics.keys().cloned().collect::<Vec<_>>(),
        "replay receipt cohort drift"
    );
    let evaluator = Path::new(&receipt.evaluator);
    validate_native_elf(evaluator, "recorded replay evaluator")?;
    ensure!(
        sha256_file(evaluator)? == receipt.evaluator_sha256,
        "recorded replay evaluator hash drift"
    );
    let datasets = panel
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut observed = 0usize;
    let mut valid = 0usize;
    for row in &receipt.receipt_files {
        let path = Path::new(&row.path);
        ensure!(sha256_file(path)? == row.sha256, "replay file hash drift");
        let dataset = datasets
            .get(row.dataset_hash.as_str())
            .context("replay receipt references a dataset outside the panel")?;
        let (file_rows, file_valid) = validate_replay_jsonl(
            path,
            request,
            dataset,
            semantics.keys().map(String::as_str).collect(),
        )?;
        ensure!(
            file_rows == row.receipts && file_valid == row.valid,
            "replay file aggregate drift"
        );
        observed += file_rows;
        valid += file_valid;
    }
    ensure!(
        observed == receipt.observed_receipts
            && receipt.observed_receipts == receipt.expected_receipts
            && valid == receipt.valid_receipts,
        "replay aggregate counts drift"
    );
    Ok(())
}
