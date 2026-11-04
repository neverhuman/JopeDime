pub(crate) fn validate_all(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let expected_coverage = if is_forecast3_gen20(run_id) {
        Some(validate_coverage_binding()?)
    } else {
        None
    };
    let root = campaign_root(run_id);
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.datasets.len() != 42
        || manifest.config != base_campaign_config(run_id)
        || manifest.coverage_binding != expected_coverage
    {
        bail!("invalid TimeSeries42 campaign manifest");
    }
    for spec in &manifest.datasets {
        if is_forecast3_gen20(run_id)
            && (spec.accepted_external_reports.len()
                != BASELINE_MODELS
                    .iter()
                    .filter(|model| !declared_baseline_skip(&spec.dataset_hash, model))
                    .count()
                || spec.comparator_lock_sha256.len() != 64)
        {
            bail!(
                "TimeSeries42 dataset {} is not bound to the forecast3 comparator",
                spec.dataset_hash
            );
        }
        let (bank, _) = load_or_create_seed_bank(&root, spec)?;
        if bank.seeds.len() != POPULATION {
            bail!("TimeSeries42 seed bank changed for {}", spec.dataset_hash);
        }
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 dataset result for {}",
                spec.dataset_hash
            );
        }
        if result.completed_generations == 0
            || result.completed_generations < manifest.config.minimum_generations.max(1)
            || result.completed_generations > manifest.config.generations
            || (!result.strict_winner
                && result.completed_generations != manifest.config.generations)
            || (is_forecast3_gen20(run_id) && result.strict_winner != result.best_beats_official)
            || (is_forecast3_gen20(run_id) && !result.strict_winner && result.status != "exhausted")
        {
            bail!(
                "invalid TimeSeries42 generation accounting for {}",
                spec.dataset_hash
            );
        }
    }
    let prior_scan = root
        .join("terminal-summary.v1.json")
        .is_file()
        .then(|| load_json::<TerminalSummary>(&root.join("terminal-summary.v1.json")).ok())
        .flatten()
        .and_then(|summary| summary.sotarena_scan_output);
    let summary = rebuild_summary(&root, run_id, prior_scan)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 terminal summary accounting mismatch");
    }
    println!(
        "{}",
        json!({
            "schema": "JopeTimeseries42Validation.v1",
            "status": "complete",
            "datasets": summary.dataset_count,
            "wins": summary.wins,
            "losses": summary.losses,
            "ties": summary.ties,
            "exhausted": summary.exhausted,
            "valid_evaluations": summary.valid_evaluations,
        })
    );
    Ok(())
}

pub(crate) fn validate_gen30(base_run_id: &str, run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    if base_run_id == run_id {
        bail!("TimeSeries42 gen30 continuation run id must differ from the base run id");
    }
    let base = load_base_campaign(base_run_id)?;
    let root = campaign_root(run_id);
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.base_run_id.as_deref() != Some(base_run_id)
        || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
        || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
        || manifest.continuation_terminal_generation != Some(GEN30_GENERATIONS - 1)
        || manifest.config != CampaignConfig::gen30()
        || manifest.datasets != base.manifest.datasets
    {
        bail!("invalid TimeSeries42 gen30 campaign manifest");
    }
    for spec in &manifest.datasets {
        let (_bank, _) = load_existing_seed_bank(&base.root, spec)?;
        for generation in 0..GENERATIONS {
            let slots = load_generation_slots(
                &base.root,
                base_run_id,
                &base.campaign_sha256,
                spec,
                generation,
            )?;
            if slots.len() != POPULATION {
                bail!("invalid base slot count for {}", spec.dataset_hash);
            }
        }
        let base_result: DatasetResult =
            load_json(&final_result_path(&base.root, &spec.dataset_hash))?;
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 gen30 dataset result for {}",
                spec.dataset_hash
            );
        }
        validate_gen30_dataset_result(&result, &base_result, base_run_id)?;
    }
    let prior_scan = root
        .join("terminal-summary.v1.json")
        .is_file()
        .then(|| load_json::<TerminalSummary>(&root.join("terminal-summary.v1.json")).ok())
        .flatten()
        .and_then(|summary| summary.sotarena_scan_output);
    let summary = rebuild_summary(&root, run_id, prior_scan)?;
    if summary.base_run_id.as_deref() != Some(base_run_id)
        || summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 gen30 terminal summary accounting mismatch");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "{}",
        json!({
            "schema": "JopeTimeseries42Gen30Validation.v1",
            "status": "complete",
            "base_run_id": base_run_id,
            "run_id": run_id,
            "datasets": summary.dataset_count,
            "wins": summary.wins,
            "losses": summary.losses,
            "ties": summary.ties,
            "exhausted": summary.exhausted,
            "inherited_base_wins": summary.inherited_base_wins.unwrap_or(0),
            "improved_base_wins": summary.improved_base_wins.unwrap_or(0),
            "continuation_strict_wins": summary.continuation_strict_wins.unwrap_or(0),
            "valid_evaluations": summary.valid_evaluations,
            "dataset_reports_written": reports_written,
            "dataset_report_file": JOPE_TIMESERIES_REPORT_FILE,
        })
    );
    Ok(())
}

pub(crate) fn batch_worker(
    manifest_path: &Path,
    output_dir: &Path,
    corpus_eval: &Path,
    timeout_seconds: u64,
) -> Result<()> {
    let started = unix_millis();
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if !matches!(
        manifest.get("schema").and_then(Value::as_str),
        Some(CORPUS_SHARD_SCHEMA | CORPUS_SHARD_SCHEMA_V2)
    ) || manifest.get("task").and_then(Value::as_str) != Some("timeseries")
    {
        bail!("TimeSeries42 worker manifest is not a corpus_eval time-series shard");
    }
    fs::create_dir_all(output_dir)?;
    let status = Command::new(corpus_eval)
        .arg("--manifest")
        .arg(manifest_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--threads")
        .arg(SHARD_SIZE.to_string())
        .arg("--timeout-seconds")
        .arg(timeout_seconds.to_string())
        .status()
        .with_context(|| format!("run corpus_eval {}", corpus_eval.display()))?;
    if !status.success() {
        let failure = json!({
            "schema": "JopeTimeseries42WorkerFailure.v1",
            "manifest_sha256": manifest_sha256,
            "exit_status": status.code(),
            "failed_at_unix_ms": unix_millis(),
        });
        fs::write(
            output_dir.join("batch-worker-failure.v1.json"),
            canonical_json(&failure)?,
        )?;
        bail!("corpus_eval failed with {status}");
    }
    let dataset = manifest
        .get("dataset_hash")
        .and_then(Value::as_str)
        .context("worker manifest lacks dataset_hash")?;
    let receipt_path = output_dir.join(format!("{dataset}.receipts.jsonl"));
    let done_path = output_dir.join(format!("{dataset}.done"));
    let receipt_bytes = fs::read(&receipt_path)?;
    let done_bytes = fs::read(&done_path)?;
    let receipts = read_receipts(&receipt_path)?;
    let worker = WorkerReceipt {
        schema: WORKER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        dataset_hash: dataset.to_string(),
        manifest_sha256,
        receipt_sha256: sha256_bytes(&receipt_bytes),
        done_sha256: sha256_bytes(&done_bytes),
        rows: receipts.len(),
        valid_rows: receipts.iter().filter(|receipt| receipt.valid).count(),
        started_at_unix_ms: started,
        finished_at_unix_ms: unix_millis(),
    };
    fs::write(
        output_dir.join("batch-worker-receipt.v1.json"),
        canonical_json(&worker)?,
    )?;
    Ok(())
}
