pub(crate) fn run_all(run_id: &str, resume: bool) -> Result<()> {
    let (manifest, staging, campaign_sha256) = initialize(run_id, resume)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        validate_all(run_id)?;
        let summary: TerminalSummary = load_json(&root.join("terminal-summary.v1.json"))?;
        write_timeseries_dataset_reports(
            &manifest,
            &summary,
            &root.join("terminal-summary.v1.json"),
        )?;
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    let allow_generation_zero_win = is_forecast3_gen20(run_id);
    println!(
        "TimeSeries42 dataset_workers={dataset_workers} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}"
    );
    let queue = Arc::new(Mutex::new(VecDeque::from(manifest.datasets.clone())));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..dataset_workers {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let staging = staging.clone();
            let release_sha256 = manifest.release_binary_sha256.clone();
            let campaign_sha256 = campaign_sha256.clone();
            let root = root.clone();
            let run_id = run_id.to_string();
            let config = manifest.config.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &config,
                    allow_generation_zero_win,
                ) {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", spec.dataset_hash));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let scan_output = run_sotarena_scan(&root).ok();
    let summary = rebuild_summary(&root, run_id, scan_output)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 did not finish with 42 terminal datasets");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 complete: wins={} losses={} ties={} exhausted={} valid_evaluations={} dataset_reports_written={}",
        summary.wins, summary.losses, summary.ties, summary.exhausted, summary.valid_evaluations, reports_written
    );
    Ok(())
}

fn validate_gen30_dataset_result(
    result: &DatasetResult,
    base_result: &DatasetResult,
    base_run_id: &str,
) -> Result<()> {
    if result.base_run_id.as_deref() != Some(base_run_id)
        || result.base_status.as_deref() != Some(base_result.status.as_str())
        || result.base_strict_winner != Some(base_result.strict_winner)
        || result.base_best_jope_mase != base_result.best_jope_mase
        || result.continuation_status.is_none()
        || result.best_evidence_run_id.is_none()
        || result.completed_generations < GEN30_START_GENERATION + 1
        || result.completed_generations > GEN30_GENERATIONS
    {
        bail!(
            "invalid TimeSeries42 gen30 continuation fields for {}",
            result.dataset_hash
        );
    }
    match result.continuation_status.as_deref() {
        Some("inherited_base_win") => {
            if !base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_some_and(|generation| generation >= GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() != Some(base_run_id)
                || result.completed_generations != GEN30_GENERATIONS
            {
                bail!(
                    "invalid inherited base win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("base_win_improved") => {
            if !base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_none_or(|generation| generation < GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() == Some(base_run_id)
                || result.completed_generations != GEN30_GENERATIONS
            {
                bail!(
                    "invalid improved base win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("gen30_strict_win") => {
            if base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_none_or(|generation| generation < GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() == Some(base_run_id)
            {
                bail!(
                    "invalid gen30 strict win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("loss" | "tie" | "exhausted") => {
            if result.strict_winner || result.completed_generations != GEN30_GENERATIONS {
                bail!(
                    "invalid terminal non-win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some(other) => bail!(
            "unknown TimeSeries42 gen30 continuation status {other} for {}",
            result.dataset_hash
        ),
        None => unreachable!("checked above"),
    }
    Ok(())
}
