pub(crate) fn run_gen30(base_run_id: &str, run_id: &str, resume: bool) -> Result<()> {
    let (base, manifest, staging, campaign_sha256) = initialize_gen30(base_run_id, run_id, resume)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        validate_gen30(base_run_id, run_id)?;
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    println!(
        "TimeSeries42 gen30 dataset_workers={dataset_workers} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}"
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
            let base = base.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_gen30_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &base,
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
        .expect("TimeSeries42 gen30 error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 gen30 retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let scan_output = run_sotarena_scan(&root).ok();
    let summary = rebuild_summary(&root, run_id, scan_output)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 gen30 did not finish with 42 terminal datasets");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 gen30 complete: wins={} losses={} ties={} exhausted={} inherited_base_wins={} improved_base_wins={} continuation_strict_wins={} valid_evaluations={} dataset_reports_written={}",
        summary.wins,
        summary.losses,
        summary.ties,
        summary.exhausted,
        summary.inherited_base_wins.unwrap_or(0),
        summary.improved_base_wins.unwrap_or(0),
        summary.continuation_strict_wins.unwrap_or(0),
        summary.valid_evaluations,
        reports_written
    );
    Ok(())
}

pub(crate) fn run_winall(
    source_run_id: &str,
    run_id: &str,
    resume: bool,
    max_generations: Option<usize>,
    lineage_salt: Option<&str>,
) -> Result<()> {
    let max_generations = max_generations.unwrap_or(DEFAULT_WINALL_GENERATIONS);
    let lineage_salt = lineage_salt.unwrap_or("");
    let (source, manifest, staging, campaign_sha256) =
        initialize_winall(source_run_id, run_id, resume, max_generations, lineage_salt)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        let summary = rebuild_summary(&root, run_id, None)?;
        let reports_written = write_timeseries_dataset_reports(
            &manifest,
            &summary,
            &root.join("terminal-summary.v1.json"),
        )?;
        println!(
            "TimeSeries42 win-all already complete: source_wins={} continuation_wins={} total_wins={} remaining_losses={} ties={} exhausted={} dataset_reports_written={}",
            source.summary.wins,
            summary.wins,
            source.summary.wins + summary.wins,
            summary.losses,
            summary.ties,
            summary.exhausted,
            reports_written
        );
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    println!(
        "TimeSeries42 win-all dataset_workers={dataset_workers} source_non_winners={} start_generation={} max_generations={} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}",
        manifest.datasets.len(),
        manifest.continuation_start_generation.unwrap_or(GEN30_GENERATIONS),
        max_generations
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
            let source = source.clone();
            let lineage_salt = manifest.config.lineage_salt.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_winall_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &source,
                    max_generations,
                    &lineage_salt,
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
        .expect("TimeSeries42 win-all error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 win-all retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let summary = rebuild_summary(&root, run_id, None)?;
    if summary.dataset_count != manifest.datasets.len()
        || summary.wins + summary.losses + summary.ties + summary.exhausted
            != manifest.datasets.len()
    {
        bail!("TimeSeries42 win-all did not finish with terminal source non-winners");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 win-all complete: source_wins={} continuation_wins={} total_wins={} remaining_losses={} ties={} exhausted={} valid_evaluations={} dataset_reports_written={}",
        source.summary.wins,
        summary.wins,
        source.summary.wins + summary.wins,
        summary.losses,
        summary.ties,
        summary.exhausted,
        summary.valid_evaluations,
        reports_written
    );
    Ok(())
}
