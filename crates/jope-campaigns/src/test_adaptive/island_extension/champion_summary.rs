use super::*;

pub(super) fn make_champion_summary(
    manifest: &ChampionManifest,
    results: Vec<ChampionDatasetResult>,
    root: &Path,
    best_solutions_sha256: String,
) -> Result<ChampionSummary> {
    let initial: IslandSummary = serde_json::from_slice(&fs::read(
        island_root(&manifest.base_run_id).join(SUMMARY_FILE),
    )?)?;
    Ok(ChampionSummary {
        schema: CHAMPION_SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        initial_island_run_id: manifest.initial_island_run_id.clone(),
        champion_run_id: manifest.champion_run_id.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        ordered_dataset_ids: manifest.ordered_dataset_ids.clone(),
        dataset_count: results.len(),
        inherited_champion_count: initial.cumulative_surpassed_count,
        continuation_champion_count: results.len(),
        cumulative_surpassed_count: initial.cumulative_surpassed_count + results.len(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        best_solutions_file: BEST_SOLUTIONS_FILE.to_string(),
        best_solutions_sha256,
        results,
        artifacts: collect_artifacts_excluding(root, CHAMPION_SUMMARY_FILE)?,
    })
}

pub(crate) fn run_champion_continuation(
    registry: &Path,
    base_run_id: &str,
    resume: bool,
) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_island_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let release_binary_sha256 = current_binary_sha256()?;
    let root = champion_root(base_run_id);
    let manifest_path = root.join(CHAMPION_MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("champion continuation already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: ChampionManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let by_hash = specs
            .iter()
            .map(|spec| (spec.dataset_hash.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let cohort = recorded
            .ordered_dataset_ids
            .iter()
            .map(|dataset| {
                by_hash
                    .get(dataset.as_str())
                    .copied()
                    .context("champion cohort dataset is absent from Golden42")
            })
            .collect::<Result<Vec<_>>>()?;
        scq_batch::validate_staging(
            &staging,
            &recorded.release_binary_sha256,
            &cohort.into_iter().cloned().collect::<Vec<_>>(),
        )?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_champion_manifest(
            base_run_id,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            recorded.started_at_unix_ms,
        )?;
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("champion manifest, ancestors, staging, or binary changed");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("champion continuation directory already exists");
        }
        let bindings = champion_bindings(base_run_id)?;
        let by_hash = specs
            .iter()
            .map(|spec| (spec.dataset_hash.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let cohort = bindings
            .iter()
            .map(|binding| {
                by_hash
                    .get(binding.dataset_hash.as_str())
                    .copied()
                    .context("champion cohort dataset is absent from Golden42")
            })
            .collect::<Result<Vec<_>>>()?;
        let release_binary = std::env::current_exe()?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &cohort.into_iter().cloned().collect::<Vec<_>>(),
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_champion_manifest(
            base_run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            unix_millis(),
        )?;
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(CHAMPION_SUMMARY_FILE).is_file() {
        return validate_champion_continuation(registry, base_run_id);
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let completed_champions = std::sync::atomic::AtomicUsize::new(0);
    let results = thread::scope(|scope| -> Result<Vec<ChampionDatasetResult>> {
        let mut handles = Vec::with_capacity(manifest.datasets.len());
        for (index, binding) in manifest.datasets.iter().enumerate() {
            let spec = *by_hash
                .get(binding.dataset_hash.as_str())
                .context("champion manifest dataset is absent from Golden42")?;
            eprintln!(
                "champion continuation dataset {}/{} {}",
                index + 1,
                manifest.datasets.len(),
                binding.dataset_hash
            );
            let counter = &completed_champions;
            let manifest = &manifest;
            handles.push(scope.spawn(move || {
                let result = run_champion_dataset(
                    registry,
                    manifest,
                    binding,
                    spec,
                    resume,
                    counter.load(std::sync::atomic::Ordering::Relaxed),
                    index,
                )?;
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(result)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("champion dataset controller panicked"))?
            })
            .collect()
    })?;
    let best_solutions = make_champion_best_solutions(&manifest, &results)?;
    let best_solutions_bytes = canonical_json(&best_solutions)?;
    immutable_write(&root.join(BEST_SOLUTIONS_FILE), &best_solutions_bytes)?;
    let summary = make_champion_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary.cumulative_surpassed_count != 42 {
        bail!("champion continuation stopped before 42/42");
    }
    immutable_write(
        &root.join(CHAMPION_SUMMARY_FILE),
        &canonical_json(&summary)?,
    )?;
    validate_champion_continuation(registry, base_run_id)
}

pub(crate) fn validate_champion_continuation(registry: &Path, base_run_id: &str) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_island_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let root = champion_root(base_run_id);
    let manifest_bytes = fs::read(root.join(CHAMPION_MANIFEST_FILE))?;
    let manifest: ChampionManifest = serde_json::from_slice(&manifest_bytes)?;
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let cohort = manifest
        .ordered_dataset_ids
        .iter()
        .map(|dataset| {
            by_hash
                .get(dataset.as_str())
                .copied()
                .context("champion cohort dataset is absent from Golden42")
        })
        .collect::<Result<Vec<_>>>()?;
    scq_batch::validate_staging(
        &staging,
        &manifest.release_binary_sha256,
        &cohort.into_iter().cloned().collect::<Vec<_>>(),
    )?;
    let expected_manifest = expected_champion_manifest(
        base_run_id,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    )?;
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected_manifest {
        bail!("champion continuation manifest does not reconstruct");
    }
    let mut results = Vec::with_capacity(manifest.datasets.len());
    let mut expected_fresh = Vec::new();
    for binding in &manifest.datasets {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("champion dataset is absent from Golden42")?;
        results.push(validate_champion_dataset(
            registry, &manifest, binding, spec,
        )?);
        let dataset_root = root.join("datasets").join(&binding.dataset_hash);
        let identity = champion_identity(&manifest, spec);
        for (checkpoint, _) in load_champion_checkpoints(
            &dataset_root,
            &identity,
            &binding.generation_29_checkpoint_sha256,
        )? {
            for outcome in checkpoint
                .islands
                .iter()
                .flat_map(|island| &island.slots)
                .filter(|outcome| outcome.cache_source.is_none())
            {
                expected_fresh.push(scq_batch::ExpectedFresh {
                    dataset_hash: binding.dataset_hash.clone(),
                    generation: outcome.generation,
                    slot: outcome.global_slot,
                    attempt: outcome.attempt,
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    node: outcome.node.clone(),
                    scq_job_id: outcome.scq_job_id.clone(),
                });
            }
        }
    }
    scq_batch::validate_batch_records(
        &root,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &manifest.ordered_dataset_ids,
        &expected_fresh,
    )?;
    let best_bytes = fs::read(root.join(BEST_SOLUTIONS_FILE))?;
    let recorded_best: Golden42BestSolutions = serde_json::from_slice(&best_bytes)?;
    let expected_best = make_champion_best_solutions(&manifest, &results)?;
    if best_bytes != canonical_json(&recorded_best)?
        || recorded_best != expected_best
        || recorded_best.dataset_count != 42
        || recorded_best
            .solutions
            .iter()
            .any(|solution| solution.benchmark_status != "surpassed_external_benchmark")
    {
        bail!("42/42 champion best-solutions map does not reconstruct");
    }
    let summary_bytes = fs::read(root.join(CHAMPION_SUMMARY_FILE))?;
    let recorded: ChampionSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary =
        make_champion_summary(&manifest, results, &root, sha256_bytes(&best_bytes))?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.cumulative_surpassed_count != 42
        || recorded.continuation_champion_count != recorded.dataset_count
    {
        bail!("champion continuation summary does not reconstruct");
    }
    eprintln!(
        "validated Golden42 champion continuation: {} inherited + {} continuation champions = 42/42; {} logical evaluations, {} fresh evaluations; map {}, summary {}",
        recorded.inherited_champion_count,
        recorded.continuation_champion_count,
        recorded.total_logical_evaluations,
        recorded.total_fresh_evaluations,
        sha256_bytes(&best_bytes),
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}
