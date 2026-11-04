use super::*;

pub(super) fn island_specs<'a>(
    base_run_id: &str,
    specs: &'a [DatasetSpec],
) -> Result<Vec<&'a DatasetSpec>> {
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    prior_no_win_ids(base_run_id)?
        .iter()
        .map(|dataset| {
            by_hash
                .get(dataset.as_str())
                .copied()
                .context("island cohort dataset is absent from Golden42")
        })
        .collect()
}

pub(crate) fn run_island_extension(registry: &Path, base_run_id: &str, resume: bool) -> Result<()> {
    validate_run_id(base_run_id)?;
    // This audit runs before the immutable manifest is created. It begins by
    // independently validating the base panel, so it preflights both prior
    // campaigns.
    validate_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let release_binary_sha256 = current_binary_sha256()?;
    let root = island_root(base_run_id);
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("island extension already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: IslandManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let cohort = island_specs(base_run_id, &specs)?;
        let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
        scq_batch::validate_staging(&staging, &recorded.release_binary_sha256, &cohort)?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_manifest(
            base_run_id,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            &specs,
            recorded.started_at_unix_ms,
        )?;
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("island manifest, prior artifacts, cohort, populations, or binary changed");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("island extension run directory already exists");
        }
        let cohort = island_specs(base_run_id, &specs)?;
        let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
        let release_binary = std::env::current_exe()?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &cohort,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            base_run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            &specs,
            unix_millis(),
        )?;
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(SUMMARY_FILE).is_file() {
        return validate_island_extension(registry, base_run_id);
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let completed_new_leaders = std::sync::atomic::AtomicUsize::new(0);
    let manifest_ref = &manifest;
    let results = thread::scope(|scope| -> Result<Vec<IslandDatasetResult>> {
        let mut handles = Vec::with_capacity(manifest_ref.datasets.len());
        for (index, binding) in manifest_ref.datasets.iter().enumerate() {
            let spec = *by_hash
                .get(binding.dataset_hash.as_str())
                .context("island manifest dataset is absent from Golden42")?;
            eprintln!(
                "island extension dataset {}/{} {}",
                index + 1,
                manifest_ref.datasets.len(),
                binding.dataset_hash
            );
            let counter = &completed_new_leaders;
            let manifest = manifest_ref;
            handles.push(scope.spawn(move || {
                let result = super::dataset::run_dataset(
                    registry,
                    manifest,
                    binding,
                    spec,
                    resume,
                    counter.load(std::sync::atomic::Ordering::Relaxed),
                    index,
                )?;
                if result.outcome == "new_leader_found" {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(result)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("island dataset controller panicked"))?
            })
            .collect()
    })?;
    let best_solutions = make_best_solutions(&manifest, &specs, &results)?;
    let best_solutions_bytes = canonical_json(&best_solutions)?;
    immutable_write(&root.join(BEST_SOLUTIONS_FILE), &best_solutions_bytes)?;
    let summary = super::summary::make_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary.dataset_count != 7
        || summary.total_logical_evaluations != 8_400
        || summary.cumulative_surpassed_count > 42
    {
        bail!("island summary campaign accounting mismatch");
    }
    immutable_write(&root.join(SUMMARY_FILE), &canonical_json(&summary)?)?;
    validate_island_extension(registry, base_run_id)
}

pub(crate) fn validate_island_extension(registry: &Path, base_run_id: &str) -> Result<()> {
    validate_run_id(base_run_id)?;
    // This also validates the complete base panel before opening the prior
    // extension artifacts.
    validate_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let root = island_root(base_run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: IslandManifest = serde_json::from_slice(&manifest_bytes)?;
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let cohort = island_specs(base_run_id, &specs)?;
    let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &cohort)?;
    let expected = expected_manifest(
        base_run_id,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        &specs,
        manifest.started_at_unix_ms,
    )?;
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("island manifest does not reconstruct from immutable prior campaigns");
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let mut results = Vec::with_capacity(manifest.datasets.len());
    let mut expected_fresh = Vec::new();
    for binding in &manifest.datasets {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("island dataset is absent from Golden42")?;
        results.push(super::dataset::validate_dataset(
            registry, &manifest, binding, spec,
        )?);
        let dataset_root = root.join("datasets").join(&binding.dataset_hash);
        let identity = island_identity(&manifest, spec);
        for (checkpoint, _) in load_island_checkpoints(
            &dataset_root,
            &identity,
            &binding.prior_terminal_checkpoint_sha256,
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
    let best_solutions_bytes = fs::read(root.join(BEST_SOLUTIONS_FILE))?;
    let recorded_best_solutions: Golden42BestSolutions =
        serde_json::from_slice(&best_solutions_bytes)?;
    let expected_best_solutions = make_best_solutions(&manifest, &specs, &results)?;
    if best_solutions_bytes != canonical_json(&recorded_best_solutions)?
        || recorded_best_solutions != expected_best_solutions
        || recorded_best_solutions.dataset_count != 42
        || recorded_best_solutions.prior_winner_count != 35
        || recorded_best_solutions.island_selected_count != 7
    {
        bail!("Golden42 best-solutions map does not reconstruct");
    }
    let summary_bytes = fs::read(root.join(SUMMARY_FILE))?;
    let recorded: IslandSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary = super::summary::make_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.dataset_count != 7
        || recorded.total_logical_evaluations != 8_400
        || recorded.cumulative_surpassed_count != EXISTING_SURPASSED + recorded.new_leaders
    {
        bail!("island extension summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated four-island extension for {} datasets: {} new leaders, {} still no leader, {} qualifying genomes, {} logical evaluations, {} fresh evaluations, cumulative surpassed {}/42; summary {}",
        recorded.dataset_count,
        recorded.new_leaders,
        recorded.still_no_leader,
        recorded.total_qualifying_genomes,
        recorded.total_logical_evaluations,
        recorded.total_fresh_evaluations,
        recorded.cumulative_surpassed_count,
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}
