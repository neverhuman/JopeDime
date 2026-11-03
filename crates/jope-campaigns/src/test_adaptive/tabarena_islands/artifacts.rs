use super::execution::{prepare_tabarena_lane, PreparedTabArenaLane};
use super::*;

fn expected_manifest(
    run_id: &str,
    datasets: &[EncodedDataset],
    local_leader_artifacts: Vec<ArtifactHash>,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    staging_sha256: &str,
    started_at_unix_ms: u128,
) -> TabArenaManifest {
    let ordered_datasets = datasets
        .iter()
        .map(|dataset| dataset.spec.dataset_hash.clone())
        .collect::<Vec<_>>();
    TabArenaManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        source_root: source_root_str().to_string(),
        canonical_scores_sha256: sha256_file(Path::new(canonical_scores_str())).unwrap(),
        local_leader_artifacts,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: staging_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::aggressive_wave(),
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_datasets),
        ordered_datasets,
        datasets: datasets
            .iter()
            .map(|dataset| dataset.binding.clone())
            .collect(),
        started_at_unix_ms,
    }
}

fn make_summary(
    run_id: &str,
    results: Vec<TabArenaResult>,
    root: &Path,
) -> Result<TabArenaSummary> {
    Ok(TabArenaSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        dataset_count: results.len(),
        leaders_beaten: results
            .iter()
            .filter(|result| result.margin_vs_leader > 0.0)
            .count(),
        total_generations: results
            .iter()
            .map(|result| result.generations_evaluated)
            .sum(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        results,
        artifacts: collect_artifacts_excluding(root, SUMMARY_FILE)?,
    })
}

pub(crate) fn run_tabarena_lite_islands(
    run_id: &str,
    resume: bool,
    dataset_selector: Option<&str>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let datasets = select_datasets(datasets, dataset_selector)?;
    let registry = root.join("encoded");
    let release_binary_sha256 = current_binary_sha256()?;
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TabArena island campaign exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: TabArenaManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let specs = datasets
            .iter()
            .map(|dataset| dataset.spec.clone())
            .collect::<Vec<_>>();
        scq_batch::validate_staging(&staging, &recorded.release_binary_sha256, &specs)?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_manifest(
            run_id,
            &datasets,
            leader_artifacts,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            recorded.started_at_unix_ms,
        );
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("TabArena island manifest, leaders, data, staging, or binary changed");
        }
        recorded
    } else {
        let specs = datasets
            .iter()
            .map(|dataset| dataset.spec.clone())
            .collect::<Vec<_>>();
        let staging = scq_batch::preflight_and_stage(
            &registry,
            &std::env::current_exe()?,
            &release_binary_sha256,
            &specs,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            run_id,
            &datasets,
            leader_artifacts,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            unix_millis(),
        );
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(SUMMARY_FILE).is_file() {
        return validate_tabarena_lite_islands(run_id);
    }
    let results = run_tabarena_dataset_waves(&root, &registry, &manifest, &datasets, resume)?;
    let summary = make_summary(run_id, results, &root)?;
    if summary.dataset_count != manifest.datasets.len()
        || summary.leaders_beaten != manifest.datasets.len()
    {
        bail!("TabArena campaign ended before every selected leader was beaten");
    }
    immutable_write(&root.join(SUMMARY_FILE), &canonical_json(&summary)?)?;
    validate_tabarena_lite_islands(run_id)
}

pub(crate) fn recover_tabarena_lite_result(run_id: &str, dataset: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: TabArenaManifest = serde_json::from_slice(&manifest_bytes)?;
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let selector = manifest.ordered_datasets.join(",");
    let datasets = select_datasets(datasets, Some(&selector))?;
    let dataset_index = datasets
        .iter()
        .position(|encoded| encoded.spec.dataset_hash == dataset)
        .with_context(|| format!("TabArena dataset {dataset} is not in campaign {run_id}"))?;
    let registry = root.join("encoded");
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let specs = datasets
        .iter()
        .map(|encoded| encoded.spec.clone())
        .collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &specs)?;
    scq_batch::verify_remote_staging(&staging)?;
    let expected = expected_manifest(
        run_id,
        &datasets,
        leader_artifacts,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    );
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("TabArena manifest does not reconstruct");
    }
    match prepare_tabarena_lane(
        &root,
        &registry,
        &manifest,
        &datasets[dataset_index],
        dataset_index,
        true,
    )? {
        PreparedTabArenaLane::Finished(_, result) => {
            eprintln!(
                "recovered TabArena result: {} generation {} RMSE {:.17} leader {:.17}",
                result.dataset,
                result.stopping_generation,
                result.selected_rmse,
                result.leader_rmse
            );
            Ok(())
        }
        PreparedTabArenaLane::Active(_) => {
            bail!("TabArena dataset has no leader-beating checkpoint to recover")
        }
    }
}

fn validate_dataset(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
) -> Result<(TabArenaResult, Vec<scq_batch::ExpectedFresh>)> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    if fs::read(root.join("campaign.v1.json"))? != canonical_json(&identity)? {
        bail!("TabArena dataset identity mismatch");
    }
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let checkpoints = load_checkpoints(&root, &identity)?;
    if checkpoints.is_empty() || !checkpoints.last().unwrap().0.leader_beaten {
        bail!("TabArena dataset has no complete leader-beating checkpoint");
    }
    let mut expected_population =
        initial_population(registry, &root, &encoded.spec, &manifest.run_id)?;
    let mut cumulative_fresh = 0;
    let mut expected_fresh = Vec::new();
    for (generation, (checkpoint, _)) in checkpoints.iter().enumerate() {
        if checkpoint.slots.iter().enumerate().any(|(slot, outcome)| {
            outcome.generation != generation
                || outcome.slot != slot
                || outcome.member != expected_population[slot]
        }) {
            bail!("TabArena logical population does not reconstruct");
        }
        let mut fresh = 0;
        for outcome in &checkpoint.slots {
            let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
            if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                bail!("TabArena candidate receipt hash mismatch");
            }
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let request: CandidateRequest = serde_json::from_slice(&fs::read(
                root.join("requests")
                    .join(&receipt.purpose)
                    .join(format!("generation-{:03}", receipt.generation))
                    .join(format!(
                        "slot-{:03}-attempt-{}.v1.json",
                        receipt.slot, receipt.attempt
                    )),
            )?)?;
            validate_receipt_against_request(&receipt, &request, &test)?;
            if outcome.cache_source.is_none() {
                fresh += 1;
                expected_fresh.push(scq_batch::ExpectedFresh {
                    dataset_hash: encoded.spec.dataset_hash.clone(),
                    generation,
                    slot: outcome.slot,
                    attempt: outcome.attempt,
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    node: outcome.node.clone(),
                    scq_job_id: outcome.scq_job_id.clone(),
                });
            }
        }
        cumulative_fresh += fresh;
        let expected = make_checkpoint(
            &identity,
            generation,
            checkpoint.previous_checkpoint_sha256.clone(),
            checkpoint.slots.clone(),
            fresh,
            cumulative_fresh,
        )?;
        if *checkpoint != expected {
            bail!("TabArena ranking, migration, cache, or next population changed");
        }
        expected_population = checkpoint.next_population.clone();
        if checkpoint.leader_beaten && generation + 1 != checkpoints.len() {
            bail!("TabArena dataset continued after beating its leader");
        }
    }
    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: TabArenaResult = serde_json::from_slice(&result_bytes)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = &terminal.slots[terminal.best_slot];
    if result_bytes != canonical_json(&result)?
        || result.schema != RESULT_SCHEMA
        || result.benchmark_valid
        || result.dataset != encoded.spec.dataset_hash
        || result.stopping_generation != terminal.generation
        || result.generations_evaluated != checkpoints.len()
        || result.selected_rmse.to_bits() != terminal.best_rmse.to_bits()
        || result.margin_vs_leader <= 0.0
        || result.semantic_sha256 != selected.member.semantic_sha256
        || result.logical_evaluations != checkpoints.len() * TOTAL_POPULATION
        || result.fresh_evaluations != terminal.cumulative_fresh_evaluations
        || result.replay.len() != scq_batch::NODES.len()
    {
        bail!("TabArena final result mismatch");
    }
    for (slot, replay) in result.replay.iter().enumerate() {
        let receipt: CandidateReceipt =
            serde_json::from_slice(&fs::read(root.join(&replay.receipt_file))?)?;
        if receipt.node != scq_batch::NODES[slot]
            || receipt.semantic_sha256 != result.semantic_sha256
            || receipt.predictions_sha256 != result.prediction_sha256
        {
            bail!("TabArena cross-node replay binding mismatch");
        }
    }
    Ok((result, expected_fresh))
}

pub(crate) fn validate_tabarena_lite_islands(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: TabArenaManifest = serde_json::from_slice(&manifest_bytes)?;
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let selector = manifest.ordered_datasets.join(",");
    let datasets = select_datasets(datasets, Some(&selector))?;
    let registry = root.join("encoded");
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let specs = datasets
        .iter()
        .map(|dataset| dataset.spec.clone())
        .collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &specs)?;
    let expected = expected_manifest(
        run_id,
        &datasets,
        leader_artifacts,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    );
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("TabArena manifest does not reconstruct");
    }
    let mut results = Vec::new();
    let mut expected_fresh = Vec::new();
    for dataset in &datasets {
        let (result, mut fresh) = validate_dataset(&root, &registry, &manifest, dataset)?;
        results.push(result);
        expected_fresh.append(&mut fresh);
    }
    scq_batch::validate_batch_records(
        &root,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &manifest.ordered_datasets,
        &expected_fresh,
    )?;
    let summary_bytes = fs::read(root.join(SUMMARY_FILE))?;
    let recorded: TabArenaSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary = make_summary(run_id, results, &root)?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.dataset_count != datasets.len()
        || recorded.leaders_beaten != datasets.len()
    {
        bail!("TabArena summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated TabArena Lite islands: {} leaders beaten in {} generations and {} logical evaluations",
        recorded.leaders_beaten, recorded.total_generations, recorded.total_logical_evaluations
    );
    Ok(())
}
