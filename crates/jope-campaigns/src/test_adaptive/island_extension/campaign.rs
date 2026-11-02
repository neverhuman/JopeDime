use super::*;

pub(super) fn prior_history(
    base_run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(GenerationCheckpoint, String)>> {
    let base_dataset_root = run_root(base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let base_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(base_dataset_root.join("campaign.v1.json"))?)?;
    let base = load_checkpoints(&base_dataset_root, &base_identity)?;
    let prior_root = extension_root(base_run_id);
    let prior_manifest: ExtensionManifest =
        serde_json::from_slice(&fs::read(prior_root.join(EXTENSION_MANIFEST_FILE))?)?;
    let binding = prior_manifest
        .datasets
        .iter()
        .find(|binding| binding.dataset_hash == spec.dataset_hash)
        .context("dataset is absent from the generations 10-19 manifest")?;
    let dataset_root = prior_root.join("datasets").join(&spec.dataset_hash);
    let prior_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(dataset_root.join("campaign.v1.json"))?)?;
    let extension = load_extension_checkpoints(
        &dataset_root,
        &prior_identity,
        &binding.base_terminal_checkpoint_sha256,
    )?;
    if extension.len() != EXTENSION_GENERATIONS {
        bail!(
            "dataset {} has incomplete generations 10-19",
            spec.dataset_hash
        );
    }
    combined_checkpoints(&base, &extension)
}

pub(super) fn prior_no_win_ids(base_run_id: &str) -> Result<Vec<String>> {
    let root = extension_root(base_run_id);
    let manifest: ExtensionManifest =
        serde_json::from_slice(&fs::read(root.join(EXTENSION_MANIFEST_FILE))?)?;
    let summary: ExtensionSummary =
        serde_json::from_slice(&fs::read(root.join(EXTENSION_SUMMARY_FILE))?)?;
    let outcomes = summary
        .results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result.outcome.as_str()))
        .collect::<HashMap<_, _>>();
    let selected = manifest
        .ordered_dataset_ids
        .iter()
        .filter(|dataset| {
            outcomes.get(dataset.as_str()).copied() == Some("no_win_after_20_total_generations")
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.len() != 7 || summary.new_leaders + summary.still_no_leader != summary.dataset_count
    {
        bail!("generations 10-19 do not reconstruct the ordered seven-dataset no-win cohort");
    }
    let base_summary: PanelSummary =
        serde_json::from_slice(&fs::read(run_root(base_run_id).join(super::SUMMARY_FILE))?)?;
    if base_summary.early_wins + summary.new_leaders != EXISTING_SURPASSED {
        bail!("prior campaigns do not reconstruct 35 surpassed datasets");
    }
    Ok(selected)
}

pub(super) fn dataset_binding(
    base_run_id: &str,
    spec: &DatasetSpec,
) -> Result<IslandDatasetBinding> {
    let prior_root = extension_root(base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let campaign_file = prior_root.join("campaign.v1.json");
    let result_file = prior_root.join("final-result.v1.json");
    let terminal_file = prior_root
        .join("checkpoints")
        .join("generation-019.v1.json");
    let combined = prior_history(base_run_id, spec)?;
    let populations = generation_20_populations(&combined, spec.features)?;
    Ok(IslandDatasetBinding {
        dataset_hash: spec.dataset_hash.clone(),
        prior_campaign_file: campaign_file.to_string_lossy().into_owned(),
        prior_campaign_sha256: sha256_file(&campaign_file)?,
        prior_final_result_file: result_file.to_string_lossy().into_owned(),
        prior_final_result_sha256: sha256_file(&result_file)?,
        prior_terminal_checkpoint_file: terminal_file.to_string_lossy().into_owned(),
        prior_terminal_checkpoint_sha256: sha256_file(&terminal_file)?,
        generation_20_island_population_hashes: populations
            .iter()
            .enumerate()
            .map(|(island, population)| {
                Ok(IslandPopulationHash {
                    island,
                    sha256: sha256_bytes(&canonical_json(population)?),
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

pub(super) fn expected_manifest(
    base_run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    scq_staging_receipt_sha256: &str,
    specs: &[DatasetSpec],
    started_at_unix_ms: u128,
) -> Result<IslandManifest> {
    let ordered_dataset_ids = prior_no_win_ids(base_run_id)?;
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let datasets = ordered_dataset_ids
        .iter()
        .map(|dataset| {
            dataset_binding(
                base_run_id,
                by_hash
                    .get(dataset.as_str())
                    .context("island cohort dataset is absent from the Golden42 panel")?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let base_root = run_root(base_run_id);
    let prior_root = extension_root(base_run_id);
    Ok(IslandManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        prior_extension_run_id: extension_run_id(base_run_id),
        island_extension_run_id: island_run_id(base_run_id),
        base_panel_manifest_sha256: sha256_file(&base_root.join("panel-manifest.v1.json"))?,
        base_panel_summary_sha256: sha256_file(&base_root.join(super::SUMMARY_FILE))?,
        base_winner_map_sha256: sha256_file(&base_root.join(WINNER_MAP_FILE))?,
        prior_extension_manifest_sha256: sha256_file(&prior_root.join(EXTENSION_MANIFEST_FILE))?,
        prior_extension_summary_sha256: sha256_file(&prior_root.join(EXTENSION_SUMMARY_FILE))?,
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
        ordered_dataset_ids,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: scq_staging_receipt_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::fixed(),
        config: IslandConfig::fixed(),
        migration_policy: MigrationPolicy::fixed(),
        rng_policy: RngPolicy::fixed(),
        datasets,
        started_at_unix_ms,
    })
}

pub(super) fn island_identity(
    manifest: &IslandManifest,
    spec: &DatasetSpec,
) -> IslandCampaignIdentity {
    IslandCampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.island_extension_run_id.clone(),
        dataset: spec.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        config: manifest.config.clone(),
        migration_policy: manifest.migration_policy.clone(),
        rng_policy: manifest.rng_policy.clone(),
        selection_objective: "continue immutable generations 0-19 on four isolated islands; rank directly by test RMSE (intentionally test-adaptive)".to_string(),
    }
}

pub(super) fn worker_identity(identity: &IslandCampaignIdentity) -> CampaignIdentity {
    let mut value = super::identity(
        &identity.run_id,
        identity.release_binary_sha256.clone(),
        &identity.dataset,
    );
    value.population = TOTAL_POPULATION;
    value.elites = ISLAND_ELITES;
    value.parent_pool = ISLAND_PARENT_POOL;
    value.generations = GENERATIONS;
    value.selection_objective = identity.selection_objective.clone();
    value
}

pub(super) fn island_outcome_from_evaluation(
    generation: usize,
    island: usize,
    local_slot: usize,
    population: IslandPopulationMember,
    evaluation: &CachedEvaluation,
    cached: bool,
) -> IslandSlotOutcome {
    IslandSlotOutcome {
        generation,
        island,
        local_slot,
        global_slot: global_slot(island, local_slot),
        member: population.member,
        parents: population.parents,
        valid: evaluation.receipt.valid(evaluation.receipt.test_rows),
        metrics: evaluation.receipt.metrics.clone(),
        failure: evaluation.receipt.failure.clone(),
        cache_source: cached.then(|| evaluation.source.clone()),
        receipt_file: evaluation.receipt_file.clone(),
        receipt_sha256: evaluation.receipt_sha256.clone(),
        node: evaluation.receipt.node.clone(),
        scq_job_id: evaluation.receipt.scq_job_id.clone(),
        attempt: evaluation.receipt.attempt,
    }
}

// Keep island identity, population state and execution budget explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_island_generation(
    identity: &CampaignIdentity,
    registry: &Path,
    campaign_root: &Path,
    scq_binary_sha256: &str,
    dataset_index: usize,
    generation: usize,
    populations: Vec<Vec<IslandPopulationMember>>,
    cache: &mut HashMap<String, CachedEvaluation>,
    test: &RegressionData,
) -> Result<(Vec<Vec<IslandSlotOutcome>>, usize)> {
    if populations.len() != ISLANDS
        || populations
            .iter()
            .any(|population| population.len() != ISLAND_POPULATION)
    {
        bail!("island generation does not contain four complete populations");
    }
    let mut outcomes = vec![None; TOTAL_POPULATION];
    let mut pending = BTreeMap::<String, Vec<(usize, usize, IslandPopulationMember)>>::new();
    for (island, population) in populations.into_iter().enumerate() {
        for (local_slot, member) in population.into_iter().enumerate() {
            let slot = global_slot(island, local_slot);
            if let Some(evaluation) = cache.get(&member.member.semantic_sha256) {
                outcomes[slot] = Some(island_outcome_from_evaluation(
                    generation, island, local_slot, member, evaluation, true,
                ));
            } else {
                pending
                    .entry(member.member.semantic_sha256.clone())
                    .or_default()
                    .push((island, local_slot, member));
            }
        }
    }
    let fresh = pending.len();
    let pending = pending.into_values().collect::<Vec<_>>();
    let representatives = pending
        .iter()
        .map(|entries| {
            let (island, local_slot, population) = &entries[0];
            (global_slot(*island, *local_slot), population.member.clone())
        })
        .collect::<Vec<_>>();
    let evaluations = scq_batch::evaluate_batches(
        campaign_root,
        identity,
        registry,
        &identity.binary_sha256,
        scq_binary_sha256,
        dataset_index,
        "candidate",
        generation,
        &representatives,
        test,
    )?;
    if evaluations.len() != pending.len() {
        bail!("SCQ batch result count does not match fresh candidate count");
    }
    for (entries, evaluation) in pending.into_iter().zip(evaluations) {
        cache.insert(
            evaluation.receipt.semantic_sha256.clone(),
            evaluation.clone(),
        );
        for (index, (island, local_slot, population)) in entries.into_iter().enumerate() {
            let slot = global_slot(island, local_slot);
            outcomes[slot] = Some(island_outcome_from_evaluation(
                generation,
                island,
                local_slot,
                population,
                &evaluation,
                index > 0,
            ));
        }
    }
    let flat = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("all island slots evaluated"))
        .collect::<Vec<_>>();
    let islands = (0..ISLANDS)
        .map(|island| flat[global_slot(island, 0)..global_slot(island + 1, 0)].to_vec())
        .collect();
    Ok((islands, fresh))
}

pub(super) fn checkpoint_path(root: &Path, generation: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

pub(super) fn migration_relative(generation: usize) -> String {
    format!("migrations/generation-{generation:03}.v1.json")
}

pub(super) fn load_island_checkpoints(
    root: &Path,
    identity: &IslandCampaignIdentity,
    prior_terminal_sha256: &str,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = prior_terminal_sha256.to_string();
    for generation in START_GENERATION..END_GENERATION {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: IslandGenerationCheckpoint = serde_json::from_slice(&bytes)?;
        validate_checkpoint_envelope(&checkpoint, identity, generation, &previous)
            .with_context(|| format!("invalid island checkpoint {}", path.display()))?;
        if bytes != canonical_json(&checkpoint)? {
            bail!("noncanonical island checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = sha256.clone();
        checkpoints.push((checkpoint, sha256));
    }
    for generation in (START_GENERATION + checkpoints.len())..END_GENERATION {
        if checkpoint_path(root, generation).is_file() {
            bail!("island checkpoint history contains a gap before generation {generation}");
        }
    }
    Ok(checkpoints)
}

pub(super) fn validate_checkpoint_envelope(
    checkpoint: &IslandGenerationCheckpoint,
    identity: &IslandCampaignIdentity,
    generation: usize,
    previous: &str,
) -> Result<()> {
    if checkpoint.schema != CHECKPOINT_SCHEMA
        || checkpoint.benchmark_valid
        || checkpoint.run_id != identity.run_id
        || checkpoint.dataset_hash != identity.dataset.dataset_hash
        || checkpoint.generation != generation
        || checkpoint.previous_checkpoint_sha256 != previous
        || checkpoint.logical_slots != TOTAL_POPULATION
        || checkpoint.islands.len() != ISLANDS
        || checkpoint.islands.iter().any(|island| {
            island.logical_slots != ISLAND_POPULATION
                || island.slots.len() != ISLAND_POPULATION
                || (generation + 1 < END_GENERATION
                    && island.next_population.len() != ISLAND_POPULATION)
                || (generation + 1 == END_GENERATION && !island.next_population.is_empty())
        })
    {
        bail!("island checkpoint envelope mismatch");
    }
    Ok(())
}

pub(super) fn validate_island_receipt_binding(
    identity: &IslandCampaignIdentity,
    outcome: &IslandSlotOutcome,
    receipt: &CandidateReceipt,
    request: &CandidateRequest,
) -> Result<()> {
    if receipt.run_id != identity.run_id
        || receipt.dataset_hash != identity.dataset.dataset_hash
        || receipt.binary_sha256 != identity.release_binary_sha256
        || request.binary_sha256 != identity.release_binary_sha256
        || receipt.semantic_sha256 != outcome.member.semantic_sha256
        || outcome.valid != receipt.valid(identity.dataset.test_rows)
        || outcome.metrics != receipt.metrics
        || outcome.failure != receipt.failure
    {
        bail!("island receipt provenance or binary mismatch");
    }
    Ok(())
}

pub(super) fn write_or_verify(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        Ok(())
    } else {
        immutable_write(path, bytes)
    }
}

pub(super) fn qualifies(outcome: &IslandSlotOutcome, g0_rmse: f64, leader_rmse: f64) -> bool {
    outcome.member.evolved_descendant
        && outcome.valid
        && outcome.rmse() < g0_rmse
        && outcome.rmse() < leader_rmse
}

pub(super) fn genome_location(outcome: &IslandSlotOutcome) -> IslandGenomeLocation {
    IslandGenomeLocation {
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

pub(super) type IslandEvolution = (
    Vec<Vec<IslandPopulationMember>>,
    Option<(String, String, MigrationRecord)>,
);

pub(super) fn evolve_islands(
    identity: &IslandCampaignIdentity,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<IslandEvolution> {
    if generation + 1 == END_GENERATION {
        return Ok((vec![Vec::new(); ISLANDS], None));
    }
    let (parents, migration) = if MIGRATION_GENERATIONS.contains(&generation) {
        let (parents, record) = migrate_rankings(
            &identity.run_id,
            &identity.dataset.dataset_hash,
            generation,
            ranked,
        )?;
        let relative = migration_relative(generation);
        let bytes = canonical_json(&record)?;
        let sha256 = sha256_bytes(&bytes);
        (parents, Some((relative, sha256, record)))
    } else {
        (ranked.to_vec(), None)
    };
    let next = parents
        .iter()
        .enumerate()
        .map(|(island, ranked)| {
            next_island_population(ranked, generation + 1, island, identity.dataset.features)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((next, migration))
}
