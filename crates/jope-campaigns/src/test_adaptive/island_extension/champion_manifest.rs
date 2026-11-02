use super::*;

pub(super) fn champion_bindings(base_run_id: &str) -> Result<Vec<ChampionDatasetBinding>> {
    let root = island_root(base_run_id);
    let summary: IslandSummary = serde_json::from_slice(&fs::read(root.join(SUMMARY_FILE))?)?;
    let mut bindings = Vec::new();
    for result in &summary.results {
        if result.outcome == "new_leader_found" {
            continue;
        }
        let result_path = root
            .join("datasets")
            .join(&result.dataset_hash)
            .join("final-result.v1.json");
        let checkpoint_path = root
            .join("datasets")
            .join(&result.dataset_hash)
            .join("checkpoints/generation-029.v1.json");
        bindings.push(ChampionDatasetBinding {
            dataset_hash: result.dataset_hash.clone(),
            initial_result_file: artifact_name(&result_path),
            initial_result_sha256: sha256_file(&result_path)?,
            generation_29_checkpoint_file: artifact_name(&checkpoint_path),
            generation_29_checkpoint_sha256: sha256_file(&checkpoint_path)?,
        });
    }
    if EXISTING_SURPASSED + summary.new_leaders + bindings.len() != 42 {
        bail!("sealed island block does not partition all 42 datasets");
    }
    Ok(bindings)
}

pub(super) fn expected_champion_manifest(
    base_run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    staging_sha256: &str,
    started_at_unix_ms: u128,
) -> Result<ChampionManifest> {
    let initial_root = island_root(base_run_id);
    let bindings = champion_bindings(base_run_id)?;
    let ordered_dataset_ids = bindings
        .iter()
        .map(|binding| binding.dataset_hash.clone())
        .collect::<Vec<_>>();
    Ok(ChampionManifest {
        schema: CHAMPION_MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        initial_island_run_id: island_run_id(base_run_id),
        champion_run_id: champion_run_id(base_run_id),
        initial_manifest_sha256: sha256_file(&initial_root.join(MANIFEST_FILE))?,
        initial_summary_sha256: sha256_file(&initial_root.join(SUMMARY_FILE))?,
        initial_best_solutions_sha256: sha256_file(
            &initial_root.join(BEST_SOLUTIONS_FILE),
        )?,
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
        ordered_dataset_ids,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: staging_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::fixed(),
        config: IslandConfig::champion(),
        migration_policy: MigrationPolicy::champion(),
        rng_policy: RngPolicy::fixed(),
        complexity_schedule:
            "depth 1 at generations 30-39, depth 2 at 40-49, depth 3 at 50-59, depth 4 thereafter; crossover probability rises from 0.25 to 0.55"
                .to_string(),
        stop_policy:
            "complete each generation; stop a dataset only after an evolved valid genome has RMSE strictly below both generation-0 and the external leader"
                .to_string(),
        datasets: bindings,
        started_at_unix_ms,
    })
}

pub(super) fn champion_identity(
    manifest: &ChampionManifest,
    spec: &DatasetSpec,
) -> IslandCampaignIdentity {
    IslandCampaignIdentity {
        schema: CHAMPION_CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.champion_run_id.clone(),
        dataset: spec.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        config: manifest.config.clone(),
        migration_policy: manifest.migration_policy.clone(),
        rng_policy: manifest.rng_policy.clone(),
        selection_objective: "test-target RMSE; continue until strict external-leader crossing"
            .to_string(),
    }
}

pub(super) fn load_initial_island_checkpoints(
    base_run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let root = island_root(base_run_id);
    let manifest: IslandManifest = serde_json::from_slice(&fs::read(root.join(MANIFEST_FILE))?)?;
    let binding = manifest
        .datasets
        .iter()
        .find(|binding| binding.dataset_hash == spec.dataset_hash)
        .context("champion dataset is absent from initial island manifest")?;
    let identity = island_identity(&manifest, spec);
    let checkpoints = load_island_checkpoints(
        &root.join("datasets").join(&spec.dataset_hash),
        &identity,
        &binding.prior_terminal_checkpoint_sha256,
    )?;
    if checkpoints.len() != GENERATIONS {
        bail!("initial island history is incomplete");
    }
    Ok(checkpoints)
}

pub(super) fn generation_30_populations(
    initial: &[(IslandGenerationCheckpoint, String)],
    features: usize,
) -> Result<Vec<Vec<IslandPopulationMember>>> {
    let terminal = &initial
        .last()
        .context("initial island history has no terminal checkpoint")?
        .0;
    if terminal.generation != CHAMPION_START_GENERATION - 1 {
        bail!("champion continuation source is not generation 29");
    }
    terminal
        .islands
        .iter()
        .enumerate()
        .map(|(island, state)| {
            let mut ranked = state.slots.clone();
            ranked.sort_by(rank_island_outcomes);
            next_champion_population(&ranked, CHAMPION_START_GENERATION, island, features)
        })
        .collect()
}

pub(super) fn evolve_champion_islands(
    identity: &IslandCampaignIdentity,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
    terminal: bool,
) -> Result<IslandEvolution> {
    if terminal {
        return Ok((vec![Vec::new(); ISLANDS], None));
    }
    let (parents, migration) = if generation > CHAMPION_START_GENERATION && generation % 2 == 1 {
        let (parents, record) = migrate_rankings_champion(
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
        .map(|(island, island_ranked)| {
            next_champion_population(
                island_ranked,
                generation + 1,
                island,
                identity.dataset.features,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((next, migration))
}

pub(super) fn make_champion_checkpoint(
    identity: &IslandCampaignIdentity,
    root: &Path,
    generation: usize,
    previous_checkpoint_sha256: String,
    outcomes: Vec<Vec<IslandSlotOutcome>>,
    cumulative_fresh_evaluations: usize,
    g0_rmse: f64,
) -> Result<(IslandGenerationCheckpoint, Vec<Vec<IslandPopulationMember>>)> {
    if outcomes.len() != ISLANDS
        || outcomes
            .iter()
            .any(|island| island.len() != ISLAND_POPULATION)
    {
        bail!("cannot checkpoint incomplete champion outcomes");
    }
    let mut ranked = outcomes.clone();
    for island in &mut ranked {
        island.sort_by(rank_island_outcomes);
    }
    let valid_candidates = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.valid)
        .count();
    let fresh_evaluations = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.cache_source.is_none())
        .count();
    let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
    all_ranked.sort_by(rank_island_outcomes);
    let qualifying_genomes = outcomes
        .iter()
        .flatten()
        .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
        .map(genome_location)
        .collect::<Vec<_>>();
    let terminal = !qualifying_genomes.is_empty();
    let (next, migration) = evolve_champion_islands(identity, generation, &ranked, terminal)?;
    if let Some((relative, _, record)) = &migration {
        write_or_verify(&root.join(relative), &canonical_json(record)?)?;
    }
    let islands = outcomes
        .into_iter()
        .enumerate()
        .map(|(island, slots)| {
            let fresh = slots
                .iter()
                .filter(|outcome| outcome.cache_source.is_none())
                .count();
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            IslandGenerationState {
                island,
                logical_slots: ISLAND_POPULATION,
                fresh_evaluations: fresh,
                valid_candidates: valid,
                best_local_slot: ranked[island][0].local_slot,
                best_global_slot: ranked[island][0].global_slot,
                best_semantic_sha256: ranked[island][0].member.semantic_sha256.clone(),
                best_rmse: ranked[island][0].rmse(),
                ranked_local_slots: ranked[island]
                    .iter()
                    .map(|outcome| outcome.local_slot)
                    .collect(),
                slots,
                next_population: next[island].clone(),
            }
        })
        .collect();
    Ok((
        IslandGenerationCheckpoint {
            schema: CHAMPION_CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: identity.run_id.clone(),
            dataset_hash: identity.dataset.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256,
            logical_slots: TOTAL_POPULATION,
            fresh_evaluations,
            cumulative_fresh_evaluations,
            valid_candidates,
            fresh_g0_rmse: g0_rmse,
            global_best: genome_location(&all_ranked[0]),
            global_best_rmse: all_ranked[0].rmse(),
            qualifying_descendant_win: terminal,
            qualifying_genomes,
            result_label: terminal.then(|| "new_leader_found".to_string()),
            migration_file: migration.as_ref().map(|value| value.0.clone()),
            migration_sha256: migration.as_ref().map(|value| value.1.clone()),
            islands,
        },
        next,
    ))
}
