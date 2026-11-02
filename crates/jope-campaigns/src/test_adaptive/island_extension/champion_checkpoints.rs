use super::*;

pub(super) fn load_champion_checkpoints(
    root: &Path,
    identity: &IslandCampaignIdentity,
    initial_terminal_sha256: &str,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = initial_terminal_sha256.to_string();
    let mut generation = CHAMPION_START_GENERATION;
    loop {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: IslandGenerationCheckpoint = serde_json::from_slice(&bytes)?;
        let terminal = checkpoint.qualifying_descendant_win;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHAMPION_CHECKPOINT_SCHEMA
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
                    || island.next_population.len() != if terminal { 0 } else { ISLAND_POPULATION }
            })
        {
            bail!("invalid champion checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = sha256.clone();
        checkpoints.push((checkpoint, sha256));
        generation += 1;
        if terminal {
            if checkpoint_path(root, generation).is_file() {
                bail!("champion history continues after a strict leader crossing");
            }
            break;
        }
    }
    Ok(checkpoints)
}

pub(super) fn validate_champion_history(
    root: &Path,
    identity: &IslandCampaignIdentity,
    initial: &[(IslandGenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
) -> Result<usize> {
    let mut expected_population = generation_30_populations(initial, identity.dataset.features)?;
    let lineage_checkpoints = initial
        .iter()
        .cloned()
        .chain(checkpoints.iter().cloned())
        .collect::<Vec<_>>();
    let combined = prior_history(
        &identity.run_id[..identity
            .run_id
            .find("-champions-g30-plus")
            .context("invalid champion run id")?],
        &identity.dataset,
    )?;
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut cumulative_fresh = 0usize;
    let mut observed = BTreeMap::<(usize, usize), IslandSlotOutcome>::new();
    for (offset, (checkpoint, _)) in checkpoints.iter().enumerate() {
        let generation = CHAMPION_START_GENERATION + offset;
        if checkpoint.generation != generation {
            bail!("champion checkpoint generation sequence mismatch");
        }
        let mut outcomes = Vec::with_capacity(ISLANDS);
        for (island, expected_island) in expected_population[..ISLANDS].iter().enumerate() {
            let state = &checkpoint.islands[island];
            let terminal = checkpoint.qualifying_descendant_win;
            if state.island != island
                || state.logical_slots != ISLAND_POPULATION
                || state.slots.len() != ISLAND_POPULATION
                || state.next_population.len() != if terminal { 0 } else { ISLAND_POPULATION }
            {
                bail!("incomplete champion island {island} in generation {generation}");
            }
            let mut fresh = 0usize;
            let mut valid = 0usize;
            for (local_slot, expected) in expected_island[..ISLAND_POPULATION].iter().enumerate() {
                let outcome = &state.slots[local_slot];
                if outcome.generation != generation
                    || outcome.island != island
                    || outcome.local_slot != local_slot
                    || outcome.global_slot != global_slot(island, local_slot)
                    || outcome.member != expected.member
                    || outcome.parents != expected.parents
                    || outcome.member.ancestry.parents
                        != outcome
                            .parents
                            .iter()
                            .map(|parent| parent.semantic_sha256.clone())
                            .collect::<Vec<_>>()
                {
                    bail!("champion population or slot mapping mismatch");
                }
                for parent in &outcome.parents {
                    validate_population_parent(
                        &combined,
                        &lineage_checkpoints,
                        generation,
                        parent,
                    )?;
                }
                let canonical = canonicalize_for_features(
                    &outcome.member.canonical_genome,
                    identity.dataset.features,
                )?;
                if canonical.canonical != outcome.member.canonical_genome
                    || canonical.semantic_sha256 != outcome.member.semantic_sha256
                    || canonical.complexity != outcome.member.complexity
                    || canonical.rust_source_len != outcome.member.rust_source_len
                {
                    bail!("champion checkpoint contains a noncanonical genome");
                }
                let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                    bail!("champion receipt hash mismatch");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                let request: CandidateRequest =
                    serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
                validate_receipt_against_request(&receipt, &request, test)?;
                validate_island_receipt_binding(identity, outcome, &receipt, &request)?;
                if let Some(source) = &outcome.cache_source {
                    if source.generation > generation
                        || (source.generation == generation && source.slot >= outcome.global_slot)
                    {
                        bail!("champion cache source is not earlier");
                    }
                    let source_outcome = observed
                        .get(&(source.generation, source.slot))
                        .context("champion cache source is absent")?;
                    if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                        || source_outcome.receipt_file != outcome.receipt_file
                        || source_outcome.receipt_sha256 != outcome.receipt_sha256
                    {
                        bail!("champion cache source mismatch");
                    }
                } else {
                    if receipt.generation != generation || receipt.slot != outcome.global_slot {
                        bail!("fresh champion receipt has the wrong slot");
                    }
                    fresh += 1;
                }
                valid += usize::from(outcome.valid);
                observed.insert((generation, outcome.global_slot), outcome.clone());
            }
            let mut ranked = state.slots.clone();
            ranked.sort_by(rank_island_outcomes);
            if valid != state.valid_candidates
                || valid < ISLAND_MIN_VALID
                || fresh != state.fresh_evaluations
                || state.ranked_local_slots
                    != ranked
                        .iter()
                        .map(|outcome| outcome.local_slot)
                        .collect::<Vec<_>>()
                || state.best_local_slot != ranked[0].local_slot
                || state.best_global_slot != ranked[0].global_slot
                || state.best_semantic_sha256 != ranked[0].member.semantic_sha256
                || state.best_rmse.to_bits() != ranked[0].rmse().to_bits()
            {
                bail!("champion island ranking or validity mismatch");
            }
            outcomes.push(state.slots.clone());
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        let fresh = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.cache_source.is_none())
            .count();
        cumulative_fresh += fresh;
        let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
        all_ranked.sort_by(rank_island_outcomes);
        let qualifying = outcomes
            .iter()
            .flatten()
            .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
            .map(genome_location)
            .collect::<Vec<_>>();
        if checkpoint.valid_candidates != valid
            || valid < GLOBAL_MIN_VALID
            || checkpoint.fresh_evaluations != fresh
            || checkpoint.cumulative_fresh_evaluations != cumulative_fresh
            || checkpoint.fresh_g0_rmse.to_bits() != g0_rmse.to_bits()
            || checkpoint.global_best != genome_location(&all_ranked[0])
            || checkpoint.global_best_rmse.to_bits() != all_ranked[0].rmse().to_bits()
            || checkpoint.qualifying_genomes != qualifying
            || checkpoint.qualifying_descendant_win != !qualifying.is_empty()
            || checkpoint.result_label
                != (!qualifying.is_empty()).then(|| "new_leader_found".to_string())
        {
            bail!("champion global accounting mismatch");
        }
        let mut ranked = outcomes;
        for island in &mut ranked {
            island.sort_by(rank_island_outcomes);
        }
        let (expected_next, migration) = evolve_champion_islands(
            identity,
            generation,
            &ranked,
            checkpoint.qualifying_descendant_win,
        )?;
        match migration {
            Some((relative, sha256, record)) => {
                let bytes = fs::read(root.join(&relative))?;
                if checkpoint.migration_file.as_deref() != Some(&relative)
                    || checkpoint.migration_sha256.as_deref() != Some(&sha256)
                    || sha256_bytes(&bytes) != sha256
                    || bytes != canonical_json(&record)?
                {
                    bail!("champion migration record mismatch");
                }
            }
            None if checkpoint.migration_file.is_none()
                && checkpoint.migration_sha256.is_none() => {}
            None => bail!("unexpected champion migration binding"),
        }
        if checkpoint
            .islands
            .iter()
            .enumerate()
            .any(|(island, state)| state.next_population != expected_next[island])
        {
            bail!("champion next population mismatch");
        }
        expected_population = expected_next;
    }
    Ok(cumulative_fresh)
}

pub(super) fn champion_generation_progress(
    dataset_hash: &str,
    checkpoint: &IslandGenerationCheckpoint,
    completed_champions: usize,
) {
    let island_text = checkpoint
        .islands
        .iter()
        .map(|island| {
            format!(
                "i{} valid {}/30 best {:.17}",
                island.island, island.valid_candidates, island.best_rmse
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let cached = TOTAL_POPULATION - checkpoint.fresh_evaluations;
    let migration = if checkpoint.migration_file.is_some() {
        "ring top-2 complete"
    } else {
        "none"
    };
    let mut node_counts = BTreeMap::<&str, usize>::new();
    for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
        if outcome.cache_source.is_none() {
            *node_counts.entry(outcome.node.as_str()).or_default() += 1;
        }
    }
    let nodes = node_counts
        .into_iter()
        .map(|(node, count)| format!("{node}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "champion continuation {dataset_hash} generation {} depth {}: {island_text}; global best {:.17}; fresh {} cache {cached}; migration {migration}; nodes [{nodes}]; cumulative surpassed {}/42",
        checkpoint.generation,
        champion_complexity_level(checkpoint.generation),
        checkpoint.global_best_rmse,
        checkpoint.fresh_evaluations,
        EXISTING_SURPASSED + completed_champions + usize::from(checkpoint.qualifying_descendant_win),
    );
}

pub(super) fn validate_champion_replays(
    root: &Path,
    identity: &IslandCampaignIdentity,
    result: &ChampionDatasetResult,
    test: &RegressionData,
) -> Result<()> {
    if result.replay.len() != scq_batch::NODES.len() {
        bail!("champion replay count mismatch");
    }
    let original: CandidateReceipt =
        serde_json::from_slice(&fs::read(root.join(&result.champion.receipt_file))?)?;
    let original_rmse = original
        .metrics
        .as_ref()
        .context("champion source receipt failed")?
        .rmse;
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!("champion replay receipt hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request: CandidateRequest =
            serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
        validate_receipt_against_request(&receipt, &request, test)?;
        let rmse = receipt
            .metrics
            .as_ref()
            .context("champion cross-node replay failed")?
            .rmse;
        let difference = original
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if receipt.purpose != "replay-final"
            || receipt.generation != result.stopping_generation
            || receipt.slot != slot
            || receipt.run_id != identity.run_id
            || receipt.binary_sha256 != identity.release_binary_sha256
            || receipt.semantic_sha256 != result.champion.semantic_sha256
            || receipt.node != scq_batch::NODES[slot]
            || (rmse - original_rmse).abs() > 1e-12
            || difference > 1e-10
            || replay.rmse.to_bits() != rmse.to_bits()
        {
            bail!("champion replay disagreement");
        }
    }
    Ok(())
}

pub(super) fn validate_champion_dataset(
    registry: &Path,
    manifest: &ChampionManifest,
    binding: &ChampionDatasetBinding,
    spec: &DatasetSpec,
) -> Result<ChampionDatasetResult> {
    if sha256_file(Path::new(&binding.initial_result_file))? != binding.initial_result_sha256
        || sha256_file(Path::new(&binding.generation_29_checkpoint_file))?
            != binding.generation_29_checkpoint_sha256
    {
        bail!("champion source artifacts changed");
    }
    let root = champion_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let identity = champion_identity(manifest, spec);
    let campaign_bytes = fs::read(root.join("campaign.v1.json"))?;
    let recorded_identity: IslandCampaignIdentity = serde_json::from_slice(&campaign_bytes)?;
    if campaign_bytes != canonical_json(&recorded_identity)?
        || recorded_identity != identity
        || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
    {
        bail!("champion dataset identity mismatch");
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let initial = load_initial_island_checkpoints(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_champion_checkpoints(&root, &identity, &binding.generation_29_checkpoint_sha256)?;
    if checkpoints
        .last()
        .is_none_or(|checkpoint| !checkpoint.0.qualifying_descendant_win)
    {
        bail!("champion dataset has no terminal leader crossing");
    }
    let cumulative_fresh =
        validate_champion_history(&root, &identity, &initial, &checkpoints, &test)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let all_checkpoints = initial
        .iter()
        .cloned()
        .chain(checkpoints.iter().cloned())
        .collect::<Vec<_>>();
    let champion = island_genome_record(
        &root,
        &combined,
        &all_checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: ChampionDatasetResult = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&result)?
        || result.schema != CHAMPION_RESULT_SCHEMA
        || result.benchmark_valid
        || result.base_run_id != manifest.base_run_id
        || result.champion_run_id != manifest.champion_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != "new_leader_found"
        || result.start_generation != CHAMPION_START_GENERATION
        || result.stopping_generation != terminal.generation
        || result.generations_evaluated != checkpoints.len()
        || result.logical_evaluations != checkpoints.len() * TOTAL_POPULATION
        || result.fresh_evaluations != cumulative_fresh
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.champion != champion
        || result.champion.rmse >= spec.leader_rmse
        || result.release_binary_sha256 != manifest.release_binary_sha256
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!("champion final result mismatch");
    }
    validate_champion_replays(&root, &identity, &result, &test)?;
    Ok(result)
}
