use super::*;

pub(super) fn make_checkpoint(
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
        bail!("cannot checkpoint incomplete island outcomes");
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
    let (next, migration) = evolve_islands(identity, generation, &ranked)?;
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
            Ok(IslandGenerationState {
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
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let checkpoint = IslandGenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
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
        qualifying_descendant_win: !qualifying_genomes.is_empty(),
        result_label: (!qualifying_genomes.is_empty()).then(|| "new_leader_found".to_string()),
        qualifying_genomes,
        migration_file: migration.as_ref().map(|value| value.0.clone()),
        migration_sha256: migration.as_ref().map(|value| value.1.clone()),
        islands,
    };
    Ok((checkpoint, next))
}

pub(super) fn receipt_request_path(root: &Path, receipt: &CandidateReceipt) -> PathBuf {
    root.join("requests")
        .join(&receipt.purpose)
        .join(format!("generation-{:03}", receipt.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            receipt.slot, receipt.attempt
        ))
}

pub(super) fn validate_population_parent(
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    generation: usize,
    parent: &IslandParentRef,
) -> Result<()> {
    if parent.generation + 1 != generation {
        bail!("island lineage parent is not from the preceding generation");
    }
    if parent.generation < START_GENERATION {
        if parent.island.is_some()
            || parent.global_slot != parent.local_slot
            || combined
                .get(parent.generation)
                .and_then(|checkpoint| checkpoint.0.slots.get(parent.local_slot))
                .is_none_or(|outcome| outcome.member.semantic_sha256 != parent.semantic_sha256)
        {
            bail!("island lineage does not bind to generations 0-19");
        }
    } else {
        let island = parent.island.context("island parent has no island")?;
        if global_slot(island, parent.local_slot) != parent.global_slot
            || checkpoints
                .get(parent.generation - START_GENERATION)
                .and_then(|checkpoint| checkpoint.0.islands.get(island))
                .and_then(|state| state.slots.get(parent.local_slot))
                .is_none_or(|outcome| {
                    outcome.member.semantic_sha256 != parent.semantic_sha256
                        || outcome.global_slot != parent.global_slot
                })
        {
            bail!("island lineage parent location mismatch");
        }
    }
    Ok(())
}

pub(super) fn validate_checkpoint_history(
    root: &Path,
    identity: &IslandCampaignIdentity,
    binding: &IslandDatasetBinding,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
) -> Result<usize> {
    let mut expected_population = generation_20_populations(combined, identity.dataset.features)?;
    let expected_hashes = expected_population
        .iter()
        .enumerate()
        .map(|(island, population)| {
            Ok(IslandPopulationHash {
                island,
                sha256: sha256_bytes(&canonical_json(population)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if expected_hashes != binding.generation_20_island_population_hashes {
        bail!("generation 20 island population hashes changed");
    }
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut cumulative_fresh = 0;
    let mut observed = BTreeMap::<(usize, usize), IslandSlotOutcome>::new();
    for (offset, (checkpoint, _)) in checkpoints.iter().enumerate() {
        let generation = START_GENERATION + offset;
        if checkpoint.generation != generation {
            bail!("island checkpoint generation sequence mismatch");
        }
        let mut outcomes = Vec::with_capacity(ISLANDS);
        for (island, expected_island) in expected_population[..ISLANDS].iter().enumerate() {
            let state = &checkpoint.islands[island];
            if state.island != island
                || state.logical_slots != ISLAND_POPULATION
                || state.slots.len() != ISLAND_POPULATION
                || state.next_population.len()
                    != if generation + 1 < END_GENERATION {
                        ISLAND_POPULATION
                    } else {
                        0
                    }
            {
                bail!("incomplete 30-slot island {island} in generation {generation}");
            }
            let mut fresh = 0;
            let mut valid = 0;
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
                    bail!("island population or global-slot mapping mismatch");
                }
                for parent in &outcome.parents {
                    validate_population_parent(combined, checkpoints, generation, parent)?;
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
                    bail!("island checkpoint contains a noncanonical genome");
                }
                let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                    bail!("island receipt hash mismatch");
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
                        bail!("island cache source is not earlier in current-binary history");
                    }
                    let source_outcome = observed
                        .get(&(source.generation, source.slot))
                        .context("island cache source is absent")?;
                    if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                        || source_outcome.receipt_file != outcome.receipt_file
                        || source_outcome.receipt_sha256 != outcome.receipt_sha256
                    {
                        bail!("island cache source mismatch");
                    }
                } else {
                    if receipt.generation != generation || receipt.slot != outcome.global_slot {
                        bail!("fresh island receipt has the wrong logical slot");
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
                bail!("island ranking or validity mismatch");
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
            bail!("island global accounting or qualifying-genome list mismatch");
        }
        let mut ranked = outcomes;
        for island in &mut ranked {
            island.sort_by(rank_island_outcomes);
        }
        let (expected_next, migration) = evolve_islands(identity, generation, &ranked)?;
        match migration {
            Some((relative, sha256, record)) => {
                let bytes = fs::read(root.join(&relative))?;
                if checkpoint.migration_file.as_deref() != Some(&relative)
                    || checkpoint.migration_sha256.as_deref() != Some(&sha256)
                    || sha256_bytes(&bytes) != sha256
                    || bytes != canonical_json(&record)?
                {
                    bail!("island migration record mismatch");
                }
            }
            None if checkpoint.migration_file.is_none()
                && checkpoint.migration_sha256.is_none() => {}
            None => bail!("unexpected island migration binding"),
        }
        if checkpoint
            .islands
            .iter()
            .enumerate()
            .any(|(island, state)| state.next_population != expected_next[island])
        {
            bail!("island next population mismatch");
        }
        expected_population = expected_next;
    }
    Ok(cumulative_fresh)
}
