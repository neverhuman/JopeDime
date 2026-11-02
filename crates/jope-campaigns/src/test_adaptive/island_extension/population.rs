use super::*;

pub(super) fn generation_20_populations(
    combined: &[(GenerationCheckpoint, String)],
    features: usize,
) -> Result<Vec<Vec<IslandPopulationMember>>> {
    if combined.len() != START_GENERATION {
        bail!("prior history does not contain generations 0-19");
    }
    let terminal = &combined.last().unwrap().0;
    if terminal.generation != START_GENERATION - 1 {
        bail!("prior terminal checkpoint is not generation 19");
    }
    let mut ranked = terminal.slots.clone();
    ranked.sort_by(rank_outcomes);
    let mut seen = HashSet::new();
    let distinct_elites = ranked
        .iter()
        .filter(|outcome| seen.insert(outcome.member.semantic_sha256.clone()))
        .take(ISLANDS * ISLAND_ELITES)
        .cloned()
        .collect::<Vec<_>>();
    if distinct_elites.len() != ISLANDS * ISLAND_ELITES {
        bail!("generation 19 has fewer than 16 distinct valid elites");
    }
    if ranked.len() < GENERATION_20_PARENT_POOL
        || ranked[..GENERATION_20_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("generation 20 parent pool is incomplete or invalid");
    }
    let parents = ranked[..GENERATION_20_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    let mut populations = Vec::with_capacity(ISLANDS);
    for island in 0..ISLANDS {
        let mut population = Vec::with_capacity(ISLAND_POPULATION);
        for rank in (island..ISLANDS * ISLAND_ELITES).step_by(ISLANDS) {
            let source = &distinct_elites[rank];
            let mut member = source.member.clone();
            member.ancestry = Ancestry {
                operation: "island_elite".to_string(),
                parents: vec![source.member.semantic_sha256.clone()],
            };
            population.push(IslandPopulationMember {
                member,
                parents: vec![old_parent_ref(START_GENERATION - 1, source)],
            });
        }
        for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
            let mut produced = None;
            for retry in 0..100 {
                let candidate = evolved_member(
                    &parents,
                    START_GENERATION,
                    island,
                    local_slot,
                    retry,
                    features,
                );
                if let Ok((member, parent_indices)) = candidate {
                    let refs = parent_indices
                        .into_iter()
                        .map(|index| old_parent_ref(START_GENERATION - 1, &ranked[index]))
                        .collect();
                    produced = Some(IslandPopulationMember {
                        member,
                        parents: refs,
                    });
                    break;
                }
            }
            population.push(produced.context("could not create generation 20 island descendant")?);
        }
        populations.push(population);
    }
    Ok(populations)
}

pub(super) fn next_island_population(
    ranked: &[IslandSlotOutcome],
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<IslandPopulationMember>> {
    if ranked.len() != ISLAND_POPULATION
        || ranked[..ISLAND_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("island {island} parent pool is incomplete or invalid");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for source in ranked.iter().take(ISLAND_ELITES) {
        let mut member = source.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![source.member.semantic_sha256.clone()],
        };
        next.push(IslandPopulationMember {
            member,
            parents: vec![island_parent_ref(source)],
        });
    }
    let parents = ranked[..ISLAND_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100 {
            if let Ok((member, parent_indices)) =
                evolved_member(&parents, generation, island, local_slot, retry, features)
            {
                let refs = parent_indices
                    .into_iter()
                    .map(|index| island_parent_ref(&ranked[index]))
                    .collect();
                produced = Some(IslandPopulationMember {
                    member,
                    parents: refs,
                });
                break;
            }
        }
        next.push(produced.context("could not create valid island descendant")?);
    }
    Ok(next)
}

pub(super) fn next_champion_population(
    ranked: &[IslandSlotOutcome],
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<IslandPopulationMember>> {
    if ranked.len() != ISLAND_POPULATION
        || ranked[..ISLAND_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("champion island {island} parent pool is incomplete or invalid");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for source in ranked.iter().take(ISLAND_ELITES) {
        let mut member = source.member.clone();
        member.ancestry = Ancestry {
            operation: "champion_elite".to_string(),
            parents: vec![source.member.semantic_sha256.clone()],
        };
        next.push(IslandPopulationMember {
            member,
            parents: vec![island_parent_ref(source)],
        });
    }
    let parents = ranked[..ISLAND_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100 {
            if let Ok((member, parent_indices)) =
                champion_evolved_member(&parents, generation, island, local_slot, retry, features)
            {
                produced = Some(IslandPopulationMember {
                    member,
                    parents: parent_indices
                        .into_iter()
                        .map(|index| island_parent_ref(&ranked[index]))
                        .collect(),
                });
                break;
            }
        }
        next.push(produced.context("could not create valid champion descendant")?);
    }
    Ok(next)
}

pub(super) fn migration_genome(outcome: &IslandSlotOutcome, rank: usize) -> MigrationGenome {
    MigrationGenome {
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        rank,
        member: outcome.member.clone(),
    }
}

pub(super) fn migrate_rankings(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if !MIGRATION_GENERATIONS.contains(&generation) {
        bail!("invalid ring migration generation {generation}");
    }
    migrate_rankings_with_policy(
        run_id,
        dataset_hash,
        generation,
        ranked,
        MigrationPolicy::fixed(),
    )
}

pub(super) fn migrate_rankings_champion(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if generation < CHAMPION_START_GENERATION + 1 || generation.is_multiple_of(2) {
        bail!("invalid champion ring migration generation {generation}");
    }
    migrate_rankings_with_policy(
        run_id,
        dataset_hash,
        generation,
        ranked,
        MigrationPolicy::champion(),
    )
}

pub(super) fn migrate_rankings_with_policy(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
    policy: MigrationPolicy,
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if ranked.len() != ISLANDS
        || ranked
            .iter()
            .any(|island| island.len() != ISLAND_POPULATION)
    {
        bail!("invalid ring migration input for generation {generation}");
    }
    let original = ranked.to_vec();
    let mut migrated = ranked.to_vec();
    let mut transfers = Vec::with_capacity(ISLANDS * 2);
    for source_island in 0..ISLANDS {
        let destination_island = (source_island + 1) % ISLANDS;
        for source_rank in 0..2 {
            let destination_rank = ISLAND_POPULATION - 2 + source_rank;
            let migrant = &original[source_island][source_rank];
            let replaced = &original[destination_island][destination_rank];
            transfers.push(MigrationTransfer {
                source_island,
                destination_island,
                source_rank,
                destination_replaced_rank: destination_rank,
                migrant: migration_genome(migrant, source_rank),
                replaced: migration_genome(replaced, destination_rank),
            });
            migrated[destination_island][destination_rank] = migrant.clone();
        }
    }
    for island in &mut migrated {
        island.sort_by(rank_island_outcomes);
    }
    Ok((
        migrated,
        MigrationRecord {
            schema: MIGRATION_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset_hash: dataset_hash.to_string(),
            generation,
            policy,
            transfers,
        },
    ))
}
