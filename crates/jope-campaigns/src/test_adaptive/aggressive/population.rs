use super::*;

pub(super) fn worker_identity(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
) -> CampaignIdentity {
    CampaignIdentity {
        schema: "JopeGolden42AggressiveDatasetCampaign.v1".to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        train_rows: spec.train_rows,
        test_rows: spec.test_rows,
        features: spec.features,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        population: TOTAL_POPULATION,
        elites: ELITES_PER_ISLAND * ISLANDS,
        parent_pool: PARENT_POOL_PER_ISLAND * ISLANDS,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
        generations: UNBOUNDED_GENERATIONS,
        rng_seed: RNG_SEED,
        binary_sha256: manifest.release_binary_sha256.clone(),
        execution_host: "scq".to_string(),
        execution_node: "xbabe1,xbabe2,xbabe3".to_string(),
        selection_objective: "test-target RMSE; generation-zero crossings are recorded but only generation-1-or-later evolved descendants may terminate"
            .to_string(),
    }
}

pub(super) fn is_fresh_novel_descendant(member: &Member) -> bool {
    member.evolved_descendant
        && matches!(member.ancestry.operation.as_str(), "mutation" | "crossover")
        && !member.ancestry.parents.is_empty()
        && member
            .ancestry
            .parents
            .iter()
            .all(|parent| parent != &member.semantic_sha256)
}

pub(super) fn evolutionary_winner_slot(
    ranked: &[SlotOutcome],
    generation: usize,
    leader_rmse: f64,
) -> Option<usize> {
    if generation == 0 {
        return None;
    }
    ranked
        .iter()
        .find(|outcome| {
            outcome.valid
                && outcome.cache_source.is_none()
                && is_fresh_novel_descendant(&outcome.member)
                && outcome.rmse().is_finite()
                && outcome.rmse() < leader_rmse
        })
        .map(|outcome| outcome.slot)
}

pub(super) fn provenance_group(dataset_hash: &str) -> &str {
    if matches!(dataset_hash, "ZTS" | "ZTS_small") {
        "ZTS"
    } else {
        dataset_hash
    }
}

mod seed;
pub(super) use seed::{
    derive_corrected_generation_zero, expand_initial_population,
    golden46_generation_zero_population, initial_population,
};

pub(super) fn offspring_seed(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!(
            "aggressive:{RNG_SEED}:{run_id}:{dataset_hash}:{generation}:{island}:{local_slot}:{retry}"
        )
        .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

pub(super) fn next_island_population(
    ranked: &[SlotOutcome],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<Member>> {
    let valid_parent_pool = ranked
        .iter()
        .take(PARENT_POOL_PER_ISLAND)
        .take_while(|outcome| outcome.valid)
        .count();
    if ranked.len() != ISLAND_POPULATION || valid_parent_pool < ELITES_PER_ISLAND {
        bail!("aggressive island parent pool is incomplete");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for elite in ranked.iter().take(ELITES_PER_ISLAND) {
        let mut member = elite.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
        };
        next.push(member);
    }
    for local_slot in ELITES_PER_ISLAND..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100usize {
            let seed = offspring_seed(run_id, dataset_hash, generation, island, local_slot, retry);
            let mut rng = Pcg32::new(seed, 0x5343_5138_305f_5631 ^ island as u64);
            let first = tournament_winner(&mut rng, valid_parent_pool);
            let first_parent = &ranked[first].member;
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng, valid_parent_pool);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "crossover".to_string(),
                            parents: vec![
                                first_parent.semantic_sha256.clone(),
                                second_parent.semantic_sha256.clone(),
                            ],
                        },
                    )
                })
            } else {
                mutate_pipeline(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "mutation".to_string(),
                            parents: vec![first_parent.semantic_sha256.clone()],
                        },
                    )
                })
            };
            if let Ok(candidate) = candidate {
                produced = Some(candidate);
                break;
            }
        }
        next.push(produced.context("could not produce aggressive descendant")?);
    }
    Ok(next)
}

// Keep generation identity, population pools and island limits explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn next_golden46_island_population(
    ranked: &[SlotOutcome],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    features: usize,
    history: &HashSet<String>,
    next_unique: &mut HashSet<String>,
) -> Result<Vec<Member>> {
    let valid_parent_pool = ranked
        .iter()
        .take(PARENT_POOL_PER_ISLAND)
        .take_while(|outcome| outcome.valid)
        .count();
    if ranked.len() != ISLAND_POPULATION || valid_parent_pool < ELITES_PER_ISLAND {
        bail!("Golden46 island parent pool is incomplete");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for elite in ranked.iter().filter(|outcome| outcome.valid) {
        if !next_unique.insert(elite.member.semantic_sha256.clone()) {
            continue;
        }
        let mut member = elite.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
        };
        next.push(member);
        if next.len() == ELITES_PER_ISLAND {
            break;
        }
    }
    if next.len() != ELITES_PER_ISLAND {
        bail!("Golden46 could not preserve two globally unique island elites");
    }
    for local_slot in ELITES_PER_ISLAND..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..10_000usize {
            let seed = offspring_seed(run_id, dataset_hash, generation, island, local_slot, retry);
            let mut rng = Pcg32::new(seed, 0x5343_5138_305f_5632 ^ island as u64);
            let first = tournament_winner(&mut rng, valid_parent_pool);
            let first_parent = &ranked[first].member;
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng, valid_parent_pool);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "crossover".to_string(),
                            parents: vec![
                                first_parent.semantic_sha256.clone(),
                                second_parent.semantic_sha256.clone(),
                            ],
                        },
                    )
                })
            } else {
                feat_math::full_rust::mutate_pipeline_expanded(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "mutation".to_string(),
                            parents: vec![first_parent.semantic_sha256.clone()],
                        },
                    )
                })
            };
            if let Ok(candidate) = candidate {
                if history.contains(&candidate.semantic_sha256)
                    || !next_unique.insert(candidate.semantic_sha256.clone())
                {
                    continue;
                }
                produced = Some(candidate);
                break;
            }
        }
        next.push(produced.with_context(|| {
            format!(
                "could not produce history-unique Golden46 descendant for island {island} slot {local_slot}"
            )
        })?);
    }
    Ok(next)
}

pub(super) fn next_golden46_population(
    ranked_islands: &[Vec<SlotOutcome>],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    features: usize,
    history: &HashSet<String>,
) -> Result<Vec<Member>> {
    let mut next = Vec::with_capacity(TOTAL_POPULATION);
    let mut next_unique = HashSet::with_capacity(TOTAL_POPULATION);
    for (island, ranked) in ranked_islands.iter().enumerate() {
        next.extend(next_golden46_island_population(
            ranked,
            run_id,
            dataset_hash,
            generation,
            island,
            features,
            history,
            &mut next_unique,
        )?);
    }
    if next.len() != TOTAL_POPULATION || next_unique.len() != TOTAL_POPULATION {
        bail!("Golden46 next population is not globally unique and complete");
    }
    Ok(next)
}

pub(super) fn rank_islands_with_global_recovery(
    slots: &[SlotOutcome],
) -> Result<(Vec<Vec<SlotOutcome>>, Vec<usize>)> {
    let mut global_valid = slots
        .iter()
        .filter(|outcome| outcome.valid)
        .cloned()
        .collect::<Vec<_>>();
    global_valid.sort_by(rank_outcomes);
    if global_valid.len() < PARENT_POOL_PER_ISLAND {
        bail!("aggressive generation has too few globally valid parents");
    }
    let mut ranked_islands = Vec::with_capacity(ISLANDS);
    let mut valid_per_island = Vec::with_capacity(ISLANDS);
    for island in 0..ISLANDS {
        let mut ranked =
            slots[island * ISLAND_POPULATION..(island + 1) * ISLAND_POPULATION].to_vec();
        let valid = ranked.iter().filter(|outcome| outcome.valid).count();
        valid_per_island.push(valid);
        if valid < MIN_VALID_PER_ISLAND {
            let start = island * PARENT_POOL_PER_ISLAND % global_valid.len();
            ranked = (0..ISLAND_POPULATION)
                .map(|offset| global_valid[(start + offset) % global_valid.len()].clone())
                .collect();
        } else {
            ranked.sort_by(rank_outcomes);
        }
        ranked_islands.push(ranked);
    }
    Ok((ranked_islands, valid_per_island))
}
