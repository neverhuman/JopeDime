use super::*;

pub(super) fn island_run_id(base_run_id: &str) -> String {
    format!("{base_run_id}-extension-g20-g29-islands4x30")
}

pub(super) fn island_root(base_run_id: &str) -> PathBuf {
    run_root(&island_run_id(base_run_id))
}

pub(super) fn champion_run_id(base_run_id: &str) -> String {
    format!("{base_run_id}-champions-g30-plus-islands4x30")
}

pub(super) fn champion_root(base_run_id: &str) -> PathBuf {
    run_root(&champion_run_id(base_run_id))
}

pub(super) fn global_slot(island: usize, local_slot: usize) -> usize {
    island * ISLAND_POPULATION + local_slot
}

#[cfg(test)]
pub(super) fn island_slot(global: usize) -> Result<(usize, usize)> {
    if global >= TOTAL_POPULATION {
        bail!("global island slot {global} is out of range");
    }
    Ok((global / ISLAND_POPULATION, global % ISLAND_POPULATION))
}

pub(super) fn rank_island_outcomes(
    left: &IslandSlotOutcome,
    right: &IslandSlotOutcome,
) -> Ordering {
    left.rmse()
        .total_cmp(&right.rmse())
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .rust_source_len
                .cmp(&right.member.rust_source_len)
        })
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
        .then_with(|| left.global_slot.cmp(&right.global_slot))
}

pub(super) fn island_offspring_seed(
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
) -> u64 {
    let text = format!("{RNG_SEED}:{generation}:{island}:{local_slot}:{retry}");
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

pub(super) fn island_sequence(island: usize) -> u64 {
    let digest = Sha256::digest(format!("{RNG_SEED}:island-stream:{island}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

pub(super) fn old_parent_ref(generation: usize, outcome: &SlotOutcome) -> IslandParentRef {
    IslandParentRef {
        generation,
        island: None,
        local_slot: outcome.slot,
        global_slot: outcome.slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

pub(super) fn island_parent_ref(outcome: &IslandSlotOutcome) -> IslandParentRef {
    IslandParentRef {
        generation: outcome.generation,
        island: Some(outcome.island),
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

pub(super) fn evolved_member(
    parents: &[Member],
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
    features: usize,
) -> Result<(Member, Vec<usize>)> {
    let seed = island_offspring_seed(generation, island, local_slot, retry);
    let mut rng = Pcg32::new(seed, island_sequence(island));
    let first = tournament_winner(&mut rng, parents.len());
    let first_parent = &parents[first];
    if rng.next_f64() < 0.25 {
        let second = tournament_winner(&mut rng, parents.len());
        let second_parent = &parents[second];
        crossover_pipelines(
            &first_parent.canonical_genome,
            &second_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
        )
        .map(|genome| {
            (
                Member::evolved(
                    genome,
                    Ancestry {
                        operation: "crossover".to_string(),
                        parents: vec![
                            first_parent.semantic_sha256.clone(),
                            second_parent.semantic_sha256.clone(),
                        ],
                    },
                ),
                vec![first, second],
            )
        })
        .map_err(anyhow::Error::msg)
    } else {
        mutate_pipeline(
            &first_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map(|genome| {
            (
                Member::evolved(
                    genome,
                    Ancestry {
                        operation: "mutation".to_string(),
                        parents: vec![first_parent.semantic_sha256.clone()],
                    },
                ),
                vec![first],
            )
        })
        .map_err(anyhow::Error::msg)
    }
}

pub(super) fn champion_complexity_level(generation: usize) -> usize {
    1 + generation
        .saturating_sub(CHAMPION_START_GENERATION)
        .div_euclid(10)
        .min(3)
}

pub(super) fn champion_evolved_member(
    parents: &[Member],
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
    features: usize,
) -> Result<(Member, Vec<usize>)> {
    let level = champion_complexity_level(generation);
    let seed = island_offspring_seed(generation, island, local_slot, retry);
    let mut rng = Pcg32::new(seed, island_sequence(island));
    let first = tournament_winner(&mut rng, parents.len());
    let first_parent = &parents[first];
    let crossover_probability = (0.25 + 0.10 * (level.saturating_sub(1) as f64)).min(0.55);
    let (mut genome, parent_indices, operation) = if rng.next_f64() < crossover_probability {
        let second = tournament_winner(&mut rng, parents.len());
        let genome = crossover_pipelines(
            &first_parent.canonical_genome,
            &parents[second].canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
        )
        .map_err(anyhow::Error::msg)?;
        (genome, vec![first, second], "crossover")
    } else {
        let genome = mutate_pipeline(
            &first_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map_err(anyhow::Error::msg)?;
        (genome, vec![first], "mutation")
    };
    for _ in 1..level {
        genome = mutate_pipeline(
            &genome.canonical,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map_err(anyhow::Error::msg)?;
    }
    let parent_hashes = parent_indices
        .iter()
        .map(|index| parents[*index].semantic_sha256.clone())
        .collect();
    Ok((
        Member::evolved(
            genome,
            Ancestry {
                operation: format!("champion_{operation}_depth_{level}"),
                parents: parent_hashes,
            },
        ),
        parent_indices,
    ))
}
