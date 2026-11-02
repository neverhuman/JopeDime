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

pub(super) fn derive_corrected_generation_zero(
    parent_seed: &SeedProvenance,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<(Vec<Member>, Vec<SelectedGenerationZeroSeed>)> {
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    let mut candidates = Vec::with_capacity(TOTAL_POPULATION);
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    for slot in 0..TOTAL_POPULATION {
        let island = slot / ISLAND_POPULATION;
        let local_slot = slot % ISLAND_POPULATION;
        let mut produced = None;
        for retry in 0..10_000usize {
            let steps = 1 + (slot + retry) % 4;
            let mut source = parent_seed.canonical_genome.clone();
            let mut evolved = None;
            let mut valid_chain = true;
            for step in 0..steps {
                let digest = Sha256::digest(
                    format!(
                        "golden46-corrected-g0:{run_id}:{}:{island}:{local_slot}:{retry}:{step}",
                        spec.dataset_hash
                    )
                    .as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                let Ok(next) =
                    feat_math::full_rust::mutate_pipeline_expanded(&source, seed, spec.features)
                else {
                    valid_chain = false;
                    break;
                };
                source = next.canonical.clone();
                evolved = Some(next);
            }
            if !valid_chain {
                continue;
            }
            let evolved = evolved.context("generation-zero mutation chain was empty")?;
            if evolved.semantic_sha256 == parent_seed.semantic_sha256
                || !seen.insert(evolved.semantic_sha256.clone())
            {
                continue;
            }
            produced = Some(evolved);
            break;
        }
        let evolved = produced.with_context(|| {
            format!("could not derive unique corrected generation-zero candidate {slot}")
        })?;
        let member = Member::evolved(
            evolved,
            Ancestry {
                operation: "generation_zero_mutation".to_string(),
                parents: vec![parent_seed.semantic_sha256.clone()],
            },
        );
        candidates.push(SelectedGenerationZeroSeed {
            island,
            local_slot,
            seed_class: "unique_full_genome_mutation".to_string(),
            provenance: SeedProvenance {
                slot,
                canonical_genome: member.canonical_genome.clone(),
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: member.complexity,
                rust_source_len: member.rust_source_len,
                source_kind: "corrected_generation_zero_candidate".to_string(),
                source_file: parent_seed.source_file.clone(),
                source_file_sha256: parent_seed.source_file_sha256.clone(),
                source_record: slot,
                source_rank: slot + 1,
                source_score: None,
            },
        });
        population.push(member);
    }
    Ok((population, candidates))
}

pub(super) fn golden46_generation_zero_population(
    _registry: &Path,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    let checkpoint_dir =
        dataset_root(GOLDEN46_CORRECTED_PARENT_RUN, &spec.dataset_hash).join("checkpoints");
    let mut paths = fs::read_dir(&checkpoint_dir)
        .with_context(|| {
            format!(
                "read corrected-parent checkpoints for {}",
                spec.dataset_hash
            )
        })?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut parent = None::<(SeedProvenance, String)>;
    for path in paths {
        let bytes = fs::read(&path)?;
        let checkpoint: AggressiveCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.run_id != GOLDEN46_CORRECTED_PARENT_RUN
            || checkpoint.dataset_hash != spec.dataset_hash
        {
            bail!("invalid corrected-parent checkpoint {}", path.display());
        }
        for outcome in checkpoint.slots.iter().filter(|outcome| outcome.valid) {
            let rmse = outcome.rmse();
            if !rmse.is_finite()
                || parent
                    .as_ref()
                    .is_some_and(|(seed, _)| seed.source_score.is_some_and(|best| rmse >= best))
            {
                continue;
            }
            let digest = sha256_bytes(&bytes);
            parent = Some((
                SeedProvenance {
                    slot: outcome.slot,
                    canonical_genome: outcome.member.canonical_genome.clone(),
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    complexity: outcome.member.complexity,
                    rust_source_len: outcome.member.rust_source_len,
                    source_kind: "corrected_test_adaptive_parent".to_string(),
                    source_file: path.to_string_lossy().into_owned(),
                    source_file_sha256: digest.clone(),
                    source_record: outcome.slot,
                    source_rank: 1,
                    source_score: Some(rmse),
                },
                digest,
            ));
        }
    }
    let (parent_seed, parent_checkpoint_sha256) =
        parent.context("corrected Golden46 parent run has no valid candidate")?;
    let (population, candidates) = derive_corrected_generation_zero(&parent_seed, run_id, spec)?;
    let population_sha256 = sha256_bytes(&canonical_json(&candidates)?);
    let snapshot = GenerationZeroSeedSnapshot {
        schema: "JopeGolden46GenerationZeroSeeds.v2".to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        provenance_group: provenance_group(&spec.dataset_hash).to_string(),
        arm: "corrected-test-adaptive-continuation".to_string(),
        strategy: "single best preserved v2 parent; 36 deterministic unique full-genome mutations"
            .to_string(),
        parent_run_id: GOLDEN46_CORRECTED_PARENT_RUN.to_string(),
        parent_checkpoint_sha256,
        parent_seed,
        population_sha256,
        generation_zero_can_terminate: false,
        candidates,
    };
    let snapshot_path = dataset_root(run_id, &spec.dataset_hash).join(GENERATION_ZERO_SEEDS_FILE);
    atomic_immutable_write(&snapshot_path, &canonical_json(&snapshot)?)?;
    Ok(population)
}

pub(super) fn initial_population(
    registry: &Path,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    if is_golden46_run(run_id) {
        return golden46_generation_zero_population(registry, run_id, spec);
    }
    let seed_path = dataset_root(run_id, &spec.dataset_hash).join("transfer-seeds.v1.json");
    let base = load_or_create_transfer_seed_manifest(&seed_path, registry, spec)?
        .seeds
        .iter()
        .take(ISLAND_POPULATION)
        .map(Member::from_seed)
        .collect::<Vec<_>>();
    expand_initial_population(&base, spec.features)
}

pub(super) fn expand_initial_population(base: &[Member], features: usize) -> Result<Vec<Member>> {
    if base.is_empty() || base.len() > ISLAND_POPULATION {
        bail!("aggressive seed baseline does not fit one island");
    }
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        for local_slot in 0..ISLAND_POPULATION {
            if island == 0 && local_slot < base.len() {
                let member = base[local_slot].clone();
                if !seen.insert(member.semantic_sha256.clone()) {
                    bail!("aggressive seed baseline contains a duplicate genome");
                }
                population.push(member);
                continue;
            }
            let parent = &base[local_slot % base.len()];
            let mut produced = None;
            for retry in 0..1000usize {
                let digest = Sha256::digest(
                    format!("aggressive:g0:{island}:{local_slot}:{retry}").as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                if let Ok(genome) = mutate_pipeline(&parent.canonical_genome, seed, features) {
                    if seen.insert(genome.semantic_sha256.clone()) {
                        produced = Some(Member::evolved(
                            genome,
                            Ancestry {
                                operation: "aggressive_generation_zero_mutation".to_string(),
                                parents: vec![parent.semantic_sha256.clone()],
                            },
                        ));
                        break;
                    }
                }
            }
            population.push(
                produced.with_context(|| {
                    format!("could not derive island {island} slot {local_slot}")
                })?,
            );
        }
    }
    if population.len() != TOTAL_POPULATION {
        bail!("aggressive initial population is incomplete");
    }
    Ok(population)
}

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
