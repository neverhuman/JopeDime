use super::*;

pub(super) fn initial_population(
    registry: &Path,
    root: &Path,
    spec: &DatasetSpec,
    run_id: &str,
) -> Result<Vec<Member>> {
    let manifest = load_or_create_transfer_seed_manifest(
        &root.join("transfer-seeds.v1.json"),
        registry,
        spec,
    )?;
    initial_population_from_seeds(&manifest.seeds, run_id, &spec.dataset_hash)
}

pub(super) fn initial_population_from_seeds(
    seeds: &[SeedProvenance],
    run_id: &str,
    dataset_hash: &str,
) -> Result<Vec<Member>> {
    if seeds.len() < TOTAL_POPULATION {
        bail!(
            "TabArena island initialization requires at least {TOTAL_POPULATION} frozen \
             transfer seeds, got {}",
            seeds.len()
        );
    }
    // The selection below assumes distinct seeds: a repeated semantic hash
    // would silently place the same genome in two island slots.
    let mut distinct = HashSet::with_capacity(seeds.len());
    if let Some(repeat) = seeds
        .iter()
        .find(|seed| !distinct.insert(seed.semantic_sha256.as_str()))
    {
        bail!(
            "TabArena island initialization requires distinct transfer seeds; slot {} repeats \
             semantic hash {}",
            repeat.slot,
            repeat.semantic_sha256
        );
    }
    const STRONG_SEEDS: usize = TOTAL_POPULATION / 2;
    let mut diverse = (STRONG_SEEDS..seeds.len()).collect::<Vec<_>>();
    diverse.sort_by_key(|index| {
        sha256_bytes(
            format!("{run_id}:{dataset_hash}:{}", seeds[*index].semantic_sha256).as_bytes(),
        )
    });
    diverse.truncate(TOTAL_POPULATION - STRONG_SEEDS);
    let mut selected = Vec::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        for local in 0..STRONG_SEEDS / ISLANDS {
            selected.push(island + local * ISLANDS);
        }
        for local in 0..diverse.len() / ISLANDS {
            selected.push(diverse[island + local * ISLANDS]);
        }
    }
    if selected.len() != TOTAL_POPULATION {
        bail!("TabArena island initialization produced an incomplete seed selection");
    }
    Ok(selected
        .into_iter()
        .map(|index| {
            let seed = &seeds[index];
            Member {
                canonical_genome: seed.canonical_genome.clone(),
                semantic_sha256: seed.semantic_sha256.clone(),
                complexity: seed.complexity,
                rust_source_len: seed.rust_source_len,
                evolved_descendant: false,
                ancestry: Ancestry {
                    operation: "tabarena_seed".to_string(),
                    parents: Vec::new(),
                },
            }
        })
        .collect())
}

fn is_fresh_novel_descendant(member: &Member) -> bool {
    member.evolved_descendant
        && matches!(member.ancestry.operation.as_str(), "mutation" | "crossover")
        && !member.ancestry.parents.is_empty()
        && member
            .ancestry
            .parents
            .iter()
            .all(|parent| parent != &member.semantic_sha256)
}

pub(super) fn offspring_seed(
    run_id: &str,
    dataset: &str,
    generation: usize,
    island: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!("tabarena:{RNG_SEED}:{run_id}:{dataset}:{generation}:{island}:{slot}:{retry}")
            .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

pub(super) fn next_population(
    run_id: &str,
    dataset: &str,
    generation: usize,
    features: usize,
    slots: &[SlotOutcome],
) -> Result<Vec<Member>> {
    if slots.len() != TOTAL_POPULATION {
        bail!("TabArena checkpoint population is incomplete");
    }
    let mut ranked = (0..ISLANDS)
        .map(|island| {
            let mut values = slots[island * POPULATION..(island + 1) * POPULATION].to_vec();
            values.sort_by(rank_outcomes);
            values
        })
        .collect::<Vec<_>>();
    if generation % 2 == 1 {
        let original = ranked.clone();
        for (source, original_island) in original[..ISLANDS].iter().enumerate() {
            let destination = (source + 1) % ISLANDS;
            for rank in 0..2 {
                ranked[destination][POPULATION - 2 + rank] = original_island[rank].clone();
            }
            ranked[destination].sort_by(rank_outcomes);
        }
    }
    let mut global_ranked = slots.iter().filter(|slot| slot.valid).collect::<Vec<_>>();
    global_ranked.sort_by(|left, right| rank_outcomes(left, right));
    if global_ranked.len() < MIN_VALID_GLOBAL {
        bail!("TabArena generation has too few globally valid parents");
    }
    let mut next = Vec::with_capacity(TOTAL_POPULATION);
    for (island, ranked_island) in ranked[..ISLANDS].iter().enumerate() {
        let local_parents = ranked_island
            .iter()
            .take_while(|outcome| outcome.valid)
            .take(PARENT_POOL)
            .collect::<Vec<_>>();
        let parents = if local_parents.len() >= MIN_VALID_ISLAND {
            local_parents
        } else {
            let start = island * PARENT_POOL % global_ranked.len();
            (0..PARENT_POOL.min(global_ranked.len()))
                .map(|offset| global_ranked[(start + offset) % global_ranked.len()])
                .collect::<Vec<_>>()
        };
        for source in parents.iter().take(ELITES) {
            let mut member = source.member.clone();
            member.ancestry = Ancestry {
                operation: "elite".to_string(),
                parents: vec![source.member.semantic_sha256.clone()],
            };
            next.push(member);
        }
        for local_slot in ELITES..POPULATION {
            let mut produced = None;
            for retry in 0..100 {
                let seed =
                    offspring_seed(run_id, dataset, generation + 1, island, local_slot, retry);
                let mut rng = Pcg32::new(seed, 0x5441_4241_5245_4e41 ^ island as u64);
                let first = tournament_winner(&mut rng, parents.len());
                let first_parent = &parents[first].member;
                let candidate = if rng.next_f64() < 0.25 {
                    let second = tournament_winner(&mut rng, parents.len());
                    let second_parent = &parents[second].member;
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
                if let Ok(member) = candidate {
                    produced = Some(member);
                    break;
                }
            }
            next.push(produced.context("could not create TabArena island descendant")?);
        }
    }
    Ok(next)
}

pub(super) fn checkpoint_path(root: &Path, generation: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("generation-{generation:06}.v1.json"))
}

pub(super) fn load_checkpoints(
    root: &Path,
    identity: &CampaignIdentity,
) -> Result<Vec<(TabArenaCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    for generation in 0usize.. {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: TabArenaCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || checkpoint.run_id != identity.run_id
            || checkpoint.dataset != identity.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256
                != checkpoints
                    .last()
                    .map(|checkpoint: &(TabArenaCheckpoint, String)| checkpoint.1.clone())
            || checkpoint.logical_slots != TOTAL_POPULATION
            || checkpoint.slots.len() != TOTAL_POPULATION
            || checkpoint.next_population.len() != TOTAL_POPULATION
        {
            bail!("TabArena checkpoint envelope or chain mismatch");
        }
        checkpoints.push((checkpoint, sha256_bytes(&bytes)));
    }
    Ok(checkpoints)
}

pub(super) fn load_cache(
    root: &Path,
    checkpoints: &[(TabArenaCheckpoint, String)],
) -> Result<HashMap<String, CachedEvaluation>> {
    let mut cache = HashMap::new();
    for (checkpoint, _) in checkpoints {
        for outcome in &checkpoint.slots {
            if outcome.cache_source.is_none() {
                let bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&bytes) != outcome.receipt_sha256 {
                    bail!("TabArena cached receipt hash mismatch");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
                cache.insert(
                    outcome.member.semantic_sha256.clone(),
                    CachedEvaluation {
                        source: CacheSource {
                            generation: outcome.generation,
                            slot: outcome.slot,
                        },
                        receipt_file: outcome.receipt_file.clone(),
                        receipt_sha256: outcome.receipt_sha256.clone(),
                        receipt,
                    },
                );
            }
        }
    }
    Ok(cache)
}

pub(super) fn make_checkpoint(
    identity: &CampaignIdentity,
    generation: usize,
    previous: Option<String>,
    slots: Vec<SlotOutcome>,
    fresh: usize,
    cumulative_fresh: usize,
) -> Result<TabArenaCheckpoint> {
    let island_valid_candidates = (0..ISLANDS)
        .map(|island| {
            slots[island * POPULATION..(island + 1) * POPULATION]
                .iter()
                .filter(|slot| slot.valid)
                .count()
        })
        .collect::<Vec<_>>();
    let island_best_rmse = (0..ISLANDS)
        .map(|island| {
            slots[island * POPULATION..(island + 1) * POPULATION]
                .iter()
                .map(SlotOutcome::rmse)
                .min_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY)
        })
        .collect::<Vec<_>>();
    if island_valid_candidates.iter().sum::<usize>() < MIN_VALID_GLOBAL {
        bail!("TabArena generation does not meet the global validity threshold");
    }
    let mut ranked = slots.clone();
    ranked.sort_by(rank_outcomes);
    let discovered = discovery_winner(&ranked, generation, identity.leader_rmse);
    let best = discovered.unwrap_or(&ranked[0]);
    let next_population = next_population(
        &identity.run_id,
        &identity.dataset_hash,
        generation,
        identity.features,
        &slots,
    )?;
    Ok(TabArenaCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: identity.run_id.clone(),
        dataset: identity.dataset_hash.clone(),
        generation,
        previous_checkpoint_sha256: previous,
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations: fresh,
        cumulative_fresh_evaluations: cumulative_fresh,
        valid_candidates: island_valid_candidates.iter().sum(),
        island_valid_candidates,
        island_best_rmse,
        best_slot: best.slot,
        best_rmse: best.rmse(),
        leader_rmse: identity.leader_rmse,
        leader_beaten: discovered.is_some(),
        migration_complete: generation % 2 == 1,
        slots,
        next_population,
    })
}

pub(super) fn discovery_winner(
    ranked: &[SlotOutcome],
    generation: usize,
    leader_rmse: f64,
) -> Option<&SlotOutcome> {
    (generation > 0)
        .then(|| {
            ranked.iter().find(|slot| {
                slot.valid
                    && slot.rmse() < leader_rmse
                    && slot.cache_source.is_none()
                    && is_fresh_novel_descendant(&slot.member)
            })
        })
        .flatten()
}

pub(super) fn replay_records(
    campaign: &Path,
    registry: &Path,
    identity: &CampaignIdentity,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<ReplayRecord>> {
    let evaluations = scq_batch::replay_across_nodes(
        campaign,
        identity,
        registry,
        &identity.binary_sha256,
        scq_binary_sha256,
        generation,
        champion,
        test,
    )?;
    let baseline = evaluations[0].receipt.predictions.clone();
    let baseline_rmse = evaluations[0]
        .receipt
        .metrics
        .as_ref()
        .context("TabArena replay failed")?
        .rmse;
    evaluations
        .into_iter()
        .map(|evaluation| {
            let rmse = evaluation
                .receipt
                .metrics
                .as_ref()
                .context("TabArena replay failed")?
                .rmse;
            let maximum_prediction_difference = baseline
                .iter()
                .zip(&evaluation.receipt.predictions)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            if (rmse - baseline_rmse).abs() > 1e-12 || maximum_prediction_difference > 1e-10 {
                bail!("TabArena cross-node replay disagreement");
            }
            Ok(ReplayRecord {
                receipt_file: evaluation.receipt_file,
                receipt_sha256: evaluation.receipt_sha256,
                rmse,
                maximum_prediction_difference,
            })
        })
        .collect()
}
