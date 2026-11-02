use super::*;

pub(super) fn run_dataset(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
) -> Result<DatasetResult> {
    let (_, test) = read_campaign_data(Path::new(&manifest.registry_root), spec)?;
    let checkpoints = load_checkpoints(&manifest.run_id, spec)?;
    for (checkpoint, _) in &checkpoints {
        validate_checkpoint_receipts(manifest, spec, checkpoint, &test)?;
    }
    if let Some((terminal, digest)) = checkpoints.last() {
        if terminal.winner_slot.is_some() {
            let (_, confirmation_sha256) = confirm_winner(manifest, spec, terminal, digest, &test)?;
            return finish_dataset(manifest, spec, terminal, digest, Some(confirmation_sha256));
        }
    }

    let mut population = checkpoints
        .last()
        .map(|(checkpoint, _)| checkpoint.next_population.clone())
        .unwrap_or(initial_population(
            Path::new(&manifest.registry_root),
            &manifest.run_id,
            spec,
        )?);
    let mut previous = checkpoints.last().map(|(_, digest)| digest.clone());
    let mut generation = checkpoints.len();
    loop {
        if population.len() != TOTAL_POPULATION {
            bail!("aggressive generation population is incomplete");
        }
        validate_preflight_receipt(&preflight()?)?;
        let identity = worker_identity(manifest, spec);
        let members = population.into_iter().enumerate().collect::<Vec<_>>();
        let evaluations = scq_batch::evaluate_aggressive_batches(
            &root(&manifest.run_id),
            &identity,
            Path::new(&manifest.registry_root),
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            generation,
            &members,
            &test,
        )?;
        let mut slots = members
            .into_iter()
            .zip(evaluations)
            .map(|((slot, member), evaluation)| {
                outcome_from_evaluation(generation, slot, member, &evaluation, false)
            })
            .collect::<Vec<_>>();
        slots.sort_by_key(|outcome| outcome.slot);
        if slots
            .iter()
            .enumerate()
            .any(|(slot, outcome)| outcome.slot != slot)
        {
            bail!("aggressive completion order affected logical slots");
        }
        let (ranked_islands, valid_per_island) = rank_islands_with_global_recovery(&slots)?;
        let mut global_ranked = slots.clone();
        global_ranked.sort_by(rank_outcomes);
        let winner_slot = evolutionary_winner_slot(&global_ranked, generation, spec.leader_rmse);
        let next_population = if winner_slot.is_none() {
            ranked_islands
                .iter()
                .enumerate()
                .map(|(island, ranked)| {
                    next_island_population(
                        ranked,
                        &manifest.run_id,
                        &spec.dataset_hash,
                        generation + 1,
                        island,
                        spec.features,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            Vec::new()
        };
        let checkpoint = AggressiveCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            test_adaptive: true,
            run_id: manifest.run_id.clone(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256: previous.clone(),
            logical_slots: TOTAL_POPULATION,
            fresh_evaluations: 0,
            cache_hits: 0,
            cumulative_fresh_evaluations: 0,
            cumulative_cache_hits: 0,
            unique_population: 0,
            valid_per_island,
            ranked_slots: global_ranked.iter().map(|outcome| outcome.slot).collect(),
            best_slot: global_ranked[0].slot,
            best_rmse: global_ranked[0].rmse(),
            winner_slot,
            slots,
            next_population: next_population.clone(),
        };
        let bytes = canonical_json(&checkpoint)?;
        let digest = sha256_bytes(&bytes);
        atomic_immutable_write(
            &checkpoint_path(&manifest.run_id, &spec.dataset_hash, generation),
            &bytes,
        )?;
        eprintln!(
            "aggressive {} generation {generation}: best RMSE {:.17}, winner {:?}",
            spec.dataset_hash, checkpoint.best_rmse, checkpoint.winner_slot
        );
        if checkpoint.winner_slot.is_some() {
            let (_, confirmation_sha256) =
                confirm_winner(manifest, spec, &checkpoint, &digest, &test)?;
            return finish_dataset(
                manifest,
                spec,
                &checkpoint,
                &digest,
                Some(confirmation_sha256),
            );
        }
        previous = Some(digest);
        population = next_population;
        generation = generation
            .checked_add(1)
            .context("aggressive generation counter overflow")?;
    }
}

pub(super) struct DatasetLane {
    pub(super) dataset_index: usize,
    pub(super) spec: DatasetSpec,
    pub(super) test: RegressionData,
    pub(super) population: Vec<Member>,
    pub(super) cache: HashMap<String, CachedEvaluation>,
    pub(super) history: HashSet<String>,
    pub(super) cumulative_fresh_evaluations: usize,
    pub(super) cumulative_cache_hits: usize,
    pub(super) previous: Option<String>,
    pub(super) generation: usize,
}

pub(super) enum PreparedLane {
    Finished(usize, DatasetResult),
    Active(DatasetLane),
}

pub(super) fn prepare_lane(
    manifest: &AggressiveManifest,
    dataset_index: usize,
    spec: &DatasetSpec,
) -> Result<PreparedLane> {
    let (_, test) = read_campaign_data(Path::new(&manifest.registry_root), spec)?;
    let checkpoints = load_checkpoints(&manifest.run_id, spec)?;
    for (checkpoint, _) in &checkpoints {
        validate_checkpoint_receipts(manifest, spec, checkpoint, &test)?;
    }
    if let Some((terminal, digest)) = checkpoints.last() {
        if terminal.winner_slot.is_some() {
            let (_, confirmation_sha256) = confirm_winner(manifest, spec, terminal, digest, &test)?;
            return Ok(PreparedLane::Finished(
                dataset_index,
                finish_dataset(manifest, spec, terminal, digest, Some(confirmation_sha256))?,
            ));
        }
    }
    let mut cache = HashMap::<String, CachedEvaluation>::new();
    let mut history = HashSet::<String>::new();
    for (checkpoint, _) in &checkpoints {
        for outcome in &checkpoint.slots {
            let receipt_path =
                dataset_root(&manifest.run_id, &spec.dataset_hash).join(&outcome.receipt_file);
            let receipt_bytes = fs::read(&receipt_path)?;
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let source = outcome.cache_source.clone().unwrap_or(CacheSource {
                generation: checkpoint.generation,
                slot: outcome.slot,
            });
            cache
                .entry(outcome.member.semantic_sha256.clone())
                .or_insert(CachedEvaluation {
                    source,
                    receipt_file: outcome.receipt_file.clone(),
                    receipt_sha256: outcome.receipt_sha256.clone(),
                    receipt,
                });
            history.insert(outcome.member.semantic_sha256.clone());
        }
    }
    let cumulative_fresh_evaluations = checkpoints
        .last()
        .map_or(0, |(checkpoint, _)| checkpoint.cumulative_fresh_evaluations);
    let cumulative_cache_hits = checkpoints
        .last()
        .map_or(0, |(checkpoint, _)| checkpoint.cumulative_cache_hits);
    Ok(PreparedLane::Active(DatasetLane {
        dataset_index,
        spec: spec.clone(),
        test,
        population: checkpoints
            .last()
            .map(|(checkpoint, _)| checkpoint.next_population.clone())
            .unwrap_or(initial_population(
                Path::new(&manifest.registry_root),
                &manifest.run_id,
                spec,
            )?),
        cache,
        history,
        cumulative_fresh_evaluations,
        cumulative_cache_hits,
        previous: checkpoints.last().map(|(_, digest)| digest.clone()),
        generation: checkpoints.len(),
    }))
}

pub(super) enum CompletedLaneGeneration {
    Finished(usize, DatasetResult),
    Active(DatasetLane),
}

pub(super) fn complete_lane_generation(
    manifest: &AggressiveManifest,
    mut lane: DatasetLane,
    members: Vec<(usize, Member)>,
    evaluations: Vec<CachedEvaluation>,
) -> Result<CompletedLaneGeneration> {
    let generation = lane.generation;
    let mut fresh_by_slot = evaluations
        .into_iter()
        .map(|evaluation| (evaluation.source.slot, evaluation))
        .collect::<HashMap<_, _>>();
    let mut slots = Vec::with_capacity(TOTAL_POPULATION);
    for (slot, member) in members {
        if let Some(evaluation) = fresh_by_slot.remove(&slot) {
            if lane.history.contains(&member.semantic_sha256)
                || evaluation.receipt.semantic_sha256 != member.semantic_sha256
            {
                bail!("Golden46 fresh evaluation repeated a historical genome");
            }
            lane.history.insert(member.semantic_sha256.clone());
            lane.cache
                .insert(member.semantic_sha256.clone(), evaluation.clone());
            slots.push(outcome_from_evaluation(
                generation,
                slot,
                member,
                &evaluation,
                false,
            ));
        } else {
            let evaluation = lane
                .cache
                .get(&member.semantic_sha256)
                .context("Golden46 logical elite is absent from the evaluation cache")?;
            slots.push(outcome_from_evaluation(
                generation, slot, member, evaluation, true,
            ));
        }
    }
    if !fresh_by_slot.is_empty() {
        bail!("Golden46 worker returned an evaluation for an unknown logical slot");
    }
    slots.sort_by_key(|outcome| outcome.slot);
    if slots
        .iter()
        .enumerate()
        .any(|(slot, outcome)| outcome.slot != slot)
    {
        bail!("aggressive completion order affected logical slots");
    }
    let (ranked_islands, valid_per_island) = rank_islands_with_global_recovery(&slots)?;
    let mut global_ranked = slots.clone();
    global_ranked.sort_by(rank_outcomes);
    let winner_slot = evolutionary_winner_slot(&global_ranked, generation, lane.spec.leader_rmse);
    let next_population = if winner_slot.is_none() {
        next_golden46_population(
            &ranked_islands,
            &manifest.run_id,
            &lane.spec.dataset_hash,
            generation + 1,
            lane.spec.features,
            &lane.history,
        )?
    } else {
        Vec::new()
    };
    let unique_population = slots
        .iter()
        .map(|outcome| outcome.member.semantic_sha256.as_str())
        .collect::<HashSet<_>>()
        .len();
    let cache_hits = slots
        .iter()
        .filter(|outcome| outcome.cache_source.is_some())
        .count();
    let fresh_evaluations = TOTAL_POPULATION - cache_hits;
    if unique_population != TOTAL_POPULATION
        || fresh_evaluations + cache_hits != TOTAL_POPULATION
        || (generation == 0 && cache_hits != 0)
    {
        bail!("Golden46 completed generation is not unique/cache-accounted");
    }
    lane.cumulative_fresh_evaluations += fresh_evaluations;
    lane.cumulative_cache_hits += cache_hits;
    let checkpoint = AggressiveCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: lane.spec.dataset_hash.clone(),
        generation,
        previous_checkpoint_sha256: lane.previous.clone(),
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations,
        cache_hits,
        cumulative_fresh_evaluations: lane.cumulative_fresh_evaluations,
        cumulative_cache_hits: lane.cumulative_cache_hits,
        unique_population,
        valid_per_island,
        ranked_slots: global_ranked.iter().map(|outcome| outcome.slot).collect(),
        best_slot: global_ranked[0].slot,
        best_rmse: global_ranked[0].rmse(),
        winner_slot,
        slots,
        next_population: next_population.clone(),
    };
    let bytes = canonical_json(&checkpoint)?;
    let digest = sha256_bytes(&bytes);
    atomic_immutable_write(
        &checkpoint_path(&manifest.run_id, &lane.spec.dataset_hash, generation),
        &bytes,
    )?;
    eprintln!(
        "aggressive {} generation {generation}: best RMSE {:.17}, winner {:?}",
        lane.spec.dataset_hash, checkpoint.best_rmse, checkpoint.winner_slot
    );
    if checkpoint.winner_slot.is_some() {
        let (_, confirmation_sha256) =
            confirm_winner(manifest, &lane.spec, &checkpoint, &digest, &lane.test)?;
        return Ok(CompletedLaneGeneration::Finished(
            lane.dataset_index,
            finish_dataset(
                manifest,
                &lane.spec,
                &checkpoint,
                &digest,
                Some(confirmation_sha256),
            )?,
        ));
    }
    lane.previous = Some(digest);
    lane.population = next_population;
    lane.generation = generation
        .checked_add(1)
        .context("aggressive generation counter overflow")?;
    Ok(CompletedLaneGeneration::Active(lane))
}

pub(super) fn run_dataset_waves(manifest: &AggressiveManifest) -> Result<Vec<DatasetResult>> {
    let mut results = vec![None; manifest.datasets.len()];
    let mut next_dataset = 0usize;
    let mut active = Vec::<DatasetLane>::new();
    while next_dataset < manifest.datasets.len() || !active.is_empty() {
        while active.len() < scq_batch::AGGRESSIVE_DATASET_LANES
            && next_dataset < manifest.datasets.len()
        {
            let spec = &manifest.datasets[next_dataset];
            eprintln!(
                "aggressive dataset {}/{} {}",
                next_dataset + 1,
                manifest.datasets.len(),
                spec.dataset_hash
            );
            match prepare_lane(manifest, next_dataset, spec)? {
                PreparedLane::Finished(index, result) => results[index] = Some(result),
                PreparedLane::Active(lane) => active.push(lane),
            }
            next_dataset += 1;
        }
        if active.is_empty() {
            continue;
        }
        // The immutable launch/resume preflight above establishes the resource
        // floor. Per-node SCQ leases and the post-lease occupancy check in
        // `run_batch_flavored` serialize later waves with other campaign
        // controllers. Re-running the global "queue must be empty" preflight
        // here makes two otherwise compatible campaigns terminate whenever
        // their generation boundaries do not align.
        let identities = active
            .iter()
            .map(|lane| worker_identity(manifest, &lane.spec))
            .collect::<Vec<_>>();
        let member_sets = active
            .iter_mut()
            .map(|lane| {
                std::mem::take(&mut lane.population)
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if member_sets
            .iter()
            .any(|members| members.len() != TOTAL_POPULATION)
        {
            bail!("aggressive generation population is incomplete");
        }
        let fresh_member_sets = member_sets
            .iter()
            .zip(&active)
            .map(|(members, lane)| {
                members
                    .iter()
                    .filter(|(_, member)| !lane.cache.contains_key(&member.semantic_sha256))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if fresh_member_sets.iter().any(Vec::is_empty) {
            bail!("Golden46 generation contains no fresh candidates");
        }
        let lane_requests = (0..active.len())
            .map(|index| {
                (
                    &identities[index],
                    active[index].generation,
                    fresh_member_sets[index].as_slice(),
                    &active[index].test,
                )
            })
            .collect::<Vec<_>>();
        let evaluation_sets = scq_batch::evaluate_aggressive_wave(
            &root(&manifest.run_id),
            Path::new(&manifest.registry_root),
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            &lane_requests,
        )?;
        let previous_active = std::mem::take(&mut active);
        for ((lane, members), evaluations) in previous_active
            .into_iter()
            .zip(member_sets)
            .zip(evaluation_sets)
        {
            match complete_lane_generation(manifest, lane, members, evaluations)? {
                CompletedLaneGeneration::Finished(index, result) => results[index] = Some(result),
                CompletedLaneGeneration::Active(lane) => active.push(lane),
            }
        }
    }
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("aggressive dataset {index} produced no result"))
        })
        .collect()
}
