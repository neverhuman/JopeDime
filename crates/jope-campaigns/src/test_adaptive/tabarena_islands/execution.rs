use super::*;

#[cfg(any())]
fn run_dataset(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
    dataset_index: usize,
    resume: bool,
) -> Result<TabArenaResult> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let result_path = root.join("final-result.v1.json");
    if result_path.is_file() {
        return serde_json::from_slice(&fs::read(result_path)?).context("read TabArena result");
    }
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    let identity_path = root.join("campaign.v1.json");
    if identity_path.is_file() {
        if !resume || fs::read(&identity_path)? != canonical_json(&identity)? {
            bail!("TabArena dataset exists with a different identity; pass --resume only for an exact match");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&identity_path, &canonical_json(&identity)?)?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let mut checkpoints = load_checkpoints(&root, &identity)?;
    if checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.leader_beaten)
    {
        bail!("TabArena leader-beating checkpoint is missing its final result");
    }
    let mut cache = load_cache(&root, &checkpoints)?;
    let mut population = checkpoints
        .last()
        .map(|checkpoint| checkpoint.0.next_population.clone())
        .map_or_else(
            || initial_population(registry, &root, &encoded.spec, &manifest.run_id),
            Ok,
        )?;
    let mut previous = checkpoints.last().map(|checkpoint| checkpoint.1.clone());
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    for generation in checkpoints.len().. {
        let (slots, fresh) = evaluate_population(
            campaign,
            registry,
            &identity,
            &manifest.scq_binary_sha256,
            dataset_index,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        cumulative_fresh += fresh;
        let checkpoint = make_checkpoint(
            &identity,
            generation,
            previous.clone(),
            slots,
            fresh,
            cumulative_fresh,
        )?;
        let bytes = canonical_json(&checkpoint)?;
        let hash = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        eprintln!(
            "TabArena {} generation {}: islands valid {:?} best {:?}; global {:.17} leader {:.17}; fresh {} cache {}; migration {}; beaten {}",
            encoded.spec.dataset_hash,
            generation,
            checkpoint.island_valid_candidates,
            checkpoint.island_best_rmse,
            checkpoint.best_rmse,
            checkpoint.leader_rmse,
            checkpoint.fresh_evaluations,
            TOTAL_POPULATION - checkpoint.fresh_evaluations,
            checkpoint.migration_complete,
            checkpoint.leader_beaten,
        );
        previous = Some(hash.clone());
        population = checkpoint.next_population.clone();
        checkpoints.push((checkpoint, hash));
        if checkpoints.last().unwrap().0.leader_beaten {
            break;
        }
    }
    let checkpoint = &checkpoints.last().unwrap().0;
    let selected = &checkpoint.slots[checkpoint.best_slot];
    let receipt_bytes = fs::read(root.join(&selected.receipt_file))?;
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let replay = replay_records(
        campaign,
        registry,
        &identity,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &selected.member,
        &test,
    )?;
    let result = TabArenaResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset: encoded.spec.dataset_hash.clone(),
        stopping_generation: checkpoint.generation,
        generations_evaluated: checkpoint.generation + 1,
        leader_models: encoded.spec.leader_models.clone(),
        leader_rmse: encoded.spec.leader_rmse,
        selected_rmse: selected.rmse(),
        margin_vs_leader: encoded.spec.leader_rmse - selected.rmse(),
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        selected_slot: selected.slot,
        prediction_sha256: receipt.predictions_sha256,
        receipt_sha256: selected.receipt_sha256.clone(),
        binary_sha256: manifest.release_binary_sha256.clone(),
        logical_evaluations: checkpoints.len() * TOTAL_POPULATION,
        fresh_evaluations: checkpoint.cumulative_fresh_evaluations,
        replay,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(&result_path, &canonical_json(&result)?)?;
    Ok(result)
}

struct TabArenaLane {
    dataset_index: usize,
    encoded: EncodedDataset,
    root: PathBuf,
    identity: CampaignIdentity,
    test: RegressionData,
    population: Vec<Member>,
    previous: Option<String>,
    cumulative_fresh: usize,
    generation: usize,
    cache: HashMap<String, CachedEvaluation>,
    checkpoints: Vec<(TabArenaCheckpoint, String)>,
    started_at_unix_ms: u128,
}

// Preserve the validated local/remote artifact payload layout without changing execution.
#[allow(clippy::large_enum_variant)]
enum PreparedTabArenaLane {
    Finished(usize, TabArenaResult),
    Active(TabArenaLane),
}

fn prepare_tabarena_lane(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
    dataset_index: usize,
    resume: bool,
) -> Result<PreparedTabArenaLane> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let result_path = root.join("final-result.v1.json");
    if result_path.is_file() {
        return Ok(PreparedTabArenaLane::Finished(
            dataset_index,
            serde_json::from_slice(&fs::read(result_path)?)?,
        ));
    }
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    let identity_path = root.join("campaign.v1.json");
    if identity_path.is_file() {
        if !resume || fs::read(&identity_path)? != canonical_json(&identity)? {
            bail!("TabArena dataset exists with a different identity; pass --resume only for an exact match");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&identity_path, &canonical_json(&identity)?)?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let checkpoints = load_checkpoints(&root, &identity)?;
    let cache = load_cache(&root, &checkpoints)?;
    let population = checkpoints
        .last()
        .map(|checkpoint| checkpoint.0.next_population.clone())
        .map_or_else(
            || initial_population(registry, &root, &encoded.spec, &manifest.run_id),
            Ok,
        )?;
    let lane = TabArenaLane {
        dataset_index,
        encoded: encoded.clone(),
        root,
        identity,
        test,
        population,
        previous: checkpoints.last().map(|checkpoint| checkpoint.1.clone()),
        cumulative_fresh: checkpoints
            .last()
            .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations),
        generation: checkpoints.len(),
        cache,
        checkpoints,
        started_at_unix_ms,
    };
    if lane
        .checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.leader_beaten)
    {
        if !resume {
            bail!("TabArena leader-beating checkpoint is missing its final result");
        }
        return Ok(PreparedTabArenaLane::Finished(
            dataset_index,
            finish_tabarena_lane(campaign, registry, manifest, &lane)?,
        ));
    }
    Ok(PreparedTabArenaLane::Active(lane))
}

struct PendingTabArenaGeneration {
    lane: TabArenaLane,
    outcomes: Vec<Option<SlotOutcome>>,
    groups: Vec<Vec<(usize, Member)>>,
    representatives: Vec<(usize, Member)>,
    fresh: usize,
}

fn prepare_tabarena_generation(mut lane: TabArenaLane) -> Result<PendingTabArenaGeneration> {
    if lane.population.len() != TOTAL_POPULATION {
        bail!("TabArena generation population is incomplete");
    }
    let mut outcomes = vec![None; TOTAL_POPULATION];
    let mut pending = BTreeMap::<String, Vec<(usize, Member)>>::new();
    for (slot, member) in std::mem::take(&mut lane.population).into_iter().enumerate() {
        if let Some(evaluation) = lane.cache.get(&member.semantic_sha256) {
            outcomes[slot] = Some(outcome_from_evaluation(
                lane.generation,
                slot,
                member,
                evaluation,
                true,
            ));
        } else {
            pending
                .entry(member.semantic_sha256.clone())
                .or_default()
                .push((slot, member));
        }
    }
    let groups = pending.into_values().collect::<Vec<_>>();
    let representatives = groups
        .iter()
        .map(|group| group[0].clone())
        .collect::<Vec<_>>();
    Ok(PendingTabArenaGeneration {
        lane,
        outcomes,
        fresh: groups.len(),
        groups,
        representatives,
    })
}

fn finish_tabarena_lane(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    lane: &TabArenaLane,
) -> Result<TabArenaResult> {
    let checkpoint = &lane
        .checkpoints
        .last()
        .context("TabArena lane has no terminal checkpoint")?
        .0;
    let selected = &checkpoint.slots[checkpoint.best_slot];
    let receipt_bytes = fs::read(lane.root.join(&selected.receipt_file))?;
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let replay = replay_records(
        campaign,
        registry,
        &lane.identity,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &selected.member,
        &lane.test,
    )?;
    let result = TabArenaResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset: lane.encoded.spec.dataset_hash.clone(),
        stopping_generation: checkpoint.generation,
        generations_evaluated: checkpoint.generation + 1,
        leader_models: lane.encoded.spec.leader_models.clone(),
        leader_rmse: lane.encoded.spec.leader_rmse,
        selected_rmse: selected.rmse(),
        margin_vs_leader: lane.encoded.spec.leader_rmse - selected.rmse(),
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        selected_slot: selected.slot,
        prediction_sha256: receipt.predictions_sha256,
        receipt_sha256: selected.receipt_sha256.clone(),
        binary_sha256: manifest.release_binary_sha256.clone(),
        logical_evaluations: lane.checkpoints.len() * TOTAL_POPULATION,
        fresh_evaluations: checkpoint.cumulative_fresh_evaluations,
        replay,
        started_at_unix_ms: lane.started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &lane.root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

// Preserve the validated local/remote artifact payload layout without changing execution.
#[allow(clippy::large_enum_variant)]
enum CompletedTabArenaGeneration {
    Finished(usize, TabArenaResult),
    Active(TabArenaLane),
}

fn complete_tabarena_generation(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    mut pending: PendingTabArenaGeneration,
    evaluations: Vec<CachedEvaluation>,
) -> Result<CompletedTabArenaGeneration> {
    let generation = pending.lane.generation;
    let by_slot = evaluations
        .into_iter()
        .map(|evaluation| (evaluation.source.slot, evaluation))
        .collect::<HashMap<_, _>>();
    for group in pending.groups {
        let evaluation = by_slot
            .get(&group[0].0)
            .context("TabArena fresh representative has no evaluation")?;
        pending.lane.cache.insert(
            evaluation.receipt.semantic_sha256.clone(),
            evaluation.clone(),
        );
        for (index, (slot, member)) in group.into_iter().enumerate() {
            pending.outcomes[slot] = Some(outcome_from_evaluation(
                generation,
                slot,
                member,
                evaluation,
                index > 0,
            ));
        }
    }
    let slots = pending
        .outcomes
        .into_iter()
        .map(|outcome| outcome.context("missing TabArena slot outcome"))
        .collect::<Result<Vec<_>>>()?;
    pending.lane.cumulative_fresh += pending.fresh;
    let checkpoint = make_checkpoint(
        &pending.lane.identity,
        generation,
        pending.lane.previous.clone(),
        slots,
        pending.fresh,
        pending.lane.cumulative_fresh,
    )?;
    let bytes = canonical_json(&checkpoint)?;
    let hash = sha256_bytes(&bytes);
    immutable_write(&checkpoint_path(&pending.lane.root, generation), &bytes)?;
    eprintln!(
        "TabArena {} generation {}: islands valid {:?} best {:?}; global {:.17} leader {:.17}; fresh {} cache {}; migration {}; beaten {}",
        pending.lane.encoded.spec.dataset_hash,
        generation,
        checkpoint.island_valid_candidates,
        checkpoint.island_best_rmse,
        checkpoint.best_rmse,
        checkpoint.leader_rmse,
        checkpoint.fresh_evaluations,
        TOTAL_POPULATION - checkpoint.fresh_evaluations,
        checkpoint.migration_complete,
        checkpoint.leader_beaten,
    );
    pending.lane.previous = Some(hash.clone());
    pending.lane.population = checkpoint.next_population.clone();
    pending.lane.checkpoints.push((checkpoint, hash));
    if pending.lane.checkpoints.last().unwrap().0.leader_beaten {
        let result = finish_tabarena_lane(campaign, registry, manifest, &pending.lane)?;
        return Ok(CompletedTabArenaGeneration::Finished(
            pending.lane.dataset_index,
            result,
        ));
    }
    pending.lane.generation = generation
        .checked_add(1)
        .context("TabArena generation counter overflow")?;
    Ok(CompletedTabArenaGeneration::Active(pending.lane))
}

pub(crate) fn run_tabarena_dataset_waves(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    datasets: &[EncodedDataset],
    resume: bool,
) -> Result<Vec<TabArenaResult>> {
    let mut results = vec![None; datasets.len()];
    let mut next_dataset = 0usize;
    let mut active = Vec::<TabArenaLane>::new();
    while next_dataset < datasets.len() || !active.is_empty() {
        while active.len() < DATASET_LANES && next_dataset < datasets.len() {
            let encoded = &datasets[next_dataset];
            eprintln!(
                "TabArena dataset {}/{} {}",
                next_dataset + 1,
                datasets.len(),
                encoded.spec.dataset_hash
            );
            match prepare_tabarena_lane(
                campaign,
                registry,
                manifest,
                encoded,
                next_dataset,
                resume,
            )? {
                PreparedTabArenaLane::Finished(index, result) => results[index] = Some(result),
                PreparedTabArenaLane::Active(lane) => active.push(lane),
            }
            next_dataset += 1;
        }
        if active.is_empty() {
            continue;
        }
        let pending = std::mem::take(&mut active)
            .into_iter()
            .map(prepare_tabarena_generation)
            .collect::<Result<Vec<_>>>()?;
        let (cached_only, pending): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|pending| pending.representatives.is_empty());
        for pending in cached_only {
            match complete_tabarena_generation(campaign, registry, manifest, pending, Vec::new())? {
                CompletedTabArenaGeneration::Finished(index, result) => {
                    results[index] = Some(result)
                }
                CompletedTabArenaGeneration::Active(lane) => active.push(lane),
            }
        }
        if pending.is_empty() {
            continue;
        }
        let lane_requests = pending
            .iter()
            .map(|pending| {
                (
                    &pending.lane.identity,
                    pending.lane.generation,
                    pending.representatives.as_slice(),
                    &pending.lane.test,
                )
            })
            .collect::<Vec<_>>();
        let evaluation_sets = scq_batch::evaluate_aggressive_wave(
            campaign,
            registry,
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            &lane_requests,
        )?;
        for (pending, evaluations) in pending.into_iter().zip(evaluation_sets) {
            match complete_tabarena_generation(campaign, registry, manifest, pending, evaluations)?
            {
                CompletedTabArenaGeneration::Finished(index, result) => {
                    results[index] = Some(result)
                }
                CompletedTabArenaGeneration::Active(lane) => active.push(lane),
            }
        }
    }
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("TabArena dataset {index} produced no result"))
        })
        .collect()
}
