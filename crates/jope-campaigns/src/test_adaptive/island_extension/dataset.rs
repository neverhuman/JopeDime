use super::*;

// Keep dataset identity, provenance and final generation state explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    identity: &IslandCampaignIdentity,
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
    started_at_unix_ms: u128,
) -> Result<IslandDatasetResult> {
    if checkpoints.len() != GENERATIONS {
        bail!("cannot finalize an incomplete island dataset");
    }
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let replay = replay_island_champion(
        &worker_identity(identity),
        registry,
        &island_root(&manifest.base_run_id),
        &manifest.scq_binary_sha256,
        END_GENERATION - 1,
        &selected.member,
        test,
    )?;
    let qualifying_descendants = collect_qualifying_records(
        root,
        combined,
        checkpoints,
        &identity.dataset,
        &identity.release_binary_sha256,
    )?;
    let first = qualifying_descendants
        .iter()
        .min_by_key(|record| (record.generation, record.island, record.global_slot));
    let result = IslandDatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: identity.run_id.clone(),
        dataset_hash: identity.dataset.dataset_hash.clone(),
        outcome: if first.is_some() {
            "new_leader_found".to_string()
        } else {
            "no_win_after_30_total_generations".to_string()
        },
        start_generation: START_GENERATION,
        stopping_generation: END_GENERATION - 1,
        generations_evaluated: GENERATIONS,
        total_generations_evaluated: END_GENERATION,
        leader_models: identity.dataset.leader_models.clone(),
        leader_rmse: identity.dataset.leader_rmse,
        fresh_g0_rmse: combined[0].0.g0_best_rmse,
        first_winning_generation: first.map(|record| record.generation),
        first_winning_island: first.map(|record| record.island),
        generation_29_selected_genome: island_genome_record(
            root,
            combined,
            checkpoints,
            selected,
            combined[0].0.g0_best_rmse,
            identity.dataset.leader_rmse,
            &identity.release_binary_sha256,
        )?,
        qualifying_descendants,
        logical_evaluations: GENERATIONS * TOTAL_POPULATION,
        fresh_evaluations: terminal.cumulative_fresh_evaluations,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        replay,
        release_binary_sha256: identity.release_binary_sha256.clone(),
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

pub(super) fn generation_progress(
    dataset_hash: &str,
    checkpoint: &IslandGenerationCheckpoint,
    completed_new_leaders: usize,
    already_won: bool,
) {
    let island_text = checkpoint
        .islands
        .iter()
        .map(|island| {
            format!(
                "i{} valid {}/{} best {:.17}",
                island.island, island.valid_candidates, ISLAND_POPULATION, island.best_rmse
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let migration = checkpoint
        .migration_file
        .as_deref()
        .map_or("none", |_| "ring top-2 complete");
    let cumulative = EXISTING_SURPASSED
        + completed_new_leaders
        + usize::from(already_won || checkpoint.qualifying_descendant_win);
    let cached = checkpoint.logical_slots - checkpoint.fresh_evaluations;
    let mut node_counts = BTreeMap::new();
    for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
        if outcome.cache_source.is_none() {
            *node_counts.entry(outcome.node.as_str()).or_insert(0usize) += 1;
        }
    }
    let node_text = node_counts
        .into_iter()
        .map(|(node, count)| format!("{node}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let batches = checkpoint.fresh_evaluations.div_ceil(scq_batch::SCQ_CPUS);
    eprintln!(
        "island extension {dataset_hash} generation {}: {island_text}; global best {:.17}; fresh {} cache {cached}; migration {migration}; SCQ batches {batches} nodes [{node_text}]; cumulative surpassed {cumulative}/42",
        checkpoint.generation,
        checkpoint.global_best_rmse,
        checkpoint.fresh_evaluations,
    );
}

pub(super) fn run_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    binding: &IslandDatasetBinding,
    spec: &DatasetSpec,
    resume: bool,
    completed_new_leaders: usize,
    dataset_index: usize,
) -> Result<IslandDatasetResult> {
    let root = island_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    if root.join("final-result.v1.json").is_file() {
        return validate_dataset(registry, manifest, binding, spec);
    }
    let identity = island_identity(manifest, spec);
    let campaign_path = root.join("campaign.v1.json");
    if campaign_path.is_file() {
        if !resume {
            bail!("island dataset already exists; pass --resume");
        }
        let bytes = fs::read(&campaign_path)?;
        let recorded: IslandCampaignIdentity = serde_json::from_slice(&bytes)?;
        if recorded != identity || bytes != canonical_json(&identity)? {
            bail!("island dataset identity changed on resume");
        }
        if fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)? {
            bail!("island dataset prior binding changed on resume");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&campaign_path, &canonical_json(&identity)?)?;
        immutable_write(
            &root.join("prior-binding.v1.json"),
            &canonical_json(binding)?,
        )?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, spec)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_island_checkpoints(&root, &identity, &binding.prior_terminal_checkpoint_sha256)?;
    validate_checkpoint_history(&root, &identity, binding, &combined, &checkpoints, &test)?;
    if checkpoints.len() == GENERATIONS {
        return finalize_dataset(
            registry,
            manifest,
            &identity,
            &root,
            &combined,
            &checkpoints,
            &test,
            started_at_unix_ms,
        );
    }
    let mut cache = load_island_cache(&root, &checkpoints)?;
    let mut population = checkpoints.last().map_or_else(
        || generation_20_populations(&combined, spec.features),
        |(checkpoint, _)| {
            Ok(checkpoint
                .islands
                .iter()
                .map(|island| island.next_population.clone())
                .collect())
        },
    )?;
    let mut previous = checkpoints.last().map_or_else(
        || binding.prior_terminal_checkpoint_sha256.clone(),
        |checkpoint| checkpoint.1.clone(),
    );
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    let mut already_won = checkpoints
        .iter()
        .any(|checkpoint| checkpoint.0.qualifying_descendant_win);
    let worker = worker_identity(&identity);
    let mut completed = checkpoints;
    for generation in (START_GENERATION + completed.len())..END_GENERATION {
        let (outcomes, fresh) = evaluate_island_generation(
            &worker,
            registry,
            &island_root(&manifest.base_run_id),
            &manifest.scq_binary_sha256,
            dataset_index,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        for (island, slots) in outcomes.iter().enumerate() {
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            if valid < ISLAND_MIN_VALID {
                bail!("generation {generation} island {island} has only {valid} valid candidates");
            }
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        if valid < GLOBAL_MIN_VALID {
            bail!("generation {generation} has only {valid} globally valid candidates");
        }
        cumulative_fresh += fresh;
        let (checkpoint, next) = make_checkpoint(
            &identity,
            &root,
            generation,
            previous.clone(),
            outcomes,
            cumulative_fresh,
            combined[0].0.g0_best_rmse,
        )?;
        if checkpoint.fresh_evaluations != fresh {
            bail!("fresh island evaluation accounting mismatch");
        }
        let bytes = canonical_json(&checkpoint)?;
        let sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        generation_progress(
            &spec.dataset_hash,
            &checkpoint,
            completed_new_leaders,
            already_won,
        );
        already_won |= checkpoint.qualifying_descendant_win;
        previous = sha256.clone();
        population = next;
        completed.push((checkpoint, sha256));
    }
    finalize_dataset(
        registry,
        manifest,
        &identity,
        &root,
        &combined,
        &completed,
        &test,
        started_at_unix_ms,
    )
}

pub(super) fn validate_replays(
    root: &Path,
    identity: &IslandCampaignIdentity,
    result: &IslandDatasetResult,
    test: &RegressionData,
) -> Result<()> {
    if result.replay.len() != 3 {
        bail!("island final replay count mismatch");
    }
    let mut receipts = Vec::new();
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!("island replay receipt hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request: CandidateRequest =
            serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
        validate_receipt_against_request(&receipt, &request, test)?;
        if receipt.purpose != "replay-final"
            || receipt.generation != END_GENERATION - 1
            || receipt.slot != slot
            || receipt.run_id != identity.run_id
            || receipt.binary_sha256 != identity.release_binary_sha256
            || receipt.semantic_sha256 != result.generation_29_selected_genome.semantic_sha256
            || receipt.node != scq_batch::NODES[slot]
        {
            bail!("island replay provenance mismatch");
        }
        receipts.push(receipt);
    }
    let base = &receipts[0];
    let base_rmse = base.metrics.as_ref().context("island replay failed")?.rmse;
    for (index, receipt) in receipts.iter().enumerate() {
        let rmse = receipt
            .metrics
            .as_ref()
            .context("island replay failed")?
            .rmse;
        let difference = base
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if (rmse - base_rmse).abs() > 1e-12
            || difference > 1e-10
            || result.replay[index].rmse.to_bits() != rmse.to_bits()
            || result.replay[index].maximum_prediction_difference.to_bits() != difference.to_bits()
        {
            bail!("island replay disagreement");
        }
    }
    Ok(())
}

pub(super) fn validate_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    binding: &IslandDatasetBinding,
    spec: &DatasetSpec,
) -> Result<IslandDatasetResult> {
    let root = island_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    if sha256_file(Path::new(&binding.prior_campaign_file))? != binding.prior_campaign_sha256
        || sha256_file(Path::new(&binding.prior_final_result_file))?
            != binding.prior_final_result_sha256
        || sha256_file(Path::new(&binding.prior_terminal_checkpoint_file))?
            != binding.prior_terminal_checkpoint_sha256
    {
        bail!("dataset {} prior artifacts changed", spec.dataset_hash);
    }
    let identity = island_identity(manifest, spec);
    let campaign_bytes = fs::read(root.join("campaign.v1.json"))?;
    let recorded_identity: IslandCampaignIdentity = serde_json::from_slice(&campaign_bytes)?;
    if recorded_identity != identity
        || campaign_bytes != canonical_json(&identity)?
        || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
    {
        bail!("dataset {} island identity mismatch", spec.dataset_hash);
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_island_checkpoints(&root, &identity, &binding.prior_terminal_checkpoint_sha256)?;
    if checkpoints.len() != GENERATIONS {
        bail!(
            "dataset {} has incomplete island checkpoints",
            spec.dataset_hash
        );
    }
    let cumulative_fresh =
        validate_checkpoint_history(&root, &identity, binding, &combined, &checkpoints, &test)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let expected_selected = island_genome_record(
        &root,
        &combined,
        &checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let qualifying = collect_qualifying_records(
        &root,
        &combined,
        &checkpoints,
        spec,
        &identity.release_binary_sha256,
    )?;
    let first = qualifying
        .iter()
        .min_by_key(|record| (record.generation, record.island, record.global_slot));
    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: IslandDatasetResult = serde_json::from_slice(&result_bytes)?;
    let expected_outcome = if first.is_some() {
        "new_leader_found"
    } else {
        "no_win_after_30_total_generations"
    };
    if result_bytes != canonical_json(&result)?
        || result.schema != RESULT_SCHEMA
        || result.benchmark_valid
        || result.base_run_id != manifest.base_run_id
        || result.island_extension_run_id != manifest.island_extension_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != expected_outcome
        || result.start_generation != START_GENERATION
        || result.stopping_generation != END_GENERATION - 1
        || result.generations_evaluated != GENERATIONS
        || result.total_generations_evaluated != END_GENERATION
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.fresh_g0_rmse.to_bits() != combined[0].0.g0_best_rmse.to_bits()
        || result.first_winning_generation != first.map(|record| record.generation)
        || result.first_winning_island != first.map(|record| record.island)
        || result.generation_29_selected_genome != expected_selected
        || result.qualifying_descendants != qualifying
        || result.logical_evaluations != GENERATIONS * TOTAL_POPULATION
        || result.fresh_evaluations != cumulative_fresh
        || result.release_binary_sha256 != manifest.release_binary_sha256
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!("dataset {} island final result mismatch", spec.dataset_hash);
    }
    validate_replays(&root, &identity, &result, &test)?;
    Ok(result)
}
