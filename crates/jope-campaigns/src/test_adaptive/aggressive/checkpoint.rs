use super::*;

pub(crate) fn load_checkpoints(
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(AggressiveCheckpoint, String)>> {
    let checkpoint_dir = dataset_root(run_id, &spec.dataset_hash).join("checkpoints");
    let mut generations = if checkpoint_dir.is_dir() {
        fs::read_dir(&checkpoint_dir)?
            .map(|entry| {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let generation = name
                    .strip_prefix("generation-")
                    .and_then(|value| value.strip_suffix(".v1.json"))
                    .context("unexpected aggressive checkpoint filename")?
                    .parse::<usize>()?;
                Ok(generation)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    generations.sort_unstable();
    for (expected, actual) in generations.iter().enumerate() {
        if *actual != expected {
            bail!("aggressive checkpoint chain contains a gap at generation {expected}");
        }
    }
    let mut checkpoints = Vec::<(AggressiveCheckpoint, String)>::new();
    let mut previous = None;
    let mut history = HashSet::<String>::new();
    let mut cumulative_fresh_evaluations = 0usize;
    let mut cumulative_cache_hits = 0usize;
    for generation in generations {
        if checkpoints
            .last()
            .is_some_and(|(checkpoint, _)| checkpoint.winner_slot.is_some())
        {
            bail!("aggressive checkpoint exists after a terminal winner");
        }
        let path = checkpoint_path(run_id, &spec.dataset_hash, generation);
        let bytes = fs::read(&path)?;
        let checkpoint: AggressiveCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || !checkpoint.test_adaptive
            || checkpoint.run_id != run_id
            || checkpoint.dataset_hash != spec.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.logical_slots != TOTAL_POPULATION
            || checkpoint.slots.len() != TOTAL_POPULATION
            || checkpoint
                .slots
                .iter()
                .enumerate()
                .any(|(slot, outcome)| outcome.slot != slot || outcome.generation != generation)
        {
            bail!("invalid aggressive generation barrier {generation}");
        }
        let (ranked_islands, valid_per_island) =
            rank_islands_with_global_recovery(&checkpoint.slots)?;
        let mut global_ranked = checkpoint.slots.clone();
        global_ranked.sort_by(rank_outcomes);
        let winner_slot = evolutionary_winner_slot(&global_ranked, generation, spec.leader_rmse);
        if is_golden46_run(run_id) {
            let unique_population = checkpoint
                .slots
                .iter()
                .map(|outcome| outcome.member.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len();
            let cache_hits = checkpoint
                .slots
                .iter()
                .filter(|outcome| outcome.cache_source.is_some())
                .count();
            let fresh_evaluations = TOTAL_POPULATION - cache_hits;
            if unique_population != TOTAL_POPULATION
                || checkpoint.unique_population != unique_population
                || checkpoint.fresh_evaluations != fresh_evaluations
                || checkpoint.cache_hits != cache_hits
                || (generation == 0 && cache_hits != 0)
                || checkpoint.slots.iter().any(|outcome| {
                    outcome.cache_source.is_none()
                        && history.contains(&outcome.member.semantic_sha256)
                })
                || checkpoint.slots.iter().any(|outcome| {
                    outcome.cache_source.is_some()
                        && !history.contains(&outcome.member.semantic_sha256)
                })
            {
                bail!("Golden46 generation {generation} uniqueness/cache barrier changed");
            }
            cumulative_fresh_evaluations += fresh_evaluations;
            cumulative_cache_hits += cache_hits;
            if checkpoint.cumulative_fresh_evaluations != cumulative_fresh_evaluations
                || checkpoint.cumulative_cache_hits != cumulative_cache_hits
            {
                bail!("Golden46 generation {generation} cumulative cache accounting changed");
            }
        }
        history.extend(
            checkpoint
                .slots
                .iter()
                .map(|outcome| outcome.member.semantic_sha256.clone()),
        );
        let expected_next = if winner_slot.is_some() {
            Vec::new()
        } else if is_golden46_run(run_id) {
            next_golden46_population(
                &ranked_islands,
                run_id,
                &spec.dataset_hash,
                generation + 1,
                spec.features,
                &history,
            )?
        } else {
            ranked_islands
                .iter()
                .enumerate()
                .map(|(island, ranked)| {
                    next_island_population(
                        ranked,
                        run_id,
                        &spec.dataset_hash,
                        generation + 1,
                        island,
                        spec.features,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        if checkpoint.valid_per_island != valid_per_island
            || checkpoint.ranked_slots
                != global_ranked
                    .iter()
                    .map(|outcome| outcome.slot)
                    .collect::<Vec<_>>()
            || checkpoint.best_slot != global_ranked[0].slot
            || checkpoint.best_rmse.to_bits() != global_ranked[0].rmse().to_bits()
            || checkpoint.winner_slot != winner_slot
            || checkpoint.next_population != expected_next
        {
            bail!("aggressive generation {generation} selection barrier changed");
        }
        let digest = sha256_bytes(&bytes);
        previous = Some(digest.clone());
        checkpoints.push((checkpoint, digest));
    }
    Ok(checkpoints)
}

pub(crate) fn validate_checkpoint_receipts(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    test: &RegressionData,
) -> Result<()> {
    for outcome in &checkpoint.slots {
        let receipt_path =
            dataset_root(&manifest.run_id, &spec.dataset_hash).join(&outcome.receipt_file);
        let receipt_bytes = fs::read(&receipt_path)?;
        if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
            bail!("aggressive slot receipt hash changed");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt_bytes != canonical_json(&receipt)? {
            bail!("aggressive slot receipt is not canonical");
        }
        let request_path = dataset_root(&manifest.run_id, &spec.dataset_hash)
            .join("requests")
            .join(&receipt.purpose)
            .join(format!("generation-{:03}", receipt.generation))
            .join(format!(
                "slot-{:03}-attempt-{}.v1.json",
                receipt.slot, receipt.attempt
            ));
        let request_bytes = fs::read(&request_path)?;
        let request: CandidateRequest = serde_json::from_slice(&request_bytes)?;
        if request_bytes != canonical_json(&request)? {
            bail!("aggressive slot request is not canonical");
        }
        validate_receipt_against_request(&receipt, &request, test)?;
        let source = outcome.cache_source.clone().unwrap_or(CacheSource {
            generation: checkpoint.generation,
            slot: outcome.slot,
        });
        let primary = (source.slot + source.generation) % scq_batch::NODES.len();
        let expected_node = match outcome.attempt {
            0 => scq_batch::NODES[primary],
            1 => scq_batch::NODES[(primary + 1) % scq_batch::NODES.len()],
            _ => bail!("aggressive slot used more than one infrastructure retry"),
        };
        if receipt.test_adaptive != Some(true)
            || receipt.purpose != "aggressive-candidate"
            || receipt.run_id != manifest.run_id
            || receipt.binary_sha256 != manifest.release_binary_sha256
            || receipt.dataset_hash != spec.dataset_hash
            || receipt.train_sha256 != spec.train_sha256
            || receipt.test_sha256 != spec.test_sha256
            || receipt.generation != source.generation
            || receipt.slot != source.slot
            || receipt.attempt != outcome.attempt
            || receipt.node != expected_node
            || receipt.semantic_sha256 != outcome.member.semantic_sha256
            || receipt.canonical_genome != outcome.member.canonical_genome
            || receipt.metrics != outcome.metrics
            || receipt.failure != outcome.failure
            || receipt.valid(receipt.test_rows) != outcome.valid
            || receipt.node != outcome.node
            || receipt.scq_job_id != outcome.scq_job_id
        {
            bail!("aggressive slot outcome is not fully bound to its request and receipt");
        }
    }
    Ok(())
}

pub(crate) fn confirm_winner(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    checkpoint_sha256: &str,
    test: &RegressionData,
) -> Result<(Confirmation, String)> {
    let path = confirmation_path(&manifest.run_id, &spec.dataset_hash);
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let confirmation: Confirmation = serde_json::from_slice(&bytes)?;
        let winner_slot = checkpoint.winner_slot.context("checkpoint has no winner")?;
        if bytes != canonical_json(&confirmation)?
            || confirmation.schema != CONFIRMATION_SCHEMA
            || confirmation.benchmark_valid
            || !confirmation.test_adaptive
            || confirmation.run_id != manifest.run_id
            || confirmation.dataset_hash != spec.dataset_hash
            || confirmation.generation != checkpoint.generation
            || confirmation.slot != winner_slot
            || confirmation.checkpoint_sha256 != checkpoint_sha256
            || confirmation.nodes
                != scq_batch::NODES
                    .iter()
                    .map(|node| (*node).to_string())
                    .collect::<Vec<_>>()
            || confirmation.receipt_files.len() != scq_batch::NODES.len()
            || confirmation.receipt_sha256.len() != scq_batch::NODES.len()
        {
            bail!("winner confirmation changed on resume");
        }
        return Ok((confirmation, sha256_bytes(&bytes)));
    }
    let winner_slot = checkpoint.winner_slot.context("checkpoint has no winner")?;
    let winner = &checkpoint.slots[winner_slot];
    let evaluations = scq_batch::replay_aggressive_across_nodes(
        &root(&manifest.run_id),
        &worker_identity(manifest, spec),
        Path::new(&manifest.registry_root),
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &winner.member,
        test,
    )?;
    if evaluations.len() != scq_batch::NODES.len() {
        bail!("winner confirmation did not return all three nodes");
    }
    let first = &evaluations[0].receipt;
    let first_metrics = first.metrics.as_ref().context("winner replay failed")?;
    let winner_receipt: CandidateReceipt = serde_json::from_slice(&fs::read(
        dataset_root(&manifest.run_id, &spec.dataset_hash).join(&winner.receipt_file),
    )?)?;
    let winner_metrics = winner_receipt
        .metrics
        .as_ref()
        .context("winning generation receipt failed")?;
    if winner_metrics.rmse.to_bits() != first_metrics.rmse.to_bits()
        || winner_receipt.predictions.len() != first.predictions.len()
        || winner_receipt
            .predictions
            .iter()
            .zip(&first.predictions)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        bail!("winner confirmation differs from the completed winning generation");
    }
    let prediction_bits = first
        .predictions
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    for evaluation in &evaluations[1..] {
        let metrics = evaluation
            .receipt
            .metrics
            .as_ref()
            .context("winner replay failed")?;
        if metrics.rmse.to_bits() != first_metrics.rmse.to_bits()
            || evaluation.receipt.predictions.len() != first.predictions.len()
            || evaluation
                .receipt
                .predictions
                .iter()
                .zip(&first.predictions)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            bail!("winner confirmation bits differ across nodes");
        }
    }
    let confirmation = Confirmation {
        schema: CONFIRMATION_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        generation: checkpoint.generation,
        slot: winner_slot,
        checkpoint_sha256: checkpoint_sha256.to_string(),
        prediction_bits_sha256: sha256_bytes(&prediction_bits),
        rmse_bits: first_metrics.rmse.to_bits(),
        nodes: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt.node.clone())
            .collect(),
        receipt_files: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt_file.clone())
            .collect(),
        receipt_sha256: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt_sha256.clone())
            .collect(),
    };
    let bytes = canonical_json(&confirmation)?;
    let digest = sha256_bytes(&bytes);
    atomic_immutable_write(&path, &bytes)?;
    Ok((confirmation, digest))
}

pub(crate) fn finish_dataset(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    checkpoint_sha256: &str,
    confirmation_sha256: Option<String>,
) -> Result<DatasetResult> {
    let path = result_path(&manifest.run_id, &spec.dataset_hash);
    let winner_slot = checkpoint
        .winner_slot
        .context("cannot finalize an aggressive dataset without a strict winner")?;
    let confirmation_sha256 = confirmation_sha256
        .context("cannot finalize an aggressive winner without three-node confirmation")?;
    let selected = &checkpoint.slots[winner_slot];
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        outcome: "confirmed_strict_win".to_string(),
        stopping_generation: checkpoint.generation,
        leader_rmse: spec.leader_rmse,
        best_rmse: selected.rmse(),
        best_slot: selected.slot,
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        terminal_checkpoint_sha256: checkpoint_sha256.to_string(),
        confirmation_sha256: Some(confirmation_sha256),
    };
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let existing: DatasetResult = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&existing)? || existing != result {
            bail!("immutable aggressive dataset result changed on resume");
        }
        return Ok(existing);
    }
    atomic_immutable_write(&path, &canonical_json(&result)?)?;
    Ok(result)
}
