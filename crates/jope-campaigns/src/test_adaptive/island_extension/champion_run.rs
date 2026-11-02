use super::*;

pub(super) fn run_champion_dataset(
    registry: &Path,
    manifest: &ChampionManifest,
    binding: &ChampionDatasetBinding,
    spec: &DatasetSpec,
    resume: bool,
    completed_champions: usize,
    dataset_index: usize,
) -> Result<ChampionDatasetResult> {
    let campaign_root = champion_root(&manifest.base_run_id);
    let root = campaign_root.join("datasets").join(&spec.dataset_hash);
    if root.join("final-result.v1.json").is_file() {
        return validate_champion_dataset(registry, manifest, binding, spec);
    }
    let identity = champion_identity(manifest, spec);
    let campaign_path = root.join("campaign.v1.json");
    if campaign_path.is_file() {
        if !resume {
            bail!("champion dataset already exists; pass --resume");
        }
        if fs::read(&campaign_path)? != canonical_json(&identity)?
            || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
        {
            bail!("champion dataset identity changed on resume");
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
    let initial = load_initial_island_checkpoints(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_champion_checkpoints(&root, &identity, &binding.generation_29_checkpoint_sha256)?;
    validate_champion_history(&root, &identity, &initial, &checkpoints, &test)?;
    if checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.qualifying_descendant_win)
    {
        return validate_champion_dataset(registry, manifest, binding, spec);
    }
    let mut cache = load_island_cache(&root, &checkpoints)?;
    let mut population = checkpoints.last().map_or_else(
        || generation_30_populations(&initial, spec.features),
        |checkpoint| {
            Ok(checkpoint
                .0
                .islands
                .iter()
                .map(|island| island.next_population.clone())
                .collect())
        },
    )?;
    let mut previous = checkpoints.last().map_or_else(
        || binding.generation_29_checkpoint_sha256.clone(),
        |checkpoint| checkpoint.1.clone(),
    );
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    let mut completed = checkpoints;
    let worker = worker_identity(&identity);
    loop {
        let generation = CHAMPION_START_GENERATION + completed.len();
        let (outcomes, fresh) = evaluate_island_generation(
            &worker,
            registry,
            &campaign_root,
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
                bail!(
                    "champion generation {generation} island {island} has only {valid} valid candidates"
                );
            }
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        if valid < GLOBAL_MIN_VALID {
            bail!("champion generation {generation} has only {valid} valid candidates");
        }
        cumulative_fresh += fresh;
        let combined = prior_history(&manifest.base_run_id, spec)?;
        let (checkpoint, next) = make_champion_checkpoint(
            &identity,
            &root,
            generation,
            previous.clone(),
            outcomes,
            cumulative_fresh,
            combined[0].0.g0_best_rmse,
        )?;
        if checkpoint.fresh_evaluations != fresh {
            bail!("fresh champion evaluation accounting mismatch");
        }
        let bytes = canonical_json(&checkpoint)?;
        let sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        champion_generation_progress(&spec.dataset_hash, &checkpoint, completed_champions);
        let won = checkpoint.qualifying_descendant_win;
        previous = sha256.clone();
        population = next;
        completed.push((checkpoint, sha256));
        if won {
            break;
        }
    }
    let terminal = &completed.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let all_checkpoints = initial
        .iter()
        .cloned()
        .chain(completed.iter().cloned())
        .collect::<Vec<_>>();
    let champion = island_genome_record(
        &root,
        &combined,
        &all_checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let replay = replay_island_champion(
        &worker,
        registry,
        &campaign_root,
        &manifest.scq_binary_sha256,
        terminal.generation,
        &selected.member,
        &test,
    )?;
    let result = ChampionDatasetResult {
        schema: CHAMPION_RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        champion_run_id: manifest.champion_run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        outcome: "new_leader_found".to_string(),
        start_generation: CHAMPION_START_GENERATION,
        stopping_generation: terminal.generation,
        generations_evaluated: completed.len(),
        logical_evaluations: completed.len() * TOTAL_POPULATION,
        fresh_evaluations: cumulative_fresh,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        champion,
        replay,
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    validate_champion_dataset(registry, manifest, binding, spec)
}

pub(super) fn make_champion_best_solutions(
    manifest: &ChampionManifest,
    results: &[ChampionDatasetResult],
) -> Result<Golden42BestSolutions> {
    let initial_path = island_root(&manifest.base_run_id).join(BEST_SOLUTIONS_FILE);
    let initial: Golden42BestSolutions = serde_json::from_slice(&fs::read(&initial_path)?)?;
    let by_dataset = results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let mut solutions = Vec::with_capacity(initial.solutions.len());
    for solution in initial.solutions {
        if solution.benchmark_status == "surpassed_external_benchmark" {
            solutions.push(solution);
            continue;
        }
        let result = by_dataset
            .get(solution.dataset_hash.as_str())
            .context("unresolved Golden42 solution has no champion result")?;
        let source = champion_root(&manifest.base_run_id)
            .join("datasets")
            .join(&result.dataset_hash)
            .join("final-result.v1.json");
        solutions.push(Golden42BestSolution {
            dataset_hash: result.dataset_hash.clone(),
            source_campaign: manifest.champion_run_id.clone(),
            source_artifact: artifact_name(&source),
            source_artifact_sha256: sha256_file(&source)?,
            benchmark_status: benchmark_status(result.champion.rmse, result.leader_rmse),
            leader_rmse: result.leader_rmse,
            selected_rmse: result.champion.rmse,
            margin_vs_leader: result.leader_rmse - result.champion.rmse,
            canonical_genome: result.champion.canonical_genome.clone(),
            semantic_sha256: result.champion.semantic_sha256.clone(),
            generation: result.champion.generation,
            island: Some(result.champion.island),
            slot: result.champion.global_slot,
            binary_sha256: result.champion.binary_sha256.clone(),
            prediction_sha256: result.champion.prediction_sha256.clone(),
            receipt_sha256: result.champion.receipt_sha256.clone(),
            replay_receipt_sha256: result
                .replay
                .iter()
                .map(|replay| replay.receipt_sha256.clone())
                .collect(),
        });
    }
    if solutions.len() != 42
        || solutions
            .iter()
            .any(|solution| solution.benchmark_status != "surpassed_external_benchmark")
        || solutions
            .iter()
            .map(|solution| solution.dataset_hash.as_str())
            .collect::<Vec<_>>()
            != initial
                .ordered_dataset_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
    {
        bail!("champion best-solutions map is not an ordered 42/42 winner map");
    }
    Ok(Golden42BestSolutions {
        schema: BEST_SOLUTIONS_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: manifest.champion_run_id.clone(),
        ordered_dataset_ids: initial.ordered_dataset_ids,
        ordered_cohort_sha256: initial.ordered_cohort_sha256,
        dataset_count: 42,
        prior_winner_count: 42 - results.len(),
        island_selected_count: results.len(),
        solutions,
    })
}
