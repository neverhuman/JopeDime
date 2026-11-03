use super::evaluation::StudyEvalContext;
use super::*;

pub(crate) fn checkpoint_path(root: &Path, dataset: &str, generation: usize) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("generations")
        .join(format!("generation-{generation:03}.v2.json"))
}

pub(crate) fn selection_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("selection-complete.v2.json")
}

pub(crate) fn champion_validation_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("champion-fixed-grid.v2.json")
}

pub(crate) fn dataset_validation_path(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset)
        .join("validation-success.v2.json")
}

pub(crate) fn run_dataset(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    resume: bool,
    pool: &ThreadPool,
) -> Result<(String, String)> {
    let (splits, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    let (generation_zero, generation_zero_digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    if split_digest != dataset.split_manifest_sha256
        || generation_zero_digest != dataset.generation_zero_sha256
        || splits.splits.len() != manifest.generations
        || generation_zero.members.len() != POPULATION
    {
        bail!("{} preparation receipts are invalid", dataset.dataset_hash);
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let evaluator = StudyEvalContext::new(&data, dataset, &splits)?;
    let mut population = generation_zero.members;
    let mut previous = None;
    let mut checkpoint_hashes = Vec::with_capacity(manifest.generations);
    let mut champions = Vec::with_capacity(manifest.generations);
    let mut campaign_history = population
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    if campaign_history.len() != POPULATION {
        bail!("generation zero repeats semantics");
    }

    for generation in 0..manifest.generations {
        evaluator.begin_generation(generation);
        let path = checkpoint_path(root, &dataset.dataset_hash, generation);
        if path.exists() {
            if !resume {
                bail!("generation checkpoint exists; pass --resume");
            }
            let (checkpoint, digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(&path)?;
            if checkpoint.generation != generation
                || checkpoint.previous_checkpoint_sha256 != previous
                || checkpoint.population.len() != POPULATION
            {
                bail!("resume checkpoint chain drift at generation {generation}");
            }
            champions.push(checkpoint.champion_semantic_sha256.clone());
            checkpoint_hashes.push(digest.clone());
            previous = Some(digest);
            if generation + 1 < manifest.generations {
                population = checkpoint.next_population.clone();
                for member in population.iter().skip(ELITES) {
                    if !campaign_history.insert(member.semantic_sha256.clone()) {
                        bail!("resume campaign history repeats a descendant");
                    }
                }
            }
            continue;
        }
        let scored = score_population(
            root,
            manifest,
            dataset,
            &evaluator,
            &splits,
            generation,
            &population,
            pool,
        )?;
        let rankable = scored
            .iter()
            .filter(|member| member.robust_fitness.is_some())
            .count();
        if rankable < MIN_RANKABLE {
            bail!(
                "{} generation {} has only {} rankable members",
                dataset.dataset_hash,
                generation,
                rankable
            );
        }
        let champion = scored[0].member.semantic_sha256.clone();
        let next_population = if generation + 1 < manifest.generations {
            refill_population(
                &manifest.run_id,
                &dataset.dataset_hash,
                dataset.features,
                generation + 1,
                &scored,
                &mut campaign_history,
            )?
        } else {
            Vec::new()
        };
        let cache_hits = scored
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| link.cache_hit)
            .count();
        let cache_misses = scored
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| !link.cache_hit)
            .count();
        let checkpoint = TrainOnlyGenerationCheckpoint {
            schema: "TrainOnlyGenerationCheckpoint.v2".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            train_sha256: dataset.train_sha256.clone(),
            encoding_sha256: dataset.encoding.context_sha256.clone(),
            checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
            generation,
            previous_checkpoint_sha256: previous.clone(),
            split_manifest_sha256: split_digest.clone(),
            ranked_semantic_sha256: scored
                .iter()
                .map(|member| member.member.semantic_sha256.clone())
                .collect(),
            champion_semantic_sha256: champion.clone(),
            rankable_members: rankable,
            cache_hits,
            cache_misses,
            invalid_members: POPULATION - rankable,
            population: scored,
            next_population: next_population.clone(),
        };
        let digest = write_receipt(&path, &checkpoint)?;
        champions.push(champion);
        checkpoint_hashes.push(digest.clone());
        previous = Some(digest);
        population = next_population;
        evaluator.sweep();
    }
    let final_checkpoint_path =
        checkpoint_path(root, &dataset.dataset_hash, manifest.generations - 1);
    let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) =
        read_receipt(&final_checkpoint_path)?;
    let final_member = &final_checkpoint.population[0].member;
    let cumulative_sha256 = receipt_digest(&json!({
        "generation_checkpoint_sha256": checkpoint_hashes,
        "generation_champions": champions,
        "split_manifest_sha256": split_digest,
        "generation_zero_sha256": generation_zero_digest,
    }))?;
    let selection = TrainOnlySelectionComplete {
        schema: "TrainOnlySelectionComplete.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        encoding_sha256: dataset.encoding.context_sha256.clone(),
        checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
        split_manifest_sha256: split_digest.clone(),
        generation_checkpoint_sha256: checkpoint_hashes,
        generation_champions: champions.clone(),
        final_champion_semantic_sha256: final_member.semantic_sha256.clone(),
        final_champion_genome: final_member.canonical_genome.clone(),
        cumulative_sha256,
    };
    let selection_path = selection_path(root, &dataset.dataset_hash);
    let selection_digest = write_receipt(&selection_path, &selection)?;

    let mut fixed_grid = Vec::with_capacity(manifest.generations);
    evaluator.begin_generation(manifest.generations);
    for (generation, semantic) in champions.iter().enumerate() {
        let (checkpoint, _): (TrainOnlyGenerationCheckpoint, _) =
            read_receipt(&checkpoint_path(root, &dataset.dataset_hash, generation))?;
        let member = checkpoint
            .population
            .iter()
            .find(|member| member.member.semantic_sha256 == *semantic)
            .context("frozen generation champion is absent")?
            .member
            .clone();
        let mut split_r2 = Vec::with_capacity(manifest.generations);
        for split in &splits.splits {
            let (observation, _) =
                evaluate_member_split(root, manifest, dataset, &evaluator, &member, split)?;
            split_r2.push(observation.r2.with_context(|| {
                format!("post-hoc champion failed: {:?}", observation.failure_reason)
            })?);
        }
        fixed_grid.push(ChampionFixedGrid {
            generation,
            semantic_sha256: member.semantic_sha256,
            canonical_genome: member.canonical_genome,
            robust_fitness: paired_robust_fitness(&split_r2)?,
            worst_split_r2: split_r2.iter().copied().min_by(f64::total_cmp).unwrap(),
            split_r2,
        });
    }
    let champion_validation = TrainOnlyChampionValidation {
        schema: "TrainOnlyChampionValidation.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        selection_complete_sha256: selection_digest.clone(),
        split_manifest_sha256: split_digest,
        champions: fixed_grid,
    };
    let validation_digest = write_receipt(
        &champion_validation_path(root, &dataset.dataset_hash),
        &champion_validation,
    )?;
    evaluator.sweep();
    Ok((selection_digest, validation_digest))
}

pub(crate) fn run(
    manifest_path: &Path,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let (manifest, manifest_digest) = validate_manifest_contract(manifest_path)?;
    if selected_dataset.is_none() {
        validate_prepared_inputs(&manifest)?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build study evaluation worker pool")?;
    let root = manifest_path
        .parent()
        .context("study manifest has no parent")?;
    let mut selection_hashes = BTreeMap::new();
    let mut validation_hashes = BTreeMap::new();
    let datasets = manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|wanted| dataset.dataset_hash == wanted))
        .collect::<Vec<_>>();
    if datasets.is_empty() {
        bail!("requested dataset is not in the frozen study manifest");
    }
    for dataset in datasets {
        if selected_dataset.is_some() {
            validate_execution_inputs(&manifest, dataset)?;
        }
        match run_dataset(root, &manifest, dataset, resume, &pool) {
            Ok((selection, validation)) => {
                selection_hashes.insert(dataset.dataset_hash.clone(), selection);
                validation_hashes.insert(dataset.dataset_hash.clone(), validation);
            }
            Err(error) => {
                let failure_path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join("failure.v2.json");
                let failure = json!({
                    "schema": "TrainOnlyDatasetFailure.v2",
                    "run_id": manifest.run_id,
                    "dataset_hash": dataset.dataset_hash,
                    "error": format!("{error:#}"),
                    "population_unchanged": true,
                    "dataset_not_replaced": true,
                });
                write_receipt(&failure_path, &failure)?;
                return Err(error);
            }
        }
    }
    if let Some(dataset) = selected_dataset {
        return Ok(selection_path(root, dataset));
    }
    let cumulative_sha256 = receipt_digest(&json!({
        "manifest_sha256": manifest_digest,
        "selection_complete_sha256": selection_hashes,
        "champion_validation_sha256": validation_hashes,
    }))?;
    let cohort = TrainOnlyCohortSelectionComplete {
        schema: "TrainOnlyCohortSelectionComplete.v2".to_string(),
        run_id: manifest.run_id.clone(),
        manifest_sha256: manifest_digest,
        dataset_ids: manifest.selected_dataset_ids,
        selection_complete_sha256: selection_hashes,
        champion_validation_sha256: validation_hashes,
        cumulative_sha256,
    };
    let path = root.join("selection-complete.v2.json");
    write_receipt(&path, &cohort)?;
    Ok(path)
}
