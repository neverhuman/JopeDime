fn validate_g20_generation_replay(
    root: &Path,
    dataset: &DatasetInput,
    manifest: &TrainOnlyRedteamManifest,
    data: &RegressionData,
    g20: &G20Complete,
    workers: usize,
) -> Result<()> {
    let (parent_archive, _) = load_parent_archive(dataset)?;
    let mut history = parent_archive
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let parent_checkpoint: Value = read_receipt::<Value>(&dataset.parent_generation_paths[9])?.0;
    let parent_ranked = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .context("parent generation nine lacks population")?
        .iter()
        .map(|value| {
            Ok(LegacyScoredCandidate {
                candidate: candidate_from_parent(value)?,
                split_r2: value
                    .get("split_r2")
                    .and_then(Value::as_array)
                    .context("parent score lacks split_r2")?
                    .iter()
                    .map(|value| value.as_f64().context("invalid parent split score"))
                    .collect::<Result<Vec<_>>>()?,
                robust_fitness: value.get("robust_fitness").and_then(Value::as_f64),
                worst_split_r2: value.get("worst_split_r2").and_then(Value::as_f64),
                failure_reason: value
                    .get("failure_reason")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut expected_population = legacy_refill(
        &manifest.parent.run_id,
        &dataset.dataset_hash,
        10,
        &parent_ranked,
        &mut history,
    )?;
    let mut previous = None;
    for (offset, generation) in G20_GENERATIONS.enumerate() {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20")
            .join(format!("generation-{generation:03}.v1.json"));
        let (checkpoint, digest): (G20GenerationReceipt, _) = read_receipt(&path)?;
        if digest != g20.extension_generation_sha256[offset]
            || checkpoint.generation != generation
            || checkpoint.previous_generation_sha256 != previous
            || checkpoint.parent_generation_nine_sha256 != dataset.parent_generation_sha256[9]
            || !checkpoint.development_only
            || checkpoint.confirmatory_valid
            || checkpoint.population.len() != 60
            || checkpoint.ranked_semantic_sha256
                != checkpoint
                    .population
                    .iter()
                    .map(|row| row.candidate.semantic_sha256.clone())
                    .collect::<Vec<_>>()
            || checkpoint.champion_semantic_sha256
                != checkpoint.population[0].candidate.semantic_sha256
            || expected_population
                .iter()
                .map(|candidate| &candidate.semantic_sha256)
                .collect::<Vec<_>>()
                != checkpoint
                    .population
                    .iter()
                    .map(|row| &row.candidate.semantic_sha256)
                    .collect::<Vec<_>>()
        {
            bail!("G20 deterministic checkpoint replay drift");
        }
        let rescored = score_legacy_population(
            &data,
            &dataset.dataset_hash,
            generation,
            &expected_population,
            workers,
        )?;
        if receipt_digest(&rescored)? != receipt_digest(&checkpoint.population)? {
            bail!("G20 uncached fitness replay drift");
        }
        let expected_next = if generation < 19 {
            legacy_refill(
                &manifest.parent.run_id,
                &dataset.dataset_hash,
                generation + 1,
                &rescored,
                &mut history,
            )?
        } else {
            Vec::new()
        };
        if receipt_digest(&expected_next)? != receipt_digest(&checkpoint.next_population)? {
            bail!("G20 deterministic offspring replay drift");
        }
        expected_population = expected_next;
        previous = Some(digest);
    }
    Ok(())
}

fn validate_smart_generation_replay(
    root: &Path,
    dataset: &DatasetInput,
    manifest: &TrainOnlyRedteamManifest,
    smart: &SmartComplete,
    screening: &ScreeningReceipt,
) -> Result<()> {
    validate_initial_populations(
        &screening.initial_populations,
        ISLAND_SIZE,
        &["dl", "transfer", "mixed"],
        manifest
            .config
            .smart_v3
            .progressive_screen
            .max_per_terminal_head_family,
    )?;
    let mut histories = screening
        .initial_populations
        .iter()
        .map(|(island, rows)| {
            (
                island.clone(),
                rows.iter()
                    .map(|candidate| candidate.semantic_sha256.clone())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut populations = screening.initial_populations.clone();
    let mut previous = None;
    let mut aggregate_counts = BTreeMap::new();
    let mut structural_niches = HashSet::new();
    let mut previous_best: Option<f64> = None;
    let mut stagnant = 0usize;
    for generation in 0..smart.generations_completed {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart")
            .join(format!("generation-{generation:03}.v1.json"));
        let (checkpoint, digest): (SmartGenerationReceipt, _) = read_receipt(&path)?;
        if digest != smart.generation_sha256[generation]
            || checkpoint.generation != generation
            || checkpoint.previous_generation_sha256 != previous
            || !checkpoint.development_only
            || checkpoint.confirmatory_valid
            || checkpoint.populations.len() != ISLANDS
            || checkpoint
                .populations
                .values()
                .any(|rows| rows.len() != ISLAND_SIZE)
            || populations.iter().any(|(island, expected)| {
                expected
                    .iter()
                    .map(|candidate| &candidate.semantic_sha256)
                    .collect::<Vec<_>>()
                    != checkpoint.populations[island]
                        .iter()
                        .map(|row| &row.candidate.semantic_sha256)
                        .collect::<Vec<_>>()
            })
        {
            bail!("Smart-v3 deterministic generation replay drift");
        }
        for island in ["dl", "transfer", "mixed"] {
            let order = nsga_order(&checkpoint.populations[island]);
            let archive_path = root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("smart")
                .join("archive")
                .join(format!("{island}-generation-{generation:03}.v1.json"));
            let (archive, archive_digest): (IslandArchiveReceipt, _) = read_receipt(&archive_path)?;
            if checkpoint.archive_sha256.get(island) != Some(&archive_digest)
                || !archive.development_only
                || archive.confirmatory_valid
                || archive.candidates.len() != 3
                || archive
                    .candidates
                    .iter()
                    .map(|row| &row.candidate.semantic_sha256)
                    .collect::<Vec<_>>()
                    != order[..3]
                        .iter()
                        .map(|index| {
                            &checkpoint.populations[island][*index]
                                .candidate
                                .semantic_sha256
                        })
                        .collect::<Vec<_>>()
            {
                bail!("Smart-v3 immutable archive drift");
            }
        }
        let best_loss = checkpoint
            .populations
            .values()
            .flatten()
            .map(|candidate| candidate.robust_loss)
            .min_by(f64::total_cmp)
            .context("Smart-v3 checkpoint has no scored candidates")?;
        let mut new_niche = false;
        for candidates in checkpoint.populations.values() {
            for (index, candidate) in candidates.iter().enumerate() {
                if is_nondominated(candidates, index) {
                    new_niche |= structural_niches.insert(structural_niche(&candidate.candidate)?);
                }
            }
        }
        let improved = previous_best.is_none_or(|previous| {
            (previous - best_loss) / previous.abs().max(1e-15)
                >= manifest.config.smart_v3.minimum_relative_improvement
        });
        if improved || new_niche {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        let can_stop = generation + 1 >= manifest.config.smart_v3.minimum_generations
            && stagnant >= manifest.config.smart_v3.stagnation_generations;
        let last = generation + 1 == manifest.config.smart_v3.maximum_generations;
        if checkpoint.best_robust_loss != best_loss
            || checkpoint.new_nondominated_structural_niche != new_niche
            || checkpoint.consecutive_stagnant_generations != stagnant
            || checkpoint.stopped != (can_stop || last)
        {
            bail!("Smart-v3 stopping-rule replay drift");
        }
        let mut expected_mutations = BTreeMap::new();
        let expected_next = if checkpoint.stopped {
            BTreeMap::new()
        } else {
            ["dl", "transfer", "mixed"]
                .into_iter()
                .map(|island| {
                    let (next, counts) = smart_refill(
                        manifest,
                        &dataset.dataset_hash,
                        island,
                        generation + 1,
                        &checkpoint.populations[island],
                        (island == "mixed").then(|| {
                            (
                                checkpoint.populations["dl"].as_slice(),
                                checkpoint.populations["transfer"].as_slice(),
                            )
                        }),
                        histories.get_mut(island).unwrap(),
                    )?;
                    for (operator, count) in counts {
                        *expected_mutations.entry(operator).or_default() += count;
                    }
                    Ok((island.to_string(), next))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        };
        if receipt_digest(&expected_next)? != receipt_digest(&checkpoint.next_populations)?
            || expected_mutations != checkpoint.mutation_counts
        {
            bail!("Smart-v3 deterministic offspring replay drift");
        }
        for (operator, count) in checkpoint.mutation_counts {
            *aggregate_counts.entry(operator).or_default() += count;
        }
        populations = checkpoint.next_populations;
        previous = Some(digest);
        previous_best = Some(best_loss);
    }
    if aggregate_counts != smart.mutation_counts {
        bail!("Smart-v3 mutation ledger drift");
    }
    Ok(())
}
