fn run_smart_dataset(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    resume: bool,
    workers: usize,
) -> Result<(PathBuf, String)> {
    let output = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart-v3-complete.v1.json");
    if output.exists() {
        if !resume {
            bail!("Smart-v3 output exists; pass --resume");
        }
        let (_, digest): (SmartComplete, _) = read_receipt(&output)?;
        return Ok((output, digest));
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let (bank, bank_digest): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    if bank_digest != dataset.fold_bank_sha256 {
        bail!("Smart-v3 fold-bank drift");
    }
    let (screening, screening_digest, _) =
        build_screening(root, manifest, dataset, &data, &bank, workers)?;
    let mut populations = screening.initial_populations;
    let mut histories = populations
        .iter()
        .map(|(island, candidates)| {
            (
                island.clone(),
                candidates
                    .iter()
                    .map(|candidate| candidate.semantic_sha256.clone())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut immutable_archive = BTreeMap::<String, HashMap<String, RankedCandidate>>::from([
        ("dl".to_string(), HashMap::new()),
        ("transfer".to_string(), HashMap::new()),
        ("mixed".to_string(), HashMap::new()),
    ]);
    let mut structural_niches = HashSet::new();
    let mut previous = None;
    let mut generation_digests = Vec::new();
    let mut mutation_counts = BTreeMap::new();
    let mut previous_best = None;
    let mut stagnant = 0;
    let mut stopped_early = false;
    let mut generations_completed = 0;
    for generation in 0..manifest.config.smart_v3.maximum_generations {
        let generation_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart")
            .join(format!("generation-{generation:03}.v1.json"));
        if generation_path.exists() {
            if !resume {
                bail!("Smart-v3 generation exists; pass --resume");
            }
            let (checkpoint, digest): (SmartGenerationReceipt, _) = read_receipt(&generation_path)?;
            if checkpoint.schema != "TrainOnlyRedteamSmartGeneration.v1"
                || checkpoint.run_id != manifest.run_id
                || checkpoint.dataset_hash != dataset.dataset_hash
                || !checkpoint.development_only
                || checkpoint.confirmatory_valid
                || checkpoint.generation != generation
                || checkpoint.previous_generation_sha256 != previous
            {
                bail!("Smart-v3 resume chain drift");
            }
            for (island, archive_digest) in &checkpoint.archive_sha256 {
                let archive_path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join("smart")
                    .join("archive")
                    .join(format!("{island}-generation-{generation:03}.v1.json"));
                let (archive, digest): (IslandArchiveReceipt, _) = read_receipt(&archive_path)?;
                if &digest != archive_digest
                    || !archive.development_only
                    || archive.confirmatory_valid
                {
                    bail!("Smart-v3 archive link drift");
                }
                for candidate in archive.candidates {
                    immutable_archive
                        .get_mut(island)
                        .unwrap()
                        .insert(candidate.candidate.semantic_sha256.clone(), candidate);
                }
            }
            for (operator, count) in checkpoint.mutation_counts {
                *mutation_counts.entry(operator).or_default() += count;
            }
            previous_best = Some(checkpoint.best_robust_loss);
            stagnant = checkpoint.consecutive_stagnant_generations;
            populations = checkpoint.next_populations;
            for (island, candidates) in &checkpoint.populations {
                histories
                    .get_mut(island)
                    .context("Smart-v3 resume island history missing")?
                    .extend(
                        candidates
                            .iter()
                            .map(|candidate| candidate.candidate.semantic_sha256.clone()),
                    );
                for (index, candidate) in candidates.iter().enumerate() {
                    if is_nondominated(candidates, index) {
                        structural_niches.insert(structural_niche(&candidate.candidate)?);
                    }
                }
            }
            for (island, candidates) in &populations {
                histories
                    .get_mut(island)
                    .context("Smart-v3 resume next-island history missing")?
                    .extend(
                        candidates
                            .iter()
                            .map(|candidate| candidate.semantic_sha256.clone()),
                    );
            }
            previous = Some(digest.clone());
            generation_digests.push(digest);
            generations_completed = generation + 1;
            if checkpoint.stopped {
                stopped_early = true;
                break;
            }
            continue;
        }
        let mut scored_populations = BTreeMap::new();
        let mut archive_sha256 = BTreeMap::new();
        let mut best_loss = f64::INFINITY;
        let mut new_niche = false;
        for island in ["dl", "transfer", "mixed"] {
            let population = populations
                .get(island)
                .with_context(|| format!("Smart-v3 missing island {island}"))?;
            let scored = evaluate_candidates(
                root,
                manifest,
                dataset,
                &data,
                "search",
                &bank.search,
                population,
                workers,
            )?;
            if scored.len() != ISLAND_SIZE {
                bail!("Smart-v3 island size drift");
            }
            let order = nsga_order(&scored);
            best_loss = best_loss.min(
                scored
                    .iter()
                    .map(|candidate| candidate.robust_loss)
                    .min_by(f64::total_cmp)
                    .unwrap(),
            );
            for index in order
                .iter()
                .take_while(|index| is_nondominated(&scored, **index))
            {
                new_niche |= structural_niches.insert(structural_niche(&scored[*index].candidate)?);
            }
            let archived = order
                .iter()
                .take(3)
                .map(|index| scored[*index].clone())
                .collect::<Vec<_>>();
            for candidate in &archived {
                immutable_archive.get_mut(island).unwrap().insert(
                    candidate.candidate.semantic_sha256.clone(),
                    candidate.clone(),
                );
            }
            let archive_receipt = IslandArchiveReceipt {
                schema: "TrainOnlyRedteamSmartArchive.v1".to_string(),
                run_id: manifest.run_id.clone(),
                dataset_hash: dataset.dataset_hash.clone(),
                island: island.to_string(),
                generation,
                candidates: archived,
                development_only: true,
                confirmatory_valid: false,
            };
            let archive_path = root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("smart")
                .join("archive")
                .join(format!("{island}-generation-{generation:03}.v1.json"));
            archive_sha256.insert(
                island.to_string(),
                write_receipt(&archive_path, &archive_receipt)?,
            );
            scored_populations.insert(island.to_string(), scored);
        }
        let improved = previous_best
            .is_none_or(|previous| (previous - best_loss) / previous.abs().max(1e-15) >= 0.001);
        if improved || new_niche {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        let can_stop = generation + 1 >= manifest.config.smart_v3.minimum_generations
            && stagnant >= manifest.config.smart_v3.stagnation_generations;
        let last = generation + 1 == manifest.config.smart_v3.maximum_generations;
        let mut next_populations = BTreeMap::new();
        let mut generation_mutations = BTreeMap::new();
        if !can_stop && !last {
            for island in ["dl", "transfer", "mixed"] {
                let (next, counts) = smart_refill(
                    manifest,
                    &dataset.dataset_hash,
                    island,
                    generation + 1,
                    &scored_populations[island],
                    (island == "mixed").then(|| {
                        (
                            scored_populations["dl"].as_slice(),
                            scored_populations["transfer"].as_slice(),
                        )
                    }),
                    histories.get_mut(island).unwrap(),
                )?;
                next_populations.insert(island.to_string(), next);
                for (operator, count) in counts {
                    *generation_mutations.entry(operator.clone()).or_default() += count;
                    *mutation_counts.entry(operator).or_default() += count;
                }
            }
        }
        let checkpoint = SmartGenerationReceipt {
            schema: "TrainOnlyRedteamSmartGeneration.v1".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            generation,
            previous_generation_sha256: previous.clone(),
            populations: scored_populations,
            next_populations: next_populations.clone(),
            archive_sha256,
            best_robust_loss: best_loss,
            new_nondominated_structural_niche: new_niche,
            consecutive_stagnant_generations: stagnant,
            mutation_counts: generation_mutations,
            stopped: can_stop || last,
            development_only: true,
            confirmatory_valid: false,
        };
        let digest = write_receipt(&generation_path, &checkpoint)?;
        previous = Some(digest.clone());
        previous_best = Some(best_loss);
        generation_digests.push(digest);
        populations = next_populations;
        generations_completed = generation + 1;
        if can_stop {
            stopped_early = true;
            break;
        }
    }
    if generations_completed < manifest.config.smart_v3.minimum_generations {
        bail!("Smart-v3 stopped before its minimum generation count");
    }
    let mut archive_top20_by_island = BTreeMap::new();
    for island in ["dl", "transfer", "mixed"] {
        let mut candidates = immutable_archive
            .remove(island)
            .unwrap()
            .into_values()
            .collect::<Vec<_>>();
        candidates.sort_by(rank_loss);
        candidates.truncate(20);
        archive_top20_by_island.insert(island.to_string(), candidates);
    }
    let complete = SmartComplete {
        schema: "TrainOnlyRedteamSmartComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        generations_completed,
        stopped_early,
        screening_sha256: screening_digest,
        generation_sha256: generation_digests,
        archive_top20_by_island,
        mutation_counts,
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &complete)?;
    Ok((output, digest))
}
