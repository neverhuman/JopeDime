use super::*;

pub(crate) fn run_g20_dataset(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    resume: bool,
    workers: usize,
) -> Result<(PathBuf, String)> {
    let output = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("g20-extension-complete.v1.json");
    if output.exists() {
        if !resume {
            bail!("G20 output exists; pass --resume");
        }
        let (_, digest): (G20Complete, _) = read_receipt(&output)?;
        return Ok((output, digest));
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let (mut archive, _) = load_parent_archive(dataset)?;
    let parent_checkpoint: Value = read_receipt::<Value>(&dataset.parent_generation_paths[9])?.0;
    let parent_population = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .context("parent generation nine lacks population")?;
    let parent_ranked = parent_population
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
    if parent_ranked
        .first()
        .and_then(|candidate| candidate.robust_fitness)
        .is_none()
    {
        bail!("parent generation-nine champion is not rankable");
    }
    let mut history = archive
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut population = legacy_refill(
        &manifest.parent.run_id,
        &dataset.dataset_hash,
        10,
        &parent_ranked,
        &mut history,
    )?;
    let mut previous = None;
    let mut generation_digests = Vec::new();
    let mut raw_generation_19 = None;
    for generation in G20_GENERATIONS {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20")
            .join(format!("generation-{generation:03}.v1.json"));
        if path.exists() {
            if !resume {
                bail!("G20 generation exists; pass --resume");
            }
            let (checkpoint, digest): (G20GenerationReceipt, _) = read_receipt(&path)?;
            if checkpoint.schema != "TrainOnlyRedteamG20Generation.v1"
                || checkpoint.run_id != manifest.run_id
                || checkpoint.dataset_hash != dataset.dataset_hash
                || !checkpoint.development_only
                || checkpoint.confirmatory_valid
                || checkpoint.previous_generation_sha256 != previous
                || checkpoint.generation != generation
            {
                bail!("G20 resume chain drift");
            }
            previous = Some(digest.clone());
            generation_digests.push(digest);
            archive.extend(
                checkpoint
                    .population
                    .iter()
                    .map(|candidate| candidate.candidate.clone()),
            );
            history.extend(
                checkpoint
                    .population
                    .iter()
                    .map(|candidate| candidate.candidate.semantic_sha256.clone()),
            );
            history.extend(
                checkpoint
                    .next_population
                    .iter()
                    .map(|candidate| candidate.semantic_sha256.clone()),
            );
            if generation == 19 {
                raw_generation_19 = checkpoint
                    .population
                    .first()
                    .map(|candidate| candidate.candidate.clone());
            } else {
                population = checkpoint.next_population;
            }
            continue;
        }
        let scored = score_legacy_population(
            &data,
            &dataset.dataset_hash,
            generation,
            &population,
            workers,
        )?;
        if scored
            .iter()
            .filter(|candidate| candidate.robust_fitness.is_some())
            .count()
            < 30
        {
            bail!("G20 generation {generation} has fewer than 30 rankable candidates");
        }
        let next_population = if generation < 19 {
            legacy_refill(
                &manifest.parent.run_id,
                &dataset.dataset_hash,
                generation + 1,
                &scored,
                &mut history,
            )?
        } else {
            Vec::new()
        };
        let checkpoint = G20GenerationReceipt {
            schema: "TrainOnlyRedteamG20Generation.v1".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            generation,
            parent_generation_nine_sha256: dataset.parent_generation_sha256[9].clone(),
            previous_generation_sha256: previous.clone(),
            ranked_semantic_sha256: scored
                .iter()
                .map(|candidate| candidate.candidate.semantic_sha256.clone())
                .collect(),
            champion_semantic_sha256: scored[0].candidate.semantic_sha256.clone(),
            population: scored.clone(),
            next_population: next_population.clone(),
            development_only: true,
            confirmatory_valid: false,
        };
        let digest = write_receipt(&path, &checkpoint)?;
        archive.extend(scored.into_iter().map(|candidate| candidate.candidate));
        if generation == 19 {
            // The last appended population is ranked, so its first element is
            // 59 positions before the tail.
            raw_generation_19 = archive.get(archive.len().saturating_sub(60)).cloned();
        }
        previous = Some(digest.clone());
        generation_digests.push(digest);
        population = next_population;
    }
    let mut seen = HashSet::new();
    archive.retain(|candidate| seen.insert(candidate.semantic_sha256.clone()));
    let (bank, _): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    let mut ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        &data,
        "search",
        &bank.search,
        &archive,
        workers,
    )?;
    ranked.truncate(10);
    let complete = G20Complete {
        schema: "TrainOnlyRedteamG20Complete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        parent_generation_sha256: dataset.parent_generation_sha256.clone(),
        extension_generation_sha256: generation_digests,
        raw_generation_19: raw_generation_19.context("G20 raw generation-19 champion missing")?,
        top_archive_search_oof: ranked,
        archive_unique_candidates: archive.len(),
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &complete)?;
    Ok((output, digest))
}
