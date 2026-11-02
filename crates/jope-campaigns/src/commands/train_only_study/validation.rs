use super::*;

pub(super) fn validate_generation_zero(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    ranking: &GlobalRankingReceipt,
) -> Result<TrainOnlyGenerationZero> {
    let (generation_zero, digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    if digest != dataset.generation_zero_sha256
        || generation_zero.schema != "TrainOnlyGenerationZero.v2"
        || generation_zero.run_id != manifest.run_id
        || generation_zero.dataset_hash != dataset.dataset_hash
        || generation_zero.train_sha256 != dataset.train_sha256
        || generation_zero.encoding_sha256 != dataset.encoding.context_sha256
        || generation_zero.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || generation_zero.dl_proposals_sha256 != manifest.dl_proposals_sha256
        || generation_zero.transfer_ranking_sha256 != dataset.transfer_ranking_sha256
        || generation_zero.members.len() != POPULATION
    {
        bail!("generation-zero receipt drift for {}", dataset.dataset_hash);
    }
    let mut semantics = HashSet::new();
    let rank_by_semantic = ranking
        .genomes
        .iter()
        .map(|genome| (genome.semantic_sha256.as_str(), genome.rank))
        .collect::<HashMap<_, _>>();
    let mut previous_transfer_rank = None;
    for (slot, member) in generation_zero.members.iter().enumerate() {
        if member.slot != slot
            || semantic_identity(&member.canonical_genome) != member.semantic_sha256
            || !genome_compatible(&member.canonical_genome, dataset.features)
            || member.complexity != genome_complexity(&GenomeIr::parse(&member.canonical_genome)?)
            || !semantics.insert(member.semantic_sha256.as_str())
        {
            bail!("generation-zero canonical identity drift");
        }
        if slot < DL_QUOTA {
            if member.source != "latest_checkpoint_dl"
                || member.ancestry.roots != BTreeSet::from(["dl".to_string()])
            {
                bail!("generation-zero DL quota/source drift");
            }
        } else {
            if member.source != "target_clean_cross_dataset_transfer"
                || member.ancestry.roots != BTreeSet::from(["transfer".to_string()])
            {
                bail!("generation-zero transfer quota/source drift");
            }
            let rank = *rank_by_semantic
                .get(member.semantic_sha256.as_str())
                .context("transfer seed is absent from target-clean ranking")?;
            if previous_transfer_rank.is_some_and(|previous| rank <= previous) {
                bail!("generation-zero transfer seeds do not walk the target-clean ranking");
            }
            previous_transfer_rank = Some(rank);
        }
    }
    Ok(generation_zero)
}

pub(super) fn validate_observation_link(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    member: &GenomeMember,
    split: &SplitReceipt,
    link: &ObservationLink,
) -> Result<TrainOnlyFitnessObservation> {
    let (observation, digest): (TrainOnlyFitnessObservation, _) = read_receipt(&link.receipt_path)?;
    let expected_key = cache_key(
        &dataset.train_sha256,
        &member.semantic_sha256,
        &split.split_sha256,
    )?;
    if digest != link.receipt_sha256
        || link.split_generation != split.generation
        || link.split_sha256 != split.split_sha256
        || link.cache_key != expected_key
        || observation.schema != "TrainOnlyFitnessObservation.v2"
        || observation.run_id != manifest.run_id
        || observation.dataset_hash != dataset.dataset_hash
        || observation.train_sha256 != dataset.train_sha256
        || observation.encoding_sha256 != dataset.encoding.context_sha256
        || observation.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || observation.canonical_genome != member.canonical_genome
        || observation.semantic_sha256 != member.semantic_sha256
        || observation.split_generation != split.generation
        || observation.split_sha256 != split.split_sha256
        || observation.kernel_version != KERNEL_VERSION
        || observation.cache_key != expected_key
    {
        bail!("fitness observation identity drift");
    }
    if observation.failure_reason.is_none()
        && (observation.r2.is_none() || observation.rmse.is_none() || observation.mae.is_none())
    {
        bail!("successful fitness observation lacks metrics");
    }
    if !link.cache_hit {
        let genome = GenomeIr::parse(&member.canonical_genome)?;
        let replay = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate(&genome, split)
        }));
        match replay {
            Ok(Ok((rmse, r2, mae, _, _)))
                if observation.failure_reason.is_none()
                    && observation.rmse == Some(rmse)
                    && observation.r2 == Some(r2)
                    && observation.mae == Some(mae) => {}
            Ok(Err(_)) | Err(_) if observation.failure_reason.is_some() => {}
            _ => bail!("fitness observation metric replay drift"),
        }
    }
    Ok(observation)
}

pub(super) fn validate_dataset_run(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    ranking: &GlobalRankingReceipt,
    pool: &ThreadPool,
) -> Result<String> {
    let data = read_regression_csv(&dataset.train_path)?;
    let rebuilt_splits = build_splits(
        &manifest.run_id,
        &dataset.dataset_hash,
        &dataset.train_sha256,
        &data.y,
        manifest.generations,
    )?;
    let (splits, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    if split_digest != dataset.split_manifest_sha256
        || receipt_digest(&rebuilt_splits)? != split_digest
    {
        bail!("split replay drift for {}", dataset.dataset_hash);
    }
    let generation_zero = validate_generation_zero(manifest, dataset, ranking)?;
    let evaluator = StudyEvalContext::new(&data, dataset, &splits)?;
    let mut history = generation_zero
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut previous = None;
    let mut checkpoint_hashes = Vec::new();
    let mut champions = Vec::new();
    for generation in 0..manifest.generations {
        evaluator.begin_generation(generation);
        let path = checkpoint_path(root, &dataset.dataset_hash, generation);
        let (checkpoint, digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(&path)?;
        if checkpoint.schema != "TrainOnlyGenerationCheckpoint.v2"
            || checkpoint.run_id != manifest.run_id
            || checkpoint.dataset_hash != dataset.dataset_hash
            || checkpoint.train_sha256 != dataset.train_sha256
            || checkpoint.encoding_sha256 != dataset.encoding.context_sha256
            || checkpoint.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.split_manifest_sha256 != split_digest
            || checkpoint.population.len() != POPULATION
            || checkpoint.rankable_members < MIN_RANKABLE
        {
            bail!("generation checkpoint contract drift at generation {generation}");
        }
        let mut semantics = HashSet::new();
        for scored in &checkpoint.population {
            if !semantics.insert(scored.member.semantic_sha256.as_str())
                || semantic_identity(&scored.member.canonical_genome)
                    != scored.member.semantic_sha256
                || scored.observations.len() != generation + 1
                || (generation > 0
                    && (scored.member.slot < ELITES)
                        != (scored.member.ancestry.operation == "elite"))
            {
                bail!("population identity or paired-history drift");
            }
        }
        let checked = pool.install(|| {
            checkpoint
                .population
                .par_iter()
                .map(|scored| -> Result<(usize, usize)> {
                    let mut r2 = Vec::new();
                    let mut cache_hits = 0usize;
                    let mut cache_misses = 0usize;
                    for (split, link) in splits.splits[..=generation]
                        .iter()
                        .zip(&scored.observations)
                    {
                        let observation = validate_observation_link(
                            manifest,
                            dataset,
                            &evaluator,
                            &scored.member,
                            split,
                            link,
                        )?;
                        if generation == 0 && link.cache_hit {
                            bail!("generation zero unexpectedly reused an evaluation");
                        }
                        if generation > 0 {
                            let expected_hit = scored.member.ancestry.operation == "elite"
                                && split.generation < generation;
                            if link.cache_hit != expected_hit {
                                bail!("elite/new-descendant cache accounting drift");
                            }
                        }
                        if link.cache_hit {
                            cache_hits += 1;
                        } else {
                            cache_misses += 1;
                        }
                        if let Some(score) = observation.r2 {
                            r2.push(score);
                        }
                    }
                    let expected_fitness = if r2.len() == generation + 1 {
                        Some(paired_robust_fitness(&r2)?)
                    } else {
                        None
                    };
                    let expected_worst = expected_fitness.map(|_| {
                        r2.iter()
                            .copied()
                            .min_by(f64::total_cmp)
                            .expect("nonempty history")
                    });
                    if scored.split_r2 != r2
                        || scored.robust_fitness != expected_fitness
                        || scored.worst_split_r2 != expected_worst
                    {
                        bail!("paired robust fitness replay drift");
                    }
                    Ok((cache_hits, cache_misses))
                })
                .collect::<Vec<_>>()
        });
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;
        for row in checked {
            let (member_hits, member_misses) = row?;
            cache_hits += member_hits;
            cache_misses += member_misses;
        }
        let mut replay_ranked = checkpoint.population.clone();
        replay_ranked.sort_by(rank_scored);
        let replay_semantics = replay_ranked
            .iter()
            .map(|member| member.member.semantic_sha256.clone())
            .collect::<Vec<_>>();
        if replay_semantics != checkpoint.ranked_semantic_sha256
            || checkpoint.champion_semantic_sha256 != replay_semantics[0]
            || checkpoint.cache_hits != cache_hits
            || checkpoint.cache_misses != cache_misses
            || checkpoint.invalid_members != POPULATION - checkpoint.rankable_members
            || checkpoint.rankable_members
                != checkpoint
                    .population
                    .iter()
                    .filter(|member| member.robust_fitness.is_some())
                    .count()
        {
            bail!("ranking, winner, or cache summary replay drift");
        }
        if generation + 1 < manifest.generations {
            let mut replay_history = history.clone();
            let expected_next = refill_population(
                &manifest.run_id,
                &dataset.dataset_hash,
                dataset.features,
                generation + 1,
                &replay_ranked,
                &mut replay_history,
            )?;
            if checkpoint.next_population != expected_next
                || checkpoint.next_population[..ELITES]
                    .iter()
                    .map(|member| member.semantic_sha256.as_str())
                    .collect::<Vec<_>>()
                    != replay_ranked[..ELITES]
                        .iter()
                        .map(|member| member.member.semantic_sha256.as_str())
                        .collect::<Vec<_>>()
            {
                bail!("top-three survival or deterministic offspring replay drift");
            }
            history = replay_history;
        } else if !checkpoint.next_population.is_empty() {
            bail!("generation nine must not have a next population");
        }
        champions.push(checkpoint.champion_semantic_sha256.clone());
        checkpoint_hashes.push(digest.clone());
        previous = Some(digest);
        evaluator.sweep();
    }

    let selection_path = selection_path(root, &dataset.dataset_hash);
    let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path)?;
    let final_checkpoint: TrainOnlyGenerationCheckpoint =
        read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
            root,
            &dataset.dataset_hash,
            manifest.generations - 1,
        ))?
        .0;
    if selection.schema != "TrainOnlySelectionComplete.v2"
        || selection.run_id != manifest.run_id
        || selection.dataset_hash != dataset.dataset_hash
        || selection.train_sha256 != dataset.train_sha256
        || selection.encoding_sha256 != dataset.encoding.context_sha256
        || selection.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
        || selection.split_manifest_sha256 != split_digest
        || selection.generation_checkpoint_sha256 != checkpoint_hashes
        || selection.generation_champions != champions
        || selection.final_champion_semantic_sha256
            != final_checkpoint.population[0].member.semantic_sha256
        || selection.final_champion_genome != final_checkpoint.population[0].member.canonical_genome
    {
        bail!("selection-complete replay drift");
    }
    let expected_cumulative = receipt_digest(&json!({
        "generation_checkpoint_sha256": checkpoint_hashes,
        "generation_champions": champions,
        "split_manifest_sha256": split_digest,
        "generation_zero_sha256": dataset.generation_zero_sha256,
    }))?;
    if selection.cumulative_sha256 != expected_cumulative {
        bail!("selection cumulative digest drift");
    }
    let (fixed, _): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if fixed.schema != "TrainOnlyChampionValidation.v2"
        || fixed.run_id != manifest.run_id
        || fixed.dataset_hash != dataset.dataset_hash
        || fixed.selection_complete_sha256 != selection_digest
        || fixed.split_manifest_sha256 != split_digest
        || fixed.champions.len() != manifest.generations
    {
        bail!("champion fixed-grid receipt drift");
    }
    evaluator.begin_generation(manifest.generations);
    let fixed_checks = pool.install(|| {
        fixed
            .champions
            .par_iter()
            .map(|champion| -> Result<()> {
                let genome = GenomeIr::parse(&champion.canonical_genome)?;
                let replay_r2 = splits
                    .splits
                    .iter()
                    .map(|split| {
                        evaluator
                            .evaluate(&genome, split)
                            .map(|(_, r2, _, _, _)| r2)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if champion.generation >= manifest.generations
                    || champion.semantic_sha256
                        != selection.generation_champions[champion.generation]
                    || semantic_identity(&champion.canonical_genome) != champion.semantic_sha256
                    || champion.split_r2.len() != manifest.generations
                    || champion.split_r2 != replay_r2
                    || champion.robust_fitness != paired_robust_fitness(&champion.split_r2)?
                    || champion.worst_split_r2
                        != champion
                            .split_r2
                            .iter()
                            .copied()
                            .min_by(f64::total_cmp)
                            .unwrap()
                {
                    bail!("champion fixed-grid score replay drift");
                }
                Ok(())
            })
            .collect::<Vec<_>>()
    });
    for check in fixed_checks {
        check?;
    }
    evaluator.sweep();
    Ok(selection_digest)
}

pub(super) fn validate_dataset_shard(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    manifest_digest: &str,
    cohort: &TrainOnlyCohortSelectionComplete,
    cohort_digest: &str,
    dataset: &DatasetStudyEntry,
    pool: &ThreadPool,
) -> Result<(String, String)> {
    let (ranking, _): (GlobalRankingReceipt, _) = read_receipt(&dataset.transfer_ranking_path)?;
    let selection_digest = validate_dataset_run(root, manifest, dataset, &ranking, pool)?;
    if cohort.selection_complete_sha256.get(&dataset.dataset_hash) != Some(&selection_digest) {
        bail!("cohort dataset selection digest drift");
    }
    let (_, champion_digest): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&champion_digest) {
        bail!("cohort champion-validation digest drift");
    }
    let receipt = TrainOnlyDatasetValidationReceipt {
        schema: "TrainOnlyDatasetValidationReceipt.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        manifest_sha256: manifest_digest.to_string(),
        cohort_selection_complete_sha256: cohort_digest.to_string(),
        selection_complete_sha256: selection_digest.clone(),
        champion_validation_sha256: champion_digest,
        kernel_version: KERNEL_VERSION.to_string(),
        metric_replay_passed: true,
        population_audit_passed: true,
        paired_history_audit_passed: true,
        cache_audit_passed: true,
    };
    let digest = write_receipt(
        &dataset_validation_path(root, &dataset.dataset_hash),
        &receipt,
    )?;
    Ok((selection_digest, digest))
}

pub(super) fn read_dataset_validation_shard(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    manifest_digest: &str,
    cohort: &TrainOnlyCohortSelectionComplete,
    cohort_digest: &str,
    dataset: &DatasetStudyEntry,
) -> Result<(String, String)> {
    let (receipt, digest): (TrainOnlyDatasetValidationReceipt, _) =
        read_receipt(&dataset_validation_path(root, &dataset.dataset_hash))?;
    let (_, selection_digest): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path(root, &dataset.dataset_hash))?;
    let (_, champion_digest): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    if receipt.schema != "TrainOnlyDatasetValidationReceipt.v2"
        || receipt.study_design != STUDY_DESIGN
        || receipt.run_id != manifest.run_id
        || receipt.dataset_hash != dataset.dataset_hash
        || receipt.manifest_sha256 != manifest_digest
        || receipt.cohort_selection_complete_sha256 != cohort_digest
        || receipt.selection_complete_sha256 != selection_digest
        || receipt.champion_validation_sha256 != champion_digest
        || receipt.kernel_version != KERNEL_VERSION
        || !receipt.metric_replay_passed
        || !receipt.population_audit_passed
        || !receipt.paired_history_audit_passed
        || !receipt.cache_audit_passed
        || cohort.selection_complete_sha256.get(&dataset.dataset_hash) != Some(&selection_digest)
        || cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&champion_digest)
    {
        bail!(
            "dataset validation shard drift for {}",
            dataset.dataset_hash
        );
    }
    Ok((selection_digest, digest))
}

pub(crate) fn validate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_execution_inputs(&manifest, dataset)?;
    } else {
        validate_prepared_inputs(&manifest)?;
    }
    let (exclusions, exclusion_digest): (CorpusExclusions, _) =
        read_receipt(&manifest.corpus_exclusions_path)?;
    if exclusion_digest != manifest.corpus_exclusions_sha256
        || manifest.checkpoint_overlap_disclosure.checkpoint_sha256
            != manifest.checkpoint.checkpoint_sha256
        || manifest.checkpoint_overlap_disclosure.zero_shot_valid
        || !manifest
            .checkpoint_overlap_disclosure
            .checkpoint_was_trained_on_golden42_identities
    {
        bail!("checkpoint-overlap disclosure audit drift");
    }
    let overlapping_ids = exclusions.dataset_ids.iter().collect::<HashSet<_>>();
    let overlapping_families = exclusions.source_families.iter().collect::<HashSet<_>>();
    let overlapping_targets = exclusions.target_identities.iter().collect::<HashSet<_>>();
    for dataset in manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|wanted| dataset.dataset_hash == wanted))
    {
        let record = manifest
            .eligibility
            .iter()
            .find(|record| record.dataset_hash == dataset.dataset_hash)
            .context("selected dataset lacks eligibility record")?;
        if !record.eligible || !record.selected || !record.reasons.is_empty() {
            bail!("selected dataset eligibility audit failed");
        }
        if record.checkpoint_exact_overlap != overlapping_ids.contains(&dataset.dataset_hash)
            || record.checkpoint_source_family_overlap
                != overlapping_families.contains(&dataset.source_family)
            || record.checkpoint_target_identity_overlap
                != overlapping_targets.contains(&dataset.target_identity)
        {
            bail!("selected dataset checkpoint-overlap disclosure drift");
        }
        let (ranking, ranking_digest): (GlobalRankingReceipt, _) =
            read_receipt(&dataset.transfer_ranking_path)?;
        if ranking_digest != dataset.transfer_ranking_sha256
            || ranking.schema != "TrainOnlyTransferGenomeRanking.v2"
            || ranking.run_id != manifest.run_id
            || ranking.target_dataset_id != dataset.dataset_hash
            || ranking.target_source_family != dataset.source_family
            || ranking.observation_db_sha256 != manifest.observation_db_sha256
            || ranking.source_catalog_sha256 != manifest.source_catalog_sha256
            || ranking.excluded_source_family != dataset.source_family
            || !ranking.excluded_dataset_ids.contains(&dataset.dataset_hash)
            || !ranking.direct_dl_sources_excluded
        {
            bail!("target transfer-ranking identity drift");
        }
        for (rank, genome) in ranking.genomes.iter().enumerate() {
            if genome.rank != rank
                || genome
                    .source_dataset_ids
                    .iter()
                    .any(|source| ranking.excluded_dataset_ids.contains(source))
                || genome.source_families.contains(&dataset.source_family)
            {
                bail!("target/source-family transfer exclusion drift");
            }
        }
        let mut blob = File::open(&dataset.encoding.blob_path)?;
        blob.seek(SeekFrom::Start(dataset.encoding.offset_bytes as u64))?;
        let mut bytes = vec![0_u8; dataset.encoding.length_bytes];
        blob.read_exact(&mut bytes)?;
        if sha256_bytes(&bytes) != dataset.encoding.context_sha256 {
            bail!("filtered context slice audit failed");
        }
    }
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if cohort.schema != "TrainOnlyCohortSelectionComplete.v2"
        || cohort.run_id != manifest.run_id
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
        || cohort.selection_complete_sha256.len() != manifest.datasets.len()
        || cohort.champion_validation_sha256.len() != manifest.datasets.len()
    {
        bail!("cohort selection-complete receipt drift");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build study validation worker pool")?;
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_dataset_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
            &pool,
        )?;
        return Ok(dataset_validation_path(&root, wanted));
    }
    let mut selections = BTreeMap::new();
    let mut dataset_validations = BTreeMap::new();
    for dataset in &manifest.datasets {
        let path = dataset_validation_path(&root, &dataset.dataset_hash);
        let (selection_digest, validation_digest) = if path.exists() {
            read_dataset_validation_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
            )?
        } else {
            validate_dataset_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
                &pool,
            )?
        };
        selections.insert(dataset.dataset_hash.clone(), selection_digest);
        dataset_validations.insert(dataset.dataset_hash.clone(), validation_digest);
    }
    let expected_cohort_cumulative = receipt_digest(&json!({
        "manifest_sha256": manifest_digest,
        "selection_complete_sha256": cohort.selection_complete_sha256,
        "champion_validation_sha256": cohort.champion_validation_sha256,
    }))?;
    if cohort.cumulative_sha256 != expected_cohort_cumulative {
        bail!("cohort cumulative digest drift");
    }
    let receipt = TrainOnlyValidationReceipt {
        schema: "TrainOnlyValidationReceipt.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure: manifest.checkpoint_overlap_disclosure.clone(),
        run_id: manifest.run_id.clone(),
        manifest_sha256: manifest_digest,
        cohort_selection_complete_sha256: cohort_digest,
        dataset_selection_sha256: selections,
        dataset_validation_sha256: dataset_validations,
        benchmark_valid: true,
        checkpoint_disclosure_audit_passed: true,
        transfer_exclusion_audit_passed: true,
        firewall_audit_passed: true,
        population_audit_passed: true,
        paired_history_audit_passed: true,
        cache_audit_passed: true,
        validated_dataset_count: manifest.datasets.len(),
    };
    let path = root.join("validation-success.v2.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}

pub(super) fn require_validated_score_gate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<(
    PathBuf,
    TrainOnlyStudyManifest,
    String,
    TrainOnlyValidationReceipt,
    String,
)> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let validation_path = root.join("validation-success.v2.json");
    if !validation_path.exists() {
        bail!(
            "post-hoc scoring requires a successful independent validation receipt: {}",
            validation_path.display()
        );
    }

    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    validate_prepared_inputs(&manifest)?;
    let (validation, validation_digest): (TrainOnlyValidationReceipt, _) =
        read_receipt(&validation_path)?;
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if validation.schema != "TrainOnlyValidationReceipt.v2"
        || validation.study_design != STUDY_DESIGN
        || validation.zero_shot_valid
        || validation.run_id != run_id
        || validation.manifest_sha256 != manifest_digest
        || validation.cohort_selection_complete_sha256 != cohort_digest
        || !validation.benchmark_valid
        || !validation.checkpoint_disclosure_audit_passed
        || !validation.transfer_exclusion_audit_passed
        || !validation.firewall_audit_passed
        || !validation.population_audit_passed
        || !validation.paired_history_audit_passed
        || !validation.cache_audit_passed
        || validation.validated_dataset_count != manifest.datasets.len()
        || validation.dataset_selection_sha256.len() != manifest.datasets.len()
        || validation.dataset_validation_sha256.len() != manifest.datasets.len()
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
    {
        bail!("validation receipt does not authorize post-hoc scoring");
    }
    for dataset in &manifest.datasets {
        let (shard_selection_digest, shard_digest) = read_dataset_validation_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
        )?;
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
            || cohort.selection_complete_sha256.get(&dataset.dataset_hash)
                != Some(&selection_digest)
            || shard_selection_digest != selection_digest
            || validation
                .dataset_validation_sha256
                .get(&dataset.dataset_hash)
                != Some(&shard_digest)
            || selection.generation_checkpoint_sha256.len() != manifest.generations
        {
            bail!(
                "validated selection digest drift for {}",
                dataset.dataset_hash
            );
        }
        let (_, fixed_digest): (TrainOnlyChampionValidation, _) =
            read_receipt(&champion_validation_path(&root, &dataset.dataset_hash))?;
        if cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&fixed_digest) {
            bail!(
                "validated champion-grid digest drift for {}",
                dataset.dataset_hash
            );
        }
        for (generation, expected_digest) in
            selection.generation_checkpoint_sha256.iter().enumerate()
        {
            let (_, digest): (TrainOnlyGenerationCheckpoint, _) =
                read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
            if &digest != expected_digest {
                bail!(
                    "validated generation checkpoint digest drift for {} generation {}",
                    dataset.dataset_hash,
                    generation
                );
            }
        }
    }
    Ok((
        root,
        manifest,
        manifest_digest,
        validation,
        validation_digest,
    ))
}
