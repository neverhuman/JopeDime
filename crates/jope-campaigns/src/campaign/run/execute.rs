use super::*;

pub(crate) fn execute_campaign(
    config: &CampaignConfig,
    train: &RegressionData,
    mut population: Vec<PopulationMember>,
    initial_population_sha256: &str,
    artifact_root: &Path,
) -> Result<SelectionFreeze> {
    validate_campaign_config(config, train, &population)?;
    let identity = CampaignIdentity {
        schema: CAMPAIGN_SCHEMA,
        evaluator: EVALUATOR,
        dataset_hash: &config.dataset,
        train_sha256: &config.train_sha256,
        train_rows: train.y.len(),
        generations: config.generations,
        population: config.population,
        cv_folds: config.cv_folds,
        confirm_folds: config.confirm_folds,
        elites: config.elites,
        parent_pool: config.parent_pool,
        tournament: config.tournament,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        seed: config.seed,
        rayon_threads: config.rayon_threads,
        member_execution: "one dedicated OS thread per population member",
        dedicated_member_threads: config.population,
        initial_population_sha256,
        quick_holdout_fold: QUICK_CV5_HOLDOUT_FOLD,
        final_cv7_candidates: config.final_cv7_candidates,
    };
    immutable_write(
        &artifact_root.join("campaign.v1.json"),
        &canonical_json(&identity)?,
    )?;
    publish_initial_population(artifact_root, &population, initial_population_sha256)?;
    let live_logger = LiveLogger::create(
        artifact_root,
        config.generations.saturating_mul(config.population),
        config.population,
        config.rayon_threads,
    )?;
    let cv5_manifest = build_fold_manifest(
        &config.dataset,
        &config.train_sha256,
        train.y.len(),
        config.cv_folds,
    )?;
    let cv5_manifest_bytes = canonical_json(&cv5_manifest)?;
    let cv5_manifest_sha256 = sha256_bytes(&cv5_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv5.v1.json"),
        &cv5_manifest_bytes,
    )?;

    let ledger_path = artifact_root.join("slots.v1.jsonl");
    let mut ledger_file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&ledger_path)
        .with_context(|| format!("create append-only ledger {}", ledger_path.display()))?;
    let campaign_started = Instant::now();
    let mut cache = HashMap::<String, CachedEvaluation>::new();
    let mut terminal_slots = 0usize;
    let mut unique_evaluations = 0usize;
    let mut generation_zero_best = None;
    let mut incumbent_history = Vec::<Incumbent>::new();
    let mut incumbent: Option<Incumbent> = None;
    let mut final_ranked = Vec::<SlotLedger>::new();

    for generation in 0..config.generations {
        let evaluated = evaluate_population(
            generation,
            &population,
            train,
            config,
            artifact_root,
            &cache,
            &live_logger,
        )?;
        let misses = evaluated.iter().filter(|item| item.cache_miss).count();
        unique_evaluations += misses;
        let mut slots = Vec::with_capacity(config.population);
        for (slot, evaluation) in evaluated.into_iter().enumerate() {
            terminal_slots += 1;
            let source = evaluation.cached.clone();
            cache
                .entry(population[slot].semantic_sha256.clone())
                .or_insert_with(|| source.clone());
            let elapsed = campaign_started.elapsed().as_secs_f64();
            let result = source.result.as_ref();
            slots.push(SlotLedger {
                schema: LEDGER_SCHEMA.to_string(),
                generation,
                slot,
                canonical_genome: population[slot].canonical_genome.clone(),
                semantic_sha256: population[slot].semantic_sha256.clone(),
                ancestry: population[slot].ancestry.clone(),
                head_family: source.head_family.clone(),
                complexity: source.complexity,
                valid: result.is_some(),
                rejection_reason: source.rejection_reason.clone(),
                cache_disposition: if evaluation.cache_miss { "miss" } else { "hit" }.to_string(),
                cache_source_generation: (!evaluation.cache_miss)
                    .then_some(source.source_generation),
                cache_source_slot: (!evaluation.cache_miss).then_some(source.source_slot),
                artifacts: source.artifacts.clone(),
                fold_r2: result.map(|value| value.fold_r2.clone()),
                pooled_oof_r2: result.map(|value| value.pooled_oof_r2),
                pooled_oof_rmse: result.map(|value| value.pooled_oof_rmse),
                robust_fitness: result.map(|value| value.robust_fitness),
                fold_fit_seconds: result.map_or_else(Vec::new, |value| value.fit_seconds.clone()),
                fold_predict_seconds: result
                    .map_or_else(Vec::new, |value| value.predict_seconds.clone()),
                cumulative_unique_evaluations: unique_evaluations,
                cumulative_terminal_slots: terminal_slots,
                cumulative_throughput_slots_per_second: if elapsed > 0.0 {
                    terminal_slots as f64 / elapsed
                } else {
                    0.0
                },
            });
        }
        let mut ranked = slots
            .iter()
            .filter(|slot| slot.rankable())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_slots);
        if ranked.len() < config.parent_pool {
            bail!(
                "generation {generation} produced only {} valid genomes; exact tournament selection requires {}",
                ranked.len(),
                config.parent_pool
            );
        }
        let best = ranked.first().unwrap();
        let best_fitness = best.robust_fitness.unwrap();
        if generation == 0 {
            generation_zero_best = Some(best_fitness);
        }
        if incumbent
            .as_ref()
            .is_none_or(|prior| best_fitness > prior.robust_fitness)
        {
            incumbent = Some(Incumbent {
                generation,
                semantic_sha256: best.semantic_sha256.clone(),
                robust_fitness: best_fitness,
            });
        }
        incumbent_history.push(incumbent.clone().unwrap());

        let next_population = if generation + 1 < config.generations {
            refill_population(config, generation + 1, &ranked)?
        } else {
            Vec::new()
        };
        for slot in &slots {
            ledger_file.write_all(&canonical_json(slot)?)?;
        }
        ledger_file.flush()?;
        ledger_file.sync_all()?;
        sync_directory(artifact_root)?;
        let ledger_sha256 = sha256_file(&ledger_path)?;
        let checkpoint = Checkpoint {
            schema: CHECKPOINT_SCHEMA,
            generation,
            terminal_slots,
            unique_evaluations,
            ledger_sha256: &ledger_sha256,
            next_population: &next_population,
        };
        immutable_write(
            &artifact_root
                .join("checkpoints")
                .join(format!("generation-{generation:03}.v1.json")),
            &canonical_json(&checkpoint)?,
        )?;
        let summary = generation_summary(
            generation,
            &slots,
            generation_zero_best.unwrap(),
            &incumbent_history,
            &next_population,
            terminal_slots,
            unique_evaluations,
            &ledger_sha256,
        )?;
        immutable_write(
            &artifact_root
                .join("summaries")
                .join(format!("generation-{generation:03}.v1.json")),
            &canonical_json(&summary)?,
        )?;
        eprintln!(
            "[jain-exact-cv] generation={generation} valid={} cache_hits={} best={best_fitness:.12} ledger={ledger_sha256}",
            ranked.len(),
            slots.iter().filter(|slot| slot.cache_disposition == "hit").count(),
        );
        final_ranked = ranked;
        population = next_population;
    }
    drop(ledger_file);

    if final_ranked.len() < config.final_cv7_candidates {
        bail!(
            "final generation has only {} quick-score candidates; full CV7 requires {}",
            final_ranked.len(),
            config.final_cv7_candidates
        );
    }
    let cv7_manifest = build_fold_manifest(
        &config.dataset,
        &config.train_sha256,
        train.y.len(),
        config.confirm_folds,
    )?;
    let cv7_manifest_bytes = canonical_json(&cv7_manifest)?;
    let cv7_manifest_sha256 = sha256_bytes(&cv7_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv7.v1.json"),
        &cv7_manifest_bytes,
    )?;
    let cv7_fold_ids = (0..config.confirm_folds).collect::<Vec<_>>();
    let cv7_finalists = thread::scope(|scope| -> Result<Vec<FinalistCv7>> {
        let mut handles = Vec::with_capacity(config.final_cv7_candidates);
        for (quick_rank, slot) in final_ranked
            .iter()
            .take(config.final_cv7_candidates)
            .enumerate()
        {
            let thread_name = format!("jain-cv7-member-{quick_rank:02}");
            let cv5_manifest_sha256 = &cv5_manifest_sha256;
            let cv7_manifest_sha256 = &cv7_manifest_sha256;
            let cv7_fold_ids = &cv7_fold_ids;
            let live_logger = &live_logger;
            handles.push(
                thread::Builder::new()
                    .name(thread_name)
                    .spawn_scoped(scope, move || -> Result<FinalistCv7> {
                        let quick_cv5_fold0 = selection_evidence_from_slot(
                            slot,
                            config.cv_folds,
                            vec![QUICK_CV5_HOLDOUT_FOLD],
                            &cv5_manifest_sha256,
                        )?;
                        let evaluated =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let genome = GenomeIr::parse(&slot.canonical_genome)?;
                                exact_cv(&genome, train, config.confirm_folds, &config.dataset)
                            }));
                        let (cv7, rejection_reason) = match evaluated {
                            Ok(Ok(result)) => (
                                Some(selection_evidence_from_result(
                                    result,
                                    config.confirm_folds,
                                    cv7_fold_ids.clone(),
                                    &cv7_manifest_sha256,
                                )),
                                None,
                            ),
                            Ok(Err(error)) => (None, Some(format!("cv7: {error:#}"))),
                            Err(payload) => (
                                None,
                                Some(format!("cv7 panic: {}", panic_message(payload.as_ref()))),
                            ),
                        };
                        let finalist = FinalistCv7 {
                            quick_rank,
                            generation: config.generations - 1,
                            slot: slot.slot,
                            canonical_genome: slot.canonical_genome.clone(),
                            semantic_sha256: slot.semantic_sha256.clone(),
                            complexity: slot.complexity,
                            quick_cv5_fold0,
                            cv7,
                            rejection_reason,
                        };
                        live_logger.record_cv7_finalist(&finalist)?;
                        Ok(finalist)
                    })
                    .with_context(|| {
                        format!("spawn dedicated CV7 thread for quick rank {quick_rank}")
                    })?,
            );
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|payload| {
                    anyhow::anyhow!(
                        "dedicated CV7 member thread panicked: {}",
                        panic_message(payload.as_ref())
                    )
                })?
            })
            .collect()
    })?;
    for finalist in &cv7_finalists {
        immutable_write(
            &artifact_root
                .join("finalists")
                .join(format!("quick-rank-{:02}.cv7.v1.json", finalist.quick_rank)),
            &canonical_json(finalist)?,
        )?;
    }
    let mut cv7_ranked = cv7_finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    cv7_ranked.sort_by(rank_finalists);
    let selected = cv7_ranked
        .first()
        .copied()
        .context("none of the top quick-score finalists completed full CV7")?;
    let ledger_sha256 = sha256_file(&ledger_path)?;
    let selection = SelectionFreeze {
        schema: SELECTION_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        dataset_hash: config.dataset.clone(),
        train_sha256: config.train_sha256.clone(),
        initial_population_sha256: initial_population_sha256.to_string(),
        generation: config.generations - 1,
        slot: selected.slot,
        canonical_genome: selected.canonical_genome.clone(),
        semantic_sha256: selected.semantic_sha256.clone(),
        complexity: selected.complexity,
        ranking_authority:
            "top 10 by quick CV5 fold-0 robust fitness; final selection by full CV7 robust fitness descending, complexity ascending, semantic_sha256 ascending"
                .to_string(),
        ledger_sha256,
        quick_cv5_fold0: selected.quick_cv5_fold0.clone(),
        cv7: selected.cv7.clone().unwrap(),
        cv7_finalists,
    };
    publish_selection_freeze(artifact_root, &selection)?;
    live_logger.record_campaign_complete(&selection)?;
    make_readonly(&ledger_path)?;
    live_logger.seal()?;
    Ok(selection)
}
