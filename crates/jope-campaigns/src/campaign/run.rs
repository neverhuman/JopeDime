use super::*;

pub(super) fn live_best_cmp(left: &LiveBest, right: &LiveBest) -> Ordering {
    right
        .robust_fitness
        .total_cmp(&left.robust_fitness)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(crate) fn run(args: JainRegressorArgs) -> Result<()> {
    validate_common_exact_flags(&args)?;
    if args.exact_cv_phase2 {
        run_phase2(&args)
    } else {
        run_phase1(&args)
    }
}

pub(super) fn validate_common_exact_flags(args: &JainRegressorArgs) -> Result<()> {
    if args.fitness != "exact-cv" {
        bail!("the exact-CV runner requires --fitness exact-cv");
    }
    if !args.root.is_absolute() {
        bail!("--root must be absolute for --fitness exact-cv");
    }
    let artifact_root = args
        .exact_cv_dir
        .as_ref()
        .context("--fitness exact-cv requires --exact-cv-dir")?;
    if !artifact_root.is_absolute() {
        bail!("--exact-cv-dir must be absolute");
    }
    if args.dataset.len() != 1 || args.dataset[0] != DATASET {
        bail!("--fitness exact-cv requires exactly --dataset {DATASET}");
    }
    if args.hashes_file.is_some() || args.limit.is_some() {
        bail!("--fitness exact-cv forbids --hashes-file and --limit");
    }
    if args.overwrite
        || args.skip_existing
        || args.compound_existing
        || args.targets_file.is_some()
        || args.time_budget_secs != 0.0
    {
        bail!("--fitness exact-cv forbids overwrite/skip/compound/targets/time-budget controls");
    }
    if args.gpu_kernel_configs != 0
        || args.gpu_screen
        || args.gpu_cache_dir.is_some()
        || args.gpr_proposal_receipt.is_some()
    {
        bail!("--fitness exact-cv requires --gpu-kernel-configs 0 and forbids GPU proposal/screen flags");
    }
    if args.rung0_exact_surrogate_dedup
        || args.rung0_trace_dir.is_some()
        || args.candidate_block_dir.is_some()
        || args.candidate_block_only
        || args.calib
        || args.hard_ten_priors
        || args.lab_hurdle2_train_only
    {
        bail!("--fitness exact-cv forbids proxy, rung-trace, calibration, candidate-block, and lab lanes");
    }
    if args.dry_run || args.status_jsonl.is_some() {
        bail!("--fitness exact-cv forbids legacy dry-run/status output");
    }
    Ok(())
}

pub(super) fn run_phase1(args: &JainRegressorArgs) -> Result<()> {
    if args.gens != GENERATIONS || args.pop != POPULATION || args.folds != CV5_FOLDS {
        bail!(
            "phase one requires --gens {GENERATIONS}, --pop {POPULATION}, and --folds {CV5_FOLDS}"
        );
    }
    if args.confirm_folds != CV7_FOLDS {
        bail!("phase one requires --confirm-folds {CV7_FOLDS}");
    }
    if !args.exact_initial_population || args.seed_genomes_file.is_none() {
        bail!("phase one requires --exact-initial-population and --seed-genomes-file");
    }
    if !args.skip_holdout_metrics {
        bail!("phase one requires --skip-holdout-metrics");
    }
    if args.no_predictions {
        bail!("phase one does not accept legacy --no-predictions");
    }
    let expected_population_sha256 = args
        .initial_population_sha256
        .as_deref()
        .context("phase one requires --initial-population-sha256")?;
    validate_sha256("--initial-population-sha256", expected_population_sha256)?;
    let artifact_root = args.exact_cv_dir.as_ref().unwrap();
    require_fresh_artifact_root(artifact_root)?;

    // Deliberately name and open only train.csv in phase one.
    let train_path = args.root.join(DATASET).join("train.csv");
    validate_file_hash(&train_path, TRAIN_SHA256, "train.csv")?;
    let train = read_regression_csv(&train_path)
        .with_context(|| format!("load pinned train data {}", train_path.display()))?;
    if train.y.len() != TRAIN_ROWS {
        bail!(
            "pinned train.csv must contain {TRAIN_ROWS} rows, found {}",
            train.y.len()
        );
    }
    let seed_path = args.seed_genomes_file.as_ref().unwrap();
    let seed_bytes = fs::read(seed_path)
        .with_context(|| format!("read exact generation zero {}", seed_path.display()))?;
    if sha256_bytes(&seed_bytes) != expected_population_sha256 {
        bail!("ordered generation-zero file differs from --initial-population-sha256");
    }
    if !seed_bytes.ends_with(b"\n") {
        bail!("ordered generation-zero file must be newline-terminated");
    }
    let initial = parse_initial_population(&seed_bytes, POPULATION)?;
    let provenance_path = seed_path.with_extension("provenance.json");
    let provenance_bytes = fs::read(&provenance_path).with_context(|| {
        format!(
            "read required generation-zero provenance {}",
            provenance_path.display()
        )
    })?;
    let provenance = validate_generation_zero_provenance(
        &provenance_bytes,
        expected_population_sha256,
        POPULATION,
    )?;

    fs::create_dir_all(artifact_root)
        .with_context(|| format!("create exact-CV root {}", artifact_root.display()))?;
    sync_directory(artifact_root)?;
    immutable_write(
        &artifact_root.join("initial-population-provenance.v1.json"),
        &canonical_json(&provenance)?,
    )?;
    let threads = init_runtime_threads("jain-exact-cv", args.threads).threads;
    eprintln!(
        "[jain-exact-cv] phase1 dataset={DATASET} generations={GENERATIONS} population={POPULATION} threads={threads}"
    );
    let selection = execute_campaign(
        &CampaignConfig::production(args.seed, threads),
        &train,
        initial,
        expected_population_sha256,
        artifact_root,
    )?;
    println!("{}", serde_json::to_string(&selection)?);
    Ok(())
}

pub(super) fn run_phase2(args: &JainRegressorArgs) -> Result<()> {
    if args.exact_initial_population
        || args.seed_genomes_file.is_some()
        || args.initial_population_sha256.is_some()
    {
        bail!("phase two forbids seed and initial-population flags");
    }
    if args.skip_holdout_metrics {
        bail!("phase two must not use --skip-holdout-metrics");
    }
    let artifact_root = args.exact_cv_dir.as_ref().unwrap();

    // Receipt validation is intentionally complete before test_path is even
    // constructed, so malformed phase-two requests cannot probe test data.
    let (selection, selection_sha256) = validate_selection_freeze(artifact_root)?;
    if selection.dataset_hash != DATASET
        || selection.train_sha256 != TRAIN_SHA256
        || selection.generation != GENERATIONS - 1
        || selection.quick_cv5_fold0.fold_plan_folds != CV5_FOLDS
        || selection.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
        || selection.cv7.fold_plan_folds != CV7_FOLDS
        || selection.cv7.evaluated_fold_ids != (0..CV7_FOLDS).collect::<Vec<_>>()
        || selection.cv7_finalists.len() != FINAL_CV7_CANDIDATES
    {
        bail!("selection freeze is not for the pinned exact-CV campaign");
    }
    validate_file_hash(
        &args.root.join(DATASET).join("train.csv"),
        TRAIN_SHA256,
        "train.csv",
    )?;

    let test_path = args.root.join(DATASET).join("test.csv");
    validate_file_hash(&test_path, TEST_SHA256, "test.csv")?;
    let train = read_regression_csv(&args.root.join(DATASET).join("train.csv"))?;
    let test = read_regression_csv(&test_path)?;
    if train.y.len() != TRAIN_ROWS || test.y.len() != TEST_ROWS {
        bail!(
            "phase two requires {TRAIN_ROWS} train and {TEST_ROWS} test rows, found {} and {}",
            train.y.len(),
            test.y.len()
        );
    }
    if train.features != test.features {
        bail!("train/test feature-count mismatch");
    }
    let genome =
        GenomeIr::parse(&selection.canonical_genome).context("parse frozen selected genome")?;
    if genome.canonical() != selection.canonical_genome
        || sha256_bytes(selection.canonical_genome.as_bytes()) != selection.semantic_sha256
    {
        bail!("selection freeze canonical genome identity is invalid");
    }

    let fit_started = Instant::now();
    let model = fit(&genome, &train.x, &train.y).context("fit frozen genome on all train rows")?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    let predict_started = Instant::now();
    let predictions = model.predict(&test.x);
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    if predictions.len() != TEST_ROWS || predictions.iter().any(|value| !value.is_finite()) {
        bail!("frozen genome did not produce exactly {TEST_ROWS} finite predictions");
    }

    let phase2_dir = artifact_root.join("phase2");
    fs::create_dir_all(&phase2_dir)?;
    let predictions_path = phase2_dir.join("predictions.csv");
    if predictions_path.exists() {
        bail!("phase-two predictions already exist; exact-CV outputs are immutable");
    }
    write_predictions_atomic(&predictions_path, &predictions)?;
    let durable_predictions = read_predictions(&predictions_path)?;
    if durable_predictions.len() != TEST_ROWS
        || durable_predictions.iter().any(|value| !value.is_finite())
    {
        bail!("durable prediction file is not exactly {TEST_ROWS} finite values");
    }
    let (test_rmse, test_r2, test_mae) = independent_metrics(&test.y, &durable_predictions)?;
    let predictions_sha256 = sha256_file(&predictions_path)?;
    let receipt = Phase2Receipt {
        schema: PHASE2_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        dataset_hash: DATASET.to_string(),
        train_sha256: TRAIN_SHA256.to_string(),
        test_sha256: TEST_SHA256.to_string(),
        selection_sha256,
        semantic_sha256: selection.semantic_sha256,
        canonical_genome: selection.canonical_genome,
        train_rows: TRAIN_ROWS,
        test_rows: TEST_ROWS,
        predictions_file: "phase2/predictions.csv".to_string(),
        predictions_sha256,
        fit_seconds,
        predict_seconds,
        test_rmse,
        test_r2,
        test_mae,
        target_rmse: TARGET_RMSE,
        tabfm_ensemble_rmse: TABFM_RMSE,
        archived_jope_rmse: ARCHIVED_JOPE_RMSE,
        signed_margin_vs_target: TARGET_RMSE - test_rmse,
        signed_margin_vs_tabfm: TABFM_RMSE - test_rmse,
        signed_margin_vs_archived_jope: ARCHIVED_JOPE_RMSE - test_rmse,
        beats_target_strict: if test_rmse < TARGET_RMSE { "YES" } else { "NO" }.to_string(),
    };
    let receipt_bytes = canonical_json(&receipt)?;
    immutable_write(&phase2_dir.join("receipt.v1.json"), &receipt_bytes)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

pub(super) fn execute_campaign(
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
