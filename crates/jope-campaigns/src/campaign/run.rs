use super::*;

mod execute;
pub(crate) use execute::execute_campaign;

pub(crate) fn live_best_cmp(left: &LiveBest, right: &LiveBest) -> Ordering {
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

pub(crate) fn validate_common_exact_flags(args: &JainRegressorArgs) -> Result<()> {
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

pub(crate) fn run_phase1(args: &JainRegressorArgs) -> Result<()> {
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

pub(crate) fn run_phase2(args: &JainRegressorArgs) -> Result<()> {
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
