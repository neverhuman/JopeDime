use super::super::StudyEvalContext;
use super::*;

pub(crate) fn baseline_metric(path: &Path, dataset_hash: &str, algorithm: &str) -> Result<f64> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read baseline {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
        || report.get("status").and_then(Value::as_str) != Some("completed")
    {
        bail!("baseline identity/status mismatch: {}", path.display());
    }
    let rmse = report
        .pointer("/metrics/rmse")
        .and_then(Value::as_f64)
        .context("baseline report lacks numeric metrics.rmse")?;
    if !rmse.is_finite() || rmse < 0.0 {
        bail!("baseline RMSE is invalid: {}", path.display());
    }
    Ok(rmse)
}

pub(crate) fn optional_baseline_metric(
    path: &Path,
    dataset_hash: &str,
    algorithm: &str,
) -> Result<Option<f64>> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read baseline {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
    {
        bail!("baseline identity mismatch: {}", path.display());
    }
    match report.get("status").and_then(Value::as_str) {
        Some("completed") => baseline_metric(path, dataset_hash, algorithm).map(Some),
        Some("failed") => Ok(None),
        _ => bail!("baseline status mismatch: {}", path.display()),
    }
}

pub(crate) fn prior_aspfm_metric(path: &Path, dataset_hash: &str) -> Result<f64> {
    let report: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read prior ASPFM report {}", path.display()))?,
    )?;
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("schema").and_then(Value::as_str) != Some("JopeDimeDatasetReport.v3")
        || report.get("status").and_then(Value::as_str) != Some("completed")
        || report.get("benchmark_valid").and_then(Value::as_bool) != Some(true)
        || report.pointer("/algorithm/name").and_then(Value::as_str) != Some("jope-dime")
        || report.get("protocol").and_then(Value::as_str) != Some("aspfm-generated")
        || report
            .pointer("/generation_contract/algorithm_input")
            .and_then(Value::as_str)
            != Some("dataset-training-data")
        || report
            .pointer("/generation_contract/algorithm_output")
            .and_then(Value::as_str)
            != Some("task-specific-executable-algorithm")
        || report
            .pointer("/generation_contract/test_targets_used_for_generation")
            .and_then(Value::as_bool)
            != Some(false)
    {
        bail!(
            "prior ASPFM report identity/status mismatch: {}",
            path.display()
        );
    }
    let rmse = report
        .pointer("/metrics/rmse")
        .and_then(Value::as_f64)
        .context("prior ASPFM report lacks numeric metrics.rmse")?;
    if !rmse.is_finite() || rmse < 0.0 {
        bail!("prior ASPFM RMSE is invalid: {}", path.display());
    }
    Ok(rmse)
}

pub(crate) fn stronger_tabfm(tabfm: Option<f64>, ensemble: f64) -> (&'static str, f64) {
    if tabfm.is_some_and(|value| value <= ensemble) {
        ("tabfm_regressor", tabfm.unwrap())
    } else {
        ("tabfm_ensemble_regressor", ensemble)
    }
}

pub(super) fn load_or_create_baselines(
    root: &Path,
    run_id: &str,
    datasets: &[DatasetStudyEntry],
) -> Result<(BaselineManifest, String)> {
    let path = root.join("baselines.v2.json");
    if path.exists() {
        let (manifest, digest): (BaselineManifest, _) = read_receipt(&path)?;
        if manifest.schema != "TrainOnlyBaselineManifest.v2"
            || manifest.run_id != run_id
            || manifest
                .entries
                .iter()
                .map(|entry| &entry.dataset_hash)
                .collect::<Vec<_>>()
                != datasets
                    .iter()
                    .map(|entry| &entry.dataset_hash)
                    .collect::<Vec<_>>()
        {
            bail!("pinned baseline manifest identity drift");
        }
        for entry in &manifest.entries {
            if sha256_file(&entry.prior_aspfm_report_path)? != entry.prior_aspfm_report_sha256
                || prior_aspfm_metric(&entry.prior_aspfm_report_path, &entry.dataset_hash)?
                    != entry.prior_aspfm_rmse
                || sha256_file(&entry.tabpfn_v3_report_path)? != entry.tabpfn_v3_report_sha256
                || sha256_file(&entry.tabfm_report_path)? != entry.tabfm_report_sha256
                || sha256_file(&entry.tabfm_ensemble_report_path)?
                    != entry.tabfm_ensemble_report_sha256
                || baseline_metric(
                    &entry.tabpfn_v3_report_path,
                    &entry.dataset_hash,
                    "tabpfn_v3_regressor",
                )? != entry.tabpfn_v3_rmse
                || optional_baseline_metric(
                    &entry.tabfm_report_path,
                    &entry.dataset_hash,
                    "tabfm_regressor",
                )? != entry.tabfm_rmse
                || baseline_metric(
                    &entry.tabfm_ensemble_report_path,
                    &entry.dataset_hash,
                    "tabfm_ensemble_regressor",
                )? != entry.tabfm_ensemble_rmse
                || (
                    entry.stronger_tabfm_algorithm.as_str(),
                    entry.stronger_tabfm_rmse,
                ) != stronger_tabfm(entry.tabfm_rmse, entry.tabfm_ensemble_rmse)
            {
                bail!("pinned baseline content drift for {}", entry.dataset_hash);
            }
        }
        return Ok((manifest, digest));
    }

    let mut entries = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let dataset_hash = &dataset.dataset_hash;
        let tabpfn = &dataset.baseline_reports.tabpfn_v3_path;
        let tabfm = &dataset.baseline_reports.tabfm_path;
        let tabfm_ensemble = &dataset.baseline_reports.tabfm_ensemble_path;
        let prior_aspfm = tabpfn
            .parent()
            .context("TabPFN baseline report has no dataset directory")?
            .join("jope-dime-regressor.json");
        let tabfm_rmse = optional_baseline_metric(tabfm, dataset_hash, "tabfm_regressor")?;
        let tabfm_ensemble_rmse =
            baseline_metric(tabfm_ensemble, dataset_hash, "tabfm_ensemble_regressor")?;
        let (stronger_tabfm_algorithm, stronger_tabfm_rmse) =
            stronger_tabfm(tabfm_rmse, tabfm_ensemble_rmse);
        entries.push(BaselineEntry {
            dataset_hash: dataset_hash.clone(),
            prior_aspfm_report_sha256: sha256_file(&prior_aspfm)?,
            prior_aspfm_rmse: prior_aspfm_metric(&prior_aspfm, dataset_hash)?,
            prior_aspfm_report_path: absolute_path(&prior_aspfm)?,
            tabpfn_v3_report_sha256: sha256_file(tabpfn)?,
            tabpfn_v3_rmse: baseline_metric(tabpfn, dataset_hash, "tabpfn_v3_regressor")?,
            tabpfn_v3_report_path: absolute_path(tabpfn)?,
            tabfm_report_sha256: sha256_file(tabfm)?,
            tabfm_rmse,
            tabfm_report_path: absolute_path(tabfm)?,
            tabfm_ensemble_report_sha256: sha256_file(tabfm_ensemble)?,
            tabfm_ensemble_rmse,
            tabfm_ensemble_report_path: absolute_path(tabfm_ensemble)?,
            stronger_tabfm_algorithm: stronger_tabfm_algorithm.to_string(),
            stronger_tabfm_rmse,
        });
    }
    let manifest = BaselineManifest {
        schema: "TrainOnlyBaselineManifest.v2".to_string(),
        run_id: run_id.to_string(),
        entries,
    };
    let digest = write_receipt(&path, &manifest)?;
    Ok((manifest, digest))
}

pub(super) fn score_frozen_champion(
    member: &GenomeMember,
    train: &RegressionData,
    test: &RegressionData,
    features: usize,
    generation: usize,
) -> Result<GenerationTestScore> {
    if train.features != features
        || test.features != features
        || semantic_identity(&member.canonical_genome) != member.semantic_sha256
        || !genome_compatible(&member.canonical_genome, features)
    {
        bail!("frozen champion data/genome compatibility failure");
    }
    let genome = GenomeIr::parse(&member.canonical_genome)?;
    let started = Instant::now();
    let model = fit(&genome, &train.x, &train.y)?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let predictions = model.predict(&test.x)?;
    let predict_seconds = started.elapsed().as_secs_f64();
    let (rmse, r2, mae) = metrics(&test.y, &predictions)?;
    Ok(GenerationTestScore {
        generation,
        semantic_sha256: member.semantic_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        rmse,
        r2,
        mae,
        fit_seconds,
        predict_seconds,
    })
}

pub(super) fn load_or_create_final_cv_reranks(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    validation: &TrainOnlyValidationReceipt,
    validation_digest: &str,
    workers: usize,
) -> Result<BTreeMap<String, (TrainOnlyFinalCvRerank, String)>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("final-cv-{index}"))
        .build()
        .context("build final CV worker pool")?;
    let mut reranks = BTreeMap::new();
    for dataset in &manifest.datasets {
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
        {
            bail!("final CV selection is not covered by the validation receipt");
        }
        let final_generation = manifest.generations - 1;
        let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
            &checkpoint_path(root, &dataset.dataset_hash, final_generation),
        )?;
        if selection.generation_checkpoint_sha256[final_generation] != checkpoint_digest
            || selection.generation_champions[final_generation]
                != checkpoint.champion_semantic_sha256
            || checkpoint.ranked_semantic_sha256.len() < FINAL_CV_TOP_K
        {
            bail!("final CV source ranking drift for {}", dataset.dataset_hash);
        }
        let train = read_regression_csv(&dataset.train_path)?;
        let split_manifest = build_final_cv_splits(
            &manifest.run_id,
            &dataset.dataset_hash,
            &dataset.train_sha256,
            &train.y,
        )?;
        let evaluator = StudyEvalContext::new(&train, dataset, &split_manifest)?;
        evaluator.begin_generation(0);
        let source_candidates = checkpoint.ranked_semantic_sha256[..FINAL_CV_TOP_K]
            .iter()
            .enumerate()
            .map(|(original_rank, semantic)| {
                let scored = checkpoint
                    .population
                    .iter()
                    .find(|candidate| candidate.member.semantic_sha256 == *semantic)
                    .with_context(|| {
                        format!(
                            "final top-10 member {} absent for {}",
                            semantic, dataset.dataset_hash
                        )
                    })?;
                Ok((original_rank, scored.member.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidates = pool.install(|| {
            source_candidates
                .par_iter()
                .map(|(original_rank, member)| {
                    let evaluated =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                            if semantic_identity(&member.canonical_genome) != member.semantic_sha256
                                || !genome_compatible(&member.canonical_genome, dataset.features)
                            {
                                bail!("final CV genome compatibility failure");
                            }
                            let genome = GenomeIr::parse(&member.canonical_genome)?;
                            let mut fold_r2 = Vec::with_capacity(FINAL_CV_FOLDS);
                            let mut fold_rmse = Vec::with_capacity(FINAL_CV_FOLDS);
                            for split in &split_manifest.splits {
                                let (rmse, r2, _, _, _) = evaluator.evaluate(&genome, split)?;
                                fold_r2.push(r2);
                                fold_rmse.push(rmse);
                            }
                            Ok((fold_r2, fold_rmse))
                        }));
                    let (fold_r2, fold_rmse, robust_fitness, worst_fold_r2, failure_reason) =
                        match evaluated {
                            Ok(Ok((fold_r2, fold_rmse))) => {
                                let fitness = paired_robust_fitness(&fold_r2)?;
                                let worst = fold_r2
                                    .iter()
                                    .copied()
                                    .min_by(f64::total_cmp)
                                    .context("final CV score list is empty")?;
                                (fold_r2, fold_rmse, Some(fitness), Some(worst), None)
                            }
                            Ok(Err(error)) => (
                                Vec::new(),
                                Vec::new(),
                                None,
                                None,
                                Some(format!("{error:#}")),
                            ),
                            Err(payload) => (
                                Vec::new(),
                                Vec::new(),
                                None,
                                None,
                                Some(
                                    payload
                                        .downcast_ref::<&str>()
                                        .map(|value| format!("panic: {value}"))
                                        .or_else(|| {
                                            payload
                                                .downcast_ref::<String>()
                                                .map(|value| format!("panic: {value}"))
                                        })
                                        .unwrap_or_else(|| "panic: non-string payload".to_string()),
                                ),
                            ),
                        };
                    Ok(FinalCvCandidate {
                        original_rank: *original_rank,
                        semantic_sha256: member.semantic_sha256.clone(),
                        canonical_genome: member.canonical_genome.clone(),
                        complexity: member.complexity,
                        fold_r2,
                        fold_rmse,
                        robust_fitness,
                        worst_fold_r2,
                        failure_reason,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        evaluator.sweep();
        candidates.sort_by(rank_final_cv);
        let selected = candidates
            .first()
            .filter(|candidate| candidate.robust_fitness.is_some())
            .context("none of the final top-10 candidates completed CV")?;
        let selected_semantic_sha256 = selected.semantic_sha256.clone();
        let selected_canonical_genome = selected.canonical_genome.clone();
        let selected_robust_fitness = selected.robust_fitness.unwrap();
        let selected_worst_fold_r2 = selected.worst_fold_r2.unwrap();
        let selected_original_rank = selected.original_rank;
        let receipt = TrainOnlyFinalCvRerank {
            schema: "TrainOnlyFinalCvRerank.v2".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_receipt_sha256: validation_digest.to_string(),
            selection_complete_sha256: selection_digest,
            final_generation_checkpoint_sha256: checkpoint_digest,
            train_sha256: dataset.train_sha256.clone(),
            kernel_version: KERNEL_VERSION.to_string(),
            top_k: FINAL_CV_TOP_K,
            folds: FINAL_CV_FOLDS,
            split_manifest,
            candidates,
            selected_semantic_sha256,
            selected_canonical_genome,
            selected_robust_fitness,
            selected_worst_fold_r2,
            selected_original_rank,
            selection_frozen_before_test_access: true,
        };
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("final-top10-cv-rerank.v2.json");
        let digest = write_receipt(&path, &receipt)?;
        reranks.insert(dataset.dataset_hash.clone(), (receipt, digest));
    }
    Ok(reranks)
}
