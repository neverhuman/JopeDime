use super::*;

pub(super) fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("median requires finite values");
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

pub(super) fn bootstrap_interval(values: &[f64]) -> Result<[f64; 2]> {
    if values.len() != 10 {
        bail!("study bootstrap is pinned to ten datasets");
    }
    let mut rng = DeterministicRng::new(STUDY_SEED ^ 0x6273_7472_6170);
    let mut estimates = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let sample = (0..values.len())
            .map(|_| values[(rng.next_u64() as usize) % values.len()])
            .collect::<Vec<_>>();
        estimates.push(median(&sample)?);
    }
    estimates.sort_by(f64::total_cmp);
    Ok([estimates[249], estimates[9749]])
}

pub(super) fn paired_sign_test(values: &[f64]) -> f64 {
    let positive = values.iter().filter(|value| **value > 0.0).count();
    let negative = values.iter().filter(|value| **value < 0.0).count();
    let n = positive + negative;
    if n == 0 {
        return 1.0;
    }
    let tail = positive.min(negative);
    let mut coefficient = 1.0;
    let mut probability = 1.0;
    for k in 1..=tail {
        coefficient *= (n + 1 - k) as f64 / k as f64;
        probability += coefficient;
    }
    (2.0 * probability / 2_f64.powi(n as i32)).min(1.0)
}

pub(super) fn cohort_elo(datasets: &[DatasetReport]) -> Result<Vec<EloRating>> {
    if datasets.is_empty() {
        bail!("Elo requires at least one dataset");
    }
    const SYSTEMS: [&str; 5] = [
        "ours_gen9",
        "ours_top10_5fold_cv",
        "prior_aspfm",
        "tabfm_ensemble",
        "tabpfn_v3",
    ];
    const INITIAL: f64 = 1500.0;
    const K: f64 = 32.0;
    let mut ratings = [INITIAL; SYSTEMS.len()];
    let mut wins = [0usize; SYSTEMS.len()];
    let mut losses = [0usize; SYSTEMS.len()];
    let mut ties = [0usize; SYSTEMS.len()];
    for dataset in datasets {
        let rmse = [
            dataset.generation_nine_rmse,
            dataset.cv_rerank_rmse,
            dataset.prior_aspfm_rmse,
            dataset.tabfm_ensemble_rmse,
            dataset.tabpfn_v3_rmse,
        ];
        if rmse.iter().any(|value| !value.is_finite() || *value < 0.0) {
            bail!("Elo received an invalid RMSE");
        }
        let before = ratings;
        let mut delta = [0.0; SYSTEMS.len()];
        for left in 0..SYSTEMS.len() {
            for right in (left + 1)..SYSTEMS.len() {
                let outcome = match rmse[left].total_cmp(&rmse[right]) {
                    Ordering::Less => {
                        wins[left] += 1;
                        losses[right] += 1;
                        1.0
                    }
                    Ordering::Greater => {
                        losses[left] += 1;
                        wins[right] += 1;
                        0.0
                    }
                    Ordering::Equal => {
                        ties[left] += 1;
                        ties[right] += 1;
                        0.5
                    }
                };
                let expected = 1.0 / (1.0 + 10_f64.powf((before[right] - before[left]) / 400.0));
                let change = K * (outcome - expected);
                delta[left] += change;
                delta[right] -= change;
            }
        }
        for index in 0..SYSTEMS.len() {
            ratings[index] += delta[index];
        }
    }
    let mut result = SYSTEMS
        .iter()
        .enumerate()
        .map(|(index, system)| EloRating {
            system: (*system).to_string(),
            rating: ratings[index],
            wins: wins[index],
            losses: losses[index],
            ties: ties[index],
        })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| {
        right
            .rating
            .total_cmp(&left.rating)
            .then_with(|| left.system.cmp(&right.system))
    });
    Ok(result)
}

pub(super) fn panel_points(values: &[f64], left: f64, top: f64, width: f64, height: f64) -> String {
    panel_points_scaled(values, values, left, top, width, height)
}

pub(super) fn panel_points_scaled(
    values: &[f64],
    scale: &[f64],
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) -> String {
    let minimum = scale.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = scale.iter().copied().max_by(f64::total_cmp).unwrap();
    let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
    let low = minimum - padding;
    let high = maximum + padding;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = left + width * index as f64 / (values.len() - 1).max(1) as f64;
            let y = top + height * (high - value) / (high - low);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn evolution_svg(
    title: &str,
    validation: &[f64],
    test: &[f64],
    tabpfn: f64,
    tabfm: Option<f64>,
    tabfm_ensemble: f64,
) -> String {
    let mut rmse_scale = test.to_vec();
    rmse_scale.extend([tabpfn, tabfm_ensemble]);
    if let Some(tabfm) = tabfm {
        rmse_scale.push(tabfm);
    }
    let validation_points = panel_points(validation, 80.0, 70.0, 820.0, 210.0);
    let test_points = panel_points_scaled(test, &rmse_scale, 80.0, 370.0, 820.0, 210.0);
    let baseline_y = |value: f64| {
        let minimum = rmse_scale.iter().copied().min_by(f64::total_cmp).unwrap();
        let maximum = rmse_scale.iter().copied().max_by(f64::total_cmp).unwrap();
        let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
        370.0 + 210.0 * (maximum + padding - value) / (maximum + padding - (minimum - padding))
    };
    let tabfm_line = tabfm.map_or_else(
        || {
            "<text x=\"905\" y=\"350\" font-family=\"sans-serif\" font-size=\"12\">TabFM unavailable</text>"
                .to_string()
        },
        |value| {
            let y = baseline_y(value);
            format!("<line x1=\"80\" y1=\"{y:.2}\" x2=\"900\" y2=\"{y:.2}\" stroke=\"#d97706\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM</text>")
        },
    );
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"650\" viewBox=\"0 0 960 650\">\n\
<rect width=\"960\" height=\"650\" fill=\"white\"/>\n\
<text x=\"480\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"20\">{title}</text>\n\
<text x=\"20\" y=\"175\" transform=\"rotate(-90 20 175)\" font-family=\"sans-serif\" font-size=\"14\">10-split robust validation fitness</text>\n\
<line x1=\"80\" y1=\"280\" x2=\"900\" y2=\"280\" stroke=\"#999\"/><polyline points=\"{validation_points}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\"/>\n\
<text x=\"20\" y=\"490\" transform=\"rotate(-90 20 490)\" font-family=\"sans-serif\" font-size=\"14\">post-hoc test RMSE</text>\n\
<line x1=\"80\" y1=\"580\" x2=\"900\" y2=\"580\" stroke=\"#999\"/><polyline points=\"{test_points}\" fill=\"none\" stroke=\"#059669\" stroke-width=\"3\"/>\n\
<line x1=\"80\" y1=\"{pfn_y:.2}\" x2=\"900\" y2=\"{pfn_y:.2}\" stroke=\"#dc2626\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{pfn_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabPFN v3</text>\n\
{tabfm_line}\n\
<line x1=\"80\" y1=\"{fm_ensemble_y:.2}\" x2=\"900\" y2=\"{fm_ensemble_y:.2}\" stroke=\"#7c3aed\" stroke-dasharray=\"4 4\"/><text x=\"905\" y=\"{fm_ensemble_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM ensemble</text>\n\
<text x=\"480\" y=\"625\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">generation 0–9</text>\n</svg>\n",
        pfn_y = baseline_y(tabpfn),
        fm_ensemble_y = baseline_y(tabfm_ensemble),
    )
}

pub(super) fn report_dataset(
    root: &Path,
    dataset: &DatasetStudyEntry,
    posthoc: &TrainOnlyPosthocScore,
) -> Result<(DatasetReport, Vec<f64>)> {
    let (fixed, _): (TrainOnlyChampionValidation, _) =
        read_receipt(&champion_validation_path(root, &dataset.dataset_hash))?;
    let (selection, _): (TrainOnlySelectionComplete, _) =
        read_receipt(&selection_path(root, &dataset.dataset_hash))?;
    let final_checkpoint: TrainOnlyGenerationCheckpoint =
        read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
            root,
            &dataset.dataset_hash,
            posthoc.scores.len() - 1,
        ))?
        .0;
    let final_member = final_checkpoint
        .population
        .iter()
        .find(|member| member.member.semantic_sha256 == selection.final_champion_semantic_sha256)
        .context("final champion missing from generation-nine checkpoint")?;
    let ancestry = if final_member.member.ancestry.roots == BTreeSet::from(["dl".to_string()]) {
        "dl"
    } else if final_member.member.ancestry.roots == BTreeSet::from(["transfer".to_string()]) {
        "transfer"
    } else {
        "mixed"
    };
    let test_rmse = posthoc
        .scores
        .iter()
        .map(|score| score.rmse)
        .collect::<Vec<_>>();
    let validation = fixed
        .champions
        .iter()
        .map(|champion| champion.robust_fitness)
        .collect::<Vec<_>>();
    if test_rmse.is_empty() || test_rmse.len() != validation.len() {
        bail!("dataset report needs matching validation and test generations");
    }
    let final_generation = test_rmse.len() - 1;
    let relative = (test_rmse[0] - test_rmse[final_generation]) / test_rmse[0];
    if !relative.is_finite() {
        bail!("generation-zero relative improvement is undefined");
    }
    let mut best_baseline_algorithm = "prior_aspfm".to_string();
    let mut best_baseline_rmse = posthoc.prior_aspfm_rmse;
    if posthoc.tabpfn_v3_rmse < best_baseline_rmse {
        best_baseline_algorithm = "tabpfn_v3".to_string();
        best_baseline_rmse = posthoc.tabpfn_v3_rmse;
    }
    if posthoc.stronger_tabfm_rmse < best_baseline_rmse {
        best_baseline_algorithm = posthoc.stronger_tabfm_algorithm.clone();
        best_baseline_rmse = posthoc.stronger_tabfm_rmse;
    }
    let generation_nine_relative_gap_to_best_baseline =
        (test_rmse[final_generation] - best_baseline_rmse) / best_baseline_rmse;
    let cv_rerank_relative_gap_to_best_baseline =
        (posthoc.final_cv_rerank_score.rmse - best_baseline_rmse) / best_baseline_rmse;

    let report_root = root.join("reports").join(&dataset.dataset_hash);
    fs::create_dir_all(&report_root)?;
    immutable_write(
        &report_root.join("evolution.svg"),
        evolution_svg(
            &format!("{} checkpoint-assisted evolution", dataset.dataset_hash),
            &validation,
            &test_rmse,
            posthoc.tabpfn_v3_rmse,
            posthoc.tabfm_rmse,
            posthoc.tabfm_ensemble_rmse,
        )
        .as_bytes(),
    )?;

    let mut trends = String::from(
        "generation,validation_robust_fitness,test_rmse,unique_genomes,dl_rooted,transfer_rooted,mixed_rooted,invalid_rate,cache_hit_rate,fit_seconds,predict_seconds\n",
    );
    for generation in 0..test_rmse.len() {
        let checkpoint: TrainOnlyGenerationCheckpoint =
            read_receipt::<TrainOnlyGenerationCheckpoint>(&checkpoint_path(
                root,
                &dataset.dataset_hash,
                generation,
            ))?
            .0;
        let mut dl = 0;
        let mut transfer = 0;
        let mut mixed = 0;
        for member in &checkpoint.population {
            if member.member.ancestry.roots == BTreeSet::from(["dl".to_string()]) {
                dl += 1;
            } else if member.member.ancestry.roots == BTreeSet::from(["transfer".to_string()]) {
                transfer += 1;
            } else {
                mixed += 1;
            }
        }
        let total_observations = checkpoint.cache_hits + checkpoint.cache_misses;
        let (fit_seconds, predict_seconds) = checkpoint
            .population
            .iter()
            .flat_map(|member| &member.observations)
            .filter(|link| !link.cache_hit)
            .try_fold((0.0, 0.0), |(fit_sum, predict_sum), link| {
                let (observation, _): (TrainOnlyFitnessObservation, _) =
                    read_receipt(&link.receipt_path)?;
                Ok::<_, anyhow::Error>((
                    fit_sum + observation.fit_seconds,
                    predict_sum + observation.predict_seconds,
                ))
            })?;
        writeln!(
            trends,
            "{generation},{},{},{},{dl},{transfer},{mixed},{},{},{fit_seconds},{predict_seconds}",
            validation[generation],
            test_rmse[generation],
            checkpoint
                .population
                .iter()
                .map(|member| &member.member.semantic_sha256)
                .collect::<HashSet<_>>()
                .len(),
            checkpoint.invalid_members as f64 / POPULATION as f64,
            if total_observations == 0 {
                0.0
            } else {
                checkpoint.cache_hits as f64 / total_observations as f64
            },
        )?;
    }
    immutable_write(
        &report_root.join("generation-trends.csv"),
        trends.as_bytes(),
    )?;
    let tabfm_display = posthoc.tabfm_rmse.map_or_else(
        || "unavailable (source report failed)".to_string(),
        |value| format!("{value:.12}"),
    );
    let markdown = format!(
        "# Dataset {}\n\n\
Final ancestry: `{ancestry}`.\n\n\
Generation 0 test RMSE: `{:.12}`. Final-generation test RMSE: `{:.12}` ({:+.2}% relative improvement).\n\n\
Prior ASPFM RMSE: `{:.12}`. TabPFN v3 RMSE: `{:.12}`. TabFM RMSE: `{}`. TabFM ensemble RMSE: `{:.12}`. The stronger TabFM comparison is `{}` at `{:.12}`. Strict final wins: prior ASPFM `{}`, TabPFN `{}`, stronger TabFM `{}`, both fixed baselines `{}`.\n\n\
The independently frozen top-10/five-fold CV rerank selected `{}` and scored test RMSE `{:.12}`. Strict CV-rerank wins: prior ASPFM `{}`, TabPFN `{}`, stronger TabFM `{}`, both fixed baselines `{}`.\n\n\
This is an exploratory checkpoint-assisted transfer study. The checkpoint has seen Golden42 identities, so these results are not zero-shot generalization evidence.\n\n\
The entire test trajectory was generated post-hoc after selection and independent validation. The top-10 CV choice was also sealed before any test path or baseline metric was opened.\n",
        dataset.dataset_hash,
        test_rmse[0],
        test_rmse[final_generation],
        relative * 100.0,
        posthoc.prior_aspfm_rmse,
        posthoc.tabpfn_v3_rmse,
        tabfm_display,
        posthoc.tabfm_ensemble_rmse,
        posthoc.stronger_tabfm_algorithm,
        posthoc.stronger_tabfm_rmse,
        posthoc.beats_prior_aspfm,
        posthoc.beats_tabpfn_v3,
        posthoc.beats_stronger_tabfm,
        posthoc.beats_both,
        posthoc.final_cv_rerank_score.semantic_sha256,
        posthoc.final_cv_rerank_score.rmse,
        posthoc.cv_rerank_beats_prior_aspfm,
        posthoc.cv_rerank_beats_tabpfn_v3,
        posthoc.cv_rerank_beats_stronger_tabfm,
        posthoc.cv_rerank_beats_both,
    );
    immutable_write(&report_root.join("report.md"), markdown.as_bytes())?;
    Ok((
        DatasetReport {
            dataset_hash: dataset.dataset_hash.clone(),
            generation_zero_validation_fitness: validation[0],
            generation_nine_validation_fitness: validation[final_generation],
            validation_fitness_change: validation[final_generation] - validation[0],
            generation_zero_rmse: test_rmse[0],
            generation_nine_rmse: test_rmse[final_generation],
            relative_rmse_improvement: relative,
            prior_aspfm_rmse: posthoc.prior_aspfm_rmse,
            tabpfn_v3_rmse: posthoc.tabpfn_v3_rmse,
            tabfm_rmse: posthoc.tabfm_rmse,
            tabfm_ensemble_rmse: posthoc.tabfm_ensemble_rmse,
            stronger_tabfm_algorithm: posthoc.stronger_tabfm_algorithm.clone(),
            stronger_tabfm_rmse: posthoc.stronger_tabfm_rmse,
            best_baseline_algorithm,
            best_baseline_rmse,
            generation_nine_relative_gap_to_best_baseline,
            cv_rerank_relative_gap_to_best_baseline,
            beats_tabpfn_v3: posthoc.beats_tabpfn_v3,
            beats_stronger_tabfm: posthoc.beats_stronger_tabfm,
            beats_both: posthoc.beats_both,
            beats_prior_aspfm: posthoc.beats_prior_aspfm,
            cv_rerank_semantic_sha256: posthoc.final_cv_rerank_score.semantic_sha256.clone(),
            cv_rerank_rmse: posthoc.final_cv_rerank_score.rmse,
            cv_rerank_beats_tabpfn_v3: posthoc.cv_rerank_beats_tabpfn_v3,
            cv_rerank_beats_stronger_tabfm: posthoc.cv_rerank_beats_stronger_tabfm,
            cv_rerank_beats_both: posthoc.cv_rerank_beats_both,
            cv_rerank_beats_prior_aspfm: posthoc.cv_rerank_beats_prior_aspfm,
            final_ancestry: ancestry.to_string(),
        },
        validation,
    ))
}

pub(crate) fn report(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    let (root, manifest, _, _, validation_digest) =
        require_validated_score_gate(study_root, run_id, sotarena_root, workers)?;
    let (index, index_digest): (TrainOnlyPosthocIndex, _) =
        read_receipt(&root.join("posthoc-scores.v2.json"))?;
    let (baselines, baseline_digest): (BaselineManifest, _) =
        read_receipt(&root.join("baselines.v2.json"))?;
    if index.schema != "TrainOnlyPosthocIndex.v2"
        || index.run_id != run_id
        || index.validation_receipt_sha256 != validation_digest
        || index.baseline_manifest_sha256 != baseline_digest
        || index.dataset_ids != manifest.selected_dataset_ids
        || index.posthoc_score_sha256.len() != manifest.datasets.len()
        || baselines.entries.len() != manifest.datasets.len()
    {
        bail!("post-hoc index does not cover the validated cohort");
    }
    let expected_index_cumulative = receipt_digest(&json!({
        "validation_receipt_sha256": index.validation_receipt_sha256,
        "baseline_manifest_sha256": index.baseline_manifest_sha256,
        "posthoc_score_sha256": index.posthoc_score_sha256,
    }))?;
    if expected_index_cumulative != index.cumulative_sha256 {
        bail!("post-hoc index cumulative hash drift");
    }

    let mut dataset_reports = Vec::with_capacity(manifest.datasets.len());
    let mut validations = Vec::with_capacity(manifest.datasets.len());
    let mut tests = Vec::with_capacity(manifest.datasets.len());
    for dataset in &manifest.datasets {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("posthoc-score.v2.json");
        let (posthoc, digest): (TrainOnlyPosthocScore, _) = read_receipt(&path)?;
        let (cv_rerank, cv_rerank_digest): (TrainOnlyFinalCvRerank, _) = read_receipt(
            &root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("final-top10-cv-rerank.v2.json"),
        )?;
        if index.posthoc_score_sha256.get(&dataset.dataset_hash) != Some(&digest)
            || posthoc.schema != "TrainOnlyPosthocScore.v2"
            || posthoc.run_id != run_id
            || posthoc.dataset_hash != dataset.dataset_hash
            || posthoc.validation_receipt_sha256 != validation_digest
            || posthoc.baseline_manifest_sha256 != baseline_digest
            || posthoc.scores.len() != manifest.generations
            || posthoc.final_cv_rerank_sha256 != cv_rerank_digest
            || cv_rerank.schema != "TrainOnlyFinalCvRerank.v2"
            || cv_rerank.dataset_hash != dataset.dataset_hash
            || cv_rerank.validation_receipt_sha256 != validation_digest
            || cv_rerank.selected_semantic_sha256 != posthoc.final_cv_rerank_score.semantic_sha256
            || !cv_rerank.selection_frozen_before_test_access
            || !posthoc.test_trajectory_post_selection
        {
            bail!("post-hoc dataset receipt drift");
        }
        let (dataset_report, validation) = report_dataset(&root, dataset, &posthoc)?;
        validations.push(validation);
        tests.push(
            posthoc
                .scores
                .iter()
                .map(|score| score.rmse)
                .collect::<Vec<_>>(),
        );
        dataset_reports.push(dataset_report);
    }
    let relative = dataset_reports
        .iter()
        .map(|dataset| dataset.relative_rmse_improvement)
        .collect::<Vec<_>>();
    let wins_vs_tabpfn_v3 = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_tabpfn_v3)
        .count();
    let wins_vs_stronger_tabfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_stronger_tabfm)
        .count();
    let wins_vs_both = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_both)
        .count();
    let wins_vs_prior_aspfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.beats_prior_aspfm)
        .count();
    let cv_rerank_wins_vs_tabpfn_v3 = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_tabpfn_v3)
        .count();
    let cv_rerank_wins_vs_stronger_tabfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_stronger_tabfm)
        .count();
    let cv_rerank_wins_vs_both = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_both)
        .count();
    let cv_rerank_wins_vs_prior_aspfm = dataset_reports
        .iter()
        .filter(|dataset| dataset.cv_rerank_beats_prior_aspfm)
        .count();
    let status = if wins_vs_both >= 5 {
        "checkpoint_assisted_claim_supported"
    } else {
        "checkpoint_assisted_claim_not_supported"
    };
    let elo_ratings = cohort_elo(&dataset_reports)?;
    let receipt = TrainOnlyStudyReport {
        schema: "TrainOnlyStudyReport.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure: manifest.checkpoint_overlap_disclosure.clone(),
        run_id: run_id.to_string(),
        validation_receipt_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        posthoc_score_sha256: index.posthoc_score_sha256.clone(),
        status: status.to_string(),
        wins_vs_tabpfn_v3,
        wins_vs_stronger_tabfm,
        wins_vs_both,
        wins_vs_prior_aspfm,
        cv_rerank_wins_vs_tabpfn_v3,
        cv_rerank_wins_vs_stronger_tabfm,
        cv_rerank_wins_vs_both,
        cv_rerank_wins_vs_prior_aspfm,
        median_relative_rmse_improvement: median(&relative)?,
        bootstrap_interval_95: bootstrap_interval(&relative)?,
        paired_sign_test_p_value: paired_sign_test(&relative),
        elo_methodology: "Five systems start at 1500 with K=32. Each frozen dataset is one simultaneous round-robin round; lower RMSE wins, exact equality ties, and all pair deltas use pre-round ratings. Dataset order is the frozen manifest order. The comparison set is our generation-9 champion, our train-only top-10/five-fold-CV choice, prior ASPFM (`jope-dime`), TabFM ensemble, and TabPFN v3.".to_string(),
        elo_ratings,
        test_trajectories_post_selection: true,
        datasets: dataset_reports,
    };
    let report_root = root.join("reports");
    fs::create_dir_all(&report_root)?;
    let report_path = report_root.join("study-report.v2.json");
    let report_digest = write_receipt(&report_path, &receipt)?;

    let cohort_validation = (0..manifest.generations)
        .map(|generation| {
            median(
                &validations
                    .iter()
                    .map(|values| values[generation])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let cohort_test = (0..manifest.generations)
        .map(|generation| {
            median(
                &tests
                    .iter()
                    .map(|values| values[generation])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let median_tabpfn = median(
        &receipt
            .datasets
            .iter()
            .map(|dataset| dataset.tabpfn_v3_rmse)
            .collect::<Vec<_>>(),
    )?;
    let median_tabfm = median(
        &receipt
            .datasets
            .iter()
            .filter_map(|dataset| dataset.tabfm_rmse)
            .collect::<Vec<_>>(),
    )?;
    let median_tabfm_ensemble = median(
        &receipt
            .datasets
            .iter()
            .map(|dataset| dataset.tabfm_ensemble_rmse)
            .collect::<Vec<_>>(),
    )?;
    immutable_write(
        &report_root.join("cohort-evolution.svg"),
        evolution_svg(
            "Golden42 checkpoint-assisted cohort medians",
            &cohort_validation,
            &cohort_test,
            median_tabpfn,
            Some(median_tabfm),
            median_tabfm_ensemble,
        )
        .as_bytes(),
    )?;
    let mut table = String::from(
        "| Dataset | Validation fitness 0→9 | Gen 0 RMSE | Gen 9 RMSE | Top-10 CV RMSE | Best baseline | Gen 9 gap | CV gap | Gen 9 win | CV win | Ancestry |\n|---|---:|---:|---:|---:|---|---:|---:|:---:|:---:|---|\n",
    );
    let mut baseline_table = String::from(
        "| Dataset | Prior ASPFM | TabPFN v3 | TabFM | TabFM ensemble | Stronger TabFM |\n|---|---:|---:|---:|---:|---|\n",
    );
    for dataset in &receipt.datasets {
        let tabfm_display = dataset
            .tabfm_rmse
            .map_or_else(|| "unavailable".to_string(), |value| format!("{value:.8}"));
        writeln!(
            table,
            "| {} | {:.4}→{:.4} ({:+.4}) | {:.8} | {:.8} | {:.8} | {} ({:.8}) | {:+.2}% | {:+.2}% | {} | {} | {} |",
            dataset.dataset_hash,
            dataset.generation_zero_validation_fitness,
            dataset.generation_nine_validation_fitness,
            dataset.validation_fitness_change,
            dataset.generation_zero_rmse,
            dataset.generation_nine_rmse,
            dataset.cv_rerank_rmse,
            dataset.best_baseline_algorithm,
            dataset.best_baseline_rmse,
            dataset.generation_nine_relative_gap_to_best_baseline * 100.0,
            dataset.cv_rerank_relative_gap_to_best_baseline * 100.0,
            dataset.beats_both,
            dataset.cv_rerank_beats_both,
            dataset.final_ancestry,
        )?;
        writeln!(
            baseline_table,
            "| {} | {:.8} | {:.8} | {} | {:.8} | {} ({:.8}) |",
            dataset.dataset_hash,
            dataset.prior_aspfm_rmse,
            dataset.tabpfn_v3_rmse,
            tabfm_display,
            dataset.tabfm_ensemble_rmse,
            dataset.stronger_tabfm_algorithm,
            dataset.stronger_tabfm_rmse,
        )?;
    }
    let mut elo_table =
        String::from("| Rank | System | Elo | W | L | T |\n|---:|---|---:|---:|---:|---:|\n");
    for (rank, rating) in receipt.elo_ratings.iter().enumerate() {
        writeln!(
            elo_table,
            "| {} | {} | {:.1} | {} | {} | {} |",
            rank + 1,
            rating.system,
            rating.rating,
            rating.wins,
            rating.losses,
            rating.ties,
        )?;
    }
    let elo_leader = receipt.elo_ratings.first().context("Elo table is empty")?;
    let gen9_elo_rank = receipt
        .elo_ratings
        .iter()
        .position(|rating| rating.system == "ours_gen9")
        .context("generation-nine system is absent from Elo")?
        + 1;
    let cv_elo_rank = receipt
        .elo_ratings
        .iter()
        .position(|rating| rating.system == "ours_top10_5fold_cv")
        .context("CV-rerank system is absent from Elo")?
        + 1;
    let markdown = format!(
        "# Checkpoint-Assisted 10-Dataset Regression Evolution Study\n\n\
Status: **{}**.\n\n\
The generation-9 champions won strictly against prior ASPFM on {}/10 datasets, TabPFN v3 on {}/10, the per-dataset stronger TabFM variant on {}/10, and both fixed baselines on {}/10. The fixed success threshold was at least 5/10 wins against both fixed baselines.\n\n\
The secondary final-top-10/five-fold-CV strategy won strictly against prior ASPFM on {}/10 datasets, TabPFN v3 on {}/10, the stronger TabFM variant on {}/10, and both fixed baselines on {}/10. This secondary result does not replace the fixed primary criterion.\n\n\
Median generation-0-to-9 relative RMSE improvement: {:+.2}% (deterministic 95% bootstrap interval {:+.2}% to {:+.2}%; paired sign-test p={:.6}).\n\n\
This experiment is exploratory: the accepted checkpoint has seen Golden42 identities. `zero_shot_valid` is `false`, and this report makes no zero-shot generalization claim.\n\n\
All selection receipts were sealed and independently validated before any test path was opened or baseline metric was used. The top-10 CV rerank was then computed from training data and sealed before those paths or metrics were opened. Test trajectories are post-hoc and never influenced either selection strategy.\n\n\
## Dataset outcomes\n\n\
{}\n\n\
Negative gaps mean our RMSE is lower than the best baseline for that dataset.\n\n\
## Baseline detail\n\n\
{}\n\n\
## Cohort Elo\n\n\
{}\n\n\
The descriptive five-system Elo leader is `{}` at `{:.1}`. Our generation-9 system ranks `{}` and our top-10/five-fold-CV system ranks `{}` on this frozen ten-dataset cohort. This is a within-cohort pace check; its numeric ratings are not directly comparable to SOTArena's official 42-dataset Elo scale.\n\n\
{}\n\n\
Report receipt: `{}`. Post-hoc index receipt: `{}`.\n",
        receipt.status,
        receipt.wins_vs_prior_aspfm,
        receipt.wins_vs_tabpfn_v3,
        receipt.wins_vs_stronger_tabfm,
        receipt.wins_vs_both,
        receipt.cv_rerank_wins_vs_prior_aspfm,
        receipt.cv_rerank_wins_vs_tabpfn_v3,
        receipt.cv_rerank_wins_vs_stronger_tabfm,
        receipt.cv_rerank_wins_vs_both,
        receipt.median_relative_rmse_improvement * 100.0,
        receipt.bootstrap_interval_95[0] * 100.0,
        receipt.bootstrap_interval_95[1] * 100.0,
        receipt.paired_sign_test_p_value,
        table,
        baseline_table,
        elo_table,
        elo_leader.system,
        elo_leader.rating,
        gen9_elo_rank,
        cv_elo_rank,
        receipt.elo_methodology,
        report_digest,
        index_digest,
    );
    immutable_write(&report_root.join("study-report.md"), markdown.as_bytes())?;
    Ok(report_path)
}
