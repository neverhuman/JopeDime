use super::*;

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
