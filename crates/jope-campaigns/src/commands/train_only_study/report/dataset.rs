use super::*;

#[allow(private_interfaces)]
pub(crate) fn report_dataset(
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
