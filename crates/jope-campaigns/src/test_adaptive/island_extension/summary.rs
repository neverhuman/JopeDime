use super::*;

pub(super) fn artifact_name(path: &Path) -> String {
    path.strip_prefix(campaign_state_root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

pub(super) fn benchmark_status(rmse: f64, leader_rmse: f64) -> String {
    if rmse < leader_rmse {
        "surpassed_external_benchmark".to_string()
    } else {
        "did_not_surpass_external_benchmark".to_string()
    }
}

pub(super) fn make_best_solutions(
    manifest: &IslandManifest,
    specs: &[DatasetSpec],
    island_results: &[IslandDatasetResult],
) -> Result<Golden42BestSolutions> {
    let base_root = run_root(&manifest.base_run_id);
    let base_summary: PanelSummary =
        serde_json::from_slice(&fs::read(base_root.join(super::SUMMARY_FILE))?)?;
    let base_map: WinnerMap = serde_json::from_slice(&fs::read(base_root.join(WINNER_MAP_FILE))?)?;
    let prior_root = extension_root(&manifest.base_run_id);
    let prior_summary: ExtensionSummary =
        serde_json::from_slice(&fs::read(prior_root.join(EXTENSION_SUMMARY_FILE))?)?;
    let specs = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let base_genomes = base_map
        .datasets
        .iter()
        .map(|entry| (entry.dataset_hash.as_str(), &entry.final_selected_genome))
        .collect::<HashMap<_, _>>();
    let extension_results = prior_summary
        .results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let island_results = island_results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let mut solutions = Vec::with_capacity(42);
    for dataset in &base_summary.datasets {
        let hash = dataset.dataset_hash.as_str();
        let spec = specs
            .get(hash)
            .context("best-solutions dataset is absent from Golden42")?;
        if dataset.outcome == "early_win" {
            let genome = base_genomes
                .get(hash)
                .context("base winner is absent from winner map")?;
            let source = base_root
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            let result: DatasetResult = serde_json::from_slice(&fs::read(&source)?)?;
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.base_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: None,
                slot: genome.slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        } else if let Some(result) = extension_results
            .get(hash)
            .filter(|result| result.outcome == "new_leader_found")
        {
            let genome = &result.final_selected_genome;
            let source = prior_root
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.prior_extension_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: None,
                slot: genome.slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        } else {
            let result = island_results
                .get(hash)
                .context("unresolved dataset is absent from island results")?;
            let genome = &result.generation_29_selected_genome;
            let source = island_root(&manifest.base_run_id)
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.island_extension_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: Some(genome.island),
                slot: genome.global_slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        }
    }
    if solutions.len() != 42
        || solutions
            .iter()
            .zip(&base_summary.datasets)
            .any(|(solution, summary)| solution.dataset_hash != summary.dataset_hash)
        || solutions
            .iter()
            .take(35)
            .filter(|solution| solution.benchmark_status == "surpassed_external_benchmark")
            .count()
            > 35
    {
        bail!("Golden42 best-solutions map ordering or membership mismatch");
    }
    let prior_winner_count = solutions
        .iter()
        .filter(|solution| solution.source_campaign != manifest.island_extension_run_id)
        .count();
    if prior_winner_count != EXISTING_SURPASSED {
        bail!("Golden42 best-solutions map does not preserve 35 prior winners");
    }
    Ok(Golden42BestSolutions {
        schema: BEST_SOLUTIONS_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: manifest.island_extension_run_id.clone(),
        ordered_dataset_ids: base_summary
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_hash.clone())
            .collect(),
        ordered_cohort_sha256: base_summary.ordered_cohort_sha256,
        dataset_count: solutions.len(),
        prior_winner_count,
        island_selected_count: solutions.len() - prior_winner_count,
        solutions,
    })
}

pub(super) fn make_summary(
    manifest: &IslandManifest,
    results: Vec<IslandDatasetResult>,
    root: &Path,
    best_solutions_sha256: String,
) -> Result<IslandSummary> {
    let new_leaders = results
        .iter()
        .filter(|result| result.outcome == "new_leader_found")
        .count();
    Ok(IslandSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        prior_extension_run_id: manifest.prior_extension_run_id.clone(),
        island_extension_run_id: manifest.island_extension_run_id.clone(),
        base_panel_manifest_sha256: manifest.base_panel_manifest_sha256.clone(),
        base_panel_summary_sha256: manifest.base_panel_summary_sha256.clone(),
        base_winner_map_sha256: manifest.base_winner_map_sha256.clone(),
        prior_extension_manifest_sha256: manifest.prior_extension_manifest_sha256.clone(),
        prior_extension_summary_sha256: manifest.prior_extension_summary_sha256.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        ordered_dataset_ids: manifest.ordered_dataset_ids.clone(),
        ordered_cohort_sha256: manifest.ordered_cohort_sha256.clone(),
        dataset_count: results.len(),
        existing_surpassed_count: EXISTING_SURPASSED,
        new_leaders,
        still_no_leader: results.len() - new_leaders,
        cumulative_surpassed_count: EXISTING_SURPASSED + new_leaders,
        total_qualifying_genomes: results
            .iter()
            .map(|result| result.qualifying_descendants.len())
            .sum(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        best_solutions_file: BEST_SOLUTIONS_FILE.to_string(),
        best_solutions_sha256,
        results,
        artifacts: collect_artifacts_excluding(root, SUMMARY_FILE)?,
    })
}
