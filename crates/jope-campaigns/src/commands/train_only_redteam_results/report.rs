use super::*;

fn score_for<'a>(receipt: &'a DatasetScoreReceipt, variant: &str) -> Result<&'a FrozenTestScore> {
    receipt
        .variants
        .iter()
        .find(|score| score.variant == variant)
        .with_context(|| format!("score receipt lacks variant {variant}"))
}

pub(super) fn median(mut values: Vec<f64>) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("median needs finite values");
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

pub(crate) fn report_run(
    study_root: &Path,
    run_id: &str,
    _sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    let (root, manifest, _, _, validation_digest) =
        super::validation::require_global_validation(study_root, run_id)?;
    let (index, index_digest): (ScoreIndex, _) = read_receipt(&root.join("scores.v1.json"))?;
    if index.schema != "TrainOnlyRedteamScoreIndex.v1"
        || index.run_id != run_id
        || index.validation_sha256 != validation_digest
        || index.dataset_score_sha256.len() != manifest.datasets.len()
        || !index.development_only
        || index.confirmatory_valid
        || index.test_reuse_disclosure != DISCLOSURE
    {
        bail!("score index drift");
    }
    let mut scores = Vec::new();
    for dataset in &manifest.datasets {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("scores.v1.json");
        let (receipt, digest): (DatasetScoreReceipt, _) = read_receipt(&path)?;
        if index.dataset_score_sha256.get(&dataset.dataset_hash) != Some(&digest)
            || receipt.schema != "TrainOnlyRedteamDatasetScore.v1"
            || receipt.run_id != run_id
            || receipt.dataset_hash != dataset.dataset_hash
            || receipt.validation_sha256 != validation_digest
            || receipt.baseline_manifest_sha256 != index.baseline_manifest_sha256
            || receipt.oracle_index_sha256 != index.oracle_index_sha256
            || receipt.variants.len() != 7
            || !receipt.selection_frozen_before_test_access
            || !receipt.development_only
            || receipt.confirmatory_valid
            || receipt.test_reuse_disclosure != DISCLOSURE
        {
            bail!("dataset score receipt drift");
        }
        scores.push(receipt);
    }

    let mut fair_scores = BTreeMap::new();
    let mut oracle_scores = BTreeMap::new();
    let mut wins = WinCounts {
        tabpfn_v3: 0,
        stronger_tabfm: 0,
        both_tabpfn_and_stronger_tabfm: 0,
        prior_train_only: 0,
        generation_9_control: 0,
        oracle_aspfm_envelope: 0,
    };
    let mut datasets = Vec::new();
    let mut final_values = Vec::new();
    let mut gen9_values = Vec::new();
    for receipt in &scores {
        let final_score = score_for(receipt, FINAL_VARIANT)?;
        let gen9 = score_for(receipt, GEN9_VARIANT)?;
        wins.tabpfn_v3 += usize::from(final_score.rmse < receipt.baselines.tabpfn_v3_rmse);
        wins.stronger_tabfm +=
            usize::from(final_score.rmse < receipt.baselines.stronger_tabfm_rmse);
        wins.both_tabpfn_and_stronger_tabfm += usize::from(
            final_score.rmse < receipt.baselines.tabpfn_v3_rmse
                && final_score.rmse < receipt.baselines.stronger_tabfm_rmse,
        );
        wins.prior_train_only +=
            usize::from(final_score.rmse < receipt.baselines.prior_train_only_rmse);
        wins.generation_9_control += usize::from(final_score.rmse < gen9.rmse);
        wins.oracle_aspfm_envelope +=
            usize::from(final_score.rmse < receipt.baselines.oracle_aspfm_rmse);
        let fair = BTreeMap::from([
            (FINAL_METHOD_ID.to_string(), final_score.rmse),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .prior_train_only
                    .method_id
                    .clone(),
                receipt.baselines.prior_train_only_rmse,
            ),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .tabpfn_v3
                    .method_id
                    .clone(),
                receipt.baselines.tabpfn_v3_rmse,
            ),
            (
                manifest
                    .config
                    .authoritative_baselines
                    .tabfm
                    .method_id
                    .clone(),
                receipt.baselines.stronger_tabfm_rmse,
            ),
        ]);
        let mut oracle = fair.clone();
        oracle.insert(
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .method_id
                .clone(),
            receipt.baselines.oracle_aspfm_rmse,
        );
        fair_scores.insert(receipt.dataset_hash.clone(), fair);
        oracle_scores.insert(receipt.dataset_hash.clone(), oracle);
        final_values.push(final_score.rmse);
        gen9_values.push(gen9.rmse);
        let (selector, _): (execution::SelectorComplete, _) =
            read_receipt(&selector_path(&root, &receipt.dataset_hash))?;
        datasets.push(ReportDataset {
            dataset_hash: receipt.dataset_hash.clone(),
            final_rmse: final_score.rmse,
            generation_9_control_rmse: gen9.rmse,
            prior_train_only_rmse: receipt.baselines.prior_train_only_rmse,
            tabpfn_v3_rmse: receipt.baselines.tabpfn_v3_rmse,
            stronger_tabfm_rmse: receipt.baselines.stronger_tabfm_rmse,
            oracle_aspfm_rmse: receipt.baselines.oracle_aspfm_rmse,
            final_variant: selector.final_variant.name,
            ensemble_accepted: selector.ensemble_accepted,
            development_only: true,
            confirmatory_valid: false,
        });
    }
    let fair_ladder = bradley_terry_elo(&fair_scores, &manifest.config.scoring)?;
    let oracle_reference_ladder = bradley_terry_elo(&oracle_scores, &manifest.config.scoring)?;
    let median_final_rmse = median(final_values)?;
    let median_generation_9_control_rmse = median(gen9_values)?;
    let development_strategy_improved = wins.both_tabpfn_and_stronger_tabfm
        >= manifest.config.scoring.success_wins_required
        && fair_ladder
            .first()
            .is_some_and(|rating| rating.method_id == FINAL_METHOD_ID)
        && median_final_rmse < median_generation_9_control_rmse;
    let labels = BTreeMap::from([
        (
            FINAL_METHOD_ID.to_string(),
            "train-only red-team final method".to_string(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .prior_train_only
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .prior_train_only
                .label
                .clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .tabpfn_v3
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .tabpfn_v3
                .label
                .clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .tabfm
                .method_id
                .clone(),
            manifest.config.authoritative_baselines.tabfm.label.clone(),
        ),
        (
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .method_id
                .clone(),
            manifest
                .config
                .authoritative_baselines
                .oracle_aspfm
                .label
                .clone(),
        ),
    ]);
    let report = StudyReport {
        schema: "TrainOnlyRedteamReport.v1".to_string(),
        run_id: run_id.to_string(),
        validation_sha256: validation_digest,
        score_index_sha256: index_digest,
        fair_ladder,
        oracle_reference_ladder,
        wins,
        median_final_rmse,
        median_generation_9_control_rmse,
        development_strategy_improved,
        oracle_comparison_determines_success: false,
        authoritative_method_labels: labels,
        datasets,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let json_path = root.join("report.v1.json");
    write_receipt(&json_path, &report)?;

    let mut markdown = format!(
        "# Train-only red-team development study\n\n\
         `development_only=true`; `confirmatory_valid=false`.\n\n\
         {}\n\n\
         Development strategy improved: `{}`. The oracle-reference comparison is descriptive and does not determine success.\n\n\
         Strict wins — TabPFN v3: {}/10; stronger per-dataset TabFM: {}/10; both on the same dataset: {}/10; prior train-only Jope Dime Regressor: {}/10; current Gen9 control: {}/10; ASPFM oracle envelope: {}/10.\n\n\
         Median final RMSE: `{:.12}`; median current Gen9 control RMSE: `{:.12}`.\n\n\
         ## Fair Bradley–Terry Elo ladder\n\n\
         | Rank | Method | Elo |\n|---:|---|---:|\n",
        DISCLOSURE,
        report.development_strategy_improved,
        report.wins.tabpfn_v3,
        report.wins.stronger_tabfm,
        report.wins.both_tabpfn_and_stronger_tabfm,
        report.wins.prior_train_only,
        report.wins.generation_9_control,
        report.wins.oracle_aspfm_envelope,
        report.median_final_rmse,
        report.median_generation_9_control_rmse,
    );
    for (rank, rating) in report.fair_ladder.iter().enumerate() {
        markdown.push_str(&format!(
            "| {} | {} | {:.6} |\n",
            rank + 1,
            report.authoritative_method_labels[&rating.method_id],
            rating.rating
        ));
    }
    markdown.push_str(
        "\n## Oracle-reference Bradley–Terry Elo ladder\n\n\
         The `jope-dime` row below is the best-test ASPFM envelope over 63,108 Golden42 candidate evaluations; it is not the prior train-only Jope Dime Regressor and cannot determine study success.\n\n\
         | Rank | Method | Elo |\n|---:|---|---:|\n",
    );
    for (rank, rating) in report.oracle_reference_ladder.iter().enumerate() {
        markdown.push_str(&format!(
            "| {} | {} | {:.6} |\n",
            rank + 1,
            report.authoritative_method_labels[&rating.method_id],
            rating.rating
        ));
    }
    markdown.push_str(
        "\n## Per-dataset results\n\n\
         Every row is `development_only=true`, `confirmatory_valid=false`.\n\n\
         | Dataset | Final | Gen9 control | Prior train-only | TabPFN v3 | Stronger TabFM | ASPFM oracle | Final selector |\n\
         |---|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for dataset in &report.datasets {
        markdown.push_str(&format!(
            "| {} | {:.12} | {:.12} | {:.12} | {:.12} | {:.12} | {:.12} | {} |\n",
            dataset.dataset_hash,
            dataset.final_rmse,
            dataset.generation_9_control_rmse,
            dataset.prior_train_only_rmse,
            dataset.tabpfn_v3_rmse,
            dataset.stronger_tabfm_rmse,
            dataset.oracle_aspfm_rmse,
            dataset.final_variant,
        ));
    }
    let markdown_path = root.join("report.v1.md");
    immutable_write(&markdown_path, markdown.as_bytes())?;
    Ok(markdown_path)
}
