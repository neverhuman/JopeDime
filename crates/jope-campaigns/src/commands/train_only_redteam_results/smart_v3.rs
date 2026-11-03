// jankurai:allow repo-rot.path.fake-versioned-source reason=active-v3-redteam-results-module-is-coverage-inventoried owner=jope-campaigns expires=2027-04-01
use super::*;
use super::{
    report::median,
    scoring::{finite_rmse, predict_variant, sum_squared_error},
};

fn smart_comparison_baselines(
    entry: &Value,
    dataset_hash: &str,
) -> Result<SmartComparisonBaselines> {
    if entry.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash) {
        bail!("baseline dataset identity drift");
    }
    let stronger_tabfm_method_id = value_string(entry, "/stronger_tabfm_algorithm")?;
    let stronger_source = match stronger_tabfm_method_id.as_str() {
        "tabfm_regressor" => ("/tabfm_report_path", "/tabfm_report_sha256"),
        "tabfm_ensemble_regressor" => (
            "/tabfm_ensemble_report_path",
            "/tabfm_ensemble_report_sha256",
        ),
        _ => bail!("stronger TabFM identity drift"),
    };
    for (path_pointer, hash_pointer) in [
        ("/prior_aspfm_report_path", "/prior_aspfm_report_sha256"),
        ("/tabpfn_v3_report_path", "/tabpfn_v3_report_sha256"),
        stronger_source,
    ] {
        let path =
            execution::relocated_input_path(&PathBuf::from(value_string(entry, path_pointer)?));
        let expected = value_string(entry, hash_pointer)?;
        if sha256_file(&path)? != expected {
            bail!("authoritative baseline source drift");
        }
    }
    Ok(SmartComparisonBaselines {
        prior_train_only_rmse: finite_rmse(entry, "/prior_aspfm_rmse")?,
        tabpfn_v3_rmse: finite_rmse(entry, "/tabpfn_v3_rmse")?,
        stronger_tabfm_method_id,
        stronger_tabfm_rmse: finite_rmse(entry, "/stronger_tabfm_rmse")?,
    })
}

pub(crate) fn regression_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
    let squared_error = sum_squared_error(truth, predictions)?;
    let mae = truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).abs())
        .sum::<f64>()
        / truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let total_squared_error = truth
        .iter()
        .map(|truth| (truth - mean).powi(2))
        .sum::<f64>();
    if total_squared_error <= 0.0 {
        bail!("R2 requires a nonconstant test target");
    }
    Ok((
        (squared_error / truth.len() as f64).sqrt(),
        mae,
        1.0 - squared_error / total_squared_error,
    ))
}

fn rmse_comparison(smart: f64, baseline: f64, tie_relative_tolerance: f64) -> &'static str {
    let tolerance = tie_relative_tolerance * smart.abs().max(baseline.abs());
    if (smart - baseline).abs() <= tolerance {
        "tie"
    } else if smart < baseline {
        "win"
    } else {
        "loss"
    }
}

pub(crate) fn smart_v3_aggregate(
    comparisons: &[(f64, f64)],
    tie_relative_tolerance: f64,
    minimum_wins_required: usize,
) -> Result<SmartV3Aggregate> {
    let aggregate =
        smart_v3_baseline_aggregate(comparisons, tie_relative_tolerance, minimum_wins_required)?;
    Ok(SmartV3Aggregate {
        wins: aggregate.wins,
        ties: aggregate.ties,
        losses: aggregate.losses,
        median_relative_rmse_change: aggregate.median_relative_rmse_change,
        geometric_mean_rmse_ratio: aggregate.geometric_mean_rmse_ratio,
        minimum_wins_required: aggregate.minimum_wins_required,
        smart_v3_improved: aggregate.smart_v3_beats_baseline,
    })
}

pub(crate) fn smart_v3_baseline_aggregate(
    comparisons: &[(f64, f64)],
    tie_relative_tolerance: f64,
    minimum_wins_required: usize,
) -> Result<SmartV3BaselineAggregate> {
    if comparisons.is_empty() || !tie_relative_tolerance.is_finite() || tie_relative_tolerance < 0.0
    {
        bail!("Smart-v3 aggregate inputs are invalid");
    }
    let mut wins = 0;
    let mut ties = 0;
    let mut losses = 0;
    let mut changes = Vec::with_capacity(comparisons.len());
    let mut log_ratios = Vec::with_capacity(comparisons.len());
    for &(smart, prior) in comparisons {
        if !smart.is_finite() || !prior.is_finite() || smart <= 0.0 || prior <= 0.0 {
            bail!("Smart-v3 RMSE ratios require positive finite values");
        }
        let tolerance = tie_relative_tolerance * smart.abs().max(prior.abs());
        if (smart - prior).abs() <= tolerance {
            ties += 1;
        } else if smart < prior {
            wins += 1;
        } else {
            losses += 1;
        }
        changes.push((smart - prior) / prior);
        log_ratios.push((smart / prior).ln());
    }
    let geometric_mean_rmse_ratio =
        (log_ratios.iter().sum::<f64>() / log_ratios.len() as f64).exp();
    Ok(SmartV3BaselineAggregate {
        wins,
        ties,
        losses,
        median_relative_rmse_change: median(changes)?,
        geometric_mean_rmse_ratio,
        minimum_wins_required,
        smart_v3_beats_baseline: wins >= minimum_wins_required && geometric_mean_rmse_ratio < 1.0,
    })
}

fn smart_selector_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("smart-v3-selector-complete.v1.json")
}

fn validate_smart_selectors(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
) -> Result<Vec<(execution::SmartV3SelectorComplete, String, String)>> {
    let selector_files = fs::read_dir(root.join("datasets"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join("smart-v3-selector-complete.v1.json"))
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    if selector_files.len() != DATASET_COUNT || manifest.datasets.len() != DATASET_COUNT {
        bail!("Smart-v3 scoring requires exactly ten frozen selector receipts");
    }
    let mut selectors = Vec::with_capacity(DATASET_COUNT);
    let mut nodes = BTreeSet::new();
    for dataset in &manifest.datasets {
        let path = smart_selector_path(root, &dataset.dataset_hash);
        let (selector, selector_digest): (execution::SmartV3SelectorComplete, _) =
            read_receipt(&path)?;
        let smart_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart-v3-complete.v1.json");
        let (smart, smart_digest): (execution::SmartComplete, _) = read_receipt(&smart_path)?;
        if selector.schema != "TrainOnlyRedteamSmartV3SelectorComplete.v1"
            || selector.run_id != manifest.run_id
            || selector.dataset_hash != dataset.dataset_hash
            || selector.smart_complete_sha256 != smart_digest
            || selector.candidates.is_empty()
            || !selector.selection_frozen_before_test_access
            || !selector.development_only
            || selector.confirmatory_valid
            || smart.schema != "TrainOnlyRedteamSmartComplete.v1"
            || smart.run_id != manifest.run_id
            || smart.dataset_hash != dataset.dataset_hash
            || !smart.development_only
            || smart.confirmatory_valid
        {
            bail!("Smart-v3 selector receipt drift");
        }
        let execution = selector
            .scq_execution
            .as_ref()
            .context("Smart-v3 selector lacks SCQ execution provenance")?;
        let expected_node = &manifest.config.scq.evaluation_ownership[&dataset.dataset_hash];
        if &execution.expected_node != expected_node
            || execution.scheduler_reported_node != execution.expected_node
            || execution.scq_job_id.is_empty()
            || execution.workers != manifest.config.scq.evaluation_cpus
            || execution.cpus_reserved != manifest.config.scq.evaluation_cpus
            || execution.ram_gb != manifest.config.scq.evaluation_ram_gb
            || !execution.non_preemptible
            || execution.max_runtime_hours != manifest.config.scq.max_runtime_hours
        {
            bail!("Smart-v3 selector SCQ provenance drift");
        }
        let assigned_node = execution.expected_node.clone();
        nodes.insert(assigned_node.clone());
        selectors.push((selector, selector_digest, assigned_node));
    }
    let expected_nodes = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if nodes != expected_nodes {
        bail!("Smart-v3 selectors do not cover all four assigned SCQ nodes");
    }
    Ok(selectors)
}

pub(crate) fn score_smart_v3_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    let root = run_root(study_root, run_id)?;
    let (manifest, manifest_digest) =
        execution::validate_manifest(&root.join("manifest.redteam.v1.json"))?;

    // Freeze and validate all ten selectors before constructing any baseline
    // or test path. Evolution jobs therefore have no code path to either.
    let selectors = validate_smart_selectors(&root, &manifest)?;

    let baseline_path = manifest
        .parent
        .manifest_path
        .parent()
        .context("parent manifest has no run root")?
        .join("baselines.v2.json");
    let (baseline_manifest, baseline_digest): (Value, _) = read_receipt(&baseline_path)?;
    let baseline_entries = baseline_manifest
        .get("entries")
        .and_then(Value::as_array)
        .context("authoritative baseline manifest lacks entries")?;
    if baseline_manifest.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyBaselineManifest.v2")
        || baseline_entries.len() != DATASET_COUNT
    {
        bail!("authoritative baseline manifest drift");
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .ok();
    let selector_by_dataset = selectors
        .into_iter()
        .map(|(selector, digest, node)| (selector.dataset_hash.clone(), (selector, digest, node)))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::with_capacity(DATASET_COUNT);
    let mut prior_comparisons = Vec::with_capacity(DATASET_COUNT);
    let mut tabpfn_comparisons = Vec::with_capacity(DATASET_COUNT);
    let mut stronger_tabfm_comparisons = Vec::with_capacity(DATASET_COUNT);
    for dataset in &manifest.datasets {
        let (selector, selector_digest, assigned_scq_node) =
            &selector_by_dataset[&dataset.dataset_hash];
        let baseline = baseline_entries
            .iter()
            .find(|entry| {
                entry.get("dataset_hash").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("authoritative baseline row is absent")?;
        let baselines = smart_comparison_baselines(baseline, &dataset.dataset_hash)?;
        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.features != dataset.features || test.features != dataset.features {
            bail!("train/test feature shape drift");
        }
        let predictions = predict_variant(&selector.final_variant, &train, &test)?;
        let (test_rmse, test_mae, test_r2) = regression_metrics(&test.y, &predictions)?;
        if baselines.prior_train_only_rmse <= 0.0
            || baselines.tabpfn_v3_rmse <= 0.0
            || baselines.stronger_tabfm_rmse <= 0.0
        {
            bail!("Smart-v3 comparisons require positive baseline RMSEs");
        }
        let relative_rmse_change =
            (test_rmse - baselines.prior_train_only_rmse) / baselines.prior_train_only_rmse;
        let prior_comparison = rmse_comparison(
            test_rmse,
            baselines.prior_train_only_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        let relative_rmse_change_tabpfn_v3 =
            (test_rmse - baselines.tabpfn_v3_rmse) / baselines.tabpfn_v3_rmse;
        let tabpfn_v3_comparison = rmse_comparison(
            test_rmse,
            baselines.tabpfn_v3_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        let relative_rmse_change_stronger_tabfm =
            (test_rmse - baselines.stronger_tabfm_rmse) / baselines.stronger_tabfm_rmse;
        let stronger_tabfm_comparison = rmse_comparison(
            test_rmse,
            baselines.stronger_tabfm_rmse,
            manifest.config.scoring.tie_relative_tolerance,
        );
        prior_comparisons.push((test_rmse, baselines.prior_train_only_rmse));
        tabpfn_comparisons.push((test_rmse, baselines.tabpfn_v3_rmse));
        stronger_tabfm_comparisons.push((test_rmse, baselines.stronger_tabfm_rmse));
        rows.push(SmartV3TestRow {
            dataset_hash: dataset.dataset_hash.clone(),
            final_variant: selector.final_variant.name.clone(),
            assigned_scq_node: assigned_scq_node.clone(),
            selector_sha256: selector_digest.clone(),
            train_sha256: dataset.train_sha256.clone(),
            test_sha256,
            test_rmse,
            test_mae,
            test_r2,
            previous_train_only_rmse: baselines.prior_train_only_rmse,
            relative_rmse_change,
            tabpfn_v3_rmse: baselines.tabpfn_v3_rmse,
            relative_rmse_change_tabpfn_v3,
            stronger_tabfm_method_id: baselines.stronger_tabfm_method_id,
            stronger_tabfm_rmse: baselines.stronger_tabfm_rmse,
            relative_rmse_change_stronger_tabfm,
            prior_comparison: prior_comparison.to_string(),
            tabpfn_v3_comparison: tabpfn_v3_comparison.to_string(),
            stronger_tabfm_comparison: stronger_tabfm_comparison.to_string(),
            development_only: true,
            confirmatory_valid: false,
        });
    }
    if rows.len() != DATASET_COUNT {
        bail!("Smart-v3 scoring did not produce exactly ten test-result rows");
    }
    let aggregate = smart_v3_aggregate(
        &prior_comparisons,
        manifest.config.scoring.tie_relative_tolerance,
        manifest.config.scoring.success_wins_required,
    )?;
    let baseline_comparisons = SmartV3AggregateComparisons {
        previous_train_only: smart_v3_baseline_aggregate(
            &prior_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
        tabpfn_v3: smart_v3_baseline_aggregate(
            &tabpfn_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
        stronger_tabfm: smart_v3_baseline_aggregate(
            &stronger_tabfm_comparisons,
            manifest.config.scoring.tie_relative_tolerance,
            manifest.config.scoring.success_wins_required,
        )?,
    };
    let results = SmartV3TestResults {
        schema: "TrainOnlyRedteamSmartV3TestResults.v1".to_string(),
        run_id: run_id.to_string(),
        manifest_sha256: manifest_digest,
        baseline_manifest_sha256: baseline_digest,
        selector_count: selector_by_dataset.len(),
        test_result_rows: rows,
        aggregate,
        baseline_comparisons,
        selection_frozen_before_test_access: true,
        test_paths_opened_only_by_this_command: true,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let json_path = root.join("smart-v3-test-results.json");
    write_receipt(&json_path, &results)?;

    let mut csv = String::from(
        "dataset_hash,final_variant,assigned_scq_node,test_rmse,test_mae,test_r2,previous_train_only_rmse,relative_rmse_change,tabpfn_v3_rmse,relative_rmse_change_tabpfn_v3,stronger_tabfm_method_id,stronger_tabfm_rmse,relative_rmse_change_stronger_tabfm,prior_comparison,tabpfn_v3_comparison,stronger_tabfm_comparison,development_only,confirmatory_valid\n",
    );
    for row in &results.test_result_rows {
        csv.push_str(&format!(
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{},{},{},true,false\n",
            row.dataset_hash,
            row.final_variant,
            row.assigned_scq_node,
            row.test_rmse,
            row.test_mae,
            row.test_r2,
            row.previous_train_only_rmse,
            row.relative_rmse_change,
            row.tabpfn_v3_rmse,
            row.relative_rmse_change_tabpfn_v3,
            row.stronger_tabfm_method_id,
            row.stronger_tabfm_rmse,
            row.relative_rmse_change_stronger_tabfm,
            row.prior_comparison,
            row.tabpfn_v3_comparison,
            row.stronger_tabfm_comparison,
        ));
    }
    immutable_write(&root.join("smart-v3-test-results.csv"), csv.as_bytes())?;

    let mut markdown = format!(
        "# Smart-v3 CPU development results\n\n\
         `development_only=true`; `confirmatory_valid=false`.\n\n\
         {}\n\n\
         Smart-v3 improved: `{}`. Acceptance requires at least {}/10 wins over the previous train-only method and a geometric-mean RMSE ratio below 1.0.\n\n\
         Wins/ties/losses: `{}/{}/{}`. Median relative RMSE change: `{:.6}%`. Geometric-mean Smart-v3/prior RMSE ratio: `{:.12}`.\n\n\
         ## Independent baseline comparisons\n\n\
         | Baseline | Strict wins | Ties | Losses | Median relative RMSE change | Geometric-mean Smart-v3/baseline RMSE | Smart-v3 beats baseline |\n\
         |---|---:|---:|---:|---:|---:|---|\n\
         | Previous train-only Jope | {} | {} | {} | {:.6}% | {:.12} | {} |\n\
         | TabPFN v3 | {} | {} | {} | {:.6}% | {:.12} | {} |\n\
         | Stronger per-dataset TabFM | {} | {} | {} | {:.6}% | {:.12} | {} |\n\n\
         | Dataset | SCQ node | Final variant | RMSE | MAE | R² | Previous train-only RMSE | Relative change | TabPFN v3 RMSE | Relative change | Stronger TabFM RMSE | Relative change | Result / TabPFN / TabFM |\n\
         |---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        DISCLOSURE,
        results.aggregate.smart_v3_improved,
        results.aggregate.minimum_wins_required,
        results.aggregate.wins,
        results.aggregate.ties,
        results.aggregate.losses,
        100.0 * results.aggregate.median_relative_rmse_change,
        results.aggregate.geometric_mean_rmse_ratio,
        results.baseline_comparisons.previous_train_only.wins,
        results.baseline_comparisons.previous_train_only.ties,
        results.baseline_comparisons.previous_train_only.losses,
        100.0 * results.baseline_comparisons.previous_train_only.median_relative_rmse_change,
        results.baseline_comparisons.previous_train_only.geometric_mean_rmse_ratio,
        results.baseline_comparisons.previous_train_only.smart_v3_beats_baseline,
        results.baseline_comparisons.tabpfn_v3.wins,
        results.baseline_comparisons.tabpfn_v3.ties,
        results.baseline_comparisons.tabpfn_v3.losses,
        100.0 * results.baseline_comparisons.tabpfn_v3.median_relative_rmse_change,
        results.baseline_comparisons.tabpfn_v3.geometric_mean_rmse_ratio,
        results.baseline_comparisons.tabpfn_v3.smart_v3_beats_baseline,
        results.baseline_comparisons.stronger_tabfm.wins,
        results.baseline_comparisons.stronger_tabfm.ties,
        results.baseline_comparisons.stronger_tabfm.losses,
        100.0 * results.baseline_comparisons.stronger_tabfm.median_relative_rmse_change,
        results.baseline_comparisons.stronger_tabfm.geometric_mean_rmse_ratio,
        results.baseline_comparisons.stronger_tabfm.smart_v3_beats_baseline,
    );
    for row in &results.test_result_rows {
        markdown.push_str(&format!(
            "| {} | {} | {} | {:.12} | {:.12} | {:.12} | {:.12} | {:.6}% | {:.12} | {:.6}% | {:.12} ({}) | {:.6}% | {}/{}/{} |\n",
            row.dataset_hash,
            row.assigned_scq_node,
            row.final_variant,
            row.test_rmse,
            row.test_mae,
            row.test_r2,
            row.previous_train_only_rmse,
            100.0 * row.relative_rmse_change,
            row.tabpfn_v3_rmse,
            100.0 * row.relative_rmse_change_tabpfn_v3,
            row.stronger_tabfm_rmse,
            row.stronger_tabfm_method_id,
            100.0 * row.relative_rmse_change_stronger_tabfm,
            row.prior_comparison,
            row.tabpfn_v3_comparison,
            row.stronger_tabfm_comparison,
        ));
    }
    immutable_write(&root.join("smart-v3-test-results.md"), markdown.as_bytes())?;
    Ok(json_path)
}
