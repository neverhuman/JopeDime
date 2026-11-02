use super::*;

pub(super) fn relation(left: f64, right: f64) -> &'static str {
    match left.total_cmp(&right) {
        Ordering::Less => "win",
        Ordering::Equal => "tie",
        Ordering::Greater => "loss",
    }
}

pub(super) fn baseline_comparisons(
    jope_rmse: f64,
    baselines: &BTreeMap<String, f64>,
) -> Result<(Value, f64, Vec<String>, String, String)> {
    let best = baselines.values().copied().fold(f64::INFINITY, f64::min);
    let best_models = baselines
        .iter()
        .filter_map(|(model, score)| (*score == best).then_some(model.clone()))
        .collect::<Vec<_>>();
    let mut comparisons = Map::new();
    for model in BASELINES {
        let baseline_rmse = baselines[*model];
        comparisons.insert(
            (*model).to_string(),
            json!({
                "rmse": baseline_rmse,
                "baseline_minus_jope_rmse": baseline_rmse - jope_rmse,
                "jope_outcome": relation(jope_rmse, baseline_rmse),
            }),
        );
    }
    Ok((
        Value::Object(comparisons),
        best,
        best_models,
        relation(jope_rmse, best).to_string(),
        relation(jope_rmse, baselines["TabFM ensemble"]).to_string(),
    ))
}

pub(super) fn score_dataset(
    registry: &Path,
    hash: &str,
    freeze: &ValidatedFreeze,
    baselines: &BTreeMap<String, f64>,
) -> Result<ScoredDataset> {
    // This function is called only after all 42 training-only freezes validate.
    let directory = registry.join(hash);
    let train_path = directory.join("train.csv");
    let test_path = directory.join("test.csv");
    let train = read_regression_csv(&train_path)?;
    let test = read_regression_csv(&test_path)?;
    if train.features != test.features {
        bail!("train/test feature mismatch for {hash}");
    }
    let timed = exact_cv::fit_predict_timed(&freeze.canonical_genome, &train, &test)?;
    let (rmse, r2, mae) = exact_cv::metrics(&test.y, &timed.values)?;
    let prediction_bytes = prediction_bytes(&timed.values)?;
    let prediction_sha256 = sha256_bytes(&prediction_bytes);
    let (comparisons, best_rmse, best_models, overall_outcome, tabfm_outcome) =
        baseline_comparisons(rmse, baselines)?;
    let test_sha256 = hash_file(&test_path)?;
    let report = json!({
        "schema": "JopeDimeDatasetReport.v1",
        "algorithm": {
            "name": MODEL,
            "dataset_specific": true,
            "canonical_genome": freeze.canonical_genome,
            "semantic_sha256": freeze.semantic_sha256,
            "complexity": freeze.complexity,
        },
        "dataset_hash": hash,
        "status": "completed",
        "train_rows": train.y.len(),
        "test_rows": test.y.len(),
        "n_features": train.features,
        "metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
            "total_seconds": timed.fit_seconds + timed.predict_seconds,
        },
        "prediction": {
            "rows": timed.values.len(),
            "sha256": prediction_sha256,
        },
        "selection_proxy": {
            "authority_label": "CV7 training-only selection proxy; not final test authority",
            "generation": freeze.generation,
            "slot": freeze.slot,
            "ranking_authority": freeze.ranking_authority,
            "quick_cv5_fold0": {
                "pooled_oof_r2": freeze.quick_pooled_oof_r2,
                "pooled_oof_rmse": freeze.quick_pooled_oof_rmse,
                "robust_fitness": freeze.quick_robust_fitness,
                "fold_manifest_sha256": freeze.cv5_fold_manifest_sha256,
            },
            "full_cv7": {
                "fold_r2": freeze.cv7_fold_r2,
                "pooled_oof_r2": freeze.cv7_pooled_oof_r2,
                "pooled_oof_rmse": freeze.cv7_pooled_oof_rmse,
                "robust_fitness": freeze.cv7_robust_fitness,
                "fit_seconds": freeze.cv7_fit_seconds,
                "predict_seconds": freeze.cv7_predict_seconds,
                "fold_manifest_sha256": freeze.cv7_fold_manifest_sha256,
            },
        },
        "comparisons": {
            "baselines": comparisons,
            "best_baseline": {"models": best_models, "rmse": best_rmse},
            "strict_overall": {
                "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
                "outcome": overall_outcome,
                "margin_best_baseline_minus_jope_rmse": best_rmse - rmse,
            },
            "tabfm_head_to_head": {
                "outcome": tabfm_outcome,
                "tabfm_rmse": baselines["TabFM ensemble"],
                "tabfm_minus_jope_rmse": baselines["TabFM ensemble"] - rmse,
            },
        },
        "provenance": {
            "selection_sha256": freeze.selection_sha256,
            "ledger_sha256": freeze.ledger_sha256,
            "train_sha256": freeze.train_sha256,
            "test_sha256": test_sha256,
            "panel_text_sha256": PANEL_SHA256,
            "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
            "evolutionary_slots": freeze.evolutionary_slots,
            "cv7_finalist_receipts": freeze.finalist_receipts,
        },
    });
    let report_bytes = canonical_json(&report)?;
    let report_sha256 = sha256_bytes(&report_bytes);
    let mut scores = baselines.clone();
    scores.insert(MODEL.to_string(), rmse);
    let outcome = json!({
        "dataset_hash": hash,
        "genome_semantic_sha256": freeze.semantic_sha256,
        "test_metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
        },
        "prediction_sha256": prediction_sha256,
        "report_sha256": report_sha256,
        "best_baseline_rmse": best_rmse,
        "best_baseline_models": best_models,
        "strict_overall_outcome": overall_outcome,
        "tabfm_head_to_head_outcome": tabfm_outcome,
        "rmse_by_algorithm": scores,
    });
    Ok(ScoredDataset {
        dataset_hash: hash.to_string(),
        predictions: timed.values,
        prediction_bytes,
        prediction_sha256,
        report_bytes,
        report_sha256,
        scores,
        outcome,
    })
}
