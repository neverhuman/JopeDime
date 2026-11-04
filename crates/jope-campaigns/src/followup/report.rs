use super::*;

pub(super) fn source_outcomes(source: &Value) -> Result<BTreeMap<String, Value>> {
    source["outcomes"]
        .as_array()
        .context("source ledger outcomes are missing")?
        .iter()
        .map(|row| {
            let hash = row["dataset_hash"]
                .as_str()
                .context("source outcome has no dataset hash")?;
            Ok((hash.to_string(), row.clone()))
        })
        .collect()
}

pub(super) fn original_report(registry: &Path, hash: &str) -> Result<Value> {
    let path = registry.join(hash).join("reports").join(REPORT_FILE);
    let report: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if report["schema"] != "JopeDimeDatasetReport.v1"
        || report["dataset_hash"].as_str() != Some(hash)
        || report["provenance"]["evolutionary_slots"].as_u64() != Some(600)
        || report["provenance"]["cv7_finalist_receipts"].as_u64() != Some(10)
        || report["comparisons"]["baselines"]
            .as_object()
            .map(|value| value.len())
            != Some(6)
    {
        bail!("original published report is invalid for {hash}");
    }
    Ok(report)
}

pub(super) fn follow_up_comparison(
    old: &Value,
    new: &ValidatedFreeze,
    new_test_rmse: f64,
) -> Result<Value> {
    let old_genome = old["algorithm"]["canonical_genome"]
        .as_str()
        .context("old report genome is missing")?;
    let old_semantic = old["algorithm"]["semantic_sha256"]
        .as_str()
        .context("old report semantic SHA-256 is missing")?;
    let old_cv7_fitness = old["selection_proxy"]["full_cv7"]["robust_fitness"]
        .as_f64()
        .context("old report CV7 fitness is missing")?;
    let old_cv7_rmse = old["selection_proxy"]["full_cv7"]["pooled_oof_rmse"]
        .as_f64()
        .context("old report CV7 RMSE is missing")?;
    let old_test_rmse = old["metrics"]["rmse"]
        .as_f64()
        .context("old report test RMSE is missing")?;
    Ok(json!({
        "label": "exploratory, test-adaptive follow-up on the 35 original strict losses",
        "exploratory_reused_test": true,
        "generation_10": {
            "generation": 9,
            "canonical_genome": old_genome,
            "semantic_sha256": old_semantic,
            "cv7_robust_fitness": old_cv7_fitness,
            "cv7_pooled_oof_rmse": old_cv7_rmse,
            "test_rmse": old_test_rmse,
        },
        "generation_30": {
            "generation": new.generation,
            "canonical_genome": new.canonical_genome,
            "semantic_sha256": new.semantic_sha256,
            "cv7_robust_fitness": new.cv7_robust_fitness,
            "cv7_pooled_oof_rmse": new.cv7_pooled_oof_rmse,
            "test_rmse": new_test_rmse,
        },
        "deltas_30_minus_10": {
            "cv7_robust_fitness": new.cv7_robust_fitness - old_cv7_fitness,
            "cv7_pooled_oof_rmse": new.cv7_pooled_oof_rmse - old_cv7_rmse,
            "test_rmse": new_test_rmse - old_test_rmse,
        },
        "selected_genome_changed": old_semantic != new.semantic_sha256,
    }))
}

pub(super) fn build_v2_report(
    old: &Value,
    new_freeze: &ValidatedFreeze,
    mut generated: Value,
) -> Result<Vec<u8>> {
    let new_test_rmse = generated["metrics"]["rmse"]
        .as_f64()
        .context("generated test RMSE is missing")?;
    generated["schema"] = json!("JopeDimeDatasetReport.v2");
    generated["follow_up"] = follow_up_comparison(old, new_freeze, new_test_rmse)?;
    generated["provenance"]["exploratory_reused_test"] = json!(true);
    generated["provenance"]["source_panel_ledger_sha256"] = json!(ORIGINAL_LEDGER_SHA256);
    generated["provenance"]["follow_up_panel_id"] = json!(FOLLOW_UP_PANEL_ID);
    generated["provenance"]["historical_evolutionary_slots_for_dataset"] = json!(2_400);
    canonical_json(&generated)
}
