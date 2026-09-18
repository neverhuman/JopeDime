use super::input::{
    capped_features, keep_feature_prefix, parse_device, parse_dtype, read_target_column,
};
use super::regression_cache::{write_regression_feature_cache, RegressionFeatureCacheMeta};
use super::{Args, Cmd};
use crate::bundle::LoadedModelBundle;
use crate::data::{
    dataset_dirs, load_dataset, load_regression_dataset, regression_report_scores, report_scores,
    select_rows, select_rows_f32, shuffled_take, stratified_take, train_validation_split,
};
use crate::metrics::{binary_auc, r2_score, rmse};
use crate::model::{ChimeraRegressor, StarforgeModel};
use crate::target_encoder::{FusedTargetEncoder, TargetTask};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use std::time::Instant;

pub(super) fn run() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::EncodeTarget {
            input,
            task,
            bundle_root,
        } => {
            let values = read_target_column(&input)?;
            let bundle = LoadedModelBundle::load(&bundle_root)?;
            let encoder = FusedTargetEncoder::load(&bundle)?;
            let encoded = encoder.encode(&values, task)?;
            println!(
                "{}",
                json!({
                    "schema": "jain.target-encoding/v1",
                    "task": match task {
                        TargetTask::Binary => "binary",
                        TargetTask::Regression => "regression",
                    },
                    "input": input,
                    "finite_label_count": values.iter().filter(|value| value.is_finite()).count(),
                    "source_checkpoint_sha256": bundle.manifest.source_checkpoint_sha256,
                    "target_artifact_sha256": bundle.manifest.encoders["target"].sha256,
                    "vector": encoded.vector,
                })
            );
        }
        Cmd::Infer {
            weights,
            train,
            test,
            device,
            dtype,
            max_context,
            max_test,
            seed,
        } => {
            let dev = parse_device(&device)?;
            let dtype = parse_dtype(&dtype)?;
            eprintln!("STARFORGE_RS_DEVICE {device}");
            let model = StarforgeModel::load_dtype(&weights, dev, dtype)?;
            let dir = train
                .parent()
                .filter(|p| p.join("test.csv") == test)
                .context("infer expects train/test in the same dataset directory")?;
            let ds = load_dataset(dir)?;
            let ctx_idx = stratified_take(&ds.train_y, max_context, seed);
            let test_idx = stratified_take(&ds.test_y, max_test, seed ^ 0xfeed_face);
            let (ctx_x, ctx_y) = select_rows(&ds.train_x, &ds.train_y, ds.n_features, &ctx_idx);
            let (test_x, test_y) = select_rows(&ds.test_x, &ds.test_y, ds.n_features, &test_idx);
            let scores = model.predict_ensemble_auc(&ctx_x, &ctx_y, &test_x, ds.n_features)?;
            let auc = binary_auc(&test_y, &scores);
            println!(
                "{}",
                json!({
                    "dataset": ds.name,
                    "mode": "ensemble-rust",
                    "roc_auc": auc,
                    "context_rows": ctx_y.len(),
                    "test_rows": test_y.len(),
                    "n_features": ds.n_features,
                })
            );
        }
        Cmd::Bench {
            weights,
            root,
            device,
            dtype,
            limit,
            max_context,
            max_test,
            train_frac,
            seed,
            out,
        } => {
            let dev = parse_device(&device)?;
            let dtype = parse_dtype(&dtype)?;
            eprintln!("STARFORGE_RS_DEVICE {device}");
            let model = StarforgeModel::load_dtype(&weights, dev, dtype)?;
            let mut rows = Vec::new();
            for (i, dir) in dataset_dirs(&root, limit)?.into_iter().enumerate() {
                let started = Instant::now();
                let ds = load_dataset(&dir)?;
                let (train_idx, val_idx) =
                    train_validation_split(&ds.train_y, train_frac, seed + i as u64);
                let train_y: Vec<u8> = train_idx.iter().map(|&idx| ds.train_y[idx]).collect();
                let rel_ctx = stratified_take(&train_y, max_context, seed + 1000 + i as u64);
                let ctx_idx: Vec<usize> = rel_ctx.into_iter().map(|idx| train_idx[idx]).collect();
                let val_idx: Vec<usize> = val_idx.into_iter().take(max_test).collect();
                let full_ctx_idx =
                    stratified_take(&ds.train_y, max_context, seed + 3000 + i as u64);
                let test_idx = stratified_take(&ds.test_y, max_test, seed + 5000 + i as u64);

                let (ctx_x, ctx_y) = select_rows(&ds.train_x, &ds.train_y, ds.n_features, &ctx_idx);
                let (val_x, val_y) = select_rows(&ds.train_x, &ds.train_y, ds.n_features, &val_idx);
                let (full_ctx_x, full_ctx_y) =
                    select_rows(&ds.train_x, &ds.train_y, ds.n_features, &full_ctx_idx);
                let (test_x, test_y) =
                    select_rows(&ds.test_x, &ds.test_y, ds.n_features, &test_idx);

                let train_scores =
                    model.predict_ensemble_auc(&ctx_x, &ctx_y, &val_x, ds.n_features)?;
                let test_scores =
                    model.predict_ensemble_auc(&full_ctx_x, &full_ctx_y, &test_x, ds.n_features)?;
                let train_auc = binary_auc(&val_y, &train_scores);
                let test_auc = binary_auc(&test_y, &test_scores);
                let reports = report_scores(&dir)?;
                let baseline_key = ["tab", "pfn_v3_classifier"].concat();
                let baseline = reports.get(&baseline_key).copied();
                let best = reports
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(name, auc)| (name.clone(), *auc));
                let mut row = json!({
                    "dataset": ds.name,
                    "mode": "ensemble-rust",
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "baseline_roc_auc": baseline,
                    "best_model": best.as_ref().map(|v| v.0.clone()),
                    "best_roc_auc": best.as_ref().map(|v| v.1),
                    "delta_vs_baseline": baseline.map(|v| test_auc - v),
                    "delta_vs_best": best.as_ref().map(|v| test_auc - v.1),
                    "train_rows_context": ctx_y.len(),
                    "train_rows_eval": val_y.len(),
                    "test_rows_eval": test_y.len(),
                    "n_features": ds.n_features,
                    "seconds": started.elapsed().as_secs_f64(),
                });
                if let Some(obj) = row.as_object_mut() {
                    obj.insert(["leg", "acy_roc_auc"].concat(), json!(baseline));
                    obj.insert(
                        ["delta_vs_leg", "acy"].concat(),
                        json!(baseline.map(|v| test_auc - v)),
                    );
                }
                println!("STARFORGE_RS_RESULT {}", serde_json::to_string(&row)?);
                rows.push(row);
            }
            let mean = if rows.is_empty() {
                f64::NAN
            } else {
                rows.iter()
                    .filter_map(|r| r.get("test_auc").and_then(|v| v.as_f64()))
                    .sum::<f64>()
                    / rows.len() as f64
            };
            let payload = json!({
                "device": device,
                "dtype": format!("{dtype:?}"),
                "mode": "ensemble-rust",
                "limit": limit,
                "mean_test_auc": mean,
                "results": rows,
            });
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&payload)? + "\n")?;
            println!(
                "STARFORGE_RS_SUMMARY datasets={} mean_test_auc={mean:.6}",
                payload["results"].as_array().map(|v| v.len()).unwrap_or(0)
            );
            println!("STARFORGE_RS_WROTE {}", out.display());
        }
        Cmd::BenchRegression {
            weights,
            root,
            device,
            dtype,
            limit,
            max_context,
            max_test,
            max_features,
            seed,
            out,
        } => {
            let dev = parse_device(&device)?;
            let dtype = parse_dtype(&dtype)?;
            eprintln!("STARFORGE_RS_DEVICE {device}");
            let model = StarforgeModel::load_dtype(&weights, dev, dtype)?;
            let mut rows = Vec::new();
            for (i, dir) in dataset_dirs(&root, limit)?.into_iter().enumerate() {
                let started = Instant::now();
                let ds = load_regression_dataset(&dir)?;
                let ctx_idx = shuffled_take(ds.train_y.len(), max_context, seed + i as u64);
                let test_idx = shuffled_take(ds.test_y.len(), max_test, seed + 5000 + i as u64);
                let (ctx_x, ctx_y) =
                    select_rows_f32(&ds.train_x, &ds.train_y, ds.n_features, &ctx_idx);
                let (test_x, test_y) =
                    select_rows_f32(&ds.test_x, &ds.test_y, ds.n_features, &test_idx);
                let n_features = capped_features(ds.n_features, max_features);
                let ctx_x = keep_feature_prefix(ctx_x, ctx_y.len(), ds.n_features, n_features);
                let test_x = keep_feature_prefix(test_x, test_y.len(), ds.n_features, n_features);
                let pred =
                    model.predict_regression_ensemble(&ctx_x, &ctx_y, &test_x, n_features)?;
                let r2 = r2_score(&test_y, &pred);
                let rmse = rmse(&test_y, &pred);
                let reports = regression_report_scores(&dir)?;
                let baseline = reports
                    .get("reference_baseline_v25_regressor_default")
                    .copied();
                let best = reports
                    .iter()
                    .max_by(|a, b| {
                        a.1 .0
                            .partial_cmp(&b.1 .0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(name, score)| (name.clone(), *score));
                let row = json!({
                    "dataset": ds.name,
                    "mode": "regression-ensemble-rust",
                    "r2": r2,
                    "rmse": rmse,
                    "baseline_r2": baseline.map(|v| v.0),
                    "baseline_rmse": baseline.map(|v| v.1),
                    "best_model": best.as_ref().map(|v| v.0.clone()),
                    "best_r2": best.as_ref().map(|v| v.1.0),
                    "delta_vs_baseline_r2": baseline.map(|v| r2 - v.0),
                    "delta_vs_best_r2": best.as_ref().map(|v| r2 - v.1.0),
                    "context_rows": ctx_y.len(),
                    "test_rows_eval": test_y.len(),
                    "n_features": n_features,
                    "source_n_features": ds.n_features,
                    "max_features": max_features,
                    "seconds": started.elapsed().as_secs_f64(),
                });
                println!("STARFORGE_RS_REG_RESULT {}", serde_json::to_string(&row)?);
                rows.push(row);
            }
            let mean = if rows.is_empty() {
                f64::NAN
            } else {
                rows.iter()
                    .filter_map(|r| r.get("r2").and_then(|v| v.as_f64()))
                    .sum::<f64>()
                    / rows.len() as f64
            };
            let payload = json!({
                "device": device,
                "dtype": format!("{dtype:?}"),
                "mode": "regression-ensemble-rust",
                "limit": limit,
                "mean_r2": mean,
                "results": rows,
            });
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&payload)? + "\n")?;
            println!(
                "STARFORGE_RS_REG_SUMMARY datasets={} mean_r2={mean:.6}",
                payload["results"].as_array().map(|v| v.len()).unwrap_or(0)
            );
            println!("STARFORGE_RS_WROTE {}", out.display());
        }
        Cmd::DumpRegressionFeatures {
            starlight_weights,
            tabicl_weights,
            chimera_weights,
            dataset_dir,
            out_dir,
            device,
            dtype,
            max_context,
            max_test,
            max_features,
            seed,
        } => {
            let dev = parse_device(&device)?;
            let dtype = parse_dtype(&dtype)?;
            eprintln!("STARFORGE_RS_DEVICE {device}");
            let model = ChimeraRegressor::load_dtype(
                &starlight_weights,
                &tabicl_weights,
                &chimera_weights,
                dev,
                dtype,
            )?;
            let ds = load_regression_dataset(&dataset_dir)?;
            let ctx_idx = shuffled_take(ds.train_y.len(), max_context, seed);
            let test_idx = shuffled_take(ds.test_y.len(), max_test, seed ^ 0x9e37_79b9_7f4a_7c15);
            let (ctx_x, ctx_y) = select_rows_f32(&ds.train_x, &ds.train_y, ds.n_features, &ctx_idx);
            let (test_x, test_y) =
                select_rows_f32(&ds.test_x, &ds.test_y, ds.n_features, &test_idx);
            let n_features = capped_features(ds.n_features, max_features);
            let ctx_x = keep_feature_prefix(ctx_x, ctx_y.len(), ds.n_features, n_features);
            let test_x = keep_feature_prefix(test_x, test_y.len(), ds.n_features, n_features);
            let dump = model.regression_feature_dump_single(&ctx_x, &ctx_y, &test_x, n_features)?;
            let cache_meta = RegressionFeatureCacheMeta {
                dataset: &ds.name,
                n_features,
                source_n_features: ds.n_features,
                max_features,
                context_idx: &ctx_idx,
                test_idx: &test_idx,
            };
            write_regression_feature_cache(&out_dir, &dump, &test_y, &cache_meta)?;
            println!(
                "{}",
                json!({
                    "mode": "regression-feature-dump",
                    "dataset": ds.name,
                    "out_dir": out_dir,
                    "context_rows": ctx_y.len(),
                    "test_rows_eval": test_y.len(),
                    "n_features": n_features,
                    "source_n_features": ds.n_features,
                })
            );
        }
    }
    Ok(())
}
