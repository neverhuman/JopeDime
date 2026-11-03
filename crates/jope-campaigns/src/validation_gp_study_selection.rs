//! Rust port of `jope.validation_gp_study_selection`.
//!
//! The Python study runner still owns its single caller. `SelectionRuntime`
//! supplies the existing evaluation-ledger and event-log boundaries while this
//! module owns candidate scoring, exact fitting, ensemble aggregation, and
//! incumbent selection.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ROBUST_SCORE_CLIP: f64 = -1.0;
const ROBUST_VARIANCE_LAMBDA: f64 = 0.25;

#[derive(Clone, Debug)]
pub struct SelectionConfig {
    pub root: PathBuf,
    pub source_root: PathBuf,
    pub out_dir: PathBuf,
    pub jain_bin: PathBuf,
    pub study_tag: String,
    pub robust_folds: usize,
    pub robust_workers: usize,
}

/// Adapter for the study runner's evaluation ledger and event log.
pub trait SelectionRuntime {
    fn evaluate_regression_with_ledger(
        &mut self,
        rows: &[Value],
        ledger: &Path,
        workers: usize,
    ) -> Result<Vec<(usize, Value)>>;

    fn emit(&mut self, dataset: &str, stage: &str, event: &str, payload: Value) -> Result<()>;
}

pub struct StudySelection<R> {
    pub config: SelectionConfig,
    pub robust_path: PathBuf,
    pub runtime: R,
}

impl<R: SelectionRuntime> StudySelection<R> {
    /// Score ledger results in original candidate order, then rank by robust
    /// score, lower confidence bound, minimum fold score, and cost.
    pub fn robust_rerank(
        &mut self,
        dataset: &str,
        entry: &Value,
        candidates: &[String],
    ) -> Result<Vec<Value>> {
        let train_csv = entry.get("train_csv").cloned().unwrap_or(Value::Null);
        let rows: Vec<Value> = candidates
            .iter()
            .map(|genome| {
                json!({
                    "engine": "featmath",
                    "dataset_hash": dataset,
                    "train_csv": train_csv,
                    "genome": genome,
                    "folds": self.config.robust_folds,
                })
            })
            .collect();
        let ledger = self.config.out_dir.join("robust_cv_ledger.jsonl");
        let evaluations = self.runtime.evaluate_regression_with_ledger(
            &rows,
            &ledger,
            self.config.robust_workers,
        )?;
        let mut results = std::collections::BTreeMap::new();
        for (idx, result) in evaluations {
            results.insert(idx, result);
        }

        let mut output = Vec::with_capacity(candidates.len());
        for (idx, genome) in candidates.iter().enumerate() {
            let result = results.get(&idx).unwrap_or(&Value::Null);
            let cv = result.get("cv").filter(|v| v.is_object());
            let folds: Vec<f64> = cv
                .and_then(|v| v.get("fold_r2"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(value_f64)
                .filter(|v| v.is_finite())
                .collect();
            let oof = cv
                .and_then(|v| v.get("oof_r2"))
                .cloned()
                .unwrap_or(Value::Null);
            let score = value_f64(&oof)
                .filter(|v| v.is_finite())
                .map(|v| robust_score(v, &folds));
            let timings = result.get("timings").filter(|v| v.is_object());
            let min_fold = folds.iter().copied().reduce(f64::min);
            let row = json!({
                "dataset": dataset,
                "rank_input": idx,
                "genome": genome,
                "oof_r2": oof,
                "fold_r2": folds,
                "robust_score": score,
                "lcb95": lcb95(&folds),
                "min_fold_r2": min_fold,
                "cost_ms": timings.and_then(|v| v.get("total_ms")).cloned().unwrap_or(Value::Null),
                "cache_hit": truthy(result.get("cache_hit")),
                "errors": result.get("errors").filter(|v| truthy(Some(v))).cloned().unwrap_or_else(|| json!([])),
            });
            append_jsonl(&self.robust_path, &row)?;
            output.push(row);
        }

        let mut ranked: Vec<Value> = output
            .into_iter()
            .filter(|row| {
                row.get("robust_score")
                    .and_then(value_f64)
                    .is_some_and(f64::is_finite)
            })
            .collect();
        ranked.sort_by(robust_sort_order);
        for (rank, row) in ranked.iter_mut().enumerate() {
            row["robust_rank"] = json!(rank + 1);
        }
        self.runtime.emit(
            dataset,
            "robust_cv",
            "done",
            json!({"candidates": candidates.len(), "ranked": ranked.len()}),
        )?;
        Ok(ranked)
    }

    pub fn final_exact_fits(
        &self,
        dataset: &str,
        entry: &Value,
        single: &Value,
        members: &[Value],
    ) -> Result<Value> {
        let final_dir = self.config.out_dir.join("final_exact_fits").join(dataset);
        let pred_dir = self.config.out_dir.join("predictions").join(dataset);
        fs::create_dir_all(&final_dir)?;
        fs::create_dir_all(&pred_dir)?;
        let single_name = self.model_name(dataset, "single_primary");
        let single_fit = self.fit_genome(
            dataset,
            value_string(single, "genome")?,
            &single_name,
            &final_dir,
            &pred_dir,
        )?;
        let mut fits = Vec::with_capacity(members.len());
        for (idx, row) in members.iter().enumerate() {
            let name = self.model_name(dataset, &format!("ensemble_m{}", idx + 1));
            fits.push(self.fit_genome(
                dataset,
                value_string(row, "genome")?,
                &name,
                &final_dir,
                &pred_dir,
            )?);
        }
        let (ensemble_path, metrics) =
            self.ensemble_predictions(dataset, entry, &fits, &pred_dir)?;
        Ok(json!({
            "single": single_fit,
            "members": fits,
            "ensemble": {
                "predictions_path": ensemble_path,
                "metrics": metrics,
                "member_count": fits.len(),
            },
        }))
    }

    pub fn fit_genome(
        &self,
        dataset: &str,
        genome: &str,
        model_name: &str,
        final_dir: &Path,
        pred_dir: &Path,
    ) -> Result<Value> {
        let log_path = final_dir.join(format!("{model_name}.log"));
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let log = File::create(&log_path)?;
        let error_log = log.try_clone()?;
        let status = Command::new(&self.config.jain_bin)
            .arg("fit-genome")
            .arg("--root")
            .arg(&self.config.root)
            .arg("--dataset")
            .arg(dataset)
            .arg("--genome")
            .arg(genome)
            .arg("--model-name")
            .arg(model_name)
            .arg("--overwrite")
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(error_log))
            .status()?;
        let rc = status.code().unwrap_or(-1);
        if rc != 0 {
            bail!(
                "fit-genome failed for {dataset} {model_name}; rc={rc}; log={}",
                log_path.display()
            );
        }

        let report_src = self
            .config
            .root
            .join(dataset)
            .join("reports")
            .join(format!("{model_name}.json"));
        let pred_src = self
            .config
            .root
            .join(dataset)
            .join("predictions")
            .join(format!("{model_name}.csv"));
        if !report_src.exists() || !pred_src.exists() {
            bail!("fit-genome missing report or predictions for {dataset} {model_name}");
        }
        let report_dst = final_dir.join(format!("{model_name}.json"));
        let pred_dst = pred_dir.join(format!("{model_name}.csv"));
        fs::copy(&report_src, &report_dst)?;
        fs::copy(&pred_src, &pred_dst)?;
        let report: Value = serde_json::from_slice(&fs::read(&report_dst)?)
            .with_context(|| format!("invalid fit report {}", report_dst.display()))?;
        let metrics = report
            .get("metrics")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        Ok(json!({
            "model_name": model_name,
            "genome": genome,
            "report_path": report_dst,
            "predictions_path": pred_dst,
            "metrics": metrics,
        }))
    }

    pub fn ensemble_predictions(
        &self,
        dataset: &str,
        entry: &Value,
        fits: &[Value],
        pred_dir: &Path,
    ) -> Result<(PathBuf, Value)> {
        if fits.is_empty() {
            bail!("{dataset}: no ensemble members");
        }
        let mut predictions = Vec::with_capacity(fits.len());
        for fit in fits {
            predictions.push(read_prediction_file(Path::new(value_string(
                fit,
                "predictions_path",
            )?))?);
        }
        let n = predictions[0].len();
        for (fit, values) in fits.iter().zip(&predictions) {
            if values.len() != n {
                bail!(
                    "{dataset}: prediction row-count mismatch for {}: {} vs {n}",
                    value_string(fit, "model_name")?,
                    values.len()
                );
            }
        }
        let mean_predictions: Vec<f64> = (0..n)
            .map(|i| {
                predictions.iter().map(|values| values[i]).sum::<f64>() / predictions.len() as f64
            })
            .collect();
        let targets_path = Path::new(value_string(entry, "test_csv")?);
        let targets = read_targets(targets_path)?;
        if targets.len() != n {
            bail!(
                "{dataset}: ensemble predictions {n} rows but test target has {} rows",
                targets.len()
            );
        }
        let path = pred_dir.join(format!(
            "{}.csv",
            self.model_name(dataset, "ensemble_secondary")
        ));
        let contents = mean_predictions
            .iter()
            .map(|value| format!("{value:.16e}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, format!("{contents}\n"))?;
        Ok((path, regression_metrics(&targets, &mean_predictions)?))
    }

    pub fn incumbent(&self, dataset: &str) -> Value {
        let reports_dir = self.config.source_root.join(dataset).join("reports");
        let mut paths: Vec<PathBuf> = fs::read_dir(reports_dir)
            .into_iter()
            .flatten()
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect();
        paths.sort();
        let mut best = json!({"model": null, "report": null, "rmse": null, "r2": null});
        for path in paths {
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if stem.starts_with(&self.config.study_tag) {
                continue;
            }
            let Ok(report_bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(report) = serde_json::from_slice::<Value>(&report_bytes) else {
                continue;
            };
            let Some(metrics) = report.get("metrics").filter(|v| v.is_object()) else {
                continue;
            };
            let Some(rmse) = metrics
                .get("rmse")
                .and_then(value_f64)
                .filter(|v| v.is_finite())
            else {
                continue;
            };
            let current = best.get("rmse").and_then(value_f64);
            if current.is_none_or(|current| rmse < current) {
                best = json!({
                        "model": stem,
                "report": path.display().to_string(),
                        "rmse": rmse,
                        "r2": metrics.get("r2").cloned().unwrap_or(Value::Null),
                    });
            }
        }
        best
    }

    fn model_name(&self, dataset: &str, suffix: &str) -> String {
        format!("{}__{dataset}__{suffix}", self.config.study_tag)
    }
}

mod helpers;
use helpers::*;

#[cfg(test)]
mod tests;
