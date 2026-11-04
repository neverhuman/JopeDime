//! One-shot, in-place v3.1 corpus label preparation.
//!
//! The Python CLI remains the active entry point; this module provides the Rust
//! implementation while callers are migrated.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const RUNTIME_CAP_MS: f64 = 600_000.0;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrepV31Report {
    pub rows: usize,
    pub datasets: usize,
    pub rows_ranked: usize,
    pub rows_censored_runtime: usize,
}

/// Stamp rank percentile, task type, global RWR weight and runtime labels.
///
/// `write = false` computes the same report without changing the corpus or
/// creating its backup, matching the Python dry-run path.
pub fn prep_v31_labels(path: impl AsRef<Path>, write: bool) -> Result<PrepV31Report> {
    let path = path.as_ref();
    let text =
        fs::read_to_string(path).with_context(|| format!("read corpus {}", path.display()))?;
    let mut rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).context("parse corpus JSONL row"))
        .collect::<Result<Vec<_>>>()?;

    let mut by_dataset: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut scores = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let object = row
            .as_object()
            .context("corpus row must be a JSON object")?;
        let metadata = object.get("metadata").filter(|value| truthy(value));
        let score_value = metadata
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("score_robust").or_else(|| object.get("score")))
            .or_else(|| object.get("score"));
        let score = score_value.and_then(finite_number);
        scores.push(score);
        if score.is_some() {
            let dataset = object
                .get("dataset_id")
                .map(python_string)
                .unwrap_or_else(|| "None".to_owned());
            by_dataset.entry(dataset).or_default().push(index);
        }
    }

    let mut rows_ranked = 0;
    for indices in by_dataset.values() {
        let mut values = indices
            .iter()
            .map(|&index| scores[index].expect("grouped finite scores"))
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let count = indices.len();
        for &index in indices {
            let score = scores[index].expect("grouped finite scores");
            let percentile = if count == 1 {
                0.5
            } else {
                let lower = values.iter().filter(|&&value| value < score).count();
                let equal = values.iter().filter(|&&value| value == score).count();
                let value = (lower as f64 + 0.5 * (equal as f64 - 1.0)) / (count - 1) as f64;
                value.clamp(0.0, 1.0)
            };
            set_metadata(
                &mut rows[index],
                "rank_pct",
                Value::from(round_six(percentile)),
            )?;
            rows_ranked += 1;
        }
    }

    let dataset_ids = rows
        .iter()
        .map(|row| {
            row.get("dataset_id")
                .map(python_string)
                .unwrap_or_else(|| "None".to_owned())
        })
        .collect::<Vec<_>>();
    let provenances = rows
        .iter()
        .map(|row| {
            let metadata = row.get("metadata").filter(|value| truthy(value));
            metadata
                .and_then(Value::as_object)
                .and_then(|metadata| metadata.get("provenance"))
                .filter(|value| truthy(value))
                .map(python_string)
                .unwrap_or_else(|| "gp_report".to_owned())
        })
        .collect::<Vec<_>>();
    let weights = rwr_weights(&dataset_ids, &scores, &provenances);

    for (row, weight) in rows.iter_mut().zip(weights) {
        let task_type = row
            .get("metadata")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("task_type"))
            .filter(|value| truthy(value))
            .cloned()
            .unwrap_or_else(|| Value::String("regression".to_owned()));
        set_metadata(row, "task_type", task_type)?;
        set_metadata(row, "rwr_weight_global", Value::from(round_six(weight)))?;
        let cost = row.get("cost_ms").and_then(finite_number);
        if let Some(cost) = cost {
            let censored = cost >= RUNTIME_CAP_MS;
            set_metadata(
                row,
                "runtime_label",
                Value::from(round_six(cost.min(RUNTIME_CAP_MS).ln_1p())),
            )?;
            set_metadata(row, "runtime_censored", Value::Bool(censored))?;
        }
    }
    let rows_censored_runtime = rows
        .iter()
        .filter(|row| {
            row.get("metadata")
                .and_then(Value::as_object)
                .and_then(|metadata| metadata.get("runtime_censored"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .count();

    if write {
        let backup = backup_path(path);
        if !backup.exists() {
            fs::copy(path, &backup)
                .with_context(|| format!("backup corpus to {}", backup.display()))?;
        }
        let mut output = String::new();
        for row in &rows {
            output.push_str(&serde_json::to_string(row).context("serialize corpus JSONL row")?);
            output.push('\n');
        }
        fs::write(path, output).with_context(|| format!("write corpus {}", path.display()))?;
    }

    Ok(PrepV31Report {
        rows: rows.len(),
        datasets: by_dataset.len(),
        rows_ranked,
        rows_censored_runtime,
    })
}

fn backup_path(path: &Path) -> PathBuf {
    path.with_extension("v3.bak.jsonl")
}

fn set_metadata(row: &mut Value, key: &str, value: Value) -> Result<()> {
    let object = row
        .as_object_mut()
        .context("corpus row must be a JSON object")?;
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Map::new()));
    metadata
        .as_object_mut()
        .with_context(|| format!("metadata must be an object to set {key}"))?
        .insert(key.to_owned(), value);
    Ok(())
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn finite_number(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(string) => string.trim().parse().ok()?,
        Value::Bool(boolean) => f64::from(u8::from(*boolean)),
        _ => return None,
    };
    number.is_finite().then_some(number)
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(_) => "{}".to_owned(),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::String(value) => format!("'{value}'"),
        _ => python_string(value),
    }
}

fn round_six(value: f64) -> f64 {
    (value * 1_000_000.0).round_ties_even() / 1_000_000.0
}

fn rwr_weights(dataset_ids: &[String], scores: &[Option<f64>], provenances: &[String]) -> Vec<f64> {
    let mut weights = provenances
        .iter()
        .map(|provenance| source_multiplier(provenance))
        .collect::<Vec<_>>();
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, (dataset, score)) in dataset_ids.iter().zip(scores).enumerate() {
        if score.is_some_and(f64::is_finite) {
            grouped.entry(dataset).or_default().push(index);
        }
    }
    for indices in grouped.values() {
        if indices.len() < 2 {
            continue;
        }
        let values = indices
            .iter()
            .map(|&index| scores[index].expect("grouped finite scores"))
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let tau = (0.5 * variance.sqrt()).max(0.05);
        let exps = values
            .iter()
            .map(|value| ((value - mean) / tau).min(30.0).exp())
            .collect::<Vec<_>>();
        let total = exps.iter().sum::<f64>();
        for (&index, exponential) in indices.iter().zip(exps) {
            weights[index] *= indices.len() as f64 * exponential / total;
        }
    }
    weights
}

fn source_multiplier(source: &str) -> f64 {
    let exact = match source {
        "loop" | "gp_report" | "self" => Some(1.0),
        "loop_old" => Some(0.5),
        "historical_import" | "seed" => Some(0.25),
        "champion" => Some(1.5),
        _ => None,
    };
    exact.unwrap_or_else(|| {
        let base = source.split_once("_g").map_or(source, |(base, _)| base);
        match base {
            "self" => 1.0,
            "champion" => 1.5,
            _ => 0.25,
        }
    })
}

#[cfg(test)]
#[path = "prep_v31_labels/tests.rs"]
mod tests;
