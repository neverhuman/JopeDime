use super::{ROBUST_SCORE_CLIP, ROBUST_VARIANCE_LAMBDA};
use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub(super) fn robust_score(oof_r2: f64, folds: &[f64]) -> f64 {
    let clipped = oof_r2.max(ROBUST_SCORE_CLIP);
    let variance = if folds.len() >= 2 {
        let mean = folds.iter().sum::<f64>() / folds.len() as f64;
        let squared = folds
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (squared / folds.len() as f64).sqrt()
    } else {
        0.0
    };
    clipped - ROBUST_VARIANCE_LAMBDA * variance.min(1.0)
}

pub(super) fn lcb95(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std = if values.len() >= 2 {
        let squared = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (squared / values.len() as f64).sqrt()
    } else {
        0.0
    };
    Some(mean - 1.96 * std / (values.len() as f64).sqrt())
}

pub(super) fn robust_sort_order(left: &Value, right: &Value) -> Ordering {
    compare_desc(left, right, "robust_score")
        .then_with(|| compare_desc(left, right, "lcb95"))
        .then_with(|| compare_desc(left, right, "min_fold_r2"))
        .then_with(|| compare_asc(left, right, "cost_ms"))
}

pub(super) fn compare_desc(left: &Value, right: &Value, key: &str) -> Ordering {
    option_f64(right.get(key))
        .unwrap_or(f64::NEG_INFINITY)
        .total_cmp(&option_f64(left.get(key)).unwrap_or(f64::NEG_INFINITY))
}

pub(super) fn compare_asc(left: &Value, right: &Value, key: &str) -> Ordering {
    option_f64(left.get(key))
        .unwrap_or(f64::INFINITY)
        .total_cmp(&option_f64(right.get(key)).unwrap_or(f64::INFINITY))
}

pub(super) fn option_f64(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(value_f64)
        .filter(|number| number.is_finite())
}

pub(super) fn value_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn value_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing or non-string {key}"))
}

pub(super) fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().is_some_and(|v| v != 0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

pub(super) fn append_jsonl(path: &Path, row: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, row)?;
    file.write_all(b"\n")?;
    Ok(())
}

pub(super) fn read_prediction_file(path: &Path) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path)?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .with_context(|| format!("invalid prediction in {}", path.display()))
        })
        .collect()
}

pub(super) fn read_targets(path: &Path) -> Result<Vec<f64>> {
    let file = File::open(path)?;
    let mut targets = Vec::new();
    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = line
            .rsplit(',')
            .next()
            .unwrap_or_default()
            .trim()
            .parse::<f64>();
        match value {
            Ok(value) => targets.push(value),
            Err(_) if line_idx == 0 => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(targets)
}

pub(super) fn regression_metrics(targets: &[f64], predictions: &[f64]) -> Result<Value> {
    let n = targets.len().min(predictions.len());
    if n == 0 || targets.len() != predictions.len() {
        bail!(
            "target/prediction length mismatch: {} vs {}",
            targets.len(),
            predictions.len()
        );
    }
    let mse = (0..n)
        .map(|idx| (targets[idx] - predictions[idx]).powi(2))
        .sum::<f64>()
        / n as f64;
    let rmse = mse.sqrt();
    let mean = targets.iter().sum::<f64>() / n as f64;
    let total = targets
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let r2 = if total > 0.0 {
        1.0 - (mse * n as f64) / total
    } else {
        0.0
    };
    let mae = (0..n)
        .map(|idx| (targets[idx] - predictions[idx]).abs())
        .sum::<f64>()
        / n as f64;
    let mut metrics = Map::new();
    metrics.insert("rmse".into(), json!(rmse));
    metrics.insert("r2".into(), json!(r2));
    metrics.insert("mae".into(), json!(mae));
    metrics.insert("n".into(), json!(n));
    Ok(Value::Object(metrics))
}
