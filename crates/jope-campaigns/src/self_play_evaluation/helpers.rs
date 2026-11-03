use serde_json::{Map, Value};
use std::cmp::Ordering;

const ROBUST_SCORE_CLIP: f64 = -1.0;
const ROBUST_VARIANCE_LAMBDA: f64 = 0.25;

/// Python `robust_score`: clip OOF R² at -1 and subtract one quarter of the
/// population standard deviation of finite folds, capped at one.
pub fn robust_score(oof_r2: Option<f64>, fold_r2: Option<&[f64]>) -> Option<f64> {
    let oof = oof_r2.filter(|value| value.is_finite())?;
    let clipped = oof.max(ROBUST_SCORE_CLIP);
    let finite: Vec<f64> = fold_r2
        .unwrap_or_default()
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let std = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / finite.len() as f64)
            .sqrt()
    };
    Some(clipped - ROBUST_VARIANCE_LAMBDA * std.min(1.0))
}

pub(super) fn finite(value: Option<&Value>) -> Option<f64> {
    value
        .filter(|value| !value.is_boolean())
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

pub(super) fn field<'a>(row: &'a Value, key: &str) -> Option<&'a Value> {
    row.as_object()?.get(key)
}

pub(super) fn set(row: &mut Value, key: &str, value: Value) {
    if !row.is_object() {
        *row = Value::Object(Map::new());
    }
    row.as_object_mut()
        .expect("object inserted")
        .insert(key.into(), value);
}

pub(super) fn optional_number(value: Option<f64>) -> Value {
    value
        .and_then(serde_json::Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

pub(super) fn float_field(row: &Value, key: &str) -> Option<f64> {
    finite(field(row, key))
}

fn score_or_negative_infinity(score: f64) -> f64 {
    if score == 0.0 {
        f64::NEG_INFINITY
    } else {
        score
    }
}

pub(super) fn choose_deployed_candidate(rows: &[Value]) -> Option<usize> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (index, row) in rows.iter().enumerate() {
        let Some(score) = float_field(row, "score_robust") else {
            continue;
        };
        // Python's `value or -inf` treats both missing and numeric zero as absent.
        let oof = float_field(row, "_oof")
            .filter(|value| *value != 0.0)
            .unwrap_or(f64::NEG_INFINITY);
        let score = score_or_negative_infinity(score);
        match best {
            None => best = Some((index, score, oof)),
            Some((_, best_score, best_oof))
                if score.total_cmp(&best_score) == Ordering::Greater
                    || (score == best_score && oof.total_cmp(&best_oof) == Ordering::Greater) =>
            {
                best = Some((index, score, oof));
            }
            _ => {}
        }
    }
    best.map(|(index, _, _)| index)
}

pub(super) fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

pub(super) fn round_six(value: f64) -> f64 {
    // Match Python round(value, 6), including ties-to-even for representable halves.
    let scale = 1_000_000.0;
    let scaled = value * scale;
    if !scaled.is_finite() || scaled.abs() >= 2_f64.powi(52) {
        return value;
    }
    let floor = scaled.floor();
    let fraction = scaled - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || (floor as i128) % 2 != 0 {
        floor + 1.0
    } else {
        floor
    };
    rounded / scale
}

pub(super) fn score_rank_percentile(scores: &mut [f64], score: f64) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }
    scores.sort_by(f64::total_cmp);
    let lower = scores.partition_point(|value| *value < score);
    let upper = scores.partition_point(|value| *value <= score);
    ((lower as f64 + 0.5 * (upper - lower) as f64) / scores.len() as f64).clamp(0.0, 1.0)
}
