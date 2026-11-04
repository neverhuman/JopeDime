use super::formatting::numeric;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

pub type CandidateKey = (String, String);
pub type EvaluationMaps = (HashMap<CandidateKey, Value>, HashMap<String, Value>);
pub type GenomeStage = (String, Vec<(String, String)>);
pub type GeneEntry = (String, Value, Option<Value>);

pub fn metric_values(rows: &[Value], key: &str, fallback: Option<&str>) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| {
            let object = row.as_object()?;
            let primary = object.get(key).and_then(numeric);
            primary.or_else(|| fallback.and_then(|name| object.get(name).and_then(numeric)))
        })
        .filter(|value| value.is_finite())
        .collect()
}

pub fn latest_rank_status(rows: &[Value]) -> Vec<Map<String, Value>> {
    for row in rows.iter().rev() {
        if let Some(status) = row.get("rank_status").and_then(Value::as_array) {
            return status
                .iter()
                .filter_map(|value| value.as_object().cloned())
                .collect();
        }
    }
    Vec::new()
}

pub fn spark(values: &[f64], width: isize, label: &str) -> String {
    let prefix = if label.is_empty() {
        String::new()
    } else {
        format!("{label}: ")
    };
    if values.is_empty() {
        return if label.is_empty() {
            String::new()
        } else {
            format!("{prefix}waiting")
        };
    }
    let start = slice_start(values.len(), -width);
    let values = &values[start..];
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() <= 1e-9 * lo.abs().max(hi.abs()) {
        return prefix + &"▁".repeat(values.len());
    }
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let glyphs: String = values
        .iter()
        .map(|value| {
            let index = (((value - lo) / (hi - lo) * (chars.len() - 1) as f64) as usize)
                .min(chars.len() - 1);
            chars[index]
        })
        .collect();
    prefix + &glyphs
}

pub fn trend_arrow(values: &[f64]) -> &'static str {
    if values.len() < 2 {
        return "•";
    }
    let earlier = values[values.len().saturating_sub(4)];
    let delta = values[values.len() - 1] - earlier;
    if delta > 1e-4 {
        "▲"
    } else if delta < -1e-4 {
        "▼"
    } else {
        "→"
    }
}

pub fn meter(value: f64, scale: f64, width: usize) -> String {
    let filled = bounded_fill(value, scale, width);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(width - filled))
}

pub fn bar(value: f64, scale: f64, width: usize) -> String {
    let filled = bounded_fill(value, scale, width);
    format!("[{}{}]", "#".repeat(filled), ".".repeat(width - filled))
}

pub fn row_genome(row: &Map<String, Value>) -> String {
    ["canonical_genome", "input_genome", "genome"]
        .iter()
        .filter_map(|key| row.get(*key))
        .find(|value| truthy(value))
        .map(python_string)
        .unwrap_or_default()
}

pub fn candidate_key(row: &Map<String, Value>) -> Option<CandidateKey> {
    let genome = row_genome(row);
    if genome.is_empty() {
        return None;
    }
    let dataset = ["dataset_hash", "dataset_id"]
        .iter()
        .filter_map(|key| row.get(*key))
        .find(|value| truthy(value))
        .map(python_string)
        .unwrap_or_default();
    Some((dataset, genome))
}

pub fn evaluation_maps(evaluations: &[Value]) -> EvaluationMaps {
    let mut by_key = HashMap::new();
    let mut by_genome = HashMap::new();
    for row in evaluations {
        let Some(object) = row.as_object() else {
            continue;
        };
        if let Some((dataset, genome)) = candidate_key(object) {
            by_key.insert((dataset, genome.clone()), row.clone());
            by_genome.insert(genome, row.clone());
        }
    }
    (by_key, by_genome)
}

pub fn matching_eval(row: &Map<String, Value>, maps: &EvaluationMaps) -> Option<Value> {
    let (by_key, by_genome) = maps;
    if let Some(key) = candidate_key(row) {
        if let Some(found) = by_key.get(&key) {
            return Some(found.clone());
        }
    }
    let genome = row_genome(row);
    if genome.is_empty() {
        None
    } else {
        by_genome.get(&genome).cloned()
    }
}

pub fn gene_stream_entries(
    proposals: &[Value],
    candidate_audit: &[Value],
    evaluations: &[Value],
    limit: isize,
) -> Vec<GeneEntry> {
    let maps = evaluation_maps(evaluations);
    let audit_count = (3isize).max(limit.div_euclid(2)) as usize;
    let audit_rows: Vec<Value> = tail(candidate_audit, audit_count)
        .iter()
        .filter(|row| {
            row.as_object()
                .is_some_and(|object| !row_genome(object).is_empty())
        })
        .cloned()
        .collect();
    let proposal_count = (4isize).max(limit - audit_rows.len() as isize) as usize;
    let proposal_rows: Vec<Value> = tail(proposals, proposal_count)
        .iter()
        .filter(|row| {
            row.as_object()
                .is_some_and(|object| !row_genome(object).is_empty())
        })
        .cloned()
        .collect();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut entries = Vec::new();
    for (source, rows) in [("audit", audit_rows), ("gene", proposal_rows)] {
        for row in rows {
            let Some(object) = row.as_object() else {
                continue;
            };
            let key = candidate_key(object);
            let dedupe = key
                .as_ref()
                .map(|(dataset, genome)| (source.to_owned(), dataset.clone(), genome.clone()))
                .unwrap_or_else(|| (source.to_owned(), String::new(), row_genome(object)));
            if !seen.insert(dedupe) {
                continue;
            }
            let matched = matching_eval(object, &maps);
            entries.push((source.to_owned(), row, matched));
        }
    }
    entries[slice_start(entries.len(), -limit)..].to_vec()
}

pub fn eval_delta(row: Option<&Value>) -> Option<f64> {
    let object = row?.as_object()?;
    for key in [
        "oof_delta_vs_incumbent",
        "delta_vs_incumbent",
        "cv_delta_vs_incumbent",
        "median_delta_vs_incumbent",
        "actual_delta",
    ] {
        if let Some(value) = object.get(key).and_then(numeric) {
            return Some(value);
        }
    }
    let cv = object.get("cv").and_then(Value::as_object);
    ["oof_delta_vs_incumbent", "delta_vs_incumbent", "oof_delta"]
        .iter()
        .find_map(|key| cv.and_then(|map| map.get(*key)).and_then(numeric))
}

pub fn code_token_total(row: &Map<String, Value>) -> Option<i64> {
    let tokens = row.get("code_tokens")?;
    if let Some(object) = tokens.as_object() {
        if let Some(total) = object.get("total").and_then(numeric) {
            return Some(total as i64);
        }
        {
            let mut subtotal: i64 = 0;
            let mut any = false;
            for value in object.values() {
                if let Some(number) = numeric(value) {
                    subtotal = subtotal.saturating_add(number as i64);
                    any = true;
                }
            }
            return any.then_some(subtotal);
        }
    }
    numeric(tokens).map(|value| value as i64)
}

pub fn freshest_age(ages: &[Option<f64>]) -> Option<f64> {
    ages.iter().flatten().copied().reduce(f64::min)
}

pub fn latest_algorithm_row(state: &Map<String, Value>) -> Option<Value> {
    let mut rows: Vec<&Value> = Vec::new();
    for key in ["proposals", "candidate_audit", "evaluations"] {
        if let Some(values) = state.get(key).and_then(Value::as_array) {
            rows.extend(values.iter().rev().filter(|row| row.is_object()));
        }
    }
    let preferred = rows.iter().find(|row| {
        row.as_object().is_some_and(|object| {
            !row_genome(object).is_empty()
                && object.get("valid") != Some(&Value::Bool(false))
                && object.get("candidate_selected") != Some(&Value::Bool(false))
        })
    });
    preferred
        .or_else(|| {
            rows.iter().find(|row| {
                row.as_object()
                    .is_some_and(|object| !row_genome(object).is_empty())
            })
        })
        .map(|row| (**row).clone())
}

fn bounded_fill(value: f64, scale: f64, width: usize) -> usize {
    let ratio = value / scale.max(1e-9);
    if ratio.is_nan() || ratio <= 0.0 {
        0
    } else if ratio >= 1.0 {
        width
    } else {
        (ratio * width as f64) as usize
    }
}

fn tail<T>(values: &[T], count: usize) -> &[T] {
    &values[values.len().saturating_sub(count)..]
}

fn slice_start(length: usize, start: isize) -> usize {
    if start >= 0 {
        (start as usize).min(length)
    } else {
        length.saturating_sub(start.unsigned_abs())
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => value.to_string(),
    }
}
