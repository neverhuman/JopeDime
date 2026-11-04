//! Candidate selection and model grammar-spec fallback from `jope.loop_selection`.

use anyhow::{bail, Result};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Use checkpoint-stored grammar specs when present, otherwise load repository defaults.
pub fn model_specs<T: Clone>(stored: Option<&[T]>, fallback: impl FnOnce() -> Vec<T>) -> Vec<T> {
    match stored.filter(|specs| !specs.is_empty()) {
        Some(specs) => specs.to_vec(),
        None => fallback(),
    }
}

/// Rank by surrogate mean with engine/head diversity, an engine floor, then uncertainty.
///
/// Both sorts are stable, matching Python's `sorted` behavior for equal numeric keys.
pub fn select_candidates(
    pool: &[Value],
    cv_top_m: usize,
    cv_uncertain_m: usize,
) -> Result<Vec<Value>> {
    let mut by_mu: Vec<&Value> = pool.iter().collect();
    for candidate in pool {
        numeric_key(candidate, "mu")?;
        numeric_key(candidate, "sigma")?;
        if candidate.get("engine").is_none() {
            bail!("candidate missing required key 'engine'");
        }
    }
    by_mu.sort_by(|left, right| descending_numeric(left, right, "mu"));

    let mut selected = Vec::new();
    let mut seen_diversity = HashSet::<(String, String)>::new();
    let engines_present: HashSet<String> = pool
        .iter()
        .map(|candidate| python_string(&candidate["engine"]))
        .collect();

    for candidate in &by_mu {
        if selected.len() >= cv_top_m {
            break;
        }
        let diversity = (
            python_string(&candidate["engine"]),
            python_string(candidate.get("head_type").unwrap_or(&Value::Null)),
        );
        if seen_diversity.insert(diversity) {
            selected.push((**candidate).clone());
        }
    }

    for engine in engines_present {
        if !selected
            .iter()
            .any(|candidate| candidate["engine"].as_str() == Some(engine.as_str()))
        {
            if let Some(candidate) = by_mu.iter().copied().find(|candidate| {
                candidate["engine"].as_str() == Some(engine.as_str())
                    && !selected.contains(candidate)
            }) {
                selected.push(candidate.to_owned());
            }
        }
    }

    let mut rest: Vec<&Value> = pool
        .iter()
        .filter(|candidate| !selected.contains(*candidate))
        .collect();
    rest.sort_by(|left, right| descending_numeric(left, right, "sigma"));
    selected.extend(
        rest.into_iter()
            .take(cv_uncertain_m)
            .map(|candidate| candidate.to_owned()),
    );
    Ok(selected)
}

fn numeric_key(candidate: &Value, key: &str) -> Result<f64> {
    candidate
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("candidate key '{key}' must be numeric"))
}

fn descending_numeric(left: &Value, right: &Value, key: &str) -> Ordering {
    let left = numeric_key(left, key).unwrap_or(f64::NAN);
    let right = numeric_key(right, key).unwrap_or(f64::NAN);
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests;
