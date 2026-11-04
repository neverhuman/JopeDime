//! Rust port of `jope.split_integrity_manifest`.
//!
//! Python remains the active compatibility surface through `jope.split_integrity`; this module
//! preserves the manifest wire shape, validation messages, Python rounding, and seeded shuffle.

use anyhow::{anyhow, ensure, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

mod canonical;
mod random;
mod validation;

use canonical::{canonical_sha256, python_float};
pub use validation::validate_split_manifest;
use validation::{is_json_integer, nonnegative_json_integer};

pub const SPLIT_MANIFEST_V1: &str = "PrimeDatasetSplitManifest.v1";
pub const TRAIN_SPLIT: &str = "train";
pub const DEVELOPMENT_SPLIT: &str = "development";
pub const TEMPORAL_AUDIT_SPLIT: &str = "temporal_audit";

/// Only explicit train authority may influence the policy corpus.
pub fn is_training_authority_label(row: &Value) -> bool {
    authority_split(row) == TRAIN_SPLIT
}

/// Build a deterministic family-disjoint split over current train authority.
pub fn build_split_manifest(
    labels: &[Value],
    source_families: &BTreeMap<String, String>,
    development_fraction: &Value,
    temporal_audit_fraction: &Value,
    seed: &Value,
    legacy_development_dataset_ids: &[Value],
) -> Result<Value> {
    validate_seed(seed)?;
    let development_value = fraction_value(development_fraction, "development_fraction")?;
    let audit_value = fraction_value(temporal_audit_fraction, "temporal_audit_fraction")?;
    let train_labels = labels
        .iter()
        .filter(|row| is_training_authority_label(row))
        .collect::<Vec<_>>();
    let datasets = train_labels
        .iter()
        .map(|row| dataset(row))
        .filter(|dataset| !dataset.is_empty())
        .collect::<BTreeSet<_>>();
    let missing_family = datasets
        .iter()
        .filter(|dataset| !source_families.contains_key(*dataset))
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut first_committed = BTreeMap::<String, u64>::new();
    let mut missing_time = BTreeSet::new();
    for dataset_id in &datasets {
        let first = train_labels
            .iter()
            .filter(|row| dataset(row) == *dataset_id)
            .filter_map(|row| committed_at_ms(row))
            .min();
        if let Some(first) = first {
            first_committed.insert(dataset_id.clone(), first);
        } else {
            missing_time.insert(dataset_id.clone());
        }
    }

    let mut reasons = BTreeSet::new();
    if !missing_family.is_empty() {
        reasons.insert("missing_source_family".to_owned());
    }
    if !missing_time.is_empty() {
        reasons.insert("missing_committed_at_ms".to_owned());
    }
    if datasets.is_empty() {
        reasons.insert("no_train_authority_datasets".to_owned());
    }
    let legacy = legacy_development_dataset_ids
        .iter()
        .map(python_string)
        .collect::<BTreeSet<_>>();
    if legacy.iter().any(|dataset| !datasets.contains(dataset)) {
        reasons.insert("legacy_development_dataset_missing".to_owned());
    }

    let mut families = BTreeMap::<String, Vec<String>>::new();
    if reasons.is_empty() {
        for dataset_id in &datasets {
            families
                .entry(source_families[dataset_id].clone())
                .or_default()
                .push(dataset_id.clone());
        }
        if families.len() < 3 {
            reasons.insert("fewer_than_three_source_families".to_owned());
        }
    }

    let mut development_families = BTreeSet::new();
    if reasons.is_empty() && !legacy.is_empty() {
        for (family, members) in &families {
            let selected = members
                .iter()
                .filter(|member| legacy.contains(*member))
                .count();
            if selected > 0 && selected != members.len() {
                reasons.insert("legacy_development_family_overlap".to_owned());
            } else if selected > 0 {
                development_families.insert(family.clone());
            }
        }
        if development_families.is_empty() {
            reasons.insert("legacy_development_split_empty".to_owned());
        }
    }

    if !reasons.is_empty() {
        let manifest = manifest_payload(
            "incomplete",
            seed,
            development_fraction,
            temporal_audit_fraction,
            BTreeMap::new(),
            MissingInputs {
                source_family: missing_family,
                committed_at_ms: missing_time,
                reasons,
            },
            legacy,
        )?;
        validate_split_manifest(&manifest)?;
        return Ok(manifest);
    }

    let family_time = families
        .iter()
        .map(|(family, members)| {
            let first = members
                .iter()
                .filter_map(|dataset| first_committed.get(dataset))
                .copied()
                .min()
                .expect("each family member has a committed timestamp");
            (family.clone(), first)
        })
        .collect::<BTreeMap<_, _>>();
    let audit_candidates = families
        .keys()
        .filter(|family| !development_families.contains(*family))
        .cloned()
        .collect::<Vec<_>>();
    let mut audit_count = py_round_count(audit_value * audit_candidates.len() as f64);
    audit_count = audit_count
        .max(1)
        .min(audit_candidates.len().saturating_sub(1));
    let mut ranked = audit_candidates;
    ranked.sort_by(|left, right| (family_time[left], left).cmp(&(family_time[right], right)));
    let audit_families = ranked
        .into_iter()
        .rev()
        .take(audit_count)
        .collect::<BTreeSet<_>>();
    let mut remaining = families
        .keys()
        .filter(|family| {
            !audit_families.contains(*family) && !development_families.contains(*family)
        })
        .cloned()
        .collect::<Vec<_>>();
    if development_families.is_empty() {
        random::PythonRandom::from_seed(seed_integer(seed)?).shuffle(&mut remaining);
        let mut development_count = py_round_count(development_value * remaining.len() as f64);
        development_count = development_count
            .max(1)
            .min(remaining.len().saturating_sub(1));
        development_families.extend(remaining.iter().take(development_count).cloned());
    }
    let train_families = families
        .keys()
        .filter(|family| {
            !audit_families.contains(*family) && !development_families.contains(*family)
        })
        .collect::<Vec<_>>();
    if train_families.is_empty() || development_families.is_empty() || audit_families.is_empty() {
        let manifest = manifest_payload(
            "incomplete",
            seed,
            development_fraction,
            temporal_audit_fraction,
            BTreeMap::new(),
            MissingInputs {
                source_family: BTreeSet::new(),
                committed_at_ms: BTreeSet::new(),
                reasons: ["unable_to_form_three_nonempty_family_splits".to_owned()]
                    .into_iter()
                    .collect(),
            },
            legacy,
        )?;
        validate_split_manifest(&manifest)?;
        return Ok(manifest);
    }

    let mut assignments = BTreeMap::new();
    for (family, members) in families {
        let split = if audit_families.contains(&family) {
            TEMPORAL_AUDIT_SPLIT
        } else if development_families.contains(&family) {
            DEVELOPMENT_SPLIT
        } else {
            TRAIN_SPLIT
        };
        for dataset_id in members {
            assignments.insert(
                dataset_id.clone(),
                json!({
                    "source_family": family,
                    "split": split,
                    "first_committed_at_ms": first_committed[&dataset_id],
                }),
            );
        }
    }
    let manifest = manifest_payload(
        "complete",
        seed,
        development_fraction,
        temporal_audit_fraction,
        assignments,
        MissingInputs {
            source_family: BTreeSet::new(),
            committed_at_ms: BTreeSet::new(),
            reasons: BTreeSet::new(),
        },
        legacy,
    )?;
    validate_split_manifest(&manifest)?;
    Ok(manifest)
}

struct MissingInputs {
    source_family: BTreeSet<String>,
    committed_at_ms: BTreeSet<String>,
    reasons: BTreeSet<String>,
}

fn manifest_payload(
    status: &str,
    seed: &Value,
    development_fraction: &Value,
    temporal_audit_fraction: &Value,
    assignments: BTreeMap<String, Value>,
    missing: MissingInputs,
    legacy_development_dataset_ids: BTreeSet<String>,
) -> Result<Value> {
    let audit_times = assignments
        .values()
        .filter(|record| record["split"] == TEMPORAL_AUDIT_SPLIT)
        .filter_map(|record| nonnegative_json_integer(&record["first_committed_at_ms"]))
        .collect::<Vec<_>>();
    let assignment_map = assignments.into_iter().collect::<Map<_, _>>();
    let mut payload = json!({
        "schema": SPLIT_MANIFEST_V1,
        "status": status,
        "seed": seed,
        "development_fraction": development_fraction,
        "temporal_audit_fraction": temporal_audit_fraction,
        "temporal_audit_cutoff_ms": audit_times.into_iter().min(),
        "assignments": assignment_map,
        "missing_source_family": missing.source_family,
        "missing_committed_at_ms": missing.committed_at_ms,
        "incomplete_reasons": missing.reasons,
        "legacy_development_dataset_ids": legacy_development_dataset_ids,
    });
    let digest = canonical_sha256(&payload)?;
    payload["manifest_sha256"] = Value::String(digest);
    Ok(payload)
}

fn label_body(row: &Value) -> &Value {
    row.get("label")
        .filter(|label| label.is_object())
        .unwrap_or(row)
}

fn dataset(row: &Value) -> String {
    let label = label_body(row);
    ["dataset", "dataset_hash", "dataset_id"]
        .into_iter()
        .filter_map(|field| label.get(field))
        .find(|value| python_truthy(value))
        .map(python_string)
        .unwrap_or_default()
}

fn authority_split(row: &Value) -> String {
    label_body(row)
        .get("split")
        .filter(|value| !value.is_null())
        .map(python_string)
        .map(|value| value.trim().to_lowercase().replace('-', "_"))
        .unwrap_or_else(|| TRAIN_SPLIT.to_owned())
}

fn committed_at_ms(row: &Value) -> Option<u64> {
    let value = row
        .get("committed_at_ms")
        .filter(|value| python_truthy(value))
        .or_else(|| row.get("_committed_at_ms"))?;
    nonnegative_json_integer(value)
}

fn python_truthy(value: &Value) -> bool {
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
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) if value.is_i64() || value.is_u64() => value.to_string(),
        Value::Number(value) => value.as_f64().map(python_float).unwrap_or_default(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn fraction_value(value: &Value, field: &str) -> Result<f64> {
    let number = value.as_f64();
    ensure!(
        number.is_some_and(f64::is_finite),
        "split manifest {field} must be finite"
    );
    let number = number.expect("checked finite fraction");
    ensure!(
        0.0 < number && number < 1.0,
        "split manifest {field} must be in (0, 1)"
    );
    Ok(number)
}

fn validate_seed(seed: &Value) -> Result<()> {
    ensure!(
        is_json_integer(seed),
        "split manifest seed must be an integer"
    );
    Ok(())
}

fn seed_integer(seed: &Value) -> Result<u64> {
    if let Some(seed) = seed.as_i64() {
        Ok(seed.unsigned_abs())
    } else if let Some(seed) = seed.as_u64() {
        Ok(seed)
    } else {
        Err(anyhow!("split manifest seed must be an integer"))
    }
}

fn py_round_count(value: f64) -> usize {
    let lower = value.floor();
    let fraction = value - lower;
    let rounded = if fraction < 0.5 {
        lower
    } else if fraction > 0.5 || (lower as u64) % 2 == 1 {
        lower + 1.0
    } else {
        lower
    };
    rounded as usize
}

#[cfg(test)]
#[path = "split_integrity_manifest_builder/tests.rs"]
mod tests;
