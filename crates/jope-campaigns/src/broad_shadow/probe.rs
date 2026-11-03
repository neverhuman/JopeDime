//! Exact ten-slot probe construction with deterministic duplicate refill.

use anyhow::{bail, ensure, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};

mod validation;
use validation::validate_probe_slate;

const PROBE_SLATE_SCHEMA: &str = "ProbeSlate.v1";
const ANCHOR_COUNT: usize = 4;
const PILOT_DATASET_COUNT: usize = 2_000;
const PILOT_GENOMES_PER_DATASET: usize = 10;
const SLOT_ROLES: [(&str, usize); 4] = [
    ("anchor", 4),
    ("transfer", 2),
    ("policy", 2),
    ("diverse", 2),
];
const SLATE_FIELDS: &[&str] = &[
    "schema",
    "slate_sha256",
    "catalog_sha256",
    "created_at_ms",
    "seed",
    "screening_protocol",
    "exact_protocol",
    "anchors",
    "datasets",
];
const ANCHOR_FIELDS: &[&str] = &[
    "anchor_index",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "algorithm_family",
    "performance_quartile",
    "median_runtime_seconds",
    "real_dataset_coverage",
];
const DATASET_FIELDS: &[&str] = &["dataset_hash", "synthetic_split", "slots"];
const SLOT_FIELDS: &[&str] = &[
    "slot",
    "role",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "source_dataset_hash",
    "anchor_index",
];
const PROTOCOL_FIELDS: &[&str] = &[
    "fidelity",
    "folds",
    "row_cap",
    "timeout_seconds",
    "omp_threads",
    "train_only",
];

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_hash(value: &Value) -> Result<String> {
    crate::selfplay_data::canonical_json_sha256(value)
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{name} has unknown or missing fields"))
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing field {key}"))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("{name} must be a string"))
}

fn exact_fields(value: &Value, expected: &[&str], name: &str) -> Result<()> {
    let map = object(value, name)?;
    ensure!(
        map.len() == expected.len() && expected.iter().all(|key| map.contains_key(*key)),
        "{name} has unknown or missing fields"
    );
    Ok(())
}

fn sha256_field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let digest = text(value, name)?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be a lowercase SHA-256 digest"
    );
    Ok(digest)
}

fn dataset_hash<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let digest = text(value, name)?;
    ensure!(
        matches!(digest.len(), 16 | 64)
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be a 16- or 64-character lowercase dataset hash"
    );
    Ok(digest)
}

fn slug<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let value = text(value, name)?;
    let mut previous_was_alphanumeric = false;
    for byte in value.bytes() {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
            previous_was_alphanumeric = true;
        } else if matches!(byte, b'-' | b'_' | b'.') && previous_was_alphanumeric {
            previous_was_alphanumeric = false;
        } else {
            bail!("{name} must be a normalized lowercase slug");
        }
    }
    ensure!(
        previous_was_alphanumeric,
        "{name} must be a normalized lowercase slug"
    );
    Ok(value)
}

fn positive_integer(value: &Value, name: &str) -> Result<u64> {
    let integer = value
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("{name} must be a positive integer"))?;
    ensure!(integer > 0, "{name} must be a positive integer");
    Ok(integer)
}

fn integer(value: &Value, name: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("{name} must be an integer"))
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => value.to_string(),
    }
}

fn candidate_order<'a>(
    candidates: &'a [Value],
    seed: i64,
    dataset: &str,
    role: &str,
) -> Result<Vec<&'a Value>> {
    let mut ordered = candidates
        .iter()
        .map(|candidate| {
            let semantic = python_string(field(candidate, "semantic_sha256")?);
            let key = format!("{seed}\0{dataset}\0{role}\0{semantic}");
            Ok((sha256_hex(key.as_bytes()), candidate))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(ordered
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect())
}

fn take_unique(
    candidates: &[Value],
    count: usize,
    used: &mut HashSet<String>,
    seed: i64,
    dataset: &str,
    role: &str,
) -> Result<Vec<Value>> {
    let mut selected = Vec::new();
    for candidate in candidate_order(candidates, seed, dataset, role)? {
        let semantic = python_string(field(candidate, "semantic_sha256")?);
        if !used.insert(semantic) {
            continue;
        }
        selected.push(candidate.clone());
        if selected.len() == count {
            return Ok(selected);
        }
    }
    bail!("dataset {dataset} cannot fill {count} unique {role} slots")
}

/// Build and validate the deterministic, content-hashed ten-slot probe slate.
pub fn build_probe_slate(
    catalog_sha256: &str,
    anchors: &[Value],
    pilot_datasets: &[Value],
    candidate_pools: &Value,
    seed: i64,
    created_at_ms: u64,
    production: bool,
) -> Result<Value> {
    let mut sorted_pilots = pilot_datasets.iter().collect::<Vec<_>>();
    sorted_pilots.sort_by_key(|row| {
        field(row, "dataset_hash")
            .map(python_string)
            .unwrap_or_default()
    });
    let pools_by_dataset = candidate_pools
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("candidate pools must be a mapping"))?;
    let mut datasets = Vec::new();
    for row in sorted_pilots {
        let dataset = python_string(field(row, "dataset_hash")?);
        let pools = pools_by_dataset
            .get(&dataset)
            .ok_or_else(|| anyhow::anyhow!("candidate pools are missing dataset {dataset}"))?;
        let pools = object(pools, "candidate pools")?;
        let mut used = anchors
            .iter()
            .map(|anchor| field(anchor, "semantic_sha256").map(python_string))
            .collect::<Result<HashSet<_>>>()?;
        let mut slots = anchors
            .iter()
            .enumerate()
            .map(|(index, anchor)| -> Result<Value> {
                Ok(json!({
                    "slot": index,
                    "role": "anchor",
                    "engine": field(anchor, "engine")?,
                    "semantic_sha256": field(anchor, "semantic_sha256")?,
                    "canonical_genome": field(anchor, "canonical_genome")?,
                    "source_dataset_hash": null,
                    "anchor_index": index,
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let refill = pools
            .get("refill")
            .map(as_sequence)
            .transpose()?
            .unwrap_or(&[]);
        for (role, count) in SLOT_ROLES.iter().skip(1) {
            let requested = pools
                .get(*role)
                .map(as_sequence)
                .transpose()?
                .unwrap_or(&[]);
            let mut selected = Vec::new();
            for source in [requested, refill] {
                let needed = *count - selected.len();
                if needed == 0 {
                    break;
                }
                let mut available = Vec::new();
                for candidate in source {
                    let semantic = python_string(field(candidate, "semantic_sha256")?);
                    if !used.contains(&semantic) {
                        available.push(candidate.clone());
                    }
                }
                if available.is_empty() {
                    continue;
                }
                selected.extend(take_unique(
                    &available,
                    needed.min(available.len()),
                    &mut used,
                    seed,
                    &dataset,
                    role,
                )?);
            }
            ensure!(
                selected.len() == *count,
                "dataset {dataset} cannot deterministically refill {role}"
            );
            for candidate in selected {
                let slot_index = slots.len();
                slots.push(json!({
                    "slot": slot_index,
                    "role": role,
                    "engine": field(&candidate, "engine")?,
                    "semantic_sha256": field(&candidate, "semantic_sha256")?,
                    "canonical_genome": field(&candidate, "canonical_genome")?,
                    "source_dataset_hash": candidate.get("source_dataset_hash").cloned().unwrap_or(Value::Null),
                    "anchor_index": null,
                }));
            }
        }
        datasets.push(json!({
            "dataset_hash": dataset,
            "synthetic_split": field(row, "split")?,
            "slots": slots,
        }));
    }
    let mut slate = json!({
        "schema": PROBE_SLATE_SCHEMA,
        "catalog_sha256": catalog_sha256,
        "created_at_ms": created_at_ms,
        "seed": seed,
        "screening_protocol": {
            "fidelity": "screening", "folds": 3, "row_cap": 768,
            "timeout_seconds": 30, "omp_threads": 1, "train_only": true,
        },
        "exact_protocol": {
            "fidelity": "exact", "folds": 5, "row_cap": null,
            "timeout_seconds": null, "omp_threads": 1, "train_only": true,
        },
        "anchors": anchors,
        "datasets": datasets,
    });
    let digest = canonical_hash(&slate)?;
    slate
        .as_object_mut()
        .expect("constructed object")
        .insert("slate_sha256".into(), json!(digest));
    validate_probe_slate(&slate, production)?;
    Ok(slate)
}

fn as_sequence(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| anyhow::anyhow!("candidate pool must be a sequence"))
}

#[cfg(test)]
mod tests;
