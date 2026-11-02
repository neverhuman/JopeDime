use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(manifest: &Value) -> Result<(), String> {
    let object = manifest
        .as_object()
        .ok_or("split manifest has unknown or missing fields")?;
    let expected = [
        "schema",
        "status",
        "seed",
        "development_fraction",
        "temporal_audit_fraction",
        "temporal_audit_cutoff_ms",
        "assignments",
        "missing_source_family",
        "missing_committed_at_ms",
        "incomplete_reasons",
        "legacy_development_dataset_ids",
        "manifest_sha256",
    ];
    if object.len() != expected.len() || expected.iter().any(|k| !object.contains_key(*k)) {
        return Err("split manifest has unknown or missing fields".into());
    }
    if object["schema"] != "PrimeDatasetSplitManifest.v1" {
        return Err("unsupported split manifest schema".into());
    }
    let claimed = object["manifest_sha256"]
        .as_str()
        .ok_or("split manifest content hash is invalid")?;
    let mut payload = object.clone();
    payload.remove("manifest_sha256");
    let canonical = super::python_canonical_json(&Value::Object(payload));
    use sha2::{Digest, Sha256};
    if hex::encode(Sha256::digest(canonical.as_bytes())) != claimed {
        return Err("split manifest content hash differs".into());
    }
    let status = object["status"]
        .as_str()
        .ok_or("split manifest status is invalid")?;
    if !["complete", "incomplete"].contains(&status) {
        return Err("split manifest status is invalid".into());
    }
    if !object["seed"].is_i64() && !object["seed"].is_u64() {
        return Err("split manifest seed must be an integer".into());
    }
    for field in ["development_fraction", "temporal_audit_fraction"] {
        let x = object[field]
            .as_f64()
            .ok_or_else(|| format!("split manifest {field} must be finite"))?;
        if !x.is_finite() {
            return Err(format!("split manifest {field} must be finite"));
        }
        if !(0.0 < x && x < 1.0) {
            return Err(format!("split manifest {field} must be in (0, 1)"));
        }
    }
    let assignments = object["assignments"]
        .as_object()
        .ok_or("split manifest assignments must be an object")?;
    let mut family_split = BTreeMap::<String, String>::new();
    let mut observed = BTreeSet::new();
    let mut audit = Vec::<i64>::new();
    for (dataset, record) in assignments {
        if dataset.is_empty() {
            return Err("split manifest dataset identity is empty".into());
        }
        let r = record
            .as_object()
            .ok_or("split assignment has unknown or missing fields")?;
        if r.len() != 3
            || !["source_family", "split", "first_committed_at_ms"]
                .iter()
                .all(|k| r.contains_key(*k))
        {
            return Err("split assignment has unknown or missing fields".into());
        }
        let family = r["source_family"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("split assignment source_family is empty")?;
        let split = r["split"]
            .as_str()
            .ok_or("split assignment has an invalid split")?;
        if !["train", "development", "temporal_audit"].contains(&split) {
            return Err("split assignment has an invalid split".into());
        }
        let time = r["first_committed_at_ms"]
            .as_i64()
            .filter(|n| *n >= 0)
            .ok_or("split assignment timestamp is invalid")?;
        if let Some(previous) = family_split.insert(family.to_string(), split.to_string()) {
            if previous != split {
                return Err("source family overlaps multiple splits".into());
            }
        }
        observed.insert(split.to_string());
        if split == "temporal_audit" {
            audit.push(time);
        }
    }
    for field in [
        "missing_source_family",
        "missing_committed_at_ms",
        "incomplete_reasons",
        "legacy_development_dataset_ids",
    ] {
        let values = object[field]
            .as_array()
            .ok_or_else(|| format!("split manifest {field} must be strings"))?;
        let strings = values
            .iter()
            .map(|v| {
                v.as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| format!("split manifest {field} must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sorted = strings.iter().copied().collect::<BTreeSet<_>>();
        if strings.len() != sorted.len() || strings.iter().copied().ne(sorted.iter().copied()) {
            return Err(format!("split manifest {field} is not canonical"));
        }
    }
    let expected_cutoff = audit
        .into_iter()
        .min()
        .map(Value::from)
        .unwrap_or(Value::Null);
    if object["temporal_audit_cutoff_ms"] != expected_cutoff {
        return Err("temporal audit cutoff differs from assignments".into());
    }
    if status == "complete" {
        for field in [
            "incomplete_reasons",
            "missing_source_family",
            "missing_committed_at_ms",
        ] {
            if !object[field].as_array().is_some_and(Vec::is_empty) {
                return Err("complete split manifest reports incomplete inputs".into());
            }
        }
        if observed
            != BTreeSet::from([
                "development".to_string(),
                "temporal_audit".to_string(),
                "train".to_string(),
            ])
        {
            return Err("complete split manifest requires all three splits".into());
        }
    } else if !assignments.is_empty() {
        return Err("incomplete split manifest cannot assign datasets".into());
    }
    Ok(())
}

pub(super) fn assignment<'a>(manifest: &'a Value, dataset: &str) -> Option<&'a Value> {
    manifest.get("assignments")?.get(dataset)
}
