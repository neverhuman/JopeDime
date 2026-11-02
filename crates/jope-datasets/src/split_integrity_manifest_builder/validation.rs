use super::canonical::canonical_sha256;
use super::{DEVELOPMENT_SPLIT, SPLIT_MANIFEST_V1, TEMPORAL_AUDIT_SPLIT, TRAIN_SPLIT};
use anyhow::{anyhow, ensure, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Validate the exact v1 manifest contract, including its canonical content hash.
pub fn validate_split_manifest(manifest: &Value) -> Result<()> {
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
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let object = manifest
        .as_object()
        .ok_or_else(|| anyhow!("split manifest has unknown or missing fields"))?;
    ensure!(
        object.keys().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        "split manifest has unknown or missing fields"
    );
    ensure!(
        manifest["schema"] == SPLIT_MANIFEST_V1,
        "unsupported split manifest schema"
    );
    ensure!(
        manifest["status"] == "complete" || manifest["status"] == "incomplete",
        "split manifest status is invalid"
    );
    ensure!(
        is_json_integer(&manifest["seed"]),
        "split manifest seed must be an integer"
    );
    for field in ["development_fraction", "temporal_audit_fraction"] {
        let value = manifest[field].as_f64();
        ensure!(
            value.is_some_and(f64::is_finite),
            "split manifest {field} must be finite"
        );
        ensure!(
            value.is_some_and(|value| 0.0 < value && value < 1.0),
            "split manifest {field} must be in (0, 1)"
        );
    }

    let mut digest_payload = manifest.clone();
    digest_payload
        .as_object_mut()
        .expect("validated manifest object")
        .remove("manifest_sha256");
    let claimed_digest = manifest["manifest_sha256"].as_str().unwrap_or("");
    ensure!(
        claimed_digest == canonical_sha256(&digest_payload)?,
        "split manifest content hash differs"
    );

    let assignments = manifest["assignments"]
        .as_object()
        .ok_or_else(|| anyhow!("split manifest assignments must be an object"))?;
    let assignable = [TRAIN_SPLIT, DEVELOPMENT_SPLIT, TEMPORAL_AUDIT_SPLIT]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut family_split = BTreeMap::<String, String>::new();
    let mut observed_splits = BTreeSet::new();
    let mut audit_times = Vec::new();
    for (dataset, record) in assignments {
        ensure!(
            !dataset.is_empty(),
            "split manifest dataset identity is empty"
        );
        let record_object = record
            .as_object()
            .ok_or_else(|| anyhow!("split assignment has unknown or missing fields"))?;
        let fields = ["source_family", "split", "first_committed_at_ms"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        ensure!(
            record_object
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == fields,
            "split assignment has unknown or missing fields"
        );
        let family = record["source_family"].as_str().unwrap_or("");
        ensure!(
            !family.is_empty(),
            "split assignment source_family is empty"
        );
        let split = record["split"].as_str().unwrap_or("");
        ensure!(
            assignable.contains(split),
            "split assignment has an invalid split"
        );
        let committed = nonnegative_json_integer(&record["first_committed_at_ms"]);
        ensure!(committed.is_some(), "split assignment timestamp is invalid");
        let committed = committed.expect("checked timestamp");
        if let Some(prior) = family_split.get(family) {
            ensure!(prior == split, "source family overlaps multiple splits");
        } else {
            family_split.insert(family.to_owned(), split.to_owned());
        }
        observed_splits.insert(split);
        if split == TEMPORAL_AUDIT_SPLIT {
            audit_times.push(committed);
        }
    }

    for field in [
        "missing_source_family",
        "missing_committed_at_ms",
        "incomplete_reasons",
        "legacy_development_dataset_ids",
    ] {
        let values = manifest[field]
            .as_array()
            .ok_or_else(|| anyhow!("split manifest {field} must be strings"))?;
        let mut strings = Vec::with_capacity(values.len());
        for value in values {
            let value = value.as_str().unwrap_or("");
            ensure!(!value.is_empty(), "split manifest {field} must be strings");
            strings.push(value);
        }
        let mut sorted = strings.clone();
        sorted.sort_unstable();
        sorted.dedup();
        ensure!(strings == sorted, "split manifest {field} is not canonical");
    }
    let expected_cutoff = audit_times.into_iter().min();
    ensure!(
        json_integer_equals(&manifest["temporal_audit_cutoff_ms"], expected_cutoff),
        "temporal audit cutoff differs from assignments"
    );
    if manifest["status"] == "complete" {
        ensure!(
            manifest["incomplete_reasons"] == json!([])
                && manifest["missing_source_family"] == json!([])
                && manifest["missing_committed_at_ms"] == json!([]),
            "complete split manifest reports incomplete inputs"
        );
        ensure!(
            observed_splits == assignable,
            "complete split manifest requires all three splits"
        );
    } else {
        ensure!(
            assignments.is_empty(),
            "incomplete split manifest cannot assign datasets"
        );
    }
    Ok(())
}

pub(super) fn is_json_integer(value: &Value) -> bool {
    value
        .as_number()
        .is_some_and(|number| number.is_i64() || number.is_u64())
}

pub(super) fn nonnegative_json_integer(value: &Value) -> Option<u64> {
    let number = value.as_number()?;
    if number.is_u64() {
        number.as_u64()
    } else if number.is_i64() {
        number
            .as_i64()
            .and_then(|integer| u64::try_from(integer).ok())
    } else {
        None
    }
}

fn json_integer_equals(value: &Value, expected: Option<u64>) -> bool {
    match expected {
        None => value.is_null(),
        Some(expected) => {
            value.as_u64() == Some(expected)
                || i64::try_from(expected)
                    .ok()
                    .is_some_and(|expected| value.as_i64() == Some(expected))
                || value
                    .as_f64()
                    .is_some_and(|actual| actual == expected as f64)
        }
    }
}
