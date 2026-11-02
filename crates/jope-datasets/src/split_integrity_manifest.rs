//! Frozen split-manifest validation shared by the row partition boundary.

use super::split_integrity_partition::{
    python_number, SplitIntegrityError, ASSIGNABLE_SPLITS, SPLIT_MANIFEST_V1, TEMPORAL_AUDIT_SPLIT,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Validate the complete frozen manifest using the same fail-closed rules as Python.
pub fn validate_split_manifest(manifest: &Value) -> Result<(), SplitIntegrityError> {
    const EXPECTED: [&str; 12] = [
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
    let object = manifest
        .as_object()
        .ok_or_else(|| SplitIntegrityError::new("split manifest has unknown or missing fields"))?;
    let expected: BTreeSet<_> = EXPECTED.into_iter().collect();
    if object.len() != expected.len() || object.keys().any(|key| !expected.contains(key.as_str())) {
        return Err(SplitIntegrityError::new(
            "split manifest has unknown or missing fields",
        ));
    }
    if object.get("schema").and_then(Value::as_str) != Some(SPLIT_MANIFEST_V1) {
        return Err(SplitIntegrityError::new(
            "unsupported split manifest schema",
        ));
    }
    let status = object.get("status").and_then(Value::as_str);
    if !matches!(status, Some("complete" | "incomplete")) {
        return Err(SplitIntegrityError::new("split manifest status is invalid"));
    }
    let seed = object.get("seed").expect("required key set");
    if !is_integer(seed) {
        return Err(SplitIntegrityError::new(
            "split manifest seed must be an integer",
        ));
    }
    for field in ["development_fraction", "temporal_audit_fraction"] {
        let value = object.get(field).expect("required key set");
        let fraction = value.as_f64().filter(|_| !value.is_boolean());
        if match fraction {
            Some(fraction) => !fraction.is_finite(),
            None => true,
        } {
            return Err(SplitIntegrityError::new(format!(
                "split manifest {field} must be finite"
            )));
        }
        let fraction = fraction.expect("checked above");
        if !(0.0 < fraction && fraction < 1.0) {
            return Err(SplitIntegrityError::new(format!(
                "split manifest {field} must be in (0, 1)"
            )));
        }
    }

    let claimed_digest = object
        .get("manifest_sha256")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut digest_payload = manifest.clone();
    digest_payload
        .as_object_mut()
        .expect("manifest object checked")
        .remove("manifest_sha256");
    if claimed_digest != canonical_sha256(&digest_payload)? {
        return Err(SplitIntegrityError::new(
            "split manifest content hash differs",
        ));
    }

    let assignments = object
        .get("assignments")
        .and_then(Value::as_object)
        .ok_or_else(|| SplitIntegrityError::new("split manifest assignments must be an object"))?;
    let mut family_split = BTreeMap::<&str, &str>::new();
    let mut observed_splits = BTreeSet::<&str>::new();
    let mut audit_times = Vec::<u64>::new();
    for (dataset, record) in assignments {
        if dataset.is_empty() {
            return Err(SplitIntegrityError::new(
                "split manifest dataset identity is empty",
            ));
        }
        let record = record.as_object().ok_or_else(|| {
            SplitIntegrityError::new("split assignment has unknown or missing fields")
        })?;
        let record_keys: BTreeSet<_> = record.keys().map(String::as_str).collect();
        let expected_record = BTreeSet::from(["source_family", "split", "first_committed_at_ms"]);
        if record_keys != expected_record {
            return Err(SplitIntegrityError::new(
                "split assignment has unknown or missing fields",
            ));
        }
        let family = record
            .get("source_family")
            .and_then(Value::as_str)
            .filter(|family| !family.is_empty())
            .ok_or_else(|| SplitIntegrityError::new("split assignment source_family is empty"))?;
        let split = record
            .get("split")
            .and_then(Value::as_str)
            .ok_or_else(|| SplitIntegrityError::new("split assignment has an invalid split"))?;
        if !ASSIGNABLE_SPLITS.contains(&split) {
            return Err(SplitIntegrityError::new(
                "split assignment has an invalid split",
            ));
        }
        let committed = record
            .get("first_committed_at_ms")
            .filter(|value| !value.is_boolean())
            .and_then(Value::as_u64)
            .ok_or_else(|| SplitIntegrityError::new("split assignment timestamp is invalid"))?;
        if family_split
            .insert(family, split)
            .is_some_and(|prior| prior != split)
        {
            return Err(SplitIntegrityError::new(
                "source family overlaps multiple splits",
            ));
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
        let values = object.get(field).and_then(Value::as_array).ok_or_else(|| {
            SplitIntegrityError::new(format!("split manifest {field} must be strings"))
        })?;
        let mut strings = Vec::with_capacity(values.len());
        for value in values {
            let value = value
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    SplitIntegrityError::new(format!("split manifest {field} must be strings"))
                })?;
            strings.push(value);
        }
        if strings.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SplitIntegrityError::new(format!(
                "split manifest {field} is not canonical"
            )));
        }
    }

    audit_times.sort_unstable();
    let expected_cutoff = audit_times.first().copied();
    let cutoff_matches = match (expected_cutoff, object.get("temporal_audit_cutoff_ms")) {
        (None, Some(Value::Null)) => true,
        (Some(expected), Some(value)) => !value.is_boolean() && value.as_u64() == Some(expected),
        _ => false,
    };
    if !cutoff_matches {
        return Err(SplitIntegrityError::new(
            "temporal audit cutoff differs from assignments",
        ));
    }
    if status == Some("complete") {
        for field in [
            "incomplete_reasons",
            "missing_source_family",
            "missing_committed_at_ms",
        ] {
            if !object[field].as_array().is_some_and(Vec::is_empty) {
                return Err(SplitIntegrityError::new(
                    "complete split manifest reports incomplete inputs",
                ));
            }
        }
        if observed_splits != BTreeSet::from(ASSIGNABLE_SPLITS) {
            return Err(SplitIntegrityError::new(
                "complete split manifest requires all three splits",
            ));
        }
    } else if !assignments.is_empty() {
        return Err(SplitIntegrityError::new(
            "incomplete split manifest cannot assign datasets",
        ));
    }
    Ok(())
}

fn is_integer(value: &Value) -> bool {
    !value.is_boolean() && (value.as_i64().is_some() || value.as_u64().is_some())
}

pub(super) fn canonical_sha256(value: &Value) -> Result<String, SplitIntegrityError> {
    let mut encoded = String::new();
    write_python_json(value, &mut encoded)?;
    Ok(crate::hash::sha256_hex(encoded.as_bytes()))
}

fn write_python_json(value: &Value, output: &mut String) -> Result<(), SplitIntegrityError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(value) => output.push_str(&python_number(value)),
        Value::String(value) => write_python_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_python_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_python_json_string(key, output);
                output.push(':');
                write_python_json(&values[key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn write_python_json_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c <= '\u{1f}' => output.push_str(&format!("\\u{:04x}", c as u32)),
            c if c <= '\u{7e}' => output.push(c),
            c if (c as u32) <= 0xffff => output.push_str(&format!("\\u{:04x}", c as u32)),
            c => {
                let scalar = c as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    output.push('"');
}
