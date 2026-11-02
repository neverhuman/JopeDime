use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SPLIT_MANIFEST_V1: &str = "PrimeDatasetSplitManifest.v1";
pub const TRAIN_SPLIT: &str = "train";
pub const DEVELOPMENT_SPLIT: &str = "development";
pub const TEMPORAL_AUDIT_SPLIT: &str = "temporal_audit";
const ASSIGNABLE_SPLITS: [&str; 3] = [TRAIN_SPLIT, DEVELOPMENT_SPLIT, TEMPORAL_AUDIT_SPLIT];
const REPORT_ONLY_SPLITS: [&str; 4] = ["arena", "sealed", "double_holdout", TEMPORAL_AUDIT_SPLIT];

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SplitIntegrityError(String);

impl SplitIntegrityError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

pub use crate::split_integrity_manifest::validate_split_manifest;

/// Return the frozen assignment after validating the entire manifest.
pub fn assignment_for_dataset<'a>(
    manifest: &'a Value,
    dataset: &str,
) -> Result<&'a Value, SplitIntegrityError> {
    validate_split_manifest(manifest)?;
    let record = manifest
        .get("assignments")
        .and_then(Value::as_object)
        .and_then(|assignments| assignments.get(dataset))
        .filter(|record| record.is_object());
    record.ok_or_else(|| {
        SplitIntegrityError::new(format!(
            "dataset {} is absent from frozen split assignments",
            python_repr_string(dataset)
        ))
    })
}

/// Python's insertion-ordered count mapping, serialized with the same JSON object field name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderedCounts(Vec<(String, usize)>);

impl OrderedCounts {
    fn increment(&mut self, key: &str) {
        if let Some((_, count)) = self.0.iter_mut().find(|(existing, _)| existing == key) {
            *count += 1;
        } else {
            self.0.push((key.to_owned(), 1));
        }
    }

    pub fn get(&self, key: &str) -> Option<usize> {
        self.0
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, count)| *count)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, usize)> {
        self.0.iter().map(|(key, count)| (key.as_str(), *count))
    }
}

impl Serialize for OrderedCounts {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, count) in &self.0 {
            map.serialize_entry(key, count)?;
        }
        map.end()
    }
}

/// Row-preserving result of the fail-closed training boundary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartitionedRows<'a> {
    pub train: Vec<&'a Value>,
    pub development: Vec<&'a Value>,
    pub report_only: Vec<&'a Value>,
    pub report_only_counts: OrderedCounts,
    pub split_manifest_sha256: Option<String>,
    pub compatibility: &'static str,
}

/// Partition serialized `TrainingRow` objects without mutating or reordering them.
pub fn partition_training_rows(rows: &[Value]) -> Result<PartitionedRows<'_>, SplitIntegrityError> {
    let mut train = Vec::new();
    let mut development = Vec::new();
    let mut report_only = Vec::new();
    let mut report_only_counts = OrderedCounts::default();
    let mut dataset_split = BTreeMap::<String, String>::new();
    let mut family_split = BTreeMap::<String, String>::new();
    let mut bound_hash: Option<String> = None;
    let mut bound_present = false;
    let mut unbound_model_rows = false;

    for row in rows {
        let empty_metadata = Map::new();
        let metadata = match row.get("metadata") {
            None | Some(Value::Null) => &empty_metadata,
            Some(Value::Object(metadata)) => metadata,
            Some(value) if is_python_falsy(value) => &empty_metadata,
            Some(_) => {
                return Err(SplitIntegrityError::new(
                    "TrainingRow metadata must be an object",
                ));
            }
        };
        let raw_split = metadata
            .get("split")
            .cloned()
            .unwrap_or_else(|| Value::String(TRAIN_SPLIT.to_owned()));
        let mut split = normalize_split(&python_str(&raw_split));
        if split == "validation" {
            split = DEVELOPMENT_SPLIT.to_owned();
        }

        let mut authority_hints = BTreeSet::new();
        for key in [
            "authority_split",
            "dataset_split",
            "evidence_split",
            "source_split",
        ] {
            if let Some(value) = metadata.get(key).filter(|value| !value.is_null()) {
                authority_hints.insert(normalize_split(&python_str(value)));
            }
        }
        let tier_value = metadata
            .get("evidence_tier")
            .filter(|value| !is_python_falsy(value));
        let tier = tier_value.map_or_else(String::new, |value| {
            python_str(value).replace('-', "_").to_lowercase()
        });
        if tier.starts_with("arena_") {
            authority_hints.insert("arena".to_owned());
        }
        for report_split in REPORT_ONLY_SPLITS {
            if tier.contains(report_split) {
                authority_hints.insert(report_split.to_owned());
            }
        }
        let report_only_splits: BTreeSet<String> = REPORT_ONLY_SPLITS
            .iter()
            .map(|split| (*split).to_owned())
            .collect();
        let report_hints: BTreeSet<_> = authority_hints
            .intersection(&report_only_splits)
            .cloned()
            .collect();
        if !report_hints.is_empty() {
            if report_hints.len() != 1 {
                return Err(SplitIntegrityError::new(
                    "TrainingRow has conflicting report-only authority",
                ));
            }
            split = report_hints.into_iter().next().expect("one report hint");
        }
        let is_assignable = ASSIGNABLE_SPLITS.contains(&split.as_str());
        let is_report_only = REPORT_ONLY_SPLITS.contains(&split.as_str());
        if !is_assignable && !is_report_only {
            return Err(SplitIntegrityError::new(format!(
                "unsupported TrainingRow split {}",
                python_repr(&raw_split)
            )));
        }
        if split == TEMPORAL_AUDIT_SPLIT || is_report_only {
            report_only.push(row);
            report_only_counts.increment(&split);
        } else if split == TRAIN_SPLIT {
            train.push(row);
        } else {
            development.push(row);
        }

        let schema = metadata
            .get("split_manifest_schema")
            .filter(|value| !value.is_null());
        let Some(schema) = schema else {
            if split == TRAIN_SPLIT || split == DEVELOPMENT_SPLIT {
                unbound_model_rows = true;
            }
            continue;
        };
        bound_present = true;
        if schema.as_str() != Some(SPLIT_MANIFEST_V1) {
            return Err(SplitIntegrityError::new(
                "TrainingRow uses an unsupported split manifest",
            ));
        }
        let family = metadata
            .get("source_family")
            .and_then(Value::as_str)
            .filter(|family| !family.is_empty())
            .ok_or_else(|| {
                SplitIntegrityError::new("manifest-bound TrainingRow lacks source_family")
            })?;
        if metadata.get("split_assignment") != Some(&Value::String(split.clone())) {
            return Err(SplitIntegrityError::new(
                "TrainingRow split differs from its assignment",
            ));
        }
        let digest = metadata
            .get("split_manifest_sha256")
            .and_then(Value::as_str)
            .filter(|digest| digest.chars().count() == 64)
            .ok_or_else(|| {
                SplitIntegrityError::new("TrainingRow split manifest hash is invalid")
            })?;
        if bound_hash.as_deref().is_some_and(|prior| prior != digest) {
            return Err(SplitIntegrityError::new(
                "TrainingRows mix different split manifests",
            ));
        }
        bound_hash.get_or_insert_with(|| digest.to_owned());
        let dataset = row.get("dataset_id").map(python_str).unwrap_or_default();
        if dataset_split
            .insert(dataset, split.clone())
            .is_some_and(|prior| prior != split)
        {
            return Err(SplitIntegrityError::new(
                "dataset is reassigned across splits",
            ));
        }
        if family_split
            .insert(family.to_owned(), split.clone())
            .is_some_and(|prior| prior != split)
        {
            return Err(SplitIntegrityError::new(
                "source family overlaps multiple splits",
            ));
        }
    }

    if bound_present && unbound_model_rows {
        return Err(SplitIntegrityError::new(
            "manifest-bound training cannot mix unbound train/development rows",
        ));
    }
    Ok(PartitionedRows {
        train,
        development,
        report_only,
        report_only_counts,
        split_manifest_sha256: bound_hash,
        compatibility: if bound_present {
            "bound_v1"
        } else {
            "historical_unbound"
        },
    })
}

fn normalize_split(value: &str) -> String {
    value.trim().to_lowercase().replace('-', "_")
}

fn is_python_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(value) => {
            value.as_i64() == Some(0) || value.as_u64() == Some(0) || value.as_f64() == Some(0.0)
        }
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
    }
}

fn python_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => python_number(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&Value::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::String(value) => python_repr_string(value),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(value) => python_number(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_repr_string(key), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(value.len() + 2);
    repr.push(quote);
    for character in value.chars() {
        match character {
            '\\' => repr.push_str("\\\\"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\t' => repr.push_str("\\t"),
            c if c == quote => {
                repr.push('\\');
                repr.push(c);
            }
            c => repr.push(c),
        }
    }
    repr.push(quote);
    repr
}

fn python_number(value: &Number) -> String {
    let rendered = value.to_string();
    if let Some((mantissa, exponent)) = rendered
        .split_once('e')
        .or_else(|| rendered.split_once('E'))
    {
        let exponent: i32 = exponent.parse().unwrap_or(0);
        if exponent >= 0 {
            format!("{mantissa}e+{exponent}")
        } else {
            format!("{mantissa}e-{:02}", exponent.unsigned_abs())
        }
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(dataset: &str, metadata: Value) -> Value {
        json!({"dataset_id": dataset, "metadata": metadata})
    }

    fn valid_manifest() -> Value {
        let mut manifest = json!({
            "schema": SPLIT_MANIFEST_V1,
            "status": "complete",
            "seed": 17,
            "development_fraction": 0.2,
            "temporal_audit_fraction": 0.1,
            "temporal_audit_cutoff_ms": 30,
            "assignments": {
                "train-dataset": {"source_family": "train-family", "split": "train", "first_committed_at_ms": 10},
                "development-dataset": {"source_family": "development-family", "split": "development", "first_committed_at_ms": 20},
                "audit-dataset": {"source_family": "audit-family", "split": "temporal_audit", "first_committed_at_ms": 30}
            },
            "missing_source_family": [],
            "missing_committed_at_ms": [],
            "incomplete_reasons": [],
            "legacy_development_dataset_ids": [],
            "manifest_sha256": ""
        });
        let mut digest_payload = manifest.clone();
        digest_payload
            .as_object_mut()
            .unwrap()
            .remove("manifest_sha256");
        let digest = crate::split_integrity_manifest::canonical_sha256(&digest_payload).unwrap();
        manifest["manifest_sha256"] = Value::String(digest);
        manifest
    }

    #[test]
    fn assignment_lookup_validates_full_manifest_and_preserves_record() {
        let manifest = valid_manifest();
        validate_split_manifest(&manifest).unwrap();
        assert_eq!(
            assignment_for_dataset(&manifest, "audit-dataset").unwrap()["split"],
            "temporal_audit"
        );
        assert_eq!(
            assignment_for_dataset(&manifest, "missing")
                .unwrap_err()
                .to_string(),
            "dataset 'missing' is absent from frozen split assignments"
        );
    }

    #[test]
    fn training_rows_keep_source_order_and_separate_report_only_evidence() {
        let rows = vec![
            row("train-1", json!({"split": " TRAIN "})),
            row("validation-1", json!({"split": "validation"})),
            row("sealed-1", json!({"split": "sealed"})),
            row("audit-1", json!({"split": "temporal-audit"})),
        ];
        let partition = partition_training_rows(&rows).unwrap();
        assert_eq!(partition.train, vec![&rows[0]]);
        assert_eq!(partition.development, vec![&rows[1]]);
        assert_eq!(partition.report_only, vec![&rows[2], &rows[3]]);
        assert_eq!(partition.report_only_counts.get("sealed"), Some(1));
        assert_eq!(partition.report_only_counts.get("temporal_audit"), Some(1));
        assert_eq!(partition.compatibility, "historical_unbound");
    }

    // Mirrors test_unselected_task_cannot_hide_forbidden_authority: any row's conflicting
    // report-only authority must fail even when a caller would later select other rows.
    #[test]
    fn conflicting_report_only_authority_fails_closed() {
        let rows = vec![
            row("train", json!({"split": "train"})),
            row(
                "other-task",
                json!({"split": "train", "authority_split": "sealed", "evidence_split": "arena"}),
            ),
        ];
        assert_eq!(
            partition_training_rows(&rows).unwrap_err().to_string(),
            "TrainingRow has conflicting report-only authority"
        );
    }

    #[test]
    fn bound_rows_reject_unbound_model_rows_and_cross_split_families() {
        let digest = "a".repeat(64);
        let bound = |dataset: &str, split: &str, family: &str| {
            row(
                dataset,
                json!({
                    "split": split,
                    "split_manifest_schema": SPLIT_MANIFEST_V1,
                    "split_assignment": split,
                    "split_manifest_sha256": digest,
                    "source_family": family
                }),
            )
        };
        let rows = vec![
            bound("train", "train", "family-a"),
            row("legacy", json!({"split": "development"})),
        ];
        assert_eq!(
            partition_training_rows(&rows).unwrap_err().to_string(),
            "manifest-bound training cannot mix unbound train/development rows"
        );
        let rows = vec![
            bound("train", "train", "shared"),
            bound("development", "development", "shared"),
        ];
        assert_eq!(
            partition_training_rows(&rows).unwrap_err().to_string(),
            "source family overlaps multiple splits"
        );
    }
}
