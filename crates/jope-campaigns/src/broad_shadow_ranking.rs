//! Canonical score normalization and broad receipt ranking helpers.

use anyhow::{bail, Result};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const RECEIPT_SCHEMA: &str = "BroadEvaluationReceipt.v1";
const RECEIPT_FIELDS: &[&str] = &[
    "schema",
    "receipt_sha256",
    "dataset_hash",
    "train_sha256",
    "semantic_sha256",
    "canonical_genome",
    "evaluator_release_sha256",
    "fold_plan_sha256",
    "vocabulary_sha256",
    "fidelity",
    "protocol",
    "valid",
    "failure",
    "fold_r2",
    "oof_r2",
    "robust_score",
    "runtime_seconds",
    "label_authority",
    "test_table_accessed",
    "selection_channel",
    "recorded_at_ms",
];
const PROTOCOL_FIELDS: &[&str] = &[
    "fidelity",
    "folds",
    "row_cap",
    "timeout_seconds",
    "omp_threads",
    "train_only",
];

/// Build a schema envelope and hash its compact, key-sorted JSON payload.
pub fn build_hashed(schema: &str, hash_field: &str, fields: Map<String, Value>) -> Result<Value> {
    let mut payload_fields = Map::new();
    payload_fields.insert("schema".to_owned(), Value::String(schema.to_owned()));
    for (key, value) in fields {
        payload_fields.insert(key, value);
    }
    let mut payload = Value::Object(payload_fields);
    let digest = content_sha256(&payload)?;
    payload
        .as_object_mut()
        .expect("object created above")
        .insert(hash_field.to_owned(), Value::String(digest));
    Ok(payload)
}

/// Median/MAD normalization used by anchor signatures and score targets.
pub fn normalize_score_within_dataset(values: &[f64]) -> Result<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("score must be finite");
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let center = median(values.to_vec());
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    let scale = (1.4826 * median(deviations)).max(1e-6);
    Ok(values
        .iter()
        .map(|value| (value - center) / scale)
        .collect())
}

/// Validate and group valid receipts without crossing dataset or protocol identity.
pub fn group_receipts_for_ranking(
    receipts: &[Value],
) -> Result<BTreeMap<(String, String, String), Vec<Value>>> {
    let mut grouped = BTreeMap::new();
    for receipt in receipts {
        validate_broad_receipt(receipt)?;
        if !receipt["valid"].as_bool().unwrap_or(false) {
            continue;
        }
        let protocol_hash = content_sha256(&receipt["protocol"])?;
        let key = (
            receipt["dataset_hash"].as_str().unwrap().to_owned(),
            receipt["fidelity"].as_str().unwrap().to_owned(),
            protocol_hash,
        );
        grouped
            .entry(key)
            .or_insert_with(Vec::new)
            .push(receipt.clone());
    }
    Ok(grouped)
}

fn validate_broad_receipt(receipt: &Value) -> Result<()> {
    let object = receipt
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("broad evaluation receipt has unknown or missing fields"))?;
    if object.len() != RECEIPT_FIELDS.len()
        || RECEIPT_FIELDS
            .iter()
            .any(|field| !object.contains_key(*field))
    {
        bail!("broad evaluation receipt has unknown or missing fields");
    }
    if receipt["schema"] != RECEIPT_SCHEMA {
        bail!("unsupported broad evaluation receipt schema");
    }
    let mut unhashed = receipt.clone();
    unhashed.as_object_mut().unwrap().remove("receipt_sha256");
    if receipt["receipt_sha256"].as_str() != Some(content_sha256(&unhashed)?.as_str()) {
        bail!("broad evaluation receipt content hash differs");
    }
    validate_digest(&receipt["dataset_hash"], "dataset_hash", Some(&[16, 64]))?;
    for field in [
        "train_sha256",
        "semantic_sha256",
        "evaluator_release_sha256",
        "fold_plan_sha256",
        "vocabulary_sha256",
    ] {
        validate_digest(&receipt[field], field, None)?;
    }
    let genome = receipt["canonical_genome"]
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("receipt semantic hash does not bind canonical genome"))?;
    if hex::encode(Sha256::digest(genome.as_bytes())) != receipt["semantic_sha256"] {
        bail!("receipt semantic hash does not bind canonical genome");
    }
    let fidelity = receipt["fidelity"]
        .as_str()
        .filter(|value| matches!(*value, "screening" | "exact"))
        .ok_or_else(|| anyhow::anyhow!("receipt fidelity must be screening or exact"))?;
    validate_protocol(&receipt["protocol"], fidelity)?;
    if receipt["label_authority"] != "train_only_oof_r2" || receipt["test_table_accessed"] != false
    {
        bail!("test-table or benchmark labels are forbidden");
    }
    if receipt["selection_channel"]
        .as_str()
        .is_none_or(|value| value.trim().is_empty())
    {
        bail!("selection_channel must be nonempty");
    }
    positive_integer(&receipt["recorded_at_ms"], "recorded_at_ms", false)?;
    let runtime = optional_finite(&receipt["runtime_seconds"], "runtime_seconds", true)?;
    let valid = receipt["valid"]
        .as_bool()
        .ok_or_else(|| anyhow::anyhow!("receipt valid flag must be boolean"))?;
    let folds = receipt["fold_r2"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("fold_r2 must be a list"))?;
    if valid {
        if !receipt["failure"].is_null() {
            bail!("valid receipt cannot carry a failure");
        }
        let expected_folds = receipt["protocol"]["folds"].as_u64().unwrap_or(0) as usize;
        if folds.len() != expected_folds {
            bail!("valid receipt fold vector differs from its protocol");
        }
        let fold_values = folds
            .iter()
            .map(|value| finite(value, "fold_r2", false))
            .collect::<Result<Vec<_>>>()?;
        let oof = finite(&receipt["oof_r2"], "oof_r2", false)?;
        let robust = finite(&receipt["robust_score"], "robust_score", false)?;
        let expected = robust_score(oof, &fold_values);
        if (robust - expected).abs() > (1e-12 * robust.abs().max(expected.abs())).max(1e-12) {
            bail!("receipt robust score is not canonical");
        }
        if runtime.is_none() {
            bail!("valid receipt requires measured runtime");
        }
    } else {
        let failure = receipt["failure"].as_object();
        if !failure.is_some_and(|value| {
            value.len() == 2
                && value
                    .get("class")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
                && value.get("message").and_then(Value::as_str).is_some()
        }) {
            bail!("invalid receipt requires a typed failure");
        }
        if !receipt["oof_r2"].is_null() || !receipt["robust_score"].is_null() || !folds.is_empty() {
            bail!("failure receipts cannot carry finite score labels");
        }
    }
    Ok(())
}

fn validate_protocol(protocol: &Value, expected: &str) -> Result<()> {
    let object = protocol
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{expected} protocol has unknown or missing fields"))?;
    if object.len() != PROTOCOL_FIELDS.len()
        || PROTOCOL_FIELDS
            .iter()
            .any(|field| !object.contains_key(*field))
    {
        bail!("{expected} protocol has unknown or missing fields");
    }
    if protocol["fidelity"] != expected
        || protocol["train_only"] != true
        || protocol["omp_threads"] != 1
    {
        bail!("{expected} protocol violates train-only/OMP authority");
    }
    if expected == "screening" {
        if protocol["folds"] != 3 || protocol["row_cap"] != 768 || protocol["timeout_seconds"] != 30
        {
            bail!("screening protocol must be CV3/768 rows/30 seconds");
        }
    } else if protocol["folds"] != 5
        || !protocol["row_cap"].is_null()
        || !protocol["timeout_seconds"].is_null()
    {
        bail!("exact protocol must be full-train CV5 without screening caps");
    }
    Ok(())
}

fn validate_digest(value: &Value, name: &str, lengths: Option<&[usize]>) -> Result<()> {
    let valid = value.as_str().is_some_and(|text| {
        lengths.map_or(text.len() == 64, |lengths| lengths.contains(&text.len()))
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if !valid {
        if name == "dataset_hash" {
            bail!("dataset_hash must be a 16- or 64-character lowercase dataset hash");
        }
        bail!("{name} must be a lowercase SHA-256 digest");
    }
    Ok(())
}

fn positive_integer(value: &Value, name: &str, allow_zero: bool) -> Result<u64> {
    let number = value
        .as_u64()
        .filter(|number| if allow_zero { true } else { *number > 0 });
    number.ok_or_else(|| {
        anyhow::anyhow!(
            "{name} must be a {} integer",
            if allow_zero {
                "nonnegative"
            } else {
                "positive"
            }
        )
    })
}

fn finite(value: &Value, name: &str, minimum_zero: bool) -> Result<f64> {
    let number = value
        .as_f64()
        .filter(|number| number.is_finite() && (!minimum_zero || *number >= 0.0));
    number.ok_or_else(|| anyhow::anyhow!("{name} must be finite"))
}

fn optional_finite(value: &Value, name: &str, minimum_zero: bool) -> Result<Option<f64>> {
    if value.is_null() {
        Ok(None)
    } else {
        finite(value, name, minimum_zero).map(Some)
    }
}

fn robust_score(oof: f64, folds: &[f64]) -> f64 {
    let mean = folds.iter().sum::<f64>() / folds.len() as f64;
    let deviation = (folds
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / folds.len() as f64)
        .sqrt();
    oof.max(-1.0) - 0.25 * deviation.min(1.0)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn content_sha256(value: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::{
        build_hashed, content_sha256, group_receipts_for_ranking, normalize_score_within_dataset,
        RECEIPT_SCHEMA,
    };
    use serde_json::{json, Map, Value};
    use sha2::{Digest, Sha256};

    #[test]
    fn normalized_scores_use_median_and_mad() {
        assert_eq!(
            normalize_score_within_dataset(&[]).unwrap(),
            Vec::<f64>::new()
        );
        assert_eq!(
            normalize_score_within_dataset(&[1.0, 2.0, 3.0]).unwrap(),
            vec![-1.0 / 1.4826, 0.0, 1.0 / 1.4826]
        );
        assert!(normalize_score_within_dataset(&[f64::NAN]).is_err());
    }

    #[test]
    fn hashed_envelopes_bind_schema_and_fields() {
        let mut fields = Map::new();
        fields.insert("answer".to_owned(), json!(42));
        let value = build_hashed("Example.v1", "sha256", fields).unwrap();
        assert_eq!(value["schema"], "Example.v1");
        assert_eq!(value["answer"], 42);
        assert_eq!(value["sha256"].as_str().unwrap().len(), 64);
        let mut unhashed = value.clone();
        unhashed.as_object_mut().unwrap().remove("sha256");
        assert_eq!(value["sha256"], content_sha256(&unhashed).unwrap());
    }

    #[test]
    fn ranking_validates_receipts_before_grouping() {
        assert!(group_receipts_for_ranking(&[json!({"valid": true})]).is_err());
        let empty: Vec<Value> = Vec::new();
        assert!(group_receipts_for_ranking(&empty).unwrap().is_empty());
    }

    #[test]
    fn valid_receipts_group_by_dataset_fidelity_and_protocol_hash() {
        let genome = "(pipe X (ridge))";
        let mut fields = Map::new();
        fields.insert("dataset_hash".to_owned(), json!("a".repeat(16)));
        fields.insert("train_sha256".to_owned(), json!("b".repeat(64)));
        fields.insert(
            "semantic_sha256".to_owned(),
            json!(hex::encode(Sha256::digest(genome.as_bytes()))),
        );
        fields.insert("canonical_genome".to_owned(), json!(genome));
        fields.insert("evaluator_release_sha256".to_owned(), json!("c".repeat(64)));
        fields.insert("fold_plan_sha256".to_owned(), json!("d".repeat(64)));
        fields.insert("vocabulary_sha256".to_owned(), json!("e".repeat(64)));
        fields.insert("fidelity".to_owned(), json!("screening"));
        fields.insert(
            "protocol".to_owned(),
            json!({
                "fidelity": "screening", "folds": 3, "row_cap": 768,
                "timeout_seconds": 30, "omp_threads": 1, "train_only": true
            }),
        );
        fields.insert("valid".to_owned(), json!(true));
        fields.insert("failure".to_owned(), Value::Null);
        fields.insert("fold_r2".to_owned(), json!([1.0, 1.0, 1.0]));
        fields.insert("oof_r2".to_owned(), json!(1.0));
        fields.insert("robust_score".to_owned(), json!(1.0));
        fields.insert("runtime_seconds".to_owned(), json!(0.1));
        fields.insert("label_authority".to_owned(), json!("train_only_oof_r2"));
        fields.insert("test_table_accessed".to_owned(), json!(false));
        fields.insert("selection_channel".to_owned(), json!("anchor"));
        fields.insert("recorded_at_ms".to_owned(), json!(1));
        let receipt = build_hashed(RECEIPT_SCHEMA, "receipt_sha256", fields).unwrap();
        let grouped = group_receipts_for_ranking(std::slice::from_ref(&receipt)).unwrap();
        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped.values().next().unwrap(), &vec![receipt]);
    }
}
