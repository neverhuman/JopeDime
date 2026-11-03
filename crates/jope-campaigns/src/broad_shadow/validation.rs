use super::canonical::{content_sha256, sha256};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

pub(super) const BROAD_RECEIPT_SCHEMA: &str = "BroadEvaluationReceipt.v1";
pub(super) const STORE_VERIFICATION_SCHEMA: &str = "BroadReceiptStoreVerification.v1";
pub(super) const LABEL_AUTHORITY: &str = "train_only_oof_r2";
const RECEIPT_FIELDS: [&str; 21] = [
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
pub(super) fn receipt_dedup_key(receipt: &Value) -> Result<(String, String, String, String)> {
    Ok((
        string_field(receipt, "dataset_hash")?.to_owned(),
        string_field(receipt, "semantic_sha256")?.to_owned(),
        string_field(receipt, "evaluator_release_sha256")?.to_owned(),
        string_field(receipt, "fold_plan_sha256")?.to_owned(),
    ))
}

pub(super) fn validate_broad_receipt(receipt: &Value) -> Result<()> {
    let fields = receipt
        .as_object()
        .context("broad evaluation receipt has unknown or missing fields")?;
    ensure!(
        fields.len() == RECEIPT_FIELDS.len()
            && RECEIPT_FIELDS
                .iter()
                .all(|field| fields.contains_key(*field)),
        "broad evaluation receipt has unknown or missing fields"
    );
    ensure!(
        receipt.get("schema") == Some(&json!(BROAD_RECEIPT_SCHEMA)),
        "unsupported broad evaluation receipt schema"
    );
    let claimed_hash = receipt
        .get("receipt_sha256")
        .and_then(Value::as_str)
        .context("receipt_sha256 must be a lowercase SHA-256 digest")?;
    ensure!(
        is_sha256(claimed_hash),
        "receipt_sha256 must be a lowercase SHA-256 digest"
    );
    let mut unhashed = receipt.clone();
    unhashed
        .as_object_mut()
        .expect("checked object")
        .remove("receipt_sha256");
    ensure!(
        content_sha256(&unhashed)? == claimed_hash,
        "broad evaluation receipt content hash differs"
    );
    let dataset = receipt
        .get("dataset_hash")
        .and_then(Value::as_str)
        .context("dataset_hash must be a 16- or 64-character lowercase dataset hash")?;
    ensure!(
        (dataset.len() == 16 || dataset.len() == 64) && is_lower_hex(dataset),
        "dataset_hash must be a 16- or 64-character lowercase dataset hash"
    );
    for field in [
        "train_sha256",
        "semantic_sha256",
        "evaluator_release_sha256",
        "fold_plan_sha256",
        "vocabulary_sha256",
    ] {
        ensure!(
            receipt
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(is_sha256),
            "{field} must be a lowercase SHA-256 digest"
        );
    }
    let genome = receipt
        .get("canonical_genome")
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        !genome.is_empty()
            && sha256(genome.as_bytes()) == string_field(receipt, "semantic_sha256")?,
        "receipt semantic hash does not bind canonical genome"
    );
    let fidelity = receipt
        .get("fidelity")
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        fidelity == "screening" || fidelity == "exact",
        "receipt fidelity must be screening or exact"
    );
    validate_protocol(receipt.get("protocol").unwrap_or(&Value::Null), fidelity)?;
    ensure!(
        receipt.get("label_authority") == Some(&json!(LABEL_AUTHORITY))
            && receipt.get("test_table_accessed") == Some(&Value::Bool(false)),
        "test-table or benchmark labels are forbidden"
    );
    ensure!(
        receipt
            .get("selection_channel")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty()),
        "selection_channel must be nonempty"
    );
    ensure!(
        positive_integer(receipt.get("recorded_at_ms"))?,
        "recorded_at_ms must be a positive integer"
    );
    let runtime = optional_finite(receipt.get("runtime_seconds"))?;
    if let Some(runtime) = runtime {
        ensure!(runtime >= 0.0, "runtime_seconds must be finite");
    }
    let valid = receipt
        .get("valid")
        .and_then(Value::as_bool)
        .context("receipt valid flag must be boolean")?;
    if valid {
        ensure!(
            receipt.get("failure") == Some(&Value::Null),
            "valid receipt cannot carry a failure"
        );
        let folds = receipt
            .get("fold_r2")
            .and_then(Value::as_array)
            .context("valid receipt fold vector differs from its protocol")?;
        ensure!(
            folds.len() == if fidelity == "screening" { 3 } else { 5 },
            "valid receipt fold vector differs from its protocol"
        );
        let fold_values = folds
            .iter()
            .map(|value| finite(value, "fold_r2"))
            .collect::<Result<Vec<_>>>()?;
        let oof = finite(receipt.get("oof_r2").unwrap_or(&Value::Null), "oof_r2")?;
        let robust = finite(
            receipt.get("robust_score").unwrap_or(&Value::Null),
            "robust_score",
        )?;
        let expected = robust_score(oof, &fold_values)?;
        ensure!(
            (robust - expected).abs() <= (1e-12 * robust.abs().max(expected.abs())).max(1e-12),
            "receipt robust score is not canonical"
        );
        ensure!(runtime.is_some(), "valid receipt requires measured runtime");
    } else {
        let failure = receipt
            .get("failure")
            .and_then(Value::as_object)
            .context("invalid receipt requires a typed failure")?;
        ensure!(
            failure.len() == 2
                && failure.contains_key("class")
                && failure.contains_key("message")
                && failure["class"]
                    .as_str()
                    .is_some_and(|value| !value.trim().is_empty())
                && failure["message"].is_string(),
            "invalid receipt requires a typed failure"
        );
        ensure!(
            receipt.get("oof_r2") == Some(&Value::Null)
                && receipt.get("robust_score") == Some(&Value::Null)
                && receipt.get("fold_r2") == Some(&json!([])),
            "failure receipts cannot carry finite score labels"
        );
    }
    Ok(())
}

pub(super) fn robust_score(oof_r2: f64, fold_r2: &[f64]) -> Result<f64> {
    ensure!(oof_r2.is_finite(), "oof_r2 must be finite");
    ensure!(!fold_r2.is_empty(), "fold_r2 must be nonempty");
    ensure!(
        fold_r2.iter().all(|value| value.is_finite()),
        "fold_r2 must be finite"
    );
    let mean = fold_r2.iter().sum::<f64>() / fold_r2.len() as f64;
    let variance = fold_r2
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / fold_r2.len() as f64;
    let population_std = variance.sqrt();
    Ok(oof_r2.max(-1.0) - 0.25 * population_std.min(1.0))
}

fn finite(value: &Value, field: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .with_context(|| format!("{field} must be finite"))
}

fn optional_finite(value: Option<&Value>) -> Result<Option<f64>> {
    match value {
        Some(Value::Null) | None => Ok(None),
        Some(value) => {
            let number = finite(value, "runtime_seconds")?;
            ensure!(number >= 0.0, "runtime_seconds must be finite");
            Ok(Some(number))
        }
    }
}

fn validate_protocol(value: &Value, fidelity: &str) -> Result<()> {
    const FIELDS: [&str; 6] = [
        "fidelity",
        "folds",
        "row_cap",
        "timeout_seconds",
        "omp_threads",
        "train_only",
    ];
    let protocol = value
        .as_object()
        .context("protocol has unknown or missing fields")?;
    ensure!(
        protocol.len() == FIELDS.len() && FIELDS.iter().all(|field| protocol.contains_key(*field)),
        "{fidelity} protocol has unknown or missing fields"
    );
    ensure!(
        protocol.get("fidelity").and_then(Value::as_str) == Some(fidelity)
            && protocol.get("train_only") == Some(&Value::Bool(true))
            && protocol.get("omp_threads").and_then(Value::as_i64) == Some(1),
        "{fidelity} protocol violates train-only/OMP authority"
    );
    if fidelity == "screening" {
        ensure!(
            numeric_equals(protocol.get("folds"), 3.0)
                && numeric_equals(protocol.get("row_cap"), 768.0)
                && numeric_equals(protocol.get("timeout_seconds"), 30.0),
            "screening protocol must be CV3/768 rows/30 seconds"
        );
    } else {
        ensure!(
            numeric_equals(protocol.get("folds"), 5.0)
                && protocol.get("row_cap") == Some(&Value::Null)
                && protocol.get("timeout_seconds") == Some(&Value::Null),
            "exact protocol must be full-train CV5 without screening caps"
        );
    }
    Ok(())
}

fn numeric_equals(value: Option<&Value>, expected: f64) -> bool {
    value.and_then(Value::as_f64) == Some(expected)
}

fn positive_integer(value: Option<&Value>) -> Result<bool> {
    Ok(value.and_then(Value::as_u64).is_some_and(|value| value > 0))
}

pub(super) fn bool_field(value: &Value, field: &str) -> Result<bool> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .with_context(|| format!("{field} must be boolean"))
}

pub(super) fn string_field(value: &Value, field: impl AsRef<str>) -> Result<&str> {
    let field = field.as_ref();
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} must be a string"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && is_lower_hex(value)
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
