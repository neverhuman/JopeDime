use super::canonical::{canonical_json_bytes, content_sha256};
#[cfg(test)]
use super::canonical::{python_float, sha256};
use super::validation::{
    bool_field, receipt_dedup_key, robust_score, string_field, validate_broad_receipt,
    BROAD_RECEIPT_SCHEMA, LABEL_AUTHORITY, STORE_VERIFICATION_SCHEMA,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub struct ReceiptInput {
    pub dataset_hash: String,
    pub train_sha256: String,
    pub semantic_sha256: String,
    pub canonical_genome: String,
    pub evaluator_release_sha256: String,
    pub fold_plan_sha256: String,
    pub vocabulary_sha256: String,
    pub fidelity: String,
    pub valid: bool,
    pub fold_r2: Option<Vec<f64>>,
    pub oof_r2: Option<f64>,
    pub runtime_seconds: Option<f64>,
    pub failure: Option<Map<String, Value>>,
    pub selection_channel: String,
    pub recorded_at_ms: Option<u64>,
}

pub fn build_receipt(input: ReceiptInput) -> Result<Value> {
    let protocol = if input.fidelity == "screening" {
        json!({
            "fidelity": "screening",
            "folds": 3,
            "row_cap": 768,
            "timeout_seconds": 30,
            "omp_threads": 1,
            "train_only": true,
        })
    } else {
        json!({
            "fidelity": "exact",
            "folds": 5,
            "row_cap": null,
            "timeout_seconds": null,
            "omp_threads": 1,
            "train_only": true,
        })
    };
    let folds = input.fold_r2.unwrap_or_default();
    let robust = if input.valid {
        input
            .oof_r2
            .map(|oof| robust_score(oof, &folds))
            .transpose()?
    } else {
        None
    };
    let recorded_at_ms = match input.recorded_at_ms {
        Some(value) if value != 0 => value,
        _ => now_ms()?,
    };
    let mut receipt = json!({
        "schema": BROAD_RECEIPT_SCHEMA,
        "dataset_hash": input.dataset_hash,
        "train_sha256": input.train_sha256,
        "semantic_sha256": input.semantic_sha256,
        "canonical_genome": input.canonical_genome,
        "evaluator_release_sha256": input.evaluator_release_sha256,
        "fold_plan_sha256": input.fold_plan_sha256,
        "vocabulary_sha256": input.vocabulary_sha256,
        "fidelity": input.fidelity,
        "protocol": protocol,
        "valid": input.valid,
        "failure": input.failure,
        "fold_r2": folds,
        "oof_r2": input.oof_r2,
        "robust_score": robust,
        "runtime_seconds": input.runtime_seconds,
        "label_authority": LABEL_AUTHORITY,
        "test_table_accessed": false,
        "selection_channel": input.selection_channel,
        "recorded_at_ms": recorded_at_ms,
    });
    let digest = content_sha256(&receipt)?;
    receipt
        .as_object_mut()
        .context("receipt builder did not create an object")?
        .insert("receipt_sha256".to_owned(), Value::String(digest));
    validate_broad_receipt(&receipt)?;
    Ok(receipt)
}

#[derive(Clone, Debug)]
pub struct ReceiptStore {
    pub root: PathBuf,
    pub receipts: PathBuf,
}

impl ReceiptStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let receipts = root.join("receipts");
        fs::create_dir_all(&receipts)
            .with_context(|| format!("create receipt directory {}", receipts.display()))?;
        Ok(Self { root, receipts })
    }

    pub fn key_sha256(receipt: &Value) -> Result<String> {
        let key = receipt_dedup_key(receipt)?;
        content_sha256(&json!([key.0, key.1, key.2, key.3]))
    }

    pub fn append(&self, receipt: &Value) -> Result<(PathBuf, bool)> {
        validate_broad_receipt(receipt)?;
        let key = Self::key_sha256(receipt)?;
        let final_path = self.receipts.join(format!("{key}.json"));
        let mut payload = canonical_json_bytes(receipt)?;
        payload.push(b'\n');
        if final_path.exists() {
            ensure!(
                fs::read(&final_path)? == payload,
                "receipt dedup key has conflicting immutable evidence"
            );
            return Ok((final_path, false));
        }

        static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let mut temporary = None;
        for _ in 0..16 {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path =
                self.receipts
                    .join(format!(".{key}.{}.{}.tmp", std::process::id(), sequence));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    temporary = Some((path, file));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        let (temporary_path, mut file) =
            temporary.context("could not allocate a unique receipt temporary file")?;
        let result = (|| -> Result<bool> {
            file.write_all(&payload)?;
            file.sync_all()?;
            drop(file);
            let created = match fs::hard_link(&temporary_path, &final_path) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => return Err(error.into()),
            };
            ensure!(
                fs::read(&final_path)? == payload,
                "concurrent receipt append conflicts"
            );
            File::open(&self.receipts)?.sync_all()?;
            Ok(created)
        })();
        let _ = fs::remove_file(&temporary_path);
        Ok((final_path, result?))
    }

    pub fn import_all(&self, receipts: impl IntoIterator<Item = Value>) -> Result<Value> {
        let (mut inserted, mut duplicates) = (0_u64, 0_u64);
        for receipt in receipts {
            let (_, created) = self.append(&receipt)?;
            inserted += if created { 1 } else { 0 };
            duplicates += if created { 0 } else { 1 };
        }
        Ok(json!({"inserted": inserted, "duplicates": duplicates}))
    }

    pub fn iter(&self) -> Result<Vec<Value>> {
        let mut paths = fs::read_dir(&self.receipts)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.retain(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        });
        paths.sort();
        let mut receipts = Vec::with_capacity(paths.len());
        for path in paths {
            let value: Value = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("decode receipt {}", path.display()))?;
            validate_broad_receipt(&value)?;
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            ensure!(
                stem == Self::key_sha256(&value)?,
                "receipt filename does not bind dedup key: {}",
                path.display()
            );
            receipts.push(value);
        }
        Ok(receipts)
    }

    pub fn verify(&self) -> Result<Value> {
        let mut keys = BTreeSet::new();
        let mut digests = Vec::new();
        let (mut valid, mut failures) = (0_u64, 0_u64);
        for receipt in self.iter()? {
            let key = receipt_dedup_key(&receipt)?;
            ensure!(
                keys.insert(key),
                "receipt store contains a duplicate scientific key"
            );
            digests.push(string_field(&receipt, "receipt_sha256")?.to_owned());
            if bool_field(&receipt, "valid")? {
                valid += 1;
            } else {
                failures += 1;
            }
        }
        Ok(json!({
            "schema": STORE_VERIFICATION_SCHEMA,
            "receipts": keys.len(),
            "valid": valid,
            "failures": failures,
            "store_sha256": content_sha256(&Value::Array(
                digests.into_iter().map(Value::String).collect()
            ))?,
            "duplicate_keys": 0,
        }))
    }
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis()
        .try_into()?)
}

#[cfg(test)]
#[path = "receipts/tests.rs"]
mod tests;
