//! Validator for the broad dataset catalog contract.
//!
//! This ports `python/src/jope/broad_shadow/contracts_catalog.py`. The Python
//! facade remains the active caller until its importers are migrated.

use anyhow::{bail, Result};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const BROAD_CATALOG_SCHEMA: &str = "BroadDatasetCatalog.v1";
pub const DATASET_EMBEDDING_DIMENSIONS: u64 = 1536;
pub const TARGET_SUMMARY_DIMENSIONS: u64 = 512;
pub const REAL_SPLIT_COUNTS: [(&str, u64); 4] = [
    ("cold_development", 119),
    ("double_holdout", 49),
    ("prospective_gate", 64),
    ("train", 2_090),
];
const REAL_SPLITS: [&str; 4] = [
    "cold_development",
    "double_holdout",
    "prospective_gate",
    "train",
];
const SYNTHETIC_SPLITS: [&str; 3] = ["test", "train", "validation"];
const CATALOG_FIELDS: &[&str] = &[
    "schema",
    "catalog_sha256",
    "created_at_ms",
    "dataset_embedding_dimensions",
    "target_encoding_dimensions",
    "foundation_cache_count",
    "hyperion_complete_count",
    "real_split_counts",
    "synthetic_split_counts",
    "synthetic_family_split_counts",
    "datasets",
];
const CATALOG_ROW_FIELDS: &[&str] = &[
    "dataset_hash",
    "train_sha256",
    "domain",
    "source_family",
    "source_family_authority_sha256",
    "generator_lineage",
    "split",
    "rows",
    "features",
    "target_shape",
    "dataset_encoding",
    "target_encoding",
    "foundation_complete",
    "hyperion_complete",
];
const ENCODING_FIELDS: &[&str] = &["sha256", "path", "offset_bytes", "dimensions", "dtype"];

/// Validate a `BroadDatasetCatalog.v1` payload, including optional production floors.
pub fn validate_broad_catalog(payload: &Value, production: bool) -> Result<()> {
    let manifest = exact_fields(payload, CATALOG_FIELDS, "broad dataset catalog")?;
    if manifest["schema"] != BROAD_CATALOG_SCHEMA {
        bail!("unsupported broad dataset catalog schema");
    }
    content_hash(manifest, "catalog_sha256", "broad dataset catalog")?;
    positive_integer(&manifest["created_at_ms"], "created_at_ms", false)?;
    if manifest["dataset_embedding_dimensions"].as_u64() != Some(DATASET_EMBEDDING_DIMENSIONS)
        || manifest["target_encoding_dimensions"].as_u64() != Some(TARGET_SUMMARY_DIMENSIONS)
    {
        bail!("broad catalog requires frozen 1536+512 encodings");
    }
    let foundation_count = positive_integer(
        &manifest["foundation_cache_count"],
        "foundation_cache_count",
        true,
    )?;
    let declared_hyperion = positive_integer(
        &manifest["hyperion_complete_count"],
        "hyperion_complete_count",
        true,
    )?;
    let rows = manifest["datasets"]
        .as_array()
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| anyhow::anyhow!("broad catalog datasets must be a nonempty list"))?;

    let mut previous: Option<&str> = None;
    let mut real_counts = BTreeMap::<String, u64>::new();
    let mut synthetic_counts = BTreeMap::<String, u64>::new();
    let mut family_split = BTreeMap::<String, String>::new();
    let mut hyperion_count = 0;
    let mut foundation_complete = 0;
    for value in rows {
        let row = exact_fields(value, CATALOG_ROW_FIELDS, "broad catalog row")?;
        let dataset = dataset_hash(&row["dataset_hash"])?;
        if previous.is_some_and(|prior| prior >= dataset) {
            bail!("broad catalog datasets must be strictly sorted and unique");
        }
        previous = Some(dataset);
        sha256(&row["train_sha256"], "train_sha256")?;
        let family = slug(&row["source_family"], "source_family")?;
        sha256(
            &row["source_family_authority_sha256"],
            "source_family_authority_sha256",
        )?;
        positive_integer(&row["rows"], "rows", false)?;
        positive_integer(&row["features"], "features", false)?;
        slug(&row["target_shape"], "target_shape")?;
        validate_encoding_ref(
            &row["dataset_encoding"],
            "dataset_encoding",
            DATASET_EMBEDDING_DIMENSIONS,
        )?;
        validate_encoding_ref(
            &row["target_encoding"],
            "target_encoding",
            TARGET_SUMMARY_DIMENSIONS,
        )?;
        let foundation = row["foundation_complete"]
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("catalog completion flags must be booleans"))?;
        let hyperion = row["hyperion_complete"]
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("catalog completion flags must be booleans"))?;
        foundation_complete += u64::from(foundation);
        hyperion_count += u64::from(hyperion);

        let domain = row["domain"].as_str().unwrap_or("");
        let split = row["split"].as_str().unwrap_or("");
        match domain {
            "real" => {
                if !REAL_SPLITS.contains(&split) || !row["generator_lineage"].is_null() {
                    bail!("real datasets require a real split and no generator lineage");
                }
                *real_counts.entry(split.to_owned()).or_default() += 1;
            }
            "synthetic" => {
                if !SYNTHETIC_SPLITS.contains(&split) {
                    bail!("synthetic dataset split is invalid");
                }
                let lineage = slug(&row["generator_lineage"], "generator_lineage")?;
                if family_split
                    .insert(lineage.to_owned(), split.to_owned())
                    .is_some_and(|prior| prior != split)
                {
                    bail!("normalized generator lineage crosses synthetic splits");
                }
                *synthetic_counts.entry(split.to_owned()).or_default() += 1;
            }
            _ => bail!("dataset domain must be real or synthetic"),
        }
        if family.is_empty() {
            bail!("validated source family unexpectedly empty");
        }
    }

    if manifest["real_split_counts"] != count_object(&REAL_SPLITS, &real_counts)
        || manifest["synthetic_split_counts"] != count_object(&SYNTHETIC_SPLITS, &synthetic_counts)
        || manifest["synthetic_family_split_counts"] != family_count_object(&family_split)
    {
        bail!("broad catalog declared counts differ from its rows");
    }
    if declared_hyperion != hyperion_count {
        bail!("broad catalog Hyperion count differs from its rows");
    }
    if foundation_count < foundation_complete {
        bail!("foundation cache count is smaller than bound catalog encodings");
    }
    if production {
        let declared_real = &manifest["real_split_counts"];
        let expected_real = count_object(
            &REAL_SPLITS,
            &REAL_SPLIT_COUNTS
                .iter()
                .map(|(key, count)| ((*key).to_owned(), *count))
                .collect(),
        );
        if declared_real != &expected_real {
            bail!("production real split must remain 2090/119/64/49");
        }
        if SYNTHETIC_SPLITS
            .iter()
            .any(|split| synthetic_counts.get(*split).copied().unwrap_or(0) == 0)
            || SYNTHETIC_SPLITS
                .iter()
                .any(|split| !family_split.values().any(|assigned| assigned == split))
        {
            bail!("production synthetic authority requires all three splits");
        }
        if foundation_count < 27_875 {
            bail!("production catalog must bind the 27,875 foundation cache");
        }
        if hyperion_count < 24_177 {
            bail!("pilot cannot start below the frozen Hyperion-ready floor");
        }
    }
    Ok(())
}

fn exact_fields<'a>(
    value: &'a Value,
    expected: &[&str],
    name: &str,
) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{name} has unknown or missing fields"))?;
    let keys: BTreeSet<_> = object.keys().map(String::as_str).collect();
    if keys != expected.iter().copied().collect() {
        bail!("{name} has unknown or missing fields");
    }
    Ok(object)
}

fn positive_integer(value: &Value, name: &str, allow_zero: bool) -> Result<u64> {
    let number = value
        .as_u64()
        .filter(|n| allow_zero || *n > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{name} must be a {} integer",
                if allow_zero {
                    "nonnegative"
                } else {
                    "positive"
                }
            )
        })?;
    Ok(number)
}

fn sha256<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let text = value
        .as_str()
        .filter(|text| valid_hex(text, 64))
        .ok_or_else(|| anyhow::anyhow!("{name} must be a lowercase SHA-256 digest"))?;
    Ok(text)
}

fn dataset_hash<'a>(value: &'a Value) -> Result<&'a str> {
    let text = value
        .as_str()
        .filter(|text| (text.len() == 16 || text.len() == 64) && valid_hex(text, text.len()))
        .ok_or_else(|| {
            anyhow::anyhow!("dataset_hash must be a 16- or 64-character lowercase dataset hash")
        })?;
    Ok(text)
}

fn valid_hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn slug<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let text = value
        .as_str()
        .filter(|text| {
            !text.is_empty()
                && text.split(['-', '_', '.']).all(|part| {
                    !part.is_empty()
                        && part
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
                })
        })
        .ok_or_else(|| anyhow::anyhow!("{name} must be a normalized lowercase slug"))?;
    Ok(text)
}

fn validate_encoding_ref(value: &Value, name: &str, dimensions: u64) -> Result<()> {
    let row = exact_fields(value, ENCODING_FIELDS, name)?;
    sha256(&row["sha256"], &format!("{name}.sha256"))?;
    safe_relative_path(&row["path"], &format!("{name}.path"))?;
    let offset = positive_integer(&row["offset_bytes"], &format!("{name}.offset_bytes"), true)?;
    if offset % 4 != 0 {
        bail!("{name}.offset_bytes must be float32 aligned");
    }
    if row["dimensions"].as_u64() != Some(dimensions) || row["dtype"] != "float32_le" {
        bail!("{name} dimensions or dtype differs from frozen authority");
    }
    Ok(())
}

fn safe_relative_path(value: &Value, name: &str) -> Result<()> {
    let path = value
        .as_str()
        .filter(|path| !path.is_empty() && !path.starts_with('/'))
        .ok_or_else(|| anyhow::anyhow!("{name} must be a nonempty relative path"))?;
    if path.split('/').any(|part| part == "..") {
        bail!("{name} must be a normal relative path");
    }
    Ok(())
}

fn content_hash(payload: &Map<String, Value>, field: &str, name: &str) -> Result<()> {
    let claimed = sha256(&payload[field], field)?;
    let mut unhashed = payload.clone();
    unhashed.remove(field);
    let bytes = serde_json::to_vec(&Value::Object(unhashed))?;
    let actual = hex::encode(Sha256::digest(bytes));
    if claimed != actual {
        bail!("{name} content hash differs");
    }
    Ok(())
}

fn count_object(keys: &[&str], counts: &BTreeMap<String, u64>) -> Value {
    let map = keys
        .iter()
        .map(|key| {
            (
                (*key).to_owned(),
                Value::from(counts.get(*key).copied().unwrap_or(0)),
            )
        })
        .collect::<Map<_, _>>();
    Value::Object(map)
}

fn family_count_object(families: &BTreeMap<String, String>) -> Value {
    let counts = SYNTHETIC_SPLITS
        .iter()
        .map(|split| {
            (
                (*split).to_owned(),
                families
                    .values()
                    .filter(|assigned| assigned.as_str() == *split)
                    .count() as u64,
            )
        })
        .collect();
    count_object(&SYNTHETIC_SPLITS, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_or_extra_catalog_fields() {
        let error =
            validate_broad_catalog(&serde_json::json!({"schema": BROAD_CATALOG_SCHEMA}), false)
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "broad dataset catalog has unknown or missing fields"
        );
    }

    #[test]
    fn rejects_unaligned_encoding_offsets() {
        let encoding = serde_json::json!({
            "sha256": "a".repeat(64), "path": "cache/encoding.bin", "offset_bytes": 2,
            "dimensions": DATASET_EMBEDDING_DIMENSIONS, "dtype": "float32_le"
        });
        let error =
            validate_encoding_ref(&encoding, "dataset_encoding", DATASET_EMBEDDING_DIMENSIONS)
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "dataset_encoding.offset_bytes must be float32 aligned"
        );
    }

    #[test]
    fn rejects_parent_traversal_in_encoding_paths() {
        let error = safe_relative_path(
            &Value::String("cache/../secret".into()),
            "dataset_encoding.path",
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "dataset_encoding.path must be a normal relative path"
        );
    }
}
