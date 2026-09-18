//! Build a catalog row from a dataset directory's `meta.json` and the hashed files.

use crate::catalog::{
    Alias, CatalogRow, FileEntry, Files, Generator, Ingest, IngestSource, License, LicenseStatus,
    Origin, Provenance, Shape, Split, Target, Task, CATALOG_SCHEMA_ID,
};
use crate::hash::FileDigest;
use serde_json::Value;

/// The three digests of one dataset directory.
#[derive(Clone, Debug)]
pub struct Hashed {
    pub train: FileDigest,
    pub test: FileDigest,
    pub meta: FileDigest,
}

/// Why a directory could not become a catalog row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reject {
    pub reason: &'static str,
    pub detail: String,
}

fn reject(reason: &'static str, detail: impl Into<String>) -> Reject {
    Reject {
        reason,
        detail: detail.into(),
    }
}

fn first_str<'a>(meta: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| meta.get(k).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
}

fn first_u64(meta: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|k| meta.get(k).and_then(Value::as_u64))
}

fn nested_str<'a>(meta: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(meta, |v, k| v.get(k))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn nested_u64(meta: &Value, path: &[&str]) -> Option<u64> {
    path.iter()
        .try_fold(meta, |v, k| v.get(k))
        .and_then(Value::as_u64)
}

/// Licence evidence for whole platforms whose terms were verified once (WP-D6 backfills
/// per-dataset licences; everything else stays `unknown` with the upstream string as a note).
pub fn license_from_meta(platform: Option<&str>, source_license: Option<&str>) -> License {
    let note = source_license.map(|s| format!("source_license: {s}"));
    match platform {
        Some("pmlb") => License {
            spdx: Some("MIT".into()),
            status: LicenseStatus::Recorded,
            evidence_url: Some("https://github.com/EpistasisLab/pmlb/blob/master/LICENSE".into()),
            text_sha256: None,
            note,
        },
        _ => License::unknown(note),
    }
}

/// Task named by `meta.json` (`task_type`), if any.
pub fn task_from_meta(meta: &Value) -> Option<Task> {
    first_str(meta, &["task_type", "task"]).and_then(Task::parse)
}

/// Build the row for directory `dir_name` (which must match `^(synth_)?[0-9a-f]{16}$`).
///
/// Rejects when `meta.json` has no `sha256`, when `sha256[:16]` differs from the directory
/// hash, or when `hash_algorithm` is present and does not start with `sha256:first16`.
pub fn build_row(
    dir_name: &str,
    task: Task,
    meta: &Value,
    hashed: &Hashed,
    ingest: Ingest,
) -> Result<CatalogRow, Reject> {
    let dir_hash = crate::dataset_dir_hash(dir_name).ok_or_else(|| {
        reject(
            "dir_name_pattern",
            format!("{dir_name} is not (synth_)?<16hex>"),
        )
    })?;
    let Some(sha) = first_str(meta, &["sha256"]) else {
        return Err(reject("meta_sha256_missing", "meta.json has no sha256"));
    };
    if sha.len() != 64
        || !sha
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(reject("meta_sha256_malformed", sha));
    }
    if &sha[..16] != dir_hash {
        return Err(reject(
            "meta_sha256_mismatch",
            format!("meta.sha256[:16]={} dir={dir_hash}", &sha[..16]),
        ));
    }
    if let Some(alg) = first_str(meta, &["hash_algorithm"]) {
        // Some binary sets spell it "sha256:first16 over source hashes, target, ..."; the
        // algorithm is the same.
        if !alg.starts_with("sha256:first16") {
            return Err(reject("hash_algorithm_unexpected", alg));
        }
    }
    if let Some(declared) = first_str(meta, &["dataset_hash"]) {
        if declared != dir_hash {
            return Err(reject(
                "meta_dataset_hash_mismatch",
                format!("meta.dataset_hash={declared} dir={dir_hash}"),
            ));
        }
    }
    let generator = first_str(meta, &["generator_name"]).map(|name| Generator {
        name: name.to_string(),
        version: first_str(meta, &["generator_version"])
            .unwrap_or("unknown")
            .to_string(),
        seed: first_u64(meta, &["generator_seed"]),
        attempt_seed: first_u64(meta, &["generator_attempt_seed"]),
        recipe_sha256: meta
            .get("generator_recipe")
            .filter(|v| !v.is_null())
            .map(|v| crate::hash::sha256_hex(canonical_json(v).as_bytes())),
        script_sha256: None,
    });
    let origin = match first_str(meta, &["origin"]) {
        Some("synthetic") => Origin::Synthetic,
        Some("real") => Origin::Real,
        _ if generator.is_some() || dir_name.starts_with("synth_") => Origin::Synthetic,
        _ => Origin::Real,
    };
    let generator = match (origin, generator) {
        (Origin::Synthetic, None) => Some(Generator {
            name: "unknown".into(),
            version: "unknown".into(),
            seed: None,
            attempt_seed: None,
            recipe_sha256: None,
            script_sha256: None,
        }),
        (Origin::Real, _) => None,
        (_, g) => g,
    };
    let columns = hashed.train.columns;
    if columns == 0 || hashed.test.columns != columns {
        return Err(reject(
            "column_mismatch",
            format!(
                "train columns {} test columns {}",
                columns, hashed.test.columns
            ),
        ));
    }
    let categorical = meta
        .get("feature_encodings")
        .and_then(Value::as_object)
        .map(|m| {
            m.values()
                .filter(|e| {
                    e.get("kind")
                        .and_then(Value::as_str)
                        .is_some_and(|k| k != "numeric")
                })
                .count() as u64
        })
        .unwrap_or(0);
    let platform = first_str(meta, &["source_platform", "source"]);
    let dataset_id = first_str(meta, &["source_dataset_id", "source_id"])
        .map(str::to_string)
        .or_else(|| {
            meta.get("source_openml_data_id")
                .and_then(Value::as_u64)
                .map(|n| n.to_string())
        });
    let license = license_from_meta(platform, first_str(meta, &["source_license"]));
    let file = |d: &FileDigest, csv: bool| FileEntry {
        sha256: d.sha256.clone(),
        bytes: d.bytes,
        rows: csv.then_some(d.rows),
        columns: csv.then_some(d.columns),
        has_header: csv.then_some(false),
    };
    Ok(CatalogRow {
        schema: CATALOG_SCHEMA_ID.into(),
        id: dir_name.to_string(),
        dataset_hash: dir_hash.to_string(),
        canonical_sha256: sha.to_string(),
        display_name: first_str(meta, &["display_name", "source_dataset_name"]).map(str::to_string),
        task,
        origin,
        shape: Shape {
            rows: hashed.train.rows + hashed.test.rows,
            features: columns - 1,
            categorical_features: categorical,
            columns,
        },
        target: Target {
            name: first_str(meta, &["target_column"])
                .unwrap_or("target")
                .to_string(),
            index: -1,
            target_type: task.target_type(),
        },
        files: Files {
            train: file(&hashed.train, true),
            test: file(&hashed.test, true),
            meta: Some(file(&hashed.meta, false)),
        },
        split: Split {
            method: nested_str(meta, &["sampling", "method"])
                .or_else(|| nested_str(meta, &["split", "method"]))
                .unwrap_or("unknown")
                .to_string(),
            seed: nested_u64(meta, &["sampling", "seed"])
                .or_else(|| nested_u64(meta, &["split", "seed"])),
            train_rows: hashed.train.rows,
            test_rows: hashed.test.rows,
        },
        provenance: Provenance {
            platform: platform.map(str::to_string),
            dataset_id,
            url: first_str(meta, &["source_url"]).map(str::to_string),
            retrieved_at: first_str(meta, &["retrieved_at", "created_at"]).map(str::to_string),
            dedupe_fingerprint: first_str(meta, &["dedupe_fingerprint"]).map(str::to_string),
            citation: first_str(meta, &["citation"]).map(str::to_string),
            license,
        },
        generator,
        collections: Vec::new(),
        aliases: Vec::new(),
        ingest: Some(ingest),
    })
}

/// Deterministic JSON (serde_json keeps object keys in insertion order; sort them).
pub fn canonical_json(v: &Value) -> String {
    fn sort(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<_> = m.keys().collect();
                keys.sort();
                Value::Object(keys.into_iter().map(|k| (k.clone(), sort(&m[k]))).collect())
            }
            Value::Array(a) => Value::Array(a.iter().map(sort).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(v)).unwrap_or_default()
}

/// Aliases for a duplicate copy of a row's files.
pub fn aliases_for(relpath: &str, hashed: &Hashed, source: IngestSource) -> Vec<Alias> {
    let src = serde_json::to_value(source)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string));
    [
        ("train.csv", &hashed.train),
        ("test.csv", &hashed.test),
        ("meta.json", &hashed.meta),
    ]
    .into_iter()
    .map(|(name, d)| Alias {
        path: format!("{relpath}/{name}"),
        sha256: d.sha256.clone(),
        source: src.clone(),
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn digest(sha: &str, rows: u64, columns: u64) -> FileDigest {
        FileDigest {
            sha256: sha.repeat(64),
            bytes: 10,
            rows,
            columns,
            first_line: String::new(),
        }
    }

    fn hashed() -> Hashed {
        Hashed {
            train: digest("a", 8, 4),
            test: digest("b", 2, 4),
            meta: digest("c", 0, 0),
        }
    }

    fn ingest() -> Ingest {
        Ingest {
            source: IngestSource::Quant,
            relpath: "regression/x".into(),
            ingested_at: "t".into(),
            tool_version: "v".into(),
        }
    }

    #[test]
    fn real_row() {
        let meta = json!({
            "sha256": "1cb97b1d7314d43dee07a1f78f16d74ea27faa0653fd135dfe0084ada3369517",
            "hash_algorithm": "sha256:first16", "origin": "real", "source_platform": "pmlb",
            "source_dataset_id": "229_pwLinear", "source_license": "MIT (PMLB)",
            "sampling": {"method": "quota", "seed": 495737872},
            "feature_encodings": {"a": {"kind": "numeric"}, "b": {"kind": "categorical_codes"}}
        });
        let row = build_row(
            "1cb97b1d7314d43d",
            Task::Regression,
            &meta,
            &hashed(),
            ingest(),
        )
        .unwrap();
        assert_eq!(
            row.shape,
            Shape {
                rows: 10,
                features: 3,
                categorical_features: 1,
                columns: 4
            }
        );
        assert_eq!(row.provenance.license.status, LicenseStatus::Recorded);
        assert_eq!(row.split.seed, Some(495_737_872));
        assert!(row.generator.is_none());
        assert!(
            row.invariant_errors().is_empty(),
            "{:?}",
            row.invariant_errors()
        );
        let v = serde_json::to_value(&row).unwrap();
        assert!(CatalogRow::validate_value(&v).is_ok());
    }

    #[test]
    fn synthetic_row_and_rejects() {
        let meta = json!({
            "sha256": "00015d6f62b800ebb08ec8ad9eb107b48234370b37db350d5f8242e8387ea694",
            "generator_name": "synthetic_binary_mixed_tabular", "generator_version": "0.2.0",
            "generator_seed": 1717423427, "generator_recipe": {"b": 1, "a": 2}
        });
        let row = build_row("00015d6f62b800eb", Task::Binary, &meta, &hashed(), ingest()).unwrap();
        assert_eq!(row.origin, Origin::Synthetic);
        let g = row.generator.as_ref().unwrap();
        assert_eq!(
            g.recipe_sha256.as_deref(),
            Some(crate::hash::sha256_hex(b"{\"a\":2,\"b\":1}").as_str())
        );
        assert!(row.invariant_errors().is_empty());

        let r = build_row(
            "00015d6f62b800eb",
            Task::Binary,
            &json!({}),
            &hashed(),
            ingest(),
        )
        .unwrap_err();
        assert_eq!(r.reason, "meta_sha256_missing");
        let r =
            build_row("00015d6f62b800ec", Task::Binary, &meta, &hashed(), ingest()).unwrap_err();
        assert_eq!(r.reason, "meta_sha256_mismatch");
        let mut bad = meta.clone();
        bad["hash_algorithm"] = json!("md5");
        assert_eq!(
            build_row("00015d6f62b800eb", Task::Binary, &bad, &hashed(), ingest())
                .unwrap_err()
                .reason,
            "hash_algorithm_unexpected"
        );
    }
}
