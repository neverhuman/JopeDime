//! Export the v2 direct multi-horizon reduction of every timeseries dataset
//! as encoder-shape tables (headerless numeric CSV, target last), so the
//! standard encoding pipeline (encode_batch.py) can produce sidecars for the
//! timeseries corpus. Raw series violate the encoder's assumptions (order-
//! carrying rows, no scaling, 8..=9,976 rows); the reduction moves temporal
//! structure into columns and scales targets into MASE units.
//!
//! Encoder-only concession: NaN feature cells (lags reaching beyond history)
//! are written as 0.0 — the same value feat-math's compatibility input fill
//! uses — because the encoder's minmax prep cannot carry NaN. The engine
//! never reads these exports; corpus_eval reduces from the raw corpus itself.
//!
//! Usage:
//!   jope_timeseries_export --registry "$JOPE_REGISTRY/../timeseries" \
//!                          --out "$JOPE_REGISTRY/../timeseries_reduced_v2"

use anyhow::{bail, Context, Result};
use crate::paths;
use crate::timeseries::{
    make_direct_forecast_rows, make_direct_reduction, read_timeseries_csv,
    validate_timeseries_data, TimeSeriesManifestSpec, TS_REDUCTION_V2,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn finite_cell(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn write_table(path: &Path, rows: &[Vec<f64>], targets: &[f64]) -> Result<()> {
    let mut out = String::new();
    for (row, target) in rows.iter().zip(targets) {
        for value in row {
            out.push_str(&format!("{:?},", finite_cell(*value)));
        }
        out.push_str(&format!("{target:?}\n"));
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

fn string_array(meta: &Value, key: &str) -> Vec<String> {
    meta.get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn export_dataset(source: &Path, destination: &Path) -> Result<(usize, usize)> {
    let meta_bytes = fs::read(source.join("meta.json"))?;
    let meta: Value = serde_json::from_slice(&meta_bytes)?;
    let required = |key: &str| -> Result<usize> {
        meta.get(key)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .with_context(|| format!("meta.json missing {key}"))
    };
    let spec = TimeSeriesManifestSpec {
        horizon: required("horizon")?,
        seasonality: required("seasonality")?,
        frequency: meta
            .get("frequency")
            .and_then(Value::as_str)
            .context("meta.json missing frequency")?
            .to_owned(),
        target_is_final_column: true,
        known_future_covariates: string_array(&meta, "known_dynamic_columns"),
        past_only_covariates: Vec::new(),
    };
    let train = read_timeseries_csv(&source.join("train.csv"))?;
    let test = read_timeseries_csv(&source.join("test.csv"))?;
    let data = validate_timeseries_data(train, test, &meta, &spec)?;
    let reduction = make_direct_reduction(&data)?;
    let forecast_rows = make_direct_forecast_rows(&data)?;
    let scaled_test_y: Vec<f64> = data
        .test
        .y
        .iter()
        .map(|value| value / reduction.target_scale)
        .collect();

    fs::create_dir_all(destination)?;
    write_table(&destination.join("train.csv"), &reduction.x, &reduction.y)?;
    write_table(
        &destination.join("test.csv"),
        &forecast_rows,
        &scaled_test_y,
    )?;
    let out_meta = json!({
        "task_type": "timeseries",
        "problem_type": "timeseries",
        "reduction_version": TS_REDUCTION_V2,
        "target_is_final_column": true,
        "headers_removed_from_csv": true,
        "nan_fill": 0.0,
        "target_scale": "divided by mase_denominator (MASE units)",
        "mase_denominator": data.mase_denominator,
        "horizon": data.horizon,
        "seasonality": data.seasonality,
        "frequency": data.frequency,
        "n_features": reduction.feature_names.len(),
        "feature_columns": reduction.feature_names,
        "counts": {
            "train_rows": reduction.y.len(),
            "test_rows": scaled_test_y.len(),
            "n_features": reduction.feature_names.len(),
        },
        "source": {
            "dataset_dir": source.display().to_string(),
            "train_sha256": sha256_file(&source.join("train.csv"))?,
            "test_sha256": sha256_file(&source.join("test.csv"))?,
            "meta_sha256": format!("{:x}", Sha256::digest(&meta_bytes)),
        },
    });
    let mut out_meta = out_meta;
    carry_synthetic_provenance(&meta, &mut out_meta);
    fs::write(
        destination.join("meta.json"),
        serde_json::to_string_pretty(&out_meta)?,
    )?;
    Ok((reduction.y.len(), scaled_test_y.len()))
}

/// Keys that must survive the reduction for a SYNTHETIC dataset.
///
/// `source_family` (`selfplay_data.rs:133-165`) branches on `synthetic_origin`
/// and then hashes `generator_recipe`. The reduced meta is rebuilt from scratch,
/// so without this a `synth_*` dataset arrives with no recipe, falls through to
/// hashing the whole meta -- which contains `source.dataset_dir` and per-dataset
/// shas -- and lands in a family of exactly one. That is the same defect that
/// left 16,890 synthetic regression datasets in 16,890 distinct families, making
/// the 10 % family-disjoint validation split hold out singletons and measure
/// nothing.
const SYNTHETIC_PROVENANCE_KEYS: [&str; 7] = [
    "origin",
    "synthetic_origin",
    "source_real_data",
    "generator_name",
    "generator_version",
    "generator_recipe",
    "generator_instance",
];

/// Copy synthetic provenance from the raw meta into the reduced meta.
///
/// Deliberately gated on `synthetic_origin == true`: the 42 real reduced metas
/// must stay **byte-identical**, because their `meta_sha256` is pinned inside
/// already-written `cache-manifest.v1.json` files and any drift fails the
/// expansion cache's identity check.
fn carry_synthetic_provenance(source_meta: &Value, out_meta: &mut Value) {
    if source_meta.get("synthetic_origin").and_then(Value::as_bool) != Some(true) {
        return;
    }
    let Some(target) = out_meta.as_object_mut() else {
        return;
    };
    for key in SYNTHETIC_PROVENANCE_KEYS {
        if let Some(value) = source_meta.get(key) {
            target.insert(key.to_owned(), value.clone());
        }
    }
}

pub fn main(args: Vec<String>) -> Result<()> {
    let mut registry = paths::quant_root().join("timeseries");
    let mut out_root = paths::quant_root().join("timeseries_reduced_v2");
    let mut arguments = args.into_iter().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments.next().context("flag requires a value")?;
        match flag.as_str() {
            "--registry" => registry = PathBuf::from(value),
            "--out" => out_root = PathBuf::from(value),
            other => bail!("unknown flag {other}"),
        }
    }
    let mut ids: Vec<String> = fs::read_dir(&registry)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    ids.sort();
    let (mut exported, mut failed) = (0usize, 0usize);
    for id in &ids {
        match export_dataset(&registry.join(id), &out_root.join(id)) {
            Ok((train_rows, test_rows)) => {
                exported += 1;
                println!("{id} train={train_rows} test={test_rows}");
            }
            Err(error) => {
                failed += 1;
                eprintln!("FAIL {id}: {error:#}");
            }
        }
    }
    println!(
        "exported={exported} failed={failed} out={}",
        out_root.display()
    );
    if failed > 0 {
        bail!("{failed} dataset(s) failed to export");
    }
    Ok(())
}
