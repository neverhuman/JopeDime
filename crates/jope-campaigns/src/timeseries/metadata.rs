use super::{TimeSeriesManifestSpec, TimeSeriesTable};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;

pub(super) fn validate_manifest_spec(spec: &TimeSeriesManifestSpec) -> Result<()> {
    if spec.horizon == 0 {
        bail!("time-series horizon must be positive");
    }
    if spec.seasonality == 0 {
        bail!("time-series seasonality must be positive");
    }
    if spec.frequency.trim().is_empty() {
        bail!("time-series frequency must be nonempty");
    }
    if !spec.target_is_final_column {
        bail!("time-series manifests must declare target_is_final_column=true");
    }
    if !spec.past_only_covariates.is_empty() {
        bail!("past-only covariates are not allowed in time-series CSV features");
    }
    let mut seen = BTreeSet::new();
    for name in &spec.known_future_covariates {
        if name.trim().is_empty() {
            bail!("known-future covariate names must be nonempty");
        }
        if !seen.insert(name) {
            bail!("duplicate known-future covariate '{name}'");
        }
    }
    Ok(())
}

pub(super) fn validate_meta(
    meta: &Value,
    train: &TimeSeriesTable,
    test: &TimeSeriesTable,
    spec: &TimeSeriesManifestSpec,
) -> Result<()> {
    if meta.get("task_type").and_then(Value::as_str) != Some("timeseries") {
        bail!("meta_json task_type must be timeseries");
    }
    if meta.get("target_is_final_column").and_then(Value::as_bool) != Some(true) {
        bail!("meta_json target_is_final_column must be true");
    }
    require_usize(meta, "horizon", spec.horizon)?;
    require_usize(meta, "seasonality", spec.seasonality)?;
    if meta.get("frequency").and_then(Value::as_str) != Some(spec.frequency.as_str()) {
        bail!("meta_json frequency does not match manifest frequency");
    }
    if let Some(features) = optional_usize(meta, "n_features") {
        if features != train.features {
            bail!(
                "meta_json n_features {features} does not match CSV width {}",
                train.features
            );
        }
    }
    if let Some(features) = number_at(meta, &["counts", "n_features"]) {
        if features as usize != train.features {
            bail!(
                "meta_json counts.n_features {} does not match CSV width {}",
                features as usize,
                train.features
            );
        }
    }
    check_optional_usize(meta, "n_train_rows", train.y.len())?;
    check_optional_usize(meta, "n_test_rows", test.y.len())?;
    check_nested_optional_usize(meta, &["counts", "train_rows"], train.y.len())?;
    check_nested_optional_usize(meta, &["counts", "test_rows"], test.y.len())?;

    let feature_columns = string_array_at(meta, &["feature_columns"])?;
    let known_dynamic = string_array_at(meta, &["known_dynamic_columns"])?;
    if feature_columns.len() != train.features {
        bail!(
            "meta_json feature_columns length {} does not match CSV width {}",
            feature_columns.len(),
            train.features
        );
    }
    if feature_columns != known_dynamic {
        bail!("time-series CSV features must be exactly known-future covariates");
    }
    if spec.known_future_covariates != known_dynamic {
        bail!("manifest known_future_covariates do not match meta_json known_dynamic_columns");
    }
    let past_excluded = string_array_at(meta, &["past_dynamic_columns_excluded"])?;
    if feature_columns
        .iter()
        .any(|column| past_excluded.iter().any(|past| past == column))
    {
        bail!("past-only covariate appears in time-series feature columns");
    }
    if let Some(policy) = meta
        .get("cleaning")
        .and_then(|value| value.get("past_only_covariates"))
        .and_then(Value::as_str)
    {
        if policy != "excluded" {
            bail!("meta_json cleaning.past_only_covariates must be excluded");
        }
    }
    Ok(())
}

fn require_usize(meta: &Value, field: &str, expected: usize) -> Result<()> {
    let Some(actual) = optional_usize(meta, field) else {
        bail!("meta_json missing numeric field {field}");
    };
    if actual != expected {
        bail!("meta_json {field} {actual} does not match manifest {expected}");
    }
    Ok(())
}

fn optional_usize(meta: &Value, field: &str) -> Option<usize> {
    meta.get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn check_optional_usize(meta: &Value, field: &str, expected: usize) -> Result<()> {
    if let Some(actual) = optional_usize(meta, field) {
        if actual != expected {
            bail!("meta_json {field} {actual} does not match CSV rows {expected}");
        }
    }
    Ok(())
}

fn check_nested_optional_usize(meta: &Value, path: &[&str], expected: usize) -> Result<()> {
    if let Some(actual) = number_at(meta, path) {
        if actual as usize != expected {
            bail!(
                "meta_json {} {} does not match CSV rows {expected}",
                path.join("."),
                actual as usize
            );
        }
    }
    Ok(())
}

pub(super) fn number_at(meta: &Value, path: &[&str]) -> Option<f64> {
    let mut cursor = meta;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_f64()
}

fn string_array_at(meta: &Value, path: &[&str]) -> Result<Vec<String>> {
    let mut cursor = meta;
    for key in path {
        let Some(next) = cursor.get(*key) else {
            return Ok(Vec::new());
        };
        cursor = next;
    }
    let Some(values) = cursor.as_array() else {
        bail!("meta_json {} must be an array", path.join("."));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("meta_json {} contains a non-string", path.join(".")))
        })
        .collect()
}
