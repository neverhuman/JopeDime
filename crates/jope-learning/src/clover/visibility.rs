use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Visibility {
    pub task: String,
    pub family: String,
    pub split_sha256: String,
    pub fold: u8,
    pub available_through: u64,
    pub fitting_rows: Vec<String>,
    pub forbidden_rows: BTreeSet<String>,
    /// Target-supervised encoders may see only the declared fitting labels.
    pub training_labels: bool,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub task: String,
    pub family: String,
    pub role: String,
    pub split_sha256: String,
    pub fold: u8,
    pub available_at: u64,
    pub features: Vec<f64>,
    pub label: f64,
}

#[derive(Debug, Serialize)]
pub struct EncoderInput {
    pub features: Vec<Vec<f64>>,
    pub labels: Option<Vec<f64>>,
    pub visibility_sha256: String,
}

/// Project before fitting preprocessing or calling an encoder. Only fitting payloads
/// are looked up. The roster/metadata must come from the trusted split authority;
/// this function cannot authenticate arbitrary external encoder code or feature lineage.
pub fn project_encoder_input(
    visibility: &Visibility,
    rows: &BTreeMap<String, Observation>,
) -> Result<EncoderInput> {
    ensure!(
        !visibility.task.trim().is_empty()
            && !visibility.family.trim().is_empty()
            && is_hex64(&visibility.split_sha256)
            && visibility.fold < 5,
        "incomplete visibility identity"
    );
    let ids: BTreeSet<_> = visibility.fitting_rows.iter().collect();
    ensure!(
        !ids.is_empty() && ids.len() == visibility.fitting_rows.len(),
        "empty or duplicate fitting rows"
    );
    ensure!(
        ids.iter()
            .all(|id| !visibility.forbidden_rows.contains(*id)),
        "forbidden fitting row"
    );
    let fitting = visibility
        .fitting_rows
        .iter()
        .map(|id| {
            ensure!(!id.trim().is_empty(), "empty row identity");
            let row = rows
                .get(id)
                .with_context(|| format!("missing fitting row {id}"))?;
            ensure!(
                row.task == visibility.task
                    && row.family == visibility.family
                    && row.role == "training"
                    && row.split_sha256 == visibility.split_sha256
                    && row.fold == visibility.fold
                    && row.available_at <= visibility.available_through,
                "label/future boundary violation: {id}"
            );
            ensure!(
                !row.features.is_empty() && row.features.iter().all(|v| v.is_finite()),
                "invalid features"
            );
            if visibility.training_labels {
                ensure!(row.label.is_finite(), "invalid fitting label");
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?;
    let width = fitting[0].features.len();
    ensure!(
        fitting.iter().all(|r| r.features.len() == width),
        "feature width mismatch"
    );
    Ok(EncoderInput {
        features: fitting.iter().map(|row| row.features.clone()).collect(),
        labels: visibility
            .training_labels
            .then(|| fitting.iter().map(|row| row.label).collect()),
        visibility_sha256: canonical_sha256(visibility)?,
    })
}
