//! CPU Hyperion encode of an admitted dataset. Requires the served weight bundle; does not download.

use crate::kernel::read_dataset;
use crate::schema::{Dataset, Kind, Split};
use anyhow::{bail, Context, Result};
use hyperion::data::Matrix;
use hyperion::hyperion::{
    default_manifest_path, default_weights_path, fit_transform_pair, HyperionConfig, HyperionMode,
    WEIGHTS_DIR_ENV,
};
use hyperion::progress::NullSink;
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(Debug, Serialize)]
pub struct EncodeReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub input_cols: usize,
    pub output_cols: usize,
    pub weights: String,
}

pub fn encode_dataset(path: &Path) -> Result<EncodeReport> {
    // Bounded, no-symlink read plus full schema validation, as for fit/search/route.
    let data = read_dataset(path)?;
    let names: Vec<String> = data
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect();
    for column in &data.columns {
        if column.kind != Kind::Integer || column.nullable {
            bail!("encode currently admits integer, non-null features only");
        }
    }
    let train = matrix(&data, Split::Train)?;
    // The hold-out is the validation split only; there is no fallback to test rows.
    let hold = matrix(&data, Split::Validation)?;
    let cfg = HyperionConfig {
        mode: HyperionMode::On,
        weights_path: default_weights_path(),
        manifest_path: default_manifest_path(),
        ..HyperionConfig::default()
    };
    let encoded = fit_transform_pair(&train, &hold, &names, &cfg, &NullSink, Instant::now())
        .with_context(|| {
            format!(
                "Hyperion CPU encode needs {WEIGHTS_DIR_ENV} (directory with model.safetensors and manifest.json)"
            )
        })?;
    anyhow::ensure!(
        encoded.state.is_some(),
        "Hyperion produced no encoder state"
    );
    Ok(EncodeReport {
        schema: "jope.encode.v1",
        ok: true,
        dataset_id: data.id,
        train_rows: encoded.train.n,
        hold_rows: encoded.hold.n,
        input_cols: train.c,
        output_cols: encoded.train.c,
        weights: cfg.weights_path.display().to_string(),
    })
}

fn matrix(data: &Dataset, split: Split) -> Result<Matrix> {
    let rows: Vec<_> = data.rows.iter().filter(|row| row.split == split).collect();
    anyhow::ensure!(!rows.is_empty(), "no {split:?} rows");
    let n = rows.len();
    let c = data.columns.len();
    let mut values = vec![0f32; n * c];
    for (index, row) in rows.iter().enumerate() {
        anyhow::ensure!(row.features.len() == c, "feature width");
        for (column, feature) in row.features.iter().enumerate() {
            let value = feature
                .as_i64()
                .with_context(|| format!("non-integer feature in {}", data.id))?;
            values[column * n + index] = value as f32;
        }
    }
    Ok(Matrix::new(values, n, c))
}
