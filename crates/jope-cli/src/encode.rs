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
/// Summary of encoding one admitted dataset with the configured Hyperion weights.
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

/// Encode a dataset file using the served CPU Hyperion path; this does not download weights.
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

/// Explicit landmark options select Dope. Omitted options retain the served Hyperion path.
pub fn encode_dataset_with_landmarks(
    path: &Path,
    landmarks: &jope_runtime::landmark_args::LandmarkArgs,
    device: &str,
) -> Result<EncodeReport> {
    landmarks.validate()?;
    if !landmarks.enabled() {
        return encode_dataset(path);
    }
    #[cfg(not(feature = "dope-encoding"))]
    {
        let _ = (path, device);
        bail!("landmark encoding requires a dope-cuda build")
    }
    #[cfg(feature = "dope-encoding")]
    {
        dope_encode_dataset(path, landmarks, device)
    }
}

#[cfg(feature = "dope-encoding")]
fn dope_encode_dataset(
    path: &Path,
    landmarks: &jope_runtime::landmark_args::LandmarkArgs,
    device: &str,
) -> Result<EncodeReport> {
    use starforge::dope_encoder::{
        encode_training, product_session_options, start_training_with_options, DopeEncoder, Head,
        TrainingInput,
    };
    let companion = landmarks
        .landmark_receipt
        .clone()
        .unwrap_or_else(|| path.with_extension("landmark-encoding.json"));
    let protected = landmarks.protected_inputs();
    let mut inputs = vec![path];
    inputs.extend(protected.iter().map(std::path::PathBuf::as_path));
    if let Some(events) = &landmarks.landmark_events {
        inputs.push(events);
    }
    jope_runtime::output_safety::refuse_aliasing(&companion, &inputs)?;
    if let Some(events) = &landmarks.landmark_events {
        let mut inputs = vec![path, companion.as_path()];
        inputs.extend(protected.iter().map(std::path::PathBuf::as_path));
        jope_runtime::output_safety::refuse_aliasing(events, &inputs)?;
    }
    let data = read_dataset(path)?;
    let rows: Vec<_> = data
        .rows
        .iter()
        .filter(|r| r.split == Split::Train)
        .collect();
    let width = data.columns.len();
    let mut categories = Vec::new();
    let mut maps = Vec::new();
    for (j, column) in data.columns.iter().enumerate() {
        let map: std::collections::BTreeMap<String, usize> = if column.kind == Kind::Categorical {
            let values: std::collections::BTreeSet<_> = rows
                .iter()
                .filter_map(|r| r.features[j].as_str().map(str::to_owned))
                .collect();
            values
                .into_iter()
                .enumerate()
                .map(|(i, s)| (s, i))
                .collect()
        } else {
            Default::default()
        };
        categories.push((column.kind == Kind::Categorical).then_some(map.len()));
        maps.push(map);
    }
    let x = rows
        .iter()
        .map(|row| {
            (0..width)
                .map(|j| {
                    if data.columns[j].kind == Kind::Categorical {
                        row.features[j]
                            .as_str()
                            .and_then(|s| maps[j].get(s))
                            .map_or(f64::NAN, |v| *v as f64)
                    } else {
                        row.features[j].as_f64().unwrap_or(f64::NAN)
                    }
                })
                .collect()
        })
        .collect();
    let classes: std::collections::BTreeMap<_, _> = rows
        .iter()
        .map(|r| r.target)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    let regression = matches!(
        data.task,
        crate::schema::Task::Regression | crate::schema::Task::TimeSeries
    );
    let y = rows
        .iter()
        .map(|r| {
            if regression {
                r.target as f64
            } else {
                classes[&r.target] as f64
            }
        })
        .collect();
    let head = if regression {
        Head::Regression
    } else if classes.len() == 2 {
        Head::Binary
    } else {
        Head::Multiclass(classes.len())
    };
    let input = TrainingInput {
        x,
        y,
        ids: rows.iter().map(|r| r.id as u64).collect(),
        categories,
        head,
        raw_cells: Some(
            rows.iter()
                .map(|r| {
                    r.features
                        .iter()
                        .map(|v| match v {
                            serde_json::Value::Null => String::new(),
                            serde_json::Value::String(s) => s.clone(),
                            v => v.to_string(),
                        })
                        .collect()
                })
                .collect(),
        ),
        fold: Some("encode-fit".into()),
        source_digest: Some(jope_runtime::sha256_hex(&serde_json::to_vec(&(
            data.task,
            data.columns
                .iter()
                .map(|c| (&c.name, c.kind, c.nullable))
                .collect::<Vec<_>>(),
            &rows,
        ))?)),
    };
    let weights = landmarks
        .dope_weights
        .as_ref()
        .context("Dope weights required (--dope-weights or DOPE_WEIGHTS)")?;
    let selected = if device == "cpu" {
        candle_cpu()
    } else {
        let index = device
            .strip_prefix("cuda:")
            .context("device must be cpu or cuda:N")?
            .parse()?;
        DopeEncoder::cuda(index)?
    };
    let encoder = DopeEncoder::load(weights, &selected)?;
    let options = product_session_options(landmarks, &encoder)?;
    let session = start_training_with_options(input, encoder, landmarks, options)?;
    let result = encode_training(session, landmarks.landmark_events.as_deref())?;
    let companion = landmarks
        .landmark_receipt
        .clone()
        .unwrap_or_else(|| path.with_extension("landmark-encoding.json"));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&companion)?;
    serde_json::to_writer_pretty(file, &result.final_snapshot.receipt)?;
    anyhow::ensure!(
        matches!(
            result.outcome,
            starforge::dope_encoder::SessionOutcome::Complete
        ),
        "encoding ended with {:?}; last snapshot saved in {}",
        result.outcome,
        companion.display()
    );
    Ok(EncodeReport {
        schema: "jope.encode.v1",
        ok: true,
        dataset_id: data.id,
        train_rows: rows.len(),
        hold_rows: data
            .rows
            .iter()
            .filter(|r| r.split == Split::Validation)
            .count(),
        input_cols: width,
        output_cols: 512,
        weights: weights.display().to_string(),
    })
}

#[cfg(feature = "dope-encoding")]
fn candle_cpu() -> starforge::dope_encoder::DopeDevice {
    starforge::dope_encoder::DopeDevice::Cpu
}
