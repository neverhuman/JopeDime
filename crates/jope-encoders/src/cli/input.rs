use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::path::Path;

pub(super) fn read_target_column(path: &Path) -> Result<Vec<f32>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening target input {}", path.display()))?;
    let mut values = Vec::new();
    for (row, record) in reader.records().enumerate() {
        let record =
            record.with_context(|| format!("reading row {} from {}", row + 1, path.display()))?;
        let Some(field) = record.iter().next_back() else {
            continue;
        };
        match field.trim().parse::<f32>() {
            Ok(value) => values.push(value),
            Err(_) if values.is_empty() && row == 0 => {}
            Err(_) => anyhow::bail!(
                "target input {} has a non-numeric final column at row {}",
                path.display(),
                row + 1
            ),
        }
    }
    if values.is_empty() {
        anyhow::bail!("target input {} has no numeric labels", path.display());
    }
    Ok(values)
}

pub(super) fn parse_device(value: &str) -> Result<Device> {
    if value == "cpu" {
        Ok(Device::Cpu)
    } else if let Some(rest) = value.strip_prefix("cuda:") {
        let ordinal = rest
            .parse::<usize>()
            .with_context(|| format!("invalid CUDA device ordinal {rest:?}"))?;
        Ok(Device::new_cuda(ordinal)?)
    } else if value == "cuda" {
        Ok(Device::new_cuda(0)?)
    } else {
        anyhow::bail!("unsupported device {value:?}; use cpu or cuda:N")
    }
}

pub(super) fn parse_dtype(value: &str) -> Result<DType> {
    match value {
        "f32" | "float32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        other => anyhow::bail!("unsupported dtype {other:?}; use f32 or bf16"),
    }
}

pub(super) fn capped_features(n_features: usize, max_features: usize) -> usize {
    if max_features == 0 {
        n_features
    } else {
        n_features.min(max_features)
    }
}

pub(super) fn keep_feature_prefix(
    x: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    keep_features: usize,
) -> Vec<f32> {
    if keep_features >= n_features {
        return x;
    }
    let mut out = Vec::with_capacity(n_rows * keep_features);
    for row in x.chunks_exact(n_features).take(n_rows) {
        out.extend_from_slice(&row[..keep_features]);
    }
    out
}
