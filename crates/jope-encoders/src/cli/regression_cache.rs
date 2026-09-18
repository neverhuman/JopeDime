use crate::model::{RegressionExpertFeatures, RegressionFeatureDump};
use anyhow::Result;
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub(super) struct RegressionFeatureCacheMeta<'a> {
    pub(super) dataset: &'a str,
    pub(super) n_features: usize,
    pub(super) source_n_features: usize,
    pub(super) max_features: usize,
    pub(super) context_idx: &'a [usize],
    pub(super) test_idx: &'a [usize],
}

pub(super) fn write_regression_feature_cache(
    out_dir: &Path,
    dump: &RegressionFeatureDump,
    y_true: &[f32],
    meta: &RegressionFeatureCacheMeta<'_>,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    write_expert_feature_arrays(out_dir, "starlight", &dump.starlight)?;
    write_expert_feature_arrays(out_dir, "tabicl", &dump.tabicl)?;
    write_npy_f32(
        &out_dir.join("gate_meta.npy"),
        &dump.gate_meta,
        &[dump.gate_meta.len()],
    )?;
    write_npy_f32(&out_dir.join("y_true.npy"), y_true, &[y_true.len()])?;
    let pred: Vec<f32> = dump.prediction.iter().map(|&v| v as f32).collect();
    write_npy_f32(&out_dir.join("rust_prediction.npy"), &pred, &[pred.len()])?;

    let meta = json!({
        "format": "cmaes_cheat.regression_feature_cache.v1",
        "dataset": meta.dataset,
        "n_features": meta.n_features,
        "source_n_features": meta.source_n_features,
        "max_features": meta.max_features,
        "context_rows": meta.context_idx.len(),
        "eval_rows": meta.test_idx.len(),
        "context_idx": meta.context_idx,
        "test_idx": meta.test_idx,
        "arrays": {
            "starlight": {
                "mean": "starlight_mean.npy",
                "hidden": "starlight_hidden.npy",
                "quantiles": "starlight_quantiles.npy",
                "conf": "starlight_conf.npy"
            },
            "tabicl": {
                "mean": "tabicl_mean.npy",
                "hidden": "tabicl_hidden.npy",
                "quantiles": "tabicl_quantiles.npy",
                "conf": "tabicl_conf.npy"
            },
            "gate_meta": "gate_meta.npy",
            "y_true": "y_true.npy",
            "rust_prediction": "rust_prediction.npy"
        }
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)? + "\n",
    )?;
    Ok(())
}

fn write_expert_feature_arrays(
    out_dir: &Path,
    prefix: &str,
    features: &RegressionExpertFeatures,
) -> Result<()> {
    write_npy_f32(
        &out_dir.join(format!("{prefix}_mean.npy")),
        &features.mean,
        &[features.n_rows],
    )?;
    write_npy_f32(
        &out_dir.join(format!("{prefix}_hidden.npy")),
        &features.hidden,
        &[features.n_rows, features.hidden_dim],
    )?;
    write_npy_f32(
        &out_dir.join(format!("{prefix}_quantiles.npy")),
        &features.quantiles,
        &[features.n_rows, features.quantile_dim],
    )?;
    write_npy_f32(
        &out_dir.join(format!("{prefix}_conf.npy")),
        &features.conf,
        &[features.n_rows, features.conf_dim],
    )?;
    Ok(())
}

fn write_npy_f32(path: &Path, values: &[f32], shape: &[usize]) -> Result<()> {
    let expected = shape.iter().copied().product::<usize>();
    if values.len() != expected {
        anyhow::bail!(
            "cannot write {}: shape {:?} expects {} values, got {}",
            path.display(),
            shape,
            expected,
            values.len()
        );
    }
    let shape_text = npy_shape_text(shape);
    let header_dict =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    let mut header = header_dict.into_bytes();
    let pad = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');
    if header.len() > u16::MAX as usize {
        anyhow::bail!("NumPy header too large for {}", path.display());
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(&header)?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn npy_shape_text(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        dims => {
            let joined = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({joined})")
        }
    }
}
