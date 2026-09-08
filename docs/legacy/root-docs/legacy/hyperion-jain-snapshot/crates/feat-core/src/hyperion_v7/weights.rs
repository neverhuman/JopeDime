//! v7 weight loading (safetensors only — jain ships `.safetensors`; the reference's torch-ckpt
//! conversion path is not vendored).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

/// The tensors the inference forward needs (reference filter + the confidence head).
pub fn is_inference_key(k: &str) -> bool {
    k.starts_with("val.")
        || k == "scale_emb.weight"
        || k.starts_with("row_pool.")
        || k.starts_with("col_pool.")
        || k.starts_with("blocks.")
        || k.starts_with("emb_head.")
        || k.starts_with("cell_mlp.")
        || k.starts_with("h_real.")
}

pub fn load_tensor_map(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading Hyperion v7 safetensors {}", path.display()))?;
    Ok(tensors
        .into_iter()
        .filter(|(k, _)| is_inference_key(k))
        .collect())
}
