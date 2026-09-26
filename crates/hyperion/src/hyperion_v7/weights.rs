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

/// [`load_tensor_map`] over safetensors bytes the caller already holds, for example bytes it has
/// just authenticated: the tensors are parsed from that buffer and no file is reopened.
pub fn tensor_map_from_bytes(bytes: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load_buffer(bytes, device)
        .context("parsing Hyperion v7 safetensors bytes")?;
    Ok(tensors
        .into_iter()
        .filter(|(k, _)| is_inference_key(k))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_and_path_loaders_keep_the_same_inference_tensors() {
        let tensors = [
            ("val.0.weight", vec![1f32, 2., 3., 4.]),
            ("blocks.0.lnr.bias", vec![0.5, -0.5]),
            ("optimizer.state", vec![9., 9.]),
        ]
        .into_iter()
        .map(|(name, values)| {
            let n = values.len();
            (
                name.to_string(),
                Tensor::from_vec(values, n, &Device::Cpu).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
        let path = std::env::temp_dir().join(format!(
            "hyperion-weights-bytes-{}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let by_path = load_tensor_map(&path, &Device::Cpu).unwrap();
        // Replacing the file after the bytes were taken cannot reach the bytes loader.
        std::fs::write(&path, b"replaced").unwrap();
        let by_bytes = tensor_map_from_bytes(&bytes, &Device::Cpu).unwrap();
        std::fs::remove_file(&path).unwrap();
        let names = |m: &HashMap<String, Tensor>| {
            let mut names = m.keys().cloned().collect::<Vec<_>>();
            names.sort();
            names
        };
        assert_eq!(names(&by_bytes), ["blocks.0.lnr.bias", "val.0.weight"]);
        assert_eq!(names(&by_bytes), names(&by_path));
        for (name, tensor) in &by_bytes {
            assert_eq!(
                tensor.to_vec1::<f32>().unwrap(),
                by_path[name].to_vec1::<f32>().unwrap()
            );
        }
        assert!(tensor_map_from_bytes(b"replaced", &Device::Cpu).is_err());
    }
}
