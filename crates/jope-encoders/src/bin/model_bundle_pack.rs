//! Convert the frozen JOPE preprocessing arrays to the runtime safetensors contract.
//!
//! This is a packaging tool, not a runtime dependency: deployed code reads only the
//! generated `preprocessing.safetensors` file.  Keeping the conversion in Rust avoids
//! making NumPy or a Python importer part of the product path.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    projection: PathBuf,
    #[arg(long)]
    standardization: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Standardization {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let standardization: Standardization = serde_json::from_slice(
        &std::fs::read(&args.standardization)
            .with_context(|| format!("read {}", args.standardization.display()))?,
    )?;
    if standardization.mean.len() != 512 || standardization.scale.len() != 512 {
        bail!("JOPE standardization arrays must have 512 entries");
    }

    let file = std::fs::File::open(&args.projection)
        .with_context(|| format!("open {}", args.projection.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("open projection NPZ")?;
    let mut read_member = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
        let mut member = archive
            .by_name(&format!("{name}.npy"))
            .with_context(|| format!("projection member {name}.npy"))?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut member, &mut bytes)?;
        read_npy_f32(&bytes).with_context(|| format!("decode projection member {name}.npy"))
    };
    let (components, components_shape) = read_member("components")?;
    let (pca_mean, pca_shape) = read_member("pca_mean")?;
    let (foundation_mean, foundation_shape) = read_member("z_mean")?;
    let (foundation_scale, foundation_scale_shape) = read_member("z_scale")?;
    if components_shape != [1024, 2048]
        || pca_shape != [2048]
        || foundation_shape != [1024]
        || foundation_scale_shape != [1024]
    {
        bail!(
            "unexpected JOPE projection shapes: components={components_shape:?}, pca_mean={pca_shape:?}, z_mean={foundation_shape:?}, z_scale={foundation_scale_shape:?}"
        );
    }
    if foundation_scale
        .iter()
        .any(|value| !value.is_finite() || value.abs() < 1e-12)
    {
        bail!("JOPE projection contains a non-finite or zero scale");
    }

    let device = Device::Cpu;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "hyperion_mean".into(),
        Tensor::from_vec(standardization.mean, (512,), &device)?,
    );
    tensors.insert(
        "hyperion_scale".into(),
        Tensor::from_vec(standardization.scale, (512,), &device)?,
    );
    tensors.insert(
        "foundation_pca_mean".into(),
        Tensor::from_vec(pca_mean, (2048,), &device)?,
    );
    tensors.insert(
        "foundation_components".into(),
        Tensor::from_vec(components, (1024, 2048), &device)?,
    );
    tensors.insert(
        "foundation_mean".into(),
        Tensor::from_vec(foundation_mean, (1024,), &device)?,
    );
    tensors.insert(
        "foundation_scale".into(),
        Tensor::from_vec(foundation_scale, (1024,), &device)?,
    );
    let tensors = tensors
        .into_iter()
        .map(|(name, tensor)| tensor.to_dtype(DType::F32).map(|tensor| (name, tensor)))
        .collect::<candle_core::Result<HashMap<_, _>>>()?;
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    candle_core::safetensors::save(&tensors, &args.out)
        .with_context(|| format!("write {}", args.out.display()))?;
    Ok(())
}

fn read_npy_f32(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("invalid NPY magic");
    }
    let major = bytes[6];
    let (header_len, header_start): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated NPY header");
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        other => bail!("unsupported NPY version {other}"),
    };
    let header_end = header_start
        .checked_add(header_len)
        .context("NPY header overflow")?;
    if header_end > bytes.len() {
        bail!("truncated NPY header");
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])?;
    if !header.contains("'descr': '<f4'") {
        bail!("expected little-endian f32 NPY");
    }
    let fortran = if header.contains("'fortran_order': False") {
        false
    } else if header.contains("'fortran_order': True") {
        true
    } else {
        bail!("NPY storage order is missing");
    };
    let shape_start = header.find("'shape':").context("NPY shape missing")? + 8;
    let shape_text = &header[shape_start..]
        .split_once(')')
        .map(|(value, _)| value)
        .context("NPY shape malformed")?;
    let shape = shape_text
        .trim_start_matches([' ', '('])
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .context("invalid NPY dimension")
        })
        .collect::<Result<Vec<_>>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .context("NPY shape overflow")?;
    let payload_len = count.checked_mul(4).context("NPY payload overflow")?;
    if bytes.len() < header_end + payload_len {
        bail!("truncated NPY payload");
    }
    let values = bytes[header_end..header_end + payload_len]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let values = if fortran && shape.len() > 1 {
        let mut converted = vec![0.0; values.len()];
        for (c_index, slot) in converted.iter_mut().enumerate() {
            let mut remainder = c_index;
            let mut coordinates = vec![0usize; shape.len()];
            for (axis, &dimension) in shape.iter().enumerate().rev() {
                coordinates[axis] = remainder % dimension;
                remainder /= dimension;
            }
            let mut f_index = coordinates[0];
            let mut stride = 1;
            for axis in 1..shape.len() {
                stride *= shape[axis - 1];
                f_index += coordinates[axis] * stride;
            }
            *slot = values[f_index];
        }
        converted
    } else {
        values
    };
    Ok((values, shape))
}
