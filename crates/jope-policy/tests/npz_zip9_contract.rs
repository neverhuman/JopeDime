//! CPU serialization contract for the security-patched Candle / ZIP dependency graph.
//! Uses bit comparisons, since float equality cannot distinguish signed zero or NaN payloads.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self> {
        // Cargo supplies this integration-test scratch directory inside the selected target.
        // Atomic creation refuses an existing directory; Drop only removes our owned child.
        let parent = Path::new(env!("CARGO_TARGET_TMPDIR"));
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = parent.join(format!("npz-zip9-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn f32_bits(tensor: &Tensor) -> candle_core::Result<Vec<u32>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn f64_bits(tensor: &Tensor) -> candle_core::Result<Vec<u64>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<f64>()?
        .into_iter()
        .map(f64::to_bits)
        .collect())
}

fn write_member(path: &Path, payload: &[u8], method: zip::CompressionMethod) -> Result<()> {
    let mut zip = zip::ZipWriter::new(File::create(path)?);
    let options = zip::write::SimpleFileOptions::default().compression_method(method);
    zip.start_file("weights.npy", options)?;
    zip.write_all(payload)?;
    zip.finish()?;
    Ok(())
}

#[test]
fn npz_zip9_preserves_names_shapes_float_bits_and_refuses_truncation() -> Result<()> {
    let scratch = Scratch::new()?;
    // Finite values, both signed zeros, a subnormal, infinity and a quiet NaN payload.
    let expected32 = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0xc060_0000,
        0x7f80_0000,
        0x7fc0_1234,
    ];
    let expected64 = [
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x4009_21fb_5444_2d18,
        0x7ff8_0000_0000_4321,
    ];
    let weights = Tensor::from_vec(
        expected32
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
        (2, 3),
        &Device::Cpu,
    )?;
    let offset = Tensor::from_vec(
        expected64
            .iter()
            .copied()
            .map(f64::from_bits)
            .collect::<Vec<_>>(),
        (2, 2),
        &Device::Cpu,
    )?;
    let archive = scratch.path("roundtrip.npz");
    Tensor::write_npz(
        &[("encoder.weight", &weights), ("offset/β", &offset)],
        &archive,
    )?;
    let loaded = Tensor::read_npz(&archive)?;
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].0, "encoder.weight");
    assert_eq!(loaded[1].0, "offset/β");
    assert_eq!(loaded[0].1.dtype(), DType::F32);
    assert_eq!(loaded[0].1.dims(), &[2, 3]);
    assert_eq!(f32_bits(&loaded[0].1)?, expected32);
    assert_eq!(loaded[1].1.dtype(), DType::F64);
    assert_eq!(loaded[1].1.dims(), &[2, 2]);
    assert_eq!(f64_bits(&loaded[1].1)?, expected64);

    let selected = Tensor::read_npz_by_name(&archive, &["offset/β", "encoder.weight"])?;
    assert_eq!(selected.len(), 2);
    assert_eq!(f64_bits(&selected[0])?, expected64);
    assert_eq!(f32_bits(&selected[1])?, expected32);
    assert!(Tensor::read_npz_by_name(&archive, &["absent"]).is_err());

    let npy = scratch.path("weights.npy");
    weights.write_npy(&npy)?;
    let loaded_npy = Tensor::read_npy(&npy)?;
    assert_eq!(loaded_npy.dims(), &[2, 3]);
    assert_eq!(loaded_npy.dtype(), DType::F32);
    assert_eq!(f32_bits(&loaded_npy)?, expected32);
    let npy_bytes = fs::read(&npy)?;

    // The reader must also accept a compressed ZIP9 member, not only Candle's stored output.
    let compressed = scratch.path("compressed.npz");
    write_member(&compressed, &npy_bytes, zip::CompressionMethod::Deflated)?;
    let loaded_compressed = Tensor::read_npz(&compressed)?;
    assert_eq!(loaded_compressed.len(), 1);
    assert_eq!(loaded_compressed[0].0, "weights");
    assert_eq!(loaded_compressed[0].1.dims(), &[2, 3]);
    assert_eq!(f32_bits(&loaded_compressed[0].1)?, expected32);

    // Remove the ZIP central directory: container corruption must be an error.
    let archive_bytes = fs::read(&archive)?;
    let truncated_zip = scratch.path("truncated-container.npz");
    fs::write(&truncated_zip, &archive_bytes[..archive_bytes.len() / 2])?;
    assert!(Tensor::read_npz(&truncated_zip).is_err());

    // Rebuild a structurally valid ZIP with its own correct CRC, but one missing NPY data byte.
    // This isolates tensor payload refusal from ZIP integrity failure.
    let truncated_payload = &npy_bytes[..npy_bytes.len() - 1];
    let truncated_npy = scratch.path("truncated-payload.npy");
    fs::write(&truncated_npy, truncated_payload)?;
    assert!(Tensor::read_npy(&truncated_npy).is_err());
    let malformed_npz = scratch.path("truncated-payload.npz");
    write_member(
        &malformed_npz,
        truncated_payload,
        zip::CompressionMethod::Stored,
    )?;
    assert!(Tensor::read_npz(&malformed_npz).is_err());
    assert!(Tensor::read_npz_by_name(&malformed_npz, &["weights"]).is_err());
    Ok(())
}
