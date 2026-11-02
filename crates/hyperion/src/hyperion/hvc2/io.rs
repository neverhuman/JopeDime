use super::format::{
    canonical_header_bytes, digest_hex, sha256_f32_le, Hvc2Artifact, Hvc2FileBinding, Hvc2Header,
    Hvc2PairReceipt, HVC2_MAGIC, HVC2_RECEIPT_FILENAME, HVC2_VERSION, MAX_HEADER_BYTES,
    PREAMBLE_BYTES,
};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn write_hvc2_atomic(
    path: &Path,
    header: &Hvc2Header,
    row_major_values: &[f32],
) -> Result<Hvc2FileBinding> {
    header.validate()?;
    let expected_filename = header.split.filename();
    if path.file_name().and_then(|value| value.to_str()) != Some(expected_filename) {
        bail!("HVC2 split must be written as {expected_filename}");
    }
    let (rows, columns) = header.shape()?;
    let expected_values = rows
        .checked_mul(columns)
        .context("HVC2 shape overflows usize")?;
    if row_major_values.len() != expected_values {
        bail!("HVC2 write payload shape mismatch");
    }
    if row_major_values.iter().any(|value| !value.is_finite()) {
        bail!("HVC2 write payload contains a non-finite value");
    }
    let observed_payload_sha256 = sha256_f32_le(row_major_values);
    if observed_payload_sha256 != header.payload_sha256 {
        bail!("HVC2 write payload hash does not match its header");
    }
    let header_bytes = canonical_header_bytes(header)?;
    let parent = path.parent().context("HVC2 output path has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating HVC2 directory {}", parent.display()))?;
    if path.exists() {
        bail!(
            "refusing to overwrite existing HVC2 file {}",
            path.display()
        );
    }
    let staging = staging_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staging)
            .with_context(|| format!("creating staged HVC2 {}", staging.display()))?;
        file.write_all(&HVC2_MAGIC)?;
        file.write_all(&HVC2_VERSION.to_le_bytes())?;
        file.write_all(
            &u64::try_from(header_bytes.len())
                .context("HVC2 header length does not fit u64")?
                .to_le_bytes(),
        )?;
        file.write_all(&header_bytes)?;
        for value in row_major_values {
            file.write_all(&value.to_le_bytes())?;
        }
        file.sync_all()?;
        drop(file);
        install_no_clobber(&staging, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&staging);
    write_result?;
    hvc2_file_binding(path)
}

pub fn read_hvc2(path: &Path) -> Result<Hvc2Artifact> {
    let mut file = File::open(path).with_context(|| format!("opening HVC2 {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .context("truncated HVC2 magic")?;
    if magic != HVC2_MAGIC {
        bail!("invalid HVC2 magic");
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .context("truncated HVC2 version")?;
    if u16::from_le_bytes(version) != HVC2_VERSION {
        bail!("unsupported HVC2 file version");
    }
    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len)
        .context("truncated HVC2 header length")?;
    let header_len = u64::from_le_bytes(header_len);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!("invalid HVC2 header length {header_len}");
    }
    let header_len_usize = usize::try_from(header_len).context("HVC2 header is too large")?;
    let mut header_bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut header_bytes)
        .context("truncated HVC2 JSON header")?;
    let header: Hvc2Header =
        serde_json::from_slice(&header_bytes).context("parsing HVC2 JSON header")?;
    header.validate()?;
    if canonical_header_bytes(&header)? != header_bytes {
        bail!("HVC2 JSON header is not canonical");
    }
    let expected_file_bytes = PREAMBLE_BYTES
        .checked_add(header_len)
        .and_then(|bytes| bytes.checked_add(header.payload_bytes))
        .context("HVC2 total length overflows u64")?;
    if file_bytes != expected_file_bytes {
        if file_bytes < expected_file_bytes {
            bail!("truncated HVC2 payload");
        }
        bail!("HVC2 file has trailing bytes");
    }
    let value_count =
        usize::try_from(header.payload_bytes / 4).context("HVC2 value count does not fit usize")?;
    let mut values = Vec::with_capacity(value_count);
    let mut digest = Sha256::new();
    for index in 0..value_count {
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes)
            .with_context(|| format!("truncated HVC2 value {index}"))?;
        digest.update(bytes);
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            bail!("HVC2 payload contains a non-finite value at index {index}");
        }
        values.push(value);
    }
    let observed_payload_sha256 = digest_hex(digest.finalize());
    if observed_payload_sha256 != header.payload_sha256 {
        bail!("HVC2 payload SHA-256 mismatch");
    }
    Ok(Hvc2Artifact { header, values })
}

pub fn hvc2_file_binding(path: &Path) -> Result<Hvc2FileBinding> {
    let artifact = read_hvc2(path)?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("HVC2 filename is not UTF-8")?
        .to_string();
    if filename != artifact.header.split.filename() {
        bail!("HVC2 filename does not match its split");
    }
    Ok(Hvc2FileBinding {
        filename,
        sha256: sha256_file(path)?,
        bytes: std::fs::metadata(path)?.len(),
        header: artifact.header,
    })
}

pub fn write_hvc2_receipt_atomic(path: &Path, receipt: &Hvc2PairReceipt) -> Result<()> {
    if path.file_name().and_then(|value| value.to_str()) != Some(HVC2_RECEIPT_FILENAME) {
        bail!("HVC2 pair receipt must be named {HVC2_RECEIPT_FILENAME}");
    }
    receipt.validate()?;
    let parent = path.parent().context("HVC2 receipt path has no parent")?;
    verify_hvc2_pair(parent, receipt)?;
    if path.exists() {
        bail!(
            "refusing to overwrite existing HVC2 receipt {}",
            path.display()
        );
    }
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    std::fs::create_dir_all(parent)?;
    let staging = staging_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staging)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        install_no_clobber(&staging, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&staging);
    write_result
}

pub fn read_hvc2_receipt(path: &Path) -> Result<Hvc2PairReceipt> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading HVC2 receipt {}", path.display()))?;
    let receipt: Hvc2PairReceipt =
        serde_json::from_slice(&bytes).context("parsing HVC2 pair receipt")?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_hvc2_pair(directory: &Path, receipt: &Hvc2PairReceipt) -> Result<()> {
    receipt.validate()?;
    for expected in [&receipt.train_cache, &receipt.test_cache] {
        let path = directory.join(&expected.filename);
        let actual = hvc2_file_binding(&path)?;
        if &actual != expected {
            bail!("HVC2 file {} does not match its receipt", path.display());
        }
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest_hex(digest.finalize()))
}

fn staging_path(path: &Path) -> Result<PathBuf> {
    static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().context("atomic output path has no parent")?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("atomic output filename is not UTF-8")?;
    Ok(parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        NEXT_STAGING.fetch_add(1, Ordering::Relaxed)
    )))
}

fn install_no_clobber(staging: &Path, destination: &Path) -> Result<()> {
    std::fs::hard_link(staging, destination).with_context(|| {
        format!(
            "installing {} without overwriting {}",
            staging.display(),
            destination.display()
        )
    })
}
