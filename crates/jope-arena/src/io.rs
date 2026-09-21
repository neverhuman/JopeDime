//! Small file helpers: transparent `.gz` / `.zst` inputs, streaming SHA-256, JSON.
//!
//! Every input file may be committed as `<name>`, `<name>.gz` or `<name>.zst` (the regression
//! fixture keeps the 3.4 MB `scores.csv` under the 500 KiB cap as `scores.csv.zst`); readers here
//! accept all three and callers never see the difference.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

/// Compressed spellings tried after the plain path, in order.
pub const COMPRESSED_SUFFIXES: [&str; 2] = [".gz", ".zst"];

/// The on-disk file behind `path`: `path` itself, `path.gz` or `path.zst`.
pub fn resolve_input(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    COMPRESSED_SUFFIXES.iter().find_map(|suffix| {
        let mut candidate = path.as_os_str().to_owned();
        candidate.push(suffix);
        let candidate = PathBuf::from(candidate);
        candidate.is_file().then_some(candidate)
    })
}

/// Read `path` (or its `.gz` / `.zst` spelling, decompressed) fully.
pub fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let actual = resolve_input(path)
        .with_context(|| format!("missing input {} (nor .gz / .zst)", path.display()))?;
    let mut file = File::open(&actual).with_context(|| format!("open {}", actual.display()))?;
    let mut bytes = Vec::new();
    if actual.extension().is_some_and(|ext| ext == "gz") {
        flate2::read::GzDecoder::new(file)
            .read_to_end(&mut bytes)
            .with_context(|| format!("gunzip {}", actual.display()))?;
    } else if actual.extension().is_some_and(|ext| ext == "zst") {
        ruzstd::decoding::StreamingDecoder::new(&mut file)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .with_context(|| format!("zstd header {}", actual.display()))?
            .read_to_end(&mut bytes)
            .with_context(|| format!("unzstd {}", actual.display()))?;
    } else {
        file.read_to_end(&mut bytes)
            .with_context(|| format!("read {}", actual.display()))?;
    }
    Ok(bytes)
}

/// Read `path` (or `path.gz`) as UTF-8 text.
pub fn read_text(path: &Path) -> Result<String> {
    String::from_utf8(read_bytes(path)?).with_context(|| format!("{} is not UTF-8", path.display()))
}

/// Parse `path` (or `path.gz`) as JSON.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&read_bytes(path)?).with_context(|| format!("parse {}", path.display()))
}

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Streaming SHA-256 and byte count of the file at `path` (no `.gz` fallback: hashes are of the
/// bytes actually on disk).
pub fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

/// Write `bytes` as `path.gz`.
pub fn write_gz(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::best());
    encoder.write_all(bytes)?;
    encoder.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gz_inputs_are_transparent() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("a.csv");
        let gz = dir.path().join("b.csv.gz");
        std::fs::write(&plain, b"x,y\n1,2\n").unwrap();
        write_gz(&gz, b"x,y\n1,2\n").unwrap();
        assert_eq!(read_bytes(&plain).unwrap(), b"x,y\n1,2\n");
        assert_eq!(
            read_bytes(&dir.path().join("b.csv")).unwrap(),
            b"x,y\n1,2\n"
        );
        assert_eq!(read_bytes(&gz).unwrap(), b"x,y\n1,2\n");
        let missing = read_bytes(&dir.path().join("c.csv")).unwrap_err();
        assert!(missing.to_string().contains(".gz / .zst"), "{missing}");
        let zst = dir.path().join("d.csv.zst");
        std::fs::write(
            &zst,
            ruzstd::encoding::compress_to_vec(&b"x,y\n1,2\n"[..], ruzstd::encoding::CompressionLevel::Fastest),
        )
        .unwrap();
        assert_eq!(read_bytes(&dir.path().join("d.csv")).unwrap(), b"x,y\n1,2\n");
        assert_eq!(resolve_input(&dir.path().join("d.csv")), Some(zst));
        let (sha, bytes) = sha256_file(&plain).unwrap();
        assert_eq!(bytes, 8);
        assert_eq!(sha, sha256_hex(b"x,y\n1,2\n"));
    }
}
