//! The ingest sources and their mapping into `jope.hwm/v1` / `jope.score/v1`
//! (the mapping table in docs/learning.md).

pub mod champions;
pub mod genebank;
pub mod pipeline;
pub mod snapshot;
pub mod sotarena;

use crate::canon::sha256_hex;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Streaming sha256 of a file (for bundle ids of large inputs such as genebank.db).
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Performs the bytes SHA-256 operation.
pub fn bytes_sha256(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}
