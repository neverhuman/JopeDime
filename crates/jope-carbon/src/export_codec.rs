//! Bounded `CZIP\x01` gzip framing inside `CUEN\x01(ExportArtifact)` envelopes (`ADR-0007`).
//!
//! Compresses large standalone source exports (`>= 16 KiB`) before `AES-256-GCM` encryption so
//! dense kernel/ensemble float tables shrink `6x-8x` on the wire while keeping `source_sha256`
//! bound to the exact uncompressed source bytes, and enforces a strict `64 MiB` decompression cap.

use std::io::{Read, Write};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use zeroize::Zeroizing;

use crate::error::CarbonErrCtx as _;

/// `5`-byte inner magic header marking gzip-compressed `ExportArtifact` plaintext inside `CUEN\x01`.
pub const CZIP_MAGIC: &[u8; 5] = b"CZIP\x01";

/// Minimum plaintext export size (`16 KiB`) before attempting `CZIP\x01` compression.
pub const EXPORT_COMPRESS_MIN_BYTES: usize = 16_384;

/// Maximum uncompressed export byte ceiling (`64 MiB`) enforced during bounded gunzip.
pub const MAX_DECOMPRESSED_EXPORT_BYTES: u64 = 64 * 1024 * 1024;

/// Compress `plaintext` with `CZIP\x01` + gzip (`Compression::fast()`) when `plaintext.len() >= 16 KiB`
/// and compression strictly reduces payload size; otherwise return a `Zeroizing` copy of `plaintext`.
pub fn maybe_compress_export_payload(plaintext: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
    if plaintext.len() < EXPORT_COMPRESS_MIN_BYTES {
        return Ok(Zeroizing::new(plaintext.to_vec()));
    }
    let mut framed = Zeroizing::new(Vec::with_capacity(plaintext.len() / 2));
    framed.extend_from_slice(CZIP_MAGIC);
    {
        let mut enc = GzEncoder::new(&mut *framed, Compression::fast());
        enc.write_all(plaintext)
            .err_ctx("gzip compress export payload")?;
        enc.finish().err_ctx("finish gzip export payload")?;
    }
    if framed.len() < plaintext.len() {
        Ok(framed)
    } else {
        Ok(Zeroizing::new(plaintext.to_vec()))
    }
}

/// Transparently gunzip `CZIP\x01`-prefixed `ExportArtifact` plaintext with a strict `64 MiB` bound,
/// or return `payload` unchanged when unframed.
pub fn decompress_export_payload_bounded(payload: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
    let Some(gz_body) = payload.strip_prefix(CZIP_MAGIC) else {
        if payload.len() as u64 > MAX_DECOMPRESSED_EXPORT_BYTES {
            return Err(format!(
                "export payload exceeds {MAX_DECOMPRESSED_EXPORT_BYTES} byte limit"
            ));
        }
        return Ok(Zeroizing::new(payload.to_vec()));
    };
    let mut out = Zeroizing::new(Vec::with_capacity(gz_body.len().saturating_mul(4)));
    let decoder = GzDecoder::new(gz_body);
    decoder
        .take(MAX_DECOMPRESSED_EXPORT_BYTES.saturating_add(1))
        .read_to_end(&mut out)
        .err_ctx("gunzip CZIP\\x01 export payload")?;
    if out.len() as u64 > MAX_DECOMPRESSED_EXPORT_BYTES {
        return Err(format!(
            "decompressed CZIP\\x01 export payload exceeds {MAX_DECOMPRESSED_EXPORT_BYTES} byte ceiling"
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn czip_codec_roundtrips_large_source_and_preserves_small_unframed() {
        let small = b"def predict_rows(rows): return [0.0 for _ in rows]\n";
        let encoded_small = maybe_compress_export_payload(small).expect("encode small");
        assert!(!encoded_small.starts_with(CZIP_MAGIC));
        let decoded_small =
            decompress_export_payload_bounded(&encoded_small).expect("decode small");
        assert_eq!(&*decoded_small, small);

        let large = "    0.1234567890123456, -0.9876543210987654,\n".repeat(200_000);
        assert!(large.len() > 8_000_000);
        let encoded_large = maybe_compress_export_payload(large.as_bytes()).expect("encode large");
        assert!(encoded_large.starts_with(CZIP_MAGIC));
        assert!(
            encoded_large.len() < 500_000,
            "expected >16x compression on repetitive float source, got {} bytes",
            encoded_large.len()
        );
        let decoded_large =
            decompress_export_payload_bounded(&encoded_large).expect("decode large");
        assert_eq!(decoded_large.as_slice(), large.as_bytes());
    }
}
