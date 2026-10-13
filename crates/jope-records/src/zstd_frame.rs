//! Zstandard frames through the pure-Rust `ruzstd` this crate already uses for `.zst` Elo inputs.
//!
//! A thin, dependency-free surface for other workspace crates (the S3 retention packs in
//! `jope-learning`), so they get zstd without a new dependency edge in `Cargo.lock`.
//! `ruzstd` 0.8 compresses at its `Fastest` level, roughly zstd level 1; any zstd decoder reads
//! the output.

use std::io::Read;

/// One zstd frame holding `data` (ruzstd `Fastest`, about zstd level 1).
pub fn compress(data: &[u8]) -> Vec<u8> {
    ruzstd::encoding::compress_to_vec(data, ruzstd::encoding::CompressionLevel::Fastest)
}

/// Decodes one zstd frame. `limit` caps the decoded size so a hostile frame cannot exhaust
/// memory; a frame that would decode past it is an error.
pub fn decompress(frame: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut source = frame;
    let decoder = ruzstd::decoding::StreamingDecoder::new(&mut source)
        .map_err(|e| format!("zstd frame header: {e}"))?;
    let mut out = Vec::new();
    decoder
        .take(limit as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| format!("zstd frame body: {e}"))?;
    if out.len() > limit {
        return Err(format!("zstd frame decodes past the {limit}-byte limit"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_limit() {
        let data: Vec<u8> = (0..200_000u32)
            .flat_map(|i| (i % 251).to_le_bytes())
            .collect();
        let frame = compress(&data);
        assert!(frame.len() < data.len(), "repetitive data compresses");
        assert_eq!(decompress(&frame, data.len()).unwrap(), data);
        assert!(decompress(&frame, data.len() - 1).is_err());
        assert!(decompress(b"not a zstd frame", 1024).is_err());
        assert_eq!(decompress(&compress(b""), 0).unwrap(), Vec::<u8>::new());
    }
}
