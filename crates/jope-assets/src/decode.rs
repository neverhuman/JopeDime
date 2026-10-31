//! Asset codecs. Every codec is lossless (a packaging-only identity class, SPECIFICATION.md §8.2):
//! decoding returns the source file's bytes exactly, which the reader checks against the entry's
//! sha256.
//!
//! `shuffle+zstd` is for safetensors files: the header is kept as it is, and each tensor's data is
//! rearranged into byte planes by its dtype's width (byte 0 of every element, then byte 1, …), so
//! the sign/exponent bytes of floating-point weights sit together and compress. `zstd` frames the
//! bytes unchanged; `raw` stores them as they are (for incompressible assets).
//!
//! Decoding authenticates the stored bytes against the manifest *before* any of them reaches the
//! decompressor, then bounds every frame (declared content size, window, no dictionary, no
//! trailing bytes) and checks the decoded sha256.

use crate::{limits, sha256_hex, AssetError, Entry, Frame};
use serde::{Deserialize, Serialize};

/// Lossless storage codec recorded for each manifest entry.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Codec {
    /// Reorder safetensors tensor bytes into byte planes, then compress with zstd.
    #[serde(rename = "shuffle+zstd")]
    ShuffleZstd,
    /// Store the source bytes in a bounded zstd frame.
    #[serde(rename = "zstd")]
    Zstd,
    /// Store the source bytes without compression.
    #[serde(rename = "raw")]
    Raw,
}

/// One tensor's data range and element width, from a safetensors header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    width: usize,
}

fn dtype_width(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" | "U64" | "C64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        _ => 1,
    }
}

/// The tensor spans of a safetensors file, or `None` when the bytes are not a well-formed one.
fn spans(bytes: &[u8]) -> Option<Vec<Span>> {
    let header_len = usize::try_from(u64::from_le_bytes(bytes.get(0..8)?.try_into().ok()?)).ok()?;
    let data_start = 8usize.checked_add(header_len)?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(bytes.get(8..data_start)?).ok()?;
    let mut spans = Vec::new();
    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info.get("dtype")?.as_str()?;
        let offsets = info.get("data_offsets")?.as_array()?;
        let (a, b) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
        let start = data_start.checked_add(usize::try_from(a).ok()?)?;
        let end = data_start.checked_add(usize::try_from(b).ok()?)?;
        if start > end || end > bytes.len() {
            return None;
        }
        let mut width = dtype_width(dtype);
        if !(end - start).is_multiple_of(width) {
            width = 1;
        }
        spans.push(Span { start, end, width });
    }
    spans.sort_by_key(|s| s.start);
    if spans.windows(2).any(|w| w[0].end > w[1].start) {
        return None;
    }
    Some(spans)
}

fn shuffle_span(input: &[u8], output: &mut [u8], width: usize) {
    let n = input.len() / width;
    for (i, element) in input.chunks_exact(width).enumerate() {
        for (k, byte) in element.iter().enumerate() {
            output[k * n + i] = *byte;
        }
    }
}

fn unshuffle_span(input: &[u8], output: &mut [u8], width: usize) {
    let n = input.len() / width;
    for (i, element) in output.chunks_exact_mut(width).enumerate() {
        for (k, byte) in element.iter_mut().enumerate() {
            *byte = input[k * n + i];
        }
    }
}

/// Byte-plane shuffle of a safetensors file; `None` when it does not parse as one.
pub fn shuffle_safetensors(bytes: &[u8]) -> Option<Vec<u8>> {
    let spans = spans(bytes)?;
    let mut out = bytes.to_vec();
    for span in spans.iter().filter(|s| s.width > 1) {
        shuffle_span(
            &bytes[span.start..span.end],
            &mut out[span.start..span.end],
            span.width,
        );
    }
    Some(out)
}

/// Inverse of [`shuffle_safetensors`]. The header is stored unshuffled, so the spans are read
/// from the shuffled bytes themselves.
pub fn unshuffle_safetensors(shuffled: &[u8]) -> Result<Vec<u8>, String> {
    let spans = spans(shuffled).ok_or("shuffled asset does not carry a safetensors header")?;
    let mut out = shuffled.to_vec();
    for span in spans.iter().filter(|s| s.width > 1) {
        unshuffle_span(
            &shuffled[span.start..span.end],
            &mut out[span.start..span.end],
            span.width,
        );
    }
    Ok(out)
}

/// Check a zstd frame header against the frame's manifest record before decoding: the zstd
/// magic, no dictionary, a declared content size equal to `raw_len`, and a bounded window.
pub fn check_zstd_header(stored: &[u8], raw_len: u64) -> Result<(), String> {
    const MAGIC: [u8; 4] = 0xFD2F_B528u32.to_le_bytes();
    if stored.get(0..4) != Some(&MAGIC[..]) {
        return Err("not a zstd frame".into());
    }
    let descriptor = *stored.get(4).ok_or("truncated zstd frame header")?;
    let fcs_flag = descriptor >> 6;
    let single_segment = descriptor & 0x20 != 0;
    if descriptor & 0x08 != 0 {
        return Err("zstd reserved bit set".into());
    }
    if descriptor & 0x03 != 0 {
        return Err("zstd frame names a dictionary".into());
    }
    let mut at = 5usize;
    let window = if single_segment {
        None
    } else {
        let byte = *stored.get(at).ok_or("truncated zstd frame header")?;
        at += 1;
        let log = 10 + u64::from(byte >> 3);
        let base = 1u64 << log;
        Some(base + (base / 8) * u64::from(byte & 7))
    };
    let fcs_len = match (fcs_flag, single_segment) {
        (0, false) => return Err("zstd frame does not declare its content size".into()),
        (0, true) => 1,
        (1, _) => 2,
        (2, _) => 4,
        _ => 8,
    };
    let field = stored
        .get(at..at + fcs_len)
        .ok_or("truncated zstd frame header")?;
    let mut buf = [0u8; 8];
    buf[..fcs_len].copy_from_slice(field);
    let mut content = u64::from_le_bytes(buf);
    if fcs_len == 2 {
        content += 256;
    }
    if content != raw_len {
        return Err(format!(
            "zstd frame declares {content} bytes, the manifest {raw_len}"
        ));
    }
    if window.unwrap_or(content) > limits::MAX_ZSTD_WINDOW {
        return Err("zstd window exceeds the bound".into());
    }
    Ok(())
}

fn decode_frame(codec: Codec, stored: &[u8], slot: &mut [u8]) -> Result<(), String> {
    match codec {
        Codec::Raw => {
            if stored.len() != slot.len() {
                return Err("raw frame length differs from its raw_len".into());
            }
            slot.copy_from_slice(stored);
        }
        Codec::Zstd | Codec::ShuffleZstd => {
            use std::io::Read;
            check_zstd_header(stored, slot.len() as u64)?;
            let mut source = stored;
            {
                let mut decoder = ruzstd::decoding::StreamingDecoder::new(&mut source)
                    .map_err(|e| format!("zstd frame header: {e:?}"))?;
                decoder
                    .read_exact(slot)
                    .map_err(|e| format!("zstd frame body: {e}"))?;
                let mut probe = [0u8; 1];
                if decoder.read(&mut probe).map_err(|e| e.to_string())? != 0 {
                    return Err("zstd frame is longer than its raw_len".into());
                }
            }
            if !source.is_empty() {
                return Err("bytes follow the zstd frame".into());
            }
        }
    }
    Ok(())
}

/// Decode one entry. `read_frame` returns a frame's stored bytes. The stored bytes are hashed in
/// order and compared with `stored_sha256` before any frame is decompressed; frames then decode in
/// parallel, and the result is compared with `sha256`.
pub fn decode_entry(
    entry: &Entry,
    read_frame: &(dyn Fn(&Frame) -> std::io::Result<Vec<u8>> + Sync),
) -> Result<Vec<u8>, AssetError> {
    use rayon::prelude::*;
    use sha2::{Digest, Sha256};
    let fail = |reason: String| AssetError::Decode {
        id: entry.id.clone(),
        reason,
    };
    // `Manifest::check` bounded every length used below.
    let mut stored = Vec::with_capacity(entry.frames.len());
    let mut hasher = Sha256::new();
    for frame in &entry.frames {
        let bytes = read_frame(frame)?;
        hasher.update(&bytes);
        stored.push(bytes);
    }
    if hex::encode(hasher.finalize()) != entry.stored_sha256 {
        return Err(AssetError::StoredHash {
            id: entry.id.clone(),
        });
    }
    let total = usize::try_from(entry.bytes)
        .map_err(|_| fail("asset too large for this platform".into()))?;
    let mut out = vec![0u8; total];
    let mut slots = Vec::with_capacity(entry.frames.len());
    let mut rest: &mut [u8] = &mut out;
    for (frame, bytes) in entry.frames.iter().zip(&stored) {
        let raw = usize::try_from(frame.raw_len).map_err(|_| fail("frame too large".into()))?;
        if raw > rest.len() {
            return Err(fail("frames exceed the asset's size".into()));
        }
        let (head, tail) = rest.split_at_mut(raw);
        slots.push((bytes, head));
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(fail("frames do not cover the asset".into()));
    }
    slots
        .into_par_iter()
        .try_for_each(|(bytes, slot)| decode_frame(entry.codec, bytes, slot))
        .map_err(fail)?;
    drop(stored);
    if entry.codec == Codec::ShuffleZstd {
        out = unshuffle_safetensors(&out).map_err(fail)?;
    }
    if sha256_hex(&out) != entry.sha256 {
        return Err(AssetError::EntryHash {
            id: entry.id.clone(),
        });
    }
    Ok(out)
}

/// Stored frames in order: (stored bytes, decoded length).
#[cfg(feature = "pack")]
pub type StoredFrames = Vec<(Vec<u8>, u64)>;

/// Encode an asset into frames of at most `frame_bytes` decoded bytes. Returns the codec actually
/// used (a non-safetensors file asked for `shuffle+zstd` falls back to `zstd`, and a compressed
/// form that is not smaller falls back to `raw`) and the stored frames in order.
#[cfg(feature = "pack")]
pub fn encode(
    requested: Codec,
    bytes: &[u8],
    frame_bytes: usize,
    level: i32,
) -> std::io::Result<(Codec, StoredFrames)> {
    use rayon::prelude::*;
    let frame_bytes = frame_bytes.clamp(1, limits::MAX_FRAME_RAW_BYTES as usize);
    let (codec, transformed) = match requested {
        Codec::ShuffleZstd => match shuffle_safetensors(bytes) {
            Some(shuffled) => (Codec::ShuffleZstd, shuffled),
            None => (Codec::Zstd, bytes.to_vec()),
        },
        other => (other, bytes.to_vec()),
    };
    let chunks: Vec<&[u8]> = transformed.chunks(frame_bytes).collect();
    let frames = chunks
        .par_iter()
        .map(|chunk| -> std::io::Result<(Vec<u8>, u64)> {
            let stored = match codec {
                Codec::Raw => chunk.to_vec(),
                Codec::Zstd | Codec::ShuffleZstd => {
                    let mut compressor = zstd::bulk::Compressor::new(level)?;
                    compressor.include_checksum(true)?;
                    compressor.include_contentsize(true)?;
                    compressor.include_dictid(false)?;
                    compressor.compress(chunk)?
                }
            };
            Ok((stored, chunk.len() as u64))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let stored: usize = frames.iter().map(|(s, _)| s.len()).sum();
    if codec != Codec::Raw && stored >= bytes.len() {
        return encode(Codec::Raw, bytes, frame_bytes, level);
    }
    Ok((codec, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small safetensors file with the given (name, dtype, element width, count) tensors.
    fn safetensors(tensors: &[(&str, &str, usize, usize)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (index, (name, dtype, width, count)) in tensors.iter().enumerate() {
            let start = data.len();
            for i in 0..width * count {
                data.push(((i * 31 + index * 7) % 251) as u8);
            }
            header.insert(
                (*name).to_owned(),
                serde_json::json!({"dtype": dtype, "shape": [count], "data_offsets": [start, data.len()]}),
            );
        }
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let header = serde_json::to_vec(&header).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn shuffle_round_trips_every_width() {
        let file = safetensors(&[
            ("a", "F32", 4, 33),
            ("b", "BF16", 2, 17),
            ("c", "F64", 8, 5),
            ("d", "U8", 1, 9),
        ]);
        let shuffled = shuffle_safetensors(&file).unwrap();
        assert_ne!(shuffled, file);
        assert_eq!(shuffled.len(), file.len());
        assert_eq!(unshuffle_safetensors(&shuffled).unwrap(), file);
    }

    #[test]
    fn non_safetensors_bytes_are_not_shuffled() {
        assert!(shuffle_safetensors(b"{\"not\": \"safetensors\"}").is_none());
        assert!(shuffle_safetensors(&[1, 2, 3]).is_none());
    }

    fn entry_for(
        codec: Codec,
        frames: &crate::decode::StoredFrames,
        raw: &[u8],
    ) -> (Entry, Vec<u8>) {
        use sha2::{Digest, Sha256};
        let mut index = Vec::new();
        let mut blob = Vec::new();
        let mut hasher = Sha256::new();
        for (stored, raw_len) in frames {
            let offset = blob.len().next_multiple_of(4096) as u64;
            blob.resize(offset as usize, 0);
            index.push(Frame {
                offset,
                len: stored.len() as u64,
                raw_len: *raw_len,
            });
            blob.extend_from_slice(stored);
            hasher.update(stored);
        }
        let entry = Entry {
            id: "model/x/w.safetensors".into(),
            role: "model".into(),
            component: "x".into(),
            origin: "x/w.safetensors".into(),
            sha256: sha256_hex(raw),
            bytes: raw.len() as u64,
            codec,
            stored_sha256: hex::encode(hasher.finalize()),
            stored_bytes: frames.iter().map(|(s, _)| s.len() as u64).sum(),
            frames: index,
            license: crate::License {
                spdx: "MIT".into(),
                redistribution: "permitted".into(),
            },
            backends: vec!["cpu".into()],
        };
        (entry, blob)
    }

    #[test]
    fn frames_decode_to_the_original_bytes() {
        let file = safetensors(&[("w", "F32", 4, 4000), ("b", "F16", 2, 300)]);
        for codec in [Codec::ShuffleZstd, Codec::Zstd, Codec::Raw] {
            let (used, frames) = encode(codec, &file, 1000, 3).unwrap();
            assert_eq!(used, codec);
            let (entry, blob) = entry_for(used, &frames, &file);
            let read =
                |f: &Frame| Ok(blob[f.offset as usize..(f.offset + f.len) as usize].to_vec());
            assert_eq!(decode_entry(&entry, &read).unwrap(), file);
        }
    }

    #[test]
    fn json_asked_to_shuffle_falls_back_and_incompressible_bytes_stay_raw() {
        let json = br#"{"x": 1, "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}"#
            .repeat(40);
        let (used, _) = encode(Codec::ShuffleZstd, &json, 1 << 20, 3).unwrap();
        assert_eq!(used, Codec::Zstd);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let noise: Vec<u8> = (0..5000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect();
        let (used, frames) = encode(Codec::Zstd, &noise, 1 << 20, 3).unwrap();
        assert_eq!(used, Codec::Raw);
        assert_eq!(frames[0].0, noise);
    }

    #[test]
    fn header_check_refuses_wrong_size_dictionary_and_garbage() {
        let data = vec![5u8; 10_000];
        let frame = zstd::bulk::compress(&data, 3).unwrap();
        assert!(check_zstd_header(&frame, 10_000).is_ok());
        assert!(check_zstd_header(&frame, 9_999).is_err());
        assert!(check_zstd_header(b"PK\x03\x04", 1).is_err());
        let mut with_dict = frame.clone();
        with_dict[4] |= 0x01;
        assert!(check_zstd_header(&with_dict, 10_000).is_err());
    }
}
