//! Raw Snappy (the block format Parquet pages use, no framing), hand-written because no Snappy
//! crate is in `Cargo.lock`. S3 Inventory writes its Parquet files with Snappy pages.
//!
//! The decoder follows the format description in google/snappy `format_description.txt`: a
//! varint uncompressed length, then literal and copy elements. Every length and offset is
//! bounds-checked and the output must be exactly the declared length. The compressor is a
//! small greedy matcher used to build test fixtures; any valid Snappy stream decodes the same.

use anyhow::{bail, ensure, Context, Result};

fn le(b: &[u8]) -> usize {
    b.iter()
        .rev()
        .fold(0usize, |acc, &x| (acc << 8) | usize::from(x))
}

/// Decodes one raw Snappy block whose declared length must not exceed `limit`.
pub fn decompress(src: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut len = 0usize;
    let mut p = 0usize;
    for shift in (0..35).step_by(7) {
        let b = *src.get(p).context("snappy: truncated length")?;
        p += 1;
        len |= usize::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        ensure!(shift < 28, "snappy: length varint too long");
    }
    ensure!(
        len <= limit,
        "snappy: declared length {len} exceeds the cap {limit}"
    );
    let mut out: Vec<u8> = Vec::with_capacity(len);
    while p < src.len() {
        let tag = src[p];
        p += 1;
        let (n, off) = match tag & 3 {
            0 => {
                let mut n = usize::from(tag >> 2);
                if n >= 60 {
                    let k = n - 59;
                    ensure!(p + k <= src.len(), "snappy: truncated literal length");
                    n = le(&src[p..p + k]);
                    p += k;
                }
                let n = n
                    .checked_add(1)
                    .context("snappy: literal length overflow")?;
                let end = p.checked_add(n).context("snappy: literal overflow")?;
                ensure!(end <= src.len(), "snappy: literal runs past the input");
                ensure!(
                    out.len() + n <= len,
                    "snappy: literal runs past the declared length"
                );
                out.extend_from_slice(&src[p..end]);
                p = end;
                continue;
            }
            1 => {
                ensure!(p < src.len(), "snappy: truncated copy");
                let n = 4 + usize::from((tag >> 2) & 7);
                let off = (usize::from(tag >> 5) << 8) | usize::from(src[p]);
                p += 1;
                (n, off)
            }
            2 => {
                ensure!(p + 2 <= src.len(), "snappy: truncated copy");
                let off = le(&src[p..p + 2]);
                p += 2;
                (1 + usize::from(tag >> 2), off)
            }
            _ => {
                ensure!(p + 4 <= src.len(), "snappy: truncated copy");
                let off = le(&src[p..p + 4]);
                p += 4;
                (1 + usize::from(tag >> 2), off)
            }
        };
        if off == 0 || off > out.len() {
            bail!(
                "snappy: copy offset {off} outside the {} bytes written",
                out.len()
            );
        }
        ensure!(
            out.len() + n <= len,
            "snappy: copy runs past the declared length"
        );
        // Copies may overlap their own output (run-length style), so go byte by byte.
        let start = out.len() - off;
        for i in 0..n {
            let b = out[start + i];
            out.push(b);
        }
    }
    ensure!(
        out.len() == len,
        "snappy: decoded {} bytes, header says {len}",
        out.len()
    );
    Ok(out)
}

fn literal(out: &mut Vec<u8>, lit: &[u8]) {
    let mut rest = lit;
    while !rest.is_empty() {
        let n = rest.len().min(1 << 16);
        let m = n - 1;
        if m < 60 {
            out.push((m as u8) << 2);
        } else if m < 256 {
            out.push(60 << 2);
            out.push(m as u8);
        } else {
            out.push(61 << 2);
            out.extend_from_slice(&(m as u16).to_le_bytes());
        }
        out.extend_from_slice(&rest[..n]);
        rest = &rest[n..];
    }
}

/// A valid raw Snappy block: greedy 4-byte matches within 64 KiB, emitted as 1- or 2-byte-offset
/// copies, everything else as literals.
pub fn compress(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 2 + 16);
    let mut v = src.len();
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
    let mut table = vec![usize::MAX; 1 << 14];
    let hash = |w: &[u8]| -> usize {
        let x = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        (x.wrapping_mul(0x1e35_a7bd) >> 18) as usize
    };
    let (mut i, mut lit_start) = (0usize, 0usize);
    while i + 4 <= src.len() {
        let h = hash(&src[i..i + 4]);
        let cand = table[h];
        table[h] = i;
        if cand != usize::MAX && i - cand <= 0xffff && src[cand..cand + 4] == src[i..i + 4] {
            literal(&mut out, &src[lit_start..i]);
            let mut n = 4;
            while i + n < src.len() && n < 64 && src[cand + n] == src[i + n] {
                n += 1;
            }
            let off = i - cand;
            if (4..=11).contains(&n) && off < 2048 {
                out.push(1 | (((n - 4) as u8) << 2) | (((off >> 8) as u8) << 5));
                out.push(off as u8);
            } else {
                out.push(2 | (((n - 1) as u8) << 2));
                out.extend_from_slice(&(off as u16).to_le_bytes());
            }
            i += n;
            lit_start = i;
        } else {
            i += 1;
        }
    }
    literal(&mut out, &src[lit_start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_built_stream_uses_every_element_kind() {
        // len 26 | literal "abcd" | copy1 len 4 off 4 | copy2 len 6 off 2 (overlapping)
        // | literal(60-form) "xyz" | copy4 len 9 off 26-17=... built below.
        let mut s = vec![26u8];
        s.extend_from_slice(&[3 << 2, b'a', b'b', b'c', b'd']); // "abcd"
        s.extend_from_slice(&[1, 4]); // copy1: len 4, off 4 -> "abcd"
        s.extend_from_slice(&[2 | (5 << 2), 2, 0]); // copy2: len 6, off 2 -> "cdcdcd"
        s.extend_from_slice(&[60 << 2, 2, b'x', b'y', b'z']); // literal len 3 via 1-byte length
        s.extend_from_slice(&[3 | (8 << 2), 17, 0, 0, 0]); // copy4: len 9, off 17
        let out = decompress(&s, 1 << 20).unwrap();
        let mut want = b"abcdabcdcdcdcdxyz".to_vec();
        let start = want.len() - 17;
        for i in 0..9 {
            want.push(want[start + i]);
        }
        assert_eq!(out, want);
        assert_eq!(out.len(), 26);
    }

    #[test]
    fn hostile_streams_are_refused() {
        assert!(
            decompress(&[5, 1 << 2, b'a', b'b'], 100).is_err(),
            "short output"
        );
        assert!(
            decompress(&[4, 1, 4], 100).is_err(),
            "copy before any output"
        );
        assert!(
            decompress(&[200, 1], 100).is_err(),
            "declared length over cap"
        );
        assert!(
            decompress(&[3, 9 << 2, b'a'], 100).is_err(),
            "literal past input"
        );
        assert!(
            decompress(&[2, 1 << 2, b'a', b'b', 1, 2], 100).is_err(),
            "past length"
        );
        assert!(decompress(&[], 100).is_err());
    }

    #[test]
    fn compressor_round_trips() {
        let mut data = Vec::new();
        for i in 0..20_000u32 {
            data.extend_from_slice(
                format!("experiments/gp-teacher-v6/contexts/{:04}/", i % 97).as_bytes(),
            );
            data.extend_from_slice(&i.to_le_bytes());
        }
        let c = compress(&data);
        assert!(c.len() < data.len() / 3, "{} vs {}", c.len(), data.len());
        assert_eq!(decompress(&c, data.len()).unwrap(), data);
        for n in [0usize, 1, 3, 4, 59, 60, 61, 255, 256, 70_000] {
            let d: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
            assert_eq!(decompress(&compress(&d), n).unwrap(), d, "n={n}");
        }
    }
}
