//! Bounded custody-v2 gzip/TAR verification. Paths are inspected, never extracted.
use crate::{Error, Result};
use flate2::{read::MultiGzDecoder, Compression, GzBuilder};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    sync::Arc,
};

pub const MAX_PAYLOAD: u64 = 64 * 1024 * 1024;
pub const MAX_COMPRESSED: u64 = 128 * 1024 * 1024;
pub const MAX_MEMBERS: usize = 512;

#[derive(Clone, Debug)]
pub struct Blob {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

pub fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn io(error: std::io::Error) -> Error {
    invalid(&format!("custody archive I/O: {error}"))
}

pub fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn expected(blobs: &[Blob]) -> Result<(BTreeMap<String, &Blob>, u64)> {
    check(
        !blobs.is_empty() && blobs.len() <= MAX_MEMBERS,
        "bounded blob roster",
    )?;
    let mut result = BTreeMap::new();
    let mut payload = 0u64;
    for blob in blobs {
        check(
            is_digest(&blob.sha256) && blob.name == format!("members/{}", blob.sha256),
            "safe content member",
        )?;
        check(
            result.insert(blob.name.clone(), blob).is_none(),
            "duplicate blob declaration",
        )?;
        payload = payload
            .checked_add(blob.bytes)
            .ok_or_else(|| invalid("payload overflow"))?;
    }
    check(payload <= MAX_PAYLOAD, "bounded unpacked bytes")?;
    Ok((result, payload))
}

fn octal(bytes: &[u8]) -> Result<u64> {
    let value = std::str::from_utf8(bytes)
        .map_err(|_| invalid("invalid TAR number"))?
        .trim_matches(|c| c == '\0' || c == ' ');
    if value.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(value, 8).map_err(|_| invalid("invalid TAR number"))
}

fn text(bytes: &[u8]) -> Result<&str> {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).map_err(|_| invalid("invalid TAR name"))
}

#[derive(Default, Clone)]
struct Pax {
    path: Option<String>,
    size: Option<u64>,
}

fn pax(raw: &[u8], state: &mut Pax) -> Result<()> {
    let mut pos = 0;
    while pos < raw.len() {
        let space = raw[pos..]
            .iter()
            .position(|b| *b == b' ')
            .ok_or_else(|| invalid("invalid PAX length"))?
            + pos;
        let digits = &raw[pos..space];
        check(
            !digits.is_empty() && digits.iter().all(u8::is_ascii_digit),
            "invalid PAX length",
        )?;
        let size: usize = text(digits)?
            .parse()
            .map_err(|_| invalid("invalid PAX length"))?;
        let end = pos
            .checked_add(size)
            .ok_or_else(|| invalid("PAX size overflow"))?;
        check(
            end <= raw.len() && end > space + 2 && raw[end - 1] == b'\n',
            "invalid PAX record",
        )?;
        let record = std::str::from_utf8(&raw[space + 1..end - 1])
            .map_err(|_| invalid("invalid PAX text"))?;
        let (key, value) = record
            .split_once('=')
            .ok_or_else(|| invalid("invalid PAX record"))?;
        match key {
            "path" => state.path = Some(value.to_owned()),
            "size" => {
                check(
                    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                    "invalid PAX size",
                )?;
                state.size = Some(value.parse().map_err(|_| invalid("invalid PAX size"))?);
            }
            "linkpath" => return Err(invalid("archive links refused")),
            _ => {} // Other PAX metadata cannot affect the inspected name/type/size.
        }
        pos = end;
    }
    Ok(())
}

/// Authenticate the outer bytes, entire gzip stream, and every declared payload.
pub fn verify(mut reader: impl Read, outer_sha: &str, blobs: &[Blob]) -> Result<()> {
    check(is_digest(outer_sha), "content-addressed pack SHA")?;
    let (expected, payload) = expected(blobs)?;
    let mut compressed = Vec::new();
    reader
        .by_ref()
        .take(MAX_COMPRESSED + 1)
        .read_to_end(&mut compressed)
        .map_err(io)?;
    check(
        compressed.len() as u64 <= MAX_COMPRESSED,
        "bounded compressed pack",
    )?;
    check(digest(&compressed) == outer_sha, "pack SHA mismatch")?;
    let limit = payload + 1024 * blobs.len() as u64 + 10240;
    let mut decoded = Vec::new();
    MultiGzDecoder::new(compressed.as_slice())
        .take(limit + 1)
        .read_to_end(&mut decoded)
        .map_err(io)?;
    check(decoded.len() as u64 <= limit, "bounded expanded tar bytes")?;
    let mut pos = 0usize;
    let mut seen = BTreeSet::new();
    let mut global = Pax::default();
    let mut local = Pax::default();
    loop {
        check(pos + 512 <= decoded.len(), "missing TAR completion")?;
        let header = &decoded[pos..pos + 512];
        if header.iter().all(|b| *b == 0) {
            check(
                decoded.len() - pos >= 1024 && (decoded.len() - pos).is_multiple_of(512),
                "incomplete TAR completion",
            )?;
            check(
                decoded[pos..].iter().all(|b| *b == 0),
                "undeclared TAR tail",
            )?;
            check(
                local.path.is_none() && local.size.is_none(),
                "unconsumed PAX metadata",
            )?;
            break;
        }
        let checksum: u64 = header
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if (148..156).contains(&i) {
                    32
                } else {
                    u64::from(*b)
                }
            })
            .sum();
        check(
            octal(&header[148..156])? == checksum,
            "TAR checksum mismatch",
        )?;
        check(
            &header[257..263] == b"ustar\0" || &header[257..263] == b"ustar ",
            "TAR format refused",
        )?;
        let name = text(&header[..100])?;
        let prefix = text(&header[345..500])?;
        let header_name = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}/{name}")
        };
        let raw_size = octal(&header[124..136])?;
        let kind = header[156];
        let size = if matches!(kind, b'x' | b'g') {
            raw_size
        } else {
            local.size.or(global.size).unwrap_or(raw_size)
        };
        let size = usize::try_from(size).map_err(|_| invalid("TAR size overflow"))?;
        let begin = pos + 512;
        let end = begin
            .checked_add(size)
            .ok_or_else(|| invalid("TAR size overflow"))?;
        let padded = end
            .checked_add((512 - size % 512) % 512)
            .ok_or_else(|| invalid("TAR size overflow"))?;
        check(padded <= decoded.len(), "truncated TAR member")?;
        check(
            decoded[end..padded].iter().all(|b| *b == 0),
            "invalid TAR padding",
        )?;
        let body = &decoded[begin..end];
        if matches!(kind, b'x' | b'g') {
            if kind == b'g' {
                pax(body, &mut global)?;
            } else {
                pax(body, &mut local)?;
            }
        } else {
            check(
                matches!(kind, 0 | b'0'),
                "unexpected or duplicate archive member",
            )?;
            let name = local
                .path
                .as_ref()
                .or(global.path.as_ref())
                .unwrap_or(&header_name);
            let blob = expected
                .get(name)
                .ok_or_else(|| invalid("unexpected or duplicate archive member"))?;
            check(
                seen.insert(name.clone()),
                "unexpected or duplicate archive member",
            )?;
            check(size as u64 == blob.bytes, "archive size mismatch")?;
            check(digest(body) == blob.sha256, "member SHA mismatch")?;
            local = Pax::default();
        }
        pos = padded;
    }
    check(seen.len() == expected.len(), "missing archive member")
}

/// Produce the same v2 CAS member layout; gzip bytes need not equal another codec's output.
pub fn encode(blobs: &BTreeMap<String, Arc<[u8]>>) -> Result<Vec<u8>> {
    let roster: Vec<_> = blobs
        .iter()
        .map(|(sha, body)| Blob {
            name: format!("members/{sha}"),
            sha256: sha.clone(),
            bytes: body.len() as u64,
        })
        .collect();
    expected(&roster)?;
    let mut tar = Vec::new();
    for (sha, body) in blobs {
        check(digest(body) == *sha, "member SHA mismatch")?;
        let mut header = [0u8; 512];
        let name = format!("members/{sha}");
        header[..name.len()].copy_from_slice(name.as_bytes());
        for (range, value) in [
            (100..108, 0o600u64),
            (108..116, 0),
            (116..124, 0),
            (124..136, body.len() as u64),
            (136..148, 0),
        ] {
            let value = format!("{:0width$o}\0", value, width = range.len() - 1);
            header[range].copy_from_slice(value.as_bytes());
        }
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum: u64 = header.iter().map(|b| u64::from(*b)).sum();
        header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
        tar.extend_from_slice(&header);
        tar.extend_from_slice(body);
        tar.resize(tar.len() + (512 - body.len() % 512) % 512, 0);
    }
    tar.resize(tar.len() + 1024, 0);
    tar.resize(tar.len().div_ceil(10240) * 10240, 0);
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::default());
    encoder.write_all(&tar).map_err(io)?;
    let compressed = encoder.finish().map_err(io)?;
    check(
        compressed.len() as u64 <= MAX_COMPRESSED,
        "bounded compressed pack",
    )?;
    Ok(compressed)
}
