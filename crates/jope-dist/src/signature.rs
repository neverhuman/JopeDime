//! Signature files: what `cosign sign-blob --output-signature` writes (base64 of an ASN.1 DER
//! ECDSA signature), raw DER, or the 64-byte `r || s` form, normalized to the low-S fixed form a
//! package carries (SPECIFICATION.md §10.3). No private key is ever read here.
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Decode standard base64 (padding optional, whitespace ignored).
pub fn base64_decode(text: &str) -> Result<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0;
    let mut padding = false;
    for byte in text.bytes().filter(|b| !b.is_ascii_whitespace()) {
        if byte == b'=' {
            padding = true;
            continue;
        }
        if padding {
            bail!("base64 data after padding");
        }
        let value = ALPHABET
            .iter()
            .position(|c| *c == byte)
            .context("invalid base64 character")? as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Standard base64 with padding.
pub fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A signature in any accepted encoding, as the fixed low-S form.
pub fn normalize(bytes: &[u8]) -> Result<[u8; 64]> {
    if bytes.first() == Some(&0x30) {
        if let Ok(fixed) = jope_assets::verify::normalize_der(bytes) {
            return Ok(fixed);
        }
    }
    if bytes.len() == 64 {
        let der = p256_fixed_to_der(bytes)?;
        return jope_assets::verify::normalize_der(&der).map_err(Into::into);
    }
    let text = std::str::from_utf8(bytes).context("signature is neither DER, r||s nor base64")?;
    let decoded = base64_decode(text.trim())?;
    if decoded.len() == 64 {
        return normalize(&decoded);
    }
    jope_assets::verify::normalize_der(&decoded).map_err(Into::into)
}

/// The 64-byte `r || s` form as DER (so one path normalizes every input).
fn p256_fixed_to_der(fixed: &[u8]) -> Result<Vec<u8>> {
    fn integer(bytes: &[u8]) -> Vec<u8> {
        let trimmed: &[u8] = {
            let start = bytes
                .iter()
                .position(|b| *b != 0)
                .unwrap_or(bytes.len() - 1);
            &bytes[start..]
        };
        let mut out = vec![0x02];
        let pad = trimmed[0] & 0x80 != 0;
        out.push((trimmed.len() + usize::from(pad)) as u8);
        if pad {
            out.push(0);
        }
        out.extend_from_slice(trimmed);
        out
    }
    let (r, s) = fixed.split_at(32);
    let body: Vec<u8> = [integer(r), integer(s)].concat();
    let mut der = vec![0x30, body.len() as u8];
    der.extend(body);
    Ok(der)
}

/// Read and normalize a signature file into the package's fixed 64-byte low-S form.
///
/// The file may contain DER, raw `r || s`, or base64-encoded signature bytes. I/O and
/// normalization failures include the path in their error context.
pub fn read(path: &Path) -> Result<[u8; 64]> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    normalize(&bytes).with_context(|| format!("signature {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_assets::verify::{normalize_der, sign_with_development_key};

    #[test]
    fn every_encoding_normalizes_to_the_same_fixed_signature() {
        let der = sign_with_development_key(b"message");
        let fixed = normalize_der(&der).unwrap();
        assert_eq!(normalize(&der).unwrap(), fixed);
        assert_eq!(normalize(&fixed).unwrap(), fixed);
        assert_eq!(normalize(base64_encode(&der).as_bytes()).unwrap(), fixed);
        assert_eq!(
            normalize(format!("{}\n", base64_encode(&der)).as_bytes()).unwrap(),
            fixed
        );
        assert!(normalize(b"not a signature").is_err());
        assert_eq!(base64_decode(&base64_encode(b"ab")).unwrap(), b"ab");
        assert!(base64_decode("QQ==QQ").is_err());
    }
}
