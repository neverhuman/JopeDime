//! Canonical JSON and sha256 helpers.
//!
//! Canonical JSON = compact serialisation with object keys sorted lexicographically (bytewise),
//! no whitespace, floats in serde_json's shortest round-trip form. `serde_json::Value` objects
//! are `BTreeMap`s in this workspace (no `preserve_order`), so serialising through `Value`
//! yields exactly that; a test below fails if that assumption ever changes.

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Performs the SHA-256 hex operation and returns an error if it cannot be completed.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// `value` as canonical JSON bytes.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let v = serde_json::to_value(value)?;
    Ok(serde_json::to_vec(&v)?)
}

/// sha256 of the canonical JSON of `value`.
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_hex(&canonical_json(value)?))
}

/// Returns whether hex64.
pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn value_objects_sort_their_keys() {
        // serde_json without `preserve_order`: canonical order falls out of BTreeMap.
        assert_eq!(
            String::from_utf8(canonical_json(&json!({"b": 1, "a": {"z": 0, "y": 0}})).unwrap())
                .unwrap(),
            r#"{"a":{"y":0,"z":0},"b":1}"#
        );
    }

    #[test]
    fn sha256_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(is_hex64(&sha256_hex(b"")));
        assert!(!is_hex64("unbound"));
    }
}
