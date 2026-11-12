//! Lowercase SHA-256 hex used by service-local helpers.
//!
//! The bytes are hashed by `jope_runtime::sha256_hex`, which is lowercase hex.

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    jope_runtime::sha256_hex(bytes)
}

/// Lowercase hex of bytes that are already a digest. This does not hash.
pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    jope_runtime::encode_hex(bytes)
}

/// Sixty-four lowercase hex characters. Uppercase is rejected.
pub(crate) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn sha256_hex_matches_the_known_digest_of_abc() {
        let digest = sha256_hex(b"abc");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(super::is_sha256_hex(&digest));
        assert!(!super::is_sha256_hex(&digest.to_ascii_uppercase()));
        assert!(!super::is_sha256_hex("abc"));
        assert_eq!(super::encode_hex(&[0x0a, 0xff]), "0aff");
    }
}
