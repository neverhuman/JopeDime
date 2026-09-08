use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value)
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))
}

pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = canonical_json_bytes(value)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

pub fn bytes_sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn validate_sha256(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{name} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

pub(crate) fn validate_schema(actual: &str, expected: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("schema must be {expected}, found {actual}"));
    }
    Ok(())
}
