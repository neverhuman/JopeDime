use super::*;

pub(crate) fn validate_sha256_hex(sha: &str) -> Result<()> {
    ensure!(
        sha.len() == 64 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "invalid expected sha256 {sha:?}: must be 64 lowercase hex digits"
    );
    Ok(())
}

pub(crate) fn validate_key_and_expected_hash(rel: &str, expected_sha256: &str) -> Result<()> {
    validate_sha256_hex(expected_sha256)?;
    ensure!(!rel.is_empty(), "object key must not be empty");
    ensure!(
        !rel.split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == ".."),
        "invalid object key {rel:?}: empty or traversal segments are forbidden"
    );
    if let Some(key_hash) = rel.strip_prefix("blobs/sha256/") {
        ensure!(
            key_hash == expected_sha256,
            "content-addressed key {rel} does not match expected sha256 {expected_sha256}"
        );
    }
    Ok(())
}

pub(crate) fn backend_name(scheme: &ObjectStoreScheme) -> String {
    match scheme {
        ObjectStoreScheme::Local => "local".to_owned(),
        ObjectStoreScheme::Memory => "memory".to_owned(),
        ObjectStoreScheme::AmazonS3 => "amazon_s3".to_owned(),
        ObjectStoreScheme::GoogleCloudStorage => "google_cloud_storage".to_owned(),
        ObjectStoreScheme::MicrosoftAzure => "microsoft_azure".to_owned(),
        ObjectStoreScheme::Http => "http".to_owned(),
        _ => "custom".to_owned(),
    }
}

pub(crate) fn is_non_reconcilable_put_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_ascii_lowercase();
    msg.contains("accessdenied")
        || msg.contains("access denied")
        || msg.contains("permissiondenied")
        || msg.contains("permission denied")
        || msg.contains("forbidden")
        || msg.contains("unauthenticated")
        || msg.contains("unauthorized")
        || msg.contains("invalidaccesskeyid")
        || msg.contains("signaturedoesnotmatch")
        || msg.contains("before-write")
        || msg.contains("denied")
}
