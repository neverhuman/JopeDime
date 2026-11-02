use super::*;
use sha2::{Digest, Sha256};

pub(super) fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

pub(super) fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(hex::encode(Sha256::digest(canonical_bytes(value)?)))
}

pub(super) fn domain_sha256(domain: &str, fields: impl Serialize) -> Result<String> {
    canonical_sha256(&serde_json::json!({"domain": domain, "fields": fields}))
}

pub(super) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(super) fn durable_write_new(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    let parent = path.parent().context("durable path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    sync_directory(parent)
}
