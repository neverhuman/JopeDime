use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Serialize;

use super::contracts::{raw_sha256, validate_prediction_binding, PredictionCacheKeyV1};
use crate::search_v3::{canonical_json_bytes, validate_sha256, PredictionObjectV1};

#[derive(Debug, Clone)]
pub struct ArtifactStoreV1 {
    root: PathBuf,
}

impl ArtifactStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn object_path(&self, digest: &str) -> PathBuf {
        self.root
            .join("objects/sha256")
            .join(&digest[..2])
            .join(format!("{digest}.json"))
    }

    fn logical_path(&self, logical_identity: &str) -> PathBuf {
        self.root
            .join("identities")
            .join(format!("{logical_identity}.ref"))
    }

    fn receipt_path(&self, receipt_identity: &str) -> PathBuf {
        self.root
            .join("receipts")
            .join(format!("{receipt_identity}.json"))
    }

    fn create_exact(path: &Path, bytes: &[u8]) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                file.write_all(bytes)
                    .map_err(|error| format!("write {}: {error}", path.display()))?;
                file.sync_all()
                    .map_err(|error| format!("sync {}: {error}", path.display()))?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let present = fs::read(path)
                    .map_err(|read| format!("read existing {}: {read}", path.display()))?;
                if present == bytes {
                    Ok(())
                } else {
                    Err(format!("immutable output collision at {}", path.display()))
                }
            }
            Err(error) => Err(format!("create {}: {error}", path.display())),
        }
    }

    /// Publishes a content-addressed object and binds one immutable logical
    /// identity to it. Retrying identical bytes is idempotent; another output
    /// for the same identity is rejected.
    pub fn publish<T: Serialize>(
        &self,
        logical_identity: &str,
        value: &T,
    ) -> Result<String, String> {
        validate_sha256("logical artifact identity", logical_identity)?;
        let bytes = canonical_json_bytes(value)?;
        let digest = raw_sha256(&bytes);
        Self::create_exact(&self.object_path(&digest), &bytes)?;
        Self::create_exact(
            &self.logical_path(logical_identity),
            format!("{digest}\n").as_bytes(),
        )?;
        Ok(digest)
    }

    pub fn load<T: DeserializeOwned>(&self, logical_identity: &str) -> Result<T, String> {
        validate_sha256("logical artifact identity", logical_identity)?;
        let reference = fs::read_to_string(self.logical_path(logical_identity))
            .map_err(|error| format!("read artifact identity: {error}"))?;
        let digest = reference.trim();
        validate_sha256("artifact object reference", digest)?;
        let bytes = fs::read(self.object_path(digest))
            .map_err(|error| format!("read artifact object: {error}"))?;
        if raw_sha256(&bytes) != digest {
            return Err("content-addressed artifact is corrupt".into());
        }
        serde_json::from_slice(&bytes).map_err(|error| format!("decode artifact object: {error}"))
    }

    pub fn contains(&self, logical_identity: &str) -> Result<bool, String> {
        validate_sha256("logical artifact identity", logical_identity)?;
        Ok(self.logical_path(logical_identity).is_file())
    }

    pub fn object_sha256(&self, logical_identity: &str) -> Result<String, String> {
        validate_sha256("logical artifact identity", logical_identity)?;
        let reference = fs::read_to_string(self.logical_path(logical_identity))
            .map_err(|error| format!("read artifact identity: {error}"))?;
        let digest = reference.trim().to_string();
        validate_sha256("artifact object reference", &digest)?;
        Ok(digest)
    }

    /// Append-only receipt identity. There is deliberately no overwrite or
    /// truncate path.
    pub fn append_receipt<T: Serialize>(
        &self,
        receipt_identity: &str,
        value: &T,
    ) -> Result<String, String> {
        validate_sha256("receipt identity", receipt_identity)?;
        let bytes = canonical_json_bytes(value)?;
        Self::create_exact(&self.receipt_path(receipt_identity), &bytes)?;
        Ok(raw_sha256(&bytes))
    }
}

#[derive(Debug, Clone)]
pub struct PredictionCacheV1 {
    store: ArtifactStoreV1,
}

impl PredictionCacheV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            store: ArtifactStoreV1::new(root),
        }
    }

    pub fn publish(
        &self,
        key: &PredictionCacheKeyV1,
        prediction: &PredictionObjectV1,
    ) -> Result<String, String> {
        validate_prediction_binding(key, prediction)?;
        self.store.publish(&key.identity()?, prediction)
    }

    pub fn load(&self, key: &PredictionCacheKeyV1) -> Result<Option<PredictionObjectV1>, String> {
        key.validate()?;
        let identity = key.identity()?;
        if !self.store.logical_path(&identity).is_file() {
            return Ok(None);
        }
        let prediction: PredictionObjectV1 = self.store.load(&identity)?;
        validate_prediction_binding(key, &prediction)?;
        Ok(Some(prediction))
    }
}
