//! The verified asset-source interface model loaders consume (SPECIFICATION.md §9).
//!
//! A loader receives [`VerifiedBytes`] for a logical id rather than a path. Only this crate can
//! construct one, and only after the bytes matched the authenticated manifest's sha256, so holding
//! one proves the package check happened. Loaders still check their own semantic identity
//! (tensor inventory, architecture, binding); the two checks are different and neither replaces
//! the other.

use crate::{AssetError, Entry, Manifest};
use std::sync::Arc;

pub trait VerifiedAssetSource: Send + Sync {
    fn manifest(&self) -> &Manifest;
    fn read_verified(&self, id: &str) -> Result<VerifiedBytes, AssetError>;
}

/// Decoded asset bytes that matched the manifest.
#[derive(Clone, Debug)]
pub struct VerifiedBytes {
    bytes: Arc<[u8]>,
    id: String,
    sha256: String,
}

impl VerifiedBytes {
    pub(crate) fn new(entry: &Entry, bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes.into(),
            id: entry.id.clone(),
            sha256: entry.sha256.clone(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The bytes, copied out only if another holder shares them.
    pub fn to_vec(&self) -> Vec<u8> {
        self.bytes.to_vec()
    }
}
