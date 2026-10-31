//! The verified asset-source interface model loaders consume (SPECIFICATION.md §9).
//!
//! A loader receives [`VerifiedBytes`] for a logical id rather than a path. Only this crate can
//! construct one, and only after the bytes matched the authenticated manifest's sha256, so holding
//! one proves the package check happened. Loaders still check their own semantic identity
//! (tensor inventory, architecture, binding); the two checks are different and neither replaces
//! the other.

use crate::{AssetError, Entry, Manifest};
use std::sync::Arc;

/// Read authenticated assets by logical id without exposing package storage paths.
pub trait VerifiedAssetSource: Send + Sync {
    /// Return the authenticated manifest describing this source.
    fn manifest(&self) -> &Manifest;
    /// Read an asset and verify its bytes against the manifest digest.
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

    /// Borrow the authenticated decoded bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the logical manifest id of these bytes.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Return the lowercase SHA-256 digest verified for these bytes.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The bytes, copied out only if another holder shares them.
    pub fn to_vec(&self) -> Vec<u8> {
        self.bytes.to_vec()
    }
}
