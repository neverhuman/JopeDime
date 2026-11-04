/// Default maximum object size (`256 MiB`) permitted for in-memory readback verification.
pub const DEFAULT_MAX_VERIFIED_READBACK_BYTES: u64 = 256 * 1024 * 1024;

/// Whether publication requires a backend physical `VersionId` or permits unversioned local/memory
/// fixtures (`file://`, `memory://`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionRequirement {
    /// Allow unversioned `file://` or `memory://` fixture stores (recording their unversioned
    /// status truthfully), while still requiring `VersionId` if the backend is `AmazonS3` or
    /// returns a version on `PUT`.
    AllowUnversionedFixture,
    /// Require a non-empty physical `VersionId` returned with the readback body and verified via
    /// exact-version `GET`; refuse any backend or response that lacks a physical `VersionId`.
    RequirePhysicalVersionId,
}

/// Options governing verified immutable publication and readback bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerificationOptions {
    pub version_requirement: VersionRequirement,
    pub max_readback_bytes: u64,
}

impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            version_requirement: VersionRequirement::AllowUnversionedFixture,
            max_readback_bytes: DEFAULT_MAX_VERIFIED_READBACK_BYTES,
        }
    }
}

impl VerificationOptions {
    pub fn require_physical_version_id() -> Self {
        Self {
            version_requirement: VersionRequirement::RequirePhysicalVersionId,
            max_readback_bytes: DEFAULT_MAX_VERIFIED_READBACK_BYTES,
        }
    }

    pub fn with_max_readback_bytes(mut self, max_readback_bytes: u64) -> Self {
        self.max_readback_bytes = max_readback_bytes;
        self
    }
}
