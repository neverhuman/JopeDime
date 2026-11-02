use super::*;

/// How an immutable object reached verified publication closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationStatus {
    /// Object was newly created by `put_if_absent` and verified on readback.
    Created,
    /// Object already existed before or during `put_if_absent` (create race) and its body
    /// was verified byte-for-byte on readback.
    ExistingVerified,
    /// The `put_if_absent` response was lost or ambiguous after write, and bounded readback
    /// proved the exact expected bytes (and required physical version) are present.
    ReconciledAfterAmbiguousPut,
}

impl PublicationStatus {
    /// Map to [`PutOutcome`] for legacy counter compatibility (`MirrorReport`, `PublishReport`).
    pub fn as_put_outcome(self) -> PutOutcome {
        match self {
            PublicationStatus::Created | PublicationStatus::ReconciledAfterAmbiguousPut => {
                PutOutcome::Created
            }
            PublicationStatus::ExistingVerified => PutOutcome::Exists,
        }
    }
}

/// Truthful physical-version availability classification for a verified object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalVersionStatus {
    /// Backend returned a non-empty physical `VersionId` with the readback body and exact-version
    /// readback confirmed those identical bytes.
    VerifiedVersionId,
    /// Local filesystem store (`file://`) verified content bytes; backend has no S3 `VersionId`.
    UnversionedLocalFixture,
    /// In-memory store (`memory://`) verified content bytes; backend has no S3 `VersionId`.
    UnversionedMemoryFixture,
    /// Another unversioned fixture backend verified content bytes without a physical `VersionId`.
    UnversionedBackend,
}

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

/// Data-only serializable receipt and reference for a content-verified immutable object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedObjectRef {
    pub schema: String,
    pub store_url: String,
    pub backend: String,
    pub key: String,
    pub resolved_key: String,
    pub expected_sha256: String,
    pub observed_sha256: String,
    pub expected_bytes: u64,
    pub observed_bytes: u64,
    /// Opaque concurrency token (`ETag`) returned with the readback body; never treated as a
    /// content SHA-256 or a physical `VersionId`.
    pub e_tag: Option<String>,
    /// Physical backend `VersionId` returned with the readback body (`None` on unversioned
    /// `file://` or `memory://` stores).
    pub physical_version_id: Option<String>,
    pub version_status: PhysicalVersionStatus,
    pub status: PublicationStatus,
    /// Always `false` in local/injected fixture execution; live S3 durability qualification
    /// requires a separate authorized gate.
    pub live_s3_qualified: bool,
}
