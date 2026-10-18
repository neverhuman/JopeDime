//! Verified immutable object publication (`I46`, contract `jope.verified-object-ref.v1`).
//!
//! Publication succeeds only when the stored object is read back and proven to match the
//! expected byte length and SHA-256 content digest, with an explicit physical-version binding
//! (`VersionId`) separate from any opaque concurrency token (`ETag`).
//!
//! - Preexisting objects (`Exists` or pre-upload check) are verified by full content hash,
//!   never by `HEAD` size alone.
//! - Newly created objects (`Created`) are verified by reading back the stored body and its
//!   associated version metadata from one `GET`.
//! - Ambiguous / lost-response PUT errors reconcile only when a bounded readback proves the
//!   intended key holds the exact expected bytes and satisfies the version requirement;
//!   access denials and pre-write errors never reconcile as publication.
//! - Existing wrong immutable objects are never overwritten, repaired in place, or deleted.

use crate::hash::sha256_hex;
use crate::store::{ObjectStoreScheme, PutOutcome, Store};
use anyhow::{anyhow, bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Canonical wire-contract identifier for verified object references (`I02` / `I46`).
pub const VERIFIED_OBJECT_CONTRACT_ID: &str = "jope.verified-object-ref.v1";

/// Default maximum object size (`256 MiB`) permitted for in-memory readback verification.
pub const DEFAULT_MAX_VERIFIED_READBACK_BYTES: u64 = 256 * 1024 * 1024;

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

/// Proof witness that an immutable object in [`Store`] has been verified by content, byte length,
/// and physical version policy.
///
/// The private field prevents construction outside this module's verification functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedObject {
    reference: VerifiedObjectRef,
    _private: (),
}

impl VerifiedObject {
    /// Borrow the underlying serializable [`VerifiedObjectRef`].
    pub fn reference(&self) -> &VerifiedObjectRef {
        &self.reference
    }

    /// Consume this witness and return the serializable [`VerifiedObjectRef`].
    pub fn into_reference(self) -> VerifiedObjectRef {
        self.reference
    }

    /// Legacy [`PutOutcome`] corresponding to how this verified object was published.
    pub fn put_outcome(&self) -> PutOutcome {
        self.reference.status.as_put_outcome()
    }
}

/// Read a local source file once into an immutable memory buffer and verify that those exact bytes
/// match `expected_sha256` and `expected_bytes`.
///
/// Uploading the returned `Vec<u8>` closes the prehash-to-upload TOCTOU gap:
/// - If the file was mutated before or during this snapshot read, the digest check fails before any
///   upload or manifest write.
/// - If the file on disk is mutated after this snapshot returns, the caller uploads the frozen
///   verified `Vec<u8>` rather than re-reading the mutated path.
pub fn snapshot_and_verify_source_file(
    path: &Path,
    expected_sha256: &str,
    expected_bytes: u64,
) -> Result<Vec<u8>> {
    validate_sha256_hex(expected_sha256)?;
    let bytes = std::fs::read(path).map_err(|e| anyhow!("read {}: {e}", path.display()))?;
    let actual_bytes = bytes.len() as u64;
    let actual_sha256 = sha256_hex(&bytes);
    if actual_sha256 != expected_sha256 || actual_bytes != expected_bytes {
        bail!(
            "{}: {} ({} bytes) differs from the manifest ({} , {} bytes)",
            path.display(),
            actual_sha256,
            actual_bytes,
            expected_sha256,
            expected_bytes
        );
    }
    Ok(bytes)
}

/// Publish an immutable object with default [`VerificationOptions`] (`AllowUnversionedFixture`,
/// `256 MiB` readback bound) and verify its stored content and version metadata.
pub async fn publish_verified_object(
    store: &Store,
    rel: &str,
    bytes: &[u8],
    expected_sha256: &str,
    expected_bytes: u64,
) -> Result<VerifiedObject> {
    publish_verified_object_with_options(
        store,
        rel,
        bytes,
        expected_sha256,
        expected_bytes,
        VerificationOptions::default(),
    )
    .await
}

/// Publish an immutable object create-only (`If-None-Match: *`) and verify the stored content,
/// byte length, and physical version binding before returning [`VerifiedObject`].
pub async fn publish_verified_object_with_options(
    store: &Store,
    rel: &str,
    bytes: &[u8],
    expected_sha256: &str,
    expected_bytes: u64,
    opts: VerificationOptions,
) -> Result<VerifiedObject> {
    validate_key_and_expected_hash(rel, expected_sha256)?;
    ensure!(
        expected_bytes <= opts.max_readback_bytes,
        "{rel}: expected size {expected_bytes} bytes exceeds max verified readback bound ({} bytes)",
        opts.max_readback_bytes
    );

    let payload_bytes = bytes.len() as u64;
    let payload_sha256 = sha256_hex(bytes);
    ensure!(
        payload_bytes == expected_bytes && payload_sha256 == expected_sha256,
        "{rel}: upload buffer ({payload_sha256}, {payload_bytes} bytes) does not match expected ({expected_sha256}, {expected_bytes} bytes)"
    );

    match store.put_if_absent_with_result(rel, bytes.to_vec()).await {
        Ok((PutOutcome::Created, put_version)) => {
            let promised_version = put_version
                .as_ref()
                .and_then(|v| v.version.as_deref())
                .map(str::to_owned);
            verify_stored_object(
                store,
                rel,
                ExpectedContent {
                    slice: Some(bytes),
                    sha256: expected_sha256,
                    bytes: expected_bytes,
                },
                promised_version.as_deref(),
                PublicationStatus::Created,
                opts,
            )
            .await
        }
        Ok((PutOutcome::Exists, _)) => {
            verify_stored_object(
                store,
                rel,
                ExpectedContent {
                    slice: Some(bytes),
                    sha256: expected_sha256,
                    bytes: expected_bytes,
                },
                None,
                PublicationStatus::ExistingVerified,
                opts,
            )
            .await
        }
        Err(put_err) => {
            if is_non_reconcilable_put_error(&put_err) {
                return Err(put_err);
            }
            match verify_stored_object(
                store,
                rel,
                ExpectedContent {
                    slice: Some(bytes),
                    sha256: expected_sha256,
                    bytes: expected_bytes,
                },
                None,
                PublicationStatus::ReconciledAfterAmbiguousPut,
                opts,
            )
            .await
            {
                Ok(verified) => Ok(verified),
                Err(reconcile_err) => Err(anyhow!(
                    "{put_err:#}; reconciliation readback failed: {reconcile_err:#}"
                )),
            }
        }
    }
}

/// Re-verify an already-published object at `rel` (and optional exact `required_version`) against
/// `expected_sha256` and `expected_bytes`.
pub async fn verify_existing_object_at_version(
    store: &Store,
    rel: &str,
    expected_sha256: &str,
    expected_bytes: u64,
    required_version: Option<&str>,
    opts: VerificationOptions,
) -> Result<VerifiedObject> {
    validate_key_and_expected_hash(rel, expected_sha256)?;
    ensure!(
        expected_bytes <= opts.max_readback_bytes,
        "{rel}: expected size {expected_bytes} bytes exceeds max verified readback bound ({} bytes)",
        opts.max_readback_bytes
    );
    verify_stored_object(
        store,
        rel,
        ExpectedContent {
            slice: None,
            sha256: expected_sha256,
            bytes: expected_bytes,
        },
        required_version,
        PublicationStatus::ExistingVerified,
        opts,
    )
    .await
}

struct ExpectedContent<'a> {
    slice: Option<&'a [u8]>,
    sha256: &'a str,
    bytes: u64,
}

async fn verify_stored_object(
    store: &Store,
    rel: &str,
    expected: ExpectedContent<'_>,
    promised_version: Option<&str>,
    status: PublicationStatus,
    opts: VerificationOptions,
) -> Result<VerifiedObject> {
    let expected_sha256 = expected.sha256;
    let expected_bytes = expected.bytes;
    let expected_slice = expected.slice;
    let observed = store
        .get_observed_version(rel, promised_version)
        .await
        .map_err(|e| {
            anyhow!(
                "{rel}: verification readback failed in {}: {e:#}",
                store.url
            )
        })?
        .ok_or_else(|| {
            anyhow!(
                "{rel}: absent on verification readback in {} (promised_version={:?})",
                store.url,
                promised_version
            )
        })?;

    let observed_bytes = observed.bytes.len() as u64;
    ensure!(
        observed_bytes <= opts.max_readback_bytes,
        "{rel}: observed readback size {observed_bytes} bytes in {} exceeds bound {} bytes",
        store.url,
        opts.max_readback_bytes
    );

    let observed_sha256 = sha256_hex(&observed.bytes);
    let content_matches = observed_bytes == expected_bytes
        && observed_sha256 == expected_sha256
        && expected_slice.is_none_or(|b| observed.bytes.as_slice() == b);

    if !content_matches {
        bail!(
            "{rel}: exists in {} with different bytes (expected sha256 {expected_sha256} / {expected_bytes} bytes, observed sha256 {observed_sha256} / {observed_bytes} bytes); immutable objects cannot be overwritten",
            store.url
        );
    }

    let observed_version_id = observed
        .version()
        .filter(|v| !v.is_empty())
        .map(str::to_owned);

    if let Some(expected_ver) = promised_version {
        ensure!(
            observed_version_id.as_deref() == Some(expected_ver),
            "{rel}: physical VersionId mismatch in {} (expected version {expected_ver:?}, observed {:?})",
            store.url,
            observed_version_id
        );
    }

    let backend_scheme = store.backend();
    let backend_label = backend_name(&backend_scheme);

    let version_status = match &observed_version_id {
        Some(ver) => {
            // When promised_version was None (e.g. Exists or ReconciledAfterAmbiguousPut on a
            // versioned store), also verify that an explicit version-pinned GET for `ver` returns
            // the exact same bytes and version.
            if promised_version.is_none() {
                let pinned = store
                    .get_observed_version(rel, Some(ver))
                    .await
                    .map_err(|e| {
                        anyhow!(
                            "{rel}: exact-version readback for version {ver} failed in {}: {e:#}",
                            store.url
                        )
                    })?
                    .ok_or_else(|| {
                        anyhow!(
                            "{rel}: version {ver} disappeared on exact-version readback in {}",
                            store.url
                        )
                    })?;
                ensure!(
                    pinned.version() == Some(ver.as_str())
                        && pinned.bytes.len() as u64 == expected_bytes
                        && sha256_hex(&pinned.bytes) == expected_sha256,
                    "{rel}: exact-version readback for {ver} in {} did not match verified bytes",
                    store.url
                );
            }
            PhysicalVersionStatus::VerifiedVersionId
        }
        None => {
            if opts.version_requirement == VersionRequirement::RequirePhysicalVersionId
                || backend_scheme == ObjectStoreScheme::AmazonS3
            {
                bail!(
                    "{rel}: backend {} ({backend_label}) returned no physical VersionId (e_tag={:?} is an opaque concurrency token, not a VersionId)",
                    store.url,
                    observed.e_tag()
                );
            }
            match backend_scheme {
                ObjectStoreScheme::Local => PhysicalVersionStatus::UnversionedLocalFixture,
                ObjectStoreScheme::Memory => PhysicalVersionStatus::UnversionedMemoryFixture,
                _ => PhysicalVersionStatus::UnversionedBackend,
            }
        }
    };

    Ok(VerifiedObject {
        reference: VerifiedObjectRef {
            schema: VERIFIED_OBJECT_CONTRACT_ID.to_owned(),
            store_url: store.url.clone(),
            backend: backend_label,
            key: rel.to_owned(),
            resolved_key: observed.key_str().to_owned(),
            expected_sha256: expected_sha256.to_owned(),
            observed_sha256,
            expected_bytes,
            observed_bytes,
            e_tag: observed.e_tag().map(str::to_owned),
            physical_version_id: observed_version_id,
            version_status,
            status,
            live_s3_qualified: false,
        },
        _private: (),
    })
}

fn validate_sha256_hex(sha: &str) -> Result<()> {
    ensure!(
        sha.len() == 64 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "invalid expected sha256 {sha:?}: must be 64 lowercase hex digits"
    );
    Ok(())
}

fn validate_key_and_expected_hash(rel: &str, expected_sha256: &str) -> Result<()> {
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

fn backend_name(scheme: &ObjectStoreScheme) -> String {
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

fn is_non_reconcilable_put_error(err: &anyhow::Error) -> bool {
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
