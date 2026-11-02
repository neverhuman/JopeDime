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

mod options;

pub use options::{VerificationOptions, VersionRequirement, DEFAULT_MAX_VERIFIED_READBACK_BYTES};

mod contract;
pub use contract::{PhysicalVersionStatus, PublicationStatus, VerifiedObjectRef};

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

mod validation;
use validation::*;
