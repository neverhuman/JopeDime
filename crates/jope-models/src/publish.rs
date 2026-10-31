//! `publish`: upload a bundle's blobs create-only (`If-None-Match: *`), then its manifest last.
//! Existing blobs are never rewritten; every existing or newly created blob and manifest is
//! verified by full SHA-256 content and byte length (plus physical `VersionId` when required)
//! before publication reports success.

use crate::manifest::BundleManifest;
use anyhow::{anyhow, Context, Result};
use jope_datasets::blobs::blob_key;
use jope_datasets::hash::{digest_content_async, sha256_hex};
use jope_datasets::store::Store;
use jope_datasets::verified_object::{
    publish_verified_object_with_options, snapshot_and_verify_source_file, PublicationStatus,
    VerificationOptions, VerifiedObjectRef,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Verified outcome of publishing one model bundle.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishReport {
    /// Bundle identifier published.
    pub bundle_id: String,
    /// Number of newly created blob objects.
    pub blobs_created: usize,
    /// Number of existing, content-verified blob objects reused.
    pub blobs_existing: usize,
    /// Total bytes uploaded for newly created blobs.
    pub bytes_uploaded: u64,
    /// `created` or `identical` (already published with the same bytes).
    pub manifest: String,
    /// Verified object references for each bundle file, in manifest order.
    pub verified_blobs: Vec<VerifiedObjectRef>,
    /// Verified object reference for the bundle manifest itself (`Some` on completed publish).
    pub verified_manifest: Option<VerifiedObjectRef>,
}

/// Callback type invoked around source file snapshotting during [`publish`].
pub type SnapshotHook<'a> = &'a mut dyn FnMut(&str, &Path);

/// Optional deterministic hooks for testing source mutation barriers during [`publish`].
#[derive(Default)]
pub struct PublishHooks<'a> {
    /// Invoked after the initial prehash pass completes, immediately before reading a file's
    /// immutable upload snapshot.
    pub on_before_snapshot: Option<SnapshotHook<'a>>,
    /// Invoked after a file's immutable upload snapshot has been read and verified in memory,
    /// immediately before uploading it to the store.
    pub on_after_snapshot: Option<SnapshotHook<'a>>,
}

/// Hash every `<from>/<path>` against the manifest first (nothing is uploaded when any file
/// differs), verify and upload each blob through [`jope_datasets::verified_object`], then publish
/// and verify the manifest last.
pub async fn publish(store: &Store, manifest_path: &Path, from: &Path) -> Result<PublishReport> {
    publish_with_options(store, manifest_path, from, VerificationOptions::default()).await
}

/// [`publish`] with explicit [`VerificationOptions`] (e.g. requiring physical `VersionId` on
/// versioned backends).
pub async fn publish_with_options(
    store: &Store,
    manifest_path: &Path,
    from: &Path,
    opts: VerificationOptions,
) -> Result<PublishReport> {
    let mut hooks = PublishHooks::default();
    publish_with_options_and_hooks(store, manifest_path, from, opts, &mut hooks).await
}

/// [`publish_with_options`] with deterministic source-mutation test hooks.
pub async fn publish_with_options_and_hooks(
    store: &Store,
    manifest_path: &Path,
    from: &Path,
    opts: VerificationOptions,
    hooks: &mut PublishHooks<'_>,
) -> Result<PublishReport> {
    let bytes = std::fs::read(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest = BundleManifest::parse(&bytes)
        .map_err(|e| anyhow!("{}: {}", manifest_path.display(), e.join("; ")))?;

    // 1. Prehash every source file before touching the store so any upfront mismatch aborts
    //    before uploading a single blob.
    for f in &manifest.files {
        let src = from.join(&f.path);
        let d = digest_content_async(src.clone())
            .await
            .with_context(|| format!("hash {}", src.display()))?;
        if d.sha256 != f.sha256 || d.bytes != f.bytes {
            return Err(anyhow!(
                "{}: {} ({} bytes) differs from the manifest ({} , {} bytes)",
                src.display(),
                d.sha256,
                d.bytes,
                f.sha256,
                f.bytes
            ));
        }
    }

    let mut report = PublishReport {
        bundle_id: manifest.bundle_id.clone(),
        ..Default::default()
    };

    // 2. For each file, take a verified in-memory snapshot and verify its publication in the
    //    store by full content hash (never by HEAD size alone).
    for f in &manifest.files {
        let src = from.join(&f.path);
        if let Some(hook) = hooks.on_before_snapshot.as_deref_mut() {
            hook(&f.path, &src);
        }
        let snapshot = snapshot_and_verify_source_file(&src, &f.sha256, f.bytes)?;
        if let Some(hook) = hooks.on_after_snapshot.as_deref_mut() {
            hook(&f.path, &src);
        }

        let key = blob_key(&f.sha256);
        let verified =
            publish_verified_object_with_options(store, &key, &snapshot, &f.sha256, f.bytes, opts)
                .await?;

        match verified.reference().status {
            PublicationStatus::Created | PublicationStatus::ReconciledAfterAmbiguousPut => {
                report.blobs_created += 1;
                report.bytes_uploaded += f.bytes;
            }
            PublicationStatus::ExistingVerified => {
                report.blobs_existing += 1;
            }
        }
        report.verified_blobs.push(verified.into_reference());
    }

    // 3. Publish the manifest last, only after every blob reference is content-verified.
    let key = crate::manifest_key(&manifest.bundle_id);
    let manifest_sha256 = sha256_hex(&bytes);
    let manifest_len = bytes.len() as u64;
    let verified_manifest = publish_verified_object_with_options(
        store,
        &key,
        &bytes,
        &manifest_sha256,
        manifest_len,
        opts,
    )
    .await
    .map_err(|e| {
        let msg = format!("{e:#}");
        if msg.contains("different bytes") {
            anyhow!(
                "{key} already exists in {} with different content; bundles are immutable — \
                 publish under a new bundle_id ({msg})",
                store.url
            )
        } else {
            e
        }
    })?;

    report.manifest = match verified_manifest.reference().status {
        PublicationStatus::Created | PublicationStatus::ReconciledAfterAmbiguousPut => {
            "created".into()
        }
        PublicationStatus::ExistingVerified => "identical".into(),
    };
    report.verified_manifest = Some(verified_manifest.into_reference());
    Ok(report)
}
