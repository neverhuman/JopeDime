//! `publish`: upload a bundle's blobs create-only (`If-None-Match: *`), then its manifest last.
//! Existing blobs are never rewritten; an existing manifest must be byte-identical.

use crate::manifest::BundleManifest;
use anyhow::{anyhow, Context, Result};
use jope_datasets::blobs::blob_key;
use jope_datasets::hash::digest_file;
use jope_datasets::store::{PutOutcome, Store};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct PublishReport {
    pub bundle_id: String,
    pub blobs_created: usize,
    pub blobs_existing: usize,
    pub bytes_uploaded: u64,
    /// `created` or `identical` (already published with the same bytes).
    pub manifest: String,
}

/// Hash every `<from>/<path>` against the manifest first (nothing is uploaded when any file
/// differs), upload the missing blobs, then the manifest.
pub async fn publish(store: &Store, manifest_path: &Path, from: &Path) -> Result<PublishReport> {
    let bytes = std::fs::read(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest = BundleManifest::parse(&bytes)
        .map_err(|e| anyhow!("{}: {}", manifest_path.display(), e.join("; ")))?;
    for f in &manifest.files {
        let src = from.join(&f.path);
        let d = digest_file(&src).with_context(|| format!("hash {}", src.display()))?;
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
    for f in &manifest.files {
        let key = blob_key(&f.sha256);
        if let Some(size) = store.size(&key).await? {
            if size != f.bytes {
                return Err(anyhow!(
                    "{key}: exists in {} with {size} bytes, manifest says {}",
                    store.url,
                    f.bytes
                ));
            }
            report.blobs_existing += 1;
            continue;
        }
        let data = std::fs::read(from.join(&f.path))?;
        match store.put_if_absent(&key, data).await? {
            PutOutcome::Created => {
                report.blobs_created += 1;
                report.bytes_uploaded += f.bytes;
            }
            PutOutcome::Exists => report.blobs_existing += 1,
        }
    }
    let key = crate::manifest_key(&manifest.bundle_id);
    report.manifest = match store.put_if_absent(&key, bytes.clone()).await? {
        PutOutcome::Created => "created".into(),
        PutOutcome::Exists => {
            let existing = store.get_bytes(&key).await?.unwrap_or_default();
            if existing == bytes {
                "identical".into()
            } else {
                return Err(anyhow!(
                    "{key} already exists in {} with different content; bundles are immutable — \
                     publish under a new bundle_id",
                    store.url
                ));
            }
        }
    };
    Ok(report)
}
