//! `fetch`: resolve a bundle manifest (committed `models/bundles/<id>.json`, then the cache,
//! then the store), print its notices and licence lines, refuse restricted files unless
//! accepted, download blobs by sha256 and materialise `bundles/<id>/…` only after verification.

use crate::cache::ModelsCache;
use crate::index::Repo;
use crate::manifest::{BundleFile, BundleManifest};
use anyhow::{anyhow, Result};
use futures_util::stream::{self, StreamExt};
use jope_datasets::blobs::PARALLEL_GETS;
use jope_datasets::store::Store;
use std::io::Write;

/// Where a fetched bundle manifest was resolved from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestSource {
    /// The committed manifest in the repository checkout.
    Repo,
    /// The local manifest cache.
    Cache,
    /// The configured model store.
    Store,
}

/// Selection and license options for fetching a bundle.
#[derive(Clone, Debug, Default)]
pub struct FetchOptions {
    /// `--component` selectors (exact paths or directory prefixes); empty = every file.
    pub components: Vec<String>,
    /// Permit fetching files marked `redistribution: prohibited` in their license metadata.
    pub accept_restricted: bool,
}

/// Counts and failures produced while resolving and materializing a bundle.
#[derive(Clone, Debug, Default)]
pub struct FetchReport {
    /// Bundle identifier requested.
    pub bundle_id: String,
    /// Number of manifest files selected.
    pub files_selected: usize,
    /// Number of files downloaded during this fetch.
    pub files_fetched: usize,
    /// Number of files already present in the blob cache.
    pub files_cached: usize,
    /// Total bytes downloaded during this fetch.
    pub bytes: u64,
    /// Restricted files that were fetched because `--accept-restricted` was given.
    pub restricted_accepted: Vec<String>,
    /// Per-file download or materialization errors.
    pub errors: Vec<String>,
}

/// Resolves manifests and verified bundle blobs across repository, cache, and store.
pub struct Fetcher<'a> {
    /// Local bundle and content-addressed blob cache.
    pub cache: &'a ModelsCache,
    /// `None` = offline: only the cache is served.
    pub store: Option<&'a Store>,
    /// The checkout whose committed manifests take precedence over the store.
    pub repo: Option<&'a Repo>,
}

impl Fetcher<'_> {
    /// The manifest for `bundle_id`: committed first, then cached, then the store.
    pub async fn manifest(
        &self,
        bundle_id: &str,
    ) -> Result<(Vec<u8>, BundleManifest, ManifestSource)> {
        if let Some(repo) = self.repo {
            if let Some((bytes, m)) = repo.read_manifest(bundle_id)? {
                return Ok((bytes, m, ManifestSource::Repo));
            }
        }
        if let Some((bytes, m)) = self.cache.read_manifest(bundle_id)? {
            return Ok((bytes, m, ManifestSource::Cache));
        }
        let Some(store) = self.store else {
            return Err(anyhow!(
                "{bundle_id}: no committed or cached manifest and offline"
            ));
        };
        let key = crate::manifest_key(bundle_id);
        let bytes = store
            .get_bytes(&key)
            .await?
            .ok_or_else(|| anyhow!("{bundle_id}: no manifest at {key} in {}", store.url))?;
        let m = BundleManifest::parse(&bytes).map_err(|e| anyhow!("{key}: {}", e.join("; ")))?;
        if m.bundle_id != bundle_id {
            return Err(anyhow!("{key}: bundle_id {} != {bundle_id}", m.bundle_id));
        }
        Ok((bytes, m, ManifestSource::Store))
    }

    /// Print notices and licence lines to `out`, refuse restricted files unless accepted,
    /// download and verify every selected blob, then link the bundle directory.
    pub async fn fetch(
        &self,
        bundle_id: &str,
        opts: &FetchOptions,
        out: &mut dyn Write,
    ) -> Result<FetchReport> {
        let (bytes, manifest, source) = self.manifest(bundle_id).await?;
        let selected: Vec<&BundleFile> = manifest.select(&opts.components)?;
        writeln!(
            out,
            "bundle {} ({} of {} files, {} bytes; manifest from {})",
            manifest.bundle_id,
            selected.len(),
            manifest.files.len(),
            selected.iter().map(|f| f.bytes).sum::<u64>(),
            match source {
                ManifestSource::Repo => "the checkout",
                ManifestSource::Cache => "the cache",
                ManifestSource::Store => "the store",
            }
        )?;
        for n in &manifest.notices {
            writeln!(out, "--- notice {} ---", n.path)?;
            writeln!(out, "{}", n.text.trim_end())?;
        }
        writeln!(out, "--- files ---")?;
        for f in &selected {
            writeln!(out, "{}", f.license_line())?;
        }
        let restricted: Vec<String> = selected
            .iter()
            .filter(|f| f.restricted())
            .map(|f| f.path.clone())
            .collect();
        if !restricted.is_empty() && !opts.accept_restricted {
            return Err(anyhow!(
                "{}: {} file(s) carry `redistribution: prohibited` ({}); nothing downloaded — \
                 pass --accept-restricted or select other components with --component",
                manifest.bundle_id,
                restricted.len(),
                restricted.join(", ")
            ));
        }
        let mut report = FetchReport {
            bundle_id: manifest.bundle_id.clone(),
            files_selected: selected.len(),
            restricted_accepted: restricted,
            ..Default::default()
        };
        let blobs = self.cache.blobs();
        let results: Vec<(String, u64, Result<bool>)> = stream::iter(selected.iter().map(|f| {
            let blobs = blobs.clone();
            async move {
                let r = blobs.ensure(self.store, &f.sha256, f.bytes).await;
                (f.path.clone(), f.bytes, r)
            }
        }))
        .buffer_unordered(PARALLEL_GETS)
        .collect()
        .await;
        for (path, bytes, r) in results {
            match r {
                Ok(true) => {
                    report.files_fetched += 1;
                    report.bytes += bytes;
                }
                Ok(false) => report.files_cached += 1,
                Err(e) => report.errors.push(format!("{path}: {e:#}")),
            }
        }
        if report.errors.is_empty() {
            if let Err(e) = self.cache.materialize(&manifest, &bytes, &selected) {
                report.errors.push(format!("{}: {e:#}", manifest.bundle_id));
            }
        }
        Ok(report)
    }
}
