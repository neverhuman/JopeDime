//! The local dataset cache: fetch (download + verify + hardlink), link (seed from a local tree)
//! and verify (re-hash blobs, check links).

use crate::blobs::{link_or_copy, BlobDir, PARALLEL_GETS};
use crate::catalog::{CatalogRow, Collection, Task};
use crate::paths::Cache;
use crate::store::Store;
use anyhow::{anyhow, Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub use crate::blobs::RESUME_THRESHOLD;

#[derive(Clone, Debug, Default)]
pub struct FetchReport {
    pub datasets: usize,
    pub files_fetched: usize,
    pub files_cached: usize,
    pub errors: Vec<String>,
    /// The exact catalog the rows were resolved from, for a pinned fetch.
    pub catalog: Option<CatalogPin>,
}

/// An exact published catalog version and the SHA-256 of its
/// `catalog.jsonl` bytes (W1 route R4). A pinned resolution never reads
/// `LATEST`, never substitutes cached rows or cached manifests, and serves a
/// cached catalog copy only when its bytes match the pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogPin {
    pub version: String,
    pub sha256: String,
}

impl CatalogPin {
    pub fn new(version: &str, sha256: &str) -> Result<Self> {
        let numbered = version
            .strip_prefix('v')
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
        if !numbered {
            return Err(anyhow!("catalog version {version:?} is not v<N>"));
        }
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(anyhow!("catalog digest {sha256:?} is not a sha256"));
        }
        Ok(Self {
            version: version.to_string(),
            sha256: sha256.to_string(),
        })
    }
}

fn parse_catalog(text: &str) -> Result<Vec<CatalogRow>> {
    let mut rows = Vec::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let value: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("catalog line {}", i + 1))?;
        rows.push(
            CatalogRow::validate_value(&value)
                .map_err(|e| anyhow!("catalog line {}: {}", i + 1, e.join("; ")))?,
        );
    }
    Ok(rows)
}

#[derive(Clone, Debug, Default)]
pub struct LinkReport {
    pub datasets: usize,
    pub collections: Vec<String>,
    pub skipped: usize,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    pub blobs: usize,
    pub datasets: usize,
    pub errors: Vec<String>,
}

pub struct Fetcher<'a> {
    pub cache: &'a Cache,
    pub store: Option<&'a Store>,
}

/// Copy a local file into the blob store under its sha256 (atomic via `.partial`).
pub fn store_blob(cache: &Cache, src: &Path, sha256: &str) -> Result<PathBuf> {
    BlobDir::new(cache.blobs_dir()).store_file(src, sha256, false)
}

/// Write `by-id/<hash>/{train.csv,test.csv,meta.json,manifest.json}` from cached blobs.
pub fn materialize(cache: &Cache, row: &CatalogRow) -> Result<PathBuf> {
    let dir = cache.by_id_dir(&row.dataset_hash);
    std::fs::create_dir_all(&dir)?;
    for (name, f) in row.files.entries() {
        let blob = cache.blob_path(&f.sha256);
        if !blob.is_file() {
            return Err(anyhow!("{}: blob {} is not in the cache", row.id, f.sha256));
        }
        link_or_copy(&blob, &dir.join(name))?;
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(row)?)?;
    Ok(dir)
}

pub fn read_manifest(path: &Path) -> Result<CatalogRow> {
    let text = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&text).with_context(|| format!("parse {}", path.display()))?;
    CatalogRow::validate_value(&value).map_err(|e| anyhow!("{}: {}", path.display(), e.join("; ")))
}

/// Every manifest under `by-id/`.
pub fn cached_rows(cache: &Cache) -> Vec<CatalogRow> {
    let mut rows = Vec::new();
    if let Ok(rd) = std::fs::read_dir(cache.registry_dir()) {
        let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        dirs.sort();
        for d in dirs {
            if let Ok(row) = read_manifest(&d.join("manifest.json")) {
                rows.push(row);
            }
        }
    }
    rows
}

impl Fetcher<'_> {
    fn offline(&self) -> bool {
        self.store.is_none()
    }

    async fn remote_text(&self, rel: &str) -> Result<Option<String>> {
        match self.store {
            Some(s) => s.get_string(rel).await,
            None => Ok(None),
        }
    }

    /// The manifest for one dataset. While the store is reachable its manifest
    /// is authoritative: a cached manifest that differs from it is stale and is
    /// not served (W1 route R4). Offline, or for a dataset the store does not
    /// publish (seeded by `link`), the cached manifest is what exists; pin a
    /// catalog ([`Fetcher::fetch_pinned`]) to exclude that path.
    pub async fn manifest(&self, dataset_hash: &str) -> Result<CatalogRow> {
        let cached = self.cache.by_id_dir(dataset_hash).join("manifest.json");
        let rel = format!("datasets/{dataset_hash}/manifest.json");
        let remote = self.remote_text(&rel).await?;
        let Some(text) = remote else {
            if cached.is_file() {
                return read_manifest(&cached);
            }
            return Err(if self.offline() {
                anyhow!("{dataset_hash}: not in the cache and offline")
            } else {
                anyhow!("{dataset_hash}: no manifest at {rel}")
            });
        };
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let row =
            CatalogRow::validate_value(&value).map_err(|e| anyhow!("{rel}: {}", e.join("; ")))?;
        if row.dataset_hash != dataset_hash {
            return Err(anyhow!("{rel} names dataset {}", row.dataset_hash));
        }
        Ok(row)
    }

    /// The rows of exactly the pinned catalog version (W1 route R4). `LATEST`
    /// is never read and cached rows are never substituted. A cached copy is
    /// served only when its bytes match the pin; otherwise the store's copy is
    /// verified against the pin and installed, or, offline, the request fails.
    pub async fn pinned_catalog(&self, pin: &CatalogPin) -> Result<Vec<CatalogRow>> {
        let local = self
            .cache
            .catalog_dir()
            .join(&pin.version)
            .join("catalog.jsonl");
        let bytes = match std::fs::read(&local) {
            Ok(bytes) if crate::hash::sha256_hex(&bytes) == pin.sha256 => bytes,
            cached => {
                let rel = format!("catalog/{}/catalog.jsonl", pin.version);
                let remote = match self.store {
                    Some(store) => store.get_bytes(&rel).await?,
                    None => None,
                };
                match remote {
                    Some(bytes) if crate::hash::sha256_hex(&bytes) == pin.sha256 => {
                        let dir = local.parent().context("catalog directory")?;
                        std::fs::create_dir_all(dir)?;
                        let partial = local.with_extension("jsonl.partial");
                        std::fs::write(&partial, &bytes)?;
                        std::fs::rename(&partial, &local)?;
                        bytes
                    }
                    Some(_) => {
                        return Err(anyhow!(
                            "catalog {} in the store differs from the pinned digest",
                            pin.version
                        ))
                    }
                    None => {
                        return Err(anyhow!(
                            "catalog {} with the pinned digest is {}{}",
                            pin.version,
                            if cached.is_ok() {
                                "not the cached copy (it differs)"
                            } else {
                                "not cached"
                            },
                            if self.offline() {
                                " and the fetch is offline"
                            } else {
                                " and absent from the store"
                            }
                        ))
                    }
                }
            }
        };
        let text = String::from_utf8(bytes).context("catalog is not UTF-8")?;
        parse_catalog(&text)
    }

    /// Resolve a dataset id or task against exactly the pinned catalog and
    /// fetch it. Cached manifests are never consulted; a cached manifest that
    /// differs from its pinned row is replaced by materialization.
    /// Collections are refused: they carry their own versions.
    pub async fn fetch_pinned(&self, selector: &str, pin: &CatalogPin) -> Result<FetchReport> {
        let catalog = self.pinned_catalog(pin).await?;
        let rows = if let Some(hash) = crate::dataset_dir_hash(selector) {
            vec![catalog
                .into_iter()
                .find(|r| r.dataset_hash == hash)
                .ok_or_else(|| anyhow!("{selector} is not in pinned catalog {}", pin.version))?]
        } else if let Some(task) = Task::parse(selector) {
            catalog.into_iter().filter(|r| r.task == task).collect()
        } else {
            return Err(anyhow!(
                "a pinned fetch resolves dataset ids and tasks; collection {selector} has its own versions"
            ));
        };
        let mut report = self.fetch_rows(rows).await?;
        report.catalog = Some(pin.clone());
        Ok(report)
    }

    /// The latest catalog (`catalog/LATEST` -> `catalog/v<N>/catalog.jsonl`), cache first.
    pub async fn catalog(&self) -> Result<Vec<CatalogRow>> {
        let dir = self.cache.catalog_dir();
        let latest_local = std::fs::read_to_string(dir.join("LATEST"))
            .ok()
            .map(|s| s.trim().to_string());
        let latest_remote = if self.offline() {
            None
        } else {
            self.remote_text("catalog/LATEST")
                .await?
                .map(|s| s.trim().to_string())
        };
        let version = latest_remote.or(latest_local);
        let Some(version) = version else {
            return Ok(cached_rows(self.cache));
        };
        let local = dir.join(&version).join("catalog.jsonl");
        let text = match std::fs::read_to_string(&local) {
            Ok(t) => t,
            Err(_) => {
                let t = self
                    .remote_text(&format!("catalog/{version}/catalog.jsonl"))
                    .await?
                    .ok_or_else(|| anyhow!("catalog {version} is missing"))?;
                std::fs::create_dir_all(local.parent().unwrap_or(&dir))?;
                std::fs::write(&local, &t)?;
                std::fs::write(dir.join("LATEST"), format!("{version}\n"))?;
                t
            }
        };
        parse_catalog(&text)
    }

    /// A collection document by name, cache first.
    pub async fn collection(&self, name: &str) -> Result<Collection> {
        let dir = self.cache.collections_dir().join(name);
        let latest_local = std::fs::read_to_string(dir.join("LATEST"))
            .ok()
            .map(|s| s.trim().to_string());
        let latest_remote = if self.offline() {
            None
        } else {
            self.remote_text(&format!("collections/{name}/LATEST"))
                .await?
                .map(|s| s.trim().to_string())
        };
        let version = latest_remote.or(latest_local).ok_or_else(|| {
            anyhow!(
                "collection {name}: unknown{}",
                if self.offline() { " (offline)" } else { "" }
            )
        })?;
        let local = dir.join(format!("{version}.json"));
        let text = match std::fs::read_to_string(&local) {
            Ok(t) => t,
            Err(_) => {
                let t = self
                    .remote_text(&format!("collections/{name}/{version}.json"))
                    .await?
                    .ok_or_else(|| anyhow!("collection {name}/{version} is missing"))?;
                std::fs::create_dir_all(&dir)?;
                std::fs::write(&local, &t)?;
                std::fs::write(dir.join("LATEST"), format!("{version}\n"))?;
                t
            }
        };
        let value: serde_json::Value = serde_json::from_str(&text)?;
        Collection::validate_value(&value)
            .map_err(|e| anyhow!("collection {name}: {}", e.join("; ")))
    }

    /// Resolve `<id>`, `<task>` or `<collection>` to rows.
    pub async fn resolve(&self, selector: &str) -> Result<Vec<CatalogRow>> {
        if let Some(hash) = crate::dataset_dir_hash(selector) {
            return Ok(vec![self.manifest(hash).await?]);
        }
        if let Some(task) = Task::parse(selector) {
            return Ok(self
                .catalog()
                .await?
                .into_iter()
                .filter(|r| r.task == task)
                .collect());
        }
        let coll = self.collection(selector).await?;
        // One manifest GET per member, up to PARALLEL_GETS in flight instead of one round trip
        // at a time. `buffered` yields in member order, so the rows keep the collection's order
        // and the first failing member (in order) is the error reported, as before.
        use futures_util::stream::{self, StreamExt, TryStreamExt};
        stream::iter(&coll.members)
            .map(|member| async move {
                let hash = crate::dataset_dir_hash(member)
                    .ok_or_else(|| anyhow!("collection member {member} is not a dataset id"))?;
                self.manifest(hash).await
            })
            .buffered(PARALLEL_GETS)
            .try_collect()
            .await
    }

    async fn ensure_blob(&self, sha256: &str, bytes: u64) -> Result<bool> {
        BlobDir::new(self.cache.blobs_dir())
            .ensure(self.store, sha256, bytes)
            .await
    }

    /// Fetch every file of every resolved row into the cache and materialize `by-id/`.
    pub async fn fetch(&self, selector: &str) -> Result<FetchReport> {
        let rows = self.resolve(selector).await?;
        self.fetch_rows(rows).await
    }

    async fn fetch_rows(&self, rows: Vec<CatalogRow>) -> Result<FetchReport> {
        use futures_util::stream::{self, StreamExt};
        // A cached manifest that is not the resolved row is stale: its consumer
        // copies are withdrawn before anything is served, and materialization
        // below replaces them from the resolved row.
        for row in &rows {
            let dir = self.cache.by_id_dir(&row.dataset_hash);
            let cached = dir.join("manifest.json");
            if cached.is_file() && read_manifest(&cached).ok().as_ref() != Some(row) {
                for name in ["manifest.json", "train.csv", "test.csv", "meta.json"] {
                    match std::fs::remove_file(dir.join(name)) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e).context("withdraw stale consumer copy"),
                    }
                }
            }
        }
        let mut report = FetchReport {
            datasets: rows.len(),
            ..Default::default()
        };
        let mut wanted: BTreeSet<(String, u64)> = BTreeSet::new();
        for row in &rows {
            for (_, f) in row.files.entries() {
                wanted.insert((f.sha256.clone(), f.bytes));
            }
        }
        let results: Vec<(String, Result<bool>)> =
            stream::iter(wanted.into_iter().map(|(sha, bytes)| async move {
                let r = self.ensure_blob(&sha, bytes).await;
                (sha, r)
            }))
            .buffer_unordered(PARALLEL_GETS)
            .collect()
            .await;
        let mut failed = BTreeSet::new();
        for (sha, r) in results {
            match r {
                Ok(true) => report.files_fetched += 1,
                Ok(false) => report.files_cached += 1,
                Err(e) => {
                    report.errors.push(format!("{sha}: {e:#}"));
                    failed.insert(sha);
                }
            }
        }
        for row in &rows {
            let unverified: Vec<&str> = row
                .files
                .entries()
                .into_iter()
                .filter(|(_, f)| failed.contains(&f.sha256))
                .map(|(name, _)| name)
                .collect();
            if !unverified.is_empty() {
                // A consumer copy of an object that could not be verified is never left
                // servable (it may be a hard link to the stale blob just removed).
                let dir = self.cache.by_id_dir(&row.dataset_hash);
                for name in &unverified {
                    match std::fs::remove_file(dir.join(name)) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => report
                            .errors
                            .push(format!("{}: remove unverified {name}: {e}", row.id)),
                    }
                }
                report.errors.push(format!(
                    "{}: not materialized; required objects failed verification: {unverified:?}",
                    row.id
                ));
                continue;
            }
            if let Err(e) = materialize(self.cache, row) {
                report.errors.push(format!("{}: {e:#}", row.id));
            }
        }
        Ok(report)
    }
}

mod link;
mod verify;

pub use link::link_from;
pub use verify::verify_cache;
