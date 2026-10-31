//! The local model cache under `$JOPE_HOME/models`: `blobs/sha256/<sha256>` (shared
//! [`BlobDir`] layout), `bundles/<id>/<path>` hardlinks plus the embedded notices, and the
//! bundle manifests under `manifests/<id>.json` (outside the bundle directory, whose paths
//! belong to the bundle — Hyperion v7 ships its own `manifest.json`).

use crate::manifest::{BundleFile, BundleManifest};
use anyhow::{anyhow, Context, Result};
use jope_datasets::blobs::{link_or_copy, same_file, BlobDir};
use jope_datasets::hash::digest_content;
use jope_datasets::paths::jope_home;
use std::path::{Path, PathBuf};

/// Local model bundle and content-addressed blob cache.
#[derive(Clone, Debug)]
pub struct ModelsCache {
    /// Root directory containing the local bundle and content-addressed blob caches.
    pub root: PathBuf,
}

/// Result of checking the locally materialized files for one bundle.
#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    /// Bundle identifier that was checked.
    pub bundle_id: String,
    /// Files listed by the manifest and present in the bundle directory.
    pub files_checked: usize,
    /// Files listed by the manifest but not fetched (e.g. `--component` fetches).
    pub files_absent: usize,
    /// Verification problems found; an empty list means every present file checked successfully.
    pub errors: Vec<String>,
}

impl ModelsCache {
    /// `$JOPE_HOME/models`.
    pub fn from_env() -> ModelsCache {
        ModelsCache {
            root: jope_home().join("models"),
        }
    }

    /// Create a cache rooted at the supplied directory.
    pub fn at(root: impl Into<PathBuf>) -> ModelsCache {
        ModelsCache { root: root.into() }
    }

    /// Return the shared content-addressed blob directory.
    pub fn blobs(&self) -> BlobDir {
        BlobDir::new(self.root.join("blobs").join("sha256"))
    }

    /// Return the root directory for materialized bundles.
    pub fn bundles_dir(&self) -> PathBuf {
        self.root.join("bundles")
    }

    /// Return the materialized directory for `bundle_id`.
    pub fn bundle_dir(&self, bundle_id: &str) -> PathBuf {
        self.bundles_dir().join(bundle_id)
    }

    /// Return the directory containing cached bundle manifests.
    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests")
    }

    /// Return the cached manifest path for `bundle_id`.
    pub fn manifest_path(&self, bundle_id: &str) -> PathBuf {
        self.manifests_dir().join(format!("{bundle_id}.json"))
    }

    /// Return the materialized path for a bundle-relative file.
    pub fn file_path(&self, bundle_id: &str, path: &str) -> PathBuf {
        self.bundle_dir(bundle_id).join(path)
    }

    /// The cached manifest, or `None` when the bundle was never materialised.
    pub fn read_manifest(&self, bundle_id: &str) -> Result<Option<(Vec<u8>, BundleManifest)>> {
        let p = self.manifest_path(bundle_id);
        if !p.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let m = BundleManifest::parse(&bytes)
            .map_err(|e| anyhow!("{}: {}", p.display(), e.join("; ")))?;
        Ok(Some((bytes, m)))
    }

    /// Bundle ids with a cached manifest, sorted.
    pub fn list(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(self.manifests_dir())
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "json"))
                    .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        ids
    }

    /// Link `files` (whose blobs must already be verified in the cache) at
    /// `bundles/<id>/<path>`, write every embedded notice, then the manifest last.
    pub fn materialize(
        &self,
        manifest: &BundleManifest,
        manifest_bytes: &[u8],
        files: &[&BundleFile],
    ) -> Result<PathBuf> {
        let dir = self.bundle_dir(&manifest.bundle_id);
        std::fs::create_dir_all(&dir)?;
        let blobs = self.blobs();
        for f in files {
            let blob = blobs.path(&f.sha256);
            let size = std::fs::metadata(&blob).map(|m| m.len()).ok();
            if size != Some(f.bytes) {
                return Err(anyhow!("{}: blob {} is not in the cache", f.path, f.sha256));
            }
            link_or_copy(&blob, &dir.join(&f.path))?;
        }
        for n in &manifest.notices {
            let p = dir.join(&n.path);
            if std::fs::read(&p).ok().as_deref() != Some(n.text.as_bytes()) {
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&p, n.text.as_bytes())?;
            }
        }
        let target = self.manifest_path(&manifest.bundle_id);
        std::fs::create_dir_all(self.manifests_dir())?;
        let partial = self
            .manifests_dir()
            .join(format!("{}.json.partial", manifest.bundle_id));
        std::fs::write(&partial, manifest_bytes)?;
        std::fs::rename(&partial, &target)?;
        Ok(dir)
    }

    /// Verify every materialized file and backing blob against the cached manifest, and check
    /// notices.
    ///
    /// When `quick` is true, compare byte lengths only; otherwise compute SHA-256 digests.
    pub fn verify(&self, bundle_id: &str, quick: bool) -> Result<VerifyReport> {
        let (_, manifest) = self
            .read_manifest(bundle_id)?
            .ok_or_else(|| anyhow!("{bundle_id}: not in the cache {}", self.root.display()))?;
        let mut report = VerifyReport {
            bundle_id: bundle_id.to_string(),
            ..Default::default()
        };
        let dir = self.bundle_dir(bundle_id);
        let blobs = self.blobs();
        for f in &manifest.files {
            let p = dir.join(&f.path);
            if !p.is_file() {
                report.files_absent += 1;
                continue;
            }
            report.files_checked += 1;
            check_file(&p, &f.sha256, f.bytes, quick, &f.path, &mut report.errors);
            let blob = blobs.path(&f.sha256);
            if !same_file(&p, &blob) {
                if blob.is_file() {
                    check_file(
                        &blob,
                        &f.sha256,
                        f.bytes,
                        quick,
                        &format!("blob {}", f.sha256),
                        &mut report.errors,
                    );
                } else {
                    report
                        .errors
                        .push(format!("{}: blob {} is missing", f.path, f.sha256));
                }
            }
        }
        for n in &manifest.notices {
            match std::fs::read(dir.join(&n.path)) {
                Ok(b) if b == n.text.as_bytes() => {}
                Ok(_) => report
                    .errors
                    .push(format!("notice {}: text differs", n.path)),
                Err(e) => report.errors.push(format!("notice {}: {e}", n.path)),
            }
        }
        Ok(report)
    }
}

fn check_file(
    p: &Path,
    sha256: &str,
    bytes: u64,
    quick: bool,
    label: &str,
    errors: &mut Vec<String>,
) {
    if quick {
        match std::fs::metadata(p) {
            Ok(m) if m.len() == bytes => {}
            Ok(m) => errors.push(format!("{label}: {} bytes, manifest {bytes}", m.len())),
            Err(e) => errors.push(format!("{label}: {e}")),
        }
        return;
    }
    match digest_content(p) {
        Ok(d) if d.sha256 == sha256 && d.bytes == bytes => {}
        Ok(d) => errors.push(format!(
            "{label}: hashes to {} ({} bytes), manifest {sha256} ({bytes} bytes)",
            d.sha256, d.bytes
        )),
        Err(e) => errors.push(format!("{label}: {e}")),
    }
}
