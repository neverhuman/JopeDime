//! `publish`: snapshot a results directory into an object store as
//! `scores/sotarena/<track>/<version>/…` (plan: "published benchmark results as submitted,
//! byte-identical, with sha256; immutable per version").
//!
//! The store is any `object_store` URL (`file:///…` locally and in tests; `s3://bucket` with the
//! `s3` feature). Every file under the results directory is copied with its relative path; a
//! `MANIFEST.json` (`jope.arena-publish/v1`) lists path, bytes and SHA-256 of each; and a version
//! prefix that already holds a `MANIFEST.json` is refused — corrections are a new version.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};

use crate::io::sha256_file;

/// One published file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublishedFile {
    /// Path relative to the version prefix.
    pub path: String,
    /// Byte count.
    pub bytes: u64,
    /// SHA-256.
    pub sha256: String,
}

/// `MANIFEST.json`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublishManifest {
    /// `jope.arena-publish/v1`.
    pub schema: String,
    /// Track id.
    pub track_id: String,
    /// Version label.
    pub version: String,
    /// Object-store prefix the files were written under.
    pub prefix: String,
    /// The source directory as given.
    pub source_dir: String,
    /// Every file, sorted by path.
    pub files: Vec<PublishedFile>,
}

/// The store prefix for a track version.
pub fn version_prefix(track_id: &str, version: &str) -> String {
    format!("scores/sotarena/{track_id}/{version}")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

/// Validate the label a version is published under.
pub fn validate_version(version: &str) -> Result<()> {
    ensure!(
        !version.is_empty()
            && version
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.'),
        "version label {version:?} must be [A-Za-z0-9._-]+"
    );
    ensure!(version != "." && version != "..", "invalid version label");
    Ok(())
}

/// Parse a store URL. `file://` needs nothing; `s3://` needs the `s3` feature.
pub fn open_store(url: &str) -> Result<(Box<dyn ObjectStore>, StorePath)> {
    let parsed = object_store::parse_url(&url.parse().with_context(|| format!("store URL {url}"))?)
        .with_context(|| {
            format!(
                "unsupported store URL {url} (file:// always works; s3:// needs the `s3` feature)"
            )
        })?;
    Ok(parsed)
}

/// Copy `results_dir` under `<store>/<base>/scores/sotarena/<track>/<version>/`.
pub async fn publish(
    store: &dyn ObjectStore,
    base: &StorePath,
    results_dir: &Path,
    track_id: &str,
    version: &str,
) -> Result<PublishManifest> {
    validate_version(version)?;
    ensure!(
        results_dir.is_dir(),
        "{} is not a directory",
        results_dir.display()
    );
    ensure!(
        results_dir.join("scores.csv").is_file() || results_dir.join("scores.csv.gz").is_file(),
        "{} has no scores.csv; refusing to publish a directory that is not a results snapshot",
        results_dir.display()
    );
    let prefix = version_prefix(track_id, version);
    let prefix_path = base
        .parts()
        .chain(StorePath::from(prefix.as_str()).parts())
        .collect::<StorePath>();
    let manifest_key = prefix_path.clone().join("MANIFEST.json");
    if store.head(&manifest_key).await.is_ok() {
        bail!(
            "{prefix} is already published (MANIFEST.json exists); versions are immutable — publish a new version"
        );
    }
    let mut files = Vec::new();
    walk(results_dir, &mut files)?;
    let mut published = Vec::with_capacity(files.len());
    for path in files {
        let relative = path
            .strip_prefix(results_dir)
            .expect("walked under results_dir")
            .to_string_lossy()
            .into_owned();
        let (sha256, bytes) = sha256_file(&path)?;
        let key = relative
            .split('/')
            .fold(prefix_path.clone(), |acc, part| acc.join(part));
        let payload = PutPayload::from(std::fs::read(&path)?);
        store
            .put_opts(
                &key,
                payload,
                PutOptions {
                    mode: PutMode::Create,
                    ..PutOptions::default()
                },
            )
            .await
            .with_context(|| format!("put {key}"))?;
        published.push(PublishedFile {
            path: relative,
            bytes,
            sha256,
        });
    }
    published.sort_by(|a, b| a.path.cmp(&b.path));
    let manifest = PublishManifest {
        schema: "jope.arena-publish/v1".to_owned(),
        track_id: track_id.to_owned(),
        version: version.to_owned(),
        prefix: prefix_path.to_string(),
        source_dir: results_dir.display().to_string(),
        files: published,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    store
        .put_opts(
            &manifest_key,
            PutPayload::from(bytes),
            PutOptions {
                mode: PutMode::Create,
                ..PutOptions::default()
            },
        )
        .await
        .with_context(|| format!("put {manifest_key}"))?;
    Ok(manifest)
}

/// Read back a published `MANIFEST.json` and re-hash every object it lists.
pub async fn verify_published(
    store: &dyn ObjectStore,
    base: &StorePath,
    track_id: &str,
    version: &str,
) -> Result<PublishManifest> {
    let prefix_path = base
        .parts()
        .chain(StorePath::from(version_prefix(track_id, version).as_str()).parts())
        .collect::<StorePath>();
    let manifest_key = prefix_path.clone().join("MANIFEST.json");
    let bytes = store
        .get(&manifest_key)
        .await
        .with_context(|| format!("get {manifest_key}"))?
        .bytes()
        .await?;
    let manifest: PublishManifest = serde_json::from_slice(&bytes)?;
    for file in &manifest.files {
        let key = file
            .path
            .split('/')
            .fold(prefix_path.clone(), |acc, part| acc.join(part));
        let bytes = store
            .get(&key)
            .await
            .with_context(|| format!("get {key}"))?
            .bytes()
            .await?;
        ensure!(
            bytes.len() as u64 == file.bytes,
            "{key}: {} bytes, manifest says {}",
            bytes.len(),
            file.bytes
        );
        let sha = crate::io::sha256_hex(&bytes);
        ensure!(
            sha == file.sha256,
            "{key}: sha256 {sha}, manifest says {}",
            file.sha256
        );
    }
    Ok(manifest)
}

/// Blocking wrappers for the CLI.
pub fn publish_blocking(
    store_url: &str,
    results_dir: &Path,
    track_id: &str,
    version: &str,
) -> Result<PublishManifest> {
    let (store, base) = open_store(store_url)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(publish(
        store.as_ref(),
        &base,
        results_dir,
        track_id,
        version,
    ))
}

/// Blocking [`verify_published`].
pub fn verify_published_blocking(
    store_url: &str,
    track_id: &str,
    version: &str,
) -> Result<PublishManifest> {
    let (store, base) = open_store(store_url)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(verify_published(store.as_ref(), &base, track_id, version))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_store_publish_is_byte_exact_and_immutable() {
        let src = tempfile::tempdir().unwrap();
        let results = src.path().join("results/tr");
        std::fs::create_dir_all(results.join("receipts")).unwrap();
        std::fs::write(results.join("scores.csv"), b"h\n1\n").unwrap();
        std::fs::write(results.join("receipts/a.json"), b"{}").unwrap();
        let store_dir = tempfile::tempdir().unwrap();
        let url = format!("file://{}", store_dir.path().display());
        let manifest = publish_blocking(&url, &results, "tr", "2026-09-19").unwrap();
        assert!(
            manifest.prefix.ends_with("scores/sotarena/tr/2026-09-19"),
            "{}",
            manifest.prefix
        );
        assert_eq!(
            manifest
                .files
                .iter()
                .map(|f| f.path.as_str())
                .collect::<Vec<_>>(),
            ["receipts/a.json", "scores.csv"]
        );
        let on_disk = store_dir
            .path()
            .join("scores/sotarena/tr/2026-09-19/scores.csv");
        assert_eq!(std::fs::read(on_disk).unwrap(), b"h\n1\n");
        let written: PublishManifest = serde_json::from_slice(
            &std::fs::read(
                store_dir
                    .path()
                    .join("scores/sotarena/tr/2026-09-19/MANIFEST.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(written, manifest);
        assert_eq!(
            verify_published_blocking(&url, "tr", "2026-09-19").unwrap(),
            manifest
        );
        let again = publish_blocking(&url, &results, "tr", "2026-09-19").unwrap_err();
        assert!(again.to_string().contains("already published"), "{again}");
        std::fs::write(
            store_dir
                .path()
                .join("scores/sotarena/tr/2026-09-19/scores.csv"),
            b"h\n2\n",
        )
        .unwrap();
        assert!(verify_published_blocking(&url, "tr", "2026-09-19").is_err());
        assert!(validate_version("v 1").is_err());
        assert!(validate_version("..").is_err());
        let not_results = tempfile::tempdir().unwrap();
        assert!(publish_blocking(&url, not_results.path(), "tr", "v2")
            .unwrap_err()
            .to_string()
            .contains("no scores.csv"));
    }
}
