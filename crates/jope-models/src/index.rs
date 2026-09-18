//! The committed `models/` directory: `manifest.json` (`jope.model-index/v1`), `bundles/<id>.json`
//! and `PINNED`.

use crate::manifest::BundleManifest;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const INDEX_SCHEMA_ID: &str = "jope.model-index/v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IndexRow {
    pub bundle_id: String,
    /// Repository-relative path of the bundle manifest (`models/bundles/<id>.json`).
    pub manifest: String,
    pub manifest_sha256: String,
    pub s3_key: String,
    pub files: usize,
    pub bytes: u64,
    #[serde(default)]
    pub licenses: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Index {
    pub schema: String,
    pub bucket: String,
    pub bundles: Vec<IndexRow>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A checkout whose `models/` directory holds the committed manifests.
#[derive(Clone, Debug)]
pub struct Repo {
    pub root: PathBuf,
}

impl Repo {
    pub fn at(root: impl Into<PathBuf>) -> Repo {
        Repo { root: root.into() }
    }

    /// `Some` when `<root>/models/manifest.json` exists.
    pub fn detect(root: &Path) -> Option<Repo> {
        root.join("models/manifest.json")
            .is_file()
            .then(|| Repo::at(root))
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn bundles_dir(&self) -> PathBuf {
        self.models_dir().join("bundles")
    }

    pub fn index_path(&self) -> PathBuf {
        self.models_dir().join("manifest.json")
    }

    pub fn pinned_path(&self) -> PathBuf {
        self.models_dir().join("PINNED")
    }

    pub fn manifest_path(&self, bundle_id: &str) -> PathBuf {
        self.bundles_dir().join(format!("{bundle_id}.json"))
    }

    pub fn read_index(&self) -> Result<Index> {
        let p = self.index_path();
        let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let index: Index =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", p.display()))?;
        if index.schema != INDEX_SCHEMA_ID {
            return Err(anyhow!(
                "{}: schema {} != {INDEX_SCHEMA_ID}",
                p.display(),
                index.schema
            ));
        }
        Ok(index)
    }

    /// Bundle ids in `models/PINNED` (blank lines and `#` comments skipped).
    pub fn read_pinned(&self) -> Result<Vec<String>> {
        let p = self.pinned_path();
        let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect())
    }

    /// The committed manifest bytes and document, or `None` when this checkout has none.
    pub fn read_manifest(&self, bundle_id: &str) -> Result<Option<(Vec<u8>, BundleManifest)>> {
        let p = self.manifest_path(bundle_id);
        if !p.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let m = BundleManifest::parse(&bytes)
            .map_err(|e| anyhow!("{}: {}", p.display(), e.join("; ")))?;
        if m.bundle_id != bundle_id {
            return Err(anyhow!(
                "{}: bundle_id {} != file name",
                p.display(),
                m.bundle_id
            ));
        }
        Ok(Some((bytes, m)))
    }

    /// Every committed bundle manifest, sorted by file name.
    pub fn manifest_files(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(self.bundles_dir())
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "json"))
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }
}
