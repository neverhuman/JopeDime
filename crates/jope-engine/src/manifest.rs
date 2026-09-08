use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub const SEED: u64 = 20260907;
pub const ROW_LIMIT: usize = 50_000;
pub fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut state = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        state.update(&buffer[..n]);
    }
    Ok(hex::encode(state.finalize()))
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(std::io::BufReader::with_capacity(
        64 * 1024,
        File::open(path)?,
    ))
    .with_context(|| format!("parse {}", path.display()))
}
/// A sibling temporary file plus fsync/rename prevents readers from seeing partial state.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("output requires a parent directory")?;
    fs::create_dir_all(parent)?;
    let tmp = path.with_extension(format!("partial.{}", std::process::id()));
    let mut file = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    atomic_write(path, &serde_json::to_vec_pretty(value)?)
}
pub fn relative(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        bail!("unsafe relative path: {}", path.display());
    }
    Ok(())
}
pub fn resolve(root: &Path, path: &Path) -> Result<PathBuf> {
    relative(path)?;
    // Reject symlinks in each existing component, including the leaf.
    let mut current = root.canonicalize()?;
    for part in path.components() {
        current.push(part);
        if fs::symlink_metadata(&current).is_ok_and(|m| m.file_type().is_symlink()) {
            bail!("symlink in workspace path: {}", current.display());
        }
    }
    Ok(current)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Admission {
    Approved,
    Excluded,
    Unresolved,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub path: PathBuf,
    pub sha256: String,
    pub source_url: String,
    pub finding: String,
}
impl Evidence {
    pub fn verify(&self) -> Result<()> {
        if self.finding.trim().is_empty()
            || self.source_url.trim().is_empty()
            || file_hash(&self.path)? != self.sha256
        {
            bail!("invalid provenance evidence: {}", self.path.display());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub key: String,
    pub admission: Admission,
    pub family: String,
    pub reason: String,
    pub evidence: Vec<Evidence>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub dataset: Option<String>,
    pub role: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferManifest {
    pub schema: String,
    pub files: Vec<FileEntry>,
}
impl TransferManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema != "parallel.transfer.v1" {
            bail!("unsupported transfer schema");
        }
        let mut paths = std::collections::BTreeSet::new();
        for f in &self.files {
            relative(&f.destination)?;
            if !paths.insert(&f.destination) {
                bail!("duplicate transfer destination");
            }
            if f.sha256.len() != 64 || hex::decode(&f.sha256).is_err() {
                bail!("invalid SHA256");
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    pub rows: usize,
    pub columns: usize,
    pub target_index: usize,
    pub header: bool,
    pub missing_features: usize,
    pub invalid_targets: usize,
    pub duplicate_rows: usize,
    pub conflicting_features: usize,
    pub target_values: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub task: String,
    pub source_dir: PathBuf,
    pub source_key: String,
    pub family: String,
    pub parents: Vec<String>,
    pub admission: Admission,
    pub reason: String,
    pub evidence: Vec<Evidence>,
    pub metadata_sha256: String,
    pub train: Option<TableStats>,
    pub test: Option<TableStats>,
    pub split_feature_overlap: usize,
    pub issues: Vec<String>,
    pub meta_partition: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub schema: String,
    pub seed: u64,
    pub roots: Vec<PathBuf>,
    pub datasets: Vec<Dataset>,
    pub artifact_counts: std::collections::BTreeMap<String, u64>,
    pub missing_roots: Vec<PathBuf>,
    pub complete: bool,
}
