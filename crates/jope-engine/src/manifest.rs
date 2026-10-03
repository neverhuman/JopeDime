use anyhow::{bail, ensure, Context, Result};
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
    if let Some(result) = crate::native_input_cache::hash_if_bound(path) {
        return result;
    }
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
/// Read `path` once, require the SHA-256 of exactly those bytes to be
/// `expected`, and return them: the verified object is the consumed object
/// (W1 route R8). Callers parse the returned bytes; they never reopen the path.
pub fn read_verified(path: &Path, expected: &str, what: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("read {what} {}", path.display()))?;
    ensure!(
        hash(&bytes) == expected,
        "{what} changed: its bytes differ from the bound identity"
    );
    after_verification(path);
    Ok(bytes)
}
/// Read `path` once and return its bytes with the SHA-256 of exactly those
/// bytes, for a receiver that records the digest of what it consumes rather
/// than checking a bound one (W1 route R8). Callers parse the returned bytes;
/// they never reopen the path.
pub fn read_hashed(path: &Path, what: &str) -> Result<(Vec<u8>, String)> {
    let bytes = fs::read(path).with_context(|| format!("read {what} {}", path.display()))?;
    let digest = hash(&bytes);
    after_verification(path);
    Ok((bytes, digest))
}
/// [`read_verified`] against a recorded binding (a JSON value, so a missing
/// or malformed binding is refused the same way) and with the receiver's own
/// refusal wording, for receivers whose error text is part of their contract.
pub fn read_bound(path: &Path, expected: &serde_json::Value, refusal: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        expected.as_str() == Some(hash(&bytes).as_str()),
        "{refusal}"
    );
    after_verification(path);
    Ok(bytes)
}
/// Test-only instrumentation point between a digest check and consumption of
/// the checked object (W1 consumer-map route R8). Compiles to nothing outside
/// this crate's own unit-test build; it grants and changes nothing.
#[inline]
pub(crate) fn after_verification(_path: &Path) {
    #[cfg(test)]
    verification_hook::fire(_path);
}
#[cfg(test)]
pub(crate) mod verification_hook {
    use std::{cell::RefCell, path::Path};
    type Hook = Box<dyn FnMut(&Path)>;
    thread_local!(static HOOK: RefCell<Option<Hook>> = RefCell::new(None));
    /// Install `hook` for this thread until the returned guard drops.
    pub(crate) fn install(hook: impl FnMut(&Path) + 'static) -> Guard {
        HOOK.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
        Guard
    }
    pub(crate) struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|h| *h.borrow_mut() = None);
        }
    }
    pub(super) fn fire(path: &Path) {
        let hook = HOOK.with(|h| h.borrow_mut().take());
        if let Some(mut hook) = hook {
            hook(path);
            HOOK.with(|h| {
                let mut slot = h.borrow_mut();
                if slot.is_none() {
                    *slot = Some(hook);
                }
            });
        }
    }
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(std::io::BufReader::with_capacity(
        64 * 1024,
        File::open(path)?,
    ))
    .with_context(|| format!("parse {}", path.display()))
}
/// A sibling temporary file plus fsync/rename prevents readers from seeing partial state.
/// The temporary name is unique per call (process, time and a process-wide
/// counter), so a temporary left by a killed writer, even one whose process id
/// is later reused, and a concurrent writer of the same file never collide.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .context("output requires a parent directory")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .context("output requires a file name")?
        .to_string_lossy();
    let tmp = parent.join(format!(
        ".{name}.tmp-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
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
