use super::prelude::*;

pub const TASKS: [&str; 4] = [
    crate::task::Task::ALL[0].as_str(),
    crate::task::Task::ALL[1].as_str(),
    crate::task::Task::ALL[2].as_str(),
    crate::task::Task::ALL[3].as_str(),
];
pub const REQUIRED_ENCODINGS: [&str; 3] =
    ["hyperion3train128dim.vs", "train3.vs", "train_target.vs"];
pub const OPTIONAL_ENCODINGS: [&str; 1] = ["hyperion3train.vs"];
pub const ENCODER_PROTOCOL: &str = "hyperion-train-context-v3";
pub const CONTEXT_PROTOCOL: &str = "jope-dataset1536-target512-f32le-v1";
pub const TRANCHE_MAX: usize = 2_048;

/// Sealed zero-shot carve-out per task (D7). The legacy tasks keep their
/// frozen 256; the new tasks get carve-outs sized to their far smaller
/// corpora (multiclass 252 real datasets, timeseries 42), topped up toward
/// 256 only from NEWLY admitted datasets so the zero-shot guarantee holds.
pub fn sealed_target(task: &str) -> usize {
    match task {
        "regression" | "binary" => 256,
        "multiclass" => 64,
        "timeseries" => 8,
        _ => 0,
    }
}

/// On-disk corpus directory for a task string (D1): the multiclass corpus
/// keeps its legacy `multi_label` directory name. Unknown task strings pass
/// through unchanged (callers validate against [`TASKS`] first).
pub fn corpus_dir_name(task: &str) -> &str {
    match crate::task::TaskKind::parse(task) {
        Some(kind) => kind.corpus_dir_name(),
        None => task,
    }
}
pub const DATASET_DIM: usize = 1_536;
pub const TARGET_DIM: usize = 512;
pub const CONTEXT_FLOATS: usize = DATASET_DIM + TARGET_DIM;
pub const CONTEXT_STRIDE: usize = CONTEXT_FLOATS * 4;

pub(super) fn load_v1_quarantine(paths: &ExpansionPaths) -> Result<Option<V1EncodingQuarantineV1>> {
    let root = paths.cache_root.join("incident-quarantine");
    let index = root.join("index.v1.json");
    if !root.exists() {
        return Ok(None);
    }
    ensure!(
        index.is_file(),
        "V1 incident quarantine exists without its frozen index"
    );
    let quarantine: V1EncodingQuarantineV1 = serde_json::from_value(read_json(&index)?)?;
    quarantine.validate()?;
    Ok(Some(quarantine))
}

pub(super) fn v1_identity_denied(
    quarantine: Option<&V1EncodingQuarantineV1>,
    identity: &str,
) -> bool {
    quarantine.is_some_and(|index| index.denies_identity(identity))
}

#[derive(Clone, Debug)]
pub struct ExpansionPaths {
    pub source_root: PathBuf,
    pub mirror_root: PathBuf,
    pub cache_root: PathBuf,
    pub catalog_root: PathBuf,
    pub receipt_root: PathBuf,
    pub sealed_path: PathBuf,
    pub standardization: PathBuf,
}

impl Default for ExpansionPaths {
    fn default() -> Self {
        let flywheel = paths::flywheel_root();
        Self {
            source_root: paths::quant_root(),
            mirror_root: paths::quant_mirror_root(),
            cache_root: paths::flywheel_encodings_root(),
            catalog_root: flywheel.join("catalog/v2"),
            receipt_root: flywheel.join("receipts/mirror-conflicts"),
            sealed_path: flywheel.join("catalog/private/sealed-expanded-v1.json"),
            standardization: paths::flywheel_encodings_root().join(
                "foundation-v1/standardization/sha256/daac224cf7c6d13a1bf74fc3bb7defd36f33bc3f30db35d3e881c16ede976d1f.json",
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InventoryEntry {
    pub task: String,
    pub dataset_id: String,
    pub dataset_hash: String,
    pub source: PathBuf,
    pub source_family: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub meta_sha256: String,
    pub mirror: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MirrorConflict {
    pub task: String,
    pub dataset_id: String,
    pub source: PathBuf,
    pub mirror: PathBuf,
    pub disagreements: BTreeMap<String, Value>,
    pub conflict_sha256: String,
    pub receipt: PathBuf,
}

#[derive(Clone, Debug)]
pub struct AuditInventory {
    pub inspected: usize,
    pub entries: Vec<InventoryEntry>,
    pub conflicts: Vec<MirrorConflict>,
}

impl InventoryEntry {
    pub fn dataset_identity(&self) -> DatasetIdentity {
        DatasetIdentity {
            task: self.task.clone(),
            dataset_hash: self.dataset_hash.clone(),
            train_sha256: self.train_sha256.clone(),
            test_sha256: self.test_sha256.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheManifest {
    pub schema: String,
    pub immutable: bool,
    pub task: String,
    pub dataset_id: String,
    pub dataset_hash: String,
    pub source: PathBuf,
    pub source_family: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub encoder_protocol: String,
    pub encoder_protocol_sha256: String,
    pub train_only_encoding: bool,
    /// Which numerical backend actually produced these bytes.
    ///
    /// `encoder_protocol_sha256` binds only the protocol string and the
    /// train-only flag -- NOT the device class, model weights, or library
    /// versions. Measured on identical input, a CUDA encode (3s) and a CPU
    /// encode (344s) of the same dataset produce DIFFERENT bytes, and
    /// `cache_path` is keyed only by task+dataset_id, so the two cannot
    /// coexist and nothing records which one a given cache holds. The
    /// encoder also falls back to CPU on GPU-memory pressure, so a retry can
    /// change corpus identity while every identity label stays the same.
    ///
    /// This field does not fix that -- it makes it AUDITABLE, which must come
    /// first: `None` means "produced before attestation existed, backend
    /// unknown", and every new encode records what it actually ran on. It is
    /// `serde(default)` because ~25,000 existing manifests predate it and
    /// `validate_cache` must keep accepting them unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_backend: Option<String>,
    pub files: BTreeMap<String, String>,
    pub created_at_ms: u128,
}

impl CacheManifest {
    pub fn cache_identity(&self) -> CacheIdentity {
        CacheIdentity {
            dataset: DatasetIdentity {
                task: self.task.clone(),
                dataset_hash: self.dataset_hash.clone(),
                train_sha256: self.train_sha256.clone(),
                test_sha256: self.test_sha256.clone(),
            },
            dataset_id: self.dataset_id.clone(),
            source_family: self.source_family.clone(),
            encoder_protocol_sha256: self.encoder_protocol_sha256.clone(),
            encoding_hashes: self
                .files
                .iter()
                .filter(|(name, _)| {
                    REQUIRED_ENCODINGS.contains(&name.as_str())
                        || OPTIONAL_ENCODINGS.contains(&name.as_str())
                })
                .map(|(name, digest)| (name.clone(), digest.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct SealedCohort {
    schema: String,
    immutable: bool,
    classification: String,
    training_ingestion_eligible: bool,
    counts: BTreeMap<String, usize>,
    entries: Vec<InventoryEntry>,
    created_at_ms: u128,
    cohort_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Standardization {
    dim: usize,
    mean: Vec<f32>,
    scale: Vec<f32>,
    foundation: FoundationSpec,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct FoundationSpec {
    cache_dir: PathBuf,
    projection_file: PathBuf,
    raw_dim: usize,
    proj_dim: usize,
}

pub(super) struct FileLock(File);

impl FileLock {
    fn exclusive(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        ensure!(file.lock().is_ok(), "lock failed: {}", path.display());
        Ok(Self(file))
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}
