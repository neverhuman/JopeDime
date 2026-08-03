//! Rust-only expanded dataset cache and append-only catalog implementation.
//!
//! The authoritative candidate tree is always compared with its local mirror.
//! Cache writes are locked, staged, content-verified, and atomically renamed.
//! Missing train encodings fail closed unless an explicit native Rust encoder is
//! supplied.  No command in this module invokes Python.

use crate::selfplay_data::{
    canonical_json_sha256, source_family, CacheIdentity, CatalogEntryIdentity, CatalogIdentity,
    DatasetIdentity,
};
use crate::selfplay_runtime::{atomic_json, now_ms, read_json, sha256_bytes, sha256_file};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use zip::ZipArchive;

pub const TASKS: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];
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
        let flywheel = PathBuf::from("/home/ubuntu/jope-flywheel/test_exact_gen0");
        Self {
            source_root: PathBuf::from("/home/ubuntu/remote_super/quant"),
            mirror_root: PathBuf::from("/home/ubuntu/quant"),
            cache_root: PathBuf::from("/home/ubuntu/jope-flywheel/encodings"),
            catalog_root: flywheel.join("catalog/v2"),
            receipt_root: flywheel.join("receipts/mirror-conflicts"),
            sealed_path: flywheel.join("catalog/private/sealed-expanded-v1.json"),
            standardization: PathBuf::from(
                "/home/ubuntu/jope/data/processed/active_v3_dual_20260706/standardization.json",
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
struct SealedCohort {
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
struct Standardization {
    dim: usize,
    mean: Vec<f32>,
    scale: Vec<f32>,
    foundation: FoundationSpec,
}

#[derive(Clone, Debug, Deserialize)]
struct FoundationSpec {
    cache_dir: PathBuf,
    projection_file: PathBuf,
    raw_dim: usize,
    proj_dim: usize,
}

struct FileLock(File);

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
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        ensure!(result == 0, "lock failed: {}", path.display());
        Ok(Self(file))
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn valid_dataset_id(value: &str) -> bool {
    let hash = value.strip_prefix("synth_").unwrap_or(value);
    hash.len() == 16
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_meta(path: &Path) -> Result<Value> {
    read_json(path).with_context(|| format!("read dataset metadata {}", path.display()))
}

fn mirror_dataset(paths: &ExpansionPaths, task: &str, dataset_id: &str) -> Option<PathBuf> {
    [
        paths
            .mirror_root
            .join(corpus_dir_name(task))
            .join(dataset_id),
        paths.mirror_root.join(dataset_id),
    ]
    .into_iter()
    .find(|candidate| candidate.is_dir())
}

fn write_conflict_receipt(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
    mirror: &Path,
    disagreements: &BTreeMap<String, Value>,
) -> Result<(String, PathBuf)> {
    let stable = json!({
        "schema": "JopeExpandedMirrorConflict.v1",
        "task": task,
        "dataset_id": dataset_id,
        "source": source,
        "mirror": mirror,
        "disagreements": disagreements,
    });
    let digest = canonical_json_sha256(&stable)?;
    let mut payload = stable;
    payload["conflict_sha256"] = Value::String(digest.clone());
    payload["first_observed_at_ms"] = json!(now_ms());
    let destination = paths.receipt_root.join(format!("{digest}.json"));
    if destination.is_file() {
        let existing = read_json(&destination)?;
        ensure!(
            existing["conflict_sha256"].as_str() == Some(&digest)
                && existing["task"] == payload["task"]
                && existing["dataset_id"] == payload["dataset_id"]
                && existing["source"] == payload["source"]
                && existing["mirror"] == payload["mirror"]
                && existing["disagreements"] == payload["disagreements"],
            "mirror conflict receipt identity drift: {}",
            destination.display()
        );
    } else {
        atomic_json(&destination, &payload)?;
    }
    Ok((digest, destination))
}

enum MirrorEvidence {
    Absent,
    Matched(PathBuf),
    Conflict(MirrorConflict),
}

fn compare_mirror_evidence(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
) -> Result<MirrorEvidence> {
    let Some(mirror) = mirror_dataset(paths, task, dataset_id) else {
        return Ok(MirrorEvidence::Absent);
    };
    let mut disagreements = BTreeMap::new();
    for name in ["train.csv", "test.csv"] {
        let authoritative = source.join(name);
        let mirrored = mirror.join(name);
        let source_hash = sha256_file(&authoritative)?;
        let mirror_hash = mirrored
            .is_file()
            .then(|| sha256_file(&mirrored))
            .transpose()?;
        if mirror_hash.as_deref() != Some(source_hash.as_str()) {
            disagreements.insert(
                name.to_owned(),
                json!({"source": source_hash, "mirror": mirror_hash}),
            );
        }
    }
    if disagreements.is_empty() {
        return Ok(MirrorEvidence::Matched(mirror));
    }
    let (conflict_sha256, receipt) =
        write_conflict_receipt(paths, task, dataset_id, source, &mirror, &disagreements)?;
    Ok(MirrorEvidence::Conflict(MirrorConflict {
        task: task.to_owned(),
        dataset_id: dataset_id.to_owned(),
        source: source.to_owned(),
        mirror,
        disagreements,
        conflict_sha256,
        receipt,
    }))
}

/// Compare an authoritative candidate against the optional local mirror.
///
/// Any missing or disagreeing mirrored train/test file produces a durable
/// receipt and returns an error. An absent mirror is allowed and recorded.
pub fn compare_mirror(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
) -> Result<Option<PathBuf>> {
    match compare_mirror_evidence(paths, task, dataset_id, source)? {
        MirrorEvidence::Absent => Ok(None),
        MirrorEvidence::Matched(mirror) => Ok(Some(mirror)),
        MirrorEvidence::Conflict(conflict) => {
            bail!(
                "{task}:{dataset_id}: mirror content conflict; receipt={}",
                conflict.receipt.display()
            )
        }
    }
}

fn candidate_directories(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<(String, PathBuf)>> {
    if let Some(task) = task_filter {
        ensure!(TASKS.contains(&task), "unsupported task: {task}");
    }
    let mut candidates = Vec::new();
    for task in TASKS {
        if task_filter.is_some_and(|filter| filter != task) {
            continue;
        }
        let task_root = paths.source_root.join(corpus_dir_name(task));
        // The legacy corpus roots are mandatory; a new task's root may not
        // exist on every host yet and simply contributes no candidates.
        if !task_root.is_dir() && !matches!(task, "regression" | "binary") {
            continue;
        }
        let mut directories = fs::read_dir(&task_root)
            .with_context(|| format!("read authoritative root {}", task_root.display()))?
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .collect::<Vec<_>>();
        directories.sort_by_key(|entry| entry.file_name());
        for directory in directories {
            let dataset_id = directory.file_name().to_string_lossy().to_string();
            if !valid_dataset_id(&dataset_id) {
                continue;
            }
            let source = directory.path();
            if !["train.csv", "test.csv", "meta.json"]
                .iter()
                .all(|name| source.join(name).is_file())
            {
                continue;
            }
            candidates.push((task.to_owned(), source));
            if limit.is_some_and(|maximum| candidates.len() >= maximum) {
                return Ok(candidates);
            }
        }
    }
    Ok(candidates)
}

/// Audit every selected authoritative dataset, collecting all mirror conflicts
/// rather than stopping at the first disagreement.
pub fn audit_discover(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<AuditInventory> {
    let candidates = candidate_directories(paths, task_filter, limit)?;
    let mut entries = Vec::with_capacity(candidates.len());
    let mut conflicts = Vec::new();
    for (task, source) in &candidates {
        let dataset_id = source
            .file_name()
            .and_then(|value| value.to_str())
            .context("candidate dataset path has no UTF-8 identity")?
            .to_owned();
        let mirror = match compare_mirror_evidence(paths, task, &dataset_id, source)? {
            MirrorEvidence::Absent => None,
            MirrorEvidence::Matched(mirror) => Some(mirror),
            MirrorEvidence::Conflict(conflict) => {
                conflicts.push(conflict);
                continue;
            }
        };
        let meta_path = source.join("meta.json");
        let meta = read_meta(&meta_path)?;
        entries.push(InventoryEntry {
            task: task.clone(),
            dataset_id: dataset_id.clone(),
            dataset_hash: dataset_id.clone(),
            source: source.clone(),
            source_family: source_family(&meta, &dataset_id)?,
            train_sha256: sha256_file(&source.join("train.csv"))?,
            test_sha256: sha256_file(&source.join("test.csv"))?,
            meta_sha256: sha256_file(&meta_path)?,
            mirror,
        });
    }
    conflicts.sort_by(|left, right| {
        (&left.task, &left.dataset_id).cmp(&(&right.task, &right.dataset_id))
    });
    Ok(AuditInventory {
        inspected: candidates.len(),
        entries,
        conflicts,
    })
}

pub fn write_conflict_summary(paths: &ExpansionPaths, audit: &AuditInventory) -> Result<PathBuf> {
    ensure!(
        !audit.conflicts.is_empty(),
        "cannot write an empty mirror conflict summary"
    );
    let conflicts = audit
        .conflicts
        .iter()
        .map(|conflict| {
            json!({
                "task": conflict.task,
                "dataset_id": conflict.dataset_id,
                "source": conflict.source,
                "mirror": conflict.mirror,
                "disagreements": conflict.disagreements,
                "conflict_sha256": conflict.conflict_sha256,
                "receipt": conflict.receipt,
            })
        })
        .collect::<Vec<_>>();
    let mut payload = json!({
        "schema": "JopeExpandedMirrorConflictSummary.v1",
        "authoritative_root": paths.source_root,
        "mirror_root": paths.mirror_root,
        "inspected": audit.inspected,
        "matched_or_unmirrored": audit.entries.len(),
        "conflict_count": conflicts.len(),
        "conflicts": conflicts,
    });
    let digest = canonical_json_sha256(&payload)?;
    payload["summary_sha256"] = Value::String(digest.clone());
    let summary = paths
        .receipt_root
        .join("summaries")
        .join(format!("{digest}.json"));
    if summary.is_file() {
        ensure!(
            read_json(&summary)? == payload,
            "mirror conflict summary content drift"
        );
    } else {
        atomic_json(&summary, &payload)?;
    }
    atomic_json(
        &paths.receipt_root.join("latest-summary.json"),
        &json!({
            "schema": "JopeExpandedMirrorConflictSummaryPointer.v1",
            "summary": summary,
            "summary_sha256": digest,
            "conflict_count": audit.conflicts.len(),
            "inspected": audit.inspected,
        }),
    )?;
    Ok(summary)
}

pub fn discover(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<InventoryEntry>> {
    let audit = audit_discover(paths, task_filter, limit)?;
    if let Some(conflict) = audit.conflicts.first() {
        bail!(
            "{}:{}: mirror content conflict; receipt={}",
            conflict.task,
            conflict.dataset_id,
            conflict.receipt.display()
        );
    }
    Ok(audit.entries)
}

pub fn cache_path(paths: &ExpansionPaths, task: &str, dataset_id: &str) -> PathBuf {
    paths
        .cache_root
        .join(format!("{task}-expanded"))
        .join(dataset_id)
}

fn copy_reflink_or_copy(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let input = File::open(source)?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(input.metadata()?.permissions().mode())
        .open(destination)?;
    // Linux FICLONE; unsupported filesystems return an error and fall back to
    // buffered std::fs::copy.
    let cloned =
        unsafe { libc::ioctl(output.as_raw_fd(), 0x4004_9409_u64, input.as_raw_fd()) } == 0;
    drop(output);
    if !cloned {
        fs::copy(source, destination)?;
        fs::set_permissions(destination, input.metadata()?.permissions())?;
    }
    Ok(())
}

fn encoder_protocol_sha256() -> Result<String> {
    canonical_json_sha256(&json!({"protocol": ENCODER_PROTOCOL, "train_only": true}))
}

pub fn validate_cache(path: &Path, expected: Option<&InventoryEntry>) -> Result<CacheManifest> {
    let manifest_path = path.join("cache-manifest.v1.json");
    let manifest: CacheManifest = serde_json::from_value(read_json(&manifest_path)?)?;
    ensure!(
        manifest.schema == "JopeExpandedEncodingCache.v1"
            && manifest.immutable
            && manifest.encoder_protocol == ENCODER_PROTOCOL
            && manifest.train_only_encoding,
        "expanded cache envelope is invalid: {}",
        path.display()
    );
    ensure!(
        manifest.encoder_protocol_sha256 == encoder_protocol_sha256()?,
        "expanded cache encoder protocol drift"
    );
    if let Some(entry) = expected {
        ensure!(
            manifest.task == entry.task
                && manifest.dataset_id == entry.dataset_id
                && manifest.dataset_hash == entry.dataset_hash
                && manifest.train_sha256 == entry.train_sha256
                && manifest.test_sha256 == entry.test_sha256
                && manifest.source_family == entry.source_family,
            "expanded cache identity drift: {}",
            path.display()
        );
    }
    for required in [
        "train.csv",
        "test.csv",
        "meta.json",
        "signature_manifest.json",
        "hyperion3train128dim.vs",
        "train3.vs",
        "train_target.vs",
    ] {
        ensure!(
            manifest.files.contains_key(required),
            "expanded cache inventory is missing {required}"
        );
    }
    for (name, digest) in &manifest.files {
        let file = path.join(name);
        ensure!(
            file.is_file() && sha256_file(&file)? == *digest,
            "expanded cache content drift: {}",
            file.display()
        );
    }
    manifest.cache_identity().validate()?;
    Ok(manifest)
}

fn is_legacy_cache(path: &Path) -> Result<bool> {
    let value = read_json(&path.join("cache-manifest.v1.json"))?;
    Ok(
        value["schema"].as_str() == Some("JopeExpandedEncodingCache.v1")
            && value["immutable"].as_bool() == Some(true)
            && value["train_only_encoding"].as_bool() == Some(true)
            && value["dataset_id"].is_null()
            && value["source_family"].is_null()
            && value["train_sha256"].is_null()
            && value["test_sha256"].is_null(),
    )
}

/// Upgrade the first-generation sidecar-only cache envelope in place.
///
/// The old encoder produced trustworthy sidecars, but its manifest omitted
/// dataset identity, source CSVs, and the signature manifest required by the
/// current activation contract. Every old manifest/file hash is checked before
/// any new file is written; the only added data files come from the audited
/// authoritative source.
fn repair_legacy_cache(path: &Path, entry: &InventoryEntry) -> Result<CacheManifest> {
    ensure!(
        path.is_dir(),
        "legacy cache directory is absent: {}",
        path.display()
    );
    ensure!(
        !path.symlink_metadata()?.file_type().is_symlink(),
        "legacy cache directory must not be a symlink: {}",
        path.display()
    );
    let legacy = read_json(&path.join("cache-manifest.v1.json"))?;
    ensure!(
        legacy["schema"].as_str() == Some("JopeExpandedEncodingCache.v1")
            && legacy["immutable"].as_bool() == Some(true)
            && legacy["train_only_encoding"].as_bool() == Some(true),
        "existing cache is not a recognized legacy envelope: {}",
        path.display()
    );
    ensure!(
        legacy["task"].as_str() == Some(entry.task.as_str())
            && legacy["dataset_hash"].as_str() == Some(entry.dataset_hash.as_str())
            && legacy["source"].as_str() == entry.source.to_str(),
        "legacy cache identity drift: {}",
        path.display()
    );
    ensure!(
        legacy["encoder_protocol"].as_str() == Some(ENCODER_PROTOCOL),
        "legacy cache encoder protocol drift: {}",
        path.display()
    );
    let legacy_files = legacy["files"]
        .as_object()
        .context("legacy cache manifest lacks files")?;
    for (name, digest) in legacy_files {
        let expected = digest
            .as_str()
            .with_context(|| format!("legacy cache file hash is invalid: {name}"))?;
        let file = path.join(name);
        ensure!(
            file.is_file() && sha256_file(&file)? == expected,
            "legacy cache content drift: {}",
            file.display()
        );
    }
    for name in REQUIRED_ENCODINGS {
        validate_sidecar(&path.join(name), name)
            .with_context(|| format!("legacy cache sidecar validation failed: {name}"))?;
    }
    for name in SIDECAR_NAMES {
        let file = path.join(name);
        if file.is_file() {
            validate_sidecar(&file, name)
                .with_context(|| format!("legacy cache sidecar validation failed: {name}"))?;
        }
    }
    ensure!(
        sha256_file(&entry.source.join("train.csv"))? == entry.train_sha256
            && sha256_file(&entry.source.join("test.csv"))? == entry.test_sha256
            && sha256_file(&entry.source.join("meta.json"))? == entry.meta_sha256,
        "authoritative source changed during legacy cache repair"
    );

    for name in ["train.csv", "test.csv", "meta.json"] {
        let destination = path.join(name);
        let expected = match name {
            "train.csv" => &entry.train_sha256,
            "test.csv" => &entry.test_sha256,
            "meta.json" => &entry.meta_sha256,
            _ => unreachable!(),
        };
        if destination.is_file() {
            ensure!(
                sha256_file(&destination)? == *expected,
                "legacy cache data drift: {}",
                destination.display()
            );
            continue;
        }
        let temporary = path.join(format!(
            ".{}.repair-{}-{}",
            name,
            std::process::id(),
            now_ms()
        ));
        copy_reflink_or_copy(&entry.source.join(name), &temporary)?;
        let result = (|| -> Result<()> {
            ensure!(
                sha256_file(&temporary)? == *expected,
                "repaired cache data hash drift: {}",
                temporary.display()
            );
            fs::rename(&temporary, &destination)?;
            Ok(())
        })();
        if result.is_err() {
            cleanup_staging(&temporary);
        }
        result?;
    }

    let encoding_hashes = REQUIRED_ENCODINGS
        .into_iter()
        .chain(OPTIONAL_ENCODINGS)
        .filter(|name| path.join(name).is_file())
        .map(|name| Ok((name.to_owned(), sha256_file(&path.join(name))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let signature = json!({
        "schema": "JopeExpandedSignatureManifest.v1",
        "task": entry.task,
        "dataset_id": entry.dataset_id,
        "train_only": true,
        "encoder_protocol": ENCODER_PROTOCOL,
        "encoder_protocol_sha256": encoder_protocol_sha256()?,
        "files": encoding_hashes,
    });
    let signature_path = path.join("signature_manifest.json");
    if signature_path.is_file() {
        ensure!(
            read_json(&signature_path)? == signature,
            "legacy cache signature manifest drift: {}",
            signature_path.display()
        );
    } else {
        atomic_json(&signature_path, &signature)?;
    }

    let files = fs::read_dir(path)?
        .filter_map(std::result::Result::ok)
        .filter(|item| {
            item.file_type().is_ok_and(|kind| kind.is_file())
                && item.file_name() != "cache-manifest.v1.json"
        })
        .map(|item| {
            let name = item.file_name().to_string_lossy().to_string();
            Ok((name, sha256_file(&item.path())?))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    ensure!(
        files["train.csv"] == entry.train_sha256
            && files["test.csv"] == entry.test_sha256
            && files["meta.json"] == entry.meta_sha256,
        "repaired cache data inventory drift"
    );
    let manifest = CacheManifest {
        schema: "JopeExpandedEncodingCache.v1".to_owned(),
        immutable: true,
        task: entry.task.clone(),
        dataset_id: entry.dataset_id.clone(),
        dataset_hash: entry.dataset_hash.clone(),
        source: entry.source.clone(),
        source_family: entry.source_family.clone(),
        train_sha256: entry.train_sha256.clone(),
        test_sha256: entry.test_sha256.clone(),
        encoder_protocol: ENCODER_PROTOCOL.to_owned(),
        encoder_protocol_sha256: encoder_protocol_sha256()?,
        train_only_encoding: true,
        files,
        created_at_ms: now_ms(),
    };
    manifest.cache_identity().validate()?;
    atomic_json(
        &path.join("cache-manifest.v1.json"),
        &serde_json::to_value(&manifest)?,
    )?;
    validate_cache(path, Some(entry))
}

fn verify_native_encoder(path: &Path) -> Result<()> {
    ensure!(path.is_file(), "Rust encoder is absent: {}", path.display());
    let mut header = [0_u8; 4];
    File::open(path)?.read_exact(&mut header)?;
    ensure!(
        header == [0x7f, b'E', b'L', b'F'],
        "encoder must be a native Rust binary (ELF), never a Python/script entrypoint"
    );
    Ok(())
}

fn run_native_encoder(encoder: &Path, source: &Path, destination: &Path, task: &str) -> Result<()> {
    verify_native_encoder(encoder)?;
    let output = Command::new(encoder)
        .args([
            "--train-csv",
            source.join("train.csv").to_string_lossy().as_ref(),
            "--output-dir",
            destination.to_string_lossy().as_ref(),
            "--task",
            task,
            "--train-only",
        ])
        .output()
        .with_context(|| format!("run native Hyperion encoder {}", encoder.display()))?;
    ensure!(
        output.status.success(),
        "native Hyperion encoder failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn cleanup_staging(path: &Path) {
    if path.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else if path.exists() {
        let _ = fs::remove_file(path);
    }
}

pub fn materialize_cache(
    paths: &ExpansionPaths,
    entry: &InventoryEntry,
    rust_encoder: Option<&Path>,
) -> Result<CacheManifest> {
    entry.dataset_identity().validate()?;
    // Recheck immediately before materialization; an audit result is not a
    // permission to ignore later mirror drift.
    compare_mirror(paths, &entry.task, &entry.dataset_id, &entry.source)?;
    ensure!(
        sha256_file(&entry.source.join("train.csv"))? == entry.train_sha256
            && sha256_file(&entry.source.join("test.csv"))? == entry.test_sha256,
        "authoritative dataset changed after discovery"
    );
    let final_path = cache_path(paths, &entry.task, &entry.dataset_id);
    let lock_path = final_path
        .parent()
        .context("cache path has no parent")?
        .join(".locks")
        .join(format!("{}.lock", entry.dataset_id));
    let _lock = FileLock::exclusive(&lock_path)?;
    if final_path.is_dir() {
        match validate_cache(&final_path, Some(entry)) {
            Ok(manifest) => return Ok(manifest),
            Err(validation_error) => {
                if is_legacy_cache(&final_path)? {
                    return repair_legacy_cache(&final_path, entry).with_context(|| {
                        format!(
                            "upgrade legacy expanded cache {} after validation failure: {validation_error}",
                            final_path.display()
                        )
                    });
                }
                return Err(validation_error);
            }
        }
    }
    let staging = final_path.with_file_name(format!(
        ".{}.tmp-{}-{}",
        entry.dataset_id,
        std::process::id(),
        now_ms()
    ));
    fs::create_dir_all(&staging)?;
    let result = (|| -> Result<CacheManifest> {
        for name in ["train.csv", "test.csv", "meta.json"] {
            copy_reflink_or_copy(&entry.source.join(name), &staging.join(name))?;
        }
        for name in REQUIRED_ENCODINGS.into_iter().chain(OPTIONAL_ENCODINGS) {
            let source_file = entry.source.join(name);
            if source_file.is_file() {
                copy_reflink_or_copy(&source_file, &staging.join(name))?;
            }
        }
        let missing = REQUIRED_ENCODINGS
            .into_iter()
            .filter(|name| !staging.join(name).is_file())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let encoder = rust_encoder.with_context(|| {
                format!(
                    "{}:{} missing {}; no native Rust Hyperion encoder supplied",
                    entry.task,
                    entry.dataset_id,
                    missing.join(",")
                )
            })?;
            run_native_encoder(encoder, &entry.source, &staging, &entry.task)?;
        }
        for name in REQUIRED_ENCODINGS {
            ensure!(
                staging.join(name).is_file(),
                "native encoder did not produce {name}"
            );
        }
        let encoding_hashes = REQUIRED_ENCODINGS
            .into_iter()
            .chain(OPTIONAL_ENCODINGS)
            .filter(|name| staging.join(name).is_file())
            .map(|name| Ok((name.to_owned(), sha256_file(&staging.join(name))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        atomic_json(
            &staging.join("signature_manifest.json"),
            &json!({
                "schema": "JopeExpandedSignatureManifest.v1",
                "task": entry.task,
                "dataset_id": entry.dataset_id,
                "train_only": true,
                "encoder_protocol": ENCODER_PROTOCOL,
                "encoder_protocol_sha256": encoder_protocol_sha256()?,
                "files": encoding_hashes,
            }),
        )?;
        let files = fs::read_dir(&staging)?
            .filter_map(std::result::Result::ok)
            .filter(|item| item.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|item| {
                let name = item.file_name().to_string_lossy().to_string();
                Ok((name, sha256_file(&item.path())?))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            files["train.csv"] == entry.train_sha256
                && files["test.csv"] == entry.test_sha256
                && files["meta.json"] == entry.meta_sha256,
            "staged data hash validation failed"
        );
        let manifest = CacheManifest {
            schema: "JopeExpandedEncodingCache.v1".to_owned(),
            immutable: true,
            task: entry.task.clone(),
            dataset_id: entry.dataset_id.clone(),
            dataset_hash: entry.dataset_hash.clone(),
            source: entry.source.clone(),
            source_family: entry.source_family.clone(),
            train_sha256: entry.train_sha256.clone(),
            test_sha256: entry.test_sha256.clone(),
            encoder_protocol: ENCODER_PROTOCOL.to_owned(),
            encoder_protocol_sha256: encoder_protocol_sha256()?,
            train_only_encoding: true,
            files,
            created_at_ms: now_ms(),
        };
        manifest.cache_identity().validate()?;
        atomic_json(
            &staging.join("cache-manifest.v1.json"),
            &serde_json::to_value(&manifest)?,
        )?;
        validate_cache(&staging, Some(entry))?;
        fs::create_dir_all(final_path.parent().context("cache path has no parent")?)?;
        fs::rename(&staging, &final_path)?;
        validate_cache(&final_path, Some(entry))
    })();
    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
}

fn deterministic_sealed_key(seed: u64, row: &InventoryEntry) -> [u8; 32] {
    Sha256::digest(
        format!(
            "{seed}|sealed|{}|{}|{}",
            row.task, row.source_family, row.dataset_hash
        )
        .as_bytes(),
    )
    .into()
}

fn cohort_hash(cohort: &SealedCohort) -> Result<String> {
    let value = json!({
        "schema": cohort.schema,
        "immutable": cohort.immutable,
        "classification": cohort.classification,
        "training_ingestion_eligible": cohort.training_ingestion_eligible,
        "counts": cohort.counts,
        "entries": cohort.entries,
        "created_at_ms": cohort.created_at_ms,
    });
    canonical_json_sha256(&value)
}

fn validate_sealed_cohort(cohort: &SealedCohort) -> Result<()> {
    ensure!(
        cohort.schema == "JopeSealedExpandedCohort.v1"
            && cohort.immutable
            && cohort.classification == "sealed_zero_shot"
            && !cohort.training_ingestion_eligible,
        "sealed cohort envelope drift"
    );
    ensure!(
        cohort.cohort_sha256 == cohort_hash(cohort)?,
        "sealed cohort content drift"
    );
    let mut identities = HashSet::new();
    let mut families = HashSet::new();
    for entry in &cohort.entries {
        entry.dataset_identity().validate()?;
        ensure!(
            identities.insert((entry.task.as_str(), entry.dataset_hash.as_str())),
            "sealed cohort repeats a dataset identity"
        );
        ensure!(
            families.insert(entry.source_family.as_str()),
            "sealed cohort repeats a source family"
        );
    }
    let mut expected_total = 0_usize;
    for task in TASKS {
        let declared = cohort.counts.get(task).copied().unwrap_or(0);
        let observed = cohort.entries.iter().filter(|row| row.task == task).count();
        let target = sealed_target(task);
        // Legacy tasks must carry their full frozen carve-out. The new tasks
        // are absent from cohorts frozen before their carve-out exists (D7):
        // zero is legal for them, any other shortfall is drift.
        let legacy = matches!(task, "regression" | "binary");
        let complete = declared == target || (!legacy && declared == 0);
        ensure!(
            observed == declared && complete,
            "{task}: sealed cohort count drift"
        );
        expected_total += declared;
    }
    ensure!(
        cohort.entries.len() == expected_total,
        "sealed cohort total count drift"
    );
    Ok(())
}

/// Freeze exactly `sealed_target(task)` family-disjoint datasets per task in
/// a private file (256 per legacy task; D7 carve-outs for the new tasks).
///
/// The returned value deliberately contains only counts and the cohort digest;
/// identities and families never appear in normal status output.
pub fn freeze_sealed(
    paths: &ExpansionPaths,
    entries: &[InventoryEntry],
    seed: u64,
) -> Result<Value> {
    if paths.sealed_path.is_file() {
        let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
        validate_sealed_cohort(&cohort)?;
        return Ok(json!({"cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts}));
    }
    let mut selected = Vec::with_capacity(TASKS.iter().map(|task| sealed_target(task)).sum());
    let mut used_families = HashSet::new();
    let mut counts = BTreeMap::new();
    for task in TASKS {
        let target = sealed_target(task);
        let mut candidates = entries
            .iter()
            .filter(|entry| entry.task == task)
            .cloned()
            .collect::<Vec<_>>();
        candidates.sort_by_key(|row| deterministic_sealed_key(seed, row));
        for row in candidates {
            if used_families.insert(row.source_family.clone()) {
                selected.push(row);
                if selected.iter().filter(|row| row.task == task).count() == target {
                    break;
                }
            }
        }
        ensure!(
            selected.iter().filter(|row| row.task == task).count() == target,
            "{task}: insufficient family-disjoint candidates for sealed cohort"
        );
        counts.insert(task.to_owned(), target);
    }
    let mut cohort = SealedCohort {
        schema: "JopeSealedExpandedCohort.v1".to_owned(),
        immutable: true,
        classification: "sealed_zero_shot".to_owned(),
        training_ingestion_eligible: false,
        counts,
        entries: selected,
        created_at_ms: now_ms(),
        cohort_sha256: String::new(),
    };
    cohort.cohort_sha256 = cohort_hash(&cohort)?;
    validate_sealed_cohort(&cohort)?;
    atomic_json(&paths.sealed_path, &serde_json::to_value(&cohort)?)?;
    fs::set_permissions(&paths.sealed_path, fs::Permissions::from_mode(0o600))?;
    if let Some(parent) = paths.sealed_path.parent() {
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }
    Ok(json!({"cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts}))
}

type DatasetKeys = HashSet<(String, String)>;
type FamilyKeys = HashSet<String>;

fn load_sealed_exclusions(paths: &ExpansionPaths) -> Result<(DatasetKeys, FamilyKeys)> {
    ensure!(
        paths.sealed_path.is_file(),
        "sealed cohort must be frozen before adaptive activation"
    );
    let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
    validate_sealed_cohort(&cohort)?;
    Ok((
        cohort
            .entries
            .iter()
            .map(|row| (row.task.clone(), row.dataset_hash.clone()))
            .collect(),
        cohort
            .entries
            .iter()
            .map(|row| row.source_family.clone())
            .collect(),
    ))
}

/// Remove private sealed identities and their entire source families before an
/// inventory can feed cache materialization or adaptive admission.
pub fn adaptive_inventory(
    paths: &ExpansionPaths,
    entries: &[InventoryEntry],
) -> Result<Vec<InventoryEntry>> {
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    Ok(entries
        .iter()
        .filter(|row| {
            !sealed_identities.contains(&(row.task.clone(), row.dataset_hash.clone()))
                && !sealed_families.contains(&row.source_family)
        })
        .cloned()
        .collect())
}

fn read_vs_matrix(path: &Path, expected_width: Option<usize>) -> Result<Vec<Vec<f32>>> {
    let input = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    let mut width = expected_width;
    for (line_index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row = line
            .trim()
            .split(',')
            .map(|cell| {
                let value: f32 = cell.trim().parse()?;
                ensure!(value.is_finite(), "non-finite .vs value");
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("{}:{}", path.display(), line_index + 1))?;
        let expected = *width.get_or_insert(row.len());
        ensure!(
            row.len() == expected,
            "{}:{} expected {expected} floats, found {}",
            path.display(),
            line_index + 1,
            row.len()
        );
        rows.push(row);
    }
    ensure!(!rows.is_empty(), "{}: empty .vs matrix", path.display());
    Ok(rows)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len().saturating_sub(1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

fn hyperion_signature(path: &Path) -> Result<Vec<f32>> {
    let rows = read_vs_matrix(path, Some(128))?;
    let count = rows.len() as f64;
    let mut result = Vec::with_capacity(512);
    for statistic in 0..4 {
        for column in 0..128 {
            let mut values = rows
                .iter()
                .map(|row| f64::from(row[column]))
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f64>() / count;
            let value = match statistic {
                0 => mean,
                1 => (values
                    .iter()
                    .map(|value| (value - mean) * (value - mean))
                    .sum::<f64>()
                    / count)
                    .sqrt(),
                2 => {
                    values.sort_by(f64::total_cmp);
                    quantile(&values, 0.10)
                }
                3 => {
                    values.sort_by(f64::total_cmp);
                    quantile(&values, 0.90)
                }
                _ => unreachable!(),
            };
            result.push(value as f32);
        }
    }
    Ok(result)
}

fn target_summary(path: &Path) -> Result<Vec<f32>> {
    let rows = read_vs_matrix(path, None)?;
    let width = rows[0].len();
    let mut output = vec![0.0_f32; TARGET_DIM];
    for column in 0..width.min(TARGET_DIM) {
        output[column] =
            (rows.iter().map(|row| f64::from(row[column])).sum::<f64>() / rows.len() as f64) as f32;
    }
    Ok(output)
}

#[derive(Debug)]
struct NpyArray {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn parse_npy(mut reader: impl Read) -> Result<NpyArray> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic[..6] == b"\x93NUMPY", "invalid NPY magic");
    let major = magic[6];
    let header_len = if major <= 1 {
        let mut bytes = [0_u8; 2];
        reader.read_exact(&mut bytes)?;
        u16::from_le_bytes(bytes) as usize
    } else {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        u32::from_le_bytes(bytes) as usize
    };
    let mut header = vec![0_u8; header_len];
    reader.read_exact(&mut header)?;
    let header = String::from_utf8(header)?;
    ensure!(
        header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
        "only little-endian f32 NPY arrays are supported"
    );
    let fortran_order =
        header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true");
    ensure!(
        fortran_order
            || header.contains("'fortran_order': False")
            || header.contains("\"fortran_order\": false"),
        "NPY header has no fortran_order flag"
    );
    let shape_marker = header.find("shape").context("NPY header has no shape")?;
    let open = header[shape_marker..]
        .find('(')
        .map(|position| position + shape_marker)
        .context("NPY shape has no opening parenthesis")?;
    let close = header[open..]
        .find(')')
        .map(|position| position + open)
        .context("NPY shape has no closing parenthesis")?;
    let shape = header[open + 1..close]
        .split(',')
        .filter_map(|value| {
            let value = value.trim();
            (!value.is_empty()).then(|| value.parse::<usize>())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(!shape.is_empty(), "NPY shape is empty");
    let count = shape.iter().product::<usize>();
    let mut bytes = vec![0_u8; count * 4];
    reader.read_exact(&mut bytes)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "NPY contains non-finite values"
    );
    // Pinned artifacts (e.g. the sklearn PCA projection npz) may be stored
    // column-major; callers assume row-major, so reorder rather than reject.
    let values = if fortran_order && shape.len() > 1 {
        let mut strides = vec![1_usize; shape.len()];
        for axis in 1..shape.len() {
            strides[axis] = strides[axis - 1] * shape[axis - 1];
        }
        let mut c_order = vec![0.0_f32; count];
        for (c_index, slot) in c_order.iter_mut().enumerate() {
            let mut remainder = c_index;
            let mut f_index = 0;
            for (axis, dim) in shape.iter().enumerate().rev() {
                f_index += (remainder % dim) * strides[axis];
                remainder /= dim;
            }
            *slot = values[f_index];
        }
        c_order
    } else {
        values
    };
    Ok(NpyArray { shape, values })
}

fn read_npy(path: &Path) -> Result<NpyArray> {
    parse_npy(File::open(path).with_context(|| format!("open NPY {}", path.display()))?)
}

fn read_npz_arrays(path: &Path) -> Result<BTreeMap<String, NpyArray>> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    let mut arrays = BTreeMap::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        if !member.name().ends_with(".npy") {
            continue;
        }
        let name = member
            .name()
            .rsplit('/')
            .next()
            .unwrap_or(member.name())
            .trim_end_matches(".npy")
            .to_owned();
        arrays.insert(name, parse_npy(&mut member)?);
    }
    Ok(arrays)
}

fn required_array<'a>(arrays: &'a BTreeMap<String, NpyArray>, name: &str) -> Result<&'a NpyArray> {
    arrays
        .get(name)
        .with_context(|| format!("projection NPZ lacks {name}"))
}

fn encode_context(
    cache: &Path,
    dataset_hash: &str,
    standardization_path: &Path,
) -> Result<Vec<u8>> {
    let standardization: Standardization =
        serde_json::from_value(read_json(standardization_path)?)?;
    ensure!(
        standardization.dim == 512
            && standardization.mean.len() == 512
            && standardization.scale.len() == 512,
        "standardization shape drift"
    );
    let mut signature = hyperion_signature(&cache.join("hyperion3train128dim.vs"))?;
    for (index, value) in signature.iter_mut().enumerate() {
        let scale = if standardization.scale[index].abs() < 1.0e-12 {
            1.0
        } else {
            standardization.scale[index]
        };
        *value = (*value - standardization.mean[index]) / scale;
    }
    let raw = read_npy(
        &standardization
            .foundation
            .cache_dir
            .join(format!("{dataset_hash}.npy")),
    )
    .with_context(|| format!("{dataset_hash}: foundation encoding is absent"))?;
    ensure!(
        raw.values.len() == standardization.foundation.raw_dim,
        "foundation raw dimension drift"
    );
    let arrays = read_npz_arrays(&standardization.foundation.projection_file)?;
    let components = required_array(&arrays, "components")?;
    let pca_mean = required_array(&arrays, "pca_mean")?;
    let z_mean = required_array(&arrays, "z_mean")?;
    let z_scale = required_array(&arrays, "z_scale")?;
    let raw_dim = standardization.foundation.raw_dim;
    let proj_dim = standardization.foundation.proj_dim;
    ensure!(
        components.shape == [proj_dim, raw_dim]
            && pca_mean.values.len() == raw_dim
            && z_mean.values.len() == proj_dim
            && z_scale.values.len() == proj_dim,
        "foundation projection shape drift"
    );
    let centered = raw
        .values
        .iter()
        .zip(&pca_mean.values)
        .map(|(value, mean)| value - mean)
        .collect::<Vec<_>>();
    let mut foundation = Vec::with_capacity(proj_dim);
    for row in components.values.chunks_exact(raw_dim) {
        let projected = centered
            .iter()
            .zip(row)
            .map(|(value, component)| value * component)
            .sum::<f32>();
        let index = foundation.len();
        ensure!(
            z_scale.values[index].abs() >= 1.0e-12,
            "foundation scale is zero"
        );
        foundation.push((projected - z_mean.values[index]) / z_scale.values[index]);
    }
    let target = target_summary(&cache.join("train_target.vs"))?;
    ensure!(
        signature.len() + foundation.len() == DATASET_DIM && target.len() == TARGET_DIM,
        "context dimension drift"
    );
    let mut output = Vec::with_capacity(CONTEXT_STRIDE);
    for value in signature.into_iter().chain(foundation).chain(target) {
        ensure!(value.is_finite(), "context contains non-finite values");
        output.extend_from_slice(&value.to_le_bytes());
    }
    ensure!(output.len() == CONTEXT_STRIDE, "context byte stride drift");
    Ok(output)
}

fn activated_identities(catalog_root: &Path) -> Result<HashSet<(String, String)>> {
    let ledger = catalog_root.join("activations.jsonl");
    if !ledger.is_file() {
        return Ok(HashSet::new());
    }
    let mut identities = HashSet::new();
    for line in BufReader::new(File::open(ledger)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let receipt: Value = serde_json::from_str(&line)?;
        let catalog = receipt["catalog"]
            .as_str()
            .context("activation receipt lacks catalog")?;
        let value = read_json(Path::new(catalog))?;
        for row in value["entries"]
            .as_array()
            .context("catalog entries are absent")?
        {
            identities.insert((
                row["dataset"]["task"]
                    .as_str()
                    .context("catalog task is absent")?
                    .to_owned(),
                row["dataset"]["dataset_hash"]
                    .as_str()
                    .context("catalog dataset hash is absent")?
                    .to_owned(),
            ));
        }
    }
    Ok(identities)
}

fn existing_activation(catalog_root: &Path, tranche_id: &str) -> Result<Option<Value>> {
    let ledger = catalog_root.join("activations.jsonl");
    if !ledger.is_file() {
        return Ok(None);
    }
    for line in BufReader::new(File::open(ledger)?).lines() {
        let value: Value = serde_json::from_str(&line?)?;
        if value["tranche_id"].as_str() != Some(tranche_id) {
            continue;
        }
        let catalog = Path::new(
            value["catalog"]
                .as_str()
                .context("activation receipt lacks catalog")?,
        );
        let data = Path::new(
            value["data"]
                .as_str()
                .context("activation receipt lacks data")?,
        );
        ensure!(
            sha256_file(catalog)?
                == value["catalog_sha256"]
                    .as_str()
                    .context("activation receipt lacks catalog hash")?
                && sha256_file(data)?
                    == value["data_sha256"]
                        .as_str()
                        .context("activation receipt lacks data hash")?,
            "activation receipt content drift"
        );
        return Ok(Some(value));
    }
    Ok(None)
}

fn family_split(family: &str) -> &'static str {
    let digest = Sha256::digest(family.as_bytes());
    if u16::from_be_bytes([digest[0], digest[1]]) < (u16::MAX / 10) {
        "validation"
    } else {
        "train"
    }
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().context("output path has no parent")?)?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("data"),
        std::process::id(),
        now_ms()
    ));
    let mut output = File::create(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn activate_tranche(
    paths: &ExpansionPaths,
    tranche_id: &str,
    requested: &[InventoryEntry],
) -> Result<Value> {
    ensure!(!tranche_id.is_empty(), "tranche id is absent");
    ensure!(
        !requested.is_empty() && requested.len() <= TRANCHE_MAX,
        "adaptive tranche must contain 1..={TRANCHE_MAX} datasets"
    );
    let _lock = FileLock::exclusive(&paths.catalog_root.join(".activation.lock"))?;
    if let Some(receipt) = existing_activation(&paths.catalog_root, tranche_id)? {
        return Ok(receipt);
    }
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    let activated = activated_identities(&paths.catalog_root)?;
    let mut seen = HashSet::new();
    let mut contexts = Vec::with_capacity(requested.len());
    let mut entries = Vec::with_capacity(requested.len());
    for (index, row) in requested.iter().enumerate() {
        let identity = (row.task.clone(), row.dataset_hash.clone());
        ensure!(seen.insert(identity.clone()), "tranche repeats a dataset");
        ensure!(
            !sealed_identities.contains(&identity) && !sealed_families.contains(&row.source_family),
            "adaptive tranche intersects the sealed cohort"
        );
        ensure!(
            !activated.contains(&identity),
            "dataset was already activated in an earlier tranche"
        );
        compare_mirror(paths, &row.task, &row.dataset_id, &row.source)?;
        let cache = cache_path(paths, &row.task, &row.dataset_id);
        let manifest = validate_cache(&cache, Some(row))?;
        let context = encode_context(&cache, &row.dataset_hash, &paths.standardization)?;
        let context_sha256 = sha256_bytes(&context);
        let cache_manifest_sha256 = sha256_file(&cache.join("cache-manifest.v1.json"))?;
        let encoding_hashes = manifest
            .files
            .iter()
            .filter(|(name, _)| {
                REQUIRED_ENCODINGS.contains(&name.as_str())
                    || OPTIONAL_ENCODINGS.contains(&name.as_str())
            })
            .map(|(name, digest)| (name.clone(), digest.clone()))
            .collect::<BTreeMap<_, _>>();
        entries.push(CatalogEntryIdentity {
            dataset: row.dataset_identity(),
            source_family: row.source_family.clone(),
            cache_manifest_sha256,
            encoder_protocol_sha256: manifest.encoder_protocol_sha256,
            context_protocol_sha256: canonical_json_sha256(&json!({"protocol": CONTEXT_PROTOCOL}))?,
            context_sha256,
            encoding_hashes,
            catalog_index: index,
            split: family_split(&row.source_family).to_owned(),
        });
        contexts.push(context);
    }
    let blob = contexts.concat();
    let data_sha256 = sha256_bytes(&blob);
    let task_counts = TASKS
        .into_iter()
        .filter_map(|task| {
            let count = entries
                .iter()
                .filter(|entry| entry.dataset.task == task)
                .count();
            (count != 0).then(|| (task.to_owned(), count))
        })
        .collect::<BTreeMap<_, _>>();
    let identity = CatalogIdentity {
        row_count: entries.len(),
        task_counts: task_counts.clone(),
        data_sha256: data_sha256.clone(),
        entries: entries.clone(),
    };
    identity.validate()?;
    fs::create_dir_all(&paths.catalog_root)?;
    let data_path = paths.catalog_root.join(format!("{data_sha256}.f32.bin"));
    if data_path.is_file() {
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "expanded context blob content drift"
        );
    } else {
        atomic_bytes(&data_path, &blob)?;
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "expanded context blob write validation failed"
        );
    }
    let catalog_path = paths.catalog_root.join(format!("{tranche_id}.json"));
    let existing_catalog = catalog_path
        .is_file()
        .then(|| read_json(&catalog_path))
        .transpose()?;
    let activated_at_ms = existing_catalog
        .as_ref()
        .and_then(|value| value["activated_at_ms"].as_u64())
        .map(u128::from)
        .unwrap_or_else(now_ms);
    let mut payload = json!({
        "schema": "JopeExpandedCatalog.v2",
        "immutable": true,
        "append_only": true,
        "tranche_id": tranche_id,
        "task_counts": task_counts,
        "row_count": entries.len(),
        "dataset_dimensions": DATASET_DIM,
        "target_dimensions": TARGET_DIM,
        "record_stride_bytes": CONTEXT_STRIDE,
        "context_protocol": CONTEXT_PROTOCOL,
        "data_sha256": data_sha256,
        "source_family_disjoint": true,
        "entries": entries,
        "activated_at_ms": activated_at_ms,
    });
    payload["catalog_content_sha256"] = Value::String(canonical_json_sha256(&payload)?);
    if let Some(existing) = existing_catalog {
        ensure!(
            existing == payload,
            "refusing to rewrite immutable expanded tranche"
        );
    } else {
        atomic_json(&catalog_path, &payload)?;
    }
    let receipt = json!({
        "schema": "JopeExpandedCatalogActivation.v2",
        "tranche_id": tranche_id,
        "catalog": catalog_path,
        "catalog_sha256": sha256_file(&catalog_path)?,
        "data": data_path,
        "data_sha256": data_sha256,
        "row_count": requested.len(),
        "task_counts": task_counts,
        "activated_at_ms": payload["activated_at_ms"],
    });
    let ledger_path = paths.catalog_root.join("activations.jsonl");
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(&ledger_path)?;
    serde_json::to_writer(&mut ledger, &receipt)?;
    ledger.write_all(b"\n")?;
    ledger.sync_all()?;
    atomic_json(
        &paths.catalog_root.join("current.json"),
        &json!({
            "schema": "JopeExpandedCatalogPointer.v2",
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path)?,
            "data": data_path,
            "data_sha256": data_sha256,
            "task_counts": task_counts,
            "row_count": requested.len(),
            "tranche_id": tranche_id,
        }),
    )?;
    Ok(receipt)
}

pub fn validate_catalog_pointer(paths: &ExpansionPaths) -> Result<Value> {
    let pointer_path = paths.catalog_root.join("current.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        pointer["schema"].as_str() == Some("JopeExpandedCatalogPointer.v2"),
        "expanded catalog pointer schema drift"
    );
    let catalog_path = Path::new(
        pointer["catalog"]
            .as_str()
            .context("expanded catalog pointer lacks catalog")?,
    );
    let data_path = Path::new(
        pointer["data"]
            .as_str()
            .context("expanded catalog pointer lacks data")?,
    );
    ensure!(
        sha256_file(catalog_path)?
            == pointer["catalog_sha256"]
                .as_str()
                .context("pointer lacks catalog hash")?,
        "expanded catalog pointer hash drift"
    );
    ensure!(
        sha256_file(data_path)?
            == pointer["data_sha256"]
                .as_str()
                .context("pointer lacks data hash")?,
        "expanded catalog data hash drift"
    );
    let payload = read_json(catalog_path)?;
    let entries: Vec<CatalogEntryIdentity> = serde_json::from_value(payload["entries"].clone())?;
    let identity = CatalogIdentity {
        row_count: payload["row_count"]
            .as_u64()
            .context("catalog lacks row count")? as usize,
        task_counts: serde_json::from_value(payload["task_counts"].clone())?,
        data_sha256: payload["data_sha256"]
            .as_str()
            .context("catalog lacks data hash")?
            .to_owned(),
        entries,
    };
    identity.validate()?;
    ensure!(
        fs::metadata(data_path)?.len() as usize == identity.row_count * CONTEXT_STRIDE,
        "expanded catalog context stride drift"
    );
    let mut input = BufReader::new(File::open(data_path)?);
    let mut raw = vec![0_u8; CONTEXT_STRIDE];
    for entry in &identity.entries {
        input.read_exact(&mut raw)?;
        ensure!(
            sha256_bytes(&raw) == entry.context_sha256,
            "expanded catalog context slice drift"
        );
    }
    let mut trailing = [0_u8; 1];
    ensure!(
        input.read(&mut trailing)? == 0,
        "expanded catalog data has trailing bytes"
    );
    Ok(json!({
        "healthy": true,
        "tranche_id": pointer["tranche_id"],
        "row_count": identity.row_count,
        "task_counts": identity.task_counts,
        "data_sha256": identity.data_sha256,
    }))
}

pub fn status(paths: &ExpansionPaths) -> Result<Value> {
    let mut counts = BTreeMap::new();
    let mut valid = BTreeMap::new();
    let mut invalid = BTreeMap::new();
    for task in TASKS {
        let root = paths.cache_root.join(format!("{task}-expanded"));
        let directories = if root.is_dir() {
            fs::read_dir(&root)?
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .filter(|entry| valid_dataset_id(&entry.file_name().to_string_lossy()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut good = 0;
        let mut bad = 0;
        for directory in &directories {
            if validate_cache(&directory.path(), None).is_ok() {
                good += 1;
            } else {
                bad += 1;
            }
        }
        counts.insert(task.to_owned(), directories.len());
        valid.insert(task.to_owned(), good);
        invalid.insert(task.to_owned(), bad);
    }
    let sealed = if paths.sealed_path.is_file() {
        let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
        validate_sealed_cohort(&cohort)?;
        json!({"frozen": true, "cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts})
    } else {
        json!({"frozen": false})
    };
    let catalog = if paths.catalog_root.join("current.json").is_file() {
        validate_catalog_pointer(paths)?
    } else {
        json!({"healthy": false, "active": false})
    };
    let activation_count = if paths.catalog_root.join("activations.jsonl").is_file() {
        BufReader::new(File::open(paths.catalog_root.join("activations.jsonl"))?)
            .lines()
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .count()
    } else {
        0
    };
    Ok(json!({
        "schema": "JopeExpandedDatasetStatus.v2",
        "cache_directories": counts,
        "valid_caches": valid,
        "invalid_caches": invalid,
        "sealed": sealed,
        "active_catalog": catalog,
        "activation_count": activation_count,
    }))
}

pub fn load_inventory_file(path: &Path) -> Result<Vec<InventoryEntry>> {
    let value = read_json(path)?;
    let entries = if value.is_array() {
        value
    } else {
        value["entries"].clone()
    };
    serde_json::from_value(entries).context("inventory file has invalid entries")
}

pub fn write_inventory_file(path: &Path, entries: &[InventoryEntry]) -> Result<Value> {
    let counts = TASKS
        .into_iter()
        .map(|task| {
            (
                task.to_owned(),
                entries.iter().filter(|entry| entry.task == task).count(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut payload = json!({
        "schema": "JopeExpandedInventory.v2",
        "authoritative_root": "/home/ubuntu/remote_super/quant",
        "mirror_root": "/home/ubuntu/quant",
        "row_count": entries.len(),
        "task_counts": counts,
        "entries": entries,
        "created_at_ms": now_ms(),
    });
    payload["inventory_sha256"] = Value::String(canonical_json_sha256(&payload)?);
    atomic_json(path, &payload)?;
    Ok(json!({
        "inventory": path,
        "inventory_sha256": payload["inventory_sha256"],
        "row_count": entries.len(),
        "task_counts": counts,
    }))
}

pub fn select_unactivated_cached(
    paths: &ExpansionPaths,
    inventory: &[InventoryEntry],
    limit: usize,
) -> Result<Vec<InventoryEntry>> {
    ensure!(
        (1..=TRANCHE_MAX).contains(&limit),
        "limit must be 1..={TRANCHE_MAX}"
    );
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    let activated = activated_identities(&paths.catalog_root)?;
    let mut selected = Vec::new();
    for row in inventory {
        let identity = (row.task.clone(), row.dataset_hash.clone());
        if sealed_identities.contains(&identity)
            || sealed_families.contains(&row.source_family)
            || activated.contains(&identity)
        {
            continue;
        }
        let cache = cache_path(paths, &row.task, &row.dataset_id);
        if validate_cache(&cache, Some(row)).is_ok()
            && paths.standardization.is_file()
            && read_json(&paths.standardization)?
                .get("foundation")
                .and_then(|value| value.get("cache_dir"))
                .and_then(Value::as_str)
                .is_some_and(|root| {
                    Path::new(root)
                        .join(format!("{}.npy", row.dataset_hash))
                        .is_file()
                })
        {
            selected.push(row.clone());
            if selected.len() == limit {
                break;
            }
        }
    }
    ensure!(
        !selected.is_empty(),
        "no fully encoded adaptive datasets are ready"
    );
    Ok(selected)
}

const LIVE_POINTER_SCHEMA: &str = "JopeTestOnlyContextPointer.v1";
const LIVE_CATALOG_SCHEMA: &str = "JopeTestOnlyContextCatalog.v1";
pub const SIDECAR_NAMES: [&str; 4] = [
    "hyperion3train128dim.vs",
    "hyperion3train.vs",
    "train3.vs",
    "train_target.vs",
];

fn read_live_pointer(live_root: &Path) -> Result<(Value, Value, Vec<u8>)> {
    let pointer_path = live_root.join("current.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        pointer["schema"].as_str() == Some(LIVE_POINTER_SCHEMA),
        "live catalog pointer schema drift"
    );
    let catalog_path = Path::new(
        pointer["catalog"]
            .as_str()
            .context("live pointer lacks catalog")?,
    );
    ensure!(
        sha256_file(catalog_path)?
            == pointer["catalog_sha256"]
                .as_str()
                .context("live pointer lacks catalog hash")?,
        "live catalog file hash drift"
    );
    let catalog = read_json(catalog_path)?;
    ensure!(
        catalog["schema"].as_str() == Some(LIVE_CATALOG_SCHEMA),
        "live catalog schema drift"
    );
    let data_path = Path::new(
        pointer["data"]
            .as_str()
            .context("live pointer lacks data")?,
    );
    let blob = fs::read(data_path)?;
    let data_sha256 = pointer["data_sha256"]
        .as_str()
        .context("live pointer lacks data hash")?;
    ensure!(
        sha256_bytes(&blob) == data_sha256 && catalog["data_sha256"].as_str() == Some(data_sha256),
        "live catalog blob hash drift"
    );
    let entries = catalog["entries"]
        .as_array()
        .context("live catalog entries are absent")?;
    ensure!(
        blob.len() == entries.len() * CONTEXT_STRIDE
            && catalog["row_count"].as_u64() == Some(entries.len() as u64),
        "live catalog row/stride drift"
    );
    for entry in entries {
        let index = entry["catalog_index"]
            .as_u64()
            .context("live entry lacks catalog index")? as usize;
        let offset = index * CONTEXT_STRIDE;
        ensure!(
            sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE])
                == entry["context_sha256"]
                    .as_str()
                    .context("live entry lacks context hash")?,
            "live catalog context slice drift at index {index}"
        );
    }
    Ok((pointer, catalog, blob))
}

struct TrancheLoad {
    tranche_id: String,
    catalog_path: PathBuf,
    catalog_sha256: String,
    entries: Vec<CatalogEntryIdentity>,
    blob: Vec<u8>,
}

fn load_activated_tranches(catalog_root: &Path, exclude: &[String]) -> Result<Vec<TrancheLoad>> {
    let ledger = catalog_root.join("activations.jsonl");
    let mut loads: Vec<TrancheLoad> = Vec::new();
    let mut seen_ids = HashSet::new();
    if ledger.is_file() {
        for line in BufReader::new(File::open(&ledger)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let receipt: Value = serde_json::from_str(&line)?;
            let tranche_id = receipt["tranche_id"]
                .as_str()
                .context("activation receipt lacks tranche id")?
                .to_owned();
            if !seen_ids.insert(tranche_id.clone()) {
                continue;
            }
            if exclude.contains(&tranche_id) {
                continue;
            }
            let catalog_path = PathBuf::from(
                receipt["catalog"]
                    .as_str()
                    .context("activation receipt lacks catalog")?,
            );
            let catalog_sha256 = receipt["catalog_sha256"]
                .as_str()
                .context("activation receipt lacks catalog hash")?
                .to_owned();
            ensure!(
                sha256_file(&catalog_path)? == catalog_sha256,
                "{tranche_id}: tranche catalog file hash drift"
            );
            let payload = read_json(&catalog_path)?;
            ensure!(
                payload["schema"].as_str() == Some("JopeExpandedCatalog.v2"),
                "{tranche_id}: tranche catalog schema drift"
            );
            let entries: Vec<CatalogEntryIdentity> =
                serde_json::from_value(payload["entries"].clone())?;
            let data_path = PathBuf::from(
                receipt["data"]
                    .as_str()
                    .context("activation receipt lacks data")?,
            );
            let blob = fs::read(&data_path)?;
            ensure!(
                sha256_bytes(&blob)
                    == receipt["data_sha256"]
                        .as_str()
                        .context("activation receipt lacks data hash")?
                    && blob.len() == entries.len() * CONTEXT_STRIDE,
                "{tranche_id}: tranche blob drift"
            );
            for entry in &entries {
                let offset = entry.catalog_index * CONTEXT_STRIDE;
                ensure!(
                    sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE]) == entry.context_sha256,
                    "{tranche_id}: tranche context slice drift at {}",
                    entry.catalog_index
                );
            }
            loads.push(TrancheLoad {
                tranche_id,
                catalog_path,
                catalog_sha256,
                entries,
                blob,
            });
        }
    }
    for excluded in exclude {
        ensure!(
            seen_ids.contains(excluded),
            "excluded tranche {excluded} is not in the activation ledger"
        );
    }
    Ok(loads)
}

fn append_history(live_root: &Path, previous_pointer: &Value) -> Result<()> {
    let mut record = json!({
        "schema": "JopeLiveCatalogPointerHistory.v1",
        "replaced_at_ms": now_ms(),
        "pointer": previous_pointer,
    });
    record["record_sha256"] = Value::String(canonical_json_sha256(&record)?);
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(live_root.join("pointer-history.jsonl"))?;
    serde_json::to_writer(&mut ledger, &record)?;
    ledger.write_all(b"\n")?;
    ledger.sync_all()?;
    Ok(())
}

/// Fold every activated v2 tranche into the live v1-schema catalog.
///
/// v1 entries pass through unmodified; v2 entries are appended in ledger order
/// with `current_catalog: false`, cache-local CSV paths, family-inherited
/// splits, and freshly verified context slices. The pointer flip is atomic and
/// the replaced pointer is retained in `pointer-history.jsonl`.
pub fn merge_live(
    paths: &ExpansionPaths,
    live_root: &Path,
    exclude: &[String],
    dry_run: bool,
) -> Result<Value> {
    let _lock = (!dry_run)
        .then(|| FileLock::exclusive(&live_root.join(".merge.lock")))
        .transpose()?;
    let (pointer, catalog, v1_blob) = read_live_pointer(live_root)?;
    let v1_entries = catalog["entries"]
        .as_array()
        .context("live catalog entries are absent")?
        .clone();
    let mut identities = HashSet::new();
    let mut live_contexts: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut family_splits: BTreeMap<String, String> = BTreeMap::new();
    for entry in &v1_entries {
        let task = entry["task"].as_str().context("v1 entry lacks task")?;
        let dataset = entry["dataset_hash"]
            .as_str()
            .context("v1 entry lacks dataset hash")?;
        ensure!(
            identities.insert((task.to_owned(), dataset.to_owned())),
            "live catalog repeats {task}:{dataset}"
        );
        live_contexts.insert(
            (task.to_owned(), dataset.to_owned()),
            entry["context_sha256"]
                .as_str()
                .context("v1 entry lacks context hash")?
                .to_owned(),
        );
        let family = entry["source_family"]
            .as_str()
            .context("v1 entry lacks source family")?;
        let split = entry["split"].as_str().context("v1 entry lacks split")?;
        if let Some(existing) = family_splits.get(family) {
            ensure!(
                existing == split,
                "live catalog family {family} straddles train/validation"
            );
        } else {
            family_splits.insert(family.to_owned(), split.to_owned());
        }
    }
    let tranches = load_activated_tranches(&paths.catalog_root, exclude)?;
    let mut merged_entries = v1_entries.clone();
    let mut blob = v1_blob;
    let mut inherited_splits = 0_u64;
    let mut appended = Vec::new();
    let mut already_merged = Vec::new();
    for tranche in &tranches {
        // A tranche whose every identity is already live with an identical
        // context was merged by an earlier run; skip it so merge-live stays
        // idempotent. A partial or context-differing overlap fails closed.
        let merged_count = tranche
            .entries
            .iter()
            .filter(|entry| {
                live_contexts
                    .get(&(
                        entry.dataset.task.clone(),
                        entry.dataset.dataset_hash.clone(),
                    ))
                    .is_some_and(|context| *context == entry.context_sha256)
            })
            .count();
        if merged_count == tranche.entries.len() {
            already_merged.push(Value::String(tranche.tranche_id.clone()));
            continue;
        }
        ensure!(
            merged_count == 0,
            "{}: tranche is partially present in the live catalog",
            tranche.tranche_id
        );
        for entry in &tranche.entries {
            let identity = (
                entry.dataset.task.clone(),
                entry.dataset.dataset_hash.clone(),
            );
            ensure!(
                !live_contexts.contains_key(&identity),
                "{}: {}:{} exists in the live catalog with a different context",
                tranche.tranche_id,
                entry.dataset.task,
                entry.dataset.dataset_hash
            );
            ensure!(
                identities.insert(identity.clone()),
                "{}: duplicate dataset identity {}:{}",
                tranche.tranche_id,
                entry.dataset.task,
                entry.dataset.dataset_hash
            );
            let split = if let Some(v1_split) = family_splits.get(&entry.source_family) {
                if *v1_split != entry.split {
                    inherited_splits += 1;
                }
                v1_split.clone()
            } else {
                family_splits.insert(entry.source_family.clone(), entry.split.clone());
                entry.split.clone()
            };
            ensure!(
                family_splits.get(&entry.source_family) == Some(&split),
                "{}: family {} straddles train/validation",
                tranche.tranche_id,
                entry.source_family
            );
            let cache = cache_path(paths, &entry.dataset.task, &entry.dataset.dataset_hash);
            let train_csv = cache.join("train.csv");
            let test_csv = cache.join("test.csv");
            ensure!(
                sha256_file(&train_csv)? == entry.dataset.train_sha256
                    && sha256_file(&test_csv)? == entry.dataset.test_sha256,
                "{}: expanded cache CSV drift for {}",
                tranche.tranche_id,
                entry.dataset.dataset_hash
            );
            let metric = crate::task::TaskKind::parse(&entry.dataset.task)
                .with_context(|| {
                    format!("unknown task in expanded tranche: {}", entry.dataset.task)
                })?
                .primary_metric();
            let catalog_index = merged_entries.len();
            merged_entries.push(json!({
                "catalog_index": catalog_index,
                "context_sha256": entry.context_sha256,
                "current_catalog": false,
                "dataset_hash": entry.dataset.dataset_hash,
                "metric": metric,
                "source_catalog_index": Value::Null,
                "source_family": entry.source_family,
                "split": split,
                "task": entry.dataset.task,
                "test_csv": test_csv,
                "test_sha256": entry.dataset.test_sha256,
                "train_csv": train_csv,
                "train_sha256": entry.dataset.train_sha256,
            }));
        }
        blob.extend_from_slice(&tranche.blob);
        appended.push(json!({
            "tranche_id": tranche.tranche_id,
            "catalog": tranche.catalog_path,
            "catalog_sha256": tranche.catalog_sha256,
            "row_count": tranche.entries.len(),
        }));
    }
    if appended.is_empty() {
        return Ok(json!({
            "schema": "JopeLiveCatalogMergeResult.v1",
            "nothing_to_merge": true,
            "dry_run": dry_run,
            "row_count": v1_entries.len(),
            "already_merged_tranches": already_merged,
        }));
    }
    ensure!(
        blob.len() == merged_entries.len() * CONTEXT_STRIDE,
        "merged blob stride drift"
    );
    for entry in &merged_entries {
        let index = entry["catalog_index"].as_u64().context("index absent")? as usize;
        let offset = index * CONTEXT_STRIDE;
        ensure!(
            entry["context_sha256"].as_str()
                == Some(sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE]).as_str()),
            "merged context slice drift at index {index}"
        );
    }
    let mut train_families = HashSet::new();
    let mut validation_families = HashSet::new();
    for entry in &merged_entries {
        let family = entry["source_family"].as_str().context("family absent")?;
        match entry["split"].as_str().context("split absent")? {
            "validation" => validation_families.insert(family.to_owned()),
            _ => train_families.insert(family.to_owned()),
        };
    }
    ensure!(
        train_families.is_disjoint(&validation_families),
        "merged catalog families straddle train/validation"
    );
    let data_sha256 = sha256_bytes(&blob);
    let task_counts = TASKS
        .into_iter()
        .filter_map(|task| {
            let count = merged_entries
                .iter()
                .filter(|entry| entry["task"].as_str() == Some(task))
                .count();
            (count != 0).then(|| (task.to_owned(), count))
        })
        .collect::<BTreeMap<_, _>>();
    let regression_count = merged_entries
        .iter()
        .filter(|entry| entry["task"].as_str() == Some("regression"))
        .count();
    let binary_count = merged_entries
        .iter()
        .filter(|entry| entry["task"].as_str() == Some("binary"))
        .count();
    let mut merged_catalog = catalog.clone();
    merged_catalog["entries"] = Value::Array(merged_entries);
    merged_catalog["row_count"] = json!(
        v1_entries.len()
            + appended
                .iter()
                .map(|row| row["row_count"].as_u64().unwrap_or(0) as usize)
                .sum::<usize>()
    );
    merged_catalog["regression_count"] = json!(regression_count);
    merged_catalog["binary_count"] = json!(binary_count);
    merged_catalog["task_counts"] = json!(task_counts);
    merged_catalog["data_sha256"] = Value::String(data_sha256.clone());
    merged_catalog["source_family_disjoint"] = Value::Bool(true);
    merged_catalog["expanded_tranches"] = Value::Array(appended.clone());
    merged_catalog["merged_from_catalog_sha256"] = pointer["catalog_sha256"].clone();
    let catalog_content_sha256 = canonical_json_sha256(&merged_catalog)?;
    let summary = json!({
        "schema": "JopeLiveCatalogMergeResult.v1",
        "nothing_to_merge": false,
        "dry_run": dry_run,
        "row_count": merged_catalog["row_count"],
        "regression_count": regression_count,
        "binary_count": binary_count,
        "task_counts": task_counts,
        "inherited_splits": inherited_splits,
        "tranches": appended,
        "already_merged_tranches": already_merged,
        "data_sha256": data_sha256,
        "catalog_content_sha256": catalog_content_sha256,
    });
    if dry_run {
        return Ok(summary);
    }
    let data_path = live_root.join(format!("{data_sha256}.f32.bin"));
    if data_path.is_file() {
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "merged blob content drift on disk"
        );
    } else {
        atomic_bytes(&data_path, &blob)?;
    }
    let catalog_path = live_root.join(format!("{catalog_content_sha256}.catalog.json"));
    if catalog_path.is_file() {
        ensure!(
            read_json(&catalog_path)? == merged_catalog,
            "refusing to rewrite a differing merged catalog"
        );
    } else {
        atomic_json(&catalog_path, &merged_catalog)?;
    }
    append_history(live_root, &pointer)?;
    atomic_json(
        &live_root.join("current.json"),
        &json!({
            "schema": LIVE_POINTER_SCHEMA,
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path)?,
            "data": data_path,
            "data_sha256": data_sha256,
        }),
    )?;
    Ok(summary)
}

fn validate_sidecar(path: &Path, name: &str) -> Result<String> {
    let rows = match name {
        "hyperion3train128dim.vs" => read_vs_matrix(path, Some(128))?,
        _ => read_vs_matrix(path, None)?,
    };
    if name == "train_target.vs" {
        ensure!(
            rows.len() == 1 && rows[0].len() == TARGET_DIM,
            "{}: train_target.vs must be exactly 1x{TARGET_DIM}",
            path.display()
        );
    }
    sha256_file(path)
}

/// Additively install encoding sidecars into the authoritative registry.
///
/// Existing registry files are never overwritten: identical content is skipped,
/// differing content fails closed. Every action appends a sha receipt line.
pub fn install_sidecars(
    paths: &ExpansionPaths,
    from: &Path,
    task: &str,
    receipts: &Path,
) -> Result<Value> {
    ensure!(TASKS.contains(&task), "unsupported task: {task}");
    if let Some(parent) = receipts.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(receipts)?;
    fn record(
        ledger: &mut File,
        task: &str,
        dataset: &str,
        file: &str,
        digest: &str,
        action: &str,
    ) -> Result<()> {
        serde_json::to_writer(
            &mut *ledger,
            &json!({
                "schema": "JopeSidecarInstallReceipt.v1",
                "task": task,
                "dataset_id": dataset,
                "file": file,
                "sha256": digest,
                "action": action,
                "at_ms": now_ms(),
            }),
        )?;
        ledger.write_all(b"\n")?;
        Ok(())
    }
    let mut directories = fs::read_dir(from)
        .with_context(|| format!("read sidecar source {}", from.display()))?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| valid_dataset_id(&entry.file_name().to_string_lossy()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    directories.sort();
    let mut installed = 0_u64;
    let mut skipped = 0_u64;
    let mut missing_registry = 0_u64;
    for directory in &directories {
        let dataset_id = directory
            .file_name()
            .and_then(|value| value.to_str())
            .context("sidecar source has no UTF-8 identity")?
            .to_owned();
        let registry = paths
            .source_root
            .join(corpus_dir_name(task))
            .join(&dataset_id);
        if !registry.is_dir() {
            missing_registry += 1;
            record(
                &mut ledger,
                task,
                &dataset_id,
                "-",
                "-",
                "missing_registry_dir",
            )?;
            continue;
        }
        for name in SIDECAR_NAMES {
            let source = directory.join(name);
            if !source.is_file() {
                continue;
            }
            let source_sha256 = validate_sidecar(&source, name)?;
            let destination = registry.join(name);
            if destination.is_file() {
                let existing = sha256_file(&destination)?;
                if existing == source_sha256 {
                    skipped += 1;
                    record(
                        &mut ledger,
                        task,
                        &dataset_id,
                        name,
                        &existing,
                        "skipped_identical",
                    )?;
                    continue;
                }
                record(&mut ledger, task, &dataset_id, name, &existing, "conflict")?;
                ledger.sync_all()?;
                bail!(
                    "{task}:{dataset_id}: refusing to overwrite {name} (registry {existing} != source {source_sha256})"
                );
            }
            copy_reflink_or_copy(&source, &destination)?;
            ensure!(
                sha256_file(&destination)? == source_sha256,
                "{task}:{dataset_id}: installed sidecar hash drift for {name}"
            );
            installed += 1;
            record(
                &mut ledger,
                task,
                &dataset_id,
                name,
                &source_sha256,
                "installed",
            )?;
        }
    }
    ledger.sync_all()?;
    Ok(json!({
        "schema": "JopeSidecarInstallResult.v1",
        "task": task,
        "datasets": directories.len(),
        "installed": installed,
        "skipped_identical": skipped,
        "missing_registry_dir": missing_registry,
        "receipts": receipts,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn paths(temp: &TempDir) -> ExpansionPaths {
        ExpansionPaths {
            source_root: temp.path().join("remote"),
            mirror_root: temp.path().join("mirror"),
            cache_root: temp.path().join("cache"),
            catalog_root: temp.path().join("catalog"),
            receipt_root: temp.path().join("receipts"),
            sealed_path: temp.path().join("private/sealed.json"),
            standardization: temp.path().join("standardization.json"),
        }
    }

    fn dataset(root: &Path, task: &str, id: &str, seed: &str) {
        let path = root.join(task).join(id);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("train.csv"), format!("x,y\n1,{seed}\n")).unwrap();
        fs::write(path.join("test.csv"), format!("x,y\n2,{seed}\n")).unwrap();
        atomic_json(
            &path.join("meta.json"),
            &json!({"origin":"synthetic","generator_recipe":{"kind":task,"seed":seed}}),
        )
        .unwrap();
        fs::write(
            path.join("hyperion3train128dim.vs"),
            "0,".repeat(127) + "0\n",
        )
        .unwrap();
        fs::write(path.join("train3.vs"), "0\n").unwrap();
        fs::write(
            path.join("train_target.vs"),
            format!(
                "{}\n",
                (0..TARGET_DIM).map(|_| "0").collect::<Vec<_>>().join(",")
            ),
        )
        .unwrap();
    }

    #[test]
    fn mirror_conflict_is_fail_closed_and_receipted() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "0014b9dad5725a1e";
        dataset(&paths.source_root, "regression", id, "1");
        dataset(&paths.mirror_root, "regression", id, "2");
        let error = discover(&paths, Some("regression"), None).unwrap_err();
        assert!(error.to_string().contains("mirror content conflict"));
        assert_eq!(fs::read_dir(&paths.receipt_root).unwrap().count(), 1);
    }

    #[test]
    fn exhaustive_audit_receipts_every_conflict_and_writes_stable_summary() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        for (id, source_seed, mirror_seed) in [
            ("0014b9dad5725a1e", "1", "9"),
            ("006bca140f93e411", "2", "8"),
            ("00709d315efacf0c", "3", "3"),
        ] {
            dataset(&paths.source_root, "regression", id, source_seed);
            dataset(&paths.mirror_root, "regression", id, mirror_seed);
        }
        let first = audit_discover(&paths, Some("regression"), None).unwrap();
        assert_eq!(first.inspected, 3);
        assert_eq!(first.entries.len(), 1);
        assert_eq!(first.conflicts.len(), 2);
        assert_eq!(first.conflicts[0].dataset_id, "0014b9dad5725a1e");
        assert_eq!(first.conflicts[1].dataset_id, "006bca140f93e411");
        assert!(first
            .conflicts
            .iter()
            .all(|conflict| conflict.receipt.is_file()));
        let first_summary = write_conflict_summary(&paths, &first).unwrap();
        let first_bytes = fs::read(&first_summary).unwrap();

        let second = audit_discover(&paths, Some("regression"), None).unwrap();
        let second_summary = write_conflict_summary(&paths, &second).unwrap();
        assert_eq!(first_summary, second_summary);
        assert_eq!(first_bytes, fs::read(second_summary).unwrap());
        let pointer = read_json(&paths.receipt_root.join("latest-summary.json")).unwrap();
        assert_eq!(pointer["conflict_count"], 2);
        assert_eq!(pointer["inspected"], 3);
    }

    #[test]
    fn cache_is_atomic_and_validates_all_hashes() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "0014b9dad5725a1e";
        dataset(&paths.source_root, "regression", id, "1");
        let entry = discover(&paths, Some("regression"), None)
            .unwrap()
            .remove(0);
        let manifest = materialize_cache(&paths, &entry, None).unwrap();
        assert!(manifest.train_only_encoding);
        let cache = cache_path(&paths, "regression", id);
        validate_cache(&cache, Some(&entry)).unwrap();
        assert_eq!(
            materialize_cache(&paths, &entry, None).unwrap(),
            manifest,
            "a complete cache must be resumed without rewriting"
        );
        fs::write(cache.join("train.csv"), "drift").unwrap();
        assert!(validate_cache(&cache, Some(&entry)).is_err());
    }

    #[test]
    fn legacy_sidecar_cache_is_upgraded_with_verified_source_files() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "0014b9dad5725a1e";
        dataset(&paths.source_root, "timeseries_reduced_v2", id, "1");
        let entry = discover(&paths, Some("timeseries"), None)
            .unwrap()
            .remove(0);
        materialize_cache(&paths, &entry, None).unwrap();
        let cache = cache_path(&paths, "timeseries", id);
        let current = read_json(&cache.join("cache-manifest.v1.json")).unwrap();
        let old_files = current["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| {
                !matches!(
                    name.as_str(),
                    "train.csv" | "test.csv" | "signature_manifest.json"
                )
            })
            .map(|(name, digest)| (name.clone(), digest.clone()))
            .collect::<serde_json::Map<_, _>>();
        fs::remove_file(cache.join("cache-manifest.v1.json")).unwrap();
        fs::remove_file(cache.join("signature_manifest.json")).unwrap();
        fs::remove_file(cache.join("train.csv")).unwrap();
        fs::remove_file(cache.join("test.csv")).unwrap();
        atomic_json(
            &cache.join("cache-manifest.v1.json"),
            &json!({
                "schema": "JopeExpandedEncodingCache.v1",
                "task": entry.task,
                "source": entry.source,
                "dataset_hash": entry.dataset_hash,
                "encoder_protocol": ENCODER_PROTOCOL,
                "train_only_encoding": true,
                "immutable": true,
                "staged_by": "legacy-fixture",
                "files": old_files,
            }),
        )
        .unwrap();

        let manifest = materialize_cache(&paths, &entry, None).unwrap();
        assert_eq!(manifest.dataset_id, id);
        assert!(cache.join("train.csv").is_file());
        assert!(cache.join("test.csv").is_file());
        assert!(cache.join("signature_manifest.json").is_file());
        validate_cache(&cache, Some(&entry)).unwrap();
    }

    #[test]
    fn missing_encodings_fail_without_leaving_a_partial_cache() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "0014b9dad5725a1e";
        dataset(&paths.source_root, "binary", id, "1");
        fs::remove_file(
            paths
                .source_root
                .join("binary")
                .join(id)
                .join("train_target.vs"),
        )
        .unwrap();
        let entry = discover(&paths, Some("binary"), None).unwrap().remove(0);
        let error = materialize_cache(&paths, &entry, None).unwrap_err();
        assert!(error
            .to_string()
            .contains("no native Rust Hyperion encoder"));
        assert!(!cache_path(&paths, "binary", id).exists());
    }

    #[test]
    fn native_encoder_rejects_scripts() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("encoder.py");
        fs::write(&path, "#!/usr/bin/env python3\n").unwrap();
        assert!(verify_native_encoder(&path).is_err());
    }

    #[test]
    fn npy_parser_reads_f32_and_shape() {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_vec();
        while (bytes.len() + 2 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.5_f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0_f32).to_le_bytes());
        let parsed = parse_npy(bytes.as_slice()).unwrap();
        assert_eq!(parsed.shape, vec![2]);
        assert_eq!(parsed.values, vec![1.5, -2.0]);
    }

    #[test]
    fn sealed_status_never_exposes_dataset_identities() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let mut entries = Vec::new();
        for task in TASKS {
            for index in 0..sealed_target(task) {
                entries.push(InventoryEntry {
                    task: task.to_owned(),
                    dataset_id: format!("{index:016x}"),
                    dataset_hash: format!("{index:016x}"),
                    source: PathBuf::from("private"),
                    source_family: format!("{task}-family-{index}"),
                    train_sha256: "a".repeat(64),
                    test_sha256: "b".repeat(64),
                    meta_sha256: "c".repeat(64),
                    mirror: None,
                });
            }
        }
        let redacted = freeze_sealed(&paths, &entries, 7).unwrap();
        let encoded = serde_json::to_string(&redacted).unwrap();
        assert!(!encoded.contains("dataset_id"));
        assert!(!encoded.contains("family-"));
        assert_eq!(
            fs::metadata(&paths.sealed_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        let adaptive = adaptive_inventory(&paths, &entries).unwrap();
        assert!(adaptive.is_empty());
    }

    #[test]
    fn family_split_is_deterministic_and_disjoint() {
        let family = "synthetic:abc";
        assert_eq!(family_split(family), family_split(family));
        let split = family_split(family);
        assert!(matches!(split, "train" | "validation"));
    }

    fn npy_bytes(shape: &[usize], values: &[f32]) -> Vec<u8> {
        let mut output = b"\x93NUMPY\x01\x00".to_vec();
        let shape = if shape.len() == 1 {
            format!("{},", shape[0])
        } else {
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}")
                .into_bytes();
        while (output.len() + 2 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        output.extend_from_slice(&(header.len() as u16).to_le_bytes());
        output.extend_from_slice(&header);
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
        output
    }

    fn write_projection(path: &Path) {
        use zip::write::SimpleFileOptions;
        let file = File::create(path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, shape, values) in [
            (
                "components.npy",
                vec![1024, 2048],
                vec![0.0_f32; 1024 * 2048],
            ),
            ("pca_mean.npy", vec![2048], vec![0.0_f32; 2048]),
            ("z_mean.npy", vec![1024], vec![0.0_f32; 1024]),
            ("z_scale.npy", vec![1024], vec![1.0_f32; 1024]),
        ] {
            archive.start_file(name, options).unwrap();
            archive.write_all(&npy_bytes(&shape, &values)).unwrap();
        }
        archive.finish().unwrap();
    }

    #[test]
    fn tranche_activation_is_hashed_validated_and_idempotent() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "feedb9dad5725a1e";
        dataset(&paths.source_root, "regression", id, "1");
        let entry = discover(&paths, Some("regression"), None)
            .unwrap()
            .remove(0);
        materialize_cache(&paths, &entry, None).unwrap();

        let foundation = temp.path().join("foundation");
        fs::create_dir_all(&foundation).unwrap();
        fs::write(
            foundation.join(format!("{id}.npy")),
            npy_bytes(&[2048], &vec![0.0; 2048]),
        )
        .unwrap();
        let projection = temp.path().join("projection.npz");
        write_projection(&projection);
        atomic_json(
            &paths.standardization,
            &json!({
                "dim": 512,
                "mean": vec![0.0_f32; 512],
                "scale": vec![1.0_f32; 512],
                "foundation": {
                    "cache_dir": foundation,
                    "projection_file": projection,
                    "raw_dim": 2048,
                    "proj_dim": 1024,
                }
            }),
        )
        .unwrap();
        let mut sealed = Vec::new();
        for task in TASKS {
            for index in 0..sealed_target(task) {
                sealed.push(InventoryEntry {
                    task: task.to_owned(),
                    dataset_id: format!("{index:016x}"),
                    dataset_hash: format!("{index:016x}"),
                    source: PathBuf::from("private"),
                    source_family: format!("{task}-sealed-{index}"),
                    train_sha256: "a".repeat(64),
                    test_sha256: "b".repeat(64),
                    meta_sha256: "c".repeat(64),
                    mirror: None,
                });
            }
        }
        freeze_sealed(&paths, &sealed, 7).unwrap();
        let first = activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(
            first,
            activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap()
        );
        let status = validate_catalog_pointer(&paths).unwrap();
        assert_eq!(status["healthy"], true);
        assert_eq!(status["row_count"], 1);
        assert_eq!(
            fs::read_to_string(paths.catalog_root.join("activations.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    /// Full activation fixture: one regression dataset, materialized cache,
    /// foundation/projection/standardization stubs, sealed cohort, activated
    /// tranche. Returns the entry and the activation receipt.
    fn activated_fixture(
        temp: &TempDir,
        id: &str,
        tranche_id: &str,
    ) -> (ExpansionPaths, InventoryEntry, Value) {
        let paths = paths(temp);
        dataset(&paths.source_root, "regression", id, "1");
        let entry = discover(&paths, Some("regression"), None)
            .unwrap()
            .remove(0);
        materialize_cache(&paths, &entry, None).unwrap();
        let foundation = temp.path().join("foundation");
        fs::create_dir_all(&foundation).unwrap();
        fs::write(
            foundation.join(format!("{id}.npy")),
            npy_bytes(&[2048], &vec![0.0; 2048]),
        )
        .unwrap();
        let projection = temp.path().join("projection.npz");
        write_projection(&projection);
        atomic_json(
            &paths.standardization,
            &json!({
                "dim": 512,
                "mean": vec![0.0_f32; 512],
                "scale": vec![1.0_f32; 512],
                "foundation": {
                    "cache_dir": foundation,
                    "projection_file": projection,
                    "raw_dim": 2048,
                    "proj_dim": 1024,
                }
            }),
        )
        .unwrap();
        let mut sealed = Vec::new();
        for task in TASKS {
            for index in 0..sealed_target(task) {
                sealed.push(InventoryEntry {
                    task: task.to_owned(),
                    dataset_id: format!("{index:016x}"),
                    dataset_hash: format!("{index:016x}"),
                    source: PathBuf::from("private"),
                    source_family: format!("{task}-sealed-{index}"),
                    train_sha256: "a".repeat(64),
                    test_sha256: "b".repeat(64),
                    meta_sha256: "c".repeat(64),
                    mirror: None,
                });
            }
        }
        freeze_sealed(&paths, &sealed, 7).unwrap();
        let receipt = activate_tranche(&paths, tranche_id, std::slice::from_ref(&entry)).unwrap();
        (paths, entry, receipt)
    }

    fn write_live_catalog(live_root: &Path, rows: &[(&str, &str, &str, &str)]) -> Value {
        fs::create_dir_all(live_root).unwrap();
        let mut blob = Vec::new();
        let mut entries = Vec::new();
        for (index, (task, dataset, family, split)) in rows.iter().enumerate() {
            let slice = vec![(index as u8).wrapping_add(1); CONTEXT_STRIDE];
            let context_sha256 = sha256_bytes(&slice);
            blob.extend_from_slice(&slice);
            entries.push(json!({
                "catalog_index": index,
                "context_sha256": context_sha256,
                "dataset_hash": dataset,
                "metric": if *task == "binary" { "auc" } else { "r2" },
                "source_catalog_index": index,
                "source_family": family,
                "split": split,
                "task": task,
                "test_csv": format!("/live/{dataset}/test.csv"),
                "test_sha256": "t".repeat(64),
                "train_csv": format!("/live/{dataset}/train.csv"),
                "train_sha256": "u".repeat(64),
            }));
        }
        let data_sha256 = sha256_bytes(&blob);
        let data_path = live_root.join(format!("{data_sha256}.f32.bin"));
        fs::write(&data_path, &blob).unwrap();
        let catalog = json!({
            "schema": "JopeTestOnlyContextCatalog.v1",
            "immutable": true,
            "dataset_dimensions": DATASET_DIM,
            "target_dimensions": TARGET_DIM,
            "record_stride_bytes": CONTEXT_STRIDE,
            "row_count": rows.len(),
            "regression_count": rows.iter().filter(|row| row.0 == "regression").count(),
            "binary_count": rows.iter().filter(|row| row.0 == "binary").count(),
            "data_sha256": data_sha256,
            "standardization_sha256": "s".repeat(64),
            "regression_source_data_sha256": "r".repeat(64),
            "source_family_disjoint": true,
            "entries": entries,
        });
        let catalog_path = live_root.join("live.catalog.json");
        atomic_json(&catalog_path, &catalog).unwrap();
        let pointer = json!({
            "schema": "JopeTestOnlyContextPointer.v1",
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path).unwrap(),
            "data": data_path,
            "data_sha256": data_sha256,
        });
        atomic_json(&live_root.join("current.json"), &pointer).unwrap();
        pointer
    }

    #[test]
    fn merge_live_without_tranches_is_a_readonly_noop() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let live_root = temp.path().join("live");
        write_live_catalog(
            &live_root,
            &[("regression", "1111b9dad5725a1e", "fam-a", "train")],
        );
        let before = fs::read(live_root.join("current.json")).unwrap();
        let result = merge_live(&paths, &live_root, &[], true).unwrap();
        assert_eq!(result["nothing_to_merge"], true);
        assert_eq!(before, fs::read(live_root.join("current.json")).unwrap());
        assert!(!live_root.join("pointer-history.jsonl").exists());
        assert!(!live_root.join(".merge.lock").exists());
        let result = merge_live(&paths, &live_root, &[], false).unwrap();
        assert_eq!(result["nothing_to_merge"], true);
        assert_eq!(before, fs::read(live_root.join("current.json")).unwrap());
        assert!(!live_root.join("pointer-history.jsonl").exists());
    }

    #[test]
    fn merge_live_appends_tranche_flips_pointer_and_is_idempotent() {
        let temp = TempDir::new().unwrap();
        let id = "feedb9dad5725a1e";
        let (paths, _entry, receipt) = activated_fixture(&temp, id, "tranche-merge");
        let live_root = temp.path().join("live");
        let old_pointer = write_live_catalog(
            &live_root,
            &[("regression", "1111b9dad5725a1e", "fam-a", "train")],
        );
        let old_blob = fs::read(Path::new(old_pointer["data"].as_str().unwrap())).unwrap();
        let summary = merge_live(&paths, &live_root, &[], false).unwrap();
        assert_eq!(summary["nothing_to_merge"], false);
        assert_eq!(summary["row_count"], 2);
        let pointer = read_json(&live_root.join("current.json")).unwrap();
        assert_ne!(pointer["catalog_sha256"], old_pointer["catalog_sha256"]);
        let merged = read_json(Path::new(pointer["catalog"].as_str().unwrap())).unwrap();
        let entries = merged["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        let original = read_json(Path::new(old_pointer["catalog"].as_str().unwrap())).unwrap();
        assert_eq!(entries[0], original["entries"].as_array().unwrap()[0]);
        assert_eq!(entries[1]["current_catalog"], false);
        assert_eq!(entries[1]["dataset_hash"], id);
        assert_eq!(entries[1]["catalog_index"], 1);
        assert_eq!(entries[1]["source_catalog_index"], Value::Null);
        assert!(entries[1]["train_csv"]
            .as_str()
            .unwrap()
            .contains("regression-expanded"));
        let new_blob = fs::read(Path::new(pointer["data"].as_str().unwrap())).unwrap();
        assert_eq!(new_blob.len(), 2 * CONTEXT_STRIDE);
        assert_eq!(&new_blob[..CONTEXT_STRIDE], &old_blob[..]);
        assert_eq!(
            sha256_bytes(&new_blob[CONTEXT_STRIDE..]),
            entries[1]["context_sha256"].as_str().unwrap()
        );
        let tranche_catalog = read_json(Path::new(receipt["catalog"].as_str().unwrap())).unwrap();
        assert_eq!(
            tranche_catalog["entries"][0]["context_sha256"],
            entries[1]["context_sha256"]
        );
        let history = fs::read_to_string(live_root.join("pointer-history.jsonl")).unwrap();
        assert_eq!(history.lines().count(), 1);
        assert!(history.contains(old_pointer["catalog_sha256"].as_str().unwrap()));
        // Re-running after success is a clean no-op (tranche already merged).
        let again = merge_live(&paths, &live_root, &[], false).unwrap();
        assert_eq!(again["nothing_to_merge"], true);
        assert_eq!(again["already_merged_tranches"][0], "tranche-merge");
        assert_eq!(
            fs::read_to_string(live_root.join("pointer-history.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        // Excluding the tranche also reports nothing to merge.
        let excluded = merge_live(&paths, &live_root, &["tranche-merge".to_owned()], true).unwrap();
        assert_eq!(excluded["nothing_to_merge"], true);
    }

    #[test]
    fn merge_live_inherits_v1_family_split() {
        let temp = TempDir::new().unwrap();
        let id = "feedb9dad5725a1e";
        let (paths, _entry, receipt) = activated_fixture(&temp, id, "tranche-split");
        let tranche_catalog = read_json(Path::new(receipt["catalog"].as_str().unwrap())).unwrap();
        let family = tranche_catalog["entries"][0]["source_family"]
            .as_str()
            .unwrap()
            .to_owned();
        let tranche_split = tranche_catalog["entries"][0]["split"].as_str().unwrap();
        let opposite = if tranche_split == "train" {
            "validation"
        } else {
            "train"
        };
        let live_root = temp.path().join("live");
        write_live_catalog(
            &live_root,
            &[("regression", "1111b9dad5725a1e", &family, opposite)],
        );
        let summary = merge_live(&paths, &live_root, &[], false).unwrap();
        assert_eq!(summary["inherited_splits"], 1);
        let pointer = read_json(&live_root.join("current.json")).unwrap();
        let merged = read_json(Path::new(pointer["catalog"].as_str().unwrap())).unwrap();
        assert_eq!(merged["entries"][1]["split"], opposite);
    }

    #[test]
    fn merge_live_rejects_conflicting_identity_and_v1_straddle() {
        let temp = TempDir::new().unwrap();
        let id = "feedb9dad5725a1e";
        let (paths, _entry, _receipt) = activated_fixture(&temp, id, "tranche-dup");
        let live_root = temp.path().join("live");
        write_live_catalog(&live_root, &[("regression", id, "fam-a", "train")]);
        let error = merge_live(&paths, &live_root, &[], true).unwrap_err();
        assert!(error.to_string().contains("different context"));
        let straddle_root = temp.path().join("straddle");
        write_live_catalog(
            &straddle_root,
            &[
                ("regression", "1111b9dad5725a1e", "fam-a", "train"),
                ("regression", "2222b9dad5725a1e", "fam-a", "validation"),
            ],
        );
        let error = merge_live(&paths, &straddle_root, &[], true).unwrap_err();
        assert!(error.to_string().contains("straddles"));
        let error = merge_live(&paths, &live_root, &["missing-id".to_owned()], true).unwrap_err();
        assert!(error.to_string().contains("not in the activation ledger"));
    }

    #[test]
    fn install_sidecars_is_additive_and_fail_closed() {
        let temp = TempDir::new().unwrap();
        let paths = paths(&temp);
        let id = "0aa0b9dad5725a1e";
        let registry = paths.source_root.join("binary").join(id);
        fs::create_dir_all(&registry).unwrap();
        let from = temp.path().join("staged");
        let staged = from.join(id);
        fs::create_dir_all(&staged).unwrap();
        let target_row = (0..512).map(|_| "0.5").collect::<Vec<_>>().join(",");
        fs::write(staged.join("train_target.vs"), format!("{target_row}\n")).unwrap();
        fs::write(staged.join("train3.vs"), "0.25\n").unwrap();
        let receipts = temp.path().join("receipts/sidecars.jsonl");
        let summary = install_sidecars(&paths, &from, "binary", &receipts).unwrap();
        assert_eq!(summary["installed"], 2);
        assert!(registry.join("train_target.vs").is_file());
        let again = install_sidecars(&paths, &from, "binary", &receipts).unwrap();
        assert_eq!(again["installed"], 0);
        assert_eq!(again["skipped_identical"], 2);
        fs::write(staged.join("train3.vs"), "0.75\n").unwrap();
        let error = install_sidecars(&paths, &from, "binary", &receipts).unwrap_err();
        assert!(error.to_string().contains("refusing to overwrite"));
        let lines = fs::read_to_string(&receipts).unwrap();
        assert!(lines.lines().count() >= 5);
        assert!(lines.contains("\"conflict\""));
        fs::write(
            staged.join("train_target.vs"),
            format!("{target_row}\n{target_row}\n"),
        )
        .unwrap();
        fs::remove_file(staged.join("train3.vs")).unwrap();
        let error = install_sidecars(&paths, &from, "binary", &receipts).unwrap_err();
        assert!(error.to_string().contains("1x512"));
        let missing_from = temp.path().join("staged-missing");
        fs::create_dir_all(missing_from.join("9bb0b9dad5725a1e")).unwrap();
        fs::write(
            missing_from.join("9bb0b9dad5725a1e").join("train3.vs"),
            "0.1\n",
        )
        .unwrap();
        let summary = install_sidecars(&paths, &missing_from, "binary", &receipts).unwrap();
        assert_eq!(summary["missing_registry_dir"], 1);
    }
}
