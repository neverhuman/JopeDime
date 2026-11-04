//! Atomic local checkpoint commit and complete-checkpoint loading (W6-T05).
//!
//! A checkpoint is committed in this order; a stop between any two steps
//! leaves the previously published checkpoint as the only complete one:
//!
//! 1. write every required object into a new private staging directory;
//! 2. write the manifest (`state.json`) that binds each object's digest;
//! 3. verify the staged directory exactly as a loader would, so an incomplete
//!    or inconsistent checkpoint is never exposed;
//! 4. fsync every object, the manifest and the staging directory;
//! 5. rename staging to `checkpoints/step-N` and fsync the parent;
//! 6. write a uniquely named temporary `latest.json`, fsync, rename, fsync the
//!    run directory.
//!
//! Unpublished directories are quarantined by `checkpoint_recovery::reconcile`.
//! Temporaries use a per-call unique name, so a temporary left by a killed
//! writer, even one that later reuses the same process id, never blocks
//! a later publication; they are inert and never read.
use crate::manifest::{file_hash, read_json, resolve};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

mod verification;
pub use verification::{load_latest, verify_dir};

pub const SCHEMA: &str = "parallel.checkpoint.v1";
/// Complete-commit format: the manifest also binds rank receipts, the
/// per-rank consumption journals and the list of required objects.
pub const FORMAT: &str = "parallel.checkpoint_commit.v2";
/// Successor format binding immutable lineage and intent identities and each
/// rank's exact durable journal prefix. Legacy and v2 readers remain intact.
pub const SUCCESSOR_FORMAT: &str = "parallel.checkpoint_commit.v3";
/// Objects every checkpoint must contain, with their manifest digest field.
pub const REQUIRED_OBJECTS: [(&str, &str); 4] = [
    ("model.ot", "model_sha256"),
    ("optimizer.ot", "optimizer_sha256"),
    ("optimizer.json", "optimizer_metadata_sha256"),
    ("config.json", "config_sha256"),
];
/// Fields exact resume already required before this format existed.
const RESUME_FIELDS: [&str; 13] = [
    "world",
    "global_batch",
    "corpus_sha256",
    "vocabulary_sha256",
    "seed",
    "sampler_id",
    "initialization",
    "autocast_cache_lifetime",
    "ranking_backward",
    "runtime_objective",
    "execution_partition",
    "config_sha256",
    "optimizer_metadata_sha256",
];
/// Additionally required by the complete format.
const COMPLETE_FIELDS: [&str; 10] = [
    "objective",
    "objective_sha256",
    "parameter_sha256",
    "rng",
    "sampler",
    "scheduler",
    "committed_exposure_by_rank",
    "rank_receipts",
    "required_objects",
    "checkpoint_format",
];

pub fn required_object_names() -> Vec<&'static str> {
    REQUIRED_OBJECTS.iter().map(|(name, _)| *name).collect()
}

/// Publication steps, in order. Tests stop between steps to prove that no
/// interruption exposes a partial checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    ArtifactsWritten,
    ManifestWritten,
    StagingSynced,
    Renamed,
    LatestStaged,
    LatestPublished,
}

pub const PHASES: [Phase; 6] = [
    Phase::ArtifactsWritten,
    Phase::ManifestWritten,
    Phase::StagingSynced,
    Phase::Renamed,
    Phase::LatestStaged,
    Phase::LatestPublished,
];

/// Production hook: never stops.
pub fn no_stop(_: Phase) -> Result<()> {
    Ok(())
}

fn unique_suffix() -> Result<String> {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    Ok(format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_dir(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

/// Replace `path` atomically through a uniquely named temporary.
pub fn replace_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let parent = path
        .parent()
        .context("output requires a parent directory")?;
    let name = path
        .file_name()
        .context("output requires a file name")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.tmp-{}", unique_suffix()?));
    write_new(&temporary, &serde_json::to_vec_pretty(value)?)?;
    fs::rename(&temporary, path)?;
    sync_dir(parent)
}

/// Commit checkpoint `step` of `run`. `write_artifacts` writes every
/// [`REQUIRED_OBJECTS`] entry into the staging directory it is given;
/// `manifest` returns the `state.json` value for that staging directory.
/// Returns the published `latest.json` value.
pub fn commit(
    root: &Path,
    run: &Path,
    step: u64,
    write_artifacts: impl FnOnce(&Path) -> Result<()>,
    manifest: impl FnOnce(&Path) -> Result<Value>,
    stop: &mut dyn FnMut(Phase) -> Result<()>,
) -> Result<Value> {
    commit_fenced(
        root,
        run,
        step,
        write_artifacts,
        manifest,
        &mut || Ok(()),
        stop,
    )
}

/// Commit with a fresh lease check before checkpoint rename and latest
/// publication. The caller must query the active attempt/lease in the callback;
/// a token captured before the write is not a publication fence.
pub fn commit_fenced(
    root: &Path,
    run: &Path,
    step: u64,
    write_artifacts: impl FnOnce(&Path) -> Result<()>,
    manifest: impl FnOnce(&Path) -> Result<Value>,
    fence: &mut dyn FnMut() -> Result<()>,
    stop: &mut dyn FnMut(Phase) -> Result<()>,
) -> Result<Value> {
    let checkpoints = run.join("checkpoints");
    fs::create_dir_all(&checkpoints)?;
    let dest = checkpoints.join(format!("step-{step}"));
    ensure!(!dest.exists(), "checkpoint already exists");
    let staging = checkpoints.join(format!(".step-{step}-{}", unique_suffix()?));
    fs::create_dir(&staging)?;
    write_artifacts(&staging)?;
    stop(Phase::ArtifactsWritten)?;
    let state = manifest(&staging)?;
    ensure!(
        state["step"].as_u64() == Some(step),
        "checkpoint manifest step differs from the committed step"
    );
    write_new(
        &staging.join("state.json"),
        &serde_json::to_vec_pretty(&state)?,
    )?;
    stop(Phase::ManifestWritten)?;
    // Never expose a checkpoint a loader would refuse.
    verify_dir(root, &staging, step, &Expectation::default())?;
    let staged_manifest: Value = read_json(&staging.join("state.json"))?;
    if staged_manifest["checkpoint_format"] == SUCCESSOR_FORMAT {
        sync_successor_closure(root, &staging)?;
    }
    for name in required_object_names().into_iter().chain(["state.json"]) {
        fs::File::open(staging.join(name))?.sync_all()?;
    }
    sync_dir(&staging)?;
    stop(Phase::StagingSynced)?;
    fence().context("writer fence refused checkpoint rename")?;
    fs::rename(&staging, &dest)?;
    sync_dir(&checkpoints)?;
    stop(Phase::Renamed)?;
    let latest = json!({"checkpoint":dest.strip_prefix(root)?,
        "state_sha256":file_hash(&dest.join("state.json"))?,"step":step});
    let temporary = run.join(format!(".latest.json.tmp-{}", unique_suffix()?));
    write_new(&temporary, &serde_json::to_vec_pretty(&latest)?)?;
    stop(Phase::LatestStaged)?;
    fence().context("writer fence refused latest publication")?;
    fs::rename(&temporary, run.join("latest.json"))?;
    sync_dir(run)?;
    stop(Phase::LatestPublished)?;
    Ok(latest)
}

/// Caller-known identities a loaded checkpoint must match.
#[derive(Clone, Debug, Default)]
pub struct Expectation<'a> {
    pub corpus_sha256: Option<&'a str>,
    pub vocabulary_sha256: Option<&'a str>,
    pub world: Option<usize>,
    pub global_batch: Option<usize>,
    /// Refuse checkpoints written before the complete commit format.
    pub require_complete: bool,
    /// Optional caller-pinned successor lineage and intent identities.
    /// Any lineage, intent, attempt or window expectation requires the successor format.
    pub lineage_sha256: Option<&'a str>,
    pub intent_sha256: Option<&'a str>,
    pub attempt_id: Option<&'a str>,
    pub window_id: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Written before [`FORMAT`]: the resume fields are present, but rank
    /// receipts and consumption journals are not bound.
    Legacy,
    Complete,
    /// Successor manifest with lineage, intent and immutable journal closure.
    Successor,
}

#[derive(Clone, Debug)]
pub struct Committed {
    pub dir: PathBuf,
    pub step: u64,
    pub state: Value,
    pub state_sha256: String,
    pub format: Format,
}
