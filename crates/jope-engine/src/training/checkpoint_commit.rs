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

pub const SCHEMA: &str = "parallel.checkpoint.v1";
/// Complete-commit format: the manifest also binds rank receipts, the
/// per-rank consumption journals and the list of required objects.
pub const FORMAT: &str = "parallel.checkpoint_commit.v2";
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
    for name in required_object_names().into_iter().chain(["state.json"]) {
        fs::File::open(staging.join(name))?.sync_all()?;
    }
    sync_dir(&staging)?;
    stop(Phase::StagingSynced)?;
    fs::rename(&staging, &dest)?;
    sync_dir(&checkpoints)?;
    stop(Phase::Renamed)?;
    let latest = json!({"checkpoint":dest.strip_prefix(root)?,
        "state_sha256":file_hash(&dest.join("state.json"))?,"step":step});
    let temporary = run.join(format!(".latest.json.tmp-{}", unique_suffix()?));
    write_new(&temporary, &serde_json::to_vec_pretty(&latest)?)?;
    stop(Phase::LatestStaged)?;
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Written before [`FORMAT`]: the resume fields are present, but rank
    /// receipts and consumption journals are not bound.
    Legacy,
    Complete,
}

#[derive(Clone, Debug)]
pub struct Committed {
    pub dir: PathBuf,
    pub step: u64,
    pub state: Value,
    pub state_sha256: String,
    pub format: Format,
}

fn plain_file(path: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(path)
        .with_context(|| format!("checkpoint object missing: {}", path.display()))?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "checkpoint object is not a regular file: {}",
        path.display()
    );
    Ok(())
}

/// Load the run's published checkpoint and verify it is complete.
pub fn load_latest(root: &Path, run: &Path, expect: &Expectation) -> Result<Committed> {
    let latest_path = run.join("latest.json");
    plain_file(&latest_path)?;
    let latest: Value = read_json(&latest_path)?;
    let step = latest["step"]
        .as_u64()
        .context("published checkpoint step absent")?;
    let dir = resolve(
        root,
        Path::new(
            latest["checkpoint"]
                .as_str()
                .context("published checkpoint path absent")?,
        ),
    )?;
    ensure!(
        dir == run.canonicalize()?.join(format!("checkpoints/step-{step}")),
        "published checkpoint is outside this run or names another step"
    );
    let committed = verify_dir(root, &dir, step, expect)?;
    ensure!(
        latest["state_sha256"] == committed.state_sha256.as_str(),
        "published checkpoint manifest changed"
    );
    Ok(committed)
}

fn field_present(state: &Value, field: &str) -> bool {
    state.get(field).is_some_and(|v| !v.is_null())
}

/// Verify a checkpoint directory as a loader requires it: every object
/// present and bound, the manifest complete and internally consistent, and
/// every caller expectation met. Nothing is loaded into a model.
pub fn verify_dir(root: &Path, dir: &Path, step: u64, expect: &Expectation) -> Result<Committed> {
    let meta = fs::symlink_metadata(dir)
        .with_context(|| format!("checkpoint directory missing: {}", dir.display()))?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "checkpoint directory is not a real directory"
    );
    let state_path = dir.join("state.json");
    plain_file(&state_path)?;
    let state_bytes = fs::read(&state_path)?;
    let state: Value =
        serde_json::from_slice(&state_bytes).context("torn or malformed checkpoint manifest")?;
    ensure!(
        state["schema"] == SCHEMA && state["step"].as_u64() == Some(step),
        "checkpoint manifest schema or step differs"
    );
    for field in RESUME_FIELDS {
        ensure!(
            field_present(&state, field),
            "checkpoint manifest lacks {field}"
        );
    }
    for (name, field) in REQUIRED_OBJECTS {
        let path = dir.join(name);
        plain_file(&path)?;
        ensure!(
            state[field].is_string() && file_hash(&path)? == state[field],
            "checkpoint object {name} differs from its manifest digest"
        );
    }
    let optimizer: Value = read_json(&dir.join("optimizer.json"))?;
    ensure!(
        optimizer["step"].as_u64() == Some(step)
            && optimizer["sha256"] == state["optimizer_sha256"]
            && optimizer["lr"]
                .as_f64()
                .is_some_and(|lr| lr.is_finite() && lr > 0.0)
            && optimizer["decay"]
                .as_f64()
                .is_some_and(|decay| decay.is_finite() && decay >= 0.0)
            && optimizer["backend"].is_string(),
        "optimizer metadata is inconsistent with the manifest"
    );
    let _: crate::neural::Config =
        read_json(&dir.join("config.json")).context("checkpoint configuration invalid")?;
    ensure!(
        state["seed"].is_u64() && state["sampler_id"].as_str().is_some_and(|s| !s.is_empty()),
        "checkpoint RNG or sampler identity absent"
    );
    let world = usize::try_from(state["world"].as_u64().context("checkpoint world")?)?;
    let global = usize::try_from(
        state["global_batch"]
            .as_u64()
            .context("checkpoint global batch")?,
    )?;
    let micro = usize::try_from(
        state["execution_partition"]["microbatch"]
            .as_u64()
            .context("checkpoint microbatch")?,
    )?;
    // W7-T06: the record's own schema selects v1 or v2; v1 is recomputed
    // exactly as before.
    let partition_schema = state["execution_partition"]["schema"]
        .as_str()
        .context("checkpoint execution partition schema")?;
    ensure!(
        state["execution_partition"]
            == crate::task_contract::training_partition_of(partition_schema, world, micro, global)?,
        "checkpoint execution partition is inconsistent"
    );
    let v2 = partition_schema == crate::task_contract::PARTITION_V2_SCHEMA;
    let rows_per_rank =
        crate::task_contract::partition_rows_per_rank(&state["execution_partition"])?;
    for (expected, field) in [
        (expect.corpus_sha256, "corpus_sha256"),
        (expect.vocabulary_sha256, "vocabulary_sha256"),
    ] {
        if let Some(expected) = expected {
            ensure!(
                state[field] == expected,
                "checkpoint {field} differs from the expected input"
            );
        }
    }
    ensure!(
        expect.world.is_none_or(|w| w == world) && expect.global_batch.is_none_or(|g| g == global),
        "checkpoint topology differs from the expected run"
    );
    if field_present(&state, "objective_sha256") {
        ensure!(
            state["objective_sha256"].as_str()
                == Some(super::objective::identity_digest(&state["objective"]).as_str()),
            "checkpoint objective identity does not match its digest"
        );
    }
    if state.get("committed_exposure_by_rank").is_some() {
        for rank in 0..world {
            crate::training_exposure::checkpoint_baseline_rows(&state, rank, &rows_per_rank)?;
        }
    }
    let plan = if v2 {
        let plan = super::partition::Plan::new(world, global, micro)?;
        verify_partition_state(&state, step, &plan)?;
        Some(plan)
    } else {
        None
    };
    let format = if field_present(&state, "checkpoint_format") {
        ensure!(
            state["checkpoint_format"] == FORMAT,
            "unknown checkpoint format"
        );
        verify_complete(root, &state, step, world, global, plan.as_ref())?;
        Format::Complete
    } else {
        ensure!(
            !v2,
            "a partition-v2 checkpoint must use the complete commit format"
        );
        ensure!(
            !expect.require_complete,
            "checkpoint predates the complete commit format"
        );
        Format::Legacy
    };
    Ok(Committed {
        dir: dir.to_path_buf(),
        step,
        state_sha256: crate::manifest::hash(&state_bytes),
        state,
        format,
    })
}

/// A partition-v2 checkpoint binds the v2 objective and its persisted
/// partition state: every rank's RNG and sampler state and the global
/// committed update, re-derived from the plan.
fn verify_partition_state(state: &Value, step: u64, plan: &super::partition::Plan) -> Result<()> {
    ensure!(
        state["objective"]["normalization"] == super::partition::NORMALIZATION,
        "a partition-v2 checkpoint must bind the globally normalized objective"
    );
    let persisted: super::partition::State = serde_json::from_value(
        state
            .get("partition_state")
            .filter(|v| !v.is_null())
            .context("partition-v2 checkpoint lacks its partition state")?
            .clone(),
    )
    .context("partition state malformed")?;
    persisted.verify(plan)?;
    ensure!(
        persisted.committed_step == step
            && Some(persisted.seed) == state["seed"].as_u64()
            && Some(persisted.sampling_offset)
                == Some(
                    state["initialization"]["optimizer_parent_step"]
                        .as_u64()
                        .unwrap_or(0)
                )
            && state["parameter_sha256"] == persisted.committed_update.parameter_sha256.as_str(),
        "partition state differs from the checkpoint it is bound to"
    );
    Ok(())
}

fn verify_complete(
    root: &Path,
    state: &Value,
    step: u64,
    world: usize,
    global: usize,
    plan: Option<&super::partition::Plan>,
) -> Result<()> {
    for field in COMPLETE_FIELDS {
        ensure!(
            field_present(state, field),
            "complete checkpoint manifest lacks {field}"
        );
    }
    ensure!(
        state["required_objects"] == json!(required_object_names()),
        "checkpoint required-object list differs"
    );
    ensure!(
        state["rng"].as_str().is_some_and(|s| !s.is_empty())
            && state["sampler"].as_str().is_some_and(|s| !s.is_empty()),
        "checkpoint RNG or sampler identity absent"
    );
    let receipts = state["rank_receipts"]
        .as_array()
        .context("rank receipts must be a list")?;
    let exposure = state["committed_exposure_by_rank"]
        .as_array()
        .context("committed exposure must be a list")?;
    ensure!(
        receipts.len() == world && exposure.len() == world,
        "checkpoint does not bind every rank"
    );
    for rank in 0..world {
        let entry = &receipts[rank];
        ensure!(
            entry["rank"].as_u64() == Some(rank as u64),
            "rank receipt order differs"
        );
        let path = resolve(
            root,
            Path::new(entry["path"].as_str().context("rank receipt path")?),
        )?;
        plain_file(&path)?;
        ensure!(
            file_hash(&path)? == entry["sha256"],
            "rank {rank} receipt changed"
        );
        let receipt: Value = read_json(&path)?;
        ensure!(
            receipt["rank"].as_u64() == Some(rank as u64)
                && receipt["step"].as_u64() == Some(step)
                && receipt["parameter_sha256"] == state["parameter_sha256"],
            "rank {rank} receipt does not match the committed update"
        );
        let journal = &exposure[rank]["consumption_journal"];
        ensure!(
            journal == &receipt["consumption_journal"],
            "rank {rank} consumption binding differs from its receipt"
        );
        let binding: super::consumption::Binding =
            serde_json::from_value(journal["binding"].clone())
                .context("rank consumption binding absent")?;
        let bytes = fs::read(resolve(
            root,
            Path::new(
                journal["path"]
                    .as_str()
                    .context("consumption journal path")?,
            ),
        )?)?;
        super::consumption::verify_binding(&bytes, &binding)?;
        let consumed = super::consumption::reconcile(&bytes[..binding.bytes as usize], Some(step))?;
        ensure!(
            consumed.committed_through == step
                && consumed.header.rank == rank
                && consumed.header.world == world
                && consumed.header.global_batch == global
                && Some(consumed.header.seed) == state["seed"].as_u64()
                && state["corpus_sha256"] == consumed.header.corpus_sha256.as_str(),
            "rank {rank} consumption journal does not reach the committed update"
        );
        // Partition v2: the journal ran exactly this rank's shard of the
        // checkpoint's plan; a v1 journal never binds a v2 checkpoint.
        ensure!(
            consumed.shard.as_ref() == plan.map(|p| &p.shards()[rank])
                && plan.is_none_or(|p| consumed.header.microbatch == p.microbatch_rows()),
            "rank {rank} consumption journal ran another partition"
        );
    }
    Ok(())
}
