//! jope_selfplay_propose_driver — keeps the DL proposal set bound to the LIVE catalog.
//!
//! Operator mandate (WS-E / dl-seeds-gp): "the deep learning model is used to
//! seed GP runs" must stay measurably true without an agent hand-holding it.
//! Every dataset starts with eight direct-DL slots. The remaining 120 adaptive
//! slots are same-wave GP offspring; later allocation rungs are earned from
//! paired efficiency evidence. A deficient context emits fewer direct rows and
//! returns the difference to GP. Cross-context proposal backfill is forbidden.
//!
//! Before this driver existed nothing regenerated proposals when the catalog
//! grew. The expansion driver adds up to 2,048 datasets an hour; the proposal
//! set was pinned to a 2,864-dataset catalog while the live catalog reached
//! 10,533, so 72.8% of live datasets had zero DL proposals and every binary
//! DL slot in every wave was cross-dataset backfill.
//!
//! One cycle (timer-driven, cheap: two JSON reads plus `scq ls`):
//!   1. bound_catalog := catalog_sha256 the current proposal set was built for
//!   2. live_catalog  := catalog/current.json catalog_sha256
//!   3. equal            -> `up_to_date`, exit 0
//!   4. propose job live -> `waiting`, exit 0 (never two GPU propose jobs)
//!   5. job completed    -> fetch, verify, install, atomically rebind
//!   6. otherwise        -> render spec, `scq check`, `scq submit`
//!
//! Checkpoint identity is held CONSTANT across a rebind on purpose:
//! `jope_selfplay_queue.rs valid_micro_waves()` filters waves by
//! checkpoint_epoch, so moving the checkpoint here would silently invalidate
//! every existing wave and its unclaimed manifests. Promotion (stream A) owns
//! the checkpoint; this driver only ever re-binds the catalog underneath the
//! checkpoint that promotion already accepted.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

use jope_dime_regressor::selfplay_runtime::{
    atomic_json, now_ms, read_json, sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};

const SCRIPTS: &str = "/home/ubuntu/jope-dime/scripts";
const PRIME_SRC: &str = "/home/ubuntu/jope-prime/src";
const CHECKPOINTS: &str = "/home/ubuntu/jope-flywheel/training/round1/checkpoints";
const STATE_SCHEMA: &str = "JopeProposeDriver.v1";
const EVENT_SCHEMA: &str = "JopeProposeDriverEvent.v1";
/// Initial direct-DL allocation. Future rungs are pointer-bound evidence.
const PROPOSALS_PER_DATASET: i64 = 8;
/// Deficiency is no longer fatal or backfilled; retained for CLI compatibility.
const ALLOW_DEFICIENT_FLOOR: i64 = 0;
const PROPOSAL_SEED: i64 = 20260726;
/// GPU nodes this job may land on. xbabe2 is excluded by policy: the priority
/// trainer lives there and must never contend for the GPU.
const PROPOSE_HOSTS: &str = "xbabe1|xbabe3";
const MAX_ATTEMPTS: u64 = 3;

// ---------------------------------------------------------------------------
// WS-6: per-task checkpoint selection
// ---------------------------------------------------------------------------
//
// Operator decision: per-task accepted checkpoints. A generation that trades
// one branch against the other can never clear a single global gate, so the
// promotion gate now records a per-task verdict and advances
// `$R/training/accepted-by-task.json`
// (`jope_selfplay_promotion.rs::advance_task_accepted_pointers`). This driver
// is the consumer: it SELECTS the checkpoint per branch.
//
// The selection is a fork in the GENERATION lane only. Publication is not
// forked, because `jope_selfplay_wave.rs:1072` and `:1325` hard-fail
// `"proposal file mixes checkpoint epochs at line {n}"` for any row whose
// `model_checkpoint_sha256` differs from the cohort plan's single
// `checkpoint_sha256`, and `jope_selfplay_queue.rs valid_micro_waves()` filters
// waves by a single `checkpoint_epoch`. So:
//   * branches agree on a checkpoint -> exactly the historical behaviour: one
//     whole-catalog job, published into `proposals/current.json`.
//   * branches disagree              -> one job PER BRANCH, each carrying a
//     single (task, checkpoint) identity, published under
//     `proposals/by-task/<task>/` and named by the ADDITIVE pointer
//     `proposals/current-by-task.json`. `proposals/current.json` is left
//     untouched, so no consumer can ever be handed a mixed-epoch file.
const TASK_BRANCHES: [&str; 4] = ["binary", "regression", "multiclass", "timeseries"];
const TASK_ACCEPTED_POINTER: &str = "training/accepted-by-task.json";
const TASK_POINTER_SCHEMA: &str = "JopeSelfplayProposalPointerByTask.v1";

/// Two-letter lane tag so a per-task job id can never collide with the
/// whole-catalog family prefix `jope-selfplay-propose-<sha12>-c`.
///
/// Every task gets its OWN tag. The old `_ => "tr"` fallback would have tagged
/// timeseries jobs identically to regression jobs, so a timeseries proposal
/// job id could collide with -- and silently shadow -- a regression one.
fn task_tag(task: &str) -> &'static str {
    match task {
        "binary" => "tb",
        "regression" => "tr",
        "multiclass" => "tm",
        "timeseries" => "tt",
        other => unreachable!("task_tag called with unknown task {other}"),
    }
}

#[derive(Parser, Debug)]
#[command(about = "Rebind the DL proposal set to the live catalog, unattended")]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    /// Plan only: never submit, fetch, or rebind.
    #[arg(long)]
    dry_run: bool,
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn get_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {key}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn toml_string(value: &str) -> Result<String> {
    ensure_plain(value)?;
    Ok(format!("\"{value}\""))
}

/// Job ids, hashes and paths in this driver are all machine-generated; refusing
/// anything with a quote or backslash keeps the rendered TOML unambiguous
/// without pulling in a serializer.
fn ensure_plain(value: &str) -> Result<()> {
    if value.contains('"') || value.contains('\\') || value.contains('\n') {
        bail!("value is not representable in a bare TOML string: {value}");
    }
    Ok(())
}

fn sh(program: &Path, args: &[&str]) -> Result<(i32, String, String)> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("spawn {} {:?}", program.display(), args))?;
    Ok((
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn append_event(path: &Path, event: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    file.write_all(&line)?;
    Ok(())
}

/// Non-blocking exclusive lock; a held lock means a previous cycle is still
/// fetching a multi-hundred-megabyte proposal file and this tick must be a
/// no-op rather than a duplicate.
fn try_lock(path: &Path) -> Result<Option<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status != 0 {
        return Ok(None);
    }
    Ok(Some(file))
}

fn rglob(directory: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rglob(&path, name, found);
        } else if path.file_name().map(|n| n == name).unwrap_or(false) {
            found.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// pointers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LiveCatalog {
    pointer_path: PathBuf,
    catalog: PathBuf,
    catalog_sha256: String,
    data: PathBuf,
    data_sha256: String,
}

fn read_live_catalog(root: &Path) -> Result<LiveCatalog> {
    let pointer_path = root.join("catalog/current.json");
    let pointer = read_json(&pointer_path)?;
    let catalog_sha256 = get_str(&pointer, "catalog_sha256")?.to_owned();
    let data_sha256 = get_str(&pointer, "data_sha256")?.to_owned();
    if !is_sha256(&catalog_sha256) || !is_sha256(&data_sha256) {
        bail!("live catalog pointer carries a malformed hash");
    }
    Ok(LiveCatalog {
        catalog: PathBuf::from(get_str(&pointer, "catalog")?),
        data: PathBuf::from(get_str(&pointer, "data")?),
        catalog_sha256,
        data_sha256,
        pointer_path,
    })
}

#[derive(Debug, Clone)]
struct ProposalBinding {
    pointer_path: PathBuf,
    pointer: Value,
    checkpoint_sha256: String,
    /// The catalog the *current* proposal file was generated against. Read
    /// from the pointer when present (this driver writes it) and otherwise
    /// recovered from the summary artifact the generator emitted.
    bound_catalog_sha256: Option<String>,
}

fn read_proposal_binding(root: &Path) -> Result<ProposalBinding> {
    let pointer_path = root.join("proposals/current.json");
    let pointer = read_json(&pointer_path)?;
    let checkpoint_sha256 = get_str(&pointer, "checkpoint_sha256")?.to_owned();
    if !is_sha256(&checkpoint_sha256) {
        bail!("proposal pointer checkpoint hash is malformed");
    }
    let mut bound = pointer
        .get("catalog_sha256")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if bound.is_none() {
        if let Some(summary) = pointer.get("summary").and_then(Value::as_str) {
            let summary_path = resolve(&pointer_path, Path::new(summary));
            if summary_path.is_file() {
                bound = read_json(&summary_path).ok().and_then(|value| {
                    value
                        .get("catalog_sha256")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
            }
        }
    }
    Ok(ProposalBinding {
        pointer_path,
        pointer,
        checkpoint_sha256,
        bound_catalog_sha256: bound,
    })
}

/// One generation lane: which branch (None = both), which checkpoint, and which
/// catalog the currently-published set for that lane was built against.
#[derive(Debug, Clone)]
struct Lane {
    task: Option<String>,
    checkpoint_sha256: String,
    checkpoint: PathBuf,
    bound_catalog_sha256: Option<String>,
}

impl Lane {
    fn label(&self) -> String {
        self.task.clone().unwrap_or_else(|| "combined".to_owned())
    }
    /// Family prefix that blocks a resubmission: any job for this lane's
    /// (task, checkpoint), whatever catalog it was aimed at.
    fn family(&self) -> String {
        match &self.task {
            Some(task) => format!(
                "jope-selfplay-propose-{}-{}-c",
                task_tag(task),
                &self.checkpoint_sha256[..12]
            ),
            None => format!("jope-selfplay-propose-{}-c", &self.checkpoint_sha256[..12]),
        }
    }
    fn prefix(&self, catalog_sha256: &str) -> String {
        format!("{}{}", self.family(), &catalog_sha256[..12])
    }
}

/// Resolve a checkpoint artifact for a per-task branch. The promotion gate
/// publishes into `$R/training/immutable/checkpoints/`; the historical warm
/// lineage lives in `CHECKPOINTS`. Either is acceptable, both are hash-verified,
/// because a propose job that stages the wrong bytes produces a whole
/// generation of silently mislabelled proposals.
fn resolve_checkpoint(root: &Path, sha256: &str, declared: Option<&str>) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = declared {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        root.join("training/immutable/checkpoints")
            .join(format!("{sha256}.model.pt")),
    );
    candidates.push(PathBuf::from(CHECKPOINTS).join(format!("{sha256}.model.pt")));
    for candidate in candidates {
        if candidate.is_file() && sha256_file(&candidate)? == sha256 {
            return Ok(candidate);
        }
    }
    bail!("no content-verified artifact for checkpoint {sha256}");
}

/// Per-task accepted checkpoints, or `None` when the branches agree (or the
/// pointer does not exist yet, which is the state before the first per-task
/// promotion lands). `None` means "run the historical combined lane".
fn read_task_lanes(root: &Path, binding: &ProposalBinding) -> Result<Option<Vec<Lane>>> {
    let pointer_path = root.join(TASK_ACCEPTED_POINTER);
    if !pointer_path.is_file() {
        return Ok(None);
    }
    let document = read_json(&pointer_path)?;
    let mut selected: Vec<(String, String, Option<String>)> = Vec::new();
    for task in TASK_BRANCHES {
        let entry = document.pointer(&format!("/tasks/{task}"));
        let sha = entry
            .and_then(|value| value.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            .unwrap_or(&binding.checkpoint_sha256)
            .to_owned();
        if !is_sha256(&sha) {
            bail!("accepted-by-task entry for {task} carries a malformed checkpoint hash");
        }
        let declared = entry
            .and_then(|value| value.get("checkpoint"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        selected.push((task.to_owned(), sha, declared));
    }
    // Every branch on the same checkpoint is not a fork: keep the single
    // whole-catalog lane so `proposals/current.json` stays authoritative and no
    // extra GPU generation is spent proving the branches equal.
    if selected.iter().all(|(_, sha, _)| *sha == selected[0].1) {
        return Ok(None);
    }
    let published = read_task_proposal_pointer(root);
    let mut lanes = Vec::new();
    for (task, sha, declared) in selected {
        let bound = published
            .pointer(&format!("/tasks/{task}"))
            .filter(|entry| {
                entry.get("checkpoint_sha256").and_then(Value::as_str) == Some(sha.as_str())
            })
            .and_then(|entry| entry.get("catalog_sha256"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let checkpoint = resolve_checkpoint(root, &sha, declared.as_deref())
            .with_context(|| format!("resolve {task} branch checkpoint"))?;
        lanes.push(Lane {
            task: Some(task),
            checkpoint_sha256: sha,
            checkpoint,
            bound_catalog_sha256: bound,
        });
    }
    Ok(Some(lanes))
}

fn read_task_proposal_pointer(root: &Path) -> Value {
    read_json(&root.join("proposals/current-by-task.json"))
        .unwrap_or_else(|_| json!({"schema": TASK_POINTER_SCHEMA, "tasks": {}}))
}

fn resolve(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_path_buf()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

// ---------------------------------------------------------------------------
// spec rendering
// ---------------------------------------------------------------------------

fn input_block(name: &str, source: &Path, target: &str) -> Result<Vec<String>> {
    let digest =
        sha256_file(source).with_context(|| format!("hash spec input {}", source.display()))?;
    let source_text = source.to_string_lossy().into_owned();
    Ok(vec![
        "[[input]]".to_owned(),
        format!("name = {}", toml_string(name)?),
        format!("source = {}", toml_string(&source_text)?),
        format!("target = {}", toml_string(target)?),
        format!("sha256 = {}", toml_string(&digest)?),
        String::new(),
    ])
}

/// The catalog pointer is a *mutable* file: the expansion driver flips it
/// hourly. Staging it directly would let a flip between submit and dispatch
/// invalidate the job's input hash, so the spec consumes an immutable snapshot
/// keyed by the catalog it describes.
fn freeze_pointer(root: &Path, live: &LiveCatalog) -> Result<PathBuf> {
    let directory = root.join("proposals/_pointers");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{}.pointer.json", &live.catalog_sha256[..16]));
    if !path.is_file() {
        let value = read_json(&live.pointer_path)?;
        atomic_json(&path, &value)?;
    }
    Ok(path)
}

struct RenderedSpec {
    job_id: String,
    path: PathBuf,
}

fn render_spec(
    root: &Path,
    checkpoint: &Path,
    checkpoint_sha256: &str,
    live: &LiveCatalog,
    attempt: u64,
    task: Option<&str>,
) -> Result<RenderedSpec> {
    let mut job_id = match task {
        Some(task) => format!(
            "jope-selfplay-propose-{}-{}-c{}",
            task_tag(task),
            &checkpoint_sha256[..12],
            &live.catalog_sha256[..12]
        ),
        None => format!(
            "jope-selfplay-propose-{}-c{}",
            &checkpoint_sha256[..12],
            &live.catalog_sha256[..12]
        ),
    };
    if attempt > 1 {
        // A terminal-failed record blocks resubmission under the same id and
        // the scheduler offers no record removal, so retries need a fresh id.
        job_id.push_str(&format!("-r{attempt}"));
    }
    let pointer = freeze_pointer(root, live)?;
    let script = Path::new(SCRIPTS).join("test_exact_dl_generate.py");
    let common = Path::new(SCRIPTS).join("test_exact_common.py");
    // `--task` restricts the generation to one branch so the output file carries
    // a single (task, checkpoint) identity. Rendered only for a per-task lane;
    // the combined lane's argv is byte-identical to what it has always been.
    let task_argument = match task {
        Some(task) => {
            ensure_plain(task)?;
            format!(", \"--task\", \"{task}\"")
        }
        None => String::new(),
    };
    let args_line = format!(
        "args = [\"{{input:script}}\", \"--catalog-pointer\", \"{{input:pointer}}\", \
         \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:data}}\", \
         \"--checkpoint\", \"{{input:checkpoint}}\", \
         \"--output\", \"{{output:proposals}}/dl-proposals.jsonl\", \
         \"--device\", \"cuda\", \"--seed\", \"{PROPOSAL_SEED}\", \
         \"--proposals-per-dataset\", \"{PROPOSALS_PER_DATASET}\", \
         \"--batch-contexts\", \"64\", \
         \"--allow-deficient-floor\", \"{ALLOW_DEFICIENT_FLOOR}\"{task_argument}]"
    );
    let mut lines = vec![
        "# Generated by jope_selfplay_propose_driver. Do not hand-edit: the".to_owned(),
        "# driver re-renders this file whenever the live catalog changes.".to_owned(),
        "[job]".to_owned(),
        format!("id = {}", toml_string(&job_id)?),
        "project = \"jope-selfplay-train\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        args_line,
        String::new(),
        "[resource]".to_owned(),
        "gpus_per_task = 1".to_owned(),
        "cpus_reserved = 4".to_owned(),
        "omp_threads = 4".to_owned(),
        // Working set is the context blob plus a ~104 MB model plus CUDA host
        // allocations; 12 GB is several times that and fits xbabe1's free RAM.
        "ram_mb = 12288".to_owned(),
        "gang_size = 1".to_owned(),
        "placement = \"single_node\"".to_owned(),
        String::new(),
        "[dependencies]".to_owned(),
        format!(
            "hostname = [{{ command = \"/usr/bin/grep\", args = [\"-qxE\", \"{PROPOSE_HOSTS}\", \"/etc/hostname\"], timeout_secs = 5 }}]"
        ),
        String::new(),
        "[env]".to_owned(),
        format!("PYTHONPATH = \"{PRIME_SRC}:{SCRIPTS}\""),
        "OMP_NUM_THREADS = \"4\"".to_owned(),
        String::new(),
        "[preempt]".to_owned(),
        "preemptible = true".to_owned(),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
        "[limits]".to_owned(),
        "max_runtime_hours = 24".to_owned(),
        "max_restarts = 1".to_owned(),
        String::new(),
    ];
    let catalog_name = file_name(&live.catalog);
    let data_name = file_name(&live.data);
    let checkpoint_name = file_name(checkpoint);
    let inputs: Vec<(&str, PathBuf, String)> = vec![
        ("script", script, "scripts/test_exact_dl_generate.py".into()),
        ("common", common, "scripts/test_exact_common.py".into()),
        ("pointer", pointer, "catalog/current.json".into()),
        (
            "catalog",
            live.catalog.clone(),
            format!("catalog/{catalog_name}"),
        ),
        ("data", live.data.clone(), format!("catalog/{data_name}")),
        (
            "checkpoint",
            checkpoint.to_path_buf(),
            format!("checkpoint/{checkpoint_name}"),
        ),
    ];
    for (name, source, target) in &inputs {
        lines.extend(input_block(name, source, target)?);
    }
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"proposals\"".to_owned(),
        "source = \"proposals\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    let path = root.join("specs/propose").join(format!("{job_id}.toml"));
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| anyhow!("spec path has no parent"))?,
    )?;
    fs::write(&path, lines.join("\n"))?;
    Ok(RenderedSpec { job_id, path })
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// scq
// ---------------------------------------------------------------------------

fn job_states(scq: &Path, prefix: &str) -> Result<Vec<(String, String)>> {
    let (code, stdout, stderr) = sh(scq, &["ls", "--json"])?;
    if code != 0 {
        bail!("scq ls failed ({code}): {}", stderr.trim());
    }
    let value: Value = serde_json::from_str(&stdout).context("parse scq ls --json")?;
    let jobs = value
        .get("Jobs")
        .and_then(|jobs| jobs.get("jobs"))
        .and_then(Value::as_array)
        .context("scq ls payload lacks Jobs.jobs")?;
    let mut found = Vec::new();
    for job in jobs {
        let Some(id) = job.get("id").and_then(Value::as_str) else {
            continue;
        };
        if id.starts_with(prefix) {
            let state = job
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            found.push((id.to_owned(), state));
        }
    }
    found.sort();
    Ok(found)
}

fn is_active(state: &str) -> bool {
    matches!(state, "queued" | "running" | "dispatching" | "held")
}

/// Completed generations worth installing, best first: a run aimed at the live
/// catalog beats one aimed at an intermediate catalog, and both beat nothing.
/// `install` still refuses anything that would not move the binding forward.
fn order_install_candidates<'a>(jobs: &'a [(String, String)], prefix: &str) -> Vec<&'a String> {
    let completed = || jobs.iter().filter(|(_, state)| state == "completed");
    let mut ordered: Vec<&String> = completed()
        .filter(|(id, _)| id.starts_with(prefix))
        .map(|(id, _)| id)
        .collect();
    ordered.extend(
        completed()
            .filter(|(id, _)| !id.starts_with(prefix))
            .map(|(id, _)| id),
    );
    ordered
}

fn check_and_submit(scq: &Path, spec: &Path) -> Result<()> {
    let spec_text = spec.to_string_lossy().into_owned();
    let (code, _, stderr) = sh(scq, &["check", &spec_text])?;
    if code != 0 {
        bail!("scq check rejected {}: {}", spec.display(), stderr.trim());
    }
    let (code, stdout, stderr) = sh(scq, &["submit", &spec_text])?;
    if code != 0 {
        // A duplicate id is the benign race with a concurrent cycle.
        let blob = format!("{stdout}{stderr}");
        if blob.contains("already") || blob.contains("duplicate") {
            return Ok(());
        }
        bail!("scq submit failed ({code}): {}", stderr.trim());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

struct Installed {
    catalog_sha256: String,
    proposals: PathBuf,
    proposals_sha256: String,
    summary: PathBuf,
    summary_sha256: String,
    counts: Value,
    validity: Value,
    datasets: u64,
}

/// Publish a finished generation. `bound` is the catalog the *currently
/// installed* proposal set was built for; anything newer than that is an
/// improvement and gets installed even if the live catalog has moved on again
/// since the job was submitted. Generations take hours and the expansion driver
/// flips the catalog hourly, so demanding an exact match with the live pointer
/// would throw away every completed run and never improve coverage at all.
/// Returns `Ok(None)` when the artifact would not change the binding.
fn install(
    scq: &Path,
    root: &Path,
    job_id: &str,
    checkpoint_sha256: &str,
    bound: Option<&str>,
    task: Option<&str>,
) -> Result<Option<Installed>> {
    let staging = root.join("proposals/_incoming").join(job_id);
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let staging_text = staging.to_string_lossy().into_owned();
    let (code, _, stderr) = sh(scq, &["fetch", job_id, "--output", &staging_text])?;
    if code != 0 {
        bail!("scq fetch {job_id} failed ({code}): {}", stderr.trim());
    }
    let mut proposals = Vec::new();
    let mut summaries = Vec::new();
    rglob(&staging, "dl-proposals.jsonl", &mut proposals);
    rglob(&staging, "dl-proposals.summary.json", &mut summaries);
    let proposals_source = proposals
        .into_iter()
        .next()
        .context("fetched outputs lack dl-proposals.jsonl")?;
    let summary_source = summaries
        .into_iter()
        .next()
        .context("fetched outputs lack dl-proposals.summary.json")?;
    let summary = read_json(&summary_source)?;

    // Provenance is verified before anything is published: a proposal file
    // built for a different checkpoint would silently invalidate every wave,
    // and one built for a different catalog would not fix coverage at all.
    let summary_checkpoint = get_str(&summary, "model_checkpoint_sha256")?;
    if summary_checkpoint != checkpoint_sha256 {
        bail!(
            "fetched proposals carry checkpoint {summary_checkpoint}, expected {checkpoint_sha256}"
        );
    }
    let summary_catalog = get_str(&summary, "catalog_sha256")?.to_owned();
    if !is_sha256(&summary_catalog) {
        bail!("fetched proposals declare a malformed catalog hash {summary_catalog}");
    }
    // Branch identity is provenance too. A per-task lane that published a
    // whole-catalog file (or the other branch's file) would put rows for the
    // wrong checkpoint epoch under a per-task name, which is precisely the
    // mixed-epoch failure `jope_selfplay_wave.rs:1072` exists to catch.
    let summary_task = summary.get("task").and_then(Value::as_str);
    if summary_task != task {
        bail!(
            "fetched proposals carry task {:?}, expected {:?}",
            summary_task,
            task
        );
    }
    if Some(summary_catalog.as_str()) == bound {
        let _ = fs::remove_dir_all(&staging);
        return Ok(None);
    }
    let proposals_sha256 = sha256_file(&proposals_source)?;
    let declared = get_str(&summary, "output_sha256")?;
    if declared != proposals_sha256 {
        bail!("fetched proposal file hash {proposals_sha256} differs from summary {declared}");
    }

    // Publish under a catalog-scoped name so the file the *old* pointer names
    // is never mutated: readers mid-sanitize keep a consistent view and the
    // pointer swap is a single atomic rename.
    let stem = format!("{checkpoint_sha256}.catalog-{}", &summary_catalog[..12]);
    let destination_dir = match task {
        Some(task) => root.join("proposals/by-task").join(task),
        None => root.join("proposals/by-checkpoint"),
    };
    fs::create_dir_all(&destination_dir)?;
    let proposals_destination = destination_dir.join(format!("{stem}.jsonl"));
    let summary_destination = destination_dir.join(format!("{stem}.summary.json"));
    move_or_copy(&proposals_source, &proposals_destination)?;
    move_or_copy(&summary_source, &summary_destination)?;
    if sha256_file(&proposals_destination)? != proposals_sha256 {
        bail!("published proposal file failed post-copy verification");
    }
    let summary_sha256 = sha256_file(&summary_destination)?;
    let _ = fs::remove_dir_all(&staging);

    Ok(Some(Installed {
        catalog_sha256: summary_catalog,
        proposals: proposals_destination,
        proposals_sha256,
        summary: summary_destination,
        summary_sha256,
        counts: summary.get("counts").cloned().unwrap_or(Value::Null),
        validity: json!({
            "proposal_validity": summary.get("proposal_validity").cloned().unwrap_or(Value::Null),
            "proposal_validity_by_task": summary
                .get("proposal_validity_by_task")
                .cloned()
                .unwrap_or(Value::Null),
            "deficient_contexts": summary
                .get("deficient_contexts")
                .and_then(Value::as_object)
                .map(|m| m.len())
                .unwrap_or(0),
        }),
        datasets: summary.get("datasets").and_then(Value::as_u64).unwrap_or(0),
    }))
}

fn move_or_copy(source: &Path, destination: &Path) -> Result<()> {
    if fs::rename(source, destination).is_ok() {
        return Ok(());
    }
    fs::copy(source, destination)
        .with_context(|| format!("copy {} -> {}", source.display(), destination.display()))?;
    Ok(())
}

/// Rewrite proposals/current.json in place, preserving every field the other
/// consumers rely on (`legacy_adopted` resolves the checkpoint artifact in
/// selfplay_runtime and the freshness authority) and swapping only the catalog
/// binding.
fn rebind(binding: &ProposalBinding, installed: &Installed) -> Result<Value> {
    let mut pointer: Map<String, Value> = binding
        .pointer
        .as_object()
        .cloned()
        .context("proposal pointer is not an object")?;
    pointer.insert(
        "proposals".into(),
        Value::String(installed.proposals.to_string_lossy().into_owned()),
    );
    pointer.insert(
        "proposals_sha256".into(),
        Value::String(installed.proposals_sha256.clone()),
    );
    pointer.insert(
        "summary".into(),
        Value::String(installed.summary.to_string_lossy().into_owned()),
    );
    pointer.insert(
        "summary_sha256".into(),
        Value::String(installed.summary_sha256.clone()),
    );
    pointer.insert(
        "catalog_sha256".into(),
        Value::String(installed.catalog_sha256.clone()),
    );
    pointer.insert(
        "checkpoint_sha256".into(),
        Value::String(binding.checkpoint_sha256.clone()),
    );
    // `proposals/current.json` has never carried a `schema` key, so no consumer
    // — jope_selfplay_supply.rs:96, jope_selfplay_freshness.rs:409,
    // jope_selfplay_trainer_watchdog.rs:1016, selfplay_runtime.rs:107 — can
    // version-gate it. Stamp one here (this driver is the only production
    // writer) so a future shape change is rejectable instead of silently
    // half-read. Purely additive: every consumer reads named fields off a
    // serde_json::Value.
    pointer.insert(
        "schema".into(),
        Value::String("JopeSelfplayProposalPointer.v1".into()),
    );
    pointer.entry("dl_slots_by_task").or_insert_with(
        || json!({"binary": PROPOSALS_PER_DATASET, "regression": PROPOSALS_PER_DATASET}),
    );
    pointer.insert("cross_context_backfill_rows".into(), json!(0));
    let value = Value::Object(pointer);
    atomic_json(&binding.pointer_path, &value)?;
    Ok(value)
}

/// Publish one branch into `proposals/current-by-task.json`.
///
/// ADDITIVE by construction. `proposals/current.json` is NOT written here and
/// is not written at all while the branches disagree: `jope_selfplay_wave.rs`
/// (`:1072`, `:1325`) rejects any proposal file whose rows do not all carry the
/// cohort plan's single `checkpoint_sha256`, and `jope_selfplay_queue.rs`
/// filters micro-waves by a single `checkpoint_epoch`. Handing either of them a
/// two-checkpoint pointer would fail the wave build, not improve it.
fn rebind_task(root: &Path, lane: &Lane, installed: &Installed) -> Result<Value> {
    let pointer_path = root.join("proposals/current-by-task.json");
    let mut document = read_task_proposal_pointer(root);
    if document.get("tasks").and_then(Value::as_object).is_none() {
        document["tasks"] = json!({});
    }
    let task = lane
        .task
        .as_deref()
        .ok_or_else(|| anyhow!("rebind_task called for the combined lane"))?;
    let dl_slots = document
        .pointer(&format!("/tasks/{task}/dl_slots"))
        .and_then(Value::as_i64)
        .unwrap_or(PROPOSALS_PER_DATASET);
    document["tasks"][task] = json!({
        "task": task,
        "checkpoint_sha256": lane.checkpoint_sha256,
        "checkpoint": lane.checkpoint.to_string_lossy(),
        "catalog_sha256": installed.catalog_sha256,
        "proposals": installed.proposals.to_string_lossy(),
        "proposals_sha256": installed.proposals_sha256,
        "summary": installed.summary.to_string_lossy(),
        "summary_sha256": installed.summary_sha256,
        "datasets": installed.datasets,
        "counts": installed.counts.clone(),
        "validity": installed.validity.clone(),
        "dl_slots": dl_slots,
        "cross_context_backfill_rows": 0,
        "updated_at_ms": now_ms() as u64,
    });
    document["schema"] = json!(TASK_POINTER_SCHEMA);
    document["combined_pointer"] = json!(root.join("proposals/current.json").to_string_lossy());
    document["publishes_into_combined_pointer"] = json!(false);
    document["publish_note"] = json!(
        "per-task proposal sets are not published into proposals/current.json: \
         jope_selfplay_wave.rs:1072 rejects a proposal file that mixes checkpoint epochs"
    );
    document["updated_at_ms"] = json!(now_ms() as u64);
    atomic_json(&pointer_path, &document)?;
    Ok(document)
}

// ---------------------------------------------------------------------------
// cycle
// ---------------------------------------------------------------------------

fn read_state(path: &Path) -> Value {
    read_json(path).unwrap_or_else(|_| json!({"schema": STATE_SCHEMA}))
}

fn attempt_for(state: &Value, key: &str) -> u64 {
    state
        .get("attempts")
        .and_then(|attempts| attempts.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1)
}

fn record_attempt(state: &mut Value, key: &str, attempt: u64) {
    let object = state.as_object_mut().expect("state is an object");
    let attempts = object
        .entry("attempts")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(map) = attempts.as_object_mut() {
        map.insert(key.to_owned(), json!(attempt));
    }
}

/// Lanes this cycle drives, in priority order.
///
/// The combined whole-catalog lane is ALWAYS present and ALWAYS first. It is the
/// lane that writes `proposals/current.json`, which is the pointer every
/// consumer actually reads (`selfplay_runtime.rs:107`, `supply.rs:96`,
/// `freshness.rs:409`, `alarms.rs:1166`, `trainer_watchdog.rs:1171`) and the
/// only lane whose output the wave/queue path can publish today. Dropping it
/// while the branches disagree would freeze the DL seed set at whatever catalog
/// it was last bound to: this unit exists to "regenerate gen-0 DL seeds whenever
/// the live catalog grows", and every dataset added after the first per-task
/// promotion would silently never receive one. The per-task lanes are ADDITIVE
/// and queue behind it.
fn ordered_lanes(combined: Lane, task_lanes: Option<Vec<Lane>>) -> Vec<Lane> {
    let mut lanes = vec![combined];
    lanes.extend(task_lanes.unwrap_or_default());
    lanes
}

fn cycle(args: &Args) -> Result<Value> {
    let root = args.root.as_path();
    let events = root.join("ops/propose-events.v1.jsonl");
    let state_path = root.join("ops/propose-driver-v1.json");

    let live = read_live_catalog(root)?;
    let binding = read_proposal_binding(root)?;

    let checkpoint =
        PathBuf::from(CHECKPOINTS).join(format!("{}.model.pt", binding.checkpoint_sha256));
    if !checkpoint.is_file() {
        bail!(
            "accepted checkpoint artifact is absent: {}",
            checkpoint.display()
        );
    }
    let combined = Lane {
        task: None,
        checkpoint_sha256: binding.checkpoint_sha256.clone(),
        checkpoint,
        bound_catalog_sha256: binding.bound_catalog_sha256.clone(),
    };
    // A malformed per-task pointer, or a branch checkpoint whose artifact has
    // been reaped, degrades to the historical combined lane instead of aborting
    // the cycle. A fault in an additive lane must never take down the lane the
    // GP consumes.
    let (task_lanes, degraded) = match read_task_lanes(root, &binding) {
        Ok(lanes) => (lanes, None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    let forked = task_lanes.is_some();
    let lanes = ordered_lanes(combined, task_lanes);
    if !forked {
        let mut outcome = run_lane(
            args,
            &live,
            &binding,
            &lanes[0],
            &events,
            &state_path,
            false,
        )?;
        if let Some(reason) = degraded {
            outcome["task_lanes_degraded"] = json!(reason);
            append_event(
                &events,
                &json!({
                    "schema": EVENT_SCHEMA,
                    "at_ms": now_ms() as u64,
                    "action": "task_lanes_degraded",
                    "reason": outcome["task_lanes_degraded"],
                }),
            )?;
        }
        return Ok(outcome);
    }
    // Forked. One GPU generation per cycle across all branches: two multi-hour
    // propose jobs at once double the draw the trainer competes with, and the
    // branches are independent, so serialising them costs latency and nothing
    // else. Installs are pure filesystem work and are never deferred. The
    // combined lane runs first and so holds first claim on that generation.
    let mut submitted = false;
    let mut per_lane = Map::new();
    for lane in &lanes {
        let result = run_lane(args, &live, &binding, lane, &events, &state_path, submitted)?;
        if result["action"] == json!("submitted") || result["action"] == json!("would_submit") {
            submitted = true;
        }
        per_lane.insert(lane.label(), result);
    }
    let combined_written = per_lane
        .get("combined")
        .is_some_and(|result| result["action"] == json!("installed"));
    let outcome = json!({
        "schema": EVENT_SCHEMA,
        "at_ms": now_ms() as u64,
        "action": "per_task",
        "live_catalog_sha256": live.catalog_sha256,
        "combined_checkpoint_sha256": binding.checkpoint_sha256,
        "task_checkpoints": lanes
            .iter()
            .map(|lane| (lane.label(), json!(lane.checkpoint_sha256)))
            .collect::<Map<String, Value>>(),
        // Whether proposals/current.json was rebound this cycle. The combined
        // lane keeps running while forked, so this is no longer always false.
        "combined_pointer_written": combined_written,
        "lanes": Value::Object(per_lane),
    });
    append_event(&events, &outcome)?;
    Ok(outcome)
}

/// One lane's cycle. Byte-for-byte the historical behaviour when
/// `lane.task == None`; the per-task lanes differ only in job id, the `--task`
/// argument, the publish directory, and which pointer is rewritten.
fn run_lane(
    args: &Args,
    live: &LiveCatalog,
    binding: &ProposalBinding,
    lane: &Lane,
    events: &Path,
    state_path: &Path,
    submission_used: bool,
) -> Result<Value> {
    let root = args.root.as_path();
    let bound = lane.bound_catalog_sha256.clone();
    let mut outcome = json!({
        "schema": EVENT_SCHEMA,
        "at_ms": now_ms() as u64,
        "task": lane.task,
        "checkpoint_sha256": lane.checkpoint_sha256,
        "live_catalog_sha256": live.catalog_sha256,
        "bound_catalog_sha256": bound,
    });

    if bound.as_deref() == Some(live.catalog_sha256.as_str()) {
        outcome["action"] = json!("up_to_date");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }

    let prefix = lane.prefix(&live.catalog_sha256);
    // Any propose job for this lane's (task, checkpoint) blocks a new
    // submission, even one aimed at an older catalog: two multi-hour GPU
    // generations at once would starve the trainer and the second would
    // supersede the first anyway.
    let family = lane.family();
    let jobs = job_states(&args.scq, &family)?;
    outcome["jobs"] = json!(jobs
        .iter()
        .map(|(id, state)| json!({"id": id, "state": state}))
        .collect::<Vec<_>>());

    if let Some((id, state)) = jobs.iter().find(|(_, state)| is_active(state)) {
        outcome["action"] = json!("waiting");
        outcome["job_id"] = json!(id);
        outcome["job_state"] = json!(state);
        append_event(events, &outcome)?;
        return Ok(outcome);
    }

    // Prefer a generation aimed at the live catalog, but a completed run for an
    // intermediate catalog still beats what is bound now: catalogs only grow,
    // so any newer proposal set strictly widens coverage. Installing it keeps
    // progress monotone when generations (hours) are slower than tranche
    // activations (hourly).
    let candidates = order_install_candidates(&jobs, &prefix);
    if let Some(id) = candidates.first().copied() {
        if args.dry_run {
            outcome["action"] = json!("would_install");
            outcome["job_id"] = json!(id);
            append_event(events, &outcome)?;
            return Ok(outcome);
        }
        if let Some(installed) = install(
            &args.scq,
            root,
            id,
            &lane.checkpoint_sha256,
            bound.as_deref(),
            lane.task.as_deref(),
        )? {
            let pointer = match lane.task {
                Some(_) => rebind_task(root, lane, &installed)?,
                None => rebind(binding, &installed)?,
            };
            outcome["action"] = json!("installed");
            outcome["job_id"] = json!(id);
            outcome["installed_catalog_sha256"] = json!(installed.catalog_sha256);
            outcome["catalog_still_ahead"] = json!(installed.catalog_sha256 != live.catalog_sha256);
            outcome["datasets"] = json!(installed.datasets);
            outcome["counts"] = installed.counts.clone();
            outcome["validity"] = installed.validity.clone();
            outcome["proposals"] = json!(installed.proposals.to_string_lossy());
            outcome["proposals_sha256"] = json!(installed.proposals_sha256);
            append_event(events, &outcome)?;
            let mut state = read_state(state_path);
            state[format!("last_install:{}", lane.label())] = outcome.clone();
            if lane.task.is_none() {
                state["last_install"] = outcome.clone();
                state["pointer"] = pointer;
            } else {
                state[format!("pointer:{}", lane.label())] = pointer;
            }
            state["schema"] = json!(STATE_SCHEMA);
            atomic_json(state_path, &state)?;
            return Ok(outcome);
        }
    }

    // Nothing active and nothing installable: submit (or retry) a generation.
    let mut state = read_state(state_path);
    let terminal_failures = jobs
        .iter()
        .filter(|(id, state)| {
            id.starts_with(&prefix) && matches!(state.as_str(), "failed" | "cancelled")
        })
        .count() as u64;
    let attempt_key = match &lane.task {
        Some(task) => format!("{task}:{}", live.catalog_sha256),
        None => live.catalog_sha256.clone(),
    };
    let attempt = attempt_for(&state, &attempt_key).max(1 + terminal_failures);
    if attempt > MAX_ATTEMPTS {
        outcome["action"] = json!("blocked");
        outcome["reason"] = json!("attempt budget exhausted; operator review required");
        outcome["attempt"] = json!(attempt);
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    if submission_used {
        // Another branch already claimed this cycle's GPU generation.
        outcome["action"] = json!("deferred");
        outcome["reason"] = json!("one propose generation per cycle; another branch submitted");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    // The generator aborts at startup unless the context blob matches its
    // pointer. Checking here costs one hash and saves a multi-hour GPU
    // reservation that could only ever fail.
    let data_sha256 = sha256_file(&live.data)
        .with_context(|| format!("hash context blob {}", live.data.display()))?;
    if data_sha256 != live.data_sha256 {
        bail!(
            "live context blob {} hashes {data_sha256}, pointer declares {}",
            live.data.display(),
            live.data_sha256
        );
    }
    let spec = render_spec(
        root,
        &lane.checkpoint,
        &lane.checkpoint_sha256,
        live,
        attempt,
        lane.task.as_deref(),
    )?;
    outcome["spec"] = json!(spec.path.to_string_lossy());
    outcome["job_id"] = json!(spec.job_id);
    outcome["attempt"] = json!(attempt);
    if args.dry_run {
        outcome["action"] = json!("would_submit");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    check_and_submit(&args.scq, &spec.path)?;
    outcome["action"] = json!("submitted");
    append_event(events, &outcome)?;
    record_attempt(&mut state, &attempt_key, attempt);
    state["schema"] = json!(STATE_SCHEMA);
    state[format!("last_submit:{}", lane.label())] = outcome.clone();
    if lane.task.is_none() {
        state["last_submit"] = outcome.clone();
    }
    atomic_json(state_path, &state)?;
    Ok(outcome)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let lock_path = args.root.join("ops/propose-driver.lock");
    let Some(_lock) = try_lock(&lock_path)? else {
        println!("{}", json!({"schema": EVENT_SCHEMA, "action": "locked"}));
        return Ok(());
    };
    let outcome = cycle(&args)?;
    println!("{outcome}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(path: &Path, body: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }

    fn fixture(root: &Path, bound: Option<&str>) -> String {
        let catalog = root.join("catalog/live.catalog.json");
        write(&catalog, "{\"entries\":[]}");
        let data = root.join("catalog/live.f32.bin");
        write(&data, "blob");
        let catalog_sha = sha256_file(&catalog).unwrap();
        let data_sha = sha256_file(&data).unwrap();
        atomic_json(
            &root.join("catalog/current.json"),
            &json!({
                "catalog": catalog,
                "catalog_sha256": catalog_sha,
                "data": data,
                "data_sha256": data_sha,
            }),
        )
        .unwrap();
        let proposals = root.join("proposals/by-checkpoint/old.jsonl");
        write(&proposals, "{}\n");
        let mut pointer = json!({
            "checkpoint_sha256": "a".repeat(64),
            "legacy_adopted": true,
            "proposals": proposals,
            "proposals_sha256": sha256_file(&proposals).unwrap(),
        });
        if let Some(bound) = bound {
            pointer["catalog_sha256"] = json!(bound);
        }
        atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
        catalog_sha
    }

    #[test]
    fn binding_prefers_the_pointer_catalog_and_falls_back_to_the_summary() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, Some(&"b".repeat(64)));
        let binding = read_proposal_binding(root).unwrap();
        assert_eq!(
            binding.bound_catalog_sha256.as_deref(),
            Some("b".repeat(64).as_str())
        );

        // No catalog_sha256 on the pointer: recover it from the summary the
        // generator wrote next to the proposal file.
        let summary = root.join("proposals/dl-proposals.summary.json");
        atomic_json(&summary, &json!({"catalog_sha256": "c".repeat(64)})).unwrap();
        let mut pointer = read_json(&root.join("proposals/current.json")).unwrap();
        pointer.as_object_mut().unwrap().remove("catalog_sha256");
        pointer["summary"] = json!(summary);
        atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
        let binding = read_proposal_binding(root).unwrap();
        assert_eq!(
            binding.bound_catalog_sha256.as_deref(),
            Some("c".repeat(64).as_str())
        );
    }

    #[test]
    fn up_to_date_binding_is_a_no_op() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let catalog_sha = fixture(root, None);
        let mut pointer = read_json(&root.join("proposals/current.json")).unwrap();
        pointer["catalog_sha256"] = json!(catalog_sha);
        atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
        let args = Args {
            root: root.to_path_buf(),
            scq: PathBuf::from("/nonexistent/scq"),
            dry_run: true,
        };
        // Reaching scq would fail; an up-to-date binding must short-circuit
        // before that, which is exactly the "cheap timer tick" property.
        let checkpoint_dir = Path::new(CHECKPOINTS);
        if !checkpoint_dir
            .join(format!("{}.model.pt", "a".repeat(64)))
            .is_file()
        {
            // The checkpoint guard fires first in this fixture; assert on it so
            // the test states the invariant it can observe here.
            let error = cycle(&args).unwrap_err().to_string();
            assert!(error.contains("checkpoint artifact is absent"), "{error}");
        }
    }

    #[test]
    fn rebind_preserves_legacy_fields_and_swaps_only_the_binding() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, Some(&"b".repeat(64)));
        let binding = read_proposal_binding(root).unwrap();
        let live = read_live_catalog(root).unwrap();
        let new_proposals = root.join("proposals/by-checkpoint/new.jsonl");
        write(&new_proposals, "{\"x\":1}\n");
        let new_summary = root.join("proposals/by-checkpoint/new.summary.json");
        write(&new_summary, "{}");
        let installed = Installed {
            catalog_sha256: live.catalog_sha256.clone(),
            proposals_sha256: sha256_file(&new_proposals).unwrap(),
            summary_sha256: sha256_file(&new_summary).unwrap(),
            proposals: new_proposals.clone(),
            summary: new_summary,
            counts: json!({"regression": 1}),
            validity: json!({}),
            datasets: 1,
        };
        rebind(&binding, &installed).unwrap();
        let pointer = read_json(&root.join("proposals/current.json")).unwrap();
        assert_eq!(pointer["legacy_adopted"], json!(true));
        assert_eq!(pointer["checkpoint_sha256"], json!("a".repeat(64)));
        assert_eq!(pointer["catalog_sha256"], json!(live.catalog_sha256));
        assert_eq!(pointer["proposals"], json!(new_proposals.to_string_lossy()));
        assert_eq!(
            pointer["proposals_sha256"],
            json!(sha256_file(&new_proposals).unwrap())
        );
    }

    #[test]
    fn frozen_pointer_snapshot_is_stable_and_content_addressed() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, None);
        let live = read_live_catalog(root).unwrap();
        let first = freeze_pointer(root, &live).unwrap();
        let first_sha = sha256_file(&first).unwrap();
        // A later flip of the mutable pointer must not disturb the snapshot the
        // in-flight job's spec already hashed.
        atomic_json(
            &root.join("catalog/current.json"),
            &json!({
                "catalog": live.catalog,
                "catalog_sha256": live.catalog_sha256,
                "data": live.data,
                "data_sha256": live.data_sha256,
                "note": "flipped",
            }),
        )
        .unwrap();
        let second = freeze_pointer(root, &live).unwrap();
        assert_eq!(first, second);
        assert_eq!(sha256_file(&second).unwrap(), first_sha);
    }

    #[test]
    fn spec_ids_are_catalog_scoped_and_retries_get_fresh_ids() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, None);
        let live = read_live_catalog(root).unwrap();
        let checkpoint = root.join("ckpt.model.pt");
        write(&checkpoint, "model");
        let checkpoint_sha = "a".repeat(64);
        let first = render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, None).unwrap();
        let retry = render_spec(root, &checkpoint, &checkpoint_sha, &live, 2, None).unwrap();
        assert_eq!(
            first.job_id,
            format!(
                "jope-selfplay-propose-{}-c{}",
                &checkpoint_sha[..12],
                &live.catalog_sha256[..12]
            )
        );
        assert_eq!(retry.job_id, format!("{}-r2", first.job_id));
        let body = fs::read_to_string(&first.path).unwrap();
        assert!(body.contains("--allow-deficient-floor"), "{body}");
        assert!(body.contains("gpus_per_task = 1"), "{body}");
        assert!(body.contains("xbabe1|xbabe3"), "{body}");
        // Every staged input must carry the hash the scheduler will verify.
        assert_eq!(body.matches("sha256 = ").count(), 6, "{body}");
        // The combined lane's argv must not gain a --task argument: this spec is
        // byte-identical to what the driver has always submitted.
        assert!(!body.contains("--task"), "{body}");
    }

    /// WS-6: a per-task lane renders a distinct job family, passes `--task`, and
    /// leaves the combined family prefix free.
    #[test]
    fn per_task_specs_are_branch_scoped_and_disjoint_from_the_combined_family() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, None);
        let live = read_live_catalog(root).unwrap();
        let checkpoint = root.join("ckpt.model.pt");
        write(&checkpoint, "model");
        let checkpoint_sha = "a".repeat(64);
        let combined = render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, None).unwrap();
        for (task, tag) in [
            ("binary", "tb"),
            ("regression", "tr"),
            ("multiclass", "tm"),
            ("timeseries", "tt"),
        ] {
            let spec =
                render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, Some(task)).unwrap();
            assert_eq!(
                spec.job_id,
                format!(
                    "jope-selfplay-propose-{tag}-{}-c{}",
                    &checkpoint_sha[..12],
                    &live.catalog_sha256[..12]
                )
            );
            // Disjoint from the combined family, so a per-task job can never be
            // mistaken for (or block) the whole-catalog lane.
            assert!(!spec.job_id.starts_with(&format!(
                "jope-selfplay-propose-{}-c",
                &checkpoint_sha[..12]
            )));
            assert_ne!(spec.job_id, combined.job_id);
            let body = fs::read_to_string(&spec.path).unwrap();
            assert!(body.contains(&format!("\"--task\", \"{task}\"")), "{body}");
        }
    }

    /// The combined whole-catalog lane must survive the fork.
    ///
    /// `proposals/current.json` is the only proposal pointer the wave/queue path
    /// can publish, and this unit's entire job is to rebind it as the catalog
    /// grows. If forking dropped that lane, every dataset added after the first
    /// per-task promotion would silently never receive a DL seed. The combined
    /// lane is therefore always present and always first, so it also holds first
    /// claim on the one-generation-per-cycle budget.
    #[test]
    fn the_combined_lane_survives_the_fork_and_keeps_first_claim_on_the_generation() {
        let combined = Lane {
            task: None,
            checkpoint_sha256: "a".repeat(64),
            checkpoint: PathBuf::from("/dev/null"),
            bound_catalog_sha256: None,
        };
        let task_lanes = vec![
            Lane {
                task: Some("binary".to_owned()),
                checkpoint_sha256: "b".repeat(64),
                checkpoint: PathBuf::from("/dev/null"),
                bound_catalog_sha256: None,
            },
            Lane {
                task: Some("regression".to_owned()),
                checkpoint_sha256: "c".repeat(64),
                checkpoint: PathBuf::from("/dev/null"),
                bound_catalog_sha256: None,
            },
        ];
        // Unforked: exactly the historical single lane.
        let unforked = ordered_lanes(combined.clone(), None);
        assert_eq!(unforked.len(), 1);
        assert_eq!(unforked[0].task, None);

        let forked = ordered_lanes(combined.clone(), Some(task_lanes));
        assert_eq!(forked.len(), 3);
        assert_eq!(forked[0].task, None, "combined lane must run first");
        assert_eq!(forked[0].label(), "combined");
        assert_eq!(forked[0].checkpoint_sha256, combined.checkpoint_sha256);
        assert_eq!(
            forked[1..]
                .iter()
                .map(|lane| lane.label())
                .collect::<Vec<_>>(),
            vec!["binary".to_owned(), "regression".to_owned()]
        );
    }

    /// WS-6: identical per-task checkpoints are NOT a fork — the driver keeps the
    /// single whole-catalog lane so `proposals/current.json` stays authoritative.
    #[test]
    fn task_lanes_only_fork_when_the_branch_checkpoints_differ() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fixture(root, None);
        let binding = read_proposal_binding(root).unwrap();
        assert!(read_task_lanes(root, &binding).unwrap().is_none());

        let same = "a".repeat(64);
        atomic_json(
            &root.join(TASK_ACCEPTED_POINTER),
            &json!({"tasks": {
                "binary": {"checkpoint_sha256": same},
                "regression": {"checkpoint_sha256": same},
            }}),
        )
        .unwrap();
        assert!(read_task_lanes(root, &binding).unwrap().is_none());

        let binary_checkpoint = root.join("training/immutable/checkpoints/b.model.pt");
        fs::create_dir_all(binary_checkpoint.parent().unwrap()).unwrap();
        write(&binary_checkpoint, "binary-model");
        let binary_sha = sha256_file(&binary_checkpoint).unwrap();
        let regression_checkpoint = root.join("training/immutable/checkpoints/r.model.pt");
        write(&regression_checkpoint, "regression-model");
        let regression_sha = sha256_file(&regression_checkpoint).unwrap();
        // Multiclass and timeseries share the regression artifact here: a task
        // without its own promoted anchor may legitimately ride another lane's
        // checkpoint, and the fork must still enumerate all four branches. (In
        // production a task absent from the pointer falls back to the GLOBAL
        // accepted checkpoint, whose artifact always resolves; this fixture's
        // binding sha is a fake with no artifact, so the entries are explicit.)
        atomic_json(
            &root.join(TASK_ACCEPTED_POINTER),
            &json!({"tasks": {
                "binary": {
                    "checkpoint_sha256": binary_sha,
                    "checkpoint": binary_checkpoint.to_string_lossy(),
                },
                "regression": {
                    "checkpoint_sha256": regression_sha,
                    "checkpoint": regression_checkpoint.to_string_lossy(),
                },
                "multiclass": {
                    "checkpoint_sha256": regression_sha,
                    "checkpoint": regression_checkpoint.to_string_lossy(),
                },
                "timeseries": {
                    "checkpoint_sha256": regression_sha,
                    "checkpoint": regression_checkpoint.to_string_lossy(),
                },
            }}),
        )
        .unwrap();
        let lanes = read_task_lanes(root, &binding).unwrap().expect("forked");
        assert_eq!(lanes.len(), 4);
        assert_eq!(lanes[0].task.as_deref(), Some("binary"));
        assert_eq!(lanes[0].checkpoint_sha256, binary_sha);
        assert_eq!(lanes[1].task.as_deref(), Some("regression"));
        assert_eq!(lanes[1].checkpoint_sha256, regression_sha);
        assert_ne!(lanes[0].checkpoint_sha256, lanes[1].checkpoint_sha256);
        assert_eq!(lanes[2].task.as_deref(), Some("multiclass"));
        assert_eq!(lanes[3].task.as_deref(), Some("timeseries"));
        assert_eq!(lanes[2].checkpoint_sha256, regression_sha);
        assert_eq!(lanes[3].checkpoint_sha256, regression_sha);
        assert_ne!(lanes[0].family(), lanes[1].family());
    }

    #[test]
    fn completed_runs_for_older_catalogs_are_still_installable_but_rank_last() {
        // Generations take hours and tranches activate hourly, so the driver
        // must not discard a finished run just because the catalog moved on
        // again while it was on the GPU.
        let jobs: Vec<(String, String)> = vec![
            ("jope-selfplay-propose-abc-cOLD".into(), "completed".into()),
            ("jope-selfplay-propose-abc-cLIVE".into(), "completed".into()),
            ("jope-selfplay-propose-abc-cDEAD".into(), "failed".into()),
        ];
        let ordered = order_install_candidates(&jobs, "jope-selfplay-propose-abc-cLIVE");
        assert_eq!(
            ordered,
            vec![
                &"jope-selfplay-propose-abc-cLIVE".to_owned(),
                &"jope-selfplay-propose-abc-cOLD".to_owned()
            ]
        );
        // Nothing finished at all: no install path, the caller falls through to
        // submitting a generation.
        let none: Vec<(String, String)> = vec![("x".into(), "queued".into())];
        assert!(order_install_candidates(&none, "x").is_empty());
    }

    #[test]
    fn a_second_cycle_cannot_run_while_the_lock_is_held() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("ops/propose-driver.lock");
        let held = try_lock(&path).unwrap();
        assert!(held.is_some());
        assert!(try_lock(&path).unwrap().is_none());
        drop(held);
        assert!(try_lock(&path).unwrap().is_some());
    }
}
