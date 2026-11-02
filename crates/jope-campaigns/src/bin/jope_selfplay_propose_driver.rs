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
use std::path::{Path, PathBuf};
use std::process::Command;

use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, sha256_file,
};
use jope_dime_regressor::spec_policy::PROPOSE;

#[path = "common/flock.rs"]
mod flock;

mod cycle;
pub(super) use cycle::cycle;

jope_paths::once_str!(
    fn scripts_str => paths::research_scripts_root()
);
jope_paths::once_str!(
    fn proposal_selection_policy_str => paths::campaign_manifest("selfplay/proposal-selection-policy.v1.json")
);
jope_paths::once_str!(
    fn prime_src_str => paths::runtime_bundle_pytorch()
);
jope_paths::once_str!(
    fn checkpoints_str => paths::legacy_checkpoints_root()
);
const STATE_SCHEMA: &str = "JopeProposeDriver.v1";
const EVENT_SCHEMA: &str = "JopeProposeDriverEvent.v1";
/// Initial direct-DL allocation. Future rungs are pointer-bound evidence.
const PROPOSALS_PER_DATASET: i64 = 8;
/// Deficiency is no longer fatal or backfilled; retained for CLI compatibility.
const ALLOW_DEFICIENT_FLOOR: i64 = 0;
const PROPOSAL_SEED: i64 = 20260726;
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
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
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
    if flock::exclusive(&file, false).is_err() {
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
/// lineage lives in `checkpoints_str()`. Either is acceptable, both are hash-verified,
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
    candidates.push(PathBuf::from(checkpoints_str()).join(format!("{sha256}.model.pt")));
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
    let published = read_task_proposal_pointer(root)?;
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

fn read_task_proposal_pointer(root: &Path) -> Result<Value> {
    let path = root.join("proposals/current-by-task.json");
    match fs::symlink_metadata(&path) {
        Ok(_) => read_json(&path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(json!({"schema": TASK_POINTER_SCHEMA, "tasks": {}}))
        }
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
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
    let script = Path::new(scripts_str()).join("test_exact_dl_generate.py");
    let common = Path::new(scripts_str()).join("test_exact_common.py");
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
         \"--device\", \"cuda\", \
         \"--selection-policy\", \"{{input:selection_policy}}\", \
         \"--seed\", \"{PROPOSAL_SEED}\", \
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
        PROPOSE.render_resource_block(),
        PROPOSE.render_placement_block(),
        String::new(),
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = \"{prime_src_str}:{scripts_str}\"",
            scripts_str = scripts_str(),
            prime_src_str = prime_src_str()
        ),
        format!("OMP_NUM_THREADS = \"{}\"", PROPOSE.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", PROPOSE.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
        PROPOSE.render_limits_block(),
        String::new(),
    ];
    let catalog_name = file_name(&live.catalog);
    let data_name = file_name(&live.data);
    let checkpoint_name = file_name(checkpoint);
    let inputs: Vec<(&str, PathBuf, String)> = vec![
        ("script", script, "scripts/test_exact_dl_generate.py".into()),
        ("common", common, "scripts/test_exact_common.py".into()),
        (
            "selection_policy",
            PathBuf::from(proposal_selection_policy_str()),
            "scripts/proposal-selection-policy.v1.json".into(),
        ),
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
    let mut document = read_task_proposal_pointer(root)?;
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

fn read_state(path: &Path) -> Result<Value> {
    match fs::symlink_metadata(path) {
        Ok(_) => read_json(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(json!({"schema": STATE_SCHEMA}))
        }
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
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
mod tests;
