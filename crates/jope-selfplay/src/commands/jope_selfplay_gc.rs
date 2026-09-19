//! jope_selfplay_gc — run-dir and receipts garbage collector.
//!
//! Retention rule (operator-mandated): a code-bearing artifact (a scheduler run
//! directory or a fetched receipts directory) may fall off disk ONLY once its
//! job is terminal and its evidence — canonical genome + score — is durably
//! indexed in the corpus (`ingested_jobs`), or its receipts were fetched to the
//! controller. Genomes and scores stay queryable in the corpus DB forever; the
//! bulky emitted-code copies do not.
//!
//! Audit-first: the default mode only writes a plan file under `ops/`. With
//! `--execute` the audit is re-run in-process (so the acted-on plan is always
//! fresh) and deletions are receipted line-by-batch to `ops/gc-events.v1.jsonl`.
//!
//! METRICS PRECONDITION (hard, not a timer-ordering hope): a trainer run
//! directory carries the only copy of its per-epoch train/validation losses in
//! `train_metrics.jsonl`, next to ~12 GB of checkpoints. Pruning the tensors is
//! the whole point; pruning the metrics is unacceptable. So `execute_plan`
//! archives every candidate's metrics into `training/history/` FIRST and deletes
//! only the candidates the archive provably covers — see `guard_metrics`. A
//! candidate whose metrics cannot be archived is refused and receipted, and
//! `jope_selfplay_alarms` turns that refusal into a loud alarm.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use jope_campaigns::paths;
use jope_campaigns::training_history as th;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn scq_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn default_runs_root_str => paths::scq_runs_root()
);
const LOCAL_NODE: &str = "xbabe2";
const TERMINAL_STATES: [&str; 3] = ["completed", "failed", "cancelled"];
const SAMPLE_FOR_SIZE: usize = 200;
const DELETE_BATCH: usize = 500;
/// How long the collector waits for the capture lock before refusing to prune
/// metric-bearing directories. A full capture cycle measures ~0.6 s, so 15 s is
/// many cycles' worth of headroom; waiting longer would only delay a daily job.
const LOCK_ATTEMPTS: u32 = 15;
const LOCK_BACKOFF_MS: u64 = 1_000;

#[derive(Parser)]
#[command(
    name = "jope_selfplay_gc",
    about = "run-dir and receipts GC (audit-first)"
)]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Garbage-collect scheduler run directories on one node.
    Runs {
        #[arg(long)]
        node: String,
        #[arg(long, default_value = default_runs_root_str())]
        runs_root: String,
        #[arg(long, default_value_t = 24)]
        min_age_hours: u64,
        #[arg(long, default_value_t = 500)]
        keep_newest: usize,
        #[arg(long, default_value = "jte-")]
        prefix: String,
        #[arg(long, default_value = default_root_str())]
        root: String,
        #[arg(long)]
        execute: bool,
    },
    /// Garbage-collect fetched receipts directories on the controller.
    Receipts {
        #[arg(long, default_value = default_root_str())]
        root: String,
        #[arg(long, default_value_t = 48)]
        min_age_hours: u64,
        #[arg(long, default_value_t = 2000)]
        keep_newest: usize,
        #[arg(long)]
        execute: bool,
    },
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One enumerated directory: name + mtime (ms since epoch).
#[derive(Clone, Debug)]
struct DirEntryInfo {
    name: String,
    mtime_ms: u64,
}

fn sane_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !name.starts_with('.')
}

/// Gate evaluation, pure so it is unit-testable. Returns the basis (list of
/// satisfied gates) when the directory is deletable, None otherwise.
#[allow(clippy::too_many_arguments)]
fn evaluate_gates(
    name: &str,
    prefix: &str,
    age_secs: u64,
    min_age_secs: u64,
    protected_newest: bool,
    scq_state: Option<&str>,
    ingested: bool,
    receipts_dir_exists: bool,
) -> Option<Vec<String>> {
    if !name.starts_with(prefix) || !sane_name(name) {
        return None;
    }
    if age_secs < min_age_secs || protected_newest {
        return None;
    }
    let mut basis = vec![
        format!("prefix:{prefix}"),
        format!("age_secs:{age_secs}"),
        "not_in_keep_newest".to_string(),
    ];
    match scq_state {
        None => basis.push("scheduler:pruned".to_string()),
        Some(state) if TERMINAL_STATES.contains(&state) => {
            basis.push(format!("scheduler:{state}"));
        }
        Some(_) => return None,
    }
    if ingested {
        basis.push("evidence:ingested_jobs".to_string());
    } else if receipts_dir_exists {
        basis.push("evidence:receipts_fetched".to_string());
    } else {
        return None;
    }
    Some(basis)
}

fn ssh_target(node: &str) -> Result<Vec<String>> {
    match node {
        "xbabe0" | "xbabe1" | "xbabe3" => Ok(vec![
            "ssh".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            node.into(),
        ]),
        "xbabe4" => Ok(vec![
            "ssh".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            "-p".into(),
            "2224".into(),
            "xbabe4".into(),
        ]),
        LOCAL_NODE => bail!("xbabe2 is the local host; never ssh to it"),
        other => bail!("unknown node {other}"),
    }
}

fn run_capture(program: &str, args: &[String], stdin: Option<&str>) -> Result<String> {
    let mut command = Command::new(program);
    command.args(args);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {program}"))?;
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .context("stdin unavailable")?
            .write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Byte-exact variant of `run_capture`. Archived metrics must be identical to
/// what the trainer wrote, so a lossy UTF-8 round trip is not acceptable here.
fn run_capture_bytes(program: &str, args: &[String], stdin: Option<&str>) -> Result<Vec<u8>> {
    let mut command = Command::new(program);
    command.args(args);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {program}"))?;
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .context("stdin unavailable")?
            .write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

/// Enumerate immediate subdirectories with mtimes, locally or over ssh.
fn enumerate_dirs(node: &str, runs_root: &str) -> Result<Vec<DirEntryInfo>> {
    if node == LOCAL_NODE {
        let mut rows = Vec::new();
        for entry in std::fs::read_dir(runs_root)? {
            let entry = entry.context("read local run-directory entry")?;
            let file_type = entry.file_type().context("read local run-directory type")?;
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let mtime_ms = entry
                .metadata()
                .context("read local run-directory metadata")?
                .modified()
                .context("read local run-directory mtime")?
                .duration_since(UNIX_EPOCH)
                .context("local run-directory mtime predates Unix epoch")?
                .as_millis() as u64;
            rows.push(DirEntryInfo { name, mtime_ms });
        }
        return Ok(rows);
    }
    let target = ssh_target(node)?;
    let script = format!(
        "cd {runs_root} 2>/dev/null && find . -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\\n'"
    );
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(script);
    let stdout = run_capture(&target[0], &args, None)?;
    let mut rows = Vec::new();
    for (line_number, line) in stdout.lines().enumerate() {
        let (ts, name) = line.split_once(' ').with_context(|| {
            format!("malformed remote directory row at line {}", line_number + 1)
        })?;
        let secs: f64 = ts.parse().with_context(|| {
            format!("invalid remote directory mtime at line {}", line_number + 1)
        })?;
        if !secs.is_finite() || secs < 0.0 || !sane_name(name) {
            bail!("unsafe remote directory row at line {}", line_number + 1);
        }
        rows.push(DirEntryInfo {
            name: name.to_string(),
            mtime_ms: (secs * 1000.0) as u64,
        });
    }
    Ok(rows)
}

fn parse_scq_job_states(stdout: &[u8]) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    let value = serde_json::from_slice::<Value>(stdout).context("decode SCQ job-state JSON")?;
    let jobs = value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ job-state JSON has no jobs array")?;
    for job in jobs {
        let id = job
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .context("SCQ job is missing a non-empty id")?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .context("SCQ job is missing a non-empty state")?;
        if map.insert(id.to_string(), state.to_string()).is_some() {
            bail!("SCQ job-state JSON repeats job id {id}");
        }
    }
    Ok(map)
}

fn scq_job_states() -> Result<HashMap<String, String>> {
    let output = Command::new(scq_str())
        .args(["--json", "ls"])
        .output()
        .context("execute SCQ job-state query")?;
    if !output.status.success() {
        bail!(
            "SCQ job-state query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_scq_job_states(&output.stdout)
}

/// Look up which of `names` are durably ingested. Falls back to the empty set
/// on persistent lock/errors (the receipts-dir gate still applies then).
fn ingested_set(root: &Path, names: &[String]) -> HashSet<String> {
    let mut found = HashSet::new();
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let Ok(connection) = Connection::open_with_flags(
        &corpus,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return found;
    };
    if connection.busy_timeout(Duration::from_secs(60)).is_err() {
        return found;
    }
    let Ok(mut statement) = connection.prepare("SELECT 1 FROM ingested_jobs WHERE job_id = ?")
    else {
        return found;
    };
    for name in names {
        if let Ok(mut rows) = statement.query([name]) {
            if let Ok(Some(_)) = rows.next() {
                found.insert(name.clone());
            }
        }
    }
    found
}

struct Plan {
    schema: &'static str,
    node: String,
    entries: Vec<Value>,
    candidate_names: Vec<String>,
    estimated_bytes: u64,
    foreign_prefix_counts: HashMap<String, usize>,
    total_dirs: usize,
}

fn build_runs_plan(
    node: &str,
    runs_root: &str,
    root: &Path,
    prefix: &str,
    min_age_hours: u64,
    keep_newest: usize,
) -> Result<Plan> {
    let mut dirs = enumerate_dirs(node, runs_root)?;
    let total_dirs = dirs.len();
    let mut foreign: HashMap<String, usize> = HashMap::new();
    for entry in &dirs {
        if !entry.name.starts_with(prefix) {
            let key = entry
                .name
                .split(['-', '_'])
                .next()
                .unwrap_or("other")
                .to_string();
            *foreign.entry(key).or_default() += 1;
        }
    }
    dirs.retain(|d| d.name.starts_with(prefix));
    dirs.sort_by_key(|d| std::cmp::Reverse(d.mtime_ms));
    let protected: HashSet<String> = dirs
        .iter()
        .take(keep_newest)
        .map(|d| d.name.clone())
        .collect();
    let states = scq_job_states()?;
    let names: Vec<String> = dirs.iter().map(|d| d.name.clone()).collect();
    let ingested = ingested_set(root, &names);
    let receipts_root = root.join("receipts");
    let now = now_ms();
    let min_age_secs = min_age_hours * 3600;
    let mut entries = Vec::new();
    let mut candidate_names = Vec::new();
    for dir in &dirs {
        let age_secs = now.saturating_sub(dir.mtime_ms) / 1000;
        let basis = evaluate_gates(
            &dir.name,
            prefix,
            age_secs,
            min_age_secs,
            protected.contains(&dir.name),
            states.get(&dir.name).map(String::as_str),
            ingested.contains(&dir.name),
            receipts_root.join(&dir.name).is_dir(),
        );
        if let Some(basis) = basis {
            entries.push(json!({
                "dir": format!("{runs_root}/{}", dir.name),
                "mtime_ms": dir.mtime_ms,
                "basis": basis,
            }));
            candidate_names.push(dir.name.clone());
        }
    }
    let estimated_bytes = estimate_bytes(node, runs_root, &candidate_names)?;
    Ok(Plan {
        schema: "JopeSelfplayGcPlan.v1",
        node: node.to_string(),
        entries,
        candidate_names,
        estimated_bytes,
        foreign_prefix_counts: foreign,
        total_dirs,
    })
}

/// Sample-based size estimate: du -sk over up to SAMPLE_FOR_SIZE candidates,
/// mean extrapolated over the full candidate list.
fn estimate_bytes(node: &str, runs_root: &str, names: &[String]) -> Result<u64> {
    if names.is_empty() {
        return Ok(0);
    }
    let step = (names.len() / SAMPLE_FOR_SIZE).max(1);
    let sample: Vec<&String> = names.iter().step_by(step).take(SAMPLE_FOR_SIZE).collect();
    let listing = sample
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let stdout = if node == LOCAL_NODE {
        let args = vec![
            "-c".to_string(),
            format!("cd {runs_root} && xargs -d '\\n' du -sk -- 2>/dev/null | awk '{{s+=$1}} END {{print s}}'"),
        ];
        run_capture("sh", &args, Some(&listing))?
    } else {
        let target = ssh_target(node)?;
        let mut args: Vec<String> = target[1..].to_vec();
        args.push(format!(
            "cd {runs_root} && xargs -d '\\n' du -sk -- 2>/dev/null | awk '{{s+=$1}} END {{print s}}'"
        ));
        run_capture(&target[0], &args, Some(&listing))?
    };
    let sampled_kb: u64 = stdout.trim().parse().unwrap_or(0);
    let mean = sampled_kb as f64 / sample.len().max(1) as f64;
    Ok((mean * names.len() as f64 * 1024.0) as u64)
}

fn write_plan(root: &Path, file_name: &str, plan: &Plan) -> Result<PathBuf> {
    let report = json!({
        "schema": plan.schema,
        "node": plan.node,
        "generated_at_ms": now_ms(),
        "total_dirs_seen": plan.total_dirs,
        "candidate_count": plan.entries.len(),
        "estimated_bytes": plan.estimated_bytes,
        "estimated_bytes_note": "sampled du, extrapolated",
        "foreign_prefix_counts": plan.foreign_prefix_counts,
        "entries": plan.entries,
    });
    let path = root.join("ops").join(file_name);
    let temporary = root
        .join("ops")
        .join(format!(".{file_name}.{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(&report)?)?;
    std::fs::rename(&temporary, &path)?;
    Ok(path)
}

fn append_receipt(root: &Path, record: Value) -> Result<()> {
    let path = root.join("ops/gc-events.v1.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

fn delete_batch(node: &str, runs_root: &str, batch: &[String]) -> Result<()> {
    for name in batch {
        if !sane_name(name) {
            bail!("refusing unsafe dir name {name:?}");
        }
    }
    if node == LOCAL_NODE {
        for name in batch {
            let path = Path::new(runs_root).join(name);
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
            }
        }
        return Ok(());
    }
    let listing = batch.join("\n");
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!("cd {runs_root} && xargs -d '\\n' rm -rf --"));
    run_capture(&target[0], &args, Some(&listing))?;
    Ok(())
}

/// One metrics file discovered under a candidate run directory, as
/// `(run_id, path relative to runs_root, size in bytes, mtime in ms)`.
type RemoteMetrics = (String, String, u64, u64);

/// Enumerate `train_metrics.jsonl` under the candidate directories of a remote
/// node in a single ssh round trip. The mtime comes back with the size because
/// the truncated-tail rule needs to know whether the writer is still alive.
fn remote_metrics(node: &str, runs_root: &str, names: &[String]) -> Result<Vec<RemoteMetrics>> {
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!(
        "cd {runs_root} && xargs -d '\\n' sh -c 'find \"$@\" -maxdepth 6 -name {} -printf \"%s %T@ %p\\n\" 2>/dev/null' sh",
        th::METRICS_FILE
    ));
    let listing = names.join("\n");
    let stdout = run_capture(&target[0], &args, Some(&listing))?;
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let (Some(size), Some(mtime), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if parts.next().is_some() {
            continue;
        }
        let path = path.trim_start_matches("./");
        let Some(run_id) = path.split('/').next() else {
            continue;
        };
        rows.push((
            run_id.to_string(),
            path.to_string(),
            size.parse::<u64>().unwrap_or(0),
            mtime
                .parse::<f64>()
                .map(|seconds| (seconds * 1000.0) as u64)
                .unwrap_or(0),
        ));
    }
    Ok(rows)
}

fn remote_read(node: &str, runs_root: &str, relative: &str) -> Result<Vec<u8>> {
    // Every component came from our own `find` over names we already validated
    // with `sane_name`, but re-check before it reaches a shell.
    if !relative.split('/').all(sane_name) {
        bail!("refusing unsafe remote path {relative:?}");
    }
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!("cat -- {runs_root}/{relative}"));
    run_capture_bytes(&target[0], &args, None)
}

/// Result of the metrics precondition for one GC pass.
struct MetricsGuard {
    /// Candidates the archive provably covers (or that hold no metrics at all).
    safe: Vec<String>,
    /// Candidates refused because their metrics are not durably archived.
    blocked: Vec<Value>,
    archived_rows: u64,
    archived_bytes: u64,
    /// Candidates cleared only after an abandoned trailing partial row was
    /// written off under the truncated-tail policy, with the hole recorded.
    truncated: Vec<Value>,
}

/// One metrics file under a candidate directory, local or remote.
struct GuardFile {
    /// Absolute path locally, path relative to the runs root remotely.
    source: String,
    key: th::SegmentKey,
    len: u64,
    mtime_ms: u64,
}

/// THE PRECONDITION. Archive-then-prune: every candidate's `train_metrics.jsonl`
/// is ingested into `training/history/` and re-verified here, and only
/// candidates whose rows are provably on durable storage are returned as `safe`.
///
/// Fail-closed in every direction:
/// * capture lock unavailable  -> metric-bearing candidates are refused;
/// * ingest error              -> that candidate is refused;
/// * archive still behind the source (a row mid-write) -> refused this pass;
/// * remote enumeration failed -> every `-train-` candidate on that node is
///   refused, while dirs that cannot carry trainer metrics still prune, so a
///   flaky ssh degrades throughput instead of losing data.
///
/// Note this is deliberately independent of `--prefix`: it guards whatever the
/// plan selected, so widening the collector to `jope-` (the obvious next move
/// under disk pressure) cannot silently delete a training curve.
///
/// THE ONE EXCEPTION, and it is a policy the operator set: a trainer killed
/// mid-write leaves a trailing INCOMPLETE JSONL line. That line is not data —
/// the epoch never finished being written — and refusing to prune on it leaks
/// ~12 GB per occurrence forever. So when the shortfall is exactly one
/// unterminated fragment AND the file has stopped being written, every complete
/// row is archived, the hole is recorded (`truncations.v1.jsonl`, the segment
/// state, this receipt) with the fragment kept verbatim, and the directory is
/// cleared. A shortfall that contains a newline is COMPLETE rows and still
/// blocks; so does a fragment too large to be one row, and so does a file that
/// is still being written.
fn guard_metrics(node: &str, runs_root: &str, root: &Path, names: &[String]) -> MetricsGuard {
    let mut guard = MetricsGuard {
        safe: Vec::new(),
        blocked: Vec::new(),
        archived_rows: 0,
        archived_bytes: 0,
        truncated: Vec::new(),
    };
    if names.is_empty() {
        return guard;
    }
    let archive = th::archive_root(root);

    // Discover what actually carries metrics. Local runs are walked directly;
    // remote nodes answer in one round trip.
    let mut by_run: HashMap<String, Vec<GuardFile>> = HashMap::new();
    let mut enumeration_failed = false;
    if node == LOCAL_NODE {
        for name in names {
            for found in th::scan_run_dir(Path::new(runs_root), name) {
                let mtime_ms = std::fs::metadata(&found.path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                by_run.entry(name.clone()).or_default().push(GuardFile {
                    source: found.path.to_string_lossy().into_owned(),
                    key: found.key,
                    len: found.len,
                    mtime_ms,
                });
            }
        }
    } else {
        match remote_metrics(node, runs_root, names) {
            Ok(rows) => {
                for (run_id, relative, len, mtime_ms) in rows {
                    let Some(key) = th::key_from_path(&run_id, Path::new(&relative)) else {
                        continue;
                    };
                    by_run.entry(run_id).or_default().push(GuardFile {
                        source: relative,
                        key,
                        len,
                        mtime_ms,
                    });
                }
            }
            Err(_) => enumeration_failed = true,
        }
    }

    // Hold the same lock the capture timer uses, so a tick cannot interleave
    // with the archiving we are about to do.
    let lock = th::CaptureLock::acquire(&archive, LOCK_ATTEMPTS, LOCK_BACKOFF_MS)
        .ok()
        .flatten();

    for name in names {
        let Some(files) = by_run.get(name) else {
            if enumeration_failed && name.contains(th::TRAIN_MARKER) {
                guard.blocked.push(json!({
                    "dir": name,
                    "reason": "metrics_enumeration_failed",
                    "node": node,
                }));
            } else {
                guard.safe.push(name.clone());
            }
            continue;
        };
        if lock.is_none() {
            guard.blocked.push(json!({
                "dir": name,
                "reason": "capture_lock_unavailable",
                "metrics_files": files.len(),
            }));
            continue;
        }
        let mut covered = true;
        let mut detail = Vec::new();
        for file in files {
            let GuardFile {
                source,
                key,
                len: source_len,
                mtime_ms,
            } = file;
            let reader = if node == LOCAL_NODE {
                Ok(th::AnySource::File(th::FileSource(PathBuf::from(source))))
            } else {
                remote_read(node, runs_root, source)
                    .map(|bytes| th::AnySource::Bytes(th::BytesSource(bytes)))
            };
            let display = if node == LOCAL_NODE {
                source.clone()
            } else {
                format!("{node}:{runs_root}/{source}")
            };
            let outcome = match &reader {
                Ok(reader) => th::ingest_source(&archive, key, reader, &display, node),
                Err(error) => Err(anyhow::anyhow!("{error}")),
            };
            match outcome {
                Ok(outcome) => {
                    guard.archived_rows += outcome.appended_rows;
                    guard.archived_bytes += outcome.appended_bytes;
                    // Re-read the archive file length rather than trusting the
                    // in-memory outcome: this is the durable fact.
                    let held = th::archived_bytes(&archive, key);
                    if held >= *source_len {
                        continue;
                    }
                    // Shortfall. The only kind that may be written off is a
                    // lone unterminated fragment on a file nobody is writing.
                    let reader = reader.as_ref().expect("reader present on Ok outcome");
                    let residual = th::classify_residual(reader, held, *source_len);
                    // Local sources share this clock, so their own mtime is
                    // exact. A remote mtime is another machine's opinion of the
                    // time; judging on it could write off a row that is still
                    // being written, so remote quiescence is instead "the
                    // capture timer has observed this size+mtime unchanged for
                    // long enough, measured here".
                    let quiescent = if node == LOCAL_NODE {
                        th::tail_is_abandoned(*mtime_ms, now_ms())
                    } else {
                        th::remote_tail_is_abandoned(
                            th::SegmentState::load(&th::segment_dir(&archive, key)).as_ref(),
                            &format!("{source_len}:{mtime_ms}"),
                            now_ms(),
                            th::TAIL_QUIESCENT_SECS,
                        )
                    };
                    match (&residual, quiescent) {
                        (Ok(th::Residual::PartialRow(fragment)), true) => {
                            match th::record_truncation(
                                &archive,
                                key,
                                node,
                                &display,
                                fragment,
                                *source_len,
                                *mtime_ms,
                            ) {
                                Ok(_) => guard.truncated.push(json!({
                                    "dir": name,
                                    "source": display,
                                    "source_bytes": source_len,
                                    "archived_bytes": held,
                                    "residual_bytes": fragment.len(),
                                    "source_idle_secs": now_ms().saturating_sub(*mtime_ms) / 1000,
                                    "reason": "truncated_tail_archived",
                                    "policy": "trailing incomplete row is not data; \
                                               all complete rows archived and the hole recorded",
                                })),
                                Err(error) => {
                                    // Could not make the record durable, so the
                                    // right to prune was never earned.
                                    covered = false;
                                    detail.push(json!({
                                        "source": display,
                                        "reason": "truncation_record_failed",
                                        "error": error.to_string(),
                                    }));
                                }
                            }
                        }
                        _ => {
                            covered = false;
                            detail.push(json!({
                                "source": display,
                                "source_bytes": source_len,
                                "archived_bytes": held,
                                "residual": residual
                                    .as_ref()
                                    .map(|r| r.reason())
                                    .unwrap_or("unreadable"),
                                "source_idle_secs": now_ms().saturating_sub(*mtime_ms) / 1000,
                                "quiescent": quiescent,
                            }));
                        }
                    }
                }
                Err(error) => {
                    covered = false;
                    detail.push(json!({"source": display, "error": error.to_string()}));
                }
            }
        }
        if covered {
            guard.safe.push(name.clone());
        } else {
            guard.blocked.push(json!({
                "dir": name,
                "reason": "metrics_unarchived",
                "detail": detail,
            }));
        }
    }

    if lock.is_some() {
        let _ = th::rebuild_index(root, &archive);
    }
    drop(lock);
    guard
}

fn execute_plan(node: &str, runs_root: &str, root: &Path, plan: &Plan) -> Result<(usize, usize)> {
    // ARCHIVE-THEN-PRUNE. Nothing below this line may delete a directory whose
    // metrics are not already in training/history/.
    let guard = guard_metrics(node, runs_root, root, &plan.candidate_names);
    if !guard.blocked.is_empty() || guard.archived_rows > 0 || !guard.truncated.is_empty() {
        append_receipt(
            root,
            json!({
                "ts_ms": now_ms(),
                "node": node,
                "event": "metrics_guard",
                "candidates": plan.candidate_names.len(),
                "safe": guard.safe.len(),
                "blocked": guard.blocked.len(),
                "archived_rows": guard.archived_rows,
                "archived_bytes": guard.archived_bytes,
                "blocked_detail": guard.blocked,
                "truncated_tails": guard.truncated.len(),
                "truncated_detail": guard.truncated,
            }),
        )?;
    }
    let mut deleted = 0usize;
    for batch in guard.safe.chunks(DELETE_BATCH) {
        delete_batch(node, runs_root, batch)?;
        deleted += batch.len();
        append_receipt(
            root,
            json!({
                "ts_ms": now_ms(),
                "node": node,
                "runs_root": runs_root,
                "deleted_count": batch.len(),
                "bytes_estimate": plan.estimated_bytes / plan.candidate_names.len().max(1) as u64 * batch.len() as u64,
                "first": batch.first(),
                "last": batch.last(),
            }),
        )?;
    }
    Ok((deleted, guard.blocked.len()))
}

fn cmd_runs(
    node: &str,
    runs_root: &str,
    root: &Path,
    prefix: &str,
    min_age_hours: u64,
    keep_newest: usize,
    execute: bool,
) -> Result<()> {
    let plan = build_runs_plan(node, runs_root, root, prefix, min_age_hours, keep_newest)?;
    let plan_path = write_plan(root, &format!("gc-plan-{node}.v1.json"), &plan)?;
    let mut summary = json!({
        "mode": if execute { "execute" } else { "audit" },
        "node": node,
        "plan": plan_path.to_string_lossy(),
        "total_dirs_seen": plan.total_dirs,
        "candidates": plan.candidate_names.len(),
        "estimated_bytes": plan.estimated_bytes,
        "foreign_prefix_counts": plan.foreign_prefix_counts,
    });
    if execute {
        let (deleted, blocked) = execute_plan(node, runs_root, root, &plan)?;
        summary["deleted"] = json!(deleted);
        summary["blocked_unarchived_metrics"] = json!(blocked);
    }
    println!("{summary}");
    Ok(())
}

fn cmd_receipts(root: &Path, min_age_hours: u64, keep_newest: usize, execute: bool) -> Result<()> {
    let receipts_root = root.join("receipts");
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(&receipts_root)?.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("jte-") || !sane_name(&name) {
            continue;
        }
        let mtime_ms = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        dirs.push(DirEntryInfo { name, mtime_ms });
    }
    let total_dirs = dirs.len();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.mtime_ms));
    let protected: HashSet<String> = dirs
        .iter()
        .take(keep_newest)
        .map(|d| d.name.clone())
        .collect();
    let names: Vec<String> = dirs.iter().map(|d| d.name.clone()).collect();
    let ingested = ingested_set(root, &names);
    let now = now_ms();
    let min_age_secs = min_age_hours * 3600;
    let mut entries = Vec::new();
    let mut candidates = Vec::new();
    for dir in &dirs {
        let age_secs = now.saturating_sub(dir.mtime_ms) / 1000;
        // Receipts fall off only on the strong gate: evidence is in the corpus.
        if age_secs < min_age_secs || protected.contains(&dir.name) || !ingested.contains(&dir.name)
        {
            continue;
        }
        entries.push(json!({
            "dir": receipts_root.join(&dir.name).to_string_lossy(),
            "mtime_ms": dir.mtime_ms,
            "basis": ["prefix:jte-", format!("age_secs:{age_secs}"), "not_in_keep_newest", "evidence:ingested_jobs"],
        }));
        candidates.push(dir.name.clone());
    }
    let estimated_bytes =
        estimate_bytes(LOCAL_NODE, &receipts_root.to_string_lossy(), &candidates)?;
    let plan = Plan {
        schema: "JopeSelfplayGcPlan.v1",
        node: "xbabe2:receipts".to_string(),
        entries,
        candidate_names: candidates,
        estimated_bytes,
        foreign_prefix_counts: HashMap::new(),
        total_dirs,
    };
    let plan_path = write_plan(root, "gc-plan-receipts.v1.json", &plan)?;
    let mut summary = json!({
        "mode": if execute { "execute" } else { "audit" },
        "node": "xbabe2:receipts",
        "plan": plan_path.to_string_lossy(),
        "total_dirs_seen": plan.total_dirs,
        "candidates": plan.candidate_names.len(),
        "estimated_bytes": plan.estimated_bytes,
    });
    if execute {
        let mut deleted = 0usize;
        for batch in plan.candidate_names.chunks(DELETE_BATCH) {
            for name in batch {
                let path = receipts_root.join(name);
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)?;
                }
            }
            deleted += batch.len();
            append_receipt(
                root,
                json!({
                    "ts_ms": now_ms(),
                    "node": "xbabe2:receipts",
                    "deleted_count": batch.len(),
                    "bytes_estimate": plan.estimated_bytes / plan.candidate_names.len().max(1) as u64 * batch.len() as u64,
                    "first": batch.first(),
                    "last": batch.last(),
                }),
            )?;
        }
        summary["deleted"] = json!(deleted);
    }
    println!("{summary}");
    Ok(())
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    match args.command {
        Cmd::Runs {
            node,
            runs_root,
            min_age_hours,
            keep_newest,
            prefix,
            root,
            execute,
        } => cmd_runs(
            &node,
            &runs_root,
            Path::new(&root),
            &prefix,
            min_age_hours,
            keep_newest,
            execute,
        ),
        Cmd::Receipts {
            root,
            min_age_hours,
            keep_newest,
            execute,
        } => cmd_receipts(Path::new(&root), min_age_hours, keep_newest, execute),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_prefix_and_sanity() {
        assert!(evaluate_gates("sh-123", "jte-", 999_999, 100, false, None, true, true).is_none());
        assert!(
            evaluate_gates("jte-../x", "jte-", 999_999, 100, false, None, true, true).is_none()
        );
    }

    #[test]
    fn gate_age_and_keep_newest() {
        assert!(evaluate_gates("jte-a", "jte-", 50, 100, false, None, true, true).is_none());
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, true, None, true, true).is_none());
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, false, None, true, true).is_some());
    }

    #[test]
    fn gate_scheduler_states() {
        assert!(evaluate_gates(
            "jte-a",
            "jte-",
            500,
            100,
            false,
            Some("running"),
            true,
            true
        )
        .is_none());
        let basis = evaluate_gates(
            "jte-a",
            "jte-",
            500,
            100,
            false,
            Some("completed"),
            true,
            false,
        )
        .expect("terminal deletable");
        assert!(basis.contains(&"scheduler:completed".to_string()));
        let pruned = evaluate_gates("jte-a", "jte-", 500, 100, false, None, false, true)
            .expect("pruned with receipts deletable");
        assert!(pruned.contains(&"scheduler:pruned".to_string()));
        assert!(pruned.contains(&"evidence:receipts_fetched".to_string()));
    }

    #[test]
    fn scq_state_parser_is_typed_and_fail_closed() {
        let states =
            parse_scq_job_states(br#"{"Jobs":{"jobs":[{"id":"jte-a","state":"completed"}]}}"#)
                .unwrap();
        assert_eq!(states.get("jte-a").map(String::as_str), Some("completed"));
        assert!(parse_scq_job_states(b"not-json").is_err());
        assert!(parse_scq_job_states(br#"{}"#).is_err());
        assert!(parse_scq_job_states(br#"{"jobs":[{"id":"jte-a"}]}"#).is_err());
        assert!(parse_scq_job_states(
            br#"{"jobs":[{"id":"jte-a","state":"failed"},{"id":"jte-a","state":"failed"}]}"#
        )
        .is_err());
    }

    #[test]
    fn ssh_topology_includes_xbabe0_but_not_the_local_node() {
        let target = ssh_target("xbabe0").unwrap();
        assert_eq!(target.first().map(String::as_str), Some("ssh"));
        assert_eq!(target.last().map(String::as_str), Some("xbabe0"));
        assert!(ssh_target(LOCAL_NODE).is_err());
        assert!(ssh_target("unknown").is_err());
    }

    #[test]
    fn gate_requires_evidence() {
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, false, None, false, false).is_none());
    }

    #[test]
    fn plan_and_receipt_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        std::fs::create_dir_all(root.join("ops"))?;
        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: "test".into(),
            entries: vec![json!({"dir": "x", "mtime_ms": 1, "basis": ["b"]})],
            candidate_names: vec!["jte-a".into()],
            estimated_bytes: 42,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 3,
        };
        let path = write_plan(root, "gc-plan-test.v1.json", &plan)?;
        let loaded: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_eq!(loaded["candidate_count"], 1);
        assert_eq!(loaded["schema"], "JopeSelfplayGcPlan.v1");
        append_receipt(root, json!({"ts_ms": 1, "deleted_count": 1}))?;
        let receipts = std::fs::read_to_string(root.join("ops/gc-events.v1.jsonl"))?;
        assert_eq!(receipts.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn ingested_set_fixture() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        std::fs::create_dir_all(root.join("corpus"))?;
        let connection = Connection::open(root.join("corpus/strict-test-exact.sqlite"))?;
        connection.execute_batch(
            "CREATE TABLE ingested_jobs(job_id TEXT PRIMARY KEY);
             INSERT INTO ingested_jobs VALUES ('jte-known');",
        )?;
        drop(connection);
        let found = ingested_set(root, &["jte-known".into(), "jte-unknown".into()]);
        assert!(found.contains("jte-known"));
        assert!(!found.contains("jte-unknown"));
        Ok(())
    }

    /// Build a trainer run directory the way scq lays one out, with a fake
    /// 12 GB-standin checkpoint next to the metrics.
    fn trainer_run(runs: &Path, run_id: &str, rows: &[&str]) -> PathBuf {
        let outputs = runs
            .join(run_id)
            .join("attempts/0/outputs/training/watermark-001805437");
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(outputs.join("train_metrics.jsonl"), rows.join("\n") + "\n").unwrap();
        std::fs::write(outputs.join("best.model.pt"), vec![0u8; 4096]).unwrap();
        runs.join(run_id)
    }

    #[test]
    fn guard_archives_metrics_before_the_directory_is_pruned() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-a-g8";
        let run_dir = trainer_run(
            &runs,
            run_id,
            &[
                "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}",
                "{\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}",
            ],
        );

        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: LOCAL_NODE.into(),
            entries: vec![],
            candidate_names: vec![run_id.to_string()],
            estimated_bytes: 4096,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 1,
        };
        let (deleted, blocked) = execute_plan(LOCAL_NODE, &runs.to_string_lossy(), &root, &plan)?;
        assert_eq!(deleted, 1, "the 12 GB directory must still be reclaimed");
        assert_eq!(blocked, 0);
        assert!(!run_dir.exists(), "run dir pruned");

        // The metrics outlived the directory that held them.
        let archive = th::archive_root(&root);
        let key = th::SegmentKey {
            run_id: run_id.into(),
            attempt: "0".into(),
            watermark: "watermark-001805437".into(),
        };
        let archived = std::fs::read_to_string(th::SegmentState::metrics_path(&th::segment_dir(
            &archive, &key,
        )))?;
        assert_eq!(archived.lines().count(), 2);
        assert!(archived.contains("\"val_loss\":3.3"));
        Ok(())
    }

    #[test]
    fn guard_refuses_to_prune_when_the_metrics_cannot_be_archived() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-c";
        let run_dir = trainer_run(&runs, run_id, &["{\"epoch\":1,\"val_loss\":9.6}"]);

        // Someone else holds the capture lock and never lets go: the collector
        // cannot archive, so it must not delete.
        let archive = th::archive_root(&root);
        let held = th::CaptureLock::acquire(&archive, 1, 0)?;
        assert!(held.is_some());

        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: LOCAL_NODE.into(),
            entries: vec![],
            candidate_names: vec![run_id.to_string()],
            estimated_bytes: 4096,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 1,
        };
        let guard = guard_metrics(
            LOCAL_NODE,
            &runs.to_string_lossy(),
            &root,
            &plan.candidate_names,
        );
        assert!(guard.safe.is_empty(), "must not be cleared for deletion");
        assert_eq!(guard.blocked.len(), 1);
        assert_eq!(guard.blocked[0]["reason"], "capture_lock_unavailable");
        assert!(run_dir.exists(), "directory survives a blocked guard");

        // Releasing the lock lets the next pass archive and then prune.
        drop(held);
        let (deleted, blocked) = execute_plan(LOCAL_NODE, &runs.to_string_lossy(), &root, &plan)?;
        assert_eq!((deleted, blocked), (1, 0));
        assert!(!run_dir.exists());
        Ok(())
    }

    #[test]
    fn dirs_without_metrics_are_unaffected_by_the_guard() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        // An ordinary eval run dir: no metrics, so nothing to protect.
        std::fs::create_dir_all(runs.join("jte-eval-1/attempts/0/outputs"))?;
        let guard = guard_metrics(
            LOCAL_NODE,
            &runs.to_string_lossy(),
            &root,
            &["jte-eval-1".to_string()],
        );
        assert_eq!(guard.safe, vec!["jte-eval-1".to_string()]);
        assert!(guard.blocked.is_empty());
        assert_eq!(guard.archived_rows, 0);
        Ok(())
    }

    #[test]
    fn guard_is_incremental_across_passes() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-a-g9";
        trainer_run(&runs, run_id, &["{\"epoch\":1}"]);
        let names = vec![run_id.to_string()];
        let first = guard_metrics(LOCAL_NODE, &runs.to_string_lossy(), &root, &names);
        assert_eq!(first.archived_rows, 1);
        // Nothing new on the second pass: no bytes re-read, no rows re-appended.
        let second = guard_metrics(LOCAL_NODE, &runs.to_string_lossy(), &root, &names);
        assert_eq!(second.archived_rows, 0);
        assert_eq!(second.safe.len(), 1);
        Ok(())
    }

    #[test]
    fn local_enumeration_and_delete() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(runs.join("jte-old"))?;
        std::fs::create_dir_all(runs.join("keepme"))?;
        let rows = enumerate_dirs(LOCAL_NODE, &runs.to_string_lossy())?;
        assert_eq!(rows.len(), 2);
        delete_batch(LOCAL_NODE, &runs.to_string_lossy(), &["jte-old".into()])?;
        assert!(!runs.join("jte-old").exists());
        assert!(runs.join("keepme").exists());
        Ok(())
    }
}
