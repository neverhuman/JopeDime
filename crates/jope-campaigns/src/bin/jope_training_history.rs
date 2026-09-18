//! jope_training_history — continuous capture of DL trainer per-epoch metrics
//! into the durable archive at `test_exact_gen0/training/history/`.
//!
//! Driven by jope-training-history.timer every 3 minutes. The trainer emits one
//! epoch roughly every 4 minutes, so a crashed, cancelled or garbage-collected
//! run can lose at most the epoch it was mid-write on.
//!
//! CLUSTER-WIDE. The trainer specs pin `hostname -qxE "xbabe1|xbabe2|xbabe3"`,
//! so a run can land on a node this process is not on. Capture therefore sweeps
//! the local runs root first and then pulls each remote node's trainer metrics
//! over ssh, incrementally (one enumeration round trip per node, then one more
//! only for segments whose size/mtime changed). The local path never waits on a
//! remote: every ssh is wrapped in `timeout(1)`, a node that fails is recorded
//! in `nodes.v1.json` and named in the receipt, and its already-archived
//! segments are NOT sealed — an unreachable node is never mistaken for a node
//! whose runs were pruned.
//!
//! Read-only with respect to the trainer: run directories are opened for reading
//! and nothing under `var/runs` is created, moved, locked or truncated. Always
//! exits 0 in `capture` mode — a capture unit that can fail is a second thing to
//! monitor; failures are reported through receipts and the health alarm instead.

use anyhow::Result;
use clap::{Parser, Subcommand};
use jope_dime_regressor::paths;
use jope_dime_regressor::training_history::{self as th, SourceReader};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn default_runs_root_str => paths::scq_runs_root()
);
const LOCAL_NODE: &str = th::LOCAL_NODE;
/// Remote sweep is ON by default, because a trainer landing on xbabe1 or xbabe3
/// must not depend on anyone remembering to pass a flag.
const DEFAULT_REMOTE_NODES: &str = "xbabe1,xbabe3,xbabe4";
/// A cycle that changed nothing only gets a receipt this often. At 480 cycles a
/// day an unconditional receipt would add ~43 MB/year of "nothing happened" to
/// an archive whose entire point is staying small forever. Freshness is still
/// observable every cycle because `index.v1.json` is rewritten unconditionally,
/// and that mtime is what the health alarm reads.
const IDLE_RECEIPT_SECS: u64 = 3_600;

#[derive(Parser)]
#[command(
    name = "jope_training_history",
    about = "durable capture of trainer per-epoch metrics (archive never shrinks)"
)]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Sync every trainer run's metrics into the archive and rebuild the index.
    Capture {
        #[arg(long, default_value = default_root_str())]
        root: String,
        #[arg(long, default_value = default_runs_root_str())]
        runs_root: String,
        #[arg(long, default_value = LOCAL_NODE)]
        node: String,
        /// Comma-separated nodes to sweep over ssh. xbabe2 is rejected: it is
        /// the local host and `ssh xbabe2` reaches a different machine.
        #[arg(long, default_value = DEFAULT_REMOTE_NODES)]
        remote_nodes: String,
        /// Disable the remote sweep (local-only capture).
        #[arg(long, default_value_t = false)]
        no_remote: bool,
        /// Runs root on the remote nodes.
        #[arg(long, default_value = default_runs_root_str())]
        remote_runs_root: String,
        #[arg(long, default_value_t = th::SSH_DEADLINE_SECS)]
        ssh_deadline_secs: u64,
        /// Seconds a trailing INCOMPLETE row must be untouched before it is
        /// written off. Lower values are for sandbox proofs only.
        #[arg(long, default_value_t = th::TAIL_QUIESCENT_SECS)]
        tail_quiescent_secs: u64,
    },
    /// Print the archive summary (no writes).
    Status {
        #[arg(long, default_value = default_root_str())]
        root: String,
    },
    /// Exit non-zero if any live run directory holds rows the archive lacks.
    Verify {
        #[arg(long, default_value = default_root_str())]
        root: String,
        #[arg(long, default_value = default_runs_root_str())]
        runs_root: String,
    },
}

/// One capture pass' configuration.
struct Cfg {
    root: PathBuf,
    runs_root: PathBuf,
    node: String,
    remote_nodes: Vec<String>,
    remote_runs_root: String,
    ssh_deadline_secs: u64,
    /// How long a trailing partial row must sit untouched before it counts as
    /// abandoned. Overridable so the policy can be exercised in a sandbox
    /// without a 15 minute wait; production uses the default.
    quiescent_secs: u64,
}

fn parse_nodes(spec: &str) -> (Vec<String>, Vec<Value>) {
    let mut nodes = Vec::new();
    let mut rejected = Vec::new();
    for raw in spec.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        match th::ssh_argv(name, th::SSH_DEADLINE_SECS) {
            Ok(_) => {
                if !nodes.iter().any(|existing| existing == name) {
                    nodes.push(name.to_string());
                }
            }
            Err(error) => rejected.push(json!({"node": name, "error": error.to_string()})),
        }
    }
    (nodes, rejected)
}

/// Running totals for one capture pass.
#[derive(Default)]
struct Totals {
    appended_rows: u64,
    appended_bytes: u64,
    rotated: u64,
    errors: Vec<Value>,
    advanced: Vec<Value>,
    truncations: Vec<Value>,
}

/// Apply the truncated-tail policy to whatever the source has past the archive.
///
/// A trailing INCOMPLETE row is not data: the epoch never finished being
/// written. Every complete row is already archived by `ingest_source` above
/// this call, so all that is left is to make the hole explicit and permanent,
/// which is what buys the right for the collector to prune the run directory.
/// Anything that is NOT a lone unterminated fragment — complete rows still
/// missing, or a fragment too big to be one row — is deliberately not recorded,
/// so those keep blocking.
fn apply_tail_policy<R: th::SourceReader>(
    archive: &Path,
    key: &th::SegmentKey,
    reader: &R,
    node: &str,
    source_path: &str,
    outcome: &th::IngestOutcome,
    source_mtime_ms: u64,
    abandoned: bool,
) -> Option<Value> {
    if outcome.residual_bytes == 0 {
        return None;
    }
    // Still being written: this is a mid-write row, not an abandoned one.
    if !abandoned {
        return None;
    }
    let residual =
        th::classify_residual(reader, outcome.archived_bytes, outcome.source_bytes).ok()?;
    let th::Residual::PartialRow(fragment) = residual else {
        return None;
    };
    match th::record_truncation(
        archive,
        key,
        node,
        source_path,
        &fragment,
        outcome.source_bytes,
        source_mtime_ms,
    ) {
        Ok(true) => Some(json!({
            "run_id": key.run_id,
            "attempt": key.attempt,
            "watermark": key.watermark,
            "node": node,
            "residual_bytes": fragment.len(),
            "archived_rows": outcome.archived_rows,
            "last_epoch": outcome.last_epoch,
            "source_path": source_path,
        })),
        _ => None,
    }
}

/// Local runs root sweep. Unchanged in behaviour except that an abandoned
/// trailing partial row is now recorded instead of blocking forever.
fn capture_local(
    archive: &Path,
    cfg: &Cfg,
    seen: &mut BTreeSet<th::SegmentKey>,
    totals: &mut Totals,
) -> usize {
    let run_ids = th::discover_trainer_runs(&cfg.runs_root).unwrap_or_default();
    for run_id in &run_ids {
        for found in th::scan_run_dir(&cfg.runs_root, run_id) {
            seen.insert(found.key.clone());
            let source = found.path.to_string_lossy().into_owned();
            let reader = th::FileSource(found.path.clone());
            match th::ingest_source(archive, &found.key, &reader, &source, &cfg.node) {
                Ok(outcome) => {
                    totals.appended_rows += outcome.appended_rows;
                    totals.appended_bytes += outcome.appended_bytes;
                    if outcome.rotated {
                        totals.rotated += 1;
                    }
                    if outcome.appended_rows > 0 || outcome.rotated {
                        totals.advanced.push(json!({
                            "node": cfg.node,
                            "run_id": found.key.run_id,
                            "attempt": found.key.attempt,
                            "rows": outcome.appended_rows,
                            "bytes": outcome.appended_bytes,
                            "last_epoch": outcome.last_epoch,
                            "rotated": outcome.rotated,
                        }));
                    }
                    let mtime_ms = std::fs::metadata(&found.path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    // Local: one clock, so the file's own mtime is exact.
                    let abandoned =
                        th::tail_is_abandoned_after(mtime_ms, th::now_ms(), cfg.quiescent_secs);
                    if let Some(record) = apply_tail_policy(
                        archive, &found.key, &reader, &cfg.node, &source, &outcome, mtime_ms,
                        abandoned,
                    ) {
                        totals.truncations.push(record);
                    }
                }
                Err(error) => totals.errors.push(json!({
                    "node": cfg.node,
                    "run_id": found.key.run_id,
                    "attempt": found.key.attempt,
                    "error": error.to_string(),
                })),
            }
            // The final-selection record is a few hundred bytes and is what ties
            // a curve to the checkpoint that was actually shipped.
            if let Some(parent) = found.path.parent() {
                if let Ok(bytes) = std::fs::read(parent.join(th::RESULT_FILE)) {
                    let _ = th::archive_sidecar(archive, &found.key, th::RESULT_FILE, &bytes);
                }
            }
        }
    }
    run_ids.len()
}

/// Sweep one remote node. Returns Err only when the node could not be
/// enumerated at all — a per-segment failure is recorded and the rest of the
/// node still archives, because partial coverage beats none.
fn capture_remote_node(
    archive: &Path,
    cfg: &Cfg,
    node: &str,
    seen: &mut BTreeSet<th::SegmentKey>,
    totals: &mut Totals,
) -> Result<usize> {
    let listing = th::remote_scan(node, &cfg.remote_runs_root, cfg.ssh_deadline_secs)?;
    let results: BTreeMap<th::SegmentKey, th::RemoteFound> = listing
        .results
        .iter()
        .map(|found| (found.key.clone(), found.clone()))
        .collect();

    for found in &listing.metrics {
        // Claim the segment before doing any work: a key we have seen is never
        // sealed, so a mid-pass failure cannot make a live run look pruned.
        seen.insert(found.key.clone());
        let reader = th::RemoteSource {
            node: node.to_string(),
            runs_root: cfg.remote_runs_root.clone(),
            relative: found.relative.clone(),
            len: found.len,
            deadline_secs: cfg.ssh_deadline_secs,
        };
        let source = reader.display();
        let fingerprint = found.fingerprint();
        let prior = th::SegmentState::load(&th::segment_dir(archive, &found.key));
        let held = th::archived_bytes(archive, &found.key);
        // Already on the truncation ledger: the outstanding bytes are a row
        // that will never be finished, so there is nothing left to fetch.
        let settled = prior
            .as_ref()
            .map(|state| {
                state.truncation_count > 0 && held + state.truncated_tail_bytes >= found.len
            })
            .unwrap_or(false);
        // Skipping on an unchanged size+mtime is what keeps an idle remote
        // segment free. It is ONLY safe when nothing is outstanding: a killed
        // trainer's file keeps its mtime forever, so skipping while bytes are
        // still unarchived would mean its truncated tail is never recorded and
        // the directory blocks pruning for good — the exact deadlock this
        // workstream exists to remove.
        let unchanged = prior
            .as_ref()
            .map(|state| {
                !state.source_fingerprint.is_empty() && state.source_fingerprint == fingerprint
            })
            .unwrap_or(false)
            && (held >= found.len || settled);
        if !unchanged {
            match th::ingest_source_fingerprinted(
                archive,
                &found.key,
                &reader,
                &source,
                node,
                &fingerprint,
            ) {
                Ok(outcome) => {
                    totals.appended_rows += outcome.appended_rows;
                    totals.appended_bytes += outcome.appended_bytes;
                    if outcome.rotated {
                        totals.rotated += 1;
                    }
                    if outcome.appended_rows > 0 || outcome.rotated {
                        totals.advanced.push(json!({
                            "node": node,
                            "run_id": found.key.run_id,
                            "attempt": found.key.attempt,
                            "rows": outcome.appended_rows,
                            "bytes": outcome.appended_bytes,
                            "last_epoch": outcome.last_epoch,
                            "rotated": outcome.rotated,
                        }));
                    }
                    // Remote: judged on how long OUR clock has seen the
                    // size+mtime unchanged, never on the remote clock.
                    let abandoned = th::remote_tail_is_abandoned(
                        prior.as_ref(),
                        &fingerprint,
                        th::now_ms(),
                        cfg.quiescent_secs,
                    );
                    if let Some(record) = apply_tail_policy(
                        archive,
                        &found.key,
                        &reader,
                        node,
                        &source,
                        &outcome,
                        found.mtime_ms,
                        abandoned,
                    ) {
                        totals.truncations.push(record);
                    }
                }
                Err(error) => totals.errors.push(json!({
                    "node": node,
                    "run_id": found.key.run_id,
                    "attempt": found.key.attempt,
                    "source": source,
                    "error": error.to_string(),
                })),
            }
        }
        // train-result.json came back in the same enumeration, so it costs a
        // round trip only when its size differs from what we already hold.
        if let Some(result) = results.get(&found.key) {
            if th::sidecar_bytes(archive, &found.key, th::RESULT_FILE) != result.len {
                let sidecar = th::RemoteSource {
                    node: node.to_string(),
                    runs_root: cfg.remote_runs_root.clone(),
                    relative: result.relative.clone(),
                    len: result.len,
                    deadline_secs: cfg.ssh_deadline_secs,
                };
                match sidecar.read_range(0, result.len as usize) {
                    Ok(bytes) => {
                        let _ = th::archive_sidecar(archive, &found.key, th::RESULT_FILE, &bytes);
                    }
                    Err(error) => totals.errors.push(json!({
                        "node": node,
                        "run_id": found.key.run_id,
                        "sidecar": th::RESULT_FILE,
                        "error": error.to_string(),
                    })),
                }
            }
        }
    }
    Ok(listing.metrics.len())
}

/// One capture pass. Returns the receipt written for this cycle.
fn capture(cfg: &Cfg) -> Result<Value> {
    let root = cfg.root.as_path();
    let archive = th::archive_root(root);
    // Non-blocking: if the collector is mid-archive under the same lock, this
    // tick has nothing useful to add and the next one is only 3 minutes away.
    let Some(_lock) = th::CaptureLock::acquire(&archive, 1, 0)? else {
        let record = json!({
            "ts_ms": th::now_ms(),
            "event": "skipped_locked",
            "node": cfg.node,
        });
        th::append_event(&archive, record.clone())?;
        return Ok(record);
    };

    let mut seen: BTreeSet<th::SegmentKey> = BTreeSet::new();
    let mut totals = Totals::default();

    // LOCAL FIRST, ALWAYS. Whatever the remotes do, the node this process runs
    // on is already fully archived by the time any ssh is attempted.
    let runs_scanned = capture_local(&archive, cfg, &mut seen, &mut totals);

    // Nodes whose source-of-truth we actually observed this pass. Only these
    // may have segments sealed; an unreachable node tells us nothing about
    // whether its run directories still exist.
    let mut covered: BTreeSet<String> = BTreeSet::new();
    covered.insert(cfg.node.clone());
    let mut statuses = th::load_nodes(&archive);
    let mut node_reports: Vec<Value> = Vec::new();
    let mut skipped_nodes: Vec<Value> = Vec::new();

    for node in &cfg.remote_nodes {
        let started = th::now_ms();
        let status = statuses
            .entry(node.clone())
            .or_insert_with(|| th::NodeStatus {
                node: node.clone(),
                ..Default::default()
            });
        status.node = node.clone();
        status.last_attempt_ms = started;
        match capture_remote_node(&archive, cfg, node, &mut seen, &mut totals) {
            Ok(segments) => {
                status.reachable = true;
                status.last_success_ms = th::now_ms();
                status.consecutive_failures = 0;
                status.last_error.clear();
                status.segments_seen = segments as u64;
                covered.insert(node.clone());
                node_reports.push(json!({
                    "node": node,
                    "ok": true,
                    "segments": segments,
                    "elapsed_ms": th::now_ms().saturating_sub(started),
                }));
            }
            Err(error) => {
                status.reachable = false;
                status.consecutive_failures += 1;
                status.last_error = error.to_string();
                let stale_secs = if status.last_success_ms == 0 {
                    Value::Null
                } else {
                    json!(th::now_ms().saturating_sub(status.last_success_ms) / 1000)
                };
                let report = json!({
                    "node": node,
                    "ok": false,
                    "error": error.to_string(),
                    "consecutive_failures": status.consecutive_failures,
                    "last_success_ms": status.last_success_ms,
                    "stale_secs": stale_secs,
                    "elapsed_ms": th::now_ms().saturating_sub(started),
                });
                node_reports.push(report.clone());
                skipped_nodes.push(report);
            }
        }
    }
    // Only the nodes this pass is configured to sweep make a coverage claim.
    // A node dropped from the roster must not leave a stale entry warning
    // forever about coverage nobody is asking for any more.
    statuses.retain(|node, _| cfg.remote_nodes.iter().any(|name| name == node));
    let _ = th::save_nodes(&archive, &statuses);

    // Any archived segment whose source has gone is sealed, not deleted: this is
    // the state a run reaches after the collector reclaims its 12 GB. A segment
    // belonging to a node we could not reach is left alone.
    let mut sealed = 0u64;
    let mut seal_deferred = 0u64;
    for (key, state) in th::load_segments(&archive) {
        if seen.contains(&key) {
            continue;
        }
        let owner = if state.node.is_empty() {
            cfg.node.as_str()
        } else {
            state.node.as_str()
        };
        if !covered.contains(owner) {
            seal_deferred += 1;
            continue;
        }
        if th::seal_segment(&archive, &key).is_ok() {
            sealed += 1;
        }
    }

    let index = th::rebuild_index(root, &archive)?;
    let record = json!({
        "ts_ms": th::now_ms(),
        "event": "capture",
        "node": cfg.node,
        "runs_scanned": runs_scanned,
        "segments_live": seen.len(),
        "segments_total": index["segment_count"],
        "appended_rows": totals.appended_rows,
        "appended_bytes": totals.appended_bytes,
        "rotated": totals.rotated,
        "sealed_now": sealed,
        "seal_deferred": seal_deferred,
        "archive_rows": index["total_rows"],
        "archive_bytes": index["total_bytes"],
        "advanced": totals.advanced,
        "errors": totals.errors,
        "truncations": totals.truncations,
        "nodes": node_reports,
        // Named explicitly so a pass with reduced coverage can never read as a
        // clean success.
        "nodes_skipped": skipped_nodes,
        "coverage_complete": skipped_nodes.is_empty(),
    });
    let changed = totals.appended_rows > 0
        || totals.rotated > 0
        || sealed > 0
        || !totals.errors.is_empty()
        || !totals.truncations.is_empty()
        || !skipped_nodes.is_empty();
    if changed || idle_receipt_due(&archive) {
        th::append_event(&archive, record.clone())?;
    }
    Ok(record)
}

/// True when the receipts log has had nothing written to it for long enough
/// that a heartbeat is worth the bytes.
fn idle_receipt_due(archive: &Path) -> bool {
    let Ok(modified) = std::fs::metadata(archive.join(th::EVENTS_FILE)).and_then(|m| m.modified())
    else {
        return true;
    };
    std::time::SystemTime::now()
        .duration_since(modified)
        .map(|age| age.as_secs() >= IDLE_RECEIPT_SECS)
        .unwrap_or(true)
}

/// Every live segment whose archived byte count is behind the source.
///
/// A trailing partial row that has already been RECORDED as a truncation is not
/// counted: the complete rows are archived, the hole is on the ledger, and the
/// bytes will never arrive. A partial row that has not been recorded yet (the
/// trainer is mid-write, or the tail is not yet quiescent) still counts, which
/// is what keeps `verify` honest about a live run.
fn uncovered(root: &Path, runs_root: &Path) -> Vec<Value> {
    let archive = th::archive_root(root);
    let mut behind = Vec::new();
    for run_id in th::discover_trainer_runs(runs_root).unwrap_or_default() {
        for found in th::scan_run_dir(runs_root, &run_id) {
            let held = th::archived_bytes(&archive, &found.key);
            if held >= found.len {
                continue;
            }
            let recorded = th::SegmentState::load(&th::segment_dir(&archive, &found.key))
                .map(|state| {
                    state.truncation_count > 0 && held + state.truncated_tail_bytes >= found.len
                })
                .unwrap_or(false);
            if recorded {
                continue;
            }
            behind.push(json!({
                "run_id": found.key.run_id,
                "attempt": found.key.attempt,
                "watermark": found.key.watermark,
                "source_bytes": found.len,
                "archived_bytes": held,
                "residual": th::classify_residual(
                    &th::FileSource(found.path.clone()),
                    held,
                    found.len,
                )
                .map(|r| r.reason())
                .unwrap_or("unreadable"),
                "path": found.path.to_string_lossy(),
            }));
        }
    }
    behind
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Cmd::Capture {
            root,
            runs_root,
            node,
            remote_nodes,
            no_remote,
            remote_runs_root,
            ssh_deadline_secs,
            tail_quiescent_secs,
        } => {
            let root = PathBuf::from(root);
            let (nodes, rejected) = if no_remote {
                (Vec::new(), Vec::new())
            } else {
                parse_nodes(&remote_nodes)
            };
            if !rejected.is_empty() {
                let _ = th::append_event(
                    &th::archive_root(&root),
                    json!({
                        "ts_ms": th::now_ms(),
                        "event": "remote_nodes_rejected",
                        "rejected": rejected,
                    }),
                );
            }
            let cfg = Cfg {
                root: root.clone(),
                runs_root: PathBuf::from(runs_root),
                node,
                remote_nodes: nodes,
                remote_runs_root,
                ssh_deadline_secs,
                quiescent_secs: tail_quiescent_secs,
            };
            match capture(&cfg) {
                Ok(record) => println!("{record}"),
                Err(error) => {
                    // Never fail the unit; the alarm notices a stalled archive.
                    let record = json!({
                        "ts_ms": th::now_ms(),
                        "event": "capture_failed",
                        "error": error.to_string(),
                    });
                    let _ = th::append_event(&th::archive_root(&root), record.clone());
                    println!("{record}");
                }
            }
            Ok(())
        }
        Cmd::Status { root } => {
            let archive = th::archive_root(Path::new(&root));
            let text = std::fs::read_to_string(archive.join(th::INDEX_FILE))
                .unwrap_or_else(|_| "{}".to_string());
            let index: Value = serde_json::from_str(&text).unwrap_or(json!({}));
            println!(
                "{}",
                json!({
                    "archive_root": archive.to_string_lossy(),
                    "schema": index.get("schema"),
                    "generated_at_ms": index.get("generated_at_ms"),
                    "run_count": index.get("run_count"),
                    "segment_count": index.get("segment_count"),
                    "total_rows": index.get("total_rows"),
                    "total_bytes": index.get("total_bytes"),
                })
            );
            Ok(())
        }
        Cmd::Verify { root, runs_root } => {
            let behind = uncovered(Path::new(&root), Path::new(&runs_root));
            println!("{}", json!({"uncovered": behind.len(), "segments": behind}));
            if behind.is_empty() {
                Ok(())
            } else {
                anyhow::bail!("{} live segment(s) not fully archived", behind.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use tempfile::TempDir;

    /// Local-only capture, i.e. the pre-existing behaviour, for tests that are
    /// about the local path. Remote coverage has its own tests.
    fn local(root: &Path, runs: &Path) -> Cfg {
        Cfg {
            root: root.to_path_buf(),
            runs_root: runs.to_path_buf(),
            node: "xbabe2".into(),
            remote_nodes: Vec::new(),
            remote_runs_root: runs.to_string_lossy().into_owned(),
            ssh_deadline_secs: 5,
            quiescent_secs: th::TAIL_QUIESCENT_SECS,
        }
    }

    fn write_run(runs: &Path, run_id: &str, attempt: &str, body: &str) -> PathBuf {
        let outputs = runs
            .join(run_id)
            .join("attempts")
            .join(attempt)
            .join("outputs/training/watermark-001805437");
        std::fs::create_dir_all(&outputs).unwrap();
        let path = outputs.join(th::METRICS_FILE);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn capture_is_incremental_and_survives_the_run_dir_disappearing() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g9";
        let source = write_run(
            &runs,
            run_id,
            "1",
            "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n",
        );

        let first = capture(&local(&root, &runs))?;
        assert_eq!(first["appended_rows"], 1);
        assert_eq!(first["archive_rows"], 1);

        // Second cycle, one more epoch: only the delta is read and appended.
        std::fs::write(
            &source,
            "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n{\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}\n",
        )?;
        let second = capture(&local(&root, &runs))?;
        assert_eq!(second["appended_rows"], 1);
        assert_eq!(second["archive_rows"], 2);

        // Idempotent: nothing new appended when nothing changed.
        let third = capture(&local(&root, &runs))?;
        assert_eq!(third["appended_rows"], 0);
        assert_eq!(third["archive_rows"], 2);
        assert!(uncovered(&root, &runs).is_empty());

        // The collector reclaims the 12 GB run directory. The rows stay.
        std::fs::remove_dir_all(runs.join(run_id))?;
        let after = capture(&local(&root, &runs))?;
        assert_eq!(after["archive_rows"], 2, "archive keeps the epochs");
        assert_eq!(after["sealed_now"], 1);
        let index: Value = serde_json::from_str(&std::fs::read_to_string(
            th::archive_root(&root).join(th::INDEX_FILE),
        )?)?;
        assert_eq!(index["total_rows"], 2);
        assert_eq!(index["segments"][0]["source_present"], false);
        assert_eq!(index["segments"][0]["last_epoch"], 2);
        Ok(())
    }

    #[test]
    fn verify_reports_a_run_that_is_ahead_of_the_archive() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-c";
        write_run(&runs, run_id, "0", "{\"epoch\":1}\n");
        // Nothing captured yet, so the run is uncovered.
        assert_eq!(uncovered(&root, &runs).len(), 1);
        capture(&local(&root, &runs))?;
        assert!(uncovered(&root, &runs).is_empty());
        Ok(())
    }

    #[test]
    fn multiple_attempts_are_archived_separately() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-c-r000001";
        write_run(&runs, run_id, "0", "{\"epoch\":1}\n");
        write_run(&runs, run_id, "1", "{\"epoch\":1}\n{\"epoch\":2}\n");
        let record = capture(&local(&root, &runs))?;
        assert_eq!(record["segments_live"], 2);
        assert_eq!(record["archive_rows"], 3);
        Ok(())
    }

    #[test]
    fn idle_cycles_do_not_grow_the_receipts_log() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        write_run(
            &runs,
            "jope-selfplay-train-001805437-c",
            "0",
            "{\"epoch\":1}\n",
        );
        let events = th::archive_root(&root).join(th::EVENTS_FILE);

        capture(&local(&root, &runs))?; // wrote a row -> receipted
        let after_work = std::fs::read_to_string(&events)?.lines().count();
        assert_eq!(after_work, 1);

        // Three cycles that change nothing must add nothing...
        for _ in 0..3 {
            capture(&local(&root, &runs))?;
        }
        assert_eq!(std::fs::read_to_string(&events)?.lines().count(), 1);

        // ...but the index is still refreshed every cycle, which is what the
        // health alarm reads for freshness.
        assert!(th::archive_root(&root).join(th::INDEX_FILE).exists());

        // Once the log goes stale a heartbeat lands.
        let old = SystemTime::now() - std::time::Duration::from_secs(IDLE_RECEIPT_SECS + 60);
        std::fs::File::options()
            .write(true)
            .open(&events)?
            .set_times(std::fs::FileTimes::new().set_modified(old))?;
        capture(&local(&root, &runs))?;
        assert_eq!(std::fs::read_to_string(&events)?.lines().count(), 2);
        Ok(())
    }

    /// Backdate a file so its trailing fragment reads as abandoned.
    fn age_file(path: &Path, secs: u64) {
        let when = SystemTime::now() - std::time::Duration::from_secs(secs);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
    }

    #[test]
    fn an_abandoned_partial_row_is_written_off_but_every_complete_row_survives() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g9";
        // Two finished epochs plus the row the trainer was killed mid-write on.
        let source = write_run(
            &runs,
            run_id,
            "1",
            "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n\
             {\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}\n\
             {\"epoch\":3,\"train_lo",
        );

        // While the file is still fresh the fragment is treated as mid-write:
        // nothing is written off and the segment is honestly reported behind.
        let live = capture(&local(&root, &runs))?;
        assert_eq!(live["appended_rows"], 2);
        assert_eq!(live["truncations"].as_array().unwrap().len(), 0);
        assert_eq!(uncovered(&root, &runs).len(), 1);
        assert_eq!(uncovered(&root, &runs)[0]["residual"], "partial_row");

        // Once the writer has been gone long enough, the hole is recorded.
        age_file(&source, th::TAIL_QUIESCENT_SECS + 120);
        let after = capture(&local(&root, &runs))?;
        let truncations = after["truncations"].as_array().unwrap();
        assert_eq!(truncations.len(), 1);
        assert_eq!(truncations[0]["residual_bytes"], 20);
        assert_eq!(after["archive_rows"], 2, "complete rows kept");
        // ...and the run directory is no longer reported as holding rows we lack.
        assert!(uncovered(&root, &runs).is_empty());

        // Recording is idempotent: the next tick adds nothing.
        let again = capture(&local(&root, &runs))?;
        assert_eq!(again["truncations"].as_array().unwrap().len(), 0);
        let ledger = std::fs::read_to_string(th::archive_root(&root).join(th::TRUNCATIONS_FILE))?;
        assert_eq!(ledger.lines().count(), 1);

        // A shortfall that spans COMPLETE rows is never written off, even when
        // the file is equally stale.
        let run_two = "jope-selfplay-train-001805437-c";
        let second = write_run(
            &runs,
            run_two,
            "0",
            "{\"epoch\":1,\"train_loss\":1.0}\n{\"epoch\":2,\"train_loss\":0.9}\n",
        );
        age_file(&second, th::TAIL_QUIESCENT_SECS + 120);
        // Archive only the first row, then make the archive look behind.
        capture(&local(&root, &runs))?;
        let key = th::SegmentKey {
            run_id: run_two.into(),
            attempt: "0".into(),
            watermark: "watermark-001805437".into(),
        };
        let archived =
            th::SegmentState::metrics_path(&th::segment_dir(&th::archive_root(&root), &key));
        let body = std::fs::read(&archived)?;
        let keep = body.iter().position(|b| *b == b'\n').unwrap() as u64 + 1;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&archived)?
            .set_len(keep)?;
        let reader = th::FileSource(second.clone());
        let residual = th::classify_residual(&reader, keep, std::fs::metadata(&second)?.len())?;
        assert!(
            matches!(residual, th::Residual::CompleteRowsMissing { .. }),
            "a missing complete row must never look like a truncated tail: {residual:?}"
        );
        Ok(())
    }

    #[test]
    fn an_unreachable_node_is_reported_and_its_segments_are_not_sealed() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let archive = th::archive_root(&root);
        // A segment that was captured from xbabe1 on some earlier pass.
        let key = th::SegmentKey {
            run_id: "jope-selfplay-train-001805437-b".into(),
            attempt: "0".into(),
            watermark: "watermark-001805437".into(),
        };
        th::ingest_source(
            &archive,
            &key,
            &th::BytesSource(b"{\"epoch\":1,\"train_loss\":2.0}\n".to_vec()),
            "xbabe1:/runs/x",
            "xbabe1",
        )?;

        // An unusable remote runs root fails enumeration without any ssh, which
        // is exactly the shape of an unreachable node.
        let cfg = Cfg {
            root: root.clone(),
            runs_root: runs.clone(),
            node: "xbabe2".into(),
            remote_nodes: vec!["xbabe1".into()],
            remote_runs_root: "/not a valid root".into(),
            ssh_deadline_secs: 5,
            quiescent_secs: th::TAIL_QUIESCENT_SECS,
        };
        let record = capture(&cfg)?;
        assert_eq!(
            record["coverage_complete"], false,
            "must not read as success"
        );
        assert_eq!(record["nodes_skipped"].as_array().unwrap().len(), 1);
        assert_eq!(record["nodes_skipped"][0]["node"], "xbabe1");
        assert_eq!(record["sealed_now"], 0);
        assert_eq!(record["seal_deferred"], 1);
        let state = th::SegmentState::load(&th::segment_dir(&archive, &key)).unwrap();
        assert!(
            state.source_present,
            "an unreachable node says nothing about whether its runs still exist"
        );

        // The failure is durable and visible without re-running ssh.
        let nodes = th::load_nodes(&archive);
        assert_eq!(nodes["xbabe1"].consecutive_failures, 1);
        assert!(!nodes["xbabe1"].reachable);
        assert!(!nodes["xbabe1"].last_error.is_empty());
        Ok(())
    }

    #[test]
    fn the_local_host_is_never_an_ssh_target() {
        let (nodes, rejected) = parse_nodes("xbabe1,xbabe2,xbabe3,xbabe4,xbabe1");
        assert_eq!(nodes, vec!["xbabe1", "xbabe3", "xbabe4"]);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0]["node"], "xbabe2");
        let (nodes, _) = parse_nodes(DEFAULT_REMOTE_NODES);
        assert!(!nodes.iter().any(|node| node == LOCAL_NODE));
    }

    #[test]
    fn a_concurrent_holder_makes_the_tick_skip_not_corrupt() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        write_run(
            &runs,
            "jope-selfplay-train-001805437-c",
            "0",
            "{\"epoch\":1}\n",
        );
        let archive = th::archive_root(&root);
        let held = th::CaptureLock::acquire(&archive, 1, 0)?;
        assert!(held.is_some());
        let record = capture(&local(&root, &runs))?;
        assert_eq!(record["event"], "skipped_locked");
        drop(held);
        assert_eq!(capture(&local(&root, &runs))?["appended_rows"], 1);
        Ok(())
    }
}
