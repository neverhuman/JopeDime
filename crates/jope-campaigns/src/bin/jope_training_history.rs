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

mod capture;
use capture::{capture, parse_nodes, Cfg};

/// Every live segment whose archived byte count is behind the source.
///
/// A trailing partial row that has already been RECORDED as a truncation is not
/// counted: the complete rows are archived, the hole is on the ledger, and the
/// bytes will never arrive. A partial row that has not been recorded yet (the
/// trainer is mid-write, or the tail is not yet quiescent) still counts, which
/// is what keeps `verify` honest about a live run.
fn uncovered(root: &Path, runs_root: &Path) -> Result<Vec<Value>> {
    let archive = th::archive_root(root);
    let mut behind = Vec::new();
    for run_id in th::discover_trainer_runs(runs_root)? {
        for found in th::scan_run_dir(runs_root, &run_id) {
            let held = th::archived_bytes(&archive, &found.key);
            if held >= found.len {
                continue;
            }
            let recorded = match th::SegmentState::load(&th::segment_dir(&archive, &found.key))? {
                Some(state) => {
                    state.truncation_count > 0 && held + state.truncated_tail_bytes >= found.len
                }
                None => false,
            };
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
    Ok(behind)
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
            let behind = uncovered(Path::new(&root), Path::new(&runs_root))?;
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
mod tests;
