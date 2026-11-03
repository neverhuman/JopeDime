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
use jope_dime_regressor::paths;
use jope_dime_regressor::training_history as th;
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
#[path = "jope_selfplay_gc/planning.rs"]
mod planning;
pub(crate) use planning::*;
#[path = "jope_selfplay_gc/metrics_guard.rs"]
mod metrics_guard;
pub(crate) use metrics_guard::*;
#[path = "jope_selfplay_gc/execution.rs"]
mod execution;
pub(crate) use execution::*;

#[path = "jope_selfplay_gc/commands.rs"]
mod commands;
use commands::{cmd_receipts, cmd_runs};

fn main() -> Result<()> {
    let args = Args::parse();
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
#[path = "jope_selfplay_gc/tests.rs"]
mod tests;
