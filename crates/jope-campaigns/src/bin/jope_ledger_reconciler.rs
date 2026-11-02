//! JOPE three-ledger reconciler.
//!
//! The self-play queue keeper keeps FOUR tables across TWO databases that must
//! agree, and updates them in non-atomic sequence:
//!
//!   $R/ops/backlog.sqlite       claims           (one row per manifest)
//!                               keeper_attempts  (one row per pair-attempt)
//!   $R/ops/pair-registry.sqlite pairs            (one row per protocol pair)
//!                               attempts         (one row per pair-attempt)
//!
//! Two structural divergences are known and both idle the whole cluster:
//!
//! FILL — `reserve_manifest_pairs_in_tx` (jope_selfplay_queue.rs:1333-1487)
//!   writes `pairs` and `events` only; it never inserts the first-attempt
//!   `attempts` row. The single Rust inserter is `append_native_retry_attempts`
//!   (:2258-2321), reachable only from `mark_infrastructure_retry` (:1317),
//!   which needs an already-`submitted` claim whose scq job failed with an
//!   infrastructure class. So a freshly reserved manifest reaches
//!   `update_native_attempts_submitted` (:2194-2214) with ZERO attempts rows,
//!   `changed` is 0, `changed != manifest.pairs.len()` and the cycle `bail!`s.
//!   Because that error escapes `cycle()`, ONE manifest aborts submission for
//!   the entire cluster.
//!
//! REAP — the pair registry releases a pair (`pairs.state='available'`,
//!   `manifest_sha256=NULL`) via freshness (jope_selfplay_freshness.rs:1209)
//!   and jope_queue_control.rs:3010, but nothing clears the corresponding
//!   `keeper_attempts` row out of the active set. That fossil then blocks the
//!   partial unique index `keeper_one_active_attempt` forever and
//!   `append_keeper_attempts` (:2101-2167) `bail!`s with "protocol pair has a
//!   conflicting active queue attempt".
//!
//! This binary reconciles both directions. It is bounded, idempotent,
//! receipted, and fully reversible: every mutation is journalled with its prior
//! value and `--revert <batch>` undoes exactly that batch.
//!
//! It NEVER mutates `pairs` or `claims`, and it never resurrects a pair whose
//! evaluation evidence already exists (`pairs.receipt_sha256` set, or a
//! `completed` attempts row for the same manifest).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde_json::json;

/// States in `keeper_attempts` that occupy the `keeper_one_active_attempt`
/// partial unique index. Mirrors jope_selfplay_queue.rs:2119-2121.
const KEEPER_ACTIVE_STATES: [&str; 4] = ["checked", "submitting", "submitted", "deferred_horizon"];

/// State the reaper parks a fossil keeper_attempts row in. Deliberately outside
/// every filter the keeper uses, so the row stops holding the unique index.
const KEEPER_RETIRED_STATE: &str = "retired_stale";

/// Registry pair states that the keeper treats as launch-ready
/// (jope_selfplay_queue.rs:796) and that `update_native_attempts_submitted`
/// transitions out of (:2206).
const PAIR_LAUNCH_READY: [&str; 2] = ["reserved", "infrastructure_retry"];

/// Claim states the keeper's `ready_claims` path will submit.
const SUBMITTABLE_CLAIM_STATES: [&str; 2] = ["checked", "submitting"];

/// Registry attempt states that count as "this pair already has a live attempt".
/// `reserved`/`submitted` are the partial unique index on the registry side;
/// `infrastructure_retry` is additionally matched by :2206 so a second row
/// would overshoot the count just as badly as a missing one.
const ATTEMPT_ACTIVE_STATES: [&str; 3] = ["reserved", "submitted", "infrastructure_retry"];

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Lane {
    Fill,
    Reap,
    Both,
}

#[derive(Parser, Debug)]
#[command(
    name = "jope_ledger_reconciler",
    about = "Reconcile backlog.sqlite {claims,keeper_attempts} against pair-registry.sqlite {pairs,attempts}"
)]
struct Args {
    /// Live self-play root ($R).
    #[arg(long)]
    root: PathBuf,

    /// Which lane to run.
    #[arg(long, value_enum, default_value = "both")]
    lane: Lane,

    /// Actually write. Without this the run is a read-only dry run that still
    /// emits a receipt.
    #[arg(long)]
    apply: bool,

    /// Maximum registry `attempts` rows to insert in one run.
    #[arg(long, default_value_t = 100_000)]
    max_fill: usize,

    /// Maximum `keeper_attempts` rows to retire in one run.
    #[arg(long, default_value_t = 500_000)]
    max_reap: usize,

    /// Rows per write transaction. Keeps the live-database write lock short.
    #[arg(long, default_value_t = 2_000)]
    batch_rows: usize,

    /// SQLite busy timeout in milliseconds.
    #[arg(long, default_value_t = 180_000)]
    busy_timeout_ms: u64,

    /// Which `claims.state` values the fill lane considers. These are the
    /// states `submit_claims` can pick up, so they are the only ones whose
    /// missing attempts rows can abort a keeper cycle.
    #[arg(long, default_value = "checked,submitting", value_delimiter = ',')]
    claim_states: Vec<String>,

    /// Undo a previous batch by id, using its journal.
    #[arg(long)]
    revert: Option<String>,

    /// Directory for receipts and journals.
    #[arg(long)]
    receipt_dir: Option<PathBuf>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the unix epoch")
        .as_millis() as u64
}

fn open_rw(path: &Path, busy_timeout_ms: u64) -> Result<Connection> {
    let connection = Connection::open(path)
        .with_context(|| format!("open sqlite read-write {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_millis(busy_timeout_ms))?;
    Ok(connection)
}

fn open_ro(path: &Path, busy_timeout_ms: u64) -> Result<Connection> {
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("open sqlite read-only {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_millis(busy_timeout_ms))?;
    Ok(connection)
}

/// Append-only journal. Every mutation is recorded with enough prior state to
/// be undone exactly.
struct Journal {
    handle: Option<File>,
}

impl Journal {
    fn create(path: PathBuf, apply: bool) -> Result<Self> {
        let handle = if apply {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            Some(
                File::create(&path)
                    .with_context(|| format!("create journal {}", path.display()))?,
            )
        } else {
            None
        };
        Ok(Self { handle })
    }

    fn record(&mut self, value: &serde_json::Value) -> Result<()> {
        let Some(handle) = self.handle.as_mut() else {
            return Ok(());
        };
        writeln!(handle, "{value}")?;
        Ok(())
    }

    /// The journal must be durable BEFORE the rows it describes are committed,
    /// otherwise a crash between commit and flush loses the reversal.
    fn sync(&mut self) -> Result<()> {
        if let Some(handle) = self.handle.as_mut() {
            handle.flush()?;
            handle.sync_all()?;
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
struct FillStats {
    candidate_manifests: usize,
    pairs_examined: usize,
    inserted: usize,
    skipped_already_active: usize,
    skipped_evidence: usize,
    skipped_pair_not_launch_ready: usize,
    manifests_completed: usize,
    truncated_by_limit: bool,
}

#[derive(Default, Debug)]
struct ReapStats {
    manifests_examined: usize,
    manifests_released: usize,
    retired: usize,
    truncated_by_limit: bool,
}

/// A registry pair that needs a first-attempt `attempts` row.
struct MissingAttempt {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    protocol_sha256: String,
    state: String,
    reservation_owner: String,
    manifest_path: Option<String>,
    manifest_sha256: String,
}

mod fill;
mod reap;
mod revert;

use fill::run_fill;
use reap::run_reap;
use revert::run_revert;

fn main() -> Result<()> {
    let args = Args::parse();
    let receipt_dir = args
        .receipt_dir
        .clone()
        .unwrap_or_else(|| args.root.join("ops/ledger-reconciler-receipts"));
    fs::create_dir_all(&receipt_dir)?;

    if let Some(batch_id) = args.revert.clone() {
        let report = run_revert(&args, &batch_id, &receipt_dir)?;
        println!("{report}");
        return Ok(());
    }

    let started_at_ms = now_ms();
    let batch_id = format!("b{started_at_ms}");
    let journal_path = receipt_dir.join(format!("{batch_id}.journal.jsonl"));
    let mut journal = Journal::create(journal_path.clone(), args.apply)?;

    let backlog_path = args.root.join("ops/backlog.sqlite");
    let registry_path = args.root.join("ops/pair-registry.sqlite");

    let mut fill = FillStats::default();
    let mut reap = ReapStats::default();

    if matches!(args.lane, Lane::Fill | Lane::Both) {
        let backlog = open_ro(&backlog_path, args.busy_timeout_ms)?;
        let mut registry = if args.apply {
            open_rw(&registry_path, args.busy_timeout_ms)?
        } else {
            open_ro(&registry_path, args.busy_timeout_ms)?
        };
        fill = run_fill(&backlog, &mut registry, &args, &batch_id, &mut journal)?;
    }

    if matches!(args.lane, Lane::Reap | Lane::Both) {
        let registry = open_ro(&registry_path, args.busy_timeout_ms)?;
        let mut backlog = if args.apply {
            open_rw(&backlog_path, args.busy_timeout_ms)?
        } else {
            open_ro(&backlog_path, args.busy_timeout_ms)?
        };
        reap = run_reap(&mut backlog, &registry, &args, &batch_id, &mut journal)?;
    }

    journal.sync()?;

    let receipt = json!({
        "schema": "JopeLedgerReconcilerReceipt.v1",
        "batch_id": batch_id,
        "generated_at_ms": now_ms(),
        "started_at_ms": started_at_ms,
        "apply": args.apply,
        "lane": format!("{:?}", args.lane).to_lowercase(),
        "claim_states": args.claim_states,
        "root": args.root.to_string_lossy(),
        "fill": {
            "candidate_manifests": fill.candidate_manifests,
            "manifests_completed": fill.manifests_completed,
            "pairs_examined": fill.pairs_examined,
            "attempts_inserted": fill.inserted,
            "skipped_already_active": fill.skipped_already_active,
            "skipped_evidence": fill.skipped_evidence,
            "skipped_pair_not_launch_ready": fill.skipped_pair_not_launch_ready,
            "truncated_by_limit": fill.truncated_by_limit,
        },
        "reap": {
            "manifests_examined": reap.manifests_examined,
            "manifests_released_by_registry": reap.manifests_released,
            "keeper_attempts_retired": reap.retired,
            "truncated_by_limit": reap.truncated_by_limit,
        },
        "journal": journal_path.to_string_lossy(),
        "reversal": format!(
            "jope_ledger_reconciler --root {} --revert {} --apply",
            args.root.to_string_lossy(),
            batch_id
        ),
        "keeper_active_states": KEEPER_ACTIVE_STATES,
        "attempt_active_states": ATTEMPT_ACTIVE_STATES,
    });

    let receipt_path = receipt_dir.join(format!("{batch_id}.json"));
    fs::write(
        &receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )?;
    // Stable pointer for the timer/alarms to read without globbing.
    fs::write(
        receipt_dir.join("latest.json"),
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )?;
    println!("{receipt}");
    Ok(())
}

#[cfg(test)]
mod tests;
