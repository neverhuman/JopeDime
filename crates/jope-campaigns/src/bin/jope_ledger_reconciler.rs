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

/// Manifests the keeper will try to submit this cycle. These are exactly the
/// claims that `ready_claims`/`submit_claims` can pick up, so they are the only
/// ones whose missing attempts rows can abort a cycle.
fn active_claim_manifests(backlog: &Connection, claim_states: &[String]) -> Result<Vec<String>> {
    if claim_states.is_empty() {
        bail!("--claim-states must name at least one claim state");
    }
    let placeholders = vec!["?"; claim_states.len()].join(",");
    let mut statement = backlog.prepare(&format!(
        "SELECT manifest_sha256 FROM claims WHERE state IN ({placeholders}) \
         ORDER BY manifest_sha256"
    ))?;
    let rows = statement.query_map(rusqlite::params_from_iter(claim_states), |row| {
        row.get::<_, String>(0)
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Collect the pairs of one manifest that are launch-ready in `pairs` but have
/// no live row in `attempts`. Read-only; the caller decides whether to write.
fn missing_attempts_for_manifest(
    registry: &Connection,
    manifest_sha256: &str,
    stats: &mut FillStats,
) -> Result<Vec<MissingAttempt>> {
    let mut pair_statement = registry.prepare_cached(
        "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state,reservation_owner,\
         manifest_path,manifest_sha256,receipt_sha256 FROM pairs WHERE manifest_sha256=?",
    )?;
    let mut active_statement = registry.prepare_cached(
        "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
         AND protocol_sha256=? AND state IN ('reserved','submitted','infrastructure_retry')",
    )?;
    let mut evidence_statement = registry.prepare_cached(
        "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
         AND protocol_sha256=? AND state='completed' AND manifest_sha256=?",
    )?;

    let rows = pair_statement.query_map([manifest_sha256], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, Option<String>>(6)?,
            row.get::<_, Option<String>>(7)?,
            row.get::<_, Option<String>>(8)?,
        ))
    })?;

    let mut missing = Vec::new();
    for row in rows {
        let (task, dataset_hash, semantic, protocol, state, owner, path, digest, receipt) = row?;
        stats.pairs_examined += 1;

        // Evidence already exists for this pair: leave it DONE.
        if receipt.is_some() {
            stats.skipped_evidence += 1;
            continue;
        }
        if !PAIR_LAUNCH_READY.contains(&state.as_str()) {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        }
        // The keeper matches attempts on manifest_sha256; a pair whose registry
        // row lost its manifest binding cannot be filled coherently.
        let Some(digest) = digest else {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        };
        if digest != manifest_sha256 {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        }

        let active: i64 = active_statement
            .query_row(params![task, dataset_hash, semantic, protocol], |row| {
                row.get(0)
            })?;
        if active > 0 {
            stats.skipped_already_active += 1;
            continue;
        }

        let completed: i64 = evidence_statement.query_row(
            params![task, dataset_hash, semantic, protocol, manifest_sha256],
            |row| row.get(0),
        )?;
        if completed > 0 {
            stats.skipped_evidence += 1;
            continue;
        }

        missing.push(MissingAttempt {
            task,
            dataset_hash,
            semantic_sha256: semantic,
            protocol_sha256: protocol,
            state,
            reservation_owner: owner,
            manifest_path: path,
            manifest_sha256: digest,
        });
    }
    Ok(missing)
}

fn run_fill(
    backlog: &Connection,
    registry: &mut Connection,
    args: &Args,
    batch_id: &str,
    journal: &mut Journal,
) -> Result<FillStats> {
    let mut stats = FillStats::default();
    let manifests = active_claim_manifests(backlog, &args.claim_states)?;
    stats.candidate_manifests = manifests.len();

    let mut pending: Vec<MissingAttempt> = Vec::new();
    for manifest_sha256 in &manifests {
        if stats.inserted + pending.len() >= args.max_fill {
            stats.truncated_by_limit = true;
            break;
        }
        let missing = missing_attempts_for_manifest(registry, manifest_sha256, &mut stats)?;
        stats.manifests_completed += 1;
        pending.extend(missing);

        if pending.len() >= args.batch_rows {
            let written = flush_fill(registry, args, batch_id, journal, &mut pending)?;
            stats.inserted += written;
        }
    }
    if !pending.is_empty() {
        let written = flush_fill(registry, args, batch_id, journal, &mut pending)?;
        stats.inserted += written;
    }
    Ok(stats)
}

/// Write one bounded transaction of inserts. The journal is synced BEFORE the
/// commit so a crash can never leave an unrecorded row.
fn flush_fill(
    registry: &mut Connection,
    args: &Args,
    batch_id: &str,
    journal: &mut Journal,
    pending: &mut Vec<MissingAttempt>,
) -> Result<usize> {
    if pending.is_empty() {
        return Ok(0);
    }
    if !args.apply {
        let count = pending.len();
        pending.clear();
        return Ok(count);
    }

    let detail = json!({ "reconciler": "wsA-three-ledger", "batch": batch_id }).to_string();
    let timestamp = now_ms() as i64;
    let mut inserted_ids: Vec<i64> = Vec::with_capacity(pending.len());

    let tx = registry.transaction_with_behavior(TransactionBehavior::Immediate)?;
    {
        let mut next_attempt = tx.prepare_cached(
            "SELECT COALESCE(MAX(attempt_no),0)+1 FROM attempts WHERE task=? AND dataset_hash=? \
             AND semantic_sha256=? AND protocol_sha256=?",
        )?;
        // Re-check inside the write transaction: another writer may have raced
        // us between the read scan and here. This keeps the lane idempotent.
        let mut recheck = tx.prepare_cached(
            "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
             AND protocol_sha256=? AND state IN ('reserved','submitted','infrastructure_retry')",
        )?;
        let mut insert = tx.prepare_cached(
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,\
             state,reservation_owner,manifest_path,manifest_sha256,job_id,receipt_sha256,\
             failure_class,detail_json,created_at_ms,finalized_at_ms) \
             VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,NULL)",
        )?;

        for row in pending.iter() {
            let active: i64 = recheck.query_row(
                params![
                    row.task,
                    row.dataset_hash,
                    row.semantic_sha256,
                    row.protocol_sha256
                ],
                |value| value.get(0),
            )?;
            if active > 0 {
                continue;
            }
            let attempt_no: i64 = next_attempt.query_row(
                params![
                    row.task,
                    row.dataset_hash,
                    row.semantic_sha256,
                    row.protocol_sha256
                ],
                |value| value.get(0),
            )?;
            insert.execute(params![
                row.task,
                row.dataset_hash,
                row.semantic_sha256,
                row.protocol_sha256,
                attempt_no,
                row.state,
                row.reservation_owner,
                row.manifest_path,
                row.manifest_sha256,
                detail,
                timestamp
            ])?;
            inserted_ids.push(tx.last_insert_rowid());
        }
    }

    for id in &inserted_ids {
        journal.record(&json!({ "t": "attempt_insert", "id": id }))?;
    }
    journal.sync()?;
    tx.commit()?;

    pending.clear();
    Ok(inserted_ids.len())
}

/// Manifests that still hold active `keeper_attempts` rows. Driven off the
/// covering index `keeper_attempt_manifest_state(manifest_sha256,state)` so
/// this stays an index scan rather than a 4.7 GB table walk.
fn active_keeper_manifests(backlog: &Connection, limit: usize) -> Result<BTreeMap<String, usize>> {
    let mut statement = backlog.prepare(
        "SELECT manifest_sha256, COUNT(*) FROM keeper_attempts \
         WHERE state IN ('checked','submitting','submitted','deferred_horizon') \
         GROUP BY manifest_sha256",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
    })?;
    let mut out = BTreeMap::new();
    let mut total = 0usize;
    for row in rows {
        let (manifest, count) = row?;
        if total >= limit {
            break;
        }
        total += count;
        out.insert(manifest, count);
    }
    Ok(out)
}

fn run_reap(
    backlog: &mut Connection,
    registry: &Connection,
    args: &Args,
    _batch_id: &str,
    journal: &mut Journal,
) -> Result<ReapStats> {
    let mut stats = ReapStats::default();
    let candidates = active_keeper_manifests(backlog, args.max_reap)?;
    stats.manifests_examined = candidates.len();

    // Two independent release signals. The registry one alone was NOT enough:
    // measured in production, it retired 0 rows across 297 receipts while the
    // active set regrew 0 -> 78,261 in three hours, because a normally-COMPLETED
    // pair keeps its manifest binding (pairs state='completed': 2,591,302 bound,
    // 0 unbound). "Zero bound pairs" only ever describes the retirement path,
    // never the far more common success path.
    //
    // Signal 1 (backlog, authoritative for "the work is done"): the manifest's
    // claim reached a terminal state. 'ingested' means the evidence is already
    // in the corpus, so the attempt has no business holding a slot in the
    // keeper_one_active_attempt uniqueness index. Same database as the rows we
    // retire, so there is no cross-database race.
    //
    // Signal 2 (registry, the original): zero bound pairs, i.e. freshness or
    // jope_queue_control nulled the binding when releasing the pair.
    const TERMINAL_CLAIM_STATES: [&str; 3] =
        ["ingested", "superseded_unstarted", "rejected_task_invalid"];

    // Scoped so the cached statements release their immutable borrow of
    // `backlog` before the retirement transaction below takes it mutably.
    let mut released: Vec<String> = Vec::new();
    {
        let mut claim_statement =
            backlog.prepare_cached("SELECT state FROM claims WHERE manifest_sha256=?")?;
        let mut bound_statement =
            registry.prepare_cached("SELECT COUNT(*) FROM pairs WHERE manifest_sha256=?")?;
        for manifest in candidates.keys() {
            let claim_state: Option<String> = claim_statement
                .query_row([manifest], |row| row.get(0))
                .optional()?;
            let claim_terminal = claim_state
                .as_deref()
                .is_some_and(|state| TERMINAL_CLAIM_STATES.contains(&state));
            if claim_terminal {
                released.push(manifest.clone());
                continue;
            }
            let bound: i64 = bound_statement.query_row([manifest], |row| row.get(0))?;
            if bound == 0 {
                released.push(manifest.clone());
            }
        }
    }
    stats.manifests_released = released.len();
    if released.is_empty() || !args.apply {
        stats.retired = released
            .iter()
            .map(|manifest| candidates.get(manifest).copied().unwrap_or(0))
            .sum();
        return Ok(stats);
    }

    for chunk in released.chunks(args.batch_rows.max(1)) {
        let mut journalled: Vec<serde_json::Value> = Vec::new();
        let tx = backlog.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut select = tx.prepare_cached(
                "SELECT id,state,failure_class FROM keeper_attempts \
                 WHERE manifest_sha256=? \
                 AND state IN ('checked','submitting','submitted','deferred_horizon')",
            )?;
            let mut update = tx.prepare_cached(
                "UPDATE keeper_attempts SET state=?,finalized_at_ms=? WHERE id=?",
            )?;
            let timestamp = now_ms() as i64;
            for manifest in chunk {
                let rows = select
                    .query_map([manifest], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                for (id, prior_state, prior_failure) in rows {
                    update.execute(params![KEEPER_RETIRED_STATE, timestamp, id])?;
                    journalled.push(json!({
                        "t": "keeper_retire",
                        "id": id,
                        "prior_state": prior_state,
                        "prior_failure_class": prior_failure,
                    }));
                }
            }
        }
        for entry in &journalled {
            journal.record(entry)?;
        }
        journal.sync()?;
        tx.commit()?;
        stats.retired += journalled.len();
    }
    Ok(stats)
}

fn run_revert(args: &Args, batch_id: &str, receipt_dir: &Path) -> Result<serde_json::Value> {
    let journal_path = receipt_dir.join(format!("{batch_id}.journal.jsonl"));
    let file = File::open(&journal_path)
        .with_context(|| format!("open journal {}", journal_path.display()))?;

    let mut attempt_ids: Vec<i64> = Vec::new();
    let mut keeper_rows: Vec<(i64, String, Option<String>)> = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        match value.get("t").and_then(|v| v.as_str()) {
            Some("attempt_insert") => {
                attempt_ids.push(
                    value
                        .get("id")
                        .and_then(|v| v.as_i64())
                        .context("journal attempt_insert lacks id")?,
                );
            }
            Some("keeper_retire") => {
                keeper_rows.push((
                    value
                        .get("id")
                        .and_then(|v| v.as_i64())
                        .context("journal keeper_retire lacks id")?,
                    value
                        .get("prior_state")
                        .and_then(|v| v.as_str())
                        .context("journal keeper_retire lacks prior_state")?
                        .to_owned(),
                    value
                        .get("prior_failure_class")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned),
                ));
            }
            other => bail!("unknown journal record type {other:?}"),
        }
    }

    let mut deleted = 0usize;
    let mut restored = 0usize;
    if args.apply {
        if !attempt_ids.is_empty() {
            let mut registry = open_rw(
                &args.root.join("ops/pair-registry.sqlite"),
                args.busy_timeout_ms,
            )?;
            for chunk in attempt_ids.chunks(args.batch_rows.max(1)) {
                let tx = registry.transaction_with_behavior(TransactionBehavior::Immediate)?;
                {
                    // Only remove rows this batch created and that are still
                    // untouched, so a reversal can never destroy live progress.
                    let mut delete = tx.prepare_cached(
                        "DELETE FROM attempts WHERE id=? AND state IN ('reserved','infrastructure_retry') \
                         AND job_id IS NULL AND receipt_sha256 IS NULL \
                         AND detail_json LIKE '%wsA-three-ledger%'",
                    )?;
                    for id in chunk {
                        deleted += delete.execute(params![id])?;
                    }
                }
                tx.commit()?;
            }
        }
        if !keeper_rows.is_empty() {
            let mut backlog = open_rw(&args.root.join("ops/backlog.sqlite"), args.busy_timeout_ms)?;
            for chunk in keeper_rows.chunks(args.batch_rows.max(1)) {
                let tx = backlog.transaction_with_behavior(TransactionBehavior::Immediate)?;
                {
                    let mut update = tx.prepare_cached(
                        "UPDATE keeper_attempts SET state=?,failure_class=?,finalized_at_ms=NULL \
                         WHERE id=? AND state=?",
                    )?;
                    for (id, prior_state, prior_failure) in chunk {
                        restored += update.execute(params![
                            prior_state,
                            prior_failure,
                            id,
                            KEEPER_RETIRED_STATE
                        ])?;
                    }
                }
                tx.commit()?;
            }
        }
    }

    Ok(json!({
        "schema": "JopeLedgerReconcilerRevert.v1",
        "batch_id": batch_id,
        "apply": args.apply,
        "journal": journal_path.to_string_lossy(),
        "attempt_rows_in_journal": attempt_ids.len(),
        "keeper_rows_in_journal": keeper_rows.len(),
        "attempt_rows_deleted": deleted,
        "keeper_rows_restored": restored,
    }))
}

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
mod tests {
    use super::*;

    fn fixture(root: &Path) -> Result<()> {
        fs::create_dir_all(root.join("ops"))?;
        let backlog = Connection::open(root.join("ops/backlog.sqlite"))?;
        backlog.execute_batch(
            "CREATE TABLE claims(manifest_sha256 TEXT PRIMARY KEY,manifest TEXT NOT NULL UNIQUE,
               job_id TEXT NOT NULL UNIQUE,spec TEXT NOT NULL,state TEXT NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0,updated_at_ms INTEGER NOT NULL,
               reservation_owner TEXT);
             CREATE TABLE keeper_attempts(id INTEGER PRIMARY KEY,task TEXT NOT NULL,
               dataset_hash TEXT NOT NULL,semantic_sha256 TEXT NOT NULL,
               protocol_sha256 TEXT NOT NULL,attempt_number INTEGER NOT NULL,
               state TEXT NOT NULL,owner TEXT NOT NULL,manifest_path TEXT NOT NULL,
               manifest_sha256 TEXT NOT NULL,job_id TEXT NOT NULL,failure_class TEXT,
               created_at_ms INTEGER NOT NULL,finalized_at_ms INTEGER,
               UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number));
             CREATE INDEX keeper_attempt_manifest_state ON keeper_attempts(manifest_sha256,state);",
        )?;
        let registry = Connection::open(root.join("ops/pair-registry.sqlite"))?;
        registry.execute_batch(
            "CREATE TABLE pairs(task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,canonical_genome TEXT,
               state TEXT NOT NULL,reservation_owner TEXT NOT NULL,manifest_path TEXT,
               manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
               infrastructure_retries INTEGER NOT NULL DEFAULT 0,created_at_ms INTEGER NOT NULL,
               updated_at_ms INTEGER NOT NULL,
               PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)) WITHOUT ROWID;
             CREATE TABLE attempts(id INTEGER PRIMARY KEY,task TEXT NOT NULL,
               dataset_hash TEXT NOT NULL,semantic_sha256 TEXT NOT NULL,
               protocol_sha256 TEXT NOT NULL,attempt_no INTEGER NOT NULL,state TEXT NOT NULL,
               reservation_owner TEXT NOT NULL,manifest_path TEXT,manifest_sha256 TEXT,
               job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
               detail_json TEXT NOT NULL DEFAULT '{}',created_at_ms INTEGER NOT NULL,
               finalized_at_ms INTEGER,
               UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no));
             CREATE UNIQUE INDEX one_active_attempt_per_pair
               ON attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
               WHERE state IN ('reserved','submitted');
             CREATE INDEX pair_manifest ON pairs(manifest_sha256);
             CREATE INDEX attempt_manifest ON attempts(manifest_sha256);",
        )?;
        Ok(())
    }

    fn args_for(root: &Path, lane: Lane, apply: bool) -> Args {
        Args {
            root: root.to_path_buf(),
            lane,
            apply,
            max_fill: 1_000,
            max_reap: 1_000,
            batch_rows: 16,
            busy_timeout_ms: 5_000,
            claim_states: vec!["checked".to_owned(), "submitting".to_owned()],
            revert: None,
            receipt_dir: Some(root.join("ops/receipts")),
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("jope-ledger-reconciler-{name}-{}", now_ms()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn claim_state_filter_binds_sql_shaped_states_as_values() {
        let backlog = Connection::open_in_memory().unwrap();
        backlog
            .execute_batch(
                "CREATE TABLE claims(manifest_sha256 TEXT PRIMARY KEY,manifest TEXT NOT NULL UNIQUE,
                   job_id TEXT NOT NULL UNIQUE,spec TEXT NOT NULL,state TEXT NOT NULL,
                   attempts INTEGER NOT NULL DEFAULT 0,updated_at_ms INTEGER NOT NULL,
                   reservation_owner TEXT);
                 INSERT INTO claims VALUES('m1','/1.json','job-1','/s.toml','checked',0,1,NULL);
                 INSERT INTO claims VALUES('m2','/2.json','job-2','/s.toml','submitting',0,1,NULL);
                 INSERT INTO claims VALUES('m3','/3.json','job-3','/s.toml','x'' OR ''1''=''1',0,1,NULL);",
            )
            .unwrap();
        let states = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>();
        // Only the placeholder list is formatted into the SQL; a quote-bearing state matches
        // its own literal row and cannot widen the filter.
        assert_eq!(
            active_claim_manifests(&backlog, &states(&["checked", "x' OR '1'='1"])).unwrap(),
            ["m1", "m3"]
        );
        assert!(active_claim_manifests(&backlog, &states(&["') OR 1=1 --"]))
            .unwrap()
            .is_empty());
        assert!(
            active_claim_manifests(&backlog, &states(&["checked'); DROP TABLE claims; --"]))
                .unwrap()
                .is_empty()
        );
        let rows: i64 = backlog
            .query_row("SELECT COUNT(*) FROM claims", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 3);
        assert!(active_claim_manifests(&backlog, &[]).is_err());
    }

    fn add_pair(registry: &Connection, semantic: &str, state: &str, manifest: Option<&str>) {
        registry
            .execute(
                "INSERT INTO pairs VALUES('regression','ds',?,'proto','genome',?,'owner-1',
                   '/m.json',?,NULL,NULL,NULL,0,1,1)",
                params![semantic, state, manifest],
            )
            .unwrap();
    }

    #[test]
    fn fill_inserts_exactly_one_reserved_attempt_per_bare_pair() {
        let root = temp_root("fill");
        fixture(&root).unwrap();
        let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
        backlog
            .execute(
                "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','submitting',0,1,'owner-1')",
                [],
            )
            .unwrap();
        let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
        add_pair(&registry, "sem-a", "reserved", Some("m1"));
        add_pair(&registry, "sem-b", "reserved", Some("m1"));
        drop(registry);
        drop(backlog);

        let args = args_for(&root, Lane::Fill, true);
        let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
        let backlog = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
        let mut registry = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
        let stats = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
        assert_eq!(stats.inserted, 2);
        assert_eq!(stats.pairs_examined, 2);

        // Exactly one active attempt per pair, matching what
        // update_native_attempts_submitted counts.
        let count: i64 = registry
            .query_row(
                "SELECT COUNT(*) FROM attempts WHERE manifest_sha256='m1' \
                 AND reservation_owner='owner-1' AND state IN ('reserved','infrastructure_retry')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);

        // Idempotent: a second run must insert nothing.
        let stats2 = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
        assert_eq!(stats2.inserted, 0);
        assert_eq!(stats2.skipped_already_active, 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn fill_leaves_pairs_with_evidence_alone() {
        let root = temp_root("evidence");
        fixture(&root).unwrap();
        let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
        backlog
            .execute(
                "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','checked',0,1,'owner-1')",
                [],
            )
            .unwrap();
        let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
        // Receipted pair: DONE, must not be resurrected.
        registry
            .execute(
                "INSERT INTO pairs VALUES('regression','ds','sem-a','proto','genome','reserved',
                   'owner-1','/m.json','m1',NULL,'receipt-sha',NULL,0,1,1)",
                [],
            )
            .unwrap();
        // Pair whose evaluation for THIS manifest already completed.
        add_pair(&registry, "sem-b", "reserved", Some("m1"));
        registry
            .execute(
                "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,
                   state,reservation_owner,manifest_sha256,created_at_ms)
                 VALUES('regression','ds','sem-b','proto',1,'completed','owner-1','m1',1)",
                [],
            )
            .unwrap();
        drop(registry);
        drop(backlog);

        let args = args_for(&root, Lane::Fill, true);
        let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
        let backlog = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
        let mut registry = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
        let stats = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.skipped_evidence, 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn reap_retires_only_registry_released_manifests() {
        let root = temp_root("reap");
        fixture(&root).unwrap();
        let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
        // m1: registry still binds its pair -> must be left alone.
        // m2: registry released it (no bound pair rows) -> fossil, must retire.
        backlog
            .execute_batch(
                "INSERT INTO keeper_attempts VALUES(1,'regression','ds','sem-a','proto',1,
                   'submitted','owner-1','/m.json','m1','job-1',NULL,1,NULL);
                 INSERT INTO keeper_attempts VALUES(2,'regression','ds','sem-b','proto',1,
                   'submitted','owner-2','/m2.json','m2','job-2',NULL,1,NULL);",
            )
            .unwrap();
        let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
        add_pair(&registry, "sem-a", "reserved", Some("m1"));
        // Released pair: state 'available' and manifest binding nulled out, the
        // shape freshness/jope_queue_control leave behind.
        add_pair(&registry, "sem-b", "available", None);
        drop(registry);
        drop(backlog);

        let args = args_for(&root, Lane::Reap, true);
        let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
        let registry = open_ro(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
        let mut backlog = open_rw(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
        let stats = run_reap(&mut backlog, &registry, &args, "t", &mut journal).unwrap();
        assert_eq!(stats.manifests_examined, 2);
        assert_eq!(stats.manifests_released, 1);
        assert_eq!(stats.retired, 1);

        let kept: String = backlog
            .query_row("SELECT state FROM keeper_attempts WHERE id=1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(kept, "submitted");
        let reaped: String = backlog
            .query_row("SELECT state FROM keeper_attempts WHERE id=2", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(reaped, KEEPER_RETIRED_STATE);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn revert_undoes_both_lanes_exactly() {
        let root = temp_root("revert");
        fixture(&root).unwrap();
        let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
        backlog
            .execute(
                "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','submitting',0,1,'owner-1')",
                [],
            )
            .unwrap();
        backlog
            .execute(
                "INSERT INTO keeper_attempts VALUES(2,'regression','ds','sem-b','proto',1,
                   'submitted','owner-2','/m2.json','m2','job-2',NULL,1,NULL)",
                [],
            )
            .unwrap();
        let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
        add_pair(&registry, "sem-a", "reserved", Some("m1"));
        add_pair(&registry, "sem-b", "available", None);
        drop(registry);
        drop(backlog);

        let receipts = root.join("ops/receipts");
        fs::create_dir_all(&receipts).unwrap();
        let mut args = args_for(&root, Lane::Both, true);
        let mut journal = Journal::create(receipts.join("t.journal.jsonl"), true).unwrap();
        {
            let backlog_ro = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
            let mut registry_rw = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
            let fill = run_fill(&backlog_ro, &mut registry_rw, &args, "t", &mut journal).unwrap();
            assert_eq!(fill.inserted, 1);
        }
        {
            let registry_ro = open_ro(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
            let mut backlog_rw = open_rw(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
            let reap = run_reap(&mut backlog_rw, &registry_ro, &args, "t", &mut journal).unwrap();
            assert_eq!(reap.retired, 1);
        }
        journal.sync().unwrap();

        args.revert = Some("t".to_owned());
        let report = run_revert(&args, "t", &receipts).unwrap();
        assert_eq!(report["attempt_rows_deleted"], 1);
        assert_eq!(report["keeper_rows_restored"], 1);

        let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
        let remaining: i64 = registry
            .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 0);
        let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
        let restored: String = backlog
            .query_row("SELECT state FROM keeper_attempts WHERE id=2", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(restored, "submitted");
        let _ = fs::remove_dir_all(&root);
    }
}
