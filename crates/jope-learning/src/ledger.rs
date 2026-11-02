//! Append-only, hash-chained JSONL ledgers.
//!
//! A batch is `<ledger>/<yyyy>/<mm>/<ulid>.jsonl`: one compact JSON event per line, then one
//! seal line (`jope.ledger-seal/v1`) carrying the sha256 of the event lines, the previous
//! batch's ULID and the sha256 of the previous batch *file*. `HEAD.json` names the tip. A batch is
//! written to a temporary file, fsynced and renamed; nothing is ever rewritten. `verify` walks the
//! batches in ULID order and recomputes every hash, so an *edited*, reordered, duplicated or
//! mid-chain-missing batch shows up as a break, as does a `HEAD.json` that disagrees with the tip
//! it names in batch, file hash, path or counts.
//!
//! An append is two renames: the batch's, then `HEAD.json`'s. A crash between them leaves a
//! sealed batch that the head does not name. [`Ledger::recover`] rolls such a head forward, and
//! so do [`Ledger::append`] and [`Ledger::require_intact`]. It does so only onto an exact,
//! sealed continuation of the chain the head names ([`Ledger::lagging_head`]). Anything else
//! remains a break.
//!
//! What it does **not** detect is a tail that was removed *and* `HEAD.json` rewritten to match:
//! delete the newest batches, point `HEAD.json` at the new last one with its real sha and
//! corrected counts, and the remaining chain is internally consistent, because `HEAD.json` is the
//! only anchor and it lives in the directory it anchors. No hash inside the tree can notice
//! history that is no longer in the tree. Detecting it needs an anchor from outside: today that is
//! the mirror — [`Ledger::mirror_position`] refuses a mirror whose tip this host's chain no longer
//! contains, so a truncated host cannot push and is told why — or any external record of the tip.
//! `verify` reporting zero breaks means "everything here is consistent with everything else here",
//! which is the strongest thing a self-contained chain can say, and is stated that way rather than
//! as an integrity guarantee it cannot make.

use crate::canon::sha256_hex;
use crate::civil::year_month;
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Schema identifier for seal values.
pub const SEAL_SCHEMA: &str = "jope.ledger-seal/v1";
/// Schema identifier for head values.
pub const HEAD_SCHEMA: &str = "jope.ledger-head/v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Seal` is a record for the seal contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Seal {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Ledger recorded for this seal.
    pub ledger: String,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Events recorded for this seal.
    pub events: u64,
    /// SHA-256 digest identifying the associated events.
    pub events_sha256: String,
    /// Optional prev batch value; `None` means it is not available.
    pub prev_batch: Option<String>,
    /// SHA-256 digest identifying the associated prev batch.
    pub prev_batch_sha256: Option<String>,
    /// Provenance source for this value.
    pub source: Value,
    /// Tool recorded for this seal.
    pub tool: String,
    /// Unix timestamp when this event was recorded.
    pub ts_unix: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Head` is a record for the head contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Head {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Ledger recorded for this head.
    pub ledger: String,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Batch file path relative to the ledger directory.
    pub path: String,
    /// Sha-256 recorded for this head.
    pub sha256: String,
    /// Batches recorded for this head.
    pub batches: u64,
    /// Events total recorded for this head.
    pub events_total: u64,
}

#[derive(Clone, Debug, Serialize)]
/// `BatchReceipt` is a record for the batch receipt contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct BatchReceipt {
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Filesystem path associated with this input or output.
    pub path: String,
    /// Sha-256 recorded for this batch receipt.
    pub sha256: String,
    /// Events recorded for this batch receipt.
    pub events: u64,
    /// SHA-256 digest identifying the associated prev batch.
    pub prev_batch_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `LedgerReport` is a record for the ledger report contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct LedgerReport {
    /// Ledger recorded for this ledger report.
    pub ledger: String,
    /// Batches recorded for this ledger report.
    pub batches: u64,
    /// Events recorded for this ledger report.
    pub events: u64,
    /// Optional head value; `None` means it is not available.
    pub head: Option<Head>,
    /// Collection of breaks values carried by this ledger report.
    pub breaks: Vec<Break>,
}

impl LedgerReport {
    /// Performs the ok operation.
    pub fn ok(&self) -> bool {
        self.breaks.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
/// `Break` is a record for the break contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Break {
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Detail recorded for this break.
    pub detail: String,
}

/// The result of comparing a mirror's `HEAD.json` against the local chain
/// ([`Ledger::mirror_position`]).
#[derive(Clone, Debug)]
pub enum MirrorPosition {
    /// The mirror's tip is this chain's tip (or both are empty): nothing to backfill.
    UpToDate,
    /// The mirror's tip is a batch of this chain, or the mirror has no `HEAD.json` at all.
    /// `missing` is every local batch after it, oldest first — the gap to upload before the
    /// pointer moves.
    /// Field value recorded by this public API type.
    Behind {
        /// Missing recorded for this outcome.
        missing: Vec<(String, PathBuf)>,
    },
    /// The mirror names a chain this host cannot continue. Refuse; do not overwrite.
    /// Field value recorded by this public API type.
    Diverged {
        /// Explanation accompanying this outcome.
        detail: String,
    },
}

/// `Ledger` is a record for the ledger contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Ledger {
    root: PathBuf,
    name: String,
}

/// Performs the now unix operation.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write `bytes` to `path` through a sibling temporary file, fsync, rename.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic write of a root path")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Performs the map err walk operation.
pub fn map_err_walk(e: walkdir::Error) -> anyhow::Error {
    anyhow!("{e}")
}

/// The id for the next batch: a fresh ULID, or the previous id incremented when the clock has not
/// advanced past it.
///
/// Strictly greater than `prev` whenever `prev` parses, which is what `Ledger::batches()`'s
/// ordering depends on. Callers must hold the ledger lock, so `prev` really is the batch being
/// chained onto and no other writer can mint in between.
fn next_batch_id(prev: Option<&str>) -> Result<ulid::Ulid> {
    let fresh = ulid::Ulid::new();
    let Some(prev) = prev else { return Ok(fresh) };
    let Ok(prev) = ulid::Ulid::from_string(prev) else {
        // An id that does not parse cannot be ordered against. Returning the fresh one leaves
        // `verify()` to report the malformed chain rather than inventing an order for it here.
        return Ok(fresh);
    };
    if fresh > prev {
        return Ok(fresh);
    }
    prev.increment()
        .ok_or_else(|| anyhow!("batch id space exhausted after {prev}"))
}

mod chain;
mod verification;

#[cfg(test)]
mod tests;
