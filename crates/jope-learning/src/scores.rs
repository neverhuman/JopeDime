//! The score ledger: `jope.score/v1` rows appended in sealed batches, verified, queried.

use crate::ledger::{Ledger, LedgerReport};
use crate::metric::Direction;
use crate::mirror::{Mirror, MirrorReport};
use crate::record::ScoreRecord;
use crate::state::State;
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
/// `AppendReport` is a record for the append report contract in this module: The score ledger: jope.score/v1 rows appended in sealed batches, verified, queried.
pub struct AppendReport {
    /// Provenance source for this value.
    pub source: Value,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Duplicates skipped recorded for this append report.
    pub duplicates_skipped: u64,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: Option<crate::ledger::BatchReceipt>,
    /// Optional mirror value; `None` means it is not available.
    pub mirror: Option<MirrorReport>,
    /// Seconds recorded for this append report.
    pub seconds: f64,
}

/// Append `rows` as one sealed batch. Rows whose `row_sha256` is already in the ledger are
/// skipped (re-running an append is idempotent), and reported.
///
/// `archive` writes whatever copy of the source this import keeps next to the ledger. It is a
/// callback rather than the caller's own `fs::write` for two reasons, both of which were real
/// defects: it must run **after** the divergence preflight, or a refusal that promises "nothing
/// was written" has already overwritten last week's archived `scores.csv`; and it must run
/// **under the same `scores` lock** as the ledger append, or two concurrent imports can interleave
/// an archive of one run with the batch of another.
pub fn append(
    state: &State,
    mirror: Option<&Mirror>,
    rows: Vec<ScoreRecord>,
    source: Value,
    archive: impl FnOnce(&State) -> Result<()>,
) -> Result<AppendReport> {
    let started = Instant::now();
    let _lock = state.lock("scores")?;
    let ledger = Ledger::open(state.root(), "scores");
    ledger.require_intact()?;
    // Fail closed before anything local moves; see `sync::preflight`.
    if let Some(m) = mirror {
        crate::sync::preflight(m, &ledger)?;
    }
    // Only now, past every refusal and inside the lock, may local bytes move.
    archive(state)?;
    let existing: std::collections::BTreeSet<String> = ledger
        .events()?
        .iter()
        .filter_map(|v| v["row_sha256"].as_str().map(str::to_string))
        .collect();
    let mut events = Vec::with_capacity(rows.len());
    let mut duplicates = 0u64;
    let mut seen = std::collections::BTreeSet::new();
    for r in rows {
        r.validate()
            .with_context(|| format!("row {}/{}", r.dataset_hash, r.method_id))?;
        if existing.contains(&r.row_sha256) || !seen.insert(r.row_sha256.clone()) {
            duplicates += 1;
            continue;
        }
        events.push(serde_json::to_value(&r)?);
    }
    let batch = if events.is_empty() {
        None
    } else {
        Some(ledger.append(&events, source.clone())?)
    };
    // The push is driven by `--store`, never by whether *this* run happened to seal a batch.
    //
    // Gating it on `batch` was the bug: an append whose rows are all duplicates seals nothing, so
    // the mirror was never even looked at — and since the backfill is the only thing that catches
    // a mirror up, the one command an operator would re-run to repair a mirror left behind by an
    // earlier local-only append was precisely the command that did nothing. It reported `rows: 0`,
    // `mirror: null` and exit 0, which is indistinguishable from a run with no `--store` at all.
    //
    // So: configured means pushed, and `mirror` is `Some` exactly when a mirror was configured.
    // A push that genuinely had nothing to do says so in its own counts (`batches: 0`), which no
    // longer reads as "no mirror" and cannot be mistaken for a publish.
    //
    // Nothing to put on the mirror ahead of each batch object: a score batch embeds its rows, so
    // unlike an hwm batch it names no blob and is self-contained on its own.
    let mirror_report = match mirror {
        Some(m) => {
            let mut rep = MirrorReport {
                url: m.url.clone(),
                ..Default::default()
            };
            let plan = crate::sync::plan(m, &ledger)?;
            crate::sync::push(m, &ledger, plan, &mut rep, |_, _| Ok(()))?;
            Some(rep)
        }
        None => None,
    };
    Ok(AppendReport {
        source,
        rows: events.len() as u64,
        duplicates_skipped: duplicates,
        batch,
        mirror: mirror_report,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Debug, Serialize, Default)]
/// `VerifyReport` is a record for the verify report contract in this module: The score ledger: jope.score/v1 rows appended in sealed batches, verified, queried.
pub struct VerifyReport {
    /// Ledger recorded for this verify report.
    pub ledger: LedgerReport,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Number of invalid rows represented here.
    pub invalid_rows: Vec<String>,
    /// Number of duplicate rows represented here.
    pub duplicate_rows: u64,
}

impl VerifyReport {
    /// Performs the ok operation and returns an error if it cannot be completed.
    pub fn ok(&self) -> bool {
        self.ledger.ok() && self.invalid_rows.is_empty() && self.duplicate_rows == 0
    }
    /// Performs the total breaks operation and returns an error if it cannot be completed.
    pub fn total_breaks(&self) -> usize {
        self.ledger.breaks.len() + self.invalid_rows.len() + self.duplicate_rows as usize
    }
}

/// Walk the batch chain and re-validate every row (schema, direction table, row hash).
pub fn verify(state: &State) -> Result<VerifyReport> {
    let ledger = Ledger::open(state.root(), "scores");
    let mut report = VerifyReport {
        ledger: ledger.verify()?,
        ..Default::default()
    };
    let mut seen = std::collections::BTreeSet::new();
    // Lenient for the same reason as `hwm::verify`: `ledger.verify()` above already reported any
    // batch that could not be read, and the verifier must survive it to report the rest.
    for v in ledger.events_readable()? {
        report.rows += 1;
        match serde_json::from_value::<ScoreRecord>(v.clone()) {
            Ok(r) => {
                if let Err(e) = r.validate() {
                    report
                        .invalid_rows
                        .push(format!("{}/{}: {e}", r.dataset_hash, r.method_id));
                }
                if !seen.insert(r.row_sha256.clone()) {
                    report.duplicate_rows += 1;
                }
            }
            Err(e) => report.invalid_rows.push(format!("unparseable row: {e}")),
        }
    }
    Ok(report)
}

/// Performs the rows operation and returns an error if it cannot be completed.
pub fn rows(state: &State) -> Result<Vec<ScoreRecord>> {
    Ledger::open(state.root(), "scores")
        .events()?
        .into_iter()
        .map(|v| serde_json::from_value(v).context("parse score row"))
        .collect()
}

#[derive(Clone, Debug, Serialize)]
/// `Best` is a record for the best contract in this module: The score ledger: jope.score/v1 rows appended in sealed batches, verified, queried.
pub struct Best {
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// Metric used to evaluate the represented result.
    pub metric: String,
    /// Whether larger or smaller metric values represent improvement.
    pub direction: Direction,
    /// Identifier for the associated method.
    pub method_id: String,
    /// Measured or derived value carried by this result.
    pub value: f64,
    /// Optional track value; `None` means it is not available.
    pub track: Option<String>,
    /// Optional version value; `None` means it is not available.
    pub version: Option<String>,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Methods recorded for this best.
    pub methods: u64,
}

/// Best benchmark-valid row per (dataset, metric); ties keep the earliest appended row.
pub fn best_per_dataset(
    state: &State,
    track: Option<&str>,
    version: Option<&str>,
    include_invalid: bool,
) -> Result<Vec<Best>> {
    let mut best: BTreeMap<(String, String), Best> = BTreeMap::new();
    let mut methods: BTreeMap<(String, String), std::collections::BTreeSet<String>> =
        BTreeMap::new();
    for r in rows(state)? {
        if !include_invalid && !r.benchmark_valid {
            continue;
        }
        if track.is_some_and(|t| r.source.track.as_deref() != Some(t)) {
            continue;
        }
        if version.is_some_and(|v| r.source.version.as_deref() != Some(v)) {
            continue;
        }
        let key = (r.dataset_hash.clone(), r.metric.to_string());
        methods
            .entry(key.clone())
            .or_default()
            .insert(r.method_id.clone());
        let entry = best.entry(key).or_insert_with(|| Best {
            dataset_hash: r.dataset_hash.clone(),
            metric: r.metric.to_string(),
            direction: r.direction,
            method_id: r.method_id.clone(),
            value: r.value,
            track: r.source.track.clone(),
            version: r.source.version.clone(),
            rows: 0,
            methods: 0,
        });
        entry.rows += 1;
        if r.direction.strictly_improves(r.value, entry.value) {
            entry.method_id = r.method_id.clone();
            entry.value = r.value;
            entry.track = r.source.track.clone();
            entry.version = r.source.version.clone();
        }
    }
    Ok(best
        .into_iter()
        .map(|(k, mut b)| {
            b.methods = methods.get(&k).map_or(0, |s| s.len() as u64);
            b
        })
        .collect())
}
