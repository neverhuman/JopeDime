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
pub struct AppendReport {
    pub source: Value,
    pub rows: u64,
    pub duplicates_skipped: u64,
    pub batch: Option<crate::ledger::BatchReceipt>,
    pub mirror: Option<MirrorReport>,
    pub seconds: f64,
}

/// Append `rows` as one sealed batch. Rows whose `row_sha256` is already in the ledger are
/// skipped (re-running an append is idempotent), and reported.
pub fn append(
    state: &State,
    mirror: Option<&Mirror>,
    rows: Vec<ScoreRecord>,
    source: Value,
) -> Result<AppendReport> {
    let started = Instant::now();
    let _lock = state.lock("scores")?;
    let ledger = Ledger::open(state.root(), "scores");
    ledger.require_intact()?;
    // Fail closed before anything local moves; see `sync::preflight`.
    if let Some(m) = mirror {
        crate::sync::preflight(m, &ledger)?;
    }
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
    let mirror_report = match (mirror, &batch) {
        (Some(m), Some(_)) => {
            let mut rep = MirrorReport {
                url: m.url.clone(),
                ..Default::default()
            };
            // Upload every batch the mirror is missing — not just this one — then move HEAD.json
            // conditionally on the tip `sync::plan` proved is an ancestor of this chain. A mirror
            // left behind by a local-only append catches up here instead of wedging.
            //
            // Nothing to put on the mirror ahead of each batch object: a score batch embeds its
            // rows, so unlike an hwm batch it names no blob and is self-contained on its own.
            let plan = crate::sync::plan(m, &ledger)?;
            crate::sync::push(m, &ledger, plan, &mut rep, |_, _| Ok(()))?;
            Some(rep)
        }
        _ => None,
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
pub struct VerifyReport {
    pub ledger: LedgerReport,
    pub rows: u64,
    pub invalid_rows: Vec<String>,
    pub duplicate_rows: u64,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.ledger.ok() && self.invalid_rows.is_empty() && self.duplicate_rows == 0
    }
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
    for v in ledger.events()? {
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

pub fn rows(state: &State) -> Result<Vec<ScoreRecord>> {
    Ledger::open(state.root(), "scores")
        .events()?
        .into_iter()
        .map(|v| serde_json::from_value(v).context("parse score row"))
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct Best {
    pub dataset_hash: String,
    pub metric: String,
    pub direction: Direction,
    pub method_id: String,
    pub value: f64,
    pub track: Option<String>,
    pub version: Option<String>,
    pub rows: u64,
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
