//! genebank `best_scores` (SQLite, read-only) → `jope.hwm/v1`, evaluator `unbound`.

use crate::canon::{canonical_sha256, sha256_hex};
use crate::hwm::Candidate;
use crate::metric::Metric;
use crate::record::*;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::path::Path;

/// Constant identifying the protocol.
pub const PROTOCOL: &str = "genebank-gen1";
/// Constant identifying the evaluator version.
pub const EVALUATOR_VERSION: &str = "genebank-legacy";

#[derive(Serialize)]
struct CutIdentity<'a> {
    tier: &'a str,
    protocol: &'a str,
}

/// Performs the cut ID operation and returns an error if it cannot be completed.
pub fn cut_id(tier: &str) -> Result<String> {
    canonical_sha256(&CutIdentity {
        tier,
        protocol: PROTOCOL,
    })
}

/// The row as read; its canonical JSON is the record blob.
#[derive(Clone, Debug, Serialize)]
pub struct Row {
    /// Task identity for the comparison represented by this record.
    pub task: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// Score tier recorded for this row.
    pub score_tier: String,
    /// Metric used to evaluate the represented result.
    pub metric: String,
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: f64,
    /// SHA-256 digest identifying the associated semantic.
    pub semantic_sha256: Option<String>,
    /// Canonical genome recorded for this row.
    pub canonical_genome: String,
    /// Achieved at ms recorded for this row.
    pub achieved_at_ms: i64,
    /// Optional receipt ref value; `None` means it is not available.
    pub receipt_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `GenebankScan` is a record for the genebank scan contract in this module: genebank bestscores (SQLite, read-only) → jope.hwm/v1, evaluator unbound.
pub struct GenebankScan {
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Mapped recorded for this genebank scan.
    pub mapped: u64,
    /// Collection of rejects values carried by this genebank scan.
    pub rejects: Vec<(String, String)>,
}

/// Performs the map row operation and returns an error if it cannot be completed.
pub fn map_row(row: &Row, bundle_id: &str) -> Result<Candidate> {
    let metric = Metric::parse(&row.metric)
        .with_context(|| format!("unknown genebank metric {}", row.metric))?;
    let identity = HwmIdentity {
        comparison: None,
        task: row.task.clone(),
        dataset_hash: row.dataset_hash.clone(),
        cut_id: cut_id(&row.score_tier)?,
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: PROTOCOL.into(),
    };
    let blob = crate::canon::canonical_json(row)?;
    if row.achieved_at_ms < 0 {
        bail!("negative achieved_at_ms");
    }
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256()?,
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: sha256_hex(row.canonical_genome.as_bytes()),
            dsl: row.canonical_genome.clone(),
            semantic_sha256: row
                .semantic_sha256
                .clone()
                .filter(|s| crate::canon::is_hex64(s)),
        },
        score: Score {
            metric,
            value: row.score,
            direction: metric.direction(),
            units: "dimensionless".into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: UNBOUND.into(),
            version: EVALUATOR_VERSION.into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: false,
        provenance: Provenance {
            source: "genebank".into(),
            run_id: format!(
                "best_scores/{}/{}/{}",
                row.task, row.dataset_hash, row.score_tier
            ),
            git_sha: UNBOUND.into(),
            bundle_id: bundle_id.into(),
            record_sha256: sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: (row.achieved_at_ms / 1000) as u64,
    };
    record.validate()?;
    Ok(Candidate {
        record,
        blob,
        chain_from_source: false,
    })
}

/// Read every `best_scores` row, read-only.
pub fn read_rows(db: &Path) -> Result<Vec<Row>> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open {}", db.display()))?;
    let mut stmt = conn.prepare(
        "SELECT task, dataset_hash, score_tier, metric, score, semantic_sha256, canonical_genome, \
         achieved_at_ms, receipt_ref FROM best_scores ORDER BY task, dataset_hash, score_tier",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(Row {
                task: r.get(0)?,
                dataset_hash: r.get(1)?,
                score_tier: r.get(2)?,
                metric: r.get(3)?,
                score: r.get(4)?,
                semantic_sha256: r.get(5)?,
                canonical_genome: r.get(6)?,
                achieved_at_ms: r.get(7)?,
                receipt_ref: r.get::<_, Option<String>>(8)?.filter(|s| !s.is_empty()),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Performs the scan operation and returns an error if it cannot be completed.
pub fn scan(db: &Path, bundle_id: &str) -> Result<(Vec<Candidate>, GenebankScan)> {
    let rows = read_rows(db)?;
    if rows.is_empty() {
        bail!("best_scores is empty in {}", db.display());
    }
    let mut scan = GenebankScan::default();
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        scan.rows += 1;
        match map_row(row, bundle_id) {
            Ok(c) => {
                scan.mapped += 1;
                out.push(c);
            }
            Err(e) => scan.rejects.push((
                format!("{}/{}/{}", row.task, row.dataset_hash, row.score_tier),
                format!("{e:#}"),
            )),
        }
    }
    Ok((out, scan))
}

/// Performs the scan source operation.
pub fn scan_source(db: &Path, bundle_id: &str, scan: &GenebankScan) -> serde_json::Value {
    crate::hwm::source_value(
        "genebank",
        db,
        json!({
            "bundle_id": bundle_id,
            "db_sha256": bundle_id,
            "rows": scan.rows,
            "mapped": scan.mapped,
            "unmapped": scan.rejects.len(),
            "evaluator": EVALUATOR_VERSION,
        }),
    )
}
