//! The GP results index (`jopedime.gp-results-index/v1`), aggregated per object group.
//!
//! Fail-closed checks: the schema must match, the recorded counts must agree with the rows, and
//! every scored S3 row must carry a score identity the policy knows. Scores are only ever compared
//! within one identity (`cv_geomean_rmse_ratio` is not `best_rust_rmse` is not L6).

use super::groups::GroupLookup;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for index values.
pub const INDEX_SCHEMA: &str = "jopedime.gp-results-index/v1";

#[derive(Clone, Debug, Default)]
/// `GroupIndex` is a record for the group index contract in this module: The GP results index (jopedime.gp-results-index/v1), aggregated per object group.
pub struct GroupIndex {
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Number of scored rows represented here.
    pub scored_rows: u64,
    /// Number of qualified rows represented here.
    pub qualified_rows: u64,
    /// Number of consumer recorded rows represented here.
    pub consumer_recorded_rows: u64,
    /// Mapping of by grain values carried by this group index.
    pub by_grain: BTreeMap<String, u64>,
    /// Families recorded for this group index.
    pub families: BTreeSet<String>,
    /// Score names recorded for this group index.
    pub score_names: BTreeSet<String>,
    /// Programs recorded for this group index.
    pub programs: BTreeSet<String>,
    /// Contexts recorded for this group index.
    pub contexts: BTreeSet<String>,
    /// context -> distinct scored programs (a context with two or more holds contrastive pairs).
    pub scored_by_context: BTreeMap<String, BTreeSet<String>>,
}

/// The per-group summary written into a plan row.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct IndexSummary {
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Number of scored rows represented here.
    pub scored_rows: u64,
    /// Number of qualified rows represented here.
    pub qualified_rows: u64,
    /// Number of consumer recorded rows represented here.
    pub consumer_recorded_rows: u64,
    /// Mapping of by grain values carried by this index summary.
    pub by_grain: BTreeMap<String, u64>,
    /// Collection of score names values carried by this index summary.
    pub score_names: Vec<String>,
    /// Distinct programs recorded for this index summary.
    pub distinct_programs: u64,
    /// Distinct contexts recorded for this index summary.
    pub distinct_contexts: u64,
    /// Contexts with alternatives recorded for this index summary.
    pub contexts_with_alternatives: u64,
}

impl GroupIndex {
    /// Performs the summary operation.
    pub fn summary(&self) -> IndexSummary {
        IndexSummary {
            rows: self.rows,
            scored_rows: self.scored_rows,
            qualified_rows: self.qualified_rows,
            consumer_recorded_rows: self.consumer_recorded_rows,
            by_grain: self.by_grain.clone(),
            score_names: self.score_names.iter().cloned().collect(),
            distinct_programs: self.programs.len() as u64,
            distinct_contexts: self.contexts.len() as u64,
            contexts_with_alternatives: self
                .scored_by_context
                .values()
                .filter(|p| p.len() >= 2)
                .count() as u64,
        }
    }
}

#[derive(Debug, Default)]
/// `IndexAgg` is a record for the index agg contract in this module: The GP results index (jopedime.gp-results-index/v1), aggregated per object group.
pub struct IndexAgg {
    /// Mapping of per group values carried by this index agg.
    pub per_group: BTreeMap<usize, GroupIndex>,
    /// family -> scored rows across the whole index (diversity test).
    pub scored_by_family: BTreeMap<String, u64>,
    /// Rows total recorded for this index agg.
    pub rows_total: u64,
    /// Rows s3 recorded for this index agg.
    pub rows_s3: u64,
    /// Rows local recorded for this index agg.
    pub rows_local: u64,
    /// Collection of rows unmatched values carried by this index agg.
    pub rows_unmatched: Vec<String>,
    /// Qualified total recorded for this index agg.
    pub qualified_total: u64,
}

fn s(v: &Value, f: &str) -> Option<String> {
    v.get(f).and_then(Value::as_str).map(str::to_string)
}

/// Performs the aggregate operation and returns an error if it cannot be completed.
pub fn aggregate(
    index: &Value,
    lookup: &GroupLookup,
    known_scores: &BTreeSet<String>,
) -> Result<IndexAgg> {
    ensure!(
        index.get("schema").and_then(Value::as_str) == Some(INDEX_SCHEMA),
        "GP index schema is not {INDEX_SCHEMA}"
    );
    let runs = index
        .get("runs")
        .and_then(Value::as_array)
        .context("GP index lacks runs")?;
    let counts = index.get("counts").context("GP index lacks counts")?;
    let declared = counts
        .get("runs")
        .and_then(Value::as_u64)
        .context("GP index counts lack runs")?;
    ensure!(
        declared == runs.len() as u64,
        "GP index counts.runs {declared} != {} rows",
        runs.len()
    );
    let mut agg = IndexAgg {
        rows_total: runs.len() as u64,
        ..Default::default()
    };
    for (n, r) in runs.iter().enumerate() {
        let qualified = match r.get("qualified") {
            Some(Value::Bool(b)) => *b,
            // Missing or non-boolean qualification is unknown: treat as a parse failure.
            _ => bail!("GP index row {n}: qualified is not a boolean"),
        };
        if qualified {
            agg.qualified_total += 1;
        }
        let scored = r.get("score").is_some_and(|v| !v.is_null());
        let family = s(r, "family").unwrap_or_else(|| "unknown".into());
        if scored {
            let name = s(r, "score_name")
                .with_context(|| format!("GP index row {n}: score without score_name"))?;
            if !known_scores.contains(&name) {
                bail!("GP index row {n}: unknown score identity {name:?} (fail closed)");
            }
            ensure!(
                r.get("score")
                    .and_then(Value::as_f64)
                    .is_some_and(f64::is_finite),
                "GP index row {n}: score is not a finite number"
            );
            *agg.scored_by_family.entry(family.clone()).or_default() += 1;
        }
        let (Some(bucket), Some(key)) = (s(r, "bucket"), s(r, "key")) else {
            agg.rows_local += 1;
            continue;
        };
        agg.rows_s3 += 1;
        let Some(gi) = lookup.find(&bucket, &key) else {
            agg.rows_unmatched.push(format!("s3://{bucket}/{key}"));
            continue;
        };
        let g = agg.per_group.entry(gi).or_default();
        g.rows += 1;
        if qualified {
            g.qualified_rows += 1;
        }
        let consumer = s(r, "consumer").unwrap_or_default().to_ascii_lowercase();
        if !consumer.is_empty() && !consumer.starts_with("none") && !consumer.starts_with("no ") {
            g.consumer_recorded_rows += 1;
        }
        *g.by_grain
            .entry(s(r, "grain").unwrap_or_else(|| "unknown".into()))
            .or_default() += 1;
        g.families.insert(family);
        let program = s(r, "program_sha");
        let context = s(r, "context_sha256");
        if let Some(p) = &program {
            g.programs.insert(p.clone());
        }
        if let Some(c) = &context {
            g.contexts.insert(c.clone());
        }
        if scored {
            g.scored_rows += 1;
            if let Some(name) = s(r, "score_name") {
                g.score_names.insert(name);
            }
            if let (Some(c), Some(p)) = (context, program) {
                g.scored_by_context.entry(c).or_default().insert(p);
            }
        }
    }
    if let Some(q) = counts.get("qualified_true").and_then(Value::as_u64) {
        ensure!(
            q == agg.qualified_total,
            "GP index counts.qualified_true {q} != {} rows",
            agg.qualified_total
        );
    }
    Ok(agg)
}
