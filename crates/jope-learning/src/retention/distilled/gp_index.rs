use super::*;

use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Accounting summary for [`distill_gp_results_index_with_stats`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpIndexDistillStats {
    /// Total index runs recorded for this GP index distill stats.
    pub total_index_runs: usize,
    /// Number of distilled rows represented here.
    pub distilled_rows: usize,
    /// Skipped unscored runs recorded for this GP index distill stats.
    pub skipped_unscored_runs: usize,
    /// Skipped incomplete provenance runs recorded for this GP index distill stats.
    pub skipped_incomplete_provenance_runs: usize,
    /// Reconciled s3 keys recorded for this GP index distill stats.
    pub reconciled_s3_keys: usize,
}

/// Additive complete archive. Unlike the v1 scored-only API, this preserves every supplied
/// occurrence and keeps missing producer/role/event evidence held. It grants no replay access.
pub fn import_gp_results_index_complete(
    index_bytes: &[u8],
    producer: Option<crate::replay::index_producer::ProducerInput<'_>>,
    known_scores: &BTreeSet<String>,
    dependencies: &BTreeMap<String, crate::replay::observation::DependencyEvidence>,
    historical: Option<&crate::replay::index_producer::HistoricalFixture>,
) -> Result<crate::replay::observation::ObservationArchive> {
    crate::replay::observation::import_index(
        index_bytes,
        producer,
        known_scores,
        dependencies,
        historical,
    )
}

/// Exact legacy compatibility projection, including original order/ranks/skip statistics and
/// refusals. Explicit caller role is diagnostic v1 metadata, never an I04 replay capability.
pub fn legacy_scored_projection(
    archive: &crate::replay::observation::ObservationArchive,
    diagnostic_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>, GpIndexDistillStats)> {
    let index = crate::replay::index_producer::parse_index(archive.original_index.as_bytes())?;
    distill_gp_results_index_with_stats(&index, diagnostic_split_role, known_scores)
}

/// Distills all S3 context-level GP records from a `jopedime.gp-results-index/v1` document
/// (`GP_RESULTS_INDEX.json`) into validated `gp_distilled/v1` rows and returns both the rows and
/// the set of distilled S3 object keys (`raw_keys`) after checking [`reconcile`].
pub fn distill_gp_results_index(
    index: &serde_json::Value,
    default_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>)> {
    let (rows, raw_keys, _) =
        distill_gp_results_index_with_stats(index, default_split_role, known_scores)?;
    Ok((rows, raw_keys))
}

/// Distills all S3 context-level GP records from a `jopedime.gp-results-index/v1` document
/// (`GP_RESULTS_INDEX.json`) into validated `gp_distilled/v1` rows and returns the rows,
/// reconciled S3 keys, and exact row-accounting statistics across all `runs` entries.
pub fn distill_gp_results_index_with_stats(
    index: &serde_json::Value,
    default_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>, GpIndexDistillStats)> {
    ensure!(
        index.get("schema").and_then(serde_json::Value::as_str)
            == Some("jopedime.gp-results-index/v1"),
        "expected schema \"jopedime.gp-results-index/v1\""
    );
    let runs = index
        .get("runs")
        .and_then(serde_json::Value::as_array)
        .context("gp-results-index missing runs array")?;

    type GroupKey = (String, String, Option<i64>, String, String, String, String);
    let mut grouped: BTreeMap<GroupKey, Vec<SelfPlayDistillObservation>> = BTreeMap::new();
    let mut raw_keys = BTreeSet::new();
    let mut skipped_unscored_runs = 0usize;
    let mut skipped_incomplete_provenance_runs = 0usize;

    for (idx, r) in runs.iter().enumerate() {
        let Some(score_name) = r.get("score_name").and_then(serde_json::Value::as_str) else {
            skipped_unscored_runs += 1;
            continue;
        };
        let Some(score) = r.get("score").and_then(serde_json::Value::as_f64) else {
            skipped_unscored_runs += 1;
            continue;
        };
        let (Some(bucket), Some(key), Some(obj_sha), Some(prog_sha), Some(dataset), Some(ctx_sha)) = (
            r.get("bucket").and_then(serde_json::Value::as_str),
            r.get("key").and_then(serde_json::Value::as_str),
            r.get("object_sha256").and_then(serde_json::Value::as_str),
            r.get("program_sha").and_then(serde_json::Value::as_str),
            r.get("dataset").and_then(serde_json::Value::as_str),
            r.get("context_sha256").and_then(serde_json::Value::as_str),
        ) else {
            skipped_incomplete_provenance_runs += 1;
            continue;
        };
        ensure!(
            r.get("qualified")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                == (score_name == "l6"),
            "gp-results-index row {idx}: qualified must equal (score_name == \"l6\")"
        );
        let fold = r.get("fold").and_then(serde_json::Value::as_i64);
        let family = r
            .get("family")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gp-policy-dl");
        let run_id = r
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            .or_else(|| r.get("eval").and_then(serde_json::Value::as_str))
            .or_else(|| r.get("grain").and_then(serde_json::Value::as_str))
            .unwrap_or("gp-hwm");
        let program = r
            .get("program")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let version_id = r
            .get("version_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let grain = r
            .get("grain")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);

        ensure!(
            raw_keys.insert(key.to_owned()),
            "gp-results-index row {idx}: duplicate S3 key {key:?}"
        );
        grouped
            .entry((
                dataset.to_owned(),
                ctx_sha.to_owned(),
                fold,
                score_name.to_owned(),
                family.to_owned(),
                run_id.to_owned(),
                bucket.to_owned(),
            ))
            .or_default()
            .push(SelfPlayDistillObservation {
                program_sha: prog_sha.to_owned(),
                program,
                outcome: DistilledOutcome::Ok,
                score: Some(score),
                seed_packet: grain,
                source_key: key.to_owned(),
                source_version_id: version_id,
                source_sha256: obj_sha.to_owned(),
            });
    }

    ensure!(
        !grouped.is_empty(),
        "gp-results-index contained no distillable S3 context-level GP records"
    );
    let mut rows = Vec::with_capacity(raw_keys.len());
    for ((dataset, context_sha256, fold, score_name, family, run_id, source_bucket), obs) in grouped
    {
        let spec = SelfPlayDistillSpec {
            dataset,
            context_sha256,
            fold,
            split_role: default_split_role,
            score_name,
            policylm_ckpt: None,
            campaign: None,
            run_id,
            family,
            source_bucket,
        };
        let mut batch = distill_selfplay_observations(&spec, &obs, known_scores)?;
        rows.append(&mut batch);
    }
    reconcile(&rows, &raw_keys)?;
    let stats = GpIndexDistillStats {
        total_index_runs: runs.len(),
        distilled_rows: rows.len(),
        skipped_unscored_runs,
        skipped_incomplete_provenance_runs,
        reconciled_s3_keys: raw_keys.len(),
    };
    ensure!(
        stats.total_index_runs
            == stats.distilled_rows
                + stats.skipped_unscored_runs
                + stats.skipped_incomplete_provenance_runs,
        "gp-results-index row accounting mismatch"
    );
    Ok((rows, raw_keys, stats))
}
