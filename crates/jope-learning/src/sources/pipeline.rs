//! A verified CPU pipeline run (`jope.pipeline-receipt/v2` or `v3`) → `jope.hwm/v1`.
//!
//! The receipt is re-derived from its run directory first
//! ([`jope_pipeline::evidence::verify_champion_run`]); a receipt its own evidence does not
//! reproduce never reaches the ledger. The record's comparison identity is complete:
//!
//! * `dataset_hash` — the catalog content hash the run resolved and byte-verified;
//! * `cut_id` — sha256 of the canonical JSON naming the prepared cuts: `prepared.json`'s sha256,
//!   the fold seed and partition, the searched cut indices and `train.csv`'s sha256;
//! * `protocol` — the pipeline measurement protocol plus the evaluator (search engine and tool
//!   version), the selection and budget rules, and the random-stream protocol;
//! * `comparison` — the numerical policy, the held-out stage and mean-over-cuts aggregation.
//!
//! The evaluator binary is not bound by hash (`unbound`, so the record is not replayable), and
//! the stage is the development hold-out, never `train_oof`: these records are **not**
//! teaching-eligible. `freeze_teaching` excludes them like every record without a measured
//! teaching qualification.

use crate::canon::{canonical_sha256, sha256_hex};
use crate::hwm::Candidate;
use crate::metric::Metric;
use crate::record::*;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::path::Path;

pub const SOURCE: &str = "pipeline";
/// The measurement: fit on a cut's training folds, score RMSE on its held-out fold, mean over
/// the requested cuts, complete coverage required.
pub const PROTOCOL: &str = "jope.pipeline-holdout/v1";
/// How the search derives every random stream (`search::stream`, `islands::island_seed`).
pub const RNG_PROTOCOL: &str = "splitmix64-coordinate/v1";
pub const STAGE: &str = "pipeline_holdout";
pub const AGGREGATION: &str = "mean_over_requested_cuts";
pub const NUMERICAL_POLICY: &str =
    "feat_math::fit f64 in-process; held-out RMSE in the dataset's original target units";

#[derive(Serialize)]
struct CutIdentity<'a> {
    source: &'a str,
    prepared_sha256: &'a str,
    fold_seed: u64,
    folds: u64,
    cuts: &'a [u64],
    train_sha256: &'a str,
}

/// The comparison identity of a verified receipt's measurement.
pub fn identity(receipt: &jope_pipeline::Receipt) -> Result<HwmIdentity> {
    let train = receipt
        .dataset
        .files
        .iter()
        .find(|f| f.name == "train.csv")
        .context("the receipt names no train.csv digest")?;
    let search = &receipt.search;
    Ok(HwmIdentity {
        task: receipt.dataset.task.clone(),
        dataset_hash: receipt.dataset.dataset_hash.clone(),
        cut_id: canonical_sha256(&CutIdentity {
            source: SOURCE,
            prepared_sha256: &receipt.prepared.sha256,
            fold_seed: receipt.prepared.seed,
            folds: receipt.prepared.folds,
            cuts: &receipt.prepared.cuts,
            train_sha256: &train.sha256,
        })?,
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: format!(
            "{PROTOCOL};engine={};tool={};selection={};budget_policy={};rng={RNG_PROTOCOL}",
            search.engine,
            receipt.tool_version,
            search.selection,
            search.budget_policy.as_deref().unwrap_or("none"),
        ),
        comparison: Some(ComparisonContext {
            numerical_policy: NUMERICAL_POLICY.into(),
            stage: STAGE.into(),
            aggregation: AGGREGATION.into(),
            label_role: "validation".into(),
            sealed: false,
            benchmark_valid: false,
            recipe_encoding: "canonical_pipeline_dsl".into(),
            class_order: Vec::new(),
            checkpoint_sha256: None,
        }),
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineScan {
    pub receipt: String,
    pub receipt_sha256: String,
    pub schema: String,
    pub dataset_hash: String,
    pub selected_genome: String,
    pub validation_rmse: f64,
    pub cuts: Vec<u64>,
    pub parent_receipt_sha256: Option<String>,
}

/// Verify `receipt_path` against its evidence and map its champion to one HWM candidate whose
/// blob is the receipt's exact bytes.
pub fn scan(
    receipt_path: &Path,
    bundle_id: Option<&str>,
) -> Result<(Vec<Candidate>, PipelineScan)> {
    let run = jope_pipeline::evidence::verify_champion_run(receipt_path)?;
    let blob =
        std::fs::read(receipt_path).with_context(|| format!("read {}", receipt_path.display()))?;
    ensure!(
        sha256_hex(&blob) == run.receipt_sha256,
        "{} changed while it was being verified",
        receipt_path.display()
    );
    let receipt = &run.receipt;
    let best = receipt
        .search
        .best
        .as_ref()
        .context("the receipt names no measured champion")?;
    let identity = identity(receipt)?;
    let ts_unix = crate::civil::parse_rfc3339(&receipt.created_at)
        .with_context(|| format!("created_at {:?} is not RFC 3339", receipt.created_at))?;
    let bundle = bundle_id.map(str::to_owned).unwrap_or_else(|| {
        receipt_path
            .parent()
            .and_then(Path::file_name)
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| UNBOUND.into())
    });
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256()?,
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            dsl: best.genome.clone(),
            program_sha256: best.program_sha256.clone(),
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value: best.validation_rmse,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost {
            runtime_seconds: Some(best.seconds),
            search_seconds: Some(
                receipt
                    .search
                    .cuts
                    .iter()
                    .filter_map(|c| c.accounting.as_ref().map(|a| a.active_seconds))
                    .sum(),
            ),
        },
        evaluator: Evaluator {
            sha256: UNBOUND.into(),
            version: format!("{} ({})", receipt.tool_version, receipt.search.engine),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: false,
        provenance: Provenance {
            source: SOURCE.into(),
            run_id: format!("{}/{}", receipt.dataset.id, bundle),
            git_sha: receipt.git_sha.clone().unwrap_or_else(|| UNBOUND.into()),
            bundle_id: bundle,
            record_sha256: run.receipt_sha256.clone(),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix,
    };
    record.validate()?;
    let scan = PipelineScan {
        receipt: receipt_path.display().to_string(),
        receipt_sha256: run.receipt_sha256.clone(),
        schema: receipt.schema.clone(),
        dataset_hash: receipt.dataset.dataset_hash.clone(),
        selected_genome: best.genome.clone(),
        validation_rmse: best.validation_rmse,
        cuts: receipt.prepared.cuts.clone(),
        parent_receipt_sha256: receipt
            .search
            .seeding
            .as_ref()
            .and_then(|s| s.parent_receipt_sha256.clone()),
    };
    Ok((
        vec![Candidate {
            record,
            blob,
            chain_from_source: false,
        }],
        scan,
    ))
}

pub fn scan_source(path: &Path, scan: &PipelineScan) -> serde_json::Value {
    crate::hwm::source_value(
        SOURCE,
        path,
        json!({
            "receipt_sha256": scan.receipt_sha256,
            "schema": scan.schema,
            "dataset_hash": scan.dataset_hash,
            "validation_rmse": scan.validation_rmse,
            "parent_receipt_sha256": scan.parent_receipt_sha256,
        }),
    )
}
