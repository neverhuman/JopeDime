//! A verified CPU pipeline run (`jope.pipeline-receipt/v2` or `v3`) → `jope.hwm/v1`.
//!
//! The receipt is re-derived from its run directory first
//! ([`jope_pipeline::evidence::verify_champion_run`]); a receipt its own evidence does not
//! reproduce never reaches the ledger. The record's comparison identity names:
//!
//! * `dataset_hash` — the catalog content hash the run resolved and byte-verified;
//! * `cut_id` — sha256 of the canonical JSON naming the prepared cuts: `prepared.json`'s sha256,
//!   the fold seed and partition, the searched cut indices and `train.csv`'s sha256;
//! * `protocol` — the pipeline measurement protocol plus the evaluator: search engine, tool
//!   version and the **source commit** the receipt records (`git_sha`: `--source-commit`,
//!   `JOPE_GIT_SHA` or `git rev-parse HEAD`); the selection and budget rules; and the
//!   random-stream protocol;
//! * `comparison` — the numerical policy, the held-out stage and mean-over-cuts aggregation.
//!
//! The tool version (`jope-pipeline/<workspace version>`) changes only when a release bumps the
//! workspace version, and the engine string is a constant, so the source commit is what tells one
//! evaluator from another: a receipt measured by another commit is another identity and never
//! displaces this one, and a receipt that names no commit is refused, because nothing would tell
//! its evaluator apart from any other. What the commit does not bind: uncommitted changes in the
//! tree the binary was built from, and the binary itself (`evaluator_sha256` is `unbound`, so the
//! record is not replayable). The metric is not part of the identity; `hwm::decide` compares it
//! separately (`metric-mismatch`).
//!
//! The stage is the development hold-out, never `train_oof`: these records are **not**
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

/// Constant identifying the source.
pub const SOURCE: &str = "pipeline";
/// The measurement: fit on a cut's training folds, score RMSE on its held-out fold, mean over
/// the requested cuts, complete coverage required.
pub const PROTOCOL: &str = "jope.pipeline-holdout/v1";
/// How the search derives every random stream (`search::stream`, `islands::island_seed`).
pub const RNG_PROTOCOL: &str = "splitmix64-coordinate/v1";
/// Constant identifying the stage.
pub const STAGE: &str = "pipeline_holdout";
/// Constant identifying the aggregation.
pub const AGGREGATION: &str = "mean_over_requested_cuts";
/// Constant identifying the numerical policy.
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

/// The commit that measured `receipt`, or the refusal: without it the evaluator is unbound.
pub fn source_commit(receipt: &jope_pipeline::Receipt) -> Result<&str> {
    receipt
        .git_sha
        .as_deref()
        .filter(|commit| !commit.trim().is_empty())
        .context(
            "the receipt names no source commit (git_sha): the code that measured it cannot be \
             told apart from any other, so its score cannot be compared; re-run with \
             --source-commit or from a checkout",
        )
}

/// The comparison identity of a verified receipt's measurement.
pub fn identity(receipt: &jope_pipeline::Receipt) -> Result<HwmIdentity> {
    let train = receipt
        .dataset
        .files
        .iter()
        .find(|f| f.name == "train.csv")
        .context("the receipt names no train.csv digest")?;
    let commit = source_commit(receipt)?;
    let search = &receipt.search;
    // The ledger and the champion bank score a champion by held-out RMSE; a search that ranked
    // its programs by another objective chose a champion RMSE would not have, so it is never
    // compared with them.
    if let Some(objective) = &search.objective {
        anyhow::bail!(
            "the receipt's search ranked programs by the {} objective, not held-out RMSE; the \
             high-water mark and the champion bank compare RMSE-ranked searches only",
            objective.name
        );
    }
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
            "{PROTOCOL};engine={};tool={};source_commit={commit};selection={};budget_policy={};\
             rng={RNG_PROTOCOL}",
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
/// `PipelineScan` is a record for the pipeline scan contract in this module: A verified CPU pipeline run (jope.pipeline-receipt/v2 or v3) → jope.hwm/v1.
pub struct PipelineScan {
    /// Receipt recorded for this pipeline scan.
    pub receipt: String,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// Selected genome recorded for this pipeline scan.
    pub selected_genome: String,
    /// Validation rmse recorded for this pipeline scan.
    pub validation_rmse: f64,
    /// Collection of cuts values carried by this pipeline scan.
    pub cuts: Vec<u64>,
    /// SHA-256 digest identifying the associated parent receipt.
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
            version: format!(
                "{} ({}) at {}",
                receipt.tool_version,
                receipt.search.engine,
                source_commit(receipt)?
            ),
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

/// Performs the scan source operation.
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
