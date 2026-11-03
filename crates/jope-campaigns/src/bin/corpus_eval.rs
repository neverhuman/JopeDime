//! corpus_eval — JopeFlywheel massive direct-test eval lane worker.
//!
//! Evaluates a shard of candidate genomes against one dataset's frozen
//! train/test split and writes one receipt line per candidate. Legacy shards
//! are test-adaptive and never benchmark-valid. Current self-play shards must
//! disclose reused-test selection bias, name their actual strict wave, and pin
//! the immutable seed, checkpoint, and hall-of-fame snapshots. Superseded
//! generation-zero manifests remain readable so queued historical work can
//! drain without rewriting its receipts. The regression branch mirrors
//! `scq_batch::evaluate_request_with_data` (feat_math::model::fit_predict +
//! the same sequential-accumulation rmse/r2), so its rmse is bit-comparable
//! with `JopeTestAdaptiveDatasetResult.v1::final_rmse`. The binary branch
//! mirrors `binary_cv::evaluate_test_adaptive_binary` (advanced-v3 raw string
//! table + fit_table/predict_table with TaskSpec binary/RocAuc).
//!
//! Timeout semantics: each candidate is evaluated by a private `eval-one`
//! child in its own process group. On expiry the parent sends SIGTERM, waits a
//! fixed grace period, sends SIGKILL if necessary, and reaps the group before
//! the scheduling slot is reused. A child may never publish a receipt.
//!
//! Modes:
//!   `corpus_eval --manifest <shard.json> --output-dir <dir> [--threads N]`
//!               [--timeout-seconds 300]
//!   `corpus_eval eval-one --manifest <shard.json> --candidate-index N --result <path>`
//!   `corpus_eval validate-manifest --manifest <shard.json>`
//!   `corpus_eval rewrite-binary-heads --winners-dir <dir> \
//!               --heads logit_ridge,gam_boost,tab_boost --output <jsonl>`

use anyhow::{bail, Context, Result};
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::Stage;
use feat_math::genome::GenomeIr;
use feat_math::heads::is_head_op;
use feat_math::target_transform::is_target_op;
use feat_math::{
    auc, fit_predict, fit_table, logloss, multiclass_accuracy, r2, rmse, ColumnSpec, DataSchema,
    Objective, RawTable, RowSemantics, Rows, TaskSpec,
};
use jope_dime_regressor::task::Task;
use jope_dime_regressor::timeseries::{
    make_direct_forecast_rows, make_direct_reduction, make_forecast_features,
    make_supervised_reduction, read_timeseries_csv_bytes, score_timeseries,
    validate_timeseries_data, ReductionVersion, TimeSeriesData, TimeSeriesManifestSpec,
    TimeSeriesMetrics, TimeSeriesTable,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs};

const SHARD_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
/// v2 differs from v1 only by the REQUIRED `reduction.version` pin on
/// timeseries shards. The schema-string bump is what fails closed: `Shard`
/// has no deny_unknown_fields, so an optional field on a v1 manifest would be
/// silently ignored by old binaries.
const SHARD_SCHEMA_V2: &str = "JopeFlywheelCorpusEvalShard.v2";
const RECEIPT_SCHEMA: &str = "JopeFlywheelCorpusEvalReceipt.v1";
/// Binary heads accepted by the schema-bound grammar that are NOT in the
/// legacy `heads::is_head_op` list (mirrors `pipeline::is_v3_head` extras).
const V3_EXTRA_HEADS: [&str; 9] = [
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "pls",
    "logit_ridge",
    "multiclass_ridge",
    "multiclass_ovr_logistic",
    "multiclass_diagonal_lda",
    "multiclass_nearest_centroid",
];

// ---------------------------------------------------------------------------
// wire formats
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Shard {
    schema: String,
    task: String,
    dataset_hash: String,
    train_csv: String,
    test_csv: String,
    train_sha256: String,
    test_sha256: String,
    #[serde(default)]
    checkpoint_epoch: Option<String>,
    #[serde(default)]
    checkpoint_sha256: Option<String>,
    #[serde(default)]
    catalog_sha256: Option<String>,
    #[serde(default)]
    protocol_sha256: Option<String>,
    #[serde(default)]
    generated_at_ms: Option<u128>,
    #[serde(default)]
    label_authority: Option<LabelAuthority>,
    #[serde(default)]
    meta_json: Option<String>,
    #[serde(default)]
    meta_sha256: Option<String>,
    #[serde(default)]
    horizon: Option<usize>,
    #[serde(default)]
    seasonality: Option<usize>,
    #[serde(default)]
    frequency: Option<String>,
    #[serde(default)]
    target_is_final_column: Option<bool>,
    #[serde(default, alias = "known_dynamic_columns")]
    known_future_covariates: Vec<String>,
    #[serde(default)]
    past_only_covariates: Vec<String>,
    #[serde(default)]
    reduction: Option<ReductionSpec>,
    candidates: Vec<ShardCandidate>,
}

#[derive(Deserialize)]
struct ReductionSpec {
    version: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct LabelAuthority {
    score_tier: String,
    metric: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    #[serde(default)]
    selection_bias: Option<String>,
    generation: String,
    frozen_seed_sha256: String,
    model_checkpoint_sha256: String,
    hall_of_fame_sha256: String,
}

#[derive(Clone, Deserialize)]
struct ShardCandidate {
    semantic_sha256: String,
    canonical_genome: String,
}

#[derive(Serialize)]
struct Receipt {
    schema: String,
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    canonical_genome: String,
    tier: String,
    metric: String,
    score: Option<f64>,
    secondary_metric: String,
    secondary_score: Option<f64>,
    balanced_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mae: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    smape: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wape: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mase_denominator: Option<f64>,
    /// Verbatim manifest reduction pin; absent on v1-schema shards, so every
    /// historical receipt stays byte-identical. Absence means ts-reduction-v1.
    #[serde(skip_serializing_if = "Option::is_none")]
    reduction_version: Option<String>,
    predictions_sha256: Option<String>,
    predictions_path: Option<String>,
    prediction_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    class_labels: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    probabilities_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    probabilities_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    probability_shape: Option<[usize; 2]>,
    #[serde(skip)]
    prediction_bytes: Option<Vec<u8>>,
    #[serde(skip)]
    probability_bytes: Option<Vec<u8>>,
    valid: bool,
    failure: Option<String>,
    failure_message: Option<String>,
    runtime_seconds: f64,
    train_sha256: String,
    test_sha256: String,
    node: String,
    manifest_sha256: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    selection_bias: Option<String>,
    generation: String,
    label_authority_sha256: Option<String>,
    frozen_seed_sha256: Option<String>,
    model_checkpoint_sha256: Option<String>,
    hall_of_fame_sha256: Option<String>,
    checkpoint_epoch: Option<String>,
    catalog_sha256: Option<String>,
    protocol_sha256: Option<String>,
    manifest_generated_at_ms: Option<u128>,
    created_at_ms: u128,
}

struct Data {
    x: Rows,
    y: Vec<f64>,
    features: usize,
}

#[derive(Clone, Debug)]
struct MulticlassContext {
    /// Fitted class index -> declared metadata class index.
    observed_declared: Vec<i64>,
    declared_classes: usize,
}

#[derive(Debug)]
struct DecodedMulticlassSplits {
    train: Vec<f64>,
    test: Vec<f64>,
    context: MulticlassContext,
}

enum Outcome {
    Score {
        score: f64,
        secondary: f64,
        balanced_accuracy: Option<f64>,
        timeseries_metrics: Option<TimeSeriesMetrics>,
        predictions: Vec<f64>,
        multiclass: Option<MulticlassEvidence>,
    },
    Fail {
        class: &'static str,
        message: String,
    },
}

struct MulticlassEvidence {
    classes: Vec<i64>,
    probabilities: Vec<Vec<f64>>,
}

/// The private `eval-one` wire format deliberately contains only the
/// candidate outcome.  The parent remains the sole writer of receipts and
/// prediction artifacts, so a killed child can never publish a partial
/// receipt.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EvalOneWire {
    Score {
        score: f64,
        secondary: f64,
        balanced_accuracy: Option<f64>,
        timeseries_metrics: Option<TimeSeriesMetricsWire>,
        predictions: Vec<f64>,
        classes: Option<Vec<i64>>,
        probabilities: Option<Vec<Vec<f64>>>,
    },
    Fail {
        class: String,
        message: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
struct TimeSeriesMetricsWire {
    mase: f64,
    rmse: f64,
    mse: f64,
    mae: f64,
    smape: f64,
    wape: f64,
    prediction_min: f64,
    prediction_max: f64,
    target_min: f64,
    target_max: f64,
    mase_denominator: f64,
}

impl From<&TimeSeriesMetrics> for TimeSeriesMetricsWire {
    fn from(metrics: &TimeSeriesMetrics) -> Self {
        Self {
            mase: metrics.mase,
            rmse: metrics.rmse,
            mse: metrics.mse,
            mae: metrics.mae,
            smape: metrics.smape,
            wape: metrics.wape,
            prediction_min: metrics.prediction_min,
            prediction_max: metrics.prediction_max,
            target_min: metrics.target_min,
            target_max: metrics.target_max,
            mase_denominator: metrics.mase_denominator,
        }
    }
}

impl From<TimeSeriesMetricsWire> for TimeSeriesMetrics {
    fn from(metrics: TimeSeriesMetricsWire) -> Self {
        Self {
            mase: metrics.mase,
            rmse: metrics.rmse,
            mse: metrics.mse,
            mae: metrics.mae,
            smape: metrics.smape,
            wape: metrics.wape,
            prediction_min: metrics.prediction_min,
            prediction_max: metrics.prediction_max,
            target_min: metrics.target_min,
            target_max: metrics.target_max,
            mase_denominator: metrics.mase_denominator,
        }
    }
}

impl From<Outcome> for EvalOneWire {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Score {
                score,
                secondary,
                balanced_accuracy,
                timeseries_metrics,
                predictions,
                multiclass,
            } => Self::Score {
                score,
                secondary,
                balanced_accuracy,
                timeseries_metrics: timeseries_metrics.as_ref().map(Into::into),
                predictions,
                classes: multiclass.as_ref().map(|value| value.classes.clone()),
                probabilities: multiclass.map(|value| value.probabilities),
            },
            Outcome::Fail { class, message } => Self::Fail {
                class: class.to_owned(),
                message,
            },
        }
    }
}

impl TryFrom<EvalOneWire> for Outcome {
    type Error = String;

    fn try_from(wire: EvalOneWire) -> Result<Self, Self::Error> {
        match wire {
            EvalOneWire::Score {
                score,
                secondary,
                balanced_accuracy,
                timeseries_metrics,
                predictions,
                classes,
                probabilities,
            } => {
                let multiclass = match (classes, probabilities) {
                    (None, None) => None,
                    (Some(classes), Some(probabilities)) => Some(MulticlassEvidence {
                        classes,
                        probabilities,
                    }),
                    _ => return Err("eval-one score has incomplete multiclass evidence".into()),
                };
                Ok(Self::Score {
                    score,
                    secondary,
                    balanced_accuracy,
                    timeseries_metrics: timeseries_metrics.map(Into::into),
                    predictions,
                    multiclass,
                })
            }
            EvalOneWire::Fail { class, message } => {
                let class = match class.as_str() {
                    "panic" => "panic",
                    "nonfinite" => "nonfinite",
                    "timeout" => "timeout",
                    "fit" => "fit",
                    "parse" => "parse",
                    _ => return Err(format!("eval-one returned unknown failure class {class}")),
                };
                Ok(Self::Fail { class, message })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

struct ResolvedAuthority {
    tier: String,
    metric: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    selection_bias: Option<String>,
    generation: String,
    authority_sha256: Option<String>,
    frozen_seed_sha256: Option<String>,
    model_checkpoint_sha256: Option<String>,
    hall_of_fame_sha256: Option<String>,
}

#[path = "corpus_eval/args.rs"]
mod args;
#[path = "corpus_eval/data.rs"]
mod data;
#[path = "corpus_eval/evaluate.rs"]
mod evaluate;
#[path = "corpus_eval/multiclass.rs"]
mod multiclass;
#[path = "corpus_eval/process.rs"]
mod process;
#[path = "corpus_eval/runner.rs"]
mod runner;

use args::flag_value;
use data::*;
use evaluate::*;
use multiclass::*;
use process::*;
use runner::*;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("eval-one") {
        let manifest = flag_value(&args, "--manifest").context("--manifest is required")?;
        let candidate_index = flag_value(&args, "--candidate-index")
            .context("--candidate-index is required")?
            .parse::<usize>()
            .context("--candidate-index must be an integer")?;
        let result = flag_value(&args, "--result").context("--result is required")?;
        return run_eval_one(Path::new(&manifest), candidate_index, Path::new(&result));
    }
    if args.first().map(String::as_str) == Some("validate-manifest") {
        let manifest = flag_value(&args, "--manifest").context("--manifest is required")?;
        validate_manifest(Path::new(&manifest))?;
        println!("manifest ok");
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("rewrite-binary-heads") {
        let winners_dir =
            flag_value(&args, "--winners-dir").context("--winners-dir is required")?;
        let heads = flag_value(&args, "--heads")
            .unwrap_or_else(|| "logit_ridge,gam_boost,tab_boost".to_string())
            .split(',')
            .map(|h| h.trim().to_string())
            .filter(|h| !h.is_empty())
            .collect::<Vec<_>>();
        let output = flag_value(&args, "--output").context("--output is required")?;
        return run_rewrite(Path::new(&winners_dir), &heads, Path::new(&output));
    }
    let manifest = flag_value(&args, "--manifest").context("--manifest is required")?;
    let output_dir = flag_value(&args, "--output-dir").context("--output-dir is required")?;
    let threads = flag_value(&args, "--threads")
        .map(|v| v.parse::<usize>().context("--threads must be an integer"))
        .transpose()?
        .unwrap_or(1)
        .max(1);
    let timeout_seconds = flag_value(&args, "--timeout-seconds")
        .map(|v| {
            v.parse::<u64>()
                .context("--timeout-seconds must be an integer")
        })
        .transpose()?
        .unwrap_or(300);
    run_eval(
        &PathBuf::from(manifest),
        &PathBuf::from(output_dir),
        threads,
        Duration::from_secs(timeout_seconds),
    )
}

#[cfg(test)]
#[path = "corpus_eval/tests.rs"]
mod tests;
