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
//! Timeout semantics (documented tradeoff): each candidate is evaluated on a
//! dedicated OS thread while the dispatching rayon worker blocks on
//! `recv_timeout`. On expiry the candidate is receipted as `failure:"timeout"`
//! and the evaluation thread is abandoned (it keeps burning one core until the
//! process exits — acceptable for single-shard worker processes, which exit
//! right after the shard completes). Timeout is therefore wall-clock accurate
//! for receipts but only best-effort for CPU reclamation.
//!
//! Modes:
//!   corpus_eval --manifest <shard.json> --output-dir <dir> [--threads N]
//!               [--timeout-seconds 300]
//!   corpus_eval validate-manifest --manifest <shard.json>
//!   corpus_eval rewrite-binary-heads --winners-dir <dir> \
//!               --heads logit_ridge,gam_boost,tab_boost --output <jsonl>

use anyhow::{bail, Context, Result};
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::Stage;
use feat_math::genome::GenomeIr;
use feat_math::heads::is_head_op;
use feat_math::target_transform::is_target_op;
use feat_math::{
    auc, fit_predict, fit_table, logloss, r2, rmse, ColumnSpec, DataSchema, Objective, RawTable,
    RowSemantics, Rows, TaskSpec,
};
use jope_dime_regressor::timeseries::{
    make_direct_forecast_rows, make_direct_reduction, make_forecast_features,
    make_supervised_reduction, read_timeseries_csv_bytes, score_timeseries,
    validate_timeseries_data, ReductionVersion, TimeSeriesData, TimeSeriesManifestSpec,
    TimeSeriesMetrics, TimeSeriesTable,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs, thread};

const SHARD_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
/// v2 differs from v1 only by the REQUIRED `reduction.version` pin on
/// timeseries shards. The schema-string bump is what fails closed: `Shard`
/// has no deny_unknown_fields, so an optional field on a v1 manifest would be
/// silently ignored by old binaries.
const SHARD_SCHEMA_V2: &str = "JopeFlywheelCorpusEvalShard.v2";
const RECEIPT_SCHEMA: &str = "JopeFlywheelCorpusEvalReceipt.v1";
/// Binary heads accepted by the schema-bound grammar that are NOT in the
/// legacy `heads::is_head_op` list (mirrors `pipeline::is_v3_head` extras).
const V3_EXTRA_HEADS: [&str; 5] = [
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "pls",
    "logit_ridge",
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
    #[serde(skip)]
    prediction_bytes: Option<Vec<u8>>,
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

enum Outcome {
    Score {
        score: f64,
        secondary: f64,
        balanced_accuracy: Option<f64>,
        timeseries_metrics: Option<TimeSeriesMetrics>,
        predictions: Vec<f64>,
    },
    Fail {
        class: &'static str,
        message: String,
    },
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(predictions.len() * 8);
    for prediction in predictions {
        bytes.extend_from_slice(&prediction.to_le_bytes());
    }
    bytes
}

fn balanced_accuracy(y: &[f64], predictions: &[f64]) -> f64 {
    let (mut true_positive, mut positive, mut true_negative, mut negative) =
        (0usize, 0usize, 0usize, 0usize);
    for (target, prediction) in y.iter().zip(predictions) {
        if *target == 1.0 {
            positive += 1;
            true_positive += usize::from(*prediction >= 0.5);
        } else {
            negative += 1;
            true_negative += usize::from(*prediction < 0.5);
        }
    }
    0.5 * (true_positive as f64 / positive as f64 + true_negative as f64 / negative as f64)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn short_hostname() -> String {
    Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Headerless numeric CSV, LAST column = target. Mirrors jope-dime's
/// `data::read_csv_text_core(IndicatorMode::None)` exactly: trimmed fields,
/// `str::parse::<f64>`, non-finite features silently become 0.0, targets must
/// be finite, constant width enforced.
fn read_numeric_csv(text: &str, source: &str) -> Result<Data> {
    let (mut x, mut y, mut features) = (Vec::new(), Vec::new(), None);
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("{source}:{} invalid numeric field", line_no + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() < 2 {
            bail!(
                "{source}:{} needs at least one feature and one target",
                line_no + 1
            );
        }
        let width = values.len() - 1;
        match features {
            Some(expected) if expected != width => {
                bail!(
                    "{source}:{} expected {expected} features, found {width}",
                    line_no + 1
                )
            }
            None => features = Some(width),
            _ => {}
        }
        let target = values.pop().expect("length checked");
        if !target.is_finite() {
            bail!("{source}:{} target is not finite", line_no + 1);
        }
        values.iter_mut().for_each(|v| {
            if !v.is_finite() {
                *v = 0.0;
            }
        });
        x.push(values);
        y.push(target);
    }
    let features = features.context("empty CSV")?;
    Ok(Data { x, y, features })
}

fn load_pinned_csv(path: &str, pinned_sha256: &str, label: &str) -> Result<Data> {
    let bytes = fs::read(path).with_context(|| format!("read {label} CSV {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} CSV {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    read_numeric_csv(std::str::from_utf8(&bytes)?, path)
}

fn load_pinned_timeseries_csv(
    path: &str,
    pinned_sha256: &str,
    label: &str,
) -> Result<TimeSeriesTable> {
    let bytes = fs::read(path).with_context(|| format!("read {label} CSV {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} CSV {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    read_timeseries_csv_bytes(&bytes, path)
}

fn load_pinned_json(path: &str, pinned_sha256: &str, label: &str) -> Result<serde_json::Value> {
    let bytes = fs::read(path).with_context(|| format!("read {label} JSON {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} JSON {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    serde_json::from_slice(&bytes).with_context(|| format!("parse {label} JSON {path}"))
}

/// Resolve the pinned reduction version from the shard schema. v1 manifests
/// must not carry a reduction pin (old binaries would ignore it silently);
/// v2 timeseries manifests must carry one; the pin is meaningless for other
/// tasks and is rejected there rather than ignored.
fn resolve_reduction(shard: &Shard) -> Result<ReductionVersion> {
    match shard.schema.as_str() {
        SHARD_SCHEMA => {
            if shard.reduction.is_some() {
                bail!("reduction pin requires shard schema {SHARD_SCHEMA_V2}");
            }
            Ok(ReductionVersion::V1)
        }
        SHARD_SCHEMA_V2 => {
            if shard.task == "timeseries" {
                let spec = shard.reduction.as_ref().with_context(|| {
                    format!("{SHARD_SCHEMA_V2} timeseries shards must pin reduction.version")
                })?;
                ReductionVersion::parse(&spec.version)
            } else {
                if shard.reduction.is_some() {
                    bail!("reduction pins apply only to timeseries shards");
                }
                Ok(ReductionVersion::V1)
            }
        }
        other => bail!(
            "unsupported shard schema '{other}' (expected {SHARD_SCHEMA} or {SHARD_SCHEMA_V2})"
        ),
    }
}

fn timeseries_manifest_spec(shard: &Shard) -> Result<TimeSeriesManifestSpec> {
    Ok(TimeSeriesManifestSpec {
        horizon: shard
            .horizon
            .context("timeseries manifest missing horizon")?,
        seasonality: shard
            .seasonality
            .context("timeseries manifest missing seasonality")?,
        frequency: shard
            .frequency
            .clone()
            .filter(|value| !value.trim().is_empty())
            .context("timeseries manifest missing frequency")?,
        target_is_final_column: shard
            .target_is_final_column
            .context("timeseries manifest missing target_is_final_column")?,
        known_future_covariates: shard.known_future_covariates.clone(),
        past_only_covariates: shard.past_only_covariates.clone(),
    })
}

fn load_timeseries_shard_data(shard: &Shard) -> Result<TimeSeriesData> {
    let meta_json = shard
        .meta_json
        .as_deref()
        .context("timeseries manifest missing meta_json")?;
    let meta_sha256 = shard
        .meta_sha256
        .as_deref()
        .context("timeseries manifest missing meta_sha256")?;
    let train = load_pinned_timeseries_csv(&shard.train_csv, &shard.train_sha256, "train")?;
    let test = load_pinned_timeseries_csv(&shard.test_csv, &shard.test_sha256, "test")?;
    let meta = load_pinned_json(meta_json, meta_sha256, "meta")?;
    let spec = timeseries_manifest_spec(shard)?;
    validate_timeseries_data(train, test, &meta, &spec)
}

fn binary_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// Shortest-round-trip string table (mirrors binary_cv::raw_table).
fn raw_table(features: usize, rows: &Rows) -> Result<RawTable> {
    let headers = (0..features).map(|j| format!("c{j}")).collect::<Vec<_>>();
    let cells = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, cells)
}

fn raw_table_named(headers: &[String], rows: &[Vec<f64>]) -> Result<RawTable> {
    let cells = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers.to_vec(), cells)
}

fn task_metric(task: &str) -> Option<&'static str> {
    match task {
        "regression" => Some("r2"),
        "binary" => Some("auc"),
        "timeseries" => Some("negative_mase"),
        _ => None,
    }
}

fn secondary_metric(task: &str) -> &'static str {
    match task {
        "binary" => "logloss",
        "timeseries" => "mase",
        _ => "rmse",
    }
}

fn is_head_like(op: &str) -> bool {
    is_head_op(op) || V3_EXTRA_HEADS.contains(&op)
}

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

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_strict_wave(value: &str) -> bool {
    value
        .strip_prefix("strict-wave-")
        .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

/// Resolve receipt labeling before loading any data. The default deliberately
/// preserves the legacy adaptive lane. Corrected authority is fail-closed.
/// Superseded generation-zero manifests are accepted only in their exact old
/// form so already-queued jobs can drain without receiving stronger claims.
fn resolve_authority(shard: &Shard) -> Result<ResolvedAuthority> {
    let expected_metric =
        task_metric(&shard.task).with_context(|| format!("unsupported task '{}'", shard.task))?;
    let Some(authority) = &shard.label_authority else {
        return Ok(ResolvedAuthority {
            tier: "test_exact".to_string(),
            metric: expected_metric.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            selection_bias: Some("test-exact-reused".to_string()),
            generation: "adaptive".to_string(),
            authority_sha256: None,
            frozen_seed_sha256: None,
            model_checkpoint_sha256: None,
            hall_of_fame_sha256: None,
        });
    };
    let corrected = authority.score_tier == "test_exact"
        && authority.metric == expected_metric
        && authority.test_adaptive
        && !authority.benchmark_valid
        && authority.selection_bias.as_deref() == Some("test-exact-reused")
        && is_strict_wave(&authority.generation);
    let legacy_superseded = authority.score_tier == "test_exact"
        && shard.task != "timeseries"
        && authority.metric == expected_metric
        && !authority.test_adaptive
        && authority.benchmark_valid
        && authority.selection_bias.is_none()
        && authority.generation == "zero";
    if !corrected && !legacy_superseded {
        bail!(
            "self-play authority requires score_tier=test_exact, metric={expected_metric}, \
             test_adaptive=true, benchmark_valid=false, selection_bias=test-exact-reused, \
             generation=strict-wave-NNNNNN"
        );
    }
    for (name, value) in [
        ("frozen_seed_sha256", authority.frozen_seed_sha256.as_str()),
        (
            "model_checkpoint_sha256",
            authority.model_checkpoint_sha256.as_str(),
        ),
        (
            "hall_of_fame_sha256",
            authority.hall_of_fame_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            bail!("label_authority.{name} must be a 64-character hexadecimal sha256");
        }
    }
    let freshness_fields_present = shard.checkpoint_epoch.is_some()
        || shard.checkpoint_sha256.is_some()
        || shard.catalog_sha256.is_some()
        || shard.protocol_sha256.is_some()
        || shard.generated_at_ms.is_some();
    if freshness_fields_present {
        let checkpoint = shard
            .checkpoint_sha256
            .as_deref()
            .context("fresh manifest is missing checkpoint_sha256")?;
        let epoch = shard
            .checkpoint_epoch
            .as_deref()
            .context("fresh manifest is missing checkpoint_epoch")?;
        let catalog = shard
            .catalog_sha256
            .as_deref()
            .context("fresh manifest is missing catalog_sha256")?;
        let protocol = shard
            .protocol_sha256
            .as_deref()
            .context("fresh manifest is missing protocol_sha256")?;
        if checkpoint != authority.model_checkpoint_sha256
            || epoch != format!("checkpoint-{checkpoint}")
            || !is_sha256(catalog)
            || !is_sha256(protocol)
            || shard.generated_at_ms.is_none()
        {
            bail!("fresh manifest checkpoint/catalog/protocol provenance mismatch");
        }
    }
    let authority_sha256 = sha256_hex(&serde_json::to_vec(authority)?);
    Ok(ResolvedAuthority {
        tier: authority.score_tier.clone(),
        metric: authority.metric.clone(),
        test_adaptive: authority.test_adaptive,
        benchmark_valid: authority.benchmark_valid,
        selection_bias: authority.selection_bias.clone(),
        generation: authority.generation.clone(),
        authority_sha256: Some(authority_sha256),
        frozen_seed_sha256: Some(authority.frozen_seed_sha256.clone()),
        model_checkpoint_sha256: Some(authority.model_checkpoint_sha256.clone()),
        hall_of_fame_sha256: Some(authority.hall_of_fame_sha256.clone()),
    })
}

// ---------------------------------------------------------------------------
// per-candidate evaluation
// ---------------------------------------------------------------------------

fn parse_canonical(candidate: &ShardCandidate) -> std::result::Result<GenomeIr, Outcome> {
    let genome = match GenomeIr::parse(&candidate.canonical_genome) {
        Ok(genome) => genome,
        Err(error) => {
            return Err(Outcome::Fail {
                class: "parse",
                message: format!("{error:#}"),
            })
        }
    };
    if genome.canonical() != candidate.canonical_genome {
        return Err(Outcome::Fail {
            class: "parse",
            message: "genome is not canonical".to_string(),
        });
    }
    Ok(genome)
}

fn evaluate_regression(candidate: &ShardCandidate, train: &Data, test: &Data) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    let predictions = match fit_predict(&genome, &train.x, &train.y, &test.x) {
        Ok(predictions) => predictions,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    if predictions.len() != test.y.len() || predictions.iter().any(|p| !p.is_finite()) {
        return Outcome::Fail {
            class: "nonfinite",
            message: "predictions are missing or non-finite".to_string(),
        };
    }
    let (score, secondary) = (r2(&test.y, &predictions), rmse(&test.y, &predictions));
    if !score.is_finite() || !secondary.is_finite() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "computed metric is non-finite".into(),
        };
    }
    Outcome::Score {
        score,
        secondary,
        balanced_accuracy: None,
        timeseries_metrics: None,
        predictions,
    }
}

fn evaluate_binary(
    candidate: &ShardCandidate,
    train_table: &RawTable,
    train_y: &[f64],
    test_table: &RawTable,
    test_y: &[f64],
    features: usize,
) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    let task = TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
    let schema = binary_schema(features);
    let model = match fit_table(&genome, &task, &schema, train_table, train_y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let predictions = match model.predict_table(test_table) {
        Ok(predictions) => predictions,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    if predictions.len() != test_y.len()
        || predictions
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return Outcome::Fail {
            class: "nonfinite",
            message: "probabilities are missing or outside [0, 1]".to_string(),
        };
    }
    let (score, secondary) = (auc(test_y, &predictions), logloss(test_y, &predictions));
    if !score.is_finite() || !secondary.is_finite() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "computed metric is non-finite".into(),
        };
    }
    Outcome::Score {
        score,
        secondary,
        balanced_accuracy: Some(balanced_accuracy(test_y, &predictions)),
        timeseries_metrics: None,
        predictions,
    }
}

fn terminal_head(genome: &GenomeIr) -> Option<&str> {
    match genome {
        GenomeIr::Pipeline(dsl) => dsl
            .stages
            .last()
            .filter(|stage| is_head_like(&stage.op))
            .map(|stage| stage.op.as_str()),
        _ => None,
    }
}

fn evaluate_timeseries(candidate: &ShardCandidate, data: &TimeSeriesData) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
        return Outcome::Fail {
            class: "task",
            message: "binary classification heads are not valid for timeseries".to_string(),
        };
    }
    let reduction = match make_supervised_reduction(data) {
        Ok(reduction) => reduction,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let schema = DataSchema {
        columns: reduction
            .feature_names
            .iter()
            .map(|name| ColumnSpec::numeric(name.clone()))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_table = match raw_table_named(&reduction.feature_names, &reduction.x) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let task = TaskSpec::regression(Objective::Rmse).expect("rmse is a regression objective");
    let model = match fit_table(&genome, &task, &schema, &train_table, &reduction.y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let mut history = data.train.y.clone();
    let mut predictions = Vec::with_capacity(data.horizon);
    for horizon_index in 0..data.horizon {
        let row = match make_forecast_features(data, &history, horizon_index) {
            Ok(row) => row,
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        let table = match raw_table_named(&reduction.feature_names, &[row]) {
            Ok(table) => table,
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        let prediction = match model.predict_table(&table) {
            Ok(mut values) if values.len() == 1 => values.remove(0),
            Ok(values) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("recursive prediction emitted {} rows", values.len()),
                }
            }
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        if !prediction.is_finite() {
            return Outcome::Fail {
                class: "nonfinite",
                message: "recursive prediction is non-finite".to_string(),
            };
        }
        predictions.push(prediction);
        history.push(prediction);
    }
    let metrics = match score_timeseries(data, &predictions) {
        Ok(metrics) => metrics,
        Err(error) => {
            return Outcome::Fail {
                class: "nonfinite",
                message: format!("{error:#}"),
            }
        }
    };
    Outcome::Score {
        score: -metrics.mase,
        secondary: metrics.mase,
        balanced_accuracy: None,
        timeseries_metrics: Some(metrics),
        predictions,
    }
}

/// v2 evaluation: direct multi-horizon reduction fit in MASE units, one
/// batched predict call over all H forecast rows (no prediction feedback),
/// inverse-scaled before scoring. Same fitness authority as v1 (-MASE).
fn evaluate_timeseries_v2(candidate: &ShardCandidate, data: &TimeSeriesData) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
        return Outcome::Fail {
            class: "task",
            message: "binary classification heads are not valid for timeseries".to_string(),
        };
    }
    let reduction = match make_direct_reduction(data) {
        Ok(reduction) => reduction,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let schema = DataSchema {
        columns: reduction
            .feature_names
            .iter()
            .map(|name| ColumnSpec::numeric(name.clone()))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_table = match raw_table_named(&reduction.feature_names, &reduction.x) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let task = TaskSpec::regression(Objective::Rmse).expect("rmse is a regression objective");
    let model = match fit_table(&genome, &task, &schema, &train_table, &reduction.y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let forecast_rows = match make_direct_forecast_rows(data) {
        Ok(rows) => rows,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let forecast_table = match raw_table_named(&reduction.feature_names, &forecast_rows) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let scaled = match model.predict_table(&forecast_table) {
        Ok(values) if values.len() == data.horizon => values,
        Ok(values) => {
            return Outcome::Fail {
                class: "fit",
                message: format!(
                    "direct prediction emitted {} rows for horizon {}",
                    values.len(),
                    data.horizon
                ),
            }
        }
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let predictions: Vec<f64> = scaled
        .iter()
        .map(|value| value * reduction.target_scale)
        .collect();
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        return Outcome::Fail {
            class: "nonfinite",
            message: "direct prediction is non-finite".to_string(),
        };
    }
    let metrics = match score_timeseries(data, &predictions) {
        Ok(metrics) => metrics,
        Err(error) => {
            return Outcome::Fail {
                class: "nonfinite",
                message: format!("{error:#}"),
            }
        }
    };
    Outcome::Score {
        score: -metrics.mase,
        secondary: metrics.mase,
        balanced_accuracy: None,
        timeseries_metrics: Some(metrics),
        predictions,
    }
}

/// Run `eval_fn` on a dedicated thread with a wall-clock deadline; panics
/// become `failure:"panic"`, deadline expiry becomes `failure:"timeout"` (the
/// abandoned thread is documented in the module header).
fn run_with_timeout(
    eval_fn: impl FnOnce() -> Outcome + Send + 'static,
    timeout: Duration,
) -> Outcome {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(eval_fn));
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(panic)) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "opaque panic payload".to_string());
            Outcome::Fail {
                class: "panic",
                message,
            }
        }
        Err(_) => Outcome::Fail {
            class: "timeout",
            message: format!("exceeded {} second wall-clock budget", timeout.as_secs()),
        },
    }
}

// ---------------------------------------------------------------------------
// eval mode
// ---------------------------------------------------------------------------

fn run_eval(manifest: &Path, output_dir: &Path, threads: usize, timeout: Duration) -> Result<()> {
    let manifest_bytes =
        fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let shard: Shard = serde_json::from_slice(&manifest_bytes).context("parse shard manifest")?;
    let reduction_version = resolve_reduction(&shard)?;
    if task_metric(&shard.task).is_none() {
        bail!("unsupported task '{}'", shard.task);
    }
    let authority = Arc::new(resolve_authority(&shard)?);
    let reduction_stamp = shard.reduction.as_ref().map(|spec| spec.version.clone());
    let node = short_hostname();
    let receipt = |c: &ShardCandidate, sha: String, outcome: Outcome, runtime: f64| {
        let bytes = match &outcome {
            Outcome::Score { predictions, .. } => Some(prediction_bytes(predictions)),
            Outcome::Fail { .. } => None,
        };
        let predictions_sha256 = bytes.as_ref().map(|value| sha256_hex(value));
        Receipt {
            schema: RECEIPT_SCHEMA.to_string(),
            task: shard.task.clone(),
            dataset_hash: shard.dataset_hash.clone(),
            semantic_sha256: sha.clone(),
            canonical_genome: c.canonical_genome.clone(),
            tier: authority.tier.clone(),
            metric: authority.metric.clone(),
            score: match &outcome {
                Outcome::Score { score, .. } => Some(*score),
                Outcome::Fail { .. } => None,
            },
            secondary_metric: secondary_metric(&shard.task).to_string(),
            secondary_score: match &outcome {
                Outcome::Score { secondary, .. } => Some(*secondary),
                Outcome::Fail { .. } => None,
            },
            balanced_accuracy: match &outcome {
                Outcome::Score {
                    balanced_accuracy, ..
                } => *balanced_accuracy,
                Outcome::Fail { .. } => None,
            },
            rmse: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.rmse),
                _ => None,
            },
            mse: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mse),
                _ => None,
            },
            mae: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mae),
                _ => None,
            },
            smape: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.smape),
                _ => None,
            },
            wape: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.wape),
                _ => None,
            },
            prediction_min: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.prediction_min),
                _ => None,
            },
            prediction_max: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.prediction_max),
                _ => None,
            },
            target_min: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.target_min),
                _ => None,
            },
            target_max: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.target_max),
                _ => None,
            },
            mase_denominator: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mase_denominator),
                _ => None,
            },
            reduction_version: reduction_stamp.clone(),
            predictions_sha256,
            predictions_path: bytes.as_ref().map(|_| format!("predictions/{sha}.f64le")),
            prediction_count: match &outcome {
                Outcome::Score { predictions, .. } => Some(predictions.len()),
                Outcome::Fail { .. } => None,
            },
            prediction_bytes: bytes,
            valid: matches!(outcome, Outcome::Score { .. }),
            failure: match &outcome {
                Outcome::Score { .. } => None,
                Outcome::Fail { class, .. } => Some(class.to_string()),
            },
            failure_message: match &outcome {
                Outcome::Score { .. } => None,
                Outcome::Fail { message, .. } => Some(message.clone()),
            },
            runtime_seconds: runtime,
            train_sha256: shard.train_sha256.clone(),
            test_sha256: shard.test_sha256.clone(),
            node: node.clone(),
            manifest_sha256: manifest_sha256.clone(),
            test_adaptive: authority.test_adaptive,
            benchmark_valid: authority.benchmark_valid,
            selection_bias: authority.selection_bias.clone(),
            generation: authority.generation.clone(),
            label_authority_sha256: authority.authority_sha256.clone(),
            frozen_seed_sha256: authority.frozen_seed_sha256.clone(),
            model_checkpoint_sha256: authority.model_checkpoint_sha256.clone(),
            hall_of_fame_sha256: authority.hall_of_fame_sha256.clone(),
            checkpoint_epoch: shard.checkpoint_epoch.clone(),
            catalog_sha256: shard.catalog_sha256.clone(),
            protocol_sha256: shard.protocol_sha256.clone(),
            manifest_generated_at_ms: shard.generated_at_ms,
            created_at_ms: unix_millis(),
        }
    };
    // Verified semantic identity: warn + substitute the computed hash on drift.
    let semantic_sha = |c: &ShardCandidate| {
        let computed = sha256_hex(c.canonical_genome.as_bytes());
        if computed != c.semantic_sha256 {
            eprintln!(
                "warning: candidate semantic_sha256 {} does not match sha256(canonical_genome) {computed}; using computed",
                c.semantic_sha256
            );
        }
        computed
    };

    let receipts: Vec<Receipt> = if shard.task == "timeseries" {
        let data = Arc::new(load_timeseries_shard_data(&shard)?);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        pool.install(|| {
            use rayon::prelude::*;
            shard
                .candidates
                .par_iter()
                .map(|c| {
                    let sha = semantic_sha(c);
                    let started = Instant::now();
                    let candidate = c.clone();
                    let data = Arc::clone(&data);
                    let outcome = run_with_timeout(
                        move || match reduction_version {
                            ReductionVersion::V1 => evaluate_timeseries(&candidate, &data),
                            ReductionVersion::V2 => evaluate_timeseries_v2(&candidate, &data),
                        },
                        timeout,
                    );
                    receipt(c, sha, outcome, started.elapsed().as_secs_f64())
                })
                .collect()
        })
    } else {
        let train = Arc::new(load_pinned_csv(
            &shard.train_csv,
            &shard.train_sha256,
            "train",
        )?);
        let test = Arc::new(load_pinned_csv(
            &shard.test_csv,
            &shard.test_sha256,
            "test",
        )?);
        if train.features != test.features {
            bail!(
                "train/test feature widths differ ({} vs {})",
                train.features,
                test.features
            );
        }
        // Binary dataset admission: targets bitwise 0.0/1.0 with both classes
        // present, else every candidate gets a dataset-level failure receipt.
        let dataset_failure = if shard.task == "binary" {
            validate_binary_targets(&train.y, "train")
                .or_else(|| validate_binary_targets(&test.y, "test"))
        } else {
            None
        };
        if let Some(message) = dataset_failure {
            shard
                .candidates
                .iter()
                .map(|c| {
                    let sha = semantic_sha(c);
                    receipt(
                        c,
                        sha,
                        Outcome::Fail {
                            class: "fit",
                            message: message.clone(),
                        },
                        0.0,
                    )
                })
                .collect()
        } else {
            // Binary string tables are built once per shard and shared (identical
            // bytes to a per-candidate build, so bit-parity is unaffected).
            let tables = if shard.task == "binary" {
                Some((
                    Arc::new(raw_table(train.features, &train.x)?),
                    Arc::new(raw_table(test.features, &test.x)?),
                ))
            } else {
                None
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            pool.install(|| {
                use rayon::prelude::*;
                shard
                    .candidates
                    .par_iter()
                    .map(|c| {
                        let sha = semantic_sha(c);
                        let started = Instant::now();
                        let candidate = c.clone();
                        let (train, test) = (Arc::clone(&train), Arc::clone(&test));
                        let outcome = match &tables {
                            None => run_with_timeout(
                                move || evaluate_regression(&candidate, &train, &test),
                                timeout,
                            ),
                            Some((train_table, test_table)) => {
                                let (train_table, test_table) =
                                    (Arc::clone(train_table), Arc::clone(test_table));
                                run_with_timeout(
                                    move || {
                                        evaluate_binary(
                                            &candidate,
                                            &train_table,
                                            &train.y,
                                            &test_table,
                                            &test.y,
                                            train.features,
                                        )
                                    },
                                    timeout,
                                )
                            }
                        };
                        receipt(c, sha, outcome, started.elapsed().as_secs_f64())
                    })
                    .collect()
            })
        }
    };

    fs::create_dir_all(output_dir)?;
    let prediction_dir = output_dir.join("predictions");
    fs::create_dir_all(&prediction_dir)?;
    for line in &receipts {
        if let (Some(relative), Some(expected), Some(bytes)) = (
            line.predictions_path.as_ref(),
            line.predictions_sha256.as_ref(),
            line.prediction_bytes.as_ref(),
        ) {
            if sha256_hex(bytes) != *expected {
                bail!("prediction evidence hash changed before persistence");
            }
            let path = output_dir.join(relative);
            let temporary = prediction_dir.join(format!(".{}.tmp", line.semantic_sha256));
            fs::write(&temporary, bytes)?;
            fs::rename(&temporary, path)?;
        }
    }
    let final_path = output_dir.join(format!("{}.receipts.jsonl", shard.dataset_hash));
    let tmp_path = output_dir.join(format!(".{}.receipts.jsonl.tmp", shard.dataset_hash));
    let mut buffer = Vec::new();
    for line in &receipts {
        serde_json::to_writer(&mut buffer, line)?;
        buffer.push(b'\n');
    }
    fs::write(&tmp_path, &buffer)?;
    fs::rename(&tmp_path, &final_path)?;
    let done = serde_json::json!({
        "dataset_hash": shard.dataset_hash,
        "manifest_sha256": manifest_sha256,
        "receipts": receipts.len(),
        "valid": receipts.iter().filter(|r| r.valid).count(),
        "node": node,
        "finished_at_ms": unix_millis(),
    });
    fs::write(
        output_dir.join(format!("{}.done", shard.dataset_hash)),
        done.to_string(),
    )?;
    eprintln!(
        "wrote {} receipts ({} valid) to {}",
        receipts.len(),
        receipts.iter().filter(|r| r.valid).count(),
        final_path.display()
    );
    Ok(())
}

fn validate_manifest(path: &Path) -> Result<()> {
    let shard: Shard = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .context("parse shard manifest")?;
    let reduction_version = resolve_reduction(&shard)?;
    if task_metric(&shard.task).is_none() {
        bail!("unsupported task '{}'", shard.task);
    }
    if shard.candidates.is_empty() || shard.candidates.len() > 128 {
        bail!("manifest must contain one through 128 candidates");
    }
    resolve_authority(&shard)?;
    let timeseries_schema = if shard.task == "timeseries" {
        let data = load_timeseries_shard_data(&shard)?;
        let feature_names = match reduction_version {
            ReductionVersion::V1 => make_supervised_reduction(&data)?.feature_names,
            ReductionVersion::V2 => make_direct_reduction(&data)?.feature_names,
        };
        Some(DataSchema {
            columns: feature_names
                .iter()
                .map(|name| ColumnSpec::numeric(name.clone()))
                .collect(),
            row_semantics: RowSemantics::Iid,
        })
    } else {
        None
    };
    let mut seen = std::collections::BTreeSet::new();
    for (index, candidate) in shard.candidates.iter().enumerate() {
        let computed = sha256_hex(candidate.canonical_genome.as_bytes());
        if candidate.semantic_sha256 != computed {
            bail!("candidate {index} semantic_sha256 does not match canonical genome");
        }
        if !seen.insert(computed) {
            bail!("candidate {index} duplicates an earlier canonical genome");
        }
        let genome = match parse_canonical(candidate) {
            Ok(genome) => genome,
            Err(Outcome::Fail { class, message }) => {
                bail!("candidate {index} {class}: {message}");
            }
            Err(Outcome::Score { .. }) => unreachable!("parser never returns a score"),
        };
        if shard.task == "binary" {
            let task =
                TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
            validate_v3_genome(&genome, &task, &binary_schema(8))
                .with_context(|| format!("candidate {index} is not binary-task-valid"))?;
        } else if shard.task == "timeseries" {
            if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
                bail!("candidate {index} uses a binary classification head");
            }
            let task = TaskSpec::regression(Objective::Rmse)
                .expect("rmse is a valid regression objective");
            let schema = timeseries_schema.as_ref().expect("timeseries schema built");
            validate_v3_genome(&genome, &task, schema)
                .with_context(|| format!("candidate {index} is not timeseries-task-valid"))?;
        }
    }
    Ok(())
}

/// Bitwise 0.0/1.0 targets with both classes present; any violation is a
/// dataset-level failure shared by every candidate in the shard.
fn validate_binary_targets(y: &[f64], split: &str) -> Option<String> {
    if let Some(value) = y.iter().find(|v| **v != 0.0 && **v != 1.0) {
        return Some(format!(
            "{split} target {value} is not a bitwise 0/1 class label"
        ));
    }
    let positives = y.iter().filter(|v| **v == 1.0).count();
    if positives == 0 || positives == y.len() {
        return Some(format!("{split} split does not contain both classes"));
    }
    None
}

// ---------------------------------------------------------------------------
// rewrite-binary-heads mode
// ---------------------------------------------------------------------------

fn rewrite_head(canonical: &str, head: &str) -> std::result::Result<(String, String), String> {
    let genome = GenomeIr::parse(canonical).map_err(|e| format!("parse: {e:#}"))?;
    let GenomeIr::Pipeline(mut dsl) = genome else {
        return Err("not a pipeline genome".to_string());
    };
    dsl.stages.retain(|stage| !is_target_op(&stage.op));
    let source_head = match dsl.stages.last() {
        Some(stage) if is_head_like(&stage.op) => dsl.stages.pop().expect("last stage exists").op,
        _ => "ridge-implicit".to_string(),
    };
    dsl.stages.push(Stage::new(head));
    let rewritten = GenomeIr::Pipeline(dsl).canonical();
    let reparsed = GenomeIr::parse(&rewritten).map_err(|e| format!("reparse: {e:#}"))?;
    if reparsed.canonical() != rewritten {
        return Err("rewritten genome is not canonical-stable".to_string());
    }
    // Width-independent admission check (validate_v3_genome inspects no
    // column counts, so a nominal 8-column all-numeric schema is exact).
    let task = TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
    validate_v3_genome(&reparsed, &task, &binary_schema(8))
        .map_err(|e| format!("admission: {e:#}"))?;
    Ok((rewritten, source_head))
}

fn run_rewrite(winners_dir: &Path, heads: &[String], output: &Path) -> Result<()> {
    let mut dataset_dirs = fs::read_dir(winners_dir)
        .with_context(|| format!("read winners dir {}", winners_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join("final-result.v1.json").is_file())
        .collect::<Vec<_>>();
    dataset_dirs.sort();
    if dataset_dirs.is_empty() {
        bail!(
            "no <dataset>/final-result.v1.json winners under {}",
            winners_dir.display()
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut lines = Vec::new();
    let (mut ok, mut rejected) = (0usize, 0usize);
    for dir in &dataset_dirs {
        let dataset = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final-result.v1.json"))?)?;
        let Some(winner) = value.get("canonical_genome").and_then(|v| v.as_str()) else {
            eprintln!("warning: {dataset} final-result has no canonical_genome; skipped");
            continue;
        };
        for head in heads {
            let (validation, genome, source_head) = match rewrite_head(winner, head) {
                Ok((genome, source_head)) => ("ok".to_string(), genome, source_head),
                Err(why) => {
                    rejected += 1;
                    lines.push(serde_json::json!({
                        "canonical_genome": serde_json::Value::Null,
                        "semantic_sha256": serde_json::Value::Null,
                        "source_winner_dataset": dataset,
                        "source_head": serde_json::Value::Null,
                        "binary_head": head,
                        "head_rewritten": true,
                        "validation": format!("rejected:{why}"),
                    }));
                    continue;
                }
            };
            if !seen.insert(genome.clone()) {
                continue; // deduplicate identical rewrites across winners
            }
            ok += 1;
            lines.push(serde_json::json!({
                "canonical_genome": genome,
                "semantic_sha256": sha256_hex(genome.as_bytes()),
                "source_winner_dataset": dataset,
                "source_head": source_head,
                "binary_head": head,
                "head_rewritten": true,
                "validation": validation,
            }));
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buffer = Vec::new();
    for line in &lines {
        serde_json::to_writer(&mut buffer, line)?;
        buffer.push(b'\n');
    }
    fs::write(output, buffer)?;
    eprintln!(
        "rewrote {} winners x {} heads -> {} unique ok, {} rejected ({})",
        dataset_dirs.len(),
        heads.len(),
        ok,
        rejected,
        output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
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
mod tests {
    use super::*;

    fn shard(task: &str, authority: Option<LabelAuthority>) -> Shard {
        Shard {
            schema: SHARD_SCHEMA.to_string(),
            task: task.to_string(),
            dataset_hash: "dataset".to_string(),
            train_csv: "train.csv".to_string(),
            test_csv: "test.csv".to_string(),
            train_sha256: "0".repeat(64),
            test_sha256: "1".repeat(64),
            checkpoint_epoch: None,
            checkpoint_sha256: None,
            catalog_sha256: None,
            protocol_sha256: None,
            generated_at_ms: None,
            label_authority: authority,
            meta_json: None,
            meta_sha256: None,
            horizon: None,
            seasonality: None,
            frequency: None,
            target_is_final_column: None,
            known_future_covariates: Vec::new(),
            past_only_covariates: Vec::new(),
            reduction: None,
            candidates: Vec::new(),
        }
    }

    fn corrected(metric: &str) -> LabelAuthority {
        LabelAuthority {
            score_tier: "test_exact".to_string(),
            metric: metric.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            selection_bias: Some("test-exact-reused".to_string()),
            generation: "strict-wave-000001".to_string(),
            frozen_seed_sha256: "2".repeat(64),
            model_checkpoint_sha256: "3".repeat(64),
            hall_of_fame_sha256: "4".repeat(64),
        }
    }

    fn legacy(metric: &str) -> LabelAuthority {
        LabelAuthority {
            score_tier: "test_exact".to_string(),
            metric: metric.to_string(),
            test_adaptive: false,
            benchmark_valid: true,
            selection_bias: None,
            generation: "zero".to_string(),
            frozen_seed_sha256: "2".repeat(64),
            model_checkpoint_sha256: "3".repeat(64),
            hall_of_fame_sha256: "4".repeat(64),
        }
    }

    #[test]
    fn legacy_shards_remain_adaptive() {
        let resolved = resolve_authority(&shard("regression", None)).unwrap();
        assert_eq!(resolved.tier, "test_exact");
        assert_eq!(resolved.metric, "r2");
        assert!(resolved.test_adaptive);
        assert!(!resolved.benchmark_valid);
        assert_eq!(resolved.generation, "adaptive");
    }

    #[test]
    fn corrected_wave_discloses_test_adaptation() {
        let resolved = resolve_authority(&shard("binary", Some(corrected("auc")))).unwrap();
        assert_eq!(resolved.metric, "auc");
        assert!(resolved.test_adaptive);
        assert!(!resolved.benchmark_valid);
        assert_eq!(
            resolved.selection_bias.as_deref(),
            Some("test-exact-reused")
        );
        assert_eq!(resolved.generation, "strict-wave-000001");
        assert!(resolved.authority_sha256.is_some());
    }

    #[test]
    fn superseded_generation_zero_authority_remains_replayable() {
        let resolved = resolve_authority(&shard("regression", Some(legacy("r2")))).unwrap();
        assert!(!resolved.test_adaptive);
        assert!(resolved.benchmark_valid);
        assert_eq!(resolved.generation, "zero");
    }

    #[test]
    fn missing_disclosure_or_wrong_metric_is_rejected() {
        let mut undisclosed = corrected("r2");
        undisclosed.selection_bias = None;
        assert!(resolve_authority(&shard("regression", Some(undisclosed))).is_err());
        assert!(resolve_authority(&shard("binary", Some(corrected("r2")))).is_err());
        assert!(resolve_authority(&shard("timeseries", Some(corrected("r2")))).is_err());
    }

    #[test]
    fn strict_authority_rejects_unpinned_provenance() {
        let mut authority = corrected("r2");
        authority.frozen_seed_sha256 = "not-a-hash".to_string();
        assert!(resolve_authority(&shard("regression", Some(authority))).is_err());
    }

    #[test]
    fn reduction_resolution_fails_closed_across_schema_versions() {
        use jope_dime_regressor::timeseries::{TS_REDUCTION_V1, TS_REDUCTION_V2};
        // v1 schema: no pin allowed, resolves to V1.
        let plain = shard("timeseries", None);
        assert_eq!(resolve_reduction(&plain).unwrap(), ReductionVersion::V1);
        let mut pinned_on_v1 = shard("timeseries", None);
        pinned_on_v1.reduction = Some(ReductionSpec {
            version: TS_REDUCTION_V2.to_string(),
        });
        assert!(resolve_reduction(&pinned_on_v1).is_err());
        // v2 schema: timeseries must pin; explicit v1 and v2 pins both resolve.
        let mut unpinned_v2 = shard("timeseries", None);
        unpinned_v2.schema = SHARD_SCHEMA_V2.to_string();
        assert!(resolve_reduction(&unpinned_v2).is_err());
        let mut pinned_v2 = unpinned_v2;
        pinned_v2.reduction = Some(ReductionSpec {
            version: TS_REDUCTION_V2.to_string(),
        });
        assert_eq!(resolve_reduction(&pinned_v2).unwrap(), ReductionVersion::V2);
        pinned_v2.reduction = Some(ReductionSpec {
            version: TS_REDUCTION_V1.to_string(),
        });
        assert_eq!(resolve_reduction(&pinned_v2).unwrap(), ReductionVersion::V1);
        pinned_v2.reduction = Some(ReductionSpec {
            version: "ts-reduction-v9".to_string(),
        });
        assert!(resolve_reduction(&pinned_v2).is_err());
        // non-timeseries tasks never carry a pin, on either schema.
        let mut regression_v2 = shard("regression", None);
        regression_v2.schema = SHARD_SCHEMA_V2.to_string();
        assert_eq!(
            resolve_reduction(&regression_v2).unwrap(),
            ReductionVersion::V1
        );
        regression_v2.reduction = Some(ReductionSpec {
            version: TS_REDUCTION_V2.to_string(),
        });
        assert!(resolve_reduction(&regression_v2).is_err());
        // unknown schema strings are rejected outright.
        let mut unknown = shard("timeseries", None);
        unknown.schema = "JopeFlywheelCorpusEvalShard.v3".to_string();
        assert!(resolve_reduction(&unknown).is_err());
    }

    #[test]
    fn prediction_encoding_and_balanced_accuracy_are_deterministic() {
        let predictions = [0.1, 0.7, 0.8, 0.2];
        let first = prediction_bytes(&predictions);
        let second = prediction_bytes(&predictions);
        assert_eq!(first, second);
        assert_eq!(first.len(), predictions.len() * 8);
        assert_eq!(sha256_hex(&first), sha256_hex(&second));
        assert_eq!(balanced_accuracy(&[0.0, 0.0, 1.0, 1.0], &predictions), 0.5);
    }

    #[test]
    fn binary_task_admission_rejects_regression_heads() {
        let task = TaskSpec::binary(Objective::RocAuc).unwrap();
        let regression = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let binary = GenomeIr::parse("(pipe X (logit_ridge l2=0.1))").unwrap();
        assert!(validate_v3_genome(&regression, &task, &binary_schema(8)).is_err());
        assert!(validate_v3_genome(&binary, &task, &binary_schema(8)).is_ok());
    }

    #[test]
    fn timeseries_authority_uses_negative_mase_and_stays_adaptive() {
        let resolved = resolve_authority(&shard("timeseries", None)).unwrap();
        assert_eq!(resolved.metric, "negative_mase");
        assert!(resolved.test_adaptive);
        assert!(!resolved.benchmark_valid);

        let corrected =
            resolve_authority(&shard("timeseries", Some(corrected("negative_mase")))).unwrap();
        assert_eq!(corrected.metric, "negative_mase");
        assert!(corrected.test_adaptive);
        assert!(!corrected.benchmark_valid);

        assert!(resolve_authority(&shard("timeseries", Some(legacy("negative_mase")))).is_err());
    }

    #[test]
    fn timeseries_rejects_binary_classification_heads() {
        let binary = GenomeIr::parse("(pipe X (logit_ridge l2=0.1))").unwrap();
        let logistic = GenomeIr::parse("(pipe X (logistic))").unwrap();
        let regression = GenomeIr::parse("(pipe X (ridge))").unwrap();
        assert!(matches!(terminal_head(&binary), Some("logit_ridge")));
        assert!(matches!(terminal_head(&logistic), Some("logistic")));
        assert!(matches!(terminal_head(&regression), Some("ridge")));
    }
}
