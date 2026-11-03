//! Binary-classification campaign lane (advanced-v3 kernel), v2-binary schemas.
//!
//! Binary campaign implementation. Design and operational scope are documented
//! in `docs/campaigns.md`.

use anyhow::{bail, Context, Result};
use feat_math::advanced_v3::{
    emit_bound, emit_fitted_cpp17, encode_fitted_cpp17, fit_table, validate_fitted_cpp_codec,
    validate_v3_genome, BoundModel, ColumnSpec, DataSchema, EncodedTable, FittedCppRuntime,
    Objective, RawTable, RowSemantics, TaskSpec,
};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline};
use feat_math::heads::is_head_op;
use feat_math::metrics::{
    accuracy_at, auc, average_precision, best_f1_threshold, brier, f1_at, lift_at, logloss,
    youden_threshold,
};
use feat_math::target_transform::is_target_op;
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use super::exact_cv::{
    canonical_json, genome_complexity, immutable_write, make_readonly, offspring_seed,
    require_fresh_artifact_root, robust_fitness_binary, sha256_bytes, sha256_file,
    stratified_assignments, DeterministicRng,
};
use crate::data::RegressionData;
use jope_dime_regressor::task::TaskKind;

const POPULATION: usize = 60;
const CV5_FOLDS: usize = 5;
const CV7_FOLDS: usize = 7;
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const QUICK_HOLDOUT_FOLD: usize = 0;
const FINAL_CV7_CANDIDATES: usize = 10;
const MAX_TRANSFORMS: usize = 6;
const CAMPAIGN_SCHEMA: &str = "JainExactCvCampaign.v2-binary";
const LEDGER_SCHEMA: &str = "JainExactCvSlot.v2-binary";
const FOLD_MANIFEST_SCHEMA: &str = "JainExactCvFoldManifest.v2-binary-stratified";
const SELECTION_SCHEMA: &str = "JainExactCvSelectionFreeze.v2-binary";
const SELECTION_COMPLETE_SCHEMA: &str = "JainExactCvSelectionComplete.v2-binary";
const EVALUATOR: &str = "prime-jain.advanced-v3-raw-table.exact-cv.v2-binary";
const FOLD_AUTHORITY: &str = "jope-dime.stratified_assignments(dataset,y,k).sha256.v1";
/// Pinned SHA-256 of seeds/generation-zero-binary.txt.
pub(crate) const SEED_BINARY_SHA256: &str =
    "3c003169a68e93e397cca168855e78d76119b54e04fdd60c2515c8b1e3a59950";

/// Weighted head vocabulary for offspring head reattachment. Every entry is
/// probability-capable under `validate_v3_genome` for binary tasks.
const HEADS_BINARY: [&str; 10] = [
    "logit_ridge",
    "logit_ridge",
    "logit_ridge",
    "tab_boost",
    "tab_boost",
    "tab_boost",
    "gam_boost",
    "gam_boost",
    "extra_trees",
    "logistic",
];

const BINARY_HEAD_NAMES: [&str; 6] = [
    "logit_ridge",
    "logistic",
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "portfolio",
];

fn is_binary_head(op: &str) -> bool {
    BINARY_HEAD_NAMES.contains(&op)
}

pub(crate) fn binary_task() -> TaskSpec {
    TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective")
}

/// Synthetic all-numeric schema for headerless numeric matrices.
pub(crate) fn binary_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// Shortest-round-trip string table: parses back to the exact f64 cells.
fn raw_table(features: usize, rows: &[Vec<f64>]) -> Result<RawTable> {
    let headers = (0..features).map(|j| format!("c{j}")).collect::<Vec<_>>();
    let cells = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, cells)
}

/// Probabilities must be finite and inside [0, 1]; anything else rejects the
/// slot (the binary twin of the regression lane's finite-prediction check).
fn validate_probabilities(predictions: &[f64], expected: usize) -> Result<()> {
    if predictions.len() != expected {
        bail!("prediction count differs from validation rows");
    }
    if predictions
        .iter()
        .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        bail!("produced out-of-range probabilities");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// receipts
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BinaryAncestry {
    pub(crate) operation: String,
    pub(crate) parents: Vec<String>,
    pub(crate) retry_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct BinaryMember {
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
    pub(crate) ancestry: BinaryAncestry,
}

/// Serializable fold evidence: undefined folds are `null` (never NaN, which
/// canonical JSON cannot round-trip).
fn optional_folds(fold_auc: &[f64]) -> Vec<Option<f64>> {
    fold_auc
        .iter()
        .map(|value| value.is_finite().then_some(*value))
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinarySelectionEvidence {
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: String,
    fold_auc: Vec<Option<f64>>,
    fold_defined: Vec<bool>,
    pooled_oof_auc: f64,
    pooled_oof_logloss: f64,
    pooled_oof_brier: f64,
    robust_fitness: f64,
}

impl BinarySelectionEvidence {
    fn from_result(
        result: &BinaryCvResult,
        fold_plan_folds: usize,
        evaluated_fold_ids: Vec<usize>,
        fold_manifest_sha256: &str,
    ) -> Self {
        Self {
            fold_plan_folds,
            evaluated_fold_ids,
            fold_manifest_sha256: fold_manifest_sha256.to_string(),
            fold_auc: optional_folds(&result.fold_auc),
            fold_defined: result.fold_defined.clone(),
            pooled_oof_auc: result.pooled_oof_auc,
            pooled_oof_logloss: result.pooled_oof_logloss,
            pooled_oof_brier: result.pooled_oof_brier,
            robust_fitness: result.robust_fitness,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinarySlotLedger {
    schema: String,
    generation: usize,
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    ancestry: BinaryAncestry,
    complexity: usize,
    valid: bool,
    rejection_reason: Option<String>,
    cache_disposition: String,
    fold_auc: Option<Vec<Option<f64>>>,
    fold_defined: Option<Vec<bool>>,
    pooled_oof_auc: Option<f64>,
    pooled_oof_logloss: Option<f64>,
    pooled_oof_brier: Option<f64>,
    robust_fitness: Option<f64>,
}

impl BinarySlotLedger {
    fn rankable(&self) -> bool {
        self.valid && self.robust_fitness.is_some() && self.pooled_oof_logloss.is_some()
    }
}

/// Ranking chain: fitness desc -> pooled OOF logloss asc -> complexity asc ->
/// semantic sha asc.
fn rank_binary_slots(left: &BinarySlotLedger, right: &BinarySlotLedger) -> std::cmp::Ordering {
    right
        .robust_fitness
        .unwrap()
        .total_cmp(&left.robust_fitness.unwrap())
        .then_with(|| {
            left.pooled_oof_logloss
                .unwrap()
                .total_cmp(&right.pooled_oof_logloss.unwrap())
        })
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryFoldAssignment {
    fold: usize,
    train_rows: Vec<usize>,
    validation_rows: Vec<usize>,
    train_positives: usize,
    train_negatives: usize,
    validation_positives: usize,
    validation_negatives: usize,
    train_rows_sha256: String,
    validation_rows_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryFoldManifest {
    schema: String,
    dataset_hash: String,
    dataset_train_sha256: String,
    fold_authority: String,
    folds: usize,
    rows: usize,
    positives: usize,
    negatives: usize,
    assignments: Vec<BinaryFoldAssignment>,
}

fn build_binary_fold_manifest(
    dataset: &str,
    train_sha256: &str,
    y: &[f64],
    folds: usize,
) -> Result<BinaryFoldManifest> {
    let assignments = stratified_assignments(dataset, y, folds)?;
    let mut folds_out = Vec::with_capacity(folds);
    for fold in 0..folds {
        let mut train_rows = Vec::new();
        let mut validation_rows = Vec::new();
        for (row, assigned) in assignments.iter().enumerate() {
            if *assigned == fold {
                validation_rows.push(row);
            } else {
                train_rows.push(row);
            }
        }
        let count = |rows: &[usize], class: f64| rows.iter().filter(|r| y[**r] == class).count();
        folds_out.push(BinaryFoldAssignment {
            fold,
            train_positives: count(&train_rows, 1.0),
            train_negatives: count(&train_rows, 0.0),
            validation_positives: count(&validation_rows, 1.0),
            validation_negatives: count(&validation_rows, 0.0),
            train_rows_sha256: sha256_bytes(&canonical_json(&train_rows)?),
            validation_rows_sha256: sha256_bytes(&canonical_json(&validation_rows)?),
            train_rows,
            validation_rows,
        });
    }
    let positives = y.iter().filter(|v| **v == 1.0).count();
    Ok(BinaryFoldManifest {
        schema: FOLD_MANIFEST_SCHEMA.to_string(),
        dataset_hash: dataset.to_string(),
        dataset_train_sha256: train_sha256.to_string(),
        fold_authority: FOLD_AUTHORITY.to_string(),
        folds,
        rows: y.len(),
        positives,
        negatives: y.len() - positives,
        assignments: folds_out,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BinaryThresholds {
    /// From the winner's pooled CV7 train-OOF probabilities; report-only.
    pub(crate) f1_threshold: f64,
    f1_value: f64,
    youden_threshold: f64,
    youden_value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryFinalist {
    quick_rank: usize,
    generation: usize,
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    quick_cv5_fold0: BinarySelectionEvidence,
    cv7: Option<BinarySelectionEvidence>,
    rejection_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryArtifactHashes {
    python_sha256: String,
    rust_sha256: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BinarySelectionFreeze {
    schema: String,
    evaluator: String,
    task: String,
    pub(crate) dataset_hash: String,
    pub(crate) train_sha256: String,
    initial_population_sha256: String,
    generation: usize,
    slot: usize,
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
    pub(crate) complexity: usize,
    ranking_authority: String,
    ledger_sha256: String,
    quick_cv5_fold0: BinarySelectionEvidence,
    pub(crate) cv7_pooled_oof_auc: f64,
    cv7: BinarySelectionEvidence,
    pub(crate) thresholds: BinaryThresholds,
    winner_artifacts: BinaryArtifactHashes,
    cv7_finalists: Vec<BinaryFinalist>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BinarySelectionComplete {
    schema: String,
    status: String,
    selection_file: String,
    selection_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct BinaryCampaignIdentity<'a> {
    schema: &'static str,
    evaluator: &'static str,
    task: &'static str,
    dataset_hash: &'a str,
    train_sha256: &'a str,
    train_rows: usize,
    positives: usize,
    negatives: usize,
    generations: usize,
    population: usize,
    cv_folds: usize,
    confirm_folds: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    seed: u64,
    member_execution: &'static str,
    initial_population_sha256: &'a str,
    quick_holdout_fold: usize,
    final_cv7_candidates: usize,
    fold_authority: &'static str,
}

mod fold_cache;
use fold_cache::BinaryFoldCache;
pub(crate) use fold_cache::BinaryCvResult;
#[cfg(test)]
pub(crate) use fold_cache::exact_cv_on_folds_binary;

// ---------------------------------------------------------------------------
// population
// ---------------------------------------------------------------------------

mod population;
mod selection;
pub(crate) use population::{
    binary_genome_complexity, evaluate_test_adaptive_binary, evaluate_test_adaptive_binary_cpp17,
    test_adaptive_next_population, test_adaptive_seed_population, BinaryAdaptiveRanked,
};
pub(super) use population::{
    evaluate_binary_unique, parse_binary_population, refill_binary_population, BinaryEvaluation,
};
pub(crate) use selection::{
    score_dataset_binary, select_dataset_binary, validate_freeze_binary,
};
#[cfg(test)]
#[path = "binary_cv/tests.rs"]
pub(crate) mod tests;

#[cfg(test)]
mod cli_smoke_support {
    //! Manual CLI smoke harness:
    //! `cargo test -p jope-dime-regressor -- --ignored export_binary_fixture_registry`
    //! writes the two fixture datasets as a registry for the real binary.
    use super::tests::tests::{export_registry_to, registry_root};

    #[test]
    #[ignore]
    fn export_binary_fixture_registry() {
        let root = registry_root();
        export_registry_to(&root);
        println!("binary fixture registry at {}", root.display());
    }
}
