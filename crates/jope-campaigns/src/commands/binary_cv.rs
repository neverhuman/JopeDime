//! Binary-classification campaign lane (advanced-v3 kernel), v2-binary schemas.
//!
//! Design notes:
//! - Genomes are evaluated through the public schema-bound v3 path
//!   (`feat_math::advanced_v3::fit_table` / `BoundModel::predict_table`) over a
//!   synthetic all-numeric schema (`c0..c{n-1}`, IID rows). Numeric cells are
//!   rendered with `{v:?}` (shortest round-trip), so the encoded feature block
//!   equals the numeric matrix bit-for-bit under the compatibility input
//!   config; this was chosen over a private matrix adapter because it is the
//!   only public v3 fit surface and exercises exactly the code path the
//!   emitted bound artifacts mirror.
//! - Folds are class-stratified (`stratified_assignments`: SHA-keyed, per-class
//!   ±1, fold 0 always holds a minority row) and materialized once per
//!   (fold-plan, fold) as shared string tables, mirroring the regression
//!   lane's `FoldPlanCache`.
//! - Fitness is `robust_fitness_binary` (Gini − 0.25·SD over DEFINED fold
//!   Ginis); folds whose validation split holds a single class have undefined
//!   AUC (`fold_defined = false`) and are excluded from the penalty while
//!   their rows still enter the pooled OOF.
//! - Offspring are a pure function of (parent canonicals, offspring_seed):
//!   the v3 head is detached, the legacy body is mutated/crossed with the
//!   frozen canonical operators, and a deterministically drawn
//!   probability-capable head from `HEADS_BINARY` is reattached before
//!   re-canonicalization; admission requires `validate_v3_genome`.
//! - The stage-prefix cache is regression-lane only for now (v3 fit paths
//!   differ); wiring it here is future work.
//! - Deviations from the regression lane, recorded: no core pinning, no live
//!   logger/checkpoints/generation summaries/phase2 receipts; slot-level
//!   artifacts are not published (bound emission requires a fitted model —
//!   the WINNER's bound artifacts are published instead).

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
use crate::task::TaskKind;

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
// fold slices
// ---------------------------------------------------------------------------

struct BinaryFoldSlices {
    train_table: RawTable,
    train_y: Vec<f64>,
    valid_table: RawTable,
    valid_y: Vec<f64>,
    validation_rows: Vec<usize>,
    valid_positives: usize,
    valid_negatives: usize,
}

/// Campaign-lifetime cache of stratified fold slices keyed by
/// (fold-plan size, fold); string tables are built once per fold instead of
/// once per genome evaluation.
struct BinaryFoldCache<'a> {
    train: &'a RegressionData,
    dataset: &'a str,
    slices: Mutex<HashMap<(usize, usize), Arc<BinaryFoldSlices>>>,
}

impl<'a> BinaryFoldCache<'a> {
    fn new(train: &'a RegressionData, dataset: &'a str) -> Self {
        Self {
            train,
            dataset,
            slices: Mutex::new(HashMap::new()),
        }
    }

    fn fold(&self, fold_plan_folds: usize, fold: usize) -> Result<Arc<BinaryFoldSlices>> {
        if let Some(hit) = self
            .slices
            .lock()
            .expect("binary fold-slice cache lock")
            .get(&(fold_plan_folds, fold))
        {
            return Ok(Arc::clone(hit));
        }
        let n = self.train.x.len().min(self.train.y.len());
        let assignments =
            stratified_assignments(self.dataset, &self.train.y[..n], fold_plan_folds)?;
        let mut train_x: Vec<Vec<f64>> = Vec::new();
        let mut train_y = Vec::new();
        let mut valid_x: Vec<Vec<f64>> = Vec::new();
        let mut valid_y = Vec::new();
        let mut validation_rows = Vec::new();
        for row in 0..n {
            if assignments[row] == fold {
                valid_x.push(self.train.x[row].clone());
                valid_y.push(self.train.y[row]);
                validation_rows.push(row);
            } else {
                train_x.push(self.train.x[row].clone());
                train_y.push(self.train.y[row]);
            }
        }
        let valid_positives = valid_y.iter().filter(|v| **v == 1.0).count();
        let valid_negatives = valid_y.len() - valid_positives;
        let built = Arc::new(BinaryFoldSlices {
            train_table: raw_table(self.train.features, &train_x)?,
            train_y,
            valid_table: raw_table(self.train.features, &valid_x)?,
            valid_y,
            validation_rows,
            valid_positives,
            valid_negatives,
        });
        let mut slices = self.slices.lock().expect("binary fold-slice cache lock");
        Ok(Arc::clone(
            slices.entry((fold_plan_folds, fold)).or_insert(built),
        ))
    }
}

// ---------------------------------------------------------------------------
// CV evaluation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct BinaryCvResult {
    /// Per evaluated fold; NaN when the fold's validation split is single-class.
    pub(crate) fold_auc: Vec<f64>,
    pub(crate) fold_defined: Vec<bool>,
    pub(crate) pooled_oof_auc: f64,
    pub(crate) pooled_oof_logloss: f64,
    pub(crate) pooled_oof_brier: f64,
    pub(crate) robust_fitness: f64,
    /// Wall-clock diagnostics; intentionally absent from every receipt so
    /// binary receipts stay byte-deterministic.
    #[allow(dead_code)]
    pub(crate) fit_seconds: Vec<f64>,
    #[allow(dead_code)]
    pub(crate) predict_seconds: Vec<f64>,
    /// Pooled OOF evidence for threshold freezing.
    pub(crate) oof_truth: Vec<f64>,
    pub(crate) oof_probabilities: Vec<f64>,
}

/// Public-shape binary CV entry (tests + future audit lanes); campaigns use
/// the shared-slice body directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn exact_cv_on_folds_binary(
    genome: &GenomeIr,
    train: &RegressionData,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    dataset: &str,
) -> Result<BinaryCvResult> {
    binary_cv_on_slices(
        genome,
        &BinaryFoldCache::new(train, dataset),
        fold_plan_folds,
        evaluated_folds,
        &binary_task(),
        &binary_schema(train.features),
    )
}

fn binary_cv_on_slices(
    genome: &GenomeIr,
    fold_cache: &BinaryFoldCache<'_>,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<BinaryCvResult> {
    let train = fold_cache.train;
    let n = train.x.len().min(train.y.len());
    let mut oof = vec![f64::NAN; n];
    let mut fold_auc = Vec::with_capacity(evaluated_folds.len());
    let mut fold_defined = Vec::with_capacity(evaluated_folds.len());
    let mut fit_seconds = Vec::with_capacity(evaluated_folds.len());
    let mut predict_seconds = Vec::with_capacity(evaluated_folds.len());
    if evaluated_folds.is_empty() || evaluated_folds.iter().any(|fold| *fold >= fold_plan_folds) {
        bail!("evaluated folds are empty or outside the fold plan");
    }
    for &fold in evaluated_folds {
        let slices = fold_cache.fold(fold_plan_folds, fold)?;
        if slices.valid_table.rows.is_empty() || slices.train_table.rows.len() < 8 {
            bail!("fold {fold} is empty or has fewer than eight training rows");
        }
        let fit_started = Instant::now();
        let model = fit_table(genome, task, schema, &slices.train_table, &slices.train_y)
            .with_context(|| format!("fit fold {fold}"))?;
        fit_seconds.push(fit_started.elapsed().as_secs_f64());
        let predict_started = Instant::now();
        let predictions = model
            .predict_table(&slices.valid_table)
            .with_context(|| format!("predict fold {fold}"))?;
        predict_seconds.push(predict_started.elapsed().as_secs_f64());
        validate_probabilities(&predictions, slices.valid_y.len())
            .with_context(|| format!("fold {fold}"))?;
        // Single-class validation split: AUC is undefined for the fold, its
        // rows still pool into the OOF evidence.
        if slices.valid_positives == 0 || slices.valid_negatives == 0 {
            fold_auc.push(f64::NAN);
            fold_defined.push(false);
        } else {
            let score = auc(&slices.valid_y, &predictions);
            if !score.is_finite() {
                bail!("fold {fold} AUC is non-finite despite both classes present");
            }
            fold_auc.push(score);
            fold_defined.push(true);
        }
        for (index, prediction) in slices.validation_rows.iter().copied().zip(predictions) {
            oof[index] = prediction;
        }
    }
    let mut oof_truth = Vec::new();
    let mut oof_probabilities = Vec::new();
    for (row, prediction) in oof.into_iter().enumerate() {
        if prediction.is_finite() {
            oof_truth.push(train.y[row]);
            oof_probabilities.push(prediction);
        }
    }
    if oof_truth.is_empty() {
        bail!("CV produced no held-out probabilities");
    }
    let pooled_positives = oof_truth.iter().filter(|v| **v == 1.0).count();
    if pooled_positives == 0 || pooled_positives == oof_truth.len() {
        bail!("pooled OOF evidence is single-class; AUC authority is undefined");
    }
    let pooled_oof_auc = auc(&oof_truth, &oof_probabilities);
    let pooled_oof_logloss = logloss(&oof_truth, &oof_probabilities);
    let pooled_oof_brier = brier(&oof_truth, &oof_probabilities);
    let robust_fitness = robust_fitness_binary(pooled_oof_auc, &fold_auc);
    if !pooled_oof_auc.is_finite() || !robust_fitness.is_finite() || !pooled_oof_logloss.is_finite()
    {
        bail!("pooled OOF metrics or robust fitness are non-finite");
    }
    Ok(BinaryCvResult {
        fold_auc,
        fold_defined,
        pooled_oof_auc,
        pooled_oof_logloss,
        pooled_oof_brier,
        robust_fitness,
        fit_seconds,
        predict_seconds,
        oof_truth,
        oof_probabilities,
    })
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

// ---------------------------------------------------------------------------
// population
// ---------------------------------------------------------------------------

fn parse_binary_population(
    bytes: &[u8],
    expected: usize,
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    let text = std::str::from_utf8(bytes).context("binary generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    let mut seen = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body).with_context(|| {
            format!(
                "binary generation-zero line {} does not parse",
                line_number + 1
            )
        })?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!(
                "binary generation-zero line {} is not a pipeline",
                line_number + 1
            );
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!(
                "binary generation-zero line {} is not canonical",
                line_number + 1
            );
        }
        validate_v3_genome(&genome, task, schema).with_context(|| {
            format!(
                "binary generation-zero line {} is not a valid binary genome",
                line_number + 1
            )
        })?;
        let semantic_sha256 = sha256_bytes(canonical.as_bytes());
        if !seen.insert(semantic_sha256.clone()) {
            bail!(
                "binary generation-zero line {} duplicates an earlier genome",
                line_number + 1
            );
        }
        population.push(BinaryMember {
            canonical_genome: canonical,
            semantic_sha256,
            ancestry: BinaryAncestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    if population.len() != expected {
        bail!(
            "binary population requires {expected} genomes, found {}",
            population.len()
        );
    }
    Ok(population)
}

/// Deterministic preset head stage for offspring reattachment. Consumes a
/// deterministic number of draws per head family, so the child is a pure
/// function of (parents, offspring_seed).
fn binary_head_stage(rng: &mut DeterministicRng) -> String {
    let head = HEADS_BINARY[(rng.next_u64() as usize) % HEADS_BINARY.len()];
    match head {
        "logit_ridge" => {
            let l2 = ["0.005", "0.02", "0.1", "0.5"][(rng.next_u64() as usize) % 4];
            if rng.next_u64() % 2 == 0 {
                format!("(logit_ridge class_weight=balanced l2={l2})")
            } else {
                format!("(logit_ridge l2={l2})")
            }
        }
        "tab_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "80", "120"][(rng.next_u64() as usize) % 4];
            format!("(tab_boost lr={lr} rounds={rounds})")
        }
        "gam_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "100"][(rng.next_u64() as usize) % 3];
            format!("(gam_boost lr={lr} rounds={rounds})")
        }
        "extra_trees" => {
            let trees = ["32", "64", "96"][(rng.next_u64() as usize) % 3];
            let depth = ["6", "8", "10"][(rng.next_u64() as usize) % 3];
            format!("(extra_trees depth={depth} trees={trees})")
        }
        _ => "(logistic)".to_string(),
    }
}

/// Detach the binary head so the frozen legacy operators can work on the
/// transform body; a placeholder ridge head keeps the body a valid legacy
/// pipeline for mutation/crossover.
fn legacy_body(canonical: &str) -> Result<String> {
    let genome = GenomeIr::parse(canonical).context("parse binary parent")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("binary parent is not a pipeline");
    };
    let mut stages = dsl.stages;
    if stages
        .last()
        .is_some_and(|stage| is_binary_head(&stage.op) || is_head_op(&stage.op))
    {
        stages.pop();
    }
    let mut body = String::from("(pipe X");
    for stage in &stages {
        body.push(' ');
        body.push_str(&stage.to_string());
    }
    body.push_str(" (ridge))");
    Ok(body)
}

/// Reattach a deterministically drawn binary head to a mutated legacy body.
fn with_binary_head(child_body: &str, head_stage: &str) -> Result<String> {
    let genome = GenomeIr::parse(child_body).context("parse mutated binary body")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("mutated binary body is not a pipeline");
    };
    let mut stages = dsl.stages;
    while stages
        .last()
        .is_some_and(|stage| is_head_op(&stage.op) || is_binary_head(&stage.op))
    {
        stages.pop();
    }
    // Binary genomes may not carry target stages; drop any defensively.
    stages.retain(|stage| !is_target_op(&stage.op));
    let mut text = String::from("(pipe X");
    for stage in &stages {
        text.push(' ');
        text.push_str(&stage.to_string());
    }
    text.push(' ');
    text.push_str(head_stage);
    text.push(')');
    let rebuilt = GenomeIr::parse(&text).context("parse reheaded binary child")?;
    Ok(rebuilt.canonical())
}

fn refill_binary_population(
    seed: u64,
    generation: usize,
    ranked: &[BinarySlotLedger],
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    if ranked.len() < PARENT_POOL || ELITES >= POPULATION {
        bail!("cannot refill binary population from configured parent pool/elites");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = HashSet::with_capacity(POPULATION);
    for elite in ranked.iter().take(ELITES) {
        seen.insert(elite.semantic_sha256.clone());
        next.push(BinaryMember {
            canonical_genome: elite.canonical_genome.clone(),
            semantic_sha256: elite.semantic_sha256.clone(),
            ancestry: BinaryAncestry {
                operation: "elite".to_string(),
                parents: vec![elite.semantic_sha256.clone()],
                retry_count: 0,
            },
        });
    }
    for slot in ELITES..POPULATION {
        let mut retry = 0usize;
        loop {
            if retry >= 1_000_000 {
                bail!(
                    "could not generate an admissible binary child for generation {generation} slot {slot}"
                );
            }
            let mut rng = DeterministicRng::new(offspring_seed(seed, generation, slot, retry));
            let crossover = rng.next_u64() % 4 == 0;
            let winner = |rng: &mut DeterministicRng| {
                let mut best = (rng.next_u64() as usize) % PARENT_POOL;
                for _ in 1..TOURNAMENT {
                    best = best.min((rng.next_u64() as usize) % PARENT_POOL);
                }
                best
            };
            let parent_a = &ranked[winner(&mut rng)];
            let produced = if crossover {
                let parent_b = &ranked[winner(&mut rng)];
                let body = crossover_canonical_pipelines(
                    &legacy_body(&parent_a.canonical_genome)?,
                    &legacy_body(&parent_b.canonical_genome)?,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                );
                body.map(|child| {
                    (
                        "crossover".to_string(),
                        vec![
                            parent_a.semantic_sha256.clone(),
                            parent_b.semantic_sha256.clone(),
                        ],
                        child,
                    )
                })
            } else {
                mutate_canonical_pipeline(
                    &legacy_body(&parent_a.canonical_genome)?,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                )
                .map(|child| {
                    (
                        "mutation".to_string(),
                        vec![parent_a.semantic_sha256.clone()],
                        child,
                    )
                })
            };
            let (operation, parents, child_body) = match produced {
                Ok(product) => product,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            let head_stage = binary_head_stage(&mut rng);
            let canonical = match with_binary_head(&child_body, &head_stage) {
                Ok(canonical) => canonical,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            let parsed = match GenomeIr::parse(&canonical) {
                Ok(parsed) => parsed,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            if validate_v3_genome(&parsed, task, schema).is_err() {
                retry += 1;
                continue;
            }
            let semantic_sha256 = sha256_bytes(canonical.as_bytes());
            if !seen.insert(semantic_sha256.clone()) {
                retry += 1;
                continue;
            }
            next.push(BinaryMember {
                canonical_genome: canonical,
                semantic_sha256,
                ancestry: BinaryAncestry {
                    operation,
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
    }
    if next.len() != POPULATION {
        bail!("binary refill produced a short population");
    }
    Ok(next)
}

/// Test-adaptive ranking input for the binary lane. This deliberately exposes
/// only the fields needed by the frozen offspring operator; held-out AUC is
/// supplied as the fitness value by the caller.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveRanked {
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
    pub(crate) complexity: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
}

/// Result of one full-train, held-out-test advanced-v3 binary evaluation.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveEvaluation {
    pub(crate) complexity: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
    pub(crate) predictions: Vec<f64>,
    pub(crate) fit_seconds: f64,
    pub(crate) predict_seconds: f64,
}

/// C++17 build identities accompanying one fitted binary evaluation.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveCpp17Evaluation {
    pub(crate) evaluation: BinaryAdaptiveEvaluation,
    pub(crate) source_sha256: String,
    pub(crate) codec_sha256: String,
    pub(crate) compile_seconds: f64,
}

/// Parse the exact canonical 60-member binary seed file through the same
/// admission checks used by the historical train-only binary campaign.
pub(crate) fn test_adaptive_seed_population(
    bytes: &[u8],
    features: usize,
) -> Result<Vec<BinaryMember>> {
    parse_binary_population(bytes, POPULATION, &binary_task(), &binary_schema(features))
}

pub(crate) fn binary_genome_complexity(canonical_genome: &str) -> Result<usize> {
    Ok(genome_complexity(&GenomeIr::parse(canonical_genome)?))
}

/// Refill a deterministic 60-member generation using the historical binary
/// mutation/crossover/head operators and the caller's held-out-test ranking.
pub(crate) fn test_adaptive_next_population(
    seed: u64,
    generation: usize,
    ranked: &[BinaryAdaptiveRanked],
    features: usize,
) -> Result<Vec<BinaryMember>> {
    let ranked = ranked
        .iter()
        .enumerate()
        .map(|(slot, candidate)| BinarySlotLedger {
            schema: LEDGER_SCHEMA.to_string(),
            generation: generation.saturating_sub(1),
            slot,
            canonical_genome: candidate.canonical_genome.clone(),
            semantic_sha256: candidate.semantic_sha256.clone(),
            ancestry: BinaryAncestry {
                operation: "ranked-parent".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
            complexity: candidate.complexity,
            valid: true,
            rejection_reason: None,
            cache_disposition: "test-adaptive-parent".to_string(),
            fold_auc: None,
            fold_defined: None,
            pooled_oof_auc: Some(candidate.auc),
            pooled_oof_logloss: Some(candidate.logloss),
            pooled_oof_brier: None,
            robust_fitness: Some(candidate.auc),
        })
        .collect::<Vec<_>>();
    refill_binary_population(
        seed,
        generation,
        &ranked,
        &binary_task(),
        &binary_schema(features),
    )
}

/// Fit one canonical advanced-v3 binary genome on all train rows and evaluate
/// probabilities directly on the held-out test target.
pub(crate) fn evaluate_test_adaptive_binary(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<BinaryAdaptiveEvaluation> {
    let (genome, model, fit_seconds) = fit_test_adaptive_binary(canonical_genome, train, test)?;
    let test_table = raw_table(test.features, &test.x)?;
    let predict_started = Instant::now();
    let predictions = model.predict_table(&test_table)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    finish_test_adaptive_binary(
        genome_complexity(&genome),
        predictions,
        test,
        fit_seconds,
        predict_seconds,
    )
}

/// Fit on the train split, lower the fitted model to a compiled C++17 image,
/// and use only that image to produce held-out probabilities.
pub(crate) fn evaluate_test_adaptive_binary_cpp17(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
    runtime_root: &Path,
) -> Result<BinaryAdaptiveCpp17Evaluation> {
    let (genome, model, fit_seconds) = fit_test_adaptive_binary(canonical_genome, train, test)?;
    let emitted = emit_fitted_cpp17(&model).context("emit fitted binary C++17 model")?;
    let compile_started = Instant::now();
    let codec = encode_fitted_cpp17(&model).context("compile fitted binary C++17 model")?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    validate_fitted_cpp_codec(&codec.bytes)?;
    let runtime = FittedCppRuntime::new(runtime_root)?;
    let compiled = runtime.load(&codec.bytes)?;
    let encoded_test = EncodedTable::from_numeric_rows(&test.x)?;
    let predict_started = Instant::now();
    let predictions = compiled.predict(&encoded_test)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    let evaluation = finish_test_adaptive_binary(
        genome_complexity(&genome),
        predictions,
        test,
        fit_seconds,
        predict_seconds,
    )?;
    Ok(BinaryAdaptiveCpp17Evaluation {
        evaluation,
        source_sha256: emitted.source_sha256,
        codec_sha256: codec.sha256,
        compile_seconds,
    })
}

fn fit_test_adaptive_binary(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<(GenomeIr, BoundModel, f64)> {
    if train.features != test.features {
        bail!("binary test-adaptive train/test feature widths differ");
    }
    let genome = GenomeIr::parse(canonical_genome).context("parse binary adaptive genome")?;
    if genome.canonical() != canonical_genome {
        bail!("binary test-adaptive genome is not canonical");
    }
    let task = binary_task();
    let schema = binary_schema(train.features);
    validate_v3_genome(&genome, &task, &schema)?;
    let train_rows = train.x.len().min(train.y.len());
    if train_rows != train.x.len() || train_rows == 0 || test.y.len() != test.x.len() {
        bail!("binary test-adaptive data has inconsistent row counts");
    }
    let train_table = raw_table(train.features, &train.x)?;
    let fit_started = Instant::now();
    let model = fit_table(
        &genome,
        &task,
        &schema,
        &train_table,
        &train.y[..train_rows],
    )?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    Ok((genome, model, fit_seconds))
}

fn finish_test_adaptive_binary(
    complexity: usize,
    predictions: Vec<f64>,
    test: &RegressionData,
    fit_seconds: f64,
    predict_seconds: f64,
) -> Result<BinaryAdaptiveEvaluation> {
    validate_probabilities(&predictions, test.y.len())?;
    let auc = auc(&test.y, &predictions);
    let logloss = logloss(&test.y, &predictions);
    if !auc.is_finite() || !logloss.is_finite() {
        bail!("binary test-adaptive metrics are non-finite");
    }
    Ok(BinaryAdaptiveEvaluation {
        complexity,
        auc,
        logloss,
        predictions,
        fit_seconds,
        predict_seconds,
    })
}

// ---------------------------------------------------------------------------
// campaign
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BinaryEvaluation {
    complexity: usize,
    result: Option<BinaryCvResult>,
    rejection_reason: Option<String>,
}

fn evaluate_binary_unique(
    member: &BinaryMember,
    fold_cache: &BinaryFoldCache<'_>,
    task: &TaskSpec,
    schema: &DataSchema,
) -> BinaryEvaluation {
    let rejected = |complexity, reason: String| BinaryEvaluation {
        complexity,
        result: None,
        rejection_reason: Some(reason),
    };
    let genome = match GenomeIr::parse(&member.canonical_genome) {
        Ok(genome) if matches!(genome, GenomeIr::Pipeline(_)) => genome,
        Ok(_) => return rejected(0, "not a pipeline genome".to_string()),
        Err(error) => return rejected(0, format!("parse: {error:#}")),
    };
    let canonical = genome.canonical();
    let complexity = genome_complexity(&genome);
    if canonical != member.canonical_genome
        || sha256_bytes(canonical.as_bytes()) != member.semantic_sha256
    {
        return rejected(
            complexity,
            "canonical or semantic identity mismatch".to_string(),
        );
    }
    if let Err(error) = validate_v3_genome(&genome, task, schema) {
        return rejected(complexity, format!("admission: {error:#}"));
    }
    match binary_cv_on_slices(
        &genome,
        fold_cache,
        CV5_FOLDS,
        &[QUICK_HOLDOUT_FOLD],
        task,
        schema,
    ) {
        Ok(result) => BinaryEvaluation {
            complexity,
            result: Some(result),
            rejection_reason: None,
        },
        Err(error) => rejected(complexity, format!("cv: {error:#}")),
    }
}

/// Binary panel-facing selection: mirrors the regression lane's structure
/// (quick stratified CV5 fold-0 race, CV7 confirmation for the top 10) with
/// v2-binary receipts throughout.
pub(crate) fn select_dataset_binary(
    train: &RegressionData,
    dataset: &str,
    train_sha256: &str,
    seed: u64,
    seed_bytes: &[u8],
    artifact_root: &Path,
    generations: usize,
) -> Result<serde_json::Value> {
    let seed_sha = sha256_bytes(seed_bytes);
    if seed_sha != SEED_BINARY_SHA256 {
        bail!("fixed binary seed population hash mismatch: {seed_sha}");
    }
    require_fresh_artifact_root(artifact_root)?;
    fs::create_dir_all(artifact_root)?;
    let task = binary_task();
    let schema = binary_schema(train.features);
    let mut population = parse_binary_population(seed_bytes, POPULATION, &task, &schema)?;
    let n = train.x.len().min(train.y.len());
    let positives = train.y[..n].iter().filter(|v| **v == 1.0).count();
    let identity = BinaryCampaignIdentity {
        schema: CAMPAIGN_SCHEMA,
        evaluator: EVALUATOR,
        task: TaskKind::Binary.as_str(),
        dataset_hash: dataset,
        train_sha256,
        train_rows: n,
        positives,
        negatives: n - positives,
        generations,
        population: POPULATION,
        cv_folds: CV5_FOLDS,
        confirm_folds: CV7_FOLDS,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        seed,
        member_execution: "one scoped thread per population member (no core pinning)",
        initial_population_sha256: &seed_sha,
        quick_holdout_fold: QUICK_HOLDOUT_FOLD,
        final_cv7_candidates: FINAL_CV7_CANDIDATES,
        fold_authority: FOLD_AUTHORITY,
    };
    immutable_write(
        &artifact_root.join("campaign.v2-binary.json"),
        &canonical_json(&identity)?,
    )?;
    let cv5_manifest = build_binary_fold_manifest(dataset, train_sha256, &train.y[..n], CV5_FOLDS)?;
    let cv5_manifest_bytes = canonical_json(&cv5_manifest)?;
    let cv5_manifest_sha256 = sha256_bytes(&cv5_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv5.v2-binary.json"),
        &cv5_manifest_bytes,
    )?;
    let ledger_path = artifact_root.join("slots.v2-binary.jsonl");
    let mut ledger_file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&ledger_path)
        .with_context(|| format!("create append-only binary ledger {}", ledger_path.display()))?;
    let fold_cache = BinaryFoldCache::new(train, dataset);
    let mut memo = HashMap::<String, BinaryEvaluation>::new();
    let mut final_ranked = Vec::<BinarySlotLedger>::new();

    for generation in 0..generations {
        let evaluations = thread::scope(|scope| -> Result<Vec<(bool, BinaryEvaluation)>> {
            let mut handles = Vec::with_capacity(population.len());
            for (slot, member) in population.iter().enumerate() {
                let memo = &memo;
                let fold_cache = &fold_cache;
                let task = &task;
                let schema = &schema;
                handles.push(
                    thread::Builder::new()
                        .name(format!("jain-bin-g{generation:02}-s{slot:02}"))
                        .spawn_scoped(scope, move || -> (bool, BinaryEvaluation) {
                            if let Some(hit) = memo.get(&member.semantic_sha256) {
                                return (false, hit.clone());
                            }
                            let evaluation =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    evaluate_binary_unique(member, fold_cache, task, schema)
                                }))
                                .unwrap_or_else(|_| {
                                    BinaryEvaluation {
                                        complexity: 0,
                                        result: None,
                                        rejection_reason: Some(
                                            "panic during evaluation".to_string(),
                                        ),
                                    }
                                });
                            (true, evaluation)
                        })
                        .with_context(|| {
                            format!("spawn binary member thread g{generation} s{slot}")
                        })?,
                );
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("binary member thread panicked"))
                })
                .collect()
        })?;
        let mut slots = Vec::with_capacity(POPULATION);
        for (slot, (miss, evaluation)) in evaluations.into_iter().enumerate() {
            memo.entry(population[slot].semantic_sha256.clone())
                .or_insert_with(|| evaluation.clone());
            let result = evaluation.result.as_ref();
            slots.push(BinarySlotLedger {
                schema: LEDGER_SCHEMA.to_string(),
                generation,
                slot,
                canonical_genome: population[slot].canonical_genome.clone(),
                semantic_sha256: population[slot].semantic_sha256.clone(),
                ancestry: population[slot].ancestry.clone(),
                complexity: evaluation.complexity,
                valid: result.is_some(),
                rejection_reason: evaluation.rejection_reason.clone(),
                cache_disposition: if miss { "miss" } else { "hit" }.to_string(),
                fold_auc: result.map(|r| optional_folds(&r.fold_auc)),
                fold_defined: result.map(|r| r.fold_defined.clone()),
                pooled_oof_auc: result.map(|r| r.pooled_oof_auc),
                pooled_oof_logloss: result.map(|r| r.pooled_oof_logloss),
                pooled_oof_brier: result.map(|r| r.pooled_oof_brier),
                robust_fitness: result.map(|r| r.robust_fitness),
            });
        }
        let mut ranked = slots
            .iter()
            .filter(|slot| slot.rankable())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_binary_slots);
        if ranked.len() < PARENT_POOL {
            bail!(
                "binary generation {generation} produced only {} valid genomes; selection requires {}",
                ranked.len(),
                PARENT_POOL
            );
        }
        for slot in &slots {
            ledger_file.write_all(&canonical_json(slot)?)?;
        }
        ledger_file.flush()?;
        ledger_file.sync_all()?;
        eprintln!(
            "[jain-binary-cv] generation={generation} valid={} best={:.12}",
            ranked.len(),
            ranked[0].robust_fitness.unwrap()
        );
        let next = if generation + 1 < generations {
            refill_binary_population(seed, generation + 1, &ranked, &task, &schema)?
        } else {
            Vec::new()
        };
        final_ranked = ranked;
        population = next;
    }
    drop(ledger_file);

    if final_ranked.len() < FINAL_CV7_CANDIDATES {
        bail!(
            "binary final generation has only {} candidates; CV7 requires {}",
            final_ranked.len(),
            FINAL_CV7_CANDIDATES
        );
    }
    let cv7_manifest = build_binary_fold_manifest(dataset, train_sha256, &train.y[..n], CV7_FOLDS)?;
    let cv7_manifest_bytes = canonical_json(&cv7_manifest)?;
    let cv7_manifest_sha256 = sha256_bytes(&cv7_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv7.v2-binary.json"),
        &cv7_manifest_bytes,
    )?;
    let cv7_fold_ids = (0..CV7_FOLDS).collect::<Vec<_>>();
    let mut finalists = Vec::with_capacity(FINAL_CV7_CANDIDATES);
    let mut cv7_results = HashMap::<String, BinaryCvResult>::new();
    for (quick_rank, slot) in final_ranked.iter().take(FINAL_CV7_CANDIDATES).enumerate() {
        let quick_evidence = BinarySelectionEvidence {
            fold_plan_folds: CV5_FOLDS,
            evaluated_fold_ids: vec![QUICK_HOLDOUT_FOLD],
            fold_manifest_sha256: cv5_manifest_sha256.clone(),
            fold_auc: slot.fold_auc.clone().unwrap(),
            fold_defined: slot.fold_defined.clone().unwrap(),
            pooled_oof_auc: slot.pooled_oof_auc.unwrap(),
            pooled_oof_logloss: slot.pooled_oof_logloss.unwrap(),
            pooled_oof_brier: slot.pooled_oof_brier.unwrap(),
            robust_fitness: slot.robust_fitness.unwrap(),
        };
        let genome = GenomeIr::parse(&slot.canonical_genome)?;
        let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            binary_cv_on_slices(
                &genome,
                &fold_cache,
                CV7_FOLDS,
                &cv7_fold_ids,
                &task,
                &schema,
            )
        }));
        let (cv7, rejection_reason) = match evaluated {
            Ok(Ok(result)) => {
                let evidence = BinarySelectionEvidence::from_result(
                    &result,
                    CV7_FOLDS,
                    cv7_fold_ids.clone(),
                    &cv7_manifest_sha256,
                );
                cv7_results.insert(slot.semantic_sha256.clone(), result);
                (Some(evidence), None)
            }
            Ok(Err(error)) => (None, Some(format!("cv7: {error:#}"))),
            Err(_) => (None, Some("cv7 panic".to_string())),
        };
        finalists.push(BinaryFinalist {
            quick_rank,
            generation: generations - 1,
            slot: slot.slot,
            canonical_genome: slot.canonical_genome.clone(),
            semantic_sha256: slot.semantic_sha256.clone(),
            complexity: slot.complexity,
            quick_cv5_fold0: quick_evidence,
            cv7,
            rejection_reason,
        });
    }
    for finalist in &finalists {
        immutable_write(
            &artifact_root.join("finalists").join(format!(
                "quick-rank-{:02}.cv7.v2-binary.json",
                finalist.quick_rank
            )),
            &canonical_json(finalist)?,
        )?;
    }
    let mut confirmed = finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    confirmed.sort_by(|left, right| {
        let l = left.cv7.as_ref().unwrap();
        let r = right.cv7.as_ref().unwrap();
        r.robust_fitness
            .total_cmp(&l.robust_fitness)
            .then_with(|| l.pooled_oof_logloss.total_cmp(&r.pooled_oof_logloss))
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let selected = confirmed
        .first()
        .copied()
        .context("none of the binary finalists completed full CV7")?;
    let winner_cv7 = cv7_results
        .get(&selected.semantic_sha256)
        .context("winner CV7 evidence missing")?;
    // Leak-safe, report-only thresholds from the winner's pooled CV7
    // train-OOF probabilities; frozen into the receipt, never applied to the
    // artifact's probability output.
    let (f1_threshold, f1_value) =
        best_f1_threshold(&winner_cv7.oof_truth, &winner_cv7.oof_probabilities);
    let (youden_thr, youden_value) =
        youden_threshold(&winner_cv7.oof_truth, &winner_cv7.oof_probabilities);
    let thresholds = BinaryThresholds {
        f1_threshold,
        f1_value,
        youden_threshold: youden_thr,
        youden_value,
    };
    // Winner bound artifacts: fit on the full training table, emit, publish
    // content-addressed and immutable.
    let winner_genome = GenomeIr::parse(&selected.canonical_genome)?;
    let full_table = raw_table(train.features, &train.x[..n])?;
    let winner_model = fit_table(&winner_genome, &task, &schema, &full_table, &train.y[..n])?;
    let emitted = emit_bound(&winner_model, &feat_math::transpile::EmitOptions::default())?;
    let artifact_dir = artifact_root
        .join("artifacts")
        .join(&selected.semantic_sha256);
    let python_sha256 = sha256_bytes(emitted.python.as_bytes());
    let rust_sha256 = sha256_bytes(emitted.rust.as_bytes());
    let manifest_sha256 = sha256_bytes(emitted.manifest.as_bytes());
    immutable_write(
        &artifact_dir.join(format!("{python_sha256}.py")),
        emitted.python.as_bytes(),
    )?;
    immutable_write(
        &artifact_dir.join(format!("{rust_sha256}.rs")),
        emitted.rust.as_bytes(),
    )?;
    immutable_write(
        &artifact_dir.join(format!("{manifest_sha256}.manifest.json")),
        emitted.manifest.as_bytes(),
    )?;
    let winner_artifacts = BinaryArtifactHashes {
        python_sha256,
        rust_sha256,
        manifest_sha256,
    };
    make_readonly(&ledger_path)?;
    let ledger_sha256 = sha256_file(&ledger_path)?;
    let selection = BinarySelectionFreeze {
        schema: SELECTION_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        task: TaskKind::Binary.as_str().to_string(),
        dataset_hash: dataset.to_string(),
        train_sha256: train_sha256.to_string(),
        initial_population_sha256: seed_sha,
        generation: generations - 1,
        slot: selected.slot,
        canonical_genome: selected.canonical_genome.clone(),
        semantic_sha256: selected.semantic_sha256.clone(),
        complexity: selected.complexity,
        ranking_authority: "top 10 by quick stratified-CV5 fold-0 robust fitness; final selection by full CV7 robust fitness descending, pooled OOF logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
        ledger_sha256,
        quick_cv5_fold0: selected.quick_cv5_fold0.clone(),
        cv7_pooled_oof_auc: winner_cv7.pooled_oof_auc,
        cv7: selected.cv7.clone().unwrap(),
        thresholds,
        winner_artifacts,
        cv7_finalists: finalists,
    };
    let selection_bytes = canonical_json(&selection)?;
    let selection_sha256 = sha256_bytes(&selection_bytes);
    immutable_write(
        &artifact_root.join("selection-freeze.v2-binary.json"),
        &selection_bytes,
    )?;
    let complete = BinarySelectionComplete {
        schema: SELECTION_COMPLETE_SCHEMA.to_string(),
        status: "complete".to_string(),
        selection_file: "selection-freeze.v2-binary.json".to_string(),
        selection_sha256,
    };
    immutable_write(
        &artifact_root.join("SELECTION_COMPLETE"),
        &canonical_json(&complete)?,
    )?;
    Ok(serde_json::to_value(&selection)?)
}

/// Light immutable-freeze validation for the binary lane: receipt parses,
/// identity matches, ledger hash matches the sealed file, the winner genome
/// is canonical + admissible, and the evidence is internally consistent.
pub(crate) fn validate_freeze_binary(
    artifact_root: &Path,
    dataset: &str,
    train_sha256: &str,
    features: usize,
    generations: usize,
) -> Result<BinarySelectionFreeze> {
    let bytes = fs::read(artifact_root.join("selection-freeze.v2-binary.json"))
        .with_context(|| format!("read binary freeze under {}", artifact_root.display()))?;
    let freeze: BinarySelectionFreeze =
        serde_json::from_slice(&bytes).context("parse binary selection freeze")?;
    if freeze.schema != SELECTION_SCHEMA
        || freeze.evaluator != EVALUATOR
        || freeze.dataset_hash != dataset
        || freeze.train_sha256 != train_sha256
        || freeze.generation + 1 != generations
    {
        bail!("binary freeze identity mismatch for {dataset}");
    }
    let ledger_sha = sha256_file(&artifact_root.join("slots.v2-binary.jsonl"))?;
    if ledger_sha != freeze.ledger_sha256 {
        bail!("binary ledger hash differs from frozen receipt");
    }
    let genome = GenomeIr::parse(&freeze.canonical_genome)?;
    if genome.canonical() != freeze.canonical_genome
        || sha256_bytes(freeze.canonical_genome.as_bytes()) != freeze.semantic_sha256
    {
        bail!("binary freeze genome identity mismatch");
    }
    validate_v3_genome(&genome, &binary_task(), &binary_schema(features))?;
    if !freeze.cv7.robust_fitness.is_finite() || !freeze.cv7_pooled_oof_auc.is_finite() {
        bail!("binary freeze evidence is non-finite");
    }
    let complete_bytes = fs::read(artifact_root.join("SELECTION_COMPLETE"))
        .context("binary SELECTION_COMPLETE marker missing")?;
    let complete: BinarySelectionComplete = serde_json::from_slice(&complete_bytes)?;
    if complete.schema != SELECTION_COMPLETE_SCHEMA || complete.status != "complete" {
        bail!("binary completion marker is invalid");
    }
    Ok(freeze)
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BinaryScore {
    pub(crate) dataset_hash: String,
    pub(crate) semantic_sha256: String,
    pub(crate) canonical_genome: String,
    pub(crate) complexity: usize,
    pub(crate) test_rows: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
    pub(crate) brier: f64,
    pub(crate) average_precision: f64,
    pub(crate) lift_at_10pct: f64,
    pub(crate) accuracy_at_half: f64,
    pub(crate) f1_at_half: f64,
    pub(crate) frozen_f1_threshold: f64,
    pub(crate) accuracy_at_frozen_threshold: f64,
    pub(crate) f1_at_frozen_threshold: f64,
    pub(crate) fit_seconds: f64,
    pub(crate) predict_seconds: f64,
    pub(crate) predictions: Vec<f64>,
}

/// Score one frozen binary selection on held-out test data. AUC is the sole
/// win/loss authority; everything else is report-only.
pub(crate) fn score_dataset_binary(
    artifact_root: &Path,
    dataset: &str,
    train: &RegressionData,
    train_sha256: &str,
    test: &RegressionData,
    generations: usize,
) -> Result<BinaryScore> {
    let freeze = validate_freeze_binary(
        artifact_root,
        dataset,
        train_sha256,
        train.features,
        generations,
    )?;
    if test.features != train.features {
        bail!("test width differs from train width for {dataset}");
    }
    let task = binary_task();
    let schema = binary_schema(train.features);
    let genome = GenomeIr::parse(&freeze.canonical_genome)?;
    let n = train.x.len().min(train.y.len());
    let train_table = raw_table(train.features, &train.x[..n])?;
    let fit_started = Instant::now();
    let model = fit_table(&genome, &task, &schema, &train_table, &train.y[..n])?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    let test_table = raw_table(test.features, &test.x)?;
    let predict_started = Instant::now();
    let predictions = model.predict_table(&test_table)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    validate_probabilities(&predictions, test.y.len())?;
    let thr = freeze.thresholds.f1_threshold;
    Ok(BinaryScore {
        dataset_hash: dataset.to_string(),
        semantic_sha256: freeze.semantic_sha256.clone(),
        canonical_genome: freeze.canonical_genome.clone(),
        complexity: freeze.complexity,
        test_rows: test.y.len(),
        auc: auc(&test.y, &predictions),
        logloss: logloss(&test.y, &predictions),
        brier: brier(&test.y, &predictions),
        average_precision: average_precision(&test.y, &predictions),
        lift_at_10pct: lift_at(&test.y, &predictions, 0.1),
        accuracy_at_half: accuracy_at(&test.y, &predictions, 0.5),
        f1_at_half: f1_at(&test.y, &predictions, 0.5),
        frozen_f1_threshold: thr,
        accuracy_at_frozen_threshold: accuracy_at(&test.y, &predictions, thr),
        f1_at_frozen_threshold: f1_at(&test.y, &predictions, thr),
        fit_seconds,
        predict_seconds,
        predictions,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use feat_math::Pcg32;
    use std::path::PathBuf;

    pub(crate) fn registry_root() -> PathBuf {
        std::env::temp_dir().join("jope-dime-binary-registry")
    }

    /// Deterministic fixture registry for the manual CLI smoke: interleaved
    /// train/test splits, both classes guaranteed in every file.
    pub(crate) fn export_registry_to(root: &Path) {
        let _ = fs::remove_dir_all(root);
        let (train_a, test_a) = fixture_balanced();
        write_csv(
            &root.join("b1a0f1c500000001").join("train.csv"),
            &train_a.x,
            &train_a.y,
        );
        write_csv(
            &root.join("b1a0f1c500000001").join("test.csv"),
            &test_a.x,
            &test_a.y,
        );
        let full_b = fixture_imbalanced();
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        // Class-stratified deterministic split: every 5th row PER CLASS goes
        // to test, so both files always hold >= 2 rows of each class.
        let mut class_counts = [0usize; 2];
        for (row, target) in full_b.x.iter().zip(&full_b.y) {
            let class = usize::from(*target == 1.0);
            let index = class_counts[class];
            class_counts[class] += 1;
            if index % 5 == 4 {
                test_x.push(row.clone());
                test_y.push(*target);
            } else {
                train_x.push(row.clone());
                train_y.push(*target);
            }
        }
        assert!(
            test_y.iter().filter(|v| **v == 1.0).count() >= 2,
            "smoke test split lacks positives"
        );
        assert!(train_y.iter().filter(|v| **v == 1.0).count() >= 2);
        write_csv(
            &root.join("b1a0f1c500000002").join("train.csv"),
            &train_x,
            &train_y,
        );
        write_csv(
            &root.join("b1a0f1c500000002").join("test.csv"),
            &test_x,
            &test_y,
        );
    }

    fn write_csv(path: &Path, x: &[Vec<f64>], y: &[f64]) {
        let mut text = String::new();
        for (row, target) in x.iter().zip(y) {
            for value in row {
                text.push_str(&format!("{value:?},"));
            }
            text.push_str(&format!("{target:?}\n"));
        }
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn regression_data(x: Vec<Vec<f64>>, y: Vec<f64>) -> RegressionData {
        let features = x[0].len();
        RegressionData { x, y, features }
    }

    /// Balanced nonlinear fixture: 240 train + 100 test rows, 6 features.
    fn fixture_balanced() -> (RegressionData, RegressionData) {
        let mut rng = Pcg32::new(97, 11);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..340 {
            let row: Vec<f64> = (0..6).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let score =
                (2.0 * row[0]).sin() + row[1] * row[2] - 0.3 * row[3] + 0.2 * rng.gaussian();
            x.push(row);
            y.push(if score > 0.0 { 1.0 } else { 0.0 });
        }
        let test_x = x.split_off(240);
        let test_y = y.split_off(240);
        (regression_data(x, y), regression_data(test_x, test_y))
    }

    /// Imbalanced (~9:1) nonlinear fixture: 200 rows, 5 features.
    fn fixture_imbalanced() -> RegressionData {
        let mut rng = Pcg32::new(131, 7);
        let mut x = Vec::new();
        let mut scores = Vec::new();
        for _ in 0..200 {
            let row: Vec<f64> = (0..5).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let score = row[0] * row[1] + (3.0 * row[2]).sin() + 0.15 * rng.gaussian();
            x.push(row);
            scores.push(score);
        }
        let mut sorted = scores.clone();
        sorted.sort_by(f64::total_cmp);
        let threshold = sorted[180];
        let y = scores
            .iter()
            .map(|s| if *s > threshold { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        (regression_data(x, y))
    }

    fn fixture_degenerate() -> RegressionData {
        let mut rng = Pcg32::new(41, 3);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..12 {
            x.push((0..3).map(|_| rng.next_f64() * 2.0 - 1.0).collect());
            y.push(if i < 3 { 1.0 } else { 0.0 });
        }
        regression_data(x, y)
    }

    fn temp_root(label: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("jope-dime-binary-{}-{label}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn seed_bytes() -> &'static [u8] {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/seeds/generation-zero-binary.txt"
        ))
    }

    #[test]
    fn binary_generation_zero_is_pinned_canonical_unique_and_valid() {
        assert_eq!(sha256_bytes(seed_bytes()), SEED_BINARY_SHA256);
        let population =
            parse_binary_population(seed_bytes(), POPULATION, &binary_task(), &binary_schema(6))
                .unwrap();
        assert_eq!(population.len(), POPULATION);
    }

    #[test]
    fn probability_validation_rejects_out_of_range() {
        assert!(validate_probabilities(&[0.0, 0.5, 1.0], 3).is_ok());
        assert!(validate_probabilities(&[0.5], 2).is_err());
        assert!(validate_probabilities(&[1.2, 0.5], 2).is_err());
        assert!(validate_probabilities(&[-0.1, 0.5], 2).is_err());
        assert!(validate_probabilities(&[f64::NAN, 0.5], 2).is_err());
    }

    #[test]
    fn degenerate_folds_are_marked_undefined_and_pool_finite() {
        let train = fixture_degenerate();
        let genome = GenomeIr::parse("(pipe X (logit_ridge))").unwrap();
        let evaluated = (0..CV7_FOLDS).collect::<Vec<_>>();
        let result =
            exact_cv_on_folds_binary(&genome, &train, CV7_FOLDS, &evaluated, "degenerate-fixture")
                .unwrap();
        assert_eq!(result.fold_defined.len(), CV7_FOLDS);
        assert!(
            result.fold_defined.iter().any(|defined| !defined),
            "a 12-row/3-positive CV7 must contain single-class validation folds"
        );
        assert!(result.pooled_oof_auc.is_finite());
        assert!(result.robust_fitness.is_finite());
        // Undefined folds are NaN in-memory and excluded from the penalty.
        for (auc_value, defined) in result.fold_auc.iter().zip(&result.fold_defined) {
            assert_eq!(auc_value.is_finite(), *defined);
        }
    }

    #[test]
    fn campaign_is_deterministic_and_winner_beats_linear_baseline() {
        let (train, test) = fixture_balanced();
        let seed = 0x6a70_6269_6e5f_7631_u64 ^ 20260616;
        let root_a = temp_root("determinism-a");
        let root_b = temp_root("determinism-b");
        let value_a = select_dataset_binary(
            &train,
            "b1a0f1c500000001",
            "fixture-train-sha",
            seed,
            seed_bytes(),
            &root_a,
            2,
        )
        .unwrap();
        let value_b = select_dataset_binary(
            &train,
            "b1a0f1c500000001",
            "fixture-train-sha",
            seed,
            seed_bytes(),
            &root_b,
            2,
        )
        .unwrap();
        let freeze_a = fs::read(root_a.join("selection-freeze.v2-binary.json")).unwrap();
        let freeze_b = fs::read(root_b.join("selection-freeze.v2-binary.json")).unwrap();
        assert_eq!(freeze_a, freeze_b, "binary freezes must be byte-identical");
        let ledger_a = fs::read(root_a.join("slots.v2-binary.jsonl")).unwrap();
        let ledger_b = fs::read(root_b.join("slots.v2-binary.jsonl")).unwrap();
        assert_eq!(ledger_a, ledger_b, "binary ledgers must be byte-identical");
        assert_eq!(value_a, value_b);

        // AUC sanity on held-out test data: the evolved winner must clearly
        // beat a plain linear-probability baseline on this nonlinear fixture.
        let score = score_dataset_binary(
            &root_a,
            "b1a0f1c500000001",
            &train,
            "fixture-train-sha",
            &test,
            2,
        )
        .unwrap();
        let task = binary_task();
        let schema = binary_schema(train.features);
        let baseline_genome = GenomeIr::parse("(pipe X (logit_ridge))").unwrap();
        let baseline_model = fit_table(
            &baseline_genome,
            &task,
            &schema,
            &raw_table(train.features, &train.x).unwrap(),
            &train.y,
        )
        .unwrap();
        let baseline_predictions = baseline_model
            .predict_table(&raw_table(test.features, &test.x).unwrap())
            .unwrap();
        let baseline_auc = auc(&test.y, &baseline_predictions);
        assert!(score.auc > 0.8, "winner test AUC too weak: {}", score.auc);
        assert!(
            score.auc > baseline_auc + 0.05,
            "winner {} must clearly beat the linear baseline {}",
            score.auc,
            baseline_auc
        );
        assert!(score.frozen_f1_threshold > 0.0 && score.frozen_f1_threshold < 1.0);
        // Thresholded report metrics are present and sane.
        assert!(score.f1_at_frozen_threshold >= 0.0 && score.f1_at_frozen_threshold <= 1.0);
        let _ = fs::remove_dir_all(&root_a);
        let _ = fs::remove_dir_all(&root_b);
    }

    #[test]
    fn imbalanced_campaign_and_scoring_smoke() {
        let train = fixture_imbalanced();
        let seed = 0x6a70_6269_6e5f_7631_u64 ^ 20260616 ^ 2;
        let root = temp_root("imbalanced");
        select_dataset_binary(
            &train,
            "b1a0f1c500000002",
            "fixture-train-sha-2",
            seed,
            seed_bytes(),
            &root,
            1,
        )
        .unwrap();
        let freeze = validate_freeze_binary(
            &root,
            "b1a0f1c500000002",
            "fixture-train-sha-2",
            train.features,
            1,
        )
        .unwrap();
        assert!(freeze.cv7_pooled_oof_auc.is_finite());
        // Winner bound artifacts were published content-addressed.
        let artifact_dir = root.join("artifacts").join(&freeze.semantic_sha256);
        assert!(artifact_dir.exists());
        let entries = fs::read_dir(&artifact_dir).unwrap().count();
        assert_eq!(entries, 3, "python + rust + manifest artifacts expected");
        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod cli_smoke_support {
    //! Manual CLI smoke harness: `cargo test -- --ignored export_binary_fixture_registry`
    //! writes the two fixture datasets as a registry for the real binary.
    use super::tests::{export_registry_to, registry_root};

    #[test]
    #[ignore]
    fn export_binary_fixture_registry() {
        let root = registry_root();
        export_registry_to(&root);
        println!("binary fixture registry at {}", root.display());
    }
}
