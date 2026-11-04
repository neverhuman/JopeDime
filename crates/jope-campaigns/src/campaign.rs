//! Selection-safe, full-training exact-CV evolutionary campaign.
//!
//! This module deliberately does not call the ordinary Jain search path.  In
//! phase one it only opens `train.csv`, evaluates every logical population
//! slot against the same fixed fold-0 holdout of the CV5 plan, then evaluates
//! generation nine's top ten with full CV7 and freezes the CV7 winner. Phase
//! two validates that immutable freeze before it even attempts to inspect
//! `test.csv`.

use super::jain_regressor::{read_regression_csv, RegressionData};
use crate::args::JainRegressorArgs;
use crate::runtime::init_runtime_threads;
use anyhow::{bail, Context, Result};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline};
use feat_math::transpile::{emit, Emitted};
use feat_math::{fit, fold_of, GenomeIr, Rows};
#[cfg(test)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

mod artifacts;
mod file_io;
mod live;
mod population;
mod run;
#[cfg(test)]
mod tests;

use artifacts::*;
use file_io::*;
pub(super) use live::{LiveBest, LiveLogger};
use population::*;
pub(crate) use run::run;

const DATASET: &str = "faaa420e922e4128";
const TRAIN_SHA256: &str = "7dc6777e799226009ecffa26d610122bff765af7dc39fdbe69cc1ee9e5485903";
const TEST_SHA256: &str = "0732d97401fa1b37f659159add6a5e71c4570cd589aff563d07ad9e8326e347a";
const TRAIN_ROWS: usize = 1_000;
const TEST_ROWS: usize = 300;
const GENERATIONS: usize = 10;
const POPULATION: usize = 60;
const CV5_FOLDS: usize = 5;
const CV7_FOLDS: usize = 7;
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const QUICK_CV5_HOLDOUT_FOLD: usize = 0;
const FINAL_CV7_CANDIDATES: usize = 10;
const MAX_TRANSFORMS: usize = 6;
const TARGET_RMSE: f64 = 0.07590440570425196;
const TABFM_RMSE: f64 = 0.07593213534123246;
const ARCHIVED_JOPE_RMSE: f64 = 0.07563492482356142;
const CAMPAIGN_SCHEMA: &str = "JainExactCvCampaign.v1";
const LEDGER_SCHEMA: &str = "JainExactCvSlot.v1";
const CHECKPOINT_SCHEMA: &str = "JainExactCvCheckpoint.v1";
const SUMMARY_SCHEMA: &str = "JainExactCvGenerationSummary.v1";
const FOLD_MANIFEST_SCHEMA: &str = "JainExactCvFoldManifest.v1";
const SELECTION_SCHEMA: &str = "JainExactCvSelectionFreeze.v1";
const SELECTION_COMPLETE_SCHEMA: &str = "JainExactCvSelectionComplete.v1";
const PHASE2_SCHEMA: &str = "JainExactCvPhase2.v1";
const EVALUATOR: &str = "prime-jain.native-feat-math-f64.exact-cv.v1";
const FOLD_AUTHORITY: &str = "feat_math.fold_of(dataset_hash,row,k).fnv1a.v1";

/// Shared robust-fitness authority for both exact-CV and the legacy Jain
/// reporting path: max(OOF R2, -1) - .25 * min(population SD(folds), 1).
pub(super) fn robust_fitness(oof_r2: f64, fold_r2: &[f64]) -> f64 {
    let finite = fold_r2
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let population_sd = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / finite.len() as f64)
            .sqrt()
    };
    oof_r2.max(-1.0) - 0.25 * population_sd.min(1.0)
}

#[derive(Clone, Debug)]
struct CampaignConfig {
    dataset: String,
    train_sha256: String,
    generations: usize,
    population: usize,
    cv_folds: usize,
    confirm_folds: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    final_cv7_candidates: usize,
    max_transforms: usize,
    seed: u64,
    rayon_threads: usize,
    expected_train_rows: Option<usize>,
}

impl CampaignConfig {
    fn production(seed: u64, rayon_threads: usize) -> Self {
        Self {
            dataset: DATASET.to_string(),
            train_sha256: TRAIN_SHA256.to_string(),
            generations: GENERATIONS,
            population: POPULATION,
            cv_folds: CV5_FOLDS,
            confirm_folds: CV7_FOLDS,
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            final_cv7_candidates: FINAL_CV7_CANDIDATES,
            max_transforms: MAX_TRANSFORMS,
            seed,
            rayon_threads,
            expected_train_rows: Some(TRAIN_ROWS),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Ancestry {
    operation: String,
    parents: Vec<String>,
    retry_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PopulationMember {
    canonical_genome: String,
    semantic_sha256: String,
    ancestry: Ancestry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArtifactHashes {
    python_sha256: String,
    rust_sha256: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExactCvResult {
    fold_r2: Vec<f64>,
    pooled_oof_r2: f64,
    pooled_oof_rmse: f64,
    robust_fitness: f64,
    fit_seconds: Vec<f64>,
    predict_seconds: Vec<f64>,
}

#[derive(Clone, Debug)]
struct CachedEvaluation {
    source_generation: usize,
    source_slot: usize,
    complexity: usize,
    head_family: String,
    artifacts: Option<ArtifactHashes>,
    result: Option<ExactCvResult>,
    rejection_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SlotLedger {
    schema: String,
    generation: usize,
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    ancestry: Ancestry,
    head_family: String,
    complexity: usize,
    valid: bool,
    rejection_reason: Option<String>,
    cache_disposition: String,
    cache_source_generation: Option<usize>,
    cache_source_slot: Option<usize>,
    artifacts: Option<ArtifactHashes>,
    fold_r2: Option<Vec<f64>>,
    pooled_oof_r2: Option<f64>,
    pooled_oof_rmse: Option<f64>,
    robust_fitness: Option<f64>,
    fold_fit_seconds: Vec<f64>,
    fold_predict_seconds: Vec<f64>,
    cumulative_unique_evaluations: usize,
    cumulative_terminal_slots: usize,
    cumulative_throughput_slots_per_second: f64,
}

impl SlotLedger {
    fn rankable(&self) -> bool {
        self.valid
            && self.robust_fitness.is_some_and(f64::is_finite)
            && self.pooled_oof_r2.is_some_and(f64::is_finite)
            && self
                .fold_r2
                .as_ref()
                .is_some_and(|folds| !folds.is_empty() && folds.iter().all(|v| v.is_finite()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldAssignment {
    fold: usize,
    train_rows: Vec<usize>,
    validation_rows: Vec<usize>,
    train_rows_sha256: String,
    validation_rows_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldManifest {
    schema: String,
    dataset_hash: String,
    dataset_train_sha256: String,
    fold_authority: String,
    folds: usize,
    rows: usize,
    assignments: Vec<FoldAssignment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SelectionEvidence {
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: String,
    fold_r2: Vec<f64>,
    pooled_oof_r2: f64,
    pooled_oof_rmse: f64,
    robust_fitness: f64,
    fit_seconds: Vec<f64>,
    predict_seconds: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FinalistCv7 {
    quick_rank: usize,
    generation: usize,
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    quick_cv5_fold0: SelectionEvidence,
    cv7: Option<SelectionEvidence>,
    rejection_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionFreeze {
    schema: String,
    evaluator: String,
    dataset_hash: String,
    train_sha256: String,
    initial_population_sha256: String,
    generation: usize,
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    ranking_authority: String,
    ledger_sha256: String,
    quick_cv5_fold0: SelectionEvidence,
    cv7: SelectionEvidence,
    cv7_finalists: Vec<FinalistCv7>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionComplete {
    schema: String,
    status: String,
    selection_file: String,
    selection_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CampaignIdentity<'a> {
    schema: &'static str,
    evaluator: &'static str,
    dataset_hash: &'a str,
    train_sha256: &'a str,
    train_rows: usize,
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
    rayon_threads: usize,
    member_execution: &'static str,
    dedicated_member_threads: usize,
    initial_population_sha256: &'a str,
    quick_holdout_fold: usize,
    final_cv7_candidates: usize,
}

#[derive(Clone, Debug, Serialize)]
struct Checkpoint<'a> {
    schema: &'static str,
    generation: usize,
    terminal_slots: usize,
    unique_evaluations: usize,
    ledger_sha256: &'a str,
    next_population: &'a [PopulationMember],
}

#[derive(Clone, Debug, Serialize)]
struct Incumbent {
    generation: usize,
    semantic_sha256: String,
    robust_fitness: f64,
}

#[derive(Clone, Debug, Serialize)]
struct GenerationSummary<'a> {
    schema: &'static str,
    generation: usize,
    population: usize,
    valid: usize,
    invalid: usize,
    cache_hits: usize,
    cache_misses: usize,
    fitness_quantiles: BTreeMap<String, f64>,
    best_fitness: f64,
    incumbent_pooled_oof_rmse: f64,
    cv_rmse_signed_margin_vs_tabfm: f64,
    cv_proxy_track_status: String,
    median_fitness: f64,
    improvement_from_generation_zero: f64,
    canonical_diversity: f64,
    head_mix: BTreeMap<String, usize>,
    incumbent_history: &'a [Incumbent],
    next_population: &'a [PopulationMember],
    cumulative_terminal_slots: usize,
    cumulative_unique_evaluations: usize,
    cumulative_ledger_sha256: &'a str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Phase2Receipt {
    schema: String,
    evaluator: String,
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    selection_sha256: String,
    semantic_sha256: String,
    canonical_genome: String,
    train_rows: usize,
    test_rows: usize,
    predictions_file: String,
    predictions_sha256: String,
    fit_seconds: f64,
    predict_seconds: f64,
    test_rmse: f64,
    test_r2: f64,
    test_mae: f64,
    target_rmse: f64,
    tabfm_ensemble_rmse: f64,
    archived_jope_rmse: f64,
    signed_margin_vs_target: f64,
    signed_margin_vs_tabfm: f64,
    signed_margin_vs_archived_jope: f64,
    beats_target_strict: String,
}
