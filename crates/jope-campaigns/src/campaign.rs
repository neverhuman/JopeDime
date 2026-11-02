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
mod population;
mod run;
#[cfg(test)]
mod tests;

use artifacts::*;
use file_io::*;
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

#[derive(Clone, Debug)]
struct LiveBest {
    semantic_sha256: String,
    complexity: usize,
    robust_fitness: f64,
    pooled_oof_r2: f64,
    pooled_oof_rmse: f64,
}

struct LiveState {
    file: File,
    human_file: File,
    started: Instant,
    completed_slots: usize,
    valid_slots: usize,
    invalid_slots: usize,
    cache_hits: usize,
    cv7_completed: usize,
    generation_completed: Vec<usize>,
    best: Option<LiveBest>,
    cv7_best: Option<LiveBest>,
    generation_best: Vec<Option<LiveBest>>,
}

struct LiveLogger {
    total_slots: usize,
    population: usize,
    json_path: PathBuf,
    human_path: PathBuf,
    state: Mutex<LiveState>,
}

impl LiveLogger {
    fn create(
        root: &Path,
        total_slots: usize,
        population: usize,
        rayon_threads: usize,
    ) -> Result<Self> {
        let path = root.join("live-progress.v1.jsonl");
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("create live progress log {}", path.display()))?;
        let human_path = root.join("live-progress.log");
        let mut human_file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&human_path)
            .with_context(|| format!("create human live progress log {}", human_path.display()))?;
        writeln!(
            human_file,
            "# JAIN EXACT-CV LIVE | runtime capacity {rayon_threads} threads | {population} dedicated single-threaded member workers | {total_slots} quick slots | fixed CV5 fold {QUICK_CV5_HOLDOUT_FOLD} | top {FINAL_CV7_CANDIDATES} advance to full CV7"
        )?;
        writeln!(
            human_file,
            "# TabFM RMSE reference {TABFM_RMSE:.14}; all AHEAD/BEHIND labels are CV proxies until phase-two test scoring"
        )?;
        human_file.flush()?;
        human_file.sync_data()?;
        let generations = total_slots.div_ceil(population.max(1));
        Ok(Self {
            total_slots,
            population,
            json_path: path,
            human_path,
            state: Mutex::new(LiveState {
                file,
                human_file,
                started: Instant::now(),
                completed_slots: 0,
                valid_slots: 0,
                invalid_slots: 0,
                cache_hits: 0,
                cv7_completed: 0,
                generation_completed: vec![0; generations],
                best: None,
                cv7_best: None,
                generation_best: vec![None; generations],
            }),
        })
    }

    fn record_slot(
        &self,
        generation: usize,
        slot: usize,
        member: &PopulationMember,
        evaluation: &CachedEvaluation,
        cache_miss: bool,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.completed_slots += 1;
        state.generation_completed[generation] += 1;
        if !cache_miss {
            state.cache_hits += 1;
        }
        if let Some(result) = evaluation.result.as_ref() {
            state.valid_slots += 1;
            let proposed = LiveBest {
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: evaluation.complexity,
                robust_fitness: result.robust_fitness,
                pooled_oof_r2: result.pooled_oof_r2,
                pooled_oof_rmse: result.pooled_oof_rmse,
            };
            if state
                .best
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.best = Some(proposed.clone());
            }
            if state.generation_best[generation]
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.generation_best[generation] = Some(proposed);
            }
        } else {
            state.invalid_slots += 1;
        }
        let best = state.best.clone();
        let generation_best = state.generation_best[generation].clone();
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let throughput = state.completed_slots as f64 / elapsed_seconds.max(f64::MIN_POSITIVE);
        let eta_seconds = self.total_slots.saturating_sub(state.completed_slots) as f64
            / throughput.max(f64::MIN_POSITIVE);
        let progress_percent =
            100.0 * state.completed_slots as f64 / self.total_slots.max(1) as f64;
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "slot_terminal",
            "scoring_phase": "QUICK_CV5_FOLD0",
            "cv5_holdout_fold": QUICK_CV5_HOLDOUT_FOLD,
            "generation": generation,
            "slot": slot,
            "completed_slots": state.completed_slots,
            "total_slots": self.total_slots,
            "generation_completed_slots": state.generation_completed[generation],
            "generation_total_slots": self.population,
            "progress_percent": progress_percent,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "throughput_slots_per_second": throughput,
            "cumulative_valid_slots": state.valid_slots,
            "cumulative_invalid_slots": state.invalid_slots,
            "cumulative_cache_hits": state.cache_hits,
            "semantic_sha256": member.semantic_sha256,
            "cache_disposition": if cache_miss { "miss" } else { "hit" },
            "valid": evaluation.result.is_some(),
            "rejection_reason": evaluation.rejection_reason,
            "robust_fitness": evaluation.result.as_ref().map(|result| result.robust_fitness),
            "pooled_oof_r2": evaluation.result.as_ref().map(|result| result.pooled_oof_r2),
            "pooled_oof_rmse": evaluation.result.as_ref().map(|result| result.pooled_oof_rmse),
            "incumbent_semantic_sha256": best.as_ref().map(|value| value.semantic_sha256.as_str()),
            "incumbent_robust_fitness": best.as_ref().map(|value| value.robust_fitness),
            "incumbent_pooled_oof_r2": best.as_ref().map(|value| value.pooled_oof_r2),
            "incumbent_pooled_oof_rmse": best.as_ref().map(|value| value.pooled_oof_rmse),
            "generation_best_robust_fitness": generation_best.as_ref().map(|value| value.robust_fitness),
            "generation_best_pooled_oof_rmse": generation_best.as_ref().map(|value| value.pooled_oof_rmse),
            "tabfm_test_rmse_reference": TABFM_RMSE,
            "cv_rmse_signed_margin_vs_tabfm": best.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "cv_proxy_track_status": best.as_ref().map(|value| if value.pooled_oof_rmse < TABFM_RMSE { "CV_PROXY_AHEAD" } else { "CV_PROXY_BEHIND" }),
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let status = best.as_ref().map_or("NO_VALID_INCUMBENT", |value| {
            if value.pooled_oof_rmse < TABFM_RMSE {
                "CV_PROXY_AHEAD"
            } else {
                "CV_PROXY_BEHIND"
            }
        });
        let global_fitness = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:.12}", value.robust_fitness),
        );
        let global_rmse = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:.12}", value.pooled_oof_rmse),
        );
        let tabfm_margin = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:+.12}", TABFM_RMSE - value.pooled_oof_rmse),
        );
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] QUICK g={generation:02} {:02}/{:02} | overall={:03}/{:03} ({progress_percent:5.1}%) | member_fit={} | GLOBAL_BEST fitness={} rmse={} | TabFM_proxy_margin={} {} | valid={} invalid={} cache_hits={} | {:.3} slots/s ETA={:.1}s",
            state.generation_completed[generation],
            self.population,
            state.completed_slots,
            self.total_slots,
            evaluation
                .result
                .as_ref()
                .map_or_else(|| "NA".to_string(), |result| format!("{:.12}", result.robust_fitness)),
            global_fitness,
            global_rmse,
            tabfm_margin,
            status,
            state.valid_slots,
            state.invalid_slots,
            state.cache_hits,
            throughput,
            eta_seconds,
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    fn record_cv7_finalist(&self, finalist: &FinalistCv7) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.cv7_completed += 1;
        if let Some(result) = finalist.cv7.as_ref() {
            let proposed = LiveBest {
                semantic_sha256: finalist.semantic_sha256.clone(),
                complexity: finalist.complexity,
                robust_fitness: result.robust_fitness,
                pooled_oof_r2: result.pooled_oof_r2,
                pooled_oof_rmse: result.pooled_oof_rmse,
            };
            if state
                .cv7_best
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.cv7_best = Some(proposed);
            }
        }
        let cv7_best = state.cv7_best.clone();
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "finalist_full_cv7_terminal",
            "scoring_phase": "FULL_CV7_FINALIST",
            "quick_rank": finalist.quick_rank,
            "completed_finalists": state.cv7_completed,
            "total_finalists": FINAL_CV7_CANDIDATES,
            "semantic_sha256": finalist.semantic_sha256,
            "valid": finalist.cv7.is_some(),
            "rejection_reason": finalist.rejection_reason,
            "cv7_robust_fitness": finalist.cv7.as_ref().map(|value| value.robust_fitness),
            "cv7_pooled_oof_r2": finalist.cv7.as_ref().map(|value| value.pooled_oof_r2),
            "cv7_pooled_oof_rmse": finalist.cv7.as_ref().map(|value| value.pooled_oof_rmse),
            "tabfm_test_rmse_reference": TABFM_RMSE,
            "cv7_rmse_signed_margin_vs_tabfm": finalist.cv7.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "cv7_incumbent_semantic_sha256": cv7_best.as_ref().map(|value| value.semantic_sha256.as_str()),
            "cv7_incumbent_robust_fitness": cv7_best.as_ref().map(|value| value.robust_fitness),
            "cv7_incumbent_pooled_oof_rmse": cv7_best.as_ref().map(|value| value.pooled_oof_rmse),
            "cv7_incumbent_rmse_signed_margin_vs_tabfm": cv7_best.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let status = cv7_best.as_ref().map_or("NO_VALID_CV7", |value| {
            if value.pooled_oof_rmse < TABFM_RMSE {
                "CV7_PROXY_AHEAD"
            } else {
                "CV7_PROXY_BEHIND"
            }
        });
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] FULL_CV7 {}/{} quick_rank={:02} | member_fit={} member_rmse={} | CV7_GLOBAL_BEST fitness={} rmse={} | TabFM_proxy_margin={} {}",
            state.cv7_completed,
            FINAL_CV7_CANDIDATES,
            finalist.quick_rank,
            finalist.cv7.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.robust_fitness),
            ),
            finalist.cv7.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.pooled_oof_rmse),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.robust_fitness),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.pooled_oof_rmse),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:+.12}", TABFM_RMSE - value.pooled_oof_rmse),
            ),
            status,
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    fn record_campaign_complete(&self, selection: &SelectionFreeze) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "campaign_complete",
            "elapsed_seconds": elapsed_seconds,
            "terminal_slots": state.completed_slots,
            "completed_finalists": state.cv7_completed,
            "selected_semantic_sha256": selection.semantic_sha256,
            "selected_cv7_robust_fitness": selection.cv7.robust_fitness,
            "selected_cv7_pooled_oof_rmse": selection.cv7.pooled_oof_rmse,
            "selected_cv7_rmse_signed_margin_vs_tabfm": TABFM_RMSE - selection.cv7.pooled_oof_rmse,
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] CAMPAIGN_COMPLETE slots={}/{} cv7={}/{} | WINNER={} | CV7 fitness={:.12} rmse={:.12} | TabFM_proxy_margin={:+.12} {} | FINAL_TEST_PENDING",
            state.completed_slots,
            self.total_slots,
            state.cv7_completed,
            FINAL_CV7_CANDIDATES,
            selection.semantic_sha256,
            selection.cv7.robust_fitness,
            selection.cv7.pooled_oof_rmse,
            TABFM_RMSE - selection.cv7.pooled_oof_rmse,
            if selection.cv7.pooled_oof_rmse < TABFM_RMSE {
                "CV7_PROXY_AHEAD"
            } else {
                "CV7_PROXY_BEHIND"
            },
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    fn seal(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.file.flush()?;
        state.file.sync_all()?;
        state.human_file.flush()?;
        state.human_file.sync_all()?;
        make_readonly(&self.json_path)?;
        make_readonly(&self.human_path)
    }
}
