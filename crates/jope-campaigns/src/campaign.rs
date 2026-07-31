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

fn live_best_cmp(left: &LiveBest, right: &LiveBest) -> Ordering {
    right
        .robust_fitness
        .total_cmp(&left.robust_fitness)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(crate) fn run(args: JainRegressorArgs) -> Result<()> {
    validate_common_exact_flags(&args)?;
    if args.exact_cv_phase2 {
        run_phase2(&args)
    } else {
        run_phase1(&args)
    }
}

fn validate_common_exact_flags(args: &JainRegressorArgs) -> Result<()> {
    if args.fitness != "exact-cv" {
        bail!("the exact-CV runner requires --fitness exact-cv");
    }
    if !args.root.is_absolute() {
        bail!("--root must be absolute for --fitness exact-cv");
    }
    let artifact_root = args
        .exact_cv_dir
        .as_ref()
        .context("--fitness exact-cv requires --exact-cv-dir")?;
    if !artifact_root.is_absolute() {
        bail!("--exact-cv-dir must be absolute");
    }
    if args.dataset.len() != 1 || args.dataset[0] != DATASET {
        bail!("--fitness exact-cv requires exactly --dataset {DATASET}");
    }
    if args.hashes_file.is_some() || args.limit.is_some() {
        bail!("--fitness exact-cv forbids --hashes-file and --limit");
    }
    if args.overwrite
        || args.skip_existing
        || args.compound_existing
        || args.targets_file.is_some()
        || args.time_budget_secs != 0.0
    {
        bail!("--fitness exact-cv forbids overwrite/skip/compound/targets/time-budget controls");
    }
    if args.gpu_kernel_configs != 0
        || args.gpu_screen
        || args.gpu_cache_dir.is_some()
        || args.gpr_proposal_receipt.is_some()
    {
        bail!("--fitness exact-cv requires --gpu-kernel-configs 0 and forbids GPU proposal/screen flags");
    }
    if args.rung0_exact_surrogate_dedup
        || args.rung0_trace_dir.is_some()
        || args.candidate_block_dir.is_some()
        || args.candidate_block_only
        || args.calib
        || args.hard_ten_priors
        || args.lab_hurdle2_train_only
    {
        bail!("--fitness exact-cv forbids proxy, rung-trace, calibration, candidate-block, and lab lanes");
    }
    if args.dry_run || args.status_jsonl.is_some() {
        bail!("--fitness exact-cv forbids legacy dry-run/status output");
    }
    Ok(())
}

fn run_phase1(args: &JainRegressorArgs) -> Result<()> {
    if args.gens != GENERATIONS || args.pop != POPULATION || args.folds != CV5_FOLDS {
        bail!(
            "phase one requires --gens {GENERATIONS}, --pop {POPULATION}, and --folds {CV5_FOLDS}"
        );
    }
    if args.confirm_folds != CV7_FOLDS {
        bail!("phase one requires --confirm-folds {CV7_FOLDS}");
    }
    if !args.exact_initial_population || args.seed_genomes_file.is_none() {
        bail!("phase one requires --exact-initial-population and --seed-genomes-file");
    }
    if !args.skip_holdout_metrics {
        bail!("phase one requires --skip-holdout-metrics");
    }
    if args.no_predictions {
        bail!("phase one does not accept legacy --no-predictions");
    }
    let expected_population_sha256 = args
        .initial_population_sha256
        .as_deref()
        .context("phase one requires --initial-population-sha256")?;
    validate_sha256("--initial-population-sha256", expected_population_sha256)?;
    let artifact_root = args.exact_cv_dir.as_ref().unwrap();
    require_fresh_artifact_root(artifact_root)?;

    // Deliberately name and open only train.csv in phase one.
    let train_path = args.root.join(DATASET).join("train.csv");
    validate_file_hash(&train_path, TRAIN_SHA256, "train.csv")?;
    let train = read_regression_csv(&train_path)
        .with_context(|| format!("load pinned train data {}", train_path.display()))?;
    if train.y.len() != TRAIN_ROWS {
        bail!(
            "pinned train.csv must contain {TRAIN_ROWS} rows, found {}",
            train.y.len()
        );
    }
    let seed_path = args.seed_genomes_file.as_ref().unwrap();
    let seed_bytes = fs::read(seed_path)
        .with_context(|| format!("read exact generation zero {}", seed_path.display()))?;
    if sha256_bytes(&seed_bytes) != expected_population_sha256 {
        bail!("ordered generation-zero file differs from --initial-population-sha256");
    }
    if !seed_bytes.ends_with(b"\n") {
        bail!("ordered generation-zero file must be newline-terminated");
    }
    let initial = parse_initial_population(&seed_bytes, POPULATION)?;
    let provenance_path = seed_path.with_extension("provenance.json");
    let provenance_bytes = fs::read(&provenance_path).with_context(|| {
        format!(
            "read required generation-zero provenance {}",
            provenance_path.display()
        )
    })?;
    let provenance = validate_generation_zero_provenance(
        &provenance_bytes,
        expected_population_sha256,
        POPULATION,
    )?;

    fs::create_dir_all(artifact_root)
        .with_context(|| format!("create exact-CV root {}", artifact_root.display()))?;
    sync_directory(artifact_root)?;
    immutable_write(
        &artifact_root.join("initial-population-provenance.v1.json"),
        &canonical_json(&provenance)?,
    )?;
    let threads = init_runtime_threads("jain-exact-cv", args.threads).threads;
    eprintln!(
        "[jain-exact-cv] phase1 dataset={DATASET} generations={GENERATIONS} population={POPULATION} threads={threads}"
    );
    let selection = execute_campaign(
        &CampaignConfig::production(args.seed, threads),
        &train,
        initial,
        expected_population_sha256,
        artifact_root,
    )?;
    println!("{}", serde_json::to_string(&selection)?);
    Ok(())
}

fn run_phase2(args: &JainRegressorArgs) -> Result<()> {
    if args.exact_initial_population
        || args.seed_genomes_file.is_some()
        || args.initial_population_sha256.is_some()
    {
        bail!("phase two forbids seed and initial-population flags");
    }
    if args.skip_holdout_metrics {
        bail!("phase two must not use --skip-holdout-metrics");
    }
    let artifact_root = args.exact_cv_dir.as_ref().unwrap();

    // Receipt validation is intentionally complete before test_path is even
    // constructed, so malformed phase-two requests cannot probe test data.
    let (selection, selection_sha256) = validate_selection_freeze(artifact_root)?;
    if selection.dataset_hash != DATASET
        || selection.train_sha256 != TRAIN_SHA256
        || selection.generation != GENERATIONS - 1
        || selection.quick_cv5_fold0.fold_plan_folds != CV5_FOLDS
        || selection.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
        || selection.cv7.fold_plan_folds != CV7_FOLDS
        || selection.cv7.evaluated_fold_ids != (0..CV7_FOLDS).collect::<Vec<_>>()
        || selection.cv7_finalists.len() != FINAL_CV7_CANDIDATES
    {
        bail!("selection freeze is not for the pinned exact-CV campaign");
    }
    validate_file_hash(
        &args.root.join(DATASET).join("train.csv"),
        TRAIN_SHA256,
        "train.csv",
    )?;

    let test_path = args.root.join(DATASET).join("test.csv");
    validate_file_hash(&test_path, TEST_SHA256, "test.csv")?;
    let train = read_regression_csv(&args.root.join(DATASET).join("train.csv"))?;
    let test = read_regression_csv(&test_path)?;
    if train.y.len() != TRAIN_ROWS || test.y.len() != TEST_ROWS {
        bail!(
            "phase two requires {TRAIN_ROWS} train and {TEST_ROWS} test rows, found {} and {}",
            train.y.len(),
            test.y.len()
        );
    }
    if train.features != test.features {
        bail!("train/test feature-count mismatch");
    }
    let genome =
        GenomeIr::parse(&selection.canonical_genome).context("parse frozen selected genome")?;
    if genome.canonical() != selection.canonical_genome
        || sha256_bytes(selection.canonical_genome.as_bytes()) != selection.semantic_sha256
    {
        bail!("selection freeze canonical genome identity is invalid");
    }

    let fit_started = Instant::now();
    let model = fit(&genome, &train.x, &train.y).context("fit frozen genome on all train rows")?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    let predict_started = Instant::now();
    let predictions = model.predict(&test.x);
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    if predictions.len() != TEST_ROWS || predictions.iter().any(|value| !value.is_finite()) {
        bail!("frozen genome did not produce exactly {TEST_ROWS} finite predictions");
    }

    let phase2_dir = artifact_root.join("phase2");
    fs::create_dir_all(&phase2_dir)?;
    let predictions_path = phase2_dir.join("predictions.csv");
    if predictions_path.exists() {
        bail!("phase-two predictions already exist; exact-CV outputs are immutable");
    }
    write_predictions_atomic(&predictions_path, &predictions)?;
    let durable_predictions = read_predictions(&predictions_path)?;
    if durable_predictions.len() != TEST_ROWS
        || durable_predictions.iter().any(|value| !value.is_finite())
    {
        bail!("durable prediction file is not exactly {TEST_ROWS} finite values");
    }
    let (test_rmse, test_r2, test_mae) = independent_metrics(&test.y, &durable_predictions)?;
    let predictions_sha256 = sha256_file(&predictions_path)?;
    let receipt = Phase2Receipt {
        schema: PHASE2_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        dataset_hash: DATASET.to_string(),
        train_sha256: TRAIN_SHA256.to_string(),
        test_sha256: TEST_SHA256.to_string(),
        selection_sha256,
        semantic_sha256: selection.semantic_sha256,
        canonical_genome: selection.canonical_genome,
        train_rows: TRAIN_ROWS,
        test_rows: TEST_ROWS,
        predictions_file: "phase2/predictions.csv".to_string(),
        predictions_sha256,
        fit_seconds,
        predict_seconds,
        test_rmse,
        test_r2,
        test_mae,
        target_rmse: TARGET_RMSE,
        tabfm_ensemble_rmse: TABFM_RMSE,
        archived_jope_rmse: ARCHIVED_JOPE_RMSE,
        signed_margin_vs_target: TARGET_RMSE - test_rmse,
        signed_margin_vs_tabfm: TABFM_RMSE - test_rmse,
        signed_margin_vs_archived_jope: ARCHIVED_JOPE_RMSE - test_rmse,
        beats_target_strict: if test_rmse < TARGET_RMSE { "YES" } else { "NO" }.to_string(),
    };
    let receipt_bytes = canonical_json(&receipt)?;
    immutable_write(&phase2_dir.join("receipt.v1.json"), &receipt_bytes)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn execute_campaign(
    config: &CampaignConfig,
    train: &RegressionData,
    mut population: Vec<PopulationMember>,
    initial_population_sha256: &str,
    artifact_root: &Path,
) -> Result<SelectionFreeze> {
    validate_campaign_config(config, train, &population)?;
    let identity = CampaignIdentity {
        schema: CAMPAIGN_SCHEMA,
        evaluator: EVALUATOR,
        dataset_hash: &config.dataset,
        train_sha256: &config.train_sha256,
        train_rows: train.y.len(),
        generations: config.generations,
        population: config.population,
        cv_folds: config.cv_folds,
        confirm_folds: config.confirm_folds,
        elites: config.elites,
        parent_pool: config.parent_pool,
        tournament: config.tournament,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        seed: config.seed,
        rayon_threads: config.rayon_threads,
        member_execution: "one dedicated OS thread per population member",
        dedicated_member_threads: config.population,
        initial_population_sha256,
        quick_holdout_fold: QUICK_CV5_HOLDOUT_FOLD,
        final_cv7_candidates: config.final_cv7_candidates,
    };
    immutable_write(
        &artifact_root.join("campaign.v1.json"),
        &canonical_json(&identity)?,
    )?;
    publish_initial_population(artifact_root, &population, initial_population_sha256)?;
    let live_logger = LiveLogger::create(
        artifact_root,
        config.generations.saturating_mul(config.population),
        config.population,
        config.rayon_threads,
    )?;
    let cv5_manifest = build_fold_manifest(
        &config.dataset,
        &config.train_sha256,
        train.y.len(),
        config.cv_folds,
    )?;
    let cv5_manifest_bytes = canonical_json(&cv5_manifest)?;
    let cv5_manifest_sha256 = sha256_bytes(&cv5_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv5.v1.json"),
        &cv5_manifest_bytes,
    )?;

    let ledger_path = artifact_root.join("slots.v1.jsonl");
    let mut ledger_file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&ledger_path)
        .with_context(|| format!("create append-only ledger {}", ledger_path.display()))?;
    let campaign_started = Instant::now();
    let mut cache = HashMap::<String, CachedEvaluation>::new();
    let mut terminal_slots = 0usize;
    let mut unique_evaluations = 0usize;
    let mut generation_zero_best = None;
    let mut incumbent_history = Vec::<Incumbent>::new();
    let mut incumbent: Option<Incumbent> = None;
    let mut final_ranked = Vec::<SlotLedger>::new();

    for generation in 0..config.generations {
        let evaluated = evaluate_population(
            generation,
            &population,
            train,
            config,
            artifact_root,
            &cache,
            &live_logger,
        )?;
        let misses = evaluated.iter().filter(|item| item.cache_miss).count();
        unique_evaluations += misses;
        let mut slots = Vec::with_capacity(config.population);
        for (slot, evaluation) in evaluated.into_iter().enumerate() {
            terminal_slots += 1;
            let source = evaluation.cached.clone();
            cache
                .entry(population[slot].semantic_sha256.clone())
                .or_insert_with(|| source.clone());
            let elapsed = campaign_started.elapsed().as_secs_f64();
            let result = source.result.as_ref();
            slots.push(SlotLedger {
                schema: LEDGER_SCHEMA.to_string(),
                generation,
                slot,
                canonical_genome: population[slot].canonical_genome.clone(),
                semantic_sha256: population[slot].semantic_sha256.clone(),
                ancestry: population[slot].ancestry.clone(),
                head_family: source.head_family.clone(),
                complexity: source.complexity,
                valid: result.is_some(),
                rejection_reason: source.rejection_reason.clone(),
                cache_disposition: if evaluation.cache_miss { "miss" } else { "hit" }.to_string(),
                cache_source_generation: (!evaluation.cache_miss)
                    .then_some(source.source_generation),
                cache_source_slot: (!evaluation.cache_miss).then_some(source.source_slot),
                artifacts: source.artifacts.clone(),
                fold_r2: result.map(|value| value.fold_r2.clone()),
                pooled_oof_r2: result.map(|value| value.pooled_oof_r2),
                pooled_oof_rmse: result.map(|value| value.pooled_oof_rmse),
                robust_fitness: result.map(|value| value.robust_fitness),
                fold_fit_seconds: result.map_or_else(Vec::new, |value| value.fit_seconds.clone()),
                fold_predict_seconds: result
                    .map_or_else(Vec::new, |value| value.predict_seconds.clone()),
                cumulative_unique_evaluations: unique_evaluations,
                cumulative_terminal_slots: terminal_slots,
                cumulative_throughput_slots_per_second: if elapsed > 0.0 {
                    terminal_slots as f64 / elapsed
                } else {
                    0.0
                },
            });
        }
        let mut ranked = slots
            .iter()
            .filter(|slot| slot.rankable())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_slots);
        if ranked.len() < config.parent_pool {
            bail!(
                "generation {generation} produced only {} valid genomes; exact tournament selection requires {}",
                ranked.len(),
                config.parent_pool
            );
        }
        let best = ranked.first().unwrap();
        let best_fitness = best.robust_fitness.unwrap();
        if generation == 0 {
            generation_zero_best = Some(best_fitness);
        }
        if incumbent
            .as_ref()
            .is_none_or(|prior| best_fitness > prior.robust_fitness)
        {
            incumbent = Some(Incumbent {
                generation,
                semantic_sha256: best.semantic_sha256.clone(),
                robust_fitness: best_fitness,
            });
        }
        incumbent_history.push(incumbent.clone().unwrap());

        let next_population = if generation + 1 < config.generations {
            refill_population(config, generation + 1, &ranked)?
        } else {
            Vec::new()
        };
        for slot in &slots {
            ledger_file.write_all(&canonical_json(slot)?)?;
        }
        ledger_file.flush()?;
        ledger_file.sync_all()?;
        sync_directory(artifact_root)?;
        let ledger_sha256 = sha256_file(&ledger_path)?;
        let checkpoint = Checkpoint {
            schema: CHECKPOINT_SCHEMA,
            generation,
            terminal_slots,
            unique_evaluations,
            ledger_sha256: &ledger_sha256,
            next_population: &next_population,
        };
        immutable_write(
            &artifact_root
                .join("checkpoints")
                .join(format!("generation-{generation:03}.v1.json")),
            &canonical_json(&checkpoint)?,
        )?;
        let summary = generation_summary(
            generation,
            &slots,
            generation_zero_best.unwrap(),
            &incumbent_history,
            &next_population,
            terminal_slots,
            unique_evaluations,
            &ledger_sha256,
        )?;
        immutable_write(
            &artifact_root
                .join("summaries")
                .join(format!("generation-{generation:03}.v1.json")),
            &canonical_json(&summary)?,
        )?;
        eprintln!(
            "[jain-exact-cv] generation={generation} valid={} cache_hits={} best={best_fitness:.12} ledger={ledger_sha256}",
            ranked.len(),
            slots.iter().filter(|slot| slot.cache_disposition == "hit").count(),
        );
        final_ranked = ranked;
        population = next_population;
    }
    drop(ledger_file);

    if final_ranked.len() < config.final_cv7_candidates {
        bail!(
            "final generation has only {} quick-score candidates; full CV7 requires {}",
            final_ranked.len(),
            config.final_cv7_candidates
        );
    }
    let cv7_manifest = build_fold_manifest(
        &config.dataset,
        &config.train_sha256,
        train.y.len(),
        config.confirm_folds,
    )?;
    let cv7_manifest_bytes = canonical_json(&cv7_manifest)?;
    let cv7_manifest_sha256 = sha256_bytes(&cv7_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv7.v1.json"),
        &cv7_manifest_bytes,
    )?;
    let cv7_fold_ids = (0..config.confirm_folds).collect::<Vec<_>>();
    let cv7_finalists = thread::scope(|scope| -> Result<Vec<FinalistCv7>> {
        let mut handles = Vec::with_capacity(config.final_cv7_candidates);
        for (quick_rank, slot) in final_ranked
            .iter()
            .take(config.final_cv7_candidates)
            .enumerate()
        {
            let thread_name = format!("jain-cv7-member-{quick_rank:02}");
            let cv5_manifest_sha256 = &cv5_manifest_sha256;
            let cv7_manifest_sha256 = &cv7_manifest_sha256;
            let cv7_fold_ids = &cv7_fold_ids;
            let live_logger = &live_logger;
            handles.push(
                thread::Builder::new()
                    .name(thread_name)
                    .spawn_scoped(scope, move || -> Result<FinalistCv7> {
                        let quick_cv5_fold0 = selection_evidence_from_slot(
                            slot,
                            config.cv_folds,
                            vec![QUICK_CV5_HOLDOUT_FOLD],
                            &cv5_manifest_sha256,
                        )?;
                        let evaluated =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let genome = GenomeIr::parse(&slot.canonical_genome)?;
                                exact_cv(&genome, train, config.confirm_folds, &config.dataset)
                            }));
                        let (cv7, rejection_reason) = match evaluated {
                            Ok(Ok(result)) => (
                                Some(selection_evidence_from_result(
                                    result,
                                    config.confirm_folds,
                                    cv7_fold_ids.clone(),
                                    &cv7_manifest_sha256,
                                )),
                                None,
                            ),
                            Ok(Err(error)) => (None, Some(format!("cv7: {error:#}"))),
                            Err(payload) => (
                                None,
                                Some(format!("cv7 panic: {}", panic_message(payload.as_ref()))),
                            ),
                        };
                        let finalist = FinalistCv7 {
                            quick_rank,
                            generation: config.generations - 1,
                            slot: slot.slot,
                            canonical_genome: slot.canonical_genome.clone(),
                            semantic_sha256: slot.semantic_sha256.clone(),
                            complexity: slot.complexity,
                            quick_cv5_fold0,
                            cv7,
                            rejection_reason,
                        };
                        live_logger.record_cv7_finalist(&finalist)?;
                        Ok(finalist)
                    })
                    .with_context(|| {
                        format!("spawn dedicated CV7 thread for quick rank {quick_rank}")
                    })?,
            );
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|payload| {
                    anyhow::anyhow!(
                        "dedicated CV7 member thread panicked: {}",
                        panic_message(payload.as_ref())
                    )
                })?
            })
            .collect()
    })?;
    for finalist in &cv7_finalists {
        immutable_write(
            &artifact_root
                .join("finalists")
                .join(format!("quick-rank-{:02}.cv7.v1.json", finalist.quick_rank)),
            &canonical_json(finalist)?,
        )?;
    }
    let mut cv7_ranked = cv7_finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    cv7_ranked.sort_by(rank_finalists);
    let selected = cv7_ranked
        .first()
        .copied()
        .context("none of the top quick-score finalists completed full CV7")?;
    let ledger_sha256 = sha256_file(&ledger_path)?;
    let selection = SelectionFreeze {
        schema: SELECTION_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        dataset_hash: config.dataset.clone(),
        train_sha256: config.train_sha256.clone(),
        initial_population_sha256: initial_population_sha256.to_string(),
        generation: config.generations - 1,
        slot: selected.slot,
        canonical_genome: selected.canonical_genome.clone(),
        semantic_sha256: selected.semantic_sha256.clone(),
        complexity: selected.complexity,
        ranking_authority:
            "top 10 by quick CV5 fold-0 robust fitness; final selection by full CV7 robust fitness descending, complexity ascending, semantic_sha256 ascending"
                .to_string(),
        ledger_sha256,
        quick_cv5_fold0: selected.quick_cv5_fold0.clone(),
        cv7: selected.cv7.clone().unwrap(),
        cv7_finalists,
    };
    publish_selection_freeze(artifact_root, &selection)?;
    live_logger.record_campaign_complete(&selection)?;
    make_readonly(&ledger_path)?;
    live_logger.seal()?;
    Ok(selection)
}

struct PopulationEvaluation {
    cache_miss: bool,
    cached: CachedEvaluation,
}

fn evaluate_population(
    generation: usize,
    population: &[PopulationMember],
    train: &RegressionData,
    config: &CampaignConfig,
    artifact_root: &Path,
    cache: &HashMap<String, CachedEvaluation>,
    live_logger: &LiveLogger,
) -> Result<Vec<PopulationEvaluation>> {
    if config.rayon_threads < population.len() {
        bail!(
            "resolved runtime capacity {} cannot provide one thread for each of {} population members",
            config.rayon_threads,
            population.len()
        );
    }
    thread::scope(|scope| -> Result<Vec<PopulationEvaluation>> {
        let mut handles = Vec::with_capacity(population.len());
        for (slot, member) in population.iter().enumerate() {
            let thread_name = format!("jain-member-g{generation:02}-s{slot:02}");
            handles.push(
                thread::Builder::new()
                    .name(thread_name)
                    .spawn_scoped(scope, move || -> Result<PopulationEvaluation> {
                        if let Some(cached) = cache.get(&member.semantic_sha256) {
                            let evaluation = PopulationEvaluation {
                                cache_miss: false,
                                cached: cached.clone(),
                            };
                            live_logger.record_slot(
                                generation,
                                slot,
                                member,
                                &evaluation.cached,
                                false,
                            )?;
                            return Ok(evaluation);
                        }
                        let cached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            evaluate_unique(generation, slot, member, train, config, artifact_root)
                        }))
                        .unwrap_or_else(|payload| CachedEvaluation {
                            source_generation: generation,
                            source_slot: slot,
                            complexity: genome_complexity_text(&member.canonical_genome),
                            head_family: head_family_text(&member.canonical_genome),
                            artifacts: None,
                            result: None,
                            rejection_reason: Some(format!(
                                "panic: {}",
                                panic_message(payload.as_ref())
                            )),
                        });
                        let evaluation = PopulationEvaluation {
                            cache_miss: true,
                            cached,
                        };
                        live_logger.record_slot(
                            generation,
                            slot,
                            member,
                            &evaluation.cached,
                            true,
                        )?;
                        Ok(evaluation)
                    })
                    .with_context(|| {
                        format!(
                            "spawn dedicated member thread for generation {generation} slot {slot}"
                        )
                    })?,
            );
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|payload| {
                    anyhow::anyhow!(
                        "dedicated population member thread panicked: {}",
                        panic_message(payload.as_ref())
                    )
                })?
            })
            .collect()
    })
}

fn evaluate_unique(
    generation: usize,
    slot: usize,
    member: &PopulationMember,
    train: &RegressionData,
    config: &CampaignConfig,
    artifact_root: &Path,
) -> CachedEvaluation {
    let rejected = |complexity, head_family, artifacts, reason: String| CachedEvaluation {
        source_generation: generation,
        source_slot: slot,
        complexity,
        head_family,
        artifacts,
        result: None,
        rejection_reason: Some(reason),
    };
    let genome = match GenomeIr::parse(&member.canonical_genome) {
        Ok(genome) if matches!(genome, GenomeIr::Pipeline(_)) => genome,
        Ok(_) => {
            return rejected(
                0,
                "non_pipeline".to_string(),
                None,
                "not a pipeline genome".to_string(),
            )
        }
        Err(error) => {
            return rejected(
                0,
                "parse_error".to_string(),
                None,
                format!("parse: {error:#}"),
            )
        }
    };
    let canonical = genome.canonical();
    let complexity = genome_complexity(&genome);
    let head_family = head_family(&genome);
    if canonical != member.canonical_genome
        || sha256_bytes(canonical.as_bytes()) != member.semantic_sha256
    {
        return rejected(
            complexity,
            head_family,
            None,
            "canonical or semantic identity mismatch".to_string(),
        );
    }
    let emitted = match emit(&genome, &Default::default()) {
        Ok(emitted) => emitted,
        Err(error) => return rejected(complexity, head_family, None, format!("emit: {error:#}")),
    };
    let artifacts = match publish_emitted(artifact_root, &member.semantic_sha256, &emitted) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            return rejected(
                complexity,
                head_family,
                None,
                format!("publish emitted artifacts: {error:#}"),
            )
        }
    };
    match exact_cv_on_folds(
        &genome,
        train,
        config.cv_folds,
        &[QUICK_CV5_HOLDOUT_FOLD],
        &config.dataset,
    ) {
        Ok(result) => CachedEvaluation {
            source_generation: generation,
            source_slot: slot,
            complexity,
            head_family,
            artifacts: Some(artifacts),
            result: Some(result),
            rejection_reason: None,
        },
        Err(error) => rejected(
            complexity,
            head_family,
            Some(artifacts),
            format!("cv: {error:#}"),
        ),
    }
}

fn exact_cv(
    genome: &GenomeIr,
    train: &RegressionData,
    folds: usize,
    dataset: &str,
) -> Result<ExactCvResult> {
    let evaluated_folds = (0..folds).collect::<Vec<_>>();
    exact_cv_on_folds(genome, train, folds, &evaluated_folds, dataset)
}

fn exact_cv_on_folds(
    genome: &GenomeIr,
    train: &RegressionData,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    dataset: &str,
) -> Result<ExactCvResult> {
    let n = train.x.len().min(train.y.len());
    let mut oof = vec![f64::NAN; n];
    let mut fold_r2 = Vec::with_capacity(evaluated_folds.len());
    let mut fit_seconds = Vec::with_capacity(evaluated_folds.len());
    let mut predict_seconds = Vec::with_capacity(evaluated_folds.len());
    if evaluated_folds.is_empty() || evaluated_folds.iter().any(|fold| *fold >= fold_plan_folds) {
        bail!("evaluated folds are empty or outside the fold plan");
    }
    for &fold in evaluated_folds {
        let mut train_x: Rows = Vec::new();
        let mut train_y = Vec::new();
        let mut valid_x: Rows = Vec::new();
        let mut valid_y = Vec::new();
        let mut valid_indices = Vec::new();
        for row in 0..n {
            if fold_of(dataset, row, fold_plan_folds) == fold {
                valid_x.push(train.x[row].clone());
                valid_y.push(train.y[row]);
                valid_indices.push(row);
            } else {
                train_x.push(train.x[row].clone());
                train_y.push(train.y[row]);
            }
        }
        if valid_x.is_empty() || train_x.len() < 8 {
            bail!("fold {fold} is empty or has fewer than eight training rows");
        }
        let fit_started = Instant::now();
        let model = fit(genome, &train_x, &train_y).with_context(|| format!("fit fold {fold}"))?;
        fit_seconds.push(fit_started.elapsed().as_secs_f64());
        let predict_started = Instant::now();
        let predictions = model.predict(&valid_x);
        predict_seconds.push(predict_started.elapsed().as_secs_f64());
        if predictions.len() != valid_y.len() || predictions.iter().any(|v| !v.is_finite()) {
            bail!("fold {fold} produced non-finite or wrong-length predictions");
        }
        let score = independent_metrics(&valid_y, &predictions)?.1;
        if !score.is_finite() {
            bail!("fold {fold} R2 is non-finite");
        }
        fold_r2.push(score);
        for (index, prediction) in valid_indices.into_iter().zip(predictions) {
            oof[index] = prediction;
        }
    }
    if fold_r2.len() != evaluated_folds.len() {
        bail!("CV did not evaluate every requested holdout fold");
    }
    let mut oof_truth = Vec::new();
    let mut oof_predictions = Vec::new();
    for (row, prediction) in oof.into_iter().enumerate() {
        if prediction.is_finite() {
            oof_truth.push(train.y[row]);
            oof_predictions.push(prediction);
        }
    }
    if oof_truth.is_empty() {
        bail!("CV produced no held-out predictions");
    }
    let (pooled_oof_rmse, pooled_oof_r2, _) = independent_metrics(&oof_truth, &oof_predictions)?;
    let robust_fitness = robust_fitness(pooled_oof_r2, &fold_r2);
    if !pooled_oof_r2.is_finite() || !robust_fitness.is_finite() {
        bail!("pooled OOF or robust fitness is non-finite");
    }
    Ok(ExactCvResult {
        fold_r2,
        pooled_oof_r2,
        pooled_oof_rmse,
        robust_fitness,
        fit_seconds,
        predict_seconds,
    })
}

fn refill_population(
    config: &CampaignConfig,
    generation: usize,
    ranked: &[SlotLedger],
) -> Result<Vec<PopulationMember>> {
    if ranked.len() < config.parent_pool || config.elites >= config.population {
        bail!("cannot refill exact population from configured parent pool/elites");
    }
    let mut next = Vec::with_capacity(config.population);
    let mut seen = HashSet::with_capacity(config.population);
    for elite in ranked.iter().take(config.elites) {
        seen.insert(elite.semantic_sha256.clone());
        next.push(PopulationMember {
            canonical_genome: elite.canonical_genome.clone(),
            semantic_sha256: elite.semantic_sha256.clone(),
            ancestry: Ancestry {
                operation: "elite".to_string(),
                parents: vec![elite.semantic_sha256.clone()],
                retry_count: 0,
            },
        });
    }
    for slot in config.elites..config.population {
        let mut retry = 0usize;
        loop {
            if retry >= 1_000_000 {
                bail!("could not generate a canonical-unique child for generation {generation} slot {slot}");
            }
            let mut rng =
                DeterministicRng::new(offspring_seed(config.seed, generation, slot, retry));
            let crossover = rng.next_u64() % 4 == 0;
            let parent_a =
                tournament_winner(&mut rng, ranked, config.parent_pool, config.tournament);
            let (operation, parents, child) = if crossover {
                let parent_b =
                    tournament_winner(&mut rng, ranked, config.parent_pool, config.tournament);
                let result = crossover_canonical_pipelines(
                    &parent_a.canonical_genome,
                    &parent_b.canonical_genome,
                    rng.next_u64(),
                    config.max_transforms,
                )
                .map_err(anyhow::Error::msg)?;
                (
                    "crossover".to_string(),
                    vec![
                        parent_a.semantic_sha256.clone(),
                        parent_b.semantic_sha256.clone(),
                    ],
                    result,
                )
            } else {
                let result = mutate_canonical_pipeline(
                    &parent_a.canonical_genome,
                    rng.next_u64(),
                    config.max_transforms,
                )
                .map_err(anyhow::Error::msg)?;
                (
                    "mutation".to_string(),
                    vec![parent_a.semantic_sha256.clone()],
                    result,
                )
            };
            let parsed = GenomeIr::parse(&child).context("parse generated exact-CV child")?;
            if !matches!(parsed, GenomeIr::Pipeline(_)) {
                retry += 1;
                continue;
            }
            let canonical = parsed.canonical();
            let semantic_sha256 = sha256_bytes(canonical.as_bytes());
            if !seen.insert(semantic_sha256.clone()) {
                retry += 1;
                continue;
            }
            next.push(PopulationMember {
                canonical_genome: canonical,
                semantic_sha256,
                ancestry: Ancestry {
                    operation,
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
    }
    validate_unique_population(&next, config.population)?;
    Ok(next)
}

fn tournament_winner<'a>(
    rng: &mut DeterministicRng,
    ranked: &'a [SlotLedger],
    parent_pool: usize,
    tournament: usize,
) -> &'a SlotLedger {
    let mut winner = (rng.next_u64() as usize) % parent_pool;
    for _ in 1..tournament {
        winner = winner.min((rng.next_u64() as usize) % parent_pool);
    }
    &ranked[winner]
}

#[derive(Clone, Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn offspring_seed(base: u64, generation: usize, slot: usize, retry: usize) -> u64 {
    base ^ (generation as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (slot as u64).wrapping_mul(0xa076_1d64_78bd_642f)
        ^ (retry as u64).wrapping_mul(0xe703_7ed1_a0b4_28db)
}

fn rank_slots(left: &SlotLedger, right: &SlotLedger) -> Ordering {
    right
        .robust_fitness
        .unwrap()
        .total_cmp(&left.robust_fitness.unwrap())
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

fn rank_finalists(left: &&FinalistCv7, right: &&FinalistCv7) -> Ordering {
    let left_score = left.cv7.as_ref().unwrap().robust_fitness;
    let right_score = right.cv7.as_ref().unwrap().robust_fitness;
    right_score
        .total_cmp(&left_score)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

fn selection_evidence_from_slot(
    slot: &SlotLedger,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> Result<SelectionEvidence> {
    if !slot.rankable() || slot.fold_r2.as_ref().unwrap().len() != evaluated_fold_ids.len() {
        bail!("quick-score finalist lacks complete held-out evidence");
    }
    Ok(SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: slot.fold_r2.clone().unwrap(),
        pooled_oof_r2: slot.pooled_oof_r2.unwrap(),
        pooled_oof_rmse: slot.pooled_oof_rmse.unwrap(),
        robust_fitness: slot.robust_fitness.unwrap(),
        fit_seconds: slot.fold_fit_seconds.clone(),
        predict_seconds: slot.fold_predict_seconds.clone(),
    })
}

fn selection_evidence_from_result(
    result: ExactCvResult,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> SelectionEvidence {
    SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: result.fold_r2,
        pooled_oof_r2: result.pooled_oof_r2,
        pooled_oof_rmse: result.pooled_oof_rmse,
        robust_fitness: result.robust_fitness,
        fit_seconds: result.fit_seconds,
        predict_seconds: result.predict_seconds,
    }
}

fn generation_summary<'a>(
    generation: usize,
    slots: &[SlotLedger],
    generation_zero_best: f64,
    incumbent_history: &'a [Incumbent],
    next_population: &'a [PopulationMember],
    terminal_slots: usize,
    unique_evaluations: usize,
    ledger_sha256: &'a str,
) -> Result<GenerationSummary<'a>> {
    let mut fitness = slots
        .iter()
        .filter_map(|slot| slot.rankable().then_some(slot.robust_fitness.unwrap()))
        .collect::<Vec<_>>();
    fitness.sort_by(f64::total_cmp);
    let best = fitness
        .last()
        .copied()
        .context("generation has no valid fitness")?;
    let incumbent = slots
        .iter()
        .filter(|slot| slot.rankable())
        .min_by(|left, right| rank_slots(left, right))
        .context("generation has no ranked incumbent")?;
    let incumbent_pooled_oof_rmse = incumbent.pooled_oof_rmse.unwrap();
    let median = quantile(&fitness, 0.5);
    let mut fitness_quantiles = BTreeMap::new();
    for (name, q) in [
        ("p00", 0.0),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p100", 1.0),
    ] {
        fitness_quantiles.insert(name.to_string(), quantile(&fitness, q));
    }
    let mut head_mix = BTreeMap::new();
    for slot in slots {
        *head_mix.entry(slot.head_family.clone()).or_insert(0) += 1;
    }
    let diversity = slots
        .iter()
        .map(|slot| slot.semantic_sha256.as_str())
        .collect::<HashSet<_>>()
        .len() as f64
        / slots.len() as f64;
    Ok(GenerationSummary {
        schema: SUMMARY_SCHEMA,
        generation,
        population: slots.len(),
        valid: fitness.len(),
        invalid: slots.len() - fitness.len(),
        cache_hits: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "hit")
            .count(),
        cache_misses: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "miss")
            .count(),
        fitness_quantiles,
        best_fitness: best,
        incumbent_pooled_oof_rmse,
        cv_rmse_signed_margin_vs_tabfm: TABFM_RMSE - incumbent_pooled_oof_rmse,
        cv_proxy_track_status: if incumbent_pooled_oof_rmse < TABFM_RMSE {
            "CV_PROXY_AHEAD"
        } else {
            "CV_PROXY_BEHIND"
        }
        .to_string(),
        median_fitness: median,
        improvement_from_generation_zero: best - generation_zero_best,
        canonical_diversity: diversity,
        head_mix,
        incumbent_history,
        next_population,
        cumulative_terminal_slots: terminal_slots,
        cumulative_unique_evaluations: unique_evaluations,
        cumulative_ledger_sha256: ledger_sha256,
    })
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = q * (sorted.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
}

fn parse_initial_population(bytes: &[u8], expected: usize) -> Result<Vec<PopulationMember>> {
    let text = std::str::from_utf8(bytes).context("generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body)
            .with_context(|| format!("generation-zero line {} does not parse", line_number + 1))?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!("generation-zero line {} is not a pipeline", line_number + 1);
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!("generation-zero line {} is not canonical", line_number + 1);
        }
        population.push(PopulationMember {
            semantic_sha256: sha256_bytes(canonical.as_bytes()),
            canonical_genome: canonical,
            ancestry: Ancestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    validate_unique_population(&population, expected)?;
    Ok(population)
}

fn validate_unique_population(population: &[PopulationMember], expected: usize) -> Result<()> {
    if population.len() != expected {
        bail!(
            "exact population requires {expected} slots, found {}",
            population.len()
        );
    }
    let unique = population
        .iter()
        .map(|member| member.semantic_sha256.as_str())
        .collect::<HashSet<_>>();
    if unique.len() != expected {
        bail!("exact population is not canonical-unique");
    }
    Ok(())
}

fn validate_campaign_config(
    config: &CampaignConfig,
    train: &RegressionData,
    population: &[PopulationMember],
) -> Result<()> {
    if config.generations == 0
        || config.cv_folds < 2
        || config.confirm_folds < 2
        || config.elites == 0
        || config.parent_pool > config.population
        || config.tournament != TOURNAMENT
        || config.final_cv7_candidates == 0
        || config.final_cv7_candidates > config.parent_pool
        || config.rayon_threads < config.population
    {
        bail!("invalid exact-CV campaign configuration");
    }
    if config
        .expected_train_rows
        .is_some_and(|rows| train.y.len() != rows)
    {
        bail!("training row count differs from campaign configuration");
    }
    validate_unique_population(population, config.population)
}

fn validate_generation_zero_provenance(
    bytes: &[u8],
    population_sha256: &str,
    population: usize,
) -> Result<serde_json::Value> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("parse generation-zero provenance JSON")?;
    if value.get("schema").and_then(serde_json::Value::as_str)
        != Some("JainExactCvGenerationZeroProvenance.v1")
        || value.get("dataset").and_then(serde_json::Value::as_str) != Some(DATASET)
        || value
            .get("ordered_population_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(population_sha256)
        || value.get("population").and_then(serde_json::Value::as_u64) != Some(population as u64)
    {
        bail!("generation-zero provenance does not bind the requested dataset and population");
    }
    for section in ["report_finalists", "archived_seeds"] {
        let source = value
            .get(section)
            .with_context(|| format!("generation-zero provenance lacks {section}"))?;
        let path = source
            .get("source_path")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("generation-zero provenance {section} lacks source_path"))?;
        let expected_sha256 = source
            .get("source_sha256")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("generation-zero provenance {section} lacks source_sha256"))?;
        validate_sha256("provenance source_sha256", expected_sha256)?;
        validate_file_hash(Path::new(path), expected_sha256, section)?;
    }
    Ok(value)
}

fn publish_initial_population(
    artifact_root: &Path,
    population: &[PopulationMember],
    population_sha256: &str,
) -> Result<()> {
    let manifest = serde_json::json!({
        "schema": "JainExactCvInitialPopulation.v1",
        "ordered_file_sha256": population_sha256,
        "slots": population.iter().enumerate().map(|(slot, member)| serde_json::json!({
            "slot": slot,
            "semantic_sha256": member.semantic_sha256,
            "canonical_genome": member.canonical_genome,
        })).collect::<Vec<_>>(),
    });
    immutable_write(
        &artifact_root.join("initial-population-manifest.v1.json"),
        &canonical_json(&manifest)?,
    )
}

fn build_fold_manifest(
    dataset: &str,
    train_sha256: &str,
    rows: usize,
    folds: usize,
) -> Result<FoldManifest> {
    let mut assignments = Vec::with_capacity(folds);
    for fold in 0..folds {
        let mut train_rows = Vec::new();
        let mut validation_rows = Vec::new();
        for row in 0..rows {
            if fold_of(dataset, row, folds) == fold {
                validation_rows.push(row);
            } else {
                train_rows.push(row);
            }
        }
        if train_rows.is_empty() || validation_rows.is_empty() {
            bail!("fold manifest contains an empty split");
        }
        assignments.push(FoldAssignment {
            fold,
            train_rows_sha256: sha256_bytes(&canonical_json(&train_rows)?),
            validation_rows_sha256: sha256_bytes(&canonical_json(&validation_rows)?),
            train_rows,
            validation_rows,
        });
    }
    Ok(FoldManifest {
        schema: FOLD_MANIFEST_SCHEMA.to_string(),
        dataset_hash: dataset.to_string(),
        dataset_train_sha256: train_sha256.to_string(),
        fold_authority: FOLD_AUTHORITY.to_string(),
        folds,
        rows,
        assignments,
    })
}

fn publish_emitted(root: &Path, semantic: &str, emitted: &Emitted) -> Result<ArtifactHashes> {
    let directory = root.join("artifacts").join(semantic);
    let python_sha256 = sha256_bytes(emitted.python.as_bytes());
    let rust_sha256 = sha256_bytes(emitted.rust.as_bytes());
    let manifest_sha256 = sha256_bytes(emitted.manifest.as_bytes());
    immutable_write(
        &directory.join(format!("{python_sha256}.py")),
        emitted.python.as_bytes(),
    )?;
    immutable_write(
        &directory.join(format!("{rust_sha256}.rs")),
        emitted.rust.as_bytes(),
    )?;
    immutable_write(
        &directory.join(format!("{manifest_sha256}.manifest.json")),
        emitted.manifest.as_bytes(),
    )?;
    Ok(ArtifactHashes {
        python_sha256,
        rust_sha256,
        manifest_sha256,
    })
}

fn publish_selection_freeze(root: &Path, selection: &SelectionFreeze) -> Result<()> {
    let selection_file = "selection-freeze.v1.json";
    let bytes = canonical_json(selection)?;
    let selection_sha256 = sha256_bytes(&bytes);
    immutable_write(&root.join(selection_file), &bytes)?;
    let complete = SelectionComplete {
        schema: SELECTION_COMPLETE_SCHEMA.to_string(),
        status: "COMPLETE".to_string(),
        selection_file: selection_file.to_string(),
        selection_sha256,
    };
    immutable_write(
        &root.join("SELECTION_COMPLETE"),
        &canonical_json(&complete)?,
    )?;
    sync_directory(root)
}

fn validate_selection_freeze(root: &Path) -> Result<(SelectionFreeze, String)> {
    let marker_path = root.join("SELECTION_COMPLETE");
    let marker: SelectionComplete =
        serde_json::from_slice(&fs::read(&marker_path).with_context(|| {
            format!("read immutable selection marker {}", marker_path.display())
        })?)
        .context("parse selection completion marker")?;
    if marker.schema != SELECTION_COMPLETE_SCHEMA
        || marker.status != "COMPLETE"
        || marker.selection_file != "selection-freeze.v1.json"
    {
        bail!("selection completion marker is invalid");
    }
    validate_sha256("selection_sha256", &marker.selection_sha256)?;
    let selection_path = root.join(&marker.selection_file);
    let bytes = fs::read(&selection_path)
        .with_context(|| format!("read immutable selection {}", selection_path.display()))?;
    if sha256_bytes(&bytes) != marker.selection_sha256 {
        bail!("selection freeze differs from its completion hash");
    }
    let selection: SelectionFreeze =
        serde_json::from_slice(&bytes).context("parse selection freeze")?;
    if selection.schema != SELECTION_SCHEMA
        || selection.evaluator != EVALUATOR
        || selection.quick_cv5_fold0.fold_plan_folds < 2
        || selection.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
        || selection.cv7.fold_plan_folds < 2
        || selection.cv7.evaluated_fold_ids
            != (0..selection.cv7.fold_plan_folds).collect::<Vec<_>>()
        || selection.cv7_finalists.is_empty()
        || selection.canonical_genome.trim() != selection.canonical_genome
        || sha256_bytes(selection.canonical_genome.as_bytes()) != selection.semantic_sha256
    {
        bail!("selection freeze structural validation failed");
    }
    let mut evidence = vec![&selection.quick_cv5_fold0, &selection.cv7];
    let mut finalist_semantics = HashSet::new();
    for (expected_rank, finalist) in selection.cv7_finalists.iter().enumerate() {
        if finalist.quick_rank != expected_rank
            || finalist.generation != selection.generation
            || GenomeIr::parse(&finalist.canonical_genome)
                .map(|genome| genome.canonical())
                .ok()
                .as_deref()
                != Some(finalist.canonical_genome.as_str())
            || sha256_bytes(finalist.canonical_genome.as_bytes()) != finalist.semantic_sha256
            || !finalist_semantics.insert(finalist.semantic_sha256.as_str())
            || finalist.quick_cv5_fold0.fold_plan_folds != selection.quick_cv5_fold0.fold_plan_folds
            || finalist.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
            || finalist.quick_cv5_fold0.fold_manifest_sha256
                != selection.quick_cv5_fold0.fold_manifest_sha256
            || finalist.cv7.is_some() == finalist.rejection_reason.is_some()
        {
            bail!("selection freeze contains an invalid CV7 finalist row");
        }
        if finalist.cv7.as_ref().is_some_and(|cv7| {
            cv7.fold_plan_folds != selection.cv7.fold_plan_folds
                || cv7.evaluated_fold_ids != (0..cv7.fold_plan_folds).collect::<Vec<_>>()
                || cv7.fold_manifest_sha256 != selection.cv7.fold_manifest_sha256
        }) {
            bail!("selection freeze contains inconsistent full-CV7 evidence");
        }
        evidence.push(&finalist.quick_cv5_fold0);
        if let Some(cv7) = finalist.cv7.as_ref() {
            evidence.push(cv7);
        }
    }
    for evidence in evidence {
        validate_sha256("fold_manifest_sha256", &evidence.fold_manifest_sha256)?;
        if evidence.fold_r2.len() != evidence.evaluated_fold_ids.len()
            || evidence.evaluated_fold_ids.is_empty()
            || evidence
                .evaluated_fold_ids
                .iter()
                .any(|fold| *fold >= evidence.fold_plan_folds)
            || evidence.fold_r2.iter().any(|v| !v.is_finite())
            || !evidence.pooled_oof_r2.is_finite()
            || !evidence.pooled_oof_rmse.is_finite()
            || !evidence.robust_fitness.is_finite()
            || (robust_fitness(evidence.pooled_oof_r2, &evidence.fold_r2) - evidence.robust_fitness)
                .abs()
                > 1e-12
        {
            bail!("selection freeze contains invalid CV evidence");
        }
    }
    let selected_finalist = selection
        .cv7_finalists
        .iter()
        .find(|finalist| finalist.semantic_sha256 == selection.semantic_sha256)
        .context("selected genome is absent from the CV7 finalist set")?;
    if selected_finalist.slot != selection.slot
        || selected_finalist.canonical_genome != selection.canonical_genome
        || selected_finalist.cv7.is_none()
        || canonical_json(selected_finalist.cv7.as_ref().unwrap())?
            != canonical_json(&selection.cv7)?
    {
        bail!("selected CV7 finalist evidence does not match the freeze");
    }
    let mut ranked_finalists = selection
        .cv7_finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    ranked_finalists.sort_by(rank_finalists);
    if ranked_finalists
        .first()
        .is_none_or(|winner| winner.semantic_sha256 != selection.semantic_sha256)
    {
        bail!("selection freeze did not choose the deterministic CV7 winner");
    }
    let cv5_path = root.join("folds/cv5.v1.json");
    let cv7_path = root.join("folds/cv7.v1.json");
    if sha256_file(&cv5_path)? != selection.quick_cv5_fold0.fold_manifest_sha256
        || sha256_file(&cv7_path)? != selection.cv7.fold_manifest_sha256
    {
        bail!("selection freeze fold manifest hashes do not match");
    }
    if sha256_file(&root.join("slots.v1.jsonl"))? != selection.ledger_sha256 {
        bail!("selection freeze ledger hash does not match");
    }
    Ok((selection, marker.selection_sha256))
}

fn genome_complexity(genome: &GenomeIr) -> usize {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.len(),
        _ => 0,
    }
}

fn genome_complexity_text(text: &str) -> usize {
    GenomeIr::parse(text).map_or(0, |genome| genome_complexity(&genome))
}

fn head_family(genome: &GenomeIr) -> String {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline
            .stages
            .last()
            .map_or_else(|| "ridge(default)".to_string(), |stage| stage.op.clone()),
        _ => "non_pipeline".to_string(),
    }
}

fn head_family_text(text: &str) -> String {
    GenomeIr::parse(text).map_or_else(|_| "parse_error".to_string(), |genome| head_family(&genome))
}

fn independent_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
    if truth.is_empty()
        || truth.len() != predictions.len()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("metrics require equal, non-empty, finite inputs");
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let mut squared_error = 0.0;
    let mut absolute_error = 0.0;
    let mut total_sum_squares = 0.0;
    for (&actual, &predicted) in truth.iter().zip(predictions) {
        squared_error += (actual - predicted).powi(2);
        absolute_error += (actual - predicted).abs();
        total_sum_squares += (actual - mean).powi(2);
    }
    let rmse = (squared_error / n).sqrt();
    let r2 = if total_sum_squares > 1e-24 {
        1.0 - squared_error / total_sum_squares
    } else {
        0.0
    };
    let mae = absolute_error / n;
    if !rmse.is_finite() || !r2.is_finite() || !mae.is_finite() {
        bail!("metrics are non-finite");
    }
    Ok((rmse, r2, mae))
}

fn write_predictions_atomic(path: &Path, predictions: &[f64]) -> Result<()> {
    let parent = path.parent().context("prediction path has no parent")?;
    fs::create_dir_all(parent)?;
    let staged = parent.join(format!(".predictions.{}.stage", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    for prediction in predictions {
        writeln!(file, "{prediction:.17}")?;
    }
    file.flush()?;
    file.sync_all()?;
    if path.exists() {
        bail!("immutable prediction target already exists");
    }
    fs::rename(&staged, path)?;
    sync_directory(parent)
}

fn read_predictions(path: &Path) -> Result<Vec<f64>> {
    let file = File::open(path)?;
    let mut predictions = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let value = line?
            .trim()
            .parse::<f64>()
            .with_context(|| format!("prediction line {} is invalid", line_number + 1))?;
        predictions.push(value);
    }
    Ok(predictions)
}

fn require_fresh_artifact_root(path: &Path) -> Result<()> {
    if path.exists() {
        let mut entries = fs::read_dir(path)
            .with_context(|| format!("inspect exact-CV root {}", path.display()))?;
        if entries.next().transpose()?.is_some() {
            bail!("--exact-cv-dir must be absent or empty; campaign outputs are immutable");
        }
    }
    Ok(())
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        if fs::read(path)? == bytes {
            make_readonly(path)?;
            return Ok(());
        }
        bail!("immutable artifact collision at {}", path.display());
    }
    let staged = parent.join(format!(
        ".immutable-{}-{}.stage",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    match fs::hard_link(&staged, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path)? != bytes {
                let _ = fs::remove_file(&staged);
                bail!("immutable artifact collision at {}", path.display());
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&staged);
            return Err(error).context("publish immutable artifact");
        }
    }
    fs::remove_file(&staged)?;
    make_readonly(path)?;
    sync_directory(parent)
}

fn make_readonly(path: &Path) -> Result<()> {
    let mut permissions = fs::metadata(path)
        .with_context(|| format!("inspect immutable artifact {}", path.display()))?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("seal immutable artifact {}", path.display()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn validate_file_hash(path: &Path, expected: &str, label: &str) -> Result<()> {
    if sha256_file(path).with_context(|| format!("hash pinned {label}"))? != expected {
        bail!("{label} differs from pinned SHA-256 {expected}");
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "prime-jain-exact-cv-{name}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn toy_data(rows: usize) -> RegressionData {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for row in 0..rows {
            let a = row as f64 / rows as f64;
            let b = ((row * 17) % rows) as f64 / rows as f64;
            x.push(vec![a, b, a * b]);
            y.push(2.0 * a - 0.7 * b + 0.1 * a * b);
        }
        RegressionData { x, y, features: 3 }
    }

    fn fixture_population(size: usize) -> Vec<PopulationMember> {
        (0..size)
            .map(|index| {
                let canonical = if index == 0 {
                    "(pipe X (ridge))".to_string()
                } else {
                    format!(
                        "(pipe X (winsor_clip q={}) (ridge))",
                        0.01 + index as f64 * 0.001
                    )
                };
                let canonical = GenomeIr::parse(&canonical).unwrap().canonical();
                PopulationMember {
                    semantic_sha256: sha256_bytes(canonical.as_bytes()),
                    canonical_genome: canonical,
                    ancestry: Ancestry {
                        operation: "seed".to_string(),
                        parents: Vec::new(),
                        retry_count: 0,
                    },
                }
            })
            .collect()
    }

    fn fixture_config(size: usize) -> CampaignConfig {
        CampaignConfig {
            dataset: "fixture".to_string(),
            train_sha256: "0".repeat(64),
            generations: 2,
            population: size,
            cv_folds: 5,
            confirm_folds: 7,
            elites: 2,
            parent_pool: 4,
            tournament: 3,
            final_cv7_candidates: 4,
            max_transforms: 3,
            seed: 42,
            rayon_threads: size,
            expected_train_rows: None,
        }
    }

    #[test]
    fn robust_fitness_clips_and_penalizes_population_sd() {
        assert!((robust_fitness(-2.0, &[0.0, 1.0]) + 1.125).abs() < 1e-12);
        assert_eq!(robust_fitness(0.5, &[0.5; 5]), 0.5);
        assert_eq!(robust_fitness(0.5, &[f64::NAN]), 0.5);
    }

    #[test]
    fn production_generation_zero_is_pinned_canonical_and_unique() {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../prime/runs/exact-cv-inputs/faaa420e922e4128.generation-zero.txt"
        ));
        assert_eq!(
            sha256_bytes(bytes),
            "5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105"
        );
        let population = parse_initial_population(bytes, POPULATION).unwrap();
        validate_unique_population(&population, POPULATION).unwrap();
    }

    #[test]
    fn fold_manifests_are_stable_and_cover_each_row_once() {
        for folds in [5, 7] {
            let left = build_fold_manifest("fixture", &"0".repeat(64), 103, folds).unwrap();
            let right = build_fold_manifest("fixture", &"0".repeat(64), 103, folds).unwrap();
            assert_eq!(
                canonical_json(&left).unwrap(),
                canonical_json(&right).unwrap()
            );
            let mut validation = left
                .assignments
                .iter()
                .flat_map(|fold| fold.validation_rows.iter().copied())
                .collect::<Vec<_>>();
            validation.sort_unstable();
            assert_eq!(validation, (0..103).collect::<Vec<_>>());
        }
    }

    #[test]
    fn stable_order_uses_fitness_complexity_then_semantic_hash() {
        let row = |fitness, complexity, semantic: &str| SlotLedger {
            schema: LEDGER_SCHEMA.to_string(),
            generation: 0,
            slot: 0,
            canonical_genome: "(pipe X (ridge))".to_string(),
            semantic_sha256: semantic.to_string(),
            ancestry: Ancestry {
                operation: "seed".into(),
                parents: vec![],
                retry_count: 0,
            },
            head_family: "ridge".into(),
            complexity,
            valid: true,
            rejection_reason: None,
            cache_disposition: "miss".into(),
            cache_source_generation: None,
            cache_source_slot: None,
            artifacts: None,
            fold_r2: Some(vec![fitness; 5]),
            pooled_oof_r2: Some(fitness),
            pooled_oof_rmse: Some(1.0 - fitness),
            robust_fitness: Some(fitness),
            fold_fit_seconds: vec![],
            fold_predict_seconds: vec![],
            cumulative_unique_evaluations: 0,
            cumulative_terminal_slots: 0,
            cumulative_throughput_slots_per_second: 0.0,
        };
        let mut values = vec![
            row(0.8, 3, "b"),
            row(0.9, 9, "z"),
            row(0.8, 2, "z"),
            row(0.8, 2, "a"),
        ];
        values.sort_by(rank_slots);
        assert_eq!(
            values
                .iter()
                .map(|v| v.semantic_sha256.as_str())
                .collect::<Vec<_>>(),
            vec!["z", "a", "z", "b"]
        );
    }

    #[test]
    fn refill_is_unique_replayable_and_records_ancestry() {
        let data = toy_data(80);
        let population = fixture_population(8);
        let root = temp_dir("refill");
        let config = fixture_config(8);
        let live = LiveLogger::create(&root, 8, 8, 8).unwrap();
        let evaluations = evaluate_population(
            0,
            &population,
            &data,
            &config,
            &root,
            &HashMap::new(),
            &live,
        )
        .unwrap();
        let mut ranked = evaluations
            .into_iter()
            .enumerate()
            .map(|(slot, item)| {
                let result = item.cached.result.unwrap();
                SlotLedger {
                    schema: LEDGER_SCHEMA.into(),
                    generation: 0,
                    slot,
                    canonical_genome: population[slot].canonical_genome.clone(),
                    semantic_sha256: population[slot].semantic_sha256.clone(),
                    ancestry: population[slot].ancestry.clone(),
                    head_family: item.cached.head_family,
                    complexity: item.cached.complexity,
                    valid: true,
                    rejection_reason: None,
                    cache_disposition: "miss".into(),
                    cache_source_generation: None,
                    cache_source_slot: None,
                    artifacts: item.cached.artifacts,
                    fold_r2: Some(result.fold_r2),
                    pooled_oof_r2: Some(result.pooled_oof_r2),
                    pooled_oof_rmse: Some(result.pooled_oof_rmse),
                    robust_fitness: Some(result.robust_fitness),
                    fold_fit_seconds: result.fit_seconds,
                    fold_predict_seconds: result.predict_seconds,
                    cumulative_unique_evaluations: 0,
                    cumulative_terminal_slots: 0,
                    cumulative_throughput_slots_per_second: 0.0,
                }
            })
            .collect::<Vec<_>>();
        ranked.sort_by(rank_slots);
        let left = refill_population(&config, 1, &ranked).unwrap();
        let right = refill_population(&config, 1, &ranked).unwrap();
        assert_eq!(
            canonical_json(&left).unwrap(),
            canonical_json(&right).unwrap()
        );
        validate_unique_population(&left, 8).unwrap();
        assert_eq!(
            left.iter()
                .filter(|m| m.ancestry.operation == "elite")
                .count(),
            2
        );
        assert!(left.iter().skip(2).all(|m| !m.ancestry.parents.is_empty()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn serial_and_parallel_cv_scores_and_order_agree() {
        let data = toy_data(96);
        let genomes = fixture_population(8);
        let serial = genomes
            .iter()
            .map(|member| {
                exact_cv(
                    &GenomeIr::parse(&member.canonical_genome).unwrap(),
                    &data,
                    5,
                    "fixture",
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool.install(|| {
            genomes
                .par_iter()
                .map(|member| {
                    exact_cv(
                        &GenomeIr::parse(&member.canonical_genome).unwrap(),
                        &data,
                        5,
                        "fixture",
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        });
        for (left, right) in serial.iter().zip(&parallel) {
            assert!((left.robust_fitness - right.robust_fitness).abs() <= 1e-12);
            assert!((left.pooled_oof_r2 - right.pooled_oof_r2).abs() <= 1e-12);
            assert!(left
                .fold_r2
                .iter()
                .zip(&right.fold_r2)
                .all(|(a, b)| (a - b).abs() <= 1e-12));
        }
        let mut serial_order = (0..serial.len()).collect::<Vec<_>>();
        serial_order.sort_by(|&a, &b| {
            parallel[b]
                .robust_fitness
                .total_cmp(&parallel[a].robust_fitness)
                .then_with(|| genomes[a].semantic_sha256.cmp(&genomes[b].semantic_sha256))
        });
        let mut parallel_order = (0..parallel.len()).collect::<Vec<_>>();
        parallel_order.sort_by(|&a, &b| {
            parallel[b]
                .robust_fitness
                .total_cmp(&parallel[a].robust_fitness)
                .then_with(|| genomes[a].semantic_sha256.cmp(&genomes[b].semantic_sha256))
        });
        assert_eq!(serial_order, parallel_order);
    }

    #[test]
    fn two_generation_fixture_has_exact_accounting_cache_and_artifacts() {
        let root = temp_dir("campaign");
        let data = toy_data(96);
        let population = fixture_population(8);
        let config = fixture_config(8);
        let population_bytes = population
            .iter()
            .map(|m| format!("{}\n", m.canonical_genome))
            .collect::<String>();
        let population_sha = sha256_bytes(population_bytes.as_bytes());
        let selection =
            execute_campaign(&config, &data, population, &population_sha, &root).unwrap();
        assert_eq!(selection.generation, 1);
        assert_eq!(selection.cv7_finalists.len(), 4);
        assert_eq!(selection.quick_cv5_fold0.evaluated_fold_ids, vec![0]);
        assert_eq!(selection.quick_cv5_fold0.fold_r2.len(), 1);
        assert_eq!(selection.cv7.fold_r2.len(), 7);
        let ledger = fs::read_to_string(root.join("slots.v1.jsonl")).unwrap();
        let slots = ledger
            .lines()
            .map(|line| serde_json::from_str::<SlotLedger>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(slots.len(), 16);
        assert_eq!(slots.iter().filter(|slot| slot.generation == 0).count(), 8);
        assert_eq!(slots.iter().filter(|slot| slot.generation == 1).count(), 8);
        assert!(slots
            .iter()
            .all(|slot| slot.cache_disposition == "hit" || slot.cache_disposition == "miss"));
        assert!(slots.iter().filter(|slot| slot.valid).all(|slot| slot
            .fold_r2
            .as_ref()
            .unwrap()
            .len()
            == 1));
        assert!(slots.iter().filter(|slot| slot.valid).all(|slot| {
            let artifacts = slot.artifacts.as_ref().unwrap();
            let dir = root.join("artifacts").join(&slot.semantic_sha256);
            dir.join(format!("{}.py", artifacts.python_sha256))
                .is_file()
                && dir.join(format!("{}.rs", artifacts.rust_sha256)).is_file()
                && dir
                    .join(format!("{}.manifest.json", artifacts.manifest_sha256))
                    .is_file()
        }));
        for generation in 0..2 {
            assert!(root
                .join(format!("checkpoints/generation-{generation:03}.v1.json"))
                .is_file());
            assert!(root
                .join(format!("summaries/generation-{generation:03}.v1.json"))
                .is_file());
        }
        let (_, recovered_sha) = validate_selection_freeze(&root).unwrap();
        assert_eq!(
            recovered_sha,
            sha256_file(&root.join("selection-freeze.v1.json")).unwrap()
        );
        assert!(!root.join("test.csv").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_selection_fails_without_touching_test_path() {
        let root = temp_dir("invalid-selection");
        let error = validate_selection_freeze(&root).unwrap_err().to_string();
        assert!(error.contains("selection marker"));
        assert!(!root.join("test.csv").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn predictions_round_trip_and_metrics_are_independent() {
        let root = temp_dir("predictions");
        let path = root.join("predictions.csv");
        let predictions = vec![1.0, 2.0, 3.0];
        write_predictions_atomic(&path, &predictions).unwrap();
        let reopened = read_predictions(&path).unwrap();
        assert_eq!(reopened, predictions);
        let (rmse, r2, mae) = independent_metrics(&[1.0, 1.0, 4.0], &reopened).unwrap();
        assert!((rmse - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((r2 - 2.0 / 3.0).abs() < 1e-12);
        assert!((mae - 2.0 / 3.0).abs() < 1e-12);
        let _ = fs::remove_dir_all(root);
    }
}
