//! Multi-round `gp_distilled/v1` self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
//!
//! Each outer cross-fitting fold `k = 0..K-1` maintains its own strictly isolated training-split
//! acquisition trajectory over `{ j : fold(j) != k }` across rounds `r = 0..R-1`, so held-out fold
//! `k` (`{ i : fold(i) == k }`) never enters or influences outer fold `k`'s training archive at
//! any round.
//!
//! Alongside the active surrogate-guided acquisition trajectory (`active_acquired_by_fold[k]`),
//! each outer fold also maintains a matched-budget passive-acquisition control trajectory
//! (`passive_acquired_by_fold[k]`) that starts from the identical Round 0 seed set and acquires the
//! exact same number of unvisited candidates per context per round in deterministic outcome-blind
//! hash order (`sha256("round0:{fold_salt}:{context_id}:{program_sha256}")`).
//!
//! Each round `r = 0..R-1`:
//! 1. Distills the cumulative acquired training observations per outer fold into a validated
//!    `gp_distilled/v1` Parquet shard (`retention::distilled::write`), reads the shard back
//!    (`retention::distilled::read`), verifies E3 key reconciliation (`retention::distilled::reconcile`),
//!    and reconstructs each outer fold's training sub-pools from the reloaded Parquet rows.
//! 2. Evaluates closed-loop measured-tilt leaf distributions (`measured_tilt_leaf_log_probs`) and
//!    `decomposed_support_kl` contract admissibility over each training sub-pool's base vs newly
//!    acquired support. Any training context that violates the KL / new-support-mass bound is
//!    actively frozen (`kl_frozen_contexts`) so its next-round acquisition falls back to the
//!    outcome-blind base reference order instead of surrogate selection.
//! 3. Fits `ContextualSurrogateQHat` per outer fold `k` on both the active training sub-pools and
//!    the matched-budget passive control training sub-pools (`fold != k`), scores held-out fold `k`
//!    on the full frozen shadow pools across budgets `B`, advances held-out discovery trajectories
//!    (`heldout_active_acquired` vs `heldout_passive_acquired`, never used for training) by
//!    `step_budget_per_context`, and acquires Round `r + 1` training candidates strictly within
//!    outer fold `k`'s training split (`fold != k`).
//!
//! Shadow-only: never claims `qualified = true` or `l6`.

use super::bootstrap::{cluster_bootstrap_ci, PairedInterval, Verdict};
use super::evaluate::{
    evaluate_with_fold_train_pools_and_priors, fold_of, Config as EvalConfig, Inputs, Summary,
};
use super::metrics::{score_context, ContextMetrics};
use super::pool::{verify, Manifest, Member, Pool, PoolView, Status, POOL_SCHEMA};
use super::selectors::{
    pre_screen_funnel, token_bag_features, ContextualSurrogateQHat, CvEstimate,
    ExternalProgramPrior, Features, Selector,
};
use crate::canon::{canonical_sha256, sha256_hex};
use crate::clover::expansion::{measured_tilt_leaf_log_probs, MeasuredTiltPolicyConfig};
use crate::retention::distilled::{
    self, distill_selfplay_observations, distilled_rows_to_measurement_rows, score_direction,
    Outcome, ScoreDirection, SelfPlayDistillObservation, SelfPlayDistillSpec, SplitRole,
};
use crate::support::decomposed_support_kl;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

pub const FLYWHEEL_RECEIPT_SCHEMA: &str = "jope.j9-distilled-flywheel-receipt/v2";

#[derive(Clone, Debug, Serialize)]
pub struct FlywheelConfig {
    pub eval: EvalConfig,
    pub eval_budgets: Vec<usize>,
    pub rounds: usize,
    pub initial_budget_per_context: usize,
    pub step_budget_per_context: usize,
    pub proposal_pool_k: usize,
    pub score_name: String,
    pub tilt: MeasuredTiltPolicyConfig,
    pub max_conditional_base_kl: f64,
    pub max_new_support_mass: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlywheelRoundSummary {
    pub round: usize,
    pub distilled_rows: usize,
    pub parquet_bytes: usize,
    pub parquet_sha256: String,
    pub reconciled_parquet_keys: usize,
    pub mean_train_hwm_gain: f64,
    pub passive_mean_train_hwm_gain: f64,
    pub mean_heldout_active_discovery_hwm_gain: f64,
    pub mean_heldout_passive_discovery_hwm_gain: f64,
    pub kl_admissible_contexts: usize,
    pub kl_frozen_contexts: usize,
    pub max_observed_conditional_base_kl: f64,
    pub max_observed_new_support_mass: f64,
    pub mean_archive_ce_loss: f64,
    pub heldout_by_budget: BTreeMap<usize, Summary>,
    pub passive_heldout_by_budget: BTreeMap<usize, Summary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoundLiftComparison {
    pub budget: usize,
    pub from_round: usize,
    pub to_round: usize,
    pub realized_gain_lift: PairedInterval,
    pub top1_gain_lift: PairedInterval,
    pub pairwise_accuracy_lift: Option<PairedInterval>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlywheelFoldCv {
    pub round: usize,
    pub fold: usize,
    pub active: CvEstimate,
    pub passive: CvEstimate,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlywheelReceipt {
    pub schema: &'static str,
    pub shadow_only: bool,
    pub qualified_claim: bool,
    pub l6_claim: bool,
    pub manifest_sha256: String,
    pub s3_prior_rows_by_fold: Vec<usize>,
    pub config: FlywheelConfig,
    pub rounds: Vec<FlywheelRoundSummary>,
    pub surrogate_cv: Vec<FlywheelFoldCv>,
    pub monotonic_train_hwm_gain: bool,
    pub monotonic_heldout_pairwise_accuracy: bool,
    pub all_rounds_kl_admissible: bool,
    pub round_lifts: Vec<RoundLiftComparison>,
    pub active_vs_passive_lifts: Vec<RoundLiftComparison>,
    pub active_vs_passive_discovery_lift: PairedInterval,
    pub final_round_receipt: super::evaluate::Receipt,
    pub surrogate_beats_strongest_baseline: bool,
    pub final_vs_round0_primary_verdict: Verdict,
    pub final_vs_round0_pairwise_verdict: Verdict,
    pub active_beats_passive_discovery_verdict: Verdict,
    pub active_beats_passive_primary_verdict: Verdict,
    pub active_beats_passive_pairwise_verdict: Verdict,
    pub active_discovery_established: bool,
    pub data_accumulation_lift_established: bool,
    pub flywheel_established: bool,
}

impl FlywheelReceipt {
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

pub fn known_scores() -> BTreeSet<String> {
    [
        "cv_geomean_rmse_ratio",
        "best_rust_rmse",
        "validation_rmse",
        "shadow_pool_q",
        "l6",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Builds per-outer-fold cross-family S3 `gp_distilled/v1` program prior statistics.
///
/// A row from `s3_rows` is admitted into outer fold `k`'s prior if and only if:
/// 1. `row.split_role == SplitRole::Train` and `row.outcome == Outcome::Ok`, and
/// 2. Its `parent_id` can be resolved unambiguously from `pools` (either directly via
///    `row.context_sha256` matching a pool, or via `row.dataset` mapping to a unique `parent_id`
///    among `pools`) AND that `parent_id` belongs to outer training split `fold != k`.
///
/// Any row whose `parent_id` cannot be proven to lie outside held-out fold `k` is excluded.
pub fn build_fold_s3_priors(
    pools: &[Pool],
    cfg: &EvalConfig,
    s3_rows: &[distilled::DistilledRow],
) -> Result<(Vec<ExternalProgramPrior>, Vec<usize>)> {
    ensure!(cfg.folds >= 2, "cross-fitting needs at least two folds");
    let mut ctx_to_parent_fold: BTreeMap<&str, (&str, usize)> = BTreeMap::new();
    for p in pools {
        let f = fold_of(&cfg.fold_salt, &p.parent_id, cfg.folds);
        ctx_to_parent_fold.insert(p.context_id.as_str(), (p.parent_id.as_str(), f));
    }

    let mut dset_to_parents: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut dset_to_folds: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for r in s3_rows {
        if let Some(ctx) = r.context_sha256.as_deref() {
            if let Some(&(par, f)) = ctx_to_parent_fold.get(ctx) {
                dset_to_parents
                    .entry(r.dataset.as_str())
                    .or_default()
                    .insert(par);
                dset_to_folds
                    .entry(r.dataset.as_str())
                    .or_default()
                    .insert(f);
            }
        }
    }

    let mut priors = vec![ExternalProgramPrior::default(); cfg.folds];
    let mut counts = vec![0usize; cfg.folds];
    for r in s3_rows {
        if r.split_role != SplitRole::Train || r.outcome != Outcome::Ok {
            continue;
        }
        let resolved: Option<(&str, usize)> = r
            .context_sha256
            .as_deref()
            .and_then(|ctx| ctx_to_parent_fold.get(ctx).copied())
            .or_else(|| {
                let pars = dset_to_parents.get(r.dataset.as_str())?;
                let flds = dset_to_folds.get(r.dataset.as_str())?;
                if pars.len() == 1 && flds.len() == 1 {
                    let par = *pars.iter().next()?;
                    let f = *flds.iter().next()?;
                    Some((par, f))
                } else {
                    None
                }
            });
        let Some((parent_id, row_fold)) = resolved else {
            continue;
        };
        // Continuous score quality when present (e.g. `1.0 - rmse_ratio` for bounded
        // `cv_geomean_rmse_ratio` HWM winners where `rank_ctx == 1` is constant, or `1 / (1 + s)`
        // for unbounded non-negative LowerIsBetter losses), falling back to `rank_ctx` when
        // score or known score direction is absent.
        let rank_fallback = match r.rank_ctx {
            Some(1) => 1.0,
            Some(rk) if rk > 1 => (1.0 / (rk as f64)).clamp(0.0, 1.0),
            _ => 0.75,
        };
        let target_val = match (
            r.score.filter(|s| s.is_finite()),
            r.score_name.as_deref(),
            r.score_name
                .as_deref()
                .and_then(|n| score_direction(n).ok()),
        ) {
            (Some(s), Some("cv_geomean_rmse_ratio"), Some(ScoreDirection::LowerIsBetter)) => {
                (1.0 - s.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            }
            (Some(s), _, Some(ScoreDirection::LowerIsBetter)) => 1.0 / (1.0 + s.max(0.0)),
            (Some(s), _, Some(ScoreDirection::HigherIsBetter)) => {
                1.0 / (1.0 + (-s.clamp(-20.0, 20.0)).exp())
            }
            _ => rank_fallback,
        };
        for k in 0..cfg.folds {
            if row_fold == k {
                continue;
            }
            let tot = priors[k]
                .tot_stats
                .entry(r.program_sha.clone())
                .or_insert((0.0, 0));
            tot.0 += target_val;
            tot.1 += 1;
            let par = priors[k]
                .parent_stats
                .entry((parent_id.to_owned(), r.program_sha.clone()))
                .or_insert((0.0, 0));
            par.0 += target_val;
            par.1 += 1;
            counts[k] += 1;
        }
    }
    Ok((priors, counts))
}

fn summarize_metrics(rows: &[ContextMetrics]) -> Summary {
    let n = rows.len() as f64;
    let mean = |f: fn(&ContextMetrics) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let pairwise: Vec<f64> = rows.iter().filter_map(|m| m.pairwise_accuracy).collect();
    Summary {
        contexts: rows.len(),
        mean_realized_gain: mean(|m| m.realized_gain),
        mean_top1_gain: mean(|m| m.top1_gain),
        mean_regret: mean(|m| m.regret),
        mean_normalized_regret: {
            let v: Vec<f64> = rows.iter().filter_map(|m| m.normalized_regret).collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
        },
        beats_reference_rate: mean(|m| m.beats_reference),
        pairwise_contexts: pairwise.len(),
        mean_pairwise_accuracy: (!pairwise.is_empty())
            .then(|| pairwise.iter().sum::<f64>() / pairwise.len() as f64),
    }
}

fn paired_metrics(
    pools: &[Pool],
    later: &[ContextMetrics],
    earlier: &[ContextMetrics],
    f: impl Fn(&ContextMetrics) -> Option<f64>,
    cfg: &EvalConfig,
) -> Result<Option<PairedInterval>> {
    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for ((pool, a), b) in pools.iter().zip(later).zip(earlier) {
        if let (Some(x), Some(y)) = (f(a), f(b)) {
            groups.entry(&pool.parent_id).or_default().push(x - y);
        }
    }
    if groups.values().map(Vec::len).sum::<usize>() < 2 {
        return Ok(None);
    }
    let groups: Vec<Vec<f64>> = groups.into_values().collect();
    Ok(Some(cluster_bootstrap_ci(
        &groups,
        cfg.bootstrap_draws,
        cfg.bootstrap_seed,
        cfg.alpha,
    )?))
}

fn paired_scalars(
    pools: &[Pool],
    later: &[f64],
    earlier: &[f64],
    cfg: &EvalConfig,
) -> Result<PairedInterval> {
    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for ((pool, &x), &y) in pools.iter().zip(later).zip(earlier) {
        groups.entry(&pool.parent_id).or_default().push(x - y);
    }
    let groups: Vec<Vec<f64>> = groups.into_values().collect();
    cluster_bootstrap_ci(&groups, cfg.bootstrap_draws, cfg.bootstrap_seed, cfg.alpha)
}

pub fn run_distilled_flywheel(
    pools: &[Pool],
    manifest: &Manifest,
    context_features: &Features,
    program_features: Option<&Features>,
    external: &[&dyn Selector],
    cfg: &FlywheelConfig,
    parquet_dir: Option<&Path>,
) -> Result<FlywheelReceipt> {
    run_distilled_flywheel_with_s3_prior(
        pools,
        manifest,
        context_features,
        program_features,
        external,
        None,
        cfg,
        parquet_dir,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_distilled_flywheel_with_s3_prior(
    pools: &[Pool],
    manifest: &Manifest,
    context_features: &Features,
    program_features: Option<&Features>,
    external: &[&dyn Selector],
    s3_prior_rows: Option<&[distilled::DistilledRow]>,
    cfg: &FlywheelConfig,
    parquet_dir: Option<&Path>,
) -> Result<FlywheelReceipt> {
    verify(pools, manifest)?;
    context_features
        .verify_context_descriptor()
        .context("flywheel nested/v3 context provenance unavailable")?;
    ensure!(!pools.is_empty(), "no frozen pools");
    ensure!(cfg.rounds >= 2, "flywheel requires at least 2 rounds");
    ensure!(
        cfg.initial_budget_per_context >= 2,
        "initial_budget_per_context must be >= 2"
    );
    ensure!(
        cfg.step_budget_per_context >= 1 && cfg.proposal_pool_k >= cfg.step_budget_per_context,
        "requires 1 <= step_budget_per_context <= proposal_pool_k"
    );
    ensure!(
        !cfg.eval_budgets.is_empty() && cfg.eval_budgets.iter().all(|&b| b >= 1),
        "eval_budgets must be non-empty and positive"
    );
    ensure!(
        cfg.eval_budgets.contains(&cfg.eval.budget),
        "eval_budgets must include primary eval.budget ({})",
        cfg.eval.budget
    );
    ensure!(
        cfg.score_name != "l6",
        "shadow flywheel distillation may not use score_name \"l6\""
    );
    cfg.tilt.validate()?;
    ensure!(
        cfg.max_conditional_base_kl.is_finite() && cfg.max_conditional_base_kl > 0.0,
        "max_conditional_base_kl must be finite and > 0"
    );
    ensure!(
        cfg.max_new_support_mass.is_finite()
            && cfg.max_new_support_mass > 0.0
            && cfg.max_new_support_mass < 1.0,
        "max_new_support_mass must be in (0, 1)"
    );

    let scores_set = known_scores();
    let direction = score_direction(&cfg.score_name)?;
    let derived_pf = match program_features {
        Some(_) => None,
        None => token_bag_features(pools),
    };
    let pf = program_features
        .or(derived_pf.as_ref())
        .context("program features or pool tokens required for flywheel")?;
    if let Some(dir) = parquet_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create parquet directory {}", dir.display()))?;
    }

    let manifest_sha256 = manifest.sha256()?;
    let (fold_s3_priors, s3_prior_rows_by_fold) = match s3_prior_rows {
        Some(rows) => {
            let (p, c) = build_fold_s3_priors(pools, &cfg.eval, rows)?;
            (Some(p), c)
        }
        None => (None, vec![0usize; cfg.eval.folds]),
    };

    let folds: Vec<usize> = pools
        .iter()
        .map(|p| fold_of(&cfg.eval.fold_salt, &p.parent_id, cfg.eval.folds))
        .collect();
    let refs: Vec<f64> = pools
        .iter()
        .map(|p| p.reference(cfg.eval.reference))
        .collect::<Result<_>>()?;
    let pool_idx_by_ctx: BTreeMap<&str, usize> = pools
        .iter()
        .enumerate()
        .map(|(i, p)| (p.context_id.as_str(), i))
        .collect();

    // Deterministic outcome-blind hash ordering and Round 0 seed subset per context.
    let mut hash_ordered_indices: Vec<Vec<usize>> = Vec::with_capacity(pools.len());
    let mut base_acquired: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); pools.len()];
    for (i, pool) in pools.iter().enumerate() {
        let mut order: Vec<(String, usize)> = pool
            .members
            .iter()
            .enumerate()
            .map(|(m_idx, m)| {
                let draw_hash = sha256_hex(
                    format!(
                        "round0:{}:{}:{}",
                        cfg.eval.fold_salt, pool.context_id, m.program_sha256
                    )
                    .as_bytes(),
                );
                (draw_hash, m_idx)
            })
            .collect();
        order.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let ordered_idx: Vec<usize> = order.iter().map(|&(_, idx)| idx).collect();
        // Ensure at least one Complete member is present in Round 0 so every context has a finite reference.
        let mut chosen: Vec<usize> = ordered_idx
            .iter()
            .copied()
            .take(cfg.initial_budget_per_context.min(pool.members.len()))
            .collect();
        if !chosen
            .iter()
            .any(|&idx| pool.members[idx].outcome().is_some())
        {
            if let Some(&first_ok) = ordered_idx
                .iter()
                .find(|&&idx| pool.members[idx].outcome().is_some())
            {
                if let Some(last) = chosen.last_mut() {
                    *last = first_ok;
                }
            }
        }
        for idx in chosen {
            base_acquired[i].insert(idx);
        }
        hash_ordered_indices.push(ordered_idx);
    }

    // Strictly isolated per-outer-fold training acquisition states for active surrogate and
    // matched-budget passive hash-order control (`folds[j] != k`).
    let mut active_acquired_by_fold: Vec<Vec<BTreeSet<usize>>> =
        vec![base_acquired.clone(); cfg.eval.folds];
    let mut passive_acquired_by_fold: Vec<Vec<BTreeSet<usize>>> =
        vec![base_acquired.clone(); cfg.eval.folds];

    // Strictly out-of-fold held-out discovery trajectories per context `i` (updated only by outer
    // fold `k = folds[i]`'s surrogate vs passive hash order; never read by any training fold).
    let mut heldout_active_acquired: Vec<BTreeSet<usize>> = base_acquired.clone();
    let mut heldout_passive_acquired: Vec<BTreeSet<usize>> = base_acquired.clone();

    let mut round_summaries = Vec::with_capacity(cfg.rounds);
    let mut round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>> =
        Vec::with_capacity(cfg.rounds);
    let mut passive_round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>> =
        Vec::with_capacity(cfg.rounds);
    let mut final_active_discovery_hwm = vec![0.0; pools.len()];
    let mut final_passive_discovery_hwm = vec![0.0; pools.len()];
    let mut all_rounds_kl_admissible = true;
    let mut final_train_sub_pools: Vec<Vec<Pool>> = Vec::new();
    let mut round_cv_trace: Vec<FlywheelFoldCv> = Vec::with_capacity(cfg.rounds * cfg.eval.folds);

    for r in 0..cfg.rounds {
        // 1. Distill each outer fold's currently acquired training observations into gp_distilled/v1 Parquet and read back.
        let mut round_distilled = Vec::new();
        let mut raw_keys = BTreeSet::new();
        let mut tokens_by_prog: BTreeMap<String, Option<Vec<u32>>> = BTreeMap::new();

        for (k, fold_acquired) in active_acquired_by_fold.iter().enumerate() {
            let run_id = format!("flywheel-r{r:02}-fold{k}");
            for (j, pool) in pools.iter().enumerate() {
                if folds[j] == k {
                    // Held-out fold `k` is strictly excluded from outer fold `k`'s training archive.
                    continue;
                }
                let spec = SelfPlayDistillSpec {
                    dataset: pool.parent_id.clone(),
                    context_sha256: pool.context_id.clone(),
                    fold: Some(folds[j] as i64),
                    split_role: SplitRole::Train,
                    score_name: cfg.score_name.clone(),
                    policylm_ckpt: None,
                    campaign: Some("clover-s3-distilled-flywheel".into()),
                    run_id: run_id.clone(),
                    family: "clover-selfplay".into(),
                    source_bucket: "local-shadow-pools".into(),
                };
                let mut obs = Vec::with_capacity(fold_acquired[j].len());
                for &m_idx in &fold_acquired[j] {
                    let m = &pool.members[m_idx];
                    tokens_by_prog
                        .entry(m.program_sha256.clone())
                        .or_insert_with(|| m.tokens.clone());
                    let (outcome, raw_score) = match (m.status, m.q) {
                        (Status::Complete, Some(q)) => {
                            let s = match direction {
                                ScoreDirection::LowerIsBetter => {
                                    if q == 0.0 {
                                        0.0
                                    } else {
                                        -q
                                    }
                                }
                                ScoreDirection::HigherIsBetter => q,
                            };
                            (Outcome::Ok, Some(s))
                        }
                        (Status::Timeout, _) => (Outcome::Timeout, None),
                        _ => (Outcome::Failed, None),
                    };
                    let key = format!(
                        "shadow-pools/round-{r:02}/fold-{k}/{}/{}.json",
                        pool.context_id, m.program_sha256
                    );
                    let pool_sha256 = pool.sha256()?;
                    let src_sha = canonical_sha256(&(
                        &pool_sha256,
                        r,
                        k,
                        &pool.context_id,
                        &m.program_sha256,
                        m.status,
                        m.q,
                    ))?;
                    raw_keys.insert(key.clone());
                    obs.push(SelfPlayDistillObservation {
                        program_sha: m.program_sha256.clone(),
                        program: None,
                        outcome,
                        score: raw_score,
                        seed_packet: Some(if base_acquired[j].contains(&m_idx) {
                            "round0_cold_start".into()
                        } else {
                            format!("surrogate_funnel_up_to_r{r}")
                        }),
                        source_key: key,
                        source_version_id: Some(format!("r{r}-k{k}")),
                        source_sha256: src_sha,
                    });
                }
                let batch = distill_selfplay_observations(&spec, &obs, &scores_set)?;
                round_distilled.extend(batch);
            }
        }

        let parquet_bytes = distilled::write(
            &round_distilled,
            &scores_set,
            &[("flywheel_round".into(), r.to_string())],
        )?;
        let parquet_sha256 = sha256_hex(&parquet_bytes);
        if let Some(dir) = parquet_dir {
            let shard_path = dir.join(format!("round-{r:02}.parquet"));
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&shard_path)
                .with_context(|| {
                    format!(
                        "create {} (outputs are never overwritten)",
                        shard_path.display()
                    )
                })?;
            f.write_all(&parquet_bytes)?;
            f.sync_all()?;
        }
        let reloaded = distilled::read(&parquet_bytes, &scores_set)?;
        distilled::reconcile(&reloaded, &raw_keys)?;

        // Partition reloaded rows by outer fold `run_id` and bridge each fold's training rows via distilled_rows_to_measurement_rows.
        let mut rows_by_run_id: BTreeMap<String, Vec<distilled::DistilledRow>> = BTreeMap::new();
        for row in reloaded.iter().cloned() {
            rows_by_run_id
                .entry(row.run_id.clone())
                .or_default()
                .push(row);
        }

        let mut train_sub_pools_by_fold: Vec<Vec<Pool>> = Vec::with_capacity(cfg.eval.folds);
        let mut passive_train_sub_pools_by_fold: Vec<Vec<Pool>> =
            Vec::with_capacity(cfg.eval.folds);
        let mut kl_admissible_by_fold_ctx: Vec<Vec<bool>> =
            vec![vec![true; pools.len()]; cfg.eval.folds];
        let mut train_hwm_gains = Vec::new();
        let mut passive_train_hwm_gains = Vec::new();
        let mut kl_admissible_contexts = 0usize;
        let mut kl_frozen_contexts = 0usize;
        let mut max_cond_kl = 0.0f64;
        let mut max_new_mass = 0.0f64;
        let mut ce_losses = Vec::new();

        for k in 0..cfg.eval.folds {
            let run_id = format!("flywheel-r{r:02}-fold{k}");
            let fold_rows = rows_by_run_id
                .remove(&run_id)
                .with_context(|| format!("missing reloaded Parquet rows for {run_id}"))?;
            let m_rows = distilled_rows_to_measurement_rows(
                &fold_rows,
                &cfg.score_name,
                &pools[0].protocol_sha256,
                &scores_set,
            )?;
            let mut by_ctx: BTreeMap<String, Vec<Member>> = BTreeMap::new();
            for row in m_rows {
                by_ctx.entry(row.context_id).or_default().push(Member {
                    tokens: tokens_by_prog
                        .get(&row.program_sha256)
                        .cloned()
                        .unwrap_or(None),
                    program_sha256: row.program_sha256,
                    status: row.status,
                    q: row.q,
                });
            }

            let mut fold_pools = Vec::new();
            let mut passive_fold_pools = Vec::new();
            for (j, pool) in pools.iter().enumerate() {
                if folds[j] == k {
                    continue;
                }
                let idx_check =
                    *pool_idx_by_ctx
                        .get(pool.context_id.as_str())
                        .with_context(|| {
                            format!("missing pool index for context {}", pool.context_id)
                        })?;
                debug_assert_eq!(j, idx_check);
                let mut members = by_ctx.remove(&pool.context_id).with_context(|| {
                    format!(
                        "missing distilled training sub-pool for fold {k} context {}",
                        pool.context_id
                    )
                })?;
                members.sort_by(|a, b| a.program_sha256.cmp(&b.program_sha256));
                let best_q = members
                    .iter()
                    .filter_map(Member::outcome)
                    .fold(refs[j], f64::max);
                train_hwm_gains.push((best_q - refs[j]).max(0.0));

                // Build matched-budget passive training sub-pool for outer fold `k` context `j`.
                let mut passive_members: Vec<Member> = passive_acquired_by_fold[k][j]
                    .iter()
                    .map(|&m_idx| pool.members[m_idx].clone())
                    .collect();
                passive_members.sort_by(|a, b| a.program_sha256.cmp(&b.program_sha256));
                let passive_best_q = passive_members
                    .iter()
                    .filter_map(Member::outcome)
                    .fold(refs[j], f64::max);
                passive_train_hwm_gains.push((passive_best_q - refs[j]).max(0.0));
                passive_fold_pools.push(Pool {
                    schema: POOL_SCHEMA.into(),
                    context_id: pool.context_id.clone(),
                    parent_id: pool.parent_id.clone(),
                    role: pool.role.clone(),
                    protocol_sha256: pool.protocol_sha256.clone(),
                    incumbent_q: pool.incumbent_q,
                    members: passive_members,
                });

                // Evaluate measured-tilt leaf distribution and decomposed support KL on base vs newly acquired leaves.
                let base_shas: BTreeSet<&str> = base_acquired[j]
                    .iter()
                    .map(|&idx| pool.members[idx].program_sha256.as_str())
                    .collect();
                let mut base_q = Vec::new();
                let mut arch_q = Vec::new();
                for m in &members {
                    let q_val = m.outcome().unwrap_or(refs[j]);
                    if base_shas.contains(m.program_sha256.as_str()) {
                        base_q.push(q_val);
                    } else {
                        arch_q.push(q_val);
                    }
                }
                let base_floor = base_q.iter().copied().fold(f64::INFINITY, f64::min);
                let (pol_lp, ref_lp, ce) =
                    measured_tilt_leaf_log_probs(&base_q, &arch_q, base_floor, &cfg.tilt)?;
                let is_base_leaf: Vec<bool> =
                    (0..pol_lp.len()).map(|idx| idx < base_q.len()).collect();
                let kl = decomposed_support_kl(
                    &pol_lp,
                    &ref_lp,
                    &is_base_leaf,
                    cfg.max_conditional_base_kl,
                    cfg.max_new_support_mass,
                )?;
                kl_admissible_by_fold_ctx[k][j] = kl.admissible;
                if kl.admissible {
                    kl_admissible_contexts += 1;
                } else {
                    all_rounds_kl_admissible = false;
                    kl_frozen_contexts += 1;
                }
                max_cond_kl = max_cond_kl.max(kl.conditional_base_kl);
                max_new_mass = max_new_mass.max(kl.new_support_mass);
                ce_losses.push(ce);

                fold_pools.push(Pool {
                    schema: POOL_SCHEMA.into(),
                    context_id: pool.context_id.clone(),
                    parent_id: pool.parent_id.clone(),
                    role: pool.role.clone(),
                    protocol_sha256: pool.protocol_sha256.clone(),
                    incumbent_q: pool.incumbent_q,
                    members,
                });
            }
            train_sub_pools_by_fold.push(fold_pools);
            passive_train_sub_pools_by_fold.push(passive_fold_pools);
        }

        // Compute current round's out-of-fold held-out discovery HWM gains per context `i` before advancing to Round r+1.
        let mut round_active_disc_hwm = vec![0.0; pools.len()];
        let mut round_passive_disc_hwm = vec![0.0; pools.len()];
        for (i, pool) in pools.iter().enumerate() {
            let act_best = heldout_active_acquired[i]
                .iter()
                .filter_map(|&m_idx| pool.members[m_idx].outcome())
                .fold(refs[i], f64::max);
            let pas_best = heldout_passive_acquired[i]
                .iter()
                .filter_map(|&m_idx| pool.members[m_idx].outcome())
                .fold(refs[i], f64::max);
            round_active_disc_hwm[i] = (act_best - refs[i]).max(0.0);
            round_passive_disc_hwm[i] = (pas_best - refs[i]).max(0.0);
        }
        if r + 1 == cfg.rounds {
            final_active_discovery_hwm = round_active_disc_hwm.clone();
            final_passive_discovery_hwm = round_passive_disc_hwm.clone();
        }

        // 2. Fit ContextualSurrogateQHat per outer fold k strictly on train_sub_pools_by_fold[k]
        //    (and passive_surrogate on passive_train_sub_pools_by_fold[k]), score held-out fold k
        //    on full pools, advance held-out discovery sets, and acquire Round r+1 training
        //    candidates strictly within outer fold k's training split (folds[j] != k).
        let empty_metrics = || ContextMetrics {
            realized_gain: 0.0,
            top1_gain: 0.0,
            regret: 0.0,
            normalized_regret: None,
            beats_reference: 0.0,
            pairwise_accuracy: None,
        };
        let mut metrics_for_round: BTreeMap<usize, Vec<ContextMetrics>> = cfg
            .eval_budgets
            .iter()
            .map(|&b| (b, vec![empty_metrics(); pools.len()]))
            .collect();
        let mut passive_metrics_for_round: BTreeMap<usize, Vec<ContextMetrics>> = cfg
            .eval_budgets
            .iter()
            .map(|&b| (b, vec![empty_metrics(); pools.len()]))
            .collect();

        for k in 0..cfg.eval.folds {
            let held: Vec<usize> = (0..pools.len()).filter(|&i| folds[i] == k).collect();
            let train: Vec<&Pool> = train_sub_pools_by_fold[k].iter().collect();
            let passive_train: Vec<&Pool> = passive_train_sub_pools_by_fold[k].iter().collect();
            if held.is_empty() {
                continue;
            }
            ensure!(!train.is_empty(), "fold {k} has no training pools");
            let ext_prior = fold_s3_priors.as_ref().map(|ps| &ps[k]);
            let surrogate = ContextualSurrogateQHat::fit_with_prior(
                &train,
                context_features,
                pf,
                cfg.eval.knn_k,
                cfg.eval.prior_weight,
                cfg.eval.ridge_lambda,
                ext_prior,
            )?;
            let passive_surrogate = ContextualSurrogateQHat::fit_with_prior(
                &passive_train,
                context_features,
                pf,
                cfg.eval.knn_k,
                cfg.eval.prior_weight,
                cfg.eval.ridge_lambda,
                ext_prior,
            )?;
            round_cv_trace.push(FlywheelFoldCv {
                round: r,
                fold: k,
                active: surrogate.inner_cv_estimate().clone(),
                passive: passive_surrogate.inner_cv_estimate().clone(),
            });

            // Evaluate held-out fold `k` on full pools (never used for training or training-split acquisitions).
            for &i in &held {
                let full_pool = &pools[i];
                let view = full_pool.view();
                let scores = surrogate.scores(&view)?;
                ensure!(
                    scores.len() == view.programs.len(),
                    "contextual_surrogate_qhat: one score per member"
                );
                ensure!(
                    scores.iter().all(|s| s.is_finite()),
                    "contextual_surrogate_qhat: non-finite score in {}",
                    full_pool.context_id
                );
                let passive_scores = passive_surrogate.scores(&view)?;
                ensure!(
                    passive_scores.len() == view.programs.len()
                        && passive_scores.iter().all(|s| s.is_finite()),
                    "passive_contextual_surrogate_qhat: invalid scores in {}",
                    full_pool.context_id
                );
                let outcomes = full_pool.outcomes();
                for &b in &cfg.eval_budgets {
                    let m = score_context(&outcomes, refs[i], &scores, &view.programs, b);
                    *metrics_for_round
                        .get_mut(&b)
                        .with_context(|| format!("missing active metrics for budget {b}"))?
                        .get_mut(i)
                        .with_context(|| {
                            format!("missing active metrics for budget {b}, context {i}")
                        })? = m;
                    let pm = score_context(&outcomes, refs[i], &passive_scores, &view.programs, b);
                    *passive_metrics_for_round
                        .get_mut(&b)
                        .with_context(|| format!("missing passive metrics for budget {b}"))?
                        .get_mut(i)
                        .with_context(|| {
                            format!("missing passive metrics for budget {b}, context {i}")
                        })? = pm;
                }

                // Advance strictly out-of-fold held-out discovery trajectories on context `i`.
                let unmeasured_held: Vec<usize> = (0..full_pool.members.len())
                    .filter(|m_idx| !heldout_active_acquired[i].contains(m_idx))
                    .collect();
                if !unmeasured_held.is_empty() {
                    let unmeasured_progs: Vec<&str> = unmeasured_held
                        .iter()
                        .map(|&m_idx| full_pool.members[m_idx].program_sha256.as_str())
                        .collect();
                    let unmeasured_tokens: Vec<Option<&[u32]>> = unmeasured_held
                        .iter()
                        .map(|&m_idx| full_pool.members[m_idx].tokens.as_deref())
                        .collect();
                    let rem_view = PoolView {
                        context_id: &full_pool.context_id,
                        parent_id: &full_pool.parent_id,
                        incumbent_q: full_pool.incumbent_q,
                        programs: unmeasured_progs,
                        tokens: unmeasured_tokens,
                    };
                    let eval_b = cfg.step_budget_per_context.min(unmeasured_held.len());
                    let prop_k = cfg.proposal_pool_k.max(eval_b);
                    let funnel = pre_screen_funnel(&rem_view, &surrogate, prop_k, eval_b)?;
                    let mut newly_added = 0usize;
                    for sel_sha in &funnel.selected_program_sha256 {
                        if let Ok(m_idx) = full_pool
                            .members
                            .binary_search_by(|m| m.program_sha256.as_str().cmp(sel_sha.as_str()))
                        {
                            if heldout_active_acquired[i].insert(m_idx) {
                                newly_added += 1;
                            }
                        }
                    }
                    let mut pas_added = 0usize;
                    for &m_idx in &hash_ordered_indices[i] {
                        if pas_added >= newly_added {
                            break;
                        }
                        if heldout_passive_acquired[i].insert(m_idx) {
                            pas_added += 1;
                        }
                    }
                }
            }

            // Acquire Round r+1 training candidates strictly inside outer fold `k`'s training split (`folds[j] != k`).
            for (j, train_pool) in pools.iter().enumerate() {
                if folds[j] == k {
                    continue;
                }
                let unmeasured_indices: Vec<usize> = (0..train_pool.members.len())
                    .filter(|m_idx| !active_acquired_by_fold[k][j].contains(m_idx))
                    .collect();
                if !unmeasured_indices.is_empty() {
                    let eval_b = cfg.step_budget_per_context.min(unmeasured_indices.len());
                    let mut newly_added = 0usize;
                    if kl_admissible_by_fold_ctx[k][j] {
                        let unmeasured_progs: Vec<&str> = unmeasured_indices
                            .iter()
                            .map(|&m_idx| train_pool.members[m_idx].program_sha256.as_str())
                            .collect();
                        let unmeasured_tokens: Vec<Option<&[u32]>> = unmeasured_indices
                            .iter()
                            .map(|&m_idx| train_pool.members[m_idx].tokens.as_deref())
                            .collect();
                        let rem_view = PoolView {
                            context_id: &train_pool.context_id,
                            parent_id: &train_pool.parent_id,
                            incumbent_q: train_pool.incumbent_q,
                            programs: unmeasured_progs,
                            tokens: unmeasured_tokens,
                        };
                        let prop_k = cfg.proposal_pool_k.max(eval_b);
                        let funnel = pre_screen_funnel(&rem_view, &surrogate, prop_k, eval_b)?;
                        for sel_sha in &funnel.selected_program_sha256 {
                            if let Ok(m_idx) = train_pool.members.binary_search_by(|m| {
                                m.program_sha256.as_str().cmp(sel_sha.as_str())
                            }) {
                                if active_acquired_by_fold[k][j].insert(m_idx) {
                                    newly_added += 1;
                                }
                            }
                        }
                    } else {
                        // KL guard violated: freeze policy update for this context and fall back to outcome-blind base hash order.
                        for &m_idx in &hash_ordered_indices[j] {
                            if newly_added >= eval_b {
                                break;
                            }
                            if active_acquired_by_fold[k][j].insert(m_idx) {
                                newly_added += 1;
                            }
                        }
                    }

                    // Matched-budget passive control arm acquires the exact same count of unvisited candidates in outcome-blind hash order.
                    let mut pas_added = 0usize;
                    for &m_idx in &hash_ordered_indices[j] {
                        if pas_added >= newly_added {
                            break;
                        }
                        if passive_acquired_by_fold[k][j].insert(m_idx) {
                            pas_added += 1;
                        }
                    }
                }
            }
        }

        let heldout_by_budget: BTreeMap<usize, Summary> = metrics_for_round
            .iter()
            .map(|(&b, rows)| (b, summarize_metrics(rows)))
            .collect();
        let passive_heldout_by_budget: BTreeMap<usize, Summary> = passive_metrics_for_round
            .iter()
            .map(|(&b, rows)| (b, summarize_metrics(rows)))
            .collect();

        round_summaries.push(FlywheelRoundSummary {
            round: r,
            distilled_rows: reloaded.len(),
            parquet_bytes: parquet_bytes.len(),
            parquet_sha256,
            reconciled_parquet_keys: raw_keys.len(),
            mean_train_hwm_gain: train_hwm_gains.iter().sum::<f64>() / train_hwm_gains.len() as f64,
            passive_mean_train_hwm_gain: passive_train_hwm_gains.iter().sum::<f64>()
                / passive_train_hwm_gains.len() as f64,
            mean_heldout_active_discovery_hwm_gain: round_active_disc_hwm.iter().sum::<f64>()
                / round_active_disc_hwm.len() as f64,
            mean_heldout_passive_discovery_hwm_gain: round_passive_disc_hwm.iter().sum::<f64>()
                / round_passive_disc_hwm.len() as f64,
            kl_admissible_contexts,
            kl_frozen_contexts,
            max_observed_conditional_base_kl: max_cond_kl,
            max_observed_new_support_mass: max_new_mass,
            mean_archive_ce_loss: ce_losses.iter().sum::<f64>() / ce_losses.len() as f64,
            heldout_by_budget,
            passive_heldout_by_budget,
        });
        round_metrics_by_budget.push(metrics_for_round);
        passive_round_metrics_by_budget.push(passive_metrics_for_round);
        if r + 1 == cfg.rounds {
            final_train_sub_pools = train_sub_pools_by_fold;
        }
    }

    let monotonic_train_hwm_gain = round_summaries
        .windows(2)
        .all(|w| w[1].mean_train_hwm_gain + 1e-12 >= w[0].mean_train_hwm_gain);
    let primary_b = cfg.eval.budget;
    let monotonic_heldout_pairwise_accuracy = round_summaries.windows(2).all(|w| {
        match (
            w[0].heldout_by_budget[&primary_b].mean_pairwise_accuracy,
            w[1].heldout_by_budget[&primary_b].mean_pairwise_accuracy,
        ) {
            (Some(a), Some(b)) => b + 1e-12 >= a,
            _ => false,
        }
    });

    let last_r = cfg.rounds - 1;
    let mut round_lifts = Vec::new();
    let mut active_vs_passive_lifts = Vec::new();
    for &b in &cfg.eval_budgets {
        let m0 = &round_metrics_by_budget[0][&b];
        let m_last = &round_metrics_by_budget[last_r][&b];
        let gain_lift = paired_metrics(pools, m_last, m0, |m| Some(m.realized_gain), &cfg.eval)?
            .context("paired realized_gain_lift requires at least 2 contexts")?;
        let top1_lift = paired_metrics(pools, m_last, m0, |m| Some(m.top1_gain), &cfg.eval)?
            .context("paired top1_gain_lift requires at least 2 contexts")?;
        let pw_lift = paired_metrics(pools, m_last, m0, |m| m.pairwise_accuracy, &cfg.eval)?;
        round_lifts.push(RoundLiftComparison {
            budget: b,
            from_round: 0,
            to_round: last_r,
            realized_gain_lift: gain_lift,
            top1_gain_lift: top1_lift,
            pairwise_accuracy_lift: pw_lift,
        });

        let pm_last = &passive_round_metrics_by_budget[last_r][&b];
        let ap_gain_lift =
            paired_metrics(pools, m_last, pm_last, |m| Some(m.realized_gain), &cfg.eval)?.context(
                "paired active_vs_passive realized_gain_lift requires at least 2 contexts",
            )?;
        let ap_top1_lift =
            paired_metrics(pools, m_last, pm_last, |m| Some(m.top1_gain), &cfg.eval)?
                .context("paired active_vs_passive top1_gain_lift requires at least 2 contexts")?;
        let ap_pw_lift =
            paired_metrics(pools, m_last, pm_last, |m| m.pairwise_accuracy, &cfg.eval)?;
        active_vs_passive_lifts.push(RoundLiftComparison {
            budget: b,
            from_round: last_r,
            to_round: last_r,
            realized_gain_lift: ap_gain_lift,
            top1_gain_lift: ap_top1_lift,
            pairwise_accuracy_lift: ap_pw_lift,
        });
    }

    let active_vs_passive_discovery_lift = paired_scalars(
        pools,
        &final_active_discovery_hwm,
        &final_passive_discovery_hwm,
        &cfg.eval,
    )?;

    let final_vs_round0_primary_verdict = round_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .map(|l| l.realized_gain_lift.verdict)
        .unwrap_or(Verdict::Inconclusive);
    let final_vs_round0_pairwise_verdict = round_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .and_then(|l| l.pairwise_accuracy_lift.as_ref())
        .map(|c| c.verdict)
        .unwrap_or(Verdict::Inconclusive);

    let active_beats_passive_discovery_verdict = active_vs_passive_discovery_lift.verdict;
    let active_beats_passive_primary_verdict = active_vs_passive_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .map(|l| l.realized_gain_lift.verdict)
        .unwrap_or(Verdict::Inconclusive);
    let active_beats_passive_pairwise_verdict = active_vs_passive_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .and_then(|l| l.pairwise_accuracy_lift.as_ref())
        .map(|c| c.verdict)
        .unwrap_or(Verdict::Inconclusive);

    // Evaluate Round R-1's outer-fold distilled training sub-pools against the full held-out pools
    // at the primary budget so `final_round_receipt` matches Round R-1's held-out evaluation bit-for-bit.
    let eval_inputs = Inputs {
        context_features: Some(context_features),
        program_features,
        external: external.to_vec(),
        unavailable: Vec::new(),
    };
    let mut final_round_receipt = evaluate_with_fold_train_pools_and_priors(
        pools,
        Some(&final_train_sub_pools),
        fold_s3_priors.as_deref(),
        manifest,
        &eval_inputs,
        &cfg.eval,
    )?;
    if !round_cv_trace.is_empty() {
        if let Some(sel_rec) = final_round_receipt
            .selectors
            .iter_mut()
            .find(|s| s.name == "contextual_surrogate_qhat")
        {
            sel_rec.rule.push_str(&format!(
                "; flywheel nested/v3 fits={}; see typed surrogate_cv",
                round_cv_trace.len()
            ));
        }
    }
    let surrogate_beats_strongest_baseline = final_round_receipt
        .comparisons
        .iter()
        .find(|c| {
            c.candidate == "contextual_surrogate_qhat"
                && c.baseline == final_round_receipt.strongest_baseline
        })
        .map(|c| {
            c.realized_gain.verdict == Verdict::Greater
                && c.pairwise_accuracy.as_ref().map(|p| p.verdict) == Some(Verdict::Greater)
        })
        .unwrap_or(false);

    let active_discovery_established =
        all_rounds_kl_admissible && active_beats_passive_discovery_verdict == Verdict::Greater;
    let data_accumulation_lift_established = monotonic_train_hwm_gain
        && all_rounds_kl_admissible
        && final_vs_round0_primary_verdict == Verdict::Greater
        && final_vs_round0_pairwise_verdict == Verdict::Greater;
    let flywheel_established = active_discovery_established
        && data_accumulation_lift_established
        && surrogate_beats_strongest_baseline
        && active_beats_passive_primary_verdict == Verdict::Greater
        && active_beats_passive_pairwise_verdict == Verdict::Greater;

    Ok(FlywheelReceipt {
        schema: FLYWHEEL_RECEIPT_SCHEMA,
        shadow_only: true,
        qualified_claim: false,
        l6_claim: false,
        manifest_sha256,
        s3_prior_rows_by_fold,
        config: cfg.clone(),
        rounds: round_summaries,
        surrogate_cv: round_cv_trace,
        monotonic_train_hwm_gain,
        monotonic_heldout_pairwise_accuracy,
        all_rounds_kl_admissible,
        round_lifts,
        active_vs_passive_lifts,
        active_vs_passive_discovery_lift,
        final_round_receipt,
        surrogate_beats_strongest_baseline,
        final_vs_round0_primary_verdict,
        final_vs_round0_pairwise_verdict,
        active_beats_passive_discovery_verdict,
        active_beats_passive_primary_verdict,
        active_beats_passive_pairwise_verdict,
        active_discovery_established,
        data_accumulation_lift_established,
        flywheel_established,
    })
}
