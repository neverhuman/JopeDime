//! Multi-round `gp_distilled/v1` self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
//!
//! Each round `r = 0..R-1`:
//! 1. Distills the cumulative acquired self-play observations across contexts into a validated
//!    `gp_distilled/v1` Parquet shard (`retention::distilled::write`), reads the shard back
//!    (`retention::distilled::read`), verifies E3 key reconciliation (`retention::distilled::reconcile`),
//!    and bridges the rows into `MeasurementRow`s (`distilled_rows_to_measurement_rows`).
//! 2. Checks closed-loop measured-tilt leaf distributions (`measured_tilt_leaf_log_probs`) and
//!    `decomposed_support_kl` admissibility over each context's base vs newly acquired support.
//! 3. Fits `ContextualSurrogateQHat` per parent-grouped fold `k` using only training folds `!= k`
//!    from the Round-`r` distilled sub-pools, evaluates held-out fold `k` on the full frozen
//!    shadow pools across budgets `B`, and uses `pre_screen_funnel` to acquire the next batch of
//!    unmeasured candidate programs for Round `r + 1`.
//!
//! Shadow-only: never claims `qualified = true` or `l6`.

use super::bootstrap::{cluster_bootstrap_ci, PairedInterval, Verdict};
use super::evaluate::{evaluate, fold_of, Config as EvalConfig, Inputs, Summary};
use super::metrics::{score_context, ContextMetrics};
use super::pool::{verify, Manifest, Member, Pool, PoolView, Status, POOL_SCHEMA};
use super::selectors::{
    pre_screen_funnel, token_bag_features, ContextualSurrogateQHat, Features, Selector,
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

pub const FLYWHEEL_RECEIPT_SCHEMA: &str = "jope.j9-distilled-flywheel-receipt/v1";

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
    pub reconciled_s3_keys: usize,
    pub mean_train_hwm_gain: f64,
    pub kl_admissible_contexts: usize,
    pub max_observed_conditional_base_kl: f64,
    pub max_observed_new_support_mass: f64,
    pub mean_archive_ce_loss: f64,
    pub heldout_by_budget: BTreeMap<usize, Summary>,
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
pub struct FlywheelReceipt {
    pub schema: &'static str,
    pub shadow_only: bool,
    pub qualified_claim: bool,
    pub l6_claim: bool,
    pub manifest_sha256: String,
    pub config: FlywheelConfig,
    pub rounds: Vec<FlywheelRoundSummary>,
    pub monotonic_train_hwm_gain: bool,
    pub monotonic_heldout_pairwise_accuracy: bool,
    pub all_rounds_kl_admissible: bool,
    pub round_lifts: Vec<RoundLiftComparison>,
    pub final_round_receipt: super::evaluate::Receipt,
    pub final_vs_round0_primary_verdict: Verdict,
    pub flywheel_established: bool,
}

impl FlywheelReceipt {
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

fn known_scores() -> BTreeSet<String> {
    [
        "cv_geomean_rmse_ratio",
        "best_rust_rmse",
        "validation_rmse",
        "l6",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
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

pub fn run_distilled_flywheel(
    pools: &[Pool],
    manifest: &Manifest,
    context_features: &Features,
    program_features: Option<&Features>,
    external: &[&dyn Selector],
    cfg: &FlywheelConfig,
    parquet_dir: Option<&Path>,
) -> Result<FlywheelReceipt> {
    verify(pools, manifest)?;
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
            && cfg.max_new_support_mass >= cfg.tilt.policy_new_support_cap
            && cfg.max_new_support_mass < 1.0,
        "max_new_support_mass must be in [policy_new_support_cap, 1)"
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

    let folds: Vec<usize> = pools
        .iter()
        .map(|p| fold_of(&cfg.eval.fold_salt, &p.parent_id, cfg.eval.folds))
        .collect();
    let refs: Vec<f64> = pools
        .iter()
        .map(|p| p.reference(cfg.eval.reference))
        .collect::<Result<_>>()?;

    // Track acquired member indices per pool (and which were acquired in Round 0 as base support).
    let mut acquired: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); pools.len()];
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
        // Ensure at least one Complete member is present in Round 0 so every context has a finite reference.
        let mut chosen: Vec<usize> = order
            .iter()
            .take(cfg.initial_budget_per_context.min(pool.members.len()))
            .map(|&(_, idx)| idx)
            .collect();
        if !chosen
            .iter()
            .any(|&idx| pool.members[idx].outcome().is_some())
        {
            if let Some(&(_, first_ok)) = order
                .iter()
                .find(|&&(_, idx)| pool.members[idx].outcome().is_some())
            {
                if let Some(last) = chosen.last_mut() {
                    *last = first_ok;
                }
            }
        }
        for idx in chosen {
            acquired[i].insert(idx);
            base_acquired[i].insert(idx);
        }
    }

    let mut round_summaries = Vec::with_capacity(cfg.rounds);
    let mut round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>> =
        Vec::with_capacity(cfg.rounds);
    let mut all_rounds_kl_admissible = true;

    for r in 0..cfg.rounds {
        // 1. Distill all currently acquired observations into gp_distilled/v1 Parquet and read back.
        let mut round_distilled = Vec::new();
        let mut raw_keys = BTreeSet::new();
        let mut tokens_by_prog: BTreeMap<String, Option<Vec<u32>>> = BTreeMap::new();

        for (i, pool) in pools.iter().enumerate() {
            let spec = SelfPlayDistillSpec {
                dataset: pool.parent_id.clone(),
                context_sha256: pool.context_id.clone(),
                fold: Some(folds[i] as i64),
                split_role: SplitRole::Train,
                score_name: cfg.score_name.clone(),
                policylm_ckpt: None,
                campaign: Some("clover-s3-distilled-flywheel".into()),
                run_id: format!("flywheel-round-{r:02}"),
                family: "clover-selfplay".into(),
                source_bucket: "veox-jopedime-learning-use2".into(),
            };
            let mut obs = Vec::with_capacity(acquired[i].len());
            for &m_idx in &acquired[i] {
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
                    "experiments/clover-selfplay/round-{r:02}/{}/{}.json",
                    pool.context_id, m.program_sha256
                );
                let src_sha = sha256_hex(
                    format!("{}:{}:{}:{r}", pool.context_id, m.program_sha256, m_idx).as_bytes(),
                );
                raw_keys.insert(key.clone());
                obs.push(SelfPlayDistillObservation {
                    program_sha: m.program_sha256.clone(),
                    program: None,
                    outcome,
                    score: raw_score,
                    seed_packet: Some(if base_acquired[i].contains(&m_idx) {
                        "round0_cold_start".into()
                    } else {
                        format!("surrogate_funnel_up_to_r{r}")
                    }),
                    source_key: key,
                    source_version_id: Some(format!("r{r}")),
                    source_sha256: src_sha,
                });
            }
            let batch = distill_selfplay_observations(&spec, &obs, &scores_set)?;
            round_distilled.extend(batch);
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
        let m_rows = distilled_rows_to_measurement_rows(
            &reloaded,
            &cfg.score_name,
            &pools[0].protocol_sha256,
            &scores_set,
        )?;

        // Group reloaded MeasurementRows by context_id to form the Round-r distilled sub-pools.
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

        let mut sub_pools = Vec::with_capacity(pools.len());
        let mut train_hwm_gains = Vec::with_capacity(pools.len());
        let mut kl_admissible_contexts = 0usize;
        let mut max_cond_kl = 0.0f64;
        let mut max_new_mass = 0.0f64;
        let mut ce_losses = Vec::with_capacity(pools.len());

        for (i, pool) in pools.iter().enumerate() {
            let mut members = by_ctx
                .remove(&pool.context_id)
                .with_context(|| format!("missing distilled sub-pool for {}", pool.context_id))?;
            members.sort_by(|a, b| a.program_sha256.cmp(&b.program_sha256));
            let best_q = members
                .iter()
                .filter_map(Member::outcome)
                .fold(refs[i], f64::max);
            train_hwm_gains.push((best_q - refs[i]).max(0.0));

            // Evaluate measured-tilt leaf distribution and decomposed support KL on base vs newly acquired leaves.
            let base_shas: BTreeSet<&str> = base_acquired[i]
                .iter()
                .map(|&idx| pool.members[idx].program_sha256.as_str())
                .collect();
            let mut base_q = Vec::new();
            let mut arch_q = Vec::new();
            for m in &members {
                let q_val = m.outcome().unwrap_or(refs[i]);
                if base_shas.contains(m.program_sha256.as_str()) {
                    base_q.push(q_val);
                } else {
                    arch_q.push(q_val);
                }
            }
            let (pol_lp, ref_lp, ce) =
                measured_tilt_leaf_log_probs(&base_q, &arch_q, refs[i], &cfg.tilt)?;
            let is_base_leaf: Vec<bool> = (0..pol_lp.len()).map(|idx| idx < base_q.len()).collect();
            let kl = decomposed_support_kl(
                &pol_lp,
                &ref_lp,
                &is_base_leaf,
                cfg.max_conditional_base_kl,
                cfg.max_new_support_mass,
            )?;
            if kl.admissible {
                kl_admissible_contexts += 1;
            } else {
                all_rounds_kl_admissible = false;
            }
            max_cond_kl = max_cond_kl.max(kl.conditional_base_kl);
            max_new_mass = max_new_mass.max(kl.new_support_mass);
            ce_losses.push(ce);

            sub_pools.push(Pool {
                schema: POOL_SCHEMA.into(),
                context_id: pool.context_id.clone(),
                parent_id: pool.parent_id.clone(),
                role: pool.role.clone(),
                protocol_sha256: pool.protocol_sha256.clone(),
                incumbent_q: pool.incumbent_q,
                members,
            });
        }

        // 2. Cross-fit ContextualSurrogateQHat on Round-r sub_pools and evaluate on full held-out pools.
        let mut metrics_for_round: BTreeMap<usize, Vec<ContextMetrics>> = cfg
            .eval_budgets
            .iter()
            .map(|&b| {
                (
                    b,
                    vec![
                        ContextMetrics {
                            realized_gain: 0.0,
                            top1_gain: 0.0,
                            regret: 0.0,
                            normalized_regret: None,
                            beats_reference: 0.0,
                            pairwise_accuracy: None,
                        };
                        pools.len()
                    ],
                )
            })
            .collect();
        let mut next_acquisitions: Vec<Vec<usize>> = vec![Vec::new(); pools.len()];

        for k in 0..cfg.eval.folds {
            let held: Vec<usize> = (0..pools.len()).filter(|&i| folds[i] == k).collect();
            let train: Vec<&Pool> = (0..pools.len())
                .filter(|&i| folds[i] != k)
                .map(|i| &sub_pools[i])
                .collect();
            if held.is_empty() {
                continue;
            }
            ensure!(!train.is_empty(), "fold {k} has no training pools");
            let surrogate = ContextualSurrogateQHat::fit(
                &train,
                context_features,
                pf,
                cfg.eval.knn_k,
                cfg.eval.prior_weight,
                cfg.eval.ridge_lambda,
            )?;

            for &i in &held {
                let full_pool = &pools[i];
                let view = full_pool.view();
                let scores = surrogate.scores(&view)?;
                let outcomes = full_pool.outcomes();
                for &b in &cfg.eval_budgets {
                    let m = score_context(&outcomes, refs[i], &scores, &view.programs, b);
                    metrics_for_round.get_mut(&b).unwrap()[i] = m;
                }

                // Use pre_screen_funnel over the remaining unmeasured candidates to acquire for Round r+1.
                let unmeasured_indices: Vec<usize> = (0..full_pool.members.len())
                    .filter(|m_idx| !acquired[i].contains(m_idx))
                    .collect();
                if !unmeasured_indices.is_empty() {
                    let unmeasured_progs: Vec<&str> = unmeasured_indices
                        .iter()
                        .map(|&m_idx| full_pool.members[m_idx].program_sha256.as_str())
                        .collect();
                    let unmeasured_tokens: Vec<Option<&[u32]>> = unmeasured_indices
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
                    let eval_b = cfg.step_budget_per_context.min(unmeasured_indices.len());
                    let prop_k = cfg.proposal_pool_k.max(eval_b);
                    let funnel = pre_screen_funnel(&rem_view, &surrogate, prop_k, eval_b)?;
                    for sel_sha in &funnel.selected_program_sha256 {
                        if let Ok(m_idx) = full_pool
                            .members
                            .binary_search_by(|m| m.program_sha256.as_str().cmp(sel_sha.as_str()))
                        {
                            next_acquisitions[i].push(m_idx);
                        }
                    }
                }
            }
        }

        let heldout_by_budget: BTreeMap<usize, Summary> = metrics_for_round
            .iter()
            .map(|(&b, rows)| (b, summarize_metrics(rows)))
            .collect();

        round_summaries.push(FlywheelRoundSummary {
            round: r,
            distilled_rows: reloaded.len(),
            parquet_bytes: parquet_bytes.len(),
            parquet_sha256,
            reconciled_s3_keys: raw_keys.len(),
            mean_train_hwm_gain: train_hwm_gains.iter().sum::<f64>() / train_hwm_gains.len() as f64,
            kl_admissible_contexts,
            max_observed_conditional_base_kl: max_cond_kl,
            max_observed_new_support_mass: max_new_mass,
            mean_archive_ce_loss: ce_losses.iter().sum::<f64>() / ce_losses.len() as f64,
            heldout_by_budget,
        });
        round_metrics_by_budget.push(metrics_for_round);

        for (i, next) in next_acquisitions.into_iter().enumerate() {
            acquired[i].extend(next);
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
    }

    let final_vs_round0_primary_verdict = round_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .map(|l| l.realized_gain_lift.verdict)
        .unwrap_or(Verdict::Inconclusive);

    // Run full frozen-pool evaluation at the primary budget for the final receipt comparison against baselines & externals.
    let eval_inputs = Inputs {
        context_features: Some(context_features),
        program_features: Some(pf),
        external: external.to_vec(),
        unavailable: Vec::new(),
    };
    let final_round_receipt = evaluate(pools, manifest, &eval_inputs, &cfg.eval)?;
    let flywheel_established = monotonic_train_hwm_gain
        && all_rounds_kl_admissible
        && final_vs_round0_primary_verdict == Verdict::Greater
        && final_round_receipt.ranking_learnability == "ESTABLISHED_SHADOW";

    Ok(FlywheelReceipt {
        schema: FLYWHEEL_RECEIPT_SCHEMA,
        shadow_only: true,
        qualified_claim: false,
        l6_claim: false,
        manifest_sha256: manifest.sha256()?,
        config: cfg.clone(),
        rounds: round_summaries,
        monotonic_train_hwm_gain,
        monotonic_heldout_pairwise_accuracy,
        all_rounds_kl_admissible,
        round_lifts,
        final_round_receipt,
        final_vs_round0_primary_verdict,
        flywheel_established,
    })
}
