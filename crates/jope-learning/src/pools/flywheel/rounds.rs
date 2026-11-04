use super::*;

pub(super) struct RoundResults {
    pub(super) round_summaries: Vec<FlywheelRoundSummary>,
    pub(super) round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>>,
    pub(super) passive_round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>>,
    pub(super) final_active_discovery_hwm: Vec<f64>,
    pub(super) final_passive_discovery_hwm: Vec<f64>,
    pub(super) all_rounds_kl_admissible: bool,
    pub(super) final_train_sub_pools: Vec<Vec<Pool>>,
    pub(super) round_cv_trace: Vec<FlywheelFoldCv>,
}

mod state;
use state::RoundState;

pub(super) struct RoundInput<'a> {
    pub(super) pools: &'a [Pool],
    pub(super) context_features: &'a Features,
    pub(super) pf: &'a Features,
    pub(super) scores_set: &'a BTreeSet<String>,
    pub(super) direction: ScoreDirection,
    pub(super) cfg: &'a FlywheelConfig,
    pub(super) parquet_dir: Option<&'a Path>,
    pub(super) fold_s3_priors: Option<&'a [ExternalProgramPrior]>,
}

pub(super) fn run_rounds(input: RoundInput<'_>) -> Result<RoundResults> {
    let RoundInput {
        pools,
        context_features,
        pf,
        scores_set,
        direction,
        cfg,
        parquet_dir,
        fold_s3_priors,
    } = input;
    let RoundState {
        folds,
        refs,
        pool_idx_by_ctx,
        hash_ordered_indices,
        base_acquired,
        mut active_acquired_by_fold,
        mut passive_acquired_by_fold,
        mut heldout_active_acquired,
        mut heldout_passive_acquired,
        mut round_summaries,
        mut round_metrics_by_budget,
        mut passive_round_metrics_by_budget,
        mut final_active_discovery_hwm,
        mut final_passive_discovery_hwm,
        mut all_rounds_kl_admissible,
        mut final_train_sub_pools,
        mut round_cv_trace,
    } = RoundState::new(pools, cfg)?;

    for r in 0..cfg.rounds {
        let distilled_round = super::distill::distill_round(
            pools,
            &folds,
            &active_acquired_by_fold,
            &base_acquired,
            cfg,
            direction,
            scores_set,
            parquet_dir,
            r,
        )?;
        let super::distill::DistilledRound {
            reloaded,
            raw_keys,
            parquet_bytes,
            parquet_sha256,
            tokens_by_prog,
        } = distilled_round;

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
                scores_set,
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
            let ext_prior = fold_s3_priors.map(|ps| &ps[k]);
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

    Ok(RoundResults {
        round_summaries,
        round_metrics_by_budget,
        passive_round_metrics_by_budget,
        final_active_discovery_hwm,
        final_passive_discovery_hwm,
        all_rounds_kl_admissible,
        final_train_sub_pools,
        round_cv_trace,
    })
}
