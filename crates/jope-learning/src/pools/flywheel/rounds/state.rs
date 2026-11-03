use super::*;

pub(super) struct RoundState {
    pub(super) folds: Vec<usize>,
    pub(super) refs: Vec<f64>,
    pub(super) pool_idx_by_ctx: BTreeMap<String, usize>,
    pub(super) hash_ordered_indices: Vec<Vec<usize>>,
    pub(super) base_acquired: Vec<BTreeSet<usize>>,
    pub(super) active_acquired_by_fold: Vec<Vec<BTreeSet<usize>>>,
    pub(super) passive_acquired_by_fold: Vec<Vec<BTreeSet<usize>>>,
    pub(super) heldout_active_acquired: Vec<BTreeSet<usize>>,
    pub(super) heldout_passive_acquired: Vec<BTreeSet<usize>>,
    pub(super) round_summaries: Vec<FlywheelRoundSummary>,
    pub(super) round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>>,
    pub(super) passive_round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>>,
    pub(super) final_active_discovery_hwm: Vec<f64>,
    pub(super) final_passive_discovery_hwm: Vec<f64>,
    pub(super) all_rounds_kl_admissible: bool,
    pub(super) final_train_sub_pools: Vec<Vec<Pool>>,
    pub(super) round_cv_trace: Vec<FlywheelFoldCv>,
}

impl RoundState {
    pub(super) fn new(pools: &[Pool], cfg: &FlywheelConfig) -> Result<Self> {
        let folds: Vec<usize> = pools
            .iter()
            .map(|p| fold_of(&cfg.eval.fold_salt, &p.parent_id, cfg.eval.folds))
            .collect();
        let refs: Vec<f64> = pools
            .iter()
            .map(|p| p.reference(cfg.eval.reference))
            .collect::<Result<_>>()?;
        let pool_idx_by_ctx: BTreeMap<String, usize> = pools
            .iter()
            .enumerate()
            .map(|(i, p)| (p.context_id.clone(), i))
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
        let active_acquired_by_fold: Vec<Vec<BTreeSet<usize>>> =
            vec![base_acquired.clone(); cfg.eval.folds];
        let passive_acquired_by_fold: Vec<Vec<BTreeSet<usize>>> =
            vec![base_acquired.clone(); cfg.eval.folds];

        // Strictly out-of-fold held-out discovery trajectories per context `i` (updated only by outer
        // fold `k = folds[i]`'s surrogate vs passive hash order; never read by any training fold).
        let heldout_active_acquired: Vec<BTreeSet<usize>> = base_acquired.clone();
        let heldout_passive_acquired: Vec<BTreeSet<usize>> = base_acquired.clone();

        let round_summaries = Vec::with_capacity(cfg.rounds);
        let round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>> =
            Vec::with_capacity(cfg.rounds);
        let passive_round_metrics_by_budget: Vec<BTreeMap<usize, Vec<ContextMetrics>>> =
            Vec::with_capacity(cfg.rounds);
        let final_active_discovery_hwm = vec![0.0; pools.len()];
        let final_passive_discovery_hwm = vec![0.0; pools.len()];
        let all_rounds_kl_admissible = true;
        let final_train_sub_pools: Vec<Vec<Pool>> = Vec::new();
        let round_cv_trace: Vec<FlywheelFoldCv> = Vec::with_capacity(cfg.rounds * cfg.eval.folds);

        Ok(Self {
            folds,
            refs,
            pool_idx_by_ctx,
            hash_ordered_indices,
            base_acquired,
            active_acquired_by_fold,
            passive_acquired_by_fold,
            heldout_active_acquired,
            heldout_passive_acquired,
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
}
