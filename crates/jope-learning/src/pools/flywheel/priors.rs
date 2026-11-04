use super::*;

/// Performs the known scores operation.
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
/// 1. `row.split_role == SplitRole::Train` and `row.outcome == DistilledOutcome::Ok`, and
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
        if r.split_role != SplitRole::Train || r.outcome != DistilledOutcome::Ok {
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

pub(super) fn summarize_metrics(rows: &[ContextMetrics]) -> Summary {
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

pub(super) fn paired_metrics(
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

pub(super) fn paired_scalars(
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
