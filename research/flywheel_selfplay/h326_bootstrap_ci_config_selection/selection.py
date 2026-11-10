"""H326 family-bootstrap median config selection; no data or model fitting."""
import numpy as np

INNER_DRAWS = 2000
INNER_SEED = 326001
MARGIN = 0.002
TIE = 1e-4


class CacheCoverageError(ValueError):
    pass


def require_grid_cache(picks, cids, expected_cids, configs, expected_configs):
    """Selected-config OOF alone cannot stand in for the eighteen-config grid."""
    if list(configs) != list(expected_configs) or len(configs) != 18:
        raise CacheCoverageError('MISSING_GRID_OOF: exact eighteen-config order required')
    if list(cids) != list(expected_cids) or len(set(cids)) != len(cids):
        raise CacheCoverageError('CONTEXT_CLOSURE_MISMATCH')
    a = np.asarray(picks)
    if a.shape != (18, len(cids)) or a.dtype.kind not in 'iu' or not ((a >= 0) & (a < 8)).all():
        raise CacheCoverageError('MISSING_GRID_OOF: expected int picks[18, outer_train_contexts]')
    return a


def choose(fold_gains, configs, switches, base_index):
    """Four C×F_i family-mean gain matrices; disjoint, lexicographically ordered families.

    Each draw resamples within each inner fold with RNG326001+i and its original
    family count. Config scores are medians over draws of pooled family medians.
    The acceptance distribution is the median of paired family differences using
    exactly the same family resamples. BASE is the unchanged H303 mean selector.
    """
    if len(fold_gains) != 4 or len(configs) != 18 or len(switches) != 18:
        raise ValueError('Expected four inner folds and eighteen configs')
    arrays = [np.asarray(x, dtype=np.float64) for x in fold_gains]
    if any(x.ndim != 2 or x.shape[0] != 18 or x.shape[1] == 0 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid family gain matrix')
    if not 0 <= base_index < 18:
        raise ValueError('Invalid BASE config index')
    resampled = []
    for i, x in enumerate(arrays):
        idx = np.random.default_rng(INNER_SEED + i).integers(0, x.shape[1], (INNER_DRAWS, x.shape[1]))
        resampled.append(x[:, idx])
    samples = np.concatenate(resampled, axis=2)
    scores = np.median(np.median(samples, axis=2), axis=1)
    best = float(scores.max())
    tied = [i for i, s in enumerate(scores) if s >= best - TIE]
    star = min(tied, key=lambda i: (int(switches[i]), -configs[i][1], -configs[i][2], configs[i][0]))
    pooled = np.concatenate(arrays, axis=1)
    point = float(np.median(pooled[star] - pooled[base_index]))
    boot = np.median(samples[star] - samples[base_index], axis=1)
    lcb = float(np.quantile(boot, 0.05))
    accept = bool(star != base_index and point >= MARGIN and lcb > 0)
    return dict(candidate_index=int(star), selected_index=int(star if accept else base_index), base_index=int(base_index),
                accepted=accept, abstained=not accept, paired_median=point, lcb95=lcb,
                bootstrap_median_scores=scores.tolist(), tied_candidates=tied,
                draws=INNER_DRAWS, rng_by_inner_fold=[INNER_SEED + i for i in range(4)], margin=MARGIN)


def decide(paired, gains, harmed, abstain_folds):
    pc = np.asarray(paired['per_cell'], dtype=np.float64).reshape(3, 3)
    seeds = int(sum(np.median(pc[:, s]) > 0 for s in range(3)))
    crit = dict(paired_drop10_margin_and_lcb=bool(paired['median'] >= MARGIN and paired['lcb95'] > 0),
                gain_retention_90pct=bool(np.median(gains['BSEL']) >= 0.9 * np.median(gains['BASE'])),
                harmed_nonincrease=bool(np.median(harmed['BSEL']) <= np.median(harmed['BASE'])),
                positive_seeds_ge2=bool(seeds >= 2))
    dominant = abstain_folds >= 30
    return dict(verdict='ADVANCE' if all(crit.values()) and not dominant else 'BSEL_NOT_ADVANCED',
                criteria=crit, seeds_positive=seeds, abstain_folds=int(abstain_folds), abstain_dominant=bool(dominant),
                label='ABSTAIN_DOMINANT' if dominant else None)
