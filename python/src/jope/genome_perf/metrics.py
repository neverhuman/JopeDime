"""Evaluation protocol. One definition of every reported number.

Three decisions here are load-bearing and easy to get wrong:

1. Constant-score datasets are EXCLUDED from the mean, not scored as 0.0. Spearman is undefined
   when either side has no variance; `fitness_corrector.protocol.spearman` returns 0.0 in that
   case, which silently drags the mean toward zero in proportion to how many degenerate datasets
   the corpus holds -- and ~39% of featmath datasets have a score range below 0.01.

2. Metrics are reported on SNR strata. Two thirds of featmath datasets have a within-dataset score
   spread smaller than the measurement noise on a single row's score, so a pooled mean over all
   datasets understates a good model and rewards a lucky one.

3. Pooled Pearson r across rows is ATTENUATED BY CONSTRUCTION and must not be headlined. With a
   uniform-within-group target, Var_between(y) = 0, so any between-dataset variance in the
   prediction is pure noise that only inflates the denominator. `pooled_r_centered` -- correlation
   after centering predictions per dataset -- is the interpretable single number.
"""
from __future__ import annotations

import numpy as np

from . import STUDY_SEED

MIN_GROUP = 5


def _midrank(v: np.ndarray) -> np.ndarray:
    """Tie-aware ranks (average rank within each tie block)."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman. Returns nan (not 0.0) when either side is constant -- undefined."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _midrank(np.asarray(a, float)), _midrank(np.asarray(b, float))
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def per_dataset_spearman(dataset: np.ndarray, pred: np.ndarray, truth: np.ndarray,
                         min_group: int = MIN_GROUP) -> dict[str, float]:
    """-> {dataset: rho}, omitting groups that are too small or degenerate."""
    out: dict[str, float] = {}
    for d in np.unique(dataset):
        m = dataset == d
        if m.sum() < min_group:
            continue
        rho = spearman(pred[m], truth[m])
        if np.isfinite(rho):
            out[d] = rho
    return out


def capture_at_k(dataset: np.ndarray, pred: np.ndarray, truth: np.ndarray, k: int = 1) -> float:
    """Fraction of datasets where the true best candidate is inside the model's top-k."""
    hits, n = 0, 0
    for d in np.unique(dataset):
        m = dataset == d
        if m.sum() < MIN_GROUP:
            continue
        t = truth[m]
        if t.std() < 1e-12:
            continue
        topk = np.argsort(-pred[m], kind="stable")[:k]
        hits += int(np.argmax(t) in set(topk.tolist()))
        n += 1
    return hits / n if n else float("nan")


def regret_at_1(dataset: np.ndarray, pred: np.ndarray, score: np.ndarray) -> float:
    """Mean gap between the best achievable score and the score of the model's top pick.

    Deliberately computed on the ABSOLUTE clipped score, not the percentile: regret in percentile
    units is meaningless, regret in r2 units is the decision-relevant quantity.
    """
    gaps = []
    for d in np.unique(dataset):
        m = dataset == d
        if m.sum() < MIN_GROUP:
            continue
        s = score[m]
        gaps.append(float(s.max() - s[int(np.argmax(pred[m]))]))
    return float(np.mean(gaps)) if gaps else float("nan")


def pooled_r_centered(dataset: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> float:
    """Pearson r after centering predictions within each dataset -- the honest single 'r'.

    Raw pooled r is attenuated by sqrt(Var_within(p)/Var_total(p)) because the target carries zero
    between-dataset variance. Centering removes the term that can only hurt.
    """
    p = pred.astype(np.float64).copy()
    t = truth.astype(np.float64).copy()
    for d in np.unique(dataset):
        m = dataset == d
        p[m] -= p[m].mean()
        t[m] -= t[m].mean()
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def evaluate(dataset: np.ndarray, pred: np.ndarray, truth_pct: np.ndarray, score: np.ndarray,
             snr: dict[str, float] | None = None) -> dict:
    """Full metric block for one arm, including the SNR strata."""
    per = per_dataset_spearman(dataset, pred, truth_pct)
    out = {
        "spearman_mean": float(np.mean(list(per.values()))) if per else float("nan"),
        "spearman_median": float(np.median(list(per.values()))) if per else float("nan"),
        "n_datasets_scored": len(per),
        "capture_at_1": capture_at_k(dataset, pred, truth_pct, 1),
        "capture_at_3": capture_at_k(dataset, pred, truth_pct, 3),
        "regret_at_1_r2": regret_at_1(dataset, pred, score),
        "pooled_r_centered": pooled_r_centered(dataset, pred, truth_pct),
        "pooled_r_raw_attenuated": float(np.corrcoef(pred, truth_pct)[0, 1]),
    }
    if snr:
        for name, lo, hi in (("resolvable", 2.0, np.inf), ("mid", 1.0, 2.0), ("noisy", -np.inf, 1.0)):
            vals = [r for d, r in per.items() if lo <= snr.get(d, np.nan) < hi]
            out[f"spearman_{name}"] = float(np.mean(vals)) if vals else float("nan")
            out[f"n_{name}"] = len(vals)
    return out


def paired_bootstrap_delta(dataset: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                           truth: np.ndarray, n_boot: int = 2000,
                           seed: int = STUDY_SEED) -> dict:
    """95% CI on mean-Spearman(a) - mean-Spearman(b), resampling DATASETS (the unit of analysis).

    Paired, because the arms share folds -- an unpaired comparison of two independent CIs is both
    wrong and 2-3x wider.
    """
    a = per_dataset_spearman(dataset, pred_a, truth)
    b = per_dataset_spearman(dataset, pred_b, truth)
    common = sorted(set(a) & set(b))
    if not common:
        return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "excludes_zero": False, "n": 0}
    d = np.array([a[k] - b[k] for k in common])
    rng = np.random.RandomState(seed)
    boot = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": len(common)}
