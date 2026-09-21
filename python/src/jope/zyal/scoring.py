"""Fitness: does a candidate feature block add lift over the frozen champion?

The only question this module answers is *marginal* value. A feature that correlates with the
target but duplicates something the 1,331-feature baseline already has is worth nothing, and the
whole campaign turns into an expensive way to rediscover existing columns unless that is measured
directly.

Two rules that are easy to get wrong and expensive to get wrong:

* **Screen and confirm on disjoint fold structures.** Screening ranks thousands of candidates on
  cheap 2-fold CV. Confirming on the same folds would select on noise and report the selection as
  a result. Confirmation re-runs at 5 folds with a different fold seed.

* **Survival requires a paired-bootstrap CI over DATASETS that excludes zero**, not a point
  estimate that happens to be higher. With ~2,078 groups and per-dataset Spearman std around 0.4,
  the SE of the mean is ~0.009 -- so raw deltas below ~0.02 are indistinguishable from noise, and
  a campaign generating thousands of candidates will produce many of them by chance alone.
"""

from __future__ import annotations

import numpy as np

from jope.genome_perf.metrics import capture_at_k, paired_bootstrap_delta, per_dataset_spearman

from . import CHAMPION, TIE_BAND

SCREEN_FOLDS = 2
CONFIRM_FOLDS = 5
SCREEN_FOLD_SEED = 20260720
CONFIRM_FOLD_SEED = 771_7919  # deliberately different from the screen seed


def _fold_map(datasets: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """GroupKFold by dataset: every row of a dataset lands in the same fold, always."""
    uniq = np.unique(datasets)
    rng = np.random.RandomState(seed)
    assign = {d: i % n_folds for i, d in enumerate(rng.permutation(uniq))}
    return np.array([assign[d] for d in datasets])


def _oof_predict(
    X: np.ndarray, y: np.ndarray, datasets: np.ndarray, folds: np.ndarray, n_folds: int, n_estimators: int
) -> np.ndarray:
    import lightgbm as lgb

    oof = np.zeros(len(y), dtype=np.float64)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not tr.any() or not te.any():
            continue
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=25,
            colsample_bytree=0.4,
            subsample=0.85,
            subsample_freq=1,
            # LightGBM's PyPI wheel here is CPU-only; device_type='gpu' does not error but runs
            # SLOWER than CPU (155 s vs 92 s measured). Pinning it stops anyone rediscovering that.
            device_type="cpu",
            deterministic=True,
            force_row_wise=True,
            n_jobs=8,
            random_state=SCREEN_FOLD_SEED,
            verbose=-1,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return oof


def screen(
    base: np.ndarray, candidate: np.ndarray, y: np.ndarray, datasets: np.ndarray, score: np.ndarray
) -> dict:
    """Cheap 2-fold pass: does base+candidate beat base alone? Returns raw deltas, no CI."""
    folds = _fold_map(datasets, SCREEN_FOLDS, SCREEN_FOLD_SEED)
    pred_base = _oof_predict(base, y, datasets, folds, SCREEN_FOLDS, 200)
    pred_aug = _oof_predict(np.hstack([base, candidate]), y, datasets, folds, SCREEN_FOLDS, 200)

    sp_base = per_dataset_spearman(datasets, pred_base, y)
    sp_aug = per_dataset_spearman(datasets, pred_aug, y)
    common = sorted(set(sp_base) & set(sp_aug))
    delta = float(np.mean([sp_aug[k] - sp_base[k] for k in common])) if common else float("nan")
    return {
        "spearman_base": float(np.mean(list(sp_base.values()))) if sp_base else float("nan"),
        "spearman_aug": float(np.mean(list(sp_aug.values()))) if sp_aug else float("nan"),
        "delta_spearman": delta,
        "capture10_aug": capture_at_k(datasets, pred_aug, y, 10),
        "n_candidate_cols": int(candidate.shape[1]),
        "passes_tie_band": bool(delta > TIE_BAND),
    }


def confirm(
    base: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    datasets: np.ndarray,
    score: np.ndarray,
    n_boot: int = 2000,
) -> dict:
    """Expensive 5-fold pass with a paired bootstrap over datasets. This decides survival.

    Different fold seed from `screen` on purpose: confirming on the folds a candidate was selected
    on would launder selection noise into a result.
    """
    folds = _fold_map(datasets, CONFIRM_FOLDS, CONFIRM_FOLD_SEED)
    pred_base = _oof_predict(base, y, datasets, folds, CONFIRM_FOLDS, 500)
    pred_aug = _oof_predict(np.hstack([base, candidate]), y, datasets, folds, CONFIRM_FOLDS, 500)

    boot = paired_bootstrap_delta(datasets, pred_aug, pred_base, y, n_boot=n_boot)
    sp_aug = per_dataset_spearman(datasets, pred_aug, y)
    sp_base = per_dataset_spearman(datasets, pred_base, y)
    aug_mean = float(np.mean(list(sp_aug.values()))) if sp_aug else float("nan")

    return {
        "spearman_base": float(np.mean(list(sp_base.values()))) if sp_base else float("nan"),
        "spearman_aug": aug_mean,
        "delta": boot["delta"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "excludes_zero": boot["excludes_zero"],
        "n_datasets": boot["n"],
        "capture10_base": capture_at_k(datasets, pred_base, y, 10),
        "capture10_aug": capture_at_k(datasets, pred_aug, y, 10),
        "beats_champion_spearman": bool(aug_mean > CHAMPION["within_dataset_spearman"] + TIE_BAND),
        # A point estimate above the champion is NOT survival; the CI must exclude zero.
        "survives": bool(boot["excludes_zero"] and boot["delta"] > TIE_BAND),
    }


def categorical_feedback(result: dict, redundancy: float, seconds: float) -> dict:
    """Convert numeric screening results into the ONLY thing an agent may be told.

    `assert_categorical_only` is applied by the caller before this reaches a prompt; returning
    strings here is not sufficient protection on its own, it is the first half of it.
    """
    delta = result.get("delta_spearman", 0.0)
    if not np.isfinite(delta):
        lift = "neutral"
    elif delta > 0.01:
        lift = "positive"
    elif delta < -0.005:
        lift = "negative"
    else:
        lift = "neutral"
    return {
        "lift": lift,
        "redundancy": "high" if redundancy > 0.9 else "medium" if redundancy > 0.6 else "low",
        "cost": "slow" if seconds > 60 else "fast",
        "novelty": "redundant" if redundancy > 0.95 else "novel",
    }
