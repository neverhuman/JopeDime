#!/usr/bin/env python3
"""H436 hybrid core: key lookup with a model backoff, and guarded columns (follow-up to H430; numpy, lightgbm; learners and level
handling shared with H430's keys_core).

H430 found two defects: a lookup that falls back to the fit mean is scored too pessimistically out of fold (a key
value with a single fit row has nothing to look up when that row is held out), and its ridge fallback had no output
guard. Here the lookup falls back to a tree model's prediction, the group mean is shrunk toward that prediction, and
every column is clipped to the fit target range widened by half its width.

  heads(X, y, Xq, seed)   the three column predictions for the query rows:
      key_hybrid   shrunk group mean of the best key where the key value is known, the model's prediction where it
                   is not; equal to gbm_plain when no key improves on the model
      ridge_z      ridge on z-scored features, penalty by generalised cross-validation, clipped
      gbm_plain    the backoff model itself (control)
"""
from __future__ import annotations

import numpy as np

import keys_core as K

M_GRID = (0.0, 0.5, 2.0, 8.0)
ACCEPT = 0.98          # a key is used only if its leave-one-out RMSE is below this fraction of the model's
PAIR_POOL = 12
INNER = 5
NAMES = ("key_hybrid", "ridge_z", "gbm_plain")


def _loo(inv, y, G, m, back):
    """Leave-one-out group mean shrunk toward the model's out-of-fold prediction; the model alone for a single row."""
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=G)
    ci = cnt[inv]
    den = ci - 1.0 + m
    return np.where(ci >= 2, (s[inv] - y + m * back) / np.where(den > 0, den, 1.0), back)


def _best_m(inv, y, G, back):
    best = None
    for m in M_GRID:
        r = K._rmse(_loo(inv, y, G, m, back), y)
        if best is None or r < best[0]:
            best = (r, m)
    return best


def choose_key(X: np.ndarray, y: np.ndarray, back: np.ndarray):
    """The key (one column or a pair among the best single keys) with the lowest leave-one-out RMSE when unknown or
    single-row key values fall back to `back`; None when no key beats the model by the ACCEPT margin."""
    n = len(y)
    r0 = K._rmse(back, y)
    single = {}
    for j in range(X.shape[1]):
        _tab, inv, G = K._levels((j,), X)
        if K._eligible(inv, G, n):
            single[j] = _best_m(inv, y, G, back)
    cands = [((j,), r, m) for j, (r, m) in single.items()]
    pool = sorted(single, key=lambda j: single[j][0])[:PAIR_POOL]
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            cols = (min(a, b), max(a, b))
            _tab, inv, G = K._levels(cols, X)
            if K._eligible(inv, G, n):
                r, m = _best_m(inv, y, G, back)
                cands.append((cols, r, m))
    if not cands:
        return None, r0
    cols, r, m = min(cands, key=lambda c: (c[1], c[0]))
    if not r < ACCEPT * r0:
        return None, r0
    return {"cols": cols, "m": m, "loo_rmse": r}, r0


def lookup(spec: dict, X: np.ndarray, y: np.ndarray, Xq: np.ndarray, back_q: np.ndarray) -> np.ndarray:
    tab, inv, G = K._levels(tuple(spec["cols"]), X)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=G)
    idx = K._index(tab, tuple(spec["cols"]), Xq)
    hit = idx >= 0
    i = np.where(hit, idx, 0)
    return np.where(hit, (s[i] + spec["m"] * back_q) / (cnt[i] + spec["m"]), back_q)


def _clip(p, y):
    lo, hi = float(y.min()), float(y.max())
    pad = 0.5 * (hi - lo)
    return np.clip(np.where(np.isfinite(p), p, float(y.mean())), lo - pad, hi + pad)


def heads(Xf: np.ndarray, y: np.ndarray, Xq: np.ndarray, seed: int):
    Xf, Xq, y = np.asarray(Xf, np.float64), np.asarray(Xq, np.float64), np.asarray(y, np.float64)
    n = len(y)
    mq = K.gbm(Xf, y, Xq, seed)                       # the backoff model on all fit rows
    f = K._inner_folds(n, seed + 3, INNER)
    moof = np.empty(n)
    for k in range(INNER):                            # its out-of-fold prediction of the fit rows
        tr, te = f != k, f == k
        moof[te] = K.gbm(Xf[tr], y[tr], Xf[te], seed + 100 + k)
    spec, r0 = choose_key(Xf, y, moof)
    sd = float(y.std()) or 1.0
    info = {"model_oof_u": float(-np.log(max(r0, 1e-12 * sd) / sd)), "key": None}
    out = {"gbm_plain": mq}
    if spec is None:
        out["key_hybrid"] = mq
    else:
        out["key_hybrid"] = lookup(spec, Xf, y, Xq, mq)
        info["key"] = {"cols": list(spec["cols"]), "m": spec["m"],
                       "loo_u": float(-np.log(max(spec["loo_rmse"], 1e-12 * sd) / sd))}
    out["ridge_z"] = K.ridge_plain(Xf, y, Xq)
    return {k: _clip(out[k], y) for k in NAMES}, info
