#!/usr/bin/env python3
"""H430 unit checks of keys_core on synthetic tables only (no real table is opened).  python3 -B test_keys_core.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keys_core as K  # noqa: E402


def u_of(pred, y, y_fit):
    return float(-np.log(max(np.sqrt(np.mean((pred - y) ** 2)), 1e-300) / np.sqrt(np.mean((y - y_fit.mean()) ** 2))))


def key_table(rng, n=1600, groups=200, extra=12, noise=0.0):
    g = rng.integers(0, groups, n)
    eff = rng.normal(size=groups)
    lab = rng.permutation(groups * 3)[:groups].astype(float)          # unordered integer labels
    X = np.column_stack([lab[g]] + [rng.normal(size=n) for _ in range(extra)]
                        + [rng.integers(0, 4, n).astype(float) for _ in range(3)])
    return X, eff[g] + noise * rng.normal(size=n)


def test_exact_key():
    rng = np.random.default_rng(1)
    X, y = key_table(rng)
    f, q = np.arange(len(y)) % 5 != 0, np.arange(len(y)) % 5 == 0
    out, info = K.heads(X[f], y[f], X[q], 11)
    assert info["direct"] and info["best"]["cols"] == [0], info
    seen = np.isin(X[q, 0], X[f, 0])
    assert u_of(out["key_direct"][seen], y[q][seen], y[f]) >= 6.0
    assert u_of(out["key_gbm"], y[q], y[f]) > u_of(out["gbm_plain"], y[q], y[f]) + 1.0
    assert abs(u_of(out["key_noise"], y[q], y[f]) - u_of(out["gbm_plain"], y[q], y[f])) < 0.3
    return "exact key: key_direct u >= 6 on seen keys; key_gbm beats gbm_plain by > 1 u; key_noise ~ gbm_plain"


def test_no_structure():
    worst = -9.0
    for s in range(100):
        rng = np.random.default_rng(100 + s)
        n = 600
        X = np.column_stack([rng.integers(0, 30, n).astype(float) for _ in range(5)] + [rng.normal(size=n) for _ in range(5)])
        y = X[:, 5] + rng.normal(size=n)
        sp = K.fit_keys(X, y)
        worst = max(worst, sp[0]["u"] if sp else -9.0)
    assert worst < 0.2, worst
    return f"no structure: best key u over 100 draws = {worst:.3f} (< 0.2)"


def test_no_leak():
    rng = np.random.default_rng(3)
    n = 4000
    X = np.column_stack([rng.integers(0, 40, n).astype(float), rng.integers(0, 300, n).astype(float)])
    y = rng.normal(size=n)
    worst = 0.0
    for sp in K.fit_keys(X, y):
        e, _ = K.nested_encode(sp, X, y, 5)
        worst = max(worst, abs(float(np.corrcoef(e, y)[0, 1])))
    assert worst < 0.05, worst
    return f"no leak: |corr(nested encoding, noise target)| max = {worst:.4f} (< 0.05)"


def test_pair_key():
    rng = np.random.default_rng(4)
    n = 2000
    a, b = rng.integers(0, 12, n), rng.integers(0, 12, n)
    eff = rng.normal(size=(12, 12))
    X = np.column_stack([a, b] + [rng.normal(size=n) for _ in range(6)]).astype(float)
    y = eff[a, b]
    sp = K.fit_keys(X, y)
    assert tuple(sp[0]["cols"]) == (0, 1) and sp[0]["u"] >= 4.0, (sp[0]["cols"], sp[0]["u"])
    f, q = np.arange(n) % 4 != 0, np.arange(n) % 4 == 0
    pr = K.apply_stat(K.make_stat(K.fit_keys(X[f], y[f])[0], X[f], y[f]), X[q])
    assert u_of(pr, y[q], y[f]) >= 3.0
    return f"pair key: found (0, 1) with u = {sp[0]['u']:.2f}; query u >= 3"


def test_ordered_backoff():
    rng = np.random.default_rng(5)
    n = 1500
    k = rng.integers(0, 300, n)
    y = np.sin(k / 40.0) + 0.5 * (k / 300.0)
    X = np.column_stack([k.astype(float)] + [rng.normal(size=n) for _ in range(4)])
    f = (k % 3) != 0                       # a third of the key values never appear in the fit rows
    q = ~f
    sp = K.fit_keys(X[f], y[f])[0]
    assert sp["cols"] == (0,) and sp["interp"], sp
    pr = K.apply_stat(K.make_stat(sp, X[f], y[f]), X[q])
    assert u_of(pr, y[q], y[f]) >= 3.0, u_of(pr, y[q], y[f])
    return f"ordered key, unseen values: interpolation u = {u_of(pr, y[q], y[f]):.2f} (>= 3)"


def test_finite_with_missing():
    rng = np.random.default_rng(6)
    n = 400
    X = rng.normal(size=(n, 9))
    X[:, 0] = rng.integers(0, 5, n)
    X[:, 1] = 3.0                          # constant column
    X[rng.random((n, 9)) < 0.1] = np.nan
    y = np.where(np.isfinite(X[:, 2]), X[:, 2], 0.0) + rng.normal(size=n)
    out, _ = K.heads(X[:300], y[:300], X[300:], 9)
    assert all(np.all(np.isfinite(out[k])) and len(out[k]) == 100 for k in K.NAMES)
    return "missing cells and a constant column: all five columns finite"


def test_factor_glm():
    rng = np.random.default_rng(7)
    n = 1600
    f = [rng.integers(0, c, n) for c in (2, 2, 2, 11, 13, 6)]
    eff = [rng.normal(scale=0.5, size=c) for c in (2, 2, 2, 11, 13, 6)]
    x = rng.normal(size=n)
    y = np.exp(sum(e[i] for e, i in zip(eff, f)) + 0.3 * x)
    X = np.column_stack(f + [x]).astype(float)
    tr, te = np.arange(n) % 5 != 0, np.arange(n) % 5 == 0
    pr = K.factor_glm(X[tr], y[tr], X[te], 3)
    u = u_of(pr, y[te], y[tr])
    assert u >= 3.0, u
    return f"multiplicative factors: factor_glm u = {u:.2f} (>= 3)"


if __name__ == "__main__":
    for t in (test_exact_key, test_no_structure, test_no_leak, test_pair_key, test_ordered_backoff,
              test_finite_with_missing, test_factor_glm):
        print("PASS", t())
    print("ALL PASS 7")
