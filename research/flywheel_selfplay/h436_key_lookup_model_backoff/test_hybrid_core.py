#!/usr/bin/env python3
"""H436 unit checks of hybrid_core on synthetic tables only (no real table is opened).  python3 -B test_hybrid_core.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_core as HC  # noqa: E402
import keys_core as K  # noqa: E402


def u_of(pred, y, y_fit):
    return float(-np.log(max(np.sqrt(np.mean((pred - y) ** 2)), 1e-300) / np.sqrt(np.mean((y - y_fit.mean()) ** 2))))


def singleton_key_table(rng, n=1500):
    """Exact key, group sizes 1..12, so several key values have a single fit row; plus a feature that predicts the
    group effect roughly (so the model backoff is informative)."""
    sizes = rng.integers(1, 13, 400)
    g = np.repeat(np.arange(400), sizes)[:n]
    g = g[rng.permutation(len(g))]
    eff = rng.normal(size=400)
    X = np.column_stack([g * 3.0 + 7, eff[g] + 0.3 * rng.normal(size=len(g))] + [rng.normal(size=len(g)) for _ in range(6)])
    return X, eff[g]


def test_out_of_fold_scoring():
    rng = np.random.default_rng(1)
    X, y = singleton_key_table(rng)
    n = len(y)
    fold = rng.permutation(n) % 5
    oof_h, oof_d = np.empty(n), np.empty(n)
    for k in range(5):
        fi, qi = fold != k, fold == k
        out, info = HC.heads(X[fi], y[fi], X[qi], 10 + k)
        oof_h[qi] = out["key_hybrid"]
        sp = K.fit_keys(X[fi], y[fi])[0]
        oof_d[qi] = K.apply_stat(K.make_stat(sp, X[fi], y[fi]), X[qi])
        assert info["key"] and info["key"]["cols"] == [0], info
    uh, ud = u_of(oof_h, y, y), u_of(oof_d, y, y)
    assert uh > ud + 0.5 and uh > 2.0, (uh, ud)
    return f"keys with single fit rows: out-of-fold u of key_hybrid {uh:.2f} vs H430 key_direct {ud:.2f}"


def test_exact_when_seen():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 150, 1600)
    X = np.column_stack([g.astype(float)] + [rng.normal(size=1600) for _ in range(8)])
    y = rng.normal(size=150)[g]
    f, q = np.arange(1600) % 5 != 0, np.arange(1600) % 5 == 0
    out, info = HC.heads(X[f], y[f], X[q], 3)
    seen = np.isin(X[q, 0], X[f, 0])
    assert info["key"]["m"] == 0.0 and u_of(out["key_hybrid"][seen], y[q][seen], y[f]) > 6.0
    return "exact key: key_hybrid is exact where the key value is known (m = 0)"


def test_no_key_equals_model():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.integers(0, 30, 900).astype(float)] + [rng.normal(size=900) for _ in range(6)])
    y = X[:, 1] + 0.5 * rng.normal(size=900)
    out, info = HC.heads(X[:700], y[:700], X[700:], 4)
    assert info["key"] is None and np.array_equal(out["key_hybrid"], out["gbm_plain"])
    return "no useful key: key_hybrid equals the model column"


def test_guard():
    rng = np.random.default_rng(4)
    n, p = 300, 250                                  # more columns than rows, near-collinear: ridge can extrapolate
    Z = rng.normal(size=(n + 100, 8))
    X = Z @ rng.normal(size=(8, p)) + 1e-6 * rng.normal(size=(n + 100, p))
    X[n:] *= 50.0                                    # query rows far outside the fit range
    y = Z[:, 0] + 0.1 * rng.normal(size=n + 100)
    out, _ = HC.heads(X[:n], y[:n], X[n:], 5)
    lo, hi = y[:n].min(), y[:n].max()
    pad = 0.5 * (hi - lo)
    assert all(np.all((out[k] >= lo - pad - 1e-9) & (out[k] <= hi + pad + 1e-9)) for k in HC.NAMES)
    return "query rows 50x outside the fit range: every column stays inside the guarded range"


if __name__ == "__main__":
    for t in (test_out_of_fold_scoring, test_exact_when_seen, test_no_key_equals_model, test_guard):
        print("PASS", t())
    print("ALL PASS 4")
