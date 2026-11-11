#!/usr/bin/env python3
"""H433 unit checks of formula_core on synthetic tables only (no real table is opened).  python3 -B test_formula_core.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formula_core as F  # noqa: E402


def u_of(pred, y, y_fit):
    return float(-np.log(max(np.sqrt(np.mean((pred - y) ** 2)), 1e-300) / np.sqrt(np.mean((y - y_fit.mean()) ** 2))))


def test_difference():
    rng = np.random.default_rng(1)
    n = 1600
    X = rng.normal(size=(n, 14)) * rng.uniform(0.5, 30, 14) + rng.uniform(-50, 50, 14)
    y = ((X[:, 1] - X[:, 2]) / 10.0) ** 2
    f, q = np.arange(n) % 5 != 0, np.arange(n) % 5 == 0
    t0 = time.time()
    out, info = F.heads(X[f], y[f], X[q], 3)
    assert info["survivors"] and info["survivors"][0]["name"] == "x1-x2", info["survivors"][:3]
    ui, ug, up, uc = (u_of(out[k], y[q], y[f]) for k in ("form_interp", "form_gbm", "gbm_plain", "cap"))
    assert ui >= 5.0 and ug > up + 0.5, (ui, ug, up, uc, info.get("interp"))
    return f"y = ((x1-x2)/10)^2: survivor x1-x2; form_interp u {ui:.2f}, form_gbm {ug:.2f}, gbm_plain {up:.2f}, cap {uc:.2f} ({time.time() - t0:.0f} s)"


def test_ratio():
    rng = np.random.default_rng(2)
    n = 1500
    X = rng.normal(size=(n, 10)) * 3 + 5
    X[:, 4] = rng.uniform(1, 9, n)
    y = X[:, 3] / X[:, 4] + 0.01 * rng.normal(size=n)
    f, q = np.arange(n) % 5 != 0, np.arange(n) % 5 == 0
    surv, _single, info = F.screen(X[f], y[f], 5)
    assert surv and surv[0]["spec"]["op"] in ("div", "rdiv", "lograt") and {surv[0]["spec"]["a"], surv[0]["spec"]["b"]} == {3, 4}, info["survivors"][:3]
    out, _ = F.heads(X[f], y[f], X[q], 5)
    us, up = u_of(out["form_sr"], y[q], y[f]), u_of(out["gbm_plain"], y[q], y[f])
    assert us > up + 0.5, (us, up)
    return f"y = x3/x4: best survivor {info['survivors'][0]['name']}; form_sr u {us:.2f} vs gbm_plain {up:.2f}"


def test_noise():
    worst = 0
    for s in range(20):
        rng = np.random.default_rng(300 + s)
        n = 800
        X = rng.normal(size=(n, 10))
        y = X[:, 0] + rng.normal(size=n)          # one linear column, no two-column structure
        surv, _s, _i = F.screen(X, y, 7 + s)
        worst = max(worst, len(surv))
    assert worst == 0, worst
    return "no two-column structure: 0 survivors in 20 draws"


def test_out_of_range():
    rng = np.random.default_rng(4)
    n = 700
    X = rng.uniform(1, 5, size=(n, 6))
    y = X[:, 0] * X[:, 1]
    Xq = rng.uniform(-20, 40, size=(200, 6))
    Xq[:10, 1] = 0.0
    out, _ = F.heads(X, y, Xq, 9)
    assert all(np.all(np.isfinite(out[k])) and len(out[k]) == 200 for k in F.NAMES)
    return "query rows far outside the fit range (and zeros): all six columns finite"


if __name__ == "__main__":
    for t in (test_difference, test_ratio, test_noise, test_out_of_range):
        print("PASS", t())
    print("ALL PASS 4")
