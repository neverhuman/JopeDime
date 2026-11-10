"""Data-free H320 controls: the trimmed inner objective (q95 clip of |gain|, family-equal means, drop the top 10),
the tie order, and the decision rule (ties never advance)."""

import numpy as np

import evaluate_h320 as ev
import run_h320 as r


def test_trimmed_objective_drops_top10_and_clips():
    fams = [f"f{i}" for i in range(12)]
    g = [1.0] * 10 + [0.01, 0.02]  # ten big families, two small
    # q95 of |g| = 1.0 (no clip of the big ones); after dropping the top 10 family means the rest is mean(0.01, 0.02)
    assert abs(r.trimmed_fold_objective(g, fams) - 0.015) < 1e-12
    g2 = [100.0] + [0.0] * 39  # one huge outlier context in a 40-family fold
    fams2 = [f"f{i}" for i in range(40)]
    q = float(np.quantile(np.abs(g2), 0.95))
    assert q == 0.0 and r.trimmed_fold_objective(g2, fams2) == 0.0  # clipped to q95 = 0, then dropped anyway
    assert r.trimmed_fold_objective([], []) == 0.0


def test_tie_order_and_decision():
    a = dict(tau=0.1, k=6, a=2, inner_switches=50)
    b = dict(tau=0.05, k=5, a=2, inner_switches=40)
    assert sorted([a, b], key=r.trim_key)[0] is b
    good = dict(median=0.002, lcb95=0.0005)
    assert ev.decide(good, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)[1] == "TRIM_ADVANCE"
    assert (
        ev.decide(dict(median=0.0, lcb95=0.0), [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)[1] == "TRIM_NOT_ADVANCED"
    )
    assert ev.decide(good, [0.04] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)[1] == "TRIM_NOT_ADVANCED"
    assert ev.decide(good, [0.05] * 9, [0.05] * 9, [0.10] * 9, [0.09] * 9)[1] == "TRIM_NOT_ADVANCED"
