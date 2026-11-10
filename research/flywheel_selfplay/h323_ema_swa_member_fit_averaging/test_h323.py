"""Data-free H323 controls: snapshot iterations and weights, snapshot prediction, and the per-arm decision rule
(+0.002 margin, 97.5% LCB, >= 2/3 seeds; ties never advance)."""

import numpy as np

import evaluate_h323 as ev
import run_h323 as r


class _Booster:
    def predict(self, X, num_iteration):
        return np.full(len(X), float(num_iteration))


def test_snapshots_and_weights():
    assert (
        r.snap_iters(100) == [60, 70, 80, 90, 100]
        and r.snap_iters(50) == [30, 35, 40, 45, 50]
        and r.snap_iters(1) == [1, 1, 1, 1, 1]
    )
    assert abs(sum(r.W_SWA) - 1) < 1e-12 and abs(sum(r.W_EMA) - 1) < 1e-12
    assert r.W_EMA[-1] > r.W_EMA[0] and abs(r.W_EMA[-1] / r.W_EMA[-2] - 1 / 0.7) < 1e-12
    p = r.snap_predict(_Booster(), np.zeros((2, 3)), 100, r.W_SWA)
    assert np.allclose(p, 80.0)  # mean of 60..100
    assert np.allclose(r.snap_predict(_Booster(), np.zeros((1, 3)), 100, (0, 0, 0, 0, 1)), 100.0)


def test_decision():
    good = dict(median=0.003, ci95=[0.0005, 0.006])
    pc = [0.003] * 9
    assert ev.decide(good, pc, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)[1] == "ADVANCE"
    small = dict(median=0.0015, ci95=[0.0005, 0.003])  # below the +0.002 margin
    assert ev.decide(small, pc, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)[1] == "NOT_ADVANCED"
    pc_one_seed = [0.01, -0.01, -0.01] * 3  # only seed 1 positive
    crit, v, n = ev.decide(good, pc_one_seed, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9)
    assert v == "NOT_ADVANCED" and n == 1
    assert ev.DOPE == {"composite_sha256_prefix": "c51a56d0", "loaded": False}
