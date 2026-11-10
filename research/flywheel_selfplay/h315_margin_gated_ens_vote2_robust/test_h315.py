"""Data-free H315 controls: predicted margin, quantile thresholds (switched contexts only), the gate (m=0 = no gate),
the drop-top-5 inner objective, the tie order (larger m first), the paired drop-top-10 bootstrap and the decision."""

import numpy as np

import evaluate_h315 as ev
import run_h315 as r


def test_margins_and_gate():
    outs = [dict(raw_scores=np.array([[0.0, 1.0, 0.5], [0.2, 0.1, 0.0]])) for _ in range(3)]
    m = r.margins(outs, [1, 0])
    assert np.allclose(m, [1.0, 0.0])
    thr = r.thresholds([1, 2, 0, 1], [0.1, 0.3, 9.9, 0.5])  # slot-0 contexts excluded from the quantiles
    assert thr["0"] is None and abs(thr["q50"] - 0.3) < 1e-12 and thr["q90"] > thr["q75"] > thr["q50"]
    assert r.gate([1, 2, 0], [0.1, 0.6, 0.0], 0.5) == [0, 2, 0]
    assert r.gate([1, 2, 0], [0.1, 0.6, 0.0], None) == [1, 2, 0]
    assert r.thresholds([0, 0], [0.0, 0.0])["q50"] is None


def test_drop_top_objective_and_ties():
    fold = [np.array([1.0, 1, 1, 1, 1, 0.0, 0.0])] * 4  # top-5 dropped per fold -> zeros remain
    assert r.drop_top_objective(fold) == 0.0
    a = dict(tau=0.1, k=6, a=2, m="0", inner_switches=10)
    b = dict(tau=0.1, k=6, a=2, m="q90", inner_switches=50)
    assert sorted([a, b], key=r.margin_choice_key)[0]["m"] == "q90"  # larger m wins ties first


def test_paired_drop_top_and_decision():
    rng = np.random.default_rng(1)
    Db = rng.normal(0, 0.01, size=(9, 205))
    Db[:, :10] += 0.3  # BASE: big wins + noise
    Dm = Db.copy()
    Dm[:, 10:] = np.abs(Dm[:, 10:]) * 0.0 + 0.002  # MARGIN removes the marginal noise
    p = ev.paired_drop_top(Dm, Db, draws=2000)
    assert p["median"] > 0 and p["lcb95"] > 0
    p0 = ev.paired_drop_top(Db, Db, draws=500)
    assert p0["median"] == 0 and p0["lcb95"] == 0
    crit, v = ev.decide(p, [0.05] * 9, [0.05] * 9, [0.05] * 9, [0.06] * 9)
    assert v == "MARGIN_ADVANCE"
    assert ev.decide(p0, [0.05] * 9, [0.05] * 9, [0.05] * 9, [0.06] * 9)[1] == "MARGIN_NOT_ADVANCED"  # ties never win
    assert ev.decide(p, [0.04] * 9, [0.05] * 9, [0.05] * 9, [0.06] * 9)[1] == "MARGIN_NOT_ADVANCED"  # < 90% of BASE
    assert ev.ELO_LABEL.startswith("reference = r882 slot0 (DL-Train outer)")
