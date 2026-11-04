"""Data-free H306 controls: the declared extended grid, the paired bootstrap, the edge histogram, lock stripping for
the regression and the decision rule. Toy data only."""

import numpy as np

import evaluate_h306 as ev
import run_h306 as r


def test_grid():
    assert r.XGRID == dict(tau=(0.025, 0.05, 0.10), k=(4, 5, 6), a=(2,))
    assert len(r.XGRID["tau"]) * len(r.XGRID["k"]) * len(r.XGRID["a"]) == 9


def test_paired_bootstrap():
    Dn = np.zeros((3, 205))
    p = ev.paired(Dn + 0.01, Dn, draws=2000)
    assert all(abs(x["gain"] - 0.01) < 1e-12 and x["lcb95_one_sided"] > 0 and x["wtl"] == [205, 0, 0] for x in p)
    p0 = ev.paired(Dn, Dn, draws=2000)
    assert all(x["gain"] == 0 and x["lcb95_one_sided"] == 0 and x["wtl"] == [0, 205, 0] for x in p0)


def test_edge_histogram():
    ch = [dict(tau=0.025, k=4), dict(tau=0.05, k=5), dict(tau=0.10, k=6), dict(tau=0.025, k=5)]
    h = ev.edge_histogram(ch)
    assert h["at_new_low_edge_tau_0_025"] == 2 and h["at_new_low_edge_k_4"] == 1 and h["at_upper_edge_k_6"] == 1


def test_decide_ties_never_win():
    xj = dict(criteria=dict(c3_harmed_le_35pct_each=True, c4_invalid_excess_le_3_and_mcnemar=True,
                            c5_drop_top5_lcb95_gt_0_on_2of3=True))  # fmt: skip
    tie = [dict(gain=0.0, lcb95_one_sided=0.0)] * 3
    assert ev.decide(xj, tie)[1] == "XGRID_NOT_ADVANCED"
    win = [dict(gain=0.01, lcb95_one_sided=0.002)] * 2 + [dict(gain=-0.001, lcb95_one_sided=-0.01)]
    assert ev.decide(xj, win)[1] == "XGRID_ADVANCES"


def test_strip_lock_ignores_paths_and_time():
    m = dict(ranker=dict(sha256="a", file="x"), veto=dict(sha256="b", file="y"), bags=[dict(sha256="c", file="z")],
             selected_lr=0.05, selected_n_estimators=50, strict_margin=0.1, selected_tau=0.05)  # fmt: skip
    a = dict(outer_fold=0, root_seed=30001, inner_table=[1], chosen={"tau": 0.05}, n_tied=1, members=[m], elapsed_sec=1)
    b = dict(a, elapsed_sec=99, members=[dict(m, ranker=dict(sha256="a", file="other"))])
    assert r.strip_lock(a) == r.strip_lock(b)
