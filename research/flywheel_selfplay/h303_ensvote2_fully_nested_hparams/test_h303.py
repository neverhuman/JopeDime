"""Data-free H303 controls: the declared grid and inner-fold rule, the conservative tie-break, TAU patching of H287's veto
module, the member/ensemble recomputation, the concentration audit and the decision rule. Toy data only."""

import types

import numpy as np

import evaluate_h303 as ev
import run_h303 as r


def test_grid_and_inner_folds():
    assert r.TAU_GRID == (0.05, 0.10, 0.15) and r.K_GRID == (5, 6, 7) and r.A_GRID == (2, 3) and r.TIE == 1e-4
    folds = {r.inner_fold(0, f"fam-{i}") for i in range(200)}
    assert folds == {0, 1, 2, 3} and r.inner_fold(1, "fam-x") == r.inner_fold(1, "fam-x")


def test_conservative_tiebreak():
    table = [dict(tau=0.10, k=6, a=2, inner_gain=0.05, inner_switches=100), dict(tau=0.05, k=7, a=3, inner_gain=0.04995, inner_switches=40),
             dict(tau=0.15, k=5, a=2, inner_gain=0.03, inner_switches=10)]  # fmt: skip
    best = max(x["inner_gain"] for x in table)
    tied = [x for x in table if x["inner_gain"] >= best - r.TIE]
    assert sorted(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))[0]["inner_switches"] == 40


def test_set_tau_patches_mask_default():
    def mask(scores, p_invalid, tau=0.10):
        return np.where(p_invalid >= tau, -np.inf, scores)

    veto = types.SimpleNamespace(TAU=0.10, mask=mask)
    r.set_tau(veto, 0.05)
    assert veto.TAU == 0.05 and veto.mask.__defaults__ == (0.05,)
    assert np.isinf(veto.mask(np.zeros(2), np.array([0.06, 0.0]))[0])


def test_ensemble_vote():
    ens = types.SimpleNamespace(
        vote_consensus_pick=lambda mp, a: next(
            (c for c in sorted({p for p in mp if p}) if [p for p in mp if p].count(c) >= a), 0
        )
    )
    assert r.ens_picks([[3, 0], [3, 2], [0, 2]], 2, ens) == [3, 2] and r.ens_picks(
        [[3, 0], [3, 2], [0, 2]], 3, ens
    ) == [0, 0]


def test_concentration_and_drop_top():
    D = np.zeros((3, 205))
    D[:, :5] = 1.0  # all gain in 5 families
    c = ev.concentration(D, 302150, 2000)
    assert (
        c[0]["gain_drop_top5"] == 0.0
        and c[0]["top10_share_of_total_gain"] == 1.0
        and c[0]["drop_top5_lcb95_one_sided"] == 0.0
    )
    D2 = np.full((3, 205), 0.02)
    assert ev.concentration(D2, 302150, 2000)[0]["drop_top5_lcb95_one_sided"] > 0
