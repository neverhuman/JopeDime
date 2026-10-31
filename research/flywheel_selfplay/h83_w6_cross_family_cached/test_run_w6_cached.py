"""H83 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_w6_cached as h  # noqa: E402


def planted(n_ctx=60, per=40, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(48, 2)), rng.normal(size=(16, 2))
    w = a @ b.T / 10.0  # rank-2 truth
    zc = rng.normal(size=(n_ctx, 48))
    z = np.repeat(zc, per, 0); x = rng.normal(size=(n_ctx * per, 16)); g = np.repeat(np.arange(n_ctx), per)
    r = np.einsum("ij,jk,ik->i", z, w, x) + 0.05 * rng.normal(size=len(g))
    return z, x, r, g, w


def test_bilinear_recovers_planted_low_rank_interaction():
    z, x, r, g, w = planted()
    est = h.fit_bilinear(z, x, h.centre_within(g, r), ridge=1.0, rank=8)
    assert np.linalg.matrix_rank(est, tol=1e-8) <= 8
    assert np.corrcoef(est.ravel(), w.ravel())[0, 1] > 0.9


def test_shuffled_context_placebo_loses_the_signal():
    z, x, r, g, w = planted(seed=1)
    perm = np.random.default_rng(5).permutation(60)
    zsh = np.repeat(z[::40][perm], 40, 0)
    est = h.fit_bilinear(zsh, x, h.centre_within(g, r), ridge=1.0)
    assert abs(np.corrcoef(est.ravel(), w.ravel())[0, 1]) < 0.5


def test_centre_within_removes_context_levels():
    g = np.array([0, 0, 1, 1]); v = np.array([1.0, 3.0, 10.0, 14.0])
    assert np.allclose(h.centre_within(g, v), [-1, 1, -2, 2])


def test_u4_uses_anchors_plus_top4_unrevealed_with_sha_ties():
    u = np.array([0.1, 0.2, 0.3, 0.4, 5.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    ids = [f"{i:02d}" for i in range(10)]
    score = np.array([0, 0, 0, 0, 1.0, 2.0, 2.0, 2.0, 2.0, -5.0])  # top4 unrevealed = slots 5..8, slot 9 excluded
    assert h.u4(u, score, ids) == (0.4, 0.4)
    score[9] = 3.0
    assert h.u4(u, score, ids) == (0.4, 9.0)
    score[4] = 9.0  # anchors are never chosen as "new" picks even if their score is high
    assert h.u4(u, np.r_[np.full(4, 99.0), score[4:]], ids)[1] == 9.0


def test_contrast_gate_and_bonferroni_percentile():
    draws = np.random.default_rng(0).integers(0, 50, size=(2000, 50))
    good = h.contrast(np.full(50, 0.02) + np.linspace(-0.001, 0.001, 50), draws, 2.5)
    assert good["common_gate_passed"] and good["lcb_percentile"] == 2.5
    zero = h.contrast(np.zeros(50), draws, 2.5)
    assert not zero["common_gate_passed"]  # LCB must exceed +0.001
    harm = np.full(50, 0.05); harm[:8] = -0.05
    assert not h.contrast(harm, draws, 2.5)["common_gate_passed"]  # 16% of families lose > .01


def test_summarize_verdicts():
    rng = np.random.default_rng(3)
    ctx, fam = {}, {}
    for i in range(80):
        base = rng.normal()
        ctx[f"c{i}"] = dict(PRIOR=base, TREAT=base + 0.05, SHUF=base, KNN=base, RETR=base - 0.1, CTX=base, RANDOM4=base - 0.5, U0=base - 1)
        fam[f"c{i}"] = f"f{i % 40}"
    res = h.summarize(ctx, fam)
    assert res["families"] == 40 and res["verdict"].startswith("ADVANCE")
    assert res["primary"]["TREAT - PRIOR"]["lcb_percentile"] == pytest.approx(2.5)
    for v in ctx.values():
        v["TREAT"] = v["PRIOR"]
    assert h.summarize(ctx, fam)["verdict"].startswith("NO_ADVANCE")
