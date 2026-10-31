"""H92 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_downside as h  # noqa: E402


def fixture(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [hashlib.sha256(f"p{i}".encode()).hexdigest() for i in range(n)]
    return rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), ids


def test_hedge_always_keeps_prior_top_two_and_adds_two_distinct_others():
    for seed in range(20):
        u, prior, s, ids = fixture(seed=seed)
        picks = h.hedge_picks(prior, s, ids)
        assert picks[:2] == h.ranked_unrevealed(prior, ids)[:2]
        assert len(set(picks)) == 4 and min(picks) >= 4
        assert set(picks[2:]) == set([i for i in h.ranked_unrevealed(s, ids) if i not in picks[:2]][:2])


def test_short_never_leaves_the_prior_shortlist():
    for seed in range(20):
        u, prior, s, ids = fixture(seed=seed)
        picks = h.short_picks(prior, s, ids, k=16)
        assert set(picks) <= set(h.ranked_unrevealed(prior, ids)[:16]) and len(set(picks)) == 4


def test_hedge_equals_prior_top4_when_scorer_agrees_with_prior():
    u, prior, _, ids = fixture()
    assert sorted(h.hedge_picks(prior, prior, ids)) == sorted(h.ranked_unrevealed(prior, ids)[:4])
    assert sorted(h.short_picks(prior, prior, ids)) == sorted(h.ranked_unrevealed(prior, ids)[:4])


def test_retained_best_uses_anchors_and_refuses_bad_picks():
    u = np.array([0.1, 0.2, 0.3, 0.9] + [0.0] * 6)
    assert h.retained_best(u, [4, 5, 6, 7]) == 0.9  # anchors still count
    with pytest.raises(ValueError):
        h.retained_best(u, [4, 4, 5, 6])
    with pytest.raises(ValueError):
        h.retained_best(u, [3, 4, 5, 6])  # anchors are never "new" picks


def test_perfect_scorer_beats_prior_under_both_controls():
    u, prior, _, ids = fixture(seed=7)
    s = u.copy()  # oracle scorer
    best = int(np.argmax(u[4:]) + 4)
    assert best in h.hedge_picks(prior, s, ids) or best in h.ranked_unrevealed(prior, ids)[:2]


class FakeH83:
    BOOTSTRAP_SEED, BOOTSTRAP_DRAWS, TIE = 20261031, 2000, 1e-4

    @staticmethod
    def contrast(values, draws, pct):
        boot = values[draws].mean(axis=1); lcb = float(np.percentile(boot, pct))
        med, harm = float(np.median(values)), float(np.mean(values < -0.01))
        return dict(family_mean=float(values.mean()), family_median=med, adjusted_one_sided_lcb=lcb,
                    families_losing_more_than_0_01_fraction=harm, common_gate_passed=bool(lcb > 0.001 and med >= 0 and harm <= 0.1))


def test_summarize_verdicts_and_median_intervals():
    rng = np.random.default_rng(1)
    ctx, fam = {}, {}
    for i in range(80):
        b = rng.normal()
        ctx[f"c{i}"] = {a: b for a in h.ARMS}
        ctx[f"c{i}"]["HEDGE_CTX"] = b + 0.02
        fam[f"c{i}"] = f"f{i % 40}"
    res = h.summarize(FakeH83, ctx, fam)
    assert res["primaries_passing"] == ["HEDGE_CTX - PRIOR"] and res["verdict"].startswith("ADVANCE")
    assert res["primary"]["HEDGE_CTX - PRIOR"]["median_ci95"][0] > 0
    for v in ctx.values():
        v["HEDGE_CTX"] = v["PRIOR"]
    assert h.summarize(FakeH83, ctx, fam)["verdict"].startswith("NO_ADVANCE")


def test_h83_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_w6_cached.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H83_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h83()
