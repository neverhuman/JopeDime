"""H112 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h112 as h  # noqa: E402

H106 = h.load_h106()
H81 = H106.load_h81()
S = lambda t: hashlib.sha256(t.encode()).hexdigest()


def fixture(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=(n, 16)), [S(f"p{i}") for i in range(n)]


def test_level_free_update_ignores_a_common_shift_and_only_reveals_the_first_pair():
    mu, x, ids = fixture()
    calls = []
    for shift in (0.0, 0.7):
        def reveal(sel):
            calls.append(list(sel)); return [float(mu[sel[0]]) + 0.3 + shift, float(mu[sel[1]]) - 0.3 + shift]
        p = h.select(H81, mu, x, ids, 1.0, 3, reveal)
        if shift == 0.0:
            base = p
    assert p == base  # a common outcome shift leaves every pick unchanged
    assert all(c == H81.ranked(mu, ids)[:2] for c in calls)
    assert p["s_top4"] == H81.ranked(mu, ids)[:4] and p["s_2plus2"][:2] == p["s_top4"][:2]


def test_equal_residuals_reduce_2plus2_to_top4_and_swap_negates_the_signal():
    mu, x, ids = fixture(seed=4)
    p = h.select(H81, mu, x, ids, 1.0, 3, lambda sel: [float(mu[i]) + 0.5 for i in sel])
    assert p["s_2plus2"] == p["s_top4"] == p["s_2plus2_swap"]
    first = H81.ranked(mu, ids)[:2]
    a = h.second_pair(H81, mu, x, ids, first, [float(mu[first[0]]) + 1, float(mu[first[1]]) - 1], 1.0, False)
    b = h.second_pair(H81, mu, x, ids, first, [float(mu[first[0]]) - 1, float(mu[first[1]]) + 1], 1.0, False)
    c = h.second_pair(H81, mu, x, ids, first, [float(mu[first[0]]) + 1, float(mu[first[1]]) - 1], 1.0, True)
    assert c == b and not set(a) & set(first)


def test_execution_error_in_first_pair_makes_adaptive_arms_incomplete():
    mu, x, ids = fixture()
    p = h.select(H81, mu, x, ids, 1.0, 3, lambda sel: [None, 1.0])
    assert p["s_2plus2"] is None and p["s_2plus2_swap"] is None and len(p["s_top4"]) == 4


def test_slope_is_within_context_and_clipped():
    groups = ["a"] * 4 + ["b"] * 4
    score = np.array([0, 1, 2, 3, 10, 11, 12, 13], dtype=float)
    util = np.array([5, 5.1, 5.2, 5.3, -1, -0.9, -0.8, -0.7])
    assert h.ols_slope(score, util, groups) == pytest.approx(0.1)  # level differences between contexts ignored
    assert abs(h.ols_slope(score, util * 100, groups)) <= h.CLIP  # clipped utilities bound the slope's leverage


def rows(delta=0.0, swap_delta=0.0, n=60):
    out = []
    for i in range(n):
        for seed in h.EVAL_SEEDS:
            for a in h.ARMS:
                v = 1.0 + 0.01 * i + (delta if a == "s_2plus2" else 0) + (swap_delta if a == "s_2plus2_swap" else 0)
                out.append(dict(tag="H112", context=f"c{i}", family=f"f{i % 30}", seed=seed, arm=a, status="completed", best_of_four=v))
    return out


ROSTER = [f"c{i}" for i in range(60)]


def test_analyze_needs_gate_and_placebo_and_complete_panel():
    assert h.analyze(H81, rows(0.02), ROSTER, [])["verdict"].startswith("ADVANCE")
    assert h.analyze(H81, rows(0.02, swap_delta=0.03), ROSTER, [])["verdict"] == "NO_ADVANCE"  # placebo as good
    assert h.analyze(H81, rows(0.02)[:-1], ROSTER, [])["verdict"].startswith("INCOMPLETE")
    with pytest.raises(SystemExit):
        h.analyze(H81, [dict(r, tag="DRY_RUN") for r in rows()], ROSTER, [])


def test_h106_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_h106.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H106_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h106()
