"""H96 structural tests on synthetic fixtures. They use the merged H81 selection rule; no data, models or native calls."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_w4 as h  # noqa: E402

H81 = h.load_h81()


def fixture(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=(n, 16)), [hashlib.sha256(f"p{i}".encode()).hexdigest() for i in range(n)]


def test_reveal_sees_only_the_first_pair_and_zero_residuals_reduce_to_top4():
    mu, x, ids = fixture()
    calls = []

    def reveal(sel):
        calls.append(list(sel)); return [float(mu[i]) for i in sel]  # outcomes equal the prior: zero residuals
    picks = h.select(H81, mu, x, ids, 1.0, 7, reveal)
    top = H81.ranked(mu, ids)
    assert picks["prog_top4"] == top[:4] and all(c == top[:2] for c in calls)
    assert picks["prog_2plus2"] == top[:4] and picks["prog_2plus2_swap"] == top[:4]
    assert all(len(set(p)) == 4 for p in picks.values())


def test_informative_residuals_change_the_second_pair_only():
    for seed in range(10):
        mu, x, ids = fixture(seed=seed)
        picks = h.select(H81, mu, x, ids, 1.0, 7, lambda sel: [float(mu[sel[0]]) + 1.0, float(mu[sel[1]]) - 1.0])
        assert picks["prog_2plus2"][:2] == H81.ranked(mu, ids)[:2]
        assert not set(picks["prog_2plus2"][2:]) & set(picks["prog_2plus2"][:2])


def test_execution_error_in_first_pair_makes_the_arm_incomplete():
    mu, x, ids = fixture()
    picks = h.select(H81, mu, x, ids, 1.0, 7, lambda sel: [None, 0.0])
    assert picks["prog_2plus2"] is None and picks["prog_2plus2_swap"] is None and len(picks["prog_top4"]) == 4


def test_dry_run_contexts_avoid_roster_families_and_are_deterministic():
    train = [dict(context_sha256=f"c{i}", family_id=f"f{i % 5}") for i in range(40)]
    roster = ["c0", "c1"]
    d = h.dry_contexts(train, roster)
    assert d == h.dry_contexts(train, roster) and len(d) == h.N_DRY
    assert all(int(c[1:]) % 5 not in (0, 1) for c in d)


def rows(delta):
    out = []
    for i in range(60):
        for seed in (701, 702, 703):
            for a in h.ARMS:
                v = 1.0 + 0.01 * i + (delta if a == "prog_2plus2" else 0.0)
                out.append(dict(tag="W4", context=f"c{i}", family=f"f{i % 30}", seed=seed, arm=a, status="completed",
                                best_of_four=v, native_wall_s=1.0, native_cpu_s=1.0))
    return out


ROSTER = [f"c{i}" for i in range(60)]


def analyze(rs, roster=ROSTER, shortfalls=()):
    return h.analyze(H81, rs, roster=roster, shortfalls=list(shortfalls))


def test_analyze_gate_and_dry_run_rows_are_never_analyzed():
    assert analyze(rows(0.02))["verdict"].startswith("ADVANCE")
    assert analyze(rows(0.0))["verdict"].startswith("NO_ADVANCE")
    dry = [dict(r, tag="DRY_RUN") for r in rows(0.02)]
    with pytest.raises(SystemExit):
        analyze(dry)


def test_missing_seeds_on_a_small_panel_are_incomplete_not_advance():
    small = [r for r in rows(0.05) if r["context"] in ("c0", "c1", "c2") and r["seed"] == 701]
    out = analyze(small)
    assert out["verdict"].startswith("INCOMPLETE") and out["panel_gaps"] > 0


def test_duplicated_seed_is_incomplete_not_advance():
    dup = [dict(r, seed=701) for r in rows(0.05)]
    assert analyze(dup)["verdict"].startswith("INCOMPLETE")


def test_incomplete_context_seeds_block_the_verdict():
    rs = [dict(r, status="incomplete") if r["seed"] != 701 or int(r["context"][1:]) >= 10 else r for r in rows(0.05)]
    out = analyze(rs)
    assert out["verdict"].startswith("INCOMPLETE") and out["incomplete_context_seeds"] > 0


def test_proposal_shortfall_blocks_the_verdict():
    out = analyze(rows(0.05), shortfalls=[dict(context="c0", seed=701, proposals=5)])
    assert out["verdict"].startswith("INCOMPLETE") and out["proposal_shortfalls"] == 1


def test_off_roster_context_blocks_the_verdict():
    assert analyze(rows(0.05), roster=ROSTER[:-1])["verdict"].startswith("INCOMPLETE")


def test_h81_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_prospective.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H81_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h81()
