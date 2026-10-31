"""H94 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_prospective_hedge as h  # noqa: E402


def ranked(score, ids, exclude=frozenset()):  # the merged H81 rule
    return sorted((i for i in range(len(ids)) if i not in exclude), key=lambda i: (-float(score[i]), ids[i]))


def fixture(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n), [hashlib.sha256(f"p{i}".encode()).hexdigest() for i in range(n)]


def test_hedge_keeps_prior_top_two_and_adds_ctx_top_two_of_the_rest():
    for seed in range(25):
        mp, mc, ids = fixture(seed=seed)
        p = h.picks("HEDGE_CTX", mp, mc, ids, ranked)
        assert p[:2] == ranked(mp, ids)[:2] and len(set(p)) == 4
        assert p[2:] == [i for i in ranked(mc, ids) if i not in p[:2]][:2]


def test_union_of_arms_never_exceeds_eight_fits_per_unit():
    for seed in range(25):
        mp, mc, ids = fixture(seed=seed)
        assert len({i for a in h.ARMS for i in h.picks(a, mp, mc, ids, ranked)}) <= 8


def test_hedge_reduces_to_prior_when_models_agree_and_minimum_proposals_suffice():
    mp, _, ids = fixture(n=h.MIN_PROPOSALS)
    assert sorted(h.picks("HEDGE_CTX", mp, mp, ids, ranked)) == sorted(ranked(mp, ids)[:4])
    with pytest.raises(ValueError):
        h.picks("RANDOM", mp, mp, ids, ranked)


class FakeH81:
    BOOTSTRAP_SEED, BOOTSTRAP_DRAWS, TIE = 20261031, 2000, 1e-4

    @staticmethod
    def contrast(values, draws):
        boot = values[draws].mean(axis=1); lcb = float(np.percentile(boot, 5.0))
        med, harm = float(np.median(values)), float(np.mean(values < -0.01))
        return dict(family_mean=float(values.mean()), family_median=med, bootstrap_p5_one_sided_lcb=lcb,
                    families_losing_more_than_0_01_fraction=harm, common_gate_passed=bool(lcb > 0.001 and med >= 0 and harm <= 0.1))


def rows_for(delta, n=60, incomplete=()):
    rows = []
    for i in range(n):
        for seed in (701, 702):
            for a in h.ARMS:
                ok = (i, seed) not in incomplete or a != "CTX_top4"
                v = 1.0 + 0.01 * i + (delta if a == "HEDGE_CTX" else 0.0)
                rows.append(dict(context=f"c{i}", family=f"f{i % 30}", seed=seed, arm=a, status="completed" if ok else "incomplete",
                                 retained_best=v if ok else None, native_wall_s=1.0, native_cpu_s=1.0, selection_cpu_s=0.01))
    return rows


def test_analyze_gate_and_incomplete_units_leave_the_paired_panel():
    res = h.analyze(FakeH81, rows_for(0.02, incomplete={(3, 701)}))
    assert res["verdict"].startswith("ADVANCE") and res["incomplete_context_seeds"] == 1
    assert res["primary"]["median_ci95"][0] > 0 and res["contexts"] == 60
    assert h.analyze(FakeH81, rows_for(0.0))["verdict"].startswith("NO_ADVANCE")


def test_h81_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_prospective.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H81_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h81()
