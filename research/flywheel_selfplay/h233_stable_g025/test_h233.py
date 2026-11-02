"""H233 design, pick-stability vote gate, STABLE_SHUF agreement-mask permutation, statistics, and guard controls;
all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402

_spec = importlib.util.spec_from_file_location("h233_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


def families(n=205):
    return [f"f{f:03}" for f in range(n)]


def test_cells_partition_folds_cyclically():
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert (a, b) == (k, (k + 1) % 5)
        assert sorted((a, b, *fit)) == list(range(5)) and len(fit) == 3


def test_fit_plan_is_25_per_seed_and_75_total():
    outers, inners = stat.fit_plan()
    assert len(outers) == 5 and len(inners) == 10
    for ex in outers:
        assert stat.members_for_exclusion(ex) == (0, 1, 2)
    for ex in inners:
        assert stat.members_for_exclusion(ex) == (0,)
    per_seed = sum(len(stat.members_for_exclusion(ex)) for ex in outers + inners)
    assert per_seed == 25
    assert per_seed * len(stat.SEEDS) == 75
    with pytest.raises(ValueError):
        stat.members_for_exclusion((0,))
    with pytest.raises(ValueError):
        stat.members_for_exclusion((1, 0))


def test_seed_folds_reproduce_h124_algorithm_and_differ_by_seed():
    fams = families()
    for s in stat.SEEDS:
        groups = stat.seed_folds(fams + fams[:7], s)
        expected = [
            sorted(x.tolist())
            for x in np.array_split(
                np.random.default_rng(stat.FOLD_SEEDS[s]).permutation(sorted(fams)), 5
            )
        ]
        assert groups == expected and all(len(g) == 41 for g in groups)
    assert len({json.dumps(stat.seed_folds(fams, s)) for s in stat.SEEDS}) == 3
    with pytest.raises(ValueError):
        stat.seed_folds(fams[:-1], stat.SEEDS[0])


def test_registered_fold_manifest_is_consistent_and_jaccard_bounded():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert m["wave"] == "H233"
    assert m["family_count"] == 205 and m["context_count"] == 1219
    assert m["fold_seeds"] == {str(s): stat.FOLD_SEEDS[s] for s in stat.SEEDS}
    fams = sorted(
        m["families_by_seed"][str(stat.SEEDS[0])][0]
        + sum(m["families_by_seed"][str(stat.SEEDS[0])][1:], [])
    )
    for s in stat.SEEDS:
        assert m["families_by_seed"][str(s)] == stat.seed_folds(fams, s)
        assert sum(m["fold_context_counts_by_seed"][str(s)]) == 1219
    assert m["pairwise_fold_jaccard"] == stat.fold_jaccard(
        {s: m["families_by_seed"][str(s)] for s in stat.SEEDS}
    )
    assert m["max_best_match_jaccard"] <= stat.JACCARD_MAX == 0.20
    assert len(m["max_best_match_jaccard_by_wave"]) == 24
    for w, val in m["max_best_match_jaccard_by_wave"].items():
        assert val <= 0.20, f"{w} exceeded 0.20: {val}"


def test_unanimous_pick_and_shuffle_agreement_mask_preserve_exact_vote_rate():
    b0 = np.array([1.0, 0.2, 3.0, 4.0])
    b1 = np.array([2.0, 0.1, 3.0, 4.0])
    b2 = np.array([5.0, -0.5, 3.0, 4.0])
    unanimous, picks = stat.unanimous_pick([b0, b1, b2])
    assert unanimous is True and picks == (1, 1, 1)

    b2_diff = np.array([-2.0, 0.1, 3.0, 4.0])
    unanimous2, picks2 = stat.unanimous_pick([b0, b1, b2_diff])
    assert unanimous2 is False and picks2 == (1, 1, 0)

    agree = {f"c{i:03}": (i % 3 == 0) for i in range(60)}
    shuf1 = stat.shuffle_agreement_mask(agree, stat.SEEDS[0], 0, "A")
    shuf2 = stat.shuffle_agreement_mask(agree, stat.SEEDS[0], 0, "A")
    shuf_other = stat.shuffle_agreement_mask(agree, stat.SEEDS[0], 1, "A")
    assert shuf1 == shuf2
    assert shuf1 != agree
    assert shuf1 != shuf_other
    assert sum(shuf1.values()) == sum(agree.values())
    assert np.isclose(np.mean(list(shuf1.values())), np.mean(list(agree.values())))


def test_context_arm_costs_and_h217_g025_parity():
    base = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    # All 3 members pull argmin to slot 2
    p0 = np.array([9.0, 9.0, -30.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    p1 = np.array([8.0, 8.0, -25.0, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)
    p2 = np.array([7.0, 7.0, -20.0, 7.0, 7.0, 7.0, 7.0, 7.0], dtype=np.float32)
    p2_disagree = np.array([7.0, 7.0, 7.0, -20.0, 7.0, 7.0, 7.0, 7.0], dtype=np.float32)

    # Case 1: zgap open (tau = +inf), unanimous = True, shuf_agree = False
    costs, opened, sig = stat.context_arm_costs(base, [p0, p1, p2], math.inf, False)
    assert opened == {
        "G025": True,
        "STABLE_G025": True,
        "STABLE_SHUF": False,
        "BAG_VOTE_UNGATED": True,
    }
    b0 = stat.blend(base, p0)
    assert np.array_equal(costs["G025"], b0)
    assert np.array_equal(costs["STABLE_G025"], b0)
    assert np.array_equal(costs["STABLE_SHUF"], base)
    assert np.array_equal(costs["BAG_VOTE_UNGATED"], b0)
    assert sig["unanimous"] is True and sig["member_picks"] == [2, 2, 2]

    # Case 2: zgap open (tau = +inf), unanimous = False (member 2 picks slot 3), shuf_agree = True
    costs2, opened2, sig2 = stat.context_arm_costs(base, [p0, p1, p2_disagree], math.inf, True)
    assert opened2 == {
        "G025": True,
        "STABLE_G025": False,
        "STABLE_SHUF": True,
        "BAG_VOTE_UNGATED": False,
    }
    assert np.array_equal(costs2["G025"], b0)
    assert np.array_equal(costs2["STABLE_G025"], base)
    assert np.array_equal(costs2["STABLE_SHUF"], b0)
    assert np.array_equal(costs2["BAG_VOTE_UNGATED"], base)
    assert sig2["unanimous"] is False and sig2["member_picks"] == [2, 2, 3]

    # Case 3: zgap closed (tau = -inf), unanimous = True -> STABLE_G025 and G025 closed, BAG_VOTE_UNGATED open
    costs3, opened3, _ = stat.context_arm_costs(base, [p0, p1, p2], -math.inf, True)
    assert opened3 == {
        "G025": False,
        "STABLE_G025": False,
        "STABLE_SHUF": False,
        "BAG_VOTE_UNGATED": True,
    }
    assert np.array_equal(costs3["STABLE_G025"], base)
    assert np.array_equal(costs3["BAG_VOTE_UNGATED"], b0)


def test_g025_is_bit_identical_to_h217_gated_025():
    h217_spec = importlib.util.spec_from_file_location(
        "h233_test_h217_stat", HERE.parent / "h217_gated025_confirm" / "analysis.py"
    )
    h217_stat = importlib.util.module_from_spec(h217_spec)
    h217_spec.loader.exec_module(h217_stat)
    rng = np.random.default_rng(19)
    for _ in range(50):
        base, p0, p1, p2 = rng.normal(size=(4, 40)).astype(np.float32)
        tau = float(rng.uniform(0, 1))
        costs, opened, _ = stat.context_arm_costs(base, [p0, p1, p2], tau, False)
        blended = h217_stat.blend(base, p0)
        old_open = h217_stat.gated(h217_stat.zgap(base, blended), tau)
        assert opened["G025"] == old_open
        assert np.array_equal(costs["G025"], blended if old_open else base)


def test_verdict_enforces_all_six_admission_gates_and_ties_fail():
    good_primary = dict(
        median_gain=0.0035,
        ci95=[0.0008, 0.0062],
        positive_seeds=3,
        per_seed_gain=[0.0030, 0.0035, 0.0041],
        harm_fraction=0.04,
    )
    beat = dict(median_gain=0.0012, ci95=[0.0001, 0.0025])
    tie = dict(median_gain=0.0, ci95=[-0.001, 0.001])
    val_sec = dict(median_gain=0.0025, ci95=[0.0002, 0.0045])
    plat = dict(win=True, gain=0.0025, ci95=[0.0002, 0.0045])

    v_ok = stat.verdict(good_primary, beat, beat, val_sec, plat)
    assert v_ok["verdict"] == "CANDIDATE: STABLE_G025"
    assert all(v_ok["gates"].values())

    # Tie with G025 or STABLE_SHUF -> NO_ADVANCE
    assert stat.verdict(good_primary, tie, beat, val_sec, plat)["verdict"] == "NO_ADVANCE"
    assert stat.verdict(good_primary, beat, tie, val_sec, plat)["verdict"] == "NO_ADVANCE"

    # Gain <= 0.002 -> NO_ADVANCE
    low_gain = dict(good_primary, median_gain=0.002)
    assert stat.verdict(low_gain, beat, beat, val_sec, plat)["verdict"] == "NO_ADVANCE"

    # Harm > 10% -> NO_ADVANCE
    high_harm = dict(good_primary, harm_fraction=0.105)
    assert stat.verdict(high_harm, beat, beat, val_sec, plat)["verdict"] == "NO_ADVANCE"


def test_member_weights_are_a_family_bootstrap():
    fams = np.repeat([f"f{i:03}" for i in range(120)], 192).tolist()
    assert stat.member_weights(fams, stat.SEEDS[0], (0, 1), 0) is None
    w1 = stat.member_weights(fams, stat.SEEDS[0], (0, 1), 1)
    w2 = stat.member_weights(fams, stat.SEEDS[0], (0, 1), 2)
    assert np.array_equal(w1, stat.member_weights(fams, stat.SEEDS[0], (0, 1), 1))
    assert not np.array_equal(w1, w2)
    for w in (w1, w2):
        per_family = w.reshape(120, 192)
        assert (per_family == per_family[:, :1]).all()
        assert per_family[:, 0].sum() == 120


def ok_sample(**kw):
    s = dict(
        load=1.0,
        load5=1.0,
        load_over_s=0.0,
        cpu_over_s=0.0,
        cpu=10.0,
        available=64 * 2**30,
        free=440_000_000_000,
        disk_used_fraction=0.90,
        rss=3_500_000_000,
        threads=8,
        tree_cores=7.8,
    )
    s.update(kw)
    return s


def test_monitor_worker_affinity_and_guards():
    pool = list(range(16))
    assert len(monitor.worker_affinity_for_load(35.0, pool)) == 8
    assert len(monitor.worker_affinity_for_load(50.0, pool)) == 4
    assert len(monitor.worker_affinity_for_load(58.0, pool)) == 4
    assert monitor.violations(ok_sample()) == []
    for kw in [
        dict(tree_cores=8.1),
        dict(available=19 * 2**30),
        dict(free=299_000_000_000),
        dict(rss=4 * 2**30),
    ]:
        assert len(monitor.violations(ok_sample(**kw))) == 1
    assert monitor.RUN_END == 1793682000.0  # 2026-11-03T05:00:00Z
    assert not monitor.overloaded(ok_sample(load5=60.0), paused=False)
    assert monitor.overloaded(ok_sample(load5=60.1), paused=False)
    assert monitor.overloaded(ok_sample(load5=58.0), paused=True)
    assert not monitor.overloaded(ok_sample(load5=55.0), paused=True)


@pytest.mark.parametrize("host", ["xbabe1", "xbabe3", "xbabe0"])
def test_runner_and_monitor_reject_excluded_hosts(host, tmp_path, monkeypatch):
    import runner

    assert set(runner.HOST_RULES) == {"xbabe2"}
    assert monitor.ALLOWED_HOSTS == ("xbabe2",)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    with pytest.raises(RuntimeError, match="xbabe2"):
        runner.admit_host()


class FakeH124:
    MODEL = dict(
        max_iter=30, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0
    )

    def __init__(self, feats):
        self.feats = feats

    def program(self, pools, cid):
        return self.feats[cid]

    def check_rss(self):
        pass

    def release_memory(self):
        pass


def toy_fit_inputs():
    rng = np.random.default_rng(11)
    rows, pools, context, samples, feats = [], {}, {}, {}, {}
    for i in range(10):
        cid = f"c{i:02}"
        rows.append({"context_sha256": cid, "family_id": f"f{i // 2}"})
        feats[cid] = rng.normal(size=(200, 412)).astype(np.float32)
        pools[cid] = {"u": rng.normal(size=200).astype(np.float32)}
        context[cid] = {
            "lm48_z": rng.normal(size=48).astype(np.float32),
            "dope629": rng.normal(size=629).astype(np.float32),
        }
        samples[cid] = np.sort(rng.choice(200, 192, replace=False))
    mapping = {r["context_sha256"]: i % 5 for i, r in enumerate(rows)}
    return FakeH124(feats), rows, mapping, context, pools, samples


def test_fit_unit_outer_3_members_inner_1_member_and_h217_bit_identity(tmp_path, monkeypatch):
    from threadpoolctl import threadpool_limits
    import runner

    with threadpool_limits(2):
        h124, rows, mapping, context, pools, samples = toy_fit_inputs()
        (tmp_path / "new").mkdir()
        r_outer = runner.fit_unit(
            h124, "BLEND", stat.SEEDS[0], (0, 1), rows, mapping, context, pools, samples, tmp_path / "new"
        )
        r_inner = runner.fit_unit(
            h124, "BLEND", stat.SEEDS[0], (0, 1, 2), rows, mapping, context, pools, samples, tmp_path / "new"
        )
        assert [r["member"] for r in r_outer] == [0, 1, 2]
        assert [r["member"] for r in r_inner] == [0]

        # Compare outer member 0 with H217 fit_gbm bit-for-bit
        saved = sys.modules.get("analysis")
        spec = importlib.util.spec_from_file_location(
            "analysis", HERE.parent / "h217_gated025_confirm" / "analysis.py"
        )
        h217_stat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h217_stat)
        sys.modules["analysis"] = h217_stat
        try:
            spec_r = importlib.util.spec_from_file_location(
                "h233_test_h217_runner", HERE.parent / "h217_gated025_confirm" / "runner.py"
            )
            h217_runner = importlib.util.module_from_spec(spec_r)
            spec_r.loader.exec_module(h217_runner)
        finally:
            if saved is not None:
                sys.modules["analysis"] = saved
        monkeypatch.setattr(h217_stat, "SEEDS", (stat.SEEDS[0],))
        (tmp_path / "old").mkdir()
        h217_runner.fit_gbm(
            h124, "BLEND", stat.SEEDS[0], (0, 1), rows, mapping, context, pools, samples, tmp_path / "old"
        )
        with (
            np.load(tmp_path / "old" / f"BLEND-seed{stat.SEEDS[0]}-exclude01.npz") as old,
            np.load(tmp_path / "new" / f"BLEND-seed{stat.SEEDS[0]}-exclude01-m0.npz") as m0,
        ):
            assert sorted(old.files) == sorted(m0.files)
            for k in old.files:
                assert np.array_equal(old[k], m0[k])
