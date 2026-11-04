"""H335 tests (pytest). Run twice, once per process group, inside sandbox_h335.sh on xbabe2:
  H335_GROUP=BASE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider test_h335.py
  H335_GROUP=AGG  PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider test_h335.py
No Phase T panel (P201b) or P205 context is ever evaluated: unit tests use NON-panel TRAIN contexts (third or later
contexts of a family) and the H317 deterministic fake worker (test_h317_gp.fake_runner); no real fit happens here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import h335_lab as LAB  # noqa: E402
import h335_stats as H  # noqa: E402
import test_h317_gp as T  # noqa: E402  (frozen H317 fixtures: fake_runner, HashCoder, scratch)

C, D, E, L, ST, S, G = K.C, K.D, K.E, K.L, K.ST, K.S, K.G
AGG = K.GROUP == "AGG"


@pytest.fixture(scope="module")
def rows():
    return C.train_rows()


@pytest.fixture(scope="module")
def nonpanel(rows):
    """TRAIN contexts in neither P205 (H317) nor P201b (H335 Phase T)."""
    prot = set(K.panel_p201b(rows))
    folds = C.family_folds(rows)
    for n in (24, 48, 96, 205, C.PILOT_N):
        prot |= set(C.pilot_shard(rows, folds, n=n))
    out = [r["context_sha256"] for r in rows if r["context_sha256"] not in prot]
    out.sort(key=lambda c: (K.table_size(c), C.stream_seed("h335-test", c)))
    return out


def make_rt(name: str, runner, seeds=(335101,)):
    rows = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in rows}
    bank, _ = D.build_bank(rows)
    eng = E.EvalEngine([0, 1, 2, 3], root=T.scratch(f"h335_{K.GROUP}_{name}") / "gproot", runner=runner)
    fb = S.FallbackScreener()
    return L.Runtime(eng, E.AuditLedger(eng), lambda cid: fb, T.HashCoder(), bank, fam_of, ds_of,
                     L.Store(T.scratch(f"h335_{K.GROUP}_{name}") / "run"), tuple(seeds))


# ---------------------------------------------------------------------------------------------------------------------
# Core: pins, guard extension, panel
# ---------------------------------------------------------------------------------------------------------------------
def test_pins_grammar_group_and_guard():
    assert K.verify_pins() == K.H317_PINS
    assert any(a.startswith("/home/ubuntu/jope_plan/evidence/H335/") for a in C._ALLOW_PRE_V)
    assert "/home/ubuntu/jope-data/h335-stagev/" in C._DENY_PRE_V
    assert C.LOCK_RECEIPT.name == "H335_LOCK_sha256.txt"
    assert bool(getattr(G, "LEARNERS_V1", False)) == AGG
    assert K.GRAMMAR_SHA256 == (K.GRAMMAR_PORT_SHA256 if AGG else K.H317_PINS["gp_grammar_h317.py"])
    with pytest.raises(C.GuardViolation):
        open("/home/ubuntu/jope-data/h335-stagev/run/x.json")
    with pytest.raises(C.GuardViolation):
        open("/home/ubuntu/jope_plan/evidence/H258/inputs/r882/best_dev_checkpoint.json")
    assert (HERE / "h335_stats.py").read_text()  # own evidence dir stays readable


def test_panel_p201b(rows):
    p = K.panel_p201b(rows)
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    sizes = {}
    for r in rows:
        sizes[r["family_id"]] = sizes.get(r["family_id"], 0) + 1
    assert len(p) == sum(1 for v in sizes.values() if v >= 2) == 201
    assert len({fam[c] for c in p}) == len(p)
    assert not set(p) & set(C.pilot_shard(rows, C.family_folds(rows)))
    assert p == K.panel_p201b(list(reversed(rows)))


def test_runner_refuses_other_worker(tmp_path_factory):
    d = T.scratch("fakebin")
    b = d / "notsuperset"
    b.write_bytes(b"#!/bin/sh\n")
    with pytest.raises(RuntimeError):
        K.rust_runner_cpu(b, 60.0)


# ---------------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------------
def _brute_p(Dv):
    F = len(Dv)
    t = np.mean(Dv)
    hits = 0
    for m in range(1 << F):
        s = np.array([-1.0 if (m >> i) & 1 else 1.0 for i in range(F)])
        hits += float(np.mean(s * Dv)) >= t - 1e-12
    return hits / (1 << F)


def test_exact_signflip_matches_bruteforce():
    rng = np.random.default_rng(1)
    for F in (1, 3, 8, 11):
        Dv = rng.normal(0.05, 0.1, F)
        assert H.exact_signflip_p(Dv, chunk=7) == pytest.approx(_brute_p(Dv), abs=0)
    assert H.exact_signflip_p(np.ones(3)) == 1 / 8
    assert H.exact_signflip_p(np.array([1.0, -1.0])) == 3 / 4


def _panel(gains, n_ctx=3):
    fam_of, arm, ctrl = {}, {}, {}
    for f, g in enumerate(gains):
        for j in range(n_ctx):
            c = f"{f:04d}{j:060d}"
            fam_of[c] = f"fam{f}"
            ctrl[c] = 1.0
            arm[c] = 1.0 + g
    return arm, ctrl, fam_of


def test_primary_clip_lofo_verdicts():
    arm, ctrl, fam = _panel([0.2] * 18 + [3.0])
    r = H.primary_test(arm, ctrl, fam)
    assert r["pass"] and not r["inferior"] and r["T_clipped_family_mean"] == pytest.approx((0.2 * 18 + 0.5) / 19)
    assert r["signflip_p"] == pytest.approx(1 / 2 ** 19)
    a2, c2, f2 = _panel([0.05] * 17 + [-0.4, -0.4])
    r2 = H.primary_test(a2, c2, f2)
    assert r2["lofo_min"] < r2["T_clipped_family_mean"]
    inf = H.primary_test(*_panel([-0.1] * 19))
    assert inf["inferior"] and not inf["pass"]
    tie = H.primary_test(*_panel([0.0] * 19))
    assert not tie["pass"] and not tie["inferior"]
    P, N, I0 = {"pass": True, "inferior": False}, {"pass": False, "inferior": False}, {"pass": False, "inferior": True}
    assert H.verdict(P, P) == "BETTER_MODEL"
    assert H.verdict(P, N) == "BETTER_WITH_COMPUTE"
    assert H.verdict(I0, P) == "INFERIOR_TO_INCUMBENT"
    assert H.verdict(N, P) == "INCONCLUSIVE"
    h = H.headline(arm, ctrl, fam)
    assert h["pct_rmse_reduction"] == pytest.approx(1 - math.exp(-h["family_mean_gain"]))
    assert h["families_won"] == 19
    with pytest.raises(ValueError):
        H.family_contrast(arm, {k: v for k, v in list(ctrl.items())[1:]}, fam)


def test_elo_functions_equal_frozen_h317():
    import stage_v_h317 as SV

    rng = np.random.default_rng(7)
    cids = [f"{i:064d}" for i in range(40)]
    fam = {c: f"f{i % 9}" for i, c in enumerate(cids)}
    vals = {a: {c: float(rng.normal(i * 0.05, 0.3)) for c in cids} for i, a in enumerate(["A", "B", "C", "D"])}
    V = np.array([[vals[a][c] for a in vals] for c in cids])
    np.testing.assert_array_equal(H.family_wins_tensor(V, [fam[c] for c in cids]),
                                  SV.family_wins_tensor(V, [fam[c] for c in cids]))
    W = H.family_wins_tensor(V, [fam[c] for c in cids]).sum(axis=0)
    np.testing.assert_array_equal(H.bt_ratings(W, anchor=2), SV.bt_ratings(W, anchor=2))
    a = H.elo_panel(vals, fam, "B", seed=5, n_boot=1000)
    b = SV.elo_panel(vals, fam, "B", seed=5, n_boot=1000)
    assert a == b and a["elo"]["B"]["elo"] == pytest.approx(1500.0)


def test_phase_t_decision_rule():
    cids = [f"{i:064d}" for i in range(30)]
    fam = {c: f"f{i}" for i, c in enumerate(cids)}
    rng = np.random.default_rng(3)
    base = {c: float(rng.normal(1.5, 0.5)) for c in cids}
    lift = {10: 0.0, 25: 0.10, 50: 0.20, 75: 0.205, 107: 0.21}
    agg = {gc: {"CV3@12": {c: base[c] + lift[gc] + float(rng.normal(0, 0.05)) for c in cids},
                "single": {c: base[c] + lift[gc] - 0.02 for c in cids}} for gc in lift}
    dec = H.phase_t_decision(agg, fam, tuple(lift), "CV3@12")
    assert dec["G_star"] in (50, 75, 107) and dec["G_star"] <= dec["best"]
    assert dec["stack"] == "CV3@12"
    agg2 = {gc: {"CV3@12": {c: v - 0.5 for c, v in d["CV3@12"].items()}, "single": d["single"]}
            for gc, d in agg.items()}
    assert H.phase_t_decision(agg2, fam, tuple(lift), "CV3@12")["stack"] == "single"


# ---------------------------------------------------------------------------------------------------------------------
# Units (fake worker)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not AGG, reason="AGG group only")
def test_unit_agg_budget_checkpoints_resume(nonpanel):
    cid = nonpanel[0]
    rt = make_rt("agg", T.fake_runner(fail_frac=0.0))
    rec = LAB.unit_agg(rt, "T", cid, 335101, gens=4, checkpoints=(2, 4))
    assert rec["seed_F"]["evals_used"] == 22
    assert rec["evo"]["live_per_gen"] == {i: 6 for i in range(1, 5)} or \
        rec["evo"]["live_per_gen"] == {str(i): 6 for i in range(1, 5)}
    o = rec["outputs"]
    assert set(o) == {f"AGG.g{g}.{s}" for g in (2, 4) for s in LAB.STACKS}
    assert o["AGG.g2.single"]["live_evals"] == 22 + 12 and o["AGG.g4.single"]["live_evals"] == 22 + 24
    assert all(math.isfinite(x["value"]) for x in o.values())
    assert rt.ledger.check()["outputs"] == 4
    again = LAB.unit_agg(rt, "T", cid, 335101, gens=4, checkpoints=(2, 4))
    assert again == json.loads(json.dumps(rec, sort_keys=True, default=C._json_default))
    # prefix property: a 2-generation run gives the same g2 outputs (per-generation RNG streams, same memo-free fakes)
    rt2 = make_rt("agg_prefix", T.fake_runner(fail_frac=0.0))
    short = LAB.unit_agg(rt2, "T", cid, 335101, gens=2, checkpoints=(2,))
    for s in LAB.STACKS:
        assert short["outputs"][f"AGG.g2.{s}"]["value"] == o[f"AGG.g2.{s}"]["value"]


@pytest.mark.skipif(AGG, reason="BASE group only")
def test_unit_base_matching_and_references(nonpanel):
    cid = nonpanel[1]
    rt = make_rt("base", T.fake_runner(fail_frac=0.0))
    rec = LAB.unit_base(rt, "T", cid, 335101, gens=12, checkpoints=(2, 10))
    o = rec["outputs"]
    assert set(o) == set(LAB.unit_keys("BASE", (2, 10)))
    # OLD at AGG checkpoint gc uses gen gc + 2: 8 + 6 (gc + 2) = 6 gc + 20 vs AGG 6 gc + 22
    assert o["OLD.g2.single"]["live_evals"] == 8 + 6 * 4
    assert abs(o["OLD.g10.single"]["live_evals"] - (22 + 6 * 10)) == K.LIVE_MATCH_TOL
    assert o["H317_BASE.single"]["live_evals"] == 68
    assert o["H257_REPLICA.CV1"]["live_evals"] in (23, 24)  # 8 seeds + 3 x 5 (+1 if a CV extra eval)
    assert o["BANK_NN1.single"]["live_evals"] == 0
    rt2 = make_rt("base_prefix", T.fake_runner(fail_frac=0.0))
    ten = LAB.unit_base(rt2, "T", cid, 335101, gens=10, checkpoints=(2,))
    assert ten["outputs"]["H317_BASE.single"]["value"] == o["H317_BASE.single"]["value"]
    assert ten["outputs"]["H257_REPLICA.CV1"]["value"] == o["H257_REPLICA.CV1"]["value"]


def test_failed_unit_and_collect(nonpanel):
    rt = make_rt("failed", T.fake_runner(fail_frac=0.0))
    keys = LAB.unit_keys(K.GROUP, (10,))
    rec = LAB.failed_unit(rt, "V", K.GROUP, nonpanel[2], 335101, keys, "boom")
    assert all(rec["outputs"][k]["value"] == 0.0 and rec["outputs"][k]["invalid"] for k in keys)
    with pytest.raises(L.IncompleteStage):
        LAB.collect([rt.store.root], "V", [nonpanel[2], nonpanel[3]], (335101,))


def test_fixed_design_constants():
    assert K.SEEDS_T == (335101,) and K.SEEDS_V == (335101, 335102)
    assert (K.N_PROP, K.N_LIVE, K.G_MAX, K.CHECKPOINTS) == (48, 6, 107, (10, 25, 50, 75, 107))
    assert 22 + K.N_LIVE * K.G_MAX == 664
    assert (K.CLIP, K.ALPHA, K.STACK, K.ELO_ANCHOR) == (0.5, 0.025, "CV3@12", "H257_REPLICA")
    assert [x[1] for x in K.NEW_HEAD_PRESETS] == [
        "(pipe X (xtrees))", "(pipe X (rforest))", "(pipe X (oblivious))",
        "(pipe X (cheb append=true degree=3) (oblivious))",
        "(pipe X (rank) (xtrees max_features=0.7 min_leaf=3 trees=150))",
        "(pipe X (rforest max_features=0.33 min_leaf=2 trees=200))"]


def test_duplicate_checkpoints_and_segment_isolation(nonpanel, monkeypatch):
    """Red-team HIGH-1/HIGH-2: checkpoints {10, G*} with G* = 10 must not request AUDIT twice; in Phase V a
    deterministic exception in one pipeline marks only that pipeline's outputs failed."""
    cid = nonpanel[4]
    rt = make_rt("dupcp", T.fake_runner(fail_frac=0.0))
    if AGG:
        rec = LAB.unit_agg(rt, "V", cid, 335101, gens=2, checkpoints=(2, 2))
        assert set(rec["outputs"]) == {"AGG.g2.single", "AGG.g2.CV3@12"}
        rt2 = make_rt("seg_agg", T.fake_runner(fail_frac=0.0))
        real = LAB.L._stack_output

        def boom(rt_, pipe, g, res, cand_of, cfg):
            if cfg == "CV3@12":
                raise ValueError("synthetic CV3 failure")
            return real(rt_, pipe, g, res, cand_of, cfg)

        monkeypatch.setattr(LAB.L, "_stack_output", boom)
        rec2 = LAB.unit_agg(rt2, "V", cid, 335101, gens=2, checkpoints=(2,))
        assert rec2["outputs"]["AGG.g2.CV3@12"]["failed"] and rec2["outputs"]["AGG.g2.CV3@12"]["value"] == 0.0
        assert not rec2["outputs"]["AGG.g2.single"].get("failed")
        with pytest.raises(ValueError):  # Phase T: every exception propagates
            LAB.unit_agg(make_rt("seg_agg_t", T.fake_runner(fail_frac=0.0)), "T", cid, 335101, gens=2,
                         checkpoints=(2,))
    else:
        rec = LAB.unit_base(rt, "V", cid, 335101, gens=12, checkpoints=(10, 10))
        assert set(rec["outputs"]) == set(LAB.unit_keys("BASE", (10,)))
        rt2 = make_rt("seg_base", T.fake_runner(fail_frac=0.0))
        monkeypatch.setattr(LAB.D, "bank_nn8_pick", lambda g: (_ for _ in ()).throw(ValueError("synthetic NN8")))
        rec2 = LAB.unit_base(rt2, "V", cid, 335101, gens=12, checkpoints=(10,))
        assert rec2["outputs"]["BANK_NN8_PICK.single"]["failed"] and rec2["outputs"]["BANK_NN8_PORTFOLIO.CV5"]["failed"]
        for k in ("OLD.g10.single", "H257_REPLICA.CV1", "H317_BASE.single", "BANK_NN1.single"):
            assert not rec2["outputs"][k].get("failed")
        assert rec2["outputs"]["H257_REPLICA.CV1"]["value"] == rec["outputs"]["H257_REPLICA.CV1"]["value"]


def test_integrity_errors_propagate_in_phase_v(nonpanel, monkeypatch):
    rt = make_rt("integrity", T.fake_runner(fail_frac=0.0))
    fn = LAB.unit_agg if AGG else LAB.unit_base
    monkeypatch.setattr(LAB.L, "_stack_output", lambda *a, **k: (_ for _ in ()).throw(D.BankLeakError("leak")))
    with pytest.raises(AssertionError):
        fn(rt, "V", nonpanel[5], 335101, 12, (10,))
