"""Data-free H242 controls: SVD energy truncation, matched placebo, frozen inventory, selectors, decision rule and A1
guard. No study data is read."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
import spectral as sp

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def toy():
    g = torch.Generator().manual_seed(11)
    names = ("a.w", "a.b", "a.v", "a.r1", "a.t")
    state = {
        "a.w": torch.randn(8, 12, generator=g),
        "a.b": torch.randn(8, generator=g),
        "a.v": torch.randn(6, 5, generator=g),
        "a.r1": torch.randn(1, 7, generator=g),
        "a.t": torch.randn(1, 1, 4, generator=g),
    }
    return names, state


def test_arms_and_constants_are_frozen():
    assert stat.SEEDS == (242101, 242102, 242103)
    assert stat.FOLD_SEEDS == {242101: 242141, 242102: 242142, 242103: 242143}
    assert stat.RHO_POLICIES == ("r882", "RHO99", "RHO95", "RHO90")
    assert stat.ARMS == ("CONTROL", "SVD_SELECT", "LASER_SINGLE", "RANDPROJ")
    assert stat.DECISION_ARMS == ("SVD_SELECT", "LASER_SINGLE")
    assert sp.RHOS == (0.99, 0.95, 0.90) and sp.SINGLE_RHO == 0.90
    assert stat.ADVANCE_MARGIN == 0.002 and stat.PLATEAU_BEST == 1.1389572


def test_kept_rank_energy_rule():
    s = np.array([3.0, 2.0, 1.0, 0.5])  # energies 9, 4, 1, .25 of 14.25
    assert sp.kept_rank(s, 1.0) == 4
    assert sp.kept_rank(s, 0.90) == 2  # 13/14.25 = .912
    assert sp.kept_rank(s, 0.95) == 3  # 14/14.25 = .982
    assert sp.kept_rank(s, 0.99) == 4  # needs 14.1075 > 14
    assert sp.kept_rank(np.array([1.0, 0.0]), 0.99) == 1
    with pytest.raises(ValueError):
        sp.kept_rank(s, 0.0)


def test_truncation_is_optimal_rank_k_and_full_rank_is_bit_exact():
    w = torch.randn(8, 12, generator=torch.Generator().manual_seed(3))
    out, k, full = sp.truncate(w, 0.90)
    assert full == 8 and k < full
    assert torch.linalg.matrix_rank(out.double(), atol=1e-4) == k  # float32 cast noise
    s = torch.linalg.svdvals(w.double())
    change = float(torch.linalg.matrix_norm(w.double() - out.double()))
    assert change == pytest.approx(float(torch.sqrt((s[k:] ** 2).sum())), rel=1e-5)
    same, k1, _ = sp.truncate(w, 1.0)
    assert k1 == 8 and same.numpy().tobytes() == w.numpy().tobytes()
    r1 = torch.randn(1, 7)
    assert sp.truncate(r1, 0.9)[0].numpy().tobytes() == r1.numpy().tobytes()
    with pytest.raises(ValueError):
        sp.truncate(torch.randn(4), 0.9)


def test_reduce_all_touches_only_2d_matrices():
    names, state = toy()
    out = sp.reduce_all(state, names, 0.90)
    for n in ("a.b", "a.t", "a.r1"):
        assert out[n].numpy().tobytes() == state[n].numpy().tobytes()
    assert not torch.equal(out["a.w"], state["a.w"])
    only = sp.reduce_all(state, names, 0.90, which=["a.v"])
    assert torch.equal(only["a.w"], state["a.w"])
    assert not torch.equal(only["a.v"], state["a.v"])
    assert sp.matrices(state, names) == ["a.w", "a.v", "a.r1"]


def test_placebo_matches_frobenius_change_and_is_seeded():
    names, state = toy()
    for rho in sp.RHOS:
        trunc, k, full = sp.truncate(state["a.w"], rho)
        rand = sp.placebo(state["a.w"], rho, "a.w")
        t = float(torch.linalg.matrix_norm(state["a.w"].double() - trunc.double()))
        r = float(torch.linalg.matrix_norm(state["a.w"].double() - rand.double()))
        assert r == pytest.approx(t, rel=1e-5)
        if k < full:
            assert not torch.allclose(rand, trunc)
        assert torch.equal(rand, sp.placebo(state["a.w"], rho, "a.w"))
    assert torch.equal(sp.placebo(state["a.r1"], 0.9, "a.r1"), state["a.r1"])


def test_inventory_and_singles():
    names, state = toy()
    inv = sp.inventory(state, names)
    assert [r["name"] for r in inv] == ["a.w", "a.v", "a.r1"]
    for r in inv:
        for rho in sp.RHOS:
            assert 1 <= r[f"k_{rho:g}"] <= r["full_rank"]
            assert 0 <= r[f"removed_energy_{rho:g}"] <= 1 - rho + 1e-12
    assert "a.r1" not in sp.singles(inv)


def test_frozen_spectral_inventory_file():
    inv = json.loads((HERE / "spectral_inventory.json").read_text())
    assert inv["rhos"] == list(sp.RHOS) and inv["single_rho"] == 0.9
    assert len(inv["matrices"]) == 16
    assert sum(m["full_rank"] == 1 for m in inv["matrices"]) == 3
    assert len(inv["laser_single_candidates"]) == 13
    for m in inv["matrices"]:
        assert m["name"].startswith("adapter_10tok.")
        assert not m["name"].startswith("adapter_10tok.anchor.")
        assert m["k_0.99"] >= m["k_0.95"] >= m["k_0.9"]


def test_selector_ties_and_noise_keep_r882():
    r = rows(40)
    labels = stat.RHO_POLICIES
    flat = np.ones((40, 4))
    d = stat.choose(r, flat, 1, labels)
    assert d["best"] == "r882" and d["selected"] == "r882" and not d["accepted"]
    noisy = flat.copy()
    noisy[:, 3] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1, labels)
    assert d["best"] == "RHO90" and d["selected"] == "r882"
    tiny = flat.copy()
    tiny[:, 1] += 0.0005
    assert stat.choose(r, tiny, 1, labels)["selected"] == "r882"


def test_selector_accepts_clear_gain_and_prefers_smallest_perturbation():
    r = rows(40)
    panel = np.ones((40, 4))
    panel[:, 2:] += 0.01
    d = stat.choose(r, panel, 3, stat.RHO_POLICIES)
    assert d["accepted"] and d["selected"] == "RHO95"
    with pytest.raises(ValueError):
        stat.choose(r, panel[:, :3], 3, stat.RHO_POLICIES)


def test_folds_partition_family_disjoint_and_manifest():
    r = rows()
    for seed in stat.SEEDS:
        groups = stat.folds(r, seed)
        assert sorted(sum(groups, [])) == sorted({x["family_id"] for x in r})
        for outer in range(5):
            fit, assess = stat.partition(r, outer, seed)
            assert not {x["family_id"] for x in fit} & {x["family_id"] for x in assess}
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seeds"] == {str(k): v for k, v in stat.FOLD_SEEDS.items()}
    for groups in fm["families_by_seed"].values():
        assert [len(g) for g in groups] == [41] * 5


def test_decision_rule_holm_over_two_decision_arms_only():
    r = rows(19)
    panel = np.ones((3, 19, 5))
    out = stat.summarize(r, panel)
    for arm in stat.DECISION_ARMS:
        c = out["contrasts"][arm + "_minus_r882"]
        assert c["median_gain"] == 0 and not c["advance"] and c["holm_p"] == 1.0
    assert "holm_p" not in out["contrasts"]["RANDPROJ_minus_r882"]
    assert "RANDPROJ_minus_SVD_SELECT" in out["contrasts"]
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    lift[:, :, 4] += 0.01  # a placebo lift never advances
    out = stat.summarize(r, lift)
    assert out["contrasts"]["SVD_SELECT_minus_r882"]["advance"]
    assert not out["contrasts"]["LASER_SINGLE_minus_r882"]["advance"]
    assert "advance" not in out["contrasts"]["RANDPROJ_minus_r882"]
    small = panel.copy()
    small[:, :, 2] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["SVD_SELECT_minus_r882"]["advance"]
    two = panel.copy()
    two[:2, :, 3] += 0.01  # two seeds positive, one zero: median .01 > margin
    c = stat.summarize(r, two)["contrasts"]["LASER_SINGLE_minus_r882"]
    assert c["positive_seeds"] == 2


def test_harm_plateau_and_fixed_readout():
    r = rows(19)
    panel = np.ones((3, 19, 5))
    panel[:, 0, 2] -= 1.0
    out = stat.summarize(r, panel)
    assert (
        out["contrasts"]["SVD_SELECT_minus_r882"]["harm_families_by_seed"]
        == [["f000"]] * 3
    )
    assert out["vs_plateau_best_by_seed"][0][0] == pytest.approx(1 - stat.PLATEAU_BEST)
    fixed = np.ones((19, 3))
    fixed[:, 1] += 0.01
    fr = stat.fixed_readout(r, fixed, ["X", "Y"])
    assert fr["X"]["families_wtl"] == [19, 0, 0] and fr["Y"]["families_wtl"] == [
        0,
        19,
        0,
    ]


def test_validate_launch_binding():
    launch = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        review_id=5,
        prereg_pr=7,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    sp.validate_launch(launch, "h", "m", "p", "f")
    sp.validate_launch(dict(manifest_sha256="m"), "x", "m", "p", "f", smoke=True)
    for key, bad in [("source_commit", "z"), ("review_id", 0), ("prereg_sha256", "q")]:
        with pytest.raises(RuntimeError):
            sp.validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")


def sample(t, available, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[44, 45, 46, 47],
    )


def test_a1_memory_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793680000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 1, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 32, 26 * GIB)) == ([], False)
    h = mon.Guard()
    h.update(sample(t0, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in h.update(sample(t0 + 31, 11 * GIB))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.COMPUTE_STOP == 1793691000.0  # 2026-11-03T07:30:00Z
    assert "absolute compute stop 07:30Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )
    assert "admission RAM<30GiB" in mon.admission_refusals(sample(t0, 29 * GIB))


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H242" and m["total_updates"] == 0
    assert m["checkpoint"] in m["inputs"]
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert "spectral_inventory.json" in m["source_sha256"]
