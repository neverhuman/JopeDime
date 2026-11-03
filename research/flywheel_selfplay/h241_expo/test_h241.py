"""Data-free H241 controls: extrapolation algebra, alpha selector, decision rule and A1 guard. No study data is read."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
from expo import adapter_l2, extrapolate, nonadapter_differences, validate_launch

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def toy_states():
    g = torch.Generator().manual_seed(7)
    names = ("adapter.a", "adapter.b")
    anchor = {n: torch.randn(3, 4, generator=g) for n in names}
    prev = {n: torch.randn(3, 4, generator=g) for n in names}
    return names, anchor, prev


def test_arms_alphas_and_fixed_points_are_frozen():
    assert stat.SEEDS == (241101, 241102, 241103)
    assert stat.FOLD_SEEDS == {241101: 241141, 241102: 241142, 241103: 241143}
    assert stat.ALPHAS == (0.0, 0.25, 0.5, 1.0)
    assert stat.DIRECTIONS == (782, 859)
    assert stat.ARMS == ("CONTROL", "EXPO782", "EXPO859")
    assert stat.ALL_ARMS[0] == "r882"
    assert stat.FIXED == (
        "R782_A0.25",
        "R782_A0.5",
        "R782_A1",
        "R859_A0.25",
        "R859_A0.5",
        "R859_A1",
    )
    assert stat.PRIMARY_FIXED in stat.FIXED
    assert stat.ADVANCE_MARGIN == 0.002 and stat.PLATEAU_BEST == 1.1389572


def test_alpha_zero_is_bit_exact_anchor():
    names, anchor, prev = toy_states()
    out = extrapolate(anchor, prev, 0.0, names)
    for n in names:
        assert out[n].dtype == anchor[n].dtype
        assert out[n].numpy().tobytes() == anchor[n].numpy().tobytes()
        assert out[n] is not anchor[n]


def test_extrapolation_formula_float64_and_distance():
    names, anchor, prev = toy_states()
    base = adapter_l2(prev, anchor)
    for alpha in stat.ALPHAS[1:]:
        out = extrapolate(anchor, prev, alpha, names)
        for n in names:
            want = (
                anchor[n].double() + alpha * (anchor[n].double() - prev[n].double())
            ).float()
            assert torch.equal(out[n], want)
        assert adapter_l2(anchor, out) == pytest.approx(alpha * base, rel=1e-6)
    # alpha = 1 doubles the last accepted step: theta = 2 r882 - r_prev.
    one = extrapolate(anchor, prev, 1.0, names)
    assert torch.allclose(one["adapter.a"], 2 * anchor["adapter.a"] - prev["adapter.a"])


def test_extrapolation_refuses_bad_inputs():
    names, anchor, prev = toy_states()
    with pytest.raises(ValueError):
        extrapolate(anchor, prev, -0.5, names)
    with pytest.raises(ValueError):
        extrapolate(anchor, {"adapter.a": prev["adapter.a"]}, 0.5, names)
    bad = dict(prev, **{"adapter.b": torch.zeros(4, 3)})
    with pytest.raises(ValueError):
        extrapolate(anchor, bad, 0.5, names)
    inf = dict(prev, **{"adapter.a": torch.full((3, 4), float("inf"))})
    with pytest.raises(ValueError):
        extrapolate(anchor, inf, 0.5, names)


def test_nonadapter_identity_check():
    names, anchor, prev = toy_states()
    a = dict(anchor, frozen=torch.ones(2))
    b = dict(prev, frozen=torch.ones(2))
    assert nonadapter_differences(a, b, names) == []
    b["frozen"] = torch.tensor([1.0, 2.0])
    assert nonadapter_differences(a, b, names) == ["frozen"]
    assert nonadapter_differences(a, dict(prev), names) == ["<key set differs>"]


def test_selector_ties_and_noise_keep_r882():
    r = rows(40)
    flat = np.ones((40, 4))
    d = stat.choose(r, flat, 1)
    assert d["best_alpha"] == 0.0 and d["selected_alpha"] == 0.0 and not d["accepted"]
    tiny = flat.copy()
    tiny[:, 2] += 0.0005  # below the .001 margin
    assert stat.choose(r, tiny, 1)["selected_alpha"] == 0.0
    noisy = flat.copy()
    noisy[:, 3] += np.where(
        np.arange(40) % 2 == 0, 0.5, -0.49
    )  # mean gain .005 but LCB < 0
    d = stat.choose(r, noisy, 1)
    assert d["best_alpha"] == 1.0 and d["selected_alpha"] == 0.0


def test_selector_accepts_clear_gain_and_prefers_smallest_tied_alpha():
    r = rows(40)
    panel = np.ones((40, 4))
    panel[:, 1:] += 0.01  # alphas .25/.5/1 tie
    d = stat.choose(r, panel, 3)
    assert d["accepted"] and d["best_alpha"] == 0.25 and d["selected_alpha"] == 0.25
    with pytest.raises(ValueError):
        stat.choose(r, panel[:, :3], 3)


def test_folds_partition_family_disjoint():
    r = rows()
    for seed in stat.SEEDS:
        groups = stat.folds(r, seed)
        assert sorted(sum(groups, [])) == sorted({x["family_id"] for x in r})
        for outer in range(5):
            fit, assess = stat.partition(r, outer, seed)
            assert not {x["family_id"] for x in fit} & {x["family_id"] for x in assess}
            assert len(fit) + len(assess) == 205
    assert stat.folds(r, 241101) != stat.folds(r, 241102)


def test_fold_manifest_matches_seeds():
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seeds"] == {str(k): v for k, v in stat.FOLD_SEEDS.items()}
    assert set(fm["families_by_seed"]) == {str(s) for s in stat.SEEDS}
    for groups in fm["families_by_seed"].values():
        assert [len(g) for g in groups] == [41] * 5


def test_zero_gain_never_advances_and_rule_needs_all_gates():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    out = stat.summarize(r, panel)
    for arm in stat.ARMS[1:]:
        c = out["contrasts"][arm + "_minus_r882"]
        assert c["median_gain"] == 0 and not c["advance"]
        assert c["holm_p"] == 1.0 and c["harm_families_by_seed"] == [[], [], []]
    assert out["median_elo"][0] == pytest.approx(1500)
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    out = stat.summarize(r, lift)
    c = out["contrasts"]["EXPO782_minus_r882"]
    assert c["advance"] and c["positive_seeds"] == 3
    assert not out["contrasts"]["EXPO859_minus_r882"]["advance"]
    small = panel.copy()
    small[:, :, 2] += 0.0015  # consistent but below the +0.002 margin
    assert not stat.summarize(r, small)["contrasts"]["EXPO782_minus_r882"]["advance"]


def test_harm_and_plateau_reporting():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    panel[:, 0, 3] -= 1.0  # family f000 harmed by EXPO859 in all seeds
    out = stat.summarize(r, panel)
    assert (
        out["contrasts"]["EXPO859_minus_r882"]["harm_families_by_seed"]
        == [["f000"]] * 3
    )
    assert out["vs_plateau_best_by_seed"][0][0] == pytest.approx(1 - stat.PLATEAU_BEST)


def test_fixed_readout_shape_and_identity():
    r = rows(19)
    panel = np.ones((19, 7))
    panel[:, 2] += 0.01
    out = stat.fixed_readout(r, panel)
    assert list(out) == list(stat.FIXED)
    assert out["R782_A0.5"]["gain"] == pytest.approx(0.01)
    assert out["R782_A0.5"]["families_wtl"] == [19, 0, 0]
    assert out["R782_A0.25"]["families_wtl"] == [0, 19, 0]
    with pytest.raises(ValueError):
        stat.fixed_readout(r, panel[:, :6])


def test_holm_two_directions():
    assert stat.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert stat.holm({"a": 0.03, "b": 0.02}) == {"b": 0.04, "a": 0.04}


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
    validate_launch(launch, "h", "m", "p", "f")
    validate_launch(dict(manifest_sha256="m"), "x", "m", "p", "f", smoke=True)
    for key, bad in [("source_commit", "z"), ("review_id", 0), ("prereg_sha256", "q")]:
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        validate_launch(launch, "h", "other", "p", "f")


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


def test_a1_memory_guard_pause_resume_and_hard_stops():
    g = mon.Guard()
    t0 = 1793680000.0
    assert g.update(sample(t0, 40 * GIB)) == ([], False)
    assert g.update(sample(t0 + 1, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 2, 24 * GIB))[1] is True  # resume needs >= 25 GiB
    assert g.update(sample(t0 + 3, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 20, 26 * GIB))[1] is True  # not yet 30 s sustained
    assert g.update(sample(t0 + 33, 26 * GIB)) == ([], False)
    h = mon.Guard()
    assert h.update(sample(t0, 11 * GIB)) == ([], True)
    reasons, _ = h.update(sample(t0 + 31, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in reasons
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]


def test_load_pause_and_admission_and_deadline():
    g = mon.Guard()
    t0 = 1793680000.0
    assert g.update(sample(t0, 40 * GIB, load5=109))[1] is True
    assert g.update(sample(t0 + 1, 40 * GIB, load5=104))[1] is True
    assert g.update(sample(t0 + 2, 40 * GIB, load5=100))[1] is False
    assert mon.COMPUTE_STOP == 1793689200.0  # 2026-11-03T07:00:00Z
    assert "absolute compute stop 07:00Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )
    refusals = mon.admission_refusals(sample(t0, 29 * GIB, load5=101))
    assert "admission RAM<30GiB" in refusals and "admission load5>100" in refusals
    assert mon.admission_refusals(sample(t0, 31 * GIB)) == []


def test_manifest_pins_inputs_and_directions():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H241" and m["total_updates"] == 0
    assert m["checkpoint"] in m["inputs"]
    assert set(m["previous"]) == {"782", "859"}
    assert all(p in m["inputs"] for p in m["previous"].values())
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["inputs"][m["previous"]["782"]].startswith("427b0d46")
    assert m["inputs"][m["previous"]["859"]].startswith("5802245f")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
