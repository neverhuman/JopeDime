"""Data-free H243 controls: module-group soups, matched placebo, frozen groups, selectors, greedy soup, decision rule
and A1 guard. No study data is read."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
import soup as sp

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def toy():
    g = torch.Generator().manual_seed(5)
    names = ("m.a", "m.b", "n.a", "n.b")
    anchor = {n: torch.randn(3, 4, generator=g) for n in names}
    prev = {n: torch.randn(3, 4, generator=g) for n in names}
    groups = {"M": ["m.a", "m.b"], "N": ["n.a", "n.b"]}
    return names, anchor, prev, groups


def test_constants_are_frozen():
    assert stat.SEEDS == (243101, 243102, 243103)
    assert stat.FOLD_SEEDS == {243101: 243141, 243102: 243142, 243103: 243143}
    assert stat.ARMS == ("CONTROL", "MODSWAP_SELECT", "GREEDY_SOUP", "PLACEBO")
    assert stat.DECISION_ARMS == ("MODSWAP_SELECT", "GREEDY_SOUP")
    assert sp.BETAS == (0.5, 1.0) and stat.GREEDY_MAX_INGREDIENTS == 5
    assert stat.CLIFF_CONTEXT == 27
    assert stat.ADVANCE_MARGIN == 0.002 and stat.PLATEAU_BEST == 1.1389572


def test_frozen_groups_partition_the_47_adapter_tensors():
    groups = json.loads((HERE / "groups.json").read_text())["groups"]
    assert len(groups) == 11
    flat = [n for g in groups.values() for n in g]
    assert len(flat) == 47 == len(set(flat))
    assert all(n.startswith("adapter_10tok.") and ".anchor." not in n for n in flat)
    sp.check_groups(groups, flat)
    with pytest.raises(ValueError):
        sp.check_groups(groups, flat[:-1])
    with pytest.raises(ValueError):
        sp.check_groups(dict(groups, X=[flat[0]]), flat)


def test_soup_touches_only_the_recipe_groups():
    names, anchor, prev, groups = toy()
    out = sp.soup(anchor, prev, {"M": 1.0}, groups)
    for n in ("m.a", "m.b"):
        assert out[n].numpy().tobytes() == prev[n].numpy().tobytes()
    for n in ("n.a", "n.b"):
        assert out[n].numpy().tobytes() == anchor[n].numpy().tobytes()
    half = sp.soup(anchor, prev, {"N": 0.5}, groups)
    want = (0.5 * anchor["n.a"].double() + 0.5 * prev["n.a"].double()).float()
    assert torch.equal(half["n.a"], want)
    assert torch.equal(half["m.a"], anchor["m.a"])
    same = sp.soup(anchor, prev, {}, groups)
    assert all(torch.equal(same[n], anchor[n]) for n in names)
    with pytest.raises(ValueError):
        sp.soup(anchor, prev, {"M": 0.0}, groups)


def test_group_l2_and_matched_placebo():
    names, anchor, prev, groups = toy()
    for beta in sp.BETAS:
        target = sp.group_l2(anchor, prev, groups["M"], beta)
        swap = sp.soup(anchor, prev, {"M": beta}, groups)
        assert sp.adapter_l2(anchor, swap) == pytest.approx(target, rel=1e-6)
        pl = sp.placebo(anchor, prev, groups, "M", beta)
        assert sp.adapter_l2(anchor, pl) == pytest.approx(target, rel=1e-6)
        assert torch.equal(pl["n.a"], anchor["n.a"])
        assert not torch.allclose(pl["m.a"], swap["m.a"])
        assert torch.equal(
            pl["m.a"], sp.placebo(anchor, prev, groups, "M", beta)["m.a"]
        )


def test_labels():
    assert sp.label("G01_modality", 0.5) == "G01_modality@0.5"
    assert sp.label("G01_modality", 1.0) == "G01_modality@1"
    assert stat.recipe_label({}) == "r882"
    assert stat.recipe_label({"G02": 1.0}) == "G02@1"
    assert stat.recipe_label({"G09": 0.5, "G02": 1.0}) == "SOUP:G02@1+G09@0.5"


def test_selector_strict_and_tie_order():
    r = rows(40)
    labels = ["r882", "A@0.5", "A@1", "B@0.5"]
    flat = np.ones((40, 4))
    assert stat.choose(r, flat, 1, labels)["selected"] == "r882"
    noisy = flat.copy()
    noisy[:, 2] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1, labels)
    assert d["best"] == "A@1" and d["selected"] == "r882"
    clear = flat.copy()
    clear[:, 1:] += 0.01
    assert stat.choose(r, clear, 1, labels)["selected"] == "A@0.5"


def toy_greedy(gains, pair_bonus=0.0):
    """Synthetic FIT values: each group adds its gain; recipes are additive plus an optional pair interaction."""
    r = rows(40)
    base = np.ones(40)
    singles = {}
    for g, (g05, g10) in gains.items():
        singles[(g, 0.5)] = base + g05
        singles[(g, 1.0)] = base + g10
    calls = []

    def score(recipe):
        calls.append(dict(recipe))
        total = sum(gains[g][0 if b == 0.5 else 1] for g, b in recipe.items())
        return base + total + (pair_bonus if len(recipe) >= 2 else 0.0)

    return r, base, singles, score, calls


def test_greedy_adds_in_gain_order_and_keeps_strict_improvements():
    r, base, singles, score, calls = toy_greedy(
        {
            "A": (0.004, 0.010),
            "B": (0.006, 0.003),
            "C": (0.0005, -0.01),
            "D": (-0.02, -0.03),
        }
    )
    out = stat.greedy(r, base, singles, sp.BETAS, score, 7)
    assert [t["group"] for t in out["trace"]] == ["A", "B"]  # C below .001, D negative
    assert out["recipe"] == {"A": 1.0, "B": 0.5}
    assert out["accepted"] and out["selected"] == {"A": 1.0, "B": 0.5}
    assert calls[0] == {"A": 1.0}


def test_greedy_rejects_additions_that_do_not_improve_and_falls_back():
    r, base, singles, score, _ = toy_greedy(
        {"A": (0.004, 0.010), "B": (0.006, 0.003)}, pair_bonus=-0.02
    )
    out = stat.greedy(r, base, singles, sp.BETAS, score, 7)
    assert out["recipe"] == {"A": 1.0}
    assert [t["kept"] for t in out["trace"]] == [True, False]
    r, base, singles, score, calls = toy_greedy({"A": (0.0, 0.0005)})
    out = stat.greedy(r, base, singles, sp.BETAS, score, 7)
    assert out["recipe"] == {} and out["selected"] == {} and not out["accepted"]
    assert calls == []


def test_greedy_strict_final_check_uses_lcb():
    r = rows(40)
    base = np.ones(40)
    noisy = base + np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    singles = {("A", 0.5): base, ("A", 1.0): noisy}
    out = stat.greedy(r, base, singles, sp.BETAS, lambda recipe: noisy, 3)
    assert out["recipe"] == {"A": 1.0} and out["selected"] == {} and not out["accepted"]


def test_greedy_caps_ingredients():
    gains = {f"G{i:02}": (0.002 * (i + 1), 0.0) for i in range(8)}
    r, base, singles, score, _ = toy_greedy(gains)
    out = stat.greedy(r, base, singles, sp.BETAS, score, 7)
    assert len(out["trace"]) == stat.GREEDY_MAX_INGREDIENTS
    assert out["trace"][0]["group"] == "G07"


def test_folds_and_manifest():
    r = rows()
    for seed in stat.SEEDS:
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
    assert "advance" not in out["contrasts"]["PLACEBO_minus_r882"]
    assert "PLACEBO_minus_MODSWAP_SELECT" in out["contrasts"]
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    lift[:, :, 4] += 0.01
    out = stat.summarize(r, lift)
    assert out["contrasts"]["MODSWAP_SELECT_minus_r882"]["advance"]
    assert not out["contrasts"]["GREEDY_SOUP_minus_r882"]["advance"]
    small = panel.copy()
    small[:, :, 2] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["MODSWAP_SELECT_minus_r882"][
        "advance"
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
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    h = mon.Guard()
    h.update(sample(t0, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in h.update(sample(t0 + 31, 11 * GIB))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.COMPUTE_STOP == 1793691000.0
    assert "admission RAM<30GiB" in mon.admission_refusals(sample(t0, 29 * GIB))


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H243" and m["total_updates"] == 0
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["inputs"][m["previous"]].startswith("427b0d46")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert "groups.json" in m["source_sha256"]
