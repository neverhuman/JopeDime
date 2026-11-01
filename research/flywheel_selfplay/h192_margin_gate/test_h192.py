"""H192 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import analysis as stat

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h192_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


def rows(n=205, per=2):
    return [
        {"context_sha256": f"c{f}-{i}", "family_id": f"f{f:03}"}
        for f in range(n)
        for i in range(per)
    ]


def gates_for(train, seed=0, agree_frac=0.6):
    rng = np.random.default_rng(seed)
    out = {}
    for r in train:
        agree = rng.random() < agree_frac
        u_top1 = float(rng.normal(1.0, 0.3))
        u_base = (
            u_top1 if agree else float(rng.choice([u_top1 - 0.5, u_top1 + 0.05, -2.0]))
        )
        out[r["context_sha256"]] = dict(
            base=0 if agree else 1,
            top1=0,
            margin=0.0 if agree else float(rng.exponential(0.5)),
            u_base=u_base,
            u_top1=u_top1,
        )
    return out


def folds(train):
    fams = [r["family_id"] for r in train]
    return {s: stat.family_folds(fams, seed=stat.FOLD_SEEDS[s]) for s in stat.SEEDS}


def test_registered_constants_and_distinct_seed_partitions():
    assert stat.SEEDS == (192101, 192102, 192103) and stat.BOOT_SEED == 192150
    assert stat.FOLD_SEEDS == {192101: 192141, 192102: 192142, 192103: 192143}
    f = folds(rows())
    assert len({str(g) for g in f.values()}) == 3  # partitions differ by seed
    for groups in f.values():
        a_cov = []
        for outer in range(5):
            fit, b, a = stat.partition({f"f{i:03}" for i in range(205)}, outer, groups)
            assert not fit & (a | b) and not a & b
            a_cov += list(a)
        assert sorted(a_cov) == sorted({f"f{i:03}" for i in range(205)})


def test_gate_inputs_margin_definition_and_tau0_identity_with_ties():
    full = np.array(
        [0.5, 0.2, 0.2, 1.0]
    )  # tie at the minimum: argmin is the first index (1)
    anchor = np.array([0.1, 0.9, 0.3, 0.8])  # base pick 0
    fitness, gate = np.array([0.1, 0.4, 0.3, 0.2]), np.array([True, True, True, False])
    g = stat.gate_inputs(full, anchor, fitness, gate)
    assert (
        g["base"] == 0
        and g["top1"] == 1
        and g["margin"] == pytest.approx((0.5 - 0.2) / full.std())
    )
    assert stat.gated_pick(g, 0.0) == (
        1,
        -0.4,
    )  # tau 0 = the full argmin incl. its tie-break
    assert stat.gated_pick(g, g["margin"]) == (
        0,
        -0.1,
    )  # not strictly greater -> keep base
    agree = stat.gate_inputs(full, np.array([0.9, 0.1, 0.5, 0.6]), fitness, gate)
    assert agree["margin"] == 0.0 and stat.gated_pick(agree, 5.0) == stat.gated_pick(
        agree, 0.0
    )
    flat = stat.gate_inputs(np.ones(4), anchor, fitness, gate)
    assert flat["margin"] == 0.0


def test_tau_grid_uses_fit_disagreements_only_and_selection_ignores_a():
    train = rows()
    gates = gates_for(train)
    f = folds(train)
    fit, b, a = stat.partition({r["family_id"] for r in train}, 0, f[192101])
    fit_rows = [r for r in train if r["family_id"] in fit]
    b_rows = [r for r in train if r["family_id"] in b]
    base = stat.select_cell(fit_rows, b_rows, gates)
    assert (
        base["grid"][0] == 0.0
        and base["grid"][1:]
        == stat.tau_grid([gates[r["context_sha256"]]["margin"] for r in fit_rows])[1:]
    )
    perturbed = copy.deepcopy(gates)
    for r in train:
        if r["family_id"] in a | b:  # non-FIT margins must not move the grid
            perturbed[r["context_sha256"]]["margin"] *= 10.0
    assert stat.select_cell(fit_rows, b_rows, perturbed)["grid"] == base["grid"]
    lock = stat.stage1(train, gates, f)
    a_changed = copy.deepcopy(gates)
    for cell in lock["cells"][:1]:
        for r in train:
            if r["family_id"] in set(cell["a_families"]):
                a_changed[r["context_sha256"]]["u_base"] += 7.0
                a_changed[r["context_sha256"]]["u_top1"] += 7.0
    relock = stat.stage1(train, a_changed, f)
    assert (
        relock["cells"][0] == lock["cells"][0]
        and lock["written_before_any_A_or_DL_Val_scoring"]
    )


def test_tau_star_tie_prefers_tau0_and_stage2_tau0_equals_r882():
    train = rows()
    same = {
        r["context_sha256"]: dict(base=0, top1=0, margin=0.0, u_base=1.0, u_top1=1.0)
        for r in train
    }
    lock = stat.stage1(train, same, folds(train))
    assert all(c["tau_star_index"] == 0 for c in lock["cells"])
    gates = gates_for(train, seed=3)
    lock = stat.stage1(train, gates, folds(train))
    panels, override = stat.stage2_panels(train, gates, lock)
    assert np.array_equal(panels["TAU0"], panels["R882"]) and panels["MGATE"].shape == (
        3,
        len(train),
    )
    assert (
        override["TAU0"][0] == override["TAU0"][1]
        and override["Q90"][0] <= override["Q25"][0]
    )


def test_pooled_select_m1_tie_and_verdict_gates():
    train = rows()
    base = np.ones((3, len(train)))
    sel = stat.pooled_select(train, base, {"MGATE": base + 0.01})
    assert sel["MGATE"]["admitted"] and sel["MGATE"]["holm_multiplicity"] == 1
    assert not stat.pooled_select(train, base, {"MGATE": base + 0.001})["MGATE"][
        "admitted"
    ]  # exactly the margin ties
    con = stat.median_seed_contrast(train, base + 0.01, base)
    assert stat.verdict(sel, con, True)["verdict"] == "ADVANCE"
    assert (
        stat.verdict(sel, con, False)["verdict"] == "NO_ADVANCE"
    )  # H178 receipt gate required
    flat = stat.median_seed_contrast(train, base, base)
    assert (
        stat.verdict(stat.pooled_select(train, base, {"MGATE": base}), flat, True)[
            "verdict"
        ]
        == "NO_ADVANCE"
    )


def test_h178_gate_binds_the_h177_analysis_bytes():
    receipt = dict(
        status="COMPLETE",
        null_and_harmful_admission_le_005=True,
        h177_analysis_sha256="abc",
    )
    assert stat.h178_gate(receipt, "abc") and not stat.h178_gate(receipt, "def")
    assert not stat.h178_gate(
        dict(receipt, null_and_harmful_admission_le_005=False), "abc"
    )


def test_elo_identical_is_1500_and_axis0_bootstrap_is_non_degenerate():
    train = rows()
    rng = np.random.default_rng(192999)
    base = np.tile(rng.normal(1.0, 0.3, len(train)), (3, 1))
    assert stat.elo_pair(train, base.copy(), base)["median"] == pytest.approx(1500.0)
    noisy = (
        base
        + rng.normal(0.0, 0.2, base.shape)
        + np.array([0.002, 0.0015, 0.001])[:, None]
    )
    con = stat.median_seed_contrast(train, noisy, base)
    assert con["ci95"][0] < 0.0 < con["ci95"][1]


def test_monitor_guards():
    ok = dict(
        load=20.0,
        cpu=30.0,
        available=70 * 2**30,
        free=500e9,
        disk_used_fraction=0.93,
        rss=1e9,
        threads=3,
    )
    assert monitor.violations(ok) == []
    assert monitor.violations(dict(ok, disk_used_fraction=0.938)) == [
        "disk user-visible used>=93.8%"
    ]
    assert monitor.violations(dict(ok, available=15 * 2**30)) == ["MemAvailable<16GiB"]
