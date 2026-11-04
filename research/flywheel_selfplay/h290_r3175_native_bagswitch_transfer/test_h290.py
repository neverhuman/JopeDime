import numpy as np
import statistics_h290 as s


def test_ties_are_not_sign_wins():
    assert s.sign(np.array([0.0, 1e-4, -1e-4])) == dict(
        wins=0, ties=3, losses=0, p_one_sided=1.0
    )
    assert s.sign(np.ones(6))["p_one_sided"] == 0.015625


def test_family_weighting_and_pairing():
    u = np.zeros((3, 4, 6))
    u[:, :3, 1] = 1
    u[:, 3, 1] = -1
    out, fm, ix = s.summarize(
        u, np.zeros_like(u, dtype=bool), ["a", "a", "a", "b"], draws=40
    )
    assert out["arms"]["NATIVE"]["gain"]["per_seed"] == [0.0, 0.0, 0.0]
    assert fm.shape == (3, 2, 6) and ix.shape == (40, 2)
    assert out["arms"]["BASE"]["elo"]["per_seed"] == [1500.0] * 3


def test_raw_minus2_is_separate_from_gate():
    u = np.zeros((3, 2, 6))
    u[:, :, 1] = -2
    out, _, _ = s.summarize(u, np.zeros_like(u, dtype=bool), ["a", "b"], draws=30)
    assert out["arms"]["NATIVE"]["raw_minus2"] == [2] * 3
    assert out["arms"]["NATIVE"]["gate_invalid"] == [0] * 3


def test_bt_exact_inherited_implementation():
    from dependencies import CLOSURE
    import ast

    tree = ast.parse((CLOSURE / "analysis.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "bt_ratings"
    )
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "inherited", "exec"), ns)
    w = np.array([[0.0, 5.0, 1.0], [2.0, 0.0, 6.0], [3.0, 1.0, 0.0]])
    assert np.array_equal(s.bt_ratings(w), ns["bt_ratings"](w))


def test_null_passing_prevents_transfer():
    import copy

    good = dict(
        gain=dict(median=0.03, per_seed_ci95=[[0.01, 0.04]] * 3),
        sign_test=[dict(p_one_sided=0.01)] * 3,
        harmed_loss_gt_001=[0] * 3,
        raw_minus2=[0] * 3,
    )
    bad = copy.deepcopy(good)
    bad["gain"]["median"] = 0.0
    a = {k: copy.deepcopy(bad) for k in s.ARMS}
    a["NATIVE"] = copy.deepcopy(good)
    summary = dict(arms=a, native_minus_r882_fit=dict(median=0.01))
    assert s.verdict(summary)["verdict"] == "NATIVE_TRANSFER_PASS"
    a["NULL_RV"] = copy.deepcopy(good)
    assert s.verdict(summary)["verdict"] == "NOT_TRANSFERRED_SEMI_FRESH"


def test_no_superiority_means_not_better():
    good = dict(
        gain=dict(median=0.03, per_seed_ci95=[[0.01, 0.04]] * 3),
        sign_test=[dict(p_one_sided=0.01)] * 3,
        harmed_loss_gt_001=[0] * 3,
        raw_minus2=[0] * 3,
    )
    import copy

    a = {k: copy.deepcopy(good) for k in s.ARMS}
    for k in ["PLACEBO", "NULL_RV", "NULL_LF"]:
        a[k]["gain"]["median"] = 0.0
    assert (
        s.verdict(dict(arms=a, native_minus_r882_fit=dict(median=0)))["verdict"]
        == "NATIVE_NOT_BETTER"
    )
