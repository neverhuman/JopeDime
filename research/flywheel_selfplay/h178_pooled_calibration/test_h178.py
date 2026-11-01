"""Data-free unit tests for H178 calibration of the H177 pooled cross-fit selector."""

import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calibration as cal  # noqa: E402
import monitor  # noqa: E402
import runner  # noqa: E402


def toy_cohort(n_families=205, contexts_per_family=1):
    rows = []
    for f_idx in range(n_families):
        for c_idx in range(contexts_per_family):
            cid = f"ctx_{f_idx:03d}_{c_idx:02d}"
            rows.append(dict(context_sha256=cid, family_id=f"fam_{f_idx:03d}"))
    return rows


def test_imported_h177_rule_object_and_exact_10k_holm_stepdown_counterexample():
    assert cal.pooled_b_select is cal.h177_analysis.pooled_b_select
    assert cal.H177_DECISION_RULE is cal.h177_analysis.H177_DECISION_RULE
    assert cal.resolve_h177_verdict is cal.h177_analysis.resolve_h177_verdict
    assert cal.StageOrderGuard is cal.h177_analysis.StageOrderGuard
    assert cal.verify_pooled_b_coverage is cal.h177_analysis.verify_pooled_b_coverage
    assert cal.expected_mde80_205fam is cal.h177_analysis.expected_mde80_205fam
    assert (
        cal.H177_ANALYSIS_SHA256
        == "707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4"
    )

    h177_runner_src = (HERE.parent / "h177_pooled_selector" / "runner.py").read_text()
    assert "import analysis as stat" in h177_runner_src
    assert "stat.pooled_b_select(" in h177_runner_src
    assert "stat.StageOrderGuard(" in h177_runner_src
    assert "stat.resolve_h177_verdict(" in h177_runner_src

    # Red-team exact 10k-draw counterexample (PR #1502 review 5727951409):
    # CTRL constant +0.01, ACC4 four of 205 families +0.06, ACC16/OC15 null (0.0).
    # Exact H177 returns ACC4 ADMIT (p_one_sided=0.01609839016, Holm step-down 3*p=0.04829517048 < 0.05),
    # whereas a naive Bonferroni 4*p=0.06439356 would wrongly reject ACC4.
    rows_205 = [dict(context_sha256=f"f{i:03d}", family_id=i) for i in range(205)]
    r882_zeros = np.zeros((3, 205), dtype=np.float64)
    ctrl_panel = np.full((3, 205), 0.01, dtype=np.float64)
    acc4_panel = np.zeros((3, 205), dtype=np.float64)
    acc4_panel[:, :4] = 0.06
    null_panel = np.zeros((3, 205), dtype=np.float64)
    pack_4 = {
        "CTRL": ctrl_panel,
        "ACC4": acc4_panel,
        "ACC16": null_panel,
        "OC15": null_panel,
    }
    direct_res = cal.pooled_b_select(
        rows_205, r882_zeros, pack_4, seed=cal.h177_analysis.BOOT_SEED
    )
    assert direct_res["CTRL"]["admitted"] is True
    assert direct_res["ACC4"]["admitted"] is True
    assert math.isclose(
        direct_res["ACC4"]["p_one_sided"], 0.0160983901609839, rel_tol=1e-12
    )
    assert math.isclose(
        direct_res["ACC4"]["holm_p"], 0.048295170482951705, rel_tol=1e-12
    )
    assert direct_res["ACC16"]["admitted"] is False
    assert direct_res["OC15"]["admitted"] is False

    # Pass the exact same 4-arm cohort through cal.evaluate_h177_pooled_rule at 10k draws
    bank_4 = np.stack([ctrl_panel, acc4_panel, null_panel, null_panel], axis=0)
    eval_res = cal.evaluate_h177_pooled_rule(
        bank_4,
        seed=cal.h177_analysis.BOOT_SEED,
        direction="oracle",
        fraction=0.02,
        inner_draws=1000,
        final_draws=10000,
    )
    assert eval_res["admitted_joint4"].tolist() == [True, True, False, False]
    assert math.isclose(
        float(eval_res["holm_p_joint4"][1]),
        direct_res["ACC4"]["holm_p"],
        rel_tol=1e-12,
    )
    # In the cloned-4-arm stress call ({arm: acc4_panel for arm in ARMS}), all 4 arms tie at p=0.01609839,
    # so pooled_b_select gives holm_p = 4 * p = 0.06439356 >= 0.05 and rejects ACC4.
    assert eval_res["admitted_cloned4"].tolist() == [True, False, False, False]
    assert math.isclose(
        float(eval_res["holm_p_cloned4"][1]), 0.0643935606439356, rel_tol=1e-12
    )

    # Verify bit-for-bit output identity between 1,219-context call and 205-family call
    rng = np.random.default_rng(178001)
    rows = []
    for f_idx in range(205):
        n_ctx = 6 if f_idx < 194 else 5  # 194*6 + 11*5 = 1219 contexts
        for c_idx in range(n_ctx):
            rows.append(
                dict(
                    context_sha256=f"c_{f_idx:03d}_{c_idx}",
                    family_id=f"f_{f_idx:03d}",
                )
            )
    assert len(rows) == 1219

    fam_rows = [dict(context_sha256=f"f_{i:03d}", family_id=i) for i in range(205)]
    r882_ctx = np.zeros((3, 1219), dtype=np.float64)
    r882_fam = np.zeros((3, 205), dtype=np.float64)
    ctx_panels = {}
    fam_panels = {}
    fam_labels = [r["family_id"] for r in rows]
    for arm in cal.ARMS:
        if arm == "CTRL":
            ctx_mat = np.zeros((3, 1219), dtype=np.float64)
        else:
            ctx_mat = rng.normal(0.005, 0.02, size=(3, 1219))
        ctx_panels[arm] = ctx_mat
        _, fam_mat = cal.h175_protocol.family_mean(ctx_mat, fam_labels)
        fam_panels[arm] = fam_mat

    prev_draws = cal.h177_analysis.DRAWS
    try:
        cal.h177_analysis.DRAWS = 256
        res_ctx = cal.pooled_b_select(rows, r882_ctx, ctx_panels, seed=178999)
        res_fam = cal.pooled_b_select(fam_rows, r882_fam, fam_panels, seed=178999)
    finally:
        cal.h177_analysis.DRAWS = prev_draws
    assert res_ctx == res_fam


def test_injections_never_read_A_outside_its_fold_and_stage_lock_barrier(tmp_path):
    rows = toy_cohort(205, contexts_per_family=2)
    payload = {
        r["context_sha256"]: (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([-2.0, 2.0, 0.5, -1.0]),
        )
        for r in rows
    }
    folds = cal.folds(rows)
    b_fams = set(folds[1])
    a_fams = set(folds[0])
    b_rows = [r for r in rows if r["family_id"] in b_fams]
    a_rows = [r for r in rows if r["family_id"] in a_fams]
    b_scope = {r["context_sha256"] for r in b_rows}
    view = cal.LocalOnlyPayload(payload, b_scope, role_name="B")

    # Reading any A_outer context through LocalOnlyPayload must fail immediately
    for r in a_rows:
        with pytest.raises(KeyError, match="outside B role scope read"):
            _ = view[r["context_sha256"]]

    before = cal.h175_protocol.inject(
        b_rows, view, b_scope, 175101, 0.2, "context", "oracle", "full"
    )
    # Poison A_outer utilities in underlying payload
    for r in a_rows:
        payload[r["context_sha256"]] = (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([999.0, -999.0, 500.0, -500.0]),
        )
    after = cal.h175_protocol.inject(
        b_rows, view, b_scope, 175101, 0.2, "context", "oracle", "full"
    )
    assert np.array_equal(before["delta"], after["delta"])

    # StageOrderGuard barrier check
    guard = cal.StageOrderGuard()
    with pytest.raises(RuntimeError, match="A-family confirmation forbidden before"):
        guard.require_b_locked()

    guard.mark_b_read()
    lock_file = tmp_path / "lock-pooled-b-toy.json"
    cal.save(
        lock_file,
        dict(
            wave="H177",
            pooled_b_families=205,
            decisions={arm: {"admitted": False} for arm in cal.ARMS},
        ),
    )
    guard.mark_b_locked(lock_file)
    verified = guard.require_b_locked()
    assert verified["pooled_b_families"] == 205

    # Tampering with the Stage-1 file on disk must be caught before Stage-2
    lock_file.write_text('{"wave": "H177", "pooled_b_families": 41}\n')
    with pytest.raises(RuntimeError, match="corrupted pooled_b_selection_lock.json"):
        guard.require_b_locked()


def test_clopper_pearson_exact_bounds_calibration_verdict_and_h177_receipt_gate():
    gold_cp = [
        (0, 200, 0.014867039231272056),
        (1, 200, 0.023498470434950704),
        (10, 200, 0.08333515106637196),
        (50, 200, 0.305467356649194),
        (199, 200, 0.9997435664127765),
        (200, 200, 1.0),
    ]
    for k, n, expected_upper in gold_cp:
        s = cal.clopper_pearson_summary(k, n)
        assert math.isclose(s["cp95_one_sided_upper"], expected_upper, rel_tol=1e-10)

    b0_200 = cal.clopper_pearson_summary(0, 200)
    b0_50 = cal.clopper_pearson_summary(0, 50)
    assert np.isclose(b0_200["cp95_two_sided"][1], 1.0 - 0.025 ** (1.0 / 200.0))
    assert np.isclose(b0_50["cp95_one_sided_upper"], 1.0 - 0.05 ** (1.0 / 50.0))

    prim_zero = {
        "power": {"median": 0.0, "per_seed": [0.0, 0.0, 0.0]},
        "clopper_pearson_cohort_any_arm_50": b0_50,
        "clopper_pearson_cohort_treatment3_50": b0_50,
        "clopper_pearson_marginal_arm_200": b0_200,
    }
    cl4_zero = {
        "power": {"median": 0.0, "per_seed": [0.0, 0.0, 0.0]},
        "clopper_pearson_200": b0_200,
    }
    sng_zero = {
        "power": {"median": 0.0, "per_seed": [0.0, 0.0, 0.0]},
        "clopper_pearson_max_seed_200": b0_200,
    }
    curves_pass = [
        dict(
            slate="24",
            variant="family",
            direction="oracle",
            fraction=0.0,
            rules={
                "H177_POOLED": prim_zero,
                "H177_POOLED_CLONED4_DIAGNOSTIC": cl4_zero,
                "H177_POOLED_SINGLE_SEED_DIAGNOSTIC": sng_zero,
            },
        ),
        dict(
            slate="24",
            variant="family",
            direction="random_nonoracle",
            fraction=0.05,
            rules={
                "H177_POOLED": prim_zero,
                "H177_POOLED_CLONED4_DIAGNOSTIC": cl4_zero,
                "H177_POOLED_SINGLE_SEED_DIAGNOSTIC": sng_zero,
            },
        ),
    ]
    head_ok = {
        "24": {
            "MDE80": {
                "family": {"H177_POOLED": {"status": "GRID_DETECTED", "gain": 0.016364}}
            }
        },
        "full": {
            "MDE80": {
                "family": {"H177_POOLED": {"status": "GRID_DETECTED", "gain": 0.024480}}
            }
        },
    }
    v_pass = cal.decide_verdict(head_ok, curves_pass)
    assert v_pass["calibration_verdict"] == "CALIBRATED_PASS"
    assert v_pass["power_verdict"] == "POWER_OK"
    assert v_pass["verdict"] == "CALIBRATED_PASS_POWER_OK"
    assert v_pass["null_and_harmful_admission_le_005"] is True
    assert v_pass["replayed_four_arm_holm_m"] == 4
    assert v_pass["h177_admit_counts_as_lift"] is True
    h177_check_pass = cal.resolve_h177_verdict(
        True,
        h178_receipt=v_pass["h178_receipt"],
        expected_analysis_sha256=cal.H177_ANALYSIS_SHA256,
    )
    assert (
        h177_check_pass["verdict"] == "ADVANCE"
        and h177_check_pass["lift_qualified"] is True
    )

    # Inject a harmful cell exceeding 0.05 admission rate
    curves_fail = json.loads(json.dumps(curves_pass))
    curves_fail[1]["rules"]["H177_POOLED"]["power"]["median"] = 0.055
    v_fail = cal.decide_verdict(head_ok, curves_fail)
    assert v_fail["calibration_verdict"] == "CALIBRATED_FAIL"
    assert v_fail["null_and_harmful_admission_le_005"] is False
    assert v_fail["h177_admit_counts_as_lift"] is False
    h177_check_fail = cal.resolve_h177_verdict(
        True,
        h178_receipt=v_fail["h178_receipt"],
        expected_analysis_sha256=cal.H177_ANALYSIS_SHA256,
    )
    assert (
        h177_check_fail["verdict"] == "DIAGNOSTIC"
        and h177_check_fail["lift_qualified"] is False
    )

    # Inject family MDE80 > 0.03
    head_low = json.loads(json.dumps(head_ok))
    head_low["full"]["MDE80"]["family"]["H177_POOLED"]["gain"] = 0.035
    v_low = cal.decide_verdict(head_low, curves_pass)
    assert v_low["calibration_verdict"] == "CALIBRATED_PASS"
    assert v_low["power_verdict"] == "POWER_LOW"


def test_no_optimizer_or_backward_call_in_owned_execution():
    for name in ["calibration.py", "runner.py", "monitor.py"]:
        tree = ast.parse((HERE / name).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                assert n.func.attr not in [
                    "backward",
                    "step",
                    "AdamW",
                    "Adam",
                    "SGD",
                    "LBFGS",
                ]
    text = (HERE / "runner.py").read_text()
    assert (
        "torch.optim.Optimizer.__init__=forbidden" in text.replace(" ", "")
        and "param.requires_grad_(False)" in text
    )


def test_native_baseline_gate_and_resource_violation_guards():
    values = np.zeros(91, dtype=np.float64)
    values[-1] = 91 * 1.1377941898164525
    assert float(values.mean()) == 1.1377941898164525
    runner.baseline_gate(values, values.copy())
    bad = values.copy()
    bad.view(np.uint64)[0] = 1
    with pytest.raises(RuntimeError):
        runner.baseline_gate(bad, values)

    good_sample = dict(
        load=10.0,
        cpu=20.0,
        available=32 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.92,
        rss=1_000_000_000,
        threads=1,
    )
    assert monitor.violations(good_sample) == []

    bad_disk = dict(good_sample, disk_used_fraction=0.938)
    assert "disk user-visible used>=93.8%" in monitor.violations(bad_disk)

    bad_rss = dict(good_sample, rss=6_000_000_000)
    assert "aggregate RSS>=6GB" in monitor.violations(bad_rss)


def test_full_pipeline_calibration_on_toy_cohort(monkeypatch, tmp_path):
    monkeypatch.setattr(cal.h175_protocol, "SIMULATIONS", 4)
    monkeypatch.setattr(cal.h175_protocol, "SELECTOR_DRAWS", 16)
    monkeypatch.setattr(cal.h175_protocol, "REPORT_DRAWS", 64)
    monkeypatch.setattr(cal, "SIMULATIONS", 4)
    monkeypatch.setattr(cal, "INNER_DRAWS", 16)
    monkeypatch.setattr(cal, "FINAL_DRAWS", 32)
    monkeypatch.setattr(cal, "REPORT_DRAWS", 64)
    monkeypatch.setattr(cal, "FRACTIONS", (0.0, 0.2))

    rows = toy_cohort(205, contexts_per_family=1)
    val = [dict(context_sha256="v0", family_id="vf0")]
    toy_util = np.concatenate(([1.5, 2.0], np.linspace(-2.0, 0.5, 22)))
    payload = {
        r["context_sha256"]: (np.arange(24, dtype=float), toy_util.copy())
        for r in rows + val
    }
    head, curves, verdict = cal.run_calibration(
        rows, val, payload, tmp_path, lambda: None, inner_draws=16, final_draws=32
    )
    assert len(curves) == 16
    assert len(list(tmp_path.glob("lock-pooled-b-*.json"))) == 16
    assert (tmp_path / "panels.npz").exists()
    assert (tmp_path / "simulation_metrics.json").exists()
    for cell in curves:
        if cell["fraction"] == 0.0:
            assert cell["rules"]["H177_POOLED"]["admitted_decisions"] == 0
            assert cell["rules"]["H177_POOLED"]["selected_elo"]["median"] == 1500.0
    assert verdict["calibration_verdict"] == "CALIBRATED_PASS"
    assert head["24"]["MDE80"]["family"]["H177_POOLED"]["status"] in (
        "GRID_DETECTED",
        "RIGHT_CENSORED",
    )
