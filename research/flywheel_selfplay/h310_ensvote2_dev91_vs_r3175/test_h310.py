"""Data-free H310 controls: the lock-phase dev91 guard (counter must stay 0 for TRAIN paths and refuse dev91 paths),
the decision rule (TRANSFERRED / UNDERPOWERED / NOT_TRANSFERRED, ties never win), the Elo label, the power note."""

import subprocess
import sys
import textwrap

import numpy as np

import evaluate_h310 as ev


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, cwd=str(ev.HERE)
    )


def test_dev_guard_refuses_dev_paths_and_counts_zero_for_train(tmp_path):
    (tmp_path / "train_cid.npz").write_bytes(b"x")
    (tmp_path / "devcid123.npz").write_bytes(b"x")
    r = _run(f"""
        import run_h310 as R
        R.install_dev_guard(["DEVCID123"])
        open(r"{tmp_path / "train_cid.npz"}").close()
        assert R.DEV_GUARD["denied"] == 0
        for p in (r"{tmp_path / "devcid123.npz"}", "/x/r882-secondary-costs.npz", "/x/r3175-gpu-costs.npz", "/x/round-3175.json"):
            try:
                open(p).close()
                raise SystemExit("not refused: " + p)
            except RuntimeError:
                pass
        assert R.DEV_GUARD["denied"] == 4
        print("ok")
    """)
    assert r.stdout.strip() == "ok", r.stderr


def _judge(gains, lcb, pooled_lcb, pooled_ci, d5, harm=0.1, exc=0):
    return dict(
        summary=dict(median_gain_across_seeds=float(np.median(gains)), pooled_lcb_one_sided_95=pooled_lcb,
                     per_seed_lcb_one_sided={"0.05": lcb}, pooled_ci95=pooled_ci, harmed_share_by_seed=[harm] * 3),
        concentration=[dict(gain_drop_top5=x) for x in d5],
        invalid_detail=[dict(excess=exc)] * 3,
    )  # fmt: skip


def test_decision_rule():
    ok = _judge([0.02, 0.03, 0.025], [0.005, -0.001, 0.004], 0.003, [0.004, 0.05], [0.01, 0.0, 0.02])
    assert ev.decide(ok)[1] == "DEV91_TRANSFERRED"
    assert ev.decide(dict(ok, invalid_detail=[dict(excess=2)] * 3))[1] == "DEV91_NOT_TRANSFERRED"  # c6 fails; CI half-width .023
    wide = _judge([0.02, 0.03, 0.025], [-0.01, -0.02, -0.01], -0.01, [-0.02, 0.07], [0.0, 0.0, 0.0])
    assert ev.decide(wide)[1] == "DEV91_UNDERPOWERED"
    narrow = _judge([0.002, 0.003, 0.0025], [-0.01] * 3, -0.01, [-0.01, 0.02], [0.0] * 3)
    assert ev.decide(narrow)[1] == "DEV91_NOT_TRANSFERRED"
    tie = _judge([0.001, 0.001, 0.001], [0.0005] * 3, 0.0005, [0.0005, 0.002], [0.001] * 3)
    assert ev.decide(tie)[1] == "DEV91_NOT_TRANSFERRED"  # median exactly +0.001 is not > +0.001; CI narrow
    assert ev.ELO_LABEL == "reference = r3175 slot0 (dev91)"


def test_power_note_and_invalid():
    u = np.zeros((3, 6, 2))
    u[:, :, 1] = [[0.1, 0.0, 0.2, 0.0, 0.1, 0.0]] * 3
    pn = ev.power_note(u, ["a", "a", "b", "b", "c", "c"])
    assert pn["families"] == 3 and pn["contexts"] == 6 and pn["mde_one_sided_alpha05_power80"] > 0
    pk = dict(gate_k=np.array([True, False, True]), utility_k=np.array([0.5, 0.4, -2.0]))
    assert not ev.invalid(pk, 0) and ev.invalid(pk, 1) and ev.invalid(pk, 2)
