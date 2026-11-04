"""Unit tests for H298 (h298_family_balanced_gradaccum): family-balanced sampling, grad-accum prefix, and strict gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis_h298 as a298

HERE = Path(__file__).resolve().parent


def test_sample_fambal_cids_prefix_and_uniformity() -> None:
    fam_ids = [f"fam-{i:03d}" for i in range(205)]
    fam_cids = [[f"fam-{i:03d}-c{j}" for j in range((i % 7) + 1)] for i in range(205)]
    cids4 = a298.sample_fambal_cids(fam_ids, fam_cids, 29801, 32768)
    cids1 = cids4[:8192]
    assert len(cids1) == 8192
    assert len(cids4) == 32768
    # Determinism check
    assert a298.sample_fambal_cids(fam_ids, fam_cids, 29801, 32768) == cids4
    # Check all 205 families are represented in 8192 draws
    sampled_fams = {c.rsplit("-c", 1)[0] for c in cids1}
    assert len(sampled_fams) == 205


def test_top4_positive_gain_concentration_gate() -> None:
    base = np.zeros(91, dtype=np.float64)
    # Concentrated gain: only 4 contexts carry 100% of positive gain -> fails < 80% gate
    cand_conc = np.zeros(91, dtype=np.float64)
    cand_conc[:4] = 0.05
    res_conc = a298.top4_positive_gain_concentration(cand_conc, base)
    assert res_conc["n_positive_contexts"] == 4
    assert pytest.approx(res_conc["top4_positive_share"]) == 1.0
    assert res_conc["passes_lt_80pct"] is False

    # Broad gain: 10 contexts each carry +0.02 -> top-4 share is 4/10 = 40% < 80% -> passes
    cand_broad = np.zeros(91, dtype=np.float64)
    cand_broad[:10] = 0.02
    res_broad = a298.top4_positive_gain_concentration(cand_broad, base)
    assert res_broad["n_positive_contexts"] == 10
    assert pytest.approx(res_broad["top4_positive_share"]) == 0.40
    assert res_broad["passes_lt_80pct"] is True


def test_strict_tie_never_wins_and_3_of_3_gate() -> None:
    dev_fams = [f"dev-fam-{i % 19:02d}" for i in range(91)]
    r3175_u = np.full(91, a298.R3175_DEV91_MEAN, dtype=np.float64)
    r3175_res = {
        "mean": a298.R3175_DEV91_MEAN,
        "per_context": r3175_u.tolist(),
        "selected_programs": [f"p0-{i}" for i in range(91)],
    }
    # Tie vs r3175 + 0.001 or tie vs paired control must NOT pass
    ctrl_u = r3175_u + 0.002
    ctrl_res = {
        "seed": 29801,
        "arm": "PAIRED_CONTROL",
        "mean": float(ctrl_u.mean()),
        "per_context": ctrl_u.tolist(),
        "selected_programs": [f"pc-{i}" for i in range(91)],
    }
    tie_cand = {
        "seed": 29801,
        "arm": "FAMBAL_ACC4",
        "mean": float(ctrl_u.mean()),
        "per_context": ctrl_u.tolist(),
        "selected_programs": [f"pc-{i}" for i in range(91)],
    }
    ev_tie = a298.evaluate_seed_comparison(tie_cand, ctrl_res, r3175_res, dev_fams, n_boot=500)
    assert ev_tie["gates"]["mean_gt_paired_control"] is False
    assert ev_tie["gates"]["seed_pass"] is False

    # Genuine broad improvement across all 19 families beating both r3175 + 0.001 and paired control
    win_u = r3175_u + 0.005
    win_cand = {
        "seed": 29801,
        "arm": "FAMBAL_ACC4",
        "mean": float(win_u.mean()),
        "per_context": win_u.tolist(),
        "selected_programs": [f"pw-{i}" for i in range(91)],
    }
    ev_win = a298.evaluate_seed_comparison(win_cand, ctrl_res, r3175_res, dev_fams, n_boot=500)
    assert ev_win["gates"]["seed_pass"] is True
    assert ev_win["elo_vs_r3175"]["point_elo"] > 1500.0

    # 2/3 seeds passing -> NOT_ADVANCED; 3/3 seeds passing -> DEV_ADVANCE
    s2 = dict(ev_win, seed=29802)
    s3_fail = dict(ev_tie, seed=29803)
    sum_2_of_3 = a298.summarize_arm_across_seeds([ev_win, s2, s3_fail])
    assert sum_2_of_3["all_3_of_3_pass"] is False

    s3_win = dict(ev_win, seed=29803)
    sum_3_of_3 = a298.summarize_arm_across_seeds([ev_win, s2, s3_win])
    assert sum_3_of_3["all_3_of_3_pass"] is True


def test_regression_receipt_and_prereg_batches() -> None:
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists()
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["round_3175_sha256"] == "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
    assert reg["evaluation_identity"] == "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
    assert reg["dope_sha256"] == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
    assert reg["r3175_replay"]["mean"] == a298.R3175_DEV91_MEAN
    assert reg["r3175_replay"]["selected_programs_match_91_of_91"] is True
    assert reg["t1845_control_184501_replay"]["round_10_mean"] == 1.1354387884
    assert reg["t1845_control_184501_replay"]["selected_programs_match_91_of_91"] is True

    batches_path = HERE / "preregistered_batches.json"
    assert batches_path.exists()
    digest = hashlib.sha256(batches_path.read_bytes()).hexdigest()
    assert digest == "51e93dff787ee5d38f12fc41c2fb4b8021929ed59926a708922c5340bd359f95"
    batches = json.loads(batches_path.read_text())
    for seed in ("29801", "29802", "29803"):
        assert len(batches[seed]["PAIRED_CONTROL"]) == 8
        assert len(batches[seed]["FAMBAL_ACC1"]) == 8
        assert len(batches[seed]["FAMBAL_ACC4"]) == 8
        for step in range(8):
            assert 8175 <= batches[seed]["PAIRED_CONTROL"][step]["n_presentations"] <= 8192
            assert batches[seed]["FAMBAL_ACC1"][step]["n_presentations"] == 8192
            assert batches[seed]["FAMBAL_ACC4"][step]["n_presentations"] == 32768
