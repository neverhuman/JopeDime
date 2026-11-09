"""Unit tests for H304 (h304_lcb_margin_promotion_replay)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_h304 as a304  # noqa: E402


def test_split_families_by_hash_disjoint_and_deterministic() -> None:
    fams = [f"fam_{i:02d}" for i in range(19)]
    for seed in a304.SPLIT_SEEDS:
        sel1, ev1 = a304.split_families_by_hash(fams, seed)
        sel2, ev2 = a304.split_families_by_hash(fams, seed)
        assert sel1 == sel2 and ev1 == ev2
        assert len(sel1) + len(ev1) == 19
        assert set(sel1).isdisjoint(set(ev1))


def test_repeated_200_balanced_splits_unique() -> None:
    fams = [f"fam_{i:02d}" for i in range(19)]
    splits = a304.generate_repeated_balanced_splits(fams, n_splits=200, rng_seed=a304.BOOT_SEED)
    assert len(splits) == 200
    seen = set()
    for sel, ev in splits:
        assert len(sel) == 9 and len(ev) == 10
        assert set(sel).isdisjoint(set(ev))
        seen.add(tuple(sel))
    assert len(seen) == 200


def test_tie_guard_blocks_micro_noise_promotions() -> None:
    fams = ["fam_a", "fam_b", "fam_c", "fam_d"]
    _, ctx_to_fam, fam_w = a304.build_family_index_map(fams)
    rounds = np.array([100, 101, 102], dtype=np.int64)
    # Candidate 101 is +5e-5 (within tie guard 1e-4); Candidate 102 is +0.005 (above all margins)
    utils = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.00005, 1.00005, 0.9, 0.9],
            [1.005, 1.005, 1.005, 1.005],
        ],
        dtype=np.float64,
    )
    sel_f = np.array([0, 1], dtype=np.int32)
    ev_f = np.array([2, 3], dtype=np.int32)
    sel_c = np.isin(ctx_to_fam, sel_f)
    ev_c = np.isin(ctx_to_fam, ev_f)
    sel_boot = a304.make_select_bootstrap_weights(2, 100, 42)

    for rname in a304.RULE_NAMES:
        res = a304.replay_single_rule(
            rname, rounds, utils, fam_w, sel_f, ev_f, sel_c, ev_c, sel_boot
        )
        # Round 101 must never be accepted; Round 102 must be accepted
        assert 101 not in res["accepted_rounds"]
        assert 102 in res["accepted_rounds"]


def test_regression_receipt_and_metrics_schema() -> None:
    reg = json.loads((HERE / "regression_receipt.json").read_text())
    assert reg["status"] == "PASS"
    assert reg["n_post_r3175_rounds"] == 1499
    assert abs(reg["r3175_mean"] - 1.1389572403748425) <= 1e-15
    assert abs(reg["post_r3175_mean"] - 1.1320995298853955) <= 1e-15
    assert abs(reg["post_r3175_q95"] - 1.1354387884248622) <= 1e-15

    m_path = HERE / "metrics.json"
    if m_path.exists():
        assert m_path.stat().st_size < 50 * 1024
        m = json.loads(m_path.read_text())
        assert m["wave_id"] == "H304"
        assert m["verdict"] in ("RULE_ADVANCES", "RULE_NOT_ADVANCED")
        assert m["dope_untouched"] is True
        assert m["optimizer_updates"] == 0
        assert m["gpu_used"] is False
