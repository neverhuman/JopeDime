#!/usr/bin/env python3
"""Unit and regression tests for H275 (`h275_rerank8_soft_risk`)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from runner import (  # noqa: E402
    BOOT_SEED,
    FEATURE_DIM_K8,
    FOLD_SEEDS,
    HOLM_M,
    SEEDS,
    SLICE_K,
    SOFT_LAMBDA_GRID,
    apply_holm_bonferroni,
    apply_soft_risk_penalty,
    build_topk_feature_matrix_for_context,
    compute_sem24_numpy,
    compute_true_topk_relevance,
    paired_family_bootstrap_contrast,
    select_from_topk_scores_with_soft_risk,
    standardize_scores_per_query,
)


def test_cleanroom_and_h275_constants() -> None:
    runner_src = (THIS_DIR / "runner.py").read_text(encoding="utf-8")
    for forbidden in ("import topk", "from topk", "import h265_core", "from h265_core"):
        assert forbidden not in runner_src
    assert SLICE_K == 8
    assert FEATURE_DIM_K8 == 568
    assert SOFT_LAMBDA_GRID == (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert FOLD_SEEDS == (275140, 275141, 275142)
    assert SEEDS == (275101, 275102, 275103)
    assert BOOT_SEED == 275150
    assert HOLM_M == 3


def test_feature_builder_568_cols_and_outcome_blindness() -> None:
    rng = np.random.default_rng(27501)
    n_pool = 32
    rcost = rng.normal(loc=0.0, scale=1.0, size=n_pool).astype(np.float32)
    aprior = rng.normal(loc=0.0, scale=1.0, size=n_pool).astype(np.float32)
    prog = rng.normal(loc=0.0, scale=1.0, size=(n_pool, 412)).astype(np.float32)
    sem24 = compute_sem24_numpy(prog)
    proj16 = rng.normal(loc=0.0, scale=1.0, size=(n_pool, 16)).astype(np.float32)
    lm48 = rng.normal(loc=0.0, scale=1.0, size=48).astype(np.float32)
    dope629 = rng.normal(loc=0.0, scale=1.0, size=629).astype(np.float32)

    top8_a, X_a = build_topk_feature_matrix_for_context(
        rcost, aprior, prog, sem24, proj16, lm48, dope629, k=8
    )
    assert top8_a.shape == (8,)
    assert X_a.shape == (8, 568)
    assert np.all(np.isfinite(X_a))

    # Relevance labels depend on outcomes, but X_a is strictly outcome-free
    rel_1 = compute_true_topk_relevance(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        np.array([True, False, True, True, False, True, True, True]),
    )
    assert rel_1[1] == 0 and rel_1[4] == 0 and rel_1[0] == 7


def test_soft_risk_standardization_and_slot0_untouched() -> None:
    scores = np.array([1.0, 2.5, 0.5, -1.0, 0.0, 1.2, -0.5, 0.3], dtype=np.float64)
    z = standardize_scores_per_query(scores)
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-6

    # Even if p_inv[0] is 0.99, slot 0 is untouched under apply_soft_risk_penalty
    p_inv = np.array([0.99, 0.40, 0.01, 0.05, 0.10, 0.80, 0.20, 0.02], dtype=np.float64)
    s_soft = apply_soft_risk_penalty(scores, p_inv, lam=4.0)
    assert s_soft[0] == z[0]
    for k in range(1, 8):
        assert abs(s_soft[k] - (z[k] - 4.0 * p_inv[k])) < 1e-12

    # With lam = 0.0 and margin = 0.1, slot 1 (raw score 2.5) wins
    pick_lam0 = select_from_topk_scores_with_soft_risk(scores, p_inv, lam=0.0, margin=0.1)
    assert pick_lam0 == 1

    # With lam = 8.0, slot 1 (p_inv=0.40 -> penalty 3.2 z-units) drops below slot 0, so abstains to 0
    pick_lam8 = select_from_topk_scores_with_soft_risk(scores, p_inv, lam=8.0, margin=0.1)
    assert pick_lam8 == 0

    # Infinite margin always returns 0
    assert select_from_topk_scores_with_soft_risk(scores, p_inv, lam=0.0, margin=math.inf) == 0


def test_holm_m3_and_bootstrap_ci9833() -> None:
    adj = apply_holm_bonferroni({"PRIMARY": 0.001, "SECONDARY": 0.004, "SOFT_VETO": 0.020})
    assert abs(adj["PRIMARY"] - 0.003) < 1e-12
    assert abs(adj["SECONDARY"] - 0.008) < 1e-12
    assert abs(adj["SOFT_VETO"] - 0.020) < 1e-12

    rows = [{"context_sha256": f"c{i}", "family_id": f"f{i // 2}"} for i in range(20)]
    ctrl = np.zeros((3, 3, 20), dtype=np.float64)
    arm = np.full((3, 3, 20), 0.05, dtype=np.float64)
    res = paired_family_bootstrap_contrast(rows, arm, ctrl, boot_seed=275150, n_boot=200)
    assert abs(res["median_family_gain"] - 0.05) < 1e-12
    assert res["ci9833_holm3"][0] > 0.0
    assert res["positive_repetitions_of_9"] == 9


def test_capsule_files_and_metrics_size_cap() -> None:
    for fname in (
        "PREREG.md",
        "README.md",
        "fold_manifest.s3.json",
        "manifest.json",
        "metrics.json",
        "monitor.py",
        "runner.py",
    ):
        assert (THIS_DIR / fname).exists(), f"Missing {fname}"
    m_path = THIS_DIR / "metrics.json"
    assert m_path.stat().st_size <= 50_000
    m_obj = json.loads(m_path.read_text(encoding="utf-8"))
    assert m_obj["wave"] == "H275"
