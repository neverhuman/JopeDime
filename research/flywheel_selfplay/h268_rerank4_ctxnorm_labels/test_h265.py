#!/usr/bin/env python3
"""Synthetic unit tests and artifact integrity checks for H265 (h265_top4_listwise_reranker)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from h265_frozen import (
    BOOT_SEED,
    CTX27_CID,
    CTX27_INDEX,
    FOLD_SEED,
    HOLM_M,
    LR_GRID,
    N_EST_GRID,
    SEEDS,
    build_top4_feature_matrix_for_context,
    compute_true_top4_relevance,
    fit_lgbm_ranker_200,
    paired_family_contrast,
    predict_lgbm_ranker,
    select_from_top4_scores,
    shuffle_labels_within_family,
)

STUDY_DIR = Path(
    "/home/ubuntu/jope-rescue-20260921/h268-astra/reference/h265_regression"
)


def test_manifests_and_folds_outcome_free() -> None:
    for fname in [
        "PREREG.md",
        "README.md",
        "fold_manifest.json",
        "manifest.json",
        "metrics.json",
        "monitor.py",
        "runner.py",
        "test_h265.py",
    ]:
        assert (STUDY_DIR / fname).exists(), f"Missing {fname}"

    manifest = json.loads((STUDY_DIR / "manifest.json").read_text())
    fold_manifest = json.loads((STUDY_DIR / "fold_manifest.json").read_text())
    assert manifest["wave_id"] == "h265-top4-listwise-reranker-20261103"
    assert manifest["fold_seed"] == FOLD_SEED == 265140
    assert manifest["seeds"] == list(SEEDS) == [265101, 265102, 265103]
    assert manifest["boot_seed"] == BOOT_SEED == 265150
    assert manifest["holm_m"] == HOLM_M == 2
    assert manifest["primary_contrast"] == "RERANK4 - CONTROL"
    assert manifest["secondary_contrast"] == "RERANK4_SHUF - CONTROL"
    assert list(LR_GRID) == [0.03, 0.05, 0.10]
    assert list(N_EST_GRID) == [50, 100, 200]

    assert fold_manifest["dl_train_contexts"] == 1219
    assert fold_manifest["dl_train_families"] == 205
    assert fold_manifest["dl_val_contexts"] == 91
    assert fold_manifest["dl_val_families"] == 19
    assert fold_manifest["n_cells"] == 15
    assert fold_manifest["train_val_family_overlap"] == 0
    assert fold_manifest["has_meta_test_any"] is False
    assert fold_manifest["outcomes_read"] == 0
    assert fold_manifest["dl_val_context_27"]["index"] == CTX27_INDEX == 27
    assert fold_manifest["dl_val_context_27"]["context_sha256"] == CTX27_CID
    folds = [set(f["families"]) for f in fold_manifest["folds"]]
    for i in range(5):
        assert len(folds[i]) == 41
        for j in range(i + 1, 5):
            assert len(folds[i] & folds[j]) == 0


def test_compute_true_top4_relevance_and_invalid_gate() -> None:
    # Candidate 0 fails gate (invalid = -2.0 utility -> relevance 0);
    # Candidates 1, 2, 3 pass gate with fitness [-1.2, -1.5, -1.1] -> order [2 (-1.5), 1 (-1.2), 3 (-1.1)]
    fit4 = np.array([-1.9, -1.2, -1.5, -1.1], dtype=np.float64)
    gate4 = np.array([False, True, True, True], dtype=np.bool_)
    rel = compute_true_top4_relevance(fit4, gate4)
    assert list(rel) == [0, 2, 3, 1]


def test_build_top4_features_and_margin_selector() -> None:
    rng = np.random.default_rng(265001)
    rcost = np.linspace(0.0, 2.0, 32, dtype=np.float32)
    aprior = rng.normal(size=32).astype(np.float32)
    prog_top4 = rng.normal(size=(4, 412)).astype(np.float32)
    sem24 = rng.normal(size=(32, 24)).astype(np.float32)
    proj16 = rng.normal(size=(32, 16)).astype(np.float32)
    lm48_z = rng.normal(size=48).astype(np.float32)
    dope629 = rng.normal(size=629).astype(np.float32)

    top4, x_top4 = build_top4_feature_matrix_for_context(
        rcost, aprior, prog_top4, sem24, proj16, lm48_z, dope629
    )
    assert list(top4) == [0, 1, 2, 3]
    assert x_top4.shape == (4, 560)
    assert np.all(np.isfinite(x_top4))
    # Slot 0 has H217 zgap == 0.0; slots 1..3 have H217 zgap > 0.0
    assert x_top4[0, 7] == 0.0
    assert x_top4[1, 7] > 0.0

    scores = np.array([0.10, 0.35, 0.20, -0.10], dtype=np.float64)
    assert select_from_top4_scores(scores, margin=0.10) == 1
    assert select_from_top4_scores(scores, margin=0.30) == 0
    assert select_from_top4_scores(scores, margin=math.inf) == 0


def test_shuffle_within_family_and_lgbm_ranker() -> None:
    rel_by_cid = {
        "c1": np.array([3, 2, 1, 0], dtype=np.int32),
        "c2": np.array([0, 3, 2, 1], dtype=np.int32),
    }
    fam_by_cid = {"c1": "famA", "c2": "famA"}
    shuf = shuffle_labels_within_family(
        rel_by_cid, ["c1", "c2"], fam_by_cid, seed=265101
    )
    orig_sorted = sorted(np.concatenate([rel_by_cid["c1"], rel_by_cid["c2"]]).tolist())
    shuf_sorted = sorted(np.concatenate([shuf["c1"], shuf["c2"]]).tolist())
    assert orig_sorted == shuf_sorted

    rng = np.random.default_rng(265002)
    X = rng.normal(size=(24, 560)).astype(np.float32)
    y = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 6)
    model = fit_lgbm_ranker_200(X, y, n_groups=6, lr=0.05, random_state=265002)
    p50 = predict_lgbm_ranker(model, X, num_iteration=50)
    p200 = predict_lgbm_ranker(model, X, num_iteration=200)
    assert p50.shape == (24,) and p200.shape == (24,)

    rows = [{"family_id": f"fam_{i}"} for i in range(4)]
    panel = np.zeros((3, 4, 2), dtype=np.float64)
    panel[:, :, 0] = 1.15
    panel[:, :, 1] = 1.18
    boot_ix = rng.integers(0, 4, size=(100, 4))
    res = paired_family_contrast(
        rows, panel, ["CONTROL", "RERANK4"], "RERANK4", "CONTROL", boot_ix
    )
    assert abs(res["median_family_gain"] - 0.03) < 1e-12
    assert res["positive_seeds"] == 3


def test_completed_metrics_invariants() -> None:
    metrics = json.loads((STUDY_DIR / "metrics.json").read_text())
    if metrics.get("status") != "COMPLETE":
        return
    assert metrics["verdict"] in ("ADVANCE", "NO_ADVANCE")
    base = metrics["baseline_regression"]
    assert base["r882_native91_abs_diff"] == 0.0
    assert base["r882_bit_exact_context_matches"] == 91
    assert base["top4_pick_oracle_ceiling_ctx_abs_diff"] <= 1e-12
    assert metrics["audits"]["audit_a_zero_leakage"]["pass"] is True
    assert metrics["audits"]["audit_b_zero_test_reveals"]["pass"] is True
    assert metrics["context_27"]["context_sha256"] == CTX27_CID
    elo_ctrl = metrics["dl_val_native91"]["mm_bt_elo_anchored_r882"]["CONTROL"][
        "median_elo"
    ]
    assert abs(elo_ctrl - 1500.0) <= 1e-9
