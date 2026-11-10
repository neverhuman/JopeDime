#!/usr/bin/env python3
"""Synthetic unit tests and artifact integrity checks for H261 (h261_diverse_reveal_set)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from research.flywheel_selfplay.h261_diverse_reveal_set.runner import (
    BOOT_SEED,
    FOLD_SEED,
    GATE_THRESHOLD,
    HOLM_M,
    LAMBDA_GRID,
    SEEDS,
    compute_candidate_emb40,
    compute_rev4_redundancy_stats,
    compute_top_zgap,
    extract_learner_and_dsl_heads,
    paired_family_contrast,
    pick_best_revealed_gated_cost,
    score_h261_context_deterministic,
    select_mmr4_from_top16,
    select_struct4_from_top16,
)

STUDY_DIR = Path(__file__).resolve().parent


def test_manifests_and_folds_outcome_free() -> None:
    for fname in [
        "PREREG.md",
        "README.md",
        "fold_manifest.json",
        "manifest.json",
        "metrics.json",
        "monitor.py",
        "runner.py",
        "test_h261.py",
    ]:
        assert (STUDY_DIR / fname).exists(), f"Missing {fname}"

    manifest = json.loads((STUDY_DIR / "manifest.json").read_text())
    fold_manifest = json.loads((STUDY_DIR / "fold_manifest.json").read_text())
    assert manifest["wave_id"] == "h261-diverse-reveal-set-20261103"
    assert manifest["fold_seed"] == FOLD_SEED == 261140
    assert manifest["seeds"] == list(SEEDS) == [261101, 261102, 261103]
    assert manifest["boot_seed"] == BOOT_SEED == 261150
    assert manifest["holm_m"] == HOLM_M == 2
    assert manifest["primary_contrast"] == "MMR4_CV - TOP4_PICK"
    assert manifest["secondary_contrast"] == "STRUCT4 - TOP4_PICK"
    assert GATE_THRESHOLD == 0.25
    assert list(LAMBDA_GRID) == [0.1, 0.2, 0.3, 0.5, 1.0]

    assert fold_manifest["dl_train_contexts"] == 1219
    assert fold_manifest["dl_train_families"] == 205
    assert fold_manifest["dl_val_contexts"] == 91
    assert fold_manifest["dl_val_families"] == 19
    assert fold_manifest["train_val_family_overlap"] == 0
    assert fold_manifest["has_meta_test_any"] is False
    assert fold_manifest["outcomes_read"] == 0
    folds = [set(f["families"]) for f in fold_manifest["folds"]]
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(folds[i] & folds[j]) == 0


def test_mmr4_and_struct4_selection_synthetic() -> None:
    top16 = np.arange(16, dtype=np.int64)
    rcost = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    # Ranks 0, 1, 2, 3 have nearly identical costs and identical embeddings along dim 0
    rcost[:7] = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32)
    emb = np.zeros((20, 40), dtype=np.float32)
    for r in range(4):
        emb[r, 0] = 1.0
    # Ranks 4, 5, 6 have orthogonal embeddings
    emb[4, 1] = 1.0
    emb[5, 2] = 1.0
    emb[6, 3] = 1.0
    for r in range(7, 20):
        emb[r, r % 40] = 1.0

    # With lam=0.0, MMR4 selects top-4 [0, 1, 2, 3]
    s_lam0 = select_mmr4_from_top16(top16, rcost, emb, lam=0.0)
    assert s_lam0 == [0, 1, 2, 3]

    # With lam=1.0, similarity penalty pushes MMR4 to select orthogonal candidates [0, 4, 5, 6]
    s_lam1 = select_mmr4_from_top16(top16, rcost, emb, lam=1.0)
    assert s_lam1 == [0, 4, 5, 6]
    assert len(set(s_lam1)) == 4

    sem24 = np.zeros((20, 24), dtype=np.float32)
    sem24[:, 0] = 1.0
    est_id = np.array([0, 0, 0, 0, 1, 2, 3] + [0] * 13, dtype=np.int32)
    dsl_id = np.array([10, 10, 10, 10, 20, 30, 40] + [10] * 13, dtype=np.int64)
    utility = np.linspace(1.0, 2.0, 20, dtype=np.float64)

    red_top4 = compute_rev4_redundancy_stats(
        [0, 1, 2, 3], top16, emb, sem24, est_id, dsl_id, utility
    )
    red_mmr4 = compute_rev4_redundancy_stats(
        s_lam1, top16, emb, sem24, est_id, dsl_id, utility
    )
    assert abs(red_top4["mean_cos_emb40"] - 1.0) < 1e-12
    assert abs(red_mmr4["mean_cos_emb40"] - 0.0) < 1e-12
    assert red_mmr4["distinct_est"] == 4.0
    assert red_mmr4["distinct_dsl_head"] == 4.0


def test_struct4_selection_and_fallback() -> None:
    top16 = np.arange(16, dtype=np.int64)
    # Ranks 0,1,2 have (est=0, dsl=10), rank 3 has (est=1, dsl=20), rank 4 has (est=1, dsl=21), rank 5 has (est=2, dsl=30)
    est_id = np.array([0, 0, 0, 1, 1, 2] + [0] * 10, dtype=np.int32)
    dsl_id = np.array([10, 10, 10, 20, 21, 30] + [10] * 10, dtype=np.int64)

    # Pass 1 picks distinct est_id: 0 (idx 0), 1 (idx 3), 2 (idx 5); Pass 2 picks unseen dsl_id: 21 (idx 4)
    s_struct = select_struct4_from_top16(top16, est_id, dsl_id, est_only=False)
    assert s_struct == [0, 3, 5, 4]

    # Estimator-only: Pass 1 picks [0, 3, 5], Pass 3 falls back to next unselected by cost (idx 1)
    s_est = select_struct4_from_top16(top16, est_id, dsl_id, est_only=True)
    assert s_est == [0, 3, 5, 1]


def test_extract_heads_and_pick_tiebreak() -> None:
    prog = np.zeros((3, 412), dtype=np.float32)
    prog[0, 26] = 1.0  # est 2
    prog[0, 5] = 2.0  # op 5
    prog[1, 28] = 1.0  # est 4, no ops
    prog[2, 26] = 1.0  # est 2
    prog[2, 5] = 1.0  # same active op bitmask as cand 0

    est_id, dsl_id = extract_learner_and_dsl_heads(prog)
    assert list(est_id) == [2, 4, 2]
    assert dsl_id[0] == dsl_id[2]
    assert dsl_id[0] != dsl_id[1]

    # Tie-break test on pick_best_revealed_gated_cost
    rcost = np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float32)
    fitness = np.array([-1.2, -1.5, -1.5, -1.9], dtype=np.float64)
    gate = np.array([True, True, True, False], dtype=np.bool_)
    # Cand 3 fails gate (2.0); cands 1 and 2 tie at -1.5, cand 2 has lower rcost (0.1 < 0.2)
    assert pick_best_revealed_gated_cost([0, 1, 2, 3], rcost, fitness, gate) == 2


def test_score_context_and_paired_contrast() -> None:
    rng = np.random.default_rng(261001)
    n = 32
    sem24 = rng.normal(size=(n, 24)).astype(np.float32)
    proj16 = rng.normal(size=(n, 16)).astype(np.float32)
    emb40 = compute_candidate_emb40(sem24, proj16)
    assert emb40.shape == (n, 40)
    assert np.allclose(np.linalg.norm(emb40, axis=1), 1.0, atol=1e-4)

    fitness = rng.normal(-1.1, 0.1, size=n)
    ctx = {
        "utility": -fitness,
        "fitness": fitness,
        "gate": np.ones(n, dtype=np.bool_),
        "emb40": emb40,
        "sem24": sem24,
        "est_id": np.arange(n, dtype=np.int32) % 5,
        "dsl_head_id": np.arange(n, dtype=np.int64) % 8,
    }
    rcost_gated = np.linspace(0.0, 2.0, n, dtype=np.float32)
    rcost_gated[1] = rcost_gated[0] + 1e-4
    assert compute_top_zgap(rcost_gated) < GATE_THRESHOLD
    det_g = score_h261_context_deterministic(ctx, rcost_gated)
    assert det_g["is_gated"] is True
    assert det_g["reveals_used"]["TOP4_PICK"] == 4
    assert det_g["reveals_used"]["STRUCT4"] == 4
    assert det_g["reveals_used"]["MMR4_L0.3"] == 4

    rcost_ungated = np.linspace(0.0, 1.0, n, dtype=np.float32)
    rcost_ungated[0] = -5.0
    det_u = score_h261_context_deterministic(ctx, rcost_ungated)
    assert det_u["is_gated"] is False
    assert det_u["reveals_used"]["TOP4_PICK"] == 0
    assert det_u["reveals_used"]["STRUCT4"] == 0

    rows = [{"family_id": f"fam_{i}"} for i in range(4)]
    panel = np.zeros((3, 4, 2), dtype=np.float64)
    panel[:, :, 0] = 1.20
    panel[:, :, 1] = 1.25
    boot_ix = rng.integers(0, 4, size=(100, 4))
    res = paired_family_contrast(rows, panel, ["BASE", "CAND"], "CAND", "BASE", boot_ix)
    assert abs(res["median_family_gain"] - 0.05) < 1e-12
    assert res["positive_seeds"] == 3


def test_completed_metrics_invariants() -> None:
    metrics = json.loads((STUDY_DIR / "metrics.json").read_text())
    if metrics.get("status") != "COMPLETE":
        return
    assert metrics["verdict"] in ("ADVANCE", "NO_ADVANCE")
    base = metrics["baseline_regression"]
    assert base["r882_native91_abs_diff"] <= 1e-12
    assert base["top4_pick_val_ctx_abs_diff"] <= 1e-12
    assert base["top4_pick_val_fam_abs_diff"] <= 1e-12
    assert metrics["audits"]["audit_a_zero_leakage"]["pass"] is True
    assert metrics["audits"]["audit_b_reveal_budget"]["pass"] is True
    assert metrics["audits"]["audit_c_permuted_negative_control_dl_val"]["pass"] is True
    elo_top4 = metrics["dl_val_native91"]["mm_bt_elo_anchored_top4_pick"]["TOP4_PICK"][
        "median_elo"
    ]
    assert abs(elo_top4 - 1500.0) <= 1e-9
