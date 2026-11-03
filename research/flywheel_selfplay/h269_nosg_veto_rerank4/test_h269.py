#!/usr/bin/env python3
"""Unit tests for H269 (`h269_nosg_veto_rerank4`)."""

from __future__ import annotations

import json
import math
import numpy as np

from research.flywheel_selfplay.h269_nosg_veto_rerank4.runner import (
    CAPSULE_STUDY_DIR,
    FOLD_SEEDS,
    NO_SCOREGEOM_COLS,
    NOSG_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
    VETO_TAU_GRID,
    apply_holm_bonferroni,
    build_top4_feature_matrix_for_context,
    compute_sem24_numpy,
    compute_true_top4_relevance,
    fit_lgbm_ranker_200,
    fit_veto_classifier,
    predict_lgbm_ranker,
    predict_veto_classifier,
    select_from_top4_scores,
    select_from_top4_scores_with_veto,
    shuffle_labels_within_family,
)


def test_nosg_528_feature_builder_and_score_invariance() -> None:
    rng = np.random.default_rng(26901)
    rcost = rng.uniform(0.0, 2.0, size=64).astype(np.float32)
    aprior = rng.uniform(0.0, 2.0, size=64).astype(np.float32)
    prog = rng.normal(size=(64, 412)).astype(np.float32)
    sem24 = compute_sem24_numpy(prog)
    proj16 = rng.normal(size=(64, 16)).astype(np.float32)
    lm48_z = rng.normal(size=48).astype(np.float32)
    dope629 = rng.normal(size=629).astype(np.float32)

    top4_a = np.argsort(rcost, kind="mergesort")[:4]
    idx4_a, X_a = build_top4_feature_matrix_for_context(
        rcost, aprior, prog[top4_a], sem24, proj16, lm48_z, dope629
    )
    assert X_a.shape == (4, TOTAL_FEATURE_DIM)
    X_nosg_a = X_a[:, NO_SCOREGEOM_COLS]
    assert X_nosg_a.shape == (4, NOSG_FEATURE_DIM)

    # Perturb scalar rcost magnitudes while preserving top-4 order -> NO_SCOREGEOM_COLS (32..559) must be invariant!
    rcost_b = rcost * 3.7 + 1.25
    top4_b = np.argsort(rcost_b, kind="mergesort")[:4]
    assert np.array_equal(top4_a, top4_b)
    _, X_b = build_top4_feature_matrix_for_context(
        rcost_b, aprior, prog[top4_b], sem24, proj16, lm48_z, dope629
    )
    assert np.allclose(X_a[:, NO_SCOREGEOM_COLS], X_b[:, NO_SCOREGEOM_COLS], atol=1e-6)


def test_relevance_and_veto_selection_rule() -> None:
    fit4 = np.array([0.8, 0.2, 0.5, 0.1], dtype=np.float64)
    gate4 = np.array([True, False, True, True], dtype=np.bool_)
    rel = compute_true_top4_relevance(fit4, gate4)
    assert rel[1] == 0  # invalid gate forced to 0
    assert rel[3] == 3  # best valid

    scores = np.array([0.10, 0.90, 0.30, 0.20], dtype=np.float64)
    p_inv = np.array([0.01, 0.15, 0.02, 0.01], dtype=np.float64)
    # Without veto, slot 1 is picked at margin 0.10
    assert select_from_top4_scores(scores, 0.10) == 1
    # With tau = 0.05 or 0.10, p_inv[1] = 0.15 > tau -> vetoed back to slot 0!
    assert select_from_top4_scores_with_veto(scores, p_inv, 0.05, 0.10) == 0
    assert select_from_top4_scores_with_veto(scores, p_inv, 0.10, 0.10) == 0
    # With tau = 0.20, p_inv[1] = 0.15 <= 0.20 -> allowed!
    assert select_from_top4_scores_with_veto(scores, p_inv, 0.20, 0.10) == 1
    assert select_from_top4_scores_with_veto(scores, p_inv, 0.20, math.inf) == 0
    assert VETO_TAU_GRID == (0.05, 0.10, 0.20)


def test_within_family_shuffle_and_holm_m3() -> None:
    cids = ["c1", "c2", "c3"]
    fam_by_cid = {"c1": "f1", "c2": "f1", "c3": "f2"}
    rel_map = {
        "c1": np.array([3, 2, 1, 0], dtype=np.int32),
        "c2": np.array([0, 3, 2, 1], dtype=np.int32),
        "c3": np.array([2, 3, 0, 1], dtype=np.int32),
    }
    shuf = shuffle_labels_within_family(rel_map, cids, fam_by_cid, seed=269101)
    assert set(shuf.keys()) == set(cids)
    assert sorted(np.concatenate([shuf["c1"], shuf["c2"]]).tolist()) == sorted(
        np.concatenate([rel_map["c1"], rel_map["c2"]]).tolist()
    )

    pvals = {
        "NOSG_TRAIN": 0.012,
        "NOSG_VETO_TRAIN": 0.004,
        "NOSG_VETO_VAL91": 0.020,
    }
    adj = apply_holm_bonferroni(pvals)
    assert abs(adj["NOSG_VETO_TRAIN"] - 0.012) < 1e-12
    assert abs(adj["NOSG_TRAIN"] - 0.024) < 1e-12
    assert abs(adj["NOSG_VETO_VAL91"] - 0.024) < 1e-12


def test_ranker_and_veto_classifier_smoke() -> None:
    rng = np.random.default_rng(26902)
    X = rng.normal(size=(40, NOSG_FEATURE_DIM)).astype(np.float32)
    y_rel = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 10)
    y_inv = np.zeros(40, dtype=np.int32)
    y_inv[3] = 1
    y_inv[7] = 1
    y_inv[15] = 1
    rk = fit_lgbm_ranker_200(X, y_rel, 10, lr=0.05, random_state=269101)
    preds = predict_lgbm_ranker(rk, X, num_iteration=50)
    clf = fit_veto_classifier(X, y_inv, random_state=269102)
    probs = predict_veto_classifier(clf, X)
    assert preds.shape == (40,)
    assert probs.shape == (40,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_fold_manifest_disjointness() -> None:
    fm = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    assert fm["outcomes_read"] == 0
    assert fm["fold_seeds"] == list(FOLD_SEEDS)
    assert fm["train_val_family_overlap"] == 0
    for fseed in FOLD_SEEDS:
        folds = fm["fold_schedules"][str(fseed)]["folds"]
        assert len(folds) == 5
        seen_fams: set[str] = set()
        seen_cids: set[str] = set()
        for f in folds:
            fset = set(f["family_ids"])
            cset = set(f["context_ids"])
            assert len(seen_fams.intersection(fset)) == 0
            assert len(seen_cids.intersection(cset)) == 0
            seen_fams.update(fset)
            seen_cids.update(cset)
        assert len(seen_fams) == 205
        assert len(seen_cids) == 1219
