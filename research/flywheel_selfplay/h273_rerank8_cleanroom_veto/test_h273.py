from __future__ import annotations

import json
import math
import numpy as np

from research.flywheel_selfplay.h273_rerank8_cleanroom_veto.runner import (
    CAPSULE_STUDY_DIR,
    FEATURE_DIM_K4,
    FEATURE_DIM_K8,
    FOLD_SEEDS,
    FROZEN_VETO_TAU,
    SEEDS,
    SLICE_K,
    STAGE1_FOLD_SEEDS,
    STAGE1_SEEDS,
    build_topk_feature_matrix_for_context,
    compute_sem24_numpy,
    compute_true_topk_relevance,
    fit_lgbm_ranker_200,
    fit_veto_classifier,
    predict_lgbm_ranker,
    predict_veto_classifier,
    select_from_topk_scores,
    select_from_topk_scores_with_veto,
    shuffle_labels_within_family,
)


def test_clean_room_topk_features_k8_and_k4() -> None:
    rng = np.random.default_rng(27301)
    n_pool = 32
    rcost = rng.normal(0.5, 0.2, size=n_pool).astype(np.float32)
    aprior = rng.normal(0.5, 0.2, size=n_pool).astype(np.float32)
    prog = rng.normal(0.0, 1.0, size=(n_pool, 412)).astype(np.float32)
    sem24 = compute_sem24_numpy(prog)
    proj16 = rng.normal(0.0, 1.0, size=(n_pool, 16)).astype(np.float32)
    lm48_z = rng.normal(0.0, 1.0, size=48).astype(np.float32)
    dope629 = rng.normal(0.0, 1.0, size=629).astype(np.float32)

    top8, X8 = build_topk_feature_matrix_for_context(
        rcost, aprior, prog, sem24, proj16, lm48_z, dope629, k=8
    )
    assert top8.shape == (8,)
    assert X8.shape == (8, FEATURE_DIM_K8)
    assert np.all(np.isfinite(X8))
    assert np.allclose(X8[:, 4], np.arange(8, dtype=np.float32) / 7.0)

    top4, X4 = build_topk_feature_matrix_for_context(
        rcost, aprior, prog, sem24, proj16, lm48_z, dope629, k=4
    )
    assert top4.shape == (4,)
    assert X4.shape == (4, FEATURE_DIM_K4)
    assert np.all(np.isfinite(X4))
    assert np.array_equal(top8[:4], top4)
    # Candidate program + embedding + context blocks (last 528 cols) match on slots 0..3
    assert np.allclose(X8[:4, -528:], X4[:, -528:])


def test_topk_relevance_labels_and_veto_selector() -> None:
    fitness8 = np.array([0.80, 0.10, 0.30, 0.50, 0.20, 0.90, 0.40, 0.05])
    gate8 = np.array([True, False, True, True, True, True, False, True])
    rel8 = compute_true_topk_relevance(fitness8, gate8)
    # Valid indices sorted by ascending fitness:
    # j=7 (0.05)->7, j=4 (0.20)->6, j=2 (0.30)->5, j=3 (0.50)->4, j=0 (0.80)->3, j=5 (0.90)->2
    # Gate-failed j=1, j=6 -> 0
    assert rel8.tolist() == [3, 0, 5, 4, 6, 2, 0, 7]

    scores = np.array([0.1, 1.4, 0.8, 0.2, 0.9, 0.0, 1.2, 0.5], dtype=np.float64)
    p_inv = np.array([0.01, 0.65, 0.02, 0.01, 0.03, 0.02, 0.80, 0.01], dtype=np.float64)
    # Without veto, slot 1 (score 1.4) wins at margin 0.10, picking a -2.0 invalid gate!
    assert select_from_topk_scores(scores, margin=0.10) == 1
    # With veto at tau=0.10, slot 1 has p_inv=0.65 > 0.10 so it falls back to slot 0 (r882)
    assert (
        select_from_topk_scores_with_veto(
            scores, p_inv, tau=FROZEN_VETO_TAU, margin=0.10
        )
        == 0
    )
    assert (
        select_from_topk_scores_with_veto(
            scores, p_inv, tau=FROZEN_VETO_TAU, margin=math.inf
        )
        == 0
    )


def test_shuffle_labels_within_family_k8() -> None:
    cids = ["c0", "c1", "c2", "c3"]
    fam_by_cid = {"c0": "fA", "c1": "fA", "c2": "fB", "c3": "fB"}
    rel_by_cid = {
        "c0": np.arange(8, dtype=np.int32),
        "c1": np.arange(10, 18, dtype=np.int32),
        "c2": np.arange(100, 108, dtype=np.int32),
        "c3": np.arange(200, 208, dtype=np.int32),
    }
    shuf = shuffle_labels_within_family(
        rel_by_cid, cids, fam_by_cid, seed=27399, k=SLICE_K
    )
    fam_a_orig = sorted(np.concatenate([rel_by_cid["c0"], rel_by_cid["c1"]]).tolist())
    fam_a_shuf = sorted(np.concatenate([shuf["c0"], shuf["c1"]]).tolist())
    fam_b_orig = sorted(np.concatenate([rel_by_cid["c2"], rel_by_cid["c3"]]).tolist())
    fam_b_shuf = sorted(np.concatenate([shuf["c2"], shuf["c3"]]).tolist())
    assert fam_a_orig == fam_a_shuf
    assert fam_b_orig == fam_b_shuf


def test_lgbm_ranker_k8_and_veto_clf_synthetic() -> None:
    rng = np.random.default_rng(27302)
    X = rng.normal(size=(64, FEATURE_DIM_K8)).astype(np.float32)
    y_rel = np.tile(np.arange(7, -1, -1, dtype=np.int32), 8)
    y_inv = np.zeros(64, dtype=np.int32)
    y_inv[3::8] = 1

    ranker = fit_lgbm_ranker_200(
        X, y_rel, n_groups=8, lr=0.05, random_state=27302, k=SLICE_K
    )
    preds = predict_lgbm_ranker(ranker, X, num_iteration=50)
    assert preds.shape == (64,)
    assert np.all(np.isfinite(preds))

    clf = fit_veto_classifier(X, y_inv, random_state=27303)
    probs = predict_veto_classifier(clf, X)
    assert probs.shape == (64,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_fold_manifest_and_clean_room_provenance() -> None:
    fm = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    ptr = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.s3.json").read_text(encoding="utf-8")
    )
    assert fm["outcomes_read"] == 0
    assert ptr["outcomes_read"] == 0
    assert fm["train_val_family_overlap"] == 0
    assert fm["stage1_fold_seeds"] == list(STAGE1_FOLD_SEEDS)
    assert fm["stage1_model_seeds"] == list(STAGE1_SEEDS)
    assert fm["stage2_fold_seeds"] == list(FOLD_SEEDS)
    assert fm["stage2_model_seeds"] == list(SEEDS)

    for fs in (*STAGE1_FOLD_SEEDS, *FOLD_SEEDS):
        sched = fm["fold_schedules"][str(fs)]
        seen_fams: set[str] = set()
        seen_cids: set[str] = set()
        for f in sched["folds"]:
            fset = set(f["families"])
            cset = set(f["context_ids"])
            assert len(fset) == 41
            assert len(seen_fams & fset) == 0
            assert len(seen_cids & cset) == 0
            seen_fams |= fset
            seen_cids |= cset
        assert len(seen_fams) == 205
        assert len(seen_cids) == 1219

    runner_src = (CAPSULE_STUDY_DIR / "runner.py").read_text(encoding="utf-8")
    assert "import topk" not in runner_src
    assert "import h265_core" not in runner_src
    assert "h270_rerank8_deeper_slice" not in runner_src
