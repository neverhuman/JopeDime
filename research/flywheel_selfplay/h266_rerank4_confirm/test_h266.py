from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from research.flywheel_selfplay.h266_rerank4_confirm.runner import (
    ABLATION_SPECS,
    ALL_COLS,
    FOLD_SEEDS,
    SEEDS,
    TOTAL_FEATURE_DIM,
    build_top4_feature_matrix_for_context,
    compute_sem24_numpy,
    compute_true_top4_relevance,
    fit_lgbm_ranker_200,
    paired_family_contrast,
    predict_lgbm_ranker,
    select_from_top4_scores,
    shuffle_labels_within_family,
)

STUDY_DIR = Path(__file__).resolve().parent


def test_fold_manifest_45_cells_and_zero_overlap() -> None:
    fm = json.loads((STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8"))
    assert fm["schema"] == "jope.h266.fold-manifest/v1"
    assert fm["outcomes_read"] == 0
    assert fm["fold_seeds"] == list(FOLD_SEEDS)
    assert fm["model_seeds"] == list(SEEDS)
    assert fm["n_cells_total"] == 45
    assert fm["train_val_family_overlap"] == 0

    for fseed in FOLD_SEEDS:
        sched = fm["fold_schedules"][str(fseed)]
        seen_fams: set[str] = set()
        seen_cids: set[str] = set()
        for f in sched["folds"]:
            assert f["n_families"] == 41
            fset = set(f["family_ids"])
            cset = set(f["context_ids"])
            assert not (seen_fams & fset)
            assert not (seen_cids & cset)
            seen_fams |= fset
            seen_cids |= cset
        assert len(seen_fams) == 205
        assert len(seen_cids) == 1219


def test_ablation_column_specs() -> None:
    assert len(ALL_COLS) == TOTAL_FEATURE_DIM == 560
    assert len(ABLATION_SPECS["SCORE_ONLY"]) == 32
    assert len(ABLATION_SPECS["SCORE_ONLY_20"]) == 20
    assert len(ABLATION_SPECS["NO_SCOREGEOM"]) == 528
    assert len(ABLATION_SPECS["NO_SEM24"]) == 531
    assert len(ABLATION_SPECS["NO_CTX76"]) == 484


def test_feature_builder_and_sem24() -> None:
    rng = np.random.default_rng(266001)
    prog = rng.normal(size=(32, 412)).astype(np.float32)
    sem24 = compute_sem24_numpy(prog)
    assert sem24.shape == (32, 24)
    rcost = np.linspace(-1.2, 0.5, 32, dtype=np.float32)
    aprior = np.linspace(-1.1, 0.4, 32, dtype=np.float32)
    proj16 = rng.normal(size=(32, 16)).astype(np.float32)
    lm48_z = rng.normal(size=(48,)).astype(np.float32)
    dope629 = rng.normal(size=(629,)).astype(np.float32)

    top4, X4 = build_top4_feature_matrix_for_context(
        rcost, aprior, prog[:4], sem24, proj16, lm48_z, dope629
    )
    assert top4.tolist() == [0, 1, 2, 3]
    assert X4.shape == (4, 560)
    assert np.all(np.isfinite(X4))


def test_relevance_margin_and_4d_contrast() -> None:
    fit4 = np.array([-1.10, -1.30, -1.20, -1.05], dtype=np.float64)
    gate4 = np.array([True, False, True, True], dtype=np.bool_)
    rel = compute_true_top4_relevance(fit4, gate4)
    assert rel.tolist() == [2, 0, 3, 1]

    scores = np.array([0.10, 0.35, 0.20, -0.10], dtype=np.float64)
    assert select_from_top4_scores(scores, margin=0.20) == 1
    assert select_from_top4_scores(scores, margin=0.30) == 0
    assert select_from_top4_scores(scores, margin=math.inf) == 0

    rel_by_cid = {
        "c1": np.array([3, 2, 1, 0], dtype=np.int32),
        "c2": np.array([0, 3, 2, 1], dtype=np.int32),
    }
    fam_by_cid = {"c1": "famA", "c2": "famA"}
    shuf = shuffle_labels_within_family(
        rel_by_cid, ["c1", "c2"], fam_by_cid, seed=266101
    )
    assert sorted(np.concatenate([rel_by_cid["c1"], rel_by_cid["c2"]]).tolist()) == (
        sorted(np.concatenate([shuf["c1"], shuf["c2"]]).tolist())
    )

    rng = np.random.default_rng(266002)
    X = rng.normal(size=(24, 32)).astype(np.float32)
    y = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 6)
    model = fit_lgbm_ranker_200(X, y, n_groups=6, lr=0.05, random_state=266002)
    p50 = predict_lgbm_ranker(model, X, num_iteration=50)
    assert p50.shape == (24,)

    rows = [{"family_id": f"fam_{i}"} for i in range(4)]
    panel_4d = np.zeros((3, 3, 4, 2), dtype=np.float64)
    panel_4d[:, :, :, 0] = 1.20
    panel_4d[:, :, :, 1] = 1.24
    boot_ix = rng.integers(0, 4, size=(100, 4))
    res = paired_family_contrast(
        rows,
        panel_4d,
        ["CONTROL", "RERANK4"],
        "RERANK4",
        "CONTROL",
        boot_ix,
        agg_across_folds="median",
    )
    assert abs(res["median_family_gain"] - 0.04) < 1e-12
    assert res["positive_seeds"] == 3


def test_completed_metrics_invariants() -> None:
    m_path = STUDY_DIR / "metrics.json"
    assert m_path.stat().st_size <= 50_000
    metrics = json.loads(m_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "COMPLETE":
        return
    assert metrics["verdict"] in ("ADVANCE", "NO_ADVANCE")
    assert metrics["confirmation_status"] in ("CONFIRMED", "NOT_CONFIRMED")
    base = metrics["baseline_regression"]
    assert base["r882_native91_abs_diff"] == 0.0
    assert base["r882_bit_exact_context_matches"] == 91
    assert base["top4_pick_oracle_ceiling_ctx_abs_diff"] <= 1e-12
