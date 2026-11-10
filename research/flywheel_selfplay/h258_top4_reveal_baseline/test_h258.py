from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from research.flywheel_selfplay.h258_top4_reveal_baseline import runner

STUDY_DIR = Path(__file__).resolve().parent


def test_manifests_and_folds_outcome_free() -> None:
    manifest = json.loads((STUDY_DIR / "manifest.json").read_text())
    fold_manifest = json.loads((STUDY_DIR / "fold_manifest.json").read_text())
    assert manifest["wave_id"] == "h258-top4-reveal-baseline-20261103"
    assert manifest["fold_seed"] == 258140
    assert manifest["seeds"] == [258101, 258102, 258103]
    assert manifest["boot_seed"] == 258150
    assert manifest["holm_m"] == 2
    assert manifest["primary_contrast"] == "ACTIVE_2PLUS2_HURDLE - TOP4_PICK"
    assert manifest["secondary_contrast"] == "TOP4_PICK - GATED_025"

    assert fold_manifest["dl_train_contexts"] == 1219
    assert fold_manifest["dl_train_families"] == 205
    assert fold_manifest["dl_val_contexts"] == 91
    assert fold_manifest["dl_val_families"] == 19
    assert fold_manifest["train_val_family_overlap"] == 0
    assert fold_manifest["has_meta_test_any"] is False
    folds = [set(f["families"]) for f in fold_manifest["folds"]]
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(folds[i] & folds[j]) == 0


def test_pick_best_revealed_gated_cost_and_tiebreak() -> None:
    fitness = np.array([-1.2, -1.5, -1.5, -1.9], dtype=np.float64)
    gate = np.array([True, True, True, False], dtype=np.bool_)
    # Candidate 3 fails gate (cost 2.0); candidates 1 and 2 tie at -1.5; rcost order [2, 1, 0, 3] picks 2 first
    pick = runner.pick_best_revealed_gated_cost([2, 1, 0, 3], fitness, gate)
    assert pick == 2
    pick_rev = runner.pick_best_revealed_gated_cost([1, 2, 0, 3], fitness, gate)
    assert pick_rev == 1


def test_score_h258_context_gated_and_ungated_reveal_budget() -> None:
    _, stage1_mod = runner.load_snapshot_modules()
    rng_data = np.random.default_rng(258001)
    n = 32
    fitness = rng_data.normal(-1.1, 0.1, size=n)
    gate = rng_data.uniform(0.0, 1.0, size=n) > 0.15
    pool = {
        "utility": -fitness,
        "fitness": fitness,
        "logratio": np.clip(fitness + 1.1, -1.5, 1.5),
        "gate": gate,
        "sem24": rng_data.normal(0.0, 1.0, size=(n, 24)),
    }
    sos_prior = rng_data.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
    sp_prior = rng_data.normal(0.0, 1.0, size=n).astype(np.float32)

    # Gated context: top_zgap < 0.25 (two near-tied best costs)
    rcost_gated = np.linspace(0.0, 2.0, n, dtype=np.float32)
    rcost_gated[1] = rcost_gated[0] + 1e-4
    out_g = runner.score_h258_context(
        pool,
        rcost_gated,
        sos_prior,
        sp_prior,
        np.random.default_rng(258101),
        stage1_mod,
    )
    assert out_g["is_gated"] is True
    assert out_g["snap_match"] is True
    for k in [
        "GATED_025",
        "ACTIVE_2PLUS2_HURDLE",
        "TOP4_PICK",
        "TOP2_RAND2_PICK",
        "ACTIVE_2PLUS2_PICK",
    ]:
        assert out_g["reveals_used"][k] == 4

    # Ungated context: top_zgap >= 0.25 (clear winner)
    rcost_ungated = np.linspace(0.0, 1.0, n, dtype=np.float32)
    rcost_ungated[0] = -5.0
    out_u = runner.score_h258_context(
        pool,
        rcost_ungated,
        sos_prior,
        sp_prior,
        np.random.default_rng(258101),
        stage1_mod,
    )
    assert out_u["is_gated"] is False
    assert out_u["snap_match"] is True
    for k in [
        "GATED_025",
        "ACTIVE_2PLUS2_HURDLE",
        "TOP4_PICK",
        "TOP2_RAND2_PICK",
        "ACTIVE_2PLUS2_PICK",
    ]:
        assert out_u["reveals_used"][k] == 0
        assert out_u["scores"][k] == out_u["scores"]["R882"]
    assert out_u["reveals_used"]["TOP4_PICK_UNGATED"] == 4


def test_completed_metrics_invariants() -> None:
    metrics = json.loads((STUDY_DIR / "metrics.json").read_text())
    if metrics.get("status") != "COMPLETE":
        return
    assert metrics["verdict"] == "NO_ADVANCE"
    base = metrics["baseline_regression"]
    assert base["r882_native91_abs_diff"] == 0.0
    assert base["r882_bit_exact_context_matches"] == 91
    assert metrics["audits"]["audit_a_zero_leakage"]["pass"] is True
    assert metrics["audits"]["audit_b_reveal_budget"]["pass"] is True
    assert metrics["audits"]["audit_c_permuted_negative_control_dl_val"]["pass"] is True
    prim = metrics["dl_val_native91"]["contrasts_r882"][
        "ACTIVE_2PLUS2_HURDLE - TOP4_PICK"
    ]
    assert prim["median_family_gain"] < 0.0
    assert prim["positive_seeds"] == 0
    sec = metrics["dl_val_native91"]["contrasts_r882"]["TOP4_PICK - GATED_025"]
    assert sec["median_family_gain"] > 0.09
    assert sec["positive_seeds"] == 3
