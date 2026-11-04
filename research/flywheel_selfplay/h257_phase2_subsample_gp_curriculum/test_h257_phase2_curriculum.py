"""Unit and artifact-integrity tests for H257 Phase-2 Warm-Started Subsample & GP Evolutionary Curriculum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H256_DIR = HERE.parent / "h256_pre_condense_residual_flywheel"
for p in (HERE, H256_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import semi_synthetic_columns_and_views as ssa  # noqa: E402
import batched_subsample_gp_evolution as bge  # noqa: E402


def test_semisynthetic_column_and_subsample_generator_invariants() -> None:
    rng = np.random.default_rng(257)
    nt, width = 120, 12
    x = rng.standard_normal((nt, width)).astype(np.float64)
    y = (x[:, 0] * 1.5 - 0.8 * x[:, 1] + 0.2 * rng.standard_normal(nt)).astype(np.float64)
    folds = np.array([i % 2 for i in range(nt)], dtype=np.uint8)

    x100, y100 = ssa.generate_perturbed_table_view(x, y, folds, "full100", rng, subsample_frac=0.80)
    assert x100.shape == (nt, width)
    assert np.allclose(y100, y)

    x_sub, y_sub = ssa.generate_perturbed_table_view(x, y, folds, "sub80", rng, subsample_frac=0.80)
    assert x_sub.shape == (int(round(0.80 * nt)), width)
    assert y_sub.shape == (int(round(0.80 * nt)),)
    assert np.all(np.isfinite(x_sub)) and np.all(np.isfinite(y_sub))

    x_syn, y_syn = ssa.generate_perturbed_table_view(x, y, folds, "semisyn80", rng, subsample_frac=0.80)
    assert x_syn.shape[0] == int(round(0.80 * nt))
    assert x_syn.shape[1] > width
    assert y_syn.shape == (int(round(0.80 * nt)),)
    assert np.all(np.isfinite(x_syn)) and np.all(np.isfinite(y_syn))


def test_leak_free_2fold_oof_pareto_stack_invariants() -> None:
    rng = np.random.default_rng(2570)
    y0 = rng.standard_normal(50).astype(np.float64)
    y1 = rng.standard_normal(50).astype(np.float64)
    b0 = float(np.sqrt(np.mean((y0 - float(np.mean(y1))) ** 2)))
    b1 = float(np.sqrt(np.mean((y1 - float(np.mean(y0))) ** 2)))

    # Construct 3 distinct passing candidate OOF predictions
    p_evals = []
    for k, noise_scale in enumerate((0.25, 0.28, 0.30)):
        f0_p = y0 + noise_scale * rng.standard_normal(50)
        f1_p = y1 + noise_scale * rng.standard_normal(50)
        r0 = float(np.sqrt(np.mean((f0_p - y0) ** 2))) / b0
        r1 = float(np.sqrt(np.mean((f1_p - y1) ** 2))) / b1
        lr = 0.5 * (np.log(r0) + np.log(r1))
        p_evals.append(
            {
                "status": "completed",
                "cv_gate_passed": True,
                "dsl": f"(pipe X (ridge_gcv alpha={k + 1}))",
                "program_sha256": f"sha_{k:04d}",
                "fitness": float(lr),
                "utility": float(-lr),
                "logratio": float(lr),
                "fold0_preds": f0_p,
                "fold1_preds": f1_p,
            }
        )

    res = bge.compute_leak_free_2fold_oof_stack(p_evals, y0, y1, b0, b1, max_models=3)
    assert res is not None
    assert res["status"] == "completed"
    assert res["cv_gate_passed"] is True
    assert res["k_models"] == 3
    assert abs(sum(res["w_fold0"]) - 1.0) < 1e-6
    assert abs(sum(res["w_fold1"]) - 1.0) < 1e-6
    assert res["utility"] > max(e["utility"] for e in p_evals)


def test_h257_weights_and_telemetry_artifacts() -> None:
    st_path = HERE / "weights" / "h257_phase2_curriculum_bundle.safetensors"
    assert st_path.exists(), f"missing {st_path}"
    from safetensors.numpy import load_file as load_safetensors_np

    assert st_path.stat().st_size <= 5 * 1024 * 1024
    tensors = load_safetensors_np(str(st_path))
    assert len(tensors) == 384
    for arr in tensors.values():
        assert np.all(np.isfinite(arr))

    frontier_path = HERE / "frontier_elo_comparison.json"
    assert frontier_path.exists(), f"missing {frontier_path}"
    f_doc = json.loads(frontier_path.read_text())
    assert f_doc["schema"] == "jope.h257.frontier_elo_comparison/v1"
    t_s1 = f_doc["dl_val91_same_fold_tournament_s1_utility"]
    assert "Opus5p5_Extra" in t_s1
    assert "H257_HIGH_BATCH_EVO3_STACK3" in t_s1
    assert float(t_s1["H257_HIGH_BATCH_EVO3_STACK3"]["joint_elo"]) > float(
        t_s1["Opus5p5_Extra"]["joint_elo"]
    )

    kpi_path = HERE / "kpi_telemetry.json"
    assert kpi_path.exists(), f"missing {kpi_path}"
    k_doc = json.loads(kpi_path.read_text())
    assert k_doc["schema"] == "jope.h257.kpi_telemetry/v1"
    assert k_doc["wave"] == "H257"
