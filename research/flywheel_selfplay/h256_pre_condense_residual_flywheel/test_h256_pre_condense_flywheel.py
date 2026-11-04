"""Unit and artifact-integrity tests for H256 Pre-Condense Residual Flywheel & Production Bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402
from train_and_export_h256_flagship_production import PreCondenseTop8NeuralReranker  # noqa: E402


def _make_dummy_context(seed: int = 256) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    col_mask = np.zeros(64, dtype=bool)
    col_mask[:12] = True
    mask_10 = np.ones(10, dtype=bool)
    return {
        "dope629": rng.standard_normal(629).astype(np.float32),
        "hidden256": rng.standard_normal(256).astype(np.float32),
        "col_lm_64x17": rng.standard_normal((64, 17)).astype(np.float32),
        "col_mask_64": col_mask,
        "tokens_10x32": rng.standard_normal((10, 32)).astype(np.float32),
        "mask_10": mask_10,
    }


def test_pre_condense_residual_bridge_token_shapes_and_gates() -> None:
    torch.set_num_threads(2)
    device = torch.device("cpu")
    c0 = _make_dummy_context(256)
    c1 = _make_dummy_context(257)
    scaler = FoldScaler.fit(["c0", "c1"], {"c0": c0, "c1": c1})
    ct = scaler.transform_context(c0, device)

    rng = np.random.default_rng(256)
    cand_q = torch.tensor(rng.standard_normal((1, 16, 32)).astype(np.float32), device=device)
    sem24 = torch.tensor(rng.standard_normal((1, 16, 24)).astype(np.float32), device=device)
    gate_risk = torch.tensor(rng.uniform(0.0, 0.2, size=(1, 16)).astype(np.float32), device=device)
    sp_prior = torch.tensor(rng.standard_normal((1, 16)).astype(np.float32), device=device)

    expected_tokens = {
        "B_CAPACITY_CTRL_10TOK": 10,
        "C_LEARNED_COMPACT_BRIDGE_629": 10,
        "D_PRE_CONDENSE_FULL629_20TOK": 20,
        "E_FULL629_PLUS_HIDDEN256_28TOK": 28,
        "F_FULL629_HIDDEN256_COL64X17": 92,
        "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED": 92,
    }
    for arm, exp_t in expected_tokens.items():
        model = PreCondenseResidualBridge(arm=arm, d_token=32, n_heads=4).to(device)
        model.eval()
        assert torch.all(model.prior_residual_out.weight == 0.0)
        assert torch.all(model.traj_residual_out.weight == 0.0)
        tokens, mask, scale_8 = model.encode_context_tokens(ct)
        assert tokens.shape == (1, exp_t, 32)
        assert mask.shape == (1, exp_t)
        assert scale_8.shape == (1, 8)
        delta_p, attn_ctx, bilinear_24 = model.forward_prior_and_context(
            ct, cand_q, sem24, gate_risk, sp_prior
        )
        assert delta_p.shape == (1, 16)
        assert attn_ctx.shape == (1, 16, 32)
        assert bilinear_24.shape == (1, 16, 24)
        assert torch.allclose(delta_p, torch.zeros_like(delta_p), atol=1e-7)


def test_pre_condense_top8_neural_reranker_forward() -> None:
    torch.set_num_threads(2)
    model = PreCondenseTop8NeuralReranker(in_dim=616, hidden_dim=128)
    model.eval()
    x = torch.randn(4, 8, 616, dtype=torch.float32)
    with torch.no_grad():
        scores, inv_probs = model(x)
    assert scores.shape == (4, 8)
    assert inv_probs.shape == (4, 8)
    assert torch.all(torch.isfinite(scores))
    assert torch.all((inv_probs >= 0.0) & (inv_probs <= 1.0))


def test_committed_weights_and_kpi_telemetry_integrity() -> None:
    weights_dir = HERE / "weights"
    if not weights_dir.exists():
        return

    manifest_path = weights_dir / "checkpoint_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "jope.h256.checkpoint_manifest/v1"

    kpi_path = HERE / "kpi_telemetry.json"
    assert kpi_path.exists()
    kpi = json.loads(kpi_path.read_text())
    assert kpi["schema"] == "jope.h256.training_and_validation_kpi_telemetry/v1"

    st_oof = kpi["flagship_models_detailed_partition_kpis"]["H256_STATIC_0REV_TOP8_SOFT_VETO"][
        "dl_train_oof_1219"
    ]
    st_val = kpi["flagship_models_detailed_partition_kpis"]["H256_STATIC_0REV_TOP8_SOFT_VETO"][
        "dl_val_91"
    ]
    act_oof = kpi["flagship_models_detailed_partition_kpis"]["H256_ACTIVE_2PLUS2_UNIFIED"][
        "dl_train_oof_1219"
    ]
    act_val = kpi["flagship_models_detailed_partition_kpis"]["H256_ACTIVE_2PLUS2_UNIFIED"][
        "dl_val_91"
    ]

    assert abs(float(st_oof["mean_family_utility"]) - 1.461114) < 1e-4
    assert abs(float(st_val["mean_family_utility"]) - 1.203296) < 1e-4
    assert abs(float(act_oof["mean_family_utility"]) - 1.489648) < 1e-4
    assert abs(float(act_val["mean_family_utility"]) - 1.201692) < 1e-4

    prod_1310 = torch.load(
        weights_dir / "h256_production_full_1310_regression_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert prod_1310["schema"] == "jope.h256.production_full_1310_regression_bundle/v1"
    assert "kpi_telemetry" in prod_1310
    assert "top8_lgbm_ranker_616_model_str" in prod_1310
    assert "top8_neural_reranker_state_dict" in prod_1310

    from safetensors.numpy import load_file as load_safetensors_np

    st_bundles = manifest.get("rust_safetensors_inference_bundles", {})
    expected_st = [
        "h256_production_full_1310_regression_bundle.safetensors",
        "h256_production_1219_dl_train_bundle.safetensors",
        "stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors",
        "stage1_full_s3_reg_all15796_encoder.safetensors",
        "full_s3_reg_7split_pretrained_encoder.safetensors",
        "sos_hurdle_and_sp_critic_models.safetensors",
        "golden_inference_fixture_context0.safetensors",
    ]
    for st_name in expected_st:
        assert st_name in st_bundles, f"Missing {st_name} in checkpoint_manifest.json"
        st_path = weights_dir / st_name
        assert st_path.exists(), f"Missing safetensors file {st_path}"
        assert st_path.stat().st_size <= 5 * 1024 * 1024
        tensors = load_safetensors_np(str(st_path))
        assert len(tensors) == int(st_bundles[st_name]["tensor_count"])
        for arr in tensors.values():
            assert np.all(np.isfinite(arr))
