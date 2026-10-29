"""Unit and parity tests for Phase E fused-path HPC speed-ups (NORTH_STAR NS11)."""

from __future__ import annotations

import json
import math
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from jope.fused_hpc import (
    FusedStageCudaGraphRunner,
    PinnedBatchBufferPool,
    bucket_tables_2d_by_p_tier_and_ntr,
    count_rust_tokens_out_of_loop,
    erf_approx_f64,
    exact_corr_with_yz_f64,
    fused_14_probe_corrs_with_yz_f64,
    parallel_load_dataset_splits,
    patch_fused_network_hpc,
    plan_multigpu_shards,
    precompute_champion_topk_batched_gpu,
    reuse_monotone_sort_order_f64,
    sort_valid_rows_radix_f64,
    sort_valid_values_radix_f64,
)
from jope.fused_hpc.accelerator import compute_table_seed_sha256_u32


def test_hpc1_batched_champion_retrieval_and_table_seed() -> None:
    torch.manual_seed(42)
    b, n_tpl, n_c128 = 6, 64, 32
    lm48 = torch.randn(b, 48, dtype=torch.float32)
    lm48_z = torch.randn(b, 48, dtype=torch.float32)
    champ_lm48_z = torch.randn(n_tpl, 48, dtype=torch.float32)
    # Create an intentional tie between templates 5 and 11
    champ_lm48_z[11] = champ_lm48_z[5].clone()
    champ_t_norms = torch.linalg.norm(champ_lm48_z, dim=1).clamp_min(1e-6)

    champ_lm12_128 = torch.randn(n_c128, 12, dtype=torch.float32)
    # Intentional L2 tie between indices 3 and 9, where index 9 has lexicographically smaller SHA
    champ_lm12_128[9] = champ_lm12_128[3].clone()
    shas = [f"sha_{i:04d}" for i in range(n_c128)]
    shas[3] = "zzzz_tie_later"
    shas[9] = "aaaa_tie_earlier"

    out = precompute_champion_topk_batched_gpu(
        lm48,
        lm48_z,
        champ_lm48_z,
        champ_t_norms,
        champ_lm12_128,
        champ_shas=shas,
    )

    assert out["sims_881"].shape == (b, n_tpl)
    assert out["top_t_indices_881"].shape == (b, n_tpl)
    assert out["order_128"].shape == (b, n_c128)
    assert out["top4_champ128_indices"].shape == (b, 4)

    # Check stable ascending index tie-break for 881 templates (index 5 must precede index 11)
    for row in range(b):
        order_row = out["top_t_indices_881"][row].tolist()
        assert order_row.index(5) < order_row.index(11)

        # Check SHA lexicographic tie-break for 128 champions (index 9 "aaaa..." before index 3 "zzzz...")
        order128_row = out["order_128"][row].tolist()
        assert order128_row.index(9) < order128_row.index(3)

    x_np = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.inf]], dtype=np.float32)
    y_np = np.array([0.5, -1.25], dtype=np.float32)
    seed_full = compute_table_seed_sha256_u32(x_np, y_np)
    seed_fast = compute_table_seed_sha256_u32(x_np, y_np, fast_subsample=True)
    assert seed_full == seed_fast  # small table <= 64x8 hashes identical stream
    assert 0 <= seed_full < (1 << 32)


def test_hpc2_radix_sort_and_monotone_order_reuse() -> None:
    torch.manual_seed(7)
    b, n, p = 4, 32, 8
    x_in = torch.randn(b, n, p, dtype=torch.float64)
    r_mask_bool = torch.zeros((b, n), dtype=torch.bool)
    for bi in range(b):
        r_mask_bool[bi, : (16 + bi * 4)] = True

    sorted_vals, order_x = sort_valid_rows_radix_f64(x_in, r_mask_bool)
    sorted_only = sort_valid_values_radix_f64(x_in, r_mask_bool)
    assert torch.equal(sorted_vals, sorted_only)

    # Monotone affine + clamp transform: Z_clp = clamp((X_in - mu) / sd, -8, 8)
    mu = x_in.mean(dim=1, keepdim=True)
    sd = x_in.std(dim=1, keepdim=True).clamp_min(1e-3)
    z_unc = (x_in - mu) / sd
    z_clp = torch.clamp(z_unc, -8.0, 8.0)

    reused_unc, reused_order = reuse_monotone_sort_order_f64(z_unc, order_x, r_mask_bool)
    resorted_unc, _ = sort_valid_rows_radix_f64(z_unc, r_mask_bool)
    reused_clp, _ = reuse_monotone_sort_order_f64(z_clp, order_x, r_mask_bool)
    resorted_clp, _ = sort_valid_rows_radix_f64(z_clp, r_mask_bool)

    assert torch.equal(reused_order, order_x)
    assert torch.max(torch.abs(reused_unc - resorted_unc)).item() == 0.0
    assert torch.max(torch.abs(reused_clp - resorted_clp)).item() == 0.0


def test_hpc3_fused_14_probe_corrs_parity() -> None:
    torch.manual_seed(123)
    b, n, p = 5, 48, 12
    r_mask = torch.zeros((b, n), dtype=torch.float64)
    for bi in range(b):
        r_mask[bi, : (24 + bi * 5)] = 1.0
    c_mask = torch.ones((b, p), dtype=torch.float64)
    c_mask[:, -2:] = 0.0
    rc_mask = r_mask.unsqueeze(-1) * c_mask.unsqueeze(1)
    n_b = r_mask.sum(dim=1, keepdim=True)

    z_clp = torch.clamp(torch.randn(b, n, p, dtype=torch.float64), -8.0, 8.0) * rc_mask
    y_raw = torch.randn(b, n, dtype=torch.float64) * r_mask
    y_mean = y_raw.sum(dim=1, keepdim=True) / n_b.clamp_min(1.0)
    y_std = torch.sqrt((((y_raw - y_mean) * r_mask) ** 2).sum(dim=1, keepdim=True) / n_b.clamp_min(1.0))
    y_z = ((y_raw - y_mean) / y_std.clamp_min(1e-6)) * r_mask

    c_q10 = torch.full((b, p), -1.2, dtype=torch.float64)
    c_q25 = torch.full((b, p), -0.6, dtype=torch.float64)
    c_q50 = torch.zeros((b, p), dtype=torch.float64)
    c_q75 = torch.full((b, p), 0.6, dtype=torch.float64)
    c_q90 = torch.full((b, p), 1.2, dtype=torch.float64)

    fused = fused_14_probe_corrs_with_yz_f64(
        z_clp, y_z, r_mask, n_b, c_mask, c_q10, c_q25, c_q50, c_q75, c_q90
    )

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    half_pi = 0.5 * math.pi
    one_third = 1.0 / 3.0
    ref = {
        "c_asinh": exact_corr_with_yz_f64(torch.asinh(z_clp), y_z, r_mask, n_b) * c_mask,
        "c_atan": exact_corr_with_yz_f64(torch.atan(z_clp), y_z, r_mask, n_b) * c_mask,
        "c_erf": exact_corr_with_yz_f64(erf_approx_f64(z_clp * inv_sqrt2), y_z, r_mask, n_b) * c_mask,
        "c_cbrt": exact_corr_with_yz_f64(
            torch.sign(z_clp) * torch.pow(torch.abs(z_clp), one_third), y_z, r_mask, n_b
        )
        * c_mask,
        "c_log1p": exact_corr_with_yz_f64(torch.sign(z_clp) * torch.log1p(torch.abs(z_clp)), y_z, r_mask, n_b)
        * c_mask,
        "c_tanh": exact_corr_with_yz_f64(torch.tanh(0.75 * z_clp), y_z, r_mask, n_b) * c_mask,
        "c_laplace": exact_corr_with_yz_f64(torch.expm1(-torch.abs(z_clp)), y_z, r_mask, n_b) * c_mask,
        "c_clamp": exact_corr_with_yz_f64(
            torch.clamp(z_clp, c_q10.unsqueeze(1), c_q90.unsqueeze(1)), y_z, r_mask, n_b
        )
        * c_mask,
        "c_step": exact_corr_with_yz_f64((z_clp >= c_q50.unsqueeze(1)).to(torch.float64), y_z, r_mask, n_b)
        * c_mask,
        "c_hinge": exact_corr_with_yz_f64(
            torch.clamp(z_clp - c_q75.unsqueeze(1), min=0.0)
            - torch.clamp(c_q25.unsqueeze(1) - z_clp, min=0.0),
            y_z,
            r_mask,
            n_b,
        )
        * c_mask,
        "c_sin": exact_corr_with_yz_f64(torch.sin(half_pi * z_clp), y_z, r_mask, n_b),
        "c_cos": exact_corr_with_yz_f64(torch.cos(half_pi * z_clp), y_z, r_mask, n_b),
        "c_fract": exact_corr_with_yz_f64(z_clp - torch.floor(z_clp), y_z, r_mask, n_b) * c_mask,
        "c_lin": exact_corr_with_yz_f64(z_clp, y_z, r_mask, n_b) * c_mask,
    }
    ref["c_sincos"] = torch.maximum(ref["c_sin"], ref["c_cos"]) * c_mask

    for key, ref_t in ref.items():
        diff = torch.max(torch.abs(fused[key] - ref_t)).item()
        assert diff < 1e-12, f"{key} max abs diff {diff} >= 1e-12"


def test_hpc4_pinned_pool_parallel_loader_and_out_of_loop_tokens() -> None:
    pool = PinnedBatchBufferPool(max_b=2, max_n_tr=8, max_n_ho=4, max_p=4, pin_memory=False)
    rng = np.random.default_rng(0)
    tables = [
        (
            rng.standard_normal((6, 3), dtype=np.float32),
            rng.standard_normal(6, dtype=np.float32),
            rng.standard_normal((3, 3), dtype=np.float32),
            rng.standard_normal(3, dtype=np.float32),
        ),
        (
            rng.standard_normal((10, 5), dtype=np.float32),
            rng.standard_normal(10, dtype=np.float32),
            rng.standard_normal((5, 5), dtype=np.float32),
            rng.standard_normal(5, dtype=np.float32),
        ),
        (
            rng.standard_normal((4, 2), dtype=np.float32),
            rng.standard_normal(4, dtype=np.float32),
            rng.standard_normal((2, 2), dtype=np.float32),
            rng.standard_normal(2, dtype=np.float32),
        ),
    ]
    batch = pool.collate_to_device(tables, torch.device("cpu"))
    assert batch["X_tr"].shape == (3, 10, 5)
    assert batch["X_ho"].shape == (3, 5, 5)
    assert batch["tr_row_mask"][0].sum().item() == 6.0
    assert batch["col_mask"][0].sum().item() == 3.0
    assert batch["col_mask"][1].sum().item() == 5.0

    loaded = parallel_load_dataset_splits([1, 2, 3, 4], lambda x: x * x, max_workers=2)
    assert loaded == [1, 4, 9, 16]

    sources = ["pub fn predict_row(x: &[f64]) -> f64 { x[0] }", "pub const N: usize = 4;"]
    counts = count_rust_tokens_out_of_loop(sources, encode_fn=lambda s: s.split(), max_workers=2)
    assert counts == [9, 6]


def test_hpc5_2d_bucketing_multigpu_sharding_and_network_patch() -> None:
    # 1. Verify 2D (P_tier, N_tr) shape bucketing on cohort915 manifest if present, plus synthetic cohort
    manifest_path = Path("/home/ubuntu/gpu_bench/infer_real_cohort_no_friedman_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shapes = [(int(d["rows"]), int(d["features"])) for d in manifest["datasets"]]
    else:
        rng = np.random.default_rng(99)
        shapes = []
        for i in range(915):
            n_r = int(rng.integers(64, 512))
            p_c = 1087 if (i % 31 == 0) else int(rng.integers(4, 16))
            shapes.append((n_r, p_c))

    plan = bucket_tables_2d_by_p_tier_and_ntr(shapes)
    assert len(plan.ordered_indices) == len(shapes)
    assert sorted(plan.ordered_indices) == list(range(len(shapes)))
    assert plan.padded_cells_2d < plan.padded_cells_1d
    assert plan.padding_reduction_ratio > 2.2

    # 2. Verify multi-GPU shard planner across xbabe1-3 and strict rejection of xbabe0
    items = [
        {"dataset_id": f"ds_{i:03d}", "rows": 100 + i * 10, "features": 10 + (i % 20)} for i in range(30)
    ]
    shards = plan_multigpu_shards(items)
    assert set(shards.keys()) == {"xbabe1", "xbabe2", "xbabe3"}
    assert sum(len(v) for v in shards.values()) == 30
    assert len(shards["xbabe1"]) >= len(shards["xbabe2"])

    with pytest.raises(ValueError, match="xbabe0"):
        plan_multigpu_shards(items, node_weights={"xbabe0": 0.5, "xbabe1": 0.5})

    # 3. Verify patch_fused_network_hpc locks DOPE weights/Dropout and forces Stage 2 fp32
    class DummyFusedNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dope_model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.1), nn.Linear(8, 4))
            self.router = nn.Linear(4, 2)
            self.last_amp_dtype: Any = None

        def _run_dope_mlc_v3_and_lm48(
            self, ms: torch.Tensor, col: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.dope_model(ms)
            return out, out + col.mean(dim=-1)

        def forward(self, ms: torch.Tensor, col: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.last_amp_dtype = kwargs.get("amp_dtype")
            dope_out, lm_out = self._run_dope_mlc_v3_and_lm48(ms, col)
            return self.router(dope_out + lm_out)

    net = DummyFusedNet()
    dummy_s1a = types.SimpleNamespace(_sort_valid_values_f64=None, _sort_valid_rows_f64=None)
    receipt = patch_fused_network_hpc(net, gpu_stage1a_module=dummy_s1a)

    assert receipt["dope_freeze_verified"] is True
    assert receipt["trainable_param_counts_by_submodule"]["dope_model"] == 0
    assert receipt["trainable_param_counts_by_submodule"]["router"] == 10
    assert dummy_s1a._sort_valid_rows_f64 is sort_valid_rows_radix_f64

    # Calling net.train(True) or net.dope_model[1].train(True) must keep DOPE in eval mode
    net.train(True)
    net.dope_model[1].train(True)
    assert all(not m.training for m in net.dope_model.modules())

    ms = torch.randn(3, 8)
    col = torch.randn(3, 4, 2)
    out = net(ms, col, amp_dtype=torch.float16)
    assert out.shape == (3, 2)
    assert net.last_amp_dtype == torch.float32

    # Verify FusedStageCudaGraphRunner eager fallback on CPU
    runner = FusedStageCudaGraphRunner(lambda a, b: (a + 1.0, b * 2.0), enable_cuda_graphs=True)
    r_a, r_b = runner.run_stage1a_neural(torch.ones(2, 4), torch.ones(2, 4))
    assert torch.allclose(r_a, torch.full((2, 4), 2.0))
    assert torch.allclose(r_b, torch.full((2, 4), 2.0))
