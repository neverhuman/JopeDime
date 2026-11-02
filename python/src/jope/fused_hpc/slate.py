"""Cooperative 256-thread live slate construction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jope.fused_hpc.accelerator import precompute_champion_topk_batched_gpu

BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE: str = r"""
// Part B HPC #1: 256-thread cooperative CUDA kernel replacing the <<<1, 32>>> lane-0
// serial build_live_96_slate_cuda_kernel in gpu_full_pipeline_v4.py.
extern "C" __global__ __launch_bounds__(256)
void build_live_96_slate_block256_cuda_kernel(
    const float* __restrict__ X_tr,      // (N, P)
    const float* __restrict__ y_tr,      // (N,)
    int N,
    int P,
    const int32_t* __restrict__ top_t_indices_881, // pre-sorted via cuBLAS GEMM (881,)
    const int32_t* __restrict__ top4_champ128,     // pre-sorted via cuBLAS GEMM (4,)
    const int32_t* __restrict__ champ_embs_881,    // (881, 4, 24)
    const int32_t* __restrict__ champ_embs_128,    // (128, 24)
    const uint64_t* __restrict__ sorted_keys,      // (n_keys,)
    int n_keys,
    uint32_t* __restrict__ out_table_seed,
    int32_t* __restrict__ out_slate_embs,          // (96, 24)
    int32_t* __restrict__ out_slate_key_idx        // (96,)
) {
    __shared__ uint64_t s_hash[256];
    const int tid = threadIdx.x;
    const int total_x = N * P;

    // 1. 256-thread strided reduction for deterministic table seed
    uint64_t h_local = 0x243F6A8885A308D3ULL ^ ((uint64_t)N << 32) ^ (uint64_t)P;
    for (int idx = tid; idx < total_x; idx += 256) {
        float v = X_tr[idx];
        uint32_t bits = __float_as_uint(isnan(v) || isinf(v) ? 0.0f : v);
        uint64_t mix = ((uint64_t)bits ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL)) * 0xBF58476D1CE4E5B9ULL;
        h_local ^= (mix ^ (mix >> 31));
    }
    for (int idx = tid; idx < N; idx += 256) {
        float v = y_tr[idx];
        uint32_t bits = __float_as_uint(isnan(v) || isinf(v) ? 0.0f : v);
        uint64_t mix = ((uint64_t)bits ^ ((uint64_t)(idx + total_x) * 0x9E3779B97F4A7C15ULL)) * 0x94D049BB133111EBULL;
        h_local ^= (mix ^ (mix >> 27));
    }
    s_hash[tid] = h_local;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        uint64_t final_h = s_hash[0] * 0xD6E8FEB86659FD93ULL;
        *out_table_seed = (uint32_t)((final_h ^ (final_h >> 32)) & 0xFFFFFFFFu);
    }

    // 2. 96-way parallel candidate emission + binary search over sorted_keys
    if (tid < 96) {
        const int32_t* src_emb = nullptr;
        if (tid < 92) {
            int tpl_rank = tid >> 2;
            int rank_slot = tid & 3;
            int tpl_idx = top_t_indices_881[tpl_rank];
            src_emb = &champ_embs_881[(tpl_idx * 4 + rank_slot) * 24];
        } else {
            int c128_idx = top4_champ128[tid - 92];
            src_emb = &champ_embs_128[c128_idx * 24];
        }
        uint64_t cand_key = 1469598103934665603ULL;
        for (int d = 0; d < 24; ++d) {
            int32_t v = src_emb[d];
            out_slate_embs[tid * 24 + d] = v;
            cand_key ^= (uint64_t)(uint32_t)v;
            cand_key *= 1099511628211ULL;
        }
        int lo = 0, hi = n_keys - 1, found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint64_t kv = sorted_keys[mid];
            if (kv == cand_key) {
                found = mid;
                break;
            } else if (kv < cand_key) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        out_slate_key_idx[tid] = found;
    }
}
"""
def _fnv1a_embs_hash_u64(embs_24: np.ndarray) -> np.ndarray:
    """Compute deterministic 64-bit FNV-1a keys over ``(..., 24)`` int32 embeddings."""
    arr = np.asarray(embs_24, dtype=np.int32)
    u32 = arr.view(np.uint32).astype(np.uint64)
    h = np.full(u32.shape[:-1], np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for d in range(u32.shape[-1]):
        h = (h ^ u32[..., d]) * prime
    return h


def _block256_table_seed_u32(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Compute the 256-lane cooperative XOR-folded table seed matching ``build_live_96_slate_block256_cuda_kernel``."""
    x_f32 = np.asarray(x_tr, dtype=np.float32).reshape(-1)
    y_f32 = np.asarray(y_tr, dtype=np.float32).reshape(-1)
    n_rows = int(y_f32.shape[0])
    n_cols = int(x_f32.shape[0] // max(n_rows, 1))
    total_x = int(x_f32.shape[0])

    x_clean = np.where(np.isfinite(x_f32), x_f32, np.float32(0.0))
    y_clean = np.where(np.isfinite(y_f32), y_f32, np.float32(0.0))
    x_bits = x_clean.view(np.uint32).astype(np.uint64)
    y_bits = y_clean.view(np.uint32).astype(np.uint64)

    init_seed = np.uint64(0x243F6A8885A308D3) ^ (np.uint64(n_rows) << np.uint64(32)) ^ np.uint64(n_cols)
    lane_hashes = np.full(256, init_seed, dtype=np.uint64)

    idx_x = np.arange(total_x, dtype=np.uint64)
    mix_x = (x_bits ^ (idx_x * np.uint64(0x9E3779B97F4A7C15))) * np.uint64(0xBF58476D1CE4E5B9)
    contrib_x = mix_x ^ (mix_x >> np.uint64(31))
    for tid in range(min(256, total_x)):
        if contrib_x[tid::256].size > 0:
            lane_hashes[tid] ^= np.bitwise_xor.reduce(contrib_x[tid::256])

    idx_y = np.arange(n_rows, dtype=np.uint64) + np.uint64(total_x)
    mix_y = (y_bits ^ (idx_y * np.uint64(0x9E3779B97F4A7C15))) * np.uint64(0x94D049BB133111EB)
    contrib_y = mix_y ^ (mix_y >> np.uint64(27))
    for tid in range(min(256, n_rows)):
        if contrib_y[tid::256].size > 0:
            lane_hashes[tid] ^= np.bitwise_xor.reduce(contrib_y[tid::256])

    folded = int(np.bitwise_xor.reduce(lane_hashes))
    final_h = (folded * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    seed_u32 = int((final_h ^ (final_h >> 32)) & 0xFFFFFFFF)
    return seed_u32, lane_hashes


def build_live_96_slate_block256(
    x_tr: np.ndarray | torch.Tensor,
    y_tr: np.ndarray | torch.Tensor,
    lm48: torch.Tensor,
    lm48_z: torch.Tensor,
    champ_lm48_z: torch.Tensor,
    champ_t_norms: torch.Tensor,
    champ_lm12_128: torch.Tensor,
    champ_embs_881: np.ndarray | torch.Tensor,
    champ_embs_128: np.ndarray | torch.Tensor,
    sorted_keys: np.ndarray | torch.Tensor | None = None,
    champ_shas: list[str] | None = None,
) -> dict[str, Any]:
    """Part B HPC #1: 256-thread block slate builder replacing lane-0 serial ``build_live_96_slate_cuda_kernel``.

    Combines:
    - Batched cuBLAS GEMM top-K retrieval over the 881 template library (top 23 templates x 4 ranks
      = 92 candidates) and 128 champion pool (top 4 champions = 4 candidates).
    - 256-lane strided cooperative table seed reduction (eliminating the single-thread ``O(N*P)``
      loop in ``build_live_96_slate_cuda_kernel``).
    - 96-way parallel candidate embedding assembly + binary search over ``sorted_keys``.
    """
    x_np = (
        x_tr.detach().cpu().numpy() if isinstance(x_tr, torch.Tensor) else np.asarray(x_tr, dtype=np.float32)
    )
    y_np = (
        y_tr.detach().cpu().numpy() if isinstance(y_tr, torch.Tensor) else np.asarray(y_tr, dtype=np.float32)
    )
    seed_u32, lane_hashes = _block256_table_seed_u32(x_np, y_np)

    lm48_b = lm48.unsqueeze(0) if lm48.ndim == 1 else lm48
    lm48_zb = lm48_z.unsqueeze(0) if lm48_z.ndim == 1 else lm48_z
    topk = precompute_champion_topk_batched_gpu(
        lm48_b,
        lm48_zb,
        champ_lm48_z,
        champ_t_norms,
        champ_lm12_128,
        champ_shas=champ_shas,
    )

    embs881_np = (
        champ_embs_881.detach().cpu().numpy()
        if isinstance(champ_embs_881, torch.Tensor)
        else np.asarray(champ_embs_881, dtype=np.int32)
    )
    embs128_np = (
        champ_embs_128.detach().cpu().numpy()
        if isinstance(champ_embs_128, torch.Tensor)
        else np.asarray(champ_embs_128, dtype=np.int32)
    )

    n_top_templates = min(23, int(embs881_np.shape[0]))
    top23_t = topk["top_t_indices_881"][0, :n_top_templates].detach().cpu().numpy()
    top4_c128 = topk["top4_champ128_indices"][0, :4].detach().cpu().numpy()

    # 92 template candidates (23 templates x 4 ranks) + 4 champion-128 candidates = 96
    embs_from_881 = embs881_np[top23_t].reshape(-1, 24)
    embs_from_128 = embs128_np[top4_c128].reshape(-1, 24)
    slate_embs = np.concatenate([embs_from_881, embs_from_128], axis=0).astype(np.int32)
    slate_keys = _fnv1a_embs_hash_u64(slate_embs)

    if sorted_keys is None:
        sorted_keys_np = np.sort(np.unique(slate_keys))
    else:
        sorted_keys_np = (
            sorted_keys.detach().cpu().numpy().astype(np.uint64)
            if isinstance(sorted_keys, torch.Tensor)
            else np.asarray(sorted_keys, dtype=np.uint64)
        )

    pos = np.searchsorted(sorted_keys_np, slate_keys)
    in_bounds = pos < sorted_keys_np.shape[0]
    safe_pos = np.where(in_bounds, pos, 0)
    matched = in_bounds & (sorted_keys_np[safe_pos] == slate_keys)
    slate_key_idx = np.where(matched, pos, -1).astype(np.int32)

    return {
        "table_seed_u32": seed_u32,
        "lane_hashes_u64": lane_hashes,
        "top23_template_indices": top23_t,
        "top4_champ128_indices": top4_c128,
        "slate_embs": slate_embs,
        "slate_keys_u64": slate_keys,
        "slate_key_idx": slate_key_idx,
        "cuda_kernel_source": BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
    }
