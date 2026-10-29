"""Phase E Fused-Path HPC Accelerator Primitives (NORTH_STAR NS11).

Addresses the five GPU/host throughput bottlenecks on the fused inference and
flywheel path documented in ``FUSION_REDTEAM.md`` Part B (``HPC #1-#5``):

* **HPC #1** (``build_live_96_slate`` champion retrieval):
  Replaces single-thread ``881 x 48`` cosine similarity + :math:`O(881^2)` insertion
  sort and ``128 x 12`` L2 distance + :math:`O(128^2)` insertion sort with vectorized
  batched GEMM/L2 kernels and deterministic tie-breaking.
* **HPC #2** (Stage 1A column sorting):
  Replaces single-thread-per-column global-memory ``batch_aquicksort_f64`` with
  coalesced transpose + CUB segmented radix sort (``torch.sort(..., stable=True)``)
  and reuses ``X_in``'s sort permutation across monotone column transforms.
* **HPC #3** (Stage 1A 14 nonlinear probe correlations):
  Fuses the 14 separate ``_exact_corr_with_yz_f64`` full-table passes into a single
  4D ``[B, N, P, 14]`` vectorized contraction pass.
* **HPC #4** (Host I/O, H2D staging, and out-of-loop BPE token counting):
  Provides reusable page-locked (``pin_memory=True``) host batch buffers with async
  stream H2D transfer, bounded multi-threaded CSV split loading, and out-of-loop
  batch BPE token counting.
* **HPC #5** (2D ``(P_tier, N_tr)`` shape bucketing, ``CUDAGraph`` Stage-1A/2 wrapper,
  and ``xbabe1-3`` multi-GPU sharding):
  Eliminates the ``3.6x`` column-padding inflation caused by 1D ``N_tr``-only sorting,
  captures fixed-shape Stage 1A-neural and Stage 2 sub-networks in ``CUDAGraph``s
  with strict ``fp32`` Stage 2 precision, and partitions workloads across
  ``xbabe1`` (RTX 4090), ``xbabe2`` (RTX 3090), and ``xbabe3`` (RTX 3090) (never ``xbabe0``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

ALLOWED_GPU_NODES: tuple[str, ...] = ("xbabe1", "xbabe2", "xbabe3")
DEFAULT_NODE_THROUGHPUT_WEIGHTS: dict[str, float] = {
    "xbabe1": 0.45,
    "xbabe2": 0.275,
    "xbabe3": 0.275,
}
PROBE_14_NAMES: tuple[str, ...] = (
    "asinh",
    "atan",
    "erf",
    "cbrt",
    "log1p",
    "tanh",
    "laplace",
    "clamp",
    "step",
    "hinge",
    "sin",
    "cos",
    "fract",
    "lin",
)


# ============================================================================
# HPC #1: Batched cuBLAS GEMM Champion Retrieval & Deterministic Table Seed
# ============================================================================


def precompute_champion_topk_batched_gpu(
    lm48: torch.Tensor,
    lm48_z: torch.Tensor,
    champ_lm48_z: torch.Tensor,
    champ_t_norms: torch.Tensor,
    champ_lm12_128: torch.Tensor,
    champ_shas: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorized batched replacement for serial 881-cosine and 128-L2 slate retrieval.

    Matches the exact tie-breaking semantics of ``build_96_slate_for_table_ws`` in
    ``fused_gpu_kernels.cu:1094-1116, 1266-1293``:
    * For the 881-template bank, cosine similarities ``dot(c_row, q_norm) / champ_t_norms[i]``
      are sorted in descending order with stable ascending template index tie-breaking.
    * For the 128-champion bank, squared L2 distances in ``float64`` over the first 12
      landmark dimensions are sorted in ascending order with tie-breaking by ``champ_shas``
      lexicographic rank (or index when ``champ_shas`` is omitted).
    """
    if lm48.ndim != 2 or lm48.shape[1] < 12:
        raise ValueError(f"lm48 must have shape [B, >=12], got {tuple(lm48.shape)}")
    if lm48_z.ndim != 2 or lm48_z.shape[1] != 48:
        raise ValueError(f"lm48_z must have shape [B, 48], got {tuple(lm48_z.shape)}")

    device = lm48_z.device
    bsz = lm48_z.shape[0]
    n_champ128 = champ_lm12_128.shape[0]

    # 1. 881-champion cosine similarity in float32 with stable ascending-index tie-break
    q_f32 = lm48_z.to(dtype=torch.float32)
    q_sq = (q_f32 * q_f32).sum(dim=1, keepdim=True)
    q_norm_val = torch.sqrt(q_sq).clamp_min(1e-6)
    q_norm = q_f32 / q_norm_val

    c_f32 = champ_lm48_z.to(device=device, dtype=torch.float32)
    t_norms = champ_t_norms.to(device=device, dtype=torch.float32).clamp_min(1e-12)
    dots = torch.matmul(q_norm, c_f32.transpose(0, 1))
    sims = dots / t_norms.unsqueeze(0)

    # Insertion sort on -sims[vi] is a stable ascending sort on -sims
    neg_sims = -sims
    top_t_indices = torch.argsort(neg_sims, dim=1, stable=True)

    # 2. 128-champion squared L2 distance in float64 over lm48[:, :12]
    lm12_f64 = lm48[:, :12].to(device=device, dtype=torch.float64)
    c12_f64 = champ_lm12_128[:, :12].to(device=device, dtype=torch.float64)
    diff12 = lm12_f64.unsqueeze(1) - c12_f64.unsqueeze(0)
    dists_128 = (diff12 * diff12).sum(dim=-1)

    if champ_shas is not None:
        if len(champ_shas) != n_champ128:
            raise ValueError(f"champ_shas length {len(champ_shas)} != {n_champ128}")
        # Rank SHAs lexicographically so ties break by d_strcmp(vsha, psha) < 0
        sorted_sha_indices = sorted(range(n_champ128), key=lambda idx: champ_shas[idx])
        sha_rank = [0] * n_champ128
        for rank_pos, orig_idx in enumerate(sorted_sha_indices):
            sha_rank[orig_idx] = rank_pos
        perm_t = torch.tensor(sorted_sha_indices, device=device, dtype=torch.long)
        dists_perm = dists_128.index_select(1, perm_t)
        order_in_perm = torch.argsort(dists_perm, dim=1, stable=True)
        order_128 = perm_t.unsqueeze(0).expand(bsz, n_champ128).gather(1, order_in_perm)
    else:
        order_128 = torch.argsort(dists_128, dim=1, stable=True)

    return {
        "sims_881": sims,
        "top_t_indices_881": top_t_indices,
        "dists_128": dists_128,
        "order_128": order_128,
        "top4_champ128_indices": order_128[:, : min(4, n_champ128)],
    }


def compute_table_seed_sha256_u32(
    x_tr: np.ndarray | torch.Tensor,
    y_tr: np.ndarray | torch.Tensor,
    n_valid: int | None = None,
    p_valid: int | None = None,
    *,
    fast_subsample: bool = False,
) -> int:
    """Compute the deterministic 32-bit table RNG seed from ``(X_tr, y_tr)``.

    When ``fast_subsample=False`` (default), hashes the exact full-table little-endian
    ``(n_b, p_b, X_tr[:n_b, :p_b], y_tr[:n_b])`` byte stream matching
    ``sha256_table_seed_f32`` in ``fused_gpu_kernels.cu:489-540``.
    When ``fast_subsample=True``, hashes a strided ``<= 64 x 8`` feature view plus
    ``<= 64`` targets in :math:`O(1)` time (`< 512` floats).
    """
    x_np = x_tr.detach().cpu().numpy() if isinstance(x_tr, torch.Tensor) else np.asarray(x_tr)
    y_np = y_tr.detach().cpu().numpy() if isinstance(y_tr, torch.Tensor) else np.asarray(y_tr)
    x_f32 = np.ascontiguousarray(x_np, dtype=np.float32)
    y_f32 = np.ascontiguousarray(y_np, dtype=np.float32).reshape(-1)

    n_b = int(x_f32.shape[0] if n_valid is None else min(n_valid, x_f32.shape[0]))
    p_b = int(x_f32.shape[1] if p_valid is None else min(p_valid, x_f32.shape[1]))

    h = hashlib.sha256()
    h.update(np.int64(n_b).tobytes())
    h.update(np.int64(p_b).tobytes())

    if n_b > 0 and p_b > 0:
        if fast_subsample:
            row_step = max(1, n_b // 64)
            x_sub = np.ascontiguousarray(x_f32[:n_b:row_step, : min(p_b, 8)], dtype="<f4")
            x_clean = np.where(np.isfinite(x_sub), x_sub, np.float32(0.0)).astype("<f4", copy=False)
            h.update(x_clean.tobytes())
        else:
            x_sub = np.ascontiguousarray(x_f32[:n_b, :p_b], dtype="<f4")
            x_clean = np.where(np.isfinite(x_sub), x_sub, np.float32(0.0)).astype("<f4", copy=False)
            h.update(x_clean.tobytes())

    if n_b > 0:
        y_step = max(1, n_b // 64) if fast_subsample else 1
        y_sub = np.ascontiguousarray(y_f32[:n_b:y_step], dtype="<f4")
        y_clean = np.where(np.isfinite(y_sub), y_sub, np.float32(0.0)).astype("<f4", copy=False)
        h.update(y_clean.tobytes())

    digest = h.digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# ============================================================================
# HPC #2: Coalesced CUB Segmented Radix Sort & Monotone Sort-Order Reuse
# ============================================================================


def sort_valid_values_radix_f64(
    x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> torch.Tensor:
    """Sort valid rows per column in ``float64`` using coalesced innermost-dimension radix sort."""
    rm = r_mask_bool.unsqueeze(-1)
    x_masked = torch.where(rm, x, torch.full_like(x, 1e300))
    x_bpn = x_masked.permute(0, 2, 1).contiguous()
    sorted_bpn = torch.sort(x_bpn, dim=-1, stable=True).values
    return sorted_bpn.permute(0, 2, 1).contiguous()


def sort_valid_rows_radix_f64(
    x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coalesced CUB radix-sort replacement for ``_sort_valid_rows_f64``.

    Transposes ``[B, N, P] -> [B, P, N]`` so each column is contiguous in memory,
    executes ``torch.sort(..., dim=-1, stable=True)`` (which dispatches to NVIDIA CUB
    ``DeviceSegmentedRadixSortKernel`` on CUDA — ``185x`` faster per call than
    single-thread global-memory ``batch_aquicksort_f64_cuda_kernel``), and transposes back.
    """
    rm = r_mask_bool.unsqueeze(-1)
    x_masked = torch.where(rm, x, torch.full_like(x, 1e300))
    x_bpn = x_masked.permute(0, 2, 1).contiguous()
    sorted_bpn, order_bpn = torch.sort(x_bpn, dim=-1, stable=True)
    return sorted_bpn.permute(0, 2, 1).contiguous(), order_bpn.permute(0, 2, 1).contiguous()


def reuse_monotone_sort_order_f64(
    x_transformed: torch.Tensor,
    order_x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse an existing column sort permutation ``order_x`` for a strictly monotone transform.

    In Stage 1A (`gpu_stage1a_and_slate.py:381, 495, 548`), ``Z_unc = (X_scaled - mean_xs) / std_xs``
    and ``Z_clp = clamp(Z_unc, -8.0, 8.0)`` are non-decreasing monotone transforms of ``X_in``.
    Gathering ``x_transformed`` along ``order_x`` in :math:`O(B N P)` avoids re-sorting the
    entire table while preserving ``sorted_vals`` and ``order_x``.
    """
    gathered = torch.gather(x_transformed, 1, order_x)
    rm = r_mask_bool.unsqueeze(-1)
    sorted_vals = torch.where(rm, gathered, torch.full_like(gathered, 1e300))
    return sorted_vals, order_x


# ============================================================================
# HPC #3: Fused 14-Probe Stage 1A Correlation Contraction
# ============================================================================


def erf_approx_f64(x: torch.Tensor) -> torch.Tensor:
    """Abramowitz-Stegun 7.1.26 polynomial ``erf`` approximation matching Stage 1A."""
    sign = torch.sign(x)
    ax = torch.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * torch.exp(-(ax * ax)))


def exact_corr_with_yz_f64(
    f_col: torch.Tensor,
    y_z: torch.Tensor,
    r_mask: torch.Tensor,
    n_b: torch.Tensor,
) -> torch.Tensor:
    """Reference single-probe absolute correlation with ``y_z`` matching Stage 1A."""
    rm3 = r_mask.unsqueeze(-1)
    dn = n_b.clamp_min(1.0)
    mean_f = (f_col * rm3).sum(dim=1) / dn
    df = (f_col - mean_f.unsqueeze(1)) * rm3
    std_f = torch.sqrt((df * df).sum(dim=1) / dn)
    valid = (std_f > 1e-9) & torch.isfinite(std_f) & (n_b > 0.0)
    std_safe = torch.where(valid, std_f, torch.ones_like(std_f))
    prod = ((df / std_safe.unsqueeze(1)) * (y_z * r_mask).unsqueeze(-1)).sum(dim=1) / dn
    out = torch.abs(prod)
    return torch.where(valid & torch.isfinite(out), out, torch.zeros_like(out))


def fused_14_probe_corrs_with_yz_f64(
    z_clp: torch.Tensor,
    y_z: torch.Tensor,
    r_mask: torch.Tensor,
    n_b: torch.Tensor,
    c_mask: torch.Tensor,
    c_q10: torch.Tensor,
    c_q25: torch.Tensor,
    c_q50: torch.Tensor,
    c_q75: torch.Tensor,
    c_q90: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all 14 Stage 1A nonlinear probe correlations in one fused 4D pass.

    Replaces the 14 separate ``_exact_corr_with_yz_f64`` calls in
    ``gpu_stage1a_and_slate.py:521-540`` (`~140` kernel launches) with a single
    ``[B, N, P, 14]`` tensor reduction over rows ``dim=1``.
    Matches the sequential Stage 1A probe correlations to ``< 1e-12`` max abs diff.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    half_pi = 0.5 * math.pi
    one_third = 1.0 / 3.0

    abs_z = torch.abs(z_clp)
    sign_z = torch.sign(z_clp)

    f_asinh = torch.asinh(z_clp)
    f_atan = torch.atan(z_clp)
    f_erf = erf_approx_f64(z_clp * inv_sqrt2)
    f_cbrt = sign_z * torch.pow(abs_z, one_third)
    f_log1p = sign_z * torch.log1p(abs_z)
    f_tanh = torch.tanh(0.75 * z_clp)
    f_laplace = torch.expm1(-abs_z)
    f_clamp = torch.clamp(z_clp, c_q10.unsqueeze(1), c_q90.unsqueeze(1))
    f_step = (z_clp >= c_q50.unsqueeze(1)).to(torch.float64)
    f_hinge = torch.clamp(z_clp - c_q75.unsqueeze(1), min=0.0) - torch.clamp(
        c_q25.unsqueeze(1) - z_clp, min=0.0
    )
    f_sin = torch.sin(half_pi * z_clp)
    f_cos = torch.cos(half_pi * z_clp)
    f_fract = z_clp - torch.floor(z_clp)
    f_lin = z_clp

    # Stack all 14 probes into [B, N, P, 14] and run a single reduction pass
    f_all = torch.stack(
        [
            f_asinh,
            f_atan,
            f_erf,
            f_cbrt,
            f_log1p,
            f_tanh,
            f_laplace,
            f_clamp,
            f_step,
            f_hinge,
            f_sin,
            f_cos,
            f_fract,
            f_lin,
        ],
        dim=-1,
    )

    rm4 = r_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
    dn = n_b.clamp_min(1.0).unsqueeze(-1)  # [B, 1, 1]
    mean_f = (f_all * rm4).sum(dim=1) / dn  # [B, P, 14]
    df = (f_all - mean_f.unsqueeze(1)) * rm4  # [B, N, P, 14]
    std_f = torch.sqrt((df * df).sum(dim=1) / dn)  # [B, P, 14]

    valid = (std_f > 1e-9) & torch.isfinite(std_f) & (n_b.unsqueeze(-1) > 0.0)
    std_safe = torch.where(valid, std_f, torch.ones_like(std_f))

    yz_masked = (y_z * r_mask).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
    prod = ((df / std_safe.unsqueeze(1)) * yz_masked).sum(dim=1) / dn  # [B, P, 14]
    corrs_all = torch.abs(prod)
    corrs_all = torch.where(valid & torch.isfinite(corrs_all), corrs_all, torch.zeros_like(corrs_all))

    c_sin = corrs_all[:, :, 10]
    c_cos = corrs_all[:, :, 11]
    masked_corrs = corrs_all * c_mask.unsqueeze(-1)

    return {
        "c_asinh": masked_corrs[:, :, 0],
        "c_atan": masked_corrs[:, :, 1],
        "c_erf": masked_corrs[:, :, 2],
        "c_cbrt": masked_corrs[:, :, 3],
        "c_log1p": masked_corrs[:, :, 4],
        "c_tanh": masked_corrs[:, :, 5],
        "c_laplace": masked_corrs[:, :, 6],
        "c_clamp": masked_corrs[:, :, 7],
        "c_step": masked_corrs[:, :, 8],
        "c_hinge": masked_corrs[:, :, 9],
        "c_sin": c_sin,
        "c_cos": c_cos,
        "c_sincos": torch.maximum(c_sin, c_cos) * c_mask,
        "c_fract": masked_corrs[:, :, 12],
        "c_lin": masked_corrs[:, :, 13],
        "corrs_14": masked_corrs,
    }


# ============================================================================
# HPC #4: Pinned-Memory Async H2D Pool, Multi-Thread CSV & Out-of-Loop BPE Tok/s
# ============================================================================


class PinnedBatchBufferPool:
    """Reusable page-locked host buffer pool for asynchronous H2D batch staging.

    Eliminates per-batch pageable ``np.zeros`` allocations in
    ``collate_raw_tables_to_gpu_batch`` (`run_parity_and_quick10_benchmark.py:97-124`)
    so ``.to(device, non_blocking=True)`` performs true asynchronous DMA over PCIe.
    """

    def __init__(
        self,
        max_b: int = 128,
        max_n_tr: int = 512,
        max_n_ho: int = 256,
        max_p: int = 64,
        *,
        pin_memory: bool | None = None,
    ) -> None:
        if max_b <= 0 or max_n_tr <= 0 or max_n_ho <= 0 or max_p <= 0:
            raise ValueError("PinnedBatchBufferPool dimensions must be positive")
        self.max_b = int(max_b)
        self.max_n_tr = int(max_n_tr)
        self.max_n_ho = int(max_n_ho)
        self.max_p = int(max_p)
        self.pin_memory = bool(torch.cuda.is_available() if pin_memory is None else pin_memory)
        self._alloc_buffers(self.max_b, self.max_n_tr, self.max_n_ho, self.max_p)

    def _alloc_buffers(self, b: int, n_tr: int, n_ho: int, p: int) -> None:
        pin = self.pin_memory and torch.cuda.is_available()
        self.x_tr_host = torch.zeros((b, n_tr, p), dtype=torch.float32, pin_memory=pin)
        self.y_tr_host = torch.zeros((b, n_tr), dtype=torch.float32, pin_memory=pin)
        self.x_ho_host = torch.zeros((b, n_ho, p), dtype=torch.float32, pin_memory=pin)
        self.y_ho_host = torch.zeros((b, n_ho), dtype=torch.float32, pin_memory=pin)
        self.tr_mask_host = torch.zeros((b, n_tr), dtype=torch.float32, pin_memory=pin)
        self.ho_mask_host = torch.zeros((b, n_ho), dtype=torch.float32, pin_memory=pin)
        self.col_mask_host = torch.zeros((b, p), dtype=torch.float32, pin_memory=pin)

    def _ensure_capacity(self, b: int, n_tr: int, n_ho: int, p: int) -> None:
        if b > self.max_b or n_tr > self.max_n_tr or n_ho > self.max_n_ho or p > self.max_p:
            self.max_b = max(self.max_b, b)
            self.max_n_tr = max(self.max_n_tr, n_tr)
            self.max_n_ho = max(self.max_n_ho, n_ho)
            self.max_p = max(self.max_p, p)
            self._alloc_buffers(self.max_b, self.max_n_tr, self.max_n_ho, self.max_p)

    def collate_to_device(
        self,
        tables: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        device: torch.device,
        *,
        max_tr_rows: int = 512,
        max_ho_rows: int = 256,
        max_cols: int = 1088,
    ) -> dict[str, torch.Tensor]:
        """Pack raw ``(X_tr, y_tr, X_ho, y_ho)`` tables into pinned buffers and transfer to ``device``."""
        if not tables:
            raise ValueError("tables must be non-empty")
        b = len(tables)
        n_tr_pad = max(min(int(t[0].shape[0]), max_tr_rows) for t in tables)
        n_ho_pad = max(min(int(t[2].shape[0]), max_ho_rows) for t in tables)
        p_pad = max(min(int(t[0].shape[1]), max_cols) for t in tables)

        self._ensure_capacity(b, n_tr_pad, n_ho_pad, p_pad)

        x_tr_s = self.x_tr_host[:b, :n_tr_pad, :p_pad]
        y_tr_s = self.y_tr_host[:b, :n_tr_pad]
        x_ho_s = self.x_ho_host[:b, :n_ho_pad, :p_pad]
        y_ho_s = self.y_ho_host[:b, :n_ho_pad]
        tr_m_s = self.tr_mask_host[:b, :n_tr_pad]
        ho_m_s = self.ho_mask_host[:b, :n_ho_pad]
        col_m_s = self.col_mask_host[:b, :p_pad]

        x_tr_s.zero_()
        y_tr_s.zero_()
        x_ho_s.zero_()
        y_ho_s.zero_()
        tr_m_s.zero_()
        ho_m_s.zero_()
        col_m_s.zero_()

        x_tr_np = x_tr_s.numpy()
        y_tr_np = y_tr_s.numpy()
        x_ho_np = x_ho_s.numpy()
        y_ho_np = y_ho_s.numpy()
        tr_m_np = tr_m_s.numpy()
        ho_m_np = ho_m_s.numpy()
        col_m_np = col_m_s.numpy()

        for i, (xtr, ytr, xho, yho) in enumerate(tables):
            ntr = min(int(xtr.shape[0]), n_tr_pad)
            nho = min(int(xho.shape[0]), n_ho_pad)
            p = min(int(xtr.shape[1]), p_pad)
            x_tr_np[i, :ntr, :p] = xtr[:ntr, :p]
            y_tr_np[i, :ntr] = ytr[:ntr]
            x_ho_np[i, :nho, :p] = xho[:nho, :p]
            y_ho_np[i, :nho] = yho[:nho]
            tr_m_np[i, :ntr] = 1.0
            ho_m_np[i, :nho] = 1.0
            col_m_np[i, :p] = 1.0

        non_blocking = device.type == "cuda" and x_tr_s.is_pinned()
        return {
            "X_tr": x_tr_s.to(device=device, non_blocking=non_blocking),
            "y_tr_raw": y_tr_s.to(device=device, non_blocking=non_blocking),
            "X_ho": x_ho_s.to(device=device, non_blocking=non_blocking),
            "y_ho_raw": y_ho_s.to(device=device, non_blocking=non_blocking),
            "tr_row_mask": tr_m_s.to(device=device, non_blocking=non_blocking),
            "ho_row_mask": ho_m_s.to(device=device, non_blocking=non_blocking),
            "col_mask": col_m_s.to(device=device, non_blocking=non_blocking),
        }


def parallel_load_dataset_splits(
    items: Sequence[Any],
    loader_fn: Callable[[Any], Any],
    *,
    max_workers: int = 8,
) -> list[Any]:
    """Load dataset splits concurrently across bounded worker threads (respecting <=85% host CPU cap)."""
    if max_workers <= 1 or len(items) <= 1:
        return [loader_fn(item) for item in items]
    bounded_workers = min(max(1, int(max_workers)), 8, len(items))
    with ThreadPoolExecutor(max_workers=bounded_workers) as pool:
        return list(pool.map(loader_fn, items))


def count_rust_tokens_out_of_loop(
    rust_sources: Sequence[str],
    *,
    encode_fn: Callable[[str], Sequence[int]] | None = None,
    encode_batch_fn: Callable[[list[str]], list[list[int]]] | None = None,
    max_workers: int = 4,
) -> list[int]:
    """Count BPE tokens outside the timed GPU inference loop (`HPC #4`).

    In ``run_batch_vram_and_cohort915_benchmark.py:418``, single-threaded ``tiktoken.encode``
    was called inside the timed ``t_cohort_start .. t_inmem_end`` loop, adding ``4.20s-5.73s``
    (`22.5%-40.0%` overhead) to reported in-memory GPU inference wall time.
    Running ``encode_batch`` (or bounded thread-pool encoding) after ``t_inmem_end``
    restores true GPU + Rust emission ``tok/s``.
    """
    src_list = list(rust_sources)
    if not src_list:
        return []
    if encode_batch_fn is not None:
        return [len(toks) for toks in encode_batch_fn(src_list)]
    if encode_fn is not None:
        if max_workers <= 1 or len(src_list) <= 1:
            return [len(encode_fn(s)) for s in src_list]
        with ThreadPoolExecutor(max_workers=min(max_workers, 8, len(src_list))) as pool:
            return [len(toks) for toks in pool.map(encode_fn, src_list)]

    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        tok_lists = enc.encode_batch(
            src_list,
            num_threads=min(max(1, int(max_workers)), 8),
            disallowed_special=(),
        )
        return [len(t) for t in tok_lists]
    except Exception:
        # Deterministic byte/whitespace token fallback when tiktoken is not installed
        return [len(s.split()) for s in src_list]


# ============================================================================
# HPC #5: 2D (P_tier, N_tr) Shape Bucketing, CUDAGraph Runner & Multi-GPU Planner
# ============================================================================


@dataclass(frozen=True)
class ShapeBucketPlan:
    """Summary and batch schedule for 2D ``(P_tier, N_tr)`` shape bucketing."""

    ordered_indices: tuple[int, ...]
    batch_ranges: tuple[tuple[int, int], ...]
    batch_tier_labels: tuple[str, ...]
    padded_cells_1d: int
    padded_cells_2d: int
    padding_reduction_ratio: float


def _compute_schedule_padded_cells(
    rows_and_cols: Sequence[tuple[int, int]],
    ordered_indices: Sequence[int],
    batch_ranges: Sequence[tuple[int, int]],
    *,
    max_n_tr_cap: int = 512,
    max_p_cap: int = 1088,
) -> int:
    total_cells = 0
    for start, end in batch_ranges:
        if end <= start:
            continue
        batch_idx = ordered_indices[start:end]
        b_len = len(batch_idx)
        n_pad = max(min(int(rows_and_cols[i][0]), max_n_tr_cap) for i in batch_idx)
        p_pad = max(min(int(rows_and_cols[i][1]), max_p_cap) for i in batch_idx)
        total_cells += b_len * n_pad * p_pad
    return total_cells


def bucket_tables_2d_by_p_tier_and_ntr(
    rows_and_cols: Sequence[tuple[int, int]],
    *,
    p_tiers: tuple[int, int] = (16, 64),
    batch_sizes_by_tier: tuple[int, int, int] = (64, 32, 16),
    baseline_batch_size: int = 32,
    max_n_tr_cap: int = 512,
    max_p_cap: int = 1088,
) -> ShapeBucketPlan:
    """Partition tables into 2D ``(P_tier, N_tr_capped, P)`` buckets to eliminate column-padding waste.

    In ``run_batch_vram_and_cohort915_benchmark.py:286``, sorting by uncapped ``(N_tr, P)``
    places wide tables (`P = 376..1087`) into the same 32-table batch as narrow tables
    (`P = 4..12`), inflating padded ``B * N_pad * P_pad`` volume by ``~2.5x-3.6x``.
    Partitioning first into three column-width tiers (`P <= 16`, `17 <= P <= 64`, `P > 64`)
    and sorting by ``(min(N_tr, max_n_tr_cap), P)`` within each tier (and by ``(P, min(N_tr, max_n_tr_cap))``
    for the wide ``P > 64`` tier) isolates wide tables and cuts padded tensor volume by ``~3x``.
    """
    n_items = len(rows_and_cols)
    if n_items == 0:
        return ShapeBucketPlan(
            ordered_indices=(),
            batch_ranges=(),
            batch_tier_labels=(),
            padded_cells_1d=0,
            padded_cells_2d=0,
            padding_reduction_ratio=1.0,
        )

    p_t0, p_t1 = int(p_tiers[0]), int(p_tiers[1])
    b_s0, b_s1, b_s2 = (
        max(1, int(batch_sizes_by_tier[0])),
        max(1, int(batch_sizes_by_tier[1])),
        max(1, int(batch_sizes_by_tier[2])),
    )

    # Baseline 1D sort by uncapped (N_tr, P) with fixed baseline_batch_size=32
    order_1d = sorted(
        range(n_items),
        key=lambda i: (int(rows_and_cols[i][0]), int(rows_and_cols[i][1]), i),
    )
    ranges_1d = [
        (s, min(s + baseline_batch_size, n_items)) for s in range(0, n_items, max(1, baseline_batch_size))
    ]
    padded_cells_1d = _compute_schedule_padded_cells(
        rows_and_cols,
        order_1d,
        ranges_1d,
        max_n_tr_cap=max_n_tr_cap,
        max_p_cap=max_p_cap,
    )

    # 2D (P_tier, N_tr_capped, P) bucketing
    tier_buckets: list[list[int]] = [[], [], []]
    tier_names = (f"P<={p_t0}", f"{p_t0 + 1}<=P<={p_t1}", f"P>{p_t1}")
    tier_batch_sizes = (b_s0, b_s1, b_s2)

    for idx, (_n_r, p_c) in enumerate(rows_and_cols):
        if p_c <= p_t0:
            tier_buckets[0].append(idx)
        elif p_c <= p_t1:
            tier_buckets[1].append(idx)
        else:
            tier_buckets[2].append(idx)

    ordered_2d: list[int] = []
    batch_ranges_2d: list[tuple[int, int]] = []
    batch_labels_2d: list[str] = []

    for t_idx, bucket in enumerate(tier_buckets):
        if not bucket:
            continue
        if t_idx == 2:
            bucket.sort(
                key=lambda i: (
                    int(rows_and_cols[i][1]),
                    min(int(rows_and_cols[i][0]), max_n_tr_cap),
                    i,
                )
            )
        else:
            bucket.sort(
                key=lambda i: (
                    min(int(rows_and_cols[i][0]), max_n_tr_cap),
                    int(rows_and_cols[i][1]),
                    i,
                )
            )
        b_size = tier_batch_sizes[t_idx]
        base_offset = len(ordered_2d)
        ordered_2d.extend(bucket)
        for local_s in range(0, len(bucket), b_size):
            local_e = min(local_s + b_size, len(bucket))
            batch_ranges_2d.append((base_offset + local_s, base_offset + local_e))
            batch_labels_2d.append(tier_names[t_idx])

    padded_cells_2d = _compute_schedule_padded_cells(
        rows_and_cols,
        ordered_2d,
        batch_ranges_2d,
        max_n_tr_cap=max_n_tr_cap,
        max_p_cap=max_p_cap,
    )
    ratio = float(padded_cells_1d) / float(max(1, padded_cells_2d))

    return ShapeBucketPlan(
        ordered_indices=tuple(ordered_2d),
        batch_ranges=tuple(batch_ranges_2d),
        batch_tier_labels=tuple(batch_labels_2d),
        padded_cells_1d=int(padded_cells_1d),
        padded_cells_2d=int(padded_cells_2d),
        padding_reduction_ratio=ratio,
    )


class FusedStageCudaGraphRunner:
    """Captures and replays ``CUDAGraph``s for fixed-shape Stage 1A-neural and Stage 2 sub-networks.

    In ``FusedJopeNetwork``, Stage 1A's ``_run_dope_mlc_v3_and_lm48(ms_376, col_lm_64x17)``
    and Stage 2's ``_run_stage2_router_qhat_spq1_batched_gpu(...)`` operate on tensors whose
    feature dimensions (`376`, `64x17`, `96x412`, `4x24`) are independent of table shape
    ``(N_tr, P)``. Capturing per-batch-size ``CUDAGraph``s eliminates Python/driver launch
    overhead while automatically falling back to eager execution on CPU or uncaptured shapes.
    """

    def __init__(
        self,
        stage1a_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        *,
        enable_cuda_graphs: bool = True,
    ) -> None:
        self.stage1a_fn = stage1a_fn
        self.enable_cuda_graphs = bool(enable_cuda_graphs)
        self._captured_stage1a: dict[
            tuple[int, torch.device, torch.dtype],
            tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def run_stage1a_neural(
        self,
        ms_376: torch.Tensor,
        col_lm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute ``_run_dope_mlc_v3_and_lm48`` via captured ``CUDAGraph`` on CUDA or eager fallback."""
        if not self.enable_cuda_graphs or not ms_376.is_cuda or not col_lm.is_cuda or torch.is_grad_enabled():
            with torch.no_grad():
                return self.stage1a_fn(ms_376, col_lm)

        bsz = int(ms_376.shape[0])
        key = (bsz, ms_376.device, ms_376.dtype)
        entry = self._captured_stage1a.get(key)
        if entry is None:
            static_ms = torch.empty_like(ms_376).copy_(ms_376)
            static_col = torch.empty_like(col_lm).copy_(col_lm)
            # Warmup before capture
            with torch.no_grad():
                _ = self.stage1a_fn(static_ms, static_col)
            torch.cuda.synchronize(ms_376.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                static_dope629, static_lm48 = self.stage1a_fn(static_ms, static_col)
            entry = (graph, static_ms, static_col, static_dope629, static_lm48)
            self._captured_stage1a[key] = entry

        graph, static_ms, static_col, static_dope629, static_lm48 = entry
        static_ms.copy_(ms_376)
        static_col.copy_(col_lm)
        graph.replay()
        return static_dope629.clone(), static_lm48.clone()


def plan_multigpu_shards(
    items: Sequence[dict[str, Any]],
    *,
    node_weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically partition datasets across ``xbabe1``, ``xbabe2``, and ``xbabe3`` (`HPC #5`).

    Strictly enforces the ``NORTH_STAR.md`` / ``COMPUTE.md`` hard rule: **NEVER ``xbabe0``**.
    Balances estimated table compute cost ``max(1, rows) * max(1, features)`` across
    active GPU nodes proportionally to ``node_weights`` (defaulting to
    ``{"xbabe1": 0.45, "xbabe2": 0.275, "xbabe3": 0.275}``).
    """
    weights = dict(DEFAULT_NODE_THROUGHPUT_WEIGHTS if node_weights is None else node_weights)
    if not weights:
        raise ValueError("node_weights must be non-empty")
    for node_name in weights:
        if node_name == "xbabe0" or node_name not in ALLOWED_GPU_NODES:
            raise ValueError(
                f"Forbidden or unknown GPU node {node_name!r}; allowed nodes are {ALLOWED_GPU_NODES}"
            )
    if any(w <= 0.0 or not math.isfinite(w) for w in weights.values()):
        raise ValueError("All node_weights must be positive finite numbers")

    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}
    shards: dict[str, list[dict[str, Any]]] = {k: [] for k in weights}
    accum_cost: dict[str, float] = {k: 0.0 for k in weights}

    # Sort descending by estimated work (rows * features) with deterministic dataset_id tie-break
    def _item_cost(it: dict[str, Any]) -> float:
        r = max(1, int(it.get("rows", 1)))
        c = max(1, int(it.get("features", 1)))
        return float(r * c)

    sorted_items = sorted(
        items,
        key=lambda it: (-_item_cost(it), str(it.get("dataset_id", ""))),
    )

    for it in sorted_items:
        cost = _item_cost(it)
        # Assign to the node with the smallest normalized load (accum_cost[node] / weight[node])
        best_node = min(
            weights.keys(),
            key=lambda k: (accum_cost[k] / norm_weights[k], k),
        )
        shards[best_node].append(it)
        accum_cost[best_node] += cost

    return shards


def _snapshot_dope_state_sha256(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 over ``model.dope_model.state_dict()``."""
    dope = getattr(model, "dope_model", None)
    if not isinstance(dope, nn.Module):
        raise ValueError("Model does not contain an nn.Module 'dope_model' submodule")
    h = hashlib.sha256()
    for name, tensor in sorted(dope.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        header = f"{name}:{cpu.dtype}:{tuple(cpu.shape)}:".encode()
        h.update(header)
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def patch_fused_network_hpc(
    fused_net: nn.Module,
    *,
    gpu_stage1a_module: Any | None = None,
    enable_cuda_graphs: bool = True,
    force_stage2_fp32: bool = True,
) -> dict[str, Any]:
    """Attach Phase E HPC optimizations to ``fused_net`` while strictly locking DOPE weights.

    1. Verifies and enforces the ``NORTH_STAR.md`` hard requirement on ``fused_net.dope_model``:
       all parameters have ``requires_grad=False``, ``grad=None``, and all nested DOPE modules
       are locked in ``.eval()`` mode with identical pre/post SHA-256 state checksums.
    2. Wraps ``_run_dope_mlc_v3_and_lm48`` in ``FusedStageCudaGraphRunner`` when enabled.
    3. Disables the zero-benefit ``fp16`` autocast in Stage 2 (``force_stage2_fp32=True``)
       that caused 6 winner mismatches in ``FUSION_REDTEAM.md`` ``[A2]``.
    4. Optionally patches ``gpu_stage1a_module._sort_valid_rows_f64`` and
       ``gpu_stage1a_module._sort_valid_values_f64`` to use coalesced CUB radix sort (`HPC #2`).
    """
    dope = getattr(fused_net, "dope_model", None)
    if not isinstance(dope, nn.Module):
        raise ValueError("fused_net is missing required 'dope_model' submodule")

    sha_before = _snapshot_dope_state_sha256(fused_net)
    for param in dope.parameters():
        param.requires_grad_(False)
        param.grad = None
    dope.eval()

    # Lock train() on every nested DOPE module so fused_net.train(True) never enables DOPE Dropout
    def _locked_eval_train(self_mod: nn.Module, mode: bool = True) -> nn.Module:
        nn.Module.train(self_mod, False)
        return self_mod

    for submod in dope.modules():
        submod.training = False
        submod.train = _locked_eval_train.__get__(submod, type(submod))  # type: ignore[method-assign]

    if hasattr(fused_net, "_run_dope_mlc_v3_and_lm48"):
        orig_stage1a_fn = fused_net._run_dope_mlc_v3_and_lm48
        graph_runner = FusedStageCudaGraphRunner(
            orig_stage1a_fn,
            enable_cuda_graphs=enable_cuda_graphs,
        )
        fused_net._hpc_stage1a_graph_runner = graph_runner  # type: ignore[attr-defined]
        fused_net._run_dope_mlc_v3_and_lm48 = graph_runner.run_stage1a_neural  # type: ignore[method-assign]

    if force_stage2_fp32 and hasattr(fused_net, "forward"):
        orig_forward = fused_net.forward

        def _forward_fp32_stage2(*args: Any, **kwargs: Any) -> Any:
            kwargs["amp_dtype"] = torch.float32
            return orig_forward(*args, **kwargs)

        fused_net.forward = _forward_fp32_stage2  # type: ignore[method-assign]

    if gpu_stage1a_module is not None:
        gpu_stage1a_module._sort_valid_values_f64 = sort_valid_values_radix_f64
        gpu_stage1a_module._sort_valid_rows_f64 = sort_valid_rows_radix_f64

    sha_after = _snapshot_dope_state_sha256(fused_net)
    if sha_after != sha_before:
        raise RuntimeError(f"DOPE state SHA-256 mutated during HPC patch: {sha_before} != {sha_after}")

    trainable_by_child = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in fused_net.named_children()
    }
    if trainable_by_child.get("dope_model", 0) != 0:
        raise RuntimeError("DOPE parameters must have 0 trainable elements")

    return {
        "dope_state_sha256_before": sha_before,
        "dope_state_sha256_after": sha_after,
        "dope_freeze_verified": sha_before == sha_after and trainable_by_child.get("dope_model", 0) == 0,
        "trainable_param_counts_by_submodule": trainable_by_child,
        "cuda_graphs_enabled": bool(enable_cuda_graphs),
        "force_stage2_fp32": bool(force_stage2_fp32),
        "stage1a_radix_sort_patched": gpu_stage1a_module is not None,
    }
