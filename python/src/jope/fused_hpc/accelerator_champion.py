from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch

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
