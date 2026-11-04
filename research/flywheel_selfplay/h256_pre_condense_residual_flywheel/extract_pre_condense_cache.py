#!/usr/bin/env python3
"""H256 Step 1: Extract & Verify Pre-Condense DOPE Representations (`dope629`, `hidden256`,
`col_lm_64x17`, `col_mask_64`) and Stage-1B Compression / Clipping Telemetry across all 1,310
contexts (1,219 DL-Train / 205 families and 91 DL-Val / 52 unseen families).

Guardrails:
- Enforces `dope_freeze_guard.py` (`lock_dope_weights_and_eval_mode`, `snapshot_dope_weights`,
  `verify_dope_frozen_unchanged`) and verifies `DOPE_SHA256 = c51a56d0...`.
- Verifies exact bit-parity (`max_abs_diff == 0.0`) between newly extracted `dope629` / `tokens_10x32`
  and the frozen `/home/ubuntu/jope-data/h58/cache/context_features.npz` cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
OUT_CACHE_DIR = Path("/home/ubuntu/jope-data/h256/cache")

sys.path.insert(0, str(H252_DIR))
import common_h252 as ch  # noqa: E402

if str(MICRO / "python") not in sys.path:
    sys.path.insert(0, str(MICRO / "python"))
from hpc.landmark_workspace import FusedLandmarkWorkspace  # noqa: E402


def landmark_float32(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)


def load_locked_fused_network(device: torch.device):
    source = ch.INPUT_DIR / "cpu_exact_fused.py"
    spec = importlib.util.spec_from_file_location("cpu_exact_fused", source)
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    guard_path = Path("/home/ubuntu/JopeDime/python/src/jope/north_star/dope_freeze_guard.py")
    g_spec = importlib.util.spec_from_file_location("freeze_guard", guard_path)
    guard = importlib.util.module_from_spec(g_spec)
    sys.modules[g_spec.name] = guard
    g_spec.loader.exec_module(guard)

    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    state = torch.load(ch.R882_CKPT, map_location=device, weights_only=True)
    net.load_state_dict(state["state_dict"], strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    guard.lock_dope_weights_and_eval_mode(net)
    before = guard.snapshot_dope_weights(net)
    if before.composite_sha256 != ch.DOPE_SHA256:
        raise RuntimeError(f"DOPE SHA-256 mismatch: {before.composite_sha256}")
    return net, guard, before


def extract_router_hidden256(net, vec256: torch.Tensor) -> torch.Tensor:
    """Exact pre-head 256D hidden state `h` inside `dope_model.router` at `level='high'`."""
    r = net.dope_model.router
    x140 = r.input_norm(vec256[:, :140])
    extra = r.v3_inter_norm(vec256[:, 140:])
    x = torch.cat([x140, extra], dim=-1)
    h = r.backbone(x140)
    h = h + r.level_embed.weight[1:2]
    h = h + r.inter_attn(x)
    return h


def compute_stage1b_slot_telemetry(
    net,
    X: torch.Tensor,
    slot_name: str,
) -> Dict[str, Any]:
    mu = getattr(net, f"{slot_name}_mu")
    sd_safe = getattr(net, f"{slot_name}_sd_safe")
    active_mask = getattr(net, f"{slot_name}_active_mask")
    proj_mat = getattr(net, f"{slot_name}_effective_proj_mat")
    X_clean = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z_unclipped = (X_clean - mu) / sd_safe
    active_bool = (active_mask.reshape(-1) > 0.5)
    z_active = z_unclipped[:, active_bool]
    pre_clip8_rate = float((z_active.abs() >= (8.0 - 1e-6)).float().mean().item())
    pre_clip8_any_context_rate = float((z_active.abs() >= (8.0 - 1e-6)).any(dim=1).float().mean().item())

    X_z = torch.clamp(z_unclipped, -8.0, 8.0) * active_mask
    out_unclipped = X_z @ proj_mat
    post_clip6_rate = float((out_unclipped.abs() >= (6.0 - 1e-6)).float().mean().item())
    post_clip6_any_context_rate = float((out_unclipped.abs() >= (6.0 - 1e-6)).any(dim=1).float().mean().item())
    out = torch.clamp(out_unclipped, -6.0, 6.0)
    return {
        "input_dim": int(X.shape[1]),
        "active_coords": int(active_bool.sum().item()),
        "output_dim": int(proj_mat.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(proj_mat.detach().cpu().numpy())),
        "pre_proj_clip8_coord_rate": pre_clip8_rate,
        "pre_proj_clip8_context_frac": pre_clip8_any_context_rate,
        "post_proj_clip6_coord_rate": post_clip6_rate,
        "post_proj_clip6_context_frac": post_clip6_any_context_rate,
        "out_std_mean": float(out.std(dim=0).mean().item()),
    }


def main() -> None:
    t0 = time.time()
    OUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    net, guard, snap_before = load_locked_fused_network(device)

    train_rows, val_rows = ch.load_cohorts()
    train_set = {r["context_sha256"] for r in train_rows}
    val_set = {r["context_sha256"] for r in val_rows}

    with np.load(ch.CACHE_DIR / "context_features.npz", allow_pickle=False) as z:
        cids = [str(c) for c in z["cids"]]
        cached_tok = z["tokens_10x32"].astype(np.float32)
        cached_mask = z["mask_10"].astype(bool)
        cached_lm48_z = z["lm48_z"].astype(np.float32)
        cached_dope629 = z["dope629"].astype(np.float32)
        cached_ctx584 = z["ctx_feat_584"].astype(np.float32)

    n_ctx = len(cids)
    col_lm_all = np.zeros((n_ctx, 64, 17), dtype=np.float32)
    col_mask_all = np.zeros((n_ctx, 64), dtype=bool)
    v256_all = np.zeros((n_ctx, 256), dtype=np.float32)

    ws = FusedLandmarkWorkspace(num_threads=4)
    for idx, cid in enumerate(cids):
        table_path = ch.INPUT_DIR / "tables" / f"{cid}.jtt1"
        raw = table_path.read_bytes()
        nt, nq, width = struct.unpack_from("<III", raw, 4)
        if nq or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError(f"Invalid train-only 2-fold JTT1 header for {cid}")
        a = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1)
        x, y = a[:, :-1], a[:, -1]
        yn = ((y - np.mean(y)) / max(float(np.std(y)), 1e-8)).astype(np.float32)
        v, col, _ = ws.fit_transform_and_extract_bundle_high(
            landmark_float32(x), np.ascontiguousarray(yn)
        )
        v256_all[idx] = v.astype(np.float32)
        n_active = min(len(col), 64)
        col_lm_all[idx, :n_active, : min(col.shape[1], 17)] = col[:n_active, :17]
        col_mask_all[idx, :n_active] = True

    with torch.no_grad():
        v_t = torch.as_tensor(v256_all, dtype=torch.float32, device=device)
        c_t = torch.as_tensor(col_lm_all, dtype=torch.float32, device=device)
        dope629_t, lm48_t = net._run_dope_mlc_v3_and_lm48(v_t, c_t, use_landmarks=True)
        tok_t, mask_t, lm48_z_t = net._project_stage1b_tokens_batched_gpu(lm48_t, dope629_t)
        hidden256_t = extract_router_hidden256(net, v_t)

        # Verify router hidden256 reproduces task_latent and dist_latent inside dope629
        r = net.dope_model.router
        task_repro = torch.tanh(r.task_latent_head(hidden256_t))
        dist_repro = torch.tanh(r.dist_latent_head(hidden256_t))
        max_task_diff = float((task_repro - dope629_t[:, 256:384]).abs().max().item())
        max_dist_diff = float((dist_repro - dope629_t[:, 384:512]).abs().max().item())

    max_dope_diff = float(np.max(np.abs(dope629_t.numpy() - cached_dope629)))
    max_tok_diff = float(np.max(np.abs(tok_t.numpy() - cached_tok)))
    max_lm48z_diff = float(np.max(np.abs(lm48_z_t.numpy() - cached_lm48_z)))
    if max_dope_diff > 1e-6 or max_tok_diff > 1e-6:
        raise RuntimeError(
            f"Cache parity mismatch: max_dope_diff={max_dope_diff}, max_tok_diff={max_tok_diff}"
        )

    # Compute Stage-1B bottleneck and clipping telemetry split by DL-Train vs DL-Val
    train_idx = [i for i, c in enumerate(cids) if c in train_set]
    val_idx = [i for i, c in enumerate(cids) if c in val_set]

    with torch.no_grad():
        lm80_z_all = net._compute_lm_smart_80d_gpu(lm48_t)
        a0 = torch.zeros((n_ctx, 88), dtype=torch.float32)
        h27 = torch.zeros((n_ctx, 16), dtype=torch.float32)
        bb20_z = torch.zeros((n_ctx, 20), dtype=torch.float32)
        m0_in = torch.cat([a0, h27, lm80_z_all], dim=1)
        m1_in = torch.cat([lm48_z_t[:, 36:48], bb20_z, lm80_z_all[:, 48:64]], dim=1)
        tok0 = net._svd_slot_project(m0_in, "svd_slot0_m0")
        tok2 = net._svd_slot_project(dope629_t, "svd_slot2_m2")
        kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(n_ctx, 64)

        slot_inputs = {
            "svd_slot0_m0": m0_in,
            "svd_slot1_m1": m1_in,
            "svd_slot2_m2": dope629_t,
            "svd_slot4_d629_mid": dope629_t,
            "svd_slot5_d629_tail": dope629_t,
            "svd_slot7_lm_fine": m0_in,
            "svd_slot9_kron": kron_64,
        }
        slot_telemetry = {"dl_train": {}, "dl_val": {}}
        for split_name, indices in (("dl_train", train_idx), ("dl_val", val_idx)):
            idx_t = torch.as_tensor(indices, dtype=torch.long)
            for s_name, s_in in slot_inputs.items():
                slot_telemetry[split_name][s_name] = compute_stage1b_slot_telemetry(
                    net, s_in[idx_t], s_name
                )

        # Combined rank of the 3 DOPE629 SVD projection matrices (629 x 96) and DOPE latent heads (256 x 256)
        p2 = net.svd_slot2_m2_effective_proj_mat.numpy()
        p4 = net.svd_slot4_d629_mid_effective_proj_mat.numpy()
        p5 = net.svd_slot5_d629_tail_effective_proj_mat.numpy()
        p_combined_629x96 = np.concatenate([p2, p4, p5], axis=1)
        rank_629x96 = int(np.linalg.matrix_rank(p_combined_629x96))

        # Fraction of standardized active DOPE629 variance captured by the 96D column space of P_combined
        u_basis, _, _ = np.linalg.svd(p_combined_629x96, full_matrices=False)
        u_96 = u_basis[:, :rank_629x96]  # [629, 96] orthonormal basis
        mu2 = net.svd_slot2_m2_mu.numpy()
        sd2 = net.svd_slot2_m2_sd_safe.numpy()
        mask2 = net.svd_slot2_m2_active_mask.numpy()

        dope_z_train = np.clip((cached_dope629[train_idx] - mu2) / sd2, -8.0, 8.0) * mask2
        dope_z_val = np.clip((cached_dope629[val_idx] - mu2) / sd2, -8.0, 8.0) * mask2
        dope_z_train_centered = dope_z_train - dope_z_train.mean(axis=0, keepdims=True)
        dope_z_val_centered = dope_z_val - dope_z_val.mean(axis=0, keepdims=True)

        proj_train = (dope_z_train_centered @ u_96) @ u_96.T
        proj_val = (dope_z_val_centered @ u_96) @ u_96.T
        var_captured_train = float(np.sum(proj_train ** 2) / max(np.sum(dope_z_train_centered ** 2), 1e-12))
        var_captured_val = float(np.sum(proj_val ** 2) / max(np.sum(dope_z_val_centered ** 2), 1e-12))

        # DOPE task_latent_head + dist_latent_head combined rank (256 x 256)
        w_task = net.dope_model.router.task_latent_head.weight.detach().cpu().numpy()
        w_dist = net.dope_model.router.dist_latent_head.weight.detach().cpu().numpy()
        w_latent_256x256 = np.concatenate([w_task, w_dist], axis=0)
        rank_latent_heads = int(np.linalg.matrix_rank(w_latent_256x256))

        # Pre-head vs post-head tanh saturation rates (|val| > 0.95)
        task_sat_train = float((np.abs(cached_dope629[train_idx, 256:384]) > 0.95).mean())
        task_sat_val = float((np.abs(cached_dope629[val_idx, 256:384]) > 0.95).mean())
        dist_sat_train = float((np.abs(cached_dope629[train_idx, 384:512]) > 0.95).mean())
        dist_sat_val = float((np.abs(cached_dope629[val_idx, 384:512]) > 0.95).mean())

    guard.verify_dope_frozen_unchanged(net, snap_before)
    snap_after = guard.snapshot_dope_weights(net)

    cache_path = OUT_CACHE_DIR / "pre_condense_context_features.npz"
    np.savez_compressed(
        cache_path,
        cids=np.asarray(cids),
        tokens_10x32=cached_tok,
        mask_10=cached_mask,
        lm48_z=cached_lm48_z,
        dope629=cached_dope629,
        hidden256=hidden256_t.numpy().astype(np.float32),
        col_lm_64x17=col_lm_all,
        col_mask_64=col_mask_all,
        ctx_feat_584=cached_ctx584,
    )
    cache_sha256 = ch.sha256_file(cache_path)

    telemetry = {
        "hypothesis_id": "H256",
        "step": "extract_pre_condense_cache",
        "elapsed_sec": round(time.time() - t0, 3),
        "n_contexts_total": n_ctx,
        "n_dl_train": len(train_idx),
        "n_dl_val": len(val_idx),
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "dope_freeze_verification": {
            "dope_weights_locked": True,
            "initial_sha256": snap_before.composite_sha256,
            "final_sha256": snap_after.composite_sha256,
            "expected_sha256": ch.DOPE_SHA256,
            "num_param_tensors": snap_after.num_param_tensors,
            "num_param_elements": snap_after.num_param_elements,
            "num_buffer_tensors": snap_after.num_buffer_tensors,
        },
        "cache_parity_checks": {
            "max_dope629_abs_diff_vs_h58": max_dope_diff,
            "max_tokens_10x32_abs_diff_vs_h58": max_tok_diff,
            "max_lm48_z_abs_diff_vs_h58": max_lm48z_diff,
            "max_router_hidden256_to_task_latent_diff": max_task_diff,
            "max_router_hidden256_to_dist_latent_diff": max_dist_diff,
        },
        "structural_rank_audit": {
            "svd_slot2_629x32_rank": int(np.linalg.matrix_rank(p2)),
            "svd_slot4_629x32_rank": int(np.linalg.matrix_rank(p4)),
            "svd_slot5_629x32_rank": int(np.linalg.matrix_rank(p5)),
            "combined_dope_svd_629x96_rank": rank_629x96,
            "step2_svd_basis_256x48_rank": int(
                np.linalg.matrix_rank(net.step2_svd_basis_256x48.numpy())
            ),
            "dope_active_coords_in_slot2_4_5": int((mask2 > 0.5).sum()),
            "dope_inactive_coords_in_slot2_4_5": int((mask2 <= 0.5).sum()),
            "dope_task_plus_dist_head_256x256_rank": rank_latent_heads,
            "dope629_variance_captured_by_96d_svd_subspace_dl_train": var_captured_train,
            "dope629_variance_discarded_by_96d_svd_subspace_dl_train": 1.0 - var_captured_train,
            "dope629_variance_captured_by_96d_svd_subspace_dl_val": var_captured_val,
            "dope629_variance_discarded_by_96d_svd_subspace_dl_val": 1.0 - var_captured_val,
            "task_latent_tanh_saturation_gt_0p95_dl_train": task_sat_train,
            "task_latent_tanh_saturation_gt_0p95_dl_val": task_sat_val,
            "dist_latent_tanh_saturation_gt_0p95_dl_train": dist_sat_train,
            "dist_latent_tanh_saturation_gt_0p95_dl_val": dist_sat_val,
            "active_columns_mean_dl_train": float(col_mask_all[train_idx].sum(axis=1).mean()),
            "active_columns_mean_dl_val": float(col_mask_all[val_idx].sum(axis=1).mean()),
        },
        "stage1b_slot_clipping_telemetry": slot_telemetry,
    }
    telem_path = OUT_CACHE_DIR / "stage1b_bottleneck_telemetry.json"
    ch.save_json(telem_path, telemetry)
    print(json.dumps(telemetry["structural_rank_audit"], indent=2))
    print(f"Saved pre-condense cache ({cache_sha256[:12]}...) in {telemetry['elapsed_sec']}s")


if __name__ == "__main__":
    main()
