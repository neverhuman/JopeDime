#!/usr/bin/env python3
"""H256 Thrust 1 (Gate T1): Learning Behavior Audit, Two-Speed Differential LR + Warmup + SWA,
All-Pairs Regret-Weighted Margin Loss, and Semi-Synthetic Column & Row-Subsample Augmentation.

Fixes four concrete training bottlenecks identified in the H256 learning audit:
  1. Two-Speed Differential Learning Rate (`lr_enc=3e-4`, `lr_head=1.8e-3`) + 6-Step Readout Warmup
     with frozen Stage-1 S3 encoder weights so zero-initialized readout heads establish clean
     ranking gradients before fine-tuning pretrained representations.
  2. All-28-Pairs Regret-Weighted Margin Loss (`L_all_pairs`) across all (i, j) pairs in the Top-8
     candidate slate weighted by utility regret |u_i - u_j| and a 1.5x gate-safety penalty, replacing
     single-pair (best_j vs slot 0) margin loss.
  3. Temperature-Annealed ListNet (tau: 1.5 -> 0.65) + Cosine Annealing LR + Stochastic Weight
     Averaging (SWA) over the tail of training.
  4. Semi-Synthetic Column & Row-Subsample Training Augmentation (`3x` training views per DL-Train
     context: View 0 = 100% clean, View 1 = 80% stratified row subsample, View 2 = 80% row subsample +
     semi-synthetic distractor/collinear/heavy-tail/interaction columns) with Multi-View Ranking
     Consistency KL Regularization, plus 11-view (`1x100% + 8x80% + 2xcol-jitter`) batched inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
H256_DIR = HERE.parent / "h256_pre_condense_residual_flywheel"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
H256_OUT_DIR = Path("/home/ubuntu/jope-data/h256")
OUT_DIR = Path("/home/ubuntu/jope-data/h257")
CACHE_DIR = OUT_DIR / "cache"

sys.path.insert(0, str(MICRO / "python"))
sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(H256_DIR))
sys.path.insert(0, str(HERE))

from hpc.landmark_workspace import FusedLandmarkWorkspace  # noqa: E402
import common_h252 as ch  # noqa: E402
import extract_pre_condense_cache as epc  # noqa: E402
import pretrain_full_s3_7split_bridge as p_s3  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
import train_and_export_h256_flagship_production as prod  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402


def generate_perturbed_table_view(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    view_kind: str,
    rng: np.random.Generator,
    subsample_frac: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a deterministic row-subsampled and/or semi-synthetically column-augmented view `(x_view, y_view)`.

    `view_kind`:
      - `"full100"`: 100% rows, original columns `x, y`.
      - `"sub80"`: 80% fold- and target-quantile-stratified random row subsample, original columns.
      - `"semisyn80"`: 80% stratified row subsample + synthetic distractor, collinear duplicate,
        heavy-tailed Student-t3, and nonlinear pairwise interaction columns.
    """
    nt, width = x.shape
    if view_kind == "full100":
        return x, y

    # Fold-stratified random 80% row subsample preserving target deciles
    sel_parts = []
    for f_val in (0, 1):
        f_idx = np.flatnonzero(folds == f_val)
        n_f = len(f_idx)
        n_keep = max(16, min(n_f, int(round(subsample_frac * n_f))))
        if n_keep >= n_f:
            sel_parts.append(f_idx)
        else:
            # Stratified across 8 quantile bins of y within each fold for balanced tail coverage
            y_f = y[f_idx]
            order = np.argsort(y_f, kind="mergesort")
            n_bins = min(8, n_keep)
            bins = np.array_split(order, n_bins)
            chosen_local = []
            base_per_bin = n_keep // n_bins
            rem = n_keep % n_bins
            for b_i, b_arr in enumerate(bins):
                k_b = base_per_bin + (1 if b_i < rem else 0)
                if k_b >= len(b_arr):
                    chosen_local.append(b_arr)
                else:
                    chosen_local.append(rng.choice(b_arr, size=k_b, replace=False))
            chosen = np.sort(f_idx[np.concatenate(chosen_local)])
            sel_parts.append(chosen)
    row_idx = np.sort(np.concatenate(sel_parts))
    x_sub = x[row_idx].copy()
    y_sub = y[row_idx].copy()

    if view_kind == "sub80":
        return x_sub, y_sub

    # Semi-synthetic column injection (`semisyn80`)
    n_sub = len(row_idx)
    col_means = np.mean(x_sub, axis=0, keepdims=True)
    col_stds = np.maximum(np.std(x_sub, axis=0, keepdims=True), 1e-6)
    x_z = np.clip((x_sub - col_means) / col_stds, -8.0, 8.0)

    k_syn = max(2, min(8, int(round(0.25 * width))))
    syn_cols = []
    for s_i in range(k_syn):
        mode = s_i % 4
        if mode == 0:
            # 1. Pure Gaussian distractor column
            c_new = rng.standard_normal(n_sub)
        elif mode == 1:
            # 2. Noisy collinear duplicate of a randomly chosen column
            src_j = int(rng.integers(0, width))
            noise_scale = float(rng.uniform(0.10, 0.25))
            c_new = x_z[:, src_j] + noise_scale * rng.standard_normal(n_sub)
        elif mode == 2:
            # 3. Heavy-tailed Student-t3 distractor column
            g1 = rng.standard_normal(n_sub)
            chi = np.maximum(rng.chisquare(df=3.0, size=n_sub) / 3.0, 1e-3)
            c_new = np.clip(g1 / np.sqrt(chi), -12.0, 12.0)
        else:
            # 4. Standardized pairwise quadratic interaction x_a * x_b
            ja = int(rng.integers(0, width))
            jb = int(rng.integers(0, width))
            prod_ab = x_z[:, ja] * x_z[:, jb]
            c_new = (prod_ab - float(np.mean(prod_ab))) / max(float(np.std(prod_ab)), 1e-6)
            c_new = np.clip(c_new, -10.0, 10.0)
        syn_cols.append(c_new.astype(np.float64))

    syn_mat = np.column_stack(syn_cols)
    x_aug = np.concatenate([x_sub, syn_mat], axis=1)
    # If total columns <= 64, apply a light random permutation of the last synthetic columns
    # with 25% of original columns to regularize column-order sensitivity
    if x_aug.shape[1] <= 64 and width >= 4:
        perm = np.arange(x_aug.shape[1])
        n_swap = min(k_syn, max(1, width // 4))
        orig_swap = rng.choice(width, size=n_swap, replace=False)
        syn_swap = width + rng.choice(k_syn, size=n_swap, replace=False)
        for o_i, s_i in zip(orig_swap, syn_swap):
            perm[o_i], perm[s_i] = perm[s_i], perm[o_i]
        x_aug = x_aug[:, perm]
    return x_aug, y_sub


def extract_or_load_semisynthetic_train_views(
    train_rows: List[Dict[str, Any]],
    net: Any,
    guard: Any,
    snap_before: Any,
    device: torch.device,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Extracts and caches two additional views (`sub80` and `semisyn80`) for all 1,219 DL-Train contexts."""
    cache_path = CACHE_DIR / "dl_train_semisynthetic_views.npz"
    cids = [r["context_sha256"] for r in train_rows]

    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as z:
            saved_cids = [str(c) for c in z["cids"]]
            if saved_cids == cids:
                out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"sub80": {}, "semisyn80": {}}
                for vname in ("sub80", "semisyn80"):
                    d629 = z[f"{vname}_dope629"].astype(np.float32)
                    lm48 = z[f"{vname}_lm48_z"].astype(np.float32)
                    hid = z[f"{vname}_hidden256"].astype(np.float32)
                    clm = z[f"{vname}_col_lm_64x17"].astype(np.float32)
                    cmask = z[f"{vname}_col_mask_64"].astype(bool)
                    for i, cid in enumerate(cids):
                        out[vname][cid] = {
                            "dope629": d629[i],
                            "lm48_z": lm48[i],
                            "hidden256": hid[i],
                            "col_lm_64x17": clm[i],
                            "col_mask_64": cmask[i],
                        }
                return out

    t0 = time.time()
    n_ctx = len(cids)
    ws = FusedLandmarkWorkspace(num_threads=4)
    arrays: Dict[str, Any] = {"cids": np.asarray(cids)}

    for v_idx, vname in enumerate(("sub80", "semisyn80")):
        v256_all = np.zeros((n_ctx, 256), dtype=np.float32)
        col_lm_all = np.zeros((n_ctx, 64, 17), dtype=np.float32)
        col_mask_all = np.zeros((n_ctx, 64), dtype=bool)

        for idx, cid in enumerate(cids):
            table_path = ch.INPUT_DIR / "tables" / f"{cid}.jtt1"
            raw = table_path.read_bytes()
            nt, nq, width = struct.unpack_from("<III", raw, 4)
            folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
            a = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(
                nt, width + 1
            )
            x, y = a[:, :-1], a[:, -1]
            rng = np.random.default_rng(prod.stream_seed("semisyn_train_view", cid, vname, v_idx))
            x_v, y_v = generate_perturbed_table_view(x, y, folds, vname, rng, subsample_frac=0.80)
            yn = ((y_v - np.mean(y_v)) / max(float(np.std(y_v)), 1e-8)).astype(np.float32)
            v, col, _ = ws.fit_transform_and_extract_bundle_high(
                epc.landmark_float32(x_v), np.ascontiguousarray(yn)
            )
            v256_all[idx] = v.astype(np.float32)
            n_active = min(len(col), 64)
            col_lm_all[idx, :n_active, : min(col.shape[1], 17)] = col[:n_active, :17]
            col_mask_all[idx, :n_active] = True

        with torch.no_grad():
            v_t = torch.as_tensor(v256_all, dtype=torch.float32, device=device)
            c_t = torch.as_tensor(col_lm_all, dtype=torch.float32, device=device)
            dope629_t, lm48_t = net._run_dope_mlc_v3_and_lm48(v_t, c_t, use_landmarks=True)
            _, _, lm48_z_t = net._project_stage1b_tokens_batched_gpu(lm48_t, dope629_t)
            hidden256_t = epc.extract_router_hidden256(net, v_t)

        arrays[f"{vname}_dope629"] = dope629_t.cpu().numpy().astype(np.float32)
        arrays[f"{vname}_lm48_z"] = lm48_z_t.cpu().numpy().astype(np.float32)
        arrays[f"{vname}_hidden256"] = hidden256_t.cpu().numpy().astype(np.float32)
        arrays[f"{vname}_col_lm_64x17"] = col_lm_all
        arrays[f"{vname}_col_mask_64"] = col_mask_all

    guard.verify_dope_frozen_unchanged(net, snap_before)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    print(f"[T1] Extracted & cached 2x1219 DL-Train semi-synthetic views in {time.time() - t0:.2f}s")
    return extract_or_load_semisynthetic_train_views(
        train_rows, net, guard, snap_before, device, force_recompute=False
    )


def extract_or_load_11view_batch_for_contexts(
    rows: List[Dict[str, Any]],
    cache_name: str,
    raw_ctx_clean: Dict[str, Dict[str, np.ndarray]],
    net: Any,
    guard: Any,
    snap_before: Any,
    device: torch.device,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Builds or loads an 11-view batch per context:
      - View 0: `100%` clean training data (exact match to `raw_ctx_clean[cid]`)
      - Views 1..8: `8x80%` random fold-stratified row subsamples (`sub80`)
      - Views 9..10: `2x80%` row subsamples with semi-synthetic column perturbation (`semisyn80`)
    Returns `{cid: {"dope629": [11, 629], "lm48_z": [11, 48], "hidden256": [11, 256],
                    "col_lm_64x17": [11, 64, 17], "col_mask_64": [11, 64]}}`.
    """
    cache_path = CACHE_DIR / cache_name
    cids = [r["context_sha256"] for r in rows]
    n_views = 11

    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as z:
            saved_cids = [str(c) for c in z["cids"]]
            if saved_cids == cids:
                d629 = z["dope629"].astype(np.float32)
                lm48 = z["lm48_z"].astype(np.float32)
                hid = z["hidden256"].astype(np.float32)
                clm = z["col_lm_64x17"].astype(np.float32)
                cmask = z["col_mask_64"].astype(bool)
                out: Dict[str, Dict[str, np.ndarray]] = {}
                for i, cid in enumerate(cids):
                    out[cid] = {
                        "dope629": d629[i],
                        "lm48_z": lm48[i],
                        "hidden256": hid[i],
                        "col_lm_64x17": clm[i],
                        "col_mask_64": cmask[i],
                    }
                return out

    t0 = time.time()
    n_ctx = len(cids)
    ws = FusedLandmarkWorkspace(num_threads=4)

    d629_all = np.zeros((n_ctx, n_views, 629), dtype=np.float32)
    lm48_all = np.zeros((n_ctx, n_views, 48), dtype=np.float32)
    hid_all = np.zeros((n_ctx, n_views, 256), dtype=np.float32)
    clm_all = np.zeros((n_ctx, n_views, 64, 17), dtype=np.float32)
    cmask_all = np.zeros((n_ctx, n_views, 64), dtype=bool)

    # View 0 is exact 100% data from raw_ctx_clean
    for i, cid in enumerate(cids):
        rc = raw_ctx_clean[cid]
        d629_all[i, 0] = rc["dope629"]
        lm48_all[i, 0] = rc["lm48_z"]
        hid_all[i, 0] = rc["hidden256"]
        clm_all[i, 0] = rc["col_lm_64x17"]
        cmask_all[i, 0] = rc["col_mask_64"]

    # Views 1..10: 8 sub80 + 2 semisyn80 encoded in a single batched pass
    n_sub_total = n_ctx * (n_views - 1)
    v256_flat = np.zeros((n_sub_total, 256), dtype=np.float32)
    clm_flat = np.zeros((n_sub_total, 64, 17), dtype=np.float32)
    cmask_flat = np.zeros((n_sub_total, 64), dtype=bool)

    for i, cid in enumerate(cids):
        table_path = ch.INPUT_DIR / "tables" / f"{cid}.jtt1"
        raw = table_path.read_bytes()
        nt, nq, width = struct.unpack_from("<III", raw, 4)
        folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
        a = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(
            nt, width + 1
        )
        x, y = a[:, :-1], a[:, -1]

        for b_view in range(1, n_views):
            vkind = "sub80" if b_view <= 8 else "semisyn80"
            rng = np.random.default_rng(prod.stream_seed("batch11_view", cid, b_view, vkind))
            x_v, y_v = generate_perturbed_table_view(x, y, folds, vkind, rng, subsample_frac=0.80)
            yn = ((y_v - np.mean(y_v)) / max(float(np.std(y_v)), 1e-8)).astype(np.float32)
            v, col, _ = ws.fit_transform_and_extract_bundle_high(
                epc.landmark_float32(x_v), np.ascontiguousarray(yn)
            )
            flat_idx = i * (n_views - 1) + (b_view - 1)
            v256_flat[flat_idx] = v.astype(np.float32)
            n_active = min(len(col), 64)
            clm_flat[flat_idx, :n_active, : min(col.shape[1], 17)] = col[:n_active, :17]
            cmask_flat[flat_idx, :n_active] = True

    with torch.no_grad():
        v_t = torch.as_tensor(v256_flat, dtype=torch.float32, device=device)
        c_t = torch.as_tensor(clm_flat, dtype=torch.float32, device=device)
        dope629_t, lm48_t = net._run_dope_mlc_v3_and_lm48(v_t, c_t, use_landmarks=True)
        _, _, lm48_z_t = net._project_stage1b_tokens_batched_gpu(lm48_t, dope629_t)
        hidden256_t = epc.extract_router_hidden256(net, v_t)

    d629_all[:, 1:, :] = dope629_t.cpu().numpy().reshape(n_ctx, n_views - 1, 629)
    lm48_all[:, 1:, :] = lm48_z_t.cpu().numpy().reshape(n_ctx, n_views - 1, 48)
    hid_all[:, 1:, :] = hidden256_t.cpu().numpy().reshape(n_ctx, n_views - 1, 256)
    clm_all[:, 1:, :, :] = clm_flat.reshape(n_ctx, n_views - 1, 64, 17)
    cmask_all[:, 1:, :] = cmask_flat.reshape(n_ctx, n_views - 1, 64)

    guard.verify_dope_frozen_unchanged(net, snap_before)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cids=np.asarray(cids),
        dope629=d629_all,
        lm48_z=lm48_all,
        hidden256=hid_all,
        col_lm_64x17=clm_all,
        col_mask_64=cmask_all,
    )
    print(f"[T1] Extracted & cached 11-view batch ({cache_name}) for {n_ctx} contexts in {time.time() - t0:.2f}s")
    return extract_or_load_11view_batch_for_contexts(
        rows, cache_name, raw_ctx_clean, net, guard, snap_before, device, force_recompute=False
    )


def update_packet_with_perturbed_view(
    base_pkt: Dict[str, Any],
    view_ctx: Dict[str, np.ndarray],
    s3_enc: p_s3.FullS3PreCondenseRegressionEncoder,
    s3_sc: FoldScaler,
    f_models: List[PreCondenseResidualBridge],
    g_models: List[PreCondenseResidualBridge],
    sc_bridge: FoldScaler,
    cand_q_topk: torch.Tensor,
    sem24_topk: torch.Tensor,
    gate_risk_topk: torch.Tensor,
    sp_prior_topk: torch.Tensor,
    r_sd: float,
    ck: np.ndarray,
    hc_k: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Computes an updated `[8, 616]` feature matrix for a perturbed view (`sub80` or `semisyn80`)
    of a context by re-running the Stage-1 S3 encoder and Stage-2 F/G bridges on `view_ctx`
    and updating the context/bridge columns (`492..567`, `568..579`, `580..615`).
    """
    X_new = base_pkt["X_616"].copy()
    d629 = view_ctx["dope629"].astype(np.float32)
    lm48_z = view_ctx["lm48_z"].astype(np.float32)
    hid256 = view_ctx["hidden256"].astype(np.float32)
    col_lm = view_ctx["col_lm_64x17"].astype(np.float32)
    col_mask = view_ctx["col_mask_64"].astype(np.float32)

    d_summary = np.array(
        [
            float(np.mean(d629)),
            float(np.std(d629)),
            float(np.linalg.norm(d629) / math.sqrt(len(d629))),
        ],
        dtype=np.float32,
    )
    ctx_vec76 = np.concatenate([lm48_z, d629[:25], d_summary], axis=0)
    X_new[:, 492:568] = ctx_vec76[None, :]

    # Stage-1 S3 16D embedding on view_ctx
    ct_s3 = s3_sc.transform_context_batch(
        d629[None, :],
        hid256[None, :],
        col_lm[None, :, :],
        view_ctx["col_mask_64"][None, :],
        device=device,
    )
    with torch.no_grad():
        _, s3_vec16_t = s3_enc.encode_context(ct_s3)
    s3_vec16 = s3_vec16_t[0].cpu().numpy().astype(np.float32)

    norms_11 = p_s3.compute_batch_log_norms_11(d629[None, :])[0]
    active_cols = col_lm[col_mask > 0.5] if np.any(col_mask > 0.5) else col_lm[:1]
    pre_condense_stats_9 = np.array(
        [
            float(np.mean(hid256)),
            float(np.std(hid256)),
            float(np.linalg.norm(hid256) / 16.0),
            float(np.mean(col_mask)),
            float(np.mean(active_cols)),
            float(np.std(active_cols)),
            float(np.mean(d629[256:565])),
            float(np.std(d629[256:565])),
            float(np.mean(d629[565:629][col_mask > 0.5])) if np.any(col_mask > 0.5) else 0.0,
        ],
        dtype=np.float32,
    )
    ctx_extra_36 = np.concatenate([norms_11, pre_condense_stats_9, s3_vec16], axis=0)
    X_new[:, 580:616] = ctx_extra_36[None, :]

    # Stage-2 F & G bridge candidate predictions on top-8
    ct_br = sc_bridge.transform_context_batch(
        d629[None, :],
        hid256[None, :],
        col_lm[None, :, :],
        view_ctx["col_mask_64"][None, :],
        device=device,
        tokens_10x32=np.zeros((1, 10, 32), dtype=np.float32),
        mask_10=np.ones((1, 10), dtype=bool),
    )
    with torch.no_grad():
        dp_f_list = [
            m.forward_prior_and_context(ct_br, cand_q_topk, sem24_topk, gate_risk_topk, sp_prior_topk)[0][0]
            .cpu()
            .numpy()
            .astype(np.float64)
            for m in f_models
        ]
        dp_g_list = [
            m.forward_prior_and_context(ct_br, cand_q_topk, sem24_topk, gate_risk_topk, sp_prior_topk)[0][0]
            .cpu()
            .numpy()
            .astype(np.float64)
            for m in g_models
        ]
    dp_f_k = np.mean(dp_f_list, axis=0)
    dp_g_k = np.mean(dp_g_list, axis=0)
    comb_k = ck + 0.025 * dp_f_k + 0.025 * dp_g_k + 0.010 * hc_k

    def _z(v: np.ndarray) -> np.ndarray:
        return (v - float(np.mean(v))) / max(float(np.std(v)), 1e-6)

    dp_f_z = _z(dp_f_k)
    dp_g_z = _z(dp_g_k)
    comb_z = _z(comb_k)

    for j in range(len(ck)):
        X_new[j, 568] = float(dp_f_z[j])
        X_new[j, 569] = float(dp_f_z[j] - dp_f_z[0])
        X_new[j, 570] = float(dp_g_z[j])
        X_new[j, 571] = float(dp_g_z[j] - dp_g_z[0])
        X_new[j, 578] = float(comb_z[j])
        X_new[j, 579] = float((comb_k[j] - comb_k[0]) / max(r_sd, 1e-12))

    return X_new


def train_neural_top8_reranker_v2(
    cids: List[str],
    packets: Dict[str, Dict[str, Any]],
    seed: int,
    device: torch.device,
    aug_views_X616: Optional[List[Dict[str, np.ndarray]]] = None,
    n_epochs: int = 18,
    warm_start_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[prod.PreCondenseTop8NeuralReranker, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Audited Top-8 Neural Reranker Trainer (`Stage C0 / T1`):
      - Warm-starts from converged Phase-1 `H256` weights (`warm_start_state_dict`) when provided so
        Stage C0 does not start from 0.
      - All-28-Pairs Regret-Weighted Margin Loss across all candidate pairs (i, j) with 1.5x gate-safety weight.
      - Temperature-Annealed ListNet (tau: 1.5 -> 0.65).
      - Two-Speed Differential LR (`lr_in=4.5e-4`, `lr_head=1.1e-3` when warm-started; `9e-4`/`1.8e-3` cold)
        + Cosine Annealing + SWA over final 6 epochs.
      - Semi-Synthetic Multi-View Context Augmentation + Ranking Consistency KL Loss (`L_consist`).
    """
    X_raw_clean = np.stack([packets[c]["X_616"] for c in cids], axis=0).astype(np.float32)  # [N, 8, 616]
    feat_mean = X_raw_clean.mean(axis=(0, 1), keepdims=True)
    feat_std = np.maximum(X_raw_clean.std(axis=(0, 1), keepdims=True), 1e-4)

    def _norm_x(x_arr: np.ndarray) -> np.ndarray:
        xn = np.clip((x_arr - feat_mean) / feat_std, -10.0, 10.0).astype(np.float32)
        xn[:, :, 2] = x_arr[:, :, 2]
        return xn

    X_clean_norm = _norm_x(X_raw_clean)
    X_aug_norms: List[torch.Tensor] = []
    if aug_views_X616:
        for v_dict in aug_views_X616:
            X_v = np.stack([v_dict[c] for c in cids], axis=0).astype(np.float32)
            X_aug_norms.append(torch.as_tensor(_norm_x(X_v), dtype=torch.float32, device=device))

    rel_mat = np.stack([packets[c]["rel_8"] for c in cids], axis=0).astype(np.float32)
    u_mat = np.stack([packets[c]["utility_8"] for c in cids], axis=0).astype(np.float32)
    inv_mat = np.stack([(~packets[c]["gate_8"]).astype(np.float32) for c in cids], axis=0)

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    model = prod.PreCondenseTop8NeuralReranker(in_dim=prod.ENRICHED_DIM_K8, hidden_dim=128).to(device)
    if warm_start_state_dict is not None:
        model.load_state_dict(warm_start_state_dict, strict=True)

    # Two-speed differential LR parameter groups
    in_params = list(model.in_norm.parameters()) + list(model.in_proj.parameters())
    head_params = (
        list(model.slot_attn.parameters())
        + list(model.post_norm.parameters())
        + list(model.score_head.parameters())
        + list(model.inv_head.parameters())
    )
    lr_in = 4.5e-4 if warm_start_state_dict is not None else 9.0e-4
    lr_head = 1.1e-3 if warm_start_state_dict is not None else 1.8e-3
    opt = torch.optim.AdamW(
        [
            {"params": in_params, "lr": lr_in, "weight_decay": 3e-4},
            {"params": head_params, "lr": lr_head, "weight_decay": 2e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=8.0e-5)

    X_t = torch.as_tensor(X_clean_norm, dtype=torch.float32, device=device)
    rel_t = torch.as_tensor(rel_mat, dtype=torch.float32, device=device)
    u_t = torch.as_tensor(u_mat, dtype=torch.float32, device=device)
    inv_t = torch.as_tensor(inv_mat, dtype=torch.float32, device=device)

    # Precompute all-28-pairs regret weights & adaptive margins [N, 8, 8]
    # Pair (i, j) active when u_i > u_j + 1e-4
    u_diff = u_t.unsqueeze(2) - u_t.unsqueeze(1)  # [N, 8, 8]: u_i - u_j
    pair_mask = (u_diff > 1e-4).float()
    regret_clipped = torch.clamp(u_diff, min=0.0, max=0.60)
    # 1.5x weight when the worse candidate j is invalid (fails CV gate)
    inv_j_mult = 1.0 + 0.5 * inv_t.unsqueeze(1)  # [N, 1, 8]
    # Extra 1.35x weight when j == 0 (beating R882 top-1) or i == 0 (defending R882 top-1 against worse slots)
    slot0_mult = torch.ones((1, 8, 8), dtype=torch.float32, device=device)
    slot0_mult[:, :, 0] = 1.35
    slot0_mult[:, 0, :] = 1.35
    pair_weights = pair_mask * (0.20 + regret_clipped) * inv_j_mult * slot0_mult
    pair_margins = torch.clamp(0.06 + 0.35 * regret_clipped, max=0.25)

    n_samples = len(cids)
    batch_size = 64
    rng = np.random.default_rng(seed)

    swa_state: Optional[Dict[str, torch.Tensor]] = None
    swa_count = 0
    swa_start = max(1, n_epochs - 8)
    epoch_losses: List[float] = []

    for ep in range(n_epochs):
        model.train()
        # Anneal ListNet temperature from 1.5 -> 0.65
        tau = 1.5 - (1.5 - 0.65) * (float(ep) / max(float(n_epochs - 1), 1.0))
        target_probs = F.softmax(rel_t / tau, dim=-1)

        perm = rng.permutation(n_samples)
        ep_loss_sum = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            b_t = torch.as_tensor(b_idx, dtype=torch.long, device=device)

            scores_clean, inv_clean = model(X_t[b_t])
            log_p_clean = F.log_softmax(scores_clean, dim=-1)
            loss_listnet = -(target_probs[b_t] * log_p_clean).sum(dim=-1).mean()

            # All-28-Pairs Regret-Weighted Margin Loss
            s_diff = scores_clean.unsqueeze(2) - scores_clean.unsqueeze(1)  # [B, 8, 8]: s_i - s_j
            pw_b = pair_weights[b_t]
            pm_b = pair_margins[b_t]
            loss_all_pairs = (F.relu(pm_b - s_diff) * pw_b).sum(dim=(1, 2)) / (
                pw_b.sum(dim=(1, 2)) + 1e-6
            )
            loss_pair = loss_all_pairs.mean()

            inv_b = inv_t[b_t]
            weight_b = 1.0 + 6.0 * inv_b
            loss_bce = F.binary_cross_entropy(
                inv_clean.clamp(1e-5, 1.0 - 1e-5), inv_b, weight=weight_b
            )
            loss = loss_listnet + 0.45 * loss_pair + 0.40 * loss_bce

            # Semi-synthetic multi-view augmentation + consistency KL
            if X_aug_norms:
                v_choice = int((ep + n_batches) % len(X_aug_norms))
                X_aug_b = X_aug_norms[v_choice][b_t]
                scores_aug, inv_aug = model(X_aug_b)
                log_p_aug = F.log_softmax(scores_aug, dim=-1)
                loss_ln_aug = -(target_probs[b_t] * log_p_aug).sum(dim=-1).mean()
                s_diff_aug = scores_aug.unsqueeze(2) - scores_aug.unsqueeze(1)
                loss_pair_aug = (
                    (F.relu(pm_b - s_diff_aug) * pw_b).sum(dim=(1, 2))
                    / (pw_b.sum(dim=(1, 2)) + 1e-6)
                ).mean()
                loss_bce_aug = F.binary_cross_entropy(
                    inv_aug.clamp(1e-5, 1.0 - 1e-5), inv_b, weight=weight_b
                )
                loss_aug = loss_ln_aug + 0.45 * loss_pair_aug + 0.40 * loss_bce_aug

                # Multi-view ranking consistency KL between clean and semi-synthetic view
                p_clean_det = F.softmax(scores_clean.detach() / 0.8, dim=-1)
                p_aug_det = F.softmax(scores_aug.detach() / 0.8, dim=-1)
                loss_consist = 0.5 * (
                    F.kl_div(F.log_softmax(scores_aug / 0.8, dim=-1), p_clean_det, reduction="batchmean")
                    + F.kl_div(F.log_softmax(scores_clean / 0.8, dim=-1), p_aug_det, reduction="batchmean")
                )
                loss = 0.65 * loss + 0.35 * loss_aug + 0.20 * loss_consist

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss_sum += float(loss.detach().item())
            n_batches += 1

        scheduler.step()
        epoch_losses.append(ep_loss_sum / max(n_batches, 1))

        if ep >= swa_start:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    swa_count = 1
                else:
                    for k, v in model.state_dict().items():
                        swa_state[k].add_(v.detach())
                    swa_count += 1

    if swa_state is not None and swa_count > 0:
        with torch.no_grad():
            for k in swa_state:
                swa_state[k].div_(float(swa_count))
            model.load_state_dict(swa_state, strict=True)

    model.eval()
    return model, feat_mean[0, 0], feat_std[0, 0], {
        "n_epochs": n_epochs,
        "swa_snapshots": swa_count,
        "initial_loss": float(epoch_losses[0]),
        "final_loss": float(epoch_losses[-1]),
    }


def predict_multiview_top8_reranker(
    nn_model: prod.PreCondenseTop8NeuralReranker,
    nn_mean: np.ndarray,
    nn_std: np.ndarray,
    lgb_rankers: List[Tuple[float, int, lgb.LGBMRanker]],
    veto_clf: lgb.LGBMClassifier,
    cids: List[str],
    view_packets_list: List[Dict[str, np.ndarray]],
    device: torch.device,
    view_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[float, int], np.ndarray], np.ndarray]:
    """Batched Multi-Subsample Inference across `B` views (`1x100% + 8x80% + 2xcol-jitter`):
    Evaluates `PreCondenseTop8NeuralReranker`, bagged `LGBMRanker`, and `LGBMClassifier`
    across all `B` views in a single batched pass `[N * B, 8, 616]` and returns view-aggregated
    scores and invalid-gate probabilities.
    """
    n_ctx = len(cids)
    n_views = len(view_packets_list)
    if view_weights is None:
        # View 0 (100% data) gets weight 0.35; the 10 subsample/jitter views share 0.65
        if n_views == 1:
            w = np.array([1.0], dtype=np.float64)
        else:
            w = np.full(n_views, 0.65 / float(n_views - 1), dtype=np.float64)
            w[0] = 0.35
    else:
        w = np.asarray(view_weights, dtype=np.float64)
        w = w / np.sum(w)

    # Stack [N, B, 8, 616]
    X_nb = np.stack(
        [np.stack([vp[c] for c in cids], axis=0) for vp in view_packets_list],
        axis=1,
    ).astype(np.float32)
    X_flat = X_nb.reshape(n_ctx * n_views, 8, prod.ENRICHED_DIM_K8)

    X_norm = np.clip((X_flat - nn_mean[None, None, :]) / nn_std[None, None, :], -10.0, 10.0)
    X_norm[:, :, 2] = X_flat[:, :, 2]
    with torch.no_grad():
        nn_s_flat, nn_inv_flat = nn_model(torch.as_tensor(X_norm, dtype=torch.float32, device=device))
    nn_s_nb = prod.standardize_scores_per_query(
        nn_s_flat.cpu().numpy().astype(np.float64).reshape(n_ctx, n_views, 8)
    )
    nn_inv_nb = nn_inv_flat.cpu().numpy().astype(np.float64).reshape(n_ctx, n_views, 8)

    nn_scores_agg = np.tensordot(nn_s_nb, w, axes=([1], [0]))
    nn_inv_agg = np.tensordot(nn_inv_nb, w, axes=([1], [0]))

    X_2d = X_flat.reshape(n_ctx * n_views * 8, prod.ENRICHED_DIM_K8)
    lgb_inv_nb = veto_clf.predict_proba(X_2d)[:, 1].astype(np.float64).reshape(n_ctx, n_views, 8)
    lgb_inv_agg = np.tensordot(lgb_inv_nb, w, axes=([1], [0]))

    lgb_scores_by_cfg: Dict[Tuple[float, int], np.ndarray] = {}
    for lr, n_est, r_model in lgb_rankers:
        s_flat = r_model.predict(X_2d, num_iteration=int(n_est)).astype(np.float64)
        s_nb = prod.standardize_scores_per_query(s_flat.reshape(n_ctx, n_views, 8))
        lgb_scores_by_cfg[(lr, n_est)] = np.tensordot(s_nb, w, axes=([1], [0]))

    return nn_scores_agg, nn_inv_agg, lgb_scores_by_cfg, lgb_inv_agg


def transform_context_batch(
    scaler: FoldScaler,
    d629_batch: np.ndarray,
    hid_batch: np.ndarray,
    clm_batch: np.ndarray,
    cmask_batch: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Vectorized batch version of `FoldScaler.transform_context` for `[B, ...]` arrays."""
    n_b = len(d629_batch)
    dope_z = np.clip(
        (d629_batch.astype(np.float32) - scaler.dope629_mean[None, :]) / scaler.dope629_std[None, :],
        -15.0,
        15.0,
    )
    hid_z = np.clip(
        (hid_batch.astype(np.float32) - scaler.hidden256_mean[None, :]) / scaler.hidden256_std[None, :],
        -15.0,
        15.0,
    )
    col_gates = d629_batch[:, 565:629, None].astype(np.float32)
    col18_raw = np.concatenate([clm_batch.astype(np.float32), col_gates], axis=-1)
    col18_z = np.clip(
        (col18_raw - scaler.col18_mean[None, None, :]) / scaler.col18_std[None, None, :],
        -15.0,
        15.0,
    )
    col18_z = col18_z * cmask_batch[:, :, None].astype(np.float32)
    norms_11 = p_s3.compute_batch_log_norms_11(dope_z)
    return {
        "tokens_10x32": torch.zeros((n_b, 10, 32), dtype=torch.float32, device=device),
        "mask_10": torch.ones((n_b, 10), dtype=torch.float32, device=device),
        "dope629_z": torch.as_tensor(dope_z, dtype=torch.float32, device=device),
        "block_log_norms": torch.as_tensor(norms_11, dtype=torch.float32, device=device),
        "hidden256_z": torch.as_tensor(hid_z, dtype=torch.float32, device=device),
        "col18_z": torch.as_tensor(col18_z, dtype=torch.float32, device=device),
        "col_mask_64": torch.as_tensor(cmask_batch.astype(np.float32), dtype=torch.float32, device=device),
    }


def build_perturbed_616_packets_for_view(
    cids: List[str],
    base_packets: Dict[str, Dict[str, Any]],
    view_ctx_by_cid: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Vectorized construction of `[8, 616]` perturbed packets for `cids` under `view_ctx_by_cid`.
    Updates the context-dependent blocks:
      - Cols `492..567` (`76D`): `lm48_z` (48D) + `dope629[:25]` (25D) + `d_summary` (3D)
      - Cols `580..615` (`36D`): `block_log_norms` (11D) + `pre_condense_stats_9` (9D) + `s3_vec16` (16D)
    """
    d629_batch = np.stack([view_ctx_by_cid[c]["dope629"] for c in cids], axis=0).astype(np.float32)
    lm48_batch = np.stack([view_ctx_by_cid[c]["lm48_z"] for c in cids], axis=0).astype(np.float32)
    hid_batch = np.stack([view_ctx_by_cid[c]["hidden256"] for c in cids], axis=0).astype(np.float32)
    clm_batch = np.stack([view_ctx_by_cid[c]["col_lm_64x17"] for c in cids], axis=0).astype(np.float32)
    cmask_batch = np.stack([view_ctx_by_cid[c]["col_mask_64"] for c in cids], axis=0).astype(bool)

    ckpt = torch.load(
        H256_OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt", map_location="cpu", weights_only=False
    )
    sc_dict = ckpt["scaler_policy_b"]
    scaler = FoldScaler(
        dope629_mean=np.asarray(sc_dict["dope629_mean"], dtype=np.float32),
        dope629_std=np.asarray(sc_dict["dope629_std"], dtype=np.float32),
        hidden256_mean=np.asarray(sc_dict["hidden256_mean"], dtype=np.float32),
        hidden256_std=np.asarray(sc_dict["hidden256_std"], dtype=np.float32),
        col18_mean=np.asarray(sc_dict["col18_mean"], dtype=np.float32),
        col18_std=np.asarray(sc_dict["col18_std"], dtype=np.float32),
    )
    bridge = PreCondenseResidualBridge(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
    ).to(device)
    bridge.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    bridge.eval()
    with torch.no_grad():
        ct = transform_context_batch(
            scaler, d629_batch, hid_batch, clm_batch, cmask_batch, device=device
        )
        ctx_tokens, ctx_mask, scale_8 = bridge.encode_context_tokens(ct)
        mask_exp = ctx_mask.unsqueeze(-1)
        mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
        ctx_mean = (ctx_tokens * mask_exp).sum(dim=1) / mask_sum
        ctx_masked_for_max = torch.where(
            mask_exp > 0.5, ctx_tokens, torch.full_like(ctx_tokens, -1e4)
        )
        ctx_max = ctx_masked_for_max.max(dim=1).values
        ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)
        u_ctx = bridge.ctx_bilinear_u(ctx_summary_64)  # [N, 24]
        s3_vec16_batch = torch.cat([u_ctx[:, :12], scale_8[:, :4]], dim=-1).cpu().numpy().astype(np.float32)
    s3_vec16_by_cid = {c: s3_vec16_batch[i] for i, c in enumerate(cids)}
    norms_11_batch = p_s3.compute_batch_log_norms_11(d629_batch)

    out: Dict[str, np.ndarray] = {}
    for i, cid in enumerate(cids):
        X_new = base_packets[cid]["X_616"].copy()
        d629 = d629_batch[i]
        lm48_z = lm48_batch[i]
        hid256 = hid_batch[i]
        col_lm = clm_batch[i]
        col_mask = cmask_batch[i].astype(np.float32)

        d_summary = np.array(
            [
                float(np.mean(d629)),
                float(np.std(d629)),
                float(np.linalg.norm(d629) / math.sqrt(len(d629))),
            ],
            dtype=np.float32,
        )
        ctx_vec76 = np.concatenate([lm48_z, d629[:25], d_summary], axis=0)
        X_new[:, 492:568] = ctx_vec76[None, :]

        active_cols = col_lm[col_mask > 0.5] if np.any(col_mask > 0.5) else col_lm[:1]
        pre_condense_stats_9 = np.array(
            [
                float(np.mean(hid256)),
                float(np.std(hid256)),
                float(np.linalg.norm(hid256) / 16.0),
                float(np.mean(col_mask)),
                float(np.mean(active_cols)),
                float(np.std(active_cols)),
                float(np.mean(d629[256:565])),
                float(np.std(d629[256:565])),
                float(np.mean(d629[565:629][col_mask > 0.5])) if np.any(col_mask > 0.5) else 0.0,
            ],
            dtype=np.float32,
        )
        ctx_extra_36 = np.concatenate(
            [norms_11_batch[i], pre_condense_stats_9, s3_vec16_by_cid[cid]], axis=0
        )
        X_new[:, 580:616] = ctx_extra_36[None, :]
        out[cid] = X_new
    return out


def run_gate_t1_hypothesis_test() -> Dict[str, Any]:
    """Executes Gate T1:
      1. Extracts 2x1219 DL-Train semi-synthetic views (`sub80`, `semisyn80`) and 11-view (`1x100% + 8x80% + 2xcol-jitter`)
         batches for DL-Val91.
      2. Trains `PreCondenseTop8NeuralReranker` with Two-Speed Differential LR + Temperature-Annealed ListNet +
         All-28-Pairs Regret-Weighted Margin Loss + Semi-Synthetic Multi-View Consistency KL + Cosine SWA.
      3. Trains semi-synthetically augmented 616D `LGBMRanker` & `LGBMClassifier` veto head on `3x` training views.
      4. Evaluates OOF (`1,219` DL-Train) and Held-Out (`91` DL-Val91) utility, Elo, and head-to-head vs `Opus5p5_Extra`.
    """
    t0 = time.time()
    device = torch.device("cpu")
    torch.set_num_threads(4)

    net, guard, dope_snap_before, w1_mat = prod.p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    all_rows = train_rows + val_rows
    train_cids = [r["context_sha256"] for r in train_rows]
    val_cids = [r["context_sha256"] for r in val_rows]

    raw_cache = r256.load_pre_condense_cache(prod.CACHE_PATH)
    btc = torch.load(H256_OUT_DIR / "cache" / "base_terms_and_priors_cache.pt", map_location="cpu", weights_only=False)
    base_terms = btc["base_terms"]
    pkt_cache = torch.load(H256_OUT_DIR / "cache" / "flagship_packets_616.pt", map_location="cpu", weights_only=False)
    packets = pkt_cache["packets"]
    r882_u_by_cid = pkt_cache["r882_u_by_cid"]
    preds_cache = torch.load(H256_OUT_DIR / "cache" / "outer_top8_preds_cache.pt", map_location="cpu", weights_only=False)
    priors_cache = torch.load(H256_OUT_DIR / "cache" / "stage2_bridge_priors_fg.pt", map_location="cpu", weights_only=False)
    dp_f_by_cid = priors_cache["dp_f_by_cid"]
    dp_g_by_cid = priors_cache["dp_g_by_cid"]

    h256_train_bundle = torch.load(
        H256_OUT_DIR / "checkpoints" / "h256_production_1219_dl_train_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    h256_nn_warm_state = h256_train_bundle["top8_neural_reranker_state_dict"]

    # 1. Extract semi-synthetic views for DL-Train and 11-view batch for DL-Val91
    train_aug_views = extract_or_load_semisynthetic_train_views(
        train_rows, net, guard, dope_snap_before, device
    )
    val_11views = extract_or_load_11view_batch_for_contexts(
        val_rows, "dl_val91_11view_batch_features.npz", raw_cache, net, guard, dope_snap_before, device
    )

    train_sub80_X616 = build_perturbed_616_packets_for_view(
        train_cids, packets, train_aug_views["sub80"], device
    )
    train_semisyn_X616 = build_perturbed_616_packets_for_view(
        train_cids, packets, train_aug_views["semisyn80"], device
    )

    val_view_X616_list: List[Dict[str, np.ndarray]] = []
    val_view_X568_list: List[Dict[str, np.ndarray]] = []
    val_view_X580_list: List[Dict[str, np.ndarray]] = []
    for b_idx in range(11):
        if b_idx == 0:
            v_x616 = {c: packets[c]["X_616"] for c in val_cids}
        else:
            v_ctx_map = {
                c: {
                    "dope629": val_11views[c]["dope629"][b_idx],
                    "lm48_z": val_11views[c]["lm48_z"][b_idx],
                    "hidden256": val_11views[c]["hidden256"][b_idx],
                    "col_lm_64x17": val_11views[c]["col_lm_64x17"][b_idx],
                    "col_mask_64": val_11views[c]["col_mask_64"][b_idx],
                }
                for c in val_cids
            }
            v_x616 = build_perturbed_616_packets_for_view(val_cids, packets, v_ctx_map, device)
        val_view_X616_list.append(v_x616)
        val_view_X568_list.append({c: v_x616[c][:, :568] for c in val_cids})
        val_view_X580_list.append({c: v_x616[c][:, :580] for c in val_cids})

    # Deterministic per-seed idx4
    idx4_by_seed_cid: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in prod.SEEDS}
    for s in prod.SEEDS:
        for r in all_rows:
            c = r["context_sha256"]
            rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", s, c))
            idx4_by_seed_cid[s][c] = rng_c.choice(len(base_terms[c]["rcost"]), size=4, replace=False)

    # 2. Train & evaluate across 5 family-grouped folds x 3 seeds (15 cells)
    t1_cache_path = CACHE_DIR / "gate_t1_oof_and_val_cache.pt"
    if t1_cache_path.exists():
        t1_preds = torch.load(t1_cache_path, map_location="cpu", weights_only=False)
    else:
        t1_preds = {}
        for s in prod.SEEDS:
            for k in range(5):
                ce = preds_cache[(s, k)]
                assess_cids = ce["assess_cids"]
                fit_cids = [c for c in train_cids if c not in set(assess_cids)]

                # Train Audited Neural Reranker v2 on clean + sub80 + semisyn80 views (warm-started from H256)
                nn_v2, nn_mean_v2, nn_std_v2, nn_telem = train_neural_top8_reranker_v2(
                    fit_cids,
                    packets,
                    seed=prod.stream_seed(r256.FOLD_SEED, s, k, "H257_C0_NN_V2_SEMISYN"),
                    device=device,
                    aug_views_X616=[train_sub80_X616, train_semisyn_X616],
                    n_epochs=18,
                    warm_start_state_dict=h256_nn_warm_state if s != prod.SEEDS[0] else None,
                )

                # Evaluate Audited Neural Reranker v2 on OOF (assess_cids: clean + sub80 + semisyn80)
                oof_views_x616 = [
                    {c: packets[c]["X_616"] for c in assess_cids},
                    {c: train_sub80_X616[c] for c in assess_cids},
                    {c: train_semisyn_X616[c] for c in assess_cids},
                ]
                snn_oof_clean, pnn_oof_clean = prod.predict_neural_top8_reranker(
                    nn_v2, nn_mean_v2, nn_std_v2, assess_cids, packets, device
                )
                # Multi-view OOF neural prediction across the 3 views (weights 0.50, 0.25, 0.25)
                snn_oof_mv_list = []
                pnn_oof_mv_list = []
                for vp in oof_views_x616:
                    X_raw = np.stack([vp[c] for c in assess_cids], axis=0).astype(np.float32)
                    X_norm = np.clip((X_raw - nn_mean_v2[None, None, :]) / nn_std_v2[None, None, :], -10.0, 10.0)
                    X_norm[:, :, 2] = X_raw[:, :, 2]
                    with torch.no_grad():
                        sc_v, pi_v = nn_v2(torch.as_tensor(X_norm, dtype=torch.float32, device=device))
                    snn_oof_mv_list.append(prod.standardize_scores_per_query(sc_v.cpu().numpy().astype(np.float64)))
                    pnn_oof_mv_list.append(pi_v.cpu().numpy().astype(np.float64))
                snn_oof_mv = 0.50 * snn_oof_mv_list[0] + 0.25 * snn_oof_mv_list[1] + 0.25 * snn_oof_mv_list[2]
                pnn_oof_mv = 0.50 * pnn_oof_mv_list[0] + 0.25 * pnn_oof_mv_list[1] + 0.25 * pnn_oof_mv_list[2]

                # Evaluate Audited Neural Reranker v2 on DL-Val91 (single-view vs 11-view batch!)
                snn_val_clean, pnn_val_clean = prod.predict_neural_top8_reranker(
                    nn_v2, nn_mean_v2, nn_std_v2, val_cids, packets, device
                )
                snn_val_11_list = []
                pnn_val_11_list = []
                for vp in val_view_X616_list:
                    X_raw = np.stack([vp[c] for c in val_cids], axis=0).astype(np.float32)
                    X_norm = np.clip((X_raw - nn_mean_v2[None, None, :]) / nn_std_v2[None, None, :], -10.0, 10.0)
                    X_norm[:, :, 2] = X_raw[:, :, 2]
                    with torch.no_grad():
                        sc_v, pi_v = nn_v2(torch.as_tensor(X_norm, dtype=torch.float32, device=device))
                    snn_val_11_list.append(prod.standardize_scores_per_query(sc_v.cpu().numpy().astype(np.float64)))
                    pnn_val_11_list.append(pi_v.cpu().numpy().astype(np.float64))
                w11 = np.full(11, 0.65 / 10.0, dtype=np.float64)
                w11[0] = 0.35
                snn_val_batch11 = np.tensordot(np.stack(snn_val_11_list, axis=1), w11, axes=([1], [0]))
                pnn_val_batch11 = np.tensordot(np.stack(pnn_val_11_list, axis=1), w11, axes=([1], [0]))

                # Also fit Semi-Synthetically Augmented 616D LGBMRanker & Veto Classifier on 3x training views
                X_tr_clean = np.concatenate([packets[c]["X_616"] for c in fit_cids], axis=0)
                X_tr_sub80 = np.concatenate([train_sub80_X616[c] for c in fit_cids], axis=0)
                X_tr_semisyn = np.concatenate([train_semisyn_X616[c] for c in fit_cids], axis=0)
                X_tr_3x = np.concatenate([X_tr_clean, X_tr_sub80, X_tr_semisyn], axis=0)
                y_tr_1x = np.concatenate([packets[c]["rel_8"] for c in fit_cids], axis=0)
                y_tr_3x = np.concatenate([y_tr_1x, y_tr_1x, y_tr_1x], axis=0)
                y_inv_1x = np.concatenate([(~packets[c]["gate_8"]).astype(np.int32) for c in fit_cids], axis=0)
                y_inv_3x = np.concatenate([y_inv_1x, y_inv_1x, y_inv_1x], axis=0)

                clf_616_aug = lgb.LGBMClassifier(
                    objective="binary",
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=30,
                    n_estimators=75,
                    max_bin=127,
                    subsample=0.80,
                    subsample_freq=1,
                    colsample_bytree=0.40,
                    reg_alpha=1.0,
                    reg_lambda=6.0,
                    n_jobs=8,
                    random_state=prod.stream_seed(r256.FOLD_SEED, s, k, "H257_616_AUG_VETO"),
                    verbose=-1,
                )
                clf_616_aug.fit(X_tr_3x, y_inv_3x)

                ranker_616_aug = lgb.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[1, 2, 4],
                    label_gain=list(range(8)),
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=25,
                    n_estimators=85,
                    max_bin=127,
                    subsample=0.80,
                    subsample_freq=1,
                    colsample_bytree=0.40,
                    reg_alpha=0.8,
                    reg_lambda=5.0,
                    n_jobs=8,
                    random_state=prod.stream_seed(r256.FOLD_SEED, s, k, "H257_616_AUG_RANK"),
                    verbose=-1,
                )
                ranker_616_aug.fit(X_tr_3x, y_tr_3x, group=[8] * (len(fit_cids) * 3))

                # Predict with augmented 616D GBDT on OOF (3 views) and Val (11 views)
                s616_aug_oof_views = []
                p616_aug_oof_views = []
                for vp in oof_views_x616:
                    X_v = np.concatenate([vp[c] for c in assess_cids], axis=0)
                    s_v = np.asarray(ranker_616_aug.booster_.predict(X_v, num_iteration=85), dtype=np.float64).reshape(len(assess_cids), 8)
                    p_v = np.asarray(clf_616_aug.booster_.predict(X_v), dtype=np.float64).reshape(len(assess_cids), 8)
                    s616_aug_oof_views.append(prod.standardize_scores_per_query(s_v))
                    p616_aug_oof_views.append(p_v)
                s616_aug_oof = 0.50 * s616_aug_oof_views[0] + 0.25 * s616_aug_oof_views[1] + 0.25 * s616_aug_oof_views[2]
                p616_aug_oof = 0.50 * p616_aug_oof_views[0] + 0.25 * p616_aug_oof_views[1] + 0.25 * p616_aug_oof_views[2]

                s616_aug_val_views = []
                p616_aug_val_views = []
                for vp in val_view_X616_list:
                    X_v = np.concatenate([vp[c] for c in val_cids], axis=0)
                    s_v = np.asarray(ranker_616_aug.booster_.predict(X_v, num_iteration=85), dtype=np.float64).reshape(len(val_cids), 8)
                    p_v = np.asarray(clf_616_aug.booster_.predict(X_v), dtype=np.float64).reshape(len(val_cids), 8)
                    s616_aug_val_views.append(prod.standardize_scores_per_query(s_v))
                    p616_aug_val_views.append(p_v)
                s616_aug_val_11 = np.tensordot(np.stack(s616_aug_val_views, axis=1), w11, axes=([1], [0]))
                p616_aug_val_11 = np.tensordot(np.stack(p616_aug_val_views, axis=1), w11, axes=([1], [0]))

                t1_preds[(s, k)] = {
                    "assess_cids": assess_cids,
                    "snn_oof_clean": snn_oof_clean,
                    "pnn_oof_clean": pnn_oof_clean,
                    "snn_oof_mv": snn_oof_mv,
                    "pnn_oof_mv": pnn_oof_mv,
                    "s616_aug_oof": s616_aug_oof,
                    "p616_aug_oof": p616_aug_oof,
                    "snn_val_clean": snn_val_clean,
                    "pnn_val_clean": pnn_val_clean,
                    "snn_val_batch11": snn_val_batch11,
                    "pnn_val_batch11": pnn_val_batch11,
                    "s616_aug_val_11": s616_aug_val_11,
                    "p616_aug_val_11": p616_aug_val_11,
                    "nn_telem": nn_telem,
                }
            print(f"[Stage C0 / T1] Completed 5-fold semi-synthetic + warm-started neural training for seed {s} in {time.time() - t0:.1f}s", flush=True)
        torch.save(t1_preds, t1_cache_path)

    # 3. Compare baseline H256 vs T1 Audited + Semi-Synthetic + 11-View Batch across OOF and DL-Val91
    arms_oof_u: Dict[str, Dict[int, Dict[str, float]]] = {
        a: {s: {} for s in prod.SEEDS}
        for a in (
            "H256_V1_NN_ONLY",
            "H256_T1_AUDITED_NN_ONLY",
            "H256_T1_BATCH_NN_ONLY",
            "H256_STATIC_0REV_TOP8_SOFT_VETO",
            "H256_T1_BATCH11_STATIC_0REV",
            "H256_ACTIVE_2PLUS2_UNIFIED",
            "H256_T1_BATCH11_ACTIVE_2PLUS2",
        )
    }
    arms_oof_inv: Dict[str, Dict[int, Dict[str, bool]]] = {
        a: {s: {} for s in prod.SEEDS} for a in arms_oof_u
    }
    arms_val_u: Dict[str, Dict[int, Dict[str, float]]] = {
        a: {s: {} for s in prod.SEEDS} for a in arms_oof_u
    }
    arms_val_inv: Dict[str, Dict[int, Dict[str, bool]]] = {
        a: {s: {} for s in prod.SEEDS} for a in arms_oof_u
    }

    for s in prod.SEEDS:
        val_base_sc_folds = []
        val_base_pinv_folds = []
        val_t1_sc_folds = []
        val_t1_pinv_folds = []
        val_v1_nn_sc_folds = []
        val_v1_nn_pinv_folds = []
        val_t1_nn_clean_sc_folds = []
        val_t1_nn_clean_pinv_folds = []
        val_t1_nn_b11_sc_folds = []
        val_t1_nn_b11_pinv_folds = []

        for k in range(5):
            ce = preds_cache[(s, k)]
            te = t1_preds[(s, k)]
            assess_cids = ce["assess_cids"]

            sc_base_oof, pinv_base_oof, sc_v1_nn_oof, pinv_v1_nn_oof = (
                prod.compute_bagged_multi_view_scores_and_risk(ce["oof"], assess_cids, packets)
            )
            # T1 Batch-Ensembled Multi-View Score on OOF
            b568_oof = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["oof"]["s568"].values()], axis=0)
            )
            b580_oof = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["oof"]["s580"].values()], axis=0)
            )
            b616_base_oof = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["oof"]["s616"].values()], axis=0)
            )
            b616_aug_oof = prod.standardize_scores_per_query(te["s616_aug_oof"])
            bnn_v2_clean_oof = prod.standardize_scores_per_query(te["snn_oof_clean"])
            bnn_v2_mv_oof = prod.standardize_scores_per_query(te["snn_oof_mv"])
            bbr_oof = prod.standardize_scores_per_query(
                np.stack([-packets[c]["X_616"][:, 578] for c in assess_cids], axis=0)
            )

            sc_t1_oof = (
                0.34 * b568_oof
                + 0.26 * b580_oof
                + 0.10 * b616_base_oof
                + 0.10 * b616_aug_oof
                + 0.15 * bnn_v2_mv_oof
                + 0.05 * bbr_oof
            )
            pinv_t1_oof = (
                0.38 * ce["oof"]["p568"]
                + 0.27 * ce["oof"]["p580"]
                + 0.10 * ce["oof"]["p616"]
                + 0.10 * te["p616_aug_oof"]
                + 0.15 * te["pnn_oof_mv"]
            )

            picks_v1_nn = prod.select_picks_from_scores(sc_v1_nn_oof, pinv_v1_nn_oof, lam=1.5, tau=1.0, margin=0.10)
            picks_t1_nn_clean = prod.select_picks_from_scores(bnn_v2_clean_oof, te["pnn_oof_clean"], lam=1.5, tau=1.0, margin=0.10)
            picks_t1_nn_mv = prod.select_picks_from_scores(bnn_v2_mv_oof, te["pnn_oof_mv"], lam=1.5, tau=1.0, margin=0.10)
            picks_base_st = prod.select_picks_from_scores(sc_base_oof, pinv_base_oof, lam=1.5, tau=1.0, margin=0.05)
            picks_t1_st = prod.select_picks_from_scores(sc_t1_oof, pinv_t1_oof, lam=1.5, tau=1.0, margin=0.05)

            for i_c, cid in enumerate(assess_cids):
                pkt = packets[cid]
                for arm_key, p_arr in (
                    ("H256_V1_NN_ONLY", picks_v1_nn),
                    ("H256_T1_AUDITED_NN_ONLY", picks_t1_nn_clean),
                    ("H256_T1_BATCH_NN_ONLY", picks_t1_nn_mv),
                    ("H256_STATIC_0REV_TOP8_SOFT_VETO", picks_base_st),
                    ("H256_T1_BATCH11_STATIC_0REV", picks_t1_st),
                ):
                    j_pick = int(p_arr[i_c])
                    arms_oof_u[arm_key][s][cid] = float(pkt["utility_8"][j_pick])
                    arms_oof_inv[arm_key][s][cid] = bool(not pkt["gate_8"][j_pick])

                act_base = prod.evaluate_active_2plus2_unified_on_context(
                    cid=cid,
                    bt=base_terms[cid],
                    pkt=pkt,
                    raw_ctx=raw_cache[cid],
                    idx4=idx4_by_seed_cid[s][cid],
                    static_pick_j=int(picks_base_st[i_c]),
                    static_scores_8=sc_base_oof[i_c],
                    static_pinv_8=pinv_base_oof[i_c],
                    stage2_cache={},
                    seed=s,
                    fold_k=k,
                    device=device,
                    dp_f_by_cid=dp_f_by_cid,
                    dp_g_by_cid=dp_g_by_cid,
                )
                arms_oof_u["H256_ACTIVE_2PLUS2_UNIFIED"][s][cid] = act_base["utility"]
                arms_oof_inv["H256_ACTIVE_2PLUS2_UNIFIED"][s][cid] = act_base["invalid_pick"]

                act_t1 = prod.evaluate_active_2plus2_unified_on_context(
                    cid=cid,
                    bt=base_terms[cid],
                    pkt=pkt,
                    raw_ctx=raw_cache[cid],
                    idx4=idx4_by_seed_cid[s][cid],
                    static_pick_j=int(picks_t1_st[i_c]),
                    static_scores_8=sc_t1_oof[i_c],
                    static_pinv_8=pinv_t1_oof[i_c],
                    stage2_cache={},
                    seed=s,
                    fold_k=k,
                    device=device,
                    dp_f_by_cid=dp_f_by_cid,
                    dp_g_by_cid=dp_g_by_cid,
                )
                arms_oof_u["H256_T1_BATCH11_ACTIVE_2PLUS2"][s][cid] = act_t1["utility"]
                arms_oof_inv["H256_T1_BATCH11_ACTIVE_2PLUS2"][s][cid] = act_t1["invalid_pick"]

            # Validation fold accumulation
            sc_base_val_k, pinv_base_val_k, sc_v1_nn_val_k, pinv_v1_nn_val_k = (
                prod.compute_bagged_multi_view_scores_and_risk(ce["val"], val_cids, packets)
            )
            b568_val = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s568"].values()], axis=0)
            )
            b580_val = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s580"].values()], axis=0)
            )
            b616_base_val = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s616"].values()], axis=0)
            )
            b616_aug_val_11 = prod.standardize_scores_per_query(te["s616_aug_val_11"])
            bnn_v2_clean_val = prod.standardize_scores_per_query(te["snn_val_clean"])
            bnn_v2_b11_val = prod.standardize_scores_per_query(te["snn_val_batch11"])
            bbr_val = prod.standardize_scores_per_query(
                np.stack([-packets[c]["X_616"][:, 578] for c in val_cids], axis=0)
            )

            sc_t1_val_k = (
                0.34 * b568_val
                + 0.26 * b580_val
                + 0.10 * b616_base_val
                + 0.10 * b616_aug_val_11
                + 0.15 * bnn_v2_b11_val
                + 0.05 * bbr_val
            )
            pinv_t1_val_k = (
                0.38 * ce["val"]["p568"]
                + 0.27 * ce["val"]["p580"]
                + 0.10 * ce["val"]["p616"]
                + 0.10 * te["p616_aug_val_11"]
                + 0.15 * te["pnn_val_batch11"]
            )

            val_base_sc_folds.append(prod.standardize_scores_per_query(sc_base_val_k))
            val_base_pinv_folds.append(pinv_base_val_k)
            val_t1_sc_folds.append(prod.standardize_scores_per_query(sc_t1_val_k))
            val_t1_pinv_folds.append(pinv_t1_val_k)
            val_v1_nn_sc_folds.append(prod.standardize_scores_per_query(sc_v1_nn_val_k))
            val_v1_nn_pinv_folds.append(pinv_v1_nn_val_k)
            val_t1_nn_clean_sc_folds.append(bnn_v2_clean_val)
            val_t1_nn_clean_pinv_folds.append(te["pnn_val_clean"])
            val_t1_nn_b11_sc_folds.append(bnn_v2_b11_val)
            val_t1_nn_b11_pinv_folds.append(te["pnn_val_batch11"])

        val_base_sc = np.mean(val_base_sc_folds, axis=0)
        val_base_pinv = np.mean(val_base_pinv_folds, axis=0)
        val_t1_sc = np.mean(val_t1_sc_folds, axis=0)
        val_t1_pinv = np.mean(val_t1_pinv_folds, axis=0)
        val_v1_nn_sc = np.mean(val_v1_nn_sc_folds, axis=0)
        val_v1_nn_pinv = np.mean(val_v1_nn_pinv_folds, axis=0)
        val_t1_nn_clean_sc = np.mean(val_t1_nn_clean_sc_folds, axis=0)
        val_t1_nn_clean_pinv = np.mean(val_t1_nn_clean_pinv_folds, axis=0)
        val_t1_nn_b11_sc = np.mean(val_t1_nn_b11_sc_folds, axis=0)
        val_t1_nn_b11_pinv = np.mean(val_t1_nn_b11_pinv_folds, axis=0)

        picks_val_v1_nn = prod.select_picks_from_scores(val_v1_nn_sc, val_v1_nn_pinv, lam=1.5, tau=1.0, margin=0.10)
        picks_val_t1_nn_clean = prod.select_picks_from_scores(val_t1_nn_clean_sc, val_t1_nn_clean_pinv, lam=1.5, tau=1.0, margin=0.10)
        picks_val_t1_nn_b11 = prod.select_picks_from_scores(val_t1_nn_b11_sc, val_t1_nn_b11_pinv, lam=1.5, tau=1.0, margin=0.10)
        picks_val_base_st = prod.select_picks_from_scores(val_base_sc, val_base_pinv, lam=1.5, tau=1.0, margin=0.05)
        picks_val_t1_st = prod.select_picks_from_scores(val_t1_sc, val_t1_pinv, lam=1.5, tau=1.0, margin=0.05)

        for i_v, cid in enumerate(val_cids):
            pkt = packets[cid]
            for arm_key, p_arr in (
                ("H256_V1_NN_ONLY", picks_val_v1_nn),
                ("H256_T1_AUDITED_NN_ONLY", picks_val_t1_nn_clean),
                ("H256_T1_BATCH_NN_ONLY", picks_val_t1_nn_b11),
                ("H256_STATIC_0REV_TOP8_SOFT_VETO", picks_val_base_st),
                ("H256_T1_BATCH11_STATIC_0REV", picks_val_t1_st),
            ):
                j_pick = int(p_arr[i_v])
                arms_val_u[arm_key][s][cid] = float(pkt["utility_8"][j_pick])
                arms_val_inv[arm_key][s][cid] = bool(not pkt["gate_8"][j_pick])

            act_val_base = prod.evaluate_active_2plus2_unified_on_context(
                cid=cid,
                bt=base_terms[cid],
                pkt=pkt,
                raw_ctx=raw_cache[cid],
                idx4=idx4_by_seed_cid[s][cid],
                static_pick_j=int(picks_val_base_st[i_v]),
                static_scores_8=val_base_sc[i_v],
                static_pinv_8=val_base_pinv[i_v],
                stage2_cache={},
                seed=s,
                fold_k=None,
                device=device,
                dp_f_by_cid=dp_f_by_cid,
                dp_g_by_cid=dp_g_by_cid,
            )
            arms_val_u["H256_ACTIVE_2PLUS2_UNIFIED"][s][cid] = act_val_base["utility"]
            arms_val_inv["H256_ACTIVE_2PLUS2_UNIFIED"][s][cid] = act_val_base["invalid_pick"]

            act_val_t1 = prod.evaluate_active_2plus2_unified_on_context(
                cid=cid,
                bt=base_terms[cid],
                pkt=pkt,
                raw_ctx=raw_cache[cid],
                idx4=idx4_by_seed_cid[s][cid],
                static_pick_j=int(picks_val_t1_st[i_v]),
                static_scores_8=val_t1_sc[i_v],
                static_pinv_8=val_t1_pinv[i_v],
                stage2_cache={},
                seed=s,
                fold_k=None,
                device=device,
                dp_f_by_cid=dp_f_by_cid,
                dp_g_by_cid=dp_g_by_cid,
            )
            arms_val_u["H256_T1_BATCH11_ACTIVE_2PLUS2"][s][cid] = act_val_t1["utility"]
            arms_val_inv["H256_T1_BATCH11_ACTIVE_2PLUS2"][s][cid] = act_val_t1["invalid_pick"]

    summary: Dict[str, Any] = {}
    print("\n=== GATE T1 HYPOTHESIS TEST RESULTS ===")
    for arm_key in arms_oof_u:
        oof_m = prod.compute_partition_metrics(train_rows, arms_oof_u[arm_key], arms_oof_inv[arm_key], r882_u_by_cid)
        val_m = prod.compute_partition_metrics(val_rows, arms_val_u[arm_key], arms_val_inv[arm_key], r882_u_by_cid)
        summary[arm_key] = {"dl_train_oof_1219": oof_m, "dl_val_91": val_m}
        print(
            f"{arm_key:34s} | OOF FamU={oof_m['mean_family_utility']:.6f} ({oof_m['mean_family_gain_vs_r882']:+.6f}) Inv={oof_m['mean_invalid_picks']:.1f} "
            f"| Val91 FamU={val_m['mean_family_utility']:.6f} ({val_m['mean_family_gain_vs_r882']:+.6f}) CtxU={val_m['mean_context_utility']:.6f} Inv={val_m['mean_invalid_picks']:.1f}"
        )

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    ch.save_json(CACHE_DIR / "gate_t1_summary.json", summary)
    return {
        "summary": summary,
        "arms_oof_u": arms_oof_u,
        "arms_val_u": arms_val_u,
        "val_11views": val_11views,
        "val_view_X616_list": val_view_X616_list,
    }


if __name__ == "__main__":
    run_gate_t1_hypothesis_test()
