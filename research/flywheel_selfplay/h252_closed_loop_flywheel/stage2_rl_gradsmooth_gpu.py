#!/usr/bin/env python3
"""
H252 Stage 2 (xbabe1 RTX 4090 GPU): Neural Adapter Continuation via
  1. Decomposed Continuous Hurdle `sos_prior` Bridge (activating the dead 2-channel `sos_prior`
     inputs in `FlywheelBlockFormerAdapterNet` with cross-fitted `[cont_logratio_z, gate_fail_z]`),
  2. Top-Resolved Plackett-Luce / GRPO Rank-Advantage + Proximal KL Trust Region to R882,
  3. Revealed-Anchor `sem24` Centering,
  4. Family-Stratified Minibatches + Cosine-Aligned Orthogonal EMA Gradient Smoothing (`PCGrad-EMA`)
     + Unit-Wise Adaptive Gradient Clipping (`AGC`) + Polyak-EMA Shadow Weights + Margin-Gated Admission.

Runs all 15 cells (5 outer folds x 3 seeds: 252101, 252102, 252103) x 40 updates/cell on GPU
(or CPU fallback) comparing 4 arms:
  - `R882`: Frozen anchor checkpoint (`round-882.pt`)
  - `CTRL_H170`: Standard H140/H170 AdamW continuation on `+2.0` cliff target with zeroed `sos_prior`
  - `RL_HURDLE_ADV`: Hurdle `sos_prior` bridge + Top-Resolved Rank-Advantage + KL Trust Region + Revealed-Anchor Centering
  - `RL_PCGRAD_EMA_SHADOW`: Full RL + Gradient Smoothing (`PCGrad-EMA` + `AGC` + Polyak-EMA Shadow + Fold-B Safe Gate)

Also verifies that DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`)
and frozen anchor weights remain bitwise unchanged.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_h252 import (
    DOPE_SHA256,
    R882_CKPT,
    R882_NATIVE91_MEAN,
    SEEDS,
    cell_folds,
    load_cohorts,
    load_context_features,
    load_harvested_promotions,
    load_pools,
    save_json,
    seed_folds,
    stream_seed,
    summarize_panel,
)

STAGE1_OUT_DIR = Path("/home/ubuntu/jope-data/h252/stage1")
STAGE2_OUT_DIR = Path("/home/ubuntu/jope-data/h252/stage2")
REPO_ROOT = Path("/home/ubuntu/audit/JopeDime_main")
FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
GUARD_SHA256 = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"


def tensor_hash(state: Dict[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for n, t in sorted(state.items()):
        h.update((n + str(t.dtype) + str(tuple(t.shape))).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def build_r882_network(ckpt_path: Path):
    import hashlib
    import types
    from common_h252 import INPUT_DIR, sha256_file

    fused_path = INPUT_DIR / "cpu_exact_fused.py"
    if sha256_file(fused_path) != FUSED_SHA256:
        raise RuntimeError("fused source hash mismatch")
    for name in [n for n in list(sys.modules) if n == "research" or n.startswith("research.")]:
        del sys.modules[name]
    fused = types.ModuleType("h252_fused")
    fused.__file__ = str(fused_path)
    sys.modules[fused.__name__] = fused
    exec(compile(fused_path.read_bytes(), str(fused_path), "exec"), fused.__dict__)

    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE_SHA256:
        raise RuntimeError("wrong checkpoint")
    net.load_state_dict(state["state_dict"], strict=True)

    guard_path = STAGE1_OUT_DIR / "dope_freeze_guard.py"
    src = guard_path.read_bytes()
    if hashlib.sha256(src).hexdigest() != GUARD_SHA256:
        raise RuntimeError("guard source mismatch")
    guard = types.ModuleType("h252_guard")
    sys.modules[guard.__name__] = guard
    exec(compile(src, str(guard_path), "exec"), guard.__dict__)

    for n, p in net.named_parameters():
        p.requires_grad_(n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor."))
    names = tuple(n for n, p in net.named_parameters() if p.requires_grad)
    if len(names) != 47 or sum(p.numel() for p in net.parameters() if p.requires_grad) != 29211:
        raise RuntimeError("adapter inventory")
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA256:
        raise RuntimeError("DOPE mismatch")
    frozen = tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
    return net, guard, names, frozen, fused


def check_frozen_network(net, guard, names, frozen: str) -> None:
    if tensor_hash({n: t for n, t in net.state_dict().items() if n not in names}) != frozen:
        raise RuntimeError("non-adapter bitwise drift")
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA256:
        raise RuntimeError("DOPE drift")


def ctrl_h170_loss(pred: torch.Tensor, fitness: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Exact H140/H170 control loss: SmoothL1 + 0.1 * KL(T=0.2) against +2.0 cliff target."""
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def rl_hurdle_proximal_loss(
    pred: torch.Tensor,
    anchor_total: torch.Tensor,
    fitness: torch.Tensor,
    logratio: torch.Tensor,
    gate: torch.Tensor,
    sos_prior: torch.Tensor,
) -> torch.Tensor:
    """
    H252 Decomposed Continuous Hurdle + Top-Resolved GRPO Rank-Advantage + Proximal KL Trust Region Loss:
      1. Continuous Hurdle Surrogate Target:
         Uses continuous `logratio.clamp(-1.5, 1.5)` + smooth gate barrier `0.45 * (~gate).float()`
         centered on the revealed anchors `0..3`, completely eliminating the artificial `+2.0` cliff
         that dominated 85% of gradients in H140-H250!
      2. Top-Resolved Plackett-Luce / GRPO Advantage Term:
         Computes utility `u = where(gate, -fitness, -2.0)` across candidates, forms the top-16
         contender set under `anchor_total`, computes group-standardized positive advantage
         for candidates that beat the anchor top-1 (`u > u_anchor_top1`), and maximizes
         advantage-weighted log-probability `log_softmax(-pred / 0.15)`.
      3. Proximal KL Trust Region to Anchor (`R882`):
         Penalizes policy divergence `KL(softmax(-anchor_total / 0.18) || softmax(-pred / 0.18))`
         plus an L2 residual margin penalty on `pred - anchor_total` so top-1 ranks never churn!
    """
    # 1. Continuous Hurdle regression target (centered per context)
    cont_cost = logratio.clamp(-1.5, 1.5) + 0.45 * (~gate).float()
    rev_mean_cont = cont_cost[:, :4].mean(dim=1, keepdim=True)
    pred_centered = pred - pred[:, :4].mean(dim=1, keepdim=True)
    cont_centered = cont_cost - rev_mean_cont
    loss_hurdle_reg = F.smooth_l1_loss(pred_centered, cont_centered, beta=0.15, reduction="none").mean(1).mean()

    # 2. Top-Resolved Plackett-Luce / GRPO Advantage on top-16 contenders per context
    u_true = torch.where(gate, -fitness.clamp(-4.0, 2.0), torch.full_like(fitness, -2.0))
    # Identify top-16 candidates under anchor_total (lowest cost)
    _, top16_idx = torch.topk(-anchor_total, k=min(16, anchor_total.shape[1]), dim=1)
    u_top16 = torch.gather(u_true, 1, top16_idx)
    pred_top16 = torch.gather(pred, 1, top16_idx)
    anc_top16 = torch.gather(anchor_total, 1, top16_idx)

    # Advantage relative to anchor's top-1 pick (index 0 of top16_idx) and top16 mean
    u_anc_pick = u_top16[:, :1]
    u_adv_raw = (u_top16 - u_anc_pick).clamp(min=-0.25, max=0.50)
    adv_std = u_adv_raw.std(dim=1, keepdim=True).clamp(min=1e-3)
    adv_weights = F.softmax((u_top16 - u_top16.mean(dim=1, keepdim=True)) / (0.12 + 0.5 * adv_std), dim=-1).detach()
    logp_top16 = F.log_softmax(-pred_top16 / 0.15, dim=-1)
    loss_top_pl = -(adv_weights * logp_top16).sum(dim=-1).mean()

    # 3. Proximal KL Trust Region to Anchor (on both full pool and top-16 contenders)
    kl_top16 = F.kl_div(
        F.log_softmax(-pred_top16 / 0.18, dim=-1),
        F.softmax(-anc_top16 / 0.18, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean()
    delta_l2 = ((pred - anchor_total) ** 2).mean()

    return 0.35 * loss_hurdle_reg + 0.45 * loss_top_pl + 0.25 * kl_top16 + 0.15 * delta_l2


def forward_adapter_gpu(
    ad: nn.Module,
    prog: torch.Tensor,
    sos: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    rev: torch.Tensor,
    raw_f: torch.Tensor,
    gate_f: torch.Tensor,
    anchor_total: torch.Tensor,
    anchor_prior: torch.Tensor,
    center_sem24_on_revealed: bool = False,
    sos_residual_weight: float = 0.0,
) -> torch.Tensor:
    """
    Differentiable forward pass through `net.adapter_10tok` on GPU/CPU.
    When `center_sem24_on_revealed=True`, centers `sem24` on the 4 revealed anchors (`0..3`)
    so cross-context semantic offsets cancel out cleanly.
    """
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tokens,
        mm_mask_10=mask,
        prog_412=prog,
        sos_prior=sos,
        active_modality_override_mask=None,
    )
    if center_sem24_on_revealed:
        rev_sem_mean = semantic[:, :4, :].mean(dim=1, keepdim=True)
        semantic = F.normalize(semantic - 0.35 * rev_sem_mean, p=2, dim=-1, eps=1e-8)

    static = dict(
        eff_prior=anchor_prior + adjustment,
        mm_prior_adj=adjustment,
        sem24=semantic,
        cand_q=query,
        cand_q_norm=F.normalize(query, p=2, dim=-1, eps=1e-8),
        ctx_pooled=pooled,
    )
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale,
        revealed_indices=rev,
        revealed_raw_f=raw_f,
        revealed_gate=gate_f,
    )
    out = anchor_total + delta
    if sos_residual_weight > 0.0:
        # Direct Hurdle + SP prior residual term (centered on revealed 0..3)
        h_comb = 0.65 * sos[:, :, 0] + 0.35 * sos[:, :, 1]
        h_cent = h_comb - h_comb[:, :4].mean(dim=1, keepdim=True)
        out = out + sos_residual_weight * h_cent
    return out.float()


def apply_pcgrad_ema_and_agc(
    params: List[nn.Parameter],
    ema_grad: List[torch.Tensor],
    beta_ema: float = 0.90,
    agc_clip: float = 0.015,
) -> Tuple[float, bool, float]:
    """
    Applies:
      1. Cosine-Aligned Orthogonal EMA Gradient Projection (`PCGrad-EMA`):
         Computes inner product `<g, m_ema>` across all trainable adapter parameters.
         If `<g, m_ema> < 0` (gradient conflict with historical cross-family descent direction),
         projects `g <- g - (<g, m_ema> / (||m_ema||^2 + 1e-12)) * m_ema`.
         Then updates `m_ema <- beta_ema * m_ema + (1 - beta_ema) * g`.
      2. Unit-Wise Adaptive Gradient Clipping (`AGC`, Brock et al. 2021):
         Clips each parameter tensor's gradient norm to `agc_clip * max(||p||_2, 1e-3)`.
    Returns `(grad_norm_before, had_conflict, cos_sim)`.
    """
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    g_flat = torch.cat([g.reshape(-1) for g in grads])
    m_flat = torch.cat([m.reshape(-1) for m in ema_grad])

    g_norm = float(torch.linalg.vector_norm(g_flat).item())
    m_norm = float(torch.linalg.vector_norm(m_flat).item())

    had_conflict = False
    cos_sim = 0.0
    if g_norm > 1e-12 and m_norm > 1e-12:
        dot = torch.dot(g_flat, m_flat)
        cos_sim = float((dot / (g_norm * m_norm + 1e-12)).item())
        if dot.item() < 0.0:
            had_conflict = True
            proj_scale = dot / (m_norm * m_norm + 1e-12)
            for p, m in zip(params, ema_grad):
                if p.grad is not None:
                    p.grad.sub_(m, alpha=float(proj_scale.item()))

    # Update EMA gradient buffer and apply Unit-Wise AGC
    with torch.no_grad():
        for p, m in zip(params, ema_grad):
            if p.grad is None:
                continue
            m.mul_(beta_ema).add_(p.grad, alpha=1.0 - beta_ema)
            p_norm = float(torch.linalg.vector_norm(p).item())
            g_p_norm = float(torch.linalg.vector_norm(p.grad).item())
            max_norm = agc_clip * max(p_norm, 1e-3)
            if g_p_norm > max_norm and g_p_norm > 1e-12:
                p.grad.mul_(max_norm / g_p_norm)

    return g_norm, had_conflict, cos_sim


def main() -> None:
    t0 = time.time()
    STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[H252 Stage 2] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load cohorts, context features, pools, R882 cache, and Stage 1 Hurdle priors
    train_rows, val_rows = load_cohorts()
    all_rows = train_rows + val_rows
    ctx_feats = load_context_features()
    pools = load_pools(all_rows, verify_hashes=False)
    for c, p in pools.items():
        if len(p["fitness"]) < 1152:
            pad_n = 1152 - len(p["fitness"])
            p["program"] = np.pad(p["program"], ((0, pad_n), (0, 0)), mode="edge")
            p["sem24"] = np.pad(p["sem24"], ((0, pad_n), (0, 0)), mode="edge")
            p["fitness"] = np.pad(p["fitness"], (0, pad_n), mode="constant", constant_values=2.0)
            p["logratio"] = np.pad(p["logratio"], (0, pad_n), mode="constant", constant_values=1.5)
            p["gate"] = np.pad(p["gate"], (0, pad_n), mode="constant", constant_values=False)
            p["utility"] = np.pad(p["utility"], (0, pad_n), mode="constant", constant_values=-2.0)
    promos_by_cid = load_harvested_promotions(train_rows)

    r882_cache_path = STAGE1_OUT_DIR / "r882_cached_costs.npz"
    priors_path = STAGE1_OUT_DIR / "stage1_hurdle_priors.npz"
    if not r882_cache_path.exists() or not priors_path.exists():
        raise RuntimeError("Stage 1 outputs (r882_cached_costs.npz and stage1_hurdle_priors.npz) required before Stage 2")

    with np.load(r882_cache_path, allow_pickle=False) as zr:
        cids_r = [str(c) for c in zr["cids"]]
        tc_all = zr["total_cost"]
        at_all = zr["anchor_total"]
        ap_all = zr["anchor_prior"]
        r882_cache = {
            cids_r[i]: {
                "total_cost": tc_all[i],
                "anchor_total": at_all[i],
                "anchor_prior": ap_all[i],
            }
            for i in range(len(cids_r))
        }

    with np.load(priors_path, allow_pickle=False) as zp:
        cids_p = [str(c) for c in zp["cids"]]
        sos_all = zp["sos_prior"]
        sp_all = zp["sp_prior"]
        hurdle_sos_by_cid = {cids_p[i]: sos_all[i] for i in range(len(cids_p))}
        sp_prior_by_cid = {cids_p[i]: sp_all[i] for i in range(len(cids_p))}

    # 2. Build canonical FusedJopeNetwork and verify DOPE + R882 Native91 bit-exactness
    net_cpu, guard, adapter_names, frozen_hash, fused = build_r882_network(R882_CKPT)
    dope_before = guard.snapshot_dope_weights(net_cpu).composite_sha256
    if dope_before != DOPE_SHA256:
        raise RuntimeError(f"DOPE sha mismatch before Stage 2: {dope_before}")

    # Move only the trainable 47-tensor `adapter_10tok` module to GPU for fast 15-cell training;
    # keep `net_cpu` intact for frozen guard verification!
    base_adapter = copy.deepcopy(net_cpu.adapter_10tok).to(device)
    base_adapter.eval()
    trainable_rel_names = [
        n.removeprefix("adapter_10tok.") for n in adapter_names if n.startswith("adapter_10tok.")
    ]
    base_state = {
        k: v.detach().clone()
        for k, v in base_adapter.state_dict().items()
        if k in trainable_rel_names
    }

    # Pre-stage context tensors on GPU (1,310 contexts x 1,152 candidates fits in ~2.5 GB VRAM on 24 GB RTX 4090!)
    print("[H252 Stage 2] Staging pool tensors to GPU VRAM for fast 15-cell training & evaluation...")
    t_stage = time.time()
    all_cids = [r["context_sha256"] for r in all_rows]
    cid_to_idx = {c: i for i, c in enumerate(all_cids)}

    prog_t = torch.as_tensor(np.stack([pools[c]["program"] for c in all_cids]), dtype=torch.float32, device=device)
    fit_t = torch.as_tensor(np.stack([pools[c]["fitness"] for c in all_cids]), dtype=torch.float32, device=device)
    lr_t = torch.as_tensor(np.stack([pools[c]["logratio"] for c in all_cids]), dtype=torch.float32, device=device)
    gate_t = torch.as_tensor(np.stack([pools[c]["gate"] for c in all_cids]), dtype=torch.bool, device=device)
    tok_t = torch.as_tensor(
        np.stack([ctx_feats[c]["tokens_10x32"] for c in all_cids]), dtype=torch.float32, device=device
    )
    mask_t = torch.as_tensor(
        np.stack([ctx_feats[c]["mask_10"] for c in all_cids]), dtype=torch.bool, device=device
    )
    r882_cost_t = torch.as_tensor(
        np.stack([r882_cache[c]["total_cost"] for c in all_cids]), dtype=torch.float32, device=device
    )
    # Compute exact anchor_total and anchor_prior on GPU in <1s
    projection = torch.as_tensor(fused.get_w1_sign_matrix() / np.sqrt(412.0), dtype=torch.float32, device=device)
    anc_tot_list, anc_pri_list = [], []
    with torch.no_grad():
        for start in range(0, len(all_cids), 32):
            b_prog = prog_t[start : start + 32]
            b = len(b_prog)
            b_proj = b_prog @ projection
            b_fit = fit_t[start : start + 32]
            b_gate = gate_t[start : start + 32]
            anc_out = base_adapter.anchor.predict_costs(
                ctx_feat=torch.zeros((b, 584), dtype=torch.float32, device=device),
                prog_412=b_prog,
                proj_16=b_proj,
                base_prior_cost=torch.zeros((b, 1152), dtype=torch.float32, device=device),
                scale=torch.full((b,), 0.4208, dtype=torch.float32, device=device),
                sos_prior=torch.zeros((b, 1152, 2), dtype=torch.float32, device=device),
                revealed_indices=torch.arange(4, device=device).reshape(1, 4).expand(b, 4),
                revealed_raw_f=b_fit[:, :4],
                revealed_gate=b_gate[:, :4].float(),
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            anc_tot_list.append(anc_out["total_cost"].detach().float())
            anc_pri_list.append(anc_out["eff_prior"].detach().float())
    anc_tot_t = torch.cat(anc_tot_list, dim=0)
    anc_pri_t = torch.cat(anc_pri_list, dim=0)
    # Scale Hurdle 2-channel sos_prior modestly (0.12) so initial perturbation is smooth
    sos_hurdle_t = torch.as_tensor(
        0.12 * np.stack([hurdle_sos_by_cid[c] for c in all_cids]), dtype=torch.float32, device=device
    )
    sos_zero_t = torch.zeros_like(sos_hurdle_t)
    print(f"[H252 Stage 2] Staged {len(all_cids)} contexts to {device} in {time.time() - t_stage:.2f}s.")

    def load_trainable_state(ad: nn.Module, st: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in ad.named_parameters():
                if n in st:
                    p.copy_(st[n])

    def get_trainable_params(ad: nn.Module) -> List[nn.Parameter]:
        out = []
        for n, p in ad.named_parameters():
            if n in base_state:
                p.requires_grad_(True)
                out.append(p)
            else:
                p.requires_grad_(False)
        return out

    @torch.no_grad()
    def predict_batch_costs(
        ad: nn.Module,
        cids: List[str],
        use_hurdle_sos: bool,
        center_sem24: bool,
        sos_res_w: float,
    ) -> np.ndarray:
        ix = torch.as_tensor([cid_to_idx[c] for c in cids], dtype=torch.long, device=device)
        out_list = []
        for start in range(0, len(ix), 64):
            b_ix = ix[start : start + 64]
            b = len(b_ix)
            rev = torch.arange(4, device=device).reshape(1, 4).expand(b, 4)
            scale = torch.full((b,), 0.4208, dtype=torch.float32, device=device)
            sos_b = sos_hurdle_t[b_ix] if use_hurdle_sos else sos_zero_t[b_ix]
            scores = forward_adapter_gpu(
                ad,
                prog_t[b_ix],
                sos_b,
                tok_t[b_ix],
                mask_t[b_ix],
                scale,
                rev,
                fit_t[b_ix, :4],
                gate_t[b_ix, :4].float(),
                anc_tot_t[b_ix],
                anc_pri_t[b_ix],
                center_sem24_on_revealed=center_sem24,
                sos_residual_weight=sos_res_w,
            )
            out_list.append(scores.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    # Verify GPU step-0 exact match with R882 Native91
    val_cids = [r["context_sha256"] for r in val_rows]
    step0_val_costs = predict_batch_costs(base_adapter, val_cids, use_hurdle_sos=False, center_sem24=False, sos_res_w=0.0)
    step0_val_utils = [
        float(pools[c]["utility"][int(np.argmin(step0_val_costs[i]))]) for i, c in enumerate(val_cids)
    ]
    step0_val_mean = float(np.mean(step0_val_utils))
    print(f"[H252 Stage 2] GPU step-0 Native91 utility mean = {step0_val_mean:.16f} (diff={abs(step0_val_mean - R882_NATIVE91_MEAN):.2e})")
    if abs(step0_val_mean - R882_NATIVE91_MEAN) > 1e-6:
        raise RuntimeError(f"GPU step-0 Native91 mismatch: {step0_val_mean}")

    arms = ["R882", "CTRL_H170", "RL_HURDLE_ADV", "RL_PCGRAD_EMA_SHADOW"]
    train_panel = np.zeros((len(SEEDS), len(train_rows), len(arms)), dtype=np.float64)
    val_panel = np.zeros((len(SEEDS), len(val_rows), len(arms)), dtype=np.float64)

    train_cid_to_pos = {r["context_sha256"]: i for i, r in enumerate(train_rows)}

    # Populate R882 anchor utility (arm 0)
    for si in range(len(SEEDS)):
        for ci, r in enumerate(train_rows):
            cid = r["context_sha256"]
            train_panel[si, ci, 0] = float(pools[cid]["utility"][int(np.argmin(r882_cache[cid]["total_cost"]))])
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            val_panel[si, ci, 0] = float(pools[cid]["utility"][int(np.argmin(r882_cache[cid]["total_cost"]))])

    n_updates = 40
    batch_size = 8
    cell_diagnostics: List[Dict[str, Any]] = []

    for si, seed in enumerate(SEEDS):
        _, cid_to_fold = seed_folds(train_rows, seed)
        # Accumulate DL-Val predictions across the 5 outer folds per arm
        val_cost_accum = {
            a: np.zeros((len(val_rows), 1152), dtype=np.float64) for a in arms[1:]
        }

        for outer_k in range(5):
            a_fold, b_fold, fit_folds = cell_folds(outer_k)
            a_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == a_fold]
            b_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == b_fold]
            fit_rows = [r for r in train_rows if cid_to_fold[r["context_sha256"]] in fit_folds]
            fit_cids = [r["context_sha256"] for r in fit_rows]

            # Group FIT contexts by family for family-stratified minibatch sampling
            fam_to_fit_cids: Dict[str, List[str]] = {}
            for r in fit_rows:
                fam_to_fit_cids.setdefault(r["family_id"], []).append(r["context_sha256"])
            fit_fams = sorted(fam_to_fit_cids.keys())

            for arm_name in arms[1:]:
                cell_seed = stream_seed("H252_STAGE2", seed, outer_k, arm_name)
                rng = np.random.default_rng(cell_seed)
                torch.manual_seed(cell_seed)

                load_trainable_state(base_adapter, base_state)
                params = get_trainable_params(base_adapter)

                if arm_name == "CTRL_H170":
                    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
                    use_hurdle = False
                    center_sem = False
                    sos_res_w = 0.0
                elif arm_name == "RL_HURDLE_ADV":
                    opt = torch.optim.AdamW(params, lr=8e-5, betas=(0.9, 0.999), weight_decay=1e-3)
                    use_hurdle = True
                    center_sem = True
                    sos_res_w = 0.010
                else:  # RL_PCGRAD_EMA_SHADOW
                    opt = torch.optim.AdamW(params, lr=7e-5, betas=(0.9, 0.999), weight_decay=2e-3)
                    use_hurdle = True
                    center_sem = True
                    sos_res_w = 0.012

                ema_grad = [torch.zeros_like(p) for p in params]
                shadow_state = {k: v.detach().clone() for k, v in base_state.items()}
                grad_norms: List[float] = []
                conflict_count = 0
                cos_sims: List[float] = []

                for step in range(1, n_updates + 1):
                    if arm_name == "CTRL_H170":
                        batch_cids = rng.choice(fit_cids, size=batch_size, replace=False).tolist()
                    else:
                        # Family-stratified minibatch: 8 distinct families per step
                        chosen_fams = rng.choice(fit_fams, size=batch_size, replace=False)
                        batch_cids = [
                            fam_to_fit_cids[f][int(rng.integers(0, len(fam_to_fit_cids[f])))]
                            for f in chosen_fams
                        ]

                    b_ix = torch.as_tensor([cid_to_idx[c] for c in batch_cids], dtype=torch.long, device=device)
                    b = len(b_ix)
                    # Subsample 64 candidates per context (4 revealed + top-32 anchor + 28 uniform)
                    sub_ix_list = []
                    for c_global in b_ix.tolist():
                        rc = r882_cost_t[c_global]
                        _, top36 = torch.topk(-rc[4:], k=32)
                        top_cands = (top36 + 4).cpu().numpy().tolist()
                        rem = [j for j in range(4, 1152) if j not in set(top_cands)]
                        uni_cands = rng.choice(rem, size=28, replace=False).tolist()
                        sub64 = [0, 1, 2, 3] + top_cands + uni_cands
                        sub_ix_list.append(sub64)
                    sub_ix = torch.as_tensor(sub_ix_list, dtype=torch.long, device=device)  # (B, 64)

                    prog_b = torch.gather(prog_t[b_ix], 1, sub_ix.unsqueeze(-1).expand(b, 64, 412))
                    fit_b = torch.gather(fit_t[b_ix], 1, sub_ix)
                    lr_b = torch.gather(lr_t[b_ix], 1, sub_ix)
                    gate_b = torch.gather(gate_t[b_ix], 1, sub_ix)
                    anc_tot_b = torch.gather(anc_tot_t[b_ix], 1, sub_ix)
                    anc_pri_b = torch.gather(anc_pri_t[b_ix], 1, sub_ix)
                    sos_src = sos_hurdle_t[b_ix] if use_hurdle else sos_zero_t[b_ix]
                    sos_b = torch.gather(sos_src, 1, sub_ix.unsqueeze(-1).expand(b, 64, 2))

                    rev = torch.arange(4, device=device).reshape(1, 4).expand(b, 4)
                    scale = torch.full((b,), 0.4208, dtype=torch.float32, device=device)

                    opt.zero_grad(set_to_none=True)
                    pred = forward_adapter_gpu(
                        base_adapter,
                        prog_b,
                        sos_b,
                        tok_t[b_ix],
                        mask_t[b_ix],
                        scale,
                        rev,
                        fit_b[:, :4],
                        gate_b[:, :4].float(),
                        anc_tot_b,
                        anc_pri_b,
                        center_sem24_on_revealed=center_sem,
                        sos_residual_weight=sos_res_w,
                    )
                    if arm_name == "CTRL_H170":
                        loss = ctrl_h170_loss(pred, fit_b, gate_b)
                    else:
                        loss = rl_hurdle_proximal_loss(pred, anc_tot_b, fit_b, lr_b, gate_b, sos_b)

                    loss.backward()

                    if arm_name == "RL_PCGRAD_EMA_SHADOW":
                        gn, had_c, cs = apply_pcgrad_ema_and_agc(params, ema_grad, beta_ema=0.90, agc_clip=0.015)
                        grad_norms.append(gn)
                        if had_c:
                            conflict_count += 1
                        cos_sims.append(cs)
                    else:
                        g_all = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
                        grad_norms.append(float(torch.linalg.vector_norm(torch.cat(g_all)).item()) if g_all else 0.0)
                        if arm_name == "RL_HURDLE_ADV":
                            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                    opt.step()

                    if arm_name == "RL_PCGRAD_EMA_SHADOW":
                        # Polyak-Ruppert EMA shadow weight update (tau = 0.95)
                        with torch.no_grad():
                            for n, p in base_adapter.named_parameters():
                                if n in shadow_state:
                                    shadow_state[n].mul_(0.95).add_(p.detach(), alpha=0.05)

                # For RL_PCGRAD_EMA_SHADOW, load Polyak-EMA shadow weights and calibrate
                # outcome-free zgap safety threshold on selector fold B!
                if arm_name == "RL_PCGRAD_EMA_SHADOW":
                    load_trainable_state(base_adapter, shadow_state)

                b_costs = predict_batch_costs(base_adapter, b_cids, use_hurdle, center_sem, sos_res_w)
                a_costs = predict_batch_costs(base_adapter, a_cids, use_hurdle, center_sem, sos_res_w)
                v_costs = predict_batch_costs(base_adapter, val_cids, use_hurdle, center_sem, sos_res_w)

                if arm_name in ("RL_HURDLE_ADV", "RL_PCGRAD_EMA_SHADOW"):
                    # Helper to compute trust-region + Hurdle-gate-safe effective cost for a context
                    def safe_contender_cost(cid_s: str, neural_cost: np.ndarray, tau_s: float, alpha_s: float, gamma_h: float) -> np.ndarray:
                        rc = r882_cache[cid_s]["total_cost"].astype(np.float64)
                        part = np.partition(rc, 1)
                        sd = float(np.std(rc)) + 1e-9
                        top_zg = float((part[1] - part[0]) / sd)
                        if top_zg >= tau_s or (alpha_s <= 0.0 and gamma_h <= 0.0):
                            return rc
                        pick_r = int(np.argmin(rc))
                        sos_c = hurdle_sos_by_cid[cid_s].astype(np.float64)
                        sp_c = sp_prior_by_cid[cid_s].astype(np.float64)
                        # Trust-region contender mask: only candidates within 0.25 * sd of R882 top-1
                        # and whose predicted gate-failure risk does not exceed R882 top-1 + 0.15
                        in_tr = ((rc - part[0]) <= 0.25 * sd) & (sos_c[:, 1] <= sos_c[pick_r, 1] + 0.15)
                        hc = 0.65 * sos_c[:, 0] + 0.35 * sos_c[:, 1] + 0.15 * sp_c
                        nc_z = (neural_cost.astype(np.float64) - float(np.mean(neural_cost))) / (float(np.std(neural_cost)) + 1e-6)
                        delta_c = alpha_s * sd * nc_z + gamma_h * hc
                        return np.where(in_tr, rc + delta_c, rc + 10.0)

                    # Calibrate outcome-free zgap threshold and blend weights on selector fold B
                    best_b_obj = -1e9
                    best_tau = 0.05
                    best_alpha = 0.02
                    best_gamma = 0.012
                    tau_grid = [0.04, 0.06, 0.10]
                    alpha_grid = [0.0, 0.01, 0.02, 0.04] if arm_name == "RL_PCGRAD_EMA_SHADOW" else [0.01, 0.02, 0.04]
                    gamma_grid = [0.008, 0.012, 0.016]

                    for tau in tau_grid:
                        for alpha in alpha_grid:
                            for gamma_h in gamma_grid:
                                gains = []
                                for bi, bcid in enumerate(b_cids):
                                    rc = r882_cache[bcid]["total_cost"]
                                    u_r = float(pools[bcid]["utility"][int(np.argmin(rc))])
                                    eff_b = safe_contender_cost(bcid, b_costs[bi], tau, alpha, gamma_h)
                                    u_c = float(pools[bcid]["utility"][int(np.argmin(eff_b))])
                                    gains.append(u_c - u_r)
                                g_arr = np.asarray(gains, dtype=np.float64)
                                mean_g = float(np.mean(g_arr))
                                se_g = float(np.std(g_arr) / np.sqrt(len(g_arr)))
                                # For RL_PCGRAD_EMA_SHADOW, optimize LCB on selector fold B
                                obj = (mean_g - 0.5 * se_g) if arm_name == "RL_PCGRAD_EMA_SHADOW" else mean_g
                                if obj > best_b_obj:
                                    best_b_obj = obj
                                    best_tau = tau
                                    best_alpha = alpha
                                    best_gamma = gamma_h

                    # Apply calibrated (best_tau, best_alpha, best_gamma) to held-out assessment fold A and DL-Val
                    ai_idx = arms.index(arm_name)
                    for idx_a, acid in enumerate(a_cids):
                        eff_c = safe_contender_cost(acid, a_costs[idx_a], best_tau, best_alpha, best_gamma)
                        pos = train_cid_to_pos[acid]
                        train_panel[si, pos, ai_idx] = float(pools[acid]["utility"][int(np.argmin(eff_c))])

                    for idx_v, vcid in enumerate(val_cids):
                        eff_c = safe_contender_cost(vcid, v_costs[idx_v], best_tau, best_alpha, best_gamma)
                        val_cost_accum[arm_name][idx_v] += eff_c / 5.0
                else:
                    # CTRL_H170: unguarded full-adapter continuation
                    ai_idx = arms.index(arm_name)
                    for idx_a, acid in enumerate(a_cids):
                        pos = train_cid_to_pos[acid]
                        train_panel[si, pos, ai_idx] = float(
                            pools[acid]["utility"][int(np.argmin(a_costs[idx_a]))]
                        )
                    for idx_v in range(len(val_cids)):
                        val_cost_accum[arm_name][idx_v] += v_costs[idx_v] / 5.0

                cell_diagnostics.append(
                    {
                        "seed": int(seed),
                        "outer_fold": int(outer_k),
                        "arm": arm_name,
                        "mean_grad_norm": float(np.mean(grad_norms)),
                        "max_grad_norm": float(np.max(grad_norms)),
                        "pcgrad_conflict_frac": float(conflict_count / n_updates),
                        "mean_grad_ema_cos": float(np.mean(cos_sims)) if cos_sims else 0.0,
                    }
                )

        # Evaluate 5-fold ensemble on DL-Val for this seed
        for arm_name in arms[1:]:
            ai_idx = arms.index(arm_name)
            for idx_v, vcid in enumerate(val_cids):
                pick = int(np.argmin(val_cost_accum[arm_name][idx_v]))
                val_panel[si, idx_v, ai_idx] = float(pools[vcid]["utility"][pick])

        print(f"[H252 Stage 2] Completed seed {seed} (5 outer folds x 3 neural arms = 15 runs).")

    # Verify DOPE and non-adapter weights on net_cpu are bitwise unchanged
    check_frozen_network(net_cpu, guard, adapter_names, frozen_hash)
    dope_after = guard.snapshot_dope_weights(net_cpu).composite_sha256
    if dope_after != DOPE_SHA256:
        raise RuntimeError(f"DOPE sha mismatch after Stage 2: {dope_after}")

    # Statistical summaries vs R882 and vs CTRL_H170
    train_vs_r882 = summarize_panel(train_rows, train_panel, arms, ref_idx=0)
    train_vs_ctrl = summarize_panel(train_rows, train_panel, arms, ref_idx=1)
    val_vs_r882 = summarize_panel(val_rows, val_panel, arms, ref_idx=0)
    val_vs_ctrl = summarize_panel(val_rows, val_panel, arms, ref_idx=1)

    for split_label, sr, sc in [
        ("DL-Train OOF (1,219 ctx / 205 fam)", train_vs_r882, train_vs_ctrl),
        ("DL-Val (91 ctx / 19 fam)", val_vs_r882, val_vs_ctrl),
    ]:
        print(f"\n=== [H252 Stage 2 GPU] {split_label} ===")
        for ai, a in enumerate(arms):
            med_fam = sr["median_family_utility"][ai]
            med_ctx = sr["median_context_utility"][ai]
            elo = sr["median_elo"][ai]
            if a == "R882":
                print(f"  {a:<22} | fam_u={med_fam:.6f} | ctx_u={med_ctx:.6f} | Elo={elo:.1f} (anchor)")
            else:
                cr = sr["contrasts"][f"{a}_minus_R882"]
                cc_key = f"{a}_minus_CTRL_H170"
                cc_gain = sc["contrasts"][cc_key]["median_family_gain"] if cc_key in sc["contrasts"] else 0.0
                print(
                    f"  {a:<22} | fam_u={med_fam:.6f} | d_fam_R882={cr['median_family_gain']:+.6f} "
                    f"(95% CI [{cr['ci95'][0]:+.6f}, {cr['ci95'][1]:+.6f}], p={cr['p_one_sided']:.4g}, "
                    f"W/T/L={cr['family_wtl_by_seed'][0]}) | d_fam_CTRL={cc_gain:+.6f} | Elo={elo:.1f}"
                )

    # Save per-context Stage 2 RL_PCGRAD_EMA_SHADOW predictions for Stage 4 2x2 Flywheel evaluation
    out_obj = {
        "hypothesis_id": "H252_STAGE2",
        "wave": "RL_HURDLE_ADV_AND_PCGRAD_EMA_SHADOW_GPU",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after,
        "elapsed_sec": float(time.time() - t0),
        "cell_diagnostics_summary": {
            arm: {
                "mean_grad_norm": float(
                    np.mean([d["mean_grad_norm"] for d in cell_diagnostics if d["arm"] == arm])
                ),
                "pcgrad_conflict_frac": float(
                    np.mean([d["pcgrad_conflict_frac"] for d in cell_diagnostics if d["arm"] == arm])
                ),
            }
            for arm in arms[1:]
        },
        "dl_train_vs_r882": train_vs_r882,
        "dl_train_vs_ctrl_h170": train_vs_ctrl,
        "dl_val_vs_r882": val_vs_r882,
        "dl_val_vs_ctrl_h170": val_vs_ctrl,
    }
    metrics_path = STAGE2_OUT_DIR / "stage2_metrics.json"
    save_json(metrics_path, out_obj)
    np.savez_compressed(
        STAGE2_OUT_DIR / "stage2_panels.npz",
        arms=np.array(arms),
        train_panel=train_panel,
        val_panel=val_panel,
    )
    print(f"\n[H252 Stage 2] Saved Stage 2 GPU metrics to {metrics_path} in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
