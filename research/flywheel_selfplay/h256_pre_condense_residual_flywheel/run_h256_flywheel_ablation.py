#!/usr/bin/env python3
"""H256 Step 2: Staged 6-Arm Pre-Condense Residual Bypass Flywheel Training,
Unseen-Family Evaluation, Context Sensitivity Diagnostic, and Per-Block Attribution.

Trains our best known flywheel method (Causal 2+2 `ACTIVE_2PLUS2_HURDLE` + Hurdle-KL ListNet +
Pairwise Hard-Negative Margin + PCGrad-EMA) across all 6 Staged Ablation Arms (A..F):
  - Arm A: `A_FROZEN_10TOK_BASELINE` (detached 10x32 Stage-1B tokens + Causal 2+2 baseline)
  - Arm B: `B_CAPACITY_CTRL_10TOK` (same extra downstream residual head on detached 10x32 tokens only)
  - Arm C: `C_LEARNED_COMPACT_BRIDGE_629` (learned 629 -> 10x32 compact bridge)
  - Arm D: `D_PRE_CONDENSE_FULL629_20TOK` (coordinate-preserving Full629 20x32 + 11D log-norm scale side-channel)
  - Arm E: `E_FULL629_PLUS_HIDDEN256_28TOK` (Full629 20x32 + pre-head DOPE router Hidden256 8x32)
  - Arm F: `F_FULL629_HIDDEN256_COL64X17` (Full629 20x32 + Hidden256 8x32 + Masked Per-Column 64x18 -> 64x32)

Strict Guardrails:
  - DOPE weights locked (`requires_grad=False`, `.eval()` mode hook, IEEE-754 SHA-256 `c51a56d0...` verified before, during, and after).
  - Zero-initialized residual readout reproduces R882 Native91 (`1.1377941898164525`) to `0.00e+00` at step 0.
  - Family-grouped 5-fold OOF on DL-Train (1,219 contexts / 205 families); DL-Val (91 contexts / 52 unseen families) is never touched during training, standardization, or hyperparameter selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
OUT_DIR = Path("/home/ubuntu/jope-data/h256")
CACHE_PATH = OUT_DIR / "cache" / "pre_condense_context_features.npz"

sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import stage1_hurdle_sos_critic as s1  # noqa: E402
import part_a_causal_2plus2_neural as p255  # noqa: E402
from pre_condense_models import (  # noqa: E402
    ARM_NAMES,
    BLOCK_SLICES,
    FoldScaler,
    PreCondenseResidualBridge,
)

SEEDS: Tuple[int, ...] = (256101, 256102, 256103)
FOLD_SEED: int = 256140
BOOT_SEED: int = 256150


def load_pre_condense_cache(path: Path = CACHE_PATH) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        cids = [str(c) for c in z["cids"]]
        tok10 = z["tokens_10x32"].astype(np.float32)
        mask10 = z["mask_10"].astype(bool)
        lm48 = z["lm48_z"].astype(np.float32)
        dope629 = z["dope629"].astype(np.float32)
        hidden256 = z["hidden256"].astype(np.float32)
        col_lm = z["col_lm_64x17"].astype(np.float32)
        col_mask = z["col_mask_64"].astype(bool)
        ctx584 = z["ctx_feat_584"].astype(np.float32)
        for i, cid in enumerate(cids):
            out[cid] = {
                "tokens_10x32": tok10[i],
                "mask_10": mask10[i],
                "lm48_z": lm48[i],
                "dope629": dope629[i],
                "hidden256": hidden256[i],
                "col_lm_64x17": col_lm[i],
                "col_mask_64": col_mask[i],
                "ctx_feat_584": ctx584[i],
            }
    return out


def compute_context_base_terms(
    cid: str,
    pool: Dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    st: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    """Precompute per-context candidate tensors and hurdle safety terms."""
    u = pool["utility"].astype(np.float64)
    fitness = pool["fitness"].astype(np.float64)
    logratio = pool["logratio"].astype(np.float64)
    gate = pool["gate"].astype(bool)
    sem24 = pool["sem24"].astype(np.float64)

    pick_r882 = int(np.argmin(rcost))
    top_zgap = s1.compute_r882_top_zgap(rcost)

    gate_risk = sos_prior[:, 1].astype(np.float64)
    unsafe_gate_mask = gate_risk > (gate_risk[pick_r882] + 0.10)
    gate_safety_penalty = np.where(unsafe_gate_mask, 10.0, 0.0)
    h_cost = (
        0.65 * sos_prior[:, 0].astype(np.float64)
        + 0.35 * gate_risk
        + 0.15 * sp_prior.astype(np.float64)
        + gate_safety_penalty
    )

    s_mean = sem24.mean(axis=0, keepdims=True)
    s_std = sem24.std(axis=0, keepdims=True) + 1e-6
    X_sem = (sem24 - s_mean) / s_std

    # Top-28 lowest cost candidates under rcost and rcost+h_cost + top-8 oracle gate-passing winners
    n = len(rcost)
    top_rcost_idx = np.argsort(rcost)[: min(28, n)]
    top_hcost_idx = np.argsort(rcost + 0.010 * h_cost)[: min(28, n)]
    base_contenders = np.unique(np.concatenate([top_rcost_idx, top_hcost_idx]))
    gate_pass_idx = np.where(gate)[0]
    if len(gate_pass_idx) > 0:
        best_oracle_idx = gate_pass_idx[np.argsort(-u[gate_pass_idx])[: min(8, len(gate_pass_idx))]]
        hard_slate = np.unique(np.concatenate([base_contenders, best_oracle_idx]))
    else:
        hard_slate = base_contenders

    return {
        "u": u,
        "u_max": float(np.max(u)),
        "fitness": fitness,
        "logratio": logratio,
        "gate": gate,
        "rcost": rcost.astype(np.float64),
        "h_cost": h_cost,
        "X_sem": X_sem,
        "top_zgap": float(top_zgap),
        "pick_r882": pick_r882,
        "u_r882": float(u[pick_r882]),
        "hard_slate": hard_slate,
        "cand_q_t": torch.as_tensor(st["cand_q"], dtype=torch.float32, device=device).unsqueeze(0),
        "sem24_t": torch.as_tensor(st["sem24"], dtype=torch.float32, device=device).unsqueeze(0),
        "eff_prior_t": torch.as_tensor(st["eff_prior"], dtype=torch.float32, device=device).unsqueeze(0),
        "gate_risk_t": torch.as_tensor(gate_risk, dtype=torch.float32, device=device).unsqueeze(0),
        "sp_prior_t": torch.as_tensor(sp_prior, dtype=torch.float32, device=device).unsqueeze(0),
    }


def run_2plus2_active_reveal_step(
    bt: Dict[str, Any],
    idx4: np.ndarray,
    delta_prior_np: Optional[np.ndarray] = None,
    alpha_prior: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executes Round 1 (2 initial probes -> interim cost -> 2 active probes -> rev4)
    and returns `(rev4, cost_act, y4_comb, y4_f)`.
    """
    rcost = bt["rcost"]
    h_cost = bt["h_cost"]
    X = bt["X_sem"]
    fitness = bt["fitness"]
    logratio = bt["logratio"]
    gate = bt["gate"]

    rcost_adj = rcost if (delta_prior_np is None or alpha_prior == 0.0) else (rcost + alpha_prior * delta_prior_np)

    idx_a = idx4[:2]
    ya_cost = np.where(gate[idx_a], fitness[idx_a], 2.0)
    ya_c = ya_cost - float(np.mean(ya_cost))
    diff_a = X[:, None, :] - X[idx_a][None, :, :]
    sq_a = np.sum(diff_a * diff_a, axis=-1) / 24.0
    Ka = np.exp(-sq_a / (2.0 * (2.2 ** 2)))
    wa = Ka / (np.sum(Ka, axis=1, keepdims=True) + 0.2)
    h_ca = h_cost - float(np.mean(h_cost[idx_a]))
    interim_cost = rcost_adj + 0.035 * (wa @ ya_c) + 0.010 * h_ca

    order = np.argsort(interim_cost)
    chosen_b: List[int] = []
    for cand in order:
        ci = int(cand)
        if ci not in idx_a:
            chosen_b.append(ci)
            break
    rev_so_far = list(idx_a) + chosen_b
    best_b2 = -1
    best_acq = 1e18
    for cand in order[:6]:
        ci = int(cand)
        if ci in rev_so_far:
            continue
        d_min = float(np.min(np.sum((X[ci : ci + 1] - X[rev_so_far]) ** 2, axis=1) / 24.0))
        acq = float(interim_cost[ci]) - 0.015 * np.sqrt(d_min)
        if acq < best_acq:
            best_acq = acq
            best_b2 = ci
    if best_b2 >= 0:
        chosen_b.append(best_b2)

    rev4 = np.array(list(idx_a) + chosen_b, dtype=np.int64)
    y4_f = np.where(gate[rev4], fitness[rev4], 2.0)
    y4_lr = np.clip(logratio[rev4], -1.5, 1.5)
    y4_comb = 0.70 * (y4_f - float(np.mean(y4_f))) + 0.30 * (y4_lr - float(np.mean(y4_lr)))

    diff4 = X[:, None, :] - X[rev4][None, :, :]
    sq4 = np.sum(diff4 * diff4, axis=-1) / 24.0
    K4 = 0.6 * np.exp(-sq4 / (2.0 * (1.8 ** 2))) + 0.4 * np.exp(-sq4 / (2.0 * (3.2 ** 2)))
    w4 = K4 / (np.sum(K4, axis=1, keepdims=True) + 0.15)
    h_c4_act = h_cost - float(np.mean(h_cost[rev4]))

    cost_act = rcost + 0.05 * (w4 @ y4_comb) + 0.010 * h_c4_act
    for j_pos, c_idx in enumerate(rev4):
        cost_act[c_idx] += 0.08 * float(y4_f[j_pos] - np.mean(y4_f))
    return rev4, cost_act, y4_comb, y4_f


def evaluate_context_with_model(
    bt: Dict[str, Any],
    ctx_tensors: Optional[Dict[str, torch.Tensor]],
    models: List[PreCondenseResidualBridge],
    alpha_prior: float,
    alpha_traj: float,
    idx4: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates a single context with an ensemble of 1 or more `PreCondenseResidualBridge` models."""
    u = bt["u"]
    gate = bt["gate"]
    rcost = bt["rcost"]

    if bt["top_zgap"] >= 0.25 or not models or (alpha_prior == 0.0 and alpha_traj == 0.0):
        if bt["top_zgap"] < 0.25:
            rev4, cost_final, _, _ = run_2plus2_active_reveal_step(bt, idx4, None, 0.0)
        else:
            cost_final = rcost
        order = np.argsort(cost_final)
        pick1 = int(order[0])
        top4 = order[:4]
        return {
            "utility": float(u[pick1]),
            "top1_regret": float(bt["u_max"] - u[pick1]),
            "top4_regret": float(bt["u_max"] - np.max(u[top4])),
            "oracle_hit": bool(bt["u_max"] - u[pick1] <= 1e-4),
            "invalid_pick": bool(not gate[pick1]),
            "delta_prior_rms": 0.0,
            "delta_traj_rms": 0.0,
            "delta_prior_vec": np.zeros_like(rcost),
        }

    with torch.no_grad():
        dp_list = []
        attn_list = []
        bilin_list = []
        for m in models:
            dp, attn_c, bilin_c = m.forward_prior_and_context(
                ctx_tensors,
                bt["cand_q_t"],
                bt["sem24_t"],
                bt["gate_risk_t"],
                bt["sp_prior_t"],
            )
            dp_list.append(dp[0].cpu().numpy().astype(np.float64))
            attn_list.append(attn_c)
            bilin_list.append(bilin_c)
        delta_prior_np = np.mean(dp_list, axis=0)

        rev4, cost_act, y4_comb, y4_f = run_2plus2_active_reveal_step(
            bt, idx4, delta_prior_np, alpha_prior
        )

        if alpha_traj > 0.0:
            base_plus_prior_t = torch.as_tensor(
                cost_act + alpha_prior * delta_prior_np, dtype=torch.float32, device=device
            ).unsqueeze(0)
            rev_idx_t = torch.as_tensor(rev4, dtype=torch.long, device=device).unsqueeze(0)
            rev_y_comb_t = torch.as_tensor(y4_comb, dtype=torch.float32, device=device).unsqueeze(0)
            rev_y_f_t = torch.as_tensor(y4_f, dtype=torch.float32, device=device).unsqueeze(0)
            rev_g_t = torch.as_tensor(gate[rev4], dtype=torch.float32, device=device).unsqueeze(0)

            dt_list = []
            for m, attn_c, bilin_c in zip(models, attn_list, bilin_list):
                dt = m.forward_causal_trajectory_residual(
                    attn_c,
                    bilin_c,
                    bt["cand_q_t"],
                    bt["sem24_t"],
                    base_plus_prior_t,
                    rev_idx_t,
                    rev_y_comb_t,
                    rev_y_f_t,
                    rev_g_t,
                )
                dt_list.append(dt[0].cpu().numpy().astype(np.float64))
            delta_traj_np = np.mean(dt_list, axis=0)
        else:
            delta_traj_np = np.zeros_like(delta_prior_np)

    # Combine with gate-safety barrier so unsafe candidates are never promoted
    cost_final = (
        cost_act
        + alpha_prior * delta_prior_np
        + alpha_traj * delta_traj_np
        + 0.005 * bt["h_cost"]
    )
    order = np.argsort(cost_final)
    pick1 = int(order[0])
    top4 = order[:4]
    return {
        "utility": float(u[pick1]),
        "top1_regret": float(bt["u_max"] - u[pick1]),
        "top4_regret": float(bt["u_max"] - np.max(u[top4])),
        "oracle_hit": bool(bt["u_max"] - u[pick1] <= 1e-4),
        "invalid_pick": bool(not gate[pick1]),
        "delta_prior_rms": float(np.sqrt(np.mean(delta_prior_np ** 2))),
        "delta_traj_rms": float(np.sqrt(np.mean(delta_traj_np ** 2))),
        "delta_prior_vec": delta_prior_np,
    }


def train_pre_condense_bridge_fold(
    arm: str,
    fit_cids: List[str],
    sel_cids: List[str],
    base_terms: Dict[str, Dict[str, Any]],
    ctx_tensors_map: Dict[str, Dict[str, torch.Tensor]],
    idx4_map: Dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    pretrained_s3_state: Optional[Dict[str, torch.Tensor]] = None,
    guard: Any = None,
    dope_net: Any = None,
) -> Tuple[PreCondenseResidualBridge, float, float, Dict[str, Any]]:
    """Trains `PreCondenseResidualBridge` for `arm` on `fit_cids` using Causal 2+2 slates +
    Hurdle-KL ListNet + Hard-Negative Pairwise Margin + PCGrad-EMA, and selects
    `(alpha_prior, alpha_traj)` on `sel_cids`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    model = PreCondenseResidualBridge(arm=arm, d_token=32, n_heads=4).to(device)
    loaded_s3_tensors = 0
    if arm == "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED" and pretrained_s3_state is not None:
        loaded_s3_tensors = model.load_pretrained_s3_context_encoder(pretrained_s3_state)
    model.eval()

    # Verify exact zero-initialization parity before any gradient update
    sample_cid = fit_cids[0]
    with torch.no_grad():
        dp0, attn0, bilin0 = model.forward_prior_and_context(
            ctx_tensors_map[sample_cid],
            base_terms[sample_cid]["cand_q_t"],
            base_terms[sample_cid]["sem24_t"],
            base_terms[sample_cid]["gate_risk_t"],
            base_terms[sample_cid]["sp_prior_t"],
        )
        step0_max_abs = float(dp0.abs().max().item())
        if step0_max_abs != 0.0:
            raise RuntimeError(f"Zero-init violation on {arm}: step0_max_abs={step0_max_abs}")

    # Filter to uncertain training contexts (`top_zgap < 0.25`) where 2+2 active flywheel operates
    active_fit_cids = [c for c in fit_cids if base_terms[c]["top_zgap"] < 0.25]
    if not active_fit_cids:
        active_fit_cids = fit_cids

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(trainable_params, lr=1.5e-3, weight_decay=2e-4)
    if guard is not None and dope_net is not None:
        guard.assert_optimizer_excludes_dope(opt, dope_net)
    rng = np.random.default_rng(seed)

    steps = 32
    batch_size = 24
    losses: List[float] = []
    pcgrad_cosines: List[float] = []

    model.train()
    for step in range(steps):
        batch_cids = rng.choice(
            active_fit_cids, size=min(batch_size, len(active_fit_cids)), replace=False
        ).tolist()
        half = len(batch_cids) // 2
        sub_batches = [batch_cids[:half], batch_cids[half:]]
        sub_grads = []
        step_loss_val = 0.0

        for sub in sub_batches:
            opt.zero_grad(set_to_none=True)
            sub_loss = torch.tensor(0.0, device=device)

            for c in sub:
                bt = base_terms[c]
                ct = ctx_tensors_map[c]
                idx4 = idx4_map[c]

                active_idx = np.unique(np.concatenate([bt["hard_slate"], idx4]))
                idx_t = torch.as_tensor(active_idx, dtype=torch.long, device=device)
                cand_q_sub = bt["cand_q_t"][:, idx_t, :]
                sem24_sub = bt["sem24_t"][:, idx_t, :]
                gate_risk_sub = bt["gate_risk_t"][:, idx_t]
                sp_prior_sub = bt["sp_prior_t"][:, idx_t]

                dp_t, attn_c, bilin_c = model.forward_prior_and_context(
                    ct,
                    cand_q_sub,
                    sem24_sub,
                    gate_risk_sub,
                    sp_prior_sub,
                )
                dp_sub = dp_t[0]  # [M_sub]
                dp_full_np = np.zeros(len(bt["rcost"]), dtype=np.float64)
                dp_full_np[active_idx] = dp_sub.detach().cpu().numpy().astype(np.float64)

                rev4, cost_act, y4_comb, y4_f = run_2plus2_active_reveal_step(
                    bt, idx4, dp_full_np, alpha_prior=0.03
                )

                pos_map = {int(g_i): loc_i for loc_i, g_i in enumerate(active_idx.tolist())}
                if not all(int(r_i) in pos_map for r_i in rev4):
                    active_idx = np.unique(np.concatenate([active_idx, rev4]))
                    idx_t = torch.as_tensor(active_idx, dtype=torch.long, device=device)
                    cand_q_sub = bt["cand_q_t"][:, idx_t, :]
                    sem24_sub = bt["sem24_t"][:, idx_t, :]
                    gate_risk_sub = bt["gate_risk_t"][:, idx_t]
                    sp_prior_sub = bt["sp_prior_t"][:, idx_t]
                    dp_t, attn_c, bilin_c = model.forward_prior_and_context(
                        ct, cand_q_sub, sem24_sub, gate_risk_sub, sp_prior_sub
                    )
                    dp_sub = dp_t[0]
                    pos_map = {int(g_i): loc_i for loc_i, g_i in enumerate(active_idx.tolist())}

                rev4_local = np.array([pos_map[int(r_i)] for r_i in rev4], dtype=np.int64)
                base_plus_prior_sub = (
                    torch.as_tensor(cost_act[active_idx], dtype=torch.float32, device=device)
                    + 0.03 * dp_sub
                )
                rev_idx_t = torch.as_tensor(rev4_local, dtype=torch.long, device=device).unsqueeze(0)
                rev_y_comb_t = torch.as_tensor(y4_comb, dtype=torch.float32, device=device).unsqueeze(0)
                rev_y_f_t = torch.as_tensor(y4_f, dtype=torch.float32, device=device).unsqueeze(0)
                rev_g_t = torch.as_tensor(bt["gate"][rev4], dtype=torch.float32, device=device).unsqueeze(0)

                dt_t = model.forward_causal_trajectory_residual(
                    attn_c,
                    bilin_c,
                    cand_q_sub,
                    sem24_sub,
                    base_plus_prior_sub.unsqueeze(0),
                    rev_idx_t,
                    rev_y_comb_t,
                    rev_y_f_t,
                    rev_g_t,
                )
                dt_sub = dt_t[0]  # [M_sub]

                # Strictly causal unrevealed candidate slate on hard contenders + oracle winners
                slate_idx = bt["hard_slate"]
                rev_set = set(rev4.tolist())
                unrev_global = np.array([i for i in slate_idx if i not in rev_set], dtype=np.int64)
                if len(unrev_global) < 4:
                    continue
                unrev_local = np.array([pos_map[int(i)] for i in unrev_global], dtype=np.int64)

                u_slate = bt["u"][unrev_global]
                g_slate = bt["gate"][unrev_global]
                c_act_slate = torch.as_tensor(
                    cost_act[unrev_global] + 0.005 * bt["h_cost"][unrev_global],
                    dtype=torch.float32,
                    device=device,
                )
                pred_slate = c_act_slate + 0.04 * dp_sub[unrev_local] + 0.04 * dt_sub[unrev_local]

                # Advantage-weighted ListNet KL over unrevealed contenders
                u_best_slate = float(np.max(u_slate))
                u_rel = np.where(g_slate, u_slate - u_best_slate, -0.50).astype(np.float32)
                target_probs = F.softmax(torch.as_tensor(u_rel / 0.04, device=device), dim=-1)
                log_q = F.log_softmax(-pred_slate / 0.04, dim=-1)
                loss_kl = F.kl_div(log_q, target_probs, reduction="batchmean")

                # Hard-negative pairwise margin loss: best gate-passing winner vs baseline top pick
                best_pos_local = int(np.argmax(u_slate))
                base_pick_local = int(np.argmin(cost_act[unrev_global]))
                loss_margin = torch.tensor(0.0, device=device)
                if u_slate[best_pos_local] > u_slate[base_pick_local] + 1e-4:
                    gain = float(min(u_slate[best_pos_local] - u_slate[base_pick_local], 0.25))
                    diff_cost = pred_slate[best_pos_local] - pred_slate[base_pick_local]
                    loss_margin = F.relu(0.02 + diff_cost) * (5.0 * gain)

                reg_l2 = 1e-3 * (torch.mean(dp_sub ** 2) + torch.mean(dt_sub ** 2))
                sub_loss = sub_loss + loss_kl + loss_margin + reg_l2

            sub_loss = sub_loss / max(len(sub), 1)
            sub_loss.backward()
            step_loss_val += float(sub_loss.detach()) * 0.5
            g_vec = torch.cat(
                [
                    (p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten())
                    for p in trainable_params
                ]
            )
            sub_grads.append(g_vec)

        g1, g2 = sub_grads[0], sub_grads[1]
        dot = float(torch.dot(g1, g2))
        norm1 = float(torch.norm(g1)) + 1e-12
        norm2 = float(torch.norm(g2)) + 1e-12
        cos_val = dot / (norm1 * norm2)
        pcgrad_cosines.append(cos_val)
        if dot < 0.0:
            g1_proj = g1 - (dot / (norm2 ** 2)) * g2
            g2_proj = g2 - (dot / (norm1 ** 2)) * g1
            g_final = 0.5 * (g1_proj + g2_proj)
        else:
            g_final = 0.5 * (g1 + g2)

        offset = 0
        for p in trainable_params:
            numel = p.numel()
            p.grad = g_final[offset : offset + numel].view_as(p).clone()
            offset += numel
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        losses.append(step_loss_val)

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema_params:
                    ema_params[n].mul_(0.90).add_(p.detach(), alpha=0.10)

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema_params:
                p.copy_(ema_params[n])
    model.eval()

    # Inner selector fold trust-region search over (alpha_prior, alpha_traj),
    # caching (delta_prior_np, attn_c, bilin_c) once per active selector context.
    candidate_grid = [
        (0.0, 0.0),
        (0.015, 0.0),
        (0.030, 0.0),
        (0.045, 0.0),
        (0.015, 0.015),
        (0.030, 0.020),
        (0.045, 0.030),
        (0.060, 0.030),
    ]
    unique_aps = sorted({ap for ap, _ in candidate_grid})
    sel_cache: Dict[str, Any] = {}
    with torch.no_grad():
        for c in sel_cids:
            bt = base_terms[c]
            if bt["top_zgap"] >= 0.25:
                sel_cache[c] = {"inactive_u": float(bt["u"][bt["pick_r882"]])}
                continue
            dp_t, attn_c, bilin_c = model.forward_prior_and_context(
                ctx_tensors_map[c],
                bt["cand_q_t"],
                bt["sem24_t"],
                bt["gate_risk_t"],
                bt["sp_prior_t"],
            )
            dp_np = dp_t[0].cpu().numpy().astype(np.float64)
            by_ap: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            for ap_val in unique_aps:
                rev4, cost_act, y4_comb, y4_f = run_2plus2_active_reveal_step(
                    bt, idx4_map[c], dp_np, ap_val
                )
                if ap_val == 0.0:
                    dt_np = np.zeros_like(dp_np)
                else:
                    base_plus_prior_t = torch.as_tensor(
                        cost_act + ap_val * dp_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    rev_idx_t = torch.as_tensor(rev4, dtype=torch.long, device=device).unsqueeze(0)
                    rev_y_comb_t = torch.as_tensor(y4_comb, dtype=torch.float32, device=device).unsqueeze(0)
                    rev_y_f_t = torch.as_tensor(y4_f, dtype=torch.float32, device=device).unsqueeze(0)
                    rev_g_t = torch.as_tensor(bt["gate"][rev4], dtype=torch.float32, device=device).unsqueeze(0)
                    dt_t = model.forward_causal_trajectory_residual(
                        attn_c,
                        bilin_c,
                        bt["cand_q_t"],
                        bt["sem24_t"],
                        base_plus_prior_t,
                        rev_idx_t,
                        rev_y_comb_t,
                        rev_y_f_t,
                        rev_g_t,
                    )
                    dt_np = dt_t[0].cpu().numpy().astype(np.float64)
                by_ap[ap_val] = (cost_act, dt_np)
            sel_cache[c] = {"dp_np": dp_np, "by_ap": by_ap}

    best_pair = (0.0, 0.0)
    best_sel_u = -1e18
    grid_utilities: Dict[str, float] = {}

    for ap, at in candidate_grid:
        vals: List[float] = []
        for c in sel_cids:
            bt = base_terms[c]
            sc_c = sel_cache[c]
            if "inactive_u" in sc_c:
                vals.append(sc_c["inactive_u"])
                continue
            cost_act, dt_np = sc_c["by_ap"][ap]
            if ap == 0.0 and at == 0.0:
                cost_final = cost_act
            else:
                cost_final = (
                    cost_act
                    + ap * sc_c["dp_np"]
                    + at * dt_np
                    + 0.005 * bt["h_cost"]
                )
            pick1 = int(np.argmin(cost_final))
            vals.append(float(bt["u"][pick1]))
        m_u = float(np.mean(vals))
        grid_utilities[f"{ap:.3f}_{at:.3f}"] = m_u
        if m_u > best_sel_u + 1e-7:
            best_sel_u = m_u
            best_pair = (ap, at)

    return model, best_pair[0], best_pair[1], {
        "loaded_s3_encoder_tensors": int(loaded_s3_tensors),
        "step0_max_abs_residual": step0_max_abs,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "pcgrad_cosine_mean": float(np.mean(pcgrad_cosines)),
        "pcgrad_conflict_frac": float(np.mean([c < 0.0 for c in pcgrad_cosines])),
        "selected_alpha_prior": best_pair[0],
        "selected_alpha_traj": best_pair[1],
        "sel_fold_grid_utilities": grid_utilities,
    }


def scaler_to_dict(sc: FoldScaler) -> Dict[str, np.ndarray]:
    return {
        "dope629_mean": sc.dope629_mean,
        "dope629_std": sc.dope629_std,
        "hidden256_mean": sc.hidden256_mean,
        "hidden256_std": sc.hidden256_std,
        "col18_mean": sc.col18_mean,
        "col18_std": sc.col18_std,
    }


def average_state_dicts(models: List[PreCondenseResidualBridge]) -> Dict[str, torch.Tensor]:
    keys = list(models[0].state_dict().keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [m.state_dict()[k].detach().cpu().float() for m in models]
        out[k] = torch.mean(torch.stack(tensors, dim=0), dim=0)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return float(np.sum(ra * rb) / denom) if denom > 1e-12 else 0.0


def paired_family_bootstrap(
    base_ctx: np.ndarray,
    cand_ctx: np.ndarray,
    families: List[str],
    seed: int = BOOT_SEED,
    n_draws: int = 10000,
) -> Dict[str, Any]:
    """10,000-draw paired family-cluster bootstrap + family W/T/L + exact two-sided sign test."""
    vals_2d = np.stack([np.asarray(base_ctx, dtype=np.float64), np.asarray(cand_ctx, dtype=np.float64)], axis=1)
    _, fm = ch.family_means_matrix(vals_2d, families)
    diff_fam = fm[:, 1] - fm[:, 0]
    diff_ctx = vals_2d[:, 1] - vals_2d[:, 0]
    n_f = len(diff_fam)

    rng = np.random.default_rng(seed)
    boot_ix = rng.integers(0, n_f, size=(n_draws, n_f))
    boot_means = diff_fam[boot_ix].mean(axis=1)
    ci95 = np.quantile(boot_means, [0.025, 0.975]).tolist()
    lcb90 = float(np.quantile(boot_means, 0.05))
    p_boot_one_sided = float((1 + np.sum(boot_means <= 0.0)) / (n_draws + 1))

    wins = int(np.sum(diff_fam > ch.TIE_MARGIN))
    ties = int(np.sum(np.abs(diff_fam) <= ch.TIE_MARGIN))
    losses = int(np.sum(diff_fam < -ch.TIE_MARGIN))
    n_eff = wins + losses
    if n_eff == 0:
        sign_p = 1.0
    else:
        k_min = min(wins, losses)
        tail = sum(math.comb(n_eff, i) for i in range(k_min + 1)) / (2.0 ** n_eff)
        sign_p = float(min(1.0, 2.0 * tail))

    return {
        "context_mean_gain": float(np.mean(diff_ctx)),
        "family_mean_gain": float(np.mean(diff_fam)),
        "ci95_low": float(ci95[0]),
        "ci95_high": float(ci95[1]),
        "lcb90": lcb90,
        "bootstrap_p_one_sided": p_boot_one_sided,
        "family_wins": wins,
        "family_ties": ties,
        "family_losses": losses,
        "sign_test_pvalue": sign_p,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. Load frozen network, verify DOPE SHA-256 lock, and load pre-condense cache
    net, guard, dope_snap_before, w1_mat = p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    all_rows = train_rows + val_rows
    pools = ch.load_pools(all_rows)
    r882_cache = p255.load_r882_cache_readonly(all_rows)
    raw_ctx = load_pre_condense_cache(CACHE_PATH)

    # Verify R882 Native91 baseline reproduction (`1.1377941898164525`, diff 0.0)
    val_r882_native = [
        ch.eval_native(r882_cache[r["context_sha256"]]["total_cost"], pools[r["context_sha256"]]["utility"])[0]
        for r in val_rows
    ]
    r882_val_mean = float(np.mean(val_r882_native))
    r882_diff = abs(r882_val_mean - ch.R882_NATIVE91_MEAN)
    if r882_diff != 0.0:
        raise RuntimeError(f"R882 Native91 drift: {r882_val_mean} vs {ch.R882_NATIVE91_MEAN}")

    # 2. Precompute static candidate encoder terms (`cand_q`, `sem24`, `eff_prior`) and 5-fold SOS/SP priors
    static_cache_path = OUT_DIR / "cache" / "static_adapter_terms.pt"
    if static_cache_path.exists():
        static_cache = torch.load(static_cache_path, map_location="cpu", weights_only=False)
        w1_f32 = w1_mat.astype(np.float32) / np.sqrt(412.0)
        for r in all_rows:
            c = r["context_sha256"]
            pools[c]["proj16"] = (pools[c]["program"] @ w1_f32).astype(np.float32)
    else:
        static_cache = p255.precompute_static_adapter_terms(
            net, all_rows, pools, raw_ctx, w1_mat, device
        )
        torch.save(static_cache, static_cache_path)
    promos = ch.load_harvested_promotions(train_rows)
    _, cid_to_fold = ch.seed_folds(train_rows, FOLD_SEED)
    all_train_cids = [r["context_sha256"] for r in train_rows]

    cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, raw_ctx)
    sos_priors, sp_priors, _ = s1.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats, promos, seed=FOLD_SEED
    )

    # Load Stage-1 Full-S3 7-Split Regression pretrained encoder & Policy B scaler (`11,121` regression cells)
    s3_ckpt_path = OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt"
    if not s3_ckpt_path.exists():
        raise RuntimeError(f"Missing Stage-1 Full-S3 7-Split Regression checkpoint: {s3_ckpt_path}")
    s3_ckpt = torch.load(s3_ckpt_path, map_location="cpu", weights_only=False)
    pretrained_s3_state = s3_ckpt["encoder_state_dict"]
    s3_scaler_policy_b = s3_ckpt["scaler_policy_b"]

    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_records: List[Dict[str, Any]] = []

    def record_checkpoint(path: Path, stage: str, description: str, n_tensors: int, n_params: int) -> str:
        digest = sha256_file(path)
        checkpoint_records.append(
            {
                "path": str(path),
                "stage": stage,
                "description": description,
                "sha256": digest,
                "size_bytes": int(path.stat().st_size),
                "num_tensors": int(n_tensors),
                "num_parameters": int(n_params),
            }
        )
        return digest

    record_checkpoint(
        s3_ckpt_path,
        stage="stage1_full_s3_regression_7split",
        description="Stage-1 Full-S3 7-Split Regression pretrained context encoder (11,121 Policy B / 11,086 Policy C regression cells)",
        n_tensors=len(pretrained_s3_state),
        n_params=int(sum(v.numel() for v in pretrained_s3_state.values())),
    )

    full_stage1_ckpt_path = ckpt_dir / "stage1_full_s3_reg_7split_pre_condense_92tok_full_model.pt"
    if not full_stage1_ckpt_path.exists():
        stage1_bridge_init = PreCondenseResidualBridge(
            arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
        )
        stage1_bridge_init.load_pretrained_s3_context_encoder(pretrained_s3_state)
        stage1_full_sd = {k: v.detach().cpu() for k, v in stage1_bridge_init.state_dict().items()}
        torch.save(
            {
                "schema": "jope.h256.stage1_full_s3_reg_7split_full_model/v1",
                "stage": "stage1_full_s3_regression_7split_pretraining",
                "model_task_filter": "regression_only",
                "dope_sha256": ch.DOPE_SHA256,
                "state_dict": stage1_full_sd,
                "bridge_state_dict": stage1_full_sd,
                "encoder_state_dict": pretrained_s3_state,
                "scaler_policy_b": s3_scaler_policy_b,
                "scaler_policy_c": s3_ckpt.get("scaler_policy_c", s3_scaler_policy_b),
            },
            full_stage1_ckpt_path,
        )
    record_checkpoint(
        full_stage1_ckpt_path,
        stage="stage1_full_s3_regression_7split",
        description="Complete Stage-1 Pre-Condense 92-token bridge & encoder checkpoint with Policy B/C scalers",
        n_tensors=len(pretrained_s3_state),
        n_params=int(sum(v.numel() for v in pretrained_s3_state.values())),
    )

    # Extract frozen DOPE problem encoder and base solution ranker state dicts once for the unified checkpoint
    full_net_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    dope_problem_encoder_sd = {
        k: v for k, v in full_net_sd.items() if k.startswith("_dope_mlc_v3.")
    }
    base_solution_ranker_sd = {
        k: v for k, v in full_net_sd.items() if not k.startswith("_dope_mlc_v3.")
    }

    base_terms: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        c = r["context_sha256"]
        base_terms[c] = compute_context_base_terms(
            c,
            pools[c],
            r882_cache[c]["total_cost"],
            sos_priors[c],
            sp_priors[c],
            static_cache[c],
            device,
        )

    # Precompute per-fold scalers and standardized context tensors once across all 5 folds
    fold_splits: List[Tuple[List[str], List[str], List[str]]] = []
    std_fold_scalers: List[FoldScaler] = []
    s3_fold_scalers: List[FoldScaler] = []
    std_fold_ctx_tensors: List[Dict[str, Dict[str, torch.Tensor]]] = []
    s3_fold_ctx_tensors: List[Dict[str, Dict[str, torch.Tensor]]] = []

    for k in range(5):
        a_fold, b_fold, fit_folds = ch.cell_folds(k)
        fit_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] in fit_folds]
        sel_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == b_fold]
        ass_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == a_fold]
        fold_splits.append((fit_cids, sel_cids, ass_cids))

        sc_std = FoldScaler.fit(fit_cids, raw_ctx)
        sc_s3 = sc_std.blend_with_s3_scaler(s3_scaler_policy_b, alpha_s3=0.25)
        std_fold_scalers.append(sc_std)
        s3_fold_scalers.append(sc_s3)

        all_k_cids = fit_cids + sel_cids + ass_cids
        std_fold_ctx_tensors.append({c: sc_std.transform_context(raw_ctx[c], device) for c in all_k_cids})
        s3_fold_ctx_tensors.append({c: sc_s3.transform_context(raw_ctx[c], device) for c in all_k_cids})

    scaler_train_full = FoldScaler.fit(all_train_cids, raw_ctx)
    scaler_train_s3_blended = scaler_train_full.blend_with_s3_scaler(s3_scaler_policy_b, alpha_s3=0.25)
    val_ctx_tensors_std = {
        r["context_sha256"]: scaler_train_full.transform_context(raw_ctx[r["context_sha256"]], device)
        for r in val_rows
    }
    val_ctx_tensors_s3 = {
        r["context_sha256"]: scaler_train_s3_blended.transform_context(raw_ctx[r["context_sha256"]], device)
        for r in val_rows
    }
    print(f"[H256 Stage-2 Init Done in {time.time() - t0:.2f}s] Starting 3-Seed x 5-Fold Training...", flush=True)

    # 3. Run 3-Seed x 5-Fold Family-Grouped OOF Training & DL-Val Evaluation across Arms A..G
    all_arm_names = ("R882", *ARM_NAMES)
    # Train Arm G and Arm F first so canonical Stage-2 checkpoints write to disk immediately!
    train_arm_order = (
        "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
        "F_FULL629_HIDDEN256_COL64X17",
        "E_FULL629_PLUS_HIDDEN256_28TOK",
        "D_PRE_CONDENSE_FULL629_20TOK",
        "C_LEARNED_COMPACT_BRIDGE_629",
        "B_CAPACITY_CTRL_10TOK",
    )
    val_metrics_by_seed: Dict[int, Dict[str, Dict[str, List[Any]]]] = {}
    oof_metrics_by_seed: Dict[int, Dict[str, Dict[str, List[Any]]]] = {}
    training_telemetry: Dict[str, Any] = {}
    latency_telemetry: Dict[str, Dict[str, float]] = {}

    # Retain the seed-0 fold models and scalers for Context Sensitivity and Per-Block Attribution
    seed0_fold_models: Dict[str, List[Tuple[PreCondenseResidualBridge, FoldScaler, float, float]]] = {
        arm: [] for arm in ARM_NAMES if arm != "A_FROZEN_10TOK_BASELINE"
    }

    for s_idx, seed in enumerate(SEEDS):
        val_metrics_by_seed[seed] = {
            arm: {"utility": [], "top1_regret": [], "top4_regret": [], "oracle_hit": [], "invalid_pick": []}
            for arm in all_arm_names
        }
        oof_metrics_by_seed[seed] = {
            arm: {"utility": [], "top1_regret": [], "top4_regret": [], "oracle_hit": [], "invalid_pick": []}
            for arm in all_arm_names
        }

        # Pre-draw deterministic `idx4` per context and seed so all arms see the exact same initial 2 probes
        idx4_map: Dict[str, np.ndarray] = {}
        for r in all_rows:
            c = r["context_sha256"]
            rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", seed, c))
            idx4_map[c] = rng_c.choice(len(base_terms[c]["rcost"]), size=4, replace=False)

        # Evaluate R882 and Arm A (`A_FROZEN_10TOK_BASELINE`) on OOF and DL-Val
        oof_order_cids: List[str] = []
        for k in range(5):
            _, _, ass_cids = fold_splits[k]
            oof_order_cids.extend(ass_cids)

        for c in oof_order_cids:
            bt = base_terms[c]
            p_r = bt["pick_r882"]
            top4_r = np.argsort(bt["rcost"])[:4]
            oof_metrics_by_seed[seed]["R882"]["utility"].append(bt["u_r882"])
            oof_metrics_by_seed[seed]["R882"]["top1_regret"].append(float(bt["u_max"] - bt["u_r882"]))
            oof_metrics_by_seed[seed]["R882"]["top4_regret"].append(float(bt["u_max"] - np.max(bt["u"][top4_r])))
            oof_metrics_by_seed[seed]["R882"]["oracle_hit"].append(bool(bt["u_max"] - bt["u_r882"] <= 1e-4))
            oof_metrics_by_seed[seed]["R882"]["invalid_pick"].append(bool(not bt["gate"][p_r]))

            res_a = evaluate_context_with_model(bt, None, [], 0.0, 0.0, idx4_map[c], device)
            for k_m in ("utility", "top1_regret", "top4_regret", "oracle_hit", "invalid_pick"):
                oof_metrics_by_seed[seed]["A_FROZEN_10TOK_BASELINE"][k_m].append(res_a[k_m])

        for r in val_rows:
            c = r["context_sha256"]
            bt = base_terms[c]
            p_r = bt["pick_r882"]
            top4_r = np.argsort(bt["rcost"])[:4]
            val_metrics_by_seed[seed]["R882"]["utility"].append(bt["u_r882"])
            val_metrics_by_seed[seed]["R882"]["top1_regret"].append(float(bt["u_max"] - bt["u_r882"]))
            val_metrics_by_seed[seed]["R882"]["top4_regret"].append(float(bt["u_max"] - np.max(bt["u"][top4_r])))
            val_metrics_by_seed[seed]["R882"]["oracle_hit"].append(bool(bt["u_max"] - bt["u_r882"] <= 1e-4))
            val_metrics_by_seed[seed]["R882"]["invalid_pick"].append(bool(not bt["gate"][p_r]))

            res_a = evaluate_context_with_model(bt, None, [], 0.0, 0.0, idx4_map[c], device)
            for k_m in ("utility", "top1_regret", "top4_regret", "oracle_hit", "invalid_pick"):
                val_metrics_by_seed[seed]["A_FROZEN_10TOK_BASELINE"][k_m].append(res_a[k_m])

        # Train and evaluate Arms G..B across the 5 family-grouped OOF folds
        for arm in train_arm_order:
            t_arm0 = time.perf_counter()
            is_s3_arm = arm == "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"
            fold_models: List[PreCondenseResidualBridge] = []
            fold_scalers: List[FoldScaler] = []
            fold_ap: List[float] = []
            fold_at: List[float] = []

            for k in range(5):
                fit_cids, sel_cids, ass_cids = fold_splits[k]
                scaler_k = s3_fold_scalers[k] if is_s3_arm else std_fold_scalers[k]
                ctx_tensors_k = s3_fold_ctx_tensors[k] if is_s3_arm else std_fold_ctx_tensors[k]
                fold_ckpt_path = ckpt_dir / f"stage2_{arm}_seed{seed}_fold{k}.pt"

                if fold_ckpt_path.exists():
                    saved_fold = torch.load(fold_ckpt_path, map_location="cpu", weights_only=False)
                    fold_sd = saved_fold["state_dict"]
                    model_k = PreCondenseResidualBridge(arm=arm, d_token=32, n_heads=4).to(device)
                    model_k.load_state_dict(fold_sd, strict=True)
                    model_k.eval()
                    ap_k = float(saved_fold["selected_alpha_prior"])
                    at_k = float(saved_fold["selected_alpha_traj"])
                    info_k = saved_fold.get("training_telemetry", {})
                else:
                    model_k, ap_k, at_k, info_k = train_pre_condense_bridge_fold(
                        arm,
                        fit_cids,
                        sel_cids,
                        base_terms,
                        ctx_tensors_k,
                        idx4_map,
                        seed + 100 * k + ARM_NAMES.index(arm) * 10,
                        device,
                        pretrained_s3_state=pretrained_s3_state if is_s3_arm else None,
                        guard=guard,
                        dope_net=net,
                    )
                    # Persist per-fold Stage-2 checkpoint immediately
                    fold_sd = {k_t: v_t.detach().cpu() for k_t, v_t in model_k.state_dict().items()}
                    torch.save(
                        {
                            "schema": "jope.h256.stage2_fold_checkpoint/v1",
                            "stage": "stage2_flywheel_regression",
                            "model_task_scope": "regression_only",
                            "arm": arm,
                            "seed": int(seed),
                            "fold": int(k),
                            "d_token": 32,
                            "n_heads": 4,
                            "dope_sha256": ch.DOPE_SHA256,
                            "state_dict": fold_sd,
                            "model_state_dict": fold_sd,
                            "scaler": scaler_to_dict(scaler_k),
                            "selected_alpha_prior": float(ap_k),
                            "selected_alpha_traj": float(at_k),
                            "training_telemetry": info_k,
                        },
                        fold_ckpt_path,
                    )

                training_telemetry[f"seed{seed}_{arm}_fold{k}"] = info_k
                fold_models.append(model_k)
                fold_scalers.append(scaler_k)
                fold_ap.append(ap_k)
                fold_at.append(at_k)
                if s_idx == 0:
                    seed0_fold_models[arm].append((model_k, scaler_k, ap_k, at_k))

                record_checkpoint(
                    fold_ckpt_path,
                    stage="stage2_flywheel_regression_fold",
                    description=f"Stage-2 PreCondenseResidualBridge fold checkpoint ({arm}, seed={seed}, fold={k})",
                    n_tensors=len(fold_sd),
                    n_params=int(sum(v.numel() for v in fold_sd.values())),
                )

                for c in ass_cids:
                    res_c = evaluate_context_with_model(
                        base_terms[c], ctx_tensors_k[c], [model_k], ap_k, at_k, idx4_map[c], device
                    )
                    for k_m in ("utility", "top1_regret", "top4_regret", "oracle_hit", "invalid_pick"):
                        oof_metrics_by_seed[seed][arm][k_m].append(res_c[k_m])

            # Evaluate unseen DL-Val using the 5-fold OOF ensemble (standardizing with full DL-Train scaler)
            sc_val_use = scaler_train_s3_blended if is_s3_arm else scaler_train_full
            val_tensors_use = val_ctx_tensors_s3 if is_s3_arm else val_ctx_tensors_std
            ap_ens = float(np.mean(fold_ap))
            at_ens = float(np.mean(fold_at))
            if ap_ens == 0.0 and at_ens == 0.0:
                ap_ens, at_ens = 0.025, 0.015

            t_lat0 = time.perf_counter()
            for r in val_rows:
                c = r["context_sha256"]
                res_v = evaluate_context_with_model(
                    base_terms[c], val_tensors_use[c], fold_models, ap_ens, at_ens, idx4_map[c], device
                )
                for k_m in ("utility", "top1_regret", "top4_regret", "oracle_hit", "invalid_pick"):
                    val_metrics_by_seed[seed][arm][k_m].append(res_v[k_m])
            lat_ms = (time.perf_counter() - t_lat0) * 1000.0 / len(val_rows)
            n_params = sum(p.numel() for p in fold_models[0].parameters() if p.requires_grad)
            latency_telemetry[arm] = {
                "val_per_context_latency_ms": round(lat_ms, 4),
                "trainable_params": int(n_params),
                "ensemble_alpha_prior": round(ap_ens, 4),
                "ensemble_alpha_traj": round(at_ens, 4),
            }
            guard.verify_dope_frozen_unchanged(net, dope_snap_before)

            # Save 5-fold ensemble + parameter-soup checkpoint on seed 0 immediately
            if s_idx == 0:
                avg_sd = average_state_dicts(fold_models)
                fold_sds = [
                    {k_t: v_t.detach().cpu() for k_t, v_t in m.state_dict().items()}
                    for m in fold_models
                ]
                ens_payload = {
                    "schema": "jope.h256.stage2_5fold_ensemble/v1",
                    "stage": "stage2_flywheel_regression",
                    "model_task_scope": "regression_only",
                    "arm": arm,
                    "seed": int(seed),
                    "d_token": 32,
                    "n_heads": 4,
                    "dope_sha256": ch.DOPE_SHA256,
                    "state_dict": avg_sd,
                    "bridge_state_dict": avg_sd,
                    "model_state_dict": avg_sd,
                    "fold_0_state_dict": fold_sds[0],
                    "fold_state_dicts": fold_sds,
                    "fold_scalers": [scaler_to_dict(sc) for sc in fold_scalers],
                    "full_train_scaler": scaler_to_dict(sc_val_use),
                    "fold_alpha_prior": [float(x) for x in fold_ap],
                    "fold_alpha_traj": [float(x) for x in fold_at],
                    "ensemble_alpha_prior": float(ap_ens),
                    "ensemble_alpha_traj": float(at_ens),
                }
                ens_ckpt_path = ckpt_dir / f"stage2_{arm}_5fold_ensemble.pt"
                torch.save(ens_payload, ens_ckpt_path)
                record_checkpoint(
                    ens_ckpt_path,
                    stage="stage2_flywheel_regression_ensemble",
                    description=f"Stage-2 5-fold ensemble + parameter-soup checkpoint ({arm}, seed={seed})",
                    n_tensors=len(avg_sd),
                    n_params=int(sum(v.numel() for v in avg_sd.values())),
                )

                if is_s3_arm:
                    top_stage2_path = OUT_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt"
                    torch.save(ens_payload, top_stage2_path)
                    record_checkpoint(
                        top_stage2_path,
                        stage="stage2_canonical_bridge",
                        description="Canonical Stage-2 PreCondenseResidualBridge checkpoint warm-started from Stage-1 Full-S3 7-Split Regression",
                        n_tensors=len(avg_sd),
                        n_params=int(sum(v.numel() for v in avg_sd.values())),
                    )

                    unified_path = OUT_DIR / "h256_fused_problem_and_solution_network.pt"
                    unified_payload = {
                        "schema": "jope.h256.fused_problem_and_solution_network/v1",
                        "stage": "stage1_and_stage2_complete",
                        "model_task_scope": "regression_only",
                        "dope_sha256": ch.DOPE_SHA256,
                        "problem_encoder_dope_state_dict": dope_problem_encoder_sd,
                        "base_solution_ranker_state_dict": base_solution_ranker_sd,
                        "stage1_pre_condense_encoder_state_dict": pretrained_s3_state,
                        "stage2_pre_condense_bridge_state_dict": avg_sd,
                        "stage2_fold0_bridge_state_dict": fold_sds[0],
                        "stage2_fold_state_dicts": fold_sds,
                        "state_dict": avg_sd,
                        "scaler_stage1_s3_policy_b": s3_scaler_policy_b,
                        "scaler_stage2_flywheel": scaler_to_dict(sc_val_use),
                        "ensemble_alpha_prior": float(ap_ens),
                        "ensemble_alpha_traj": float(at_ens),
                    }
                    torch.save(unified_payload, unified_path)
                    record_checkpoint(
                        unified_path,
                        stage="unified_problem_and_solution_network",
                        description="Unified self-contained Problem Encoder (locked DOPE) + Solution Ranker + Stage-1 Pre-Condense Encoder + Stage-2 Pre-Condense Residual Bridge",
                        n_tensors=len(dope_problem_encoder_sd)
                        + len(base_solution_ranker_sd)
                        + len(avg_sd),
                        n_params=int(
                            sum(v.numel() for v in dope_problem_encoder_sd.values())
                            + sum(v.numel() for v in base_solution_ranker_sd.values())
                            + sum(v.numel() for v in avg_sd.values())
                        ),
                    )
                ch.save_json(
                    ckpt_dir / "checkpoint_manifest.json",
                    {
                        "schema": "jope.h256.checkpoint_manifest/v1",
                        "hypothesis_id": "H256",
                        "model_task_scope": "regression_only",
                        "dope_sha256": ch.DOPE_SHA256,
                        "canonical_stage1_encoder_checkpoint": str(s3_ckpt_path),
                        "canonical_stage1_full_model_checkpoint": str(full_stage1_ckpt_path),
                        "canonical_stage2_bridge_checkpoint": str(
                            OUT_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt"
                        ),
                        "unified_problem_and_solution_checkpoint": str(
                            OUT_DIR / "h256_fused_problem_and_solution_network.pt"
                        ),
                        "total_checkpoints_saved": len(checkpoint_records),
                        "checkpoints": checkpoint_records,
                    },
                )
            print(
                f"  [Seed {seed} | {arm}] 5-fold train+OOF+Val done in {time.perf_counter() - t_arm0:.2f}s | "
                f"Val91 U={np.mean(val_metrics_by_seed[seed][arm]['utility']):.6f} | "
                f"OOF1219 U={np.mean(oof_metrics_by_seed[seed][arm]['utility']):.6f}",
                flush=True,
            )

    # 4. Aggregate 3-seed metrics, 10,000-draw paired family bootstrap CIs, Sign Test, and 400-iter BT Elo
    val_fams = [r["family_id"] for r in val_rows]
    oof_fams = []
    for k in range(5):
        a_fold, _, _ = ch.cell_folds(k)
        oof_fams.extend([r["family_id"] for r in train_rows if cid_to_fold[r["context_sha256"]] == a_fold])

    def summarize_partition(
        metrics_by_seed: Dict[int, Dict[str, Dict[str, List[Any]]]],
        fams: List[str],
        partition_label: str,
    ) -> Dict[str, Any]:
        u_mat = np.stack(
            [
                np.mean([metrics_by_seed[s][arm]["utility"] for s in SEEDS], axis=0)
                for arm in all_arm_names
            ],
            axis=1,
        )  # [N_ctx, N_arms]
        _, fam_mat = ch.family_means_matrix(u_mat, fams)  # [N_fams, N_arms]
        wins_tensor = ch.family_wins_tensor(u_mat, fams)
        elo_point = ch.bt_ratings(wins_tensor.mean(axis=0), anchor=0)

        rng_elo = np.random.default_rng(BOOT_SEED + 11)
        n_f = len(fam_mat)
        elo_idx = rng_elo.integers(0, n_f, size=(ch.ELO_DRAWS, n_f))
        elo_draws = ch.bt_ratings(wins_tensor[elo_idx].mean(axis=1), anchor=0)

        r882_ctx = u_mat[:, 0]
        arm_a_ctx = u_mat[:, 1]
        raw_pvals_vs_r882: Dict[str, float] = {}
        raw_pvals_vs_arm_a: Dict[str, float] = {}
        stats_vs_r882: Dict[str, Dict[str, Any]] = {}
        stats_vs_arm_a: Dict[str, Dict[str, Any]] = {}

        for j, arm in enumerate(all_arm_names):
            if j > 0:
                st_r = paired_family_bootstrap(
                    r882_ctx, u_mat[:, j], fams, seed=BOOT_SEED + j * 13
                )
                stats_vs_r882[arm] = st_r
                raw_pvals_vs_r882[arm] = st_r["sign_test_pvalue"]
            if j > 1:
                st_a = paired_family_bootstrap(
                    arm_a_ctx, u_mat[:, j], fams, seed=BOOT_SEED + 100 + j * 17
                )
                stats_vs_arm_a[arm] = st_a
                raw_pvals_vs_arm_a[arm] = st_a["sign_test_pvalue"]

        adj_p_r882 = ch.holm_bonferroni(raw_pvals_vs_r882)
        adj_p_arm_a = ch.holm_bonferroni(raw_pvals_vs_arm_a)

        out_arms: Dict[str, Any] = {}
        for j, arm in enumerate(all_arm_names):
            top1_reg = float(np.mean([metrics_by_seed[s][arm]["top1_regret"] for s in SEEDS]))
            top4_reg = float(np.mean([metrics_by_seed[s][arm]["top4_regret"] for s in SEEDS]))
            oracle_hit = float(np.mean([metrics_by_seed[s][arm]["oracle_hit"] for s in SEEDS]))
            inv_rate = float(np.mean([metrics_by_seed[s][arm]["invalid_pick"] for s in SEEDS]))
            entry: Dict[str, Any] = {
                "partition": partition_label,
                "context_mean_utility": float(np.mean(u_mat[:, j])),
                "family_mean_utility": float(np.mean(fam_mat[:, j])),
                "top1_regret": top1_reg,
                "top4_regret": top4_reg,
                "oracle_top1_hit_rate": oracle_hit,
                "invalid_gate_selection_rate": inv_rate,
                "elo": float(elo_point[j]),
                "elo_lcb95": float(np.quantile(elo_draws[:, j], 0.025)),
                "elo_ucb95": float(np.quantile(elo_draws[:, j], 0.975)),
            }
            if arm in stats_vs_r882:
                entry["vs_r882"] = {
                    **stats_vs_r882[arm],
                    "holm_bonferroni_pvalue": adj_p_r882[arm],
                }
            if arm in stats_vs_arm_a:
                entry["vs_arm_a_10tok_baseline"] = {
                    **stats_vs_arm_a[arm],
                    "holm_bonferroni_pvalue": adj_p_arm_a[arm],
                }
            out_arms[arm] = entry
        return out_arms

    dl_val_summary = summarize_partition(val_metrics_by_seed, val_fams, "DL-Val (91 contexts / 52 unseen families)")
    dl_train_oof_summary = summarize_partition(
        oof_metrics_by_seed, oof_fams, "DL-Train OOF (1219 contexts / 205 families)"
    )

    # Also summarize all canonical regression sub-splits (`strict_train_fit_881`, `dl_train_inner_203`,
    # `all_dl_train_1100`, `st_val_gate_32`, `dl_val_71`)
    oof_order_cids_full: List[str] = []
    for k in range(5):
        a_fold, _, _ = ch.cell_folds(k)
        oof_order_cids_full.extend(
            [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == a_fold]
        )
    oof_pos_by_cid = {c: i for i, c in enumerate(oof_order_cids_full)}
    val_pos_by_cid = {r["context_sha256"]: i for i, r in enumerate(val_rows)}
    cid_to_fam = {r["context_sha256"]: r["family_id"] for r in all_rows}

    with Path("/home/ubuntu/jope-data/hwv-06/sources/historical-role-identities.json").open() as f:
        hist_roles = json.load(f)["roles"]
    with Path(
        "/mnt/jope-archive/flywheel-20261023/results/h40_h45_cluster/h40_fast/h40_split_role_reconciliation_receipt_v2.json"
    ).open() as f:
        st16_cids = list(json.load(f)["authorized_st16_inner70_allowlist_v2"].keys())

    subsplit_defs = {
        "strict_train_fit_881": hist_roles["strict_train_fit_881"],
        "dl_train_inner_203": hist_roles["dl_train_inner_203"],
        "all_dl_train_1100": hist_roles["strict_train_fit_881"] + hist_roles["dl_train_inner_203"] + st16_cids,
        "st_val_gate_32": hist_roles["st_val_gate_32"],
        "dl_val_71": hist_roles["dl_val_71"],
    }
    regression_subsplits_summary: Dict[str, Any] = {}
    for sp_name, sp_cids in subsplit_defs.items():
        valid_cids = [c for c in sp_cids if (c in oof_pos_by_cid or c in val_pos_by_cid)]
        fams_sp = [cid_to_fam[c] for c in valid_cids]
        sub_metrics = {
            s: {
                arm: {
                    k_m: [
                        (
                            oof_metrics_by_seed[s][arm][k_m][oof_pos_by_cid[c]]
                            if c in oof_pos_by_cid
                            else val_metrics_by_seed[s][arm][k_m][val_pos_by_cid[c]]
                        )
                        for c in valid_cids
                    ]
                    for k_m in ("utility", "top1_regret", "top4_regret", "oracle_hit", "invalid_pick")
                }
                for arm in all_arm_names
            }
            for s in SEEDS
        }
        regression_subsplits_summary[sp_name] = summarize_partition(
            sub_metrics, fams_sp, f"{sp_name} ({len(valid_cids)} regression contexts / {len(set(fams_sp))} families)"
        )

    # 5. Context Sensitivity Diagnostic (True vs. Cross-Family Shuffled vs. Zeroed Context)
    scaler_train_full = FoldScaler.fit(all_train_cids, raw_ctx)
    scaler_train_s3_blended = scaler_train_full.blend_with_s3_scaler(s3_scaler_policy_b, alpha_s3=0.25)
    idx4_seed0 = {
        r["context_sha256"]: np.random.default_rng(
            ch.stream_seed("h256_idx4", SEEDS[0], r["context_sha256"])
        ).choice(len(base_terms[r["context_sha256"]]["rcost"]), size=4, replace=False)
        for r in all_rows
    }

    context_sensitivity: Dict[str, Any] = {}
    val_cids = [r["context_sha256"] for r in val_rows]
    # Construct a deterministic cross-family derangement of DL-Val contexts
    shuffled_val_cids = val_cids[17:] + val_cids[:17]

    for diag_arm in (
        "D_PRE_CONDENSE_FULL629_20TOK",
        "E_FULL629_PLUS_HIDDEN256_28TOK",
        "F_FULL629_HIDDEN256_COL64X17",
        "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
    ):
        models_d = [t[0] for t in seed0_fold_models[diag_arm]]
        ap_d = latency_telemetry[diag_arm]["ensemble_alpha_prior"]
        at_d = latency_telemetry[diag_arm]["ensemble_alpha_traj"]
        sc_use = scaler_train_s3_blended if diag_arm == "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED" else scaler_train_full

        u_true_list, u_shuf_list, u_zero_list = [], [], []
        rms_shift_shuf, rms_shift_zero, spearman_shuf = [], [], []

        for c_true, c_shuf in zip(val_cids, shuffled_val_cids):
            bt = base_terms[c_true]
            ct_true = sc_use.transform_context(raw_ctx[c_true], device)
            ct_shuf = sc_use.transform_context(raw_ctx[c_shuf], device)
            ct_zero = {
                k_t: (v_t.clone() if k_t in ("tokens_10x32", "mask_10", "col_mask_64") else torch.zeros_like(v_t))
                for k_t, v_t in ct_true.items()
            }

            res_true = evaluate_context_with_model(bt, ct_true, models_d, ap_d, at_d, idx4_seed0[c_true], device)
            res_shuf = evaluate_context_with_model(bt, ct_shuf, models_d, ap_d, at_d, idx4_seed0[c_true], device)
            res_zero = evaluate_context_with_model(bt, ct_zero, models_d, ap_d, at_d, idx4_seed0[c_true], device)

            u_true_list.append(res_true["utility"])
            u_shuf_list.append(res_shuf["utility"])
            u_zero_list.append(res_zero["utility"])

            if bt["top_zgap"] < 0.25:
                v_t = res_true["delta_prior_vec"]
                v_s = res_shuf["delta_prior_vec"]
                v_z = res_zero["delta_prior_vec"]
                rms_shift_shuf.append(float(np.sqrt(np.mean((v_t - v_s) ** 2))))
                rms_shift_zero.append(float(np.sqrt(np.mean((v_t - v_z) ** 2))))
                spearman_shuf.append(spearman_corr(v_t, v_s))

        st_true_vs_shuf = paired_family_bootstrap(
            np.asarray(u_shuf_list), np.asarray(u_true_list), val_fams, seed=BOOT_SEED + 501
        )
        context_sensitivity[diag_arm] = {
            "true_context_mean_utility": float(np.mean(u_true_list)),
            "shuffled_context_mean_utility": float(np.mean(u_shuf_list)),
            "zeroed_context_mean_utility": float(np.mean(u_zero_list)),
            "utility_gain_true_over_shuffled": float(np.mean(u_true_list) - np.mean(u_shuf_list)),
            "utility_gain_true_over_zeroed": float(np.mean(u_true_list) - np.mean(u_zero_list)),
            "paired_family_bootstrap_true_vs_shuffled": st_true_vs_shuf,
            "residual_rms_shift_true_vs_shuffled": float(np.mean(rms_shift_shuf)),
            "residual_rms_shift_true_vs_zeroed": float(np.mean(rms_shift_zero)),
            "mean_within_pool_spearman_true_vs_shuffled": float(np.mean(spearman_shuf)),
        }

    # 6. Per-Block Leave-One-Block-Out Attribution on Arm G (`G_FULL_S3_REG_7SPLIT_11121_PRETRAINED`)
    models_f = [t[0] for t in seed0_fold_models["G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"]]
    ap_f = latency_telemetry["G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"]["ensemble_alpha_prior"]
    at_f = latency_telemetry["G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"]["ensemble_alpha_traj"]

    blocks_to_ablate = (
        *BLOCK_SLICES.keys(),
        "block_log_norms",
        "hidden256",
        "col_lm_64x17",
    )
    block_attribution: Dict[str, Any] = {}
    full_f_vecs = {}
    full_f_u = []
    for c in val_cids:
        ct_full = scaler_train_s3_blended.transform_context(raw_ctx[c], device, ablate_block=None)
        res_f = evaluate_context_with_model(
            base_terms[c], ct_full, models_f, ap_f, at_f, idx4_seed0[c], device
        )
        full_f_vecs[c] = res_f["delta_prior_vec"]
        full_f_u.append(res_f["utility"])

    for blk in blocks_to_ablate:
        blk_u = []
        blk_rms_shift = []
        for c in val_cids:
            ct_abl = scaler_train_s3_blended.transform_context(raw_ctx[c], device, ablate_block=blk)
            res_abl = evaluate_context_with_model(
                base_terms[c], ct_abl, models_f, ap_f, at_f, idx4_seed0[c], device
            )
            blk_u.append(res_abl["utility"])
            if base_terms[c]["top_zgap"] < 0.25:
                blk_rms_shift.append(
                    float(np.sqrt(np.mean((full_f_vecs[c] - res_abl["delta_prior_vec"]) ** 2)))
                )
        block_attribution[blk] = {
            "ablated_mean_utility": float(np.mean(blk_u)),
            "utility_drop_when_ablated": float(np.mean(full_f_u) - np.mean(blk_u)),
            "residual_rms_perturbation": float(np.mean(blk_rms_shift)),
        }

    # Learned gate weights summary on Arm G
    with torch.no_grad():
        g20 = torch.mean(
            torch.stack([torch.sigmoid(m.full629_gate_logits) for m in models_f]), dim=0
        ).cpu().numpy()
        gh8 = torch.mean(
            torch.stack([torch.sigmoid(m.hidden_gate_logits) for m in models_f]), dim=0
        ).cpu().numpy()
        gcol = float(
            torch.mean(torch.stack([torch.sigmoid(m.col_gate_logit) for m in models_f])).item()
        )
    learned_gate_summary = {
        "desc256_8tok_mean_gate": float(np.mean(g20[0:8])),
        "task128_4tok_mean_gate": float(np.mean(g20[8:12])),
        "dist128_4tok_mean_gate": float(np.mean(g20[12:16])),
        "heads53_plus_norms11_2tok_mean_gate": float(np.mean(g20[16:18])),
        "col_gates64_2tok_mean_gate": float(np.mean(g20[18:20])),
        "hidden256_8tok_mean_gate": float(np.mean(gh8)),
        "col_lm_64x17_token_gate": gcol,
    }

    # 7. Final DOPE Weight-Lock Cryptographic Verification
    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    dope_snap_after = guard.snapshot_dope_weights(net)
    if dope_snap_after.composite_sha256 != ch.DOPE_SHA256:
        raise RuntimeError("DOPE SHA-256 changed after H256 training!")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    dope_receipt = {
        "hypothesis_id": "H256",
        "dope_weights_locked": True,
        "initial_sha256": dope_snap_before.composite_sha256,
        "final_sha256": dope_snap_after.composite_sha256,
        "expected_sha256": ch.DOPE_SHA256,
        "bit_identical": dope_snap_before.composite_sha256 == dope_snap_after.composite_sha256,
        "num_param_tensors": dope_snap_after.num_param_tensors,
        "num_param_elements": dope_snap_after.num_param_elements,
        "num_buffer_tensors": dope_snap_after.num_buffer_tensors,
    }
    ch.save_json(OUT_DIR / "dope_freeze_verification.json", dope_receipt)

    manifest_path = ckpt_dir / "checkpoint_manifest.json"
    manifest_sha = ch.save_json(
        manifest_path,
        {
            "schema": "jope.h256.checkpoint_manifest/v1",
            "hypothesis_id": "H256",
            "model_task_scope": "regression_only",
            "dope_sha256": ch.DOPE_SHA256,
            "canonical_stage1_encoder_checkpoint": str(s3_ckpt_path),
            "canonical_stage1_full_model_checkpoint": str(full_stage1_ckpt_path),
            "canonical_stage2_bridge_checkpoint": str(
                OUT_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt"
            ),
            "unified_problem_and_solution_checkpoint": str(
                OUT_DIR / "h256_fused_problem_and_solution_network.pt"
            ),
            "total_checkpoints_saved": len(checkpoint_records),
            "checkpoints": checkpoint_records,
        },
    )

    summary = {
        "hypothesis_id": "H256",
        "title": "Pre-Condense Residual Bypass Trained on Full S3 7-Split Regression Corpus (11,121 Policy B / 11,086 Policy C Cells) and Multi-Split Regression Flywheel with Weight-Locked DOPE",
        "model_task_scope": "regression_only",
        "elapsed_sec": round(time.time() - t0, 3),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "checkpoint_manifest_path": str(manifest_path),
        "checkpoint_manifest_sha256": manifest_sha,
        "total_checkpoints_saved": len(checkpoint_records),
        "canonical_checkpoints": {
            "stage1_encoder": str(s3_ckpt_path),
            "stage1_full_model": str(full_stage1_ckpt_path),
            "stage2_bridge": str(OUT_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt"),
            "unified_problem_and_solution_network": str(
                OUT_DIR / "h256_fused_problem_and_solution_network.pt"
            ),
        },
        "r882_native91_reproduction": {
            "expected": ch.R882_NATIVE91_MEAN,
            "observed": r882_val_mean,
            "abs_diff": r882_diff,
        },
        "dope_freeze_verification": dope_receipt,
        "dl_val_unseen_families_summary": dl_val_summary,
        "dl_train_5fold_oof_summary": dl_train_oof_summary,
        "regression_subsplits_summary": regression_subsplits_summary,
        "context_sensitivity_diagnostic": context_sensitivity,
        "per_block_leave_one_out_attribution": block_attribution,
        "learned_block_gate_activations": learned_gate_summary,
        "latency_and_capacity_telemetry": latency_telemetry,
        "fold_training_telemetry": training_telemetry,
    }
    summary_sha = ch.save_json(OUT_DIR / "summary.json", summary)
    print("H256 Summary SHA-256:", summary_sha)
    for arm in all_arm_names:
        v = dl_val_summary[arm]
        o = dl_train_oof_summary[arm]
        g32 = regression_subsplits_summary["st_val_gate_32"][arm]
        v71 = regression_subsplits_summary["dl_val_71"][arm]
        print(
            f"  {arm:38s} | DL-Val91 U={v['context_mean_utility']:.6f} (Elo={v['elo']:.1f}) | "
            f"Gate32 U={g32['context_mean_utility']:.6f} | Val71 U={v71['context_mean_utility']:.6f} | "
            f"OOF1219 U={o['context_mean_utility']:.6f} (Elo={o['elo']:.1f})"
        )


if __name__ == "__main__":
    main()
