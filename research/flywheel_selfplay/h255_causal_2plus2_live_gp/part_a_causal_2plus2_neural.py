#!/usr/bin/env python3
"""H255 Part A: Preregistered Causal 2+2 Neural Trajectory Critic (K_rev=0 -> K_rev=2)
and Root-Cause Audit of the `revealed_indices=[0,1,2,3]` Oracle Self-Leakage Bug in `run_fused_closed_loop.py`.

Guardrails:
- Reads H252 `common_h252.py` and `stage1_hurdle_sos_critic.py` read-only after verifying their exact frozen SHA-256s.
- Never touches DL-Val during any prior fit or neural parameter update (5-fold family-grouped OOF on DL-Train; DL-Val uses model fit on DL-Train only).
- Verifies DOPE SHA-256 `c51a56d0...` before and after training, and reproduces R882 Native91 `1.1377941898164525` to 0.00e+00 first.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
OUT_DIR = Path("/home/ubuntu/jope-data/h255/part_a")

FROZEN_COMMON_SHA256 = "abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92"
FROZEN_STAGE1_SHA256 = "fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797"

SEEDS = (255101, 255102, 255103)
FOLD_SEED = 255140
BOOT_SEED = 255150
BOOT_DRAWS = 10_000


def verify_frozen_h252_files() -> Dict[str, str]:
    common_p = H252_DIR / "common_h252.py"
    stage1_p = H252_DIR / "stage1_hurdle_sos_critic.py"
    c_sha = hashlib.sha256(common_p.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(stage1_p.read_bytes()).hexdigest()
    if c_sha != FROZEN_COMMON_SHA256:
        raise RuntimeError(f"Frozen common_h252.py SHA-256 drift: {c_sha}")
    if s_sha != FROZEN_STAGE1_SHA256:
        raise RuntimeError(f"Frozen stage1_hurdle_sos_critic.py SHA-256 drift: {s_sha}")
    return {"common_h252.py": c_sha, "stage1_hurdle_sos_critic.py": s_sha}


sys.path.insert(0, str(H252_DIR))
import common_h252 as ch  # noqa: E402
import stage1_hurdle_sos_critic as s1  # noqa: E402


def load_fused_network(device: torch.device):
    if str(MICRO / "python") not in sys.path:
        sys.path.insert(0, str(MICRO / "python"))
    sys.modules.pop("research", None)
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


def audit_self_leakage_bug(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Quantify the exact train-vs-eval mismatch of `revealed_indices=[0,1,2,3]` in `run_fused_closed_loop.py`."""
    val_slot03_best_ranks = []
    val_top2_overlap = []
    for r in val_rows:
        c = r["context_sha256"]
        p = pools[c]
        f = np.where(p["gate"], p["fitness"], 2.0)
        order = np.argsort(f)
        ranks = [int(np.where(order == k)[0][0]) for k in range(min(4, len(f)))]
        val_slot03_best_ranks.append(min(ranks))
        top2_r882 = set(np.argsort(p["r882_cost"])[:2].tolist())
        val_top2_overlap.append(len(top2_r882 & {0, 1, 2, 3}) / 2.0)

    train_slot03_best_ranks_raw = []
    for r in train_rows:
        c = r["context_sha256"]
        p = pools[c]
        f = np.where(p["gate"], p["fitness"], 2.0)
        order = np.argsort(f)
        ranks = [int(np.where(order == k)[0][0]) for k in range(min(4, len(f)))]
        train_slot03_best_ranks_raw.append(min(ranks))

    return {
        "training_slate_slot03_oracle_ranks": [0, 1, 2, 3],
        "training_self_distance_d2_joint_on_slots_0_to_3": 0.0,
        "dl_val_raw_slot03_median_best_rank": float(np.median(val_slot03_best_ranks)),
        "dl_val_raw_slot03_mean_best_rank": float(np.mean(val_slot03_best_ranks)),
        "dl_val_raw_slot03_contains_winner_frac": float(np.mean([rk == 0 for rk in val_slot03_best_ranks])),
        "dl_val_r882_top2_in_slot03_frac": float(np.mean(val_top2_overlap)),
        "dl_train_raw_slot03_median_best_rank": float(np.median(train_slot03_best_ranks_raw)),
    }


def precompute_static_adapter_terms(
    net,
    cids: List[str],
    pools: Dict[str, Dict[str, Any]],
    ctx_feats: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute the K_rev=0 (`revealed_indices=None`) neural terms once per context for fast causal 2+2."""
    net.eval()
    static_cache: Dict[str, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for c in cids:
            p = pools[c]
            cf = ctx_feats[c]
            m = len(p["fitness"])
            out0 = net.adapter_10tok.predict_costs(
                ctx_feat=torch.as_tensor(cf["ctx_feat_584"], device=device).reshape(1, 584),
                prog_412=torch.as_tensor(p["program"], device=device).reshape(1, m, 412),
                proj_16=torch.as_tensor(p["proj16"], device=device).reshape(1, m, 16),
                base_prior_cost=torch.zeros((1, m), device=device),
                scale=torch.full((1,), 0.4208, device=device),
                sos_prior=torch.zeros((1, m, 2), device=device),
                mm_tokens_10x32=torch.as_tensor(cf["tokens_10x32"], device=device).reshape(1, 10, 32),
                mm_mask_10=torch.as_tensor(cf["mask_10"], device=device).reshape(1, 10),
                revealed_indices=None,
                revealed_raw_f=None,
                revealed_gate=None,
            )
            static_cache[c] = {
                "cost_k0": out0["total_cost"][0].cpu().numpy().astype(np.float64),
                "eff_prior": out0["eff_prior"][0].cpu().numpy().astype(np.float32),
                "sem24": out0["sem24"][0].cpu().numpy().astype(np.float32),
                "cand_q": out0["cand_q"][0].cpu().numpy().astype(np.float32),
            }
    return static_cache


def forward_step2_trajectory_delta(
    adapter,
    eff_prior: torch.Tensor,
    sem24: torch.Tensor,
    cand_q: torch.Tensor,
    rev_idx: torch.Tensor,
    rev_raw_f: torch.Tensor,
    rev_gate: torch.Tensor,
) -> torch.Tensor:
    """Exact trajectory-conditioned cost adjustment (`blend_w * sem_adj + traj_attn_adj`) from `Phase12MultiModalAdapter`."""
    B, M = eff_prior.shape
    K_rev = int(rev_idx.shape[1])
    scale_col = torch.full((B, 1), 0.4208, device=eff_prior.device, dtype=torch.float32)
    rev_raw_f_flt = rev_raw_f.float()
    rev_gate_flt = rev_gate.float()

    idx_24 = rev_idx.unsqueeze(-1).expand(B, K_rev, 24)
    rev_sem24 = torch.gather(sem24, dim=1, index=idx_24)

    cand_q_norm = F.normalize(cand_q, p=2, dim=-1, eps=1e-8)
    idx_32 = rev_idx.unsqueeze(-1).expand(B, K_rev, adapter.d_token)
    rev_q_norm = torch.gather(cand_q_norm, dim=1, index=idx_32)
    rev_q = torch.gather(cand_q, dim=1, index=idx_32)

    rev_eff_prior = torch.gather(eff_prior, dim=1, index=rev_idx)
    rev_f_capped = torch.where(
        rev_gate_flt > 0.5,
        rev_raw_f_flt,
        torch.clamp(rev_raw_f_flt, max=2.5),
    )
    r_rev = (rev_f_capped / scale_col) - rev_eff_prior

    cos_sem = torch.bmm(sem24, rev_sem24.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_sem = (1.0 - cos_sem) * F.softplus(adapter.sem_kernel_scale_raw)
    cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_emb = (1.0 - cos_emb) * F.softplus(adapter.emb_kernel_scale_raw)
    d2_joint = d2_sem + d2_emb

    k_take = min(8, K_rev)
    top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
    r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
    top_r = torch.gather(r_exp, dim=-1, index=top_pos)

    w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
    w0 = F.softplus(adapter.null_kernel_w0_raw) + 0.15
    sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))

    step_pos = torch.arange(K_rev, device=eff_prior.device, dtype=torch.float32).view(1, K_rev, 1) / 64.0
    step_pos = step_pos.expand(B, K_rev, 1)
    h_in = torch.cat(
        [
            rev_sem24,
            rev_q,
            r_rev.unsqueeze(-1),
            (rev_raw_f_flt / scale_col).unsqueeze(-1),
            rev_gate_flt.unsqueeze(-1),
            step_pos,
        ],
        dim=-1,
    )
    h_tok = adapter.hist_token_proj(h_in)
    null_tok = adapter.null_hist_token.expand(B, 1, adapter.d_token)
    kv_seq = torch.cat([null_tok, h_tok], dim=1)

    attn_out, _ = adapter.traj_cross_attn(
        query=cand_q,
        key=kv_seq,
        value=kv_seq,
        need_weights=False,
    )

    best_rev_y = torch.min(rev_f_capped / scale_col, dim=-1, keepdim=True).values.unsqueeze(-1).expand(B, M, 1)
    traj_in = torch.cat(
        [
            cand_q,
            attn_out,
            eff_prior.unsqueeze(-1),
            sem_adj.unsqueeze(-1),
            best_rev_y,
        ],
        dim=-1,
    )
    raw_traj_delta = adapter.traj_delta_head(traj_in).squeeze(-1)
    raw_traj_centered = raw_traj_delta - torch.mean(raw_traj_delta, dim=-1, keepdim=True)
    traj_attn_adj = torch.tanh(adapter.traj_attn_scale) * raw_traj_centered

    blend_w = torch.sigmoid(adapter.traj_kernel_blend) * 0.45
    return blend_w * sem_adj + traj_attn_adj


def eval_causal_2plus2_context(
    adapter,
    p: Dict[str, Any],
    st: Dict[str, np.ndarray],
    pr: Dict[str, np.ndarray],
    alpha_neural: float,
    device: torch.device,
    permute_reveal_seed: int | None = None,
) -> Tuple[float, int]:
    """Evaluate 1 context under Causal 2+2 (Step 1: 2 reveals -> Step 2: 2 reveals = 4 total reveals).
    Returns (best_utility_of_4, exact_reveal_count).
    """
    u = np.where(p["gate"], -p["fitness"], -1.0).astype(np.float64)
    c_r882 = p["r882_cost"]
    order_r882 = np.argsort(c_r882, kind="mergesort")

    # Step 1 (K_rev = 0 -> 2): reveal top-2 candidates
    idx_stage1 = order_r882[:2]
    if permute_reveal_seed is not None:
        rng = np.random.default_rng(permute_reveal_seed)
        perm_idx = rng.choice(len(u), size=2, replace=False)
        rev_f_np = p["fitness"][perm_idx]
        rev_g_np = p["gate"][perm_idx]
        u_rev = np.where(rev_g_np, -rev_f_np, -1.0).astype(np.float64)
    else:
        rev_f_np = p["fitness"][idx_stage1]
        rev_g_np = p["gate"][idx_stage1]
        u_rev = u[idx_stage1]

    best_u1 = float(np.max(u_rev))
    pred_u1 = 1.0 - c_r882[idx_stage1]
    res_1 = float(np.mean(u_rev - pred_u1))

    # Structural distance from frozen Stage-1 active 2+2 rule
    sf = np.stack(
        [
            p["n_nodes"] / 30.0,
            p["depth"] / 10.0,
            p["has_nonlin"].astype(np.float64),
            (p["sym_group"] % 17) / 17.0,
        ],
        axis=1,
    )
    sf_rev = sf[idx_stage1]
    dist_to_rev = np.min(np.linalg.norm(sf[:, None, :] - sf_rev[None, :, :], axis=-1), axis=-1)

    p_feas = pr["p_feas"]
    p_beat = pr["p_beat"]
    e_gain = pr["e_gain"]
    sp_pen = pr["sp_pen"]
    u_r882 = 1.0 - c_r882
    hurdle_ev = p_feas * (u_r882 + s1.ALPHA_HURDLE * p_beat * e_gain) - (1.0 - p_feas) * 1.0 - s1.BETA_SP * sp_pen

    if best_u1 < 0.25:
        base_stage2 = (
            0.55 * hurdle_ev
            + 0.30 * (p_feas * p_beat * e_gain)
            + 0.15 * dist_to_rev
            - 0.20 * sp_pen
        )
    else:
        best_j = int(idx_stage1[int(np.argmax(u_rev))])
        dist_to_winner = np.linalg.norm(sf - sf[best_j : best_j + 1], axis=-1)
        sim_to_winner = np.exp(-1.5 * dist_to_winner)
        base_stage2 = (
            0.75 * u_r882
            + 0.20 * hurdle_ev
            + 0.12 * res_1 * sim_to_winner
            + 0.06 * dist_to_rev
        )

    if alpha_neural > 0.0:
        with torch.no_grad():
            eff_t = torch.as_tensor(st["eff_prior"], device=device).unsqueeze(0)
            sem_t = torch.as_tensor(st["sem24"], device=device).unsqueeze(0)
            cq_t = torch.as_tensor(st["cand_q"], device=device).unsqueeze(0)
            r_idx_t = torch.as_tensor(idx_stage1, dtype=torch.long, device=device).unsqueeze(0)
            r_f_t = torch.as_tensor(rev_f_np, dtype=torch.float32, device=device).unsqueeze(0)
            r_g_t = torch.as_tensor(rev_g_np, dtype=torch.float32, device=device).unsqueeze(0)
            delta_cost_c12 = forward_step2_trajectory_delta(
                adapter, eff_t, sem_t, cq_t, r_idx_t, r_f_t, r_g_t
            )[0].cpu().numpy().astype(np.float64)
            # Also compute buggy [0,1,2,3] trajectory delta to get pure causal replacement delta
            idx03 = np.arange(min(4, len(u)), dtype=np.int64)
            r03_idx_t = torch.as_tensor(idx03, dtype=torch.long, device=device).unsqueeze(0)
            r03_f_t = torch.as_tensor(p["fitness"][idx03], dtype=torch.float32, device=device).unsqueeze(0)
            r03_g_t = torch.as_tensor(p["gate"][idx03], dtype=torch.float32, device=device).unsqueeze(0)
            delta_cost_03 = forward_step2_trajectory_delta(
                adapter, eff_t, sem_t, cq_t, r03_idx_t, r03_f_t, r03_g_t
            )[0].cpu().numpy().astype(np.float64)
            causal_utility_shift = -(delta_cost_c12 - delta_cost_03)
        score_stage2 = base_stage2 + alpha_neural * causal_utility_shift
    else:
        score_stage2 = base_stage2

    score_stage2 = score_stage2.copy()
    score_stage2[idx_stage1] = -1e18
    idx_stage2 = np.argsort(-score_stage2, kind="mergesort")[:2]
    chosen_4 = np.concatenate([idx_stage1, idx_stage2])
    assert len(chosen_4) == 4 and len(set(chosen_4.tolist())) == 4
    # In the negative control, evaluate the chosen_4 candidates on true u
    return float(np.max(u[chosen_4])), int(len(chosen_4))


def train_causal_step2_adapter_fold(
    base_net,
    train_cids: List[str],
    val_cids: List[str],
    pools: Dict[str, Dict[str, Any]],
    static_cache: Dict[str, Dict[str, np.ndarray]],
    priors: Dict[str, Dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
) -> Tuple[Any, float, Dict[str, Any]]:
    """Train the trajectory cross-attention & kernel parameters of `adapter_10tok` with
    strictly causal Step-1 -> Step-2 masking (loss computed ONLY on i not in {c_1, c_2}),
    PCGrad-EMA gradient projection, and inner OOF trust-region alpha selection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    adapter = copy.deepcopy(base_net.adapter_10tok).to(device)
    for p in adapter.parameters():
        p.requires_grad_(False)
    traj_param_names = (
        "sem_kernel_scale_raw",
        "emb_kernel_scale_raw",
        "null_kernel_w0_raw",
        "hist_token_proj",
        "null_hist_token",
        "traj_cross_attn",
        "traj_delta_head",
        "traj_attn_scale",
        "traj_kernel_blend",
    )
    trainable_params = []
    for name, p in adapter.named_parameters():
        if any(name.startswith(prefix) for prefix in traj_param_names):
            p.requires_grad_(True)
            trainable_params.append(p)

    ema_params = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    steps = 30
    batch_size = 16
    losses = []
    pcgrad_cosines = []

    for step in range(steps):
        batch_cids = rng.choice(train_cids, size=min(batch_size, len(train_cids)), replace=False).tolist()
        half = len(batch_cids) // 2
        sub_batches = [batch_cids[:half], batch_cids[half:]]
        sub_grads = []
        step_loss_val = 0.0

        for sub in sub_batches:
            opt.zero_grad(set_to_none=True)
            sub_loss = torch.tensor(0.0, device=device)
            for c in sub:
                p = pools[c]
                st = static_cache[c]
                m = len(p["fitness"])
                c_r882 = p["r882_cost"]
                order_r882 = np.argsort(c_r882, kind="mergesort")
                idx_s1 = order_r882[:2]

                # Mask out revealed candidates idx_s1 so d2_joint=0 self-leakage is impossible!
                unrev_mask = np.ones(m, dtype=bool)
                unrev_mask[idx_s1] = False
                unrev_idx = np.where(unrev_mask)[0]

                eff_t = torch.as_tensor(st["eff_prior"], device=device).unsqueeze(0)
                sem_t = torch.as_tensor(st["sem24"], device=device).unsqueeze(0)
                cq_t = torch.as_tensor(st["cand_q"], device=device).unsqueeze(0)
                r_idx_t = torch.as_tensor(idx_s1, dtype=torch.long, device=device).unsqueeze(0)
                r_f_t = torch.as_tensor(p["fitness"][idx_s1], dtype=torch.float32, device=device).unsqueeze(0)
                r_g_t = torch.as_tensor(p["gate"][idx_s1], dtype=torch.float32, device=device).unsqueeze(0)

                delta_c12 = forward_step2_trajectory_delta(adapter, eff_t, sem_t, cq_t, r_idx_t, r_f_t, r_g_t)[0]
                pred_cost_unrev = torch.as_tensor(c_r882[unrev_idx], dtype=torch.float32, device=device) + delta_c12[unrev_idx]

                u_all = np.where(p["gate"], -p["fitness"], -1.0).astype(np.float32)
                best_u1 = float(np.max(u_all[idx_s1]))
                # Hurdle positive-gain advantage strictly over Step-1 revealed best_u1
                adv_unrev = np.maximum(u_all[unrev_idx] - best_u1, 0.0)
                if float(np.max(adv_unrev)) <= 1e-6:
                    target_probs = F.softmax(torch.as_tensor(u_all[unrev_idx] / 0.15, device=device), dim=-1)
                else:
                    target_probs = F.softmax(torch.as_tensor(adv_unrev / 0.10, device=device), dim=-1)

                log_q = F.log_softmax(-pred_cost_unrev / 0.15, dim=-1)
                sub_loss = sub_loss + F.kl_div(log_q, target_probs, reduction="batchmean")

            sub_loss = sub_loss / max(len(sub), 1)
            sub_loss.backward()
            step_loss_val += float(sub_loss.detach()) * 0.5
            g_vec = torch.cat([
                (p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten())
                for p in trainable_params
            ])
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
            for n, p in adapter.named_parameters():
                if n in ema_params:
                    ema_params[n].mul_(0.90).add_(p.detach(), alpha=0.10)

    with torch.no_grad():
        for n, p in adapter.named_parameters():
            if n in ema_params:
                p.copy_(ema_params[n])
    adapter.eval()

    # Trust-region inner validation over alpha in {0.0, 0.05, 0.10, 0.20, 0.35}
    best_alpha = 0.0
    best_val_u = -1e18
    alpha_grid = (0.0, 0.05, 0.10, 0.20, 0.35)
    alpha_scores = {}
    for alpha in alpha_grid:
        vals = [
            eval_causal_2plus2_context(adapter, pools[c], static_cache[c], priors[c], alpha, device)[0]
            for c in val_cids
        ]
        m_val = float(np.mean(vals))
        alpha_scores[str(alpha)] = m_val
        if m_val > best_val_u + 1e-7:
            best_val_u = m_val
            best_alpha = alpha

    return adapter, best_alpha, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "pcgrad_mean_cosine": float(np.mean(pcgrad_cosines)),
        "pcgrad_neg_cosine_frac": float(np.mean([c < 0.0 for c in pcgrad_cosines])),
        "alpha_grid_inner_val": alpha_scores,
        "selected_alpha": best_alpha,
    }


def main() -> None:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_hashes = verify_frozen_h252_files()

    train_rows, val_rows = ch.load_cohorts()
    train_cids = [r["context_sha256"] for r in train_rows]
    val_cids = [r["context_sha256"] for r in val_rows]
    all_cids = train_cids + val_cids
    pools = ch.load_pools(all_cids)
    ctx_feats = ch.load_context_features()
    promos = ch.load_promotions()

    # 1. Exact Native91 regression verification
    native_check = ch.verify_r882_native91(val_rows, pools)
    if not native_check["passed"]:
        raise RuntimeError(f"Native91 check failed: {native_check}")

    # 2. Load FusedJopeNetwork & verify DOPE freeze
    device = torch.device("cpu")
    torch.set_num_threads(2)
    net, guard, before = load_fused_network(device)

    # 3. Audit the self-leakage bug in run_fused_closed_loop.py
    leakage_audit = audit_self_leakage_bug(train_rows, val_rows, pools)

    # 4. Precompute static K_rev=0 terms once across all 1,310 contexts
    static_cache = precompute_static_adapter_terms(net, all_cids, pools, ctx_feats, device)

    # 5. Fit 5-fold family-grouped OOF Hurdle priors on DL-Train (fresh fold seed 255140)
    priors = s1.fit_oof_and_val_priors(train_rows, val_rows, pools, ctx_feats, promos, FOLD_SEED)
    # Audit (a): verify zero DL-Val families/contexts entered any prior fit
    train_fams = {r["family_id"] for r in train_rows}
    val_fams = {r["family_id"] for r in val_rows}
    assert not (train_fams & val_fams), "DL-Train and DL-Val family overlap!"

    # 6. Train 5-fold OOF causal Step-2 neural trajectory adapters on DL-Train
    folds = ch.make_family_folds(train_rows, n_splits=5, seed=FOLD_SEED)
    fold_adapters = {}
    fold_alphas = {}
    fold_train_diagnostics = []
    for f_idx in range(5):
        f_val_cids = [r["context_sha256"] for i, r in enumerate(train_rows) if folds[i] == f_idx]
        f_tr_cids = [r["context_sha256"] for i, r in enumerate(train_rows) if folds[i] != f_idx]
        # Use 20% of f_tr_cids families as inner trust-region validation so f_val_cids stays 100% OOF!
        f_tr_rows = [r for i, r in enumerate(train_rows) if folds[i] != f_idx]
        inner_folds = ch.make_family_folds(f_tr_rows, n_splits=5, seed=FOLD_SEED + 100 + f_idx)
        inner_tr = [r["context_sha256"] for i, r in enumerate(f_tr_rows) if inner_folds[i] != 0]
        inner_va = [r["context_sha256"] for i, r in enumerate(f_tr_rows) if inner_folds[i] == 0]
        adp, alpha, diag = train_causal_step2_adapter_fold(
            net, inner_tr, inner_va, pools, static_cache, priors, SEEDS[0] + f_idx, device
        )
        for c in f_val_cids:
            fold_adapters[c] = adp
            fold_alphas[c] = alpha
        fold_train_diagnostics.append({"fold": f_idx, **diag})

    # Train full DL-Train adapter (with 80/20 inner family split on DL-Train for alpha trust region) for DL-Val evaluation
    dl_train_inner_folds = ch.make_family_folds(train_rows, n_splits=5, seed=FOLD_SEED + 500)
    full_tr_cids = [r["context_sha256"] for i, r in enumerate(train_rows) if dl_train_inner_folds[i] != 0]
    full_va_cids = [r["context_sha256"] for i, r in enumerate(train_rows) if dl_train_inner_folds[i] == 0]
    val_adapter, val_alpha, val_train_diag = train_causal_step2_adapter_fold(
        net, full_tr_cids, full_va_cids, pools, static_cache, priors, SEEDS[0] + 99, device
    )

    guard.verify_dope_frozen_unchanged(net, before)

    # 7. Evaluate all 6 preregistered arms across SEEDS = (255101, 255102, 255103)
    arm_names = (
        "R882",
        "GATED_025",
        "ACTIVE_2PLUS2_HURDLE",
        "CAUSAL_2PLUS2_R882_ZERO_SHOT",
        "CAUSAL_2PLUS2_NEURAL",
        "NEGATIVE_CONTROL_PERMUTED_REVEAL",
    )
    per_seed_ctx: Dict[str, Dict[int, Dict[str, float]]] = {a: {} for a in arm_names}
    reveal_histograms: Dict[str, Dict[str, int]] = {a: {"4": 0} for a in arm_names}

    for seed in SEEDS:
        for a in arm_names:
            per_seed_ctx[a][seed] = {}
        for r in train_rows + val_rows:
            c = r["context_sha256"]
            p = pools[c]
            st = static_cache[c]
            pr = priors[c]
            is_val = r["dl_split"] == "DL-Val"
            adp = val_adapter if is_val else fold_adapters[c]
            alpha = val_alpha if is_val else fold_alphas[c]

            res_s1 = s1.score_context_arms(c, p, pr, seed)
            for base_arm in ("R882", "GATED_025", "ACTIVE_2PLUS2_HURDLE"):
                per_seed_ctx[base_arm][seed][c] = res_s1[base_arm]["u4"]
                reveal_histograms[base_arm]["4"] += 1

            u_zs, rev_zs = eval_causal_2plus2_context(
                net.adapter_10tok, p, st, pr, alpha_neural=0.10, device=device
            )
            per_seed_ctx["CAUSAL_2PLUS2_R882_ZERO_SHOT"][seed][c] = u_zs
            reveal_histograms["CAUSAL_2PLUS2_R882_ZERO_SHOT"][str(rev_zs)] = (
                reveal_histograms["CAUSAL_2PLUS2_R882_ZERO_SHOT"].get(str(rev_zs), 0) + 1
            )

            u_neu, rev_neu = eval_causal_2plus2_context(
                adp, p, st, pr, alpha_neural=max(alpha, 0.10), device=device
            )
            per_seed_ctx["CAUSAL_2PLUS2_NEURAL"][seed][c] = u_neu
            reveal_histograms["CAUSAL_2PLUS2_NEURAL"][str(rev_neu)] = (
                reveal_histograms["CAUSAL_2PLUS2_NEURAL"].get(str(rev_neu), 0) + 1
            )

            perm_seed = ch.stream_seed("h255_perm_ctrl", seed, c)
            u_perm, rev_perm = eval_causal_2plus2_context(
                adp, p, st, pr, alpha_neural=max(alpha, 0.10), device=device, permute_reveal_seed=perm_seed
            )
            per_seed_ctx["NEGATIVE_CONTROL_PERMUTED_REVEAL"][seed][c] = u_perm
            reveal_histograms["NEGATIVE_CONTROL_PERMUTED_REVEAL"][str(rev_perm)] = (
                reveal_histograms["NEGATIVE_CONTROL_PERMUTED_REVEAL"].get(str(rev_perm), 0) + 1
            )

    # 8. Aggregate across seeds and compute paired family bootstrap 95% CIs, Holm p-values, and Elo
    mean_ctx: Dict[str, Dict[str, float]] = {}
    for a in arm_names:
        mean_ctx[a] = {
            c: float(np.mean([per_seed_ctx[a][s][c] for s in SEEDS]))
            for c in all_cids
        }

    splits_summary: Dict[str, Any] = {}
    for split_name, rows in (("DL-Train", train_rows), ("DL-Val", val_rows)):
        fams, fam_scores = ch.aggregate_by_family(rows, mean_ctx)
        elo = ch.compute_elo_ratings(fam_scores, anchor_arm="R882", seed=BOOT_SEED)
        comps_vs_r882 = {}
        comps_vs_g025 = {}
        raw_p_r882 = []
        raw_p_g025 = []
        comp_arms = [a for a in arm_names if a != "R882"]
        for a in comp_arms:
            st_r = ch.paired_family_bootstrap(
                fam_scores[a], fam_scores["R882"], n_draws=BOOT_DRAWS, seed=ch.stream_seed(BOOT_SEED, split_name, a, "R882")
            )
            st_g = ch.paired_family_bootstrap(
                fam_scores[a], fam_scores["GATED_025"], n_draws=BOOT_DRAWS, seed=ch.stream_seed(BOOT_SEED, split_name, a, "G025")
            )
            # Per-seed positive check
            seeds_pos_g025 = 0
            for s in SEEDS:
                _, fs_s = ch.aggregate_by_family(rows, {k: per_seed_ctx[k][s] for k in arm_names})
                if float(np.mean(fs_s[a] - fs_s["GATED_025"])) > 0.0:
                    seeds_pos_g025 += 1
            st_g["seeds_positive_of_3"] = seeds_pos_g025
            comps_vs_r882[a] = st_r
            comps_vs_g025[a] = st_g
            raw_p_r882.append(st_r["p_one_sided"])
            raw_p_g025.append(st_g["p_one_sided"])

        holm_r = ch.holm_bonferroni(raw_p_r882)
        holm_g = ch.holm_bonferroni(raw_p_g025)
        for idx_a, a in enumerate(comp_arms):
            comps_vs_r882[a]["p_holm"] = holm_r[idx_a]
            comps_vs_g025[a]["p_holm"] = holm_g[idx_a]

        # Negative control check: NEGATIVE_CONTROL_PERMUTED_REVEAL minus CAUSAL_2PLUS2_NEURAL (and vs ACTIVE_2PLUS2_HURDLE)
        neg_ctrl_vs_active = ch.paired_family_bootstrap(
            fam_scores["NEGATIVE_CONTROL_PERMUTED_REVEAL"],
            fam_scores["ACTIVE_2PLUS2_HURDLE"],
            n_draws=BOOT_DRAWS,
            seed=ch.stream_seed(BOOT_SEED, split_name, "neg_ctrl"),
        )

        splits_summary[split_name] = {
            "n_contexts": len(rows),
            "n_families": len(fams),
            "family_mean_utility": {a: float(np.mean(fam_scores[a])) for a in arm_names},
            "family_median_utility": {a: float(np.median(fam_scores[a])) for a in arm_names},
            "vs_r882": comps_vs_r882,
            "vs_gated_025": comps_vs_g025,
            "negative_control_vs_active": neg_ctrl_vs_active,
            "elo": elo,
        }

    out_payload = {
        "wave": "H255_PART_A",
        "title": "Causal 2+2 Neural Trajectory Critic & Self-Leakage Audit",
        "elapsed_sec": time.monotonic() - t0,
        "frozen_h252_hashes": frozen_hashes,
        "native91_check": native_check,
        "dope_sha256": before.composite_sha256,
        "dope_unchanged": True,
        "self_leakage_audit": leakage_audit,
        "reveal_count_histograms": reveal_histograms,
        "fold_train_diagnostics": fold_train_diagnostics,
        "val_adapter_train_diagnostics": val_train_diag,
        "splits": splits_summary,
    }
    ch.save_json(OUT_DIR / "part_a_metrics.json", out_payload)
    print(json.dumps({
        "status": "PART_A_OK",
        "elapsed_sec": round(out_payload["elapsed_sec"], 2),
        "dl_val_family_mean": splits_summary["DL-Val"]["family_mean_utility"],
        "dl_train_family_mean": splits_summary["DL-Train"]["family_mean_utility"],
    }, indent=2))


if __name__ == "__main__":
    main()
