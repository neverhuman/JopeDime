#!/usr/bin/env python3
"""H255 Part A: Preregistered Causal 2+2 Neural Trajectory Critic (K_rev=0 -> K_rev=2 -> 4)
and Root-Cause Audit of the `revealed_indices=[0,1,2,3]` Oracle Self-Leakage Bug in `run_fused_closed_loop.py`.

Guardrails:
- Reads H252 `common_h252.py` and `stage1_hurdle_sos_critic.py` read-only after verifying their exact frozen SHA-256s.
- Never writes to `/home/ubuntu/jope-data/h252/`.
- Never touches DL-Val during any prior fit or neural parameter update (5-fold family-grouped OOF on DL-Train; DL-Val uses model fit on DL-Train only).
- Verifies DOPE SHA-256 `c51a56d0...` before and after training, and reproduces R882 Native91 `1.1377941898164525` to 0.00e+00 first.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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


def load_r882_cache_readonly(all_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Read R882 costs and base priors into memory without touching /home/ubuntu/jope-data/h252/stage1."""
    cids_sorted = sorted(r["context_sha256"] for r in all_rows)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(s1.R882_TRAIN_COSTS_NPZ, allow_pickle=False) as zt, np.load(
        s1.R882_VAL_COSTS_NPZ, allow_pickle=False
    ) as zv, np.load(s1.POOL_PRIORS_NPZ, allow_pickle=False) as zp:
        pri_cids = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        base_pri = zp["base_prior_cost"]
        pri_map = {
            pri_cids[i]: base_pri[offsets[i] : offsets[i + 1]].astype(np.float32)
            for i in range(len(pri_cids))
        }
        for cid in cids_sorted:
            tc = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
            bp = pri_map[cid]
            out[cid] = {
                "total_cost": tc,
                "anchor_total": tc,
                "anchor_prior": bp,
            }
    return out


def load_fused_network(device: torch.device):
    if str(MICRO / "python") not in sys.path:
        sys.path.insert(0, str(MICRO / "python"))
    sys.modules.pop("research", None)
    import rust_evaluator as evaluator  # noqa: F401
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
    return net, guard, before, evaluator.get_w1_sign_matrix()


def audit_self_leakage_bug(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    r882_cache: Dict[str, Dict[str, np.ndarray]],
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
        top2_r882 = set(np.argsort(r882_cache[c]["total_cost"])[:2].tolist())
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
    all_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    ctx_feats: Dict[str, Dict[str, np.ndarray]],
    w1_mat: np.ndarray,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute the K_rev=0 (`revealed_indices=None`) neural terms and buggy `[0,1,2,3]` trajectory delta once per context."""
    net.eval()
    static_cache: Dict[str, Dict[str, np.ndarray]] = {}
    w1_f32 = w1_mat.astype(np.float32) / np.sqrt(412.0)
    with torch.no_grad():
        for r in all_rows:
            c = r["context_sha256"]
            p = pools[c]
            cf = ctx_feats[c]
            prog = p["program"]
            m = len(prog)
            proj16 = (prog @ w1_f32).astype(np.float32)
            p["proj16"] = proj16
            out0 = net.adapter_10tok.predict_costs(
                ctx_feat=torch.as_tensor(cf["ctx_feat_584"], device=device).reshape(1, 584),
                prog_412=torch.as_tensor(prog, device=device).reshape(1, m, 412),
                proj_16=torch.as_tensor(proj16, device=device).reshape(1, m, 16),
                base_prior_cost=torch.zeros((1, m), device=device),
                scale=torch.full((1,), 0.4208, device=device),
                sos_prior=torch.zeros((1, m, 2), device=device),
                mm_tokens_10x32=torch.as_tensor(cf["tokens_10x32"], device=device).reshape(1, 10, 32),
                mm_mask_10=torch.as_tensor(cf["mask_10"], device=device).reshape(1, 10),
                revealed_indices=torch.zeros((1, 0), dtype=torch.long, device=device),
                revealed_raw_f=torch.zeros((1, 0), dtype=torch.float32, device=device),
                revealed_gate=torch.zeros((1, 0), dtype=torch.float32, device=device),
            )
            eff_prior = out0["eff_prior"][0]
            sem24 = out0["sem24"][0]
            cand_q = out0["cand_q"][0]
            idx03 = torch.arange(min(4, m), dtype=torch.long, device=device).unsqueeze(0)
            f03 = torch.as_tensor(p["fitness"][: min(4, m)], dtype=torch.float32, device=device).unsqueeze(0)
            g03 = torch.as_tensor(p["gate"][: min(4, m)], dtype=torch.float32, device=device).unsqueeze(0)
            delta03 = forward_step2_trajectory_delta(
                net.adapter_10tok,
                eff_prior.unsqueeze(0),
                sem24.unsqueeze(0),
                cand_q.unsqueeze(0),
                idx03,
                f03,
                g03,
            )[0]
            static_cache[c] = {
                "eff_prior": eff_prior.cpu().numpy().astype(np.float32),
                "sem24": sem24.cpu().numpy().astype(np.float32),
                "cand_q": cand_q.cpu().numpy().astype(np.float32),
                "delta03_r882": delta03.cpu().numpy().astype(np.float64),
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


def score_all_arms_for_context(
    cid: str,
    pool: Dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    st: Dict[str, np.ndarray],
    base_adapter,
    trained_adapter,
    alpha_neural: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Evaluate all 6 preregistered Part A arms on context `cid` using the exact same `idx4` draw:
    1. `R882`
    2. `GATED_025`
    3. `ACTIVE_2PLUS2_HURDLE` (frozen Stage-1 formula)
    4. `CAUSAL_2PLUS2_R882_ZERO_SHOT` (replaces buggy [0,1,2,3] neural trajectory conditioning with true `rev4`)
    5. `CAUSAL_2PLUS2_NEURAL` (causally trained trajectory head conditioned on true `rev4`)
    6. `NEGATIVE_CONTROL_PERMUTED_REVEAL` (revealed fitnesses on `rev4` permuted within context)
    """
    u = pool["utility"]
    fitness = pool["fitness"]
    logratio = pool["logratio"]
    gate = pool["gate"]
    sem24 = pool["sem24"].astype(np.float64)
    n = len(rcost)

    idx4 = rng.choice(n, size=4, replace=False)
    top_zgap = s1.compute_r882_top_zgap(rcost)

    pick_r882 = int(np.argmin(rcost))
    u_r882 = float(u[pick_r882])

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
    X = (sem24 - s_mean) / s_std

    # GATED_025
    if top_zgap < 0.25:
        y4_cost = np.where(gate[idx4], fitness[idx4], 2.0)
        y4_c = y4_cost - float(np.mean(y4_cost))
        diff = X[:, None, :] - X[idx4][None, :, :]
        sq = np.sum(diff * diff, axis=-1) / 24.0
        K = np.exp(-sq / (2.0 * (2.0 ** 2)))
        w = K / (np.sum(K, axis=1, keepdims=True) + 0.2)
        cost_g025 = rcost + 0.05 * (w @ y4_c)
        u_g025 = float(u[int(np.argmin(cost_g025))])
    else:
        u_g025 = u_r882

    # Step 1 (2 reveals: idx_a = idx4[:2]) -> Step 2 (2 active reveals: chosen_b) -> rev4 (4 reveals total)
    if top_zgap < 0.25:
        idx_a = idx4[:2]
        ya_cost = np.where(gate[idx_a], fitness[idx_a], 2.0)
        ya_c = ya_cost - float(np.mean(ya_cost))
        diff_a = X[:, None, :] - X[idx_a][None, :, :]
        sq_a = np.sum(diff_a * diff_a, axis=-1) / 24.0
        Ka = np.exp(-sq_a / (2.0 * (2.2 ** 2)))
        wa = Ka / (np.sum(Ka, axis=1, keepdims=True) + 0.2)
        h_ca = h_cost - float(np.mean(h_cost[idx_a]))
        interim_cost = rcost + 0.035 * (wa @ ya_c) + 0.010 * h_ca

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
        assert len(rev4) == 4 and len(set(rev4.tolist())) == 4
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
        u_act = float(u[int(np.argmin(cost_act))])

        # Causal neural trajectory conditioning on the actual 4 revealed candidates `rev4`
        with torch.no_grad():
            eff_t = torch.as_tensor(st["eff_prior"], device=device).unsqueeze(0)
            sem_t = torch.as_tensor(st["sem24"], device=device).unsqueeze(0)
            cq_t = torch.as_tensor(st["cand_q"], device=device).unsqueeze(0)
            rev_idx_t = torch.as_tensor(rev4, dtype=torch.long, device=device).unsqueeze(0)
            rev_f_t = torch.as_tensor(fitness[rev4], dtype=torch.float32, device=device).unsqueeze(0)
            rev_g_t = torch.as_tensor(gate[rev4], dtype=torch.float32, device=device).unsqueeze(0)

            delta_zs = forward_step2_trajectory_delta(
                base_adapter, eff_t, sem_t, cq_t, rev_idx_t, rev_f_t, rev_g_t
            )[0].cpu().numpy().astype(np.float64)
            delta_tr = forward_step2_trajectory_delta(
                trained_adapter, eff_t, sem_t, cq_t, rev_idx_t, rev_f_t, rev_g_t
            )[0].cpu().numpy().astype(np.float64)

        # Zero-shot causal neural trajectory update on rev4
        cost_zs = cost_act + 0.025 * (delta_zs - st["delta03_r882"])
        u_zs = float(u[int(np.argmin(cost_zs))])

        # Trained causal neural trajectory update on rev4 (gated by OOF trust-region alpha_neural)
        cost_neu = cost_act + alpha_neural * (delta_tr - st["delta03_r882"])
        u_neu = float(u[int(np.argmin(cost_neu))])

        # Negative Control: permute revealed outcomes across pool indices
        perm_idx = rng.choice(n, size=4, replace=False)
        y4_f_perm = np.where(gate[perm_idx], fitness[perm_idx], 2.0)
        y4_lr_perm = np.clip(logratio[perm_idx], -1.5, 1.5)
        y4_comb_perm = 0.70 * (y4_f_perm - float(np.mean(y4_f_perm))) + 0.30 * (
            y4_lr_perm - float(np.mean(y4_lr_perm))
        )
        cost_perm = rcost + 0.05 * (w4 @ y4_comb_perm)
        u_perm = float(u[int(np.argmin(cost_perm))])
    else:
        u_act = u_r882
        u_zs = u_r882
        u_neu = u_r882
        u_perm = u_r882

    scores = {
        "R882": u_r882,
        "GATED_025": u_g025,
        "ACTIVE_2PLUS2_HURDLE": u_act,
        "CAUSAL_2PLUS2_R882_ZERO_SHOT": u_zs,
        "CAUSAL_2PLUS2_NEURAL": u_neu,
        "NEGATIVE_CONTROL_PERMUTED_REVEAL": u_perm,
    }
    rev_counts = {k: 4 for k in scores}
    return scores, rev_counts


def train_causal_step2_adapter_fold(
    base_net,
    train_cids: List[str],
    val_cids: List[str],
    pools: Dict[str, Dict[str, Any]],
    r882_cache: Dict[str, Dict[str, np.ndarray]],
    static_cache: Dict[str, Dict[str, np.ndarray]],
    sos_priors: Dict[str, np.ndarray],
    sp_priors: Dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> Tuple[Any, float, Dict[str, Any]]:
    """Train the trajectory cross-attention & kernel parameters of `adapter_10tok` with
    strictly causal masking (loss computed ONLY on unrevealed candidates i not in rev4),
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

    steps = 24
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
                rcost = r882_cache[c]["total_cost"].astype(np.float64)
                m = len(rcost)
                rev4 = rng.choice(m, size=4, replace=False)

                # Mask out revealed candidates rev4 so d2_joint=0 self-leakage is impossible!
                unrev_mask = np.ones(m, dtype=bool)
                unrev_mask[rev4] = False
                unrev_idx = np.where(unrev_mask)[0]

                eff_t = torch.as_tensor(st["eff_prior"], device=device).unsqueeze(0)
                sem_t = torch.as_tensor(st["sem24"], device=device).unsqueeze(0)
                cq_t = torch.as_tensor(st["cand_q"], device=device).unsqueeze(0)
                r_idx_t = torch.as_tensor(rev4, dtype=torch.long, device=device).unsqueeze(0)
                r_f_t = torch.as_tensor(p["fitness"][rev4], dtype=torch.float32, device=device).unsqueeze(0)
                r_g_t = torch.as_tensor(p["gate"][rev4], dtype=torch.float32, device=device).unsqueeze(0)

                delta_rev4 = forward_step2_trajectory_delta(adapter, eff_t, sem_t, cq_t, r_idx_t, r_f_t, r_g_t)[0]
                pred_cost_unrev = (
                    torch.as_tensor(rcost[unrev_idx] - st["delta03_r882"][unrev_idx], dtype=torch.float32, device=device)
                    + delta_rev4[unrev_idx]
                )

                u_all = p["utility"].astype(np.float32)
                best_u_rev = float(np.max(u_all[rev4]))
                adv_unrev = np.maximum(u_all[unrev_idx] - best_u_rev, 0.0)
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

    # Inner trust-region validation over alpha in {0.0, 0.01, 0.025, 0.05}
    alpha_grid = (0.0, 0.01, 0.025, 0.05)
    alpha_scores = {}
    best_alpha = 0.0
    best_val_u = -1e18
    for alpha in alpha_grid:
        rng_inner = np.random.default_rng(seed + 777)
        vals = []
        for c in val_cids:
            sc, _ = score_all_arms_for_context(
                c,
                pools[c],
                r882_cache[c]["total_cost"].astype(np.float64),
                sos_priors[c],
                sp_priors[c],
                static_cache[c],
                base_net.adapter_10tok,
                adapter,
                alpha,
                rng_inner,
                device,
            )
            vals.append(sc["CAUSAL_2PLUS2_NEURAL"])
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
    all_rows = train_rows + val_rows
    pools = ch.load_pools(all_rows, verify_hashes=False)
    ctx_feats = ch.load_context_features()
    promos_by_cid = ch.load_harvested_promotions(train_rows)

    # 1. Load R882 cache read-only and verify exact Native91 = 1.1377941898164525
    r882_cache = load_r882_cache_readonly(all_rows)
    val_native_utils = [
        float(pools[r["context_sha256"]]["utility"][int(np.argmin(r882_cache[r["context_sha256"]]["total_cost"]))])
        for r in val_rows
    ]
    val_native91_mean = float(np.mean(val_native_utils))
    diff_native = abs(val_native91_mean - ch.R882_NATIVE91_MEAN)
    if diff_native > 1e-12:
        raise RuntimeError(f"Native91 mismatch: {val_native91_mean} != {ch.R882_NATIVE91_MEAN}")

    # 2. Load FusedJopeNetwork & verify DOPE freeze
    device = torch.device("cpu")
    torch.set_num_threads(2)
    net, guard, before, w1_mat = load_fused_network(device)

    # 3. Audit the self-leakage bug in run_fused_closed_loop.py
    leakage_audit = audit_self_leakage_bug(train_rows, val_rows, pools, r882_cache)

    # 4. Precompute static K_rev=0 terms once across all 1,310 contexts
    static_cache = precompute_static_adapter_terms(net, all_rows, pools, ctx_feats, w1_mat, device)

    # 5. Build 64D candidate features & fit 5-fold OOF Hurdle priors on DL-Train (fresh fold seed 255140)
    cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, ctx_feats)
    sos_priors_f0, sp_priors_f0, hurdle_diag_f0 = s1.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats, promos_by_cid, seed=FOLD_SEED
    )

    # Audit (a): verify zero DL-Val families/contexts entered any prior fit
    train_fams = {r["family_id"] for r in train_rows}
    val_fams = {r["family_id"] for r in val_rows}
    assert not (train_fams & val_fams), "DL-Train and DL-Val family overlap!"

    # 6. Train 5-fold OOF causal Step-2 neural trajectory adapters on DL-Train
    _, cid_to_fold = ch.seed_folds(train_rows, FOLD_SEED)
    fold_adapters = {}
    fold_alphas = {}
    fold_train_diagnostics = []
    for f_idx in range(5):
        f_val_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == f_idx]
        f_tr_rows = [r for r in train_rows if cid_to_fold[r["context_sha256"]] != f_idx]
        # Inner 80/20 split by family inside f_tr_rows for trust-region alpha selection
        tr_fams_sorted = sorted({r["family_id"] for r in f_tr_rows})
        inner_val_fams = set(tr_fams_sorted[::5])
        inner_tr_cids = [r["context_sha256"] for r in f_tr_rows if r["family_id"] not in inner_val_fams]
        inner_va_cids = [r["context_sha256"] for r in f_tr_rows if r["family_id"] in inner_val_fams]
        adp, alpha, diag = train_causal_step2_adapter_fold(
            net,
            inner_tr_cids,
            inner_va_cids,
            pools,
            r882_cache,
            static_cache,
            sos_priors_f0,
            sp_priors_f0,
            SEEDS[0] + f_idx,
            device,
        )
        for c in f_val_cids:
            fold_adapters[c] = adp
            fold_alphas[c] = alpha
        fold_train_diagnostics.append({"fold": f_idx, **diag})

    # Full DL-Train fit (with 80/20 inner family split on DL-Train) for DL-Val evaluation
    all_tr_fams = sorted({r["family_id"] for r in train_rows})
    full_inner_val_fams = set(all_tr_fams[::5])
    full_tr_cids = [r["context_sha256"] for r in train_rows if r["family_id"] not in full_inner_val_fams]
    full_va_cids = [r["context_sha256"] for r in train_rows if r["family_id"] in full_inner_val_fams]
    val_adapter, val_alpha, val_train_diag = train_causal_step2_adapter_fold(
        net,
        full_tr_cids,
        full_va_cids,
        pools,
        r882_cache,
        static_cache,
        sos_priors_f0,
        sp_priors_f0,
        SEEDS[0] + 99,
        device,
    )

    guard.verify_dope_frozen_unchanged(net, before)

    # 7. Evaluate all 6 preregistered arms across SEEDS = (255101, 255102, 255103)
    arm_names = [
        "R882",
        "GATED_025",
        "ACTIVE_2PLUS2_HURDLE",
        "CAUSAL_2PLUS2_R882_ZERO_SHOT",
        "CAUSAL_2PLUS2_NEURAL",
        "NEGATIVE_CONTROL_PERMUTED_REVEAL",
    ]
    train_panel = np.zeros((len(SEEDS), len(train_rows), len(arm_names)), dtype=np.float64)
    val_panel = np.zeros((len(SEEDS), len(val_rows), len(arm_names)), dtype=np.float64)
    reveal_histograms: Dict[str, Dict[str, int]] = {a: {} for a in arm_names}

    for si, seed in enumerate(SEEDS):
        sos_priors, sp_priors, _ = s1.fit_decomposed_hurdle_and_sp_priors(
            train_rows, val_rows, pools, cand_feats, promos_by_cid, seed=seed
        )
        rng_train = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            cid = r["context_sha256"]
            sc, rc = score_all_arms_for_context(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"].astype(np.float64),
                sos_priors[cid],
                sp_priors[cid],
                static_cache[cid],
                net.adapter_10tok,
                fold_adapters[cid],
                fold_alphas[cid],
                rng_train,
                device,
            )
            for ai, a in enumerate(arm_names):
                train_panel[si, ci, ai] = sc[a]
                k_str = str(rc[a])
                reveal_histograms[a][k_str] = reveal_histograms[a].get(k_str, 0) + 1

        rng_val = np.random.default_rng(seed + 1000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            sc, rc = score_all_arms_for_context(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"].astype(np.float64),
                sos_priors[cid],
                sp_priors[cid],
                static_cache[cid],
                net.adapter_10tok,
                val_adapter,
                val_alpha,
                rng_val,
                device,
            )
            for ai, a in enumerate(arm_names):
                val_panel[si, ci, ai] = sc[a]
                k_str = str(rc[a])
                reveal_histograms[a][k_str] = reveal_histograms[a].get(k_str, 0) + 1

    # 8. Summarize panels vs R882 (ref_idx=0) and vs GATED_025 (ref_idx=1)
    train_vs_r882 = ch.summarize_panel(train_rows, train_panel, arm_names, ref_idx=0, boot_seed=BOOT_SEED)
    train_vs_g025 = ch.summarize_panel(train_rows, train_panel, arm_names, ref_idx=1, boot_seed=BOOT_SEED)
    val_vs_r882 = ch.summarize_panel(val_rows, val_panel, arm_names, ref_idx=0, boot_seed=BOOT_SEED)
    val_vs_g025 = ch.summarize_panel(val_rows, val_panel, arm_names, ref_idx=1, boot_seed=BOOT_SEED)

    out_payload = {
        "wave": "H255_PART_A",
        "title": "Causal 2+2 Neural Trajectory Critic & Self-Leakage Audit",
        "elapsed_sec": time.monotonic() - t0,
        "frozen_h252_hashes": frozen_hashes,
        "r882_native91_exact": val_native91_mean,
        "r882_native91_diff": diff_native,
        "dope_sha256": before.composite_sha256,
        "dope_unchanged": True,
        "self_leakage_audit": leakage_audit,
        "reveal_count_histograms": reveal_histograms,
        "hurdle_oof_diagnostics": hurdle_diag_f0,
        "fold_train_diagnostics": fold_train_diagnostics,
        "val_adapter_train_diagnostics": val_train_diag,
        "dl_train_vs_r882": train_vs_r882,
        "dl_train_vs_gated_025": train_vs_g025,
        "dl_val_vs_r882": val_vs_r882,
        "dl_val_vs_gated_025": val_vs_g025,
    }
    ch.save_json(OUT_DIR / "part_a_metrics.json", out_payload)
    print(json.dumps({
        "status": "PART_A_OK",
        "elapsed_sec": round(out_payload["elapsed_sec"], 2),
        "arms": arm_names,
        "dl_val_median_family_utility": val_vs_r882["median_family_utility"],
        "dl_train_median_family_utility": train_vs_r882["median_family_utility"],
        "dl_val_elo": val_vs_r882["median_elo"],
    }, indent=2))


if __name__ == "__main__":
    main()
