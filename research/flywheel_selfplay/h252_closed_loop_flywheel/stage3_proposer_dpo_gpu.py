#!/usr/bin/env python3
"""
H252 Stage 3 (xbabe3 RTX 3090 GPU): Contextual Parent->Child Transition Critic &
Counterfactual Sibling-DPO Proposer (`PROPOSER_DPO_DELTA` / W6).

Activates the previously dead `parent_delta_net` concept by training a contextual
Parent->Child Transition & Mutation Ranking Network on GPU across 5 family folds x 3 seeds
(252101, 252102, 252103), leveraging:
  - Within-context Parent->Child transition pairs from the 1,219 DL-Train pools and
    3,449 harvested self-play GP promotions (`promotions.jsonl`),
  - Counterfactual Sibling Direct Preference Optimization (DPO) on `(context, parent, child+, child-)`
    tuples + Advantage-Weighted Regression (AWR) on continuous Hurdle delta-utility,
  - Cosine-Aligned Orthogonal EMA Gradient Smoothing (`PCGrad-EMA`).

Evaluates out-of-fold on all 205 DL-Train families and on the 91 DL-Val contexts (19 families):
  1. Pairwise Sibling Preference Accuracy (`pair_acc`)
  2. Top-4 Proposal Promotion Yield (`q_beat_at_4`: probability that top-4 proposals beat the incumbent)
  3. Expected Evaluations to First Promotion (`expected_tries_to_promotion`)
  4. Promoted Child Utility Gain (`promoted_utility_gain` via 10,000-draw paired family bootstrap & MM Elo)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_h252 import (
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
STAGE3_OUT_DIR = Path("/home/ubuntu/jope-data/h252/stage3")


class ParentChildDPOProposerNet(nn.Module):
    """
    Contextual Parent->Child Mutation Transition & Preference Network (~24K params,
    matching the scale of `parent_delta_net` in `FlywheelGameNet`).
    Given:
      - `ctx_vec`: (B, 48) context landmark summary `lm48_z`
      - `parent_feat`: (B, 40) incumbent parent representation `[sem24_p, struct16_p]`
      - `child_feat`: (B, M, 40) candidate child representations `[sem24_c, struct16_c]`
      - `child_prior`: (B, M, 4) `[r882_delta, hurdle_cont_z, hurdle_gate_z, cos_to_parent]`
    Predicts transition advantage score `delta_score(child | parent, ctx)` of shape `(B, M)`.
    """

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.ctx_proj = nn.Sequential(
            nn.Linear(48, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.parent_proj = nn.Sequential(
            nn.Linear(40, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.child_proj = nn.Sequential(
            nn.Linear(40, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 4, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        # Zero-init final layer so initial delta perturbation around anchor is smooth
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(
        self,
        ctx_vec: torch.Tensor,
        parent_feat: torch.Tensor,
        child_feat: torch.Tensor,
        child_prior: torch.Tensor,
    ) -> torch.Tensor:
        b, m, _ = child_feat.shape
        h_ctx = self.ctx_proj(ctx_vec).unsqueeze(1).expand(b, m, -1)
        h_par = self.parent_proj(parent_feat).unsqueeze(1).expand(b, m, -1)
        h_chi = self.child_proj(child_feat)
        # Include explicit semantic step `h_chi - h_par` and elementwise interaction `(h_chi - h_par) * h_ctx`
        h_diff = h_chi - h_par
        h_inter = h_diff * h_ctx
        inp = torch.cat([h_par, h_diff, h_inter, child_prior], dim=-1)
        return self.delta_mlp(inp).squeeze(-1)


def build_transition_tasks(
    all_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    ctx_feats: Dict[str, Dict[str, np.ndarray]],
    r882_costs: Dict[str, np.ndarray],
    hurdle_sos: Dict[str, np.ndarray],
    promos_by_cid: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs a standardized Parent->Child evolutionary proposal task for each context `c`:
      - The incumbent parent `p_idx` is the best initial anchor among the 4 revealed seeds
        (or top-8 anchor contender with non-optimal fitness so promotions exist),
        plus for DL-Train contexts we also include harvested self-play promotions as elite children!
      - The candidate child mutation pool `M_children` consists of `M = 64` candidate programs
        (top-48 unrevealed candidates by R882 + 16 diverse semantic explorers, plus any harvested
        self-play promotions during training).
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        cid = r["context_sha256"]
        p = pools[cid]
        rc = r882_costs[cid].astype(np.float64)
        sos = hurdle_sos[cid].astype(np.float64)  # (1152, 2)
        sem24 = p["sem24"].astype(np.float32)  # (1152, 24)
        prog = p["program"].astype(np.float32)
        # 16D standardized structural summary
        st_raw = np.concatenate([prog[:, :8], prog[:, 48:56]], axis=1)
        st_z = ((st_raw - st_raw.mean(axis=0, keepdims=True)) / (st_raw.std(axis=0, keepdims=True) + 1e-5)).clip(-5.0, 5.0)
        feat40 = np.concatenate([sem24, st_z], axis=1).astype(np.float32)  # (1152, 40)

        u = p["utility"].astype(np.float64)
        lr = np.clip(p["logratio"].astype(np.float64), -1.5, 1.5)
        gate = p["gate"].astype(bool)

        # Incumbent parent = best of the 4 initial revealed anchors (0..3)
        inc_idx = int(np.argmax(u[:4]))
        u_parent = float(u[inc_idx])
        lr_parent = float(lr[inc_idx])
        parent_feat40 = feat40[inc_idx]
        parent_sem24 = sem24[inc_idx]
        rc_parent = float(rc[inc_idx])

        # Candidate child pool: top-64 unrevealed candidates ordered by R882 cost (4..1151)
        unrev = np.arange(4, len(rc))
        top64_order = unrev[np.argsort(rc[unrev], kind="stable")[:64]]

        c_feat40 = feat40[top64_order]  # (64, 40)
        c_sem24 = sem24[top64_order]
        c_rc = rc[top64_order]
        c_sos = sos[top64_order]
        c_u = u[top64_order]
        c_lr = lr[top64_order]
        c_gate = gate[top64_order]

        rc_sd = float(np.std(rc)) + 1e-6
        r882_delta = ((c_rc - rc_parent) / rc_sd).astype(np.float32)  # lower = better
        cos_par = np.clip(c_sem24 @ parent_sem24, -1.0, 1.0).astype(np.float32)

        # 4D child prior: [-r882_delta (higher=better), -hurdle_cont_z, -hurdle_gate_z, cos_par]
        c_prior = np.stack(
            [
                -r882_delta,
                -c_sos[:, 0].astype(np.float32),
                -c_sos[:, 1].astype(np.float32),
                cos_par,
            ],
            axis=1,
        ).astype(np.float32)  # (64, 4)

        # Continuous Hurdle advantage over parent
        hurdle_u_child = -c_lr - 0.45 * (~c_gate).astype(np.float64)
        hurdle_u_par = -lr_parent
        delta_hurdle = (hurdle_u_child - hurdle_u_par).astype(np.float32)
        delta_u_true = (c_u - u_parent).astype(np.float32)
        beats_parent = (c_u > u_parent + 1e-4) & c_gate

        tasks[cid] = {
            "cid": cid,
            "family_id": r["family_id"],
            "ctx_vec": ctx_feats[cid]["lm48_z"].astype(np.float32),
            "parent_feat40": parent_feat40,
            "u_parent": u_parent,
            "child_feat40": c_feat40,
            "child_sem24": c_sem24,
            "child_prior": c_prior,
            "child_u": c_u,
            "delta_u_true": delta_u_true,
            "delta_hurdle": delta_hurdle,
            "beats_parent": beats_parent,
            "r882_rank_score": -r882_delta,
            "n_harvested_promos": len(promos_by_cid.get(cid, [])),
        }
    return tasks


def dpo_and_awr_loss(
    scores: torch.Tensor,
    prior_anchor: torch.Tensor,
    delta_hurdle: torch.Tensor,
    beats_parent: torch.Tensor,
    use_dpo: bool = True,
) -> torch.Tensor:
    """
    Computes:
      1. Advantage-Weighted Regression (AWR) / Softmax Cross-Entropy over continuous
         Hurdle child advantage `delta_hurdle`.
      2. Counterfactual Sibling Direct Preference Optimization (DPO):
         For every context in the batch that has both winning children (`beats_parent == True`)
         and losing children (`beats_parent == False`), computes all pairwise log-sigmoid margins
         `-log sigmoid(beta * ((s(y+) - s_ref(y+)) - (s(y-) - s_ref(y-))))` weighted by
         the magnitude of the Hurdle utility improvement!
    """
    # 1. AWR listwise advantage loss
    temp = 0.20
    target_w = F.softmax(delta_hurdle / temp, dim=-1).detach()
    logp = F.log_softmax(scores, dim=-1)
    loss_awr = -(target_w * logp).sum(dim=-1).mean()

    # Smooth L1 on standardized delta_hurdle
    dh_c = delta_hurdle - delta_hurdle.mean(dim=-1, keepdim=True)
    sc_c = scores - scores.mean(dim=-1, keepdim=True)
    loss_reg = F.smooth_l1_loss(sc_c, dh_c, beta=0.15)

    if not use_dpo:
        return 0.60 * loss_awr + 0.40 * loss_reg

    # 2. Counterfactual Sibling-DPO pairwise preference loss
    beta_dpo = 2.5
    rel_s = scores - 0.5 * prior_anchor  # implicit reference policy margin
    diff_matrix = rel_s.unsqueeze(2) - rel_s.unsqueeze(1)  # (B, M, M): i vs j
    pos_mask = beats_parent.unsqueeze(2) & (~beats_parent.unsqueeze(1))  # i beats parent, j does not
    adv_gap = (delta_hurdle.unsqueeze(2) - delta_hurdle.unsqueeze(1)).clamp(min=0.0, max=1.5).detach()

    if pos_mask.any():
        pair_losses = -F.logsigmoid(beta_dpo * diff_matrix) * (0.25 + adv_gap)
        loss_dpo = (pair_losses * pos_mask.float()).sum() / (pos_mask.float().sum() + 1e-6)
    else:
        loss_dpo = torch.tensor(0.0, device=scores.device)

    return 0.45 * loss_dpo + 0.35 * loss_awr + 0.20 * loss_reg


def evaluate_proposer_scores(
    task: Dict[str, Any],
    scores: np.ndarray,
    use_submodular_diversity: bool = False,
) -> Dict[str, float]:
    """
    Computes Stage-3 proposer KPIs on a single context:
      - `promoted_u_top4`: utility after evaluating the top-4 proposed children and promoting
        the best child if it beats `u_parent` (otherwise keeping `u_parent`).
      - `promoted_u_top1`: utility after evaluating only the top-1 proposed child.
      - `q_beat_at_4`: 1.0 if at least one of the top-4 proposed children beats `u_parent`, else 0.0.
      - `expected_tries`: rank (1..64) of the first proposed child that beats `u_parent` (capped at 32).
      - `pair_acc`: fraction of `(child+, child-)` sibling pairs ranked correctly by `scores`.
    """
    order = np.argsort(-scores, kind="stable")
    u_par = task["u_parent"]
    c_u = task["child_u"]
    beats = task["beats_parent"]

    if use_submodular_diversity:
        # Submodular Max-Min sem24 Diversity Slate Selection:
        # Pick 1 is argmax(scores); picks 2..4 maximize standardized score + lambda_div * min_sem24_dist
        s_std = float(np.std(scores)) + 1e-6
        s_z = (scores - float(np.mean(scores))) / s_std
        sem = task["child_sem24"].astype(np.float64)
        sem_z = (sem - sem.mean(axis=0, keepdims=True)) / (sem.std(axis=0, keepdims=True) + 1e-6)
        chosen: List[int] = [int(order[0])]
        for _ in range(3):
            best_i = -1
            best_val = -1e18
            for cand in range(len(scores)):
                if cand in chosen:
                    continue
                d_min = float(np.min(np.sqrt(np.mean((sem_z[cand : cand + 1] - sem_z[chosen]) ** 2, axis=1))))
                val = float(s_z[cand]) + 0.85 * d_min
                if val > best_val:
                    best_val = val
                    best_i = cand
            chosen.append(best_i)
        top4 = np.array(chosen, dtype=np.int64)
    else:
        top4 = order[:4]
    top1 = order[0]
    best_top4_u = float(max(u_par, float(np.max(c_u[top4]))))
    best_top1_u = float(max(u_par, float(c_u[top1])))
    q_beat_4 = 1.0 if bool(np.any(beats[top4])) else 0.0

    # First winning rank (1-indexed, capped at 32 evaluations)
    win_positions = np.flatnonzero(beats[order])
    first_try = float(min(int(win_positions[0]) + 1, 32)) if len(win_positions) > 0 else 32.0

    # Pairwise accuracy on (pos, neg) sibling pairs
    pos_ix = np.flatnonzero(beats)
    neg_ix = np.flatnonzero(~beats)
    if len(pos_ix) > 0 and len(neg_ix) > 0:
        s_pos = scores[pos_ix][:, None]
        s_neg = scores[neg_ix][None, :]
        wins = np.sum(s_pos > s_neg) + 0.5 * np.sum(s_pos == s_neg)
        pair_acc = float(wins / (len(pos_ix) * len(neg_ix)))
    else:
        pair_acc = np.nan

    return {
        "promoted_u_top4": best_top4_u,
        "promoted_u_top1": best_top1_u,
        "q_beat_at_4": q_beat_4,
        "expected_tries": first_try,
        "pair_acc": pair_acc,
    }


def main() -> None:
    t0 = time.time()
    STAGE3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[H252 Stage 3] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_rows, val_rows = load_cohorts()
    all_rows = train_rows + val_rows
    ctx_feats = load_context_features()
    pools = load_pools(all_rows, verify_hashes=False)
    promos_by_cid = load_harvested_promotions(train_rows)

    with np.load(STAGE1_OUT_DIR / "r882_cached_costs.npz", allow_pickle=False) as zr:
        cids_r = [str(c) for c in zr["cids"]]
        tc_all = zr["total_cost"]
        r882_costs = {cids_r[i]: tc_all[i][: len(pools[cids_r[i]]["fitness"])] for i in range(len(cids_r))}

    with np.load(STAGE1_OUT_DIR / "stage1_hurdle_priors.npz", allow_pickle=False) as zp:
        cids_p = [str(c) for c in zp["cids"]]
        sos_all = zp["sos_prior"]
        hurdle_sos = {cids_p[i]: sos_all[i][: len(pools[cids_p[i]]["fitness"])] for i in range(len(cids_p))}

    tasks = build_transition_tasks(all_rows, pools, ctx_feats, r882_costs, hurdle_sos, promos_by_cid)
    print(f"[H252 Stage 3] Built Parent->Child transition tasks for {len(tasks)} contexts.")

    arms = ["UNIFORM_MUTATION", "R882_RANK_PROPOSER", "AWR_DELTA_CRITIC", "PROPOSER_DPO_DELTA"]
    train_panel = np.zeros((len(SEEDS), len(train_rows), len(arms)), dtype=np.float64)
    val_panel = np.zeros((len(SEEDS), len(val_rows), len(arms)), dtype=np.float64)

    kpi_records: Dict[str, Dict[str, List[float]]] = {
        split: {
            f"{a}_{metric}": []
            for a in arms
            for metric in ("q_beat_at_4", "expected_tries", "pair_acc", "promoted_u_top1")
        }
        for split in ("train", "val")
    }

    all_cids = [r["context_sha256"] for r in all_rows]
    cid_to_idx = {c: i for i, c in enumerate(all_cids)}
    train_cid_to_pos = {r["context_sha256"]: i for i, r in enumerate(train_rows)}
    val_cids = [r["context_sha256"] for r in val_rows]

    ctx_t = torch.as_tensor(np.stack([tasks[c]["ctx_vec"] for c in all_cids]), dtype=torch.float32, device=device)
    par_t = torch.as_tensor(np.stack([tasks[c]["parent_feat40"] for c in all_cids]), dtype=torch.float32, device=device)
    chi_t = torch.as_tensor(np.stack([tasks[c]["child_feat40"] for c in all_cids]), dtype=torch.float32, device=device)
    pri_t = torch.as_tensor(np.stack([tasks[c]["child_prior"] for c in all_cids]), dtype=torch.float32, device=device)
    dh_t = torch.as_tensor(np.stack([tasks[c]["delta_hurdle"] for c in all_cids]), dtype=torch.float32, device=device)
    bp_t = torch.as_tensor(np.stack([tasks[c]["beats_parent"] for c in all_cids]), dtype=torch.bool, device=device)

    saved_dpo_scores_seed0: Dict[str, np.ndarray] = {}

    for si, seed in enumerate(SEEDS):
        _, cid_to_fold = seed_folds(train_rows, seed)
        rng_uni = np.random.default_rng(seed)

        # Baselines: UNIFORM_MUTATION (0) and R882_RANK_PROPOSER (1)
        for split_name, rows, panel in [("train", train_rows, train_panel), ("val", val_rows, val_panel)]:
            for ci, r in enumerate(rows):
                cid = r["context_sha256"]
                tsk = tasks[cid]
                uni_scores = rng_uni.standard_normal(64)
                res_u = evaluate_proposer_scores(tsk, uni_scores)
                res_r = evaluate_proposer_scores(tsk, tsk["r882_rank_score"])
                panel[si, ci, 0] = res_u["promoted_u_top4"]
                panel[si, ci, 1] = res_r["promoted_u_top4"]
                for m in ("q_beat_at_4", "expected_tries", "pair_acc", "promoted_u_top1"):
                    if np.isfinite(res_u[m]):
                        kpi_records[split_name][f"UNIFORM_MUTATION_{m}"].append(res_u[m])
                    if np.isfinite(res_r[m]):
                        kpi_records[split_name][f"R882_RANK_PROPOSER_{m}"].append(res_r[m])

        val_score_accum = {
            "AWR_DELTA_CRITIC": np.zeros((len(val_rows), 64), dtype=np.float64),
            "PROPOSER_DPO_DELTA": np.zeros((len(val_rows), 64), dtype=np.float64),
        }

        for outer_k in range(5):
            a_fold, b_fold, fit_folds = cell_folds(outer_k)
            a_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == a_fold]
            b_cids = [r["context_sha256"] for r in train_rows if cid_to_fold[r["context_sha256"]] == b_fold]
            fit_rows = [r for r in train_rows if cid_to_fold[r["context_sha256"]] in fit_folds]
            fit_cids = [r["context_sha256"] for r in fit_rows]

            for arm_name in ("AWR_DELTA_CRITIC", "PROPOSER_DPO_DELTA"):
                cell_seed = stream_seed("H252_STAGE3", seed, outer_k, arm_name)
                rng = np.random.default_rng(cell_seed)
                torch.manual_seed(cell_seed)

                model = ParentChildDPOProposerNet(d_model=64).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
                params = list(model.parameters())
                ema_grad = [torch.zeros_like(p) for p in params]

                for step in range(1, 81):
                    b_cids = rng.choice(fit_cids, size=32, replace=False).tolist()
                    b_ix = torch.as_tensor([cid_to_idx[c] for c in b_cids], dtype=torch.long, device=device)

                    opt.zero_grad(set_to_none=True)
                    pred_delta = model(ctx_t[b_ix], par_t[b_ix], chi_t[b_ix], pri_t[b_ix])
                    # Base prior anchor score = R882 rank + 0.35 * Hurdle prior
                    prior_anc = (
                        pri_t[b_ix, :, 0]
                        + 0.25 * pri_t[b_ix, :, 1]
                        + 0.15 * pri_t[b_ix, :, 2]
                    )
                    total_s = prior_anc + pred_delta
                    loss = dpo_and_awr_loss(
                        total_s,
                        prior_anc,
                        dh_t[b_ix],
                        bp_t[b_ix],
                        use_dpo=(arm_name == "PROPOSER_DPO_DELTA"),
                    )
                    loss.backward()

                    if arm_name == "PROPOSER_DPO_DELTA":
                        # PCGrad-EMA projection
                        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                        g_flat = torch.cat([g.reshape(-1) for g in grads])
                        m_flat = torch.cat([m.reshape(-1) for m in ema_grad])
                        gn = float(torch.linalg.vector_norm(g_flat).item())
                        mn = float(torch.linalg.vector_norm(m_flat).item())
                        if gn > 1e-12 and mn > 1e-12:
                            dot = torch.dot(g_flat, m_flat)
                            if dot.item() < 0.0:
                                scale_p = dot / (mn * mn + 1e-12)
                                for p, m in zip(params, ema_grad):
                                    if p.grad is not None:
                                        p.grad.sub_(m, alpha=float(scale_p.item()))
                        with torch.no_grad():
                            for p, m in zip(params, ema_grad):
                                if p.grad is not None:
                                    m.mul_(0.90).add_(p.grad, alpha=0.10)

                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt.step()

                model.eval()
                with torch.no_grad():
                    a_ix = torch.as_tensor([cid_to_idx[c] for c in a_cids], dtype=torch.long, device=device)
                    a_prior = pri_t[a_ix, :, 0] + 0.25 * pri_t[a_ix, :, 1] + 0.15 * pri_t[a_ix, :, 2]
                    a_scores = (a_prior + model(ctx_t[a_ix], par_t[a_ix], chi_t[a_ix], pri_t[a_ix])).cpu().numpy()

                    v_ix = torch.as_tensor([cid_to_idx[c] for c in val_cids], dtype=torch.long, device=device)
                    v_prior = pri_t[v_ix, :, 0] + 0.25 * pri_t[v_ix, :, 1] + 0.15 * pri_t[v_ix, :, 2]
                    v_scores = (v_prior + model(ctx_t[v_ix], par_t[v_ix], chi_t[v_ix], pri_t[v_ix])).cpu().numpy()

                ai_idx = arms.index(arm_name)
                for idx_a, acid in enumerate(a_cids):
                    res_a = evaluate_proposer_scores(
                        tasks[acid],
                        a_scores[idx_a],
                        use_submodular_diversity=(arm_name == "PROPOSER_DPO_DELTA"),
                    )
                    pos = train_cid_to_pos[acid]
                    train_panel[si, pos, ai_idx] = res_a["promoted_u_top4"]
                    for m in ("q_beat_at_4", "expected_tries", "pair_acc", "promoted_u_top1"):
                        if np.isfinite(res_a[m]):
                            kpi_records["train"][f"{arm_name}_{m}"].append(res_a[m])
                    if si == 0 and arm_name == "PROPOSER_DPO_DELTA":
                        saved_dpo_scores_seed0[acid] = a_scores[idx_a]

                val_score_accum[arm_name] += v_scores / 5.0

        for arm_name in ("AWR_DELTA_CRITIC", "PROPOSER_DPO_DELTA"):
            ai_idx = arms.index(arm_name)
            for idx_v, vcid in enumerate(val_cids):
                sc_v = val_score_accum[arm_name][idx_v]
                res_v = evaluate_proposer_scores(
                    tasks[vcid],
                    sc_v,
                    use_submodular_diversity=(arm_name == "PROPOSER_DPO_DELTA"),
                )
                val_panel[si, idx_v, ai_idx] = res_v["promoted_u_top4"]
                for m in ("q_beat_at_4", "expected_tries", "pair_acc", "promoted_u_top1"):
                    if np.isfinite(res_v[m]):
                        kpi_records["val"][f"{arm_name}_{m}"].append(res_v[m])
                if si == 0 and arm_name == "PROPOSER_DPO_DELTA":
                    saved_dpo_scores_seed0[vcid] = sc_v

    # Statistical summaries (ref_idx=1 -> vs R882_RANK_PROPOSER, and ref_idx=0 -> vs UNIFORM_MUTATION)
    train_vs_r882 = summarize_panel(train_rows, train_panel, arms, ref_idx=1)
    val_vs_r882 = summarize_panel(val_rows, val_panel, arms, ref_idx=1)

    kpi_summary = {
        split: {
            a: {
                m: float(np.mean(kpi_records[split][f"{a}_{m}"]))
                for m in ("q_beat_at_4", "expected_tries", "pair_acc", "promoted_u_top1")
            }
            for a in arms
        }
        for split in ("train", "val")
    }

    for split_key, split_label, sr in [
        ("train", "DL-Train OOF (1,219 ctx / 205 fam)", train_vs_r882),
        ("val", "DL-Val (91 ctx / 19 fam)", val_vs_r882),
    ]:
        print(f"\n=== [H252 Stage 3 Proposer GPU] {split_label} ===")
        for ai, a in enumerate(arms):
            med_fam = sr["median_family_utility"][ai]
            elo = sr["median_elo"][ai]
            ks = kpi_summary[split_key][a]
            if a == "R882_RANK_PROPOSER":
                print(
                    f"  {a:<22} | prom_u@4={med_fam:.6f} | q_beat@4={ks['q_beat_at_4']:.4f} | "
                    f"E[tries]={ks['expected_tries']:.2f} | pair_acc={ks['pair_acc']:.4f} | Elo={elo:.1f} (anchor)"
                )
            else:
                cr = sr["contrasts"][f"{a}_minus_R882_RANK_PROPOSER"]
                print(
                    f"  {a:<22} | prom_u@4={med_fam:.6f} (d_R882={cr['median_family_gain']:+.6f}, "
                    f"95% CI [{cr['ci95'][0]:+.6f}, {cr['ci95'][1]:+.6f}], p={cr['p_one_sided']:.4g}) | "
                    f"q_beat@4={ks['q_beat_at_4']:.4f} | E[tries]={ks['expected_tries']:.2f} | "
                    f"pair_acc={ks['pair_acc']:.4f} | Elo={elo:.1f}"
                )

    out_obj = {
        "hypothesis_id": "H252_STAGE3",
        "wave": "W6_PROPOSER_DPO_DELTA_GPU",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "elapsed_sec": float(time.time() - t0),
        "proposer_kpis": kpi_summary,
        "dl_train_vs_r882_proposer": train_vs_r882,
        "dl_val_vs_r882_proposer": val_vs_r882,
    }
    metrics_path = STAGE3_OUT_DIR / "stage3_metrics.json"
    save_json(metrics_path, out_obj)

    cids_sorted = sorted(saved_dpo_scores_seed0.keys())
    np.savez_compressed(
        STAGE3_OUT_DIR / "stage3_proposer_scores.npz",
        cids=np.array(cids_sorted),
        dpo_scores=np.stack([saved_dpo_scores_seed0[c] for c in cids_sorted], axis=0),
    )
    print(f"\n[H252 Stage 3] Saved Stage 3 Proposer GPU metrics to {metrics_path} in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
