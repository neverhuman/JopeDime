#!/usr/bin/env python3
"""H473 Part B (`I3` / `I8`): Matryoshka `[32, 64]` Two-Tower Retrieval & `base_prior_cost` Study.

Implements `TwoTowerRetriever` trained in FP32 with Matryoshka `[32, 64]` InfoNCE + Huber
utility calibration, comparing `ret_pop`, `ret_lm48_knn`, `ret_desc_2t` (`290-D`),
`ret_dope_2t` (`663-D`), `ret_ultra_packet_2t` (`859-D`), and `ret_ultra_packet_shuffled` across:
  (a) Cross-context `TRAIN`-fold champion-bank `K = 96` slate construction (`Spec 2 §6.4`)
  (b) Within-pool `K = 16` / `K = 32` shortlist & `predict_costs(..., base_prior_cost=...)` (`Spec 2 §6.3`).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from h256_token_layout_audit import (
    derange_across_families,
    family_unweighted_mean,
    verify_mem_available_gib,
)

ALPHA_2T_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


class TwoTowerRetriever(nn.Module):
    """Dual-encoder mapping context `R(D)` and candidate `[prog_std_412, sem24]` to `[32, 64]` embeddings."""

    def __init__(self, d_ctx: int, d_prog: int = 436, d_emb: int = 64, d_hidden: int = 128) -> None:
        super().__init__()
        self.d_ctx = int(d_ctx)
        self.d_prog = int(d_prog)
        self.d_emb = int(d_emb)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(max(self.d_ctx, 1), d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(d_hidden, d_emb),
        )
        self.ctx_skip = nn.Linear(max(self.d_ctx, 1), d_emb, bias=False)
        self.prog_mlp = nn.Sequential(
            nn.Linear(d_prog, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(d_hidden, d_emb),
        )
        self.prog_skip = nn.Linear(d_prog, d_emb, bias=False)
        self.prog_bias_head = nn.Linear(d_emb, 1)
        self.raw_tau_32 = nn.Parameter(torch.tensor(-1.85, dtype=torch.float32))
        self.raw_tau_64 = nn.Parameter(torch.tensor(-1.85, dtype=torch.float32))

    def tau_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.raw_tau_32) + 0.05, F.softplus(self.raw_tau_64) + 0.05

    def encode_context(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if self.d_ctx == 0:
            zeros = torch.zeros((x_ctx.shape[0], 1), dtype=torch.float32, device=x_ctx.device)
            return self.ctx_mlp(zeros)
        return self.ctx_mlp(x_ctx) + 0.25 * self.ctx_skip(x_ctx)

    def encode_programs(self, x_prog: torch.Tensor, emp_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_emb = self.prog_mlp(x_prog) + 0.25 * self.prog_skip(x_prog)
        bias = 0.35 * self.prog_bias_head(v_emb).squeeze(-1) + emp_prior
        return v_emb, bias

    def score_slates(
        self, u_ctx: torch.Tensor, v_slates: torch.Tensor, b_slates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t32, t64 = self.tau_pair()
        u32 = F.normalize(u_ctx[:, :32], p=2, dim=-1, eps=1e-8)
        v32 = F.normalize(v_slates[:, :, :32], p=2, dim=-1, eps=1e-8)
        u64 = F.normalize(u_ctx[:, :64], p=2, dim=-1, eps=1e-8)
        v64 = F.normalize(v_slates[:, :, :64], p=2, dim=-1, eps=1e-8)
        s32 = torch.einsum("bd,bsd->bs", u32, v32) / t32 + b_slates
        s64 = torch.einsum("bd,bsd->bs", u64, v64) / t64 + b_slates
        return s32, s64

    def score_bank(
        self, u_ctx: torch.Tensor, v_bank: torch.Tensor, b_bank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t32, t64 = self.tau_pair()
        u32 = F.normalize(u_ctx[:, :32], p=2, dim=-1, eps=1e-8)
        v32 = F.normalize(v_bank[:, :32], p=2, dim=-1, eps=1e-8)
        u64 = F.normalize(u_ctx[:, :64], p=2, dim=-1, eps=1e-8)
        v64 = F.normalize(v_bank[:, :64], p=2, dim=-1, eps=1e-8)
        s32 = (u32 @ v32.T) / t32 + b_bank.unsqueeze(0)
        s64 = (u64 @ v64.T) / t64 + b_bank.unsqueeze(0)
        return s32, s64


def _std_fit_apply(x_fit: np.ndarray, x_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_fit.shape[1] == 0:
        return (
            np.zeros((1, 0), dtype=np.float32),
            np.ones((1, 0), dtype=np.float32),
            np.zeros((x_out.shape[0], 0), dtype=np.float32),
        )
    mu = np.mean(x_fit.astype(np.float64), axis=0, keepdims=True)
    sd = np.std(x_fit.astype(np.float64), axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z_out = np.clip((x_out.astype(np.float64) - mu) / sd, -8.0, 8.0)
    return mu.astype(np.float32), sd.astype(np.float32), np.nan_to_num(z_out, nan=0.0).astype(np.float32)


def _prog_feat_436(prog_412: np.ndarray, sem24: np.ndarray, p_mu: np.ndarray, p_sd: np.ndarray) -> np.ndarray:
    x_log = np.sign(prog_412) * np.log1p(np.abs(prog_412))
    p_z = np.clip((x_log - p_mu) / p_sd, -8.0, 8.0).astype(np.float32)
    return np.hstack([p_z, sem24.astype(np.float32)])


def _build_train_priors_and_banks(
    tr_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    champs_by_cid: Mapping[str, str],
) -> dict[str, Any]:
    pid_z_sum: dict[str, float] = {}
    pid_cnt: dict[str, int] = {}
    pid_wins: dict[str, int] = {}
    pid_rep: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prog_samples: list[np.ndarray] = []

    for i_c, cid in enumerate(tr_cids):
        p = pools[cid]
        fit = p["fitness"]
        gate = p["gate"]
        u_cap = np.where(gate, np.clip(-fit, -2.5, 5.0), -2.5)
        mu_u = float(np.mean(u_cap))
        sd_u = max(float(np.std(u_cap)), 0.10)
        z_u = (u_cap - mu_u) / sd_u
        q75 = float(np.percentile(u_cap[gate], 75)) if np.any(gate) else -2.0
        if i_c % 4 == 0:
            prog_samples.append(p["program"][::8])
        seen: set[str] = set()
        for j, pid in enumerate(p["pids"]):
            if pid not in seen:
                seen.add(pid)
                pid_cnt[pid] = pid_cnt.get(pid, 0) + 1
                pid_z_sum[pid] = pid_z_sum.get(pid, 0.0) + float(z_u[j])
                if gate[j] and u_cap[j] >= q75:
                    pid_wins[pid] = pid_wins.get(pid, 0) + 1
                if pid not in pid_rep:
                    pid_rep[pid] = (p["program"][j], p["struct_sig"][j])

    samp_cat = np.concatenate(prog_samples, axis=0).astype(np.float64)
    samp_log = np.sign(samp_cat) * np.log1p(np.abs(samp_cat))
    p_mu = np.mean(samp_log, axis=0, keepdims=True).astype(np.float32)
    p_sd = np.maximum(np.std(samp_log, axis=0, keepdims=True), 1e-4).astype(np.float32)

    emp_prior: dict[str, float] = {
        pid: float(pid_z_sum[pid] / (cnt + 6.0)) for pid, cnt in pid_cnt.items()
    }
    champ_win_cnt: dict[str, int] = {}
    for cid in tr_cids:
        cpid = champs_by_cid.get(cid)
        if cpid is not None and cpid in pid_rep:
            champ_win_cnt[cpid] = champ_win_cnt.get(cpid, 0) + 1
        else:
            p = pools[cid]
            u = np.where(p["gate"], -p["fitness"], -2.0)
            bpid = str(p["pids"][int(np.argmax(u))])
            champ_win_cnt[bpid] = champ_win_cnt.get(bpid, 0) + 1

    champ_pids = sorted(
        champ_win_cnt.keys(),
        key=lambda pid: (-champ_win_cnt[pid], -emp_prior.get(pid, 0.0), pid),
    )
    reusable_set = {pid for pid, c in pid_cnt.items() if c >= 3 and pid_wins.get(pid, 0) >= 1} | set(champ_pids)
    reusable_pids = sorted(
        reusable_set,
        key=lambda pid: (-pid_wins.get(pid, 0), -emp_prior.get(pid, 0.0), pid),
    )
    return {
        "p_mu": p_mu,
        "p_sd": p_sd,
        "emp_prior": emp_prior,
        "pid_rep": pid_rep,
        "champ_pids": champ_pids,
        "champ_pop_scores": np.array(
            [champ_win_cnt[p] + 0.5 * emp_prior.get(p, 0.0) for p in champ_pids], dtype=np.float64
        ),
        "reusable_pids": reusable_pids,
        "reusable_pop_scores": np.array(
            [0.25 * np.log1p(pid_wins.get(p, 0)) + emp_prior.get(p, 0.0) for p in reusable_pids], dtype=np.float64
        ),
    }


def fit_two_tower_retriever(
    x_ctx_tr: np.ndarray,
    tr_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    bank_info: Mapping[str, Any],
    *,
    seed: int,
    slate_size: int = 48,
    epochs: int = 18,
) -> tuple[TwoTowerRetriever, np.ndarray, np.ndarray]:
    c_mu, c_sd, cz_tr = _std_fit_apply(x_ctx_tr, x_ctx_tr)
    p_mu, p_sd = bank_info["p_mu"], bank_info["p_sd"]
    emp_prior = bank_info["emp_prior"]

    n_tr = len(tr_cids)
    max_pool_m = max(len(pools[cid]["pids"]) for cid in tr_cids)
    s_len = min(int(slate_size), max_pool_m)
    rng = np.random.default_rng(int(seed) + 47311)
    sl_feats = np.empty((n_tr, s_len, 436), dtype=np.float32)
    sl_emp = np.empty((n_tr, s_len), dtype=np.float32)
    sl_z = np.empty((n_tr, s_len), dtype=np.float32)

    for i, cid in enumerate(tr_cids):
        p = pools[cid]
        m = len(p["pids"])
        s_eff = min(s_len, m)
        fit = p["fitness"]
        gate = p["gate"]
        u_raw = np.where(gate, np.clip(-fit, -2.5, 5.0), -2.5)
        mu_u = float(np.mean(u_raw))
        sd_u = max(float(np.std(u_raw)), 0.10)
        z_u = np.where(gate, (u_raw - mu_u) / sd_u, -3.0).astype(np.float32)

        ord_u = np.argsort(-z_u)
        k_top = min(max(4, s_eff // 3), s_eff)
        top_part = ord_u[:k_top]
        rem = ord_u[k_top:]
        if len(rem) > 0 and s_eff > k_top:
            need = s_eff - k_top
            q_take = min(max(4, s_eff // 3), need, len(rem))
            q_idx = np.linspace(0, len(rem) - 1, num=q_take, dtype=np.int64)
            q_part = rem[q_idx]
            rem2 = np.setdiff1d(rem, q_part, assume_unique=True)
            r_need = need - len(q_part)
            r_part = (
                rng.choice(rem2, size=r_need, replace=len(rem2) < r_need)
                if r_need > 0 and len(rem2) > 0
                else np.zeros(0, dtype=np.int64)
            )
            sel = np.concatenate([top_part, q_part, r_part])
        else:
            sel = top_part
        if len(sel) < s_len:
            sel = np.pad(sel, (0, s_len - len(sel)), mode="wrap")
        sel = sel[:s_len]
        sl_feats[i] = _prog_feat_436(p["program"][sel], p["struct_sig"][sel], p_mu, p_sd)
        sl_emp[i] = np.array([emp_prior.get(p["pids"][int(j)], 0.0) for j in sel], dtype=np.float32)
        sl_z[i] = z_u[sel]

    torch.manual_seed(int(seed) + 47377)
    model = TwoTowerRetriever(d_ctx=x_ctx_tr.shape[1], d_prog=436, d_emb=64, d_hidden=96)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    cz_t = torch.from_numpy(cz_tr)
    pf_t = torch.from_numpy(sl_feats)
    pe_t = torch.from_numpy(sl_emp)
    zt_t = torch.from_numpy(sl_z)

    bsz = 256
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n_tr).astype(np.int64))
        for st in range(0, n_tr, bsz):
            idx = perm[st : st + bsz]
            b_len = int(idx.shape[0])
            opt.zero_grad(set_to_none=True)
            u_b = model.encode_context(cz_t[idx])
            v_flat, b_flat = model.encode_programs(pf_t[idx].reshape(-1, 436), pe_t[idx].reshape(-1))
            s32, s64 = model.score_slates(u_b, v_flat.view(b_len, s_len, 64), b_flat.view(b_len, s_len))
            zb = zt_t[idx]
            q_soft = F.softmax(zb / 0.30, dim=-1)
            loss = torch.tensor(0.0, dtype=torch.float32)
            for w_m, sm in ((0.4, s32), (0.6, s64)):
                log_p = F.log_softmax(sm, dim=-1)
                nce = -(q_soft * log_p).sum(dim=-1).mean() - 0.5 * log_p[:, 0].mean()
                huber = F.smooth_l1_loss(sm, zb, beta=0.5)
                loss = loss + w_m * (nce + 0.6 * huber)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model, c_mu, c_sd


def _lm48_knn_champ_scores(
    lm48_tr: np.ndarray,
    tr_cids: Sequence[str],
    lm48_ev: np.ndarray,
    champ_pids: Sequence[str],
    champs_by_cid: Mapping[str, str],
    emp_prior: Mapping[str, float],
) -> np.ndarray:
    _, _, z_tr = _std_fit_apply(lm48_tr, lm48_tr)
    _, _, z_ev = _std_fit_apply(lm48_tr, lm48_ev)
    u_tr = z_tr.astype(np.float64) / np.maximum(np.linalg.norm(z_tr.astype(np.float64), axis=1, keepdims=True), 1e-9)
    u_ev = z_ev.astype(np.float64) / np.maximum(np.linalg.norm(z_ev.astype(np.float64), axis=1, keepdims=True), 1e-9)
    sims = np.maximum(u_ev @ u_tr.T, 0.0) ** 2  # [N_ev, N_tr]
    pid_to_col = {pid: j for j, pid in enumerate(champ_pids)}
    tr_champ_mat = np.zeros((len(tr_cids), len(champ_pids)), dtype=np.float64)
    for i, cid in enumerate(tr_cids):
        cpid = champs_by_cid.get(cid)
        if cpid in pid_to_col:
            tr_champ_mat[i, pid_to_col[cpid]] = 1.0
    prior_tie = np.array([1e-4 * emp_prior.get(p, 0.0) for p in champ_pids], dtype=np.float64)
    return sims @ tr_champ_mat + prior_tie[None, :]


def _eval_champ_slate_metrics(
    champ_scores_mat: np.ndarray,
    champ_pids: Sequence[str],
    reusable_scores_mat: np.ndarray,
    reusable_pids: Sequence[str],
    eval_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    k_slate: int = 96,
) -> dict[str, np.ndarray]:
    n_ev = len(eval_cids)
    c_pid_to_j = {pid: j for j, pid in enumerate(champ_pids)}
    r_pid_to_j = {pid: j for j, pid in enumerate(reusable_pids)}
    k_c = min(k_slate, len(champ_pids))
    k_r = min(k_slate, len(reusable_pids))

    rec96_c = np.full(n_ev, np.nan, dtype=np.float64)
    hit96_c = np.zeros(n_ev, dtype=np.float64)
    top1_c_u = np.zeros(n_ev, dtype=np.float64)
    rec96_r = np.zeros(n_ev, dtype=np.float64)
    top1_r_u = np.zeros(n_ev, dtype=np.float64)

    for i, cid in enumerate(eval_cids):
        p = pools[cid]
        u = np.where(p["gate"], -p["fitness"], -2.0)
        sc_c = champ_scores_mat[i]
        top_c_set = {champ_pids[idx] for idx in np.argsort(-sc_c, kind="mergesort")[:k_c].tolist()}

        sc_r = reusable_scores_mat[i]
        top_r_set = {reusable_pids[idx] for idx in np.argsort(-sc_r, kind="mergesort")[:k_r].tolist()}
        in_r = [(j_p, r_pid_to_j[pid]) for j_p, pid in enumerate(p["pids"]) if pid in r_pid_to_j]
        if in_r:
            best_r_pool_j = max(in_r, key=lambda pr: (float(u[pr[0]]), -pr[1]))[0]
            rec96_r[i] = 1.0 if p["pids"][best_r_pool_j] in top_r_set else 0.0
            pick_r_j = max(in_r, key=lambda pr: (round(float(sc_r[pr[1]]), 9), -pr[1]))[0]
            top1_r_u[i] = float(u[pick_r_j])
        else:
            top1_r_u[i] = float(u[0])

        in_c = [(j_p, c_pid_to_j[pid]) for j_p, pid in enumerate(p["pids"]) if pid in c_pid_to_j]
        if in_c:
            best_c_pool_j = max(in_c, key=lambda pr: (float(u[pr[0]]), -pr[1]))[0]
            rec96_c[i] = 1.0 if p["pids"][best_c_pool_j] in top_c_set else 0.0
            hit96_c[i] = 1.0 if any(p["pids"][j_p] in top_c_set and p["gate"][j_p] for j_p, _ in in_c) else 0.0
            pick_c_j = max(in_c, key=lambda pr: (round(float(sc_c[pr[1]]), 9), -pr[1]))[0]
            top1_c_u[i] = float(u[pick_c_j])
        else:
            top1_c_u[i] = top1_r_u[i]

    return {
        "champ_recall96": rec96_c,
        "champ_hit96": hit96_c,
        "champ_top1_u": top1_c_u,
        "reusable_recall96": rec96_r,
        "reusable_top1_u": top1_r_u,
    }


def _score_within_pool_and_prior(
    pool_scores_list: list[np.ndarray],
    eval_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    zero_rev_cost_fn: Callable[[str, np.ndarray], np.ndarray],
    alpha_2t: float,
) -> dict[str, np.ndarray]:
    n_ev = len(eval_cids)
    ora16 = np.zeros(n_ev, dtype=np.float64)
    ora32 = np.zeros(n_ev, dtype=np.float64)
    rec8_16 = np.zeros(n_ev, dtype=np.float64)
    rec8_32 = np.zeros(n_ev, dtype=np.float64)
    dir_u = np.zeros(n_ev, dtype=np.float64)
    pri_u = np.zeros(n_ev, dtype=np.float64)
    pri_gf = np.zeros(n_ev, dtype=np.float64)

    for i, cid in enumerate(eval_cids):
        p = pools[cid]
        u = np.where(p["gate"], -p["fitness"], -2.0)
        sc = pool_scores_list[i]
        order = np.argsort(-sc, kind="mergesort")
        s16 = order[: min(16, len(order))]
        s32 = order[: min(32, len(order))]
        ora16[i] = float(np.max(u[s16]))
        ora32[i] = float(np.max(u[s32]))
        top8_true = set(np.argsort(-u, kind="mergesort")[: min(8, len(u))].tolist())
        rec8_16[i] = float(len(top8_true & set(s16.tolist()))) / max(len(top8_true), 1)
        rec8_32[i] = float(len(top8_true & set(s32.tolist()))) / max(len(top8_true), 1)
        dir_u[i] = float(u[int(order[0])])

        if alpha_2t == 0.0:
            base_c = np.zeros(len(sc), dtype=np.float32)
        else:
            c_std = -(sc - float(np.mean(sc))) / max(float(np.std(sc)), 1e-6)
            base_c = (float(alpha_2t) * c_std).astype(np.float32)
        tot_c = zero_rev_cost_fn(cid, base_c)
        pk = int(np.argmin(tot_c))
        pri_gf[i] = 0.0 if bool(p["gate"][pk]) else 1.0
        pri_u[i] = float(u[pk])

    return {
        "ora16": ora16,
        "ora32": ora32,
        "rec8_16": rec8_16,
        "rec8_32": rec8_32,
        "direct_u": dir_u,
        "prior0rev_u": pri_u,
        "prior0rev_gf": pri_gf,
    }


def _encode_bank_and_pools(
    model: TwoTowerRetriever,
    c_mu: np.ndarray,
    c_sd: np.ndarray,
    x_ctx_ev: np.ndarray,
    eval_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    bank_info: Mapping[str, Any],
) -> dict[str, Any]:
    p_mu, p_sd = bank_info["p_mu"], bank_info["p_sd"]
    emp_prior = bank_info["emp_prior"]
    pid_rep = bank_info["pid_rep"]
    z_ev = np.clip((x_ctx_ev.astype(np.float64) - c_mu) / c_sd, -8.0, 8.0).astype(np.float32)
    z_ev = np.nan_to_num(z_ev, nan=0.0)

    def _bank_tensors(pids_seq: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        pr = np.stack([pid_rep[p][0] for p in pids_seq], axis=0)
        sm = np.stack([pid_rep[p][1] for p in pids_seq], axis=0)
        ft = torch.from_numpy(_prog_feat_436(pr, sm, p_mu, p_sd))
        ep = torch.tensor([emp_prior.get(p, 0.0) for p in pids_seq], dtype=torch.float32)
        return model.encode_programs(ft, ep)

    with torch.no_grad():
        u_ev = model.encode_context(torch.from_numpy(z_ev))
        vc, bc = _bank_tensors(bank_info["champ_pids"])
        vr, br = _bank_tensors(bank_info["reusable_pids"])
        sc_c32, sc_c64 = model.score_bank(u_ev, vc, bc)
        sc_r32, sc_r64 = model.score_bank(u_ev, vr, br)
        t32, t64 = model.tau_pair()
        u32_np = F.normalize(u_ev[:, :32], p=2, dim=-1, eps=1e-8).numpy()
        u64_np = F.normalize(u_ev[:, :64], p=2, dim=-1, eps=1e-8).numpy()
        t32_f, t64_f = float(t32.item()), float(t64.item())

        pool_s64: list[np.ndarray] = []
        pool_s32: list[np.ndarray] = []
        lengths = [len(pools[cid]["pids"]) for cid in eval_cids]
        all_prog = np.concatenate([pools[cid]["program"] for cid in eval_cids], axis=0)
        all_sem = np.concatenate([pools[cid]["struct_sig"] for cid in eval_cids], axis=0)
        all_ft = _prog_feat_436(all_prog, all_sem, p_mu, p_sd)
        all_ep = np.array([emp_prior.get(pid, 0.0) for cid in eval_cids for pid in pools[cid]["pids"]], dtype=np.float32)
        n_tot = len(all_ft)
        v_all = np.empty((n_tot, 64), dtype=np.float32)
        b_all = np.empty(n_tot, dtype=np.float32)
        chunk_sz = 16384
        for st_c in range(0, n_tot, chunk_sz):
            en_c = min(st_c + chunk_sz, n_tot)
            vp_c, bp_c = model.encode_programs(
                torch.from_numpy(all_ft[st_c:en_c]), torch.from_numpy(all_ep[st_c:en_c])
            )
            v_all[st_c:en_c] = vp_c.numpy()
            b_all[st_c:en_c] = bp_c.numpy()

        pos = 0
        for i, m_len in enumerate(lengths):
            vp_np = v_all[pos : pos + m_len]
            bp_np = b_all[pos : pos + m_len]
            pos += m_len
            v32 = vp_np[:, :32] / np.maximum(np.linalg.norm(vp_np[:, :32], axis=-1, keepdims=True), 1e-8)
            v64 = vp_np / np.maximum(np.linalg.norm(vp_np, axis=-1, keepdims=True), 1e-8)
            pool_s32.append((v32 @ u32_np[i]) / t32_f + bp_np)
            pool_s64.append((v64 @ u64_np[i]) / t64_f + bp_np)

    return {
        "champ64": sc_c64.numpy(),
        "champ32": sc_c32.numpy(),
        "reusable64": sc_r64.numpy(),
        "reusable32": sc_r32.numpy(),
        "pool64": pool_s64,
        "pool32": pool_s32,
    }


def _select_inner_alpha_2t(
    x_ctx_tr: np.ndarray,
    tr_cids: Sequence[str],
    tr_fams: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    champs_by_cid: Mapping[str, str],
    zero_rev_cost_fn: Callable[[str, np.ndarray], np.ndarray],
    seed: int,
) -> float:
    uniq_f = sorted(set(str(f) for f in tr_fams))
    rng = np.random.default_rng(int(seed) + 47355)
    perm_f = [uniq_f[i] for i in rng.permutation(len(uniq_f)).tolist()]
    va_set = set(perm_f[: max(1, len(uniq_f) // 5)])
    va_idx = [i for i, f in enumerate(tr_fams) if str(f) in va_set]
    in_idx = [i for i, f in enumerate(tr_fams) if str(f) not in va_set]
    if not in_idx or not va_idx:
        return 0.0
    in_cids = [tr_cids[i] for i in in_idx]
    va_cids = [tr_cids[i] for i in va_idx]
    va_fams = [tr_fams[i] for i in va_idx]
    b_in = _build_train_priors_and_banks(in_cids, pools, champs_by_cid)
    mod_in, c_mu, c_sd = fit_two_tower_retriever(
        x_ctx_tr[in_idx], in_cids, pools, b_in, seed=seed, epochs=12
    )
    enc_va = _encode_bank_and_pools(mod_in, c_mu, c_sd, x_ctx_tr[va_idx], va_cids, pools, b_in)
    best_a = 0.0
    best_u = -1e9
    for a_cand in ALPHA_2T_GRID:
        res_a = _score_within_pool_and_prior(enc_va["pool64"], va_cids, pools, zero_rev_cost_fn, a_cand)
        fam_u = family_unweighted_mean(res_a["prior0rev_u"], va_fams)
        if fam_u > best_u + 1e-9:
            best_u = fam_u
            best_a = float(a_cand)
    return best_a


def run_part_b_two_tower_study(
    train_cids: Sequence[str],
    train_families: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    champs_by_cid: Mapping[str, str],
    repr_train: Mapping[str, np.ndarray],
    folds_list: Sequence[np.ndarray],
    zero_rev_cost_fn: Callable[[str, np.ndarray], np.ndarray],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    contrast_fn: Any,
    smoke: bool = False,
) -> dict[str, Any]:
    verify_mem_available_gib(15.0, "part_b_two_tower")
    n_tr = len(train_cids)
    eval_folds = folds_list[:1] if smoke else folds_list
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)
    n_folds = int(np.max(eval_folds[0])) + 1
    denom = 1.0 if smoke else float(len(eval_folds) * len(eval_seeds))

    slate_arms = (
        "ret_pop",
        "ret_lm48_knn",
        "ret_desc_2t",
        "ret_dope_2t",
        "ret_ultra_packet_2t",
        "ret_ultra_packet_2t_m32",
        "ret_ultra_packet_shuffled",
    )
    pool_arms = (
        "incumbent_0rev",
        "ret_pop",
        "ret_desc_2t",
        "ret_dope_2t",
        "ret_ultra_packet_2t",
        "ret_ultra_packet_2t_m32",
        "ret_ultra_packet_shuffled",
    )
    sl_keys = ("champ_recall96", "champ_hit96", "champ_top1_u", "reusable_recall96", "reusable_top1_u")
    pl_keys = ("ora16", "ora32", "rec8_16", "rec8_32", "direct_u", "prior0rev_u", "prior0rev_gf")
    sl_acc = {a: {k: np.zeros(n_tr, dtype=np.float64) for k in sl_keys} for a in slate_arms}
    sl_cov = np.zeros(n_tr, dtype=np.float64)
    pl_acc = {a: {k: np.zeros(n_tr, dtype=np.float64) for k in pl_keys} for a in pool_arms}
    sel_alphas: dict[str, list[float]] = {"ret_desc_2t": [], "ret_dope_2t": [], "ret_ultra_packet_2t": []}

    for fold_arr in eval_folds:
        for k in range(1 if smoke else n_folds):
            ho_idx = np.where(fold_arr == k)[0]
            tr_idx = np.where(fold_arr != k)[0]
            tr_cids_k = [train_cids[i] for i in tr_idx.tolist()]
            ho_cids_k = [train_cids[i] for i in ho_idx.tolist()]
            tr_fams_k = [train_families[i] for i in tr_idx.tolist()]
            ho_fams_k = [train_families[i] for i in ho_idx.tolist()]
            b_info = _build_train_priors_and_banks(tr_cids_k, pools, champs_by_cid)

            pop_c_mat = np.repeat(b_info["champ_pop_scores"][None, :], len(ho_cids_k), axis=0)
            pop_r_mat = np.repeat(b_info["reusable_pop_scores"][None, :], len(ho_cids_k), axis=0)
            knn_c_mat = _lm48_knn_champ_scores(
                repr_train["lm48_z"][tr_idx], tr_cids_k, repr_train["lm48_z"][ho_idx],
                b_info["champ_pids"], champs_by_cid, b_info["emp_prior"],
            )
            pop_pool_sc = [
                np.array([b_info["emp_prior"].get(pid, 0.0) for pid in pools[c]["pids"]], dtype=np.float32)
                for c in ho_cids_k
            ]
            inc_pool_sc = [
                -zero_rev_cost_fn(c, np.zeros(len(pools[c]["pids"]), dtype=np.float32)) for c in ho_cids_k
            ]

            for sd in eval_seeds:
                donor_ho = derange_across_families(ho_fams_k, seed=20261200 + 100 * k + int(sd))
                sl_res_map: dict[str, dict[str, np.ndarray]] = {
                    "ret_pop": _eval_champ_slate_metrics(
                        pop_c_mat, b_info["champ_pids"], pop_r_mat, b_info["reusable_pids"], ho_cids_k, pools
                    ),
                    "ret_lm48_knn": _eval_champ_slate_metrics(
                        knn_c_mat, b_info["champ_pids"], pop_r_mat, b_info["reusable_pids"], ho_cids_k, pools
                    ),
                }
                pl_res_map: dict[str, dict[str, np.ndarray]] = {
                    "incumbent_0rev": _score_within_pool_and_prior(inc_pool_sc, ho_cids_k, pools, zero_rev_cost_fn, 0.0),
                    "ret_pop": _score_within_pool_and_prior(pop_pool_sc, ho_cids_k, pools, zero_rev_cost_fn, 0.0),
                }

                for arm_2t, r_key in (
                    ("ret_desc_2t", "R_desc"),
                    ("ret_dope_2t", "R_dope"),
                    ("ret_ultra_packet_2t", "DopeUltraPacket859"),
                ):
                    x_mat = repr_train[r_key]
                    a_sel = _select_inner_alpha_2t(
                        x_mat[tr_idx], tr_cids_k, tr_fams_k, pools, champs_by_cid, zero_rev_cost_fn, int(sd)
                    )
                    sel_alphas[arm_2t].append(a_sel)
                    mod, c_mu, c_sd = fit_two_tower_retriever(
                        x_mat[tr_idx], tr_cids_k, pools, b_info, seed=int(sd), epochs=10 if smoke else 18
                    )
                    enc = _encode_bank_and_pools(mod, c_mu, c_sd, x_mat[ho_idx], ho_cids_k, pools, b_info)
                    sl_res_map[arm_2t] = _eval_champ_slate_metrics(
                        enc["champ64"], b_info["champ_pids"], enc["reusable64"], b_info["reusable_pids"], ho_cids_k, pools
                    )
                    pl_res_map[arm_2t] = _score_within_pool_and_prior(
                        enc["pool64"], ho_cids_k, pools, zero_rev_cost_fn, a_sel
                    )
                    if arm_2t == "ret_ultra_packet_2t":
                        sl_res_map["ret_ultra_packet_2t_m32"] = _eval_champ_slate_metrics(
                            enc["champ32"], b_info["champ_pids"], enc["reusable32"], b_info["reusable_pids"], ho_cids_k, pools
                        )
                        pl_res_map["ret_ultra_packet_2t_m32"] = _score_within_pool_and_prior(
                            enc["pool32"], ho_cids_k, pools, zero_rev_cost_fn, a_sel
                        )
                        enc_sh = _encode_bank_and_pools(
                            mod, c_mu, c_sd, x_mat[ho_idx][donor_ho], ho_cids_k, pools, b_info
                        )
                        sl_res_map["ret_ultra_packet_shuffled"] = _eval_champ_slate_metrics(
                            enc_sh["champ64"], b_info["champ_pids"], enc_sh["reusable64"], b_info["reusable_pids"], ho_cids_k, pools
                        )
                        pl_res_map["ret_ultra_packet_shuffled"] = _score_within_pool_and_prior(
                            enc_sh["pool64"], ho_cids_k, pools, zero_rev_cost_fn, a_sel
                        )

                valid_c = ~np.isnan(sl_res_map["ret_pop"]["champ_recall96"])
                sl_cov[ho_idx[valid_c]] += 1.0 / denom
                for sa, sr in sl_res_map.items():
                    for k_m in sl_keys:
                        vals = np.nan_to_num(sr[k_m], nan=0.0)
                        sl_acc[sa][k_m][ho_idx] += vals / denom
                for pa, pr in pl_res_map.items():
                    for k_m in pl_keys:
                        pl_acc[pa][k_m][ho_idx] += pr[k_m] / denom

    mask_ev = (eval_folds[0] == 0) if smoke else np.ones(n_tr, dtype=bool)
    fams_ev = [train_families[i] for i in np.where(mask_ev)[0].tolist()]
    cov_ev = sl_cov[mask_ev] > 0.5

    champ_slate_out = {
        sa: {
            "champ_recall_at_96": round(float(np.mean(sl_acc[sa]["champ_recall96"][mask_ev][cov_ev] / sl_cov[mask_ev][cov_ev])), 6),
            "champ_hit_rate_at_96": round(float(np.mean(sl_acc[sa]["champ_hit96"][mask_ev])), 6),
            "champ_top1_family_utility": round(family_unweighted_mean(sl_acc[sa]["champ_top1_u"][mask_ev], fams_ev), 6),
            "reusable_recall_at_96": round(float(np.mean(sl_acc[sa]["reusable_recall96"][mask_ev])), 6),
            "reusable_top1_family_utility": round(family_unweighted_mean(sl_acc[sa]["reusable_top1_u"][mask_ev], fams_ev), 6),
        }
        for sa in slate_arms
    }
    within_pool_out = {
        pa: {
            "shortlist16_oracle_family_utility": round(family_unweighted_mean(pl_acc[pa]["ora16"][mask_ev], fams_ev), 6),
            "shortlist32_oracle_family_utility": round(family_unweighted_mean(pl_acc[pa]["ora32"][mask_ev], fams_ev), 6),
            "top8_pool_recall_at_16": round(float(np.mean(pl_acc[pa]["rec8_16"][mask_ev])), 6),
            "top8_pool_recall_at_32": round(float(np.mean(pl_acc[pa]["rec8_32"][mask_ev])), 6),
            "direct_2t_top1_family_utility": round(family_unweighted_mean(pl_acc[pa]["direct_u"][mask_ev], fams_ev), 6),
            "prior_0rev_family_utility": round(family_unweighted_mean(pl_acc[pa]["prior0rev_u"][mask_ev], fams_ev), 6),
            "prior_0rev_gatefail_rate": round(float(np.mean(pl_acc[pa]["prior0rev_gf"][mask_ev])), 6),
        }
        for pa in pool_arms
    }
    alpha_summary = {k: round(float(np.mean(vs)), 4) if vs else 0.0 for k, vs in sel_alphas.items()}

    contrasts = {
        "champ_top1_ultra_minus_lm48_knn": contrast_fn(
            sl_acc["ret_ultra_packet_2t"]["champ_top1_u"][mask_ev], sl_acc["ret_lm48_knn"]["champ_top1_u"][mask_ev], fams_ev
        ),
        "champ_top1_ultra_minus_dope2t": contrast_fn(
            sl_acc["ret_ultra_packet_2t"]["champ_top1_u"][mask_ev], sl_acc["ret_dope_2t"]["champ_top1_u"][mask_ev], fams_ev
        ),
        "champ_top1_ultra_minus_shuffled": contrast_fn(
            sl_acc["ret_ultra_packet_2t"]["champ_top1_u"][mask_ev], sl_acc["ret_ultra_packet_shuffled"]["champ_top1_u"][mask_ev], fams_ev
        ),
        "shortlist32_oracle_ultra_minus_dope2t": contrast_fn(
            pl_acc["ret_ultra_packet_2t"]["ora32"][mask_ev], pl_acc["ret_dope_2t"]["ora32"][mask_ev], fams_ev
        ),
        "shortlist32_oracle_ultra_minus_pop": contrast_fn(
            pl_acc["ret_ultra_packet_2t"]["ora32"][mask_ev], pl_acc["ret_pop"]["ora32"][mask_ev], fams_ev
        ),
        "shortlist32_oracle_ultra_minus_shuffled": contrast_fn(
            pl_acc["ret_ultra_packet_2t"]["ora32"][mask_ev], pl_acc["ret_ultra_packet_shuffled"]["ora32"][mask_ev], fams_ev
        ),
        "prior0rev_ultra_minus_incumbent0rev": contrast_fn(
            pl_acc["ret_ultra_packet_2t"]["prior0rev_u"][mask_ev], pl_acc["incumbent_0rev"]["prior0rev_u"][mask_ev], fams_ev
        ),
    }

    return {
        "champ_pool_overlap_context_fraction": round(float(np.mean(cov_ev)), 6),
        "selected_alpha_2t_mean": alpha_summary,
        "cross_context_champion_slate_k96": champ_slate_out,
        "within_pool_shortlist_and_prior": within_pool_out,
        "contrasts": contrasts,
    }


def eval_part_b_val91_regcheck(
    train_cids: Sequence[str],
    val_cids: Sequence[str],
    val_families: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    champs_by_cid: Mapping[str, str],
    repr_train: Mapping[str, np.ndarray],
    repr_val: Mapping[str, np.ndarray],
    zero_rev_cost_fn: Callable[[str, np.ndarray], np.ndarray],
    frozen_alphas: Mapping[str, float],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    contrast_fn: Any,
) -> dict[str, Any]:
    b_info = _build_train_priors_and_banks(train_cids, pools, champs_by_cid)
    n_v = len(val_cids)
    n_sd = float(len(seeds))
    pop_c = np.repeat(b_info["champ_pop_scores"][None, :], n_v, axis=0)
    pop_r = np.repeat(b_info["reusable_pop_scores"][None, :], n_v, axis=0)
    knn_c = _lm48_knn_champ_scores(
        repr_train["lm48_z"], train_cids, repr_val["lm48_z"],
        b_info["champ_pids"], champs_by_cid, b_info["emp_prior"],
    )
    inc_sc = [-zero_rev_cost_fn(c, np.zeros(len(pools[c]["pids"]), dtype=np.float32)) for c in val_cids]
    pop_sc = [
        np.array([b_info["emp_prior"].get(pid, 0.0) for pid in pools[c]["pids"]], dtype=np.float32)
        for c in val_cids
    ]

    sl_res: dict[str, dict[str, np.ndarray]] = {
        "ret_pop": _eval_champ_slate_metrics(pop_c, b_info["champ_pids"], pop_r, b_info["reusable_pids"], val_cids, pools),
        "ret_lm48_knn": _eval_champ_slate_metrics(knn_c, b_info["champ_pids"], pop_r, b_info["reusable_pids"], val_cids, pools),
    }
    pl_res: dict[str, dict[str, np.ndarray]] = {
        "incumbent_0rev": _score_within_pool_and_prior(inc_sc, val_cids, pools, zero_rev_cost_fn, 0.0),
        "ret_pop": _score_within_pool_and_prior(pop_sc, val_cids, pools, zero_rev_cost_fn, 0.0),
    }
    for arm_2t, r_key in (("ret_dope_2t", "R_dope"), ("ret_ultra_packet_2t", "DopeUltraPacket859")):
        a_use = float(frozen_alphas.get(arm_2t, 0.0))
        sl_res[arm_2t] = {k: np.zeros(n_v, dtype=np.float64) for k in ("champ_recall96", "champ_hit96", "champ_top1_u", "reusable_recall96", "reusable_top1_u")}
        pl_res[arm_2t] = {k: np.zeros(n_v, dtype=np.float64) for k in ("ora16", "ora32", "rec8_16", "rec8_32", "direct_u", "prior0rev_u", "prior0rev_gf")}
        if arm_2t == "ret_ultra_packet_2t":
            sl_res["ret_ultra_packet_shuffled"] = {k: np.zeros(n_v, dtype=np.float64) for k in sl_res[arm_2t]}
            pl_res["ret_ultra_packet_shuffled"] = {k: np.zeros(n_v, dtype=np.float64) for k in pl_res[arm_2t]}
        for sd in seeds:
            mod, c_mu, c_sd = fit_two_tower_retriever(repr_train[r_key], train_cids, pools, b_info, seed=int(sd), epochs=18)
            enc = _encode_bank_and_pools(mod, c_mu, c_sd, repr_val[r_key], val_cids, pools, b_info)
            sr = _eval_champ_slate_metrics(enc["champ64"], b_info["champ_pids"], enc["reusable64"], b_info["reusable_pids"], val_cids, pools)
            pr = _score_within_pool_and_prior(enc["pool64"], val_cids, pools, zero_rev_cost_fn, a_use)
            for k_m in sl_res[arm_2t]:
                sl_res[arm_2t][k_m] += np.nan_to_num(sr[k_m], nan=0.0) / n_sd
            for k_m in pl_res[arm_2t]:
                pl_res[arm_2t][k_m] += pr[k_m] / n_sd
            if arm_2t == "ret_ultra_packet_2t":
                donor_v = derange_across_families(val_families, seed=20261299 + int(sd))
                enc_sh = _encode_bank_and_pools(mod, c_mu, c_sd, repr_val[r_key][donor_v], val_cids, pools, b_info)
                sr_sh = _eval_champ_slate_metrics(enc_sh["champ64"], b_info["champ_pids"], enc_sh["reusable64"], b_info["reusable_pids"], val_cids, pools)
                pr_sh = _score_within_pool_and_prior(enc_sh["pool64"], val_cids, pools, zero_rev_cost_fn, a_use)
                for k_m in sl_res["ret_ultra_packet_shuffled"]:
                    sl_res["ret_ultra_packet_shuffled"][k_m] += np.nan_to_num(sr_sh[k_m], nan=0.0) / n_sd
                for k_m in pl_res["ret_ultra_packet_shuffled"]:
                    pl_res["ret_ultra_packet_shuffled"][k_m] += pr_sh[k_m] / n_sd

    cov_v = ~np.isnan(sl_res["ret_pop"]["champ_recall96"])
    return {
        "champ_slate_k96": {
            sa: {
                "champ_recall_at_96": round(float(np.mean(np.nan_to_num(sr["champ_recall96"][cov_v], nan=0.0))), 6) if np.any(cov_v) else 0.0,
                "champ_top1_family_utility": round(family_unweighted_mean(sr["champ_top1_u"], val_families), 6),
                "reusable_top1_family_utility": round(family_unweighted_mean(sr["reusable_top1_u"], val_families), 6),
            }
            for sa, sr in sl_res.items()
        },
        "within_pool": {
            pa: {
                "shortlist16_oracle_family_utility": round(family_unweighted_mean(pr["ora16"], val_families), 6),
                "shortlist32_oracle_family_utility": round(family_unweighted_mean(pr["ora32"], val_families), 6),
                "top8_pool_recall_at_16": round(float(np.mean(pr["rec8_16"])), 6),
                "top8_pool_recall_at_32": round(float(np.mean(pr["rec8_32"])), 6),
                "prior_0rev_family_utility": round(family_unweighted_mean(pr["prior0rev_u"], val_families), 6),
                "prior_0rev_gatefail_rate": round(float(np.mean(pr["prior0rev_gf"])), 6),
            }
            for pa, pr in pl_res.items()
        },
        "contrasts": {
            "champ_top1_ultra_minus_lm48_knn": contrast_fn(sl_res["ret_ultra_packet_2t"]["champ_top1_u"], sl_res["ret_lm48_knn"]["champ_top1_u"], val_families),
            "champ_top1_ultra_minus_dope2t": contrast_fn(sl_res["ret_ultra_packet_2t"]["champ_top1_u"], sl_res["ret_dope_2t"]["champ_top1_u"], val_families),
            "shortlist32_oracle_ultra_minus_dope2t": contrast_fn(pl_res["ret_ultra_packet_2t"]["ora32"], pl_res["ret_dope_2t"]["ora32"], val_families),
            "prior0rev_ultra_minus_incumbent0rev": contrast_fn(pl_res["ret_ultra_packet_2t"]["prior0rev_u"], pl_res["incumbent_0rev"]["prior0rev_u"], val_families),
        },
    }

