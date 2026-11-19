"""Shared Frozen-Anchor Precomputation Bank (`FlywheelGameNet` encoded ONCE per context).

Eliminates redundant `FlywheelGameNet.encode_candidates` (`[1, M, 560]` SwiGLU + MLP) calls
across Part A (`10` Stage-1A modes + `4` reveal modes), Part C.1 (`WP3`), Part C.2 (`B03`),
and `val91` `REGCHECK`, cutting CPU wall time by >100x with `< 1e-6` verified parity.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def _anchor_tail_from_encoded(
    anc: Any,
    ctx_emb: torch.Tensor,
    prog_emb: torch.Tensor,
    struct_sig: torch.Tensor,
    eff_prior: torch.Tensor,
    proj16_d: torch.Tensor,
    rev_idx: torch.Tensor,
    rev_f: torch.Tensor,
    rev_g: torch.Tensor,
) -> torch.Tensor:
    """Exact tail of `FlywheelGameNet.predict_costs` (`model.py` lines 251–384) given encoded candidates."""
    bsz, n_cands, _ = prog_emb.shape
    t_steps = int(rev_idx.shape[1])
    ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, anc.d_model)
    if t_steps == 0:
        gate_in = torch.cat(
            [ctx_exp, prog_emb, eff_prior.float().unsqueeze(-1), torch.zeros_like(eff_prior.float().unsqueeze(-1))],
            dim=-1,
        )
        fail_logit = anc.gate_head(gate_in).squeeze(-1)
        gate_pen = F.softplus(anc.gate_barrier_weight) * torch.sigmoid(fail_logit)
        return (eff_prior + gate_pen.double()).float()

    scale_d = 0.4208
    valid_mask = rev_idx >= 0
    safe_idx = rev_idx.clamp(min=0)
    idx_16 = safe_idx.unsqueeze(-1).expand(bsz, t_steps, 16)
    rev_proj_16 = torch.gather(proj16_d, dim=1, index=idx_16)
    rev_eff_prior = torch.gather(eff_prior, dim=1, index=safe_idx)

    rev_res_w1 = rev_f.double() / scale_d  # base_prior_cost == 0
    clamped_f = torch.where(rev_g > 0.5, rev_f.double(), torch.clamp(rev_f.double(), max=2.5))
    rev_res_smooth = clamped_f / scale_d - rev_eff_prior

    diff_16 = proj16_d.unsqueeze(2) - rev_proj_16.unsqueeze(1)
    dist2_16 = torch.sum(diff_16 * diff_16, dim=-1)
    dist2_16 = torch.where(valid_mask.unsqueeze(1), dist2_16, torch.full_like(dist2_16, 1e12))

    k_nn = min(5, t_steps)
    top_d2, top_order = torch.topk(dist2_16, k=k_nn, dim=-1, largest=False, sorted=True)
    w_w1 = torch.where(top_d2 < 1e11, 1.0 / (1.0 + top_d2), torch.zeros_like(top_d2))
    top_res_w1 = torch.gather(rev_res_w1.unsqueeze(1).expand(bsz, n_cands, t_steps), dim=-1, index=top_order)
    w1_residual_adj = torch.where(
        valid_mask.any(dim=-1, keepdim=True),
        torch.sum(w_w1 * top_res_w1, dim=-1) / torch.sum(w_w1, dim=-1).clamp(min=1e-12),
        torch.zeros((bsz, n_cands), dtype=torch.float64, device=prog_emb.device),
    )

    idx_24 = safe_idx.unsqueeze(-1).expand(bsz, t_steps, 24)
    rev_sig_24 = torch.gather(struct_sig, dim=1, index=idx_24)
    cos_sim = torch.einsum("bnd,btd->bnt", struct_sig, rev_sig_24)
    dist2_struct = (1.0 - cos_sim).clamp(min=0.0) * 4.0
    joint_d2 = 0.5 * dist2_16.float() + 2.0 * dist2_struct
    joint_d2 = torch.where(valid_mask.unsqueeze(1), joint_d2, torch.full_like(joint_d2, 1e9))
    min_struct_dist = torch.min(joint_d2, dim=-1).values.clamp(max=10.0) / 10.0

    k_s = min(8, t_steps)
    s_d2, s_ord = torch.topk(joint_d2.double(), k=k_s, dim=-1, largest=False, sorted=True)
    w_s = torch.where(s_d2 < 1e8, 0.7 / (0.25 + s_d2) + 0.3 / (2.0 + s_d2), torch.zeros_like(s_d2))
    s_den = torch.sum(w_s, dim=-1).clamp(min=1e-12)
    top_res_sm = torch.gather(rev_res_smooth.unsqueeze(1).expand(bsz, n_cands, t_steps), dim=-1, index=s_ord)
    struct_adj = torch.where(
        valid_mask.any(dim=-1, keepdim=True), torch.sum(w_s * top_res_sm, dim=-1) / s_den, torch.zeros_like(s_den)
    )
    top_fail = torch.gather(
        (1.0 - rev_g).double().unsqueeze(1).expand(bsz, n_cands, t_steps), dim=-1, index=s_ord
    )
    rev_gate_local = (torch.sum(w_s * top_fail, dim=-1) / s_den).float()

    idx_emb = safe_idx.unsqueeze(-1).expand(bsz, t_steps, anc.d_model)
    rev_prog_emb = torch.gather(prog_emb, dim=1, index=idx_emb)
    step_frac = (
        torch.linspace(0.1, 1.0, steps=t_steps, device=prog_emb.device).unsqueeze(0).expand(bsz, t_steps)
        if t_steps > 1
        else torch.full((bsz, 1), 0.1, device=prog_emb.device, dtype=torch.float32)
    )
    tok_feats = torch.stack(
        [
            torch.clamp(rev_res_smooth.float(), -5.0, 5.0),
            rev_g.float(),
            torch.clamp(rev_eff_prior.float(), -5.0, 5.0),
            step_frac,
        ],
        dim=-1,
    )
    mem_tokens = anc.history_token_in(torch.cat([rev_prog_emb, tok_feats], dim=-1))
    attn_out, _ = anc.cross_attn(
        query=prog_emb, key=mem_tokens, value=mem_tokens, key_padding_mask=~valid_mask, need_weights=False
    )
    summary_feats = torch.stack(
        [
            w1_residual_adj.float().clamp(-5.0, 5.0),
            struct_adj.float().clamp(-5.0, 5.0),
            min_struct_dist,
            rev_gate_local,
        ],
        dim=-1,
    )
    attn_delta = torch.tanh(anc.traj_out(torch.cat([ctx_exp, attn_out, summary_feats], dim=-1)).squeeze(-1))

    gate_in = torch.cat([ctx_exp, prog_emb, eff_prior.float().unsqueeze(-1), rev_gate_local.unsqueeze(-1)], dim=-1)
    fail_logit = anc.gate_head(gate_in).squeeze(-1) + 2.0 * rev_gate_local
    gate_pen = F.softplus(anc.gate_barrier_weight) * torch.sigmoid(fail_logit)

    w_struct = torch.sigmoid(anc.alpha_struct_kernel).double() * 0.35
    blended_res = (1.0 - w_struct) * w1_residual_adj + w_struct * struct_adj
    tot = eff_prior + blended_res + anc.alpha_attn.double() * attn_delta.double() + gate_pen.double()
    return tot.float()


def _rehydrate_prog_std_inplace(
    ad: Any,
    bank: dict[str, dict[str, Any]],
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
) -> None:
    p_mean = ad.prog_mean.detach().cpu().numpy().reshape(1, 412).astype(np.float32)
    p_std = ad.prog_std.detach().cpu().numpy().reshape(1, 412).astype(np.float32)
    for cid in ordered_cids:
        b = bank[cid]
        if "prog_std" not in b:
            prog = pools[cid]["program"]
            x_log = np.sign(prog) * np.log1p(np.abs(prog))
            b["prog_std"] = np.clip((x_log - p_mean) / p_std, -8.0, 8.0).astype(np.float32)


def build_shared_frozen_anchor_bank(
    net: Any,
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
    *,
    cache_path: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Deduplicate unique candidate programs across `ordered_cids`, run `FlywheelGameNet.encode_candidates`
    in batched chunks, and precompute exact per-context reveal modes (with optional disk cache).
    """
    from pathlib import Path
    from phase12_multimodal_model import compute_semantic_struct_sig_24d_torch, transform_program_features_torch

    net.eval()
    ad = net.adapter_10tok
    anc = ad.anchor

    if cache_path is not None and Path(cache_path).is_file():
        import hashlib

        cp = Path(cache_path)
        sha_sidecar = cp.with_suffix(".pt.sha256")
        if sha_sidecar.is_file():
            expected_sha = sha_sidecar.read_text(encoding="utf-8").strip()
            actual_sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"anchor cache SHA-256 mismatch for {cp.name}")
        raw_loaded = torch.load(cp, map_location="cpu", weights_only=True)
        if isinstance(raw_loaded, dict) and all(c in raw_loaded for c in ordered_cids):
            loaded: dict[str, dict[str, Any]] = {
                cid: {
                    k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                    for k, v in raw_loaded[cid].items()
                }
                for cid in ordered_cids
            }
            _rehydrate_prog_std_inplace(ad, loaded, ordered_cids, pools)
            return loaded

    z_ctx = torch.zeros((1, 584), dtype=torch.float32)

    # 1. Deduplicate unique programs across all pools in `ordered_cids`
    pid_to_uid: dict[str, int] = {}
    uniq_prog_list: list[np.ndarray] = []
    uniq_proj_list: list[np.ndarray] = []
    ctx_uids: dict[str, np.ndarray] = {}

    for cid in ordered_cids:
        p = pools[cid]
        pids = p["pids"]
        prog = p["program"]
        proj = p["proj16"]
        u_arr = np.empty(len(pids), dtype=np.int64)
        for j, pid in enumerate(pids):
            uid = pid_to_uid.get(pid)
            if uid is None:
                uid = len(uniq_prog_list)
                pid_to_uid[pid] = uid
                uniq_prog_list.append(prog[j])
                uniq_proj_list.append(proj[j])
            u_arr[j] = uid
        ctx_uids[cid] = u_arr

    n_uniq = len(uniq_prog_list)
    u_prog_np = np.stack(uniq_prog_list, axis=0).astype(np.float32)
    u_proj_np = np.stack(uniq_proj_list, axis=0).astype(np.float32)
    del uniq_prog_list, uniq_proj_list

    with torch.no_grad():
        ctx_emb_zero = anc.ctx_block(anc.ctx_in(z_ctx))  # [1, 128]
        u_sem24 = np.empty((n_uniq, 24), dtype=np.float32)
        u_prog_std = np.empty((n_uniq, 412), dtype=np.float32)
        u_prog_emb = torch.empty((n_uniq, anc.d_model), dtype=torch.float32)
        u_struct_sig = torch.empty((n_uniq, 24), dtype=torch.float32)
        u_eff_prior = np.empty(n_uniq, dtype=np.float64)
        u_tot_0rev = np.empty(n_uniq, dtype=np.float32)

        chunk = 4096
        for st in range(0, n_uniq, chunk):
            en = min(st + chunk, n_uniq)
            m_ch = en - st
            pt = torch.from_numpy(u_prog_np[st:en][None])
            pjt = torch.from_numpy(u_proj_np[st:en][None])
            zs = torch.zeros((1, m_ch, 2), dtype=torch.float32)

            u_sem24[st:en] = compute_semantic_struct_sig_24d_torch(pt)[0].numpy()
            u_prog_std[st:en] = transform_program_features_torch(pt, ad.prog_mean, ad.prog_std)[0].numpy()
            _, pemb, ssig, pdelta = anc.encode_candidates(z_ctx, pt, pjt, zs)
            eff_p = anc.alpha_prior.double() * pdelta.double()
            tot0 = _anchor_tail_from_encoded(
                anc,
                ctx_emb_zero,
                pemb,
                ssig,
                eff_p,
                pjt.double(),
                torch.empty((1, 0), dtype=torch.long),
                torch.empty((1, 0), dtype=torch.float32),
                torch.empty((1, 0), dtype=torch.float32),
            )
            u_prog_emb[st:en] = pemb[0]
            u_struct_sig[st:en] = ssig[0]
            u_eff_prior[st:en] = eff_p[0].numpy()
            u_tot_0rev[st:en] = tot0[0].numpy()

        bank: dict[str, dict[str, Any]] = {
            "__meta__": {
                "u_prog_emb": u_prog_emb,
                "u_struct_sig": u_struct_sig,
                "ctx_emb_zero": ctx_emb_zero,
            }
        }
        for cid in ordered_cids:
            p = pools[cid]
            uids = ctx_uids[cid]
            uids_t = torch.from_numpy(uids)
            m = len(uids)
            pemb = u_prog_emb[uids_t].unsqueeze(0)
            ssig = u_struct_sig[uids_t].unsqueeze(0)
            eff_p_np = u_eff_prior[uids]
            eff_p_t = torch.from_numpy(eff_p_np[None])
            proj_d = torch.as_tensor(p["proj16"][None], dtype=torch.float64)

            tot_w1 = _anchor_tail_from_encoded(
                anc,
                ctx_emb_zero,
                pemb,
                ssig,
                eff_p_t,
                proj_d,
                torch.zeros((1, 1), dtype=torch.long),
                torch.full((1, 1), 100.0, dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
            )
            k_r = min(4, m)
            rev4 = torch.arange(k_r, dtype=torch.long).reshape(1, k_r)
            tot_w4 = _anchor_tail_from_encoded(
                anc,
                ctx_emb_zero,
                pemb,
                ssig,
                eff_p_t,
                proj_d,
                rev4,
                torch.full((1, k_r), 100.0, dtype=torch.float32),
                torch.zeros((1, k_r), dtype=torch.float32),
            )
            rf4 = torch.as_tensor(p["fitness"][:k_r][None], dtype=torch.float32)
            rg4 = torch.as_tensor(p["gate"][:k_r][None], dtype=torch.float32)
            tot_r4 = _anchor_tail_from_encoded(
                anc, ctx_emb_zero, pemb, ssig, eff_p_t, proj_d, rev4, rf4, rg4
            )

            bank[cid] = {
                "uids": uids,
                "prog_std": u_prog_std[uids],
                "sem24": u_sem24[uids],
                "anc_eff": eff_p_np.astype(np.float32),
                "eval_anc_eff": eff_p_np.astype(np.float32),
                "anc_tot_0rev": u_tot_0rev[uids],
                "anc_tot_w0k1": tot_w1[0].numpy(),
                "anc_tot_w0k4": tot_w4[0].numpy(),
                "eval_anc_total": tot_r4[0].numpy(),
                "eval_rf4": p["fitness"][:k_r].astype(np.float32),
                "eval_rg4": p["gate"][:k_r].astype(np.float32),
            }
    return bank


def populate_pairnf_train_anchors_and_trim(
    net: Any,
    bank: dict[str, dict[str, Any]],
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
    inc_top: Mapping[str, np.ndarray],
    *,
    cache_path: Any | None = None,
) -> None:
    """Compute `train_anc_total` (`train_head4`) from deduplicated `u_prog_emb` in ~1s, free `__meta__`,
    and persist slim anchor arrays to `cache_path` (excluding bulky `prog_std`) if not yet cached.
    """
    from pathlib import Path

    anc = net.adapter_10tok.anchor
    meta = bank.get("__meta__")
    if meta is None:
        return
    u_prog_emb = meta["u_prog_emb"]
    u_struct_sig = meta["u_struct_sig"]
    ctx_emb_zero = meta["ctx_emb_zero"]

    with torch.no_grad():
        for cid in ordered_cids:
            b = bank[cid]
            p = pools[cid]
            if cid in inc_top:
                order = np.argsort(p["fitness"])
                ic = int(inc_top[cid][0])
                head_cands = [int(x) for x in order[:48]]
                if ic in head_cands[:4]:
                    head_cands.remove(ic)
                    head_cands.insert(4, ic)
                th4 = np.asarray(head_cands[:4], dtype=np.int64)
                if len(th4) == 4 and np.array_equal(th4, np.arange(4)):
                    b["train_head4"] = th4
                    b["train_anc_total"] = b["eval_anc_total"]
                    b["train_anc_eff"] = b["eval_anc_eff"]
                    b["train_rf4"] = b["eval_rf4"]
                    b["train_rg4"] = b["eval_rg4"]
                else:
                    uids_t = torch.from_numpy(b["uids"])
                    pemb = u_prog_emb[uids_t].unsqueeze(0)
                    ssig = u_struct_sig[uids_t].unsqueeze(0)
                    rev_tr = torch.as_tensor(th4[None], dtype=torch.long)
                    rf_tr = torch.as_tensor(p["fitness"][th4][None], dtype=torch.float32)
                    rg_tr = torch.as_tensor(p["gate"][th4][None], dtype=torch.float32)
                    tot_tr = _anchor_tail_from_encoded(
                        anc,
                        ctx_emb_zero,
                        pemb,
                        ssig,
                        torch.from_numpy(b["anc_eff"][None]).double(),
                        torch.as_tensor(p["proj16"][None], dtype=torch.float64),
                        rev_tr,
                        rf_tr,
                        rg_tr,
                    )
                    b["train_head4"] = th4
                    b["train_anc_total"] = tot_tr[0].numpy()
                    b["train_anc_eff"] = b["eval_anc_eff"]
                    b["train_rf4"] = p["fitness"][th4].astype(np.float32)
                    b["train_rg4"] = p["gate"][th4].astype(np.float32)
            b.pop("uids", None)
    bank.pop("__meta__", None)

    if cache_path is not None and not Path(cache_path).is_file():
        import hashlib

        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        slim = {
            cid: {
                k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                for k, v in bank[cid].items()
                if k != "prog_std"
            }
            for cid in ordered_cids
        }
        tmp = cp.with_suffix(".pt.tmp")
        torch.save(slim, tmp)
        tmp.replace(cp)
        digest = hashlib.sha256(cp.read_bytes()).hexdigest()
        cp.with_suffix(".pt.sha256").write_text(f"{digest}\n", encoding="utf-8")


def fast_adapter_predict_single(
    ad: Any,
    b_entry: Mapping[str, Any],
    tok_1x10x32: torch.Tensor,
    msk_1x10: torch.Tensor,
    reveal_mode: str,
) -> dict[str, np.ndarray]:
    """Exact single-context `FlywheelBlockFormerAdapterNet.predict_costs` using precomputed anchor bank."""
    with torch.no_grad():
        sem24 = torch.from_numpy(b_entry["sem24"][None])  # [1, M, 24]
        prog_std = torch.from_numpy(b_entry["prog_std"][None])  # [1, M, 412]
        m = sem24.shape[1]
        eff_mask = torch.cat([torch.ones((1, 1), dtype=torch.float32), msk_1x10.float()[:, 1:]], dim=1)
        gates = torch.sigmoid(ad.modality_gate_logits).view(1, ad.n_modalities, 1)
        tok = ad.modality_norm(tok_1x10x32.float() + ad.modality_type_emb.unsqueeze(0))
        tok_gated = tok * gates * eff_mask.unsqueeze(-1)
        ctx_pooled = torch.sum(tok_gated, dim=1) / torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)

        sem24_c = sem24 - torch.mean(sem24, dim=1, keepdim=True)
        prog_bot16 = ad.prog_412_bottleneck(prog_std)
        z_sos = torch.zeros((1, m, 2), dtype=torch.float32)
        cand_q = ad.cand_query_proj(torch.cat([sem24, sem24_c, prog_bot16, z_sos], dim=-1))
        mm_attn_out, _ = ad.mm_cross_attn(
            query=cand_q, key=tok_gated, value=tok_gated, key_padding_mask=(eff_mask < 0.5), need_weights=False
        )
        raw_mm = ad.mm_prior_out(
            torch.cat([mm_attn_out, ad.ctx_bilinear_u(ctx_pooled).unsqueeze(1) * ad.prog_bilinear_v(sem24), z_sos], dim=-1)
        ).squeeze(-1)
        mm_prior_adj = torch.tanh(ad.mm_prior_scale) * (raw_mm - torch.mean(raw_mm, dim=-1, keepdim=True))
        anc_eff = torch.from_numpy(b_entry["anc_eff"][None])
        eff_prior = anc_eff + mm_prior_adj

        if reveal_mode == "true_0rev":
            anc_tot = torch.from_numpy(b_entry["anc_tot_0rev"][None])
            tot = anc_tot + mm_prior_adj
            return {
                "total_cost": tot[0].numpy(),
                "eff_prior": eff_prior[0].numpy(),
                "mm_prior_adj": mm_prior_adj[0].numpy(),
                "cand_q": cand_q[0].numpy(),
                "sem24": b_entry["sem24"],
            }

        if reveal_mode == "wave0_fake_k1":
            anc_tot = torch.from_numpy(b_entry["anc_tot_w0k1"][None])
            rev_idx = torch.zeros((1, 1), dtype=torch.long)
            rev_f = torch.full((1, 1), 100.0, dtype=torch.float32)
            rev_g = torch.zeros((1, 1), dtype=torch.float32)
        elif reveal_mode == "wave0_fake_k4":
            anc_tot = torch.from_numpy(b_entry["anc_tot_w0k4"][None])
            k_r = min(4, m)
            rev_idx = torch.arange(k_r, dtype=torch.long).reshape(1, k_r)
            rev_f = torch.full((1, k_r), 100.0, dtype=torch.float32)
            rev_g = torch.zeros((1, k_r), dtype=torch.float32)
        elif reveal_mode == "real_4rev":
            anc_tot = torch.from_numpy(b_entry["eval_anc_total"][None])
            k_r = len(b_entry["eval_rf4"])
            rev_idx = torch.arange(k_r, dtype=torch.long).reshape(1, k_r)
            rev_f = torch.from_numpy(b_entry["eval_rf4"][None])
            rev_g = torch.from_numpy(b_entry["eval_rg4"][None])
        else:
            raise ValueError(f"unknown reveal_mode: {reveal_mode}")

        static_terms = {
            "eff_prior": eff_prior,
            "mm_prior_adj": mm_prior_adj,
            "sem24": sem24,
            "cand_q": cand_q,
            "cand_q_norm": F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        }
        ad_delta = ad.compute_adapter_delta_from_static_terms(
            static_terms, torch.full((1,), 0.4208, dtype=torch.float32), rev_idx, rev_f, rev_g
        )
        return {
            "total_cost": (anc_tot + ad_delta)[0].numpy(),
            "eff_prior": eff_prior[0].numpy(),
            "mm_prior_adj": mm_prior_adj[0].numpy(),
            "cand_q": cand_q[0].numpy(),
            "sem24": b_entry["sem24"],
        }
