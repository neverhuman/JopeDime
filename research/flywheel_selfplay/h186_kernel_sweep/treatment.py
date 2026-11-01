"""H184 component scorer with exactly three configurable constants; weights remain unchanged."""

DEFAULT = dict(kernel_multiplier=1.0, topk=8, null_multiplier=1.0)
CHANGES = {
    "KX0.5": ("kernel_multiplier", 0.5),
    "KX1.5": ("kernel_multiplier", 1.5),
    "KX2.0": ("kernel_multiplier", 2.0),
    "TOPK4": ("topk", 4),
    "TOPK16": ("topk", 16),
    "W0X0.5": ("null_multiplier", 0.5),
    "W0X2": ("null_multiplier", 2.0),
}


def constants(arm):
    values = DEFAULT.copy()
    if arm != "R882":
        key, value = CHANGES[arm]
        values[key] = value
    return values


def scores(anchor, components):
    return (
        anchor + (components["PRIOR"] + components["KERNEL"] + components["ATTN"])
    ).float()


def adapter_components(
    ad,
    static,
    scale,
    revealed_indices,
    revealed_raw_f,
    revealed_gate,
    *,
    kernel_multiplier=1.0,
    topk=8,
    null_multiplier=1.0,
):
    """Return {PRIOR: mm_prior_adj, KERNEL: blend_w * sem_adj, ATTN: traj_attn_adj} for K_rev > 0, replaying the live
    compute_adapter_delta_from_static_terms (whose return value is PRIOR + KERNEL + ATTN in that order)."""
    import torch
    import torch.nn.functional as F

    eff_prior = static["eff_prior"]
    mm_prior_adj = static["mm_prior_adj"]
    sem24 = static["sem24"]
    cand_q = static["cand_q"]
    cand_q_norm = static["cand_q_norm"]
    B, M, _ = sem24.shape
    K_rev = int(revealed_indices.shape[1])
    if K_rev == 0:
        raise ValueError("H184 requires revealed history (K_rev > 0)")
    scale_col = scale.float().view(B, 1).clamp(min=1e-4)
    rev_raw_f_flt = revealed_raw_f.float()
    rev_gate_flt = revealed_gate.float()
    idx_24 = revealed_indices.unsqueeze(-1).expand(B, K_rev, 24)
    rev_sem24 = torch.gather(sem24, dim=1, index=idx_24)
    idx_32 = revealed_indices.unsqueeze(-1).expand(B, K_rev, ad.d_token)
    rev_q_norm = torch.gather(cand_q_norm, dim=1, index=idx_32)
    rev_q = torch.gather(cand_q, dim=1, index=idx_32)
    rev_eff_prior = torch.gather(eff_prior, dim=1, index=revealed_indices)
    rev_f_capped = torch.where(
        rev_gate_flt > 0.5, rev_raw_f_flt, torch.clamp(rev_raw_f_flt, max=2.5)
    )
    r_rev = (rev_f_capped / scale_col) - rev_eff_prior
    cos_sem = torch.bmm(sem24, rev_sem24.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_sem = (1.0 - cos_sem) * F.softplus(ad.sem_kernel_scale_raw)
    cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_emb = (1.0 - cos_emb) * F.softplus(ad.emb_kernel_scale_raw)
    d2_joint = d2_sem + d2_emb
    k_take = min(topk, K_rev)
    top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
    r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
    top_r = torch.gather(r_exp, dim=-1, index=top_pos)
    w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
    w0 = (F.softplus(ad.null_kernel_w0_raw) + 0.15) * null_multiplier
    sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))
    step_pos = (
        torch.arange(K_rev, device=sem24.device, dtype=torch.float32).view(1, K_rev, 1)
        / 64.0
    )
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
    h_tok = ad.hist_token_proj(h_in)
    null_tok = ad.null_hist_token.expand(B, 1, ad.d_token)
    kv_seq = torch.cat([null_tok, h_tok], dim=1)
    attn_out, _ = ad.traj_cross_attn(
        query=cand_q, key=kv_seq, value=kv_seq, need_weights=False
    )
    best_rev_y = (
        torch.min(rev_f_capped / scale_col, dim=-1, keepdim=True)
        .values.unsqueeze(-1)
        .expand(B, M, 1)
    )
    traj_in = torch.cat(
        [cand_q, attn_out, eff_prior.unsqueeze(-1), sem_adj.unsqueeze(-1), best_rev_y],
        dim=-1,
    )
    raw_traj_delta = ad.traj_delta_head(traj_in).squeeze(-1)
    raw_traj_centered = raw_traj_delta - torch.mean(
        raw_traj_delta, dim=-1, keepdim=True
    )
    traj_attn_adj = torch.tanh(ad.traj_attn_scale) * raw_traj_centered
    blend_w = torch.sigmoid(ad.traj_kernel_blend) * (0.45 * kernel_multiplier)
    return {"PRIOR": mm_prior_adj, "KERNEL": blend_w * sem_adj, "ATTN": traj_attn_adj}
