"""H188 inherits the H184/H180/H176 forward, anchor-only scorer and component decomposition verbatim and adds the
prespecified kernel-INPUT variants. `kernel_term` replays the live kernel lines of
FlywheelBlockFormerAdapterNet.compute_adapter_delta_from_static_terms with three explicit substitution points
(similarity features, residual transform); with the identity substitutions it must equal the live KERNEL term bit for
bit, which the runner checks on every batch."""

import math

import torch


def adapter_forward(
    net, prog, sos, tokens, mask, scale, rev, raw, gate, anchor_total, anchor_prior
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tokens,
        mm_mask_10=mask,
        prog_412=prog,
        sos_prior=sos,
        active_modality_override_mask=None,
    )
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
        revealed_raw_f=raw,
        revealed_gate=gate,
    )
    return (anchor_total + delta).float()


def gradient_norm(params):
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or not (
            (type(launch.get("review_id")) is int and launch["review_id"] > 0)
            or (
                isinstance(launch.get("review_waiver"), str)
                and len(launch["review_waiver"].strip()) >= 20
            )
        )
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")


def zero_adapter_scores(anchor_total):
    """ZERO_ADAPTER: the adapter contribution removed. The live FlywheelBlockFormerAdapterNet.predict_costs returns
    total_cost = anchor_total + adapter_total_delta, where anchor_total is anchor.predict_costs(...) with prior
    correction, struct kernel and cross-attention on and UCB off; dropping adapter_total_delta leaves anchor_total."""
    return anchor_total.detach().float().clone()


def assert_anchor_decomposition(native, anchor_total):
    """Proof that the anchor-only scorer is what the live code produces with the adapter delta set to zero: the live
    total must equal the separately computed anchor_total plus the live adapter_total_delta, bit for bit."""
    import torch

    if "adapter_total_delta" not in native:
        raise RuntimeError(
            "live scorer did not expose adapter_total_delta (revealed history required)"
        )
    if not torch.equal(
        native["total_cost"].float(),
        anchor_total.float() + native["adapter_total_delta"].float(),
    ):
        raise RuntimeError("live total_cost != anchor_total + adapter_total_delta")
    return True


COMPONENTS = ("PRIOR", "KERNEL", "ATTN")


def static_terms(net, prog, sos, tokens, mask, anchor_prior):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tokens,
        mm_mask_10=mask,
        prog_412=prog,
        sos_prior=sos,
        active_modality_override_mask=None,
    )
    return dict(
        eff_prior=anchor_prior + adjustment,
        mm_prior_adj=adjustment,
        sem24=semantic,
        cand_q=query,
        cand_q_norm=F.normalize(query, p=2, dim=-1, eps=1e-8),
        ctx_pooled=pooled,
    )


def adapter_components(
    ad, static, scale, revealed_indices, revealed_raw_f, revealed_gate
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
    k_take = min(8, K_rev)
    top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
    r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
    top_r = torch.gather(r_exp, dim=-1, index=top_pos)
    w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
    w0 = F.softplus(ad.null_kernel_w0_raw) + 0.15
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
    blend_w = torch.sigmoid(ad.traj_kernel_blend) * 0.45
    return {"PRIOR": mm_prior_adj, "KERNEL": blend_w * sem_adj, "ATTN": traj_attn_adj}


def assert_component_sum(components, live_delta):
    """The three summands, added in the live order, must reproduce the live adapter delta bit for bit."""
    import torch

    total = components["PRIOR"] + components["KERNEL"] + components["ATTN"]
    if not torch.equal(total, live_delta):
        raise RuntimeError("component sum != live adapter delta")
    return True


def ablated_scores(anchor_total, components, drop):
    """Output-level single-component ablation: anchor_total + (sum of the kept components, live order)."""
    if drop not in COMPONENTS:
        raise ValueError("unknown component")
    kept = [components[c] for c in COMPONENTS if c != drop]
    return (anchor_total + (kept[0] + kept[1])).float()


HUBER_C = 1.345  # Huber's canonical 95%-efficiency constant
MAD_TO_SD = 1.4826


def rank_residuals(r):
    """KIN_RANKRES: within-context rank scores rescaled to the context's own residual location and scale:
    r' = mean(r) + std(r) * u / std(u), u = average-tie rank - (K+1)/2. If std(r) == 0, r is returned unchanged."""
    import torch

    k = r.shape[-1]
    if k < 2:
        return r.clone()
    order = torch.argsort(torch.argsort(r, dim=-1), dim=-1).to(r.dtype) + 1.0  # 1..K
    ties = (r.unsqueeze(-1) == r.unsqueeze(-2)).to(r.dtype)
    avg_rank = (ties * order.unsqueeze(-2)).sum(-1) / ties.sum(
        -1
    )  # average rank over tied values
    u = avg_rank - (k + 1) / 2.0
    su = u.std(dim=-1, keepdim=True, unbiased=False)
    sr = r.std(dim=-1, keepdim=True, unbiased=False)
    scaled = r.mean(dim=-1, keepdim=True) + sr * u / torch.where(
        su > 0, su, torch.ones_like(su)
    )
    return torch.where((sr > 0) & (su > 0), scaled, r)


def huber_clip_residuals(r):
    """KIN_CLIPRES: Huber winsorization around the context median at c * 1.4826 * MAD (c = 1.345); if MAD == 0 the
    residuals are returned unchanged. Large invalid-outcome residuals (failed gates enter capped at 2.5) are clipped."""
    import torch

    med = r.median(dim=-1, keepdim=True).values
    mad = (r - med).abs().median(dim=-1, keepdim=True).values
    bound = HUBER_C * MAD_TO_SD * mad
    clipped = med + torch.clamp(r - med, min=-bound, max=bound)
    return torch.where(mad > 0, clipped, r)


def kernel_term(
    ad,
    sem24,
    cand_q_norm,
    eff_prior,
    scale,
    revealed_indices,
    revealed_raw_f,
    revealed_gate,
    residual_transform=None,
):
    """The live KERNEL term (0.45 * sigmoid(blend) * sem_adj). `sem24`/`cand_q_norm` are the similarity features and
    `residual_transform` maps r_rev -> r_rev'; identity inputs reproduce the live term exactly."""
    import torch
    import torch.nn.functional as F

    B, M, d_sem = sem24.shape
    K_rev = int(revealed_indices.shape[1])
    if K_rev == 0:
        raise ValueError("H188 requires revealed history (K_rev > 0)")
    scale_col = scale.float().view(B, 1).clamp(min=1e-4)
    rev_raw_f_flt = revealed_raw_f.float()
    rev_gate_flt = revealed_gate.float()
    idx_s = revealed_indices.unsqueeze(-1).expand(B, K_rev, d_sem)
    rev_sem = torch.gather(sem24, dim=1, index=idx_s)
    idx_q = revealed_indices.unsqueeze(-1).expand(B, K_rev, cand_q_norm.shape[-1])
    rev_q_norm = torch.gather(cand_q_norm, dim=1, index=idx_q)
    rev_eff_prior = torch.gather(eff_prior, dim=1, index=revealed_indices)
    rev_f_capped = torch.where(
        rev_gate_flt > 0.5, rev_raw_f_flt, torch.clamp(rev_raw_f_flt, max=2.5)
    )
    r_rev = (rev_f_capped / scale_col) - rev_eff_prior
    if residual_transform is not None:
        r_rev = residual_transform(r_rev)
    cos_sem = torch.bmm(sem24, rev_sem.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_sem = (1.0 - cos_sem) * F.softplus(ad.sem_kernel_scale_raw)
    cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_emb = (1.0 - cos_emb) * F.softplus(ad.emb_kernel_scale_raw)
    d2_joint = d2_sem + d2_emb
    k_take = min(8, K_rev)
    top_d, top_pos = torch.topk(d2_joint, k=k_take, dim=-1, largest=False)
    r_exp = r_rev.unsqueeze(1).expand(B, M, K_rev)
    top_r = torch.gather(r_exp, dim=-1, index=top_pos)
    w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
    w0 = F.softplus(ad.null_kernel_w0_raw) + 0.15
    sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))
    blend_w = torch.sigmoid(ad.traj_kernel_blend) * 0.45
    return blend_w * sem_adj


def anchor_features(anchor, ctx_feat, prog_412, proj_16, sos_prior):
    """KIN_ANCHOR similarity features from the frozen anchor (read-only): L2-normalized structural signatures (24-d) in
    place of the adapter's sem24, and L2-normalized anchor candidate embeddings in place of cand_q_norm."""
    import torch.nn.functional as F

    _, prog_emb, struct_sig, _ = anchor.encode_candidates(
        ctx_feat, prog_412, proj_16, sos_prior
    )
    return F.normalize(struct_sig.float(), p=2, dim=-1, eps=1e-8), F.normalize(
        prog_emb.float(), p=2, dim=-1, eps=1e-8
    )


def variant_scores(anchor_total, components, kernel_variant):
    """Arm score: anchor_total + (PRIOR + KERNEL' + ATTN), live summation order, with only the kernel term replaced."""
    return (
        anchor_total + (components["PRIOR"] + kernel_variant + components["ATTN"])
    ).float()
