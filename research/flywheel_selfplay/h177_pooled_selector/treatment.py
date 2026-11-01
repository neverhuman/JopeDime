"""H177 recipe treatments (CTRL, ACC4, ACC16, OC15), None-as-zero 29,211-param gradients, and H175 launch gate."""

import math
import re
import torch
import torch.nn.functional as F

PARAMETERS = 29211
TENSORS = 47
CANDIDATE_STEP = 40
ACCUM_STEPS = {"CTRL": 1, "ACC4": 4, "ACC16": 16, "OC15": 1}
TOTAL_UPDATES = {"CTRL": 40, "ACC4": 40, "ACC16": 40, "OC15": 40}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def oc15_cycle(step, peak=1.5e-4):
    """H170 OC15 schedule at update t=1..40: 12 linear warmup endpoints (1e-5 -> 1.5e-4, beta1 0.95 -> 0.85), 28 cosine-anneal updates to 1e-6 (beta1 0.85 -> 0.95)."""
    if not 1 <= step <= CANDIDATE_STEP or peak != 1.5e-4:
        raise ValueError("fixed H170 OC15 40-step schedule only")
    if step <= 12:
        q = (step - 1) / 11.0
        return 1e-5 + (peak - 1e-5) * q, 0.95 - 0.10 * q
    q = (step - 12) / 28.0
    c = 0.5 * (1.0 + math.cos(math.pi * q))
    return 1e-6 + (peak - 1e-6) * c, 0.95 - 0.10 * c


def optimizer_settings(arm, step):
    if arm not in ACCUM_STEPS or not 1 <= step <= CANDIDATE_STEP:
        raise ValueError("unknown arm or update step")
    if arm == "OC15":
        return oc15_cycle(step, 1.5e-4)
    return 1e-4, 0.9


def apply_optimizer_settings(optimizer, arm, step):
    lr, beta1 = optimizer_settings(arm, step)
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["betas"] = (beta1, 0.999)
    return lr, beta1


H178_CALIBRATION_GATE = (
    "Any H177 ADMIT is lift only if H178 (gpu_tokbench_agy, calibration of this exact rule on the H175 bank) "
    "shows null/harmful admission <= 0.05. Otherwise H177 is DIAGNOSTIC."
)


def validate_h175_power_anchor(anchor, smoke=False):
    """Enforce that non-smoke execution waits for H175 closeout with SELECTOR_UNDERPOWERED or MIXED, both context and family-clustered MDE80s, and the H178 calibration gate."""
    if not isinstance(anchor, dict) or anchor.get("wave") != "H177":
        raise RuntimeError("invalid h175_power_anchor.json")
    if (
        anchor.get("single_fold_b_families") != 41
        or anchor.get("pooled_b_families") != 205
        or abs(float(anchor.get("family_count_ratio", 0.0)) - 5.0) > 1e-12
    ):
        raise RuntimeError("h175_power_anchor family ratio mismatch")
    if anchor.get("h178_calibration_gate") != H178_CALIBRATION_GATE:
        raise RuntimeError("h175_power_anchor missing exact H178 calibration gate")
    mde41 = anchor.get("mde80_single_fold_41_family")
    mde205 = anchor.get("expected_pooled_205_family_mde80")
    if not isinstance(mde41, dict) or not isinstance(mde205, dict):
        raise RuntimeError(
            "h175_power_anchor must include both context and family_clustered MDE80 dictionaries"
        )
    for regime in ("context", "family_clustered"):
        if not isinstance(mde41.get(regime), dict) or not isinstance(
            mde205.get(regime), dict
        ):
            raise RuntimeError(f"missing {regime} MDE80 block in h175_power_anchor")
        for slate in ("24_slate", "full_slate"):
            v41 = mde41[regime].get(slate)
            v205 = mde205[regime].get(slate)
            if (
                v41 is None
                or v205 is None
                or not math.isfinite(float(v41))
                or not math.isfinite(float(v205))
                or float(v41) <= 0.0
                or float(v205) <= 0.0
            ):
                raise RuntimeError(
                    f"non-positive or non-finite MDE80 for {regime}/{slate}"
                )
    if smoke:
        return
    verdict = anchor.get("h175_verdict")
    if verdict == "CEILING_LIKELY":
        raise RuntimeError(
            "H175 returned CEILING_LIKELY: H177 must close as NOT_RUN without executing"
        )
    if (
        anchor.get("status") != "READY_TO_LAUNCH"
        or verdict not in ("SELECTOR_UNDERPOWERED", "MIXED")
        or not isinstance(anchor.get("h175_closeout_commit"), str)
        or not HEX40_RE.match(anchor["h175_closeout_commit"])
        or anchor.get("h175_mde80_41fam") is None
        or not math.isfinite(float(anchor["h175_mde80_41fam"]))
        or float(anchor["h175_mde80_41fam"]) <= 0.0
        or anchor.get("expected_mde80_205fam") is None
        or not math.isfinite(float(anchor["expected_mde80_205fam"]))
        or float(anchor["expected_mde80_205fam"]) <= 0.0
    ):
        raise RuntimeError(
            "H177 science launch blocked until H175 closes with SELECTOR_UNDERPOWERED or MIXED and h175_power_anchor.json is updated with measured MDE80"
        )


def compute_fit_family_weights(fit_rows, excluded_cids=()):
    """Compute fit-only equal-family context weights w_c = 1 / (|F_fit| * |C_f(c)|) matching family-stratified sampling."""
    excl = set(excluded_cids)
    fam_counts = {}
    row_by_cid = {}
    for r in fit_rows:
        cid = r["context_sha256"]
        if cid in excl:
            raise ValueError("fit_rows must be disjoint from excluded_cids")
        if cid not in row_by_cid:
            row_by_cid[cid] = r["family_id"]
            fam_counts[r["family_id"]] = fam_counts.get(r["family_id"], 0) + 1
    if not row_by_cid or not fam_counts:
        raise ValueError("non-empty fit_rows required")
    ordered_cids = sorted(row_by_cid)
    n_fams = len(fam_counts)
    weights = torch.tensor(
        [1.0 / (n_fams * fam_counts[row_by_cid[c]]) for c in ordered_cids],
        dtype=torch.float64,
    )
    if not torch.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > 1e-12:
        raise ValueError("fit family weights must be finite and sum to 1.0")
    return ordered_cids, weights, n_fams


def flatten_grad_with_none_as_zero(params, expected_numel=PARAMETERS):
    """Flatten parameter gradients into a single float64 vector, treating p.grad is None as exact zeros."""
    plist = list(params)
    pieces = []
    active_count = 0
    none_count = 0
    for p in plist:
        if p.grad is None:
            none_count += 1
            pieces.append(torch.zeros(p.numel(), dtype=torch.float64, device="cpu"))
        else:
            active_count += 1
            pieces.append(p.grad.detach().cpu().double().reshape(-1))
    if active_count == 0:
        raise RuntimeError("all parameter gradients are None")
    vec = torch.cat(pieces)
    if expected_numel is not None and vec.numel() != expected_numel:
        raise ValueError(
            f"expected {expected_numel} gradient params, got {vec.numel()}"
        )
    if not torch.isfinite(vec).all():
        raise RuntimeError("nonfinite gradient vector")
    return vec, active_count, none_count


def grad_norms(params, max_norm=1.0, expected_numel=PARAMETERS):
    """Extract pre-clip None-zero-filled gradient vector, clip in-place, and return pre/post L2 norms."""
    plist = list(params)
    pre_vec, active_count, none_count = flatten_grad_with_none_as_zero(
        plist, expected_numel=expected_numel
    )
    raw_norm = float(torch.linalg.vector_norm(pre_vec))
    clip_ret = float(
        torch.nn.utils.clip_grad_norm_(plist, max_norm, error_if_nonfinite=True)
    )
    post_vec, _, _ = flatten_grad_with_none_as_zero(
        plist, expected_numel=expected_numel
    )
    clipped_norm = float(torch.linalg.vector_norm(post_vec))
    if (
        not math.isfinite(raw_norm)
        or not math.isfinite(clip_ret)
        or not math.isfinite(clipped_norm)
    ):
        raise RuntimeError("nonfinite gradient norm")
    return pre_vec, dict(
        raw_grad_l2=raw_norm,
        clipped_grad_l2=clipped_norm,
        active_grad_tensors=active_count,
        none_grad_tensors=none_count,
    )


def accumulate_microbatches(params, microbatch_losses):
    """Accumulate mean gradient across M microbatch loss callables: (1/M) * sum_{m=0..M-1} grad(L_m)."""
    plist = list(params)
    m_count = len(microbatch_losses)
    if m_count not in (1, 4, 16):
        raise ValueError("microbatch accumulation count must be in {1, 4, 16}")
    for p in plist:
        p.grad = None
    losses = []
    for fn in microbatch_losses:
        loss_val = fn()
        if not torch.isfinite(loss_val):
            raise RuntimeError("nonfinite microbatch objective")
        losses.append(float(loss_val.detach()))
        (loss_val / float(m_count)).backward()
    return float(sum(losses) / m_count)


def adapter_l2_drift(state_a, state_b, names):
    """Compute float64 L2 norm of parameter difference across the 47 trainable adapter tensors."""
    diffs = [
        (state_a[n].detach().double() - state_b[n].detach().double()).reshape(-1)
        for n in names
    ]
    return float(torch.linalg.vector_norm(torch.cat(diffs)))


def precompute_static_slate(ad, prog, sos, mask, sem24_fn=None, transform_prog_fn=None):
    """Precompute parameter-free static candidate/mask terms for bit-exact fast adapter forward passes."""
    if sem24_fn is None or transform_prog_fn is None:
        from phase12_multimodal_model import (
            compute_semantic_struct_sig_24d_torch,
            transform_program_features_torch,
        )

        sem24_fn = compute_semantic_struct_sig_24d_torch
        transform_prog_fn = transform_program_features_torch

    with torch.no_grad():
        sem24 = sem24_fn(prog)
        sem24_centered = sem24 - torch.mean(sem24, dim=1, keepdim=True)
        prog_std_412 = transform_prog_fn(prog, ad.prog_mean, ad.prog_std)
        eff_mask = mask.float().clone()
        eff_mask[:, 0] = 1.0
        mask_sum = torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
        key_pad_mask = eff_mask < 0.5
    return dict(
        sem24=sem24,
        sem24_centered=sem24_centered,
        prog_std_412=prog_std_412,
        sos=sos.float(),
        eff_mask=eff_mask,
        mask_sum=mask_sum,
        key_pad_mask=key_pad_mask,
    )


def adapter_forward_static(
    net,
    tok,
    static_slate,
    scale,
    rev,
    raw,
    gate,
    anchor_total,
    anchor_prior,
):
    """Bit-exact adapter forward pass using precomputed parameter-free candidate/mask terms."""
    ad = net.adapter_10tok
    if ad.training:
        raise RuntimeError("adapter_10tok must remain in eval mode")
    sem24 = static_slate["sem24"]
    sem24_centered = static_slate["sem24_centered"]
    prog_std_412 = static_slate["prog_std_412"]
    sos = static_slate["sos"]
    eff_mask = static_slate["eff_mask"]
    mask_sum = static_slate["mask_sum"]
    key_pad_mask = static_slate["key_pad_mask"]

    gates = torch.sigmoid(ad.modality_gate_logits).view(1, ad.n_modalities, 1)
    t = ad.modality_norm(tok.float() + ad.modality_type_emb.unsqueeze(0))
    tok_gated = t * gates * eff_mask.unsqueeze(-1)
    ctx_pooled = torch.sum(tok_gated, dim=1) / mask_sum
    prog_bot16 = ad.prog_412_bottleneck(prog_std_412)
    cand_q = ad.cand_query_proj(
        torch.cat([sem24, sem24_centered, prog_bot16, sos], dim=-1)
    )
    mm_attn_out, _ = ad.mm_cross_attn(
        query=cand_q,
        key=tok_gated,
        value=tok_gated,
        key_padding_mask=key_pad_mask,
        need_weights=False,
    )
    bilinear_16 = ad.ctx_bilinear_u(ctx_pooled).unsqueeze(1) * ad.prog_bilinear_v(sem24)
    raw_mm_delta = ad.mm_prior_out(
        torch.cat([mm_attn_out, bilinear_16, sos], dim=-1)
    ).squeeze(-1)
    raw_mm_centered = raw_mm_delta - torch.mean(raw_mm_delta, dim=-1, keepdim=True)
    adjustment = torch.tanh(ad.mm_prior_scale) * raw_mm_centered
    static = dict(
        eff_prior=anchor_prior + adjustment,
        mm_prior_adj=adjustment,
        sem24=sem24,
        cand_q=cand_q,
        cand_q_norm=F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        ctx_pooled=ctx_pooled,
    )
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale,
        revealed_indices=rev,
        revealed_raw_f=raw,
        revealed_gate=gate,
    )
    return (anchor_total + delta).float()


def adapter_forward(
    net,
    prog,
    sos,
    tokens,
    mask,
    scale,
    rev,
    raw,
    gate,
    anchor_total,
    anchor_prior,
):
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
