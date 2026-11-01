"""Shared frozen-anchor forward and zero-gradient optimizer semantics for H169."""

import torch


def gradient_norm(params):
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def materialize_missing_grads(params):
    """Treat every absent adapter gradient as an explicit zero before optimizer.step."""
    missing = []
    for index, parameter in enumerate(params):
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            missing.append(index)
    return tuple(missing)


def optimizer_for(params, arm, sgd_lr):
    if arm == "CTRL":
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4, eps=1e-8)
    if arm == "ADAM_EPS":
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4, eps=1e-3)
    if arm in ("SGDM", "SGDM_NEST"):
        return torch.optim.SGD(
            params,
            lr=sgd_lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=arm == "SGDM_NEST",
        )
    raise ValueError("unknown H169 optimizer arm: " + arm)


def adapter_forward(net, prog, sos, tokens, mask, scale, rev, raw, gate,
                    anchor_total, anchor_prior, input_site):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=input_site(tokens),
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


def adapter_drift(net, names, initial):
    return float(
        torch.sqrt(
            sum(
                (net.state_dict()[name].detach() - initial[name]).double().square().sum()
                for name in names
            )
        )
    )
