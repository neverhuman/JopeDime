"""Exact H170 AdamW and adapter math inherited unchanged from H257/H247."""

import math
import torch

CONTROL_LR, CONTROL_WD, BETAS = 1e-4, 1e-4, (0.9, 0.999)


class ControlAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW, with a no-op gradient filter."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


def gradient_norm(params):
    """None contributes zero to diagnostics; native AdamW None semantics are retained."""
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    """All 29,211 coordinates, including inactive/unchanged tensors; float64 L2."""
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )


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


def l2sp_penalty(params, initial, beta=50.0):
    return (
        0.5
        * beta
        * torch.stack(
            [(p - p0).square().sum() for p, p0 in zip(params, initial, strict=True)]
        ).sum()
    )


@torch.no_grad()
def retract(params, initial, gamma=0.25):
    for p, p0 in zip(params, initial, strict=True):
        p.sub_(gamma * (p - p0))


def head4_loss(pred, fitness, gate):
    """Six unordered pairs; invalid is worst, invalid-invalid pairs omitted.

    Raw valid costs determine ordering and the 1e-4 near-tie filter. For a
    valid/invalid pair, invalid is always worst even for valid raw cost>2.
    The utility sentinel -2 is never treated as a measured valid cost.
    """
    if pred.shape != (4,) or fitness.shape != (4,) or gate.shape != (4,):
        raise ValueError("four candidates required")
    i, j = torch.triu_indices(4, 4, offset=1)
    gi, gj = gate[i].bool(), gate[j].bool()
    validpair = gi & gj
    cost = torch.where(gate.bool(), fitness, torch.full_like(fitness, 2.0))
    gap = cost[i] - cost[j]
    mask = (gap.abs() >= 1e-4) & (validpair | (gi != gj))
    sign = torch.where(gi != gj, torch.where(gi, -1.0, 1.0), torch.sign(gap))
    losses = torch.nn.functional.softplus(sign * (pred[j] - pred[i]))
    return losses[mask].mean() if mask.any() else pred.sum() * 0.0
