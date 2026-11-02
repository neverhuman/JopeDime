"""H234 loss arms (CTRL loss vs LambdaLoss NDCG-weighted pairwise) for the r882 adapter continuation; every arm uses
the exact merged H170 CTRL AdamW recipe. Also None-safe diagnostics."""

import math
import torch

ARMS = ("CONTROL", "LAMBDAMIX", "LAMBDA")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
LAMBDA_WEIGHT = 0.5  # hw_t2220: LAMBDAMIX = CTRL loss + 0.5 * Lambda
LAMBDA_SIGMA = 1.0  # pairwise logistic scale on score differences


def make_optimizer(arm, params):
    """Identical AdamW for every arm (H234 changes only the loss)."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return torch.optim.AdamW(
        params, lr=CONTROL_LR, betas=(0.9, 0.999), weight_decay=CONTROL_WD
    )


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay) of an arm at update 1..40 (constant, identical across arms)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    return CONTROL_LR, 0.9, 0.999, CONTROL_WD


def lambda_loss(pred, fitness, gate, sigma=LAMBDA_SIGMA):
    """LambdaRank-style NDCG-weighted pairwise logistic loss (Burges 2010; the LambdaLoss framework, Wang et al. 2018),
    per context over its candidates, averaged over contexts.

    Costs: `pred` is the adapter cost (lower = picked first). Targets reuse the CTRL loss's cost target
    t = fitness clamped to [-4, 2] where the gate passes, else 2.0; utility u = -t, score s = -pred.
    Graded relevance rel = (u - min u) / (max u - min u) in [0, 1] per context, gain 2^rel - 1.
    For every pair with u_i > u_j: weight |dNDCG_ij| = |gain_i - gain_j| * |1/log2(1+r_i) - 1/log2(1+r_j)| / IDCG with
    ranks r from the current scores (detached), and loss softplus(-sigma * (s_i - s_j)). A context's loss is the
    weight-normalised mean over its pairs; contexts with no strictly ordered pair contribute 0."""
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    u = -target.double()
    s = -pred
    lo, hi = u.min(dim=1, keepdim=True).values, u.max(dim=1, keepdim=True).values
    span = hi - lo
    rel = torch.where(
        span > 0,
        (u - lo) / torch.where(span > 0, span, torch.ones_like(span)),
        torch.zeros_like(u),
    )
    gain = torch.pow(2.0, rel) - 1.0
    n = pred.shape[1]
    order = torch.argsort(s.detach(), dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1, order, torch.arange(1, n + 1, device=pred.device).expand_as(order)
    )
    disc = 1.0 / torch.log2(1.0 + ranks.double())
    ideal = torch.sort(gain, dim=1, descending=True).values
    idcg = (ideal / torch.log2(torch.arange(2, n + 2, dtype=torch.float64))).sum(
        dim=1, keepdim=True
    )
    better = (u[:, :, None] > u[:, None, :]).double()
    w = (gain[:, :, None] - gain[:, None, :]).abs() * (
        disc[:, :, None] - disc[:, None, :]
    ).abs()
    w = w * better / torch.where(idcg > 0, idcg, torch.ones_like(idcg))[:, :, None]
    pair = torch.nn.functional.softplus(-sigma * (s[:, :, None] - s[:, None, :]))
    total = w.sum(dim=(1, 2))
    per_context = torch.where(
        total > 0,
        (w * pair.double()).sum(dim=(1, 2))
        / torch.where(total > 0, total, torch.ones_like(total)),
        0.0,
    )
    return per_context.mean().to(pred.dtype)


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """CONTROL: the inherited H146 CTRL loss. LAMBDAMIX: CTRL + 0.5 * Lambda. LAMBDA: Lambda only."""
    if arm == "CONTROL":
        return ctrl_loss(pred, fitness, gate)
    if arm == "LAMBDAMIX":
        return ctrl_loss(pred, fitness, gate) + LAMBDA_WEIGHT * lambda_loss(
            pred, fitness, gate
        )
    if arm == "LAMBDA":
        return lambda_loss(pred, fitness, gate)
    raise ValueError("unknown arm")


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
        or type(launch.get("review_id")) is not int
        or launch["review_id"] <= 0
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
