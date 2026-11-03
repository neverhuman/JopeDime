"""H250 per-context standardized regression loss; all arms retain exact H170 CTRL AdamW."""

import math
import numpy as np
import torch
import torch.nn.functional as F

ARMS = ("CONTROL", "CTXZ_FB", "CTXZ")
CONTROL_LR, CONTROL_WD, BETAS = 1e-4, 1e-4, (0.9, 0.999)


class ControlAdamW(torch.optim.AdamW):
    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


def make_optimizer(arm, params):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ControlAdamW(list(params))


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm/update")
    return CONTROL_LR, BETAS[0], BETAS[1], CONTROL_WD


def failure_ceiling(fitness, gate):
    """Native invalid utility -2 is cost +2; replace invalid cost by worst valid +1."""
    fitness, gate = np.asarray(fitness), np.asarray(gate, dtype=bool)
    if not gate.any() or not np.isfinite(fitness[gate]).all():
        raise ValueError("finite non-failure label required")
    return float(fitness[gate].max()) + 1.0


def clipped_targets(fitness, gate, ceilings):
    costs = torch.where(gate, fitness, ceilings[:, None].expand_as(fitness)).detach()
    if not torch.isfinite(costs).all():
        raise ValueError("nonfinite normalized-loss targets")
    return costs


def standardize(values, epsilon):
    """Population std over each sampled24 slate; gradients pass through prediction mean/std."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive frozen epsilon required")
    if not torch.isfinite(values).all():
        raise ValueError("finite scores required")
    return (values - values.mean(1, keepdim=True)) / values.std(
        1, correction=0, keepdim=True
    ).clamp_min(epsilon)


def normalized_loss(pred, fitness, gate, epsilon, ceilings):
    target = clipped_targets(fitness, gate, ceilings)
    z_pred, z_true = standardize(pred, epsilon), standardize(target, epsilon)
    return (
        F.smooth_l1_loss(z_pred, z_true, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-z_pred / 0.2, dim=-1),
            F.softmax(-z_true / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def arm_loss(arm, ctrl_loss, pred, fitness, gate, epsilon, ceilings):
    if arm == "CONTROL":
        return ctrl_loss(pred, fitness, gate)
    if arm in ["CTXZ_FB", "CTXZ"]:
        return normalized_loss(pred, fitness, gate, epsilon, ceilings)
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
