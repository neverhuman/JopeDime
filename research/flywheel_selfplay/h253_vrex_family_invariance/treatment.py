"""H253 V-REx: unchanged H170 CTRL loss and AdamW, plus population variance of four FIT-family risks."""

import math
import torch
import torch.nn.functional as F

ARMS = ("CONTROL", "VREX_B1", "VREX_B10")
CONTROL_LR, CONTROL_WD, BETAS = 1e-4, 1e-4, (0.9, 0.999)
BETA_MULTIPLIER = {"CONTROL": 0.0, "VREX_B1": 1.0, "VREX_B10": 10.0}
VARIANCE_FLOOR = 1e-12


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
    return dict(
        optimizer="AdamW",
        lr=CONTROL_LR,
        beta1=BETAS[0],
        beta2=BETAS[1],
        weight_decay=CONTROL_WD,
        beta_multiplier=BETA_MULTIPLIER[arm],
    )


def candidate_losses(pred, fitness, gate):
    """Candidate contributions whose mean is the exact H170 context loss (KL sum preserved)."""
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return F.smooth_l1_loss(pred, target, reduction="none") + 0.1 * pred.shape[
        -1
    ] * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    )


def beta_scale(context_variances):
    import numpy as np

    values = np.asarray(context_variances, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("finite nonnegative FIT-only variance inventory required")
    median = float(np.median(values))
    return dict(
        median_context_variance=median,
        variance_floor=VARIANCE_FLOOR,
        beta_scale=1.0 / max(median, VARIANCE_FLOOR),
        contexts=len(values),
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate, families, scale):
    if arm not in ARMS or len(families) != len(pred):
        raise ValueError("arm/family inventory")
    ids = sorted(set(families))
    if len(ids) != 4 or any(families.count(f) != 2 for f in ids):
        raise ValueError("exact four families with two contexts each required")
    per_context = candidate_losses(pred, fitness, gate).mean(dim=1)
    risks = torch.stack(
        [
            per_context[[i for i, value in enumerate(families) if value == f]].mean()
            for f in ids
        ]
    )
    variance = risks.var(unbiased=False)
    base = ctrl_loss(pred, fitness, gate)
    beta = BETA_MULTIPLIER[arm] * float(scale)
    objective = base if arm == "CONTROL" else base + beta * variance
    return objective, dict(
        base_loss=float(base.detach()),
        family_variance=float(variance.detach()),
        beta=beta,
        family_ids=ids,
        family_losses=risks.detach().tolist(),
        context_losses=per_context.detach().tolist(),
    )


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
    # Owner t1250 permits local work while main is held. The sole allowed unmerged transport fix
    # must retain the merged scientific preregistration and prove zero prior model/native/update calls.
    if not launch.get("merged_source_verified"):
        if not (
            launch.get("merged_prereg_verified")
            and launch.get("preregistered_protocol_commit")
            == "144f045c585000b1eb7955eeb1fbf1eaf16b4e6f"
            and launch.get("execution_source_reviewed")
            and launch.get("configuration_only_amendment")
            and launch.get("zero_prior_model_native_optimizer_calls")
            and launch.get("scientific_AST_parity_verified")
            and launch.get("main_hold")
        ):
            raise RuntimeError(
                "unmerged transport requires merged protocol and exact-head pre-native proof"
            )
    if (
        not (
            launch.get("merged_source_verified") or launch.get("merged_prereg_verified")
        )
        or launch.get("source_commit") != head
        or type(launch.get("review_id")) is not int
        or launch["review_id"] <= 0
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
