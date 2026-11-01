"""H170 fixed one-cycle optimizer settings and None-safe diagnostics."""

import math
import torch

ARMS = ("CTRL", "OC3", "OC15", "CTRL_MEANMATCH")


def cycle(step, peak):
    """Applied update t=1..40: 12 inclusive warmup endpoints, 28 cosine-tail updates."""
    if not 1 <= step <= 40 or peak not in (3e-4, 1.5e-4):
        raise ValueError("fixed H170 schedule only")
    if step <= 12:
        q = (step - 1) / 11
        return 1e-5 + (peak - 1e-5) * q, 0.95 - 0.10 * q
    q = (step - 12) / 28
    c = 0.5 * (1 + math.cos(math.pi * q))
    return 1e-6 + (peak - 1e-6) * c, 0.95 - 0.10 * c


MEANMATCH_LR = math.fsum(cycle(t, 3e-4)[0] for t in range(1, 41)) / 40


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    if arm == "CTRL":
        return 1e-4, 0.9
    if arm == "CTRL_MEANMATCH":
        return MEANMATCH_LR, 0.9
    return cycle(step, 3e-4 if arm == "OC3" else 1.5e-4)


def apply_settings(optimizer, arm, step):
    lr, beta1 = settings(arm, step)
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["betas"] = (beta1, 0.999)
    return lr, beta1


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
