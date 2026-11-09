"""H259 exact H170 scalar loss and precision-controlled AdamW, with unchanged cached adapter forward."""

import math
import torch
from precision import PrecisionAdamW

ARMS = ("CONTROL", "BF16_AMP", "BF16_SR")


def make_optimizer(arm, params, names=None):
    return PrecisionAdamW(arm, params, names)


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("arm/update")
    return dict(
        optimizer="AdamW",
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        forward="fp32" if arm == "CONTROL" else "cpu_bf16_autocast",
        weights="bf16_SR" if arm == "BF16_SR" else "fp32_master",
        state="fp32",
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    if arm not in ARMS:
        raise ValueError("arm")
    return ctrl_loss(pred.float(), fitness.float(), gate)


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
    if smoke:
        raise RuntimeError("smoke is in-process after Native91")
    if not (
        launch.get("published_prereg_verified")
        and launch.get("head_review_verified")
        and launch.get("source_commit") == head
        and launch.get("manifest_sha256") == manifest_sha256
        and launch.get("prereg_sha256") == prereg_sha256
        and launch.get("fold_manifest_sha256") == fold_sha256
        and launch.get("prereg_published_at", "9999") < "2026-11-03T15:15:00Z"
    ):
        raise RuntimeError("exact published/reviewed prereg launch binding")
