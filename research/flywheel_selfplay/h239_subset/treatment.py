"""H239 parameter subsets; exact H170 AdamW with frozen explicit tensor membership."""

import json
import math
from pathlib import Path
import torch

ARMS = ("CONTROL", "BIAS_ONLY", "HEAD_ONLY", "LLRD")
CONTROL_LR, CONTROL_WD = 1e-4, 1e-4
INVENTORY = json.loads((Path(__file__).with_name("trainable_tensors.json")).read_text())


def make_optimizer(arm, named):
    if arm not in ARMS or set(named) != set(INVENTORY["arms"]["CONTROL"]["names"]):
        raise ValueError("arm or exact adapter inventory")
    allowed = set(INVENTORY["arms"][arm]["names"])
    for name, parameter in named.items():
        parameter.grad = None
        parameter.requires_grad_(name in allowed)
    if arm == "LLRD":
        depths = {r["name"]: r["llrd_multiplier"] for r in INVENTORY["tensors"]}
        params = [
            {"params": [p], "lr": CONTROL_LR * depths[n], "name": n}
            for n, p in named.items()
        ]
    else:
        params = [p for n, p in named.items() if n in allowed]
    return torch.optim.AdamW(
        params, lr=CONTROL_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=CONTROL_WD
    )


def check_subset(arm, named, initial):
    allowed = set(INVENTORY["arms"][arm]["names"])
    for n, p in named.items():
        if p.requires_grad != (n in allowed):
            raise RuntimeError("gradient membership changed: " + n)
        if n not in allowed and (
            p.grad is not None or not torch.equal(p.detach(), initial[n])
        ):
            raise RuntimeError("frozen subset drift: " + n)


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("arm or step")
    return CONTROL_LR, 0.9, 0.999, CONTROL_WD


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    if arm not in ARMS:
        raise ValueError("arm")
    return ctrl_loss(pred, fitness, gate)


def gradient_norm(params):
    """None contributes zero to diagnostics; native optimizer None semantics are retained."""
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
    return_hidden=False,
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
    pred = (anchor_total + delta).float()
    # H235: the per-candidate query embedding feeds the log-variance head; the mean prediction is unchanged.
    return (pred, query) if return_hidden else pred


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
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
