"""H238: all arms use the exact H170 CTRL AdamW; only LR/update budget differ."""

import math
import torch

ARMS = ("CONTROL", "LR3E5", "LR1E5_120")
LEARNING_RATES = dict(CONTROL=1e-4, LR3E5=3e-5, LR1E5_120=1e-5)
UPDATES = dict(CONTROL=40, LR3E5=40, LR1E5_120=120)
CONTROL_WD = 1e-4


def make_optimizer(arm, params):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return torch.optim.AdamW(
        params, lr=LEARNING_RATES[arm], betas=(0.9, 0.999), weight_decay=CONTROL_WD
    )


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= UPDATES[arm]:
        raise ValueError("unknown arm or update")
    return LEARNING_RATES[arm], 0.9, 0.999, CONTROL_WD


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    if arm not in ARMS:
        raise ValueError("unknown arm")
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
