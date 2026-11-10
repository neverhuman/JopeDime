"""H297 losses: cost-target normalization; failed utility -2 means cost +2."""

import torch
import torch.nn.functional as F


def normalize_targets(target, smoothing=0.05):
    median = torch.quantile(target, 0.5, dim=-1, keepdim=True)
    q1 = torch.quantile(target, 0.25, dim=-1, keepdim=True)
    q3 = torch.quantile(target, 0.75, dim=-1, keepdim=True)
    normalized = ((target - median) / (q3 - q1 + 1e-3)).clamp(-3, 3)
    return (1 - smoothing) * normalized + smoothing * normalized.mean(
        dim=-1, keepdim=True
    )


def training_loss(pred, target, arm):
    if arm == "CTXNORM_CLIP":
        target = normalize_targets(target, 0.05)
    elif arm == "CTXNORM_CLIP_NOSMOOTH":
        target = normalize_targets(target, 0.0)
    elif arm != "CONTROL":
        raise ValueError(arm)
    return F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="batchmean",
    )
