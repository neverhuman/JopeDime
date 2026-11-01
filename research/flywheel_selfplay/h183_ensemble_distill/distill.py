"""FIT-only ENS10 soft targets and paired fresh-student objective."""

import numpy as np
import torch
import torch.nn.functional as F

TEMPERATURE = 0.2


def require_fit(ids, fit_ids, excluded_ids):
    fit, excluded = set(fit_ids), set(excluded_ids)
    if fit & excluded or not set(ids) <= fit or set(ids) & excluded:
        raise ValueError("teacher/student target or batch outside FIT")


def teacher_distribution(member_costs, temperature=TEMPERATURE):
    values = np.asarray(member_costs, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] != 10 or not np.isfinite(values).all():
        raise ValueError("ten finite teacher cost panels required")
    # ENS10 is a mean-cost policy: average scores first, then normalize its logits.
    mean_cost = np.mean(values, axis=0, dtype=np.float32)
    probability = torch.softmax(-torch.from_numpy(mean_cost) / temperature, dim=-1)
    return mean_cost, probability.detach()


def student_objective(prediction, fitness, gate, rt_loss, teacher_prob=None):
    live = rt_loss(prediction, fitness, gate)
    if teacher_prob is None:
        return live, live, prediction.new_tensor(0.0)
    if teacher_prob.shape != prediction.shape or teacher_prob.requires_grad:
        raise ValueError("fixed teacher target shape/gradient mismatch")
    kl = F.kl_div(
        F.log_softmax(-prediction / TEMPERATURE, dim=-1),
        teacher_prob,
        reduction="batchmean",
    )
    return 0.5 * live + 0.5 * kl, live, kl


def validate_initial_hashes(teachers, student):
    if len(teachers) != 10 or len(set(teachers)) != 10 or student in teachers:
        raise ValueError(
            "distinct teacher initializations and separate student seed required"
        )
