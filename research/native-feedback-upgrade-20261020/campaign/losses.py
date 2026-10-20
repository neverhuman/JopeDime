"""Acquired-target losses only; no outcome lookup, model, optimizer or device setup."""
import torch
from torch.nn import functional as F


def context_loss(pred, quality, mask, concept):
    if pred.shape != quality.shape or pred.shape != mask.shape or pred.ndim != 2:
        raise ValueError('matching BxM tensors required')
    if mask.dtype != torch.bool or not (mask.sum(1) >= 2).all():
        raise ValueError('at least two supported members/context required')
    if not torch.isfinite(pred).all() or not torch.isfinite(quality[mask]).all():
        raise ValueError('nonfinite acquired value/prediction')
    if concept not in ('composite_control', 'top_listwise', 'bootstrap_thompson'):
        raise ValueError('unknown objective')
    # Padded qualities must not affect rank/group predicates.
    quality = torch.where(mask, quality, torch.zeros_like(quality))
    diff = quality[:, :, None] - quality[:, None, :]
    pairmask = (diff > 1e-8) & mask[:, :, None] & mask[:, None, :]
    count = mask.sum(1)
    ranks = (((diff > 1e-8) & mask[:, None, :]).sum(2) +
             .5 * ((diff.abs() <= 1e-8) & mask[:, None, :]).sum(2)) / count[:, None]
    robust = (F.huber_loss(pred.sigmoid(), ranks, reduction='none') * mask).sum(1) / count
    if concept == 'top_listwise':
        higher = (diff < -1e-8) & mask[:, None, :]
        equal = (diff.abs() <= 1e-8) & mask[:, None, :]
        suffix = (diff >= -1e-8) & mask[:, None, :]
        # Never differentiate all-masked logsumexp at a padded anchor.
        equal = equal | (~mask[:, :, None] & mask[:, None, :])
        suffix = suffix | (~mask[:, :, None] & mask[:, None, :])
        expanded = pred[:, None, :].expand_as(diff)
        group_log = expanded.masked_fill(~equal, -torch.inf).logsumexp(2)
        suffix_log = expanded.masked_fill(~suffix, -torch.inf).logsumexp(2)
        supported = mask & ((diff > 1e-8) & mask[:, None, :]).any(2)
        terms = torch.where(supported, suffix_log - group_log, torch.zeros_like(pred))
        weights = supported / torch.log2(higher.sum(2).float() + 2) / equal.sum(2).clamp_min(1)
        return (terms * weights).sum(1) / weights.sum(1).clamp_min(1) + .1 * robust
    pair = F.softplus(pred[:, None, :] - pred[:, :, None])
    pairing = (pair * pairmask).sum((1, 2)) / pairmask.sum((1, 2)).clamp_min(1)
    target = (ranks / .25).masked_fill(~mask, -torch.inf).softmax(1)
    logp = pred.masked_fill(~mask, -torch.inf).log_softmax(1)
    listing = -(target * torch.where(mask, logp, torch.zeros_like(logp))).sum(1)
    return .5 * pairing + .5 * listing + .25 * robust
