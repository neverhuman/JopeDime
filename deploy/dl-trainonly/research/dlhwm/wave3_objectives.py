"""Fixed wave3 objectives. Missing targets are masked before arithmetic."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def lineage_weights(groups, valid):
    """Weights have mean one on the full head support, never on a random batch."""
    out = np.zeros(len(groups), np.float32)
    _, inv, counts = np.unique(np.asarray(groups)[valid], return_inverse=True, return_counts=True)
    if len(counts):
        out[valid] = valid.sum() / len(counts) / counts[inv]
    return out


def rank_loss(mu, y, generator=None, pairs=2048):
    valid = torch.isfinite(y)
    if int(valid.sum()) < 4:
        return mu[:0].sum()
    mu, y = mu[valid], y[valid]
    i = torch.randint(len(y), (pairs,), device=y.device, generator=generator)
    j = torch.randint(len(y), (pairs,), device=y.device, generator=generator)
    sign = torch.sign(y[i] - y[j]); keep = sign != 0
    return F.softplus(-sign[keep] * (mu[i] - mu[j])[keep]).mean() if keep.any() else mu[:0].sum()


def objective(outputs, targets, spec, weights=None, generator=None):
    parts, supports = {}, {}
    for head, key in [('fitness', 'fit_mu'), ('gap', 'gap_mu'), ('beat10', 'b10')]:
        mu = outputs[key].float(); y = targets[head].float(); valid = torch.isfinite(y)
        supports[head] = int(valid.sum())
        if not spec['head_weights'][head] or not valid.any():
            parts[head] = mu[:0].sum()
            continue
        pred, yy = mu[valid], y[valid]
        if head == 'beat10':
            smooth = spec['label_smoothing']
            loss = F.binary_cross_entropy_with_logits(pred, yy * (1 - smooth) + smooth / 2, reduction='none')
        elif spec['regression_loss'] == 'mse':
            loss = (pred - yy).square()
        elif spec['regression_loss'] == 'huber':
            loss = spec['huber_scale'] * F.huber_loss(pred, yy, delta=spec['huber_delta'], reduction='none')
        elif spec['regression_loss'] == 'beta_nll':
            ls = outputs['fit_ls' if head == 'fitness' else 'gap_ls'].float()[valid]
            var = ls.exp()
            loss = .5 * (ls + (pred - yy).square() / var) * var.detach().pow(spec['beta'])
        else:
            raise ValueError('unknown objective')
        if weights is not None:
            loss = loss * weights[head][valid]
        parts[head] = loss.mean()
    parts['rank'] = rank_loss(outputs['fit_mu'].float(), targets['fitness'], generator) if spec['head_weights']['rank'] else outputs['fit_mu'][:0].sum()
    total = sum(spec['head_weights'][h] * parts[h] for h in parts)
    return total, parts, supports


def metrics(pred, targets, groups, spec):
    """Full-panel metrics; each lineage has total weight one for macro MSE/R²."""
    result = {'supports': {}}
    for head, col in [('fitness', 0), ('gap', 2)]:
        y = targets[head]; p = pred[:, col]
        mask = np.isfinite(y) & np.isfinite(p)
        result['supports'][head] = int(mask.sum())
        if head not in spec['active_heads'] or not mask.any():
            result[head + '_mse'] = result[head + '_macro_mse'] = result[head + '_r2'] = None
            continue
        yy, pp = y[mask].astype(np.float64), p[mask].astype(np.float64)
        w = lineage_weights(groups, mask)[mask].astype(np.float64)
        e2 = (yy - pp) ** 2
        result[head + '_mse'] = float(e2.mean())
        result[head + '_macro_mse'] = float(np.dot(w, e2) / w.sum())
        sst = ((yy - yy.mean()) ** 2).sum()
        result[head + '_r2'] = float(1 - e2.sum() / sst) if sst > 0 else None
    mask = np.isfinite(targets['beat10']) & np.isfinite(pred[:, 4])
    result['supports']['beat10'] = int(mask.sum())
    result['beat10_auc'] = None
    if 'beat10' in spec['active_heads'] and mask.any():
        from dlhwm.train import auc
        yy = targets['beat10'][mask]
        if len(np.unique(yy)) == 2:
            result['beat10_auc'] = auc(yy, pred[mask, 4])
    return result


def validate_spec(spec):
    for key, value in spec['head_weights'].items():
        if key not in ('fitness', 'gap', 'beat10', 'rank') or not math.isfinite(value) or value < 0:
            raise ValueError('invalid head weight')
    if not 0 <= spec['label_smoothing'] < 1 or not 0 <= spec['beta'] <= 1 or not 0 < spec['ema'] < 1:
        raise ValueError('invalid smoothing/beta/EMA')
    if not math.isfinite(spec['min_delta']) or spec['min_delta'] < 0:
        raise ValueError('invalid min_delta')
    if set(spec['head_weights']) != {'fitness', 'gap', 'beat10', 'rank'}:
        raise ValueError('missing head weights')
    cfg = spec['cfg']
    for name in ('lr', 'wd'):
        if not math.isfinite(cfg[name]) or cfg[name] < 0 or name == 'lr' and cfg[name] == 0:
            raise ValueError('invalid ' + name)
    for name in ('epochs', 'patience', 'bs'):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1:
            raise ValueError('invalid ' + name)
    if spec['regression_loss'] not in ('mse', 'huber', 'beta_nll'):
        raise ValueError('unknown regression objective')
    if spec['checkpoint'] not in ('lineage_macro_fitness_mse', 'fitness_r2_plus_beat10_auc'):
        raise ValueError('unknown checkpoint')
    if spec['sample_weight'] not in ('uniform', 'lineage'):
        raise ValueError('unknown sample weighting')
    for name in ('huber_delta', 'huber_scale'):
        if not math.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError(name)
    active = [h for h in ('fitness', 'gap', 'beat10') if spec['head_weights'][h] > 0]
    if active != spec['active_heads'] or 'fitness' not in active:
        raise ValueError('head availability mismatch')
    if spec['uncertainty_trained'] != (spec['regression_loss'] == 'beta_nll'):
        raise ValueError('uncertainty mismatch')
    if spec['sample_weight'] == 'lineage' and spec['head_weights']['rank']:
        raise ValueError('lineage-weighted ranking is undefined')
