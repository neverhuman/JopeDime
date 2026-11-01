"""H196 zero-update treatment primitives."""

import hashlib

import numpy as np
import torch


def stream_seed(*parts):
    text = ":".join(map(str, ("H196", *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def fit_scale(fit_ids, tokens, masks):
    """Population standard deviation per active token-position/channel."""
    ids = sorted(fit_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique FIT IDs required")
    x = np.stack([np.asarray(tokens[c], np.float64) for c in ids])
    active = np.stack([np.asarray(masks[c], bool) for c in ids])
    if x.shape != (len(ids), 10, 32) or active.shape != (len(ids), 10):
        raise ValueError("10x32 tokens and 10 masks required")
    if not np.isfinite(x).all():
        raise ValueError("finite features required")
    result = np.zeros((10, 32), np.float32)
    for slot in range(10):
        observed = x[active[:, slot], slot]
        if len(observed) >= 2:
            result[slot] = observed.std(axis=0, ddof=0).astype(np.float32)
    return result


def noisy_tokens(tokens, masks, scale, seed, cell, cid, draw):
    """Noise stream depends only on identities, never batch shape/order."""
    x, active = np.asarray(tokens, np.float32), np.asarray(masks, bool)
    s = np.asarray(scale, np.float32)
    if x.shape != (10, 32) or active.shape != (10,) or s.shape != x.shape:
        raise ValueError("invalid feature, mask, or scale shape")
    if not np.isfinite(s).all() or (s < 0).any() or draw not in range(16):
        raise ValueError("invalid scale/draw")
    noise = np.random.default_rng(stream_seed("noise", seed, cell, cid, draw))
    perturb = noise.standard_normal(x.shape).astype(np.float32)
    perturb *= s
    perturb *= np.float32(0.05)
    result = x.copy()
    result[active] += perturb[active]
    return result


def permutations(seed, cid, size):
    if size < 4:
        raise ValueError("four revealed candidates required")
    return [
        np.random.default_rng(stream_seed("perm", seed, cid, draw)).permutation(size)
        for draw in range(8)
    ]


def permute_kwargs(kwargs, order):
    """Order is [batch, candidate]; reveals preserve original history order."""
    p = torch.as_tensor(order, dtype=torch.int64)
    if p.ndim != 2 or p.shape != kwargs["base_prior_cost"].shape:
        raise ValueError("permutation batch shape")
    expected = torch.arange(p.shape[1]).expand_as(p)
    if not torch.equal(p.sort(dim=1).values, expected):
        raise ValueError("not a permutation")
    inverse = p.argsort(dim=1)
    result = dict(kwargs)
    for key in ("prog_412", "proj_16", "base_prior_cost", "sos_prior"):
        x = kwargs[key]
        index = p.reshape(*p.shape, *((1,) * (x.ndim - 2))).expand_as(x)
        result[key] = x.gather(1, index)
    result["revealed_indices"] = inverse.gather(1, kwargs["revealed_indices"])
    return result, inverse


def averaged_pick(draw_costs):
    values = np.asarray(draw_costs)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("finite draw-by-candidate array required")
    means = values.mean(axis=0, dtype=np.float64)
    return int(np.argmin(means)), means


def admission(ci95, holm_p, harm):
    values = np.asarray([*ci95, holm_p, harm], np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("complete finite decision required")
    if ci95[0] > ci95[1] or not 0 <= holm_p <= 1 or not 0 <= harm <= 1:
        raise ValueError("invalid decision bounds")
    return bool(ci95[0] > 0 and holm_p < 0.05 and harm <= 0.10)
