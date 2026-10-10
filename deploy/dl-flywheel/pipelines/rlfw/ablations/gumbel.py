"""Gumbel-Top-k without replacement. No value head and no sequential halving."""

from __future__ import annotations

import math
import random

GUMBEL_SEED = 7
_U_LO = 1e-12
_U_HI = 1.0 - 1e-12


def gumbel_topk(priors, k, seed=GUMBEL_SEED):
    """Return k distinct indexes. The score is log prior plus a seeded Gumbel."""
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    if type(k) is not int or k < 1 or k > len(priors):
        raise ValueError("k must lie in 1..len(priors)")
    if any((not math.isfinite(p)) or p <= 0 for p in priors):
        raise ValueError("priors must be finite and positive")
    if abs(math.fsum(priors) - 1.0) > 1e-9:
        raise ValueError("priors must sum to 1")
    rng = random.Random(seed)
    scored = []
    for index, prior in enumerate(priors):
        u = min(max(rng.random(), _U_LO), _U_HI)
        gumbel = -math.log(-math.log(u))
        scored.append((math.log(prior) + gumbel, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [index for _, index in scored[:k]]
    if len(set(chosen)) != k:
        raise RuntimeError("Gumbel-Top-k repeated an index")
    return chosen
