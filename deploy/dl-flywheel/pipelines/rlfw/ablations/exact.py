"""Exact J_mean and J_B on a fully labelled finite support.

J_B is E[max(h, q_1..q_B) - h] for B i.i.d. draws. A deterministic search
result is the realized best-of-budget, not that expectation.
"""

from __future__ import annotations

import math


def _as_distribution(values, probs):
    if not values or len(values) != len(probs):
        raise ValueError("values and probabilities must align")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("finite values required")
    if any((not math.isfinite(p)) or p < 0 for p in probs):
        raise ValueError("probabilities must be finite and nonnegative")
    total = math.fsum(probs)
    if total <= 0 or abs(total - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")
    return values, probs


def j_mean(values, probs):
    values, probs = _as_distribution(values, probs)
    return math.fsum(v * p for v, p in zip(values, probs))


def expectation_of_max(values, probs, draws):
    """E[max of ``draws`` i.i.d. samples]."""
    values, probs = _as_distribution(values, probs)
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive int")
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    cdf = 0.0
    previous = 0.0
    expectation = 0.0
    for index in order:
        cdf += probs[index]
        cdf_b = cdf ** draws
        expectation += values[index] * (cdf_b - previous)
        previous = cdf_b
    return expectation


def j_b(values, probs, incumbent, draws):
    """E[max(incumbent, max of B draws) - incumbent]."""
    if not math.isfinite(incumbent):
        raise ValueError("finite incumbent required")
    lifted = [max(v, incumbent) for v in values]
    return expectation_of_max(lifted, probs, draws) - incumbent


def realized_regret(values, incumbent):
    """max(incumbent, max of the charged programs) - incumbent."""
    if not values:
        raise ValueError("empty selection")
    if any(not math.isfinite(v) for v in values) or not math.isfinite(incumbent):
        raise ValueError("finite scores required")
    return max(incumbent, max(values)) - incumbent


def effective_support(probs):
    """exp(entropy). Uniform on k atoms has support k."""
    _, probs = _as_distribution([0.0] * len(probs), probs)
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob)
    return math.exp(entropy)


def kl_divergence(probs, anchor):
    """KL(probs || anchor). Both are distributions on the same atoms."""
    _, probs = _as_distribution([0.0] * len(probs), probs)
    _, anchor = _as_distribution([0.0] * len(anchor), anchor)
    if len(probs) != len(anchor):
        raise ValueError("KL requires a shared support")
    total = 0.0
    for prob, base in zip(probs, anchor):
        if prob == 0:
            continue
        if base <= 0:
            raise ValueError("anchor gives zero mass to a positive atom")
        total += prob * math.log(prob / base)
    return total
