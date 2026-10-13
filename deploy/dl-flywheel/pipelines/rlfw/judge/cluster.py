"""Parent-cluster paired intervals and a hierarchical power simulation.

Context-level intervals treat each context as independent. Cuts that share a
parent are one cluster. The percentile interval resamples parents with
replacement. Neither interval is a promotion, and neither spends a seal.
"""

import math
import random

from rlfw.judge.power import Z_ALPHA_95


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def context_interval(values, z=Z_ALPHA_95):
    """Normal interval for the mean. SE is the sample SD with ddof=1, over sqrt(n)."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least two paired contexts")
    if any(not _finite_number(v) for v in values) or not _finite_number(z):
        raise ValueError("paired differences must be finite")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return {
        "n": n,
        "mean": mean,
        "sd": math.sqrt(var),
        "se": se,
        "low": mean - z * se,
        "high": mean + z * se,
    }


def cluster_bootstrap_ci(groups, draws=2000, seed=20261011, alpha=0.05):
    """Percentile CI of the mean, resampling parent clusters with replacement.

    `groups` is a list of lists. A drawn parent contributes its contexts once
    per draw. The returned context interval is the independent-context interval
    on the same pooled values.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be positive")
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("every cluster needs at least one paired difference")
    if any(not _finite_number(v) for g in groups for v in g):
        raise ValueError("paired differences must be finite")
    if not _finite_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    pooled = [v for g in groups for v in g]
    sums = [sum(g) for g in groups]
    counts = [len(g) for g in groups]
    rng = random.Random(seed)
    g = len(groups)
    means = []
    for _ in range(draws):
        num = 0.0
        den = 0
        for _pick in range(g):
            i = rng.randrange(g)
            num += sums[i]
            den += counts[i]
        means.append(num / den)
    means.sort()
    lo_pos = (alpha / 2) * (draws - 1)
    hi_pos = (1.0 - alpha / 2) * (draws - 1)

    def at(pos):
        i = int(math.floor(pos))
        j = min(i + 1, draws - 1)
        w = pos - i
        return means[i] * (1.0 - w) + means[j] * w

    point = sum(pooled) / len(pooled)
    return {
        "n_contexts": len(pooled),
        "n_clusters": g,
        "cluster_sizes": counts,
        "mean": point,
        "context": context_interval(pooled),
        "cluster_bootstrap": {
            "draws": draws,
            "seed": seed,
            "alpha": alpha,
            "method": "percentile_parent_resample",
            "low": at(lo_pos),
            "high": at(hi_pos),
        },
    }


def variance_components(groups):
    """Unbalanced one-way moment estimates of within-parent and between-parent variance."""
    if len(groups) < 2 or any(len(g) < 1 for g in groups):
        raise ValueError("need at least two non-empty clusters")
    counts = [len(g) for g in groups]
    n = sum(counts)
    g = len(groups)
    if n == g:
        raise ValueError("within variance needs a cluster with two contexts")
    if any(not _finite_number(v) for g in groups for v in g):
        raise ValueError("paired differences must be finite")
    grand = sum(sum(group) for group in groups) / n
    sse = 0.0
    ssb = 0.0
    for group in groups:
        mean = sum(group) / len(group)
        sse += sum((v - mean) ** 2 for v in group)
        ssb += len(group) * (mean - grand) ** 2
    within = sse / (n - g)
    between_mean = ssb / (g - 1)
    n0 = (n - sum(c * c for c in counts) / n) / (g - 1)
    between = max(0.0, (between_mean - within) / n0)
    icc = between / (between + within) if between + within > 0 else 0.0
    return {
        "n_contexts": n,
        "n_clusters": g,
        "sigma_within": math.sqrt(within),
        "sigma_between": math.sqrt(between),
        "icc": icc,
        "n0": n0,
    }


def hierarchical_power(
    cluster_sizes,
    sigma_within,
    sigma_between,
    delta,
    sims=400,
    draws=1000,
    seed=20261011,
    alpha=0.05,
):
    """Rejection rate of a two-sided parent-cluster percentile interval.

    Each simulation draws parent effects and within-parent noise, then applies
    the same cluster bootstrap used on the observed series. The context-level
    rate uses the independent normal interval on the simulated contexts. This is
    a simulation at the supplied variance, not a measured verdict. Non-finite
    variances, effects, and invalid simulation parameters are refused before a
    rejection rate is computed.
    """
    if isinstance(sims, bool) or not isinstance(sims, int) or sims < 1:
        raise ValueError("sims and draws must be positive")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("sims and draws must be positive")
    if (
        not cluster_sizes
        or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in cluster_sizes)
    ):
        raise ValueError("cluster sizes must be positive")
    if any(not _finite_number(v) for v in (sigma_within, sigma_between, delta, alpha)):
        raise ValueError("power inputs must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if sigma_within < 0 or sigma_between < 0:
        raise ValueError("sigmas must be non-negative")
    rng = random.Random(seed)
    cluster_rejects = 0
    context_rejects = 0
    for sim in range(sims):
        groups = []
        for size in cluster_sizes:
            shift = delta + rng.gauss(0.0, sigma_between)
            groups.append([shift + rng.gauss(0.0, sigma_within) for _ in range(size)])
        report = cluster_bootstrap_ci(groups, draws=draws, seed=seed + sim + 1, alpha=alpha)
        band = report["cluster_bootstrap"]
        if band["high"] < 0.0 or band["low"] > 0.0:
            cluster_rejects += 1
        interval = report["context"]
        if interval["high"] < 0.0 or interval["low"] > 0.0:
            context_rejects += 1
    return {
        "sims": sims,
        "draws": draws,
        "seed": seed,
        "delta": delta,
        "sigma_within": sigma_within,
        "sigma_between": sigma_between,
        "n_clusters": len(cluster_sizes),
        "n_contexts": sum(cluster_sizes),
        "cluster_reject_rate": cluster_rejects / sims,
        "context_reject_rate": context_rejects / sims,
        "qualifying": False,
    }
