"""Self-normalized IS on uniform-trie audit logs. ESS below 100 is not a value.

This is a pre-screen. ``gate`` stays false: OPE does not ship a checkpoint.
Historical arms with unknown behaviour propensity are not importance-sampled.
"""

from __future__ import annotations

import math

MIN_ESS = 100
UNIFORM_ARM = "uniform_trie"
PROPENSITY_SOURCE = "arm_definition_uniform_trie"


def ess(weights):
    if not weights or any((not math.isfinite(w)) or w < 0 for w in weights):
        raise ValueError("weights must be finite and nonnegative")
    total = math.fsum(weights)
    squares = math.fsum(w * w for w in weights)
    if total <= 0 or squares <= 0:
        raise ValueError("ESS needs positive weight")
    return (total * total) / squares


def self_normalized_is(rewards, weights, min_ess=MIN_ESS):
    """SNIS of the mean reward. Below ``min_ess`` the value is null."""
    if min_ess != MIN_ESS:
        raise ValueError("ESS threshold is 100")
    if len(rewards) != len(weights) or not rewards:
        raise ValueError("rewards and weights must align and be nonempty")
    if any(not math.isfinite(r) for r in rewards):
        raise ValueError("finite rewards required")
    sample_ess = ess(weights)
    report = {
        "ess": sample_ess,
        "min_ess": MIN_ESS,
        "n": len(rewards),
        "gate": False,
        "estimator": "SNIS_J_mean",
    }
    if sample_ess < MIN_ESS:
        report["value"] = None
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = "ess_below_100"
        return report
    report["value"] = math.fsum(w * r for w, r in zip(weights, rewards)) / math.fsum(
        weights
    )
    report["verdict"] = "REPORTED"
    report["reason"] = None
    return report


def uniform_trie_ope(draws, catalog_size, min_ess=MIN_ESS):
    """Weight pi/mu only for uniform_trie draws whose mu is 1/catalog_size.

    ``pi`` is the anchor trie probability. A missing pi, a non-uniform arm, or
    a null propensity stays out of the estimator.
    """
    if type(catalog_size) is not int or catalog_size < 1:
        raise ValueError("catalog_size must be a positive int")
    mu_expected = 1.0 / catalog_size
    usable_rewards = []
    usable_weights = []
    dropped = {
        "non_randomized": 0,
        "missing_mu": 0,
        "missing_pi": 0,
        "mu_mismatch": 0,
        "not_in_catalog": 0,
    }
    for draw in draws:
        arm = draw.get("arm")
        if arm != UNIFORM_ARM or draw.get("propensity_source") != PROPENSITY_SOURCE:
            dropped["non_randomized"] += 1
            continue
        mu = draw.get("mu")
        if mu is None:
            dropped["missing_mu"] += 1
            continue
        if not math.isfinite(mu) or abs(mu - mu_expected) > 1e-15:
            dropped["mu_mismatch"] += 1
            continue
        if draw.get("in_catalog") is False:
            dropped["not_in_catalog"] += 1
            continue
        reward = draw.get("reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("OPE reward must be measured q")
        log_pi = draw.get("log_pi")
        if log_pi is None and draw.get("pi") is not None:
            pi = draw.get("pi")
            if not math.isfinite(pi) or pi <= 0:
                raise ValueError("anchor probability must be finite and positive")
            log_pi = math.log(pi)
        if log_pi is None:
            dropped["missing_pi"] += 1
            continue
        if not math.isfinite(log_pi):
            raise ValueError("anchor log-probability must be finite")
        usable_rewards.append(float(reward))
        usable_weights.append(log_pi - math.log(mu))
    report = {
        "dropped": dropped,
        "used": len(usable_rewards),
        "catalog_size": catalog_size,
        "mu": mu_expected,
        "propensity_source": PROPENSITY_SOURCE,
        "gate": False,
    }
    if not usable_rewards:
        report.update(
            value=None,
            ess=None,
            verdict="INCONCLUSIVE",
            reason="no_usable_uniform_trie_draws",
            estimator="SNIS_J_mean",
            min_ess=MIN_ESS,
            n=0,
        )
        return report
    peak = max(usable_weights)
    # SNIS and ESS are invariant to scaling. Shift so exp does not underflow.
    scaled = [math.exp(weight - peak) for weight in usable_weights]
    report.update(self_normalized_is(usable_rewards, scaled, min_ess))
    return report
