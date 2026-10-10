"""Power of a Lane B lineage-mean comparison.

The sample-size factor 7.84 is the spec's rounded two-sided normal factor for
alpha 0.05 and power 0.80 (§7.4), not a measurement from this host. An effect
of 0.2 sigma needs 196 families and 0.1 sigma needs 784. The dev panel L=54
can detect about 0.381 sigma and stays directional.

Observed judge-partition noise is unmeasured. D5's paired arena is Lane A and
INCONCLUSIVE (bench REPORT_20261009.md:349); it is not a sigma_D for G^judge.
"""

import math

SPEC_POWER_FACTOR = 7.84
Z_ALPHA_95 = 1.959963984540054  # Phi^{-1}(0.975), a normal constant, not a measurement.
SEALED_FAMILIES = 200
SEALED_CONTEXTS = 400

UNMEASURED_JUDGE_NOISE = {
    "name": "observed_judge_sigma_D",
    "sigma_D": None,
    "status": "unmeasured",
    "sigma_source": "unmeasured",
    "reason": (
        "No Lane B judge scores exist. The D5 paired arena is Lane A and "
        "INCONCLUSIVE, so it is not a judge-partition sigma."
    ),
    "citation": (
        "/home/ubuntu/jopedime_probe_exp/rlfw-20261009/REPORT_20261009.md:349"
    ),
}


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def families_needed(sigma, delta, factor=SPEC_POWER_FACTOR):
    if sigma <= 0 or delta == 0 or not math.isfinite(sigma) or not math.isfinite(delta):
        raise ValueError("sigma must be positive and delta nonzero")
    return factor * sigma * sigma / (delta * delta)


def minimum_detectable_effect(lineages, sigma=1.0, factor=SPEC_POWER_FACTOR):
    if lineages <= 0 or sigma < 0:
        raise ValueError("lineages and sigma must be positive")
    return sigma * math.sqrt(factor / lineages)


def comparison_power(lineages, sigma, delta, z_alpha=Z_ALPHA_95):
    """Two-sided normal power of a lineage-mean test. Inputs are caller-supplied."""
    if lineages <= 0 or sigma <= 0 or not math.isfinite(delta):
        raise ValueError("need a positive lineage count and sigma")
    z = abs(delta) / (sigma / math.sqrt(lineages))
    return (1.0 - norm_cdf(z_alpha - z)) + norm_cdf(-z_alpha - z)


def _hypothetical(name, lineages, delta_over_sigma):
    """Unit-sigma rows. delta_over_sigma is a hypothesis, not an observed effect."""
    sigma = 1.0
    delta = delta_over_sigma * sigma
    needed = families_needed(sigma, delta)
    return {
        "name": name,
        "sigma_source": "hypothetical_unit_sigma",
        "sigma_D": sigma,
        "delta": delta,
        "delta_in_sigma": delta_over_sigma,
        "lineages": lineages,
        "families_needed_for_power_80": needed,
        "power": None if lineages is None else comparison_power(lineages, sigma, delta),
        "mde_80": None if lineages is None else minimum_detectable_effect(lineages, sigma),
        "measured": False,
    }


def publish_document():
    """The committed table. Hypothetical rows are not measurements."""
    return {
        "alpha": 0.05,
        "factor": SPEC_POWER_FACTOR,
        "measured_judge_sigma": False,
        "power_target": 0.80,
        "rows": table_rows(),
        "schema": "jope.rlfw-judge-power/v1",
    }


def table_rows():
    rows = [
        _hypothetical("effect_0.2_sigma", None, 0.2),
        _hypothetical("effect_0.1_sigma", None, 0.1),
        _hypothetical("dev_panel_L54", 54, 0.2),
        _hypothetical("smoke_L10", 10, 0.2),
        _hypothetical("week1_L50", 50, 0.2),
        _hypothetical("internal_L100", 100, 0.2),
        _hypothetical("sealed_floor_L200", 200, 0.2),
        _hypothetical("sealed_floor_L200_effect_0.1_sigma", 200, 0.1),
        _hypothetical("sealed_L400", 400, 0.2),
    ]
    rows.append(dict(UNMEASURED_JUDGE_NOISE))
    return rows


def tier_for_counts(n_families, n_contexts):
    if n_families >= SEALED_FAMILIES and n_contexts >= SEALED_CONTEXTS:
        return "sealed_count_floor"
    if n_families >= 100 and n_contexts >= 200:
        return "internal_promotion"
    if n_families >= 50 and n_contexts >= 100:
        return "week1_directional"
    if n_families >= 10 and n_contexts >= 20:
        return "smoke_mechanics_only"
    return "below_smoke"


def claim_floor(n_families, n_contexts, power_required_families):
    """Sealed claims need the count floor, and more families when power says so."""
    if power_required_families is None:
        return False, "power requirement unmeasured"
    needed = max(SEALED_FAMILIES, math.ceil(power_required_families - 1e-12))
    if n_families < needed or n_contexts < SEALED_CONTEXTS:
        return False, f"need at least {needed} families and {SEALED_CONTEXTS} contexts"
    return True, "count floor met"
