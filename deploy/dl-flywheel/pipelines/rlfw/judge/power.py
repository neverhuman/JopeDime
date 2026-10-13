"""Power of a Lane B lineage-mean comparison.

The sample-size factor 7.84 is the spec's rounded two-sided normal factor for
alpha 0.05 and power 0.80 (§7.4), not a measurement from this host. An effect
of 0.2 sigma needs 196 families and 0.1 sigma needs 784. The dev panel L=54
can detect about 0.381 sigma and stays directional.

Observed judge-partition noise is unmeasured. D5's paired arena is Lane A and
INCONCLUSIVE (bench REPORT_20261009.md:349); it is not a sigma_D for G^judge.

`families_needed` takes a paired standard deviation: the sample SD of one
independent paired difference. CX-MET `median_se_pair` 0.05296631946624709
(owner round 0.053) is `SE_pair = max(bootstrap_se, |d0-d1|/2)` on one
log-RMSE-ratio pair (`analytics/metrics.py:189-197`). That is a standard
error. n=882 is what the formula returns only when 0.053 is misread as the
paired SD. It is not a sample size for this SE, and the scale is not an L6 cost.
"""

import math

SPEC_POWER_FACTOR = 7.84
Z_ALPHA_95 = 1.959963984540054  # Phi^{-1}(0.975), a normal constant, not a measurement.
# CX-MET DELIVERY.json measurements.median_se_pair. A standard error, not sigma_D.
CX_MET_MEDIAN_SE_PAIR = 0.05296631946624709
OWNER_ROUNDED_SE_PAIR = 0.053
L6_DELTA_STAND_IN = 0.005
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
    """sigma is a paired SD of one independent difference, not a standard error."""
    if sigma <= 0 or delta == 0 or not math.isfinite(sigma) or not math.isfinite(delta):
        raise ValueError("sigma must be positive and delta nonzero")
    return factor * sigma * sigma / (delta * delta)


def n_at_power(sigma, delta, target=0.80):
    """Smallest n whose two-sided normal power reaches `target`. sigma is a paired SD."""
    if not 0.0 < target < 1.0:
        raise ValueError("power target must lie strictly between 0 and 1")
    n = max(1, math.floor(families_needed(sigma, delta)))
    while comparison_power(n, sigma, delta) < target:
        n += 1
        if n > 10_000_000:
            raise ValueError("no n reached the power target")
    return n


def interpret_noise(value, delta, unit, target=0.80):
    """Return a sample size only when `unit` is `paired_sd`.

    `standard_error` records the n the SD formula would emit if the SE were
    misread as a paired SD, and sets that n inapplicable.
    """
    if unit not in ("paired_sd", "standard_error"):
        raise ValueError("unit must be paired_sd or standard_error")
    misread = n_at_power(value, delta, target)
    if unit == "paired_sd":
        return {
            "applicable": True,
            "unit": "paired_sd",
            "n80": misread,
            "families_needed_raw": families_needed(value, delta),
            "n80_if_misread_as_paired_sd": None,
        }
    return {
        "applicable": False,
        "unit": "standard_error",
        "n80": None,
        "families_needed_raw": None,
        "n80_if_misread_as_paired_sd": misread,
        "reason": (
            "n for 80 percent power uses a paired standard deviation. "
            "A standard error is not that input, so the misread n is not a sample size."
        ),
    }


def se_pair_unit_check():
    """The published CX-MET median is an SE. n=882 does not apply to it."""
    exact = interpret_noise(CX_MET_MEDIAN_SE_PAIR, L6_DELTA_STAND_IN, "standard_error")
    rounded = interpret_noise(OWNER_ROUNDED_SE_PAIR, L6_DELTA_STAND_IN, "standard_error")
    return {
        "citation": (
            "CX-MET lanes/CX-MET/DELIVERY.json measurements.median_se_pair; "
            "SE_pair = max(bootstrap_se, abs(d0-d1)/2) at analytics/metrics.py:189-197"
        ),
        "definition": "max(bootstrap_se, abs(d0-d1)/2) on one log-RMSE-ratio pair",
        "delta": L6_DELTA_STAND_IN,
        "exact_value": CX_MET_MEDIAN_SE_PAIR,
        "families_needed_for_power_80": None,
        "measured_l6_paired_sd": False,
        "n80": None,
        "n80_if_misread_as_paired_sd_exact": exact["n80_if_misread_as_paired_sd"],
        "n80_if_misread_as_paired_sd_owner_round": rounded["n80_if_misread_as_paired_sd"],
        "name": "cx_met_median_se_pair",
        "owner_round": OWNER_ROUNDED_SE_PAIR,
        "paired_sd": False,
        "scale": "log_rmse_ratio",
        "status": "not_a_sample_size",
        "unit": "standard_error",
    }


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
        "se_pair_unit_check": se_pair_unit_check(),
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
