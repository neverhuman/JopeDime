"""G^judge(B): judge-partition q of the one program picked from B inner disclosures.

Spec §3.2: this is the only generalization metric, and it can fall as B grows.
The lineage interval reuses the arena's parent-cluster bootstrap. Siblings are
collapsed to the parent mean before resampling, so they do not narrow the interval.
"""

import math
from collections import defaultdict

from rlfw.arena.game_spec import require_sha
from rlfw.arena.kpi import cluster_stat

from .power import SEALED_FAMILIES, claim_floor, families_needed

PRIMARY_B = 8
CURVE_B = (1, 2, 4, 8)


def select_program(disclosures, B):
    """Argmax of the first B inner-fold scores. Judge fields are refused."""
    if B not in CURVE_B:
        raise ValueError("B must be one of 1, 2, 4, 8")
    window = list(disclosures)[:B]
    if len(window) < B:
        raise ValueError("not enough inner-fold disclosures")
    for item in window:
        if "judge_q" in item or "judge_row_ids" in item or "judge_labels" in item:
            raise ValueError("judge feedback is not an inner-fold disclosure")
        require_sha(item.get("program_sha256"), "program")
        if type(item.get("inner_q_grid")) is not int:
            raise ValueError("inner disclosures carry integer micro-q only")
    best = max(window, key=lambda item: (item["inner_q_grid"], item["program_sha256"]))
    return best["program_sha256"]


def g_judge(units, *, draws=2000, seed=7):
    if not units:
        raise ValueError("G^judge needs contexts")
    groups = defaultdict(list)
    censored = 0
    for unit in units:
        if unit.get("censored") or unit.get("judge_q") is None:
            censored += 1
            continue
        value = unit["judge_q"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("judge q must be finite")
        groups[unit["lineage_id"]].append(float(value))
    stat = cluster_stat(groups, draws=draws, seed=seed)
    return {
        "B": PRIMARY_B,
        "censored_contexts": censored,
        "ci95": stat["ci95"],
        "draws": draws,
        "label": "G^judge; not Lane A inner-CV",
        "lineages": stat["lineages"],
        "mean": stat["mean"],
        "seed": seed,
        "sign_stable": censored == 0 and stat["mean"] is not None,
        "units": stat["units"],
    }


def g_curve(contexts, *, draws=200, seed=7):
    """Fixture curve. Means may fall as B grows; that is the winner's-curse check."""
    means = {}
    for B in CURVE_B:
        units = []
        for context in contexts:
            program = select_program(context["disclosures"], B)
            judge_q = context["judge_q_by_program"][program]
            units.append(
                {
                    "censored": False,
                    "judge_q": judge_q,
                    "lineage_id": context["lineage_id"],
                }
            )
        report = g_judge(units, draws=draws, seed=seed)
        report["B"] = B
        means[B] = report
    return means


def judge_claim(
    report,
    *,
    n_families,
    n_contexts,
    canaries_a2_a4,
    overlap_status,
    sigma_source,
    observed_sigma,
    registered_delta,
):
    """G3+G4. A caller cannot mark a sigma as measured; none is bundled.

    The primary comparison is one preregistered B, so the multiplicity factor
    recorded here is 1. The B curve is a winner's-curse diagnostic, not a
    second claim. Censoring has no invented numeric bound.
    """
    if sigma_source == "measured":
        raise ValueError("this package has no measured judge sigma; label it unmeasured")
    if sigma_source not in ("fixture", "unmeasured"):
        raise ValueError("sigma_source must be fixture or unmeasured")
    reasons = []
    required = None
    if sigma_source == "unmeasured" or observed_sigma is None:
        reasons.append("observed noise unmeasured")
        _floor(reasons, n_families, n_contexts, float(SEALED_FAMILIES))
    elif (
        not isinstance(observed_sigma, (int, float))
        or not isinstance(registered_delta, (int, float))
        or not math.isfinite(observed_sigma)
        or not math.isfinite(registered_delta)
        or observed_sigma < 0
        or registered_delta == 0
    ):
        raise ValueError("fixture sigma must be positive and delta nonzero")
    elif observed_sigma == 0:
        reasons.append("observed noise is degenerate zero")
        _floor(reasons, n_families, n_contexts, float(SEALED_FAMILIES))
    else:
        required = families_needed(observed_sigma, registered_delta)
        _floor(reasons, n_families, n_contexts, required)
    if not report.get("sign_stable"):
        reasons.append("censoring can reverse the sign")
    lower = report.get("ci95", [None, None])[0]
    if lower is None or lower <= 0:
        reasons.append("CI lower bound is not above 0")
    if not canaries_a2_a4:
        reasons.append("A2-A4 canaries are not passing")
    if overlap_status == "quarantine":
        reasons.append("family overlap quarantined")
    elif overlap_status != "no_overlap":
        reasons.append("overlap_unknown")
    claim = None
    if not reasons and overlap_status == "no_overlap":
        claim = "Policy-training-unseen-family lift"
    return {
        "allowed": not reasons,
        "claim": claim,
        "multiplicity": 1,
        "power_required_families": required,
        "reasons": reasons,
        "sigma_source": sigma_source,
    }


def _floor(reasons, n_families, n_contexts, required):
    ok, detail = claim_floor(n_families, n_contexts, required)
    if not ok:
        reasons.append(detail)
