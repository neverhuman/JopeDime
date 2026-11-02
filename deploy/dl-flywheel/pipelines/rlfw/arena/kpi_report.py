"""Report assembly and ranking metrics for measured arena outcomes."""

from collections import defaultdict
from itertools import combinations
import math

from .game_spec import require_sha, R5_KPI_VERSION
from .kpi import (
    LABEL,
    arm_metrics,
    cluster_stat,
    context_units,
    paired_difference,
    summarize_values,
)

def ranking_metrics(candidates, tau=0.01):
    """Predictions are compared to receipt-qualified q, never substituted for q."""
    grouped = defaultdict(dict)
    for c in candidates:
        if (
            c.get("source") not in ("archive", "fresh")
            or c.get("role") != "development"
        ):
            raise ValueError("ranking requires separately labelled development sources")
        if c.get("outcome") != "qualified_complete" or c.get("qualified") is not True:
            raise ValueError("ranking requires qualified measured candidates")
        require_sha(c.get("receipt_sha256"), "ranking receipt")
        require_sha(c.get("program_sha256"), "ranking program")
        if type(c.get("q_grid")) is not int or not math.isfinite(
            c.get("predicted_q", float("nan"))
        ):
            raise ValueError("ranking requires finite measured and predicted values")
        key = (c["source"], c["context_id"], c["lineage_id"])
        old = grouped[key].get(c["program_sha256"])
        if old is not None and (old["q_grid"], old["predicted_q"]) != (
            c["q_grid"],
            c["predicted_q"],
        ):
            raise ValueError("conflicting duplicate ranking candidate")
        grouped[key][c["program_sha256"]] = c
    values = {
        source: defaultdict(lambda: defaultdict(list))
        for source in ("archive", "fresh")
    }
    for (source, _, lineage), distinct in sorted(grouped.items()):
        programs = list(distinct.values())
        if len(programs) < 2:
            continue
        agree = eligible = concordant = discordant = tie_pred = tie_q = 0
        for x, y in combinations(programs, 2):
            dq, dp = (
                (x["q_grid"] - y["q_grid"]) / 1_000_000,
                x["predicted_q"] - y["predicted_q"],
            )
            if abs(dq) > tau and not math.isclose(
                abs(dq), tau, rel_tol=0, abs_tol=1e-12
            ):
                eligible += 1
                agree += dq * dp > 0
            if dq and dp:
                concordant += dq * dp > 0
                discordant += dq * dp < 0
            elif not dp and dq:
                tie_pred += 1
            elif not dq and dp:
                tie_q += 1
        if not eligible:
            continue
        values[source]["pairwise_accuracy"][lineage].append(agree / eligible)
        denominator = math.sqrt(
            (concordant + discordant + tie_pred) * (concordant + discordant + tie_q)
        )
        if denominator:
            values[source]["kendall_tau_b"][lineage].append(
                (concordant - discordant) / denominator
            )
        truth = min(programs, key=lambda c: (-c["q_grid"], c["program_sha256"]))
        predicted = min(
            programs, key=lambda c: (-c["predicted_q"], c["program_sha256"])
        )
        values[source]["top1_hit"][lineage].append(
            float(truth["program_sha256"] == predicted["program_sha256"])
        )
    return {
        source: {
            "label": "held out from training, reused for monitoring",
            **{
                metric: cluster_stat(values[source][metric])
                for metric in ("pairwise_accuracy", "kendall_tau_b", "top1_hit")
            },
        }
        for source in ("archive", "fresh")
    }


def compute_report(spec, rows, hwm, closed_contexts, ranking=()):
    allowed = {c.context_id: c.lineage_id for c in spec.contexts}
    closed = set(closed_contexts)
    if not closed <= allowed.keys():
        raise ValueError("unknown closed context")
    for candidate in ranking:
        if allowed.get(candidate.get("context_id")) != candidate.get("lineage_id"):
            raise ValueError(
                "ranking candidate lies outside the frozen development panel"
            )
    for row in rows:
        if (
            row.spec_sha256 != spec.sha256
            or allowed.get(row.context_id) != row.lineage_id
            or row.arm not in {a.name for a in spec.arms}
            or row.outcome not in spec.outcome_classes
        ):
            raise ValueError("result identity disagrees with frozen GameSpec")
    comparators = {r.context_id: r for r in rows if r.arm == "hwm"}
    if len(comparators) != sum(r.arm == "hwm" for r in rows):
        raise ValueError("each original incumbent is re-scored only once")
    for ctx in closed:
        registered = next(c for c in spec.contexts if c.context_id == ctx)
        comparator = comparators.get(ctx)
        if (
            comparator is not None
            and comparator.program_sha256 != registered.hwm_program_sha256
        ):
            raise ValueError("comparator is not the registered original incumbent")
        if hwm.get(ctx) != (comparator.q if comparator else None):
            raise ValueError("incumbent q requires a qualified re-execution receipt")
    sampling = [r for r in rows if r.arm != "hwm" and r.context_id in closed]
    expected = {
        (c, a.name, s, slot)
        for c in closed
        for a in spec.arms
        if a.name != "hwm"
        for s in spec.seeds
        for slot in range(1, 9)
    }
    actual = [(r.context_id, r.arm, r.seed, r.slot) for r in sampling]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("closed context is missing or duplicating matched slots")
    reports = {}
    primary_units = None
    for tau in (spec.tau, 0.005, 0.02):
        units = context_units(
            sampling,
            hwm,
            tau,
            hwm_programs={
                c.context_id: c.hwm_program_sha256
                for c in spec.contexts
                if c.hwm_program_sha256 is not None
            },
            program_parity=spec.kpi_version == R5_KPI_VERSION,
        )
        by_arm = {
            a.name: [u for u in units if u["arm"] == a.name]
            for a in spec.arms
            if a.name != "hwm"
        }
        reports[str(tau)] = {
            "arms": {name: arm_metrics(v) for name, v in by_arm.items()},
            "paired": {
                f"{a} - {b}": paired_difference(by_arm[a], by_arm[b])
                for a, b in combinations(by_arm, 2)
                if by_arm[a]
            },
        }
        if tau == spec.tau:
            primary_units = units
    count = len(closed)
    comparable = sum(hwm.get(c) is not None for c in closed)
    censored = sum(u["censored"] for u in primary_units)
    fraction = censored / len(primary_units) if primary_units else None
    lineages = len({allowed[c] for c in closed})
    headroom = [
        {
            "context_id": c,
            "lineage_id": allowed[c],
            "headroom": max(
                hwm[c],
                max(
                    (r.q for r in sampling if r.context_id == c and r.qualified),
                    default=hwm[c],
                ),
            )
            - hwm[c],
        }
        for c in sorted(closed)
        if hwm.get(c) is not None
    ]
    for u in headroom:
        u["above_tau"] = float(u["headroom"] > spec.tau)
    coverage = defaultdict(int)
    for ctx in closed:
        row = comparators.get(ctx)
        category = (
            "hwm_missing_program"
            if row is None
            else "comparable"
            if row.qualified
            else "hwm_parity_unqualified"
            if row.outcome == "parity_unqualified"
            else "hwm_parity_custody"
            if row.censor_reason == "parity_custody"
            else "hwm_infra_unknown"
            if row.outcome == "censored"
            else "hwm_program_failed"
        )
        coverage[category] += 1
    return {
        "label": LABEL
        if spec.kpi_version != R5_KPI_VERSION
        else (
            "Lane A, task-label-aware inner-CV search, dev panel "
            f"L={len({c.lineage_id for c in spec.contexts})}, directional; {spec.device} inference"
        ),
        "spec_sha256": spec.sha256,
        "pairing_sha256": spec.pairing_sha256,
        "bootstrap": 2000,
        "bootstrap_seed": 7,
        "tau": spec.tau,
        "budget": spec.budget,
        "closed_contexts": count,
        "observed_lineages": lineages,
        "registered_lineages": len({c.lineage_id for c in spec.contexts}),
        "partial_panel": count != len(spec.contexts),
        "comparable_hwm": comparable,
        "no_comparable_hwm": count - comparable,
        "hwm_coverage_categories": dict(coverage),
        "comparable_coverage": comparable / count if count else None,
        "low_hwm_coverage": bool(count and comparable / count < 0.8),
        "censored_unit_fraction": fraction,
        "baseline_size_and_coverage_gate": bool(
            count >= 54
            and lineages >= 50
            and fraction is not None
            and fraction < 0.05
            and comparable / count >= 0.8
        ),
        "integrity_and_determinism": "require separate G0/G1 and replay receipts",
        "tau_sensitivity": reports,
        "ranking": ranking_metrics(ranking),
        "headroom_lower_bound": summarize_values(headroom, "headroom"),
        "headroom_above_tau": summarize_values(headroom, "above_tau"),
        "per_context": primary_units,
    }
