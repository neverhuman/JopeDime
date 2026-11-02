"""Measured, lineage-macro KPIs with paired parent-cluster uncertainty."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
import math
import random
from statistics import fmean, median

from .game_spec import require_sha, R5_KPI_VERSION

LABEL = "Lane A, task-label-aware inner-CV search, dev panel L=54, directional"


@dataclass(frozen=True)
class SlotResult:
    context_id: str
    lineage_id: str
    arm: str
    seed: int
    slot: int
    program_sha256: str
    outcome: str
    q_grid: int | None = None
    qualified: bool = False
    cache_hit: bool = False
    duplicate: bool = False
    attempts: int = 1
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    spec_sha256: str = ""
    receipt_sha256: str | None = None
    censor_reason: str | None = None
    unknown_cost_attempts: int = 0

    def __post_init__(self):
        require_sha(self.program_sha256, "program")
        require_sha(self.spec_sha256, "game spec")
        if self.outcome not in (
            "qualified_complete",
            "program_failed",
            "timeout",
            "parity_unqualified",
            "censored",
        ):
            raise ValueError("unresolved infra/parity outcomes must be censored")
        if self.outcome == "qualified_complete":
            if not self.qualified or type(self.q_grid) is not int:
                raise ValueError("KPI scores require qualified measured q_grid")
            require_sha(self.receipt_sha256, "measured result receipt")
        elif self.q_grid is not None or self.qualified:
            raise ValueError("failure/unknown is never a q value")
        if self.slot < 1 or self.attempts not in (0, 1, 2, 3):
            raise ValueError("invalid slot or retry count")
        if any(
            not math.isfinite(v) or v < 0 for v in (self.wall_seconds, self.cpu_seconds)
        ):
            raise ValueError("invalid evaluator cost")
        if self.censor_reason is not None and (
            self.outcome != "censored"
            or self.censor_reason not in ("parity_custody", "infra_unknown")
        ):
            raise ValueError("censor reason must preserve the unresolved outcome")
        if (
            type(self.unknown_cost_attempts) is not int
            or not 0 <= self.unknown_cost_attempts <= self.attempts
        ):
            raise ValueError("unknown costs must remain explicit")

    @property
    def charged(self):
        return self.outcome != "censored"

    @property
    def q(self):
        # The deployed q_grid is an integer micro-q, not a q-unit reward.
        return self.q_grid / 1_000_000 if self.qualified else None

    def to_dict(self):
        return asdict(self)


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


@lru_cache(maxsize=8)
def cluster_draws(n, draws=2000, seed=7):
    rng = random.Random(seed)
    return tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(draws))


def cluster_stat(values_by_lineage, draws=2000, seed=7, alpha=0.05):
    """Each input item is one parent mean; siblings never become clusters."""
    values = [fmean(v) for _, v in sorted(values_by_lineage.items()) if v]
    if not values:
        return {"mean": None, "ci95": [None, None], "lineages": 0, "units": 0}
    replicates = [
        fmean(values[i] for i in sample)
        for sample in cluster_draws(len(values), draws, seed)
    ]
    return {
        "mean": fmean(values),
        "ci95": [
            percentile(replicates, alpha / 2),
            percentile(replicates, 1 - alpha / 2),
        ],
        "lineages": len(values),
        "units": sum(len(v) for v in values_by_lineage.values()),
    }


def summarize_values(units, key, predicate=lambda u: True):
    groups = defaultdict(list)
    for unit in units:
        if predicate(unit) and unit[key] is not None:
            groups[unit["lineage_id"]].append(unit[key])
    return cluster_stat(groups)


def outcome_score(best, hwm, tau=0.01):
    if best is None:
        return "LOSS", 0.0
    delta = best - hwm
    # q_grid is discrete; arithmetic at an exact registered margin is a tie.
    if math.isclose(abs(delta), tau, rel_tol=0, abs_tol=1e-12) or abs(delta) <= tau:
        return "TIE", 0.5
    return ("WIN", 1.0) if delta > tau else ("LOSS", 0.0)


def context_units(rows, hwm, tau=0.01, *, hwm_programs=None, program_parity=False):
    grouped = defaultdict(list)
    for row in rows:
        if row.outcome == "parity_unqualified" and not program_parity:
            raise ValueError("program parity requires an explicit R5 metric version")
        grouped[(row.context_id, row.arm, row.seed)].append(row)
    if hwm_programs is None:
        # Preserve callers that already bind identity alongside numeric quality.
        hwm_programs = {
            ctx: hwm[(ctx, "program_sha256")]
            for ctx, _, _ in grouped
            if (ctx, "program_sha256") in hwm
        }
    for program in hwm_programs.values():
        require_sha(program, "original program")
    units = []
    for (ctx, arm, seed), slots in sorted(grouped.items()):
        slots.sort(key=lambda r: r.slot)
        if [r.slot for r in slots] != list(range(1, 9)):
            raise ValueError(
                "a closed sampling unit requires exactly eight resolved slots"
            )
        if len({r.lineage_id for r in slots}) != 1:
            raise ValueError("lineage changed within a context")
        incumbent = hwm.get(ctx)
        best = {
            b: max((r.q for r in slots if r.slot <= b and r.qualified), default=None)
            for b in (1, 2, 4, 8)
        }
        censored = any(r.outcome == "censored" for r in slots)
        outcome, score = (
            outcome_score(best[8], incumbent, tau)
            if incumbent is not None
            else (None, None)
        )
        unit = {
            "context_id": ctx,
            "lineage_id": slots[0].lineage_id,
            "arm": arm,
            "seed": seed,
            "hwm": incumbent,
            "best_q": best,
            "censored": censored,
            "outcome": None if censored else outcome,
            "score": None if censored else score,
            "lower_score": 0.0 if censored and incumbent is not None else score,
            "upper_score": 1.0 if censored and incumbent is not None else score,
            "empty_v": float(best[8] is None),
            "valid8": float(best[8] is not None),
            "strict_win": float(outcome == "WIN") if not censored and outcome else None,
            "tie": float(outcome == "TIE") if not censored and outcome else None,
            "loss": float(outcome == "LOSS") if not censored and outcome else None,
            "regret": incumbent - best[8]
            if incumbent is not None and best[8] is not None
            else None,
            "positive_regret": max(0, incumbent - best[8])
            if incumbent is not None and best[8] is not None
            else None,
            "positive_gain": (
                max(0, best[8] - incumbent) if best[8] is not None else 0.0
            )
            if incumbent is not None and not censored
            else None,
            "failure_rate": sum(r.outcome == "program_failed" for r in slots) / 8,
            "timeout_rate": sum(r.outcome == "timeout" for r in slots) / 8,
            "censored_rate": sum(r.outcome == "censored" for r in slots) / 8,
            "distinct_shas": float(len({r.program_sha256 for r in slots})),
            "rediscovery_rate": (
                sum(r.program_sha256 == hwm_programs[ctx] for r in slots) / 8
                if ctx in hwm_programs
                else None
            ),
            "cache_hit_rate": sum(r.cache_hit for r in slots) / 8,
            "wall_seconds": None
            if any(r.unknown_cost_attempts for r in slots)
            else sum(r.wall_seconds for r in slots),
            "cpu_seconds": None
            if any(r.unknown_cost_attempts for r in slots)
            else sum(r.cpu_seconds for r in slots),
            "observed_wall_seconds_lower_bound": sum(r.wall_seconds for r in slots),
            "observed_cpu_seconds_lower_bound": sum(r.cpu_seconds for r in slots),
            "unknown_cost_attempts": sum(r.unknown_cost_attempts for r in slots),
            "qualified_completion_rate": sum(r.qualified for r in slots) / 8,
            "charged_slots": sum(r.charged for r in slots),
        }
        if program_parity:
            unit["parity_unqualified_rate"] = (
                sum(r.outcome == "parity_unqualified" for r in slots) / 8
            )
        for b in (1, 2, 4, 8):
            unit[f"best_q@{b}"] = best[b]
            unit[f"valid@{b}"] = float(best[b] is not None)
        units.append(unit)
    return units


def arm_metrics(units):
    result = {
        "win_rate": summarize_values(units, "score"),
        "censoring_bounds": [
            summarize_values(units, "lower_score")["mean"],
            summarize_values(units, "upper_score")["mean"],
        ],
        "censored_units": sum(u["censored"] for u in units),
        "no_comparable_hwm": sum(u["hwm"] is None for u in units),
    }
    for key in (
        "strict_win",
        "tie",
        "loss",
        "empty_v",
        "regret",
        "positive_regret",
        "positive_gain",
        "valid8",
        "failure_rate",
        "timeout_rate",
        "censored_rate",
        "distinct_shas",
        "rediscovery_rate",
        "cache_hit_rate",
        "wall_seconds",
        "cpu_seconds",
        "qualified_completion_rate",
    ):
        result[key] = summarize_values(units, key)
    for b in (1, 2, 4, 8):
        result[f"best_q@{b}"] = summarize_values(units, f"best_q@{b}")
        result[f"valid@{b}"] = summarize_values(units, f"valid@{b}")
    if any("parity_unqualified_rate" in u for u in units):
        if not all("parity_unqualified_rate" in u for u in units):
            raise ValueError("mixed metric versions in one arm")
        result["parity_unqualified_rate"] = summarize_values(
            units, "parity_unqualified_rate"
        )
    costs = [u["wall_seconds"] for u in units if u["wall_seconds"] is not None]
    result["median_evaluator_seconds"] = (
        median(costs) if len(costs) == len(units) and costs else None
    )
    result["unknown_cost_attempts"] = sum(u["unknown_cost_attempts"] for u in units)
    return result


def paired_difference(a, b, alpha=0.05):
    left = {(u["context_id"], u["seed"]): u for u in a}
    right = {(u["context_id"], u["seed"]): u for u in b}
    if set(left) != set(right) or not left:
        raise ValueError(
            "paired arms require exactly the same closed contexts and seeds"
        )
    differences = defaultdict(lambda: defaultdict(list))
    for key in sorted(left):
        x, y = left[key], right[key]
        if x["lineage_id"] != y["lineage_id"] or x["hwm"] != y["hwm"]:
            raise ValueError("paired lineage/incumbent mismatch")
        lineage = x["lineage_id"]
        if x["hwm"] is not None:
            differences["lower_score"][lineage].append(
                x["lower_score"] - y["upper_score"]
            )
            differences["upper_score"][lineage].append(
                x["upper_score"] - y["lower_score"]
            )
        if not x["censored"] and not y["censored"]:
            for metric in (
                "score",
                "best_q@8",
                "regret",
                "positive_regret",
                "positive_gain",
            ):
                if x[metric] is not None and y[metric] is not None:
                    differences[metric][lineage].append(x[metric] - y[metric])
        for metric in ("valid8", "wall_seconds", "cpu_seconds", "distinct_shas"):
            if x[metric] is not None and y[metric] is not None:
                differences[metric][lineage].append(x[metric] - y[metric])
    result = {
        metric: cluster_stat(differences[metric], alpha=alpha)
        for metric in (
            "score",
            "best_q@8",
            "regret",
            "positive_regret",
            "positive_gain",
            "valid8",
            "wall_seconds",
            "cpu_seconds",
            "distinct_shas",
        )
    }
    result["censoring_bounds"] = [
        cluster_stat(differences[key])["mean"] for key in ("lower_score", "upper_score")
    ]
    lower, upper = result["censoring_bounds"]
    result["censoring_decision"] = (
        "inconclusive"
        if lower is None or lower <= 0 <= upper
        else ("positive" if lower > 0 else "negative")
    )
    result["units"] = len(left)
    return result




from .kpi_report import compute_report, ranking_metrics
