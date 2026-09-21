"""G9 -- redundancy, conflict, and estimated fit cost. Prefix ``g9_``.

Pure genome-only features. Three ideas:

1. **Redundancy**: the same op appearing twice, or two different scalers stacked, is wasted
   pipeline budget. At the 7-stage cap (74.8% of featmath rows, the worst band) this is where
   the wasted stages live.
2. **Conflict**: a rank-family transform feeding a linear head is the "discretize-then-linear"
   motif that measured +0.06 (bin_onehot/quantile_bins -> ridge_gcv); ``rank`` in a later
   position measured +0.021.
3. **Cost**: a scalar proxy for how expensive the pipeline is to fit, plus how that cost is
   *distributed* across stages. Cost is a genome-level quantity that varies freely between
   genomes on the same dataset, so it survives GroupKFold-by-dataset.

The dataset-interacted version (cost x n_rows, the actual timeout-risk signal) is built
elsewhere; ``g9_cost_x_depth`` is the genome-only stand-in.
"""

from __future__ import annotations

import math

__all__ = ["feature_names", "extract", "COST_WEIGHT"]

# Cost weight per grammar cost_class. Roughly log-spaced: each step up is ~4x the fit time.
# `portfolio` sits between expensive and extreme -- it fits several cheap members, not one
# extreme model.
COST_WEIGHT: dict[str, float] = {
    "cheap": 1.0,
    "medium": 4.0,
    "expensive": 16.0,
    "extreme": 64.0,
    "portfolio": 32.0,
}
_DEFAULT_WEIGHT = COST_WEIGHT["medium"]

_NAMES: tuple[str, ...] = (
    "g9_has_duplicate_op",
    "g9_duplicate_op_count",
    "g9_has_conflicting_scalers",
    "g9_n_scaling_ops",
    "g9_rank_transform_then_linear",
    "g9_est_fit_cost",
    "g9_est_cost_per_stage",
    "g9_cost_concentration",
    "g9_max_stage_cost",
    "g9_n_cheap",
    "g9_n_medium",
    "g9_n_expensive",
    "g9_n_extreme",
    "g9_head_cost_weight",
    "g9_cost_x_depth",
)


def feature_names(engine: str = "featmath") -> list[str]:
    """Every column this module emits, in stable order.

    Identical for every engine: the columns are defined over grammar-independent concepts
    (cost classes, the ``scaling``/``rank``/``linear`` families), all of which exist in both
    the featmath and newmath grammars.
    """
    return list(_NAMES)


def _gini(weights: list[float]) -> float:
    """Population Gini coefficient of the per-stage cost weights.

    Defined as the mean absolute difference between all ordered pairs, normalised by twice
    the mean::

        G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))

    computed via the equivalent sorted-rank form (x sorted ascending, i 1-indexed)::

        G = (2 * sum_i i * x_i) / (n * sum_i x_i) - (n + 1) / n

    Properties, stated exactly so downstream readers do not over-interpret the value:
      * G = 0 when every stage costs the same (a fully cheap pipeline is always 0.0).
      * G is maximal at ``(n - 1) / n`` when one stage carries all the cost, NOT at 1.0.
        With the 7-stage featmath cap the ceiling is 6/7 = 0.857; with 2 stages it is 0.5.
        This is the standard (uncorrected) Gini -- the ``n / (n - 1)`` bias-corrected variant
        is deliberately not used because it would smuggle an inverse ``n_stages`` term into
        what is meant to be a pure concentration measure.
      * G = 0.0 for a single-stage pipeline and for a zero-total-cost pipeline.
    """
    n = len(weights)
    if n < 2:
        return 0.0
    total = math.fsum(weights)
    if total <= 0.0:
        return 0.0
    ordered = sorted(weights)
    weighted = math.fsum((i + 1) * x for i, x in enumerate(ordered))
    g = (2.0 * weighted) / (n * total) - (n + 1) / n
    # Clamp: guards float error at the endpoints only, never a real value.
    if g < 0.0:
        return 0.0
    return g if g < 1.0 else 1.0


def extract(pg, engine: str = "featmath") -> dict[str, float]:
    """One genome -> {column: float}. Never raises; all-zero defaults when ``pg`` is unparseable.

    ``pg.ok`` is False for ``(sym ...)`` genomes and for anything that is not a ``(pipe ...)``,
    and those carry no stages at all -- there is nothing to cost, so every column is 0.0.
    """
    out = {name: 0.0 for name in _NAMES}

    stages = getattr(pg, "stages", None) or []
    if not getattr(pg, "ok", False) or not stages:
        return out

    n_stages = len(stages)
    ops = [s.op for s in stages]

    # --- redundancy -------------------------------------------------------------------
    # duplicate_op_count = number of *surplus* stages, i.e. sum over ops of (count - 1),
    # which equals n_stages - n_distinct_ops. Three `rank` stages contribute 2, not 1.
    n_distinct = len(set(ops))
    dup_count = n_stages - n_distinct
    out["g9_duplicate_op_count"] = float(dup_count)
    out["g9_has_duplicate_op"] = 1.0 if dup_count > 0 else 0.0

    # --- conflict ---------------------------------------------------------------------
    # Multi-label families: winsor_clip / winsor_moments carry ('scaling', 'robust') and count
    # as scalers alongside standardize / robust_scale / norm_features.
    scaling_ops = {s.op for s in stages if "scaling" in s.families}
    n_scaling = sum(1 for s in stages if "scaling" in s.families)
    out["g9_n_scaling_ops"] = float(n_scaling)
    out["g9_has_conflicting_scalers"] = 1.0 if len(scaling_ops) >= 2 else 0.0

    head = stages[-1]
    head_is_linear = "linear" in head.families
    if head_is_linear and any("rank" in s.families for s in stages[:-1]):
        out["g9_rank_transform_then_linear"] = 1.0

    # --- cost -------------------------------------------------------------------------
    weights = [COST_WEIGHT.get(s.cost_class, _DEFAULT_WEIGHT) for s in stages]
    total = math.fsum(weights)

    counts = {"cheap": 0, "medium": 0, "expensive": 0, "extreme": 0}
    for s in stages:
        if s.cost_class in counts:
            counts[s.cost_class] += 1
    out["g9_n_cheap"] = float(counts["cheap"])
    out["g9_n_medium"] = float(counts["medium"])
    out["g9_n_expensive"] = float(counts["expensive"])
    out["g9_n_extreme"] = float(counts["extreme"])

    out["g9_est_fit_cost"] = math.log1p(total)
    out["g9_est_cost_per_stage"] = math.log1p(total / n_stages)
    out["g9_max_stage_cost"] = math.log1p(max(weights))
    out["g9_cost_concentration"] = _gini(weights)
    out["g9_head_cost_weight"] = math.log1p(COST_WEIGHT.get(head.cost_class, _DEFAULT_WEIGHT))
    out["g9_cost_x_depth"] = math.log1p(total * n_stages)

    return out
