"""Train-only label rules (PLAN §6.3, §5 P5 "Labels" and "Outcome classes", §5 P7 test 2, §6.3.2 baseline arm).

Each context is labelled from its own rows only (its ``y_train`` and its own records), so perturbing any
other context, or any record of another context, cannot change its labels. Numpy and the standard library
only (no torch).

Per context c (inner folds j in {1, 2}, PLAN §6.3):

    s_c       = 1.4826*MAD(y_train) -> IQR/1.349 -> SD(y_train)
    eps_c     = max(1e-12*max(1, |mean(y_train)|), 1e-6*s_c)
    q         = -1/2 * sum_j log(max(RMSE_j, eps_c) / max(RMSE_j,base, eps_c))      (finite by construction)
    q_opt     = clip(q, -log 1e3, +log 1e3)
    q_grid    = round(q * 1e6)                       (ties are equality on this grid, hence transitive)
    gate_pass = 1/2 * sum_j RMSE_j / RMSE_j,base <= 1.05

A detail-less completed row uses ``q = min(-cv_mean_log_ratio, q_cap)`` with the context's floor maximum
``q_cap = -1/2 * sum_j log(eps_c / max(RMSE_j,base, eps_c))``; a non-finite detail-less value is UNKNOWN.

The per-context baseline ``RMSE_j,base`` comes only from rows with inner-CV detail (``is_detailed``), which must all
agree (§5 P5 "Labels"). Any other row's baseline pair must agree with it but never establishes it, so a context
without a detailed row is ``UNKNOWN_BASELINE`` and its completed rows get no q.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

COMPLETED = "completed"
TIMEOUT_OUTCOME = "timeout@90000ms"
TIMEOUT_S = 90.0
NONFINITE = "failed:nonfinite_metric"
FAILED_PREFIX = "failed:"
ABANDONED = frozenset({
    "abandoned:runtime_error",
    "abandoned:transport_error",
    "abandoned:blocked_existing_claim",
    "abandoned:capacity_deferred_before_claim",
})
MISSING = "missing"
LOCAL_BACKENDS = frozenset({"local"})
Q_GRID_SCALE = 1_000_000
Q_CLIP = math.log(1e3)
GATE_RATIO = 1.05
FEAS_COMPLETED, FEAS_FAILED, FEAS_TIMEOUT = 0, 1, 2
CONTEXT_STATUSES = ("OK", "DEGENERATE_BASELINE", "UNKNOWN_BASELINE")
Q_STATUSES = ("known", "degenerate_baseline", "unknown_baseline", "unknown_detailless_nonfinite",
              "not_completed", "unlabelled")
COUNT_KEYS = ("rows", "labelled", "completed", "failed", "timeout", "abandoned", "missing", "nonfinite",
              "q_known", "q_unknown")
# Ranking-pair weight (PLAN §6.3.2 L_BT): min(|dq_opt|, cap) * (factor if max midrank pct >= threshold else 1).
RANK_WEIGHT_CAP = 1.0
RANK_BOOST_THRESHOLD = 0.9
RANK_BOOST_FACTOR = 2.0

_OK, _DEGENERATE, _UNKNOWN = CONTEXT_STATUSES


class LabelError(RuntimeError):
    """A row or context cannot be labelled: bad outcome string, empty or non-finite ``y_train``, baseline
    disagreement, a non-numeric metric or a record that does not belong to its context."""


# ---------------------------------------------------------------------------------------------- numeric rules

def _y_array(y: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LabelError(f"y_train is not numeric: {exc}") from None
    if arr.ndim != 1 or arr.size == 0:
        raise LabelError("y_train must be a non-empty 1-d sequence")
    if not bool(np.all(np.isfinite(arr))):
        raise LabelError("y_train contains a non-finite value")
    return arr


def robust_scale(y: Sequence[float]) -> float:
    """s_c (PLAN §6.3): 1.4826*MAD, else IQR/1.349 (linear percentiles), else the population SD (may be 0.0)."""
    arr = _y_array(y)
    s = 1.4826 * float(np.median(np.abs(arr - np.median(arr))))
    if s > 0:
        return s
    iqr = (float(np.percentile(arr, 75, method="linear")) - float(np.percentile(arr, 25, method="linear"))) / 1.349
    if iqr > 0:
        return iqr
    return float(np.std(arr, ddof=0))


def epsilon(y: Sequence[float], s_c: float) -> float:
    """eps_c (PLAN §6.3) = max(1e-12*max(1, |mean(y)|), 1e-6*s_c); always > 0."""
    arr = _y_array(y)
    s_c = float(s_c)
    if not math.isfinite(s_c) or s_c < 0:
        raise LabelError(f"s_c must be finite and >= 0, got {s_c}")
    return max(1e-12 * max(1.0, abs(float(np.mean(arr)))), 1e-6 * s_c)


def _real(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise LabelError(f"{what} must be a real number, got {x!r}")
    return float(x)


def _is_pair(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(x is not None for x in v)


def _valid_baseline(v: Any) -> tuple[float, float] | None:
    """A baseline pair counts only when it holds 2 finite values >= 0."""
    if not _is_pair(v):
        return None
    pair = (_real(v[0], "baseline_rmse_j"), _real(v[1], "baseline_rmse_j"))
    if all(math.isfinite(b) and b >= 0 for b in pair):
        return pair
    return None


def is_detailed(rec: Mapping[str, Any]) -> bool:
    """Per-fold detail exists: ``rmse_j`` and ``baseline_rmse_j`` are both length-2 lists with no None."""
    return _is_pair(rec.get("rmse_j")) and _is_pair(rec.get("baseline_rmse_j"))


def normalize_outcome(rec: Mapping[str, Any]) -> str:
    """Validate the outcome class (PLAN §5 P5); a completed detailed row whose ``rmse_j`` holds NaN, +-inf or a
    negative value becomes ``failed:nonfinite_metric``. A detail-less row is never turned into a failure."""
    outcome = rec.get("outcome")
    if not isinstance(outcome, str):
        raise LabelError(f"outcome must be a string, got {outcome!r}")
    if outcome == COMPLETED:
        if is_detailed(rec):
            values = [_real(r, "rmse_j") for r in rec["rmse_j"]]
            if not all(math.isfinite(r) and r >= 0 for r in values):
                return NONFINITE
        return COMPLETED
    if outcome.startswith(FAILED_PREFIX):
        cls = outcome[len(FAILED_PREFIX):]
        if not cls or cls.strip() != cls:
            raise LabelError(f"failed outcome needs a non-empty class: {outcome!r}")
        return outcome
    if outcome == TIMEOUT_OUTCOME or outcome in ABANDONED or outcome == MISSING:
        return outcome
    raise LabelError(f"unknown outcome {outcome!r}")


def _check_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    return eps


def q_from_detail(rmse_j: Sequence[float], base_j: Sequence[float], eps: float) -> float:
    """q = -1/2 * sum_j log(max(r_j, eps) / max(b_j, eps)) (PLAN §6.3; both floors)."""
    eps = _check_eps(eps)
    if len(rmse_j) != len(base_j):
        raise ValueError("rmse_j and base_j differ in length")
    total = 0.0
    for r, b in zip(rmse_j, base_j):
        total += math.log(max(float(r), eps) / max(float(b), eps))
    return -0.5 * total


def q_cap(base_j: Sequence[float], eps: float) -> float:
    """The context's floor maximum -1/2 * sum_j log(eps / max(b_j, eps)); equals q_from_detail of an exact fit."""
    eps = _check_eps(eps)
    total = 0.0
    for b in base_j:
        total += math.log(eps / max(float(b), eps))
    return -0.5 * total


def clip_q(q: float) -> float:
    """q_opt = clip(q, -log 1e3, +log 1e3)."""
    return min(max(q, -Q_CLIP), Q_CLIP)


def q_grid(q: float) -> int:
    """The 1e-6 tie grid: Python round(q * 1e6) (half to even)."""
    if not math.isfinite(q):
        raise ValueError(f"q_grid needs a finite value, got {q}")
    return round(q * Q_GRID_SCALE)


def midrank_pcts(values: Sequence[int]) -> list[float]:
    """Ascending midrank percentiles: ties share the average 1-based rank; pct = (midrank-1)/(n-1); n == 1 -> 0.5."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        # midrank = (start + end + 2) / 2, so (midrank - 1) / (n - 1) = (start + end) / (2 (n - 1)), one rounding.
        pct = (start + end) / (2.0 * (n - 1))
        for pos in range(start, end + 1):
            out[order[pos]] = pct
        start = end + 1
    return out


# ------------------------------------------------------------------------------------------------ label types

@dataclass(frozen=True)
class RecordLabel:
    """Labels of one (context, program) row; ``None`` means unknown, never 0."""
    context_sha256: str
    program_sha256: str
    skeleton_sha256: str
    outcome: str
    labelled: bool
    feas_class: int | None
    feas_valid: bool
    q: float | None
    q_opt: float | None
    q_grid: int | None
    q_status: str
    gate_pass: bool | None
    midrank_pct: float | None
    rt_target: float | None
    rt_censored: bool
    rt_valid: bool
    value: float | None
    value_pct: float | None
    backend: str
    shared_seed: bool

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContextLabel:
    """Per-context label state; ``counts`` uses COUNT_KEYS (completed == q_known + q_unknown)."""
    context_sha256: str
    family_id: str
    dataset_hash: str
    split_role: str
    status: str
    s_c: float
    eps_c: float
    baseline_rmse_j: tuple[float, float] | None
    q_cap: float | None
    counts: dict[str, int]

    def to_json(self) -> dict:
        doc = {k: getattr(self, k) for k in self.__dataclass_fields__}
        doc["baseline_rmse_j"] = None if self.baseline_rmse_j is None else list(self.baseline_rmse_j)
        doc["counts"] = dict(self.counts)
        return doc


@dataclass(frozen=True)
class LabelTable:
    """Labels of a corpus: contexts by sha256 and their records sorted by program_sha256."""
    contexts: dict[str, ContextLabel]
    records: dict[str, tuple[RecordLabel, ...]]

    def counts(self) -> dict[str, int]:
        total = {k: 0 for k in COUNT_KEYS}
        for ctx in self.contexts.values():
            for k in COUNT_KEYS:
                total[k] += ctx.counts[k]
        return total


# ------------------------------------------------------------------------------------------------- labelling

def _req_str(doc: Mapping[str, Any], key: str) -> str:
    value = doc.get(key)
    if not isinstance(value, str) or not value:
        raise LabelError(f"field {key!r} must be a non-empty string, got {value!r}")
    return value


def _wall_us(rec: Mapping[str, Any]) -> float | None:
    wall = rec.get("wall_us")
    if wall is None:
        return None
    w = _real(wall, "wall_us")
    if not math.isfinite(w) or w < 0:
        raise LabelError(f"wall_us must be finite and >= 0, got {wall!r}")
    return w


def label_context(ctx: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[ContextLabel, tuple[RecordLabel, ...]]:
    """Label one context from its own rows only (PLAN §6.3 rules 1-8; see the module docstring)."""
    csha = _req_str(ctx, "context_sha256")
    y = _y_array(ctx.get("y_train"))
    s_c = robust_scale(y)
    eps_c = epsilon(y, s_c)
    ordered = sorted(rows, key=lambda r: _req_str(r, "program_sha256"))
    for rec in ordered:
        if rec.get("context_sha256") != csha:
            raise LabelError(f"record {rec.get('program_sha256')} belongs to {rec.get('context_sha256')}, not {csha}")
    outcomes = [normalize_outcome(rec) for rec in ordered]

    # (2) the per-context baseline (§5 P5): established only by rows with inner-CV detail, which must all agree; a
    #     valid baseline pair on any other row must agree with it but never establishes it.
    base: tuple[float, float] | None = None
    for rec in ordered:
        pair = _valid_baseline(rec.get("baseline_rmse_j")) if is_detailed(rec) else None
        if pair is None:
            continue
        if base is None:
            base = pair
        elif pair != base:
            raise LabelError(f"baseline disagreement in context {csha}: {pair} != {base}")
    if base is not None:
        for rec in ordered:
            pair = _valid_baseline(rec.get("baseline_rmse_j"))
            if pair is not None and pair != base:
                raise LabelError(f"baseline disagreement in context {csha}: {pair} != {base}")
    if base is None:
        status = _UNKNOWN
    elif any(b < eps_c for b in base):
        status = _DEGENERATE
    else:
        status = _OK
    cap = q_cap(base, eps_c) if status == _OK else None

    partial: list[dict[str, Any]] = []
    for rec, outcome in zip(ordered, outcomes):
        backend = _req_str(rec, "backend")
        labelled = outcome not in ABANDONED and outcome != MISSING
        # (3) feasibility class.
        if outcome == COMPLETED:
            feas: int | None = FEAS_COMPLETED
        elif outcome.startswith(FAILED_PREFIX):
            feas = FEAS_FAILED
        elif outcome == TIMEOUT_OUTCOME:
            feas = FEAS_TIMEOUT
        else:
            feas = None
        detailed = is_detailed(rec)
        # (4) q, only for completed rows of OK contexts.
        q: float | None = None
        if not labelled:
            q_status = "unlabelled"
        elif outcome != COMPLETED:
            q_status = "not_completed"
        elif status == _DEGENERATE:
            q_status = "degenerate_baseline"
        elif status == _UNKNOWN:
            q_status = "unknown_baseline"
        elif detailed:
            q = q_from_detail([_real(r, "rmse_j") for r in rec["rmse_j"]], base, eps_c)
            q_status = "known"
        else:
            raw = rec.get("cv_mean_log_ratio")
            v = None if raw is None else -_real(raw, "cv_mean_log_ratio")
            if v is None or not math.isfinite(v):
                q_status = "unknown_detailless_nonfinite"
            else:
                q = min(v, cap)
                q_status = "known"
        # (5) the evaluator's arithmetic-mean guard, for completed rows only.
        gate: bool | None = None
        if outcome == COMPLETED:
            own_base = [_real(b, "baseline_rmse_j") for b in rec["baseline_rmse_j"]] if detailed else None
            if own_base is not None and all(b > 0 for b in own_base):
                r0, r1 = (_real(r, "rmse_j") for r in rec["rmse_j"])
                gate = 0.5 * (r0 / own_base[0] + r1 / own_base[1]) <= GATE_RATIO
            else:
                raw_ratio = rec.get("cv_mean_ratio")
                ratio = None if raw_ratio is None else _real(raw_ratio, "cv_mean_ratio")
                if ratio is not None and math.isfinite(ratio):
                    gate = ratio <= GATE_RATIO
        # (7) runtime target: completed exact, timeout right-censored at 90 s.
        wall = _wall_us(rec)
        if outcome == COMPLETED and wall is not None:
            rt_target: float | None = math.log1p(wall / 1e6)
            rt_censored = False
        elif outcome == TIMEOUT_OUTCOME:
            rt_target, rt_censored = math.log1p(TIMEOUT_S), True
        else:
            rt_target, rt_censored = None, False
        partial.append({
            "context_sha256": csha,
            "program_sha256": rec["program_sha256"],
            "skeleton_sha256": _req_str(rec, "skeleton_sha256"),
            "outcome": outcome,
            "labelled": labelled,
            "feas_class": feas,
            "feas_valid": feas is not None and backend in LOCAL_BACKENDS,
            "q": q,
            "q_opt": None if q is None else clip_q(q),
            "q_grid": None if q is None else q_grid(q),
            "q_status": q_status,
            "gate_pass": gate,
            "midrank_pct": None,
            "rt_target": rt_target,
            "rt_censored": rt_censored,
            "rt_valid": backend in LOCAL_BACKENDS and feas in (FEAS_COMPLETED, FEAS_TIMEOUT) and rt_target is not None,
            "value": None,
            "value_pct": None,
            "backend": backend,
            "shared_seed": bool(rec.get("shared_seed", False)),
            "_detailed": detailed,
            "_rmse_j": rec.get("rmse_j"),
        })

    # (6) within-context midrank percentiles over the q grid of known rows.
    known = [i for i, p in enumerate(partial) if p["q_status"] == "known"]
    for i, pct in zip(known, midrank_pcts([partial[i]["q_grid"] for i in known])):
        partial[i]["midrank_pct"] = pct

    # (8) baseline arm: value = 1/(1+rmse_geo/median) over known detailed rows, and its midrank percentile.
    arm = [i for i in known if partial[i]["_detailed"]]
    if arm:
        geo = [math.exp(0.5 * sum(math.log(max(_real(r, "rmse_j"), eps_c)) for r in partial[i]["_rmse_j"]))
               for i in arm]
        median = float(np.median(np.asarray(geo, dtype=np.float64)))
        values = [1.0 / (1.0 + g / median) for g in geo]
        for i, value, pct in zip(arm, values, midrank_pcts([q_grid(v) for v in values])):
            partial[i]["value"] = value
            partial[i]["value_pct"] = pct

    labels = tuple(RecordLabel(**{k: v for k, v in p.items() if not k.startswith("_")}) for p in partial)
    counts = {k: 0 for k in COUNT_KEYS}
    counts["rows"] = len(labels)
    for lab in labels:
        counts["labelled"] += lab.labelled
        counts["completed"] += lab.outcome == COMPLETED
        counts["failed"] += lab.outcome.startswith(FAILED_PREFIX)
        counts["timeout"] += lab.outcome == TIMEOUT_OUTCOME
        counts["abandoned"] += lab.outcome in ABANDONED
        counts["missing"] += lab.outcome == MISSING
        counts["nonfinite"] += lab.outcome == NONFINITE
        counts["q_known"] += lab.q_status == "known"
        counts["q_unknown"] += lab.outcome == COMPLETED and lab.q is None
    ctx_label = ContextLabel(
        context_sha256=csha,
        family_id=_req_str(ctx, "family_id"),
        dataset_hash=_req_str(ctx, "dataset_hash"),
        split_role=_req_str(ctx, "split_role"),
        status=status,
        s_c=s_c,
        eps_c=eps_c,
        baseline_rmse_j=base,
        q_cap=cap,
        counts=counts,
    )
    return ctx_label, labels


def label_corpus(contexts: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> LabelTable:
    """Group records by context and label each context independently (sorted by context_sha256)."""
    rows_by_ctx: dict[str, list[Mapping[str, Any]]] = {}
    for ctx in contexts:
        csha = _req_str(ctx, "context_sha256")
        if csha in rows_by_ctx:
            raise LabelError(f"duplicate context {csha}")
        rows_by_ctx[csha] = []
    for rec in records:
        csha = _req_str(rec, "context_sha256")
        if csha not in rows_by_ctx:
            raise LabelError(f"record {rec.get('program_sha256')} names absent context {csha}")
        rows_by_ctx[csha].append(rec)
    ctx_labels: dict[str, ContextLabel] = {}
    rec_labels: dict[str, tuple[RecordLabel, ...]] = {}
    for ctx in sorted(contexts, key=lambda c: c["context_sha256"]):
        csha = ctx["context_sha256"]
        ctx_labels[csha], rec_labels[csha] = label_context(ctx, rows_by_ctx[csha])
    return LabelTable(contexts=ctx_labels, records=rec_labels)


def ranking_pairs(members: Sequence[RecordLabel]) -> list[tuple[int, int, float]]:
    """Bradley-Terry pairs (winner, loser, w) over index pairs i < k with both q known and distinct q_grid.

    w = min(|q_opt_i - q_opt_k|, 1) * (2 if max(midrank_pct_i, midrank_pct_k) >= 0.9 else 1). Grid ties give no
    pair; since a tie is equality on the grid, the tie relation is transitive. Lexicographic (i, k) order."""
    out: list[tuple[int, int, float]] = []
    for i, a in enumerate(members):
        if a.q_status != "known":
            continue
        for k in range(i + 1, len(members)):
            b = members[k]
            if b.q_status != "known" or a.q_grid == b.q_grid:
                continue
            boost = RANK_BOOST_FACTOR if max(a.midrank_pct, b.midrank_pct) >= RANK_BOOST_THRESHOLD else 1.0
            w = min(abs(a.q_opt - b.q_opt), RANK_WEIGHT_CAP) * boost
            out.append((i, k, w) if a.q_grid > b.q_grid else (k, i, w))
    return out
