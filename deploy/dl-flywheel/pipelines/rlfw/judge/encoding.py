"""Fit-only encoding stand-in for the Lane B canary contracts.

This is not Kumo and not XGBoost. It does not load weights. Feature rank is
absolute correlation on fit rows only, which is enough to test the exposure
rules in spec §7.3 A1–A4 and packet §§2.2–2.5:

- judge row ids are never read
- hidden coordinates 0:2048 and the spread coordinates ignore query targets
- coordinate 2048 changes when only query targets change
- masking 2048–2050 is labelled "not decontamination"
"""

import math
import struct

from rlfw.arena.game_spec import digest

VECTOR_DIM = 2051
SELECTION_NOTE = "fixture-abs-correlation-stand-in; XGBoost was not run"
NOT_DECONTAMINATION = "not decontamination"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _mean(values):
    return sum(values) / len(values)


def _population_std(values):
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _corr(xs, ys):
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _percentile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def context_query_split(fit_row_ids):
    """Packet §2.3 ratio, applied to the fit pool only: about 3/4 context."""
    ids = list(fit_row_ids)
    if len(ids) < 4:
        raise ValueError("fit pool needs at least four rows so context and query both exist")
    n_context = min(max(1, (3 * len(ids)) // 4), len(ids) - 1)
    return ids[:n_context], ids[n_context:]


def _column(table, row_ids, index):
    return [table[row_id]["features"][index] for row_id in row_ids]


def _targets(table, row_ids):
    return [table[row_id]["target"] for row_id in row_ids]


def _rank(table, fit_row_ids, width):
    targets = _targets(table, fit_row_ids)
    scored = []
    for index in range(width):
        column = _column(table, fit_row_ids, index)
        variance = _population_std(column) ** 2
        scored.append((abs(_corr(column, targets)), variance, -index, index))
    scored.sort(reverse=True)
    return [item[3] for item in scored]


def _scale(value, lo, span):
    if span <= 1e-12:
        return 0.0
    return (value - lo) / span


def fit_chain(table, fit_row_ids, *, judge_row_ids, proposals, mask_diagnostics=False):
    """Build the 2051-d stand-in from fit rows. Judge ids are recorded and unread."""
    fit_ids = list(fit_row_ids)
    judge_ids = list(judge_row_ids)
    if len(set(fit_ids)) != len(fit_ids) or len(set(judge_ids)) != len(judge_ids):
        raise ValueError("duplicate row id")
    if set(fit_ids) & set(judge_ids):
        raise ValueError("judge rows leaked into the fit pool")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("proposals must be a non-empty list")
    sample = table[fit_ids[0]]["features"]
    width = len(sample)
    if width < 1 or any(len(table[row_id]["features"]) != width for row_id in fit_ids):
        raise ValueError("fit rows must share one feature width")
    context_ids, query_ids = context_query_split(fit_ids)
    selected = _rank(table, fit_ids, width)
    mins, spans = [], []
    for index in selected:
        column = _column(table, fit_ids, index)
        lo, hi = min(column), max(column)
        span = hi - lo
        if not math.isfinite(span) or span <= 1e-12:
            span = 1.0
        mins.append(lo)
        spans.append(span)
    context_matrix = []
    for row_id in context_ids:
        features = table[row_id]["features"]
        context_matrix.append([
            _scale(features[index], lo, span) for index, lo, span in zip(selected, mins, spans, strict=True)
        ])
    context_targets = _targets(table, context_ids)
    target_mean = _mean(context_targets)
    target_std = _population_std(context_targets) + 1e-8
    feature_mean = [_mean([row[col] for row in context_matrix]) for col in range(len(selected))]
    feature_std = [
        _population_std([row[col] for row in context_matrix]) + 1e-8 for col in range(len(selected))
    ]

    def standardized(row_id):
        features = table[row_id]["features"]
        scaled = [
            _scale(features[index], lo, span) for index, lo, span in zip(selected, mins, spans, strict=True)
        ]
        return [(value - center) / scale for value, center, scale in zip(scaled, feature_mean, feature_std, strict=True)]

    vector = [0.0] * VECTOR_DIM
    for slot, _index in enumerate(selected):
        column = [standardized(row_id)[slot] for row_id in context_ids]
        mean = _mean(column)
        std = _population_std(column)
        p10 = _percentile(column, 0.1)
        p90 = _percentile(column, 0.9)
        base = (slot * 4) % 512
        vector[base] += mean
        vector[512 + base] += std
        vector[1024 + base] += p10
        vector[1536 + base] += p90
    query_std = [standardized(row_id) for row_id in query_ids]
    predictions = [_mean(row) for row in query_std]
    query_targets = _targets(table, query_ids)
    q_mean = _mean(query_targets)
    q_std = _population_std(query_targets) + 1e-8
    z = [(target - q_mean) / q_std for target in query_targets]
    sse = sum((zi - pred) ** 2 for zi, pred in zip(z, predictions, strict=True))
    sst = sum(zi * zi for zi in z) + 1e-12
    skill = max(-10.0, 1.0 - sse / sst)
    spread = _population_std(predictions) if len(predictions) > 1 else 0.0
    vector[2048] = skill
    vector[2049] = spread
    vector[2050] = math.log(spread + 1e-6)
    diagnostic = (vector[2048], vector[2049], vector[2050])
    diagnostic_l2 = math.sqrt(sum(value * value for value in diagnostic))
    if mask_diagnostics:
        vector[2048] = 0.0
        vector[2049] = 0.0
        vector[2050] = 0.0
    vector = [f32(value) for value in vector]
    if len(vector) != VECTOR_DIM:
        raise ValueError("vector width")
    weights = {
        "feature_mean": [f32(value) for value in feature_mean],
        "feature_min": [f32(value) for value in mins],
        "feature_std": [f32(value) for value in feature_std],
        "fit_row_ids": fit_ids,
        "selected_indices": selected,
        "span": [f32(value) for value in spans],
        "target_mean": f32(target_mean),
        "target_std": f32(target_std),
    }
    optimization_score = f32(_mean(context_targets))
    cache_key = digest(
        {
            "mask_diagnostics": mask_diagnostics,
            "proposals": list(proposals),
            "vector": vector,
            "weights": weights,
        }
    )
    return {
        "cache_key": cache_key,
        "context_row_ids": context_ids,
        "diagnostic_l2": diagnostic_l2,
        "diagnostic_label": NOT_DECONTAMINATION if mask_diagnostics else None,
        "fit_row_ids": fit_ids,
        "judge_row_ids_ignored": judge_ids,
        "not_decontamination": bool(mask_diagnostics),
        "optimization_score": optimization_score,
        "proposals": list(proposals),
        "query_row_ids": query_ids,
        "selected_indices": selected,
        "selection_implementation": SELECTION_NOTE,
        "vector": vector,
        "weights": weights,
    }
