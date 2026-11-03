from __future__ import annotations

import math
from typing import Any


def rank_candidates(valid, n, stability_threshold, selection_key):
    """Final candidate ordering. BUG-A fix (all 7 external reviews): the deployed
    rank-0 slot is SACRED — always the global selection_key argmax. The engine
    diversity floor previously got PREPENDED (engines alphabetical → featmath
    preempted a better newmath candidate at slot 0, structurally defeating the
    routing objective and poisoning self-play credit assignment). Diversity now
    only fills slots >= 1."""
    ranked_order = sorted(range(len(valid)), key=selection_key, reverse=True)
    top = ranked_order[0]
    floor: list[int] = []
    if n >= len({row["engine"] for row in valid}):
        for engine in sorted({row["engine"] for row in valid}):
            if valid[top]["engine"] == engine:
                continue  # already represented by the global best at slot 0
            viable = [
                i
                for i in ranked_order
                if valid[i]["engine"] == engine
                and i != top
                and i not in floor
                and float(valid[i].get("p_bad") or 0.0) <= stability_threshold
                and not valid[i].get("candidate_risk_reason")
            ]
            if viable:
                floor.append(viable[0])
    return [top] + floor + [i for i in ranked_order if i != top and i not in floor]


def _rank_candidates_with_reranker(
    valid: list[dict[str, Any]],
    n: int,
    stability_threshold: float,
) -> list[int]:
    """Selector 2.0 portfolio: top reranker rows first, then uncertainty/diversity/engine slots."""
    limit = max(0, int(n))
    base_order = sorted(
        range(len(valid)),
        key=lambda i: (
            _row_numeric(
                valid[i], "selector_reranker_score", default=_row_numeric(valid[i], "selection_key")
            ),
            _row_numeric(valid[i], "selection_key"),
            _row_numeric(valid[i], "reward"),
            -int(
                valid[i].get("proposal_index") if isinstance(valid[i].get("proposal_index"), int) else 10**9
            ),
        ),
        reverse=True,
    )
    if limit <= 0:
        return base_order
    selected: list[int] = []
    selected_set: set[int] = set()

    def safe(i: int) -> bool:
        return (
            not valid[i].get("candidate_risk_reason")
            and _row_numeric(valid[i], "p_bad", default=0.0) <= stability_threshold
        )

    def add(i: int) -> bool:
        if i in selected_set:
            return False
        selected.append(i)
        selected_set.add(i)
        return len(selected) >= limit

    for i in base_order:
        if safe(i) and add(i) and len(selected) >= min(3, limit):
            break
        if len(selected) >= min(3, limit):
            break
    if len(selected) >= limit:
        return selected + [i for i in base_order if i not in selected_set]

    uncertainty = sorted(
        [i for i in base_order if i not in selected_set and safe(i)],
        key=lambda i: (_row_numeric(valid[i], "reward_std"), _row_numeric(valid[i], "selection_key")),
        reverse=True,
    )
    if uncertainty and add(uncertainty[0]):
        return selected + [i for i in base_order if i not in selected_set]

    seen_diversity = {str(valid[i].get("candidate_diversity_key") or "") for i in selected}
    diverse = [
        i
        for i in base_order
        if i not in selected_set
        and safe(i)
        and bool(valid[i].get("novel_vs_historical", True))
        and str(valid[i].get("candidate_diversity_key") or "") not in seen_diversity
    ]
    if diverse and add(diverse[0]):
        return selected + [i for i in base_order if i not in selected_set]

    engine_counts: dict[str, int] = {}
    for i in selected:
        engine = str(valid[i].get("engine") or "")
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
    engines = sorted({str(row.get("engine") or "") for row in valid})
    underrepresented = sorted(engines, key=lambda e: (engine_counts.get(e, 0), e))
    for engine in underrepresented:
        bucket = [
            i
            for i in base_order
            if i not in selected_set and safe(i) and str(valid[i].get("engine") or "") == engine
        ]
        if bucket and add(bucket[0]):
            return selected + [i for i in base_order if i not in selected_set]

    for i in base_order:
        if add(i):
            break
    return selected + [i for i in base_order if i not in selected_set]


def _candidate_diversity_key_v3(
    actions: dict[str, Any],
    head_type: Any,
    family_bitmap: Any,
) -> str:
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    first = str(transforms[0].get("op") if transforms else "none")
    head = str(
        head_type
        or ((actions.get("head") or {}).get("op") if isinstance(actions.get("head"), dict) else "")
        or "unknown"
    )
    families = family_bitmap if isinstance(family_bitmap, dict) else {}
    active = sorted(str(k) for k, v in families.items() if v)
    family = active[0] if active else first
    return f"{head}:{family}"


def _row_numeric(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default
