from __future__ import annotations

from typing import Any


_EXPLOSIVE = {
    "featmath": {"cheb", "poly_interact", "pair_prod", "bin_onehot", "rff_cos", "rff_sin"},
    "newmath": {
        "cheb",
        "poly_interact",
        "rff_cos",
        "rff_sin",
        "phase_interference",
        "iterate",
        "logistic_map",
    },
}

def candidate_risk_reason_v3(engine: str, actions: dict[str, Any]) -> str | None:
    counts: dict[str, int] = {}
    for stage in actions.get("transforms") or []:
        op = str(stage.get("op"))
        counts[op] = counts.get(op, 0) + 1
    for op, count in counts.items():
        if count >= 3 and op in _EXPLOSIVE.get(engine, set()):
            return f"repeated_explosive_transform:{op}"
    return None

