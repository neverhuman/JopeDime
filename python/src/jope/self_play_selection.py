"""Selection and candidate-record helpers for the self-play loop."""

from __future__ import annotations

import hashlib
from typing import Any

def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def deployment_elo(row: dict[str, Any]) -> float | None:
    """Canonical deployed Elo: cheap-probe first, then CV-select, then legacy top1."""
    for key in ("elo_probe", "elo_cv", "elo_top1"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _resolve_ship_decision(
    gate_shipped: bool, elo: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    prior = [deployment_elo(h) for h in history if h.get("shipped") and deployment_elo(h) is not None]
    cur = deployment_elo(elo)
    baseline = max(prior) if prior else None
    elo_record = bool(cur is not None and (baseline is None or cur >= baseline))
    elo_regressed = bool(cur is not None and baseline is not None and cur < baseline)
    return {
        "shipped": (bool(gate_shipped) and not elo_regressed) or elo_record,
        "elo_record": elo_record,
        "deployed_elo_baseline": baseline,
        "deployed_elo_delta": (cur - baseline) if cur is not None and baseline is not None else None,
        "deployed_elo_regressed": elo_regressed,
    }


def _choose_deployed_candidate(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in recs if _finite_float(r.get("score_robust")) is not None]
    if not scored:
        return None
    return max(
        scored,
        key=lambda r: (
            _finite_float(r.get("score_robust")) or float("-inf"),
            _finite_float(r.get("_oof")) or float("-inf"),
        ),
    )


def _proposal_failure_exceeded(failed: int, total: int, threshold: float | None) -> bool:
    if total <= 0 or threshold is None:
        return False
    return failed / max(1, total) > float(threshold)


def _candidate_pool_id(generation: int, dataset_hash: str, engine: str, canonical: str) -> str:
    raw = f"{generation}|{dataset_hash}|{engine}|{canonical}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _candidate_pool_record(
    *,
    generation: int,
    candidate_pool_id: str,
    policy_checkpoint: str,
    dataset_hash: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    reward_mean = _finite_float(row.get("reward_mean", row.get("reward")))
    reward_std = _finite_float(row.get("reward_std"))
    reward_lcb = _finite_float(row.get("reward_lcb"))
    if reward_lcb is None and reward_mean is not None and reward_std is not None:
        reward_lcb = reward_mean - reward_std
    final_key = row.get("selector_reranker_score")
    if final_key is None:
        final_key = row.get("selection_key")
    return {
        "generation": generation,
        "candidate_pool_id": candidate_pool_id,
        "policy_checkpoint": policy_checkpoint,
        "dataset_hash": dataset_hash,
        "engine": row.get("engine"),
        "canonical_genome": row.get("canonical_genome") or row.get("genome"),
        "decoder": row.get("decoder"),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_lcb": reward_lcb,
        "reward_percentile_in_engine": row.get("reward_percentile_in_engine"),
        "p_bad": row.get("p_bad"),
        "candidate_risk_reason": row.get("candidate_risk_reason"),
        "family_bitmap": row.get("family_bitmap"),
        "head_type": row.get("head_type"),
        "candidate_diversity_key": row.get("candidate_diversity_key"),
        "selection_key": row.get("selection_key"),
        "selection_score": row.get("selection_score"),
        "selector_reranker_score": row.get("selector_reranker_score"),
        "final_selection_key": final_key,
        "selected_for_eval_rank": None,
        "measured_oof": None,
        "score_robust": None,
        "cost_ms": None,
        "deployed": False,
    }
