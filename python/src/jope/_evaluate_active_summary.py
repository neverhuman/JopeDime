from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .data.active import parse_pipeline_actions
from .eval_ledger import EVALUATOR_VERSION


def _summary(
    proposal_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in proposal_rows if row.get("valid")]
    novel = [row for row in proposal_rows if row.get("novel")]
    invalid_reasons = Counter(
        str(error) for row in proposal_rows if not row.get("valid") for error in (row.get("errors") or [])
    )
    deltas = [
        float(row["oof_delta_vs_incumbent"])
        for row in evaluation_rows
        if isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and math.isfinite(float(row["oof_delta_vs_incumbent"]))
    ]
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in evaluation_rows:
        delta = row.get("oof_delta_vs_incumbent")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            by_dataset[str(row.get("dataset_hash"))].append(float(delta))
    best_of_n = [max(values) for values in by_dataset.values() if values]
    test_r2s = [
        float(row["test_r2"])
        for row in evaluation_rows
        if isinstance(row.get("test_r2"), (int, float)) and math.isfinite(float(row["test_r2"]))
    ]
    heads = Counter(
        str(row.get("head_type") or _head_from_genome(row.get("canonical_genome") or "")) for row in valid
    )
    eval_skip_reasons = Counter(
        str(row.get("evaluation_skip_reason")) for row in proposal_rows if row.get("evaluation_skip_reason")
    )
    eval_skipped_datasets = {
        str(row.get("dataset_hash"))
        for row in proposal_rows
        if row.get("evaluation_skip_reason") and row.get("dataset_hash") is not None
    }
    gp_evals = [
        row
        for row in evaluation_rows
        if (row.get("head_type") == "gpr" or "(gpr" in str(row.get("canonical_genome") or ""))
    ]
    reward_pairs = [
        (float(row["reward"]), float(row["oof_delta_vs_incumbent"]))
        for row in evaluation_rows
        if isinstance(row.get("reward"), (int, float))
        and isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and math.isfinite(float(row["reward"]))
        and math.isfinite(float(row["oof_delta_vs_incumbent"]))
    ]
    evaluator_versions = Counter(str(row.get("evaluator_version") or "unknown") for row in evaluation_rows)
    return {
        "proposal_count": len(proposal_rows),
        "proposal_valid_count": len(valid),
        "proposal_valid_rate": len(valid) / max(1, len(proposal_rows)),
        "proposal_novel_count": len(novel),
        "proposal_novel_rate": len(novel) / max(1, len(proposal_rows)),
        "invalid_reasons": dict(invalid_reasons.most_common(8)),
        "evaluated_count": len(evaluation_rows),
        "cache_hit_count": sum(1 for row in evaluation_rows if row.get("cache_hit")),
        "cache_hit_rate": sum(1 for row in evaluation_rows if row.get("cache_hit"))
        / max(1, len(evaluation_rows)),
        "novel_eval_count": sum(1 for row in evaluation_rows if not row.get("cache_hit")),
        "eval_skipped_dataset_count": len(eval_skipped_datasets),
        "eval_skip_reasons": dict(eval_skip_reasons.most_common()),
        "mean_delta_vs_incumbent": _mean(deltas),
        "mean_delta_vs_incumbent_clipped": _mean([max(d, -1.0) for d in deltas]),
        "median_delta_vs_incumbent": _median(deltas),
        "win_rate_vs_incumbent": sum(1 for delta in deltas if delta > 0.0) / max(1, len(deltas)),
        "best_of_n_mean_delta": _mean(best_of_n),
        "best_of_n_median_delta": _median(best_of_n),
        "holdout_test_eval_count": len(test_r2s),
        "mean_test_r2": _mean(test_r2s),
        "median_test_r2": _median(test_r2s),
        "head_distribution": dict(heads),
        "head_diversity": len([head for head, count in heads.items() if count > 0 and head != "None"]),
        "gp_proposal_count": sum(
            1
            for row in valid
            if row.get("head_type") == "gpr" or "(gpr" in str(row.get("canonical_genome") or "")
        ),
        "gp_eval_count": len(gp_evals),
        "gp_fold_r2": [
            ((row.get("cv") or {}).get("fold_r2")) for row in gp_evals if isinstance(row.get("cv"), dict)
        ],
        "reward_actual_correlation": _correlation(
            [pair[0] for pair in reward_pairs], [pair[1] for pair in reward_pairs]
        ),
        "head_outcomes": _head_outcomes(evaluation_rows),
        "worst_reward_outliers": _worst_reward_outliers(evaluation_rows),
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_versions": dict(evaluator_versions),
        "selected_dataset_count": len(selected),
        "per_engine": _per_engine_summary(proposal_rows, evaluation_rows),
    }


def _per_engine_summary(
    proposal_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in proposal_rows:
        engine = str(row.get("engine") or "featmath")
        entry = out.setdefault(
            engine,
            {"proposals": 0, "valid": 0, "novel": 0, "evaluated": 0, "wins": 0, "deltas": []},
        )
        entry["proposals"] += 1
        entry["valid"] += int(bool(row.get("valid")))
        entry["novel"] += int(bool(row.get("novel")))
    for row in evaluation_rows:
        engine = str(row.get("engine") or "featmath")
        entry = out.setdefault(
            engine,
            {"proposals": 0, "valid": 0, "novel": 0, "evaluated": 0, "wins": 0, "deltas": []},
        )
        entry["evaluated"] += 1
        delta = row.get("oof_delta_vs_incumbent")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            entry["deltas"].append(float(delta))
            entry["wins"] += int(float(delta) > 0.0)
    for entry in out.values():
        deltas = entry.pop("deltas")
        entry["valid_rate"] = entry["valid"] / max(1, entry["proposals"])
        entry["win_rate_vs_incumbent"] = entry["wins"] / max(1, entry["evaluated"])
        entry["median_delta_vs_incumbent"] = _median(deltas)
    return out


def _head_from_genome(genome: str) -> str | None:
    try:
        return str(parse_pipeline_actions(genome)["head"]["op"])
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _head_outcomes(evaluation_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_head: dict[str, list[float]] = defaultdict(list)
    for row in evaluation_rows:
        delta = row.get("oof_delta_vs_incumbent")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            continue
        head = str(row.get("head_type") or _head_from_genome(row.get("canonical_genome") or "") or "unknown")
        by_head[head].append(float(delta))
    return {
        head: {
            "count": len(values),
            "wins": sum(1 for value in values if value > 0.0),
            "win_rate": sum(1 for value in values if value > 0.0) / max(1, len(values)),
            "median_delta": _median(values),
        }
        for head, values in sorted(by_head.items())
    }


def _worst_reward_outliers(evaluation_rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        row
        for row in evaluation_rows
        if isinstance(row.get("reward"), (int, float))
        and isinstance(row.get("oof_delta_vs_incumbent"), (int, float))
        and float(row["oof_delta_vs_incumbent"]) < 0.0
    ]
    rows.sort(
        key=lambda row: (float(row.get("reward") or 0.0), -float(row.get("oof_delta_vs_incumbent") or 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        out.append(
            {
                "dataset_hash": row.get("dataset_hash"),
                "head_type": row.get("head_type") or _head_from_genome(row.get("canonical_genome") or ""),
                "reward": row.get("reward"),
                "oof_delta_vs_incumbent": row.get("oof_delta_vs_incumbent"),
                "canonical_genome": row.get("canonical_genome"),
            }
        )
    return out
