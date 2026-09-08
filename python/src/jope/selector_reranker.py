from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl

SCHEMA_VERSION = "selector_reranker_v1"

_NUMERIC_FEATURES = {
    "reward",
    "reward_mean",
    "reward_std",
    "reward_lcb",
    "reward_percentile_in_engine",
    "p_bad",
    "selection_score",
    "calibrated_expected_delta",
    "calibrated_win_rate",
    "calibrated_tail_risk",
    "transform_count",
}

_BASE_FEATURES = (
    "reward_mean",
    "reward_std",
    "reward_lcb",
    "reward_percentile_in_engine",
    "p_bad",
    "selection_score",
    "calibrated_expected_delta",
    "calibrated_win_rate",
    "calibrated_tail_risk",
)


def load_selector_reranker(path_or_data: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if path_or_data is None:
        return None
    if isinstance(path_or_data, dict):
        data = path_or_data
    else:
        data = json.loads(Path(path_or_data).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported selector reranker schema: {data.get('schema_version')!r}")
    names = data.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError("selector reranker artifact requires feature_names")
    return data


def score_candidate_with_reranker(row: dict[str, Any], reranker: dict[str, Any]) -> float | None:
    names = [str(name) for name in reranker.get("feature_names") or []]
    if not names:
        return None
    x = np.asarray([_feature_value(row, name) for name in names], dtype=np.float64)
    mean = np.asarray(reranker.get("feature_mean") or [0.0] * len(names), dtype=np.float64)
    scale = np.asarray(reranker.get("feature_scale") or [1.0] * len(names), dtype=np.float64)
    coef = np.asarray(reranker.get("coef") or [], dtype=np.float64)
    if mean.shape[0] != len(names) or scale.shape[0] != len(names) or coef.shape[0] != len(names):
        return None
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    intercept = _finite_float(reranker.get("intercept")) or 0.0
    score = float(intercept + ((x - mean) / scale) @ coef)
    return score if math.isfinite(score) else None


def train_selector_reranker(
    candidate_pool_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    l2: float = 1e-3,
) -> dict[str, Any]:
    rows = _load_measured_pool_rows(candidate_pool_jsonl)
    if not rows:
        raise ValueError("no measured candidate-pool rows found")
    feature_names = _feature_names(rows)
    x = np.asarray([[_feature_value(row, name) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["score_robust"]) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
    z = (x - mean) / scale
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "linear_ridge",
        "target": "score_robust",
        "feature_names": feature_names,
        "feature_mean": [float(v) for v in mean.tolist()],
        "feature_scale": [float(v) for v in scale.tolist()],
        "intercept": float(coef[0]),
        "coef": [float(v) for v in coef[1:].tolist()],
        "fit": {
            "row_count": len(rows),
            "dataset_count": len({str(row.get("dataset_hash")) for row in rows}),
            "l2": float(l2),
        },
    }
    artifact["replay"] = replay_selector_reranker(rows, artifact)
    return artifact


def write_selector_reranker(artifact: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replay_selector_reranker(
    rows_or_paths: list[dict[str, Any]] | str | Path | list[str | Path] | tuple[str | Path, ...],
    reranker: dict[str, Any],
    *,
    k: int = 6,
) -> dict[str, Any]:
    rows = rows_or_paths if isinstance(rows_or_paths, list) and (not rows_or_paths or isinstance(rows_or_paths[0], dict)) else _load_measured_pool_rows(rows_or_paths)  # type: ignore[arg-type]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:  # type: ignore[union-attr]
        groups[str(row.get("dataset_hash"))].append(row)
    current_regrets: list[float] = []
    reranker_regrets: list[float] = []
    captures: list[int] = []
    ndcg_values: list[float] = []
    p_bad_pairs: list[tuple[float, float]] = []
    for group in groups.values():
        measured = [row for row in group if _finite_float(row.get("score_robust")) is not None]
        if not measured:
            continue
        best = max(float(row["score_robust"]) for row in measured)
        current = min(
            measured,
            key=lambda row: int(row.get("selected_for_eval_rank") if isinstance(row.get("selected_for_eval_rank"), int) else 10**9),
        )
        scored = []
        for row in measured:
            score = score_candidate_with_reranker(row, reranker)
            scored.append((score if score is not None else float("-inf"), row))
            p_bad = _finite_float(row.get("p_bad"))
            if p_bad is not None:
                p_bad_pairs.append((p_bad, float(row["score_robust"])))
        ranked = [row for _score, row in sorted(scored, key=lambda item: item[0], reverse=True)]
        selected = ranked[0]
        current_regrets.append(best - float(current["score_robust"]))
        reranker_regrets.append(best - float(selected["score_robust"]))
        captures.append(1 if any(float(row["score_robust"]) == best for row in ranked[: max(1, int(k))]) else 0)
        ndcg_values.append(_ndcg_at_k(ranked, k=max(1, int(k))))
    return {
        "dataset_count": len(current_regrets),
        "current_regret_at_1": _mean(current_regrets),
        "reranker_regret_at_1": _mean(reranker_regrets),
        "regret_improvement": (
            (_mean(current_regrets) - _mean(reranker_regrets))
            if _mean(current_regrets) is not None and _mean(reranker_regrets) is not None
            else None
        ),
        "capture_at_6": _mean(captures),
        "ndcg_at_6": _mean(ndcg_values),
        "p_bad_score_correlation": _correlation(p_bad_pairs),
    }


def _feature_value(row: dict[str, Any], name: str) -> float:
    if name in _NUMERIC_FEATURES:
        if name == "reward_mean":
            return _finite_float(row.get("reward_mean", row.get("reward"))) or 0.0
        if name == "reward_lcb":
            reward = _finite_float(row.get("reward")) or 0.0
            std = _finite_float(row.get("reward_std")) or 0.0
            return reward - std
        return _finite_float(row.get(name)) or 0.0
    if name.startswith("engine="):
        return 1.0 if str(row.get("engine") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("head="):
        return 1.0 if str(row.get("head_type") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("decoder="):
        return 1.0 if str(row.get("decoder") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("family="):
        family = name.split("=", 1)[1]
        bitmap = row.get("family_bitmap") if isinstance(row.get("family_bitmap"), dict) else {}
        return 1.0 if bool(bitmap.get(family)) else 0.0
    return 0.0


def _load_measured_pool_rows(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _iter_paths(paths):
        for row in read_jsonl(path):
            score = _finite_float(row.get("score_robust"))
            if score is None:
                continue
            row = dict(row)
            row["score_robust"] = score
            out.append(row)
    return out


def _iter_paths(paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    engines = sorted({str(row.get("engine") or "") for row in rows if row.get("engine")})
    heads = sorted({str(row.get("head_type") or "") for row in rows if row.get("head_type")})
    decoders = sorted({str(row.get("decoder") or "") for row in rows if row.get("decoder")})
    families = sorted({
        str(k)
        for row in rows
        for k, v in (row.get("family_bitmap") if isinstance(row.get("family_bitmap"), dict) else {}).items()
        if v
    })
    names = list(_BASE_FEATURES)
    names.extend(f"engine={engine}" for engine in engines)
    names.extend(f"head={head}" for head in heads)
    names.extend(f"decoder={decoder}" for decoder in decoders)
    names.extend(f"family={family}" for family in families)
    return names


def _ndcg_at_k(rows: list[dict[str, Any]], *, k: int) -> float | None:
    gains = [max(float(row["score_robust"]), 0.0) for row in rows[:k] if _finite_float(row.get("score_robust")) is not None]
    ideal = sorted([max(float(row["score_robust"]), 0.0) for row in rows], reverse=True)[:k]
    if not gains or not ideal:
        return None
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else None


def _mean(values: list[float] | list[int] | list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else None


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None
