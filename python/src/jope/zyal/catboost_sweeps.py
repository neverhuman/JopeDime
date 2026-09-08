"""Exhaustive, leakage-bounded CatBoost screening for very wide feature banks.

Every candidate appears exactly once in each of three independently shuffled
sweeps.  Models only see 100--300 candidate columns at a time, plus a stable
anchor bridge and matched synthetic-noise controls.  Model importances are
normalized to one, noise-subtracted, calibrated across windows, and then
renormalized across the complete sweep before the three estimates are averaged.

The final feature count is chosen on a tuning partition.  A disjoint grouped
holdout is opened once for the reported out-of-sample number; it never affects
importance estimates, cut interpolation, or hyperparameters.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class SweepError(RuntimeError):
    """The exhaustive sweep contract was violated."""


class CatBoostLike(Protocol):
    def fit(self, features: np.ndarray, target: np.ndarray, **kwargs: Any) -> Any: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    def get_feature_importance(self, **kwargs: Any) -> np.ndarray: ...


ModelFactory = Callable[[int, "SweepConfig"], CatBoostLike]


@dataclass(frozen=True)
class SweepConfig:
    sweeps: int = 3
    minimum_window: int = 100
    maximum_window: int = 300
    sweep_validation_fraction: float = 0.20
    tuning_fraction: float = 0.15
    final_holdout_fraction: float = 0.15
    anchor_count: int = 64
    noise_columns: int = 8
    iterations: int = 160
    depth: int = 7
    learning_rate: float = 0.08
    l2_leaf_reg: float = 5.0
    thread_count: int = 8
    percentile_cuts: tuple[float, ...] = (50.0, 80.0, 90.0, 95.0)
    interpolation_points: int = 3
    plateau_tolerance: float = 0.002
    seed: int = 20260720

    def validate(self) -> None:
        if self.sweeps != 3:
            raise SweepError("the production screen requires exactly three complete sweeps")
        if not 1 <= self.minimum_window <= self.maximum_window:
            raise SweepError("invalid CatBoost window bounds")
        if self.minimum_window != 100 or self.maximum_window != 300:
            raise SweepError("production CatBoost windows must stay in 100..300")
        if not 0.0 < self.sweep_validation_fraction < 0.5:
            raise SweepError("sweep validation fraction must be in (0, 0.5)")
        if self.tuning_fraction <= 0 or self.final_holdout_fraction <= 0:
            raise SweepError("tuning and final grouped holdouts must be nonempty")
        if self.tuning_fraction + self.final_holdout_fraction >= 0.5:
            raise SweepError("at least half of the groups must remain for discovery")
        if self.anchor_count != 64:
            raise SweepError("the cross-window calibration bridge requires 64 anchors")
        if self.noise_columns < 2:
            raise SweepError("each window requires at least two synthetic-noise controls")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _hash_strings(values: Sequence[str]) -> str:
    return _canonical_hash(list(map(str, values)))


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Create a self-hashed JSON file once; return whether this process won."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = dict(payload)
    sealed["payload_sha256"] = _canonical_hash(sealed)
    encoded = json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise SweepError(f"sealed CatBoost artifact collision: {path}")
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _read_sealed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("payload_sha256", None)
    if expected != _canonical_hash(payload):
        raise SweepError(f"sealed CatBoost artifact was modified: {path}")
    return payload


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else 0.0


def ranking_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    metric_groups: np.ndarray,
) -> dict[str, float]:
    """Tie-aware metrics, averaged at the honest within-group authority."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    metric_groups = np.asarray(metric_groups)
    correlations: list[float] = []
    captures: list[float] = []
    regrets: list[float] = []
    for group in np.unique(metric_groups):
        index = np.flatnonzero(metric_groups == group)
        if len(index) < 3:
            continue
        group_prediction = prediction[index]
        group_target = target[index]
        correlations.append(_pearson(_ranks(group_prediction), _ranks(group_target)))
        order = np.argsort(-group_prediction, kind="stable")
        best = float(group_target.max())
        captures.append(float(float(group_target[order[:3]].max()) >= best - 1e-12))
        regrets.append(best - float(group_target[order[0]]))
    return {
        "within_group_spearman": float(np.mean(correlations)) if correlations else 0.0,
        "capture_at_3": float(np.mean(captures)) if captures else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "percentile_pearson": _pearson(prediction, target),
    }


def grouped_partition(
    groups: np.ndarray,
    *,
    seed: int,
    tuning_fraction: float,
    final_fraction: float,
) -> dict[str, np.ndarray]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 10:
        raise SweepError("at least ten split groups are required")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    final_count = max(1, int(round(len(unique) * final_fraction)))
    tuning_count = max(1, int(round(len(unique) * tuning_fraction)))
    final_groups = set(shuffled[:final_count].tolist())
    tuning_groups = set(shuffled[final_count : final_count + tuning_count].tolist())
    discovery_groups = set(shuffled[final_count + tuning_count :].tolist())
    partitions = {
        "discovery": np.asarray([item in discovery_groups for item in groups], dtype=bool),
        "tuning": np.asarray([item in tuning_groups for item in groups], dtype=bool),
        "final": np.asarray([item in final_groups for item in groups], dtype=bool),
    }
    if any(not mask.any() for mask in partitions.values()):
        raise SweepError("a grouped screening partition is empty")
    if sum(mask.astype(np.int8) for mask in partitions.values()).max() != 1:
        raise SweepError("grouped screening partitions overlap")
    return partitions


def grouped_random_80_20(
    groups: np.ndarray, eligible: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    eligible = np.asarray(eligible, dtype=bool)
    unique = np.unique(groups[eligible])
    if len(unique) < 5:
        raise SweepError("a sweep needs at least five discovery groups")
    rng = np.random.default_rng(seed)
    validation_count = max(1, int(round(0.20 * len(unique))))
    validation_groups = set(unique[rng.permutation(len(unique))[:validation_count]].tolist())
    validation = eligible & np.asarray([item in validation_groups for item in groups])
    train = eligible & ~validation
    if set(groups[train]) & set(groups[validation]):
        raise SweepError("a dataset crossed the random 80/20 sweep split")
    return train, validation


def window_partition(
    feature_count: int, *, seed: int, minimum: int = 100, maximum: int = 300
) -> list[np.ndarray]:
    """Shuffle all indices, then partition with no underfilled tail."""

    if feature_count <= 0:
        raise SweepError("no candidate features were supplied")
    rng = np.random.default_rng(seed)
    order = rng.permutation(feature_count)
    if feature_count < minimum:
        return [order]
    target = int(rng.integers(minimum, maximum + 1))
    window_count = int(round(feature_count / target))
    minimum_windows = math.ceil(feature_count / maximum)
    maximum_windows = feature_count // minimum
    window_count = min(max(window_count, minimum_windows), maximum_windows)
    sizes = np.full(window_count, feature_count // window_count, dtype=np.int64)
    sizes[: feature_count % window_count] += 1
    if sizes.min() < minimum or sizes.max() > maximum:
        raise SweepError(f"could not partition {feature_count} features into 100..300 windows")
    offsets = np.cumsum(np.concatenate(([0], sizes)))
    return [order[offsets[i] : offsets[i + 1]] for i in range(window_count)]


def _default_model_factory(seed: int, config: SweepConfig) -> CatBoostLike:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover - deployment environment dependency
        raise SweepError(
            "CatBoost is required for exhaustive feature sweeps; install the feature-foundry extra"
        ) from error
    return CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        random_strength=1.0,
        rsm=0.8,
        thread_count=config.thread_count,
        verbose=False,
        allow_writing_files=False,
    )


def _normalised_importance(model: CatBoostLike, feature_count: int) -> np.ndarray:
    raw = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64
    )
    if raw.shape != (feature_count,) or not np.isfinite(raw).all():
        raise SweepError("CatBoost returned malformed feature importance")
    raw = np.maximum(raw, 0.0)
    total = float(raw.sum())
    if total <= 0:
        raise SweepError("CatBoost returned zero total feature importance")
    result = raw / total
    result /= result.sum()
    if abs(float(result.sum()) - 1.0) > 1e-12:
        raise SweepError("normalized CatBoost feature importance does not add to one")
    return result


def _anchor_selection(
    anchors: np.ndarray | None,
    anchor_names: Sequence[str] | None,
    discovery: np.ndarray,
    count: int,
) -> tuple[np.ndarray, list[str]]:
    if anchors is None:
        return np.empty((len(discovery), 0), dtype=np.float32), []
    anchors = np.asarray(anchors)
    names = list(map(str, anchor_names or []))
    if anchors.ndim != 2 or anchors.shape[0] != len(discovery) or len(names) != anchors.shape[1]:
        raise SweepError("anchor matrix/name shape mismatch")
    if len(set(names)) != len(names):
        raise SweepError("anchor names are not unique")
    variances = np.var(np.asarray(anchors[discovery], dtype=np.float64), axis=0)
    eligible = [index for index, variance in enumerate(variances) if variance > 0]
    eligible.sort(key=lambda index: (-float(variances[index]), names[index]))
    chosen = eligible[:count]
    if len(chosen) != count:
        raise SweepError(f"expected {count} nonconstant anchors, found {len(chosen)}")
    return np.asarray(anchors[:, chosen], dtype=np.float32), [names[index] for index in chosen]


def _matched_noise(
    candidate_window: np.ndarray,
    split_groups: np.ndarray,
    *,
    seed: int,
    columns: int,
) -> tuple[np.ndarray, list[str]]:
    """Within-group shuffles preserve marginal scale without preserving signal."""

    candidate_window = np.asarray(candidate_window, dtype=np.float32)
    split_groups = np.asarray(split_groups)
    rng = np.random.default_rng(seed)
    noise = np.empty((len(candidate_window), columns), dtype=np.float32)
    for noise_index in range(columns):
        source = noise_index % candidate_window.shape[1]
        values = candidate_window[:, source].copy()
        for group in np.unique(split_groups):
            rows = np.flatnonzero(split_groups == group)
            values[rows] = values[rows][rng.permutation(len(rows))]
        noise[:, noise_index] = values
    names = [f"__synthetic_noise__{seed:010d}_{index:02d}" for index in range(columns)]
    return noise, names


def _fit_window(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    metric_groups: np.ndarray,
    *,
    seed: int,
    config: SweepConfig,
    model_factory: ModelFactory,
) -> tuple[np.ndarray, dict[str, float], float]:
    model = model_factory(seed, config)
    started = time.perf_counter()
    model.fit(
        np.asarray(features[train], dtype=np.float32),
        np.asarray(target[train], dtype=np.float64),
        eval_set=(
            np.asarray(features[validation], dtype=np.float32),
            np.asarray(target[validation], dtype=np.float64),
        ),
        use_best_model=False,
    )
    prediction = np.asarray(model.predict(features[validation]), dtype=np.float64)
    importance = _normalised_importance(model, features.shape[1])
    metrics = ranking_metrics(prediction, target[validation], metric_groups[validation])
    return importance, metrics, time.perf_counter() - started


def exhaustive_sweeps(
    candidates: np.ndarray,
    feature_names: Sequence[str],
    target: np.ndarray,
    split_groups: np.ndarray,
    *,
    metric_groups: np.ndarray | None = None,
    anchors: np.ndarray | None = None,
    anchor_names: Sequence[str] | None = None,
    config: SweepConfig | None = None,
    model_factory: ModelFactory | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all three complete CatBoost sweeps and return a durable ledger."""

    config = config or SweepConfig()
    config.validate()
    model_factory = model_factory or _default_model_factory
    candidates = np.asarray(candidates)
    feature_names = list(map(str, feature_names))
    target = np.asarray(target, dtype=np.float64)
    split_groups = np.asarray(split_groups)
    metric_groups = np.asarray(metric_groups if metric_groups is not None else split_groups)
    if candidates.ndim != 2 or candidates.shape != (len(target), len(feature_names)):
        raise SweepError("candidate matrix/name/target shape mismatch")
    if len(set(feature_names)) != len(feature_names):
        raise SweepError("candidate feature names are not unique")
    if not np.isfinite(candidates).all() or not np.isfinite(target).all():
        raise SweepError("candidate matrix and target must be finite")
    if len(split_groups) != len(target) or len(metric_groups) != len(target):
        raise SweepError("group vectors do not match candidate rows")

    partitions = grouped_partition(
        split_groups,
        seed=config.seed,
        tuning_fraction=config.tuning_fraction,
        final_fraction=config.final_holdout_fraction,
    )
    anchor_matrix, selected_anchor_names = _anchor_selection(
        anchors, anchor_names, partitions["discovery"], config.anchor_count
    )
    if set(feature_names) & set(selected_anchor_names):
        raise SweepError("candidate and anchor names overlap")

    sweep_reports: list[dict[str, Any]] = []
    complete_ledger: list[dict[str, Any]] = []
    contribution_matrix = np.zeros((config.sweeps, len(feature_names)), dtype=np.float64)
    validation_group_hashes: set[str] = set()
    checkpoint_identity = {
        "feature_names_sha256": _hash_strings(feature_names),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "split_groups_sha256": hashlib.sha256(split_groups.astype(str).tobytes()).hexdigest(),
        "anchors_sha256": _hash_strings(selected_anchor_names),
        "config": asdict(config),
    }
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for sweep in range(config.sweeps):
        sweep_seed = config.seed + 10_000 * (sweep + 1)
        for split_attempt in range(100):
            split_seed = sweep_seed + split_attempt
            train, validation = grouped_random_80_20(
                split_groups, partitions["discovery"], seed=split_seed
            )
            validation_hash = _hash_strings(
                sorted(map(str, set(split_groups[validation])))
            )
            if validation_hash not in validation_group_hashes:
                validation_group_hashes.add(validation_hash)
                break
        else:
            raise SweepError("could not produce three distinct grouped 80/20 sweep splits")
        windows = window_partition(
            len(feature_names),
            seed=sweep_seed,
            minimum=config.minimum_window,
            maximum=config.maximum_window,
        )
        seen: list[int] = []
        provisional = np.zeros(len(feature_names), dtype=np.float64)
        window_reports: list[dict[str, Any]] = []
        ledger_start = len(complete_ledger)
        for window_index, indices in enumerate(windows):
            seen.extend(map(int, indices))
            candidate_window = np.asarray(candidates[:, indices], dtype=np.float32)
            noise, noise_names = _matched_noise(
                candidate_window,
                split_groups,
                seed=sweep_seed + window_index,
                columns=config.noise_columns,
            )
            blocks = [candidate_window]
            names = [feature_names[index] for index in indices]
            if anchor_matrix.shape[1]:
                blocks.append(anchor_matrix)
                names.extend(selected_anchor_names)
            blocks.append(noise)
            names.extend(noise_names)
            window_matrix = np.column_stack(blocks).astype(np.float32, copy=False)
            checkpoint_key = _canonical_hash(
                {
                    **checkpoint_identity,
                    "sweep": sweep + 1,
                    "window": window_index,
                    "indices": list(map(int, indices)),
                    "train_groups": _hash_strings(sorted(map(str, set(split_groups[train])))),
                    "validation_groups": _hash_strings(
                        sorted(map(str, set(split_groups[validation])))
                    ),
                }
            )
            checkpoint_path = (
                checkpoint_dir
                / f"sweep-{sweep + 1:02d}"
                / f"window-{window_index:04d}-{checkpoint_key[:16]}.json"
                if checkpoint_dir is not None
                else None
            )
            cached = False
            if checkpoint_path is not None and checkpoint_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                result_sha256 = checkpoint.pop("result_sha256", None)
                if (
                    checkpoint.get("identity_sha256") != checkpoint_key
                    or result_sha256 != _canonical_hash(checkpoint)
                ):
                    raise SweepError(f"CatBoost window checkpoint was modified: {checkpoint_path}")
                importance = np.asarray(checkpoint["importance"], dtype=np.float64)
                metrics = {str(key): float(value) for key, value in checkpoint["metrics"].items()}
                seconds = float(checkpoint["seconds"])
                if (
                    importance.shape != (window_matrix.shape[1],)
                    or abs(float(importance.sum()) - 1.0) > 1e-12
                ):
                    raise SweepError(f"CatBoost window checkpoint is malformed: {checkpoint_path}")
                cached = True
            else:
                importance, metrics, seconds = _fit_window(
                    window_matrix,
                    target,
                    train,
                    validation,
                    metric_groups,
                    seed=sweep_seed + window_index,
                    config=config,
                    model_factory=model_factory,
                )
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = {
                        "identity_sha256": checkpoint_key,
                        "importance": importance.tolist(),
                        "metrics": metrics,
                        "seconds": seconds,
                    }
                    checkpoint["result_sha256"] = _canonical_hash(checkpoint)
                    encoded = json.dumps(checkpoint, sort_keys=True, indent=2, allow_nan=False) + "\n"
                    temporary = checkpoint_path.with_name(
                        f".{checkpoint_path.name}.{os.getpid()}.tmp"
                    )
                    with temporary.open("x", encoding="utf-8") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(temporary, checkpoint_path)
                    except FileExistsError:
                        if checkpoint_path.read_text(encoding="utf-8") != encoded:
                            raise SweepError(
                                f"CatBoost window checkpoint collision: {checkpoint_path}"
                            )
                    finally:
                        temporary.unlink(missing_ok=True)
            candidate_count = len(indices)
            anchor_end = candidate_count + anchor_matrix.shape[1]
            candidate_importance = importance[:candidate_count]
            anchor_importance = importance[candidate_count:anchor_end]
            noise_importance = importance[anchor_end:]
            noise_threshold = float(np.quantile(noise_importance, 0.95))
            denoised = np.maximum(candidate_importance - noise_threshold, 0.0)
            positive_anchors = anchor_importance[anchor_importance > 0]
            anchor_scale = (
                float(np.median(positive_anchors))
                if len(positive_anchors)
                else 1.0 / max(1, candidate_count)
            )
            calibrated = denoised / max(anchor_scale, 1e-12)
            if not np.any(calibrated):
                calibrated = candidate_importance * candidate_count
            provisional[indices] = calibrated
            for local_index, feature_index in enumerate(indices):
                complete_ledger.append(
                    {
                        "sweep": sweep + 1,
                        "window": window_index,
                        "feature_index": int(feature_index),
                        "feature_name": feature_names[feature_index],
                        "model_normalized_importance": float(candidate_importance[local_index]),
                        "noise_importance_p95": noise_threshold,
                        "anchor_importance_median": anchor_scale,
                        "denoised_anchor_relative": float(calibrated[local_index]),
                        "sweep_contribution": None,
                    }
                )
            window_reports.append(
                {
                    "window": window_index,
                    "candidate_columns": candidate_count,
                    "total_model_columns": len(names),
                    "feature_names_sha256": _hash_strings(names),
                    "importance_sum": float(importance.sum()),
                    "candidate_importance_sum": float(candidate_importance.sum()),
                    "anchor_importance_sum": float(anchor_importance.sum()),
                    "noise_importance_sum": float(noise_importance.sum()),
                    "noise_importance_p95": noise_threshold,
                    "metrics": metrics,
                    "seconds": seconds,
                    "checkpoint_cache_hit": cached,
                }
            )
        if sorted(seen) != list(range(len(feature_names))) or len(seen) != len(set(seen)):
            raise SweepError(f"sweep {sweep + 1} skipped or duplicated candidate columns")
        if float(provisional.sum()) <= 0:
            raise SweepError(f"sweep {sweep + 1} produced no positive feature contribution")
        contribution = provisional / provisional.sum()
        contribution /= contribution.sum()
        contribution_matrix[sweep] = contribution
        for row in complete_ledger[ledger_start:]:
            row["sweep_contribution"] = float(contribution[int(row["feature_index"])])
        if abs(float(contribution.sum()) - 1.0) > 1e-12:
            raise SweepError(f"sweep {sweep + 1} contribution does not add to one")
        summary_count = min(500, len(feature_names))
        summary_indices = np.argsort(-contribution, kind="stable")[:summary_count]
        _summary_importance, summary_metrics, summary_seconds = _fit_window(
            np.asarray(candidates[:, summary_indices], dtype=np.float32),
            target,
            train,
            validation,
            metric_groups,
            seed=sweep_seed + 9_000,
            config=config,
            model_factory=model_factory,
        )
        sweep_reports.append(
            {
                "sweep": sweep + 1,
                "seed": sweep_seed,
                "split_seed": split_seed,
                "split": "grouped_random_80_20",
                "train_groups_sha256": _hash_strings(sorted(map(str, set(split_groups[train])))),
                "validation_groups_sha256": _hash_strings(
                    sorted(map(str, set(split_groups[validation])))
                ),
                "train_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "windows": window_reports,
                "features_seen": len(seen),
                "sweep_contribution_sum": float(contribution.sum()),
                "validation_summary": {
                    "feature_count": summary_count,
                    "feature_names_sha256": _hash_strings(
                        [feature_names[index] for index in summary_indices]
                    ),
                    "metrics": summary_metrics,
                    "seconds": summary_seconds,
                },
            }
        )

    mean = contribution_matrix.mean(axis=0)
    std = contribution_matrix.std(axis=0)
    percentiles = 100.0 * (_ranks(mean) + 1.0) / len(mean)
    aggregate = [
        {
            "feature_index": index,
            "feature_name": name,
            "sweep_contributions": contribution_matrix[:, index].tolist(),
            "mean_contribution": float(mean[index]),
            "contribution_std": float(std[index]),
            "contribution_percentile": float(percentiles[index]),
        }
        for index, name in enumerate(feature_names)
    ]
    aggregate.sort(key=lambda row: (-row["mean_contribution"], row["feature_name"]))
    return {
        "schema": "ZyalCatBoostFeatureSweeps.v1",
        "config": asdict(config),
        "feature_count": len(feature_names),
        "feature_names_sha256": _hash_strings(feature_names),
        "anchors": selected_anchor_names,
        "partitions": {
            name: {
                "rows": int(mask.sum()),
                "groups_sha256": _hash_strings(sorted(map(str, set(split_groups[mask])))),
            }
            for name, mask in partitions.items()
        },
        "sweeps": sweep_reports,
        "ledger": complete_ledger,
        "aggregate": aggregate,
        "complete": len(complete_ledger) == len(feature_names) * config.sweeps,
        "all_features_once_per_sweep": True,
        "importance_normalization": "each CatBoost model and each complete sweep sum to 1",
        "partition_masks": partitions,
    }


def _fit_feature_set(
    candidates: np.ndarray,
    indices: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    metric_groups: np.ndarray,
    *,
    seed: int,
    config: SweepConfig,
    model_factory: ModelFactory,
) -> tuple[dict[str, float], float]:
    model = model_factory(seed, config)
    started = time.perf_counter()
    model.fit(
        np.asarray(candidates[train][:, indices], dtype=np.float32),
        target[train],
        eval_set=(np.asarray(candidates[validation][:, indices], dtype=np.float32), target[validation]),
        use_best_model=False,
    )
    prediction = np.asarray(
        model.predict(np.asarray(candidates[validation][:, indices], dtype=np.float32)),
        dtype=np.float64,
    )
    return ranking_metrics(prediction, target[validation], metric_groups[validation]), time.perf_counter() - started


def finalize_percentile_set(
    sweep_receipt: Mapping[str, Any],
    candidates: np.ndarray,
    feature_names: Sequence[str],
    target: np.ndarray,
    split_groups: np.ndarray,
    *,
    metric_groups: np.ndarray | None = None,
    config: SweepConfig | None = None,
    model_factory: ModelFactory | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Choose/interpolate percentile cuts, then open the final grouped holdout once."""

    config = config or SweepConfig()
    config.validate()
    model_factory = model_factory or _default_model_factory
    candidates = np.asarray(candidates)
    feature_names = list(map(str, feature_names))
    target = np.asarray(target, dtype=np.float64)
    split_groups = np.asarray(split_groups)
    metric_groups = np.asarray(metric_groups if metric_groups is not None else split_groups)
    if sweep_receipt.get("complete") is not True:
        raise SweepError("cannot finalize an incomplete feature sweep")
    if sweep_receipt.get("feature_names_sha256") != _hash_strings(feature_names):
        raise SweepError("candidate feature schema changed after the sweeps")
    partitions = sweep_receipt.get("partition_masks")
    if not isinstance(partitions, Mapping):
        raise SweepError("in-memory sweep partition masks are unavailable")
    discovery = np.asarray(partitions["discovery"], dtype=bool)
    tuning = np.asarray(partitions["tuning"], dtype=bool)
    final = np.asarray(partitions["final"], dtype=bool)
    aggregate = list(sweep_receipt["aggregate"])
    percentile_by_name = {
        str(row["feature_name"]): float(row["contribution_percentile"])
        for row in aggregate
    }
    contribution_by_name = {
        str(row["feature_name"]): float(row["mean_contribution"]) for row in aggregate
    }

    evaluated: dict[float, dict[str, Any]] = {}

    def evaluate(cut: float) -> dict[str, Any]:
        if cut in evaluated:
            return evaluated[cut]
        indices = np.asarray(
            [index for index, name in enumerate(feature_names) if percentile_by_name[name] >= cut],
            dtype=np.int64,
        )
        if not len(indices):
            indices = np.asarray(
                [max(range(len(feature_names)), key=lambda index: contribution_by_name[feature_names[index]])],
                dtype=np.int64,
            )
        metrics, seconds = _fit_feature_set(
            candidates,
            indices,
            target,
            discovery,
            tuning,
            metric_groups,
            seed=config.seed + 70_000 + int(round(cut * 10)),
            config=config,
            model_factory=model_factory,
        )
        result = {
            "percentile_cut": cut,
            "feature_count": len(indices),
            "feature_names": [feature_names[index] for index in indices],
            "feature_names_sha256": _hash_strings([feature_names[index] for index in indices]),
            "tuning_metrics": metrics,
            "seconds": seconds,
        }
        evaluated[cut] = result
        return result

    initial = [evaluate(cut) for cut in config.percentile_cuts]
    by_cut = sorted(initial, key=lambda row: row["percentile_cut"])
    best_initial = max(
        by_cut,
        key=lambda row: (
            row["tuning_metrics"]["within_group_spearman"],
            row["tuning_metrics"]["capture_at_3"],
            -row["tuning_metrics"]["mean_regret"],
            -row["feature_count"],
        ),
    )
    best_position = by_cut.index(best_initial)
    neighbor_positions = {
        max(0, best_position - 1),
        min(len(by_cut) - 1, best_position + 1),
    }.difference({best_position})
    for position in sorted(neighbor_positions):
        lower, upper = sorted(
            [best_initial["percentile_cut"], by_cut[position]["percentile_cut"]]
        )
        for point in range(1, config.interpolation_points + 1):
            cut = lower + (upper - lower) * point / (config.interpolation_points + 1)
            evaluate(round(cut, 6))

    best_spearman = max(
        row["tuning_metrics"]["within_group_spearman"] for row in evaluated.values()
    )
    plateau = [
        row
        for row in evaluated.values()
        if row["tuning_metrics"]["within_group_spearman"]
        >= best_spearman - config.plateau_tolerance
    ]
    chosen = min(
        plateau,
        key=lambda row: (
            row["feature_count"],
            -row["tuning_metrics"]["capture_at_3"],
            row["tuning_metrics"]["mean_regret"],
            -row["percentile_cut"],
        ),
    )
    chosen_indices = np.asarray(
        [feature_names.index(name) for name in chosen["feature_names"]], dtype=np.int64
    )
    train_final = discovery | tuning
    final_identity = _canonical_hash(
        {
            "feature_names_sha256": _hash_strings(chosen["feature_names"]),
            "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            "train_groups_sha256": _hash_strings(
                sorted(map(str, set(split_groups[train_final])))
            ),
            "final_groups_sha256": _hash_strings(
                sorted(map(str, set(split_groups[final])))
            ),
            "config": asdict(config),
        }
    )
    final_cache_hit = False
    reservation_path = None
    result_path = None
    if checkpoint_dir is not None:
        reservation_path = checkpoint_dir / f"final-{final_identity}.reservation.json"
        result_path = checkpoint_dir / f"final-{final_identity}.result.json"
    if result_path is not None and result_path.is_file():
        cached = _read_sealed_json(result_path)
        if cached.get("identity_sha256") != final_identity:
            raise SweepError(f"final CatBoost holdout identity changed: {result_path}")
        final_metrics = {
            str(key): float(value) for key, value in cached["metrics"].items()
        }
        final_seconds = float(cached["seconds"])
        final_cache_hit = True
    else:
        if reservation_path is not None:
            created = _write_exclusive_json(
                reservation_path,
                {
                    "identity_sha256": final_identity,
                    "state": "reserved_before_open",
                },
            )
            if not created:
                if result_path is not None and result_path.is_file():
                    cached = _read_sealed_json(result_path)
                    if cached.get("identity_sha256") != final_identity:
                        raise SweepError(
                            f"final CatBoost holdout identity changed: {result_path}"
                        )
                    final_metrics = {
                        str(key): float(value) for key, value in cached["metrics"].items()
                    }
                    final_seconds = float(cached["seconds"])
                    final_cache_hit = True
                else:
                    raise SweepError(
                        "final CatBoost holdout was reserved but no result was sealed; "
                        "refusing to reopen it"
                    )
        if not final_cache_hit:
            final_metrics, final_seconds = _fit_feature_set(
                candidates,
                chosen_indices,
                target,
                train_final,
                final,
                metric_groups,
                seed=config.seed + 90_000,
                config=config,
                model_factory=model_factory,
            )
            if result_path is not None:
                _write_exclusive_json(
                    result_path,
                    {
                        "identity_sha256": final_identity,
                        "metrics": final_metrics,
                        "seconds": final_seconds,
                    },
                )
    return {
        "schema": "ZyalCatBoostFinalFeatureSet.v1",
        "candidate_feature_count": len(feature_names),
        "evaluated_cuts": sorted(evaluated.values(), key=lambda row: row["percentile_cut"]),
        "selection_rule": (
            "smallest set within 0.002 within-group Spearman of the best tuning cut; "
            "capture@3 then regret break ties"
        ),
        "selected_percentile_cut": chosen["percentile_cut"],
        "selected_feature_count": chosen["feature_count"],
        "selected_features": chosen["feature_names"],
        "selected_features_sha256": chosen["feature_names_sha256"],
        "tuning_metrics": chosen["tuning_metrics"],
        "final_oos_metrics": final_metrics,
        "final_oos_rows": int(final.sum()),
        "final_oos_groups_sha256": _hash_strings(sorted(map(str, set(split_groups[final])))),
        "final_holdout_opened_once": True,
        "final_holdout_used_for_selection": False,
        "final_holdout_identity_sha256": final_identity,
        "final_holdout_checkpoint_cache_hit": final_cache_hit,
        "final_fit_seconds": final_seconds,
        "complete": True,
    }


def serializable_sweep_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Remove in-memory masks before writing the durable sweep receipt."""

    return {key: value for key, value in receipt.items() if key != "partition_masks"}
