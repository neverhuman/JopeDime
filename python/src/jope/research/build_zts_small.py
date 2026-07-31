#!/usr/bin/env python3
"""Build and verify the deterministic ZTS_small benchmark coreset.

The emitted CSV rows are copied byte-for-byte from the parent ZTS split.  Test
membership never informs training preprocessing, and rows never move between
splits.  Selection is target-stratified, feature-aware, and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCHEMA = "zts-small.coreset.v1"
ROW_SCHEMA = "zts-small.source-rows.v1"
SEED = 20260723
STRATA = 100
ROWS_PER_STRATUM = 10
CANDIDATES = 64
RANDOM_BASELINES = 100
PROJECTION_DIMENSIONS = 32
MIN_LEVEL_SUPPORT = 0.005
MAX_LOW_CARDINALITY = 32
QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])
DIAGNOSTIC_NAMES = (
    "feature_mean_error",
    "feature_variance_error",
    "feature_quantile_error",
    "low_cardinality_frequency_error",
    "projected_covariance_error",
    "nearest_neighbor_coverage",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def atomic_immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite changed immutable artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def read_raw_rows(path: Path) -> list[bytes]:
    rows = path.read_bytes().splitlines(keepends=True)
    if rows and not rows[-1].endswith((b"\n", b"\r")):
        rows[-1] += b"\n"
    return rows


def read_numeric_csv(path: Path) -> np.ndarray:
    values = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path} must contain features plus a final target")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite emitted values")
    return values


def target_strata(target: np.ndarray) -> np.ndarray:
    if target.shape[0] % STRATA:
        raise ValueError(f"split size {target.shape[0]} is not divisible by {STRATA}")
    order = np.lexsort((np.arange(target.shape[0]), target))
    strata = np.empty(target.shape[0], dtype=np.int16)
    size = target.shape[0] // STRATA
    for stratum in range(STRATA):
        strata[order[stratum * size : (stratum + 1) * size]] = stratum
    return strata


@dataclass(frozen=True)
class Projection:
    median: np.ndarray
    scale: np.ndarray
    matrix: np.ndarray

    def transform(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        robust = np.clip((features - self.median) / self.scale, -8.0, 8.0)
        projected = robust @ self.matrix
        return robust.astype(np.float32), projected.astype(np.float32)


def fit_projection(train_features: np.ndarray) -> Projection:
    median = np.median(train_features, axis=0)
    q25, q75 = np.quantile(train_features, [0.25, 0.75], axis=0)
    scale = q75 - q25
    mad = 1.4826 * np.median(np.abs(train_features - median), axis=0)
    standard = np.std(train_features, axis=0)
    scale = np.where(scale > 1e-12, scale, mad)
    scale = np.where(scale > 1e-12, scale, standard)
    scale = np.where(scale > 1e-12, scale, 1.0)
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(
        0.0,
        1.0 / math.sqrt(train_features.shape[1]),
        size=(train_features.shape[1], PROJECTION_DIMENSIONS),
    )
    return Projection(median=median, scale=scale, matrix=matrix)


@dataclass(frozen=True)
class Level:
    feature: int
    value: float
    support: float


def supported_low_cardinality_levels(features: np.ndarray) -> list[Level]:
    levels: list[Level] = []
    for feature in range(features.shape[1]):
        values, counts = np.unique(features[:, feature], return_counts=True)
        if len(values) > MAX_LOW_CARDINALITY:
            continue
        for value, count in zip(values, counts, strict=True):
            support = float(count / features.shape[0])
            if support >= MIN_LEVEL_SUPPORT:
                levels.append(Level(feature, float(value), support))
    return sorted(levels, key=lambda level: (level.support, level.feature, level.value))


def level_mask(features: np.ndarray, level: Level) -> np.ndarray:
    return features[:, level.feature] == level.value


def covered_levels(features: np.ndarray, indices: np.ndarray, levels: list[Level]) -> np.ndarray:
    return np.asarray(
        [bool(np.any(features[indices, level.feature] == level.value)) for level in levels],
        dtype=bool,
    )


@dataclass
class Surrogate:
    robust: np.ndarray
    projected: np.ndarray
    feature_mean: np.ndarray
    feature_second: np.ndarray
    projected_mean: np.ndarray
    projected_second: np.ndarray
    histogram_bin: np.ndarray
    histogram_target: np.ndarray
    weights: np.ndarray

    @classmethod
    def create(
        cls,
        robust: np.ndarray,
        projected: np.ndarray,
        baseline_components: np.ndarray | None = None,
    ) -> "Surrogate":
        feature_mean = robust.mean(axis=0, dtype=np.float64)
        feature_second = np.square(robust, dtype=np.float64).mean(axis=0)
        projected_mean = projected.mean(axis=0, dtype=np.float64)
        projected_second = projected.astype(np.float64).T @ projected / projected.shape[0]
        edges = np.quantile(robust, np.linspace(0.1, 0.9, 9), axis=0)
        bins = np.empty_like(robust, dtype=np.uint8)
        targets = np.empty((robust.shape[1], 10), dtype=np.float64)
        for feature in range(robust.shape[1]):
            bins[:, feature] = np.searchsorted(
                edges[:, feature],
                robust[:, feature],
                side="right",
            )
            targets[feature] = np.bincount(bins[:, feature], minlength=10) / robust.shape[0]
        if baseline_components is None:
            weights = np.ones(4, dtype=np.float64)
        else:
            weights = 1.0 / np.maximum(baseline_components, 1e-12)
        return cls(
            robust=robust,
            projected=projected,
            feature_mean=feature_mean,
            feature_second=feature_second,
            projected_mean=projected_mean,
            projected_second=projected_second,
            histogram_bin=bins,
            histogram_target=targets,
            weights=weights,
        )

    def components(self, indices: np.ndarray) -> np.ndarray:
        selected_x = self.robust[indices].astype(np.float64)
        selected_z = self.projected[indices].astype(np.float64)
        mean_error = float(np.sqrt(np.mean(np.square(selected_x.mean(axis=0) - self.feature_mean))))
        second_error = float(
            np.sqrt(np.mean(np.square(np.square(selected_x).mean(axis=0) - self.feature_second)))
        )
        projected_second = selected_z.T @ selected_z / selected_z.shape[0]
        covariance_error = float(
            np.linalg.norm(projected_second - self.projected_second)
            / max(np.linalg.norm(self.projected_second), 1e-12)
        )
        counts = np.zeros_like(self.histogram_target)
        for feature in range(self.robust.shape[1]):
            counts[feature] = np.bincount(
                self.histogram_bin[indices, feature],
                minlength=10,
            ) / len(indices)
        histogram_error = float(np.mean(np.abs(counts - self.histogram_target)))
        return np.asarray([mean_error, second_error, covariance_error, histogram_error])

    def score(self, indices: np.ndarray) -> float:
        return float(np.mean(self.components(indices) * self.weights))


def stratified_random_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for stratum in range(STRATA):
        rows = np.flatnonzero(strata == stratum)
        selected.extend(rng.choice(rows, size=ROWS_PER_STRATUM, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def forced_endpoint_indices(target: np.ndarray) -> set[int]:
    minimum = int(np.flatnonzero(target == target.min())[0])
    maximum = int(np.flatnonzero(target == target.max())[-1])
    return {minimum, maximum}


def farthest_first_indices(
    projected: np.ndarray,
    strata: np.ndarray,
    target: np.ndarray,
    candidate: int,
) -> np.ndarray:
    selected: list[int] = []
    endpoints = forced_endpoint_indices(target)
    rng = np.random.default_rng(SEED + 10_007 * (candidate + 1))
    for stratum in range(STRATA):
        rows = np.flatnonzero(strata == stratum)
        anchors = [row for row in sorted(endpoints) if strata[row] == stratum]
        if not anchors:
            centroid = projected[rows].mean(axis=0)
            distances = np.sum(np.square(projected[rows] - centroid), axis=1)
            order = np.lexsort((rows, distances))
            window = order[: max(1, len(order) // 5)]
            anchors.append(int(rows[window[candidate % len(window)]]))
        nearest = np.full(rows.shape[0], np.inf, dtype=np.float64)
        while len(anchors) < ROWS_PER_STRATUM:
            last = projected[anchors[-1]]
            nearest = np.minimum(
                nearest,
                np.sum(np.square(projected[rows] - last), axis=1),
            )
            nearest[np.isin(rows, anchors)] = -np.inf
            jitter = rng.random(rows.shape[0]) * 1e-10
            anchors.append(int(rows[np.argmax(nearest + jitter)]))

        anchor_values = projected[anchors].astype(np.float64)
        values = projected[rows].astype(np.float64)
        distance_matrix = (
            np.sum(np.square(values), axis=1, keepdims=True)
            + np.sum(np.square(anchor_values), axis=1)
            - 2.0 * values @ anchor_values.T
        )
        assignments = np.argmin(distance_matrix, axis=1)
        counts = np.bincount(assignments, minlength=len(anchors))
        cumulative = np.cumsum(counts) / len(rows)
        offset = rng.random() / ROWS_PER_STRATUM
        allocated_cells = np.searchsorted(
            cumulative,
            offset + np.arange(ROWS_PER_STRATUM) / ROWS_PER_STRATUM,
            side="right",
        )
        chosen: list[int] = []
        for anchor_slot in allocated_cells:
            anchor = anchors[int(anchor_slot)]
            members = rows[assignments == anchor_slot]
            unused_members = np.asarray(
                [int(row) for row in members if int(row) not in chosen],
                dtype=np.int64,
            )
            if anchor in endpoints and anchor not in chosen:
                medoid = anchor
            elif len(unused_members):
                medoid = int(unused_members[int(rng.integers(0, len(unused_members)))])
            else:
                unused_rows = np.asarray(
                    [int(row) for row in rows if int(row) not in chosen],
                    dtype=np.int64,
                )
                medoid = int(unused_rows[int(rng.integers(0, len(unused_rows)))])
            if medoid in chosen:
                unused = [int(row) for row in rows if int(row) not in chosen]
                medoid = min(
                    unused,
                    key=lambda row: (
                        float(np.sum(np.square(projected[row] - projected[anchor]))),
                        row,
                    ),
                )
            chosen.append(medoid)
        for endpoint in endpoints:
            if strata[endpoint] != stratum or endpoint in chosen:
                continue
            replaceable = [row for row in chosen if row not in endpoints]
            chosen[chosen.index(replaceable[0])] = endpoint
        selected.extend(chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def enforce_levels(
    features: np.ndarray,
    strata: np.ndarray,
    selected: np.ndarray,
    levels: list[Level],
    protected: set[int],
    surrogate: Surrogate,
) -> tuple[np.ndarray, list[Level]]:
    selected_set = set(int(index) for index in selected)
    by_level = [np.flatnonzero(level_mask(features, level)) for level in levels]
    for level_index, level in enumerate(levels):
        if any(features[index, level.feature] == level.value for index in selected_set):
            continue
        best: tuple[float, int, int] | None = None
        for incoming in by_level[level_index]:
            stratum = int(strata[incoming])
            outgoing_rows = [
                index
                for index in selected_set
                if strata[index] == stratum and index not in protected
            ]
            for outgoing in outgoing_rows:
                safe = True
                for prior_level in levels:
                    if features[outgoing, prior_level.feature] != prior_level.value:
                        continue
                    remaining = selected_set - {outgoing}
                    if not any(
                        features[index, prior_level.feature] == prior_level.value
                        for index in remaining
                    ):
                        safe = False
                        break
                if not safe:
                    continue
                proposal = np.asarray(sorted(selected_set - {outgoing} | {int(incoming)}))
                score = surrogate.score(proposal)
                key = (score, int(incoming), int(outgoing))
                if best is None or key < best:
                    best = key
        if best is not None:
            _, incoming, outgoing = best
            selected_set.remove(outgoing)
            selected_set.add(incoming)
    result = np.asarray(sorted(selected_set), dtype=np.int64)
    coverage = covered_levels(features, result, levels)
    missing = [level for level, present in zip(levels, coverage, strict=True) if not present]
    return result, missing


def optimize_swaps(
    features: np.ndarray,
    strata: np.ndarray,
    selected: np.ndarray,
    levels: list[Level],
    protected: set[int],
    surrogate: Surrogate,
    candidate: int,
    proposals: int,
) -> np.ndarray:
    rng = np.random.default_rng(SEED ^ (candidate + 1) * 0x9E3779B1)
    selected_set = set(int(index) for index in selected)
    current = surrogate.score(selected)
    for _ in range(proposals):
        stratum = int(rng.integers(0, STRATA))
        outgoing_pool = [
            index for index in selected_set if strata[index] == stratum and index not in protected
        ]
        incoming_pool = [
            int(index)
            for index in np.flatnonzero(strata == stratum)
            if int(index) not in selected_set
        ]
        if not outgoing_pool or not incoming_pool:
            continue
        outgoing = outgoing_pool[int(rng.integers(0, len(outgoing_pool)))]
        incoming = incoming_pool[int(rng.integers(0, len(incoming_pool)))]
        remaining = selected_set - {outgoing}
        if any(
            features[outgoing, level.feature] == level.value
            and not any(features[index, level.feature] == level.value for index in remaining)
            and features[incoming, level.feature] != level.value
            for level in levels
        ):
            continue
        proposal_set = remaining | {incoming}
        proposal = np.asarray(sorted(proposal_set), dtype=np.int64)
        score = surrogate.score(proposal)
        if score + 1e-15 < current:
            selected_set = proposal_set
            current = score
    return np.asarray(sorted(selected_set), dtype=np.int64)


def deterministic_probe_indices(rows: int, count: int = 512) -> np.ndarray:
    if rows <= count:
        return np.arange(rows, dtype=np.int64)
    return np.linspace(0, rows - 1, count, dtype=np.int64)


def nearest_neighbor_coverage(projected: np.ndarray, selected: np.ndarray) -> float:
    probe = projected[deterministic_probe_indices(projected.shape[0])].astype(np.float64)
    reference = projected[selected].astype(np.float64)
    probe_norm = np.sum(np.square(probe), axis=1, keepdims=True)
    reference_norm = np.sum(np.square(reference), axis=1)
    distances = probe_norm + reference_norm - 2.0 * probe @ reference.T
    return float(np.sqrt(np.maximum(np.min(distances, axis=1), 0.0)).mean())


def diagnostics(
    features: np.ndarray,
    robust: np.ndarray,
    projected: np.ndarray,
    selected: np.ndarray,
    levels: list[Level],
) -> dict[str, float]:
    full_mean = robust.mean(axis=0, dtype=np.float64)
    full_variance = robust.var(axis=0, dtype=np.float64)
    selected_mean = robust[selected].mean(axis=0, dtype=np.float64)
    selected_variance = robust[selected].var(axis=0, dtype=np.float64)
    mean_scale = np.sqrt(full_variance) + 1e-8
    mean_error = float(np.sqrt(np.mean(np.square((selected_mean - full_mean) / mean_scale))))
    variable = full_variance > 1e-12
    variance_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.log(
                        (selected_variance[variable] + 1e-12)
                        / (full_variance[variable] + 1e-12)
                    )
                )
            )
        )
    )
    full_quantiles = np.quantile(robust, QUANTILES, axis=0)
    selected_quantiles = np.quantile(robust[selected], QUANTILES, axis=0)
    quantile_scale = np.maximum(
        full_quantiles[-2] - full_quantiles[1],
        np.sqrt(full_variance),
    )
    quantile_scale = np.maximum(quantile_scale, 1e-8)
    quantile_error = float(
        np.mean(np.abs(selected_quantiles - full_quantiles) / quantile_scale)
    )
    frequency_errors = []
    for level in levels:
        full_frequency = float(np.mean(features[:, level.feature] == level.value))
        selected_frequency = float(np.mean(features[selected, level.feature] == level.value))
        frequency_errors.append(abs(selected_frequency - full_frequency))
    low_cardinality_error = float(np.mean(frequency_errors)) if frequency_errors else 0.0
    full_covariance = np.cov(projected.astype(np.float64), rowvar=False, ddof=0)
    selected_covariance = np.cov(projected[selected].astype(np.float64), rowvar=False, ddof=0)
    covariance_error = float(
        np.linalg.norm(selected_covariance - full_covariance)
        / max(np.linalg.norm(full_covariance), 1e-12)
    )
    return {
        "feature_mean_error": mean_error,
        "feature_variance_error": variance_error,
        "feature_quantile_error": quantile_error,
        "low_cardinality_frequency_error": low_cardinality_error,
        "projected_covariance_error": covariance_error,
        "nearest_neighbor_coverage": nearest_neighbor_coverage(projected, selected),
    }


def baseline_diagnostics(
    features: np.ndarray,
    robust: np.ndarray,
    projected: np.ndarray,
    strata: np.ndarray,
    levels: list[Level],
) -> tuple[list[dict[str, float]], np.ndarray]:
    rng = np.random.default_rng(SEED ^ 0xB453_11E)
    rows = []
    surrogate = Surrogate.create(robust, projected)
    components = []
    for _ in range(RANDOM_BASELINES):
        indices = stratified_random_indices(strata, rng)
        rows.append(diagnostics(features, robust, projected, indices, levels))
        components.append(surrogate.components(indices))
    return rows, np.median(np.asarray(components), axis=0)


def diagnostic_medians(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.median([row[name] for row in rows]))
        for name in DIAGNOSTIC_NAMES
    }


def candidate_indices(
    features: np.ndarray,
    target: np.ndarray,
    robust: np.ndarray,
    projected: np.ndarray,
    baseline_components: np.ndarray,
    candidate: int,
) -> tuple[np.ndarray, list[Level]]:
    strata = target_strata(target)
    levels = supported_low_cardinality_levels(features)
    protected = forced_endpoint_indices(target)
    surrogate = Surrogate.create(robust, projected, baseline_components)
    selected = farthest_first_indices(projected, strata, target, candidate)
    selected, _ = enforce_levels(
        features,
        strata,
        selected,
        levels,
        protected,
        surrogate,
    )
    selected = optimize_swaps(
        features,
        strata,
        selected,
        levels,
        protected,
        surrogate,
        candidate,
        proposals=220,
    )
    selected, missing = enforce_levels(
        features,
        strata,
        selected,
        levels,
        protected,
        surrogate,
    )
    return selected, missing


@dataclass(frozen=True)
class SplitSelection:
    indices: np.ndarray
    candidate: int
    diagnostics: dict[str, float]
    baseline_medians: dict[str, float]
    composite_score: float
    levels: list[Level]


def select_split(
    features: np.ndarray,
    target: np.ndarray,
    robust: np.ndarray,
    projected: np.ndarray,
) -> SplitSelection:
    strata = target_strata(target)
    levels = supported_low_cardinality_levels(features)
    random_rows, baseline_components = baseline_diagnostics(
        features,
        robust,
        projected,
        strata,
        levels,
    )
    medians = diagnostic_medians(random_rows)
    candidates: list[SplitSelection] = []
    for candidate in range(CANDIDATES):
        indices, missing = candidate_indices(
            features,
            target,
            robust,
            projected,
            baseline_components,
            candidate,
        )
        if missing:
            continue
        row = diagnostics(features, robust, projected, indices, levels)
        ratios = [row[name] / max(medians[name], 1e-12) for name in DIAGNOSTIC_NAMES]
        candidates.append(
            SplitSelection(
                indices=indices,
                candidate=candidate,
                diagnostics=row,
                baseline_medians=medians,
                composite_score=float(np.mean(ratios)),
                levels=levels,
            )
        )
    passing = [
        candidate
        for candidate in candidates
        if all(
            candidate.diagnostics[name] < candidate.baseline_medians[name]
            for name in DIAGNOSTIC_NAMES
        )
    ]
    if not passing:
        best = min(candidates, key=lambda row: (max(
            row.diagnostics[name] / max(row.baseline_medians[name], 1e-12)
            for name in DIAGNOSTIC_NAMES
        ), row.composite_score, row.candidate))
        failures = {
            name: {
                "candidate": best.diagnostics[name],
                "random_median": best.baseline_medians[name],
            }
            for name in DIAGNOSTIC_NAMES
            if best.diagnostics[name] >= best.baseline_medians[name]
        }
        raise RuntimeError(
            "no deterministic candidate beats the stratified-random median on every "
            f"diagnostic; best candidate {best.candidate} failures: {failures}"
        )
    return min(passing, key=lambda row: (row.composite_score, row.candidate))


def calibration_panel(
    train: np.ndarray,
    test: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    models = {
        "linear_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "neighbor_knn15": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=1),
        ),
        "forest_extra_trees96": ExtraTreesRegressor(
            n_estimators=96,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        ),
        "boosting_histgbm": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.07,
            l2_regularization=0.1,
            random_state=SEED,
        ),
        "kernel_rbf_features": make_pipeline(
            StandardScaler(),
            RBFSampler(gamma="scale", n_components=96, random_state=SEED),
            Ridge(alpha=1.0),
        ),
    }

    def score(train_rows: np.ndarray, test_rows: np.ndarray) -> dict[str, float]:
        result = {}
        for name, model in models.items():
            model.fit(x_train[train_rows], y_train[train_rows])
            prediction = np.asarray(model.predict(x_test[test_rows]), dtype=np.float64)
            result[name] = float(np.sqrt(np.mean(np.square(prediction - y_test[test_rows]))))
        return result

    full = score(np.arange(len(train)), np.arange(len(test)))
    small = score(train_indices, test_indices)
    names = sorted(models)
    correlation = round(
        float(
        spearmanr(
            [full[name] for name in names],
            [small[name] for name in names],
        ).statistic
        ),
        12,
    )
    return {
        "models": names,
        "full_zts_rmse": full,
        "zts_small_rmse": small,
        "full_order": sorted(names, key=lambda name: (full[name], name)),
        "small_order": sorted(names, key=lambda name: (small[name], name)),
        "spearman": correlation,
        "required_spearman": 0.90,
        "passed": correlation >= 0.90,
    }


def row_manifest(
    role: str,
    parent_path: Path,
    parent_rows: list[bytes],
    selected: np.ndarray,
) -> dict[str, Any]:
    output_rows = [parent_rows[int(index)] for index in selected]
    return {
        "schema": ROW_SCHEMA,
        "role": role,
        "parent_path": str(parent_path),
        "parent_sha256": sha256_file(parent_path),
        "parent_rows": len(parent_rows),
        "selected_rows": len(selected),
        "zero_based_parent_indices": [int(index) for index in selected],
        "rows": [
            {
                "output_row": output_row,
                "parent_row": int(parent_row),
                "sha256": sha256_bytes(output_rows[output_row].rstrip(b"\r\n")),
            }
            for output_row, parent_row in enumerate(selected)
        ],
    }


def selection_record(selection: SplitSelection, strata: np.ndarray) -> dict[str, Any]:
    quota = np.bincount(strata[selection.indices], minlength=STRATA)
    return {
        "candidate": selection.candidate,
        "composite_score": selection.composite_score,
        "zero_based_parent_indices": [int(index) for index in selection.indices],
        "target_stratum_counts": [int(value) for value in quota],
        "diagnostics": selection.diagnostics,
        "stratified_random_medians": selection.baseline_medians,
        "beats_random_median": {
            name: selection.diagnostics[name] < selection.baseline_medians[name]
            for name in DIAGNOSTIC_NAMES
        },
        "supported_low_cardinality_levels": [
            {
                "feature_index": level.feature,
                "value": level.value,
                "parent_support": level.support,
            }
            for level in selection.levels
        ],
    }


def validate_selection(
    parent: np.ndarray,
    selected: np.ndarray,
    indices: np.ndarray,
    role: str,
) -> None:
    if selected.shape != (STRATA * ROWS_PER_STRATUM, parent.shape[1]):
        raise ValueError(f"{role}: unexpected selected shape {selected.shape}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{role}: selected parent indices are not unique")
    if not np.array_equal(selected, parent[indices]):
        raise ValueError(f"{role}: selected values differ from parent rows")
    quotas = np.bincount(
        target_strata(parent[:, -1])[indices],
        minlength=STRATA,
    )
    if not np.all(quotas == ROWS_PER_STRATUM):
        raise ValueError(f"{role}: target-stratum quotas changed")
    endpoints = forced_endpoint_indices(parent[:, -1])
    if not endpoints.issubset(set(int(index) for index in indices)):
        raise ValueError(f"{role}: target endpoints are absent")


def build(parent: Path, output: Path, *, verify_only: bool = False) -> dict[str, Any]:
    train_path = parent / "train.csv"
    test_path = parent / "test.csv"
    meta_path = parent / "meta.json"
    parent_meta = json.loads(meta_path.read_text())
    if parent_meta.get("headers_removed_from_csv") is not True:
        raise ValueError("ZTS parent must be the canonical headerless artifact")
    train = read_numeric_csv(train_path)
    test = read_numeric_csv(test_path)
    if train.shape != (50_000, 148) or test.shape != (10_000, 148):
        raise ValueError(f"unexpected ZTS shape: train={train.shape}, test={test.shape}")
    projection = fit_projection(train[:, :-1])
    train_robust, train_projected = projection.transform(train[:, :-1])
    test_robust, test_projected = projection.transform(test[:, :-1])
    train_selection = select_split(
        train[:, :-1],
        train[:, -1],
        train_robust,
        train_projected,
    )
    test_selection = select_split(
        test[:, :-1],
        test[:, -1],
        test_robust,
        test_projected,
    )
    selected_train = train[train_selection.indices]
    selected_test = test[test_selection.indices]
    validate_selection(train, selected_train, train_selection.indices, "train")
    validate_selection(test, selected_test, test_selection.indices, "test")
    train_row_bytes = read_raw_rows(train_path)
    test_row_bytes = read_raw_rows(test_path)
    train_csv = b"".join(train_row_bytes[int(index)] for index in train_selection.indices)
    test_csv = b"".join(test_row_bytes[int(index)] for index in test_selection.indices)
    train_rows_manifest = row_manifest(
        "train",
        train_path,
        train_row_bytes,
        train_selection.indices,
    )
    test_rows_manifest = row_manifest(
        "test",
        test_path,
        test_row_bytes,
        test_selection.indices,
    )
    calibration = calibration_panel(
        train,
        test,
        train_selection.indices,
        test_selection.indices,
    )
    if not calibration["passed"]:
        raise RuntimeError(
            f"calibration rank preservation failed: Spearman={calibration['spearman']}"
        )
    source_hash = sha256_file(Path(__file__).resolve())
    meta = {
        "schema": SCHEMA,
        "dataset_hash": "ZTS_small",
        "dataset_number": 43,
        "source_dataset_name": "ZTS_small",
        "source_family": parent_meta.get("source_family"),
        "source_platform": parent_meta.get("source_platform"),
        "task": "regression",
        "provenance_group": "ZTS",
        "parent_dataset": "ZTS",
        "parent": {
            "path": str(parent),
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
            "meta_sha256": sha256_file(meta_path),
        },
        "selection": {
            "code_path": str(Path(__file__).resolve()),
            "code_sha256": source_hash,
            "rng_seed": SEED,
            "projection_dimensions": PROJECTION_DIMENSIONS,
            "candidate_coresets": CANDIDATES,
            "stratified_random_baselines": RANDOM_BASELINES,
            "target_strata": STRATA,
            "rows_per_stratum": ROWS_PER_STRATUM,
            "minimum_level_support": MIN_LEVEL_SUPPORT,
            "maximum_low_cardinality": MAX_LOW_CARDINALITY,
            "robust_scaling_fit_scope": "original ZTS training features only",
            "projection_fit_scope": "original ZTS training features only",
            "projection_matrix_sha256": sha256_bytes(
                projection.matrix.astype("<f8").tobytes()
            ),
            "train": selection_record(
                train_selection,
                target_strata(train[:, -1]),
            ),
            "test": selection_record(
                test_selection,
                target_strata(test[:, -1]),
            ),
        },
        "calibration_panel": calibration,
        "headers_removed_from_csv": True,
        "target_column": parent_meta["target_column"],
        "target_is_final_column": True,
        "test_has_target": True,
        "n_features": 147,
        "feature_columns": parent_meta.get("feature_columns"),
        "output_headers": parent_meta.get("output_headers"),
        "feature_scaling": parent_meta.get("feature_scaling"),
        "target_scaling": parent_meta.get("target_scaling"),
        "counts": {
            "train_rows": 1000,
            "test_rows": 1000,
            "n_features": 147,
        },
        "train_rows": 1000,
        "test_rows": 1000,
        "artifacts": {
            "train.csv": {
                "rows": 1000,
                "columns": 148,
                "sha256": sha256_bytes(train_csv),
            },
            "test.csv": {
                "rows": 1000,
                "columns": 148,
                "sha256": sha256_bytes(test_csv),
            },
            "train.source-rows.json": {
                "sha256": sha256_bytes(canonical_json(train_rows_manifest)),
            },
            "test.source-rows.json": {
                "sha256": sha256_bytes(canonical_json(test_rows_manifest)),
            },
        },
        "redistribution": {
            "raw_csv_publication": "withheld-pending-documented-rights",
            "manifest_publication": "permitted",
        },
    }
    expected = {
        "train.csv": train_csv,
        "test.csv": test_csv,
        "train.source-rows.json": canonical_json(train_rows_manifest),
        "test.source-rows.json": canonical_json(test_rows_manifest),
        "meta.json": canonical_json(meta),
    }
    if verify_only:
        for name, content in expected.items():
            path = output / name
            if not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(f"deterministic verification failed for {path}")
    else:
        for name, content in expected.items():
            atomic_immutable_write(output / name, content)
    return {
        "output": str(output),
        "train_sha256": sha256_bytes(train_csv),
        "test_sha256": sha256_bytes(test_csv),
        "meta_sha256": sha256_bytes(expected["meta.json"]),
        "train_candidate": train_selection.candidate,
        "test_candidate": test_selection.candidate,
        "calibration_spearman": calibration["spearman"],
        "verified": verify_only,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("/home/ubuntu/remote_super/quant/regression/ZTS"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/remote_super/quant/regression/ZTS_small"),
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build(args.parent.resolve(), args.output.resolve(), verify_only=args.verify),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
