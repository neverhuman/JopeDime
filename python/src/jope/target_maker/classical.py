from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import (
    TASK_BINARY,
    TASK_NAMES,
    DatasetRecord,
    bits_at_indices,
    load_manifest,
    sample_indices,
    stable_seed,
)


@dataclass
class ClassicalBaselineConfig:
    cache_dir: Path
    run_dir: Path
    window_size: int = 96
    views_per_record: int = 4
    limit_per_task: int | None = 5000
    train_pairs: int = 50_000
    val_pairs: int = 10_000
    test_pairs: int = 10_000
    retrieval_queries: int = 1000
    retrieval_candidates: int = 256
    seed: int = 17
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    positive_fraction: float = 0.50
    same_task_negative_fraction: float = 0.80
    include_cache_profile: bool = False
    models: tuple[str, ...] = ("length_only", "dummy", "sgd_logistic", "hist_gbdt", "extra_trees", "ensemble")
    n_jobs: int = -1


@dataclass(frozen=True)
class WindowFeature:
    dataset_id: str
    task: int
    features: np.ndarray


@dataclass(frozen=True)
class PairIndex:
    left: int
    right: int
    label: int


@dataclass(frozen=True)
class RetrievalQuery:
    query: int
    candidates: tuple[int, ...]


class WeightedScoreEnsemble:
    def __init__(self, models: Sequence[tuple[str, object]], weights: Sequence[float]) -> None:
        self.models = list(models)
        weights_array = np.asarray(list(weights), dtype=np.float64)
        if (
            weights_array.size != len(self.models)
            or not np.isfinite(weights_array).all()
            or float(weights_array.sum()) <= 0.0
        ):
            weights_array = np.ones(len(self.models), dtype=np.float64)
        self.weights = weights_array / max(1.0e-12, float(weights_array.sum()))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("WeightedScoreEnsemble has no fitted models")
        score = np.zeros(x.shape[0], dtype=np.float64)
        for weight, (_, model) in zip(self.weights, self.models):
            score += float(weight) * _score_model(model, x)
        score = np.clip(score, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1.0e-12 else 0.0


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0.0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(max(2, probabilities.size)))


def _trimmed_mean(sorted_values: np.ndarray, trim: float) -> float:
    n = sorted_values.size
    lo = int(math.floor(n * trim))
    hi = n - lo
    if lo >= hi:
        return float(sorted_values.mean())
    return float(sorted_values[lo:hi].mean())


def _winsorized_mean(values: np.ndarray, low: float, high: float) -> float:
    lo, hi = np.quantile(values, [low, high])
    return float(np.clip(values, lo, hi).mean())


def _gini(values: np.ndarray) -> float:
    x = np.sort(np.clip(values.astype(np.float64, copy=False), 0.0, None))
    n = x.size
    total = float(x.sum())
    if n == 0 or total <= 1.0e-12:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x) / (n * total)) - ((n + 1.0) / n))


def _theil(values: np.ndarray) -> float:
    x = np.clip(values.astype(np.float64, copy=False), 1.0e-12, None)
    mean = float(x.mean())
    if mean <= 1.0e-12:
        return 0.0
    ratio = x / mean
    return float(np.mean(ratio * np.log(ratio)))


def _pairwise_mean_abs(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    diff = np.abs(values[:, None] - values[None, :])
    return float(diff.mean())


def _uniform_distances(sorted_values: np.ndarray) -> tuple[float, float, float]:
    n = sorted_values.size
    if n == 0:
        return 0.0, 0.0, 0.0
    ecdf_hi = np.arange(1, n + 1, dtype=np.float64) / n
    ecdf_lo = np.arange(0, n, dtype=np.float64) / n
    ks = float(np.maximum(np.abs(ecdf_hi - sorted_values), np.abs(sorted_values - ecdf_lo)).max())
    cvm = float(
        (1.0 / (12.0 * n)) + np.sum((sorted_values - ((2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n))) ** 2)
    )
    wasserstein = float(np.abs(sorted_values - ((np.arange(1, n + 1) - 0.5) / n)).mean())
    return ks, cvm, wasserstein


def _finite_unit_interval(values: Iterable[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.clip(x, 0.0, 1.0)


def summarize_target_window(
    values: Iterable[float] | np.ndarray,
    *,
    task: int,
    original_length: int,
    effective_count: int,
    replacement_fraction: float,
    window_size: int | None = None,
    cache_profile: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a wide, non-neural summary vector for one sampled target window."""
    x = _finite_unit_interval(values)
    sx = np.sort(x)
    n = float(x.size)
    names: list[str] = []
    feats: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        feats.append(float(value) if math.isfinite(float(value)) else 0.0)

    add("task_is_binary", float(task == TASK_BINARY))
    add("task_is_regression", float(task != TASK_BINARY))
    add("log_original_length", math.log1p(max(0, int(original_length))))
    add("log_effective_count", math.log1p(max(0, int(effective_count))))
    add("replacement_fraction", replacement_fraction)
    add("sample_count", n)
    add("sample_count_over_window", _safe_div(n, float(window_size or x.size or 1)))

    mean = float(x.mean())
    var = float(x.var())
    std = math.sqrt(max(var, 0.0))
    centered = x - mean
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    mean_abs_dev = float(np.mean(np.abs(x - mean)))
    add("mean", mean)
    add("std", std)
    add("variance", var)
    add("sem", _safe_div(std, math.sqrt(n)))
    add("min", float(sx[0]))
    add("max", float(sx[-1]))
    add("range", float(sx[-1] - sx[0]))
    add("median", median)
    add("mean_minus_median", mean - median)
    add("mad", mad)
    add("mean_abs_dev", mean_abs_dev)
    add("coef_variation", _safe_div(std, abs(mean) + 1.0e-6))
    add("logit_mean", math.log((mean + 1.0e-6) / (1.0 - mean + 1.0e-6)))

    for power in range(2, 13):
        add(f"raw_moment_{power}", float(np.mean(x**power)))
    for power in range(2, 11):
        central = float(np.mean(centered**power))
        add(f"central_moment_{power}", central)
        if power >= 3:
            add(f"standardized_moment_{power}", _safe_div(central, (std + 1.0e-6) ** power))

    quantile_probs = np.asarray(
        [
            0.0,
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
            0.10,
            0.125,
            0.20,
            0.25,
            0.333,
            0.375,
            0.50,
            0.625,
            0.667,
            0.75,
            0.80,
            0.875,
            0.90,
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            1.0,
        ],
        dtype=np.float64,
    )
    quantiles = np.quantile(x, quantile_probs)
    qmap = {float(p): float(q) for p, q in zip(quantile_probs, quantiles)}
    for p, q in zip(quantile_probs, quantiles):
        add(f"q_{p:.3f}", float(q))
    for left, right in [(0.25, 0.75), (0.10, 0.90), (0.05, 0.95), (0.01, 0.99), (0.001, 0.999)]:
        add(f"q_range_{left:.3f}_{right:.3f}", qmap[right] - qmap[left])
    for i in range(len(quantile_probs) - 1):
        add(
            f"q_gap_{quantile_probs[i]:.3f}_{quantile_probs[i + 1]:.3f}",
            float(quantiles[i + 1] - quantiles[i]),
        )

    iqr = qmap[0.75] - qmap[0.25]
    add("iqr", iqr)
    add("bowley_skew", _safe_div(qmap[0.75] + qmap[0.25] - 2.0 * qmap[0.50], iqr + 1.0e-6))
    add("kelly_skew", _safe_div(qmap[0.90] + qmap[0.10] - 2.0 * qmap[0.50], qmap[0.90] - qmap[0.10] + 1.0e-6))
    add("moors_kurtosis", _safe_div((qmap[0.875] - qmap[0.625]) + (qmap[0.375] - qmap[0.125]), iqr + 1.0e-6))

    for trim in (0.01, 0.05, 0.10, 0.20, 0.25):
        add(f"trimmed_mean_{trim:.2f}", _trimmed_mean(sx, trim))
        add(f"winsorized_mean_{trim:.2f}", _winsorized_mean(x, trim, 1.0 - trim))

    thresholds = np.asarray(
        [
            0.0,
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            1.0,
        ]
    )
    for threshold in thresholds:
        add(f"cdf_le_{threshold:.3f}", float(np.mean(x <= threshold)))
        add(f"tail_ge_{threshold:.3f}", float(np.mean(x >= threshold)))
    add("mass_exact_zero", float(np.mean(x <= 1.0e-12)))
    add("mass_exact_one", float(np.mean(x >= 1.0 - 1.0e-12)))
    add("mass_mid_40_60", float(np.mean((x >= 0.40) & (x <= 0.60))))
    add("mass_tails_05_95", float(np.mean((x <= 0.05) | (x >= 0.95))))

    for bins in (4, 8, 16, 32, 64):
        hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
        p = hist.astype(np.float64) / max(1.0, float(hist.sum()))
        add(f"hist_{bins}_entropy", _entropy(p))
        add(f"hist_{bins}_simpson", float(1.0 - np.square(p).sum()))
        add(f"hist_{bins}_max_bin_mass", float(p.max()))
        add(f"hist_{bins}_empty_bin_fraction", float(np.mean(p <= 0.0)))
        for index, value in enumerate(p):
            add(f"hist_{bins}_bin_{index:02d}", float(value))

    hist64, _ = np.histogram(x, bins=64, range=(0.0, 1.0))
    hist64 = hist64.astype(np.float64) / max(1.0, float(hist64.sum()))
    fft = np.fft.rfft(hist64 - hist64.mean())
    for index in range(1, min(17, fft.size)):
        add(f"hist64_fft_real_{index:02d}", float(fft[index].real))
        add(f"hist64_fft_imag_{index:02d}", float(fft[index].imag))
        add(f"hist64_fft_abs_{index:02d}", float(abs(fft[index])))

    gaps = np.diff(np.concatenate([np.asarray([0.0]), sx, np.asarray([1.0])]))
    add("sorted_gap_mean", float(gaps.mean()))
    add("sorted_gap_std", float(gaps.std()))
    add("sorted_gap_min", float(gaps.min()))
    add("sorted_gap_max", float(gaps.max()))
    add("sorted_gap_endpoint_left", float(sx[0]))
    add("sorted_gap_endpoint_right", float(1.0 - sx[-1]))
    for p, q in zip(
        (0.10, 0.25, 0.50, 0.75, 0.90, 0.99), np.quantile(gaps, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    ):
        add(f"sorted_gap_q_{p:.2f}", float(q))

    add("gini", _gini(x))
    add("theil", _theil(x))
    add("pairwise_mean_abs_diff", _pairwise_mean_abs(x))
    ks, cvm, wasserstein = _uniform_distances(sx)
    add("ks_distance_uniform", ks)
    add("cvm_distance_uniform", cvm)
    add("wasserstein_l1_uniform", wasserstein)

    for decimals in (0, 1, 2, 3, 4, 6):
        rounded = np.round(x, decimals=decimals)
        _, counts = np.unique(rounded, return_counts=True)
        probs = counts.astype(np.float64) / max(1.0, float(counts.sum()))
        add(f"unique_fraction_round_{decimals}", float(counts.size / n))
        add(f"mode_fraction_round_{decimals}", float(probs.max()))
        add(f"entropy_round_{decimals}", _entropy(probs))

    y = 2.0 * x - 1.0
    for degree in range(1, 13):
        coeff = np.zeros(degree + 1, dtype=np.float64)
        coeff[degree] = 1.0
        add(f"legendre_mean_{degree:02d}", float(np.polynomial.legendre.legval(y, coeff).mean()))

    for freq in range(1, 17):
        angles = 2.0 * math.pi * freq * x
        sin_mean = float(np.sin(angles).mean())
        cos_mean = float(np.cos(angles).mean())
        add(f"ecf_sin_{freq:02d}", sin_mean)
        add(f"ecf_cos_{freq:02d}", cos_mean)
        add(f"ecf_abs_{freq:02d}", math.hypot(sin_mean, cos_mean))

    centers = np.linspace(0.0, 1.0, 33)
    for bandwidth in (0.025, 0.05, 0.10, 0.20):
        denom = 2.0 * bandwidth * bandwidth
        for index, center in enumerate(centers):
            add(f"rbf_bw_{bandwidth:.3f}_{index:02d}", float(np.exp(-((x - center) ** 2) / denom).mean()))

    binary_rate = float(np.mean(x >= 0.5))
    binary_entropy = 0.0
    if 0.0 < binary_rate < 1.0:
        binary_entropy = float(
            -(binary_rate * math.log(binary_rate) + (1.0 - binary_rate) * math.log(1.0 - binary_rate))
            / math.log(2.0)
        )
    add("binary_rate_at_05", binary_rate)
    add("binary_balance", 1.0 - abs(2.0 * binary_rate - 1.0))
    add("binary_entropy", binary_entropy)
    add("bernoulli_variance", binary_rate * (1.0 - binary_rate))

    if cache_profile is not None:
        profile = np.asarray(cache_profile, dtype=np.float64).reshape(-1)
        for index, value in enumerate(np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)):
            add(f"cache_profile_{index:02d}", float(value))

    feature_array = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return feature_array, names


def _limit_records(
    records: Sequence[DatasetRecord], limit_per_task: int | None, seed: int
) -> list[DatasetRecord]:
    if limit_per_task is None:
        return list(records)
    limited: list[DatasetRecord] = []
    for task in sorted(TASK_NAMES):
        task_records = [record for record in records if record.task == task]
        task_records.sort(key=lambda record: stable_seed("classical-limit", seed, record.dataset_id))
        limited.extend(task_records[: max(0, int(limit_per_task))])
    return limited


def _split_records(
    records: Sequence[DatasetRecord],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[DatasetRecord]]:
    splits = {"train": [], "val": [], "test": []}
    for task in sorted(TASK_NAMES):
        task_records = [record for record in records if record.task == task]
        task_records.sort(key=lambda record: stable_seed("classical-split", seed, record.dataset_id))
        n = len(task_records)
        if n == 0:
            continue
        if n == 1:
            splits["train"].extend(task_records)
            continue
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = 1 if n >= 3 else 0
            n_train = max(1, n - n_val - 1)
        splits["train"].extend(task_records[:n_train])
        splits["val"].extend(task_records[n_train : n_train + n_val])
        splits["test"].extend(task_records[n_train + n_val :])
    return splits


def _sample_window_features(
    records: Sequence[DatasetRecord],
    cfg: ClassicalBaselineConfig,
) -> tuple[list[WindowFeature], list[str]]:
    windows: list[WindowFeature] = []
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            profile = np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            for view_index in range(cfg.views_per_record):
                rng = np.random.default_rng(
                    stable_seed("classical-window", cfg.seed, record.dataset_id, view_index, cfg.window_size)
                )
                indices, effective_count, replacement_fraction = sample_indices(
                    record.length, cfg.window_size, rng
                )
                if record.task == TASK_BINARY:
                    values = bits_at_indices(npz["bits"], indices)
                else:
                    values = np.asarray(npz["values"], dtype=np.float32)[indices]
                features, names = summarize_target_window(
                    values,
                    task=record.task,
                    original_length=record.length,
                    effective_count=effective_count,
                    replacement_fraction=replacement_fraction,
                    window_size=cfg.window_size,
                    cache_profile=profile,
                )
                if feature_names is None:
                    feature_names = names
                elif len(feature_names) != len(names):
                    raise RuntimeError("Feature dimension changed while sampling windows")
                windows.append(WindowFeature(record.dataset_id, record.task, features))
    if feature_names is None:
        raise ValueError("No windows sampled")
    return windows, feature_names


def pair_feature_names(feature_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for prefix in (
        "diff",
        "absdiff",
        "sqdiff",
        "minimum",
        "maximum",
        "ratio",
        "log_absdiff",
        "product",
        "mean",
    ):
        names.extend([f"{prefix}:{name}" for name in feature_names])
    names.extend(
        [
            "global:mean_absdiff",
            "global:max_absdiff",
            "global:rmse",
            "global:cosine_similarity",
            "global:cosine_distance",
            "global:dot_product_mean",
            "global:mean_signed_diff",
            "global:std_absdiff",
        ]
    )
    return names


def pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.nan_to_num(np.asarray(left, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    right = np.nan_to_num(np.asarray(right, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    diff = left - right
    absdiff = np.abs(diff)
    denominator = np.where(np.abs(right) > 1.0e-6, right, np.where(right >= 0.0, 1.0e-6, -1.0e-6))
    ratio = np.clip(left / denominator, -1.0e4, 1.0e4)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = _safe_div(float(np.dot(left, right)), left_norm * right_norm)
    global_features = np.asarray(
        [
            float(absdiff.mean()),
            float(absdiff.max(initial=0.0)),
            math.sqrt(float(np.mean(diff * diff))),
            cosine,
            1.0 - cosine,
            float(np.mean(left * right)),
            float(diff.mean()),
            float(absdiff.std()),
        ],
        dtype=np.float32,
    )
    out = np.concatenate(
        [
            diff,
            absdiff,
            diff * diff,
            np.minimum(left, right),
            np.maximum(left, right),
            ratio,
            np.log1p(absdiff),
            left * right,
            0.5 * (left + right),
            global_features,
        ]
    )
    return np.nan_to_num(out.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def length_feature_indices(feature_names: Sequence[str]) -> list[int]:
    length_names = {
        "log_original_length",
        "log_effective_count",
        "replacement_fraction",
        "sample_count",
        "sample_count_over_window",
    }
    return [index for index, name in enumerate(feature_names) if name in length_names]


def no_length_feature_indices(feature_names: Sequence[str]) -> list[int]:
    blocked = set(length_feature_indices(feature_names))
    return [index for index in range(len(feature_names)) if index not in blocked]


def select_window_features(windows: Sequence[WindowFeature], indices: Sequence[int]) -> list[WindowFeature]:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Feature selection cannot be empty")
    return [WindowFeature(window.dataset_id, window.task, window.features[idx]) for window in windows]


def build_pair_indices(
    windows: Sequence[WindowFeature],
    pair_count: int,
    *,
    seed: int,
    positive_fraction: float,
    same_task_negative_fraction: float,
) -> list[PairIndex]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    rng = np.random.default_rng(seed)
    by_record: dict[str, list[int]] = {}
    task_by_record: dict[str, int] = {}
    for index, window in enumerate(windows):
        by_record.setdefault(window.dataset_id, []).append(index)
        task_by_record[window.dataset_id] = window.task
    positive_records = [record_id for record_id, indexes in by_record.items() if len(indexes) >= 2]
    all_records = sorted(by_record)
    records_by_task: dict[int, list[str]] = {}
    for record_id in all_records:
        records_by_task.setdefault(task_by_record[record_id], []).append(record_id)
    same_task_negative_tasks = [task for task, record_ids in records_by_task.items() if len(record_ids) >= 2]
    if not positive_records:
        raise ValueError("Need at least one record with two sampled windows for positive pairs")
    if len(all_records) < 2:
        raise ValueError("Need at least two records for negative pairs")

    pairs: list[PairIndex] = []
    positive_count = int(round(pair_count * positive_fraction))
    positive_count = min(max(1, positive_count), pair_count - 1)

    def choose_window(record_id: str) -> int:
        indexes = by_record[record_id]
        return int(indexes[rng.integers(0, len(indexes))])

    for row in range(pair_count):
        if row < positive_count:
            record_id = positive_records[int(rng.integers(0, len(positive_records)))]
            left, right = rng.choice(by_record[record_id], size=2, replace=False)
            pairs.append(PairIndex(int(left), int(right), 1))
        else:
            if same_task_negative_tasks and rng.random() < same_task_negative_fraction:
                task = same_task_negative_tasks[int(rng.integers(0, len(same_task_negative_tasks)))]
                left_record, right_record = rng.choice(records_by_task[task], size=2, replace=False)
            else:
                left_record, right_record = rng.choice(all_records, size=2, replace=False)
            pairs.append(PairIndex(choose_window(str(left_record)), choose_window(str(right_record)), 0))

    order = rng.permutation(pair_count)
    return [pairs[int(index)] for index in order]


def pair_dataset_from_indices(
    windows: Sequence[WindowFeature], pairs: Sequence[PairIndex]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([window.features for window in windows]).astype(np.float32, copy=False)
    pair_dim = pair_features(features[0], features[0]).shape[0]
    x_pairs = np.empty((len(pairs), pair_dim), dtype=np.float32)
    y = np.empty(len(pairs), dtype=np.int64)
    for row, pair in enumerate(pairs):
        x_pairs[row] = pair_features(features[pair.left], features[pair.right])
        y[row] = int(pair.label)
    return x_pairs, y


def _build_pair_dataset(
    windows: Sequence[WindowFeature],
    pair_count: int,
    *,
    seed: int,
    positive_fraction: float,
    same_task_negative_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = build_pair_indices(
        windows,
        pair_count,
        seed=seed,
        positive_fraction=positive_fraction,
        same_task_negative_fraction=same_task_negative_fraction,
    )
    return pair_dataset_from_indices(windows, pairs)


def _model_specs(model_names: Sequence[str], seed: int, n_jobs: int) -> dict[str, object]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    specs: dict[str, object] = {}
    for name in model_names:
        key = name.strip()
        if not key:
            continue
        if key == "ensemble":
            continue
        if key == "dummy":
            specs[key] = DummyClassifier(strategy="prior")
        elif key in {"sgd_logistic", "length_only"}:
            specs[key] = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=1.0e-4,
                    l1_ratio=0.05,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1.0e-4,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=8,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            )
        elif key == "hist_gbdt":
            specs[key] = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0e-3,
                random_state=seed,
            )
        elif key == "extra_trees":
            specs[key] = ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Unknown classical model {key!r}")
    if not specs:
        raise ValueError("At least one non-ensemble model must be selected")
    return specs


def _ensemble_requested(model_names: Sequence[str]) -> bool:
    return any(name.strip() == "ensemble" for name in model_names)


def _ensemble_members(model_reports: Mapping[str, object], fitted_models: Mapping[str, object]) -> list[str]:
    preferred = [name for name in ("sgd_logistic", "hist_gbdt", "extra_trees") if name in fitted_models]
    return preferred or [name for name in fitted_models if name != "dummy" and name != "length_only"]


def _validation_weights(
    model_names: Sequence[str], model_reports: Mapping[str, object], y_val: np.ndarray
) -> np.ndarray:
    weights: list[float] = []
    positive_rate = float(np.mean(y_val))
    for name in model_names:
        report = model_reports.get(name, {})
        val = report.get("val", {}) if isinstance(report, Mapping) else {}
        auc = float(val.get("roc_auc", 0.5)) if isinstance(val, Mapping) else 0.5
        ap = float(val.get("average_precision", positive_rate)) if isinstance(val, Mapping) else positive_rate
        weights.append(max(0.0, auc - 0.5) + max(0.0, ap - positive_rate))
    if not weights or sum(weights) <= 0.0:
        return np.ones(len(model_names), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def _score_model(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=np.float64)))
    else:
        scores = model.predict(x)
    return np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1.0e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    clipped = np.clip(scores, 1.0e-6, 1.0 - 1.0e-6)
    predicted = (scores >= threshold).astype(np.int64)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }
    return metrics


def rank_metrics(ranks: Sequence[int], candidate_count: int) -> dict[str, float]:
    if not ranks:
        return {}
    rank_array = np.asarray(ranks, dtype=np.float64)
    return {
        "retrieval_candidates": float(candidate_count),
        "retrieval@1": float(np.mean(rank_array <= 1)),
        "retrieval@5": float(np.mean(rank_array <= 5)),
        "retrieval@10": float(np.mean(rank_array <= 10)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "median_rank": float(np.median(rank_array)),
    }


def _top_features(model: object, names: Sequence[str], limit: int = 30) -> list[dict[str, float | str]]:
    estimator = model
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        estimator = steps[-1]
    values: np.ndarray | None = None
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
    if values is None or values.size != len(names):
        return []
    order = np.argsort(np.abs(values))[::-1][:limit]
    return [{"feature": names[int(index)], "value": float(values[int(index)])} for index in order]


def build_retrieval_queries(
    windows: Sequence[WindowFeature],
    *,
    queries: int,
    candidates: int,
    seed: int,
) -> list[RetrievalQuery]:
    if queries <= 0:
        return []
    rng = np.random.default_rng(seed)
    by_record: dict[str, list[int]] = {}
    for index, window in enumerate(windows):
        by_record.setdefault(window.dataset_id, []).append(index)
    positive_records = [record_id for record_id, indexes in by_record.items() if len(indexes) >= 2]
    all_records = sorted(by_record)
    if not positive_records or len(all_records) < 2:
        return []

    candidate_count = max(2, int(candidates))
    built: list[RetrievalQuery] = []
    for _ in range(int(queries)):
        record_id = positive_records[int(rng.integers(0, len(positive_records)))]
        query_index, positive_index = rng.choice(by_record[record_id], size=2, replace=False)
        negative_records = [candidate for candidate in all_records if candidate != record_id]
        replace = len(negative_records) < candidate_count - 1
        chosen_records = rng.choice(negative_records, size=candidate_count - 1, replace=replace)
        candidate_indexes = [int(positive_index)]
        for negative_record in chosen_records:
            indexes = by_record[str(negative_record)]
            candidate_indexes.append(int(indexes[rng.integers(0, len(indexes))]))
        built.append(RetrievalQuery(int(query_index), tuple(candidate_indexes)))
    return built


def retrieval_metrics_from_scores(
    scores_by_query: Sequence[np.ndarray], candidate_count: int
) -> dict[str, float]:
    ranks: list[int] = []
    for scores in scores_by_query:
        order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
    return rank_metrics(ranks, candidate_count)


def retrieval_metrics_for_model(
    model: object,
    windows: Sequence[WindowFeature],
    queries: Sequence[RetrievalQuery],
) -> dict[str, float]:
    if not queries:
        return {}
    features = np.stack([window.features for window in windows]).astype(np.float32, copy=False)
    scores_by_query: list[np.ndarray] = []
    for query in queries:
        x_pairs = np.stack(
            [pair_features(features[int(query.query)], features[int(index)]) for index in query.candidates]
        )
        scores_by_query.append(_score_model(model, x_pairs))
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))


def _retrieval_metrics(
    model: object,
    windows: Sequence[WindowFeature],
    *,
    queries: int,
    candidates: int,
    seed: int,
) -> dict[str, float]:
    built = build_retrieval_queries(windows, queries=queries, candidates=candidates, seed=seed)
    return retrieval_metrics_for_model(model, windows, built)


def _jsonable_config(cfg: ClassicalBaselineConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["cache_dir"] = str(cfg.cache_dir)
    data["run_dir"] = str(cfg.run_dir)
    data["models"] = list(cfg.models)
    return data


def _model_artifact_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)


def run_classical_baseline(cfg: ClassicalBaselineConfig) -> dict[str, object]:
    started = time.time()
    records = load_manifest(cfg.cache_dir, split="all", seed=cfg.seed)
    records = _limit_records(records, cfg.limit_per_task, cfg.seed)
    if len(records) < 3:
        raise ValueError("Need at least three records for train/val/test classical baseline splits")
    splits = _split_records(
        records, train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(
            f"Non-empty train/val/test splits are required; got {[len(splits[key]) for key in ('train', 'val', 'test')]}"
        )

    sampled: dict[str, list[WindowFeature]] = {}
    feature_names: list[str] | None = None
    for split_name, split_records in splits.items():
        windows, names = _sample_window_features(split_records, cfg)
        sampled[split_name] = windows
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed across splits")
    assert feature_names is not None
    pair_names = pair_feature_names(feature_names)
    length_indices = length_feature_indices(feature_names)
    if not length_indices:
        raise RuntimeError("No length feature indices were found")
    no_length_indices = no_length_feature_indices(feature_names)
    full_indices = tuple(range(len(feature_names)))
    length_indices_tuple = tuple(length_indices)
    no_length_indices_tuple = tuple(no_length_indices)

    pair_indices = {
        "train": build_pair_indices(
            sampled["train"],
            cfg.train_pairs,
            seed=stable_seed("pairs", cfg.seed, "train"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "val": build_pair_indices(
            sampled["val"],
            cfg.val_pairs,
            seed=stable_seed("pairs", cfg.seed, "val"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "test": build_pair_indices(
            sampled["test"],
            cfg.test_pairs,
            seed=stable_seed("pairs", cfg.seed, "test"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
    }
    retrieval_queries = build_retrieval_queries(
        sampled["test"],
        queries=cfg.retrieval_queries,
        candidates=cfg.retrieval_candidates,
        seed=stable_seed("retrieval", cfg.seed, "shared"),
    )

    matrix_cache: dict[
        tuple[int, ...],
        tuple[
            dict[str, list[WindowFeature]],
            list[str],
            list[str],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    def matrices_for(indices: tuple[int, ...]):
        cached = matrix_cache.get(indices)
        if cached is not None:
            return cached
        selected = {
            split_name: select_window_features(windows, indices) for split_name, windows in sampled.items()
        }
        selected_names = [feature_names[index] for index in indices]
        selected_pair_names = pair_feature_names(selected_names)
        x_train, y_train = pair_dataset_from_indices(selected["train"], pair_indices["train"])
        x_val, y_val = pair_dataset_from_indices(selected["val"], pair_indices["val"])
        x_test, y_test = pair_dataset_from_indices(selected["test"], pair_indices["test"])
        cached = (
            selected,
            selected_names,
            selected_pair_names,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
        )
        matrix_cache[indices] = cached
        return cached

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cfg.run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(name.strip() for name in cfg.models if name.strip()))
    base_requested = [name for name in requested if name != "ensemble"]
    if "length_only" not in base_requested:
        base_requested.append("length_only")
    full_specs = _model_specs(base_requested, cfg.seed, cfg.n_jobs)

    def fit_and_report(
        specs: Mapping[str, object],
        *,
        feature_indices: tuple[int, ...],
        report_prefix: str = "",
        include_ensemble: bool = False,
    ) -> dict[str, object]:
        selected, selected_names, selected_pair_names, x_train, y_train, x_val, y_val, x_test, y_test = (
            matrices_for(feature_indices)
        )
        reports: dict[str, object] = {}
        fitted: dict[str, object] = {}
        for model_name, model in specs.items():
            fit_started = time.time()
            model.fit(x_train, y_train)
            val_scores = _score_model(model, x_val)
            threshold = _best_f1_threshold(y_val, val_scores)
            test_scores = _score_model(model, x_test)
            val_metrics = _classification_metrics(y_val, val_scores, threshold)
            test_metrics = _classification_metrics(y_test, test_scores, threshold)
            retrieval = retrieval_metrics_for_model(model, selected["test"], retrieval_queries)
            artifact_name = _model_artifact_name(f"{report_prefix}{model_name}")
            model_path = model_dir / f"{artifact_name}.joblib"
            import joblib

            joblib.dump(
                {
                    "model": model,
                    "model_name": f"{report_prefix}{model_name}",
                    "feature_names": selected_names,
                    "pair_feature_names": selected_pair_names,
                    "selected_feature_indices": list(feature_indices),
                    "config": _jsonable_config(cfg),
                },
                model_path,
            )
            reports[model_name] = {
                "fit_seconds": time.time() - fit_started,
                "val": val_metrics,
                "test": test_metrics,
                "retrieval": retrieval,
                "top_features": _top_features(model, selected_pair_names),
                "model_path": str(model_path),
            }
            fitted[model_name] = model
        if include_ensemble:
            member_names = _ensemble_members(reports, fitted)
            if member_names:
                weights = _validation_weights(member_names, reports, y_val)
                ensemble = WeightedScoreEnsemble([(name, fitted[name]) for name in member_names], weights)
                val_scores = _score_model(ensemble, x_val)
                threshold = _best_f1_threshold(y_val, val_scores)
                test_scores = _score_model(ensemble, x_test)
                artifact_name = _model_artifact_name(f"{report_prefix}ensemble")
                model_path = model_dir / f"{artifact_name}.joblib"
                import joblib

                joblib.dump(
                    {
                        "model": ensemble,
                        "model_name": f"{report_prefix}ensemble",
                        "feature_names": selected_names,
                        "pair_feature_names": selected_pair_names,
                        "selected_feature_indices": list(feature_indices),
                        "member_names": member_names,
                        "weights": weights.tolist(),
                        "config": _jsonable_config(cfg),
                    },
                    model_path,
                )
                reports["ensemble"] = {
                    "fit_seconds": 0.0,
                    "val": _classification_metrics(y_val, val_scores, threshold),
                    "test": _classification_metrics(y_test, test_scores, threshold),
                    "retrieval": retrieval_metrics_for_model(ensemble, selected["test"], retrieval_queries),
                    "top_features": [],
                    "member_names": member_names,
                    "weights": weights.tolist(),
                    "model_path": str(model_path),
                }
        return reports

    model_reports: dict[str, object] = {}
    full_model_specs: dict[str, object] = {}
    for model_name, model in full_specs.items():
        indices = length_indices_tuple if model_name == "length_only" else full_indices
        model_reports.update(
            fit_and_report(
                {model_name: model},
                feature_indices=indices,
                include_ensemble=False,
            )
        )
    if _ensemble_requested(requested):
        ensemble_base = {
            name: full_specs[name]
            for name in ("sgd_logistic", "hist_gbdt", "extra_trees")
            if name in full_specs
        }
        if not ensemble_base:
            ensemble_base = {
                name: full_specs[name] for name in full_specs if name not in {"dummy", "length_only"}
            }
        if ensemble_base:
            full_model_specs = {name: model_reports[name] for name in ensemble_base if name in model_reports}
            _, _, _, _, _, x_val, y_val, _, _ = matrices_for(full_indices)
            fitted_models: dict[str, object] = {}
            import joblib

            for name in ensemble_base:
                report = model_reports.get(name, {})
                path = report.get("model_path") if isinstance(report, Mapping) else None
                if path:
                    fitted_models[name] = joblib.load(path)["model"]
            member_names = _ensemble_members(full_model_specs, fitted_models)
            if member_names:
                weights = _validation_weights(member_names, full_model_specs, y_val)
                selected, selected_names, selected_pair_names, _, _, x_val, y_val, x_test, y_test = (
                    matrices_for(full_indices)
                )
                ensemble = WeightedScoreEnsemble(
                    [(name, fitted_models[name]) for name in member_names], weights
                )
                val_scores = _score_model(ensemble, x_val)
                threshold = _best_f1_threshold(y_val, val_scores)
                test_scores = _score_model(ensemble, x_test)
                model_path = model_dir / "ensemble.joblib"
                joblib.dump(
                    {
                        "model": ensemble,
                        "model_name": "ensemble",
                        "feature_names": selected_names,
                        "pair_feature_names": selected_pair_names,
                        "selected_feature_indices": list(full_indices),
                        "member_names": member_names,
                        "weights": weights.tolist(),
                        "config": _jsonable_config(cfg),
                    },
                    model_path,
                )
                model_reports["ensemble"] = {
                    "fit_seconds": 0.0,
                    "val": _classification_metrics(y_val, val_scores, threshold),
                    "test": _classification_metrics(y_test, test_scores, threshold),
                    "retrieval": retrieval_metrics_for_model(ensemble, selected["test"], retrieval_queries),
                    "top_features": [],
                    "member_names": member_names,
                    "weights": weights.tolist(),
                    "model_path": str(model_path),
                }

    no_length_names = [name for name in base_requested if name not in {"dummy", "length_only"}]
    if not no_length_names:
        no_length_names = ["sgd_logistic"]
    no_length_specs = _model_specs(no_length_names, cfg.seed + 997, cfg.n_jobs)
    no_length_reports = fit_and_report(
        no_length_specs,
        feature_indices=no_length_indices_tuple,
        report_prefix="no_length_",
        include_ensemble=_ensemble_requested(requested),
    )

    report: dict[str, object] = {
        "config": _jsonable_config(cfg),
        "record_count": len(records),
        "split_records": {key: len(value) for key, value in splits.items()},
        "split_windows": {key: len(value) for key, value in sampled.items()},
        "window_feature_dim": len(feature_names),
        "pair_feature_dim": len(pair_names),
        "length_feature_names": [feature_names[index] for index in length_indices],
        "no_length_window_feature_dim": len(no_length_indices),
        "no_length_pair_feature_dim": len(
            pair_feature_names([feature_names[index] for index in no_length_indices])
        ),
        "feature_names": feature_names,
        "pair_feature_names": pair_names,
        "models": model_reports,
        "no_length_models": no_length_reports,
        "elapsed_seconds": time.time() - started,
    }
    report_path = cfg.run_dir / "classical_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (cfg.run_dir / "classical_metrics.jsonl").open("a", encoding="utf-8") as handle:
        for model_name, model_report in model_reports.items():
            row: dict[str, object] = {
                "phase": "classical_baseline",
                "model": model_name,
                "time": time.time(),
                "report": str(report_path),
            }
            test_metrics = model_report["test"] if isinstance(model_report, Mapping) else {}
            retrieval = model_report["retrieval"] if isinstance(model_report, Mapping) else {}
            if isinstance(test_metrics, Mapping):
                row.update({f"test_{key}": value for key, value in test_metrics.items()})
            if isinstance(retrieval, Mapping):
                row.update({f"retrieval_{key}": value for key, value in retrieval.items()})
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report
