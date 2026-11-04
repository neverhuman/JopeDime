from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .data import TASK_BINARY


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
