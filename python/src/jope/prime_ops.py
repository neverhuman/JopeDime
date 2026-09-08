"""Exact-f64 CPU reference implementations for the Prime transform cohort.

The transforms are usable for certification fixtures while the registry keeps
them quarantined from proposal generation. All randomness is derived from an
explicit seed, output width is capped before allocation, and non-finite values
are sanitized after every stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grammars.prime import maximum_output_columns, registry_snapshot

MAX_WIDTH = maximum_output_columns()
PRIME_VERSION = "prime-transforms-exact-f64-v1"
PRIME_PROTOTYPE_VERSION = "prime-prototypes-exact-f64-v3"
RADIX2_FFT_VERSION = "prime-radix2-complex-f64-v1"

_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1
_F64_EPSILON = 2.220446049250313e-16
_F64_TINY = float.fromhex("0x1.0p-1022")


def _registry_usize_bins(operator_name: str, parameter_name: str) -> tuple[int, ...]:
    operator = next(
        entry for entry in registry_snapshot() if entry["name"] == operator_name
    )
    parameter = next(
        item for item in operator["params"] if item["name"] == parameter_name
    )
    return tuple(int(item["usize"]) for item in parameter["bins"])


_TENSOR_SKETCH_WIDTHS = _registry_usize_bins("tensor_sketch", "n")
_TENSOR_SKETCH_DEGREES = _registry_usize_bins("tensor_sketch", "degree")
if not _TENSOR_SKETCH_WIDTHS or any(
    width <= 0 or width > MAX_WIDTH or width & (width - 1)
    for width in _TENSOR_SKETCH_WIDTHS
):
    raise RuntimeError("TensorSketch registry widths must be positive powers of two")


class _Pcg32:
    """Bit-exact twin of battle-runtime's PCG32 XSH-RR stream."""

    __slots__ = ("state", "inc")

    def __init__(self, seed: int, stream: int):
        self.state = 0
        self.inc = ((int(stream) << 1) | 1) & _U64_MASK
        self.next_u32()
        self.state = (self.state + (int(seed) & _U64_MASK)) & _U64_MASK
        self.next_u32()

    @classmethod
    def for_stage(cls, seed: int, in_cols: int, n_out: int) -> _Pcg32:
        stream = (((int(in_cols) & _U64_MASK) << 32) ^ int(n_out)) & _U64_MASK
        return cls(seed, stream)

    def next_u32(self) -> int:
        old = self.state
        self.state = (
            old * 6364136223846793005 + self.inc
        ) & _U64_MASK
        xorshifted = (((old >> 18) ^ old) >> 27) & _U32_MASK
        rotation = (old >> 59) & 31
        return (
            (xorshifted >> rotation)
            | (xorshifted << ((-rotation) & 31))
        ) & _U32_MASK

    def next_f64(self) -> float:
        high = self.next_u32() << 32
        low = self.next_u32()
        return ((high | low) >> 11) * (1.0 / 9007199254740992.0)

    def gaussian(self) -> float:
        u1 = max(self.next_f64(), 1e-12)
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _matrix(x: Any) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError("Prime transforms require a two-dimensional matrix")
    return np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))


def _finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _finish(x: np.ndarray, z: np.ndarray, append: bool) -> np.ndarray:
    width = max(0, min(MAX_WIDTH, z.shape[1] + (x.shape[1] if append else 0)))
    if append:
        left = min(x.shape[1], width)
        right = width - left
        # Slice before concatenate so a width-capped stage never allocates a
        # temporary wider than the runtime contract.
        return _finite(np.concatenate((x[:, :left], z[:, :right]), axis=1))
    return _finite(z[:, :width])


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    total = 0.0
    for index in range(len(left)):
        total += float(left[index]) * float(right[index])
    return total


def _fixed_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Matrix product with a portable row/column/inner reduction order."""

    if left.shape[1] != right.shape[0]:
        raise ValueError("matrix product dimensions do not match")
    out = np.empty((left.shape[0], right.shape[1]), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            total = 0.0
            for inner in range(left.shape[1]):
                total += float(left[row, inner]) * float(right[inner, column])
            out[row, column] = total
    return out


def _modified_gram_schmidt(matrix: np.ndarray) -> list[np.ndarray]:
    """Return deterministic orthonormal columns with stable degeneracy repair."""

    rows, columns = matrix.shape
    basis: list[np.ndarray] = []
    for column in range(columns):
        vector = np.asarray(matrix[:, column], dtype=np.float64).copy()
        original_norm = _dot(vector, vector)
        # Two fixed MGS passes avoid convergence-dependent behavior while
        # keeping nearly-collinear columns stable across portable twins.
        for _ in range(2):
            for previous in basis:
                projection = _dot(vector, previous)
                for row in range(rows):
                    vector[row] -= projection * previous[row]
        norm_squared = _dot(vector, vector)
        threshold = (
            _F64_EPSILON * max(1.0, original_norm) * max(1, rows)
        ) ** 2
        if not math.isfinite(norm_squared) or norm_squared <= threshold:
            # Pick the standard-basis residual with greatest norm. Strict `>`
            # means an exact tie keeps the lowest coordinate.
            best_vector = np.zeros(rows, dtype=np.float64)
            best_norm = -1.0
            for coordinate in range(rows):
                candidate = np.zeros(rows, dtype=np.float64)
                candidate[coordinate] = 1.0
                for _ in range(2):
                    for previous in basis:
                        projection = _dot(candidate, previous)
                        for row in range(rows):
                            candidate[row] -= projection * previous[row]
                candidate_norm = _dot(candidate, candidate)
                if candidate_norm > best_norm:
                    best_vector = candidate
                    best_norm = candidate_norm
            vector = best_vector
            norm_squared = best_norm
        denominator = math.sqrt(max(norm_squared, _F64_TINY))
        basis.append(vector / denominator)
    return basis


def _symmetric_gram(matrix: np.ndarray) -> np.ndarray:
    columns = matrix.shape[1]
    gram = np.empty((columns, columns), dtype=np.float64)
    for row in range(columns):
        for column in range(row + 1):
            total = 0.0
            for sample in range(len(matrix)):
                total += float(matrix[sample, row]) * float(matrix[sample, column])
            gram[row, column] = total
            gram[column, row] = total
    return gram


def _regularized_cholesky(matrix: np.ndarray, regularizer: float) -> np.ndarray:
    """Fixed-order lower Cholesky with a deterministic positive pivot floor."""

    size = matrix.shape[0]
    lower = np.zeros((size, size), dtype=np.float64)
    diagonal_scale = 1.0
    for index in range(size):
        diagonal_scale = max(diagonal_scale, abs(float(matrix[index, index])))
    pivot_floor = max(_F64_TINY, _F64_EPSILON * diagonal_scale)
    for row in range(size):
        for column in range(row + 1):
            value = float(matrix[row, column])
            if row == column:
                value += regularizer
            for inner in range(column):
                value -= lower[row, inner] * lower[column, inner]
            if row == column:
                if not math.isfinite(value) or value < pivot_floor:
                    value = pivot_floor
                lower[row, column] = math.sqrt(value)
            else:
                lower[row, column] = value / lower[column, column]
    return lower


def _forward_solve(lower: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.empty(len(right), dtype=np.float64)
    for row in range(len(right)):
        value = float(right[row])
        for column in range(row):
            value -= lower[row, column] * out[column]
        out[row] = value / lower[row, row]
    return out


def _inverse_lower(lower: np.ndarray) -> np.ndarray:
    size = len(lower)
    inverse = np.empty((size, size), dtype=np.float64)
    for column in range(size):
        right = np.zeros(size, dtype=np.float64)
        right[column] = 1.0
        inverse[:, column] = _forward_solve(lower, right)
    return inverse


def _rff(x: np.ndarray, n: int, gamma: float, seed: int, *, orthogonal: bool) -> np.ndarray:
    n = max(0, int(n))
    if n == 0:
        return np.empty((len(x), 0), dtype=np.float64)
    d = x.shape[1]
    rng = _Pcg32.for_stage(seed, d, n)
    gamma = gamma if math.isfinite(gamma) and gamma >= 0.0 else 0.0
    if orthogonal:
        rows: list[np.ndarray] = []
        if d:
            # Draw order: projection column, then input coordinate. Each block
            # has min(d, remaining) columns, so no discarded tail is drawn.
            # MGS visits columns left-to-right and consumes no random values.
            while len(rows) < n:
                block_columns = min(d, n - len(rows))
                block = np.empty((d, block_columns), dtype=np.float64)
                for column in range(block_columns):
                    for row in range(d):
                        block[row, column] = rng.gaussian()
                rows.extend(_modified_gram_schmidt(block))
            w = np.asarray(rows[:n]) * math.sqrt(2.0 * gamma * d)
        else:
            w = np.empty((n, 0), dtype=np.float64)
    else:
        w = np.empty((n, d), dtype=np.float64)
        for row in range(n):
            for column in range(d):
                w[row, column] = rng.gaussian() * math.sqrt(2.0 * gamma)
    # Phases are the final n draws after every weight block.
    phase = np.asarray(
        [2.0 * math.pi * rng.next_f64() for _ in range(n)],
        dtype=np.float64,
    )
    projection = _fixed_matmul(x, w.T)
    return math.sqrt(2.0 / n) * np.cos(projection + phase)


def _halton(count: int, dimension: int, seed: int) -> np.ndarray:
    # First primes are fixed by the transform version. A digital shift makes
    # seed changes reproducible without relying on Python's randomized hash.
    bases: list[int] = []
    candidate = 2
    while len(bases) < dimension:
        if all(candidate % p for p in range(2, int(math.sqrt(candidate)) + 1)):
            bases.append(candidate)
        candidate += 1
    out = np.empty((count, dimension), dtype=np.float64)
    for j, base in enumerate(bases):
        for i in range(count):
            value = i + 1
            factor = 1.0 / base
            acc = 0.0
            while value:
                value, digit = divmod(value, base)
                acc += digit * factor
                factor /= base
            shift = ((int(seed) * 0x9E3779B97F4A7C15 + j * 0xD1B54A32D192ED03) & ((1 << 64) - 1))
            shift = ((shift >> 11) + 0.5) / float(1 << 53)
            out[i, j] = (acc + shift) % 1.0
    return out


def _normal_ppf(u: np.ndarray) -> np.ndarray:
    # Acklam's rational inverse-normal approximation, versioned as part of the
    # Prime semantics. Clamping avoids infinities at the exact endpoints.
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    lo = u < 0.02425
    hi = u > 1.0 - 0.02425
    mid = ~(lo | hi)
    out = np.empty_like(u)
    def tail(q: np.ndarray) -> np.ndarray:
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return numerator / denominator

    q = np.sqrt(-2.0 * np.log(u[lo]))
    out[lo] = tail(q)
    q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
    out[hi] = -tail(q)
    q = u[mid] - 0.5
    r = q * q
    numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    denominator = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    out[mid] = numerator / denominator
    return out


def _rff_halton(x: np.ndarray, n: int, gamma: float, seed: int) -> np.ndarray:
    u = _halton(n, x.shape[1], seed)
    w = _normal_ppf(u) * math.sqrt(2.0 * gamma)
    phase = 2.0 * math.pi * _halton(n, 1, seed + 0xA5A5A5A5).ravel()
    projection = _fixed_matmul(x, w.T)
    return math.sqrt(2.0 / n) * np.cos(projection + phase)


def _maximin_pilot_indices(x: np.ndarray, count: int) -> np.ndarray:
    count = max(0, min(int(count), len(x)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if x.shape[1] == 0:
        return np.arange(count, dtype=np.int64)

    scale = float(np.max(np.abs(x), initial=0.0))
    scaled = x / scale if scale > 0.0 else np.zeros_like(x)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum(scaled * scaled, axis=1)))
    distances = np.sum((scaled - scaled[selected[0]]) ** 2, axis=1)
    used = {int(selected[0])}
    for i in range(1, count):
        order = np.argsort(-distances, kind="stable")
        choice = next(int(idx) for idx in order if int(idx) not in used)
        selected[i] = choice
        used.add(choice)
        distances = np.minimum(
            distances,
            np.sum((scaled - scaled[choice]) ** 2, axis=1),
        )
    return selected


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    if left.shape[1] != right.shape[1]:
        raise ValueError("RBF inputs must have the same number of columns")
    gamma = gamma if math.isfinite(gamma) and gamma >= 0.0 else 0.0
    if gamma == 0.0 or left.shape[1] == 0:
        return np.ones((len(left), len(right)), dtype=np.float64)

    scale = max(
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
    )
    if scale == 0.0:
        return np.ones((len(left), len(right)), dtype=np.float64)
    out = np.empty((len(left), len(right)), dtype=np.float64)
    multiplier = gamma * scale * scale
    for row in range(len(left)):
        for column in range(len(right)):
            distance = 0.0
            for feature in range(left.shape[1]):
                difference = (
                    float(left[row, feature]) / scale
                    - float(right[column, feature]) / scale
                )
                distance += difference * difference
            exponent = multiplier * distance
            if math.isnan(exponent) or exponent < 0.0:
                exponent = 0.0
            out[row, column] = math.exp(-exponent)
    return out


def _ridge_leverage_landmarks(
    x: np.ndarray,
    k: int,
    gamma: float,
    ridge: float,
) -> np.ndarray:
    """Select stable top approximate ridge-leverage rows.

    A deterministic maximin pilot of at most ``2 * k`` rows defines an RBF
    feature matrix. Its regularized row leverage scores approximate those of
    the full kernel without allocating the full sample-by-sample matrix.
    """

    k = max(0, min(int(k), len(x)))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    pilot_count = min(len(x), max(k, min(2 * k, 256)))
    pilot = x[_maximin_pilot_indices(x, pilot_count)]
    features = _rbf_kernel(x, pilot, gamma)
    gram = _symmetric_gram(features)
    regularizer = (
        ridge
        if math.isfinite(ridge) and ridge > 0.0
        else np.finfo(np.float64).eps
    )
    lower = _regularized_cholesky(gram, regularizer)
    scores = np.empty(len(features), dtype=np.float64)
    for row in range(len(features)):
        whitened = _forward_solve(lower, features[row])
        scores[row] = _dot(whitened, whitened)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    return np.argsort(-scores, kind="stable")[:k].astype(np.int64, copy=False)


@dataclass
class NystromState:
    landmarks: np.ndarray
    projector: np.ndarray


def _nystrom_fit(x: np.ndarray, k: int, gamma: float, ridge: float) -> NystromState:
    indices = _ridge_leverage_landmarks(x, k, gamma, ridge)
    landmarks = x[indices]
    if len(landmarks) == 0:
        return NystromState(
            landmarks=landmarks,
            projector=np.empty((0, 0), dtype=np.float64),
        )
    regularizer = (
        ridge
        if math.isfinite(ridge) and ridge > 0.0
        else np.finfo(np.float64).eps
    )
    kernel = _rbf_kernel(landmarks, landmarks, gamma)
    lower = _regularized_cholesky(kernel, regularizer)
    # K + ridge*I = L*L^T, so L^-T is a deterministic whitening factor.
    projector = _inverse_lower(lower).T
    return NystromState(landmarks=landmarks, projector=projector)


def _nystrom_apply(x: np.ndarray, state: NystromState, gamma: float) -> np.ndarray:
    return _fixed_matmul(
        _rbf_kernel(x, state.landmarks, gamma),
        state.projector,
    )


def _fwht(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    width = out.shape[-1]
    step = 1
    while step < width:
        for start in range(0, width, step * 2):
            left = out[..., start : start + step].copy()
            right = out[..., start + step : start + 2 * step].copy()
            out[..., start : start + step] = left + right
            out[..., start + step : start + 2 * step] = left - right
        step *= 2
    return out / math.sqrt(width)


def _partial_fisher_yates(width: int, rng: _Pcg32) -> np.ndarray:
    permutation = np.arange(width, dtype=np.int64)
    # Exactly `width` draws, including the final modulo-one swap. This is the
    # battle-runtime partial Fisher-Yates convention with k == width.
    for index in range(width):
        other = index + rng.next_u32() % (width - index)
        permutation[index], permutation[other] = (
            permutation[other],
            permutation[index],
        )
    return permutation


def _fastfood(x: np.ndarray, n: int, gamma: float, seed: int) -> np.ndarray:
    n = max(0, int(n))
    if n == 0:
        return np.empty((len(x), 0), dtype=np.float64)
    d = x.shape[1]
    width = 1 << (max(1, d) - 1).bit_length()
    rng = _Pcg32.for_stage(seed, d, n)
    padded = np.pad(x, ((0, 0), (0, width - d)))
    outputs = np.empty((len(x), n), dtype=np.float64)
    gamma = gamma if math.isfinite(gamma) and gamma >= 0.0 else 0.0
    for start in range(0, n, width):
        stop = min(n, start + width)
        # Fixed B/G/P/S/phase draw order. Every block is complete even when
        # only a prefix is used by the final output block.
        b = np.asarray(
            [1.0 if rng.next_u32() & 0x80000000 else -1.0 for _ in range(width)],
            dtype=np.float64,
        )
        g = np.asarray([rng.gaussian() for _ in range(width)], dtype=np.float64)
        perm = _partial_fisher_yates(width, rng)
        radial = np.empty(width, dtype=np.float64)
        for coordinate in range(width):
            chi_squared = 0.0
            for _ in range(width):
                draw = rng.gaussian()
                chi_squared += draw * draw
            radial[coordinate] = math.sqrt(chi_squared)
        phases = np.asarray(
            [2.0 * math.pi * rng.next_f64() for _ in range(width)],
            dtype=np.float64,
        )

        first = _fwht(padded * b)
        mixed = _fwht(first[:, perm] * g)
        g_norm = max(math.sqrt(_dot(g, g)), _F64_TINY)
        s = radial * math.sqrt(width) / g_norm
        projected = mixed * s * math.sqrt(2.0 * gamma)
        block = np.cos(projected + phases)
        outputs[:, start:stop] = block[:, : stop - start]
    return math.sqrt(2.0 / n) * outputs


def _hash_pair(index: int, seed: int) -> tuple[int, float]:
    value = (index + 1) * 0x9E3779B97F4A7C15 + seed * 0xD1B54A32D192ED03
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return int(value & 0x7FFFFFFFFFFFFFFF), (1.0 if value & 1 else -1.0)


def _count_sketch(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    out = np.zeros((len(x), n), dtype=np.float64)
    for column in range(x.shape[1]):
        bucket, sign = _hash_pair(column, seed)
        out[:, bucket % n] += sign * x[:, column]
    return out


# Pinned forward stage roots for lengths up to the global feature-width cap.
# Portable twins must use these exact f64 values and the loop order below,
# rather than recomputing roots with a platform-specific trigonometric library.
_RADIX2_FORWARD_ROOTS = {
    2: (-1.0, -0.0),
    4: (0.0, -1.0),
    8: (
        float.fromhex("0x1.6a09e667f3bcdp-1"),
        -float.fromhex("0x1.6a09e667f3bcdp-1"),
    ),
    16: (
        float.fromhex("0x1.d906bcf328d46p-1"),
        -float.fromhex("0x1.87de2a6aea963p-2"),
    ),
    32: (
        float.fromhex("0x1.f6297cff75cb0p-1"),
        -float.fromhex("0x1.8f8b83c69a60ap-3"),
    ),
    64: (
        float.fromhex("0x1.fd88da3d12526p-1"),
        -float.fromhex("0x1.917a6bc29b42cp-4"),
    ),
    128: (
        float.fromhex("0x1.ff621e3796d7ep-1"),
        -float.fromhex("0x1.91f65f10dd814p-5"),
    ),
    256: (
        float.fromhex("0x1.ffd886084cd0dp-1"),
        -float.fromhex("0x1.92155f7a3667ep-6"),
    ),
    512: (
        float.fromhex("0x1.fff62169b92dbp-1"),
        -float.fromhex("0x1.921d1fcdec784p-7"),
    ),
}


def _validate_radix2_width(width: int) -> None:
    if width <= 0 or width > MAX_WIDTH or width & (width - 1):
        raise ValueError(
            f"TensorSketch width must be a positive power of two no greater than {MAX_WIDTH}"
        )


def _validate_tensor_sketch_parameters(n: int, degree: int) -> None:
    _validate_radix2_width(n)
    if n not in _TENSOR_SKETCH_WIDTHS:
        raise ValueError(
            f"TensorSketch width must be one of the registry widths {_TENSOR_SKETCH_WIDTHS}"
        )
    if degree not in _TENSOR_SKETCH_DEGREES:
        raise ValueError(
            f"TensorSketch degree must be one of the registry values {_TENSOR_SKETCH_DEGREES}"
        )


def _fft_radix2_v1(
    real: Any,
    imaginary: Any | None = None,
    *,
    inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-order radix-2 complex FFT used by portable TensorSketch twins."""

    real_out = np.asarray(real, dtype=np.float64).copy()
    if real_out.ndim == 0:
        raise ValueError("radix-2 FFT input must have at least one dimension")
    width = real_out.shape[-1]
    _validate_radix2_width(width)
    if imaginary is None:
        imaginary_out = np.zeros_like(real_out)
    else:
        imaginary_out = np.asarray(imaginary, dtype=np.float64).copy()
        if imaginary_out.shape != real_out.shape:
            raise ValueError("radix-2 FFT real and imaginary shapes must match")

    rows = real_out.size // width
    real_rows = real_out.reshape(rows, width)
    imaginary_rows = imaginary_out.reshape(rows, width)

    # Bit-reversal permutation. Strict left-to-right swaps fix the mutation
    # order for a scalar Rust or standard-library Python implementation.
    reversed_index = 0
    for index in range(1, width):
        bit = width >> 1
        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1
        reversed_index ^= bit
        if index < reversed_index:
            real_left = real_rows[:, index].copy()
            real_rows[:, index] = real_rows[:, reversed_index]
            real_rows[:, reversed_index] = real_left
            imaginary_left = imaginary_rows[:, index].copy()
            imaginary_rows[:, index] = imaginary_rows[:, reversed_index]
            imaginary_rows[:, reversed_index] = imaginary_left

    length = 2
    while length <= width:
        root_real, root_imaginary = _RADIX2_FORWARD_ROOTS[length]
        if inverse:
            root_imaginary = -root_imaginary
        half = length // 2
        for start in range(0, width, length):
            twiddle_real = 1.0
            twiddle_imaginary = 0.0
            for offset in range(half):
                even = start + offset
                odd = even + half
                # Rows are independent lanes. These elementwise expressions
                # keep the scalar multiply/add order while avoiding a Python
                # loop; portable scalar twins evaluate the same operations.
                odd_real = real_rows[:, odd].copy()
                odd_imaginary = imaginary_rows[:, odd].copy()
                product_real = (
                    twiddle_real * odd_real
                    - twiddle_imaginary * odd_imaginary
                )
                product_imaginary = (
                    twiddle_real * odd_imaginary
                    + twiddle_imaginary * odd_real
                )
                even_real = real_rows[:, even].copy()
                even_imaginary = imaginary_rows[:, even].copy()
                real_rows[:, even] = even_real + product_real
                imaginary_rows[:, even] = even_imaginary + product_imaginary
                real_rows[:, odd] = even_real - product_real
                imaginary_rows[:, odd] = even_imaginary - product_imaginary
                next_twiddle_real = (
                    twiddle_real * root_real
                    - twiddle_imaginary * root_imaginary
                )
                next_twiddle_imaginary = (
                    twiddle_real * root_imaginary
                    + twiddle_imaginary * root_real
                )
                twiddle_real = next_twiddle_real
                twiddle_imaginary = next_twiddle_imaginary
        length *= 2

    if inverse:
        real_rows /= width
        imaginary_rows /= width
    return real_out, imaginary_out


def _tensor_sketch(x: np.ndarray, n: int, degree: int, seed: int) -> np.ndarray:
    _validate_tensor_sketch_parameters(n, degree)
    spectra_real = np.ones((len(x), n), dtype=np.float64)
    spectra_imaginary = np.zeros((len(x), n), dtype=np.float64)
    for table in range(degree):
        table_real, table_imaginary = _fft_radix2_v1(
            _count_sketch(x, n, seed + table * 0x9E37)
        )
        product_real = (
            spectra_real * table_real - spectra_imaginary * table_imaginary
        )
        product_imaginary = (
            spectra_real * table_imaginary + spectra_imaginary * table_real
        )
        spectra_real = product_real
        spectra_imaginary = product_imaginary
    transformed, _ = _fft_radix2_v1(
        spectra_real,
        spectra_imaginary,
        inverse=True,
    )
    return transformed / math.sqrt(n)


def fit_transform(
    name: str,
    x: Any,
    params: dict[str, Any] | None = None,
    *,
    y: Any | None = None,
) -> tuple[np.ndarray, Any | None]:
    params = dict(params or {})
    matrix = _matrix(x)
    append = bool(params.pop("append", False))
    n = int(params.get("n", 32))
    gamma = float(params.get("gamma", 1.0))
    seed = int(params.get("seed", 7))
    if name == "rff_orthogonal":
        z, state = _rff(matrix, n, gamma, seed, orthogonal=True), None
    elif name == "rff_halton":
        z, state = _rff_halton(matrix, n, gamma, seed), None
    elif name == "nystrom_leverage":
        state = _nystrom_fit(matrix, int(params.get("k", 32)), gamma, float(params.get("ridge", 1e-6)))
        z = _nystrom_apply(matrix, state, gamma)
    elif name == "fastfood_rbf":
        z, state = _fastfood(matrix, n, gamma, seed), None
    elif name == "count_sketch":
        z, state = _count_sketch(matrix, n, seed), None
    elif name == "tensor_sketch":
        z, state = _tensor_sketch(matrix, n, int(params.get("degree", 2)), seed), None
    else:
        raise ValueError(f"unknown Prime transform {name!r}")
    return _finish(matrix, _finite(z), append), state


def apply_transform(name: str, x: Any, params: dict[str, Any] | None = None, *, state: Any | None = None) -> np.ndarray:
    if name == "nystrom_leverage" and state is not None:
        params = dict(params or {})
        matrix = _matrix(x)
        append = bool(params.pop("append", False))
        z = _nystrom_apply(matrix, state, float(params.get("gamma", 1.0)))
        return _finish(matrix, _finite(z), append)
    return fit_transform(name, x, params)[0]
