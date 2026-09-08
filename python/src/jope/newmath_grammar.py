"""Sampler for newmath (jmath) genomes over the empirically-proven op subset.

Scope: the transforms that newmath's GP actually used in real evaluated pipelines plus
the runnable regression heads (ridge / gbm / ensemble_ridge_gbm). Excludes the quarantined
"zyal"/reconstructed kernel ops (seed-only, promotable:false) and the near-no-ops
`residual_boost` (identity) and `concat` (alias placeholder).

Genomes are `(pipe X <transform>* <head>)`. The sampler over-generates; callers validate
each candidate through `jope-newmath-validate` and keep only the valid ones, so type-
incompatible sequences are filtered out and output is 100% valid by construction.

Param grids follow newmath's GP jitter grids (gp.rs). The `map` op is restricted to the two
named-macro exprs the compiler recognizes verbatim (`sigmoid_col0`, `log1p_col0`).
"""

from __future__ import annotations

import random
from typing import Any

# transform op -> {param: [choices]}. Empty dict = no params.
TRANSFORMS: dict[str, dict[str, list[Any]]] = {
    "robust_scale": {},
    "rank": {},
    "norm_features": {},
    "sigmoid": {},
    "polar": {},
    "radial": {},
    "poincare": {},
    "entropy": {},
    "spectral_entropy": {},
    "fft_mag": {"axis": ["features"]},
    "rfft_mag": {"axis": ["features"]},
    "rfft_phase": {"axis": ["features"]},
    "spectral_moment": {"order": [1, 2, 3]},
    "peak_freq": {},
    "rff_cos": {"n": [16, 32, 64, 128, 256], "seed": [1, 7, 9, 13, 29, 347]},
    "rff_sin": {"n": [16, 32, 64, 128], "seed": [3, 5, 13, 29]},
    "cheb": {"degree": [2, 3, 4, 6]},
    "poly_interact": {"degree": [2, 3]},
    "proj": {"k": [8, 16, 32, 64], "seed": [1, 3, 7, 17]},
    "proj_eigen": {"iters": [8, 16, 24]},
    "svd_project": {"k": [4, 8, 16, 24, 32], "iters": [8, 16, 24]},
    "nystrom": {"k": [4, 8, 16, 32], "gamma": [0.05, 0.1, 0.2, 0.5]},
    "quantile_bins": {"bins": [3, 4, 6, 8, 12]},
    "wavelet": {"family": ["dyadic", "morlet", "mexican"], "scale": [1, 2, 3, 4]},
    "iterate": {"steps": [2, 4, 6, 8, 10], "dt": [0.05, 0.1, 0.2], "seed": [1, 7]},
    "reduce": {"op": ["mean", "var", "max", "sum", "l2", "energy"]},
    "roll": {"shift": [1, 2, 3]},
    "lag": {"k": [1, 2, 3]},
    "quasicrystal": {"phi": [1.6180339887]},
    "golden_scale": {"phi": [1.6180339887]},
    "logistic_map": {"r": [3.6, 3.8, 3.9]},
    "eigen_ground_state": {"iters": [8, 16]},
    "persistence1d": {"topk": [2, 4, 8]},
    "local_density": {"k": [4, 8, 16]},
    "gate_mix": {},
    "cross_ratio": {},
    "map": {"expr": ["sigmoid_col0", "log1p_col0"]},
}

# head op -> {param: [choices]}. All are runnable regression heads.
HEADS: dict[str, dict[str, list[Any]]] = {
    "ridge": {"alpha": [0.1, 0.5, 1.0, 3.0]},
    "gbm": {"rounds": [80, 150, 240], "depth": [2, 3, 4], "lr": [0.05, 0.1]},
    "ensemble_ridge_gbm": {},
}

TRANSFORM_LIST = list(TRANSFORMS)
HEAD_LIST = list(HEADS)
KNOWN_VALID_FALLBACK = "(pipe X (robust_scale) (ridge alpha=1.0))"
MAX_TRANSFORMS = 4


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # match newmath's f64 formatting (no trailing zeros beyond needed)
        return repr(value)
    return str(value)


def _format_stage(op: str, params: dict[str, Any]) -> str:
    parts = [op]
    for key in sorted(params):
        parts.append(f"{key}={_format_value(params[key])}")
    return "(" + " ".join(parts) + ")"


def _sample_params(grid: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(choices) for key, choices in grid.items() if choices}


def sample_genome(rng: random.Random, *, max_transforms: int = MAX_TRANSFORMS) -> str:
    """Sample one candidate newmath genome string (not guaranteed valid; validate downstream)."""
    n = rng.randint(1, max(1, max_transforms))
    ops = [rng.choice(TRANSFORM_LIST) for _ in range(n)]
    stages = [_format_stage(op, _sample_params(TRANSFORMS[op], rng)) for op in ops]
    head = rng.choice(HEAD_LIST)
    stages.append(_format_stage(head, _sample_params(HEADS[head], rng)))
    return "(pipe X " + " ".join(stages) + ")"


def sample_genomes(n: int, *, seed: int = 0, max_transforms: int = MAX_TRANSFORMS) -> list[str]:
    """Sample n candidate genome strings, deduped by string."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        g = sample_genome(rng, max_transforms=max_transforms)
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out
