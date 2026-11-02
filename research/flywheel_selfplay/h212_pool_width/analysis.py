"""Data-free H212 family-level estimands and decision gates."""

from __future__ import annotations

import hashlib

import numpy as np

SEEDS = (212001, 212002, 212003)
WIDTHS = (16, 32, 64)
PRIMARY_WIDTH = 64
FOLD_SEED = 212140
BOOT_SEED = 212150
ELO_BOOT_SEED = 212151
BOOT_DRAWS = 10_000
ELO_DRAWS = 1_000
HARM_MAX = 0.20
TIE_EPS = 0.0
ANCHOR_ELO = 1500.0


def stream_seed(seed: int, context_sha256: str) -> int:
    if seed not in SEEDS or len(context_sha256) != 64:
        raise ValueError("registered H212 seed and SHA256 context required")
    payload = f"H212:propose:{seed}:{context_sha256}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def family_folds(family_ids: list[str], seed: int = FOLD_SEED) -> list[list[str]]:
    families = sorted(set(family_ids))
    if seed != FOLD_SEED or len(families) != 205:
        raise ValueError("H212 requires its registered 205-family fold seed and roster")
    return [
        sorted(group.tolist())
        for group in np.array_split(np.random.default_rng(seed).permutation(families), 5)
    ]


def first_argmin_utility(costs, fitness, gate) -> tuple[float, int]:
    c = np.asarray(costs, dtype=np.float64)
    f = np.asarray(fitness, dtype=np.float64)
    g = np.asarray(gate, dtype=bool)
    if c.ndim != 1 or c.size == 0 or c.shape != f.shape or c.shape != g.shape:
        raise ValueError("nonempty matching candidate vectors required")
    if not np.isfinite(c).all() or not np.isfinite(f).all():
        raise ValueError("candidate costs and fitness must be finite")
    idx = int(np.argmin(c))
    return (float(-f[idx] if g[idx] else -2.0), idx)


def family_means(values, families: list[str]) -> tuple[list[str], np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    fam = np.asarray(families, dtype=str)
    if v.ndim != 1 or v.shape != fam.shape or not np.isfinite(v).all():
        raise ValueError("finite per-context values and matching family IDs required")
    order = sorted(set(families))
    if len(order) != 205:
        raise ValueError("H212 requires all 205 families")
    means = np.array([v[fam == family].mean() for family in order], dtype=np.float64)
    return order, means


def family_gain_stats(gain, bootstrap_indices) -> dict:
    g = np.asarray(gain, dtype=np.float64)
    ix = np.asarray(bootstrap_indices, dtype=np.int64)
    if g.shape != (205,) or not np.isfinite(g).all():
        raise ValueError("205 finite family gains required")
    if ix.ndim != 2 or ix.shape[1] != 205 or (ix < 0).any() or (ix >= 205).any():
        raise ValueError("bootstrap indices must sample 205 registered families")
    draws = np.median(g[ix], axis=1)
    return {
        "family_median_gain": float(np.median(g)),
        "family_mean_gain": float(np.mean(g)),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "p_one_sided_median_le_zero": float((1 + np.sum(draws <= 0.0)) / (len(draws) + 1)),
        "harm_count": int(np.sum(g < 0.0)),
        "harm_fraction": float(np.mean(g < 0.0)),
        "win_tie_loss": [int(np.sum(g > 0.0)), int(np.sum(g == 0.0)), int(np.sum(g < 0.0))],
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, returned under the original contrast keys."""
    if not p_values or any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in p_values.values()):
        raise ValueError("nonempty finite p-values in [0, 1] required")
    order = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, running, count = {}, 0.0, len(order)
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def pooled_seed_gain_stats(seed_family_gain, bootstrap_indices) -> dict:
    """Median of three equal-family seed means with a paired family bootstrap."""
    values = np.asarray(seed_family_gain, dtype=np.float64)
    ix = np.asarray(bootstrap_indices, dtype=np.int64)
    if values.shape != (len(SEEDS), 205) or not np.isfinite(values).all():
        raise ValueError("finite three-seed by 205-family gain matrix required")
    if ix.ndim != 2 or ix.shape[1] != 205 or (ix < 0).any() or (ix >= 205).any():
        raise ValueError("bootstrap indices must sample 205 registered families")
    seed_draw_means = values[:, ix].mean(axis=2)
    draws = np.median(seed_draw_means, axis=0)
    pooled_family_gain = values.mean(axis=0)
    return {
        "seed_family_mean_gain": [float(x) for x in values.mean(axis=1)],
        "median_over_three_seed_family_means": float(np.median(values.mean(axis=1))),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "p_one_sided_median_le_zero": float((1 + np.sum(draws <= 0.0)) / (len(draws) + 1)),
        "pooled_family_mean_gain": float(np.mean(pooled_family_gain)),
        "harm_fraction": float(np.mean(pooled_family_gain < 0.0)),
        "win_tie_loss": [
            int(np.sum(pooled_family_gain > 0.0)),
            int(np.sum(pooled_family_gain == 0.0)),
            int(np.sum(pooled_family_gain < 0.0)),
        ],
    }


def decide(primary_by_seed: dict[int, dict], elo_lower_by_seed: dict[int, float]) -> dict:
    if set(primary_by_seed) != set(SEEDS) or set(elo_lower_by_seed) != set(SEEDS):
        return {"verdict": "INCOMPLETE", "failed_gates": ["all_three_seeds_required"]}
    failed = []
    for seed in SEEDS:
        stats = primary_by_seed[seed]
        if stats["ci95"][0] <= 0.0:
            failed.append(f"seed{seed}_ci_lower_gt_zero")
        if stats["harm_fraction"] > HARM_MAX:
            failed.append(f"seed{seed}_harm_le_20pct")
        if elo_lower_by_seed[seed] <= ANCHOR_ELO:
            failed.append(f"seed{seed}_elo_lcb_gt_1500")
    return {"verdict": "ADVANCE" if not failed else "NO_ADVANCE", "failed_gates": failed}


def validate_width_proposals(proposals: list[dict], width: int) -> list[dict]:
    """Validate one exact width-specific output from the pinned proposer."""
    if width not in WIDTHS or len(proposals) != width:
        raise ValueError(f"exactly {width} registered proposals required")
    ids = [p.get("program_sha256") for p in proposals]
    if any(not isinstance(x, str) or len(x) != 64 for x in ids) or len(set(ids)) != width:
        raise ValueError(f"{width} unique proposal SHA256 identities required")
    return proposals
