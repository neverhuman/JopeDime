"""Fitness-v2 episode engine: never-repeating tabular episodes with a live
opponent zoo.

An episode = (task, row-subset seed, feature-subset/permutation seed,
context/query re-split, monotone y-warp) drawn fresh per evaluation. The
zoo opponents (ridge/knn/extratrees/histgbm — the generator's exact
hyperparameters, the same four estimator classes that exist in the real
archive reports) are re-fit ON the episode's own context and scored on its
queries, so opponent utilities can never be memorized either.

Task sources: the 25,205 synth_* dirs (never trained on — the backtest bed)
and, later, real/prior tiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
SYNTH_RE = re.compile(r"^synth_[0-9a-f]{16}$")

CTX_CHOICES = (128, 256, 384, 512, 700)
QRY_CHOICES = (64, 128, 256)
FEATURE_CAP = 192
WARPS = ("identity", "log1p_signed", "power", "sigmoid_bounded")

# Difficulty bands from gen_synth_regression.py `_observed_difficulty`
DIFF_EDGES = (0.12, 0.35, 0.72)
DIFF_NAMES = ("noisy", "hard", "medium", "easy")


def difficulty_bucket(zoo_best_r2: float) -> str:
    if zoo_best_r2 >= DIFF_EDGES[2]:
        return "easy"
    if zoo_best_r2 >= DIFF_EDGES[1]:
        return "medium"
    if zoo_best_r2 >= DIFF_EDGES[0]:
        return "hard"
    return "noisy"


def list_synth_tasks(limit: int | None = None, seed: int = 0) -> list[str]:
    names = sorted(d.name for d in ARCHIVE_ROOT.iterdir() if SYNTH_RE.match(d.name))
    if limit is not None and limit < len(names):
        rng = np.random.default_rng(seed)
        names = [names[i] for i in rng.choice(len(names), size=limit, replace=False)]
        names.sort()
    return names


def load_task(name: str) -> tuple[np.ndarray, np.ndarray]:
    """All rows of a task pooled (its own train/test split is irrelevant to
    episodes — we re-split freshly)."""
    root = ARCHIVE_ROOT / name
    frames = [
        pd.read_csv(root / f, header=None, dtype=np.float32).to_numpy() for f in ("train.csv", "test.csv")
    ]
    data = np.concatenate(frames, axis=0)
    return np.ascontiguousarray(data[:, :-1]), np.ascontiguousarray(data[:, -1])


def apply_warp(y: np.ndarray, warp: str, param: float) -> np.ndarray:
    if warp == "identity":
        return y
    if warp == "log1p_signed":
        return np.sign(y) * np.log1p(np.abs(y) * (1.0 + param))
    if warp == "power":
        lam = 0.5 + 1.5 * param  # in [0.5, 2.0]
        return np.sign(y) * np.abs(y) ** lam
    if warp == "sigmoid_bounded":
        scale = 1.0 + 3.0 * param
        return 1.0 / (1.0 + np.exp(-scale * (y - np.median(y))))
    raise ValueError(warp)


@dataclass
class Episode:
    task: str
    seed: int
    ctx_x: np.ndarray  # (C, F) f32, may contain NaN (model handles natively)
    ctx_y: np.ndarray  # (C,) f32, warped
    qry_x: np.ndarray  # (Q, F)
    qry_y: np.ndarray  # (Q,) f32, warped
    warp: str
    bucket: str = "unknown"
    zoo_rmse: dict[str, float] = field(default_factory=dict)


def _task_int(task: str) -> int:
    """Stable across processes (Python's hash() is salted)."""
    hex_part = task.split("_")[-1]
    try:
        return int(hex_part, 16) & (2**63 - 1)
    except ValueError:
        import hashlib

        return int(hashlib.sha256(task.encode()).hexdigest()[:15], 16)


def draw_episode(task: str, seed: int, X: np.ndarray, y: np.ndarray) -> Episode:
    """Deterministic in (task, seed); a fresh seed gives a fresh episode."""
    rng = np.random.default_rng([_task_int(task), seed])
    n, F = X.shape
    ctx_n = int(rng.choice([c for c in CTX_CHOICES if c <= max(64, n - 32)] or [max(16, n // 2)]))
    qry_n = int(rng.choice([q for q in QRY_CHOICES if q <= n - ctx_n] or [max(8, n - ctx_n)]))
    rows = rng.permutation(n)
    ctx_rows, qry_rows = rows[:ctx_n], rows[ctx_n : ctx_n + qry_n]

    n_feat = min(F, FEATURE_CAP)
    cols = rng.permutation(F)[:n_feat]

    warp = str(rng.choice(WARPS))
    param = float(rng.random())
    yw = apply_warp(y.astype(np.float64), warp, param).astype(np.float32)
    if not np.isfinite(yw).all() or float(np.nanstd(yw)) < 1e-9:
        warp, yw = "identity", y  # degenerate warp on this task: fall back

    return Episode(
        task=task,
        seed=seed,
        ctx_x=np.ascontiguousarray(X[np.ix_(ctx_rows, cols)]),
        ctx_y=np.ascontiguousarray(yw[ctx_rows]),
        qry_x=np.ascontiguousarray(X[np.ix_(qry_rows, cols)]),
        qry_y=np.ascontiguousarray(yw[qry_rows]),
        warp=warp,
    )


# ---- live zoo (gen_synth_regression.py zoo_best_r2 hyperparameters) --------


def _impute(ctx_x: np.ndarray, qry_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sklearn zoo members (except histgbm) cannot take NaN; ctx-mean impute."""
    if not (np.isnan(ctx_x).any() or np.isnan(qry_x).any()):
        return ctx_x, qry_x
    col_mean = np.nanmean(ctx_x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    fix = lambda a: np.where(np.isnan(a), col_mean, a)
    return fix(ctx_x), fix(qry_x)


def score_zoo(ep: Episode) -> dict[str, float]:
    """Fit the four opponents on the episode context, score rmse on queries.
    Checkpoint-independent: compute once per episode, reuse for every model."""
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cx, qx = _impute(ep.ctx_x.astype(np.float64), ep.qry_x.astype(np.float64))
    cy, qy = ep.ctx_y.astype(np.float64), ep.qry_y.astype(np.float64)
    n = len(cy)
    zoo = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=min(15, max(2, n // 25)), weights="distance"),
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=40, max_depth=12, min_samples_leaf=2, random_state=1, n_jobs=1
        ),
        "histgbm": HistGradientBoostingRegressor(
            max_iter=70,
            max_depth=8,
            learning_rate=0.08,
            l2_regularization=0.01,
            early_stopping=False,
            random_state=2,
        ),
    }
    out = {}
    for name, model in zoo.items():
        try:
            if name == "histgbm":  # native NaN support: use raw features
                model.fit(ep.ctx_x.astype(np.float64), cy)
                pred = model.predict(ep.qry_x.astype(np.float64))
            else:
                model.fit(cx, cy)
                pred = model.predict(qx)
            out[name] = float(np.sqrt(np.mean((pred - qy) ** 2)))
        except Exception:
            out[name] = float("nan")
    return out


def zoo_r2_best(ep: Episode) -> float:
    """Episode-level difficulty signal: best zoo r2 on this episode."""
    var = float(np.var(ep.qry_y.astype(np.float64)))
    if var <= 1e-12:
        return 0.0
    best = min((v for v in ep.zoo_rmse.values() if np.isfinite(v)), default=float("inf"))
    return 1.0 - best**2 / var if np.isfinite(best) else 0.0


REAL_RE = re.compile(r"^[0-9a-f]{16}$")


def list_real_tasks(limit: int | None = None, seed: int = 0) -> list[str]:
    names = sorted(d.name for d in ARCHIVE_ROOT.iterdir() if REAL_RE.match(d.name))
    if limit is not None and limit < len(names):
        rng = np.random.default_rng(seed)
        names = [names[i] for i in rng.choice(len(names), size=limit, replace=False)]
        names.sort()
    return names


def column_as_target(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Tier C decontamination: a random feature column becomes the target
    (the original target joins the features). The label FUNCTION is new, so
    dataset-level function memory cannot leak into the episode score; only
    the realistic feature marginals carry over — which is the point."""
    rng = np.random.default_rng([seed, 977])
    full = np.concatenate([X, y[:, None]], axis=1)
    n_cols = full.shape[1]
    candidates = []
    for c in range(n_cols - 1):  # exclude the original target as new target
        col = full[:, c]
        finite = col[np.isfinite(col)]
        if len(finite) > 0.9 * len(col) and np.std(finite) > 1e-9 and len(np.unique(finite)) >= 16:
            candidates.append(c)
    if not candidates:
        return None
    c = int(rng.choice(candidates))
    new_y = np.nan_to_num(full[:, c], nan=float(np.nanmean(full[:, c])))
    new_X = np.delete(full, c, axis=1)
    return np.ascontiguousarray(new_X), np.ascontiguousarray(new_y)


def prepare_episodes(
    tasks: list[str],
    per_task: int,
    base_seed: int,
    workers: int = 32,
    col_as_target: bool = False,
) -> list[Episode]:
    """Draw episodes and score the zoo once for each (parallel, CPU)."""
    from multiprocessing import Pool

    specs = [(t, base_seed + 7919 * k, col_as_target) for t in tasks for k in range(per_task)]
    with Pool(workers) as pool:
        episodes = pool.map(_build_one, specs, chunksize=4)
    return [e for e in episodes if e is not None]


def _build_one(spec: tuple[str, int, bool]) -> Episode | None:
    task, seed, col_as_target_flag = spec
    try:
        X, y = load_task(task)
        if col_as_target_flag:
            converted = column_as_target(X, y, seed)
            if converted is None:
                return None
            X, y = converted
        if X.shape[0] < 96 or X.shape[1] < 2:
            return None
        ep = draw_episode(task, seed, X, y)
        ep.zoo_rmse = score_zoo(ep)
        if sum(np.isfinite(v) for v in ep.zoo_rmse.values()) < 3:
            return None
        ep.bucket = difficulty_bucket(zoo_r2_best(ep))
        return ep
    except Exception:
        return None
