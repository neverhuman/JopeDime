"""Deployable corrector: closed-form standardized ridge, JSON artifact.

Mirrors src/jope/selector_reranker.py exactly in spirit: numpy-only,
JSON-serializable (no pickle), intercept unpenalized, features standardized
by train statistics stored in the artifact. Two heads share one feature
vector: `rank` (within-task z target — the primary ranking head) and
`absolute` (honest score — budget gating / calibrated reporting), plus a
bucketed residual-quantile calibration table for the absolute head.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float,
              feature_names: list[str]) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    xs = (x - mean) / scale
    n, d = xs.shape
    design = np.hstack([np.ones((n, 1)), xs])
    reg = l2 * np.eye(d + 1)
    reg[0, 0] = 0.0  # unpenalized intercept
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return {
        "feature_names": list(feature_names),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(beta[0]),
        "coef": beta[1:].tolist(),
        "l2": float(l2),
    }


def ridge_predict(head: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(head["feature_mean"])
    scale = np.asarray(head["feature_scale"])
    coef = np.asarray(head["coef"])
    return head["intercept"] + ((x - mean) / scale) @ coef


def residual_calibration(pred: np.ndarray, actual: np.ndarray,
                         n_buckets: int = 10) -> dict:
    """Empirical p10/p50/p90 of (actual - pred) per prediction decile."""
    edges = np.quantile(pred, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p10, p50, p90 = [], [], []
    for b in range(n_buckets):
        m = (pred >= edges[b]) & (pred < edges[b + 1])
        if m.sum() >= 5:
            r = actual[m] - pred[m]
            p10.append(float(np.quantile(r, 0.10)))
            p50.append(float(np.quantile(r, 0.50)))
            p90.append(float(np.quantile(r, 0.90)))
        else:
            p10.append(0.0)
            p50.append(0.0)
            p90.append(0.0)
    return {"bucket_edges": [None if not np.isfinite(e) else float(e)
                             for e in edges.tolist()],
            "residual_p10": p10, "residual_p50": p50, "residual_p90": p90}


def build_artifact(engine: str, rank_head: dict, abs_head: dict,
                   pca_bundle: dict, calibration: dict, fit_meta: dict,
                   replay: dict) -> dict:
    pca_out = {}
    for kind, pca in (pca_bundle or {}).items():
        if pca is None:
            continue
        pca_out[kind] = {"mean": np.asarray(pca["mean"]).tolist(),
                         "components": np.asarray(pca["components"]).tolist()}
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "model_type": "linear_ridge",
        "target": "honest_within_task_z",
        **{k: rank_head[k] for k in ("feature_names", "feature_mean",
                                     "feature_scale", "coef", "intercept",
                                     "l2")},
        "absolute_head": abs_head,
        "pca": pca_out,
        "calibration": calibration,
        "fit": fit_meta,
        "replay": replay,
    }


def write_artifact(artifact: dict, path: Path) -> str:
    text = json.dumps(artifact, sort_keys=True, indent=1) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_artifact(path: Path) -> dict:
    art = json.loads(Path(path).read_text())
    if art.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema: {art.get('schema_version')!r}")
    return art


def score(artifact: dict, x: np.ndarray) -> np.ndarray:
    """Rank-head score for feature matrix x (columns = feature_names order)."""
    return ridge_predict(artifact, x)


def score_absolute_with_uncertainty(artifact: dict, x: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = ridge_predict(artifact["absolute_head"], x)
    cal = artifact["calibration"]
    edges = [(-np.inf if e is None else e) for e in cal["bucket_edges"]]
    p10 = np.zeros_like(pred)
    p50 = np.zeros_like(pred)
    p90 = np.zeros_like(pred)
    for i, p in enumerate(pred):
        b = 0
        for j in range(len(edges) - 1):
            if edges[j] <= p < edges[j + 1]:
                b = j
                break
        else:
            b = len(cal["residual_p10"]) - 1
        p10[i] = p + cal["residual_p10"][b]
        p50[i] = p + cal["residual_p50"][b]
        p90[i] = p + cal["residual_p90"][b]
    return pred, p10, p50, p90


class CorrectorHandle:
    """Callable wrapper around a published corrector artifact.

    WHY THIS EXISTS
    ---------------
    `load_artifact` returns a plain dict, and `score()` takes a numpy matrix whose columns are
    already in `feature_names` order. The race worker needs neither: it has one candidate row at
    a time and wants a scalar. Before this class, `gp_worker.load_corrector` tried
    `art.mtime = mt` on that dict — an AttributeError, swallowed by a bare `except`, so
    `ctx.corrector` was ALWAYS None and `select_diverse` silently fell back to the raw proxy
    ordering for every race ever run.

    That fallback is not neutral. The proxy ANTI-correlates with realized score inside the
    finalist band: measured within-task Spearman is -0.0304 (featmath) / -0.0362 (newmath) for
    raw proxy versus +0.1280 / +0.1873 for the corrector. The winner's curse was being selected
    for.

    The handle owns the feature assembly so it cannot drift from the trainer: it rebuilds the
    SAME dict `corrector_refit._feature_vector` builds, then projects it onto the artifact's own
    `feature_names` order. A feature the artifact does not know is dropped; a feature it expects
    and the row lacks is 0.0 — the same convention the trainer uses for absent rungs.
    """

    __slots__ = ("artifact", "mtime", "ver", "_names", "_index", "_missing")

    def __init__(self, artifact: dict, mtime: float | None = None):
        if not isinstance(artifact, dict):
            raise TypeError("corrector artifact must be a dict")
        names = artifact.get("feature_names")
        if not names:
            raise ValueError("corrector artifact has no feature_names")
        self.artifact = artifact
        self.mtime = mtime
        self.ver = str(artifact.get("version") or artifact.get("created_ms") or "unknown")
        self._names = list(names)
        self._index = {n: i for i, n in enumerate(self._names)}
        self._missing = 0

    def features(self, canonical: str, row: dict | None = None) -> np.ndarray:
        """Build the artifact-ordered feature row. Mirrors corrector_refit._feature_vector."""
        from .genome_features import hashed_ngrams, structural_features

        feats = structural_features(canonical)
        if not feats:
            raise ValueError("genome produced no structural features")
        feats = dict(feats)
        feats.update(hashed_ngrams(canonical, buckets=256) or {})
        row = row or {}
        for k in ("proxy_rung0", "proxy_rung1", "proxy_rung2", "proxy_score"):
            v = row.get(k)
            feats[f"f_{k}"] = float(v) if v is not None else 0.0
            feats[f"f_{k}_missing"] = 0.0 if v is not None else 1.0

        x = np.zeros((1, len(self._names)), dtype=np.float64)
        for name, value in feats.items():
            j = self._index.get(name)
            if j is not None:
                x[0, j] = float(value)
        return x

    def score(self, canonical: str, row: dict | None = None) -> float:
        """Scalar rank-head score. Raises on malformed input — callers decide how loud to be."""
        value = float(score(self.artifact, self.features(canonical, row))[0])
        if not np.isfinite(value):
            raise ValueError("corrector produced a non-finite score")
        return value


def load_handle(path, mtime: float | None = None) -> CorrectorHandle:
    """Load a published artifact as a callable handle. Raises rather than returning None."""
    return CorrectorHandle(load_artifact(path), mtime=mtime)
