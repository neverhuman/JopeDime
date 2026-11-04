"""H317 GP track: screener interface, neural-screener adapter, deterministic test fallback, residual correction.

Interface used by every GP arm:
    Screener.score(cid, view_index_set, dsl_list) -> np.ndarray[B, M]   (higher = better predicted SEARCH utility)

Real screener: the neural track's OOF fold model,
    rapid_hypothesis_lab.load_screener(run_dir, fold).score(view_feats[B, 997], feat412[M, 412], proj16[M, 16], sem24[M, 24])
For a pilot context the model of that context's outer fold is used (never trained on it or its family).

AUDIT boundary for the context input: the full-table view cache (`$H317_DATA/views/<cid>.npz`, kind "full") includes
the AUDIT rows' X and y, so a GP target only ever gets its SEARCH-row view set (kind "search",
`$H317_DATA/views-search/<cid>.npz`, built by the neural track). Full-table views are read only for BANK contexts
(other families), as their nearest-neighbour codes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

import numpy as np  # noqa: E402

import gp_grammar_h317 as G  # noqa: E402

N_VIEWS = 32
REGIMES = ((0, 8), (8, 16), (16, 24), (24, 32))
VIEW_CTX_DIM = 997


def consensus_weights(n_views: int) -> np.ndarray:
    """H257 view weighting: canonical view 0 gets 0.35, the other views share 0.65."""
    if n_views == 1:
        return np.ones(1)
    w = np.full(n_views, 0.65 / (n_views - 1))
    w[0] = 0.35
    return w


def zscore_rows(s: np.ndarray) -> np.ndarray:
    """Per-view (row) standardization across candidates (H257 `standardize_scores_per_query`)."""
    s = np.asarray(s, np.float64)
    mu = s.mean(axis=-1, keepdims=True)
    sd = s.std(axis=-1, keepdims=True)
    return (s - mu) / np.maximum(sd, 1e-9)


class Screener:
    name = "abstract"
    is_fallback = False
    n_views = N_VIEWS

    def score(self, cid: str, view_index_set: Sequence[int], dsl_list: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "is_fallback": self.is_fallback, "n_views": self.n_views}


# ----------------------------------------------------------------------------------------------------------------------
# Deterministic fallback (tests / dry runs only; refused by gp_lab for real arms)
# ----------------------------------------------------------------------------------------------------------------------
_FB_PRIOR = {"histgbm": 0.35, "histgbm2": 0.30, "gpr": 0.25, "ksmooth": 0.15, "krr": 0.15, "ridge_gcv": 0.10,
             "ridge": 0.08, "elasticnet": 0.08, "ensemble": 0.30, "stack": 0.30, "portfolio": 0.35}


class FallbackScreener(Screener):
    """score = structural prior(head family, complexity) + view-dependent hash noise. No data, no learning."""

    name = "fallback_hash_prior"
    is_fallback = True

    def __init__(self, seed: int = 317999, noise: float = 0.3):
        self.seed, self.noise = seed, noise

    def score(self, cid, view_index_set, dsl_list):
        out = np.zeros((len(view_index_set), len(dsl_list)))
        for j, d in enumerate(dsl_list):
            prior = _FB_PRIOR.get(G.head_family(d), 0.0) - 0.01 * G.complexity(d)
            sha = G.sha256_text(d)
            for i, v in enumerate(view_index_set):
                h = C.stream_seed(self.seed, cid, int(v), sha)
                out[i, j] = prior + self.noise * ((h % 10_000) / 10_000.0 - 0.5)
        return out


# ----------------------------------------------------------------------------------------------------------------------
# SEARCH-only view features for GP contexts
# ----------------------------------------------------------------------------------------------------------------------
class SearchViewProvider:
    """[32, 997] screener input of a GP target context, keyed (cid, "search"): the neural track's SEARCH-row view set
    (`crop_shuffle_synthetic_views.context_view_feats(cid, "search")`, $H317_DATA/views-search/<cid>.npz; the file's
    `kind` field is checked there). The full-table view set (kind "full") is never used for a GP target."""

    def __init__(self, view_fn: Optional[Callable[..., np.ndarray]] = None):
        self.view_fn = view_fn
        self._lock = threading.Lock()
        self._mem: Dict[str, np.ndarray] = {}

    def _fn(self):
        if self.view_fn is None:
            self.view_fn = importlib.import_module("crop_shuffle_synthetic_views").context_view_feats
        return self.view_fn

    def __call__(self, cid: str, kind: str = "search") -> np.ndarray:
        if kind != "search":
            raise C.GuardViolation("GP screener context input must be the SEARCH-row view set")
        with self._lock:
            if cid not in self._mem:
                vf = np.asarray(self._fn()(cid, "search"), np.float32)
                if vf.shape != (N_VIEWS, VIEW_CTX_DIM):
                    raise ValueError(f"{cid[:12]}: SEARCH view set shape {vf.shape}")
                self._mem[cid] = vf
            return self._mem[cid]


# ----------------------------------------------------------------------------------------------------------------------
# Neural adapter
# ----------------------------------------------------------------------------------------------------------------------
class NeuralScreenerAdapter(Screener):
    is_fallback = False

    def __init__(self, model_obj, view_feats_fn: Callable[[str], np.ndarray], name: str = "neural_oof",
                 higher_is_better: bool = True, family_of: Optional[Dict[str, str]] = None):
        self.model = model_obj
        self.view_feats_fn = view_feats_fn
        self.family_of = family_of or {}
        self.name = name
        self.sign = 1.0 if higher_is_better else -1.0
        self._lock = threading.Lock()  # torch modules are not re-entrant across threads
        self._cache: Dict[Tuple[str, Tuple[int, ...], str], np.ndarray] = {}
        self.calls = 0
        self.scored = 0
        # Scale L (InducedSetBlock over candidates) makes a candidate's score depend on the batch: no caching then.
        models = getattr(model_obj, "models", None)
        self.cache_scores = bool(models) and all(getattr(m, "set_blocks", None) is None for m in models)

    def score(self, cid, view_index_set, dsl_list):
        """Cached per (cid, views, program) only for candidate-isolated models (exact); batch-coupled models are
        always scored on the full requested batch."""
        views = tuple(int(v) for v in view_index_set)
        if not dsl_list:
            return np.zeros((len(views), 0))
        vf = np.asarray(self.view_feats_fn(cid), np.float32)
        if vf.shape != (N_VIEWS, VIEW_CTX_DIM):
            raise ValueError(f"view feats shape {vf.shape} != {(N_VIEWS, VIEW_CTX_DIM)}")
        shas = [G.sha256_text(d) for d in dsl_list]
        use_cache = self.cache_scores
        missing = [j for j, h in enumerate(shas) if not use_cache or (cid, views, h) not in self._cache]
        if missing:
            f412, p16, s24 = G.featurize_batch([dsl_list[j] for j in missing])
            with self._lock:  # target-bound call: the neural Screener raises ScreenerLeak if cid/family was trained on
                out = self.model.score(vf[list(views)], f412, p16, s24, target_cid=cid,
                                       target_family=self.family_of.get(cid))
                self.calls += 1
                self.scored += len(missing)
            out = np.asarray(out, np.float64)
            if out.shape != (len(views), len(missing)):
                raise ValueError(f"screener returned {out.shape}")
            if not use_cache:
                return self.sign * out
            for k, j in enumerate(missing):
                self._cache[(cid, views, shas[j])] = out[:, k]
        return self.sign * np.stack([self._cache[(cid, views, h)] for h in shas], axis=1)


class FoldScreeners:
    """Pick the OOF model of the context's outer fold (family_folds seed 317140)."""

    def __init__(self, by_fold: Dict[int, Screener], fold_of: Dict[str, int]):
        self.by_fold, self.fold_of = by_fold, fold_of

    def for_context(self, cid: str) -> Screener:
        return self.by_fold[self.fold_of[cid]]


def load_neural_fold_screeners(run_dir: Path, fold_of: Dict[str, int], view_provider: Callable[[str], np.ndarray],
                               folds: Sequence[int] = range(C.N_FOLDS), seeds: Optional[Sequence[int]] = None,
                               module: str = "rapid_hypothesis_lab",
                               family_of: Optional[Dict[str, str]] = None) -> FoldScreeners:
    """Unbound OOF fold screeners (`rapid_hypothesis_lab.load_fold_screener`); every score passes target_cid, so the
    neural side re-checks that the target and its family are outside that fold model's training set."""
    lab = importlib.import_module(module)
    by_fold = {}
    for f in folds:
        obj = lab.load_fold_screener(run_dir, f, seeds=seeds, device="cpu")
        by_fold[int(f)] = NeuralScreenerAdapter(obj, view_provider, name=f"neural_oof:{Path(run_dir).name}:fold{f}",
                                                family_of=family_of)
    return FoldScreeners(by_fold, fold_of)


# ----------------------------------------------------------------------------------------------------------------------
# Residual correction from exact SEARCH results (Seed D "2+2" and the per-generation screening of both evo arms)
# ----------------------------------------------------------------------------------------------------------------------
class ResidualCorrector:
    """Calibrate z-scored screener consensus to SEARCH utility with an affine fit (slope shrunk to a prior), then
    propagate the revealed residuals to unrevealed candidates with a Nadaraya-Watson kernel over `ext_descriptor`.

    Only SEARCH utilities enter. Failures (-2.0) are clipped to `u_floor` to limit their leverage.
    """

    def __init__(self, slope_prior: float = 0.10, slope_lambda: float = 1.0, kappa: float = 0.5,
                 bandwidth: Optional[float] = None, u_floor: float = -0.5):
        self.slope_prior, self.slope_lambda, self.kappa = slope_prior, slope_lambda, kappa
        self.bandwidth, self.u_floor = bandwidth, u_floor

    def fit_predict(self, s_cand: np.ndarray, d_cand: np.ndarray, s_rev: np.ndarray, d_rev: np.ndarray,
                    u_rev: np.ndarray) -> Tuple[np.ndarray, dict]:
        s_cand = np.asarray(s_cand, np.float64)
        if len(u_rev) == 0:
            return s_cand.copy(), {"n_rev": 0}
        u = np.maximum(np.asarray(u_rev, np.float64), self.u_floor)
        s_rev = np.asarray(s_rev, np.float64)
        n = len(u)
        sm, um = s_rev.mean(), u.mean()
        var = float(np.sum((s_rev - sm) ** 2))
        cov = float(np.sum((s_rev - sm) * (u - um)))
        lam = self.slope_lambda
        b = max(0.0, (cov + lam * self.slope_prior) / (var + lam))
        a = um - b * sm
        r = u - (a + b * s_rev)
        dist2 = np.sum((np.asarray(d_cand)[:, None, :] - np.asarray(d_rev)[None, :, :]) ** 2, axis=-1)
        h = self.bandwidth
        if h is None:
            pos = dist2[dist2 > 1e-12]
            h = float(np.sqrt(np.median(pos))) if pos.size else 1.0
            h = max(h, 1e-3)
        k = np.exp(-dist2 / (2.0 * h * h))
        delta = (k @ r) / (self.kappa + k.sum(axis=1))
        pred = a + b * s_cand + delta
        return pred, {"n_rev": n, "a": a, "b": b, "h": h, "resid_sd": float(np.std(r))}


def screen_matrix(screener: Screener, cid: str, dsls: Sequence[str], views: Sequence[int] = tuple(range(N_VIEWS))
                  ) -> np.ndarray:
    """[B, M] row-z-scored screener matrix."""
    return zscore_rows(screener.score(cid, list(views), list(dsls)))


def corrected_matrix(screener: Screener, cid: str, dsls: Sequence[str], revealed: Sequence[Tuple[str, float]],
                     corrector: Optional[ResidualCorrector] = None, views: Sequence[int] = tuple(range(N_VIEWS))
                     ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Score candidates and revealed programs in one batch; returns (S_corr [B, M] in utility units, consensus [M], info).

    `revealed` = [(dsl, SEARCH utility)] only. Per-view rows get the same calibration and residual shift.
    """
    corrector = corrector or ResidualCorrector()
    rdsl = [d for d, _ in revealed]
    S = screen_matrix(screener, cid, list(dsls) + rdsl, views)
    w = consensus_weights(S.shape[0])
    cons = w @ S
    M = len(dsls)
    if not revealed:
        return S[:, :M], cons[:M], {"n_rev": 0}
    d_all = np.stack([G.ext_descriptor(d) for d in list(dsls) + rdsl])
    pred, info = corrector.fit_predict(cons[:M], d_all[:M], cons[M:], d_all[M:], np.array([u for _, u in revealed]))
    a, b = info["a"], info["b"]
    shift = pred - (a + b * cons[:M])
    S_corr = a + b * S[:, :M] + shift[None, :]
    return S_corr, pred, info
