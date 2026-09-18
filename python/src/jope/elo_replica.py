"""Fast exact replica of the official ladder Elo, vendored for dev-pool scoring.

Ported verbatim from the configured read-only fable harness (M0-verified:
official-vs-replica diff 0.0000 on dev_a; the fable red-team verified the same
replica against tools.elo on 60- and 300-dataset pools, diff 0.0000). The
official path (tools.elo.algorithm_matches + compute_elo) stays the scorer of
record for ship runs; this replica is for iteration speed on dev pools.

Semantics: round-robin logistic Elo over all per-dataset pairwise utility
(−RMSE) comparisons, 60 sweeps, K=64 averaged per player, init 1000, ties by
isclose(rel=abs=1e-12).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .prime_config import load_config

QUANT_MODELS = load_config().quant_models_root
LADDER_CACHE = QUANT_MODELS / "tools/artifacts/regression_router_cache_all_real_models_v1"

_LADDER = None


def load_ladder():
    global _LADDER
    if _LADDER is None:
        lab = np.load(LADDER_CACHE / "labels.npz", allow_pickle=True)
        _LADDER = {
            "rmse": lab["rmse"],
            "algorithms": [str(a) for a in lab["algorithms"]],
            "hashes": [str(x) for x in lab["dataset_hashes"]],
            "valid": lab["valid_mask"],
        }
        _LADDER["hidx"] = {h: i for i, h in enumerate(_LADDER["hashes"])}
    return _LADDER


def _augmented(cand_rmse: dict[str, float], hashes: list[str], name: str):
    lad = load_ladder()
    rows = [lad["hidx"][h] for h in hashes if h in lad["hidx"]]
    U = -lad["rmse"].copy()
    jc = np.full((U.shape[0], 1), np.nan)
    jv = np.zeros((U.shape[0], 1), bool)
    for h in hashes:
        if h not in lad["hidx"]:
            continue
        r = lad["hidx"][h]
        jr = cand_rmse.get(h)
        if jr is not None and math.isfinite(jr):
            jc[r, 0] = -float(jr)
            jv[r, 0] = True
    algos = lad["algorithms"] + [name]
    U_aug = np.hstack([U, jc])
    v_aug = np.hstack([lad["valid"], jv])
    return algos, U_aug, v_aug, rows


def official_elo(cand_rmse: dict[str, float], hashes: list[str], name: str = "cand") -> dict:
    """Exact official path: tools.elo.algorithm_matches + compute_elo (slow)."""
    sys.path.insert(0, str(QUANT_MODELS))
    from tools.elo import algorithm_matches, compute_elo  # read-only official import

    algos, U_aug, v_aug, rows = _augmented(cand_rmse, hashes, name)
    matches = algorithm_matches(algos, U_aug, v_aug, dataset_indices=rows)
    ratings = compute_elo(set(algos), matches)
    return _summarize(ratings, name, cand_rmse, hashes)


def fast_elo(cand_rmse: dict[str, float], hashes: list[str], name: str = "cand") -> dict:
    """Closed-form equivalent of compute_elo over algorithm_matches (verified exact)."""
    algos, U_aug, v_aug, rows = _augmented(cand_rmse, hashes, name)
    order = np.argsort(np.array(algos))  # compute_elo sorts model names
    algos_sorted = [algos[i] for i in order]
    U = U_aug[rows][:, order]
    V = np.asarray(v_aug[rows][:, order], bool) & np.isfinite(U)
    n = len(algos_sorted)
    S = np.zeros((n, n))
    C = np.zeros((n, n))
    for r in range(U.shape[0]):
        u = U[r]
        v = V[r]
        vi = np.where(v)[0]
        if len(vi) < 2:
            continue
        uu = u[vi]
        diff = uu[:, None] - uu[None, :]
        tol = np.maximum(np.abs(uu[:, None]), np.abs(uu[None, :])) * 1e-12 + 1e-12
        close = np.abs(diff) <= tol
        out = np.where(close, 0.5, (diff > 0).astype(float))
        np.fill_diagonal(out, 0.0)
        ones = np.ones((len(vi), len(vi)))
        np.fill_diagonal(ones, 0.0)
        S[np.ix_(vi, vi)] += out
        C[np.ix_(vi, vi)] += ones
    ratings = np.full(n, 1000.0)
    counts = C.sum(axis=1)
    for _ in range(60):
        E = 1.0 / (1.0 + 10.0 ** ((ratings[None, :] - ratings[:, None]) / 400.0))
        np.fill_diagonal(E, 0.0)
        delta = (S - E * C).sum(axis=1)
        ratings = ratings + 64.0 * delta / np.maximum(1, counts)
    rat = dict(zip(algos_sorted, ratings))
    return _summarize(rat, name, cand_rmse, hashes)


def per_dataset_diagnostics(cand_rmse: dict[str, float], hashes: list[str]) -> list[dict]:
    lad = load_ladder()
    out = []
    for h in hashes:
        if h not in lad["hidx"]:
            continue
        r = lad["hidx"][h]
        jr = cand_rmse.get(h)
        inc = lad["rmse"][r]
        m = np.asarray(lad["valid"][r], bool) & np.isfinite(inc)
        inc_valid = inc[m]
        if jr is not None and math.isfinite(jr) and len(inc_valid):
            beat = int((jr < inc_valid).sum())
            nn = len(inc_valid)
            p = min(max(beat / nn, 1e-4), 1 - 1e-4)
            out.append({
                "hash": h, "rmse": jr, "rank": int((inc_valid < jr).sum()) + 1, "field": nn,
                "beat": beat, "pctile": round(100 * beat / nn, 1),
                "elo_local": round(1000 + 400 * math.log10(p / (1 - p))),
            })
        else:
            out.append({"hash": h, "rmse": None, "rank": None, "field": int(len(inc_valid)),
                        "beat": 0, "pctile": 0.0, "elo_local": None})
    return out


def _summarize(ratings: dict[str, float], name: str, cand_rmse, hashes) -> dict:
    order = sorted(ratings.items(), key=lambda kv: -kv[1])
    jrank = next(i for i, (k, _) in enumerate(order) if k == name) + 1
    per = per_dataset_diagnostics(cand_rmse, hashes)
    scored = [p for p in per if p["rank"] is not None]
    return {
        "elo": round(ratings[name], 1),
        "field_rank": jrank,
        "field_size": len(order),
        "battles_won": sum(1 for p in scored if p["rank"] == 1),
        "battles_top3": sum(1 for p in scored if p["rank"] <= 3),
        "n_scored": len(scored),
        "mean_pctile": round(float(np.mean([p["pctile"] for p in scored])), 2) if scored else None,
        "median_rank": int(np.median([p["rank"] for p in scored])) if scored else None,
        "top_incumbents": [(k, round(v, 1)) for k, v in order[:5] if k != name][:5],
        "per_dataset": per,
    }
