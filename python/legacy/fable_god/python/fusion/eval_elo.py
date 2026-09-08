"""ELO bridge: place TabFUSION (+ TabFUSION_ORACLE) into the hyperion Elo ranking.

Reads the harness's precomputed utility matrix (mean-CV roc_auc per model x dataset)
READ-ONLY from ~/hyperion/pack/, appends new model columns, and re-runs the SAME
pairwise Elo (identical to hyperion384_router_elo_study.fast_elo). Writes nothing to
the archive. TabFUSION_ORACLE = per-dataset max(tabpfn_v3, tabicl) is computed from
the pack's own expert columns (protocol-consistent, no model run). TabFUSION (the
actual model) is loaded from a {hash: fused_auc} JSON if provided.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

PACK = os.environ.get("HYPERION_PACK", "/home/ubuntu/hyperion/pack")


def _outcome(left, right):
    o = np.where(left > right, 1.0, 0.0)
    o[np.isclose(left, right, rtol=1e-12, atol=1e-12)] = 0.5
    return o.astype(np.float64)


def build_index(U, V):
    """All both-valid algorithm pairs per dataset -> (left, right, outcome)."""
    L, R, O = [], [], []
    for d in range(U.shape[0]):
        valid = np.flatnonzero(V[d]); vals = U[d, valid]
        for i in range(valid.size - 1):
            ra = valid[i + 1:].astype(np.int64); rv = vals[i + 1:]
            L.append(np.full(ra.size, int(valid[i]), dtype=np.int64)); R.append(ra)
            O.append(_outcome(np.full(rv.size, vals[i]), rv))
    return (np.concatenate(L), np.concatenate(R), np.concatenate(O))


def fast_elo(n, left, right, outcome, rounds=80, init=1000.0):
    ratings = np.full(n, init, dtype=np.float64)
    counts = np.bincount(left, minlength=n) + np.bincount(right, minlength=n)
    denom = np.maximum(counts, 1).astype(np.float64)
    for _ in range(rounds):
        exp_l = 1.0 / (1.0 + 10.0 ** ((ratings[right] - ratings[left]) / 400.0))
        grad = outcome - exp_l
        delta = np.bincount(left, weights=grad, minlength=n) - np.bincount(right, weights=grad, minlength=n)
        ratings += 64.0 * delta / denom
    return ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-auc", default=None, help="JSON {hash: fused_auc} for TabFUSION")
    ap.add_argument("--extra-auc", default=None, help="JSON {name: {hash: auc}} for more cols")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    z = np.load(f"{PACK}/router_bin_targets.npz", allow_pickle=True)
    U = z["utilities"].astype(np.float64); V = z["valid_mask"].astype(bool)
    algs = [str(a) for a in z["algorithms"]]
    idx = pd.read_parquet(f"{PACK}/router_bin_index.parquet")
    sel = idx[(idx.task == "bin") & (idx.is_real) & (idx.has_router) & (idx.split == "test")]
    rows = sel.pack_row.to_numpy().astype(np.int64)
    hashes = sel.hash.astype(str).to_numpy()
    Ut, Vt = U[rows], V[rows]                              # (230, 168)
    i_pfn, i_icl = algs.index("tabpfn_v3_classifier"), algs.index("tabicl")
    print(f"{len(rows)} binary test datasets | tabpfn_v3 col {i_pfn} (mean AUC "
          f"{Ut[Vt[:, i_pfn], i_pfn].mean():.4f}), tabicl col {i_icl} "
          f"(mean AUC {Ut[Vt[:, i_icl], i_icl].mean():.4f})", flush=True)

    names = list(algs); cols = [Ut]; masks = [Vt]

    def add(name, vals, valid):
        names.append(name); cols.append(vals[:, None]); masks.append(valid[:, None])

    # TabFUSION_ORACLE: per-dataset best of the two experts (from the pack itself)
    vp, vi = Vt[:, i_pfn], Vt[:, i_icl]
    orac = np.where(vp & vi, np.maximum(Ut[:, i_pfn], Ut[:, i_icl]),
                    np.where(vp, Ut[:, i_pfn], np.where(vi, Ut[:, i_icl], 0.0)))
    add("TabFUSION_ORACLE", orac, vp | vi)

    # TabFUSION (actual model): {hash: fused_auc} joined by hash
    for path, nm in [(args.fusion_auc, "TabFUSION")]:
        if path and os.path.exists(path):
            d = json.load(open(path))
            vals = np.array([d.get(h, np.nan) for h in hashes])
            valid = ~np.isnan(vals); vals = np.nan_to_num(vals)
            add(nm, vals, valid)
            print(f"{nm}: matched {valid.sum()}/{len(hashes)} test hashes | "
                  f"mean AUC {vals[valid].mean():.4f}", flush=True)
    if args.extra_auc and os.path.exists(args.extra_auc):
        for nm, dd in json.load(open(args.extra_auc)).items():
            vals = np.array([dd.get(h, np.nan) for h in hashes])
            valid = ~np.isnan(vals); add(nm, np.nan_to_num(vals), valid)

    Uall = np.concatenate(cols, 1); Vall = np.concatenate(masks, 1)
    left, right, out = build_index(Uall, Vall)
    elo = fast_elo(len(names), left, right, out, rounds=args.rounds)
    order = np.argsort(-elo)
    rank = {int(order[r]): r + 1 for r in range(len(order))}
    hi = {"TabFUSION", "TabFUSION_ORACLE", "tabpfn_v3_classifier", "tabicl", "oracle"}
    print(f"\n{'rank':>4} {'Elo':>8}  model")
    for r, j in enumerate(order[:args.top]):
        mark = "  <<<" if names[j] in hi else ""
        print(f"{r+1:>4} {elo[j]:>8.1f}  {names[j]}{mark}", flush=True)
    print("\nkey models:")
    for nm in ["TabFUSION_ORACLE", "TabFUSION", "tabpfn_v3_classifier", "tabicl"]:
        if nm in names:
            j = names.index(nm)
            print(f"  {nm:28s} Elo {elo[j]:7.1f}  rank {rank[j]}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
