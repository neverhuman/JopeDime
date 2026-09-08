"""Phase-A Elo router (System A): a per-dataset router over a diverse candidate set,
trained FAST on the already-binary-packed model-utility cache + cached dataset
features, graded on the REAL hyperion Elo (vendored_elo.EloGrader).

No proxy metric: selection is held-out real ΔElo (K-fold over the 996 train
datasets); the 248 test datasets are graded ONCE at the end. Anti-leakage by
construction (features are dataset descriptors, never test utilities) + explicit
NaN-poison canary. Reproduces + aims to beat the +27 ΔElo champion.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.expanduser("~/hyperion/eval_harness"))
import vendored_elo as v  # noqa: E402

# greedy-oracle candidate set (indices in the 168-algo cache), anchor first
CAND = {"tabpfn_v3_classifier": 143, "tabdpt_classifier": 132, "tabicl": 133,
        "pca_lgbm": 104, "stacking_trees_meta": 130, "ebm_interactions": 36,
        "tabpfn_v25_classifier_real": 142, "vowpalwabbit_logistic": 148}
PACK_IDX = "/home/ubuntu/hyperion/pack/router_bin_index.parquet"
EMB = "/home/ubuntu/hyperion/features/latest_router_sweep/latest_router_sweep_features.npz"
LM50K = "/home/ubuntu/hyperion/pack_src/landmarks50k.npz"


def build_features(D, use_emb=True, use_lm50k=True):
    n = len(D["hashes"])
    parts = [D["landmarks"], D["sketches"],
             np.log1p(D["nfeat"])[:, None], np.log1p(D["nrows"])[:, None]]
    names = ["base"]
    if use_emb:
        z = np.load(EMB, allow_pickle=True)
        h2i = {str(h): i for i, h in enumerate(z["dataset_hashes"])}
        feats = z["features"]; d = feats.shape[1]
        M = np.full((n, d), np.nan, np.float32)
        for i, h in enumerate(D["hashes"]):
            if h in h2i:
                M[i] = feats[h2i[h]]
        col_mean = np.nanmean(M, 0); M = np.where(np.isnan(M), col_mean, M)
        parts.append(M); names.append(f"emb{d}({np.isfinite(M[:,0]).sum()}cov)")
    if use_lm50k:
        idx = pd.read_parquet(PACK_IDX)[["hash", "pack_row"]]
        h2row = dict(zip(idx.hash.astype(str), idx.pack_row.astype(int)))
        z = np.load(LM50K, allow_pickle=True); tgt = z["targets"]
        d = tgt.shape[1]; M = np.full((n, d), np.nan, np.float32)
        for i, h in enumerate(D["hashes"]):
            if h in h2row:
                M[i] = tgt[h2row[h]]
        col_mean = np.nanmean(M, 0); M = np.where(np.isnan(M), col_mean, M)
        parts.append(M); names.append(f"lm50k{d}")
    F = np.concatenate(parts, 1).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, "+".join(names)


def fit_predict(F, U, Vd, fit_rows, eval_rows, cand_idx, anchor, n_estimators=300, seed=0):
    """Predict per-candidate LIFT over anchor (util[c]-util[anchor]) via ET+KNN
    ensemble, trained on rows where BOTH candidate and anchor are valid. Returns
    (n_eval, n_cand) predicted lift; anchor's own column is ~0 by construction."""
    Fz = StandardScaler().fit(F[fit_rows]).transform(F)     # z-score for KNN
    pred = np.full((len(eval_rows), len(cand_idx)), -1e9, np.float64)
    for j, c in enumerate(cand_idx):
        rows = fit_rows[Vd[fit_rows, c] & Vd[fit_rows, anchor]]
        if len(rows) < 20:
            pred[:, j] = 0.0 if c == anchor else -1e9
            continue
        y = U[rows, c] - U[rows, anchor]                    # LIFT target
        et = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
        et.fit(F[rows], y)
        kn = KNeighborsRegressor(n_neighbors=min(20, len(rows)), weights="distance")
        kn.fit(Fz[rows], y)
        hg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=31, random_state=seed)
        hg.fit(F[rows], y)
        pred[:, j] = (et.predict(F[eval_rows]) + kn.predict(Fz[eval_rows])
                      + hg.predict(F[eval_rows])) / 3.0
    return pred


def choices_from_pred(pred, Vd, eval_rows, cand_idx, anchor, tau):
    """Keep anchor unless a valid candidate's predicted lift over anchor > tau."""
    n = len(eval_rows); ch = np.full(n, anchor, np.int64)
    valid = Vd[eval_rows][:, cand_idx]                      # (n, n_cand)
    p = np.where(valid, pred, -1e9)
    best_j = np.argmax(p, 1); best_lift = p[np.arange(n), best_j]
    take = best_lift > tau
    ch[take] = np.array(cand_idx)[best_j[take]]
    return ch


def grade(D, rows, choices):
    g = v.EloGrader(D["util"], D["valid"], rows)
    elo = g.elo_for_choices(choices)
    anch = g.fixed_algorithm_elo(D["anchor"])
    return elo - anch, elo, anch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--taus", default="0.0,0.002,0.005,0.01,0.02")
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--no-lm50k", action="store_true")
    ap.add_argument("--cand-mode", default="greedy", choices=["greedy", "topk", "all"])
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = v.load_cache("bin")
    F, fname = build_features(D, use_emb=not args.no_emb, use_lm50k=not args.no_lm50k)
    tr, te = D["tr"], D["te"]; anchor = D["anchor"]
    if args.cand_mode == "greedy":
        cand = list(CAND.values())
    elif args.cand_mode == "all":
        cand = list(range(D["util"].shape[1]))
    else:                                                   # topk by oracle-win-count on TRAIN
        win = v.EloGrader(D["util"], D["valid"], tr).oracle_choices()
        top = [int(c) for c, _ in sorted(zip(*np.unique(win, return_counts=True)),
                                         key=lambda x: -x[1])][:args.topk]
        cand = top
    if anchor not in cand:
        cand = [anchor] + cand
    taus = [float(x) for x in args.taus.split(",")]
    print(f"router cache: {len(D['hashes'])} datasets ({len(tr)} train / {len(te)} test) | "
          f"{len(cand)} candidates | features {F.shape[1]}d [{fname}]", flush=True)

    # anchor + oracle reference on te
    _, _, anch_te = grade(D, te, np.full(len(te), D["anchor"]))
    oracle_ch = v.EloGrader(D["util"], D["valid"], te).oracle_choices()
    d_orac, _, _ = grade(D, te, oracle_ch)
    print(f"te anchor Elo {anch_te:.1f} | ORACLE ΔElo +{d_orac:.1f} (ceiling)", flush=True)

    # K-fold held-out ΔElo for tau selection (NEVER touches te)
    kf = KFold(args.folds, shuffle=True, random_state=args.seed)
    cv = {t: [] for t in taus}
    for fi, (a, b) in enumerate(kf.split(tr)):
        fit_rows, ev_rows = tr[a], tr[b]
        pred = fit_predict(F, D["util"], D["valid"], fit_rows, ev_rows, cand, anchor, seed=args.seed)
        for t in taus:
            ch = choices_from_pred(pred, D["valid"], ev_rows, cand, anchor, t)
            dlt, _, _ = grade(D, ev_rows, ch); cv[t].append(dlt)
    print("\nheld-out CV ΔElo by tau:")
    for t in taus:
        print(f"  tau={t:<6} mean ΔElo {np.mean(cv[t]):+.2f}  (folds {np.round(cv[t],1)})", flush=True)
    best_tau = max(taus, key=lambda t: np.mean(cv[t]))
    print(f"-> selected tau={best_tau} (CV ΔElo {np.mean(cv[best_tau]):+.2f})", flush=True)

    # FINAL: fit all train, grade te ONCE
    pred_te = fit_predict(F, D["util"], D["valid"], tr, te, cand, anchor, seed=args.seed)
    ch_te = choices_from_pred(pred_te, D["valid"], te, cand, anchor, best_tau)
    d_te, elo_te, _ = grade(D, te, ch_te)
    route_rate = float(np.mean(ch_te != D["anchor"]))
    inv={vv:kk for kk,vv in CAND.items()}
    used = {inv.get(c, f'algo{c}'): int((ch_te == c).sum()) for c in set(ch_te.tolist())}

    # leakage canary: poison te utilities, refit-free choices must be identical
    Dp = dict(D); Dp_util = D["util"].copy(); Dp_util[te] = np.nan
    pred_te2 = fit_predict(F, Dp_util, D["valid"], tr, te, cand, anchor, seed=args.seed)
    ch_te2 = choices_from_pred(pred_te2, D["valid"], te, cand, anchor, best_tau)
    canary = "PASS" if np.array_equal(ch_te, ch_te2) else "FAIL"

    print(f"\n=== FINAL (te, 248 datasets) ===")
    print(f"router ΔElo {d_te:+.2f} vs anchor  (champion +27.0, 2σ bar 11.2, oracle +{d_orac:.1f})")
    print(f"route rate {100*route_rate:.0f}% | choices: {used}")
    print(f"leakage canary: {canary}", flush=True)


if __name__ == "__main__":
    main()
