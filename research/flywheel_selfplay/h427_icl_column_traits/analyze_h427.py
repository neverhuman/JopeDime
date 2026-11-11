#!/usr/bin/env python3
"""H427 (analysis only, no verdict): where does H424's in-context column earn its stack weight and its gain on DEV?
Stored files only: H424's DEV inference bank (SEARCH rows, no AUDIT target) and value file, H425's frozen list and
screen output. No table, no model, nothing on CONFIRM.

  analyze_h427.py --out FILE

Outcomes per DEV context:   weight = the stack weight of the model's column;  gain = clip(I - B, +-0.5) in u.
Traits (SEARCH rows only unless marked):
  log_rows        log10 of table rows
  log_features    log10 of feature columns
  lowcard_share   share of feature columns with <= 10 distinct values (a stand-in for "categorical"; no type is stored)
  missing_share   share of missing feature cells
  abs_skew        |skewness| of the target, capped at 10
  alone_search    column-alone value against the stack before deployment: out-of-fold u of the column minus the best
                  stored candidate's out-of-fold u
  copy_flag       1 if H425's tier is "copy" or "possible"
  stack_fit       (added) the best stored candidate's out-of-fold u: how well the GP candidates fit the table
  alone_audit     (hindsight only) AUDIT u of the column alone minus the stored stack's
Models: ridge regression on standardised traits; 5-fold cross-validation grouped by H426's `feature` clusters
(fold = sha256("H427-fold" NUL cluster) mod 5); penalty from an inner grouped CV over {0.1, 1, 10, 100}.
"traits" = the eight pre-deployment traits; "hindsight" = traits + alone_audit.
Noise: out-of-fold R2 = 1 - SSE / SST against the fold-training mean; its permutation null (2,000 shuffles of the
outcome, same folds, same procedure) gives p = share of shuffles with R2 at least as high.
Also: Spearman correlation of every trait with each outcome and its cluster-bootstrap 95% interval (2,000 draws);
outcome means by tercile of every trait.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import restate_h426 as R26  # noqa: E402  (clusterings from the frozen H425 screen output)

EV = Path("/home/ubuntu/jope_plan/evidence")
BANK = Path("/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz")
APPLY = EV / "H424/results/H424_DEV.apply.json"
LIST = EV / "H425/results/EXCLUSION_DEV.json"
SCREEN = EV / "H425/results/SCREEN_DEV.json"
TRAITS = ("log_rows", "log_features", "lowcard_share", "missing_share", "abs_skew", "alone_search", "copy_flag", "stack_fit")
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
K, N_PERM, N_BOOT, SEED = 5, 2000, 2000, 427101
CLIP = 0.5


def fold_of(cluster: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\0{cluster}".encode()).digest()[:8], "big") % K


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def cv_predict(X, y, folds, inner) -> np.ndarray:
    """Out-of-fold ridge predictions; the penalty of each outer fold is chosen by the inner grouped folds."""
    pred = np.zeros(len(y))
    for k in range(K):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        best, lam_best = None, LAMBDAS[0]
        for lam in LAMBDAS:
            sse = 0.0
            for j in range(K):
                a, b = tr & (inner != j), tr & (inner == j)
                if b.any() and a.any():
                    sse += float(((ridge_fit(X[a], y[a], lam)(X[b]) - y[b]) ** 2).sum())
            if best is None or sse < best:
                best, lam_best = sse, lam
        pred[te] = ridge_fit(X[tr], y[tr], lam_best)(X[te])
    return pred


def r2_oof(y, pred, folds) -> float:
    sst = sum(float(((y[folds == k] - y[folds != k].mean()) ** 2).sum()) for k in range(K) if (folds == k).any())
    return 1.0 - float(((y - pred) ** 2).sum()) / sst


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per = json.loads(APPLY.read_text())["per_context"]
    lst = json.loads(LIST.read_text())
    screen = json.loads(SCREEN.read_text())
    if hashlib.sha256(SCREEN.read_bytes()).hexdigest() != lst["screen_sha256"]:
        raise SystemExit("the screen file is not the one the frozen list was cut from")
    cids = sorted(per)
    if sorted(lst["contexts"]) != cids or len(cids) != 115:
        raise SystemExit("not the 115 DEV contexts")
    b = np.load(BANK, allow_pickle=False)
    if set(b["cohort"].tolist()) != {"DEV"} or sorted(b["cids"].tolist()) != cids:
        raise SystemExit("the bank is not the H424 DEV bank")
    ss, sa = b["s_shape"].astype(np.int64), b["a_shape"].astype(np.int64)
    s_off = np.concatenate([[0], np.cumsum(ss[:, 0] * ss[:, 1])])
    l_off = np.concatenate([[0], np.cumsum(ss[:, 0])])
    tr = {}
    for i, c in enumerate(b["cids"].tolist()):
        r, cc = ss[i].tolist()
        X = b["search"][s_off[i]:s_off[i + 1]].reshape(r, cc)[:, :-1].astype(np.float64)
        y = b["y64"][l_off[i]:l_off[i + 1]]
        nd = np.array([len(np.unique(col[np.isfinite(col)])) for col in X.T])
        sd = float(y.std())
        skew = float(np.mean((y - y.mean()) ** 3) / sd ** 3) if sd > 0 else 0.0
        p = per[c]
        tr[c] = {"log_rows": float(np.log10(r + int(sa[i][0]))), "log_features": float(np.log10(cc - 1)),
                 "lowcard_share": float((nd <= 10).mean()), "missing_share": float(np.mean(~np.isfinite(X))),
                 "abs_skew": float(min(abs(skew), 10.0)), "alone_search": float(p["oof_u_ridgeres"] - p["oof_u_best_stored"]),
                 "copy_flag": float(lst["contexts"][c]["tier"] in ("copy", "possible")), "stack_fit": float(p["oof_u_best_stored"]),
                 "alone_audit": float(p["alone_ridgeres"] - p["B"])}
    cl = R26.clusterings(screen, cids)["feature"]
    X = np.array([[tr[c][t] for t in TRAITS] for c in cids])
    Xh = np.column_stack([X, [tr[c]["alone_audit"] for c in cids]])
    out_y = {"weight": np.array([per[c]["w_added_I"][0] for c in cids]),
             "gain": np.clip(np.array([per[c]["I"] - per[c]["B"] for c in cids]), -CLIP, CLIP)}
    folds = np.array([fold_of(cl[c], "H427-fold") for c in cids])
    inner = np.array([fold_of(cl[c], "H427-inner") for c in cids])
    rng = np.random.default_rng(SEED)
    models = {}
    for oname, y in out_y.items():
        for mname, M in (("traits", X), ("hindsight", Xh)):
            pred = cv_predict(M, y, folds, inner)
            r2 = r2_oof(y, pred, folds)
            null = np.array([r2_oof(yp, cv_predict(M, yp, folds, inner), folds) for yp in (rng.permutation(y) for _ in range(N_PERM))])
            full = ridge_fit(M, y, 1.0)
            names = TRAITS + (("alone_audit",) if mname == "hindsight" else ())
            mu, sd = M.mean(0), M.std(0) + 1e-12
            Z = (M - mu) / sd
            coef = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
            models[f"{oname}.{mname}"] = {"r2_out_of_fold": r2, "spearman_out_of_fold": spearman(pred, y),
                                          "perm_p": float((null >= r2).mean()), "perm_null_median": float(np.median(null)),
                                          "perm_null_p95": float(np.quantile(null, 0.95)),
                                          "r2_in_sample_ridge1": float(1 - ((full(M) - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                                          "std_coef_ridge1": {n: float(x) for n, x in zip(names, coef)}}
    units = sorted(set(cl.values()))
    idx_of = {u: [i for i, c in enumerate(cids) if cl[c] == u] for u in units}
    uni, bins = {}, {}
    for t in TRAITS + ("alone_audit",):
        x = np.array([tr[c][t] for c in cids])
        uni[t], bins[t] = {}, {}
        for oname, y in out_y.items():
            draws = []
            for _ in range(N_BOOT):
                ii = np.concatenate([idx_of[u] for u in rng.choice(units, len(units))])
                draws.append(spearman(x[ii], y[ii]))
            lo, hi = np.quantile(draws, [0.025, 0.975])
            uni[t][oname] = {"spearman": spearman(x, y), "ci95": [float(lo), float(hi)]}
        if t == "copy_flag":
            groups = [("0", x == 0), ("1", x == 1)]
        else:
            q = np.quantile(x, [1 / 3, 2 / 3])
            groups = [("low", x <= q[0]), ("mid", (x > q[0]) & (x <= q[1])), ("high", x > q[1])]
        for g, m in groups:
            if m.any():
                bins[t][g] = {"n": int(m.sum()), "trait_mean": float(x[m].mean()),
                              **{f"{o}_mean": float(y[m].mean()) for o, y in out_y.items()},
                              **{f"{o}_se": float(y[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None for o, y in out_y.items()},
                              "used_share": float((out_y["weight"][m] > 0).mean()),
                              "wins": int((out_y["gain"][m] > 1e-4).sum()), "losses": int((out_y["gain"][m] < -1e-4).sum())}
    res = {"schema": "jope.h427.analysis/v1", "role": "analysis only; no verdict", "n": len(cids), "units": len(units),
           "outcomes": {o: {"mean": float(y.mean()), "sd": float(y.std(ddof=1))} for o, y in out_y.items()},
           "traits": {t: {"mean": float(np.mean([tr[c][t] for c in cids])), "sd": float(np.std([tr[c][t] for c in cids], ddof=1)),
                          "min": float(min(tr[c][t] for c in cids)), "max": float(max(tr[c][t] for c in cids))}
                      for t in TRAITS + ("alone_audit",)},
           "models": models, "univariate": uni, "bins": bins,
           "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BANK, APPLY, LIST, SCREEN)},
           "per_context": {c: {**tr[c], "weight": float(out_y["weight"][i]), "gain": float(out_y["gain"][i]), "cluster": cl[c]}
                           for i, c in enumerate(cids)}}
    print(f"== H427 n {res['n']} units {res['units']}  weight mean {res['outcomes']['weight']['mean']:.3f}  gain mean {res['outcomes']['gain']['mean']:+.4f}")
    for k, m in models.items():
        print(f"   model {k:18s} out-of-fold R2 {m['r2_out_of_fold']:+.3f} (shuffle median {m['perm_null_median']:+.3f}, 95th {m['perm_null_p95']:+.3f}, p {m['perm_p']:.3f}) "
              f"Spearman {m['spearman_out_of_fold']:+.2f}  coef " + " ".join(f"{n}:{v:+.3f}" for n, v in m["std_coef_ridge1"].items()))
    for t in TRAITS + ("alone_audit",):
        u = uni[t]
        print(f"   {t:14s} rho weight {u['weight']['spearman']:+.2f} [{u['weight']['ci95'][0]:+.2f}, {u['weight']['ci95'][1]:+.2f}]  "
              f"rho gain {u['gain']['spearman']:+.2f} [{u['gain']['ci95'][0]:+.2f}, {u['gain']['ci95'][1]:+.2f}]  bins "
              + " | ".join(f"{g} n{v['n']} w {v['weight_mean']:.3f} g {v['gain_mean']:+.4f} W/L {v['wins']}/{v['losses']}" for g, v in bins[t].items()))
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
