#!/usr/bin/env python3
"""H419 addendum 5, section 3: references for the monitor that are stronger than ridge, from stored H417 matrices.

The monitor's task is: context = SEARCH folds 1-4, query = SEARCH fold 0 of a FIT-SEL context. The H417 deploy stored,
for the same rows and the same folds, the out-of-fold predictions Q of each context's stack candidates (about 64 GP
programs): Q's fold-0 rows were predicted by models fitted on folds 1-4. So, with no worker run:

  gp_best_cv     the program with the lowest out-of-fold RMSE on folds 1-4, scored on fold 0
  gp_kcar        Caruana selection (meta_h416.kcar, 100 steps = KFB's rule) fitted on the fold 1-4 rows of Q,
                 scored on fold 0
  gp_oracle      the best program on fold 0 itself (in-sample upper bound for a single program)

  icl_refs.py build --val fitsel_search_val.npz --npz-dir <pulled>/h416_BASE_npz --meta-dir <stage code> --out refs.json
  icl_refs.py compare --refs refs.json --diag diag.json [...]     win rate / median of each diag recipe against each

SEARCH rows of FIT-SEL only (TRAIN-side design cohort). No AUDIT value is read.
"""
import argparse
import json
import math
import sys

import numpy as np


def u_of(pred, yq, ym):
    r = float(np.sqrt(np.mean((pred - yq) ** 2)))
    r0 = max(float(np.sqrt(np.mean((ym - yq) ** 2))), 1e-30)
    return max(-2.0, min(6.0, -math.log(max(r / r0, 1e-30))))


def build(a):
    sys.path.insert(0, a.meta_dir)
    import meta_h416 as M

    v = np.load(a.val, allow_pickle=False)
    if set(v["cohort"].tolist()) != {"FIT-SEL"}:
        raise SystemExit("the monitor bank holds a context outside FIT-SEL")
    rows, lo, preds = [], 0, {"kcar": [], "best": []}
    for i, (cid, (r, c)) in enumerate(zip(v["cids"].tolist(), v["shape"].tolist())):
        y, lab = v["y64"][lo:lo + r], v["lab"][lo:lo + r].astype(np.int64)
        lo += r
        z = np.load(f"{a.npz_dir}/{cid}.npz", allow_pickle=False)
        if z["y"].shape[0] != r or not np.array_equal(z["lab"].astype(np.int64), lab) or not np.allclose(z["y"], y):
            raise SystemExit(f"{cid}: the stored matrix does not match the monitor rows")
        Q = z["Q"]
        ok = np.all(np.isfinite(Q), 0)
        Q = Q[:, ok]
        ctx, qry = lab != 0, lab == 0
        ym = float(y[ctx].mean())
        us = [u_of(Q[qry, j], y[qry], ym) for j in range(Q.shape[1])]
        cv = np.sqrt(np.mean((Q[ctx] - y[ctx, None]) ** 2, 0))
        cnt = M.kcar(Q[ctx], y[ctx], 100)
        rows.append({"i": i, "cid": cid, "n_prog": int(Q.shape[1]), "gp_best_cv": us[int(np.argmin(cv))],
                     "gp_kcar": u_of(Q[qry] @ (cnt / cnt.sum()), y[qry], ym), "gp_oracle": float(max(us))})
        preds["kcar"].append(Q @ (cnt / cnt.sum())); preds["best"].append(Q[:, int(np.argmin(cv))])
    out = {"schema": "jope.h419.monitor-refs/v1", "n": len(rows), "rows": rows}
    with open(a.out, "x") as f:
        json.dump(out, f)
    # the two references' predictions for every SEARCH row, in the monitor bank's row order (context rows are
    # out-of-fold predictions; weights / choice use fold 1-4 rows only): bases for the stack-residual readout
    np.savez(a.out[:-5] + "_pred.npz", cids=v["cids"], shape=v["shape"],
             **{k: np.concatenate(x) for k, x in preds.items()})
    for k in ("gp_best_cv", "gp_kcar", "gp_oracle"):
        xs = sorted(r[k] for r in rows)
        print(k, "mean", round(float(np.mean(xs)), 4), "median", round(xs[len(xs) // 2], 4))


def compare(a):
    refs = {r["i"]: r for r in json.load(open(a.refs))["rows"]}
    for path in a.diag:
        d = json.load(open(path))
        print(f"== {path}  step {d['state']['step']}  h {d['state']['elapsed'] / 3600:.2f}")
        names = list(d["rows"][0]["u"])
        print("   reference medians: ridge", round(sorted(r["u_ridge"] for r in d["rows"])[len(d["rows"]) // 2], 3),
              " ".join(f"{k} {sorted(refs[i][k] for i in refs)[len(refs) // 2]:.3f}" for k in ("gp_best_cv", "gp_kcar", "gp_oracle")))
        for v in names:
            us = [r["u"][v] for r in d["rows"]]
            line = [f"median {sorted(us)[len(us) // 2]:.3f}", f"mean {np.mean(us):.3f}",
                    f"win vs ridge {np.mean([r['u'][v] > r['u_ridge'] for r in d['rows']]):.3f}"]
            for k in ("gp_best_cv", "gp_kcar", "gp_oracle"):
                line.append(f"vs {k} {np.mean([r['u'][v] > refs[r['i']][k] for r in d['rows']]):.3f}")
            print(f"   {v:16s} " + "  ".join(line))


def stack(a):
    """Does the stack use in-context columns, and does it help on a held-out fold? Caruana (kcar, 100 steps) is fitted
    on the fold 1-4 rows of Q with and without the model's out-of-fold columns and scored on fold 0."""
    sys.path.insert(0, a.meta_dir)
    import meta_h416 as M

    v, o = np.load(a.val, allow_pickle=False), np.load(a.oof, allow_pickle=False)
    if o["cids"].tolist() != v["cids"].tolist():
        raise SystemExit("the columns are not for this monitor bank")
    names = [k for k in o.files if k not in ("cids", "shape")]
    sets = [[n] for n in names] + ([names] if len(names) > 1 else [])
    res, lo = {"+".join(s_): [] for s_ in sets}, 0
    for cid, (r, c) in zip(v["cids"].tolist(), v["shape"].tolist()):
        y, lab = v["y64"][lo:lo + r], v["lab"][lo:lo + r].astype(np.int64)
        z = np.load(f"{a.npz_dir}/{cid}.npz", allow_pickle=False)
        Q = z["Q"][:, np.all(np.isfinite(z["Q"]), 0)]
        ctx, qry = lab != 0, lab == 0
        ym = float(y[ctx].mean())
        c0 = M.kcar(Q[ctx], y[ctx], 100)
        u0 = u_of(Q[qry] @ (c0 / c0.sum()), y[qry], ym)
        for s_ in sets:
            C = np.column_stack([o[n][lo:lo + r] for n in s_])
            if not np.all(np.isfinite(C)):
                res["+".join(s_)].append((u0, u0, 0.0)); continue
            P = np.column_stack([Q, C])
            c1 = M.kcar(P[ctx], y[ctx], 100)
            w = c1 / c1.sum()
            res["+".join(s_)].append((u0, u_of(P[qry] @ w, y[qry], ym), float(w[Q.shape[1]:].sum())))
        lo += r
    out = {}
    for k, rows in res.items():
        d = np.array([b - a_ for a_, b, _ in rows]); w = np.array([x[2] for x in rows])
        out[k] = {"n": len(rows), "win": float(np.mean(d > 0)), "loss": float(np.mean(d < 0)), "tie": float(np.mean(d == 0)),
                  "mean_gain": float(d.mean()), "clipped_mean_gain": float(np.clip(d, -0.5, 0.5).mean()),
                  "median_gain": float(np.median(d)), "used": float(np.mean(w > 0)), "mean_weight": float(w.mean()),
                  "mean_u_without": float(np.mean([x[0] for x in rows])), "mean_u_with": float(np.mean([x[1] for x in rows]))}
        print(k, json.dumps({kk: round(vv, 4) for kk, vv in out[k].items()}))
    if a.out:
        with open(a.out, "x") as f:
            json.dump({"schema": "jope.h419.monitor-stack/v1", "oof": a.oof, "result": out}, f)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--val", required=True); b.add_argument("--npz-dir", required=True)
    b.add_argument("--meta-dir", required=True); b.add_argument("--out", required=True); b.set_defaults(fn=build)
    c = sub.add_parser("compare")
    c.add_argument("--refs", required=True); c.add_argument("--diag", nargs="+", required=True); c.set_defaults(fn=compare)
    t = sub.add_parser("stack")
    t.add_argument("--val", required=True); t.add_argument("--npz-dir", required=True)
    t.add_argument("--meta-dir", required=True); t.add_argument("--oof", required=True); t.add_argument("--out")
    t.set_defaults(fn=stack)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
