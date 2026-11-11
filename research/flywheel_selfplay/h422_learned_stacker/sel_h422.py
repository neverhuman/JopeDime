#!/usr/bin/env python3
"""H422 learned stacker: a winsorised row-wise combiner whose strength is learned across families from held-out error.

The stored design (h416.SXB) predicts a row by the weighted MEAN of its members' predictions (bootstrap-Caruana
weights w; every member contributes its fold-model bag and its refit, half weight each). The stacker keeps w and
first pulls every member's prediction of the row back to within k member-scales of the row's weighted median:
      pred(row) = sum_i w_i * clip(P_i(row), median(row) - k_c * s_i, median(row) + k_c * s_i)
  s_i     the member's own typical (root-mean-square) deviation from the blend on the SEARCH out-of-fold rows
  k_c     one number per context, from the grid KGRID; the last grid value is infinity = the stored design exactly
  k_c = KGRID[clip(round(theta0 + sum_j theta_j * z_cj), 0, last)], z = five SEARCH-side features, standardised on
        the fitting contexts
No AUDIT target and no statistic over AUDIT rows enters a prediction: median and clipping are row by row, scales and
features come from SEARCH rows. theta is fitted on DESIGN contexts by maximising the family mean of
score = clipped AUDIT gain over the stored design + MU * tanh(gain / TAU); model class and penalty are chosen by an
inner family-grouped cross-validation; the TRAIN-side estimate is an outer family-grouped cross-fit (every context
is scored by a model that never saw its family).

  sel_h422.py fit     --cache DIR [DIR ...] --out DIR                             design contexts only
  sel_h422.py apply   --cache DIR --frozen FILE --frozen-sha256 HEX --out FILE    the frozen model on DEV
  sel_h422.py readout --apply FILE --out FILE                                     tests fixed in PREREG section 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CLIP = 0.5
MU, TAU = 0.05, 1e-3
KGRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0, float("inf"))
J = len(KGRID)
FEATS = ("oof_u", "log_eff_members", "robust_gain_search", "log_disp_over_resid", "search_opt_blend")
OUTER_SALT, INNER_SALT, K_OUTER, K_INNER = "H422-fold", "H422-inner", 5, 5
ETAS = (1e-4, 1e-3, 1e-2)
T0_GRID = np.arange(0.0, J - 1 + 1e-9, 0.5)
TJ_GRID = np.arange(-6.0, 6.0 + 1e-9, 0.25)
SWEEPS = 40
BOOT_SEED, N_BOOT = 422150, 10_000
MIN_GAIN = 0.010
LAM0_TOL = 1e-6
ARMS = ("B", "S", "S1", "A", "SA")
DESIGN_COHORTS = {"DL-Train", "FIT-SEL"}
APPLY_COHORT = "DEV"   # no command-line switch; smoke_h422.py overrides it in-process to exercise the code on FIT-SEL


# ----------------------------------------------------------------------------------------------------------------------
# per-context quantities
# ----------------------------------------------------------------------------------------------------------------------
def wquant(P: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Row-wise weighted quantile: the smallest value whose cumulative weight reaches q."""
    o = np.argsort(P, axis=1, kind="stable")
    cw = np.cumsum(w[o], axis=1)
    k = (cw >= q - 1e-12).argmax(axis=1)
    r = np.arange(len(P))
    return P[r, o[r, k]]


def _u(pred, y, b) -> float:
    return float(-np.log(max(float(np.sqrt(np.mean((pred - y) ** 2))) / float(b), 1e-12)))


def context_stats(c: dict, w: np.ndarray, with_audit_targets: bool = True) -> dict:
    """Features (SEARCH rows only) and the AUDIT predictions for every k of KGRID, of one context under weights w."""
    if len(w) < c["Q"].shape[1]:   # weights of the first m0 columns; the appended columns get 0
        w = np.concatenate([w, np.zeros(c["Q"].shape[1] - len(w))])
    s = w > 0
    ws = w[s] / w[s].sum()
    Qs, y = c["Q"][:, s], c["y"]
    ms, md = Qs @ ws, wquant(Qs, ws, 0.5)
    sd = float(y.std())
    resid = float(np.sqrt(np.mean((ms - y) ** 2)))
    disp = float(np.sqrt(np.mean(((Qs - ms[:, None]) ** 2) @ ws)))
    f = [min(_u(ms, y, sd), 6.0), float(-np.log(np.sum(ws ** 2))),
         float(np.clip(_u(0.5 * ms + 0.5 * md, y, sd) - _u(ms, y, sd), -CLIP, CLIP)),
         float(np.clip(np.log((disp + 1e-300) / (resid + 1e-300)), -6.0, 3.0)), 0.0]
    ds = md - ms   # the mean / median blend that would have been best on the SEARCH out-of-fold rows
    if float(np.mean(ds * ds)) > 0:
        f[4] = float(np.clip(-np.mean((ms - y) * ds) / np.mean(ds * ds), 0.0, 1.0))
    scale = np.sqrt(((Qs - ms[:, None]) ** 2).mean(0)) + 1e-300   # per member, SEARCH out-of-fold rows
    cols = np.concatenate([c["BAG"][:, s], c["REF"][:, s]], axis=1)
    w2, s2 = np.concatenate([ws, ws]) / 2.0, np.concatenate([scale, scale])
    med = wquant(cols, w2, 0.5)[:, None]
    P = np.stack([np.clip(cols, med - k * s2, med + k * s2) @ w2 for k in KGRID[:-1]] + [cols @ w2])
    out = {"f": f, "P": P}
    if with_audit_targets:   # scoring only
        out["U"] = [_u(p, c["ya"], c["ba"]) for p in P]
    return out


def load_cache(dirs, allowed: set):
    ctx = []
    for d in dirs:
        info = json.loads(next(Path(d).glob("cache_*.json")).read_text())
        if not set(info["cohorts"]) <= allowed:
            raise SystemExit(f"{d}: cohorts {info['cohorts']} are not allowed in this step ({sorted(allowed)})")
        for p in sorted(Path(d).glob("*.npz")):
            z = np.load(p, allow_pickle=False)
            c = {k: z[k] for k in z.files}
            for k in ("Q", "BAG", "REF"):
                c[k] = c[k].astype(np.float64)
            c["cid"], c["panel"], c["family"] = p.stem, info["tag"], str(c["family"])
            ctx.append(c)
    return ctx


# ----------------------------------------------------------------------------------------------------------------------
# model: fit, selection, cross-fit.  G[c, j] = u(KGRID[j]) - u(stored design) of context c
# ----------------------------------------------------------------------------------------------------------------------
def fold_of(family: str, salt: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\0{family}".encode()).digest()[:8], "big") % k


def score(g: np.ndarray) -> np.ndarray:
    """Per-context objective: the clipped gain plus MU times a smooth sign of the gain (win +1, loss -1)."""
    return np.clip(g, -CLIP, CLIP) + MU * np.tanh(g / TAU)


def idx_of(model: dict, F: np.ndarray) -> np.ndarray:
    if model["kind"] == "M0":
        return np.full(len(F), J - 1)
    th = np.asarray(model["theta"], float)
    Z = (F - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return np.clip(np.rint(th[0] + Z @ th[1:]), 0, J - 1).astype(int)


def take(G: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return G[np.arange(len(G)), idx] if idx.ndim == 1 else G[np.arange(len(G))[:, None], idx]


def fit_model(kind: str, eta: float, F, G) -> dict:
    mu, sd = F.mean(0), F.std(0) + 1e-9
    m = score(G).mean(0)
    best = np.flatnonzero(m >= m.max() - 1e-12)
    th = np.zeros(1 + F.shape[1])
    th[0] = float(best[-1])   # ties: the grid value closest to the stored design
    if kind == "MG":
        Z = (F - mu) / sd

        def obj(T):   # T: (C, p) candidate parameter rows
            idx = np.clip(np.rint(T[:, :1].T + Z @ T[:, 1:].T), 0, J - 1).astype(int)
            return score(take(G, idx)).mean(0) - eta * (T[:, 1:] ** 2).sum(1)

        for _ in range(SWEEPS):
            changed = False
            for k in range(len(th)):
                grid = T0_GRID if k == 0 else TJ_GRID
                T = np.repeat(th[None, :], len(grid), 0)
                T[:, k] = grid
                o = obj(T)
                b = np.flatnonzero(o >= o.max() - 1e-12)
                new = float(grid[b[np.argmin(np.abs(grid[b] - th[k]))]])   # ties: the value closest to now
                if new != th[k]:
                    th[k], changed = new, True
            if not changed:
                break
    return {"kind": kind, "eta": eta, "theta": [float(x) for x in th], "mu": [float(x) for x in mu],
            "sd": [float(x) for x in sd]}


def select_and_fit(F, G, fam) -> dict:
    """Inner family-grouped CV over {M1, MG(eta)} on the score: a gated model is taken only if it beats the constant
    model by more than one standard error of the paired difference; if the chosen model's inner out-of-fold mean
    score is not above 0 the stored design is kept (M0)."""
    inner = np.array([fold_of(f, INNER_SALT, K_INNER) for f in fam])
    cands = [("M1", 0.0)] + [("MG", e) for e in ETAS]
    oof = np.zeros((len(cands), len(F)))
    for k in range(K_INNER):
        tr, te = inner != k, inner == k
        if not te.any() or not tr.any():
            continue
        for i, (kind, eta) in enumerate(cands):
            m = fit_model(kind, eta, F[tr], G[tr])
            oof[i, te] = score(take(G[te], idx_of(m, F[te])))
    means = oof.mean(1)
    pick = 0
    g = 1 + int(np.argmax(means[1:]))
    d = oof[g] - oof[0]
    if d.mean() > d.std(ddof=1) / np.sqrt(len(d)):
        pick = g
    scores = {f"{k}:{e}": float(x) for (k, e), x in zip(cands, means)}
    if means[pick] <= 0:
        return {"kind": "M0", "eta": 0.0, "theta": [float(J - 1)] + [0.0] * F.shape[1], "mu": [0.0] * F.shape[1],
                "sd": [1.0] * F.shape[1], "inner_scores": scores}
    m = fit_model(*cands[pick], F, G)
    m["inner_scores"] = scores
    return m


def _stats():
    import train_h416 as T16

    _, H = T16._stats_mod()
    return T16, H


def _wtl(g) -> list:
    return [int((g > 1e-4).sum()), int((np.abs(g) <= 1e-4).sum()), int((g < -1e-4).sum())]


def cmd_fit(a) -> None:
    t0 = time.process_time()
    ctx = load_cache(a.cache, DESIGN_COHORTS)
    fam = [c["family"] for c in ctx]
    if len(set(fam)) != len(fam):
        raise SystemExit("one design context per family expected")
    st = [context_stats(c, c["w_sxb"]) for c in ctx]
    F = np.array([s["f"] for s in st])
    U = np.array([s["U"] for s in st])
    G = U - U[:, -1:]
    chk = max(abs(float(c["u_recomputed"]) - x) for c, x in zip(ctx, U[:, -1]))
    if chk > LAM0_TOL:   # summation order differs from the stored formula; near-exact contexts reach u of 15
        raise SystemExit(f"k = infinity does not reproduce the stored design (max difference {chk})")
    outer = np.array([fold_of(f, OUTER_SALT, K_OUTER) for f in fam])
    idx, folds = np.zeros(len(ctx), int), []
    for k in range(K_OUTER):
        tr, te = outer != k, outer == k
        m = select_and_fit(F[tr], G[tr], [f for f, t in zip(fam, tr) if t])
        idx[te] = idx_of(m, F[te])
        folds.append({"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum()), "model": m})
    g = take(G, idx)
    T16, H = _stats()
    cid = [c["cid"] for c in ctx]
    fam_of = dict(zip(cid, fam))
    pt = H.primary_test(dict(zip(cid, U[:, -1] + g)), dict(zip(cid, U[:, -1])), fam_of)
    gc = np.clip(g, -CLIP, CLIP)
    pan = np.array([c["panel"] for c in ctx])
    gate = {"clipped_mean": float(pt["T_clipped_family_mean"]), "signflip_p": float(pt["signflip_p"]),
            "mean": float(g.mean()), "median": float(np.median(g)), "wtl": _wtl(g),
            "by_panel": {p: float(gc[pan == p].mean()) for p in sorted(set(pan))},
            "stored_design_kept_in": int((idx == J - 1).sum())}
    gate["pass"] = bool(gate["clipped_mean"] > 0 and gate["signflip_p"] < 0.05)
    final = select_and_fit(F, G, fam)
    const = fit_model("M1", 0.0, F, G)
    gf, g1 = take(G, idx_of(final, F)), take(G, idx_of(const, F))
    frozen = {"schema": "jope.h422.frozen/v1", "features": list(FEATS), "kgrid": [str(k) for k in KGRID], "clip": CLIP,
              "mu": MU, "tau": TAU, "final": final, "const": const, "design_contexts": len(ctx),
              "panels": sorted(set(pan.tolist())),
              "cids_sha256": hashlib.sha256("\n".join(sorted(cid)).encode()).hexdigest(),
              "in_sample": {"final_clipped_mean": float(np.clip(gf, -CLIP, CLIP).mean()), "final_wtl": _wtl(gf),
                            "const_clipped_mean": float(np.clip(g1, -CLIP, CLIP).mean()), "const_wtl": _wtl(g1),
                            "const_k": str(KGRID[int(const["theta"][0])])}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv = {"schema": "jope.h422.crossfit/v1", "n": len(ctx), "gate": gate, "folds": folds,
          "u_sxb_mean": float(U[:, -1].mean()), "lam0_max_abs_diff": float(chk), "cpu_s": time.process_time() - t0,
          "grid_clipped_mean": {str(k): float(np.clip(G[:, j], -CLIP, CLIP).mean()) for j, k in enumerate(KGRID)},
          "grid_wtl": {str(k): _wtl(G[:, j]) for j, k in enumerate(KGRID)},
          "per_context": {c: {"family": f, "fold": int(o), "k": str(KGRID[int(i)]), "gain": float(x)}
                          for c, f, o, i, x in zip(cid, fam, outer, idx, g)}}
    with open(out / "CROSSFIT.json", "x") as f:
        json.dump(cv, f, sort_keys=True)
    with open(out / "FROZEN.json", "x") as f:
        json.dump(frozen, f, sort_keys=True, indent=1)
    print(json.dumps({"gate": gate, "final": {k: final[k] for k in ("kind", "eta", "theta")},
                      "in_sample": frozen["in_sample"],
                      "fold_models": [(x["model"]["kind"], x["model"]["eta"], x["model"]["theta"][0]) for x in folds]}))


def cmd_apply(a) -> None:
    t0 = time.process_time()
    raw = Path(a.frozen).read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.frozen_sha256:
        raise SystemExit("the frozen model file is not the one named by --frozen-sha256")
    fr = json.loads(raw)
    if fr["kgrid"] != [str(k) for k in KGRID] or fr["features"] != list(FEATS):
        raise SystemExit("the frozen model was fitted by other code")
    ctx = load_cache([a.cache], {APPLY_COHORT})
    import apex_core_meta as AM

    rows = AM.rows()
    if {rows[c["cid"]]["cohort"] for c in ctx} != {APPLY_COHORT}:
        raise SystemExit(f"the apply step runs on {APPLY_COHORT} contexts")
    per, worst, label_free = {}, 0.0, True
    for c in ctx:
        if not np.isfinite(float(c["u_record"])):
            raise SystemExit(f"{c['cid']}: no stored baseline value")
        worst = max(worst, abs(float(c["u_record"]) - float(c["u_recomputed"])))
        o = {"B": float(c["u_record"]), "family": c["family"]}
        for arm, w in (("S", c["w_sxb"]), ("SA", c["w_union"])):
            s = context_stats(c, w)
            F = np.array([s["f"]])
            i = int(idx_of(fr["final"], F)[0])
            o[arm], o[f"k_{arm}"], o[f"f_{arm}"] = float(s["U"][i]), str(KGRID[i]), s["f"]
            if arm == "S":
                o["S1"], o["stored_path_u"] = float(s["U"][int(idx_of(fr["const"], F)[0])]), float(s["U"][-1])
            else:
                o["A"] = float(s["U"][-1])
            c2 = dict(c)
            c2["ya"] = np.zeros_like(c["ya"])   # the prediction must not depend on the AUDIT targets
            s2 = context_stats(c2, w, with_audit_targets=False)
            label_free &= bool(np.array_equal(s2["P"], s["P"]) and s2["f"] == s["f"])
        o["members_S"], o["members_SA"] = int((c["w_sxb"] > 0).sum()), int((c["w_union"] > 0).sum())
        o["columns"], o["m0"] = int(c["Q"].shape[1]), int(c["m0"])
        per[c["cid"]] = o
    res = {"schema": "jope.h422.apply/v1", "n": len(ctx), "cohort": APPLY_COHORT, "frozen_sha256": a.frozen_sha256,
           "replay_max_abs_diff": float(worst),
           "lam0_max_abs_diff": float(max(abs(o["stored_path_u"] - o["B"]) for o in per.values())),
           "label_free": bool(label_free), "cpu_s": time.process_time() - t0, "per_context": per}
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("n", "replay_max_abs_diff", "lam0_max_abs_diff", "label_free", "cpu_s")}))


def cmd_readout(a) -> None:
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    fam_of = {c: per[c]["family"] for c in cids}
    if len(set(fam_of.values())) != len(cids):
        raise SystemExit("one context per family expected")
    T16, H = _stats()
    vals = {x: {c: per[c][x] for c in cids} for x in ARMS}
    pairs = (("S", "B"), ("S1", "B"), ("A", "B"), ("SA", "B"), ("SA", "S"), ("SA", "A"), ("S", "S1"))
    res = {"schema": "jope.h422.readout/v1", "n": len(cids), "cohort": ap["cohort"],
           "baseline": "stored H420 TBOTH values (B* of H421), not recomputed",
           "family_mean_u": {x: float(H.family_mean(vals[x], fam_of)) for x in ARMS},
           "pairs": {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], fam_of) for x, y in pairs},
           "k_counts": {x: {k: int(sum(per[c][f"k_{x}"] == k for c in cids)) for k in sorted({per[c][f"k_{x}"] for c in cids})}
                        for x in ("S", "SA")}}
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "B", BOOT_SEED, N_BOOT)
    res["elo_anchor_B_1500"] = elo
    res["delta_elo"] = {f"{x}_vs_{y}": T16.delta_elo(np, arms, point, boot, x, y) for x, y in pairs}
    prim, de = res["pairs"]["S_vs_B"], res["delta_elo"]["S_vs_B"]
    cond = {"gain_at_least_0.010": bool(prim["clipped_mean"] >= MIN_GAIN), "primary_test_pass": bool(prim["pass"]),
            "p_below_0.05_one_sided": bool(prim["signflip_p"] < 0.05), "elo_ci_excludes_0": bool(de["ci95"][0] > 0)}
    res["conditions"] = cond
    if prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        res["verdict"] = "SEL_HARM"
    elif cond["gain_at_least_0.010"] and cond["primary_test_pass"] and cond["elo_ci_excludes_0"]:
        res["verdict"] = "SEL_CONFIRMED"
    else:
        res["verdict"] = "SEL_NULL"
    print(f"== H422 {res['cohort']} n {res['n']} baseline: {res['baseline']}")
    print("   mean u: " + "  ".join(f"{x} {res['family_mean_u'][x]:.4f}" for x in ARMS) + f"   k of S: {res['k_counts']['S']}")
    for k, s in res["pairs"].items():
        e = res["delta_elo"][k]
        print(f"   {k:10s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']} dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]")
    print(f"   conditions {cond} verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("apply")
    p.add_argument("--cache", required=True)
    p.add_argument("--frozen", required=True)
    p.add_argument("--frozen-sha256", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_readout)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
