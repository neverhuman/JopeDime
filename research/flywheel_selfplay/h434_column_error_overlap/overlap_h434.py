#!/usr/bin/env python3
"""H434: how much do the in-context column's errors overlap with the stored candidates' errors? Stored values only.

  overlap_h434.py run --run-dir DIR --tag TBOTH --apply H424_DEV.apply.json --cols DIR --bank FILE --screen SCREEN_DEV.json
                      --need H432_DEV.need_per_table.jsonl --out FILE
  overlap_h434.py selftest

Per DEV context, from the stored out-of-fold matrix Q (64 candidates), the SEARCH targets y, the stored stack
weights of the record (no weight is refitted) and H424's stored column q:
  errors      e_c = q - y;  e_j = Q[:, j] - y;  e_b = Q @ w_stored - y (the stored blend);  e_best = the candidate
              with the lowest out-of-fold RMSE
  overlap     the uncentred correlation (cosine) of two error vectors; overlap^2 = the share of one error's squares
              that lies along the other
  bound       two predictors, blend and column, with RMSE ratio r = RMSE_c / RMSE_b and overlap rho: the
              error-minimising weight of the column is w* = (1 - rho r) / (1 + r^2 - 2 rho r); w* >= 0.10 exactly
              when rho <= rho_max(r) = (0.9 - 0.1 r^2) / (0.8 r). rho = 0 gives r <= 3, i.e. ln 3 = 1.10 u (H432's
              independent-error result, 1.09 u).
  blocked     the column is within ln 3 of the blend, yet rho > rho_max(r): overlap, not accuracy, denies it 10%
  floor       the lowest overlap with the stored blend (that candidate's own weight removed, the rest renormalised)
              among stored candidates at least as accurate as a column HANDICAP = 0.17 u weaker than today's; and
              today's column's own overlap if that is lower
  reachable   floor <= rho_max(r * exp(HANDICAP)): the most different predictor on record would earn 10% at the
              accuracy expected of a plain-readout column
Verdict on the 35 wide tables, clusters weighted equally (PREREG section 4):
  B_UNDECIDED  the bound classifies "real weight >= 0.10" correctly on fewer than 75% of the 115 tables
  B_GO         otherwise, blocked share > 1/2 and reachable share >= 1/3
  B_NO_GO      otherwise
Only SEARCH-side arrays and stored values are used; the gain table uses H424's stored AUDIT u values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WIDE, NEED, HANDICAP, BOOT_SEED, N_BOOT = 30, 0.10, 0.17, 434150, 10_000
SXB_KEY = "P3+LIB3.SXB"
LN3 = float(np.log(3.0))


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def cos(u, v) -> float:
    d = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / d) if d > 0 else 0.0


def rho_max(r: float) -> float:
    return (0.9 - 0.1 * r * r) / (0.8 * r)


def w_two(rho: float, r: float) -> float:
    d = 1.0 + r * r - 2.0 * rho * r
    return float(np.clip((1.0 - rho * r) / d, 0.0, 1.0)) if d > 1e-12 else 0.5


def measures(Q: np.ndarray, y: np.ndarray, q: np.ndarray, ws: np.ndarray) -> dict:
    E = Q - y[:, None]
    ec = q - y
    eb = Q @ ws - y
    rm = np.sqrt((E ** 2).mean(0))
    best = int(rm.argmin())
    rc, rb = rms(ec), rms(eb)
    rho_j = np.array([cos(ec, E[:, j]) for j in range(E.shape[1])])
    rho_b, r = cos(ec, eb), rc / rb
    cc, cb = ec - ec.mean(), eb - eb.mean()
    out = {"rmse_column": rc, "rmse_blend": rb, "rmse_best": float(rm[best]), "d_blend": float(np.log(r)), "d_best": float(np.log(rc / rm[best])),
           "blend_gain_over_best": float(np.log(rm[best] / rb)), "overlap_blend": rho_b, "pearson_blend": cos(cc, cb),
           "overlap_best": float(rho_j[best]), "overlap_cand_median": float(np.median(rho_j)), "overlap_cand_min": float(rho_j.min()),
           "overlap_cand_max": float(rho_j.max()), "overlap_cand_q25": float(np.quantile(rho_j, 0.25)), "overlap_cand_q75": float(np.quantile(rho_j, 0.75)),
           "nearest_is_best": bool(int(rho_j.argmax()) == best), "nearest_has_weight": bool(ws[int(rho_j.argmax())] > 0),
           "rho_max_now": rho_max(r), "w_two": w_two(rho_b, r), "within_ln3": bool(np.log(r) <= LN3),
           "need_d_best_theory": float(np.log(-4.0 * rho_b + np.sqrt(16.0 * rho_b ** 2 + 9.0)) - np.log(rm[best] / rb))}
    out["blocked"] = bool(out["within_ln3"] and rho_b > out["rho_max_now"])
    fl = []
    for h in (0.0, HANDICAP, 2 * HANDICAP):
        ok = [j for j in range(E.shape[1]) if rm[j] <= rc * np.exp(h)]
        pool = []
        for j in ok:
            w = ws.copy()
            w[j] = 0.0
            if w.sum() <= 0:
                continue
            pool.append(cos(E[:, j], Q @ (w / w.sum()) - y))
        floor = min(pool + [rho_b])
        fl.append({"handicap": h, "eligible": len(pool), "pool_floor": float(min(pool)) if pool else None,
                   "pool_q25": float(np.quantile(pool, 0.25)) if pool else None, "pool_median": float(np.median(pool)) if pool else None,
                   "floor": float(floor), "rho_needed": rho_max(r * float(np.exp(h))), "reachable": bool(floor <= rho_max(r * float(np.exp(h))))})
    out["floors"] = fl
    return out


def cmd_selftest(a) -> None:
    r = np.random.default_rng(2)
    y = r.standard_normal(500)
    shared = r.standard_normal(500)
    Q = y[:, None] + 0.5 * shared[:, None] + r.standard_normal((500, 6)) * np.linspace(0.3, 0.8, 6)
    ws = np.array([0.5, 0.3, 0.2, 0, 0, 0.0])
    m = measures(Q, y, y + 0.5 * shared + 0.6 * r.standard_normal(500), ws)
    print("selftest", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "floors"}, m["floors"][1])
    print("bound: rho_max(1.197) =", round(rho_max(1.197), 3), " r at rho 0 =", round(float(-0 + np.sqrt(9.0)), 3), " w_two(0, 3) =", round(w_two(0.0, 3.0), 3))


def cmd_run(a) -> None:
    import cache_h422 as CA
    import need_h432 as N32
    import restate_h426 as R26
    import train_h416 as T16

    _, H = T16._stats_mod()
    per = json.loads(Path(a.apply).read_text())["per_context"]
    cids = sorted(per)
    z = np.load(a.bank, allow_pickle=False)
    Fn = {str(c): int(s[1]) - 1 for c, s in zip(z["cids"], z["s_shape"])}
    if sorted(Fn) != cids:
        raise SystemExit("the bank does not hold exactly the contexts of the value file")
    cl = R26.clusterings(json.loads(Path(a.screen).read_text()), cids)["feature"]
    need = {r["cid"]: r for r in (json.loads(x) for x in Path(a.need).read_text().splitlines() if x.strip())}
    rows = {}
    for cid in cids:
        rec, st = CA.stored(a.run_dir, a.tag, cid)
        sp = rec["outputs"][SXB_KEY]["spec"]
        ws = np.zeros(len(st["shas"]))
        for s, x in zip(sp["members"], sp["w"]):
            ws[st["shas"].index(s)] = x
        q = np.load(Path(a.cols) / f"{cid}.npz", allow_pickle=False)["q_ridgeres"].astype(np.float64)
        if len(q) != len(st["y"]) or abs(float(ws.sum()) - 1.0) > 1e-6:
            raise SystemExit(f"{cid}: column or stored weights do not fit the stored matrix")
        m = measures(st["Q"], st["y"], q, ws)
        nd = N32.need_d(need[cid]["S"])
        m.update({"features": Fn[cid], "wide": bool(Fn[cid] > WIDE), "w_actual": float(per[cid]["w_added_I"][0]),
                  "gain_I_minus_B": float(per[cid]["I"] - per[cid]["B"]), "h432_need_d_S": float(nd) if np.isfinite(nd) else None,
                  "members": int((ws > 0).sum())})
        rows[cid] = m
    wide = [c for c in cids if rows[c]["wide"]]
    narrow = [c for c in cids if not rows[c]["wide"]]
    rng = np.random.default_rng(BOOT_SEED)

    def agg(val, sel) -> dict:
        g = {}
        for c in sel:
            g.setdefault(cl[c], []).append(float(val(rows[c])))
        u = np.array([np.mean(v) for _, v in sorted(g.items())])
        idx = rng.integers(0, len(u), size=(N_BOOT, len(u)))
        bm, bd = u[idx].mean(1), np.median(u[idx], axis=1)
        return {"units": len(u), "tables": len(sel), "mean": float(u.mean()), "mean_ci95": [float(x) for x in np.quantile(bm, [0.025, 0.975])],
                "median": float(np.median(u)), "median_ci95": [float(x) for x in np.quantile(bd, [0.025, 0.975])],
                "quartiles": [float(x) for x in np.quantile(u, [0.25, 0.75])], "min": float(u.min()), "max": float(u.max())}

    def block(sel) -> dict:
        o = {k: agg(lambda r, k=k: r[k], sel) for k in ("overlap_blend", "pearson_blend", "overlap_best", "overlap_cand_median", "overlap_cand_max",
                                                        "overlap_cand_min", "d_blend", "d_best", "blend_gain_over_best", "rho_max_now", "w_two", "w_actual")}
        o["share_blocked"] = agg(lambda r: float(r["blocked"]), sel)
        o["share_within_ln3"] = agg(lambda r: float(r["within_ln3"]), sel)
        o["share_nearest_is_best"] = agg(lambda r: float(r["nearest_is_best"]), sel)
        for i, name in enumerate(("handicap_0", "handicap_0.17", "handicap_0.34")):
            o[f"share_reachable_{name}"] = agg(lambda r, i=i: float(r["floors"][i]["reachable"]), sel)
            o[f"floor_{name}"] = agg(lambda r, i=i: r["floors"][i]["floor"], sel)
            o[f"rho_needed_{name}"] = agg(lambda r, i=i: r["floors"][i]["rho_needed"], sel)
        o["tables_blocked"] = int(sum(rows[c]["blocked"] for c in sel))
        o["tables_reachable"] = int(sum(rows[c]["floors"][1]["reachable"] for c in sel))
        o["tables_weight_ge_0.10"] = int(sum(rows[c]["w_actual"] >= NEED for c in sel))
        return o

    def bound_check(sel) -> dict:
        p = np.array([rows[c]["w_two"] >= NEED for c in sel])
        t = np.array([rows[c]["w_actual"] >= NEED for c in sel])
        from scipy.stats import spearmanr

        return {"tables": len(sel), "accuracy": float((p == t).mean()), "both": int((p & t).sum()), "bound_only": int((p & ~t).sum()),
                "real_only": int((~p & t).sum()), "neither": int((~p & ~t).sum()),
                "spearman_w_two_vs_real": float(spearmanr([rows[c]["w_two"] for c in sel], [rows[c]["w_actual"] for c in sel]).statistic)}

    def need_check(sel) -> dict:
        a_ = np.array([rows[c]["need_d_best_theory"] for c in sel])
        b_ = np.array([np.nan if rows[c]["h432_need_d_S"] is None else rows[c]["h432_need_d_S"] for c in sel])
        k = np.isfinite(a_) & np.isfinite(b_)
        from scipy.stats import spearmanr

        return {"tables": int(k.sum()), "median_theory": float(np.median(a_[k])), "median_h432": float(np.median(b_[k])),
                "median_abs_diff": float(np.median(np.abs(a_[k] - b_[k]))), "spearman": float(spearmanr(a_[k], b_[k]).statistic)}

    res = {"schema": "jope.h434.overlap/v1", "cohort": "DEV (development evidence, stored values)", "handicap_u": HANDICAP,
           "wide": block(wide), "narrow": block(narrow),
           "bound_check": {"all": bound_check(cids), "wide": bound_check(wide), "narrow": bound_check(narrow)},
           "need_distance_check_vs_h432_S": {"wide": need_check(wide), "narrow": need_check(narrow)}}
    bc = res["bound_check"]["all"]["accuracy"]
    c1, c2 = res["wide"]["share_blocked"]["mean"], res["wide"]["share_reachable_handicap_0.17"]["mean"]
    res["decision"] = {"bound_accuracy_all_tables": bc, "bound_ok": bool(bc >= 0.75), "share_blocked": c1, "share_reachable": c2,
                       "C1_blocked_share_above_half": bool(c1 > 0.5), "C2_reachable_share_at_least_third": bool(c2 >= 1.0 / 3.0)}
    res["verdict"] = "B_UNDECIDED" if bc < 0.75 else ("B_GO" if (c1 > 0.5 and c2 >= 1.0 / 3.0) else "B_NO_GO")

    # gain concentration with the largest table left out (H424's stored AUDIT u values)
    g = {c: rows[c]["gain_I_minus_B"] for c in cids}
    top = max(cids, key=lambda c: g[c])
    bins = (("<=10", 0, 10), ("11-20", 11, 20), ("21-30", 21, 30), ("31-100", 31, 100), (">100", 101, 10 ** 9))

    def by_width(sel) -> list:
        tot = sum(g[c] for c in sel)
        out = []
        for name, lo, hi in bins:
            s = [c for c in sel if lo <= rows[c]["features"] <= hi]
            out.append({"features": name, "tables": len(s), "gain_sum": float(sum(g[c] for c in s)), "share_of_total": float(sum(g[c] for c in s) / tot),
                        "mean_gain": float(np.mean([g[c] for c in s])) if s else None, "wins": int(sum(g[c] > 1e-4 for c in s)),
                        "losses": int(sum(g[c] < -1e-4 for c in s))})
        return out

    def test(sel, narrow_only: bool) -> dict:
        x = {c: (per[c]["I"] if (not narrow_only or not rows[c]["wide"]) else per[c]["B"]) for c in sel}
        s = T16.pair_stats(np, H, x, {c: per[c]["B"] for c in sel}, {c: cl[c] for c in sel})
        return {"units": len({cl[c] for c in sel}), **{k: s[k] for k in ("gain", "clipped_mean", "signflip_p", "lofo_min", "family_median", "wtl", "pass")}}

    rest = [c for c in cids if cl[c] != cl[top]]
    srt = sorted((g[c] for c in cids), reverse=True)
    res["gain_concentration"] = {"largest_table": {"cid": top, "features": rows[top]["features"], "gain": g[top], "share_of_total": g[top] / sum(g.values()),
                                                   "cluster_size": int(sum(cl[c] == cl[top] for c in cids))},
                                 "top5_share_of_total": float(sum(srt[:5]) / sum(srt)), "top5_gains": [float(x) for x in srt[:5]],
                                 "by_width_all": by_width(cids), "by_width_without_largest": by_width([c for c in cids if c != top]),
                                 "I_vs_B_all": test(cids, False), "I_vs_B_without_largest_cluster": test(rest, False),
                                 "N_vs_B_all": test(cids, True), "N_vs_B_without_largest_cluster": test(rest, True)}
    res["per_table"] = rows
    w, d = res["wide"], res["decision"]
    f = lambda s: f"{s['median']:.3f} [{s['median_ci95'][0]:.3f}, {s['median_ci95'][1]:.3f}] (mean {s['mean']:.3f})"
    print(f"== H434: {len(wide)} wide tables in {w['overlap_blend']['units']} clusters; {len(narrow)} narrow in {res['narrow']['overlap_blend']['units']}")
    for name, blk in (("wide", w), ("narrow", res["narrow"])):
        print(f"   {name}: overlap with blend {f(blk['overlap_blend'])}; with best candidate {f(blk['overlap_best'])}; "
              f"median candidate {f(blk['overlap_cand_median'])}; nearest candidate {f(blk['overlap_cand_max'])}")
        print(f"      distance to blend {f(blk['d_blend'])}; overlap allowed for 10% now {f(blk['rho_max_now'])}; blocked {blk['tables_blocked']}/{blk['overlap_blend']['tables']} "
              f"share {blk['share_blocked']['mean']:.3f} {[round(x, 3) for x in blk['share_blocked']['mean_ci95']]}; real weight >= 0.10 in {blk['tables_weight_ge_0.10']}")
        for hn in ("handicap_0", "handicap_0.17", "handicap_0.34"):
            print(f"      {hn}: floor {f(blk['floor_' + hn])}; needed {f(blk['rho_needed_' + hn])}; reachable share {blk['share_reachable_' + hn]['mean']:.3f} "
                  f"{[round(x, 3) for x in blk['share_reachable_' + hn]['mean_ci95']]}")
    print(f"   bound check: {res['bound_check']}")
    print(f"   need-distance check vs H432 S: {res['need_distance_check_vs_h432_S']}")
    print(f"   decision {d}  verdict {res['verdict']}")
    gc = res["gain_concentration"]
    print(f"   gain: largest table {gc['largest_table']}; top-5 share {gc['top5_share_of_total']:.3f}")
    for name in ("by_width_all", "by_width_without_largest"):
        print(f"      {name}: " + "; ".join(f"{b['features']} n {b['tables']} sum {b['gain_sum']:+.3f} share {b['share_of_total']:.2f} {b['wins']}/{b['losses']}" for b in gc[name]))
    for name in ("I_vs_B_all", "I_vs_B_without_largest_cluster", "N_vs_B_all", "N_vs_B_without_largest_cluster"):
        s = gc[name]
        print(f"      {name}: units {s['units']} clipped {s['clipped_mean']:+.4f} p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} W/T/L {s['wtl']} pass {s['pass']}")
    with open(a.out, "x") as fo:
        json.dump(res, fo, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for k in ("--run-dir", "--tag", "--apply", "--cols", "--bank", "--screen", "--need", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
