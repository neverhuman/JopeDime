#!/usr/bin/env python3
"""H435: a prospective test of H434's two-number rule as a screen before full column inference. FIT-SEL only.

  screen_h435.py run     --run-dir DIR --tag TBOTH --screen FITSEL_SCREEN.json --val VAL_BANK.npz --oof OOF.npz
                         --h20 DIR --h430 DIR --out DIR [--shard k/n]
  screen_h435.py readout --dir DIR --screen FITSEL_SCREEN.json --h424-smoke FILE --h430-apply FILE --out FILE
  screen_h435.py selftest

The rule (frozen in H434, nothing refitted here): for a column and the stored blend with RMSE ratio r and error
overlap rho (uncentred correlation of out-of-fold errors), predicted weight w = (1 - rho r) / (1 + r^2 - 2 rho r),
clipped to [0, 1]; the screen KEEPS a table for full inference when w >= 0.10.
run, per FIT-SEL context and stored column: the rule's two numbers estimated on ONE FOLD of SEARCH rows (stored
fold label 0; what one screening pass would give), on half of that fold and on all rows; and the realised weight of
the column when the stored builder (meta_h416 BCAR) is fitted on the 64 stored candidates plus that column.
Columns (all computed earlier, none here): H419's hour-12 in-context column in its two readouts (106 tables),
H424's hour-20 smoke column (8 tables), and H430's five stored columns (another session's files, read only).
Only SEARCH-side arrays are read (Q, y, fold labels); no AUDIT array is loaded and no gain is computed.
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

WIDE, NEED, BOOT_SEED, N_BOOT = 30, 0.10, 435150, 10_000
SXB_KEY = "P3+LIB3.SXB"
PRIMARY = ("icl_h12_ridgeres", "icl_h12_plain")
H430 = ("key_direct", "key_gbm", "factor_glm", "gbm_plain", "key_noise")


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def cos(u, v) -> float:
    d = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / d) if d > 0 else 0.0


def w_two(rho: float, r: float) -> float:
    d = 1.0 + r * r - 2.0 * rho * r
    return float(np.clip((1.0 - rho * r) / d, 0.0, 1.0)) if d > 1e-12 else 0.5


def est(eb: np.ndarray, ec: np.ndarray, rows) -> dict:
    b, c = eb[rows], ec[rows]
    r, rho = rms(c) / rms(b), cos(c, b)
    return {"rows": int(len(b)), "overlap": rho, "ratio": r, "w_pred": w_two(rho, r)}


def one_column(bcar, Q, y, eb, lab, q, cid) -> dict:
    ec = q - y
    f0 = np.flatnonzero(lab == 0)
    return {"all": est(eb, ec, slice(None)), "fold0": est(eb, ec, f0), "half": est(eb, ec, f0[::2]),
            "w_real": float(bcar(np.column_stack([Q, q]), y, cid)[-1])}


def cmd_selftest(a) -> None:
    import cache_h422 as CA

    r = np.random.default_rng(3)
    y = r.standard_normal(600)
    Q = y[:, None] + r.standard_normal((600, 8)) * np.linspace(0.3, 0.9, 8)
    ws = np.array([0.4, 0.3, 0.3, 0, 0, 0, 0, 0.0])
    lab = np.arange(600) % 5
    for s in (0.25, 0.6, 1.5):
        o = one_column(CA.bcar, Q, y, Q @ ws - y, lab, y + s * r.standard_normal(600), "selftest")
        print("selftest scale", s, {k: (round(v["w_pred"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in o.items()})


def cmd_run(a) -> None:
    import cache_h422 as CA

    t0, c0 = time.time(), time.process_time()
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = sorted(cfg["cids"])
    scr = json.loads(Path(a.screen).read_text())
    if scr["queries"] != ["FIT-SEL"] or sorted(scr["per_context"]) != cids:
        raise SystemExit("the screen file is not the FIT-SEL rehearsal of exactly this panel")
    v = np.load(a.val, allow_pickle=False)
    o = np.load(a.oof, allow_pickle=False)
    if set(v["cohort"].tolist()) != {"FIT-SEL"} or not np.array_equal(v["cids"], o["cids"]) or not np.array_equal(v["shape"], o["shape"]):
        raise SystemExit("the bank is not FIT-SEL or does not match the stored out-of-fold file")
    off = np.concatenate([[0], np.cumsum(v["shape"][:, 0])])
    pos = {str(c): i for i, c in enumerate(v["cids"])}
    k, n = (int(x) for x in a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for cid in cids[k::n]:
        p = out / f"{cid}.json"
        if p.exists():
            continue
        t1 = time.process_time()
        sd = Path(a.run_dir) / "stages" / f"H416_{a.tag}"
        rp = sd / f"{cid}__confirm.jsonl"
        rec = CA.last_result(rp if rp.exists() else sd / f"{cid}__dev.jsonl")
        zp = Path(a.run_dir) / f"h416_{a.tag}_npz" / f"{cid}.npz"
        if hashlib.sha256(zp.read_bytes()).hexdigest() != rec["npz"]["sha256"]:
            raise SystemExit(f"{zp}: npz sha differs from its record")
        z = np.load(zp, allow_pickle=False)
        Q, y, lab = z["Q"].astype(np.float64), z["y"].astype(np.float64), z["lab"].astype(np.int64)   # SEARCH-side arrays only
        shas = [str(s) for s in z["shas"]]
        sp = rec["outputs"][SXB_KEY]["spec"]
        ws = np.zeros(len(shas))
        for s, x in zip(sp["members"], sp["w"]):
            ws[shas.index(s)] = x
        if abs(float(ws.sum()) - 1.0) > 1e-6:
            raise SystemExit(f"{cid}: stored weights do not sum to 1")
        eb = Q @ ws - y
        F = int(scr["per_context"][cid]["width"])
        ns, na = int(len(y)), int(rec["n_audit"])
        per_pass = (35.0 + 0.72 * F) / 1600.0
        r = {"cid": cid, "features": F, "wide": bool(F > WIDE), "n_search": ns, "n_audit": na, "fold0_rows": int((lab == 0).sum()),
             "cost_full_s": 6.0 * (ns + na) * per_pass, "cost_screen_s": ns * per_pass, "columns": {}, "skipped": {}}
        i = pos[cid]
        lo, hi = int(off[i]), int(off[i + 1])
        ok12 = bool(hi - lo == ns and np.array_equal(v["lab"][lo:hi].astype(np.int64), lab) and np.allclose(v["y64"][lo:hi], y))
        r["aligned_h12"] = ok12
        cols = {}
        if ok12:
            cols["icl_h12_ridgeres"], cols["icl_h12_plain"] = o["ridgeres"][lo:hi].astype(np.float64), o["full"][lo:hi].astype(np.float64)
        p20 = Path(a.h20) / f"{cid}.npz"
        if p20.exists():
            z20 = np.load(p20, allow_pickle=False)
            if bool(z20["ok"]):
                cols["icl_h20_ridgeres"] = z20["q_ridgeres"].astype(np.float64)
        p30 = Path(a.h430) / f"{cid}.npz"
        if p30.exists():
            z30 = np.load(p30, allow_pickle=False)
            if bool(z30["ok"]):
                for name in H430:
                    cols[f"h430_{name}"] = z30[f"q_{name}"].astype(np.float64)
        for name, q in cols.items():
            if len(q) != ns or not np.all(np.isfinite(q)):
                r["skipped"][name] = "length or non-finite"
                continue
            r["columns"][name] = one_column(CA.bcar, Q, y, eb, lab, q, cid)
        r["cpu_s"] = time.process_time() - t1
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, sort_keys=True))
        tmp.rename(p)
        done += 1
    st = {"schema": "jope.h435.run/v1", "shard": a.shard, "written": done, "threads": 1, "torch_loaded": "torch" in sys.modules,
          "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    (out / f"run_status_{int(t0)}_s{k}of{n}.status").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st))


def cmd_readout(a) -> None:
    import restate_h426 as R26

    rs = {r["cid"]: r for r in (json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("*.json")))}
    cids = sorted(rs)
    scr = json.loads(Path(a.screen).read_text())
    cl = R26.clusterings(scr, cids)["feature"]
    units = sorted(set(cl.values()))
    by_unit = {u: [c for c in cids if cl[c] == u] for u in units}
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(units), size=(N_BOOT, len(units)))

    def decisions(names, sample="fold0", thr=NEED):
        d = {}
        for c in cids:
            d[c] = [(float(rs[c]["columns"][nm][sample]["w_pred"]) >= thr, float(rs[c]["columns"][nm]["w_real"]) >= NEED,
                     rs[c]["cost_full_s"], rs[c]["cost_screen_s"]) for nm in names if nm in rs[c]["columns"]]
        return d

    def score(rows) -> dict:
        if not rows:
            return {"decisions": 0}
        p, t = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
        cf, cs = np.array([x[2] for x in rows]), np.array([x[3] for x in rows])
        n = len(rows)
        tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
        return {"decisions": n, "kept": int(p.sum()), "real_weight": int(t.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "accuracy": (tp + tn) / n, "fn_per_decision": fn / n, "miss_rate_of_real": (fn / (fn + tp)) if (fn + tp) else None,
                "saved_inference": float(1.0 - (cs.sum() + cf[p].sum()) / cf.sum()),
                "saved_inference_if_screen_pass_reused": float(1.0 - (cs[~p].sum() + cf[p].sum()) / cf.sum()),
                "saved_by_count": float(1.0 - p.mean())}

    def block(names, sample="fold0", thr=NEED, sel=None) -> dict:
        d = decisions(names, sample, thr)
        use = [c for c in cids if sel is None or c in sel]
        s = score([x for c in use for x in d[c]])
        if not s["decisions"]:
            return s
        bf, bs, ba = [], [], []
        for idx in draws[:2000] if sel is not None else draws:
            rows = [x for i in idx for c in by_unit[units[i]] if (sel is None or c in sel) for x in d[c]]
            if not rows:
                continue
            b = score(rows)
            bf.append(b["fn_per_decision"]); bs.append(b["saved_inference"]); ba.append(b["accuracy"])
        q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]
        s.update({"fn_per_decision_ci95": q(bf), "saved_inference_ci95": q(bs), "accuracy_ci95": q(ba), "tables": int(sum(bool(d[c]) for c in use))})
        return s

    prim = block(PRIMARY)
    res = {"schema": "jope.h435.readout/v1", "cohort": "FIT-SEL (TRAIN side)", "tables": len(cids), "units": len(units),
           "aligned_h12": int(sum(r["aligned_h12"] for r in rs.values())), "wide_tables": int(sum(r["wide"] for r in rs.values())),
           "primary_pooled_icl_h12": prim,
           "per_column": {nm: block((nm,)) for nm in PRIMARY + ("icl_h20_ridgeres",) + tuple(f"h430_{x}" for x in H430)},
           "h430_pooled": block(tuple(f"h430_{x}" for x in H430)),
           "variants_primary": {"keep_if_pred_ge_0.05": block(PRIMARY, thr=0.05), "estimate_from_all_rows": block(PRIMARY, sample="all"),
                                "estimate_from_half_fold": block(PRIMARY, sample="half")},
           "primary_by_width": {"wide": block(PRIMARY, sel={c for c in cids if rs[c]["wide"]}), "narrow": block(PRIMARY, sel={c for c in cids if not rs[c]["wide"]})}}
    fn_ok, save_ok = prim["fn_per_decision"] <= 0.05, prim["saved_inference"] >= 0.40
    res["success"] = {"fn_at_most_1_in_20": bool(fn_ok), "saved_at_least_40pct": bool(save_ok)}
    res["verdict"] = "SCREEN_PASSES" if (fn_ok and save_ok) else "SCREEN_FAILS"

    # the plain readout against the residual readout on wide FIT-SEL tables (reported; a pre-declared flag for H434)
    def unit_stat(names_val, sel) -> dict:
        g = {}
        for c in sel:
            if names_val(c) is not None:
                g.setdefault(cl[c], []).append(names_val(c))
        u = np.array([np.mean(x) for _, x in sorted(g.items())])
        idx = rng.integers(0, len(u), size=(N_BOOT, len(u)))
        return {"units": len(u), "median": float(np.median(u)), "median_ci95": [float(x) for x in np.quantile(np.median(u[idx], axis=1), [0.025, 0.975])],
                "mean": float(u.mean()), "mean_ci95": [float(x) for x in np.quantile(u[idx].mean(1), [0.025, 0.975])]}

    side = {}
    for grp, sel in (("wide", [c for c in cids if rs[c]["wide"]]), ("narrow", [c for c in cids if not rs[c]["wide"]])):
        col = lambda c, nm, k: (rs[c]["columns"][nm]["all"][k] if nm in rs[c]["columns"] else None)
        side[grp] = {"tables": len(sel)}
        for nm in PRIMARY:
            side[grp][nm] = {"overlap": unit_stat(lambda c: col(c, nm, "overlap"), sel),
                             "distance_u": unit_stat(lambda c: (None if col(c, nm, "ratio") is None else float(np.log(col(c, nm, "ratio")))), sel),
                             "share_real_weight_ge_0.10": unit_stat(lambda c: (None if nm not in rs[c]["columns"] else float(rs[c]["columns"][nm]["w_real"] >= NEED)), sel),
                             "mean_real_weight": unit_stat(lambda c: (None if nm not in rs[c]["columns"] else rs[c]["columns"][nm]["w_real"]), sel)}
        side[grp]["plain_minus_ridgeres_distance_u"] = unit_stat(
            lambda c: (None if col(c, "icl_h12_plain", "ratio") is None else float(np.log(col(c, "icl_h12_plain", "ratio") / col(c, "icl_h12_ridgeres", "ratio")))), sel)
        side[grp]["plain_minus_ridgeres_overlap"] = unit_stat(
            lambda c: (None if col(c, "icl_h12_plain", "overlap") is None else col(c, "icl_h12_plain", "overlap") - col(c, "icl_h12_ridgeres", "overlap")), sel)
    res["plain_vs_ridgeres_hour12"] = side
    res["h434_flag"] = "REVISIT_H434" if side["wide"]["icl_h12_plain"]["share_real_weight_ge_0.10"]["mean"] >= 1.0 / 3.0 else "H434_NO_GO_STANDS"

    # replay of realised weights against the values other runs stored
    sm = json.loads(Path(a.h424_smoke).read_text())["per_context"]
    d20 = [abs(rs[c]["columns"]["icl_h20_ridgeres"]["w_real"] - float(sm[c]["w_added_I"][0])) for c in cids if "icl_h20_ridgeres" in rs[c]["columns"]]
    h3 = json.loads(Path(a.h430_apply).read_text())["per_context"]
    dg = [abs(rs[c]["columns"]["h430_gbm_plain"]["w_real"] - float(h3[c]["w_added_G1"][0])) for c in cids if "h430_gbm_plain" in rs[c]["columns"] and h3.get(c, {}).get("w_added_G1")]
    dn = [abs(rs[c]["columns"]["h430_key_noise"]["w_real"] - float(h3[c]["w_added_N1"][0])) for c in cids if "h430_key_noise" in rs[c]["columns"] and h3.get(c, {}).get("w_added_N1")]
    res["replay"] = {"h424_smoke_tables": len(d20), "h424_smoke_max_abs_diff": float(max(d20)) if d20 else None,
                     "h430_G1_tables": len(dg), "h430_G1_max_abs_diff": float(max(dg)) if dg else None,
                     "h430_N1_tables": len(dn), "h430_N1_max_abs_diff": float(max(dn)) if dn else None}
    res["cpu_s"] = float(sum(r["cpu_s"] for r in rs.values()))
    f = lambda s: (f"n {s['decisions']} kept {s['kept']} real {s['real_weight']} TP {s['tp']} FP {s['fp']} FN {s['fn']} TN {s['tn']} acc {s['accuracy']:.3f} "
                   f"FN/dec {s['fn_per_decision']:.3f} {[round(x, 3) for x in s['fn_per_decision_ci95']]} saved {s['saved_inference']:.3f} "
                   f"{[round(x, 3) for x in s['saved_inference_ci95']]} (reuse {s['saved_inference_if_screen_pass_reused']:.3f}, count {s['saved_by_count']:.3f})") if s.get("decisions") else "no decisions"
    print(f"== H435 FIT-SEL: {len(cids)} tables in {len(units)} clusters; hour-12 columns aligned on {res['aligned_h12']}; wide {res['wide_tables']}; {res['cpu_s']:.0f} CPU-s")
    print(f"   PRIMARY (two hour-12 in-context columns, one fold of rows): {f(prim)}")
    for nm, s in res["per_column"].items():
        print(f"   {nm:20s} {f(s)}")
    print(f"   h430 pooled          {f(res['h430_pooled'])}")
    for nm, s in res["variants_primary"].items():
        print(f"   variant {nm:24s} {f(s)}")
    for nm, s in res["primary_by_width"].items():
        print(f"   primary, {nm:7s} {f(s)}")
    for grp in ("wide", "narrow"):
        s = side[grp]
        g = lambda x: f"{x['median']:.3f} [{x['median_ci95'][0]:.3f}, {x['median_ci95'][1]:.3f}]"
        print(f"   {grp} ({s['tables']} tables): " + "; ".join(
            f"{nm}: overlap {g(s[nm]['overlap'])} distance {g(s[nm]['distance_u'])} weight>=0.10 share {s[nm]['share_real_weight_ge_0.10']['mean']:.3f} "
            f"{[round(x, 3) for x in s[nm]['share_real_weight_ge_0.10']['mean_ci95']]} mean weight {s[nm]['mean_real_weight']['mean']:.3f}" for nm in PRIMARY))
        print(f"      plain - ridgeres: distance {g(s['plain_minus_ridgeres_distance_u'])} overlap {g(s['plain_minus_ridgeres_overlap'])}")
    print(f"   replay {res['replay']}   h434 flag {res['h434_flag']}")
    print(f"   success {res['success']}  verdict {res['verdict']}")
    with open(a.out, "x") as fo:
        json.dump(res, fo, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for k in ("--run-dir", "--tag", "--screen", "--val", "--oof", "--h20", "--h430", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--shard", default="0/1")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("readout")
    for k in ("--dir", "--screen", "--h424-smoke", "--h430-apply", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_readout)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
