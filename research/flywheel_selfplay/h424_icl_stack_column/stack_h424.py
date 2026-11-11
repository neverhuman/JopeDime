#!/usr/bin/env python3
"""H424: the in-context model's column added to the stored stack (numpy; no GP engine, no table is opened here).

  stack_h424.py apply   --run-dir DIR --tag TBOTH --cohort DEV|FIT-SEL --bank FILE --cols DIR --out FILE [--procs N]
  stack_h424.py readout --apply FILE --out FILE

apply, per context of config_<TAG>.json (which must be exactly the bank's contexts and of the named cohort):
  the stored matrices (sha-checked against the H416 record; cache_h422.stored) are joined with the model's columns
  (cols_h424.py) after three alignment checks: SEARCH targets equal, fold labels equal, and the sha256 of the stored
  AUDIT targets equal to the one the bank computed inside the sandbox.
  B    the stored h416.SXB value of the record (not recomputed; the recomputation is the replay check)
  I    stored 64 candidates + the "ridgeres" column: the stored weight builder (meta_h416 BCAR) on 65 columns, the
       stored mix of bag and refit predictions                                                     [primary]
  also reported: the AUDIT value of the model column alone, and the weight the builder gives the added column.
  A context without a finite model column (or not reached before the inference budget ended) scores B in every arm
  and is counted.
readout: the test fixed in PREREG section 4 (verdict only for cohort DEV).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache_h422 as CA  # noqa: E402
import meta_h416 as M  # noqa: E402

SXB_KEY = "P3+LIB3.SXB"
CLIP = 0.5
BOOT_SEED, N_BOOT = 424150, 10_000
ARMS = ("B", "I")
_BANK = {}


def _u(pred, ya, ba) -> float:
    return float(M.utility(pred, ya, ba)[0])


def one(job) -> dict:
    run_dir, tag, cols_dir, cid, bank_row = job
    t0 = time.process_time()
    rec, a = CA.stored(run_dir, tag, cid)
    st = rec["outputs"][SXB_KEY]
    if st.get("invalid") or st.get("spec", {}).get("w") is None:
        raise SystemExit(f"{cid}: no stored baseline output")
    w0 = CA.bcar(a["Q"], a["y"], cid)
    ws = np.zeros(len(w0))
    for s, x in zip(st["spec"]["members"], st["spec"]["w"]):
        ws[a["shas"].index(s)] = x
    u_re = _u(0.5 * a["BAG"] @ w0 + 0.5 * a["REF"] @ w0, a["ya"], a["ba"])
    o = {"cid": cid, "family": rec["family_id"], "B": float(st["value"]), "w_diff": float(np.abs(ws - w0).max()),
         "u_diff": abs(u_re - float(st["value"])), "n_search": int(len(a["y"])), "n_audit": int(len(a["ya"]))}
    y64, lab, ya_sha = bank_row
    o["aligned"] = bool(np.array_equal(y64, a["y"]) and np.array_equal(lab.astype(np.int64), a["lab"].astype(np.int64))
                        and hashlib.sha256(np.ascontiguousarray(a["ya"], np.float64).tobytes()).hexdigest() == ya_sha)
    p = Path(cols_dir) / f"{cid}.npz"
    z = np.load(p, allow_pickle=False) if p.exists() else None
    o["column_ok"] = bool(z is not None and bool(z["ok"]) and o["aligned"]
                          and len(z["q_ridgeres"]) == len(a["y"]) and len(z["ref_ridgeres"]) == len(a["ya"]))
    if not o["column_ok"]:
        o.update({"I": o["B"], "cpu_s": time.process_time() - t0})
        return o
    for arm, names in (("I", ("ridgeres",)),):
        Q = np.column_stack([a["Q"]] + [z[f"q_{v}"] for v in names])
        BAG = np.column_stack([a["BAG"]] + [z[f"bag_{v}"] for v in names])
        REF = np.column_stack([a["REF"]] + [z[f"ref_{v}"] for v in names])
        w = CA.bcar(Q, a["y"], cid)
        o[arm] = _u(0.5 * BAG @ w + 0.5 * REF @ w, a["ya"], a["ba"])
        o[f"w_added_{arm}"] = [float(x) for x in w[len(w0):]]
        o[f"members_{arm}"] = int((w > 0).sum())
    sd = float(a["y"].std())
    for v in ("ridgeres",):
        o[f"alone_{v}"] = _u(0.5 * z[f"bag_{v}"] + 0.5 * z[f"ref_{v}"], a["ya"], a["ba"])
        o[f"oof_u_{v}"] = float(-np.log(max(float(np.sqrt(np.mean((z[f"q_{v}"] - a["y"]) ** 2))) / sd, 1e-12)))
    o["oof_u_best_stored"] = float(-np.log(max(float(np.sqrt(((a["Q"] - a["y"][:, None]) ** 2).mean(0)).min()) / sd, 1e-12)))
    o["infer_wall_s"] = float(z["wall_s"])
    o["cpu_s"] = time.process_time() - t0
    return o


def cmd_apply(a) -> None:
    import apex_core_meta as AM

    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    rows = AM.rows()
    if {rows[c]["cohort"] for c in cids} != {a.cohort}:
        raise SystemExit(f"the panel is not cohort {a.cohort}")
    b = np.load(a.bank, allow_pickle=False)
    if set(b["cohort"].tolist()) != {a.cohort} or sorted(b["cids"].tolist()) != sorted(cids):
        raise SystemExit("the bank does not hold exactly the panel's contexts")
    lo, bank = 0, {}
    for i, c in enumerate(b["cids"].tolist()):
        r = int(b["s_shape"][i][0])
        bank[c] = (b["y64"][lo:lo + r], b["lab"][lo:lo + r], str(b["ya_sha256"][i]))
        lo += r
    with Pool(a.procs) as pool:
        res = pool.map(one, [(a.run_dir, a.tag, a.cols, c, bank[c]) for c in cids], chunksize=1)
    st = [json.loads(p.read_text()) for p in sorted(Path(a.cols).glob("cols_status_*.json"))]
    out = {"schema": "jope.h424.apply/v1", "cohort": a.cohort, "n": len(res),
           "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
           "ckpt_sha256": sorted({s["ckpt_sha256"] for s in st}), "cols_status": st,
           "aligned_all": bool(all(r["aligned"] for r in res)), "columns_ok": int(sum(r["column_ok"] for r in res)),
           "replay_max_w_diff": float(max(r["w_diff"] for r in res)), "replay_max_u_diff": float(max(r["u_diff"] for r in res)),
           "cpu_s": float(sum(r["cpu_s"] for r in res)), "per_context": {r["cid"]: r for r in res}}
    with open(a.out, "x") as f:
        json.dump(out, f, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("cohort", "n", "aligned_all", "columns_ok", "replay_max_w_diff",
                                          "replay_max_u_diff", "cpu_s")}))


def _excl(g: np.ndarray) -> dict:
    """Win share with the contexts at the +-0.5 clip excluded (the pre-declared secondary)."""
    k = np.abs(g) < CLIP
    x = g[k]
    w, l = int((x > 1e-4).sum()), int((x < -1e-4).sum())
    return {"excluded_at_clip": int((~k).sum()), "n": int(k.sum()), "wins": w, "ties": int(k.sum()) - w - l, "losses": l,
            "win_share_of_decided": (w / (w + l)) if (w + l) else None,
            "win_share_ties_half": (w + 0.5 * (int(k.sum()) - w - l)) / int(k.sum()) if int(k.sum()) else None,
            "mean_gain": float(x.mean()) if len(x) else None}


def cmd_readout(a) -> None:
    import train_h416 as T16

    _, H = T16._stats_mod()
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    fam_of = {c: per[c]["family"] for c in cids}
    vals = {x: {c: per[c][x] for c in cids} for x in ARMS}
    pairs = (("I", "B"),)
    res = {"schema": "jope.h424.readout/v1", "cohort": ap["cohort"], "n": len(cids), "families": len(set(fam_of.values())),
           "baseline": "stored H420 TBOTH h416.SXB values, not recomputed", "ckpt_sha256": ap["ckpt_sha256"],
           "columns_ok": ap["columns_ok"], "family_mean_u": {x: float(H.family_mean(vals[x], fam_of)) for x in ARMS},
           "pairs": {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], fam_of) for x, y in pairs}}
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "B", BOOT_SEED, N_BOOT)
    res["delta_elo"] = {f"{x}_vs_{y}": T16.delta_elo(np, arms, point, boot, x, y) for x, y in pairs}
    res["secondary_clipped_excluded"] = {f"{x}_vs_B": _excl(np.array([per[c][x] - per[c]["B"] for c in cids])) for x in ("I",)}
    ok = [c for c in cids if per[c]["column_ok"]]
    wa = np.array([per[c]["w_added_I"][0] for c in ok]) if ok else np.zeros(0)
    res["finished_subset"] = {"n": len(ok)}   # contexts whose column was computed before the inference budget ended
    if 5 <= len(ok) < len(cids):
        res["finished_subset"]["I_vs_B"] = T16.pair_stats(np, H, {c: per[c]["I"] for c in ok}, {c: per[c]["B"] for c in ok},
                                                              {c: fam_of[c] for c in ok})
    res["column"] = {"contexts_with_weight": int((wa > 0).sum()), "mean_weight": float(wa.mean()) if len(wa) else None,
                     "mean_weight_when_used": float(wa[wa > 0].mean()) if (wa > 0).any() else None,
                     "alone_minus_B_mean": float(np.mean([per[c]["alone_ridgeres"] - per[c]["B"] for c in ok])) if ok else None,
                     "alone_beats_B": int(sum(per[c]["alone_ridgeres"] > per[c]["B"] for c in ok)) if ok else None,
                     "oof_beats_best_stored": int(sum(per[c]["oof_u_ridgeres"] > per[c]["oof_u_best_stored"] for c in ok))}
    prim = res["pairs"]["I_vs_B"]
    if ap["cohort"] != "DEV":
        res["verdict"] = "none (design-cohort code smoke)"
    elif prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        res["verdict"] = "ICL_HARM"
    elif prim["pass"]:
        res["verdict"] = "ICL_CONFIRMED"
    else:
        res["verdict"] = "ICL_NULL"
    print(f"== H424 {res['cohort']} n {res['n']} families {res['families']} columns ok {res['columns_ok']} baseline: {res['baseline']}")
    print("   mean u: " + "  ".join(f"{x} {res['family_mean_u'][x]:.4f}" for x in ARMS) + f"   column: {res['column']}")
    for k, s in res["pairs"].items():
        e, x = res["delta_elo"][k], res["secondary_clipped_excluded"][k]
        print(f"   {k:8s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']} dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}] | clipped excluded: "
              f"{x['wins']}/{x['ties']}/{x['losses']} share {x['win_share_of_decided']} gain {x['mean_gain']}")
    print(f"   verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cohort", required=True, choices=("DEV", "FIT-SEL"))
    p.add_argument("--bank", required=True)
    p.add_argument("--cols", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=16)
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_readout)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
