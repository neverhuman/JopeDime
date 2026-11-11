#!/usr/bin/env python3
"""H436: the key-lookup columns added to the stored stack (numpy; no GP engine, no table is opened here).

  stack_h436.py apply   --run-dir DIR --tag TAG --panel NAME --bank FILE --cols DIR --out FILE [--procs N]
                        [--base-cols DIR --base-name NAME[,NAME...]]
  stack_h436.py readout --apply FILE --out FILE [--units FILE] [--exclude FILE]

apply, per context of config_<TAG>.json (which must be exactly the bank's contexts):
  the stored matrices (sha-checked against the H416 record; cache_h422.stored) are joined with the H436 columns
  (cols_h436.py) after three alignment checks: SEARCH targets equal, fold labels equal, and the sha256 of the stored
  AUDIT targets equal to the one in the bank.
  B     the stored h416.SXB value of the record (not recomputed; the recomputation is the replay check)
  BASE  B, or with --base-cols the stored 64 candidates + the named columns of that directory (e.g. H430's
        key_direct,key_gbm,factor_glm), rebuilt with the stored builder
  KH    BASE's candidates + key_hybrid                                                          [primary]
  KD    BASE's candidates + key_direct of the base directory (H430's lookup with a mean backoff)
  KHR   BASE's candidates + key_hybrid + ridge_z
  G1    BASE's candidates + gbm_plain        (control: the backoff model alone)
  Every arm uses the stored weight builder (meta_h416 BCAR) and the stored mix of bag and refit predictions.
  A context whose H436 columns are missing or not ok scores BASE in every arm and is counted.
readout: clipped family-mean contrasts (h335_stats.primary_test) for KH vs KD (primary), KH vs BASE, KD vs BASE,
  KH vs B, KD vs B, KHR vs KH and G1 vs BASE, where B is the stored stack value.
  --units FILE replaces the family of each context by a unit id (JSON object cid -> unit), e.g. the H426 clusters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("H436_BASE_CODE"):     # development runs: the frozen H424 stage that holds the stored builder
    sys.path.append(os.environ["H436_BASE_CODE"])
import cache_h422 as CA  # noqa: E402
import meta_h416 as M  # noqa: E402

SXB_KEY = "P3+LIB3.SXB"
ARMS = {"KH": ("key_hybrid",), "KD": ("base:key_direct",), "KHR": ("key_hybrid", "ridge_z"), "G1": ("gbm_plain",)}
ALL = ("key_hybrid", "ridge_z", "gbm_plain")
BOOT_SEED, N_BOOT = 436150, 10_000
_BANK = {}


def _u(pred, ya, ba) -> float:
    return float(M.utility(pred, ya, ba)[0])


def one(job) -> dict:
    run_dir, tag, cols_dir, cid, bank_row, base_dir, base_name = job
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
    Q, BAG, REF = a["Q"], a["BAG"], a["REF"]
    o["BASE"], o["base_added"] = o["B"], False
    zb = None
    if base_dir:
        bp = Path(base_dir) / f"{cid}.npz"
        zb = np.load(bp, allow_pickle=False) if bp.exists() else None
        bn = base_name.split(",")          # one or several column names of the base directory
        if zb is not None and bool(zb["ok"]) and o["aligned"] and all(len(zb[f"q_{v}"]) == len(a["y"]) for v in bn):
            Q = np.column_stack([Q] + [zb[f"q_{v}"] for v in bn])
            BAG = np.column_stack([BAG] + [zb[f"bag_{v}"] for v in bn])
            REF = np.column_stack([REF] + [zb[f"ref_{v}"] for v in bn])
            wb = CA.bcar(Q, a["y"], cid)
            o["BASE"], o["base_added"] = _u(0.5 * BAG @ wb + 0.5 * REF @ wb, a["ya"], a["ba"]), True
    p = Path(cols_dir) / f"{cid}.npz"
    z = np.load(p, allow_pickle=False) if p.exists() else None
    o["column_ok"] = bool(z is not None and bool(z["ok"]) and o["aligned"]
                          and all(len(z[f"q_{v}"]) == len(a["y"]) and len(z[f"ref_{v}"]) == len(a["ya"]) for v in ALL))
    if not o["column_ok"]:
        o.update({arm: o["BASE"] for arm in ARMS})
        o["cpu_s"] = time.process_time() - t0
        return o
    nb = Q.shape[1]
    def col(kind, v):                       # "base:<name>" takes the column from the base directory
        return zb[f"{kind}_{v[5:]}"] if v.startswith("base:") else z[f"{kind}_{v}"]

    for arm, names in ARMS.items():
        if any(v.startswith("base:") for v in names) and not o["base_added"]:
            o[arm], o[f"w_added_{arm}"] = o["BASE"], [0.0] * len(names)
            continue
        Qa = np.column_stack([Q] + [col("q", v) for v in names])
        Ba = np.column_stack([BAG] + [col("bag", v) for v in names])
        Ra = np.column_stack([REF] + [col("ref", v) for v in names])
        w = CA.bcar(Qa, a["y"], cid)
        o[arm] = _u(0.5 * Ba @ w + 0.5 * Ra @ w, a["ya"], a["ba"])
        o[f"w_added_{arm}"] = [float(x) for x in w[nb:]]
    sd = float(a["y"].std())
    for v in ALL:
        o[f"alone_{v}"] = _u(0.5 * z[f"bag_{v}"] + 0.5 * z[f"ref_{v}"], a["ya"], a["ba"])
        o[f"oof_u_{v}"] = float(-np.log(max(float(np.sqrt(np.mean((z[f"q_{v}"] - a["y"]) ** 2))) / sd, 1e-12)))
    o["oof_u_best_stored"] = float(-np.log(max(float(np.sqrt(((a["Q"] - a["y"][:, None]) ** 2).mean(0)).min()) / sd, 1e-12)))
    o["cpu_s"] = time.process_time() - t0
    return o


def cmd_apply(a) -> None:
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    b = np.load(a.bank, allow_pickle=False)
    if sorted(b["cids"].tolist()) != sorted(cids) or set(b["cohort"].tolist()) != {a.panel}:
        raise SystemExit("the bank does not hold exactly the panel's contexts")
    lo, bank = 0, {}
    for i, c in enumerate(b["cids"].tolist()):
        r = int(b["s_shape"][i][0])
        bank[c] = (np.asarray(b["y64"][lo:lo + r], np.float64), b["lab"][lo:lo + r], str(b["ya_sha256"][i]))
        lo += r
    jobs = [(a.run_dir, a.tag, a.cols, c, bank[c], a.base_cols, a.base_name) for c in cids]
    with Pool(a.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    st = [json.loads(p.read_text()) for p in sorted(Path(a.cols).glob("cols_status_*.json"))]
    out = {"schema": "jope.h436.apply/v1", "panel": a.panel, "tag": a.tag, "n": len(res),
           "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(), "cols_status": st,
           "base": {"cols": a.base_cols, "name": a.base_name} if a.base_cols else None,
           "aligned_all": bool(all(r["aligned"] for r in res)), "columns_ok": int(sum(r["column_ok"] for r in res)),
           "base_added": int(sum(r["base_added"] for r in res)),
           "replay_max_w_diff": float(max(r["w_diff"] for r in res)), "replay_max_u_diff": float(max(r["u_diff"] for r in res)),
           "cpu_s": float(sum(r["cpu_s"] for r in res)), "per_context": {r["cid"]: r for r in res}}
    with open(a.out, "x") as f:
        json.dump(out, f, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("panel", "n", "aligned_all", "columns_ok", "base_added", "replay_max_w_diff",
                                          "replay_max_u_diff", "cpu_s")}))


def cmd_readout(a) -> None:
    import train_h416 as T16

    _, H = T16._stats_mod()
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    if a.exclude:                          # context-id prefixes to leave out (a pre-registered secondary reading)
        drop = [x for x in Path(a.exclude).read_text().split() if x]
        cids = [c for c in cids if not any(c.startswith(x) for x in drop)]
    unit = {c: per[c]["family"] for c in cids}
    if a.units:
        m = json.loads(Path(a.units).read_text())
        unit = {c: str(m[c]) for c in cids}
    arms = ("B", "BASE", "KH", "KD", "KHR", "G1")
    vals = {x: {c: per[c][x] for c in cids} for x in arms}
    pairs = (("KH", "KD"), ("KH", "BASE"), ("KD", "BASE"), ("KH", "B"), ("KD", "B"), ("KHR", "KH"), ("G1", "BASE"))
    res = {"schema": "jope.h436.readout/v1", "panel": ap["panel"], "tag": ap["tag"], "n": len(cids),
           "units": len(set(unit.values())), "unit_kind": "file" if a.units else "family", "base": ap["base"],
           "excluded": len(per) - len(cids),
           "columns_ok": ap["columns_ok"], "aligned_all": ap["aligned_all"],
           "replay_max_u_diff": ap["replay_max_u_diff"],
           "unit_mean_u": {x: float(H.family_mean(vals[x], unit)) for x in arms},
           "pairs": {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], unit) for x, y in pairs}}
    an, point, boot, _elo = T16.elo_boot(np, H, vals, unit, "B", BOOT_SEED, N_BOOT)
    res["delta_elo"] = {f"{x}_vs_{y}": T16.delta_elo(np, an, point, boot, x, y) for x, y in pairs}
    ok = [c for c in cids if per[c]["column_ok"]]
    res["columns"] = {}
    for i, v in enumerate(ARMS["KHR"]):
        w = np.array([per[c]["w_added_KHR"][i] for c in ok]) if ok else np.zeros(0)
        res["columns"][v] = {"contexts_with_weight": int((w > 0).sum()), "mean_weight": float(w.mean()) if len(w) else None,
                             "oof_beats_best_stored": int(sum(per[c][f"oof_u_{v}"] > per[c]["oof_u_best_stored"] for c in ok))}
    prim, ctl = res["pairs"]["KH_vs_KD"], res["pairs"]["KH_vs_B"]
    res["primary_pass"] = bool(prim["pass"])
    res["beats_stored_stack"] = bool(ctl["pass"])
    res["harm"] = bool(prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025)
    print(f"== H436 {res['panel']} tag {res['tag']} n {res['n']} units {res['units']} ({res['unit_kind']}) "
          f"columns ok {res['columns_ok']} base {res['base']}")
    print("   mean u: " + "  ".join(f"{x} {res['unit_mean_u'][x]:.4f}" for x in arms))
    for k, s in res["pairs"].items():
        e = res["delta_elo"][k]
        print(f"   {k:11s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']} dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]")
    print(f"   columns: {res['columns']}")
    print(f"   primary_pass {res['primary_pass']}  beats_stored_stack {res['beats_stored_stack']}  harm {res['harm']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--run-dir", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--panel", required=True); p.add_argument("--bank", required=True)
    p.add_argument("--cols", required=True); p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--base-cols"); p.add_argument("--base-name", default="ridgeres")
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True); p.add_argument("--out", required=True); p.add_argument("--units")
    p.add_argument("--exclude")
    p.set_defaults(fn=cmd_readout)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
