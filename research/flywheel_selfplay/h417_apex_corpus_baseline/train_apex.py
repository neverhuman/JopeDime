#!/usr/bin/env python3
"""H417: the stored H416 SXB recipe on APEX panels, and its TRAIN-side readouts.

  deploy   --panel P --run-dir R --tag T --cores L [--jobs 12] [--inner-jobs 3]
           (xbabe0 docker sandbox) deploy_h416.h416_unit at its published bytes, mode confirm, with the published H416
           design (DESIGN.json = BCAR.mix, sha-pinned), on the run dir's concatenated GS + GE + GL memo. Runtime =
           apex_core.build_rt. Panel ID2B = the first 2 P201b contexts (TRAIN; identity check against the stored H416
           confirm run, seed 335101); APEX panels use search seed 417101.
  idcheck  --run-dir R --tag T --stored-run S --stored-tag P201B
           the ID2B deploy must reproduce the stored H416 confirm records (SXB, KFB, GS.stock; |du| <= 1e-9) with 0
           physical worker runs. Proves the APEX runtime changes nothing for TRAIN contexts.
  readout  --run-dir R --tag T --out O.json
           (xbabe2, from the pulled run; numpy + h335_stats only) per entrant: family-mean AUDIT u, paired tests vs
           the single-run recipe GS.stock and SXB vs KFB, internal Elo (anchor GS.stock = 1500). Descriptive: H417
           is the baseline wave, it has no pass / fail test.
  headroom --run-dir R --tag T --out O.json
           (FIT-SEL only) per context, from the stored matrices: the best single full-refit program on AUDIT and the
           AUDIT-fitted Caruana stack (both in-sample upper bounds) against SXB. Sizes what selection can still gain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

NEURAL = {"GS": "H0.1-A", "GE": "H381-E2-S3fz", "GL": "H415-L"}
DESIGN_SHA256 = "0f0c96ad5d819db42ba1c242c814f4eac3345d9ce2744924666a89820314f8c4"  # H416 DESIGN.json on v1/next
ID_SEED, ID_TOL = 335101, 1e-9
BOOT_SEED, N_BOOT = 417150, 10_000
ENTRANTS = {"X0.GS": "GS.stock", "GE.stock": "GE.stock", "GL.stock": "GL.stock", "BAG2": "P2+LIB2.BAG_CAR",
            "BAG3": "P3+LIB3.BAG_CAR", "KFB": "P3+LIB3.KFB", "HFR": "P3+LIB3.HFR"}
SXB_KEY = "P3+LIB3.SXB"


def _design() -> dict:
    raw = (HERE / "DESIGN.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA256:
        raise SystemExit("staged DESIGN.json is not the published H416 design")
    d = json.loads(raw)
    if d.get("schema") != "jope.h416.design/v1" or d.get("design") != "BCAR.mix":
        raise SystemExit("not the H416 BCAR.mix design")
    return d


def cmd_deploy(a) -> None:
    import apex_core as A  # FIRST import of the GP stack: h335_core pins + guard

    DH, K, C, L, D = A.DH, A.K, A.C, A.L, A.D
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    A.install()
    if a.panel == "ID2B":
        p201b = json.loads((Path(C.DATA_ROOT) / "h416" / "P201b.json").read_text())
        cids, seed = sorted(p201b["cids"])[:2], ID_SEED
    else:
        cids, seed = A.panel(a.panel), A.SEARCH_SEED
    design = _design()
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    missing = [c for c in cids if not (memo / f"{c}.split.jsonl").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no concatenated memo (run deploy-prep first)")
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / NEURAL[s] for s in ("GS", "GE", "GL")}
    rt, scr, man = A.build_rt(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (seed,), nd)
    cfg = {"wave": "H417", "tag": a.tag, "mode": "confirm", "panel": a.panel, "cids": cids, "sources": list(nd),
           "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": a.timeout, "design": design,
           "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                    "max_members": DH.M.MAX_MEMBERS},
           "manifest_sha256": A.MANIFEST_SHA256,
           "code_sha256": K.code_manifest(A.code_files() + ("gp_apex.py", "train_apex.py")),
           "screener_manifest": man}
    p = run_dir / f"config_{a.tag}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items()):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)
    DH.GPH.install(0)
    npz_dir = run_dir / f"h416_{a.tag}_npz"
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(lambda c, _s: DH.h416_unit(rt, a.tag, c, scr, seed, "confirm", npz_dir, design,
                                                     a.inner_jobs), [(c, 0) for c in order], a.jobs)
    st = {"tag": a.tag, "mode": "confirm", "units": len(recs), "wall_s": time.time() - t0,
          "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "akf_ledger": DH.DP15.KF_LEDGER.check(),
          "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DH.DP.physical_by_kind(rt), "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{a.tag}_{int(t0)}.json", st)
    rt.engine.close()
    print(json.dumps({"tag": a.tag, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"], "akf_ledger": st["akf_ledger"]}))


# ----------------------------------------------------------------------------------------------------------------------
# readouts (no GP stack: train_h416's helpers, numpy, h335_stats, meta_h416)
# ----------------------------------------------------------------------------------------------------------------------
def _records(run_dir: Path, tag: str):
    import train_h416 as T16

    cfg = json.loads((run_dir / f"config_{tag}.json").read_text())
    r16, r15 = {}, {}
    for c in cfg["cids"]:
        r = T16._last_result(run_dir / "stages" / f"H416_{tag}" / f"{c}__confirm.jsonl")
        r16[c] = r
        r15[c] = T16._last_result(run_dir / "stages" / r["h415_stage"] / f"{c}__pool.jsonl")
    return T16, cfg, r16, r15


def _values(T16, r16: dict, r15: dict) -> dict:
    """{entrant: {cid: u}} + invalid counts. X0.GS = the single-run stock output (0.0 = SEARCH-mean predictor when it
    is invalid); every other invalid or missing output scores X0.GS's value (H415 / H416 rule) and is counted."""
    vals = {e: {} for e in list(ENTRANTS) + ["SXB"]}
    inv = {e: 0 for e in vals}
    for c in r16:
        o = r15[c]["outputs"]
        gs = o.get("GS.stock")
        ok = gs is not None and not gs.get("invalid") and gs["raw"].get("status") == "completed"
        x0 = float(gs["value"]) if ok else 0.0
        inv["X0.GS"] += not ok
        vals["X0.GS"][c] = x0
        for e, k in ENTRANTS.items():
            if e == "X0.GS":
                continue
            v, bad, _ = T16.out_value(r15[c], k, x0)
            vals[e][c] = v
            inv[e] += bool(bad)
        s = r16[c]["outputs"].get(SXB_KEY)
        bad = s is None or s.get("invalid") or r16[c].get("error")
        vals["SXB"][c] = x0 if bad else float(s["value"])
        inv["SXB"] += bool(bad)
    return vals, inv


def cmd_idcheck(a) -> None:
    T16, cfg, r16, r15 = _records(Path(a.run_dir), a.tag)
    out, ok = [], True
    for c in cfg["cids"]:
        s16 = T16._last_result(Path(a.stored_run) / "stages" / f"H416_{a.stored_tag}" / f"{c}__confirm.jsonl")
        s15 = T16._last_result(Path(a.stored_run) / "stages" / s16["h415_stage"] / f"{c}__pool.jsonl")
        row = {"cid": c}
        for name, mine, theirs in (("SXB", r16[c]["outputs"][SXB_KEY]["value"], s16["outputs"][SXB_KEY]["value"]),
                                   ("KFB", r16[c]["kfb_u"], s16["kfb_u"]),
                                   ("GS.stock", r15[c]["outputs"]["GS.stock"]["value"],
                                    s15["outputs"]["GS.stock"]["value"])):
            row[name] = {"mine": mine, "stored": theirs, "delta": mine - theirs}
            ok = ok and abs(mine - theirs) <= ID_TOL
        row["members_same"] = (r16[c]["outputs"][SXB_KEY]["spec"]["members"]
                               == s16["outputs"][SXB_KEY]["spec"]["members"])
        ok = ok and row["members_same"]
        out.append(row)
    phys = {}
    for p in sorted(Path(a.run_dir).glob(f"status_{a.tag}_*.json")):
        for k, v in json.loads(p.read_text())["physical_by_kind"].items():
            phys[k] = phys.get(k, 0) + int(v)
    ok = ok and sum(phys.values()) == 0
    print(json.dumps({"ok": ok, "physical_by_kind": phys, "contexts": out}, indent=1))
    if not ok:
        raise SystemExit("idcheck failed")


def cmd_readout(a) -> None:
    import apex_core_meta as AM

    T16, cfg, r16, r15 = _records(Path(a.run_dir), a.tag)
    np, H = T16._stats_mod()
    rows = AM.rows()
    fam_all = AM.family_of(cfg["cids"])
    if len(set(fam_all.values())) != len(cfg["cids"]):
        raise SystemExit("one context per family expected")
    vals_all, _ = _values(T16, r16, r15)
    res = {"schema": "jope.h417.readout/v1", "tag": a.tag, "panel": cfg["panel"], "seed": cfg["seed"],
           "cohorts": {}, "values": vals_all,
           "worker_s": {k: float(sum(r["worker_s"].get(k, 0.0) for r in list(r16.values()) + list(r15.values())))
                        for k in sorted({k for r in list(r16.values()) + list(r15.values()) for k in r["worker_s"]})}}
    groups = {co: [c for c in cfg["cids"] if rows[c]["cohort"] == co] for co in ("DEV", "FIT-SEL")}
    groups["ALL"] = list(cfg["cids"])
    for co, cids in groups.items():
        if not cids:
            continue
        fam_of = {c: fam_all[c] for c in cids}
        vals, inv = _values(T16, {c: r16[c] for c in cids}, {c: r15[c] for c in cids})
        arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "X0.GS", BOOT_SEED, N_BOOT)
        out = {"n": len(cids), "family_mean_u": {e: H.family_mean(v, fam_of) for e, v in vals.items()},
               "invalid": inv, "errors": sum(1 for c in cids if r16[c].get("error")), "elo_anchor_X0.GS_1500": elo,
               "vs_X0.GS": {e: T16.pair_stats(np, H, vals[e], vals["X0.GS"], fam_of) for e in vals if e != "X0.GS"},
               "SXB_vs_KFB": {**T16.pair_stats(np, H, vals["SXB"], vals["KFB"], fam_of),
                              "delta_elo": T16.delta_elo(np, arms, point, boot, "SXB", "KFB")}}
        res["cohorts"][co] = out
        print(f"== {co}: n {out['n']} errors {out['errors']} invalid {inv}")
        print("   family-mean u: " + "  ".join(f"{e} {v:.4f}" for e, v in out["family_mean_u"].items()))
        for e, s in out["vs_X0.GS"].items():
            print(f"   {e:9s} vs X0.GS gain {s['gain']:+.4f} ({s['pct_rmse']:+.2f}% RMSE) W/T/L {s['wtl']} "
                  f"clipped {s['clipped_mean']:+.4f} p {s['signflip_p']:.2g} pass {s['pass']} Elo {elo[e]['elo']:.1f}")
        s = out["SXB_vs_KFB"]
        print(f"   SXB vs KFB: dElo {s['delta_elo']['delta']:+.1f} {s['delta_elo']['ci95']} W/T/L {s['wtl']} "
              f"gain {s['gain']:+.4f}")
    Path(a.out).write_text(json.dumps(res, sort_keys=True))


def cmd_headroom(a) -> None:
    import apex_core_meta as AM

    T16, cfg, r16, r15 = _records(Path(a.run_dir), a.tag)
    np, H = T16._stats_mod()
    import meta_h416 as M

    rows = AM.rows()
    cids = [c for c in cfg["cids"] if rows[c]["cohort"] == "FIT-SEL"]  # DEV AUDIT rows are used for scoring alone
    if not cids:
        raise SystemExit("no FIT-SEL context in this run")
    r16 = {c: r16[c] for c in cids}
    r15 = {c: r15[c] for c in cids}
    fam_of = AM.family_of(cids)
    vals, _ = _values(T16, r16, r15)
    best1, ostack, nprog = {}, {}, {}
    for c, r in r16.items():
        if r.get("error"):
            best1[c] = ostack[c] = vals["SXB"][c]
            continue
        z = T16.load_npz(np, r, Path(a.run_dir))
        cols = [(i, p) for i, p in enumerate(z["refit"]) if p is not None and np.all(np.isfinite(p))]
        nprog[c] = len(cols)
        if not cols:
            best1[c] = ostack[c] = vals["SXB"][c]
            continue
        P = np.column_stack([p for _, p in cols])
        us = [M.utility(P[:, j], z["ya"], z["ba"])[0] for j in range(P.shape[1])]
        best1[c] = float(max(us))
        cnt = M.kcar(P, z["ya"], 100)  # fitted on AUDIT: an in-sample upper bound, never a deployable output
        ostack[c] = float(M.utility(P @ (cnt / cnt.sum()), z["ya"], z["ba"])[0])
    res = {"schema": "jope.h417.headroom/v1", "tag": a.tag, "cohort": "FIT-SEL", "n": len(cids),
           "family_mean_u": {"SXB": H.family_mean(vals["SXB"], fam_of), "KFB": H.family_mean(vals["KFB"], fam_of),
                             "oracle_best_single_refit": H.family_mean(best1, fam_of),
                             "oracle_audit_fitted_caruana": H.family_mean(ostack, fam_of)},
           "oracle_single_minus_SXB": T16.pair_stats(np, H, best1, vals["SXB"], fam_of),
           "oracle_stack_minus_SXB": T16.pair_stats(np, H, ostack, vals["SXB"], fam_of),
           "usable_programs_mean": float(np.mean(list(nprog.values()))) if nprog else 0.0,
           "values": {"best_single": best1, "oracle_stack": ostack}}
    Path(a.out).write_text(json.dumps(res, sort_keys=True))
    print(json.dumps({k: v for k, v in res.items() if k != "values"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--panel", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--inner-jobs", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p = sub.add_parser("idcheck")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--stored-run", required=True)
    p.add_argument("--stored-tag", default="P201B")
    for name in ("readout", "headroom"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--tag", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"deploy": cmd_deploy, "idcheck": cmd_idcheck, "readout": cmd_readout, "headroom": cmd_headroom}[a.cmd](a)


if __name__ == "__main__":
    main()
