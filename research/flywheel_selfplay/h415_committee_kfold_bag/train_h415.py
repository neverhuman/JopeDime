#!/usr/bin/env python3
"""H415 TRAIN driver (PREREG §6-7): deploy_h415.deploy_unit over a panel, then the fixed readout.

The deploy run dir's memo is the concatenation, per context, of the source GP runs' memos in source order (made on the
host by `run_h415.sh deploy-prep`; the source run dirs are never written). Sources (one GP seed, 415101, every arm):
GS = H0.1-A per-fold OOF screener, GE = H381-E2-S3fz, GL = H415-L. P2 = GS + GE, P3 = GS + GE + GL.

  deploy  --panel P201e|DEV2E --run-dir R --tag T --sources GS,GE[,GL] --cores L [--jobs J] [--inner-jobs 3]
          [--timeout 300]
  analyze --run-dir R --tag T --panel P201e|DEV2E --gp GS=DIR --gp GE=DIR [--gp GL=DIR] [--panel-file F] --out F
          (numpy + h335_stats only; reads the deploy stage records and the source runs' stored stage records)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

SEED = 415101
STAGE_GP, KEY_GP = "E415_AGG", "AGG.g50.CV3@12"
NEURAL = {"GS": "H0.1-A", "GE": "H381-E2-S3fz", "GL": "H415-L"}
BOOT_SEED, N_BOOT = 415150, 10_000
HOLM_ALPHA = 0.05
ELO_OUTPUTS = ("X0e", "GE.stock", "GL.stock", "P2.RUNBAG", "P3.RUNBAG", "P2+LIB2.BAG_CAR", "P3+LIB3.BAG_CAR",
               "P3+LIB3.W48", "P3+LIB3.KFB", "P3+LIB3.HFR", "P3+LIB3.KFB1", "LIB.single")


def cmd_deploy(a) -> None:
    import deploy_h415 as DP  # FIRST import of the GP stack: h335_core pins + guard

    import gp_h415 as GP

    K, C, L, D = DP.K, DP.C, DP.L, DP.D
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    srcs = a.sources.split(",")
    if srcs[:2] != ["GS", "GE"] or srcs[2:] not in ([], ["GL"]):
        raise SystemExit("sources are GS,GE or GS,GE,GL (PREREG §6 pool order)")
    cids = GP.panel(a.panel)
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    missing = [c for c in cids if not (memo / f"{c}.split.jsonl").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no concatenated memo (run deploy-prep first)")
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / NEURAL[s] for s in srcs}
    rt, scr, man = DP.DP.build_rt_t(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (SEED,), nd)
    cfg = {"tag": a.tag, "panel": a.panel, "cids": cids, "sources": srcs, "neural": {s: str(p) for s, p in nd.items()},
           "seed": SEED, "timeout_s": a.timeout, "kf": [DP.KF_SEED, DP.KF_K], "hf": [DP.HF_TOP, DP.HF_PER_HEAD,
                                                                                    DP.HF_CAP],
           "kcar_steps": DP.KCAR_STEPS, "code_sha256": K.code_manifest(DP.code_files() + ("gp_h415.py",
                                                                                         "train_h415.py")),
           "screener_manifest": man}
    p = run_dir / f"config_{a.tag}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items()):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)
    DP.GPH.install(0)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(lambda c, _s: DP.deploy_unit(rt, a.tag, c, scr, SEED, a.inner_jobs),
                          [(c, 0) for c in order], a.jobs)
    st = {"tag": a.tag, "units": len(recs), "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "akf_ledger": DP.KF_LEDGER.check(), "guard": C.guard_report(),
          "bank_counters": dict(D.BANK_COUNTERS), "physical_by_kind": DP.DP.physical_by_kind(rt),
          "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{a.tag}_{int(t0)}.json", st)
    rt.engine.close()
    print(json.dumps({"tag": a.tag, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"], "akf_ledger": st["akf_ledger"]}))


def _last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def _members(o: dict):
    sp = o.get("spec") or {}
    return sp.get("members") if isinstance(sp, dict) else None


def holm(ps: dict, alpha: float = HOLM_ALPHA) -> dict:
    """Holm step-down over {name: p}; returns {name: rejected}."""
    order = sorted(ps, key=lambda k: (ps[k], k))
    m, rej, go = len(order), {}, True
    for i, k in enumerate(order):
        go = go and ps[k] <= alpha / (m - i)
        rej[k] = go
    return rej


def cmd_analyze(a) -> None:
    # the staged h335_stats.py (next to this file in a stage) wins; the H335 source copy is the fallback
    sys.path.append("/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo")
    import numpy as np
    import h335_stats as H

    gp = dict(x.split("=", 1) for x in a.gp)
    if list(gp)[:2] != ["GS", "GE"]:
        raise SystemExit("--gp GS=... --gp GE=... [--gp GL=...]")
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    if cfg["sources"] != list(gp):
        raise SystemExit("--gp arms differ from the deploy sources")
    stored = {n: {c: _last_result(Path(d) / "stages" / STAGE_GP / f"{c}__{SEED}.jsonl")["outputs"][KEY_GP]
                  for c in cids} for n, d in gp.items()}
    recs = {c: _last_result(run_dir / "stages" / f"H415_{a.tag}" / f"{c}__pool.jsonl") for c in cids}
    fam_of = {c: recs[c]["family_id"] for c in cids}  # rt.fam_of = the TRAIN row index (C.train_rows)
    if a.panel_file:
        pf = json.loads(Path(a.panel_file).read_text())
        if sorted(pf["cids"]) != sorted(cids) or any(pf["family_of"][c] != fam_of[c] for c in cids):
            raise SystemExit("deploy records differ from the panel file (cids / families)")
    kp = recs[cids[0]]["kfb_pool"]
    if any(r["kfb_pool"] != kp for r in recs.values()):
        raise SystemExit("kfb_pool differs across contexts")
    x0 = {c: float(stored["GS"][c]["value"]) for c in cids}

    # replay check (PREREG §6 inputs / §8.2): deploy-unit stock outputs == the stored run outputs
    replay = {}
    for n in gp:
        d = [recs[c]["outputs"][f"{n}.stock"]["value"] - stored[n][c]["value"] for c in cids]
        same = sum(1 for c in cids if _members(recs[c]["outputs"][f"{n}.stock"]) == _members(stored[n][c]))
        replay[n] = {"max_abs_delta": float(max(abs(v) for v in d)), "n_nonzero": int(sum(v != 0 for v in d)),
                     "members_same": same, "n": len(cids)}

    def out_of(c, k):
        """(value, invalid, bag_fallback). X0e / <arm>.stock = the stored run outputs; BAG_<b> missing -> <b> (H410
        rule); a missing or invalid output scores the X0e value (0 gain) and is counted."""
        if k == "X0e":
            return x0[c], False, False
        if k.endswith(".stock") and k[:-6] in stored:
            o = stored[k[:-6]][c]
            return float(o["value"]), bool(o.get("invalid")), False
        o = recs[c]["outputs"]
        alts = [k] + ([k.replace("BAG_", "")] if "BAG_" in k else [])
        for i, kk in enumerate(alts):
            if kk in o:
                if o[kk].get("invalid") or o[kk]["raw"].get("status") != "completed":
                    return x0[c], True, i > 0
                return float(o[kk]["value"]), False, i > 0
        return x0[c], True, False

    allk = sorted(set().union(*[set(r["outputs"]) for r in recs.values()]) | {f"{n}.stock" for n in gp})

    def stats(k, ctrl):
        vals = {c: out_of(c, k)[0] for c in cids}
        pt = H.primary_test(vals, ctrl, fam_of)
        hl = H.headline(vals, ctrl, fam_of)
        _, Dc = H.family_contrast(vals, ctrl, fam_of, clip=H.CLIP)
        _, Du = H.family_contrast(vals, ctrl, fam_of, clip=None)
        rng = np.random.default_rng(BOOT_SEED)
        lcb = float(np.quantile(Dc[rng.integers(0, len(Dc), size=(N_BOOT, len(Dc)))].mean(axis=1), 0.025))
        harm = float((Du < -0.01).mean())
        gate = bool(lcb > 0.001 and pt["family_median"] >= 0 and harm <= 0.10 and pt["signflip_p"] < H.ALPHA
                    and pt["lofo_min"] > 0)
        return {"family_mean_gain": hl["family_mean_gain"], "pct_rmse_reduction": hl["pct_rmse_reduction"],
                "wtl": [hl["families_won"], hl["families_tied"], hl["families_lost"]], "primary_test": pt,
                "boot_lcb95_clipped": lcb, "frac_families_harmed_gt_0.01": harm, "h317_adoption_gate": gate,
                "invalid": sum(1 for c in cids if out_of(c, k)[1]),
                "bag_fallback_contexts": sum(1 for c in cids if out_of(c, k)[2])}

    rep = {"schema": "jope.h415.train-analysis/v1", "panel": a.panel, "tag": a.tag, "n_contexts": len(cids),
           "n_families": len(set(fam_of.values())), "sources": list(gp), "kfb_pool": kp, "replay": replay,
           "outputs": {k: stats(k, x0) for k in allk if k != "X0e"}}
    cand = f"{kp}.KFB"
    tests = {"primary": (cand, "X0e")}
    sec = {"S1": (cand, f"{kp}.BAG_CAR"), "S2": ("P2+LIB2.BAG_CAR", "X0e")}
    if "GL" in gp:
        sec["S3"] = ("GL.stock", "X0e")
    ctrl_of = lambda k: {c: out_of(c, k)[0] for c in cids}  # noqa: E731
    res = {}
    for name, (arm, ctrl) in {**tests, **sec}.items():
        res[name] = {"arm": arm, "ctrl": ctrl, **stats(arm, ctrl_of(ctrl))}
    hr = holm({s: res[s]["primary_test"]["signflip_p"] for s in sec})
    for s in sec:
        pt = res[s]["primary_test"]
        res[s]["holm_rejected"] = hr[s]
        res[s]["pass"] = bool(hr[s] and pt["lofo_min"] > 0 and pt["family_median"] >= 0
                              and pt["T_clipped_family_mean"] > 0)
    rep["tests"] = res
    p1 = res["primary"]["primary_test"]["pass"]
    if p1:
        verdict = "KFB_CONFIRMED_BEATS_H410" if res["S1"]["pass"] else "KFB_CONFIRMED"
    else:
        verdict = "KFB_NOT_CONFIRMED"
    rep["verdict"] = verdict
    rep["gain_components"] = [s for s in sec if res[s]["pass"]]
    elo_keys = [k for k in ELO_OUTPUTS if k == "X0e" or k in rep["outputs"]]
    rep["elo"] = H.elo_panel({k: {c: out_of(c, k)[0] for c in cids} for k in elo_keys}, fam_of, "X0e",
                             seed=BOOT_SEED, n_boot=N_BOOT)
    ws = {}
    for r in recs.values():
        for k, v in r.get("worker_s", {}).items():
            ws[k] = ws.get(k, 0.0) + float(v)
        for k, o in r["outputs"].items():
            if "akf_worker_s" in o.get("raw", {}):
                ws[f"akf.{k}"] = ws.get(f"akf.{k}", 0.0) + float(o["raw"]["akf_worker_s"])
    rep["worker_s_total"] = ws
    hf = [r.get("hf", {}) for r in recs.values()]
    rep["hf"] = {"n_candidates_mean": float(np.mean([h.get("n_candidates", 0) for h in hf])),
                 "n_usable_mean": float(np.mean([h.get("n_usable", 0) for h in hf])),
                 "kcar_members_mean": float(np.mean([len(h.get("kcar_members", [])) for h in hf])),
                 "contexts_without_usable": sum(1 for h in hf if not h.get("n_usable")),
                 "envelopes_ok": sum(1 for h in hf if h.get("envelopes", {}).get("ok")),
                 "kcar_minus_best_single_oof_u_mean": float(np.mean([h["kcar_oof_u"] - h["best_single_oof_u"]
                                                                     for h in hf if "kcar_oof_u" in h] or [math.nan]))}
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"replay: {json.dumps(replay)}")
    print(f"verdict {verdict}  components {rep['gain_components']}")
    for name, r in res.items():
        pt = r["primary_test"]
        print(f"{name:8s} {r['arm']} vs {r['ctrl']}: gain {r['family_mean_gain']:+.4f} "
              f"rmse {100 * r['pct_rmse_reduction']:+.2f}% clip {pt['T_clipped_family_mean']:+.4f} "
              f"p {pt['signflip_p']:.4g} lofo {pt['lofo_min']:+.4f} med {pt['family_median']:+.4f} "
              f"pass {r.get('pass', pt['pass'])}")
    for k, r in sorted(rep["outputs"].items(), key=lambda kv: -kv[1]["family_mean_gain"]):
        pt = r["primary_test"]
        print(f"{k:24s} gain {r['family_mean_gain']:+.4f} rmse {100 * r['pct_rmse_reduction']:+.2f}% "
              f"W/T/L {'/'.join(map(str, r['wtl']))} p {pt['signflip_p']:.3g} lofo {pt['lofo_min']:+.4f} "
              f"lcb {r['boot_lcb95_clipped']:+.4f} inv {r['invalid']}")
    for k, v in sorted(rep["elo"]["elo"].items(), key=lambda kv: -kv[1]["elo"]):
        print(f"elo {k:24s} {v['elo']:7.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--panel", required=True, choices=("P201e", "DEV2E"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--inner-jobs", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p = sub.add_parser("analyze")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--gp", action="append", required=True)
    p.add_argument("--panel-file", default="", help="optional cross-check: P201e.json (cids + family_of)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"deploy": cmd_deploy, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
