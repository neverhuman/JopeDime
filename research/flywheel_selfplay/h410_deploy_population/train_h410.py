#!/usr/bin/env python3
"""H410 TRAIN driver (PREREG §6): deploy_unit on TRAIN panels from the STORED H364 X0 and H381 E2L / E2R runs.

The run dir's memo is a concatenation of the stored runs' memos for the panel (made on the host; the stored run dirs
are never written). Sources: X0 = H364_C_xbabe0 (H0.1-A per-fold OOF screener, seed 364101); E2 = H381 E2L (S60) or
E2R (R60) (S3fz per-fold screener, seed 364101). own = X0's own pool (the primary, deployment-only contrast vs the
stored X0 AGG.g50.CV3@12); P2 = X0 + E2 (secondary).

  deploy  --panel DEV2|R6|S60|R60 --run-dir R --tag T --sources X0[,E2] --library 0|1 --cores L --jobs J
  analyze --run-dir R --tag T --panel P --out F       (numpy only; reads stage records + the stored X0 records)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

SEED = 364101
X0_RUN = Path("/home/ubuntu/jope-data/h335-cl/runs/H364_C_xbabe0")
X0_KEY = "AGG.g50.CV3@12"
NEURAL = {"X0": "/home/ubuntu/jope-data/h335-cl/neural/runs/H0.1-A",
          "E2": "/home/ubuntu/jope-data/h335-cl/neural/runs/H381-E2-S3fz"}
BOOT_SEED = 410150


def cmd_deploy(a) -> None:
    import deploy_h410 as DP  # FIRST import of the GP stack: h335_core pins + guard

    GPH, K, C, L, D = DP.GPH, DP.K, DP.C, DP.L, DP.D
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    srcs = a.sources.split(",")
    cids = GPH.panel(a.panel)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(NEURAL[s]) for s in srcs}
    rt, scr, man = DP.build_rt_t(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (SEED,), nd)
    pools = {"P1": ["X0"]} if srcs == ["X0"] else {"P1": ["X0"], "P2": srcs}
    cfg = {"tag": a.tag, "panel": a.panel, "cids": cids, "sources": srcs, "library": bool(a.library),
           "pools": pools, "own_source": "X0", "timeout_s": a.timeout, "seeds": [SEED],
           "code_sha256": K.code_manifest(DP.code_files() + ("train_h410.py",)), "screener_manifest": man}
    p = run_dir / f"config_{a.tag}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items()):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)
    GPH.install(0)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(lambda c, _s: DP.deploy_unit(rt, a.tag, c, scr, (SEED,), bool(a.library), "X0", pools,
                                                       a.inner_jobs), [(c, 0) for c in order], a.jobs)
    st = {"tag": a.tag, "units": len(recs), "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DP.physical_by_kind(rt), "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{a.tag}_{int(t0)}.json", st)
    rt.engine.close()
    print(json.dumps({"tag": a.tag, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"]}))


def _last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def cmd_analyze(a) -> None:
    sys.path.insert(0, "/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/"
                       "h335_aggressive_locked_elo")
    import numpy as np
    import h335_stats as H

    pj = json.loads(Path(a.panels).read_text())
    cids = [c for n in a.panel.split("+") for c in pj[n]["cids"]]
    fams = [f for n in a.panel.split("+") for f in pj[n]["families"]]
    fam_of = dict(zip(cids, fams))
    x0 = {c: float(_last_result(X0_RUN / "stages" / "C_AGG" / f"{c}__{SEED}.jsonl")["outputs"][X0_KEY]["value"])
          for c in cids}
    recs = {c: _last_result(Path(a.run_dir) / "stages" / f"H410_{a.tag}" / f"{c}__pool.jsonl") for c in cids}
    keys = sorted(set.intersection(*[set(r["outputs"]) for r in recs.values()]))
    replay = {c: recs[c]["outputs"][f"X0.s{SEED}.stock"]["value"] - x0[c] for c in cids}
    rep = {"schema": "jope.h410.train-analysis/v1", "panel": a.panel, "tag": a.tag, "n": len(cids),
           "replay_max_abs_delta_vs_stored_x0": max(abs(v) for v in replay.values()), "outputs": {}}
    rng_seed = BOOT_SEED
    for k in keys:
        kk = k.replace(f"own.X0.s{SEED}.", "own.")
        vals = {c: float(recs[c]["outputs"][k]["value"]) for c in cids}
        pt = H.primary_test(vals, x0, fam_of)
        hl = H.headline(vals, x0, fam_of)
        _, Dd = H.family_contrast(vals, x0, fam_of, clip=H.CLIP)
        rng = np.random.default_rng(rng_seed)
        lcb = float(np.quantile(Dd[rng.integers(0, len(Dd), size=(10000, len(Dd)))].mean(axis=1), 0.025))
        rep["outputs"][kk] = {"family_mean_gain": hl["family_mean_gain"], "pct_rmse_reduction":
                              hl["pct_rmse_reduction"], "wtl": [hl["families_won"], hl["families_tied"],
                                                                hl["families_lost"]],
                              "primary_test": pt, "boot_lcb95_clipped": lcb,
                              "invalid": sum(1 for c in cids if recs[c]["outputs"][k].get("invalid"))}
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"replay max |u - stored X0| = {rep['replay_max_abs_delta_vs_stored_x0']:.3g}")
    for k, r in sorted(rep["outputs"].items(), key=lambda kv: -kv[1]["family_mean_gain"]):
        pt = r["primary_test"]
        print(f"{k:28s} gain {r['family_mean_gain']:+.4f} rmse {100 * r['pct_rmse_reduction']:+.2f}% "
              f"W/T/L {'/'.join(map(str, r['wtl']))} clip {pt['T_clipped_family_mean']:+.4f} p {pt['signflip_p']:.3f} "
              f"lofo {pt['lofo_min']:+.4f} pass {pt['pass']} lcb {r['boot_lcb95_clipped']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--panel", required=True, choices=("DEV2", "R6", "S60", "R60"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--sources", default="X0")
    p.add_argument("--library", type=int, default=0)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--inner-jobs", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p = sub.add_parser("analyze")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--panels", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"deploy": cmd_deploy, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
