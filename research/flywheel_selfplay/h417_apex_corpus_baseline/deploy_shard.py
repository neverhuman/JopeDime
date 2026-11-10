#!/usr/bin/env python3
"""H417 deploy as context-disjoint shards of ONE run (operational addendum, 2026-11-10).

A single train_apex.py deploy process is bound by its Python thread (search replays and stacking), not by the worker
cores: it finished 23 of 221 contexts in the first hour with two thirds of its 96 cores idle. This entry runs the same
unit function (deploy_h416.h416_unit, mode confirm, pinned design) with the same config, run directory and tag, on
the contexts order[k::n] of the size-sorted panel. Shards partition by context, so every context's memo and stage
files have one writing process (the engine's documented multi-process mode, as the search arms use it). Contexts the
first process already finished are store hits. The run config must already exist and must equal this process's config.

  deploy_shard.py --panel P --run-dir R --tag T --cores L --shard k/n [--jobs 4] [--inner-jobs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_apex as TA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--cores", required=True)
    ap.add_argument("--shard", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--inner-jobs", type=int, default=3); ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    import apex_core as A  # FIRST import of the GP stack: h335_core pins + guard

    DH, K, C, L, D = A.DH, A.K, A.C, A.L, A.D
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    if a.panel == "ID2B":
        raise SystemExit("the identity panel is not sharded")
    A.install()
    cids, seed = A.panel(a.panel), A.SEARCH_SEED
    design = TA._design()
    run_dir = Path(a.run_dir)
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / TA.NEURAL[s] for s in ("GS", "GE", "GL")}
    rt, scr, man = A.build_rt(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (seed,), nd)
    cfg = {"wave": "H417", "tag": a.tag, "mode": "confirm", "panel": a.panel, "cids": cids, "sources": list(nd),
           "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": a.timeout, "design": design,
           "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                    "max_members": DH.M.MAX_MEMBERS},
           "manifest_sha256": A.MANIFEST_SHA256,
           "code_sha256": K.code_manifest(A.code_files() + ("gp_apex.py", "train_apex.py")),
           "screener_manifest": man}
    p = run_dir / f"config_{a.tag}.json"
    if not p.exists():
        raise SystemExit("no run config: start the run with train_apex.py deploy")
    old = json.loads(p.read_text())
    bad = [k for k, v in cfg.items() if old.get(k) != json.loads(json.dumps(v))]
    if bad:
        raise SystemExit(f"run config mismatch on {bad}")
    k, n = (int(x) for x in a.shard.split("/"))
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    mine = order[k::n]
    DH.GPH.install(0)
    npz_dir = run_dir / f"h416_{a.tag}_npz"
    t0 = time.time()
    recs = L.run_parallel(lambda c, _s: DH.h416_unit(rt, a.tag, c, scr, seed, "confirm", npz_dir, design,
                                                     a.inner_jobs), [(c, 0) for c in mine], a.jobs)
    st = {"tag": a.tag, "mode": "confirm", "shard": a.shard, "units": len(recs), "wall_s": time.time() - t0,
          "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "akf_ledger": DH.DP15.KF_LEDGER.check(),
          "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DH.DP.physical_by_kind(rt), "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{a.tag}_{int(t0)}_s{k}of{n}.json", st)
    print(json.dumps({"tag": a.tag, "shard": a.shard, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"]}))
    rt.engine.close()


if __name__ == "__main__":
    main()
