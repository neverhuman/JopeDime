#!/usr/bin/env python3
"""H417 GP arms on an APEX panel: the locked H335 AGG search (seed F + Evo A K = 6 live / P = 48 screened, 50
generations; stock AGG.g50.{single, CV3@12}) once per committee screener, exactly as H416's GE / GL arms ran on P201b
(unit = gp_h381.unit_lf with lf_rows 0; superset worker 2a848c8f; 60 s per-evaluation limit; 1 thread per worker).

Differences from H416, all forced by the new families: the runtime is apex_core.build_rt (fold-0 screener pair for
every APEX target, as on dev91), all three arms GS / GE / GL are new runs (no stored Phase T run exists here), and the
search seed is 417101.

  encode  --panel P [--threads N]   SEARCH-row DOPE view sets of the panel -> $H317_DATA/views-search/<cid>.npz
                                    (xbabe2 bwrap sandbox with H335_VIEWS_RW=1; frozen H317 recipe; DOPE sha checked)
  launch  --run-dir R --arm GS|GE|GL --neural-run-dir D --cpus L --shards N --workers-per-shard W --panel P
  run     (one shard; called by launch)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_core as A  # noqa: E402  (FIRST: the frozen stack, pins and guard)

import argparse  # noqa: E402

GPH, K, C, L, LAB, D = A.GPH, A.K, A.C, A.L, A.LAB, A.D

WAVE = "H417"
PHASE = "E417"  # stage label: records under stages/E417_AGG/
SEED = A.SEARCH_SEED
GENS, CHECKPOINTS = 50, (50,)
ARMS = ("GS", "GE", "GL")
CODE_FILES = A.code_files() + ("gp_apex.py",)
CONFIG_IDENTITY = ("panel", "arm", "cids", "seeds", "gens", "checkpoints", "lf_rows", "fr_k", "n_prop", "n_live",
                   "stacks", "worker_sha256", "timeout_s", "neural_run_dir", "code_sha256", "max_expanded_width",
                   "manifest_sha256")

GPH.WAVE, GPH.PHASE, GPH.SEED = WAVE, PHASE, SEED


def cmd_encode(a) -> None:
    import crop_shuffle_synthetic_views as V

    A.install()
    r = A.rows()
    cids = A.panel(a.panel)
    out = V.VIEW_SEARCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = V.DopeViewEncoder(threads=a.threads)
    done = skipped = 0
    for cid in cids:
        p = out / f"{cid}.npz"
        if V._view_file_ok(p):
            skipped += 1
            continue
        V._save_npz(p, V.encode_context(enc, cid, r[cid], "search"))
        done += 1
    after = enc.dope_sha()
    rec = {"schema": "jope.h335.search-view-encode/v1", "wave": WAVE, "host": os.uname().nodename,
           "recipe": V.RECIPE_VERSION, "kind": "search", "panel": a.panel, "cids": cids, "encoded": done,
           "skipped_existing": skipped, "threads": a.threads, "elapsed_s": round(time.time() - t0, 1),
           "dope_sha256_before": enc.dope_sha_before, "dope_sha256_after": after,
           "dope_unchanged": after == enc.dope_sha_before == C.DOPE_SHA256, "guard": C.guard_report()}
    C.write_json(out / f"encode_receipt_h417_{a.panel}_{os.uname().nodename}_{int(t0)}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "cids"}))
    if not rec["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


def _config(run_dir: Path, a, cids: List[str]) -> dict:
    cfg = {"schema": "jope.h417.gp-config/v1", "wave": WAVE, "phase": PHASE, "panel": a.panel, "arm": a.arm,
           "cids": cids, "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG",
           "seeds": [SEED], "gens": GENS, "checkpoints": list(CHECKPOINTS), "lf_rows": 0, "fr_k": 0,
           "n_prop": K.N_PROP, "n_live": K.N_LIVE, "stacks": list(LAB.STACKS),
           "worker_sha256": C.sha256_file(Path(a.worker)), "timeout_s": a.timeout,
           "neural_run_dir": str(a.neural_run_dir), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "manifest_sha256": A.MANIFEST_SHA256,
           "host": os.uname().nodename}
    p = run_dir / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        for k in CONFIG_IDENTITY:
            if old.get(k) != cfg[k]:
                raise SystemExit(f"run config mismatch on {k}")
        return old
    C.write_json(p, cfg)
    return cfg


def cmd_run(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if C.sha256_file(Path(a.worker)) != cfg["worker_sha256"]:
        raise SystemExit("worker differs from the run config")
    if K.code_manifest(CODE_FILES) != cfg["code_sha256"]:
        raise SystemExit("code differs from the run config")
    GPH.install(0)
    GPH.SEED = cfg["seeds"][0]
    cids = L.shard_cids(cfg["cids"], a.shard)
    rt, _, man = A.build_rt(run_dir, L.parse_cpus(a.cores), Path(a.worker), float(cfg["timeout_s"]), cfg["seeds"],
                            {cfg["arm"]: Path(cfg["neural_run_dir"])}, a.torch_threads)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in cfg["seeds"]]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, cfg["gens"], cfg["checkpoints"], 0,
                                                   tuple(cfg["stacks"])), work, a.jobs)
    status = {"wave": WAVE, "phase": PHASE, "arm": cfg["arm"], "shard": a.shard, "records": len(recs),
              "wall_s": time.time() - t0, "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(),
              "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS), "screener_manifest": man,
              "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / "shards" / f"status_{a.shard.replace('/', 'of')}_{int(t0)}.json", status)
    rt.engine.close()


def cmd_launch(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    A.install()
    cids = A.panel(a.panel)
    missing = [c for c in cids if not (Path(C.DATA_ROOT) / "views-search" / f"{c}.npz").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no SEARCH view set (encode first)")
    _config(run_dir, a, cids)
    cpus, w = L.parse_cpus(a.cpus), a.workers_per_shard
    if a.shards * w > len(cpus):
        raise SystemExit("not enough CPUs for shards x workers")
    procs = []
    for k in range(a.shards):
        mine = ",".join(str(c) for c in cpus[k * w:(k + 1) * w])
        env = dict(os.environ, H335_GROUP="AGG")
        env.pop("H317_LEARNERS_V1", None)
        cmd = ["taskset", "-c", mine, "nice", "-n", "19", sys.executable, "-u", str(Path(__file__).resolve()), "run",
               "--run-dir", str(run_dir), "--shard", f"{k}/{a.shards}", "--cores", mine, "--jobs", str(a.jobs),
               "--worker", a.worker, "--torch-threads", str(a.torch_threads)]
        log = open(run_dir / "shards" / f"shard{k}of{a.shards}.log", "a")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [k for k, p, _ in procs if p.wait() != 0]
    for *_, log in procs:
        log.close()
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    print(json.dumps({"wave": WAVE, "arm": a.arm, "panel": a.panel, "launched_shards": a.shards,
                      "contexts": len(cids), "seed": SEED}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode")
    p.add_argument("--panel", required=True)
    p.add_argument("--threads", type=int, default=8)
    for name in ("launch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--torch-threads", type=int, default=1)
        if name == "launch":
            p.add_argument("--arm", required=True, choices=ARMS)
            p.add_argument("--neural-run-dir", required=True)
            p.add_argument("--timeout", type=float, default=60.0)
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
            p.add_argument("--panel", required=True)
        else:
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
    a = ap.parse_args()
    {"encode": cmd_encode, "launch": cmd_launch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
