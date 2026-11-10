#!/usr/bin/env python3
"""H420 GX arm: the H417 GL search (locked H335 AGG recipe, H415-L screener, seed 417101, 50 generations, K = 6 live of
P = 48 screened, superset worker 2a848c8f, 60 s limit) with the extended generators of lang_h420 switched on. It
differs from H417's GL arm in the generators only, so `GS + GE + GX` against `GS + GE + GL` is a replace-one-run test
at equal live evaluations.

  probe   --run-dir R --cid C [--n 40]   worker acceptance: every library-extension preset and n extended programs,
                                         SEARCH evaluation on one TRAIN context; prints status counts and seconds
  launch  --run-dir R --panel P --cpus L --shards N --workers-per-shard W
  run     (one shard; called by launch)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_core as A  # noqa: E402  (FIRST: the frozen stack, pins and guard)

import argparse  # noqa: E402

import lang_h420 as LANG  # noqa: E402

GPH, K, C, L, LAB, D, G = A.GPH, A.K, A.C, A.L, A.LAB, A.D, A.G

WAVE = "H420"
PHASE = "E420"  # stage label: records under stages/E420_AGG/
SEED = A.SEARCH_SEED
GENS, CHECKPOINTS = 50, (50,)
ARM, SCREENER = "GX", "H415-L"
CODE_FILES = A.code_files() + ("lang_h420.py", "gp_h420.py")
CONFIG_IDENTITY = ("panel", "arm", "cids", "seeds", "gens", "checkpoints", "lf_rows", "fr_k", "n_prop", "n_live",
                   "stacks", "worker_sha256", "timeout_s", "neural_run_dir", "code_sha256", "max_expanded_width",
                   "manifest_sha256", "lang")

GPH.WAVE, GPH.PHASE, GPH.SEED = WAVE, PHASE, SEED


def lang_identity() -> dict:
    return {"mode": "gx", "p_ext_head": LANG.P_EXT_HEAD, "p_ext_stage": LANG.P_EXT_STAGE,
            "histgbm2_work": LANG.EXT_HISTGBM2_WORK, "periodic_max_width": LANG.PERIODIC_MAX_WIDTH,
            "gpr_kernels": list(LANG.GPR_KERNELS),
            "map_exprs": list(LANG.MAP_EXPRS), "derive_forms": list(LANG.DERIVE_FORMS)}


def _worker_default() -> str:
    return str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085")


def _neural() -> Path:
    return Path(C.DATA_ROOT) / "neural" / "runs" / SCREENER


def _config(run_dir: Path, a, cids: List[str]) -> dict:
    cfg = {"schema": "jope.h420.gp-config/v1", "wave": WAVE, "phase": PHASE, "panel": a.panel, "arm": ARM,
           "cids": cids, "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG",
           "seeds": [SEED], "gens": GENS, "checkpoints": list(CHECKPOINTS), "lf_rows": 0, "fr_k": 0,
           "n_prop": K.N_PROP, "n_live": K.N_LIVE, "stacks": list(LAB.STACKS),
           "worker_sha256": C.sha256_file(Path(a.worker)), "timeout_s": a.timeout,
           "neural_run_dir": str(_neural()), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "manifest_sha256": A.MANIFEST_SHA256,
           "lang": lang_identity(), "host": os.uname().nodename}
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
    if K.code_manifest(CODE_FILES) != cfg["code_sha256"] or cfg["lang"] != lang_identity():
        raise SystemExit("code or language settings differ from the run config")
    GPH.install(0)
    GPH.SEED = cfg["seeds"][0]
    LANG.install(G, all_threads=True)
    cids = L.shard_cids(cfg["cids"], a.shard)
    rt, _, man = A.build_rt(run_dir, L.parse_cpus(a.cores), Path(a.worker), float(cfg["timeout_s"]), cfg["seeds"],
                            {ARM: Path(cfg["neural_run_dir"])}, a.torch_threads)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in cfg["seeds"]]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, cfg["gens"], cfg["checkpoints"], 0,
                                                   tuple(cfg["stacks"])), work, a.jobs)
    status = {"wave": WAVE, "phase": PHASE, "arm": ARM, "shard": a.shard, "records": len(recs),
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
    LANG.install(G)
    print(json.dumps({"lang_selfcheck": LANG.selfcheck(G)}))
    cids = A.panel(a.panel)
    missing = [c for c in cids if not (Path(C.DATA_ROOT) / "views-search" / f"{c}.npz").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no SEARCH view set (encode first)")
    if not list(_neural().glob("seed*/fold0/model.pt")):
        raise SystemExit(f"screener {SCREENER} has no fold-0 models here")
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
    print(json.dumps({"wave": WAVE, "arm": ARM, "panel": a.panel, "launched_shards": a.shards,
                      "contexts": len(cids), "seed": SEED}))


def cmd_probe(a) -> None:
    """Worker acceptance on one TRAIN context (SEARCH evaluation only; nothing is scored or kept for a readout)."""
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    GPH.install(0)
    LANG.install(G)
    print(json.dumps({"lang_selfcheck": LANG.selfcheck(G)}))
    rt, _, _ = A.build_rt(run_dir, L.parse_cpus(a.cores), Path(a.worker), a.timeout, (SEED,), {ARM: _neural()})
    if a.cid in A.rows() or a.cid not in rt.fam_of:
        raise SystemExit("probe runs on a TRAIN context")
    ctx = rt.engine.context(a.cid, rt.fam_of[a.cid])
    progs = [(n, d) for n, d in LANG.LIB_EXT]
    rng = random.Random(420)
    with LANG.gx():
        while len(progs) < len(LANG.LIB_EXT) + a.n:
            g = G.op_synth(rng.randrange(3), rng)
            g = G.op_insert_stage(g, rng) if rng.random() < 0.5 else g
            if isinstance(g, G.Pipe) and (any(s.op in LANG.NEW_STAGE_OPS for s in g.stages) or rng.random() < 0.3):
                progs.append((f"synth{len(progs)}", G.canonical(g)))
        cands, feat_fail = [], 0
        for n, d in progs:
            c = G.safe_candidate(G.parse(d), "h420_probe:" + n, (), ctx.width, cost_exempt=True)
            if c is None:
                print(json.dumps({"name": n, "dsl": d, "status": "refused_by_grammar"}))
                continue
            try:
                LANG._STOCK["featurize"](c.dsl)
            except Exception as e:  # the wrapper falls back; count how often the Rust featurizer refuses
                feat_fail += 1
                G.featurize(c.dsl)
                print(json.dumps({"name": n, "featurizer": str(e)[:160]}))
            cands.append((n, c))
    t0 = time.time()
    res = rt.engine.search(ctx, [c for _n, c in cands])
    out = {}
    for (n, c), r in zip(cands, res):
        st = getattr(r, "status", None) or ("ok" if getattr(r, "utility", None) is not None else "none")
        out[st] = out.get(st, 0) + 1
        print(json.dumps({"name": n, "dsl": c.dsl[:150], "status": st,
                          "seconds": round(float(r.wall_s or 0.0), 2), "gate": bool(r.gate),
                          "error": (str(getattr(r, "error", "") or "")[:200])}))
    print(json.dumps({"probe": a.cid[:12], "width": ctx.width, "n": len(cands), "status_counts": out,
                      "featurizer_refusals": feat_fail, "wall_s": round(time.time() - t0, 1)}))
    rt.engine.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=_worker_default())
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--torch-threads", type=int, default=1)
        p.add_argument("--timeout", type=float, default=60.0)
        if name == "launch":
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
            p.add_argument("--panel", required=True)
        elif name == "run":
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
        else:
            p.add_argument("--cid", required=True)
            p.add_argument("--cores", required=True)
            p.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    {"launch": cmd_launch, "run": cmd_run, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
