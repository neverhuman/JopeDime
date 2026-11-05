#!/usr/bin/env python3
"""H357 GP side (H357 PREREG §3): the FROZEN H335 AGG recipe on the fresh TRAIN panel P201c.

Code root: the frozen H335 files (byte-identical to /home/ubuntu/h335-stage/code; H335 on origin/v1/next) plus this
file, bound at the H335 source path by the H335 sandboxes (H335_CODE= for sandbox_h335.sh on xbabe2;
H335_STAGE=<dir holding code/> for sandbox_h335_docker.sh on xbabe0). Only H335 functions run the search:
h335_lab.build_rt (Phase T TRAIN runtime: fold-j OOF screener, superset worker, CPU-limited runner) and
h335_lab.run_units -> unit_agg (Seed F -> Evo A K = 6 / P = 48 -> AGG.g50.{single, CV3@12}). The only differences from
H335 Phase T: the panel (P201c), the search seed (357101), G = 50 with the one checkpoint 50, AGG only (no BASE group),
and the stage label "C" (records under stages/C_AGG/).

Commands:
  panel   xbabe2, sandbox_h335.sh                        -> $H317_DATA/h357/P201c.json (cids, families; disjointness)
  encode  xbabe2, H335_VIEWS_RW=1 sandbox_h335.sh        -> $H317_DATA/views-search/<cid>.npz for P201c (+ receipt)
  launch  xbabe0, sandbox_h335_docker.sh (H335_GROUP=AGG) --run-dir R --cpus LIST --shards N --workers-per-shard W
  run     spawned by launch, one per shard
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402  (FIRST: pins, grammar group, guard)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

import h335_lab as LAB  # noqa: E402

C, D, L = K.C, K.D, K.L

WAVE = "H357"
PHASE = "C"
SEEDS_C = (357101,)
GENS_C = 50
CHECKPOINTS_C = (50,)
STACKS_C = LAB.STACKS  # ("single", "CV3@12"): the H335 output pair; H357 reads AGG.g50.CV3@12
CODE_FILES = LAB.PHASE_T_FILES + ("gp_c_h357.py",)
PANEL_FILE = Path(C.DATA_ROOT) / "h357" / "P201c.json"


def panel_p201c(rows: Sequence[dict]) -> List[str]:
    """The THIRD seeded context of every TRAIN family with >= 3 contexts, in the seeded order of P205 (first) and
    P201b (second) (h335_core._ctx_order); disjoint from both, asserted."""
    fam_ctx: Dict[str, List[str]] = {}
    for r in rows:
        fam_ctx.setdefault(r["family_id"], []).append(r["context_sha256"])
    out = sorted(c[2] for c in K._ctx_order(fam_ctx).values() if len(c) >= 3)
    p205 = set(C.pilot_shard(rows, C.family_folds(rows)))
    p201b = set(K.panel_p201b(rows))
    if (p205 | p201b) & set(out):
        raise AssertionError("P201c overlaps P205 or P201b")
    return out


def cmd_panel(_a) -> None:
    rows = C.train_rows()
    by = {r["context_sha256"]: r for r in rows}
    cids = panel_p201c(rows)
    rec = {"schema": "jope.h357.panel/v1", "panel": "P201c", "rule": "third seeded context of every TRAIN family with "
           ">= 3 contexts (h335_core._ctx_order); disjoint from P205 and P201b", "n": len(cids), "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(),
           "family_of": {c: by[c]["family_id"] for c in cids}, "dataset_of": {c: by[c]["dataset_id"] for c in cids},
           "families": len({by[c]["family_id"] for c in cids}), "guard": C.guard_report()}
    if rec["families"] != len(cids):
        raise AssertionError("P201c must hold one context per family")
    if PANEL_FILE.exists():
        old = json.loads(PANEL_FILE.read_text())
        if old["cids"] != cids:
            raise SystemExit(f"{PANEL_FILE} exists with a different panel")
        print(json.dumps({k: old[k] for k in ("n", "families", "cids_sha256")}))
        return
    C.write_json(PANEL_FILE, rec)
    print(json.dumps({k: rec[k] for k in ("n", "families", "cids_sha256")}))


def panel_cids() -> List[str]:
    cids = json.loads(PANEL_FILE.read_text())["cids"]
    if cids != panel_p201c(C.train_rows()):
        raise AssertionError(f"{PANEL_FILE} differs from the recomputed P201c")
    return cids


def cmd_encode(a) -> None:
    """= h335_encode.main for P201c (the frozen H317 view recipe, DOPE sha checked before/after)."""
    import crop_shuffle_synthetic_views as V

    rows = C.train_rows()
    by = {r["context_sha256"]: r for r in rows}
    cids = panel_cids()
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
        V._save_npz(p, V.encode_context(enc, cid, by[cid], "search"))
        done += 1
    after = enc.dope_sha()
    rec = {"schema": "jope.h335.search-view-encode/v1", "wave": WAVE, "host": os.uname().nodename,
           "recipe": V.RECIPE_VERSION, "kind": "search", "panel": "P201c", "cids": cids, "encoded": done,
           "skipped_existing": skipped, "threads": a.threads, "elapsed_s": round(time.time() - t0, 1),
           "dope_sha256_before": enc.dope_sha_before, "dope_sha256_after": after,
           "dope_unchanged": after == enc.dope_sha_before == C.DOPE_SHA256, "guard": C.guard_report()}
    C.write_json(out / f"encode_receipt_h357_P201c_{os.uname().nodename}_{int(t0)}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "cids"}))
    if not rec["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


CONFIG_IDENTITY = ("cids", "seeds", "gens", "checkpoints", "n_prop", "n_live", "stacks", "worker_sha256", "timeout_s",
                   "neural_run_dir", "code_sha256", "max_expanded_width")


def _config(run_dir: Path, a, cids: List[str]) -> dict:
    cfg = {"schema": "jope.h357.phase-c-config/v1", "wave": WAVE, "phase": PHASE, "panel": "P201c", "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG",
           "seeds": list(SEEDS_C), "gens": GENS_C, "checkpoints": list(CHECKPOINTS_C), "n_prop": K.N_PROP,
           "n_live": K.N_LIVE, "stacks": list(STACKS_C), "worker_sha256": C.sha256_file(Path(a.worker)),
           "timeout_s": a.timeout, "neural_run_dir": str(a.neural_run_dir), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "host": os.uname().nodename}
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
    cids = L.shard_cids(cfg["cids"], a.shard)
    rt = LAB.build_rt(run_dir, "T", L.parse_cpus(a.cores), Path(a.worker), float(cfg["timeout_s"]), cfg["seeds"],
                      Path(cfg["neural_run_dir"]), a.torch_threads)
    t0 = time.time()
    recs = LAB.run_units(rt, PHASE, cids, cfg["seeds"], cfg["gens"], cfg["checkpoints"], a.jobs, tuple(cfg["stacks"]))
    status = {"wave": WAVE, "phase": PHASE, "group": K.GROUP, "shard": a.shard, "records": len(recs),
              "wall_s": time.time() - t0, "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(),
              "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS), "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / "shards" / f"status_C_AGG_{a.shard.replace('/', 'of')}_{int(t0)}.json", status)
    rt.engine.close()


def cmd_launch(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    cids = panel_cids()
    _config(run_dir, a, cids)
    rt = LAB.build_rt(run_dir, "T", [0], Path(a.worker), a.timeout, SEEDS_C, Path(a.neural_run_dir), 1)
    rt.engine.close()  # screener manifest written once before any shard starts (H335 cmd_launch)
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
               "--timeout", str(a.timeout), "--worker", a.worker, "--torch-threads", str(a.torch_threads)]
        log = open(run_dir / "shards" / f"C_AGG_shard{k}of{a.shards}.log", "a")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [k for k, p, _ in procs if p.wait() != 0]
    for *_, log in procs:
        log.close()
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    print(json.dumps({"wave": WAVE, "phase": PHASE, "launched_shards": a.shards, "contexts": len(cids)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panel")
    p = sub.add_parser("encode")
    p.add_argument("--threads", type=int, default=8)
    for name in ("launch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--timeout", type=float, default=60.0)
        p.add_argument("--torch-threads", type=int, default=1)
        if name == "launch":
            p.add_argument("--neural-run-dir", default=str(Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"))
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
        else:
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
    a = ap.parse_args()
    {"panel": cmd_panel, "encode": cmd_encode, "launch": cmd_launch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
