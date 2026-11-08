#!/usr/bin/env python3
"""H415 GP runner (PREREG §3): the frozen H335 AGG recipe (= H381 `unit_lf` with lf_rows 0, which reproduced the stored
H364 X0 exactly in H381's REPLAY check) on the fresh TRAIN panel P201e, once per screener of the committee.

Only the panel, the search seed, the stage label and the screener directory differ from X0. No H317/H335/H381 file is
edited; gp_h381.py is imported unchanged (its published bytes) and its unit is called with lf_rows 0.

Commands (H335 sandbox; `panel` and `encode` on xbabe2, `launch`/`run` in sandbox_h335_docker.sh on xbabe0):
  panel                       write $H317_DATA/h415/P201e.json (create-once; reads only the TRAIN row index)
  encode [--threads N]        P201e SEARCH views with the frozen H317 recipe (DOPE sha checked before/after)
  idcheck --run-dir R         harness check on 2 stored S60 contexts: this wrapper, seed 364101, the H0.1-A screener and
                              a copy of the H364 memo must reproduce the stored X0 AGG.g50.CV3@12 exactly, 0 worker runs
  launch  --run-dir R --neural-run-dir D --cpus L --shards N --workers-per-shard W [--panel P201e|DEV2E]
  run     spawned by launch
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gp_h381 as GPH  # noqa: E402  (FIRST: h335_core pins + grammar group + guard; S3 scale registered)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

K, C, LAB, L, D, E = GPH.K, GPH.C, GPH.LAB, GPH.L, GPH.D, GPH.E

WAVE = "H415"
PHASE = "E415"  # stage label: records under stages/E415_AGG/
SEED = 415101  # one search seed for every committee arm (paired: the screener is the only difference)
GENS, CHECKPOINTS = 50, (50,)  # = the locked H335 recipe AGG.g50.CV3@12
CODE_FILES = LAB.PHASE_T_FILES + ("gp_h381.py", "gp_h415.py")
PANEL_FILE = Path(C.DATA_ROOT) / "h415" / "P201e.json"
X0_RUN = Path(C.DATA_ROOT) / "runs" / "H364_C_xbabe0"
X0_SEED, X0_KEY = 364101, "AGG.g50.CV3@12"

# gp_h381's unit reads these module globals at call time
GPH.WAVE, GPH.PHASE, GPH.SEED = WAVE, PHASE, SEED


# ----------------------------------------------------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------------------------------------------------
def panel_p201e(rows: Sequence[dict]) -> List[str]:
    """The FIFTH seeded context (h335_core._ctx_order) of every TRAIN family with >= 5 contexts; disjoint from P205
    (first), P201b (second), P201c (third) and P201d (fourth) by construction, asserted."""
    fam_ctx: Dict[str, List[str]] = {}
    for r in rows:
        fam_ctx.setdefault(r["family_id"], []).append(r["context_sha256"])
    order = K._ctx_order(fam_ctx)
    out = sorted(c[4] for c in order.values() if len(c) >= 5)
    used = set(C.pilot_shard(rows, C.family_folds(rows))) | set(K.panel_p201b(rows))
    used |= {c[2] for c in order.values() if len(c) >= 3} | {c[3] for c in order.values() if len(c) >= 4}
    if used & set(out):
        raise AssertionError("P201e overlaps P205, P201b, P201c or P201d")
    return out


def cmd_panel(_a) -> None:
    rows = C.train_rows()
    by = {r["context_sha256"]: r for r in rows}
    cids = panel_p201e(rows)
    rec = {"schema": "jope.h415.panel/v1", "panel": "P201e", "rule": "fifth seeded context of every TRAIN family with "
           ">= 5 contexts (h335_core._ctx_order); disjoint from P205, P201b, P201c and P201d", "n": len(cids),
           "cids": cids, "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(),
           "family_of": {c: by[c]["family_id"] for c in cids}, "dataset_of": {c: by[c]["dataset_id"] for c in cids},
           "families": len({by[c]["family_id"] for c in cids}), "guard": C.guard_report()}
    if rec["families"] != len(cids):
        raise AssertionError("P201e must hold one context per family")
    if PANEL_FILE.exists():
        old = json.loads(PANEL_FILE.read_text())
        if old["cids"] != cids:
            raise SystemExit(f"{PANEL_FILE} exists with a different panel")
        print(json.dumps({k: old[k] for k in ("n", "families", "cids_sha256")}))
        return
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(PANEL_FILE, rec)
    print(json.dumps({k: rec[k] for k in ("n", "families", "cids_sha256")}))


def family_of() -> Dict[str, str]:
    d = json.loads(PANEL_FILE.read_text())
    if d["cids"] != panel_p201e(C.train_rows()):
        raise AssertionError(f"{PANEL_FILE} differs from the recomputed P201e")
    return dict(d["family_of"])


def panel(name: str) -> List[str]:
    """P201e: the confirmation panel. DEV2E: the 2 P201d contexts H381 used as its never-scored smoke (DEV2), for the
    smoke of this wave's runner and deploy code (never scored here either). ID2: the first 2 S60 contexts (idcheck)."""
    if name == "P201e":
        return sorted(family_of())
    if name == "DEV2E":
        return GPH.panel("DEV2")
    if name == "ID2":
        return GPH.panel("S60")[:2]
    raise SystemExit(f"unknown panel {name}")


def fam_lookup(cids: Sequence[str]) -> Dict[str, str]:
    rows = {r["context_sha256"]: r["family_id"] for r in C.train_rows()}
    return {c: rows[c] for c in cids}


# ----------------------------------------------------------------------------------------------------------------------
# SEARCH views (screener inputs) for P201e
# ----------------------------------------------------------------------------------------------------------------------
def cmd_encode(a) -> None:
    import crop_shuffle_synthetic_views as V

    by = {r["context_sha256"]: r for r in C.train_rows()}
    cids = panel("P201e")
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
           "recipe": V.RECIPE_VERSION, "kind": "search", "panel": "P201e", "cids": cids, "encoded": done,
           "skipped_existing": skipped, "threads": a.threads, "elapsed_s": round(time.time() - t0, 1),
           "dope_sha256_before": enc.dope_sha_before, "dope_sha256_after": after,
           "dope_unchanged": after == enc.dope_sha_before == C.DOPE_SHA256, "guard": C.guard_report()}
    C.write_json(out / f"encode_receipt_h415_P201e_{os.uname().nodename}_{int(t0)}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "cids"}))
    if not rec["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


# ----------------------------------------------------------------------------------------------------------------------
# GP runs
# ----------------------------------------------------------------------------------------------------------------------
CONFIG_IDENTITY = ("cids", "seeds", "gens", "checkpoints", "lf_rows", "fr_k", "n_prop", "n_live", "stacks",
                   "worker_sha256", "timeout_s", "neural_run_dir", "code_sha256", "max_expanded_width")


def _config(run_dir: Path, a, cids: List[str], seed: int) -> dict:
    cfg = {"schema": "jope.h415.gp-config/v1", "wave": WAVE, "phase": PHASE, "panel": a.panel, "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG", "seeds": [seed],
           "gens": GENS, "checkpoints": list(CHECKPOINTS), "lf_rows": 0, "fr_k": 0, "n_prop": K.N_PROP,
           "n_live": K.N_LIVE, "stacks": list(LAB.STACKS), "worker_sha256": C.sha256_file(Path(a.worker)),
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
    GPH.install(0)
    GPH.SEED = cfg["seeds"][0]
    cids = L.shard_cids(cfg["cids"], a.shard)
    rt = LAB.build_rt(run_dir, "T", L.parse_cpus(a.cores), Path(a.worker), float(cfg["timeout_s"]), cfg["seeds"],
                      Path(cfg["neural_run_dir"]), a.torch_threads)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in cfg["seeds"]]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, cfg["gens"], cfg["checkpoints"], 0,
                                                   tuple(cfg["stacks"])), work, a.jobs)
    status = {"wave": WAVE, "phase": PHASE, "shard": a.shard, "records": len(recs), "wall_s": time.time() - t0,
              "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "guard": C.guard_report(),
              "bank_counters": dict(D.BANK_COUNTERS), "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / "shards" / f"status_{a.shard.replace('/', 'of')}_{int(t0)}.json", status)
    rt.engine.close()


def cmd_launch(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    cids = panel(a.panel)
    seed = X0_SEED if a.panel == "ID2" else SEED
    _config(run_dir, a, cids, seed)
    rt = LAB.build_rt(run_dir, "T", [0], Path(a.worker), a.timeout, (seed,), Path(a.neural_run_dir), 1)
    rt.engine.close()
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
        log = open(run_dir / "shards" / f"shard{k}of{a.shards}.log", "a")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [k for k, p, _ in procs if p.wait() != 0]
    for *_, log in procs:
        log.close()
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    print(json.dumps({"wave": WAVE, "launched_shards": a.shards, "contexts": len(cids), "seed": seed}))


def _last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def cmd_idcheck(a) -> None:
    """After `launch --panel ID2 --neural-run-dir .../H0.1-A` on a run dir whose memo is a copy of the H364 X0 memo for
    the 2 contexts: the outputs must equal the stored X0 records and the run must have made 0 physical worker runs."""
    run_dir = Path(a.run_dir)
    out = []
    for cid in panel("ID2"):
        mine = _last_result(run_dir / "stages" / f"{PHASE}_AGG" / f"{cid}__{X0_SEED}.jsonl")["outputs"][X0_KEY]
        x0 = _last_result(X0_RUN / "stages" / "C_AGG" / f"{cid}__{X0_SEED}.jsonl")["outputs"][X0_KEY]
        out.append({"cid": cid, "u": mine["value"], "x0_u": x0["value"], "delta": mine["value"] - x0["value"],
                    "spec_same": mine.get("spec") == x0.get("spec")})
    phys = 0
    for p in sorted((run_dir / "shards").glob("status_*.json")):
        phys += int(json.loads(p.read_text())["engine"].get("physical_runs", 0))
    ok = all(r["delta"] == 0.0 and r["spec_same"] for r in out) and phys == 0
    C.write_json(run_dir / "idcheck.json", {"schema": "jope.h415.idcheck/v1", "ok": ok, "physical_runs": phys,
                                            "contexts": out})
    print(json.dumps({"ok": ok, "physical_runs": phys, "contexts": out}))
    if not ok:
        raise SystemExit("idcheck failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panel")
    p = sub.add_parser("encode")
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("idcheck")
    p.add_argument("--run-dir", required=True)
    for name in ("launch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--timeout", type=float, default=60.0)
        p.add_argument("--torch-threads", type=int, default=1)
        if name == "launch":
            p.add_argument("--neural-run-dir", required=True)
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
            p.add_argument("--panel", default="P201e", choices=("P201e", "DEV2E", "ID2"))
        else:
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
    a = ap.parse_args()
    {"panel": cmd_panel, "encode": cmd_encode, "idcheck": cmd_idcheck, "launch": cmd_launch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
