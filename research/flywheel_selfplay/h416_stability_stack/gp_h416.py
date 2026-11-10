#!/usr/bin/env python3
"""H416 GP arms on TRAIN P201b (PREREG §4): the locked H335 AGG search (seed F + Evo A K = 6 live / P = 48 screened,
50 generations; stock AGG.g50.{single, CV3@12}) once per committee screener, at the seed of the stored H335 Phase T
P201b run (335101), so every arm is paired with it and the screener is the only difference.

  panel                       write $H317_DATA/h416/P201b.json (create-once; TRAIN row index + the staged P201b board
                              BOARD.json, which must list the same 201 contexts / families)
  launch  --run-dir R --neural-run-dir D --cpus L --shards N --workers-per-shard W [--panel P201b|ID2B]
  run     (one shard; called by launch)
  idcheck --run-dir R         after `launch --panel ID2B` with H0.1-A on a run dir whose memo is a copy of the stored
                              H335 Phase T memo of the 2 ID2B contexts: the outputs must equal the stored Phase T
                              records (= the P201b board scores of h335.H335_AGG) and the run must make 0 worker runs

GS (H0.1-A) is never searched again on P201b: the deploy unit replays the stored Phase T run from its memo (0 worker
runs). GE (H381-E2-S3fz) and GL (H415-L) are new runs. The unit is gp_h381.unit_lf with lf_rows 0, exactly as in
H415 (gp_h415.py imported at its published bytes for the pins, the panel helpers and the code manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gp_h415 as GP15  # noqa: E402  (FIRST: gp_h381 -> h335_core pins + guard)

GPH, K, C, L, LAB, D = GP15.GPH, GP15.K, GP15.C, GP15.L, GP15.LAB, GP15.D

WAVE = "H416"
PHASE = "E416"  # stage label: records under stages/E416_AGG/
SEED = 335101  # = the stored H335 Phase T P201b run (h335.H335_AGG on the P201b board)
GENS, CHECKPOINTS = 50, (50,)
KEY = "AGG.g50.CV3@12"
CODE_FILES = GP15.CODE_FILES + ("gp_h416.py",)
PANEL_FILE = Path(C.DATA_ROOT) / "h416" / "P201b.json"
BOARD_FILE = HERE / "board_p201b_BOARD.json"  # staged copy of research/elo/reference-board/boards/p201b-.../BOARD.json
BOARD_X0 = HERE / "board_p201b__h335.H335_AGG.jsonl"  # staged copy of the board's h335.H335_AGG scores
# The stored H335 Phase T P201b run lives in two run dirs (T_xbabe0 on xbabe0: 150 contexts; T_xbabe2 on xbabe2: 51).
# run_h416.sh tprep makes one create-once, sha-verified copy of both on xbabe0 (per context: memo + T_AGG record).
T_RUNS = (Path(C.DATA_ROOT) / "runs" / "H416_GS",)
T_STAGE = "T_AGG"
SPEC_SELECTION = ("members", "w_audit", "w_fold0", "w_fold1", "islands", "config", "fallback_single")
ID_TOL = 1e-12

GPH.WAVE, GPH.PHASE, GPH.SEED = WAVE, PHASE, SEED


# ----------------------------------------------------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------------------------------------------------
def board_contexts() -> Dict[str, dict]:
    b = json.loads(BOARD_FILE.read_text())
    if b.get("board") != "p201b-train-audit-v1" or b.get("n_contexts") != 201:
        raise AssertionError("staged board file is not p201b-train-audit-v1 (201 contexts)")
    return {c["cid"]: c for c in b["contexts"]}


def panel_p201b(rows: Sequence[dict]) -> List[str]:
    """P201b = h335_core.panel_p201b (the SECOND seeded context of every TRAIN family with >= 2 contexts): the 201
    contexts of the p201b-train-audit-v1 board, asserted (same cids and families); disjoint from P201e (H416's
    development panel), asserted."""
    out = sorted(K.panel_p201b(rows))
    bc = board_contexts()
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    if out != sorted(bc) or any(bc[c]["family_id"] != fam[c] for c in out):
        raise AssertionError("panel_p201b differs from the P201b board contexts")
    if set(out) & set(GP15.panel_p201e(rows)):
        raise AssertionError("P201b overlaps P201e")
    return out


def cmd_panel(_a) -> None:
    rows = C.train_rows()
    by = {r["context_sha256"]: r for r in rows}
    cids = panel_p201b(rows)
    rec = {"schema": "jope.h416.panel/v1", "panel": "P201b", "rule": "h335_core.panel_p201b: second seeded context of "
           "every TRAIN family with >= 2 contexts = the p201b-train-audit-v1 board contexts; disjoint from P201e",
           "n": len(cids), "cids": cids, "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(),
           "family_of": {c: by[c]["family_id"] for c in cids}, "dataset_of": {c: by[c]["dataset_id"] for c in cids},
           "families": len({by[c]["family_id"] for c in cids}),
           "board_envelope_sha256": {c: v["envelope_sha256"] for c, v in board_contexts().items()},
           "guard": C.guard_report()}
    if rec["families"] != len(cids):
        raise AssertionError("P201b must hold one context per family")
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
    if d["cids"] != panel_p201b(C.train_rows()):
        raise AssertionError(f"{PANEL_FILE} differs from the recomputed P201b")
    return dict(d["family_of"])


def panel(name: str) -> List[str]:
    """P201b: the confirmation panel. ID2B: its first 2 contexts (sorted), for the Phase T replay identity check."""
    if name == "P201b":
        return sorted(family_of())
    if name == "ID2B":
        return sorted(family_of())[:2]
    raise SystemExit(f"unknown panel {name}")


def t_run_of(cid: str) -> Path:
    """The stored H335 Phase T run dir holding cid's seed-335101 AGG record."""
    hits = [r for r in T_RUNS if (r / "stages" / T_STAGE / f"{cid}__{SEED}.jsonl").is_file()]
    if len(hits) != 1:
        raise SystemExit(f"{cid}: {len(hits)} Phase T runs hold a record")
    return hits[0]


def _last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def stored_t(cid: str) -> dict:
    return _last_result(t_run_of(cid) / "stages" / T_STAGE / f"{cid}__{SEED}.jsonl")["outputs"][KEY]


def board_x0() -> Dict[str, float]:
    recs = [json.loads(x) for x in BOARD_X0.read_text().splitlines() if x.strip()]
    if {r["model_id"] for r in recs} != {"h335.H335_AGG"} or len(recs) != 201:
        raise AssertionError("staged h335.H335_AGG board scores are not 201 records")
    return {r["cid"]: float(r["utility"]) for r in recs}


# ----------------------------------------------------------------------------------------------------------------------
# GP runs (gp_h415's launch / run with the H416 panel, phase and seed)
# ----------------------------------------------------------------------------------------------------------------------
def _config(run_dir: Path, a, cids: List[str], seed: int) -> dict:
    cfg = {"schema": "jope.h416.gp-config/v1", "wave": WAVE, "phase": PHASE, "panel": a.panel, "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG", "seeds": [seed],
           "gens": GENS, "checkpoints": list(CHECKPOINTS), "lf_rows": 0, "fr_k": 0, "n_prop": K.N_PROP,
           "n_live": K.N_LIVE, "stacks": list(LAB.STACKS), "worker_sha256": C.sha256_file(Path(a.worker)),
           "timeout_s": a.timeout, "neural_run_dir": str(a.neural_run_dir), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "host": os.uname().nodename}
    p = run_dir / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        for k in GP15.CONFIG_IDENTITY:
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
    _config(run_dir, a, cids, SEED)
    rt = LAB.build_rt(run_dir, "T", [0], Path(a.worker), a.timeout, (SEED,), Path(a.neural_run_dir), 1)
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
    print(json.dumps({"wave": WAVE, "launched_shards": a.shards, "contexts": len(cids), "seed": SEED}))


def cmd_idcheck(a) -> None:
    run_dir = Path(a.run_dir)
    bx = board_x0()
    out = []
    for cid in panel("ID2B"):
        mine = _last_result(run_dir / "stages" / f"{PHASE}_AGG" / f"{cid}__{SEED}.jsonl")["outputs"][KEY]
        t = stored_t(cid)
        ms, ts = mine.get("spec") or {}, t.get("spec") or {}
        out.append({"cid": cid, "u": mine["value"], "t_u": t["value"], "board_u": bx[cid],
                    "delta": mine["value"] - t["value"], "board_delta": mine["value"] - bx[cid],
                    "spec_same": ms == ts, "selection_same": all(ms.get(k) == ts.get(k) for k in SPEC_SELECTION)})
    phys = 0
    for p in sorted((run_dir / "shards").glob("status_*.json")):
        phys += int(json.loads(p.read_text())["engine"].get("physical_runs", 0))
    # PREREG §4: same selection (members, weights, islands, config) and |du| <= 1e-12. Contexts whose Phase T record
    # was made on xbabe2 replay on xbabe0 with ulp-level drift in the stack's SEARCH-fitness floats.
    ok = all(abs(r["delta"]) <= ID_TOL and abs(r["board_delta"]) <= ID_TOL and r["selection_same"]
             for r in out) and phys == 0
    C.write_json(run_dir / "idcheck.json", {"schema": "jope.h416.idcheck/v1", "ok": ok, "physical_runs": phys,
                                            "contexts": out})
    print(json.dumps({"ok": ok, "physical_runs": phys, "contexts": out}))
    if not ok:
        raise SystemExit("idcheck failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panel")
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
            p.add_argument("--panel", default="P201b", choices=("P201b", "ID2B"))
        else:
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
    a = ap.parse_args()
    {"panel": cmd_panel, "idcheck": cmd_idcheck, "launch": cmd_launch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
