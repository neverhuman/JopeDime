#!/usr/bin/env python3
"""H381 part B: dev91 five-context direction check of the final TRAIN arms (PREREG §5 + part-B amendment).

Guard: importing gp_h381 -> h335_core installs the H317 guard v2. This module points the Stage V unlock at its own
LOCK (DATA/stage_v/LOCK.json) and create-once receipt ns/receipts/H381_LOCK_sha256.txt (the H335 / H369 pattern):
`unlock` is one-shot, later commands use resume_stage_v. Sealed / holdout / DL-Test stay refused.
The LOCK's code_sha256 lists only code, the worker and screener models (no dev91 path: H369 attempt-1 lesson); the
dev91 inputs (tables, SEARCH views) are checked after the unlock against the H335 Stage V prep sha manifests whose own
file shas the LOCK records.

Commands (all in sandbox_h381_d91.sh on xbabe0):
  lock     --arms ARMS.json        write DATA/stage_v/LOCK.json (create-once); prints its sha256 (no dev91 read)
  unlock                           the one-shot C.unlock_stage_v()
  select                           resume; 5 dev91 contexts from 5 families (PREREG §5 rule) -> DATA/cids5.json
  run      --arm NAME --run-dir R --cores L --jobs J   resume; unit_lf on cids5 x seeds (335101, 335102), phase V
                                   runtime (fold-0 screener pair of the arm's neural dir, DL-Val SEARCH views)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gp_h381 as GPH  # noqa: E402  (FIRST: h335_core pins + guard; S3 registered)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

K, C, LAB, L, D = GPH.K, GPH.C, GPH.LAB, GPH.L, GPH.D

WAVE = "H381"
DATA = Path(os.environ.get("H381_D91", "/home/ubuntu/jope-data/h381-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H381_LOCK_sha256.txt")
VIEWS = DATA / "views-search"  # ro bind of H335 stagev-root/views-search
PREP_MANIFESTS = (Path("/home/ubuntu/jope-data/h381-d91/h335-prep/x0_prep_20261105T005754Z.tables.sha256"),
                  Path("/home/ubuntu/jope-data/h381-d91/h335-prep/x0_prep_20261105T005754Z.root.sha256"))
WORKER = Path("/home/ubuntu/jope-data/h335-cl/bin/bootstrap.s1main.superset-2a848c8f8085")
SEEDS = (335101, 335102)
SEL_SEED = 381501
CODE_FILES = ("gp_h381.py", "d91_h381.py") + tuple(LAB.PHASE_T_FILES)


def _sha(p: Path) -> str:
    return C.sha256_file(Path(p))


def arms() -> dict:
    return json.loads((DATA / "stage_v" / "ARMS.json").read_text())


def cmd_lock(a) -> None:
    lp = DATA / "stage_v" / "LOCK.json"
    if lp.exists():
        raise SystemExit(f"{lp} exists (create-once)")
    spec = json.loads(Path(a.arms).read_text())
    code = {str(HERE / f): _sha(HERE / f) for f in CODE_FILES}
    code.update({str(K.H317_SRC / n): s for n, s in K.H317_PINS.items()})
    code[str(K.GRAMMAR_PORT)] = K.GRAMMAR_PORT_SHA256
    code[str(WORKER)] = _sha(WORKER)
    for arm in spec["arms"].values():
        nd = Path(arm["neural_run_dir"])
        for s in C.TRAIN_SEEDS[:2]:
            p = nd / f"seed{s}" / "fold0" / "model.pt"
            code[str(p)] = _sha(p)
    lock = {"schema": "jope.h381.lock/v1", "wave": WAVE, "arms": spec["arms"], "seeds": list(SEEDS),
            "select_rule": f"first 5 families by stream_seed({SEL_SEED}, 'd91fam', family); in each the cid with the "
                           f"smallest stream_seed({SEL_SEED}, 'd91ctx', cid)",
            "code_sha256": code, "inputs_manifests_sha256": {str(p): _sha(p) for p in PREP_MANIFESTS},
            "prereg_commit": spec["prereg_commit"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lp.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(DATA / "stage_v" / "ARMS.json", spec)
    C.write_json(lp, lock)
    print(_sha(lp))


def _manifest_shas() -> dict:
    lock = json.loads((DATA / "stage_v" / "LOCK.json").read_text())
    out = {}
    for p, sha in lock["inputs_manifests_sha256"].items():
        if _sha(Path(p)) != sha:
            raise SystemExit(f"prep manifest changed after LOCK: {p}")
        for line in Path(p).read_text().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                out[Path(name.strip().lstrip("*")).name] = h
    return out


def cmd_unlock(_a) -> None:
    C.unlock_stage_v()
    print(json.dumps(C.guard_report()))


def cmd_select(_a) -> None:
    C.resume_stage_v()
    out = DATA / "cids5.json"
    if out.exists():
        print(out.read_text())
        return
    rows = C.dev_rows_for_stage_v()
    fam: dict = {}
    for r in rows:
        fam.setdefault(r["family_id"], []).append(r["context_sha256"])
    fams = sorted(fam, key=lambda f: C.stream_seed(SEL_SEED, "d91fam", f))[:5]
    cids = [min(fam[f], key=lambda c: C.stream_seed(SEL_SEED, "d91ctx", c)) for f in fams]
    man = _manifest_shas()
    chk = {}
    for c in cids:
        t, v = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t, "views_sha256": v, "table_in_prep": man.get(f"{c}.jtt1") == t,
                  "views_in_prep": man.get(f"{c}.npz") == v}
        if not (chk[c]["table_in_prep"] and chk[c]["views_in_prep"]):
            raise SystemExit(f"{c[:12]}: dev91 input sha not in the H335 prep manifests")
    rec = {"schema": "jope.h381.cids5/v1", "families": fams, "cids": cids, "family_of": dict(zip(cids, fams)),
           "rule": json.loads((DATA / "stage_v" / "LOCK.json").read_text())["select_rule"], "inputs": chk,
           "guard": C.guard_report()}
    fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(fd, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    os.close(fd)
    print(json.dumps({"cids": [c[:12] for c in cids], "families": fams}))


def cmd_run(a) -> None:
    C.resume_stage_v()
    arm = arms()["arms"][a.arm]
    sel = json.loads((DATA / "cids5.json").read_text())
    cids = sel["cids"]
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"schema": "jope.h381.d91-run/v1", "arm": a.arm, **arm, "cids": cids, "seeds": list(SEEDS),
           "lock_sha256": _sha(DATA / "stage_v" / "LOCK.json")}
    p = run_dir / "config.json"
    if p.exists() and json.loads(p.read_text()) != cfg:
        raise SystemExit("run config mismatch")
    C.write_json(p, cfg)
    GPH.install(int(arm["lf_rows"]))
    rt = LAB.build_rt(run_dir, "V", L.parse_cpus(a.cores), WORKER, float(arm.get("timeout_s", 60.0)), SEEDS,
                      Path(arm["neural_run_dir"]), 1, stagev_views=VIEWS)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in SEEDS]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, int(arm["gens"]), arm["checkpoints"],
                                                   int(arm["fr_k"]), tuple(LAB.STACKS)), work, a.jobs)
    status = {"wave": WAVE, "arm": a.arm, "units": len(recs), "wall_s": time.time() - t0,
              "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "guard": C.guard_report(),
              "bank_counters": dict(D.BANK_COUNTERS), "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{int(t0)}.json", status)
    rt.engine.close()
    print(json.dumps({k: status[k] for k in ("arm", "units", "wall_s")} | {"physical_runs":
                                                                            status["engine"]["physical_runs"]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--arms", required=True)
    sub.add_parser("unlock")
    sub.add_parser("select")
    p = sub.add_parser("run")
    p.add_argument("--arm", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    {"lock": cmd_lock, "unlock": cmd_unlock, "select": cmd_select, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
