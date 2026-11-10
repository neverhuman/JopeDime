#!/usr/bin/env python3
"""H410 dev91 driver (PREREG §3, §7): the 5 H381-exposed contexts only, under H410's own LOCK.

Guard: importing deploy_h410 -> gp_h381 -> h335_core installs the H317 guard v2. This module points the Stage V unlock
at DATA/stage_v/LOCK.json and the create-once receipt ns/receipts/H410_LOCK_sha256.txt; `unlock` is one-shot, later
commands use resume_stage_v. The LOCK's code_sha256 lists code, the worker and every source screener's fold-0 models
(no dev91 path). Inputs (H381 cids5.json copy, H335 prep manifests) are checked after the unlock.

Commands (sandbox_h410_d91.sh on xbabe0):
  lock     --arms ARMS.json     DATA/stage_v/LOCK.json (create-once); prints its sha256 (no dev91 read)
  unlock                        the one-shot C.unlock_stage_v()
  select                        resume; DATA/cids5.json = the H381 contexts (sha-checked copy), inputs vs prep shas
  gp       --run-dir R --cores L --jobs J      resume; C1: the AGG recipe (unit_lf, lf_rows 0, g50) with the C1
                                               screener, cids5 x seeds (335101, 335102)
  deploy   --tag T --run-dir R --cores L --jobs J   resume; deploy_unit with the arm's sources / pools / library
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h410 as DP  # noqa: E402  (FIRST: h335_core pins + guard)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

GPH, K, C, LAB, L, D = DP.GPH, DP.K, DP.C, DP.LAB, DP.L, DP.D

WAVE = "H410"
DATA = Path(os.environ.get("H410_D91", "/home/ubuntu/jope-data/h410-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H410_LOCK_sha256.txt")
VIEWS = DATA / "views-search"  # ro bind of H335 stagev-root/views-search
PREP_MANIFESTS = (DATA / "h335-prep" / "x0_prep_20261105T005754Z.tables.sha256",
                  DATA / "h335-prep" / "x0_prep_20261105T005754Z.root.sha256")
CIDS5_IN = DATA / "inputs" / "cids5_h381.json"
WORKER = Path("/home/ubuntu/jope-data/h335-cl/bin/bootstrap.s1main.superset-2a848c8f8085")
SEEDS = (335101, 335102)
CODE_FILES = DP.code_files() + ("d91_h410.py",)


def _sha(p: Path) -> str:
    return C.sha256_file(Path(p))


def lock() -> dict:
    return json.loads((DATA / "stage_v" / "LOCK.json").read_text())


def cmd_lock(a) -> None:
    lp = DATA / "stage_v" / "LOCK.json"
    if lp.exists():
        raise SystemExit(f"{lp} exists (create-once)")
    spec = json.loads(Path(a.arms).read_text())
    code = {str(HERE / f): _sha(HERE / f) for f in CODE_FILES}
    code.update({str(K.H317_SRC / n): s for n, s in K.H317_PINS.items()})
    code[str(K.GRAMMAR_PORT)] = K.GRAMMAR_PORT_SHA256
    code[str(WORKER)] = _sha(WORKER)
    for nd in spec["neural_dirs"].values():
        for s in C.TRAIN_SEEDS[:2]:
            p = Path(nd) / f"seed{s}" / "fold0" / "model.pt"
            code[str(p)] = _sha(p)
    inputs = {str(p): _sha(p) for p in PREP_MANIFESTS}
    inputs[str(CIDS5_IN)] = _sha(CIDS5_IN)
    inputs[str(DATA / "neural" / "views_c32_train1219.npz")] = _sha(DATA / "neural" / "views_c32_train1219.npz")
    lk = {"schema": "jope.h410.lock/v1", "wave": WAVE, "arms": spec["arms"], "neural_dirs": spec["neural_dirs"],
          "seeds": list(SEEDS), "select_rule": "the 5 H381-exposed contexts (H381 cids5.json, sha in inputs_sha256)",
          "code_sha256": code, "inputs_sha256": inputs, "prereg_commit": spec["prereg_commit"],
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lp.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(lp, lk)
    print(_sha(lp))


def _inputs_ok() -> dict:
    lk = lock()
    for p, sha in lk["inputs_sha256"].items():
        if _sha(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")
    out = {}
    for p in PREP_MANIFESTS:
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
    man = _inputs_ok()
    src = json.loads(CIDS5_IN.read_text())
    cids = src["cids"]
    dev = {r["context_sha256"]: r for r in C.dev_rows_for_stage_v()}
    chk = {}
    for c in cids:
        if c not in dev or dev[c]["family_id"] != src["family_of"][c]:
            raise SystemExit(f"{c[:12]}: not a dev91 context of the recorded family")
        t, v = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t, "views_sha256": v, "table_in_prep": man.get(f"{c}.jtt1") == t,
                  "views_in_prep": man.get(f"{c}.npz") == v,
                  "same_as_h381": t == src["inputs"][c]["table_sha256"] and v == src["inputs"][c]["views_sha256"]}
        if not all(chk[c][k] for k in ("table_in_prep", "views_in_prep", "same_as_h381")):
            raise SystemExit(f"{c[:12]}: dev91 input check failed")
    rec = {"schema": "jope.h410.cids5/v1", "families": src["families"], "cids": cids, "family_of": src["family_of"],
           "rule": lock()["select_rule"], "inputs": chk, "h381_cids5_sha256": _sha(CIDS5_IN),
           "guard": C.guard_report()}
    fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(fd, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    os.close(fd)
    print(json.dumps({"cids": [c[:12] for c in cids]}))


def _run_cfg(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / f"config_{cfg['tag']}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items() if k != "screener_manifest"):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)


def _status(run_dir: Path, rt, tag: str, t0: float, n: int) -> dict:
    st = {"wave": WAVE, "tag": tag, "units": n, "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "host": os.uname().nodename, "physical_by_kind": DP.physical_by_kind(rt)}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{tag}_{int(t0)}.json", st)
    return st


def cmd_gp(a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    lk = lock()
    arm = lk["arms"]["C1"]
    cids = json.loads((DATA / "cids5.json").read_text())["cids"]
    run_dir = Path(a.run_dir)
    nd = {"C1": Path(lk["neural_dirs"]["C1"])}
    rt, _, man = DP.build_rt_v(run_dir, L.parse_cpus(a.cores), WORKER, float(arm["timeout_s"]), SEEDS, nd, VIEWS)
    _run_cfg(run_dir, {"tag": "C1gp", "arm": arm, "cids": cids, "seeds": list(SEEDS), "lock_sha256":
                       _sha(DATA / "stage_v" / "LOCK.json"), "screener_manifest": man})
    GPH.install(0)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in SEEDS]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, int(arm["gens"]), [int(arm["gens"])], 0,
                                                   tuple(LAB.STACKS)), work, a.jobs)
    st = _status(run_dir, rt, "C1gp", t0, len(recs))
    rt.engine.close()
    print(json.dumps({"tag": "C1gp", "units": len(recs), "wall_s": st["wall_s"],
                      "physical_runs": st["engine"].get("physical_runs")}))


def cmd_deploy(a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    lk = lock()
    arm = lk["arms"][a.tag]
    cids = json.loads((DATA / "cids5.json").read_text())["cids"]
    run_dir = Path(a.run_dir)
    nd = {n: Path(lk["neural_dirs"][n]) for n in arm["sources"]}
    rt, scr, man = DP.build_rt_v(run_dir, L.parse_cpus(a.cores), WORKER, float(arm["timeout_s"]), SEEDS, nd, VIEWS)
    _run_cfg(run_dir, {"tag": a.tag, "arm": arm, "cids": cids, "seeds": list(SEEDS),
                       "lock_sha256": _sha(DATA / "stage_v" / "LOCK.json"), "screener_manifest": man})
    GPH.install(0)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(lambda c, _s: DP.deploy_unit(rt, a.tag, c, scr, SEEDS, bool(arm["library"]),
                                                       arm.get("own_source"), arm["pools"], a.inner_jobs),
                          [(c, 0) for c in order], a.jobs)
    st = _status(run_dir, rt, a.tag, t0, len(recs))
    rt.engine.close()
    print(json.dumps({"tag": a.tag, "units": len(recs), "wall_s": st["wall_s"],
                      "engine": {k: st["engine"].get(k) for k in ("physical_runs", "memo_hits")}}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--arms", required=True)
    sub.add_parser("unlock")
    sub.add_parser("select")
    for name in ("gp", "deploy"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--cores", required=True)
        p.add_argument("--jobs", type=int, default=10)
        if name == "deploy":
            p.add_argument("--tag", required=True)
            p.add_argument("--inner-jobs", type=int, default=4)
    a = ap.parse_args()
    {"lock": cmd_lock, "unlock": cmd_unlock, "select": cmd_select, "gp": cmd_gp, "deploy": cmd_deploy}[a.cmd](a)


if __name__ == "__main__":
    main()
