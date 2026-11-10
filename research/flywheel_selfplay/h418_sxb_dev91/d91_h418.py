#!/usr/bin/env python3
"""H418: the one dev91-audit-v1 score of the locked H416 entrant h416.SXB (PREREG.md; owner confirmation typed by
Jepson 2026-11-10T09:26Z: "confirm: score h416.SXB once on dev91-audit-v1").

The entrant is H416's confirm path unchanged: GS = the stored H335 Stage V AGG run (memo replay, no new search),
GE / GL = the H416 GP arms (gp_h381.unit_lf at gp_h416's settings) with the H381-E2-S3fz / H415-L screeners, then
deploy_h416.h416_unit in mode confirm with the locked DESIGN.json (BCAR.mix). Every H416 / H415 / H410 / H381 / H335
module is imported at its published bytes; this file only binds them to dev91 (phase V runtime, Stage V unlock, the
board's two search seeds).

Guard: importing deploy_h416 (-> deploy_h415 -> deploy_h410 -> gp_h381 -> h335_core) installs the H317 guard v2. This
module points the Stage V unlock at DATA/stage_v/LOCK.json and the create-once receipt
ns/receipts/H418_LOCK_sha256.txt (the H369 / H381 / H410 pattern). `unlock` is one-shot; every later command calls
resume_stage_v, which re-hashes every file in LOCK['code_sha256']. That list holds code, the H317 pins, the worker and
the screener models only: no dev91 path (H369 attempt-1 lesson). The dev91-side inputs (H335 prep manifests, the
manifest of the stored H335 Stage V GS files, the board file manifest) are in LOCK['inputs_sha256'] and are checked
after the unlock by every command.

Commands (sandbox_h418_d91.sh on xbabe0; prelock / unlock modes see TRAIN only):
  lock --spec SPEC.json   prelock: DATA/stage_v/LOCK.json (create-once); prints its sha256. Reads no dev91 byte.
  unlock                  the one-shot C.unlock_stage_v()
  select                  resume; DATA/cids91.json (create-once): the 91 dev91 contexts / 19 families, each table
                          and SEARCH-view sha in the H335 prep manifests
  gp --arm GE|GL --run-dir R --cores L [--jobs J]
                          resume; the H416 GP arm on the 91 contexts x seeds (335101, 335102), phase V runtime (the
                          arm screener's fold-0 pair, the H335 SEARCH views); records under stages/E416_AGG
  deploy --seed S --tag T --run-dir R --cores L [--jobs J] [--inner-jobs 3]
                          resume; deploy_h416.h416_unit (confirm, DESIGN.json) on the 91 contexts at search seed S.
                          The run dir's memo is the per-context concatenation GS + GE + GL (run_h418.sh deploy-prep).
                          One process and one run dir per seed: the H415 / H416 unit stores are keyed by context and
                          its capture dict by context, so two seeds of one context never share a process.
  inputs                  resume; re-verify LOCK['inputs_sha256'] and print the guard report (no other read)

Not here: reading any entrant value. The readout (gates, per-context values, board rows) is analyze_h418.py on
xbabe2, from sha-checked pulled copies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h416 as DH  # noqa: E402  (FIRST: deploy_h415 -> deploy_h410 -> gp_h381 -> h335_core pins + guard)
import gp_h416 as GP16  # noqa: E402  (sets gp_h381 WAVE / PHASE / SEED = H416 / E416 / 335101: the arms' stage label)
import train_h416 as T16  # noqa: E402  (constants only: NEURAL, STAGE_GP, KEY_GP; stdlib imports at module level)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

DP15, DP, GPH, K, C, LAB, L, D = DH.DP15, DH.DP, DH.GPH, DH.K, DH.C, DH.LAB, DH.L, DH.D

WAVE = "H418"
ENTRANT = "h416.SXB"
DATA = Path(os.environ.get("H418_D91", "/home/ubuntu/jope-data/h418-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H418_LOCK_sha256.txt")
LOCK_PATH = DATA / "stage_v" / "LOCK.json"
VIEWS = DATA / "views-search"  # run mode: ro bind of the H335 Stage V stagev-root/views-search
H335D = Path("/home/ubuntu/jope-data/h335-cl")
WORKER = H335D / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
WORKER_SHA256 = "2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2"
NEURAL = dict(T16.NEURAL)  # {"GS": "H0.1-A", "GE": "H381-E2-S3fz", "GL": "H415-L"}
SOURCES = ("GS", "GE", "GL")  # deploy order: P2 = GS + GE, P3 = all three (deploy_h415.deploy_unit)
ARMS = ("GE", "GL")  # new GP runs; GS is the stored H335 Stage V AGG run
SEEDS = tuple(K.SEEDS_V)  # (335101, 335102): the dev91 board's search seeds for every GP entrant (h335.*, h364.*)
GP_TIMEOUT_S = 60.0  # gp_h416 launch default (the H416 arms)
DEPLOY_TIMEOUT_S = 300.0  # train_h416 deploy default (H416 confirm config timeout_s 300.0)
DESIGN_FILE = HERE / "DESIGN.json"  # staged: the H416 DESIGN.json at its origin/v1/next bytes (PREREG part B)
DESIGN_NAME = "BCAR.mix"
N_CONTEXTS, N_FAMILIES = 91, 19
PREP_MANIFESTS = (DATA / "h335-prep" / "x0_prep_20261105T005754Z.tables.sha256",
                  DATA / "h335-prep" / "x0_prep_20261105T005754Z.root.sha256")
INPUT_MANIFESTS = (DATA / "inputs" / "gs_stage_v.sha256",  # stored H335 Stage V GS memo + V_AGG records (sha only)
                   DATA / "inputs" / "board_audit.sha256")  # dev91-audit-v1 BOARD.json + score files (sha only)
# No path above may contain an H317 guard DL-Val token (dev91, dl-val, dl_val, dlval, hold_out, hold-out): the guard
# refuses such a path before the Stage V unlock (H418 dry run 2026-11-10: `board_dev91.sha256` was refused at lock).
# the staged files hashed into LOCK['code_sha256'] (+ the H317 pins via K.code_manifest, the worker, the models)
CODE_FILES = DH.code_files() + ("gp_h415.py", "train_h415.py", "gp_h416.py", "train_h416.py", "d91_h418.py",
                                "DESIGN.json", "sandbox_h418_d91.sh", "PREREG_H418.md",
                                # the readout and the orchestration are LOCKed too (fixed before the unlock)
                                "analyze_h418.py", "selftest_h418.py", "run_h418.sh", "stage_h418.sh")

# the recipe constants this driver relies on, asserted against the imported published modules
if GPH.PHASE != "E416" or T16.STAGE_GP != {"GE": "E416_AGG", "GL": "E416_AGG"} or T16.KEY_GP != "AGG.g50.CV3@12":
    raise AssertionError("H416 stage label / output key differ from the H416 recipe")
if (GP16.GENS, tuple(GP16.CHECKPOINTS)) != (50, (50,)) or NEURAL != {"GS": "H0.1-A", "GE": "H381-E2-S3fz",
                                                                      "GL": "H415-L"}:
    raise AssertionError("H416 generations / checkpoints / screeners differ from the H416 recipe")
if SEEDS != (335101, 335102):
    raise AssertionError("h335_core.SEEDS_V is not (335101, 335102)")


def _sha(p: Path) -> str:
    return C.sha256_file(Path(p))


def lock() -> dict:
    return json.loads(LOCK_PATH.read_text())


def _write_once(p: Path, obj: dict) -> None:
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(fd, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def model_files() -> dict:
    """Each source screener's fold-0 pair (the phase V screener: no model saw a DL-Val family)."""
    out = {}
    for name in SOURCES:
        for s in C.TRAIN_SEEDS[:2]:
            p = H335D / "neural" / "runs" / NEURAL[name] / f"seed{s}" / "fold0" / "model.pt"
            out[str(p)] = _sha(p)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# lock / unlock / inputs / select
# ----------------------------------------------------------------------------------------------------------------------
def cmd_lock(a) -> None:
    if LOCK_PATH.exists():
        raise SystemExit(f"{LOCK_PATH} exists (create-once)")
    if C._STATE["stage_v"]:
        raise SystemExit("lock runs before the Stage V unlock")
    spec = json.loads(Path(a.spec).read_text())
    for k in ("prereg_commit", "stage_commit"):
        if not isinstance(spec.get(k), str) or len(spec[k]) != 40:
            raise SystemExit(f"spec needs a full {k}")
    design = json.loads(DESIGN_FILE.read_text())
    if design.get("schema") != "jope.h416.design/v1" or design.get("design") != DESIGN_NAME:
        raise SystemExit(f"staged DESIGN.json is not the H416 {DESIGN_NAME} design")
    if _sha(WORKER) != WORKER_SHA256:
        raise SystemExit("superset worker sha differs")
    code = K.code_manifest(CODE_FILES)
    code[str(WORKER)] = WORKER_SHA256
    code.update(model_files())
    inputs = {str(p): _sha(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS}
    lk = {"schema": "jope.h418.lock/v1", "wave": WAVE, "entrant": ENTRANT, "entrant_wave": "H416",
          "board": "dev91-audit-v1", "design": DESIGN_NAME, "design_sha256": _sha(DESIGN_FILE),
          "neural": NEURAL, "sources": list(SOURCES), "arms": list(ARMS), "seeds": list(SEEDS),
          "gp": {"gens": GP16.GENS, "checkpoints": list(GP16.CHECKPOINTS), "lf_rows": 0, "fr_k": 0,
                 "stacks": list(LAB.STACKS), "timeout_s": GP_TIMEOUT_S, "stage": T16.STAGE_GP["GE"]},
          "deploy": {"mode": "confirm", "timeout_s": DEPLOY_TIMEOUT_S, "memo": "per context: cat GS GE GL",
                     "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                              "max_members": DH.M.MAX_MEMBERS}},
          "contexts_rule": "C.dev_rows_for_stage_v(): all 91 dev91 contexts / 19 families",
          "code_sha256": code, "inputs_sha256": inputs, "prereg_commit": spec["prereg_commit"],
          "stage_commit": spec["stage_commit"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_once(LOCK_PATH, lk)
    print(_sha(LOCK_PATH))


def _inputs_ok() -> dict:
    """LOCK['inputs_sha256'] unchanged; returns {basename: sha256} from the H335 prep manifests."""
    lk = lock()
    for p, sha in lk["inputs_sha256"].items():
        if _sha(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")
    if sorted(lk["inputs_sha256"]) != sorted(str(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS):
        raise SystemExit("LOCK inputs differ from this driver's input list")
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


def cmd_inputs(_a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    print(json.dumps({"inputs": "ok", "lock_sha256": _sha(LOCK_PATH), "guard": C.guard_report()}))


def cmd_select(_a) -> None:
    C.resume_stage_v()
    out = DATA / "cids91.json"
    if out.exists():
        d = json.loads(out.read_text())
        print(json.dumps({"n": d["n"], "families": d["families"], "cids_sha256": d["cids_sha256"]}))
        return
    man = _inputs_ok()
    rows = C.dev_rows_for_stage_v()  # 91 contexts / 19 families, asserted by the guard module
    cids = [r["context_sha256"] for r in rows]
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    if len(cids) != N_CONTEXTS or len(set(fam.values())) != N_FAMILIES:
        raise SystemExit("dev91 is not 91 contexts / 19 families")
    chk = {}
    for c in cids:
        t, v = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t, "views_sha256": v}
        if man.get(f"{c}.jtt1") != t or man.get(f"{c}.npz") != v:
            raise SystemExit(f"{c[:12]}: dev91 table / SEARCH view sha not in the H335 prep manifests")
    import hashlib

    rec = {"schema": "jope.h418.cids91/v1", "n": len(cids), "families": len(set(fam.values())), "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "family_of": fam, "inputs": chk,
           "lock_sha256": _sha(LOCK_PATH), "guard": C.guard_report()}
    _write_once(out, rec)
    print(json.dumps({"n": rec["n"], "families": rec["families"], "cids_sha256": rec["cids_sha256"]}))


def _cids() -> list:
    d = json.loads((DATA / "cids91.json").read_text())
    if d["lock_sha256"] != _sha(LOCK_PATH) or len(d["cids"]) != N_CONTEXTS:
        raise SystemExit("cids91.json is not this LOCK's 91-context selection")
    return list(d["cids"])


def _run_cfg(p: Path, cfg: dict) -> None:
    """Create-once run config; a resume must carry identical keys (the screener manifest included)."""
    if p.exists():
        old = json.loads(p.read_text())
        bad = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
        if bad:
            raise SystemExit(f"run config mismatch on {bad}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(p, cfg)


def _status(run_dir: Path, rt, tag: str, t0: float, n: int, extra: dict) -> dict:
    st = {"wave": WAVE, "tag": tag, "units": n, "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DP.physical_by_kind(rt), "host": os.uname().nodename, **extra}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{tag}_{int(t0)}.json", st)
    return st


def _order(cids: list) -> list:
    return sorted(cids, key=lambda c: (-K.table_size(c), c))


# ----------------------------------------------------------------------------------------------------------------------
# GP arms (gp_h416.cmd_run's unit call, phase V runtime, the board's two seeds)
# ----------------------------------------------------------------------------------------------------------------------
def cmd_gp(a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    if a.arm not in ARMS:
        raise SystemExit(f"GP arms are {ARMS} (GS = the stored H335 Stage V run)")
    cids = _cids()
    run_dir = Path(a.run_dir)
    nd = H335D / "neural" / "runs" / NEURAL[a.arm]
    cfg = {"schema": "jope.h418.gp-config/v1", "wave": WAVE, "entrant_wave": "H416", "arm": a.arm,
           "phase": GPH.PHASE, "stage": T16.STAGE_GP[a.arm], "panel": "dev91", "cids": cids, "seeds": list(SEEDS),
           "group": "AGG", "gens": GP16.GENS, "checkpoints": list(GP16.CHECKPOINTS), "lf_rows": 0, "fr_k": 0,
           "n_prop": K.N_PROP, "n_live": K.N_LIVE, "stacks": list(LAB.STACKS), "timeout_s": GP_TIMEOUT_S,
           "neural_run_dir": str(nd), "worker_sha256": _sha(WORKER), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "lock_sha256": _sha(LOCK_PATH)}
    _run_cfg(run_dir / "config.json", cfg)
    GPH.install(0)
    rt = LAB.build_rt(run_dir, "V", L.parse_cpus(a.cores), WORKER, GP_TIMEOUT_S, SEEDS, nd, 1, stagev_views=VIEWS)
    t0 = time.time()
    work = [(c, s) for c in _order(cids) for s in SEEDS]
    recs = L.run_parallel(lambda c, s: GPH.unit_lf(rt, c, s, GP16.GENS, GP16.CHECKPOINTS, 0, tuple(LAB.STACKS)),
                          work, a.jobs)
    st = _status(run_dir, rt, f"gp{a.arm}", t0, len(recs), {"arm": a.arm})
    rt.engine.close()
    print(json.dumps({"arm": a.arm, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"]}))


# ----------------------------------------------------------------------------------------------------------------------
# deploy (train_h416.cmd_deploy, mode confirm, with deploy_h410.build_rt_v for the phase V runtime)
# ----------------------------------------------------------------------------------------------------------------------
def cmd_deploy(a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    seed = int(a.seed)
    if seed not in SEEDS:
        raise SystemExit(f"search seed must be one of {SEEDS}")
    if a.tag != f"D{SEEDS.index(seed) + 1}":
        raise SystemExit(f"seed {seed} deploys under tag D{SEEDS.index(seed) + 1}")
    design = json.loads(DESIGN_FILE.read_text())
    if design.get("schema") != "jope.h416.design/v1" or design.get("design") != DESIGN_NAME or \
            _sha(DESIGN_FILE) != lock()["design_sha256"]:
        raise SystemExit("staged DESIGN.json is not the locked H416 design")
    cids = _cids()
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    missing = [c for c in cids if not (memo / f"{c}.split.jsonl").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no concatenated memo (run_h418.sh deploy-prep first)")
    nd = {s: H335D / "neural" / "runs" / NEURAL[s] for s in SOURCES}
    rt, scr, man = DP.build_rt_v(run_dir, L.parse_cpus(a.cores), WORKER, DEPLOY_TIMEOUT_S, (seed,), nd, VIEWS)
    cfg = {"schema": "jope.h418.deploy-config/v1", "wave": WAVE, "entrant": ENTRANT, "tag": a.tag,
           "mode": "confirm", "panel": "dev91", "cids": cids, "sources": list(nd),
           "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": DEPLOY_TIMEOUT_S,
           "design": design, "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                                      "max_members": DH.M.MAX_MEMBERS},
           "code_sha256": K.code_manifest(CODE_FILES), "screener_manifest": man, "lock_sha256": _sha(LOCK_PATH)}
    _run_cfg(run_dir / f"config_{a.tag}.json", cfg)
    DH.GPH.install(0)
    npz_dir = run_dir / f"h416_{a.tag}_npz"
    t0 = time.time()
    recs = L.run_parallel(lambda c, _s: DH.h416_unit(rt, a.tag, c, scr, seed, "confirm", npz_dir, design,
                                                     a.inner_jobs), [(c, 0) for c in _order(cids)], a.jobs)
    st = _status(run_dir, rt, a.tag, t0, len(recs), {"seed": seed, "akf_ledger": DP15.KF_LEDGER.check()})
    rt.engine.close()
    # counts only: no entrant value is printed here
    print(json.dumps({"tag": a.tag, "seed": seed, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "errors": sum(1 for r in recs if r.get("error")), "physical_by_kind": st["physical_by_kind"],
                      "akf_ledger": st["akf_ledger"]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--spec", required=True)
    for n in ("unlock", "select", "inputs"):
        sub.add_parser(n)
    p = sub.add_parser("gp")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=12)  # gp_h416 run default
    p = sub.add_parser("deploy")
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--tag", required=True, choices=("D1", "D2"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--inner-jobs", type=int, default=3)  # train_h416 deploy default
    a = ap.parse_args()
    {"lock": cmd_lock, "unlock": cmd_unlock, "select": cmd_select, "inputs": cmd_inputs, "gp": cmd_gp,
     "deploy": cmd_deploy}[a.cmd](a)


if __name__ == "__main__":
    main()
