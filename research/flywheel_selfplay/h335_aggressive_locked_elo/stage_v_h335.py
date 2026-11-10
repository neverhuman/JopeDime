#!/usr/bin/env python3
"""H335 Stage V driver (PREREG §4-§5): LOCK writer, pinned one-shot unlock, prep, sealed audit units, stats.
Built and tested on SYNTHETIC fixtures and TRAIN data only; it has never been run on DL-Val.

Reuses the FROZEN H317 driver `stage_v_h317.py` by import, never modified: create_once / write_atomic / event,
enter_stage_v (receipt + every LOCK file + marker, then common_h317.resume_stage_v), cmd_unlock, verify_inputs,
run_dedupe, encoder_parity, encode_targets, run_regression, ShardRunLock, _slim / _r, MAX_UNIT_ATTEMPTS.
Its module root is H317_STAGEV_ROOT, set here from H335_STAGEV_ROOT (default /home/ubuntu/jope-data/h335-stagev) BEFORE
the import. Its schema check reads the module global SCHEMA_LOCK, which is rebound to jope.h335.lock/v1 (the only
runtime rebinding, like h335_core's guard extension of common_h317). Every step runs from the frozen read-only code copy
(H335_CODE, bound at the H335 source path by the sandboxes; $STAGE/code in docker).

Subcommands (hosts, mounts, staging and the exact commands: stage_v_mounts_h335.md):
  write-lock   xbabe2, sandbox_h335.sh. $H317_DATA/stage_v/LOCK.json (create-once, fsynced) from the Phase T decision;
               refuses a hard stop and any Phase T run dir whose config.json code shas differ from the current files.
               code_sha256 = the explicit runtime-code list LOCK_CODE_FILES + the frozen H317 pins + Micro featurizer
               files + worker + frontier binary + screener models + merged views + bank / TRAIN inputs; every other
               file of the code dir goes to the NON-verified docs_sha256. Prints the LOCK sha256 last.
  verify-lock-files  any host, any sandbox, before or after the unlock: rehash every code_sha256 file (+ the TRAIN
               inputs manifest; --prep: the encoder stack); refuses on any missing / changed file; never unlocks.
  unlock       xbabe2, `sandbox_h335_stagev.sh unlock` (no DL-Val byte visible): the frozen H317 cmd_unlock.
  prep         xbabe2, `sandbox_h335_stagev.sh v`: input shas; dedupe (before any outcome); encoder parity + DL-Val
               SEARCH-row view sets; the 91 LLM AUDIT envelopes (outcome-free); R882 Native91 regression (recorded at
               once, never recomputed; a FAIL stops Stage V); PREP_DONE.json; then the 4 LLM regressors through
               frontier_audit_h335.run (append-only, skip-existing) and LLM_DONE.json. --resume any number of times
               before FIRST_OUTCOME_READ (never-recorded steps only). Prints counts only.
  audit        xbabe0, `sandbox_h335_docker.sh --stage-v`: one `audit-shard` process per (group, shard) with
               H335_GROUP=AGG|BASE. Each runs its units (largest first) through h335_lab.run_units one unit at a time
               (identical records), stops taking units at the first propagated error, caps host-failure reruns at
               H317's MAX_UNIT_ATTEMPTS (durable attempt log), and writes STOPPED_INTEGRITY on an integrity violation.
               Sealed: prints counts only.
  stats        xbabe2, `sandbox_h335_stagev.sh v`: identity checks of every record, then completeness (refused while
               incomplete; --final-incomplete only once every planned shard is done or after the Stage V compute stop),
               then the create-once FIRST_OUTCOME_READ (binds the record-set sha), then the PREREG §5 readout ->
               run/metrics_v.json (< 50 KB) + run/panel_stage_v.json.
After FIRST_OUTCOME_READ no unit runs (audit refuses), prep refuses, and stats re-reads only the identical record set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402  (FIRST: pins, grammar group, guard)

STAGEV_ROOT = Path(os.environ.get("H335_STAGEV_ROOT", "/home/ubuntu/jope-data/h335-stagev"))
os.environ["H317_STAGEV_ROOT"] = str(STAGEV_ROOT)

import argparse  # noqa: E402
import calendar  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

import crop_shuffle_synthetic_views as V  # noqa: E402,F401  (binds its view / merged-view paths to $H317_DATA now)
import frontier_audit_h335 as FA  # noqa: E402
import h335_lab as LAB  # noqa: E402
import h335_stats as H  # noqa: E402
import stage_v_h317 as SV  # noqa: E402

C, L, G = K.C, K.L, K.G
if SV.STAGEV_ROOT != STAGEV_ROOT:
    raise RuntimeError(f"stage_v_h317 was imported with Stage V root {SV.STAGEV_ROOT}, not {STAGEV_ROOT}")
SCHEMA_LOCK = "jope.h335.lock/v1"
SV.SCHEMA_LOCK = SCHEMA_LOCK  # read by the frozen SV.verify_lock (enter_stage_v)
SCHEMA_METRICS = "jope.h335.stage-v-metrics/v1"
SCHEMA_DECISION = "jope.h335.phase-t-decision/v1"
assert (H.CLIP, H.ALPHA, H.TIE_TOL) == (K.CLIP, K.ALPHA, K.TIE_TOL)

# The runtime code that determines outcomes (LOCK code_sha256, verified by unlock / every resume_stage_v). Every other
# file of the code dir (PREREG.md, *.md, test_*.py, run_*.sh, stage_*.sh, h335_replay.py, h335_encode.py, *.rs) is
# recorded in the NON-verified docs_sha256.
LOCK_CODE_FILES = ("h335_core.py", "h335_lab.py", "h335_stats.py", "h335_frontier.py", "frontier_audit_h335.py",
                   "stage_v_h335.py", "gp_grammar_h317_learners_v1.py", "sandbox_h335.sh", "sandbox_h335_stagev.sh",
                   "sandbox_h335_docker.sh")
# Micro files feeding the screener featurizer (also the staged micro subset mounted at this path in the xbabe0 docker)
MICRO_CODE = ("python/rust_evaluator.py", "python/rank_prepare.py", "assets/projections/program-feature-names.json")
# production inputs, bound at import (tests redirect DATA_ROOT afterwards; the LOCK hashes these exact paths)
WORKER = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
FRONTIER = Path(C.DATA_ROOT) / "bin" / "frontier_audit_h335"  # ro in both sandboxes (== build/ copy, same sha)
FRONTIER_SHA256 = "b359807c289a7fbd8304218d8c29b31d758d552e0b1f8a2e82a748686cc22d43"
NEURAL_RUN_DIR = Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"
N_SCREENER_MODELS = 10  # 2 training seeds x 5 OOF folds (fold 0 pair scores every DL-Val target)
GP_TIMEOUT_S = 60.0
LLM_CPU_LIMIT_S, LLM_WALL_LIMIT_S = 300, 900
GROUPS = ("AGG", "BASE")
P1_CTRL, P2_CTRL = "H257_REPLICA", "OLD_MATCHED"
DESCRIPTIVE_CONTROLS = ("OLD_MATCHED_CV3", "H317_BASE", "BANK_NN8_PORTFOLIO", "BANK_NN1")
LLM_RECORDS, LLM_DONE, METRICS = "llm_audit.jsonl", "LLM_DONE.json", "metrics_v.json"
INTEGRITY_STOP = "STOPPED_INTEGRITY"
MAX_UNIT_ATTEMPTS = SV.MAX_UNIT_ATTEMPTS  # 2: a unit that was twice the culprit of an error is recorded failed
STAGE_V_COMPUTE_STOP = "2026-11-05T07:30:00Z"  # PREREG §6
STAGE_V_COMPUTE_STOP_UTC = calendar.timegm(time.strptime(STAGE_V_COMPUTE_STOP, "%Y-%m-%dT%H:%M:%SZ"))


def _utcnow() -> float:
    return time.time()


RULES = {
    "unit": "(context, search seed, group) for 91 DL-Val contexts x seeds 335101/335102 x {AGG, BASE}, largest first; "
            "AGG to G* (outputs at checkpoints {10, G*}, single + CV3@12), BASE to G* + 2 (OLD at checkpoint + 2, "
            "H317_BASE, H257_REPLICA, BANK_*); fold-0 H0.1-A OOF screener pair; superset worker, CPU-time limited",
    "per_context_value": "mean over the two search seeds of the AUDIT utility; a failed unit / segment scores u = 0 "
                         "(SEARCH-mean predictor), invalid; LLM entrants: one fit, seed 42, failure/timeout u = 0",
    "primary": "d = clip(u_arm - u_ctrl, -0.5, 0.5); family means over the 19 families; exact one-sided sign-flip over "
               "2^19 (identity included); pass iff T > 0 AND p < 0.025 AND LOFO min > 0 AND family median >= 0; "
               "inferior = the mirrored test passes (h335_stats.primary_test)",
    "sequence": "fixed: P1 = H335_AGG vs H257_REPLICA; P2 = H335_AGG vs OLD_MATCHED counts only if P1 passes; each at "
                "full alpha 0.025",
    "verdict": "P1 & P2 -> BETTER_MODEL; P1 only -> BETTER_WITH_COMPUTE; P1 inferior -> INFERIOR_TO_INCUMBENT; else "
               "INCONCLUSIVE; an incomplete run -> INCONCLUSIVE; R882 regression mismatch -> REGRESSION_FAIL",
    "adoption_secondary": "H317 contract gate on unclipped values vs H257_REPLICA and OLD_MATCHED: whole-family "
                          "bootstrap LCB95 > +0.001 (10,000 draws, RNG 335150), median >= 0, <= 10% families losing "
                          "> 0.01, sign-flip p < 0.025, LOFO > 0 (reported, not the verdict)",
    "elo": "AUDIT-Elo: family-grouped MM Bradley-Terry over the 12 pre-declared entrants (h335_stats.elo_panel == the "
           "frozen H317 functions), 400 iterations, prior 0.01, tie 1e-4, anchor H257_REPLICA = 1500, family "
           "bootstrap CI95 (10,000, RNG 335160); plus the pairwise H335_AGG vs H257_REPLICA panel; descriptive",
    "dedupe": "descriptive + sensitivity: P1/P2 recomputed without contexts whose raw_sorted max Jaccard or overlap vs "
              "any TRAIN table > 0.5; the verdict uses all 91",
    "invalid_flag": "invalid(H335_AGG) <= invalid(ctrl) + 1 for both primary controls (flag, not a gate)",
    "sealing": "unit and LLM records are identity-checked, never aggregated or printed before every record exists; "
               "stats creates FIRST_OUTCOME_READ (create-once, binds the record set sha) before the first value read",
    "final_incomplete": "an incomplete run may be read (INCONCLUSIVE) only when AUDIT_PLAN.json exists, every planned "
                        "(group, shard) has an AUDIT_DONE record and run/LLM_DONE.json exists, or after the Stage V "
                        "compute stop " + STAGE_V_COMPUTE_STOP,
    "reruns": "a unit that was the CULPRIT of MAX_UNIT_ATTEMPTS = 2 propagated in-process errors (durable culprit "
              "lines) is recorded failed (every output u = 0, invalid) on its next attempt (H317 cap); units merely in "
              "flight when a process died (start lines only) are never capped, only disclosed",
    "integrity": "guard / AUDIT ledger / bank leak / assertion / screener leak: create-once STOPPED_INTEGRITY; audit, "
                 "prep and stats refuse afterwards",
    "regression": "R882 Native91 91/91 bit-exact vs H258's pinned reference, mean 1.1377941898164525 (H317 runner "
                  "mapping, stage_v_h317.run_regression); a mismatch stops Stage V",
}


class StageVError(SV.StageVError):
    pass


def gp_dir() -> Path:
    return SV.STAGEV_ROOT / "gp"


def llm_env_dir() -> Path:
    return SV.STAGEV_ROOT / "llm-envs"


def _abs(p) -> str:
    return os.path.abspath(str(p))  # absolute, symlinks NOT resolved: every host must present the file at this path


def entrant_keys(gstar: int, stack: str) -> Dict[str, str]:
    """Pre-declared AUDIT entrant -> unit output key ("LLM:<name>" for the 4 frontier regressors)."""
    return {"H335_AGG": f"AGG.g{gstar}.{stack}", "OLD_MATCHED": f"OLD.g{gstar}.single",
            "OLD_MATCHED_CV3": f"OLD.g{gstar}.{K.STACK}", "H317_BASE": "H317_BASE.single",
            "H257_REPLICA": "H257_REPLICA.CV1", "BANK_NN8_PICK": "BANK_NN8_PICK.single",
            "BANK_NN8_PORTFOLIO": "BANK_NN8_PORTFOLIO.CV5", "BANK_NN1": "BANK_NN1.single",
            **{e: f"LLM:{e}" for e in FA.ENTRANTS}}


def refuse_if_stopped(rd: Path) -> None:
    if (rd / INTEGRITY_STOP).exists():
        raise SV.RetryRefused("STOPPED_INTEGRITY is recorded (integrity violation): Stage V stopped")
    if (rd / SV.REG_FAIL).exists():
        raise SV.RetryRefused("REGRESSION_FAIL is recorded: Stage V stopped")


# ----------------------------------------------------------------------------------------------------------------------
# LOCK
# ----------------------------------------------------------------------------------------------------------------------
def read_decision(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    dec = d.get("decision") or {}
    if d.get("schema") != SCHEMA_DECISION or "hard_stop" not in dec:
        raise StageVError(f"{path}: not an H335 Phase T decision")
    if d.get("hard_stop") or dec.get("hard_stop"):
        raise StageVError("Phase T decision is a HARD STOP: no LOCK, DL-Val stays unread")
    g, stack = int(dec["G_star_after_time_cap"]), dec["stack"]
    if g not in K.CHECKPOINTS or stack not in LAB.STACKS:
        raise StageVError(f"invalid locked setting G*={g} stack={stack!r}")
    locked = f"AGG.g{g}.{stack}"
    if d.get("locked_output", locked) != locked:
        raise StageVError(f"decision locked_output {d.get('locked_output')} != {locked}")
    return {"file": _abs(path), "sha256": C.sha256_file(Path(path)), "G_star": int(dec["G_star"]),
            "G_star_after_time_cap": g, "stack": stack, "hard_stop": False, "locked_output": locked,
            "projected_hours_v": (dec.get("projection") or {}).get("projected_hours_v"),
            "agg_invalid_share_locked": dec.get("agg_invalid_share_locked")}


def verify_phase_t(run_dirs: Sequence[Path]) -> Dict[str, dict]:
    """Every Phase T run dir ran exactly the current Phase T code (config.json code_sha256 over the H317 pins and
    h335_lab.PHASE_T_FILES == the files now at the H335 source path)."""
    if not run_dirs:
        raise StageVError("write-lock needs --phase-t-run-dirs")
    want = K.code_manifest(LAB.PHASE_T_FILES)
    out = {}
    for rd in run_dirs:
        p = Path(rd) / "config.json"
        cfg = json.loads(p.read_text())
        got = cfg.get("code_sha256") or {}
        bad = sorted(Path(k).name for k, v in want.items() if got.get(k) != v)
        if bad or cfg.get("schema") != "jope.h335.run-config/v1" or cfg.get("phase") != "T":
            raise StageVError(f"{p}: Phase T code differs from the current files: {bad[:6]}")
        out[_abs(p)] = {"sha256": C.sha256_file(p), "host": cfg.get("host"), "contexts": len(cfg.get("cids", []))}
    return out


def docs_manifest() -> Dict[str, str]:
    """NON-verified record of every other file in the code dir (docs, tests, helper scripts)."""
    return {_abs(p): C.sha256_file(p) for p in sorted(HERE.iterdir())
            if p.is_file() and not p.name.startswith(".") and p.name not in LOCK_CODE_FILES}


def lock_groups(frontier: Path, worker: Path, neural_run_dir: Path) -> Dict[str, List[str]]:
    for p, want, what in ((worker, K.SUPERSET_SHA256, "superset worker"), (frontier, FRONTIER_SHA256, "frontier binary"),
                          (SV.merged_views_path(), SV.VIEWS_MERGED_SHA256, "merged TRAIN views")):
        if C.sha256_file(Path(p)) != want:
            raise StageVError(f"{p}: {what} sha != pinned {want[:12]}…")
    models = sorted(_abs(p) for p in Path(neural_run_dir).glob("seed*/fold*/model.pt"))
    if len(models) != N_SCREENER_MODELS:
        raise StageVError(f"{neural_run_dir}: {len(models)} screener models, expected {N_SCREENER_MODELS}")
    tim = SV.write_train_inputs_manifest()  # create-once: every TRAIN table sha vs selected.json + phase pins
    return {"code": sorted(K.code_manifest(LOCK_CODE_FILES)), "micro": [_abs(C.MICRO / r) for r in MICRO_CODE],
            "worker": [_abs(worker)], "frontier": [_abs(frontier)], "screener": models,
            "views": [_abs(SV.merged_views_path())],
            "inputs": [_abs(C.RES01 / "champions.json"), _abs(C.RES01 / "winners.json"), _abs(tim)]}


def build_lock(decision: Path, phase_t_run_dirs: Sequence[Path], frontier: Path = FRONTIER, worker: Path = WORKER,
               neural_run_dir: Path = NEURAL_RUN_DIR) -> dict:
    dec = read_decision(decision)
    phase_t = verify_phase_t(phase_t_run_dirs)
    g, stack = dec["G_star_after_time_cap"], dec["stack"]
    groups = lock_groups(Path(frontier), Path(worker), Path(neural_run_dir))
    code = {k: C.sha256_file(Path(k)) for keys in groups.values() for k in keys}
    if any(code[k] != v for k, v in K.code_manifest(LOCK_CODE_FILES).items()):
        raise StageVError("an H335 source file changed while the LOCK was written")
    tim = SV.train_inputs_manifest_path()
    return {
        "schema": SCHEMA_LOCK, "wave": K.WAVE, "slug": K.SLUG, "created_utc": time.time(), "host": socket.gethostname(),
        "stage_v_runs": True,  # read by the frozen SV.cmd_unlock / verify_lock; a hard stop never produces a LOCK
        "prereg_sha256": C.sha256_file(HERE / "PREREG.md"),
        "decision": dec, "phase_t_run_configs": phase_t,
        "design": {"G_star": g, "stack": stack, "n_prop": K.N_PROP, "n_live": K.N_LIVE, "old_offset": K.OLD_OFFSET,
                   "seeds": list(K.SEEDS_V), "checkpoints": sorted({K.H317_BASE_GEN, g}), "gens_agg": g,
                   "gens_base": g + K.OLD_OFFSET, "stacks": list(LAB.STACKS), "timeout_s": GP_TIMEOUT_S,
                   "cpu_limit_factor": K.CPU_LIMIT_FACTOR, "wall_backstop_factor": K.WALL_BACKSTOP_FACTOR,
                   "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "worker": _abs(worker),
                   "worker_sha256": K.SUPERSET_SHA256,
                   "grammar_sha256": {"AGG": K.GRAMMAR_PORT_SHA256, "BASE": K.H317_PINS["gp_grammar_h317.py"]},
                   "neural_run_dir": _abs(neural_run_dir), "max_unit_attempts": MAX_UNIT_ATTEMPTS,
                   "screener": "fold-0 H0.1-A OOF pair (mean of its 2 training seeds) for every target"},
        "entrants": entrant_keys(g, stack), "elo_anchor": K.ELO_ANCHOR,
        "primary": {"P1": ["H335_AGG", P1_CTRL], "P2": ["H335_AGG", P2_CTRL]},
        "llm": {"binary": _abs(frontier), "binary_sha256": FRONTIER_SHA256, "entrants": list(FA.ENTRANTS),
                "seed": FA.SEED, "cpu_limit_s": LLM_CPU_LIMIT_S, "wall_limit_s": LLM_WALL_LIMIT_S,
                "wrapper_version": FA.WRAPPER_VERSION,
                "envelope": "common_h317.search_and_audit_envelopes(cid, table)[1].to_bytes(): fold 0 = SEARCH rows, "
                            "fold 1 = AUDIT rows, nq = 0"},
        "rules": RULES,
        "constants": {"clip": K.CLIP, "alpha": K.ALPHA, "tie": K.TIE_TOL, "boot_seed_adoption": K.BOOT_SEED,
                      "elo_boot_seed": 335160, "n_boot": 10_000, "elo_iters": H.ELO_ITERS, "elo_prior": H.ELO_PRIOR,
                      "live_match_tol": K.LIVE_MATCH_TOL, "dedupe_flag": SV.DEDUPE_FLAG,
                      "dedupe_mode": SV.DEDUPE_INTERPRETIVE, "contract_margin": 0.001,
                      "stage_v_compute_stop": STAGE_V_COMPUTE_STOP,
                      "regression": {"reference_sha256": SV.R882_REF_SHA256, "costs_sha256": SV.R882_COSTS_SHA256,
                                     "native91_mean": SV.R882_NATIVE91_MEAN, "mean_tol": SV.R882_MEAN_TOL}},
        "stage_v": {"root": str(STAGEV_ROOT), "targets": SV.N_TARGETS, "families": SV.N_TARGET_FAMILIES,
                    "units": SV.N_TARGETS * len(K.SEEDS_V) * len(GROUPS),
                    "llm_records": SV.N_TARGETS * len(FA.ENTRANTS)},
        "train_inputs": {"manifest": _abs(tim), "manifest_sha256": C.sha256_file(tim), "n_tables": SV.N_TRAIN_TABLES,
                         "selected_allowed_by_phase": SV.SELECTED_ALLOWED_BY_PHASE},
        # prep-only (xbabe2 encoder stack): verified by prep, not by resume_stage_v, so xbabe0 needs no DOPE files
        "prep_sha256": {_abs(p): C.sha256_file(p) for p in SV.dope_files()},
        "lock_code_files": list(LOCK_CODE_FILES),
        "docs_sha256": docs_manifest(),  # NOT verified by unlock / resume
        "code_groups": groups,
        "code_sha256": code,
    }


def cmd_write_lock(args) -> None:
    run_dirs = [Path(x) for x in args.phase_t_run_dirs.split(",") if x]
    lock = build_lock(Path(args.decision), run_dirs, Path(args.frontier_binary))
    sha = SV.create_once(SV.lock_path(), SV.json_bytes(lock))  # refuses to overwrite an existing LOCK
    print(json.dumps({"lock": str(SV.lock_path()), "sha256": sha, "G_star": lock["design"]["G_star"],
                      "stack": lock["design"]["stack"], "checkpoints": lock["design"]["checkpoints"],
                      "code_files": len(lock["code_sha256"]), "docs_files": len(lock["docs_sha256"])}, indent=1))
    print(sha)


def cmd_verify_lock_files(args) -> int:
    """Rehash every LOCK file at its LOCK path on THIS host (no unlock, no Stage V state): exit 0 only if all match."""
    lp = SV.lock_path()
    lock = json.loads(lp.read_text())
    if lock.get("schema") != SCHEMA_LOCK:
        raise StageVError(f"LOCK schema {lock.get('schema')} != {SCHEMA_LOCK}")
    if set(lock["code_sha256"]) != {k for ks in lock["code_groups"].values() for k in ks}:
        raise StageVError("LOCK code_groups and code_sha256 disagree")
    files = dict(lock["code_sha256"], **(lock.get("prep_sha256", {}) if args.prep else {}))
    missing = sorted(k for k in files if not Path(k).is_file())
    changed = sorted(k for k, s in files.items() if k not in missing and C.sha256_file(Path(k)) != s)
    tin = SV.verify_train_inputs(lock, args.phase) if not (missing or changed) else {}
    out = {"lock_sha256": C.sha256_file(lp), "files": len(files), "missing": len(missing), "changed": len(changed),
           "train_tables_verified": tin.get("train_tables_verified"), "host": socket.gethostname()}
    print(json.dumps(out, sort_keys=True))
    for k in missing[:20] + changed[:20]:
        print(f"LOCK FILE {'MISSING' if k in missing else 'CHANGED'}: {k}", file=sys.stderr)
    return 6 if (missing or changed) else 0


def cmd_unlock(args) -> int:
    lock = json.loads(SV.lock_path().read_text())
    if lock.get("schema") != SCHEMA_LOCK or (lock.get("decision") or {}).get("hard_stop") is not False:
        raise StageVError("not an H335 LOCK (or a hard stop): refusing the unlock")
    return SV.cmd_unlock(args)


def enter(phase: str) -> Tuple[dict, str, dict]:
    lock, sha, rep = SV.enter_stage_v(phase)  # receipt, schema, every LOCK file, marker, then resume_stage_v
    if phase == "prep":
        bad = [p for p, s in sorted(lock.get("prep_sha256", {}).items()) if C.sha256_file(Path(p)) != s]
        if bad:
            raise C.GuardViolation(f"prep: encoder-stack file(s) changed after LOCK: {bad[:5]}")
    return lock, sha, rep


def _targets(rd: Path) -> Dict[str, dict]:
    man = json.loads((rd / "targets_manifest.json").read_text())["targets"]
    if len(man) != SV.N_TARGETS or len({m["family_id"] for m in man.values()}) != SV.N_TARGET_FAMILIES:
        raise StageVError(f"targets manifest is not {SV.N_TARGETS} contexts / {SV.N_TARGET_FAMILIES} families")
    return man


# ----------------------------------------------------------------------------------------------------------------------
# Prep (xbabe2)
# ----------------------------------------------------------------------------------------------------------------------
def _dev_rows() -> List[dict]:
    sel = {r["context_sha256"]: r for r in C.load_selected()["dev"]}
    return [sel[r["context_sha256"]] for r in C.dev_rows_for_stage_v()]  # full original rows (table / pool sha)


def admit_prep(rd: Path, resume: bool) -> str:
    """'first' | 'resume'. Any number of resumes before FIRST_OUTCOME_READ; recorded steps are never recomputed."""
    rd.mkdir(parents=True, exist_ok=True)
    refuse_if_stopped(rd)
    if (rd / SV.FIRST_OUTCOME).exists():
        raise SV.RetryRefused("FIRST_OUTCOME_READ exists: prep cannot run after the first outcome read")
    if not (rd / SV.PREP_STARTED).exists():
        if resume:
            raise SV.RetryRefused("no previous prep attempt: run without --resume")
        SV.create_once(rd / SV.PREP_STARTED, f"{time.time()} {socket.gethostname()} {os.getpid()}\n".encode())
        return "first"
    if not resume:
        raise SV.RetryRefused("prep was started before: --resume completes only never-recorded steps")
    return "resume"


def write_llm_envelopes(rows: Sequence[dict]) -> Dict[str, dict]:
    """Per context the AUDIT envelope of common_h317.search_and_audit_envelopes (fold 0 = SEARCH rows, fold 1 = AUDIT
    rows, nq = 0): the frontier binary fits fold 0 and predicts fold 1. Create-once; a resume checks the bytes."""
    out = {}
    for r in sorted(rows, key=lambda r: r["context_sha256"]):
        cid = r["context_sha256"]
        env = C.read_jtt1(C.table_path(cid))
        _, audit, mask = C.search_and_audit_envelopes(cid, env)
        data = audit.to_bytes()
        p = llm_env_dir() / f"{cid}.jtt1"
        if p.exists():
            if p.read_bytes() != data:
                raise StageVError(f"{cid[:12]}: existing LLM envelope differs from a rebuild")
        else:
            SV.create_once(p, data)
        out[cid] = {"sha256": hashlib.sha256(data).hexdigest(), "nt": int(env.nt), "width": int(env.width),
                    "n_search": int((~mask).sum()), "n_audit": int(mask.sum())}
    return out


def _prep_steps(args, lock_sha: str, rows: Sequence[dict], encoder_factory: Optional[Callable], t0: float,
                mode: str) -> None:
    rd = SV.run_dir()
    reference, costs = Path(args.reference), Path(args.costs)
    if not (rd / "inputs_verified.json").exists():
        SV.write_atomic(rd / "inputs_verified.json", SV.json_bytes(SV.verify_inputs(rows, reference, costs)))
    if not (rd / "dedupe.json").exists():  # before any outcome
        rep = SV.run_dedupe(rows)
        rep["lock_sha256"] = lock_sha
        SV.write_atomic(rd / "dedupe.json", SV.json_bytes(rep))
        SV.event("dedupe_done", n_flagged={m: len(s["flagged"]) for m, s in rep["summary"].items()})
    if not (rd / "targets_manifest.json").exists():
        enc = encoder_factory() if encoder_factory else V.DopeViewEncoder(threads=args.threads)
        parity = SV.encoder_parity(enc)
        man = SV.encode_targets(enc, rows)
        dope_after = enc.dope_sha()
        if dope_after != C.DOPE_SHA256:
            raise StageVError("DOPE hash changed during encoding")
        SV.write_atomic(rd / "encoder_parity.json", SV.json_bytes(parity))
        SV.write_atomic(rd / "targets_manifest.json", SV.json_bytes({"targets": man, "dope_sha256_after": dope_after,
                                                                     "recipe": V.RECIPE_VERSION}))
    if not (rd / "llm_envelopes.json").exists():  # outcome-free; the LLM regressors run only after a regression PASS
        SV.write_atomic(rd / "llm_envelopes.json", SV.json_bytes({"envelopes": write_llm_envelopes(rows)}))
    native = json.loads((rd / "native.json").read_text()) if (rd / "native.json").exists() else None
    if native is not None and not (native.get("regression") or {}).get("pass"):
        SV._record_regression_fail(lock_sha, native.get("regression") or {}, native.get("error") or "recorded FAIL")
        raise SV.RegressionFail("R882 Native91 regression failed (recorded): Stage V stops")
    if native is None:  # a recorded PASS is never recomputed
        try:
            SV.run_regression(lock_sha, rows, reference, costs)
        except SV.RegressionFail:
            reg = json.loads((rd / "native.json").read_text()).get("regression") or {}
            print(json.dumps({"verdict": "REGRESSION_FAIL", "bit_exact_matches": reg.get("bit_exact_matches"),
                              "contexts": reg.get("contexts")}))
            raise
    files = ("inputs_verified.json", "dedupe.json", "encoder_parity.json", "targets_manifest.json", "native.json",
             "llm_envelopes.json")
    SV.create_once(rd / SV.PREP_DONE, SV.json_bytes({"utc": time.time(), "wall_s": time.time() - t0, "mode": mode,
                                                     "lock_sha256": lock_sha, "regression_pass": True,
                                                     "files": {n: C.sha256_file(rd / n) for n in files}}))
    SV.event("prep_done", wall_s=time.time() - t0)


def run_llm(lock: dict, cids: Sequence[str], jobs: int, runner: Optional[Callable] = None) -> dict:
    llm, rd = lock["llm"], SV.run_dir()
    env_man = json.loads((rd / "llm_envelopes.json").read_text())["envelopes"]
    bad = [c[:12] for c in cids if C.sha256_file(llm_env_dir() / f"{c}.jtt1") != env_man[c]["sha256"]]
    if bad:
        raise StageVError(f"LLM envelope(s) changed after prep: {bad[:5]}")
    try:
        s = (runner or FA.run)(llm_env_dir(), list(cids), rd / LLM_RECORDS, llm["binary"],
                               cpu_limit_s=llm["cpu_limit_s"], wall_limit_s=llm["wall_limit_s"], jobs=jobs,
                               binary_sha256=llm["binary_sha256"], entrants=llm["entrants"])
    except FA.EnvelopeRefused as e:
        SV.event("llm_envelopes_refused", n=len(e.summary["refused"]))
        raise StageVError(f"{len(e.summary['refused'])} LLM envelope(s) refused by the frontier binary") from None
    return {"ran": int(s["ran"]), "skipped": int(s["skipped"]),
            "status_counts": {k: int(v) for k, v in sorted(s["status_counts"].items())}}


def run_prep(args, encoder_factory: Optional[Callable] = None, llm_runner: Optional[Callable] = None) -> dict:
    t0 = time.time()
    lock, lock_sha, rep = enter("prep")
    rd = SV.run_dir()
    mode = admit_prep(rd, args.resume)
    SV.event("prep_start", mode=mode, lock_sha256=lock_sha, enter=rep,
             step="llm" if (rd / SV.PREP_DONE).exists() else "prep")
    rows = _dev_rows()
    if not (rd / SV.PREP_DONE).exists():
        _prep_steps(args, lock_sha, rows, encoder_factory, t0, mode)
    if not (rd / LLM_DONE).exists():  # resumable: frontier_audit_h335.run appends and skips recorded pairs
        t1 = time.time()
        cids = sorted(r["context_sha256"] for r in rows)
        llm = run_llm(lock, cids, args.llm_jobs, llm_runner)
        have = FA._existing_pairs(rd / LLM_RECORDS, lock["llm"]["binary_sha256"])
        if have != {(c, e) for c in cids for e in lock["llm"]["entrants"]}:
            raise StageVError(f"LLM records incomplete after the run ({len(have)} pairs)")
        SV.create_once(rd / LLM_DONE, SV.json_bytes({"utc": time.time(), "wall_s": time.time() - t1, "summary": llm,
                                                     "records_sha256": C.sha256_file(rd / LLM_RECORDS)}))
        SV.event("llm_done", records=len(have))
    else:
        llm = json.loads((rd / LLM_DONE).read_text())["summary"]
    out = {"prep": "done", "mode": mode, "regression": "PASS", "llm": llm}
    print(json.dumps(out, sort_keys=True))  # counts only: no utility is ever printed here
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Audit (xbabe0): launcher + one process per (group, shard)
# ----------------------------------------------------------------------------------------------------------------------
def admit_audit(rd: Path) -> None:
    refuse_if_stopped(rd)
    if not (rd / SV.PREP_DONE).exists() or not json.loads((rd / SV.PREP_DONE).read_text()).get("regression_pass"):
        raise SV.RetryRefused("audit needs PREP_DONE.json with a regression PASS")
    if (rd / SV.FIRST_OUTCOME).exists():
        raise SV.RetryRefused("FIRST_OUTCOME_READ exists: no unit may run after the first outcome read")


def shard_cids(man: Dict[str, dict], k: int, n: int) -> List[str]:
    """Round-robin over the largest-first order (nt x (width + 1)): deterministic, disjoint, covering, size-balanced."""
    if not 0 <= k < n:
        raise ValueError(f"shard {k}/{n}")
    order = sorted(man, key=lambda c: (-man[c]["nt"] * (man[c]["width"] + 1), c))
    return [c for i, c in enumerate(order) if i % n == k]


def _shard_lock(rd: Path, group: str, k: int, n: int) -> "SV.ShardRunLock":
    lk = SV.ShardRunLock((k, n))
    lk.path = rd / f"AUDIT_RUNNING.{group}.s{k}of{n}"  # one flock per (group, shard)
    return lk


class AttemptLog:
    """Durable unit log of one (group, shard): unit_attempts.<group>.s<k>of<n>.jsonl, one fsynced line per event.
    "start" lines (every execution start) are disclosure only. "culprit" lines are written by the job that was running
    the unit when an in-process error propagated; only they count towards MAX_UNIT_ATTEMPTS. A unit that was merely in
    flight when its process was killed (cgroup OOM, systemctl stop, reboot) leaves a start line and no culprit line, so
    it is never capped. Counts are read over every log of the group (the shard layout is fixed by AUDIT_PLAN.json)."""

    def __init__(self, rd: Path, group: str, k: int, n: int):
        self.group = group
        self.log = SV.JsonlLog(rd / f"unit_attempts.{group}.s{k}of{n}.jsonl")
        self.starts: Dict[Tuple[str, int], int] = {}
        self.culprits: Dict[Tuple[str, int], int] = {}
        self._lock = threading.Lock()
        for p in sorted(rd.glob(f"unit_attempts.{group}.*.jsonl")):
            for r in SV.read_jsonl(p)[0]:
                d = self.culprits if r.get("type") == "culprit" else self.starts
                key = (r.get("cid"), int(r.get("seed", -1)))
                d[key] = d.get(key, 0) + 1

    def started(self, cid: str, seed: int) -> int:
        with self._lock:
            return self.starts.get((cid, int(seed)), 0)

    def culprit_count(self, cid: str, seed: int) -> int:
        with self._lock:
            return self.culprits.get((cid, int(seed)), 0)

    def _append(self, d: dict, kind: str, cid: str, seed: int, **kw) -> int:
        with self._lock:
            n = d.get((cid, int(seed)), 0) + 1
            d[(cid, int(seed))] = n
        self.log.append({"type": kind, "cid": cid, "seed": int(seed), "group": self.group, "n": n, "utc": time.time(),
                         "host": socket.gethostname(), "pid": os.getpid(), **kw})
        return n

    def start(self, cid: str, seed: int) -> int:
        return self._append(self.starts, "start", cid, seed)

    def culprit(self, cid: str, seed: int, err: BaseException) -> int:
        return self._append(self.culprits, "culprit", cid, seed, error=type(err).__name__)


def run_shard_units(rt, cids: Sequence[str], seeds: Sequence[int], design: dict, jobs: int,
                    attempts: AttemptLog) -> Tuple[List[dict], dict]:
    """The units of one shard, largest first, `jobs` at a time. Each unit runs through h335_lab.run_units (so every
    record is exactly an h335_lab.run_units record). A unit that was the culprit of MAX_UNIT_ATTEMPTS propagated errors
    is recorded failed (h335_lab.failed_unit, all keys) instead of being executed again. The first propagated error
    (integrity or infrastructure) writes its culprit line and stops the queue: running units finish, pending units are
    not started."""
    gens, cps, stacks = int(design["gens_agg"]), [int(x) for x in design["checkpoints"]], tuple(design["stacks"])
    stage = f"V_{K.GROUP}"
    q: "queue.Queue[Tuple[str, int]]" = queue.Queue()
    for c in sorted(cids, key=lambda c: (-K.table_size(c), c)):
        for s in seeds:
            q.put((c, int(s)))
    stop, lk = threading.Event(), threading.Lock()
    recs: List[dict] = []
    errors: List[BaseException] = []
    info = {"executed": 0, "replayed": 0, "reruns": 0, "capped": 0}

    def job():
        while not stop.is_set():
            try:
                c, s = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = rt.store.result(stage, c, s)
                what = "replayed"
                if rec is None:
                    k = attempts.culprit_count(c, s)
                    if k >= MAX_UNIT_ATTEMPTS:
                        rec = LAB.failed_unit(rt, "V", K.GROUP, c, s, LAB.unit_keys(K.GROUP, cps, stacks),
                                              f"host failure: culprit of {k} propagated errors (MAX_UNIT_ATTEMPTS="
                                              f"{MAX_UNIT_ATTEMPTS})")
                        what = "capped"
                    else:
                        n = attempts.started(c, s)
                        attempts.start(c, s)
                        rec = LAB.run_units(rt, "V", [c], [s], gens, cps, 1, stacks)[0]
                        what = "reruns" if n else "executed"
                with lk:
                    recs.append(rec)
                    info[what] += 1
                    info["executed"] += int(what == "reruns")
            except BaseException as e:  # noqa: BLE001 - re-raised after the join
                try:
                    attempts.culprit(c, s, e)  # before stop: this unit raised; bystanders get no culprit line
                finally:
                    stop.set()
                    with lk:
                        errors.append(e)
                return

    threads = [threading.Thread(target=job, name=f"h335v-job{i}") for i in range(max(1, int(jobs)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        integ = [e for e in errors if isinstance(e, LAB.INTEGRITY_ERRORS)]
        raise (integ or errors)[0]
    info["not_started"] = q.qsize()
    return recs, info


def stop_integrity(rd: Path, where: str, e: BaseException) -> None:
    msg = f"{type(e).__name__}: {e}"[:400]
    try:
        SV.create_once(rd / INTEGRITY_STOP, SV.json_bytes({"utc": time.time(), "host": socket.gethostname(),
                                                           "pid": os.getpid(), "where": where, "error": msg}))
    except FileExistsError:
        pass
    SV.event("integrity_stop", where=where, error=msg)


def run_audit_shard(args, rt_factory: Optional[Callable] = None) -> dict:
    if args.group != K.GROUP:
        raise SystemExit(f"--group {args.group} but H335_GROUP={K.GROUP}")
    lock, lock_sha, rep = enter("audit")
    k, n = (int(x) for x in args.shard.split("/"))
    rd = SV.run_dir()
    with _shard_lock(rd, K.GROUP, k, n):
        admit_audit(rd)
        m = rd / f"AUDIT_STARTED.{K.GROUP}.s{k}of{n}"
        if m.exists() and not args.resume:
            raise SV.RetryRefused(f"{m.name} exists: --resume completes only units without a record")
        mode = "resume" if m.exists() else "first"
        if mode == "first":
            SV.create_once(m, f"{time.time()} {socket.gethostname()} {os.getpid()}\n".encode())
        man = _targets(rd)
        cids = shard_cids(man, k, n)
        for c in cids:  # the staged inputs are the prep's
            if C.sha256_file(C.table_path(c)) != man[c]["table_sha256"] or \
                    C.sha256_file(SV.views_search_dir() / f"{c}.npz") != man[c]["search_views_sha256"]:
                raise StageVError(f"{c[:12]}: staged table / SEARCH views sha != targets manifest")
        d = lock["design"]
        seeds = [int(s) for s in d["seeds"]]
        tag = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
        attempts = AttemptLog(rd, K.GROUP, k, n)
        SV.event("audit_shard_start", group=K.GROUP, shard=args.shard, mode=mode, contexts=len(cids), attempt=tag)
        t0 = time.time()
        G.MAX_EXPANDED_WIDTH = int(d["max_expanded_width"])
        cores = SV.parse_cpus(args.cores)
        rt = None
        try:
            rt = rt_factory(cores, d) if rt_factory else LAB.build_rt(
                gp_dir(), "V", cores, Path(d["worker"]), float(d["timeout_s"]), seeds, Path(d["neural_run_dir"]),
                args.torch_threads, stagev_views=SV.views_search_dir())
            recs, info = run_shard_units(rt, cids, seeds, d, args.jobs, attempts)
            ledger = rt.ledger.check()
            bank = dict(K.D.BANK_COUNTERS)
            if bank.get("same_family_bank_hits"):
                raise C.GuardViolation("same_family_bank_hits != 0")
            status = {"group": K.GROUP, "shard": args.shard, "mode": mode, "attempt": tag, "contexts": len(cids),
                      "units": len(recs), "replayed": info["replayed"], "executed": info["executed"],
                      "reruns": info["reruns"], "capped": info["capped"],
                      "failed_units": int(sum(bool(r.get("failed")) for r in recs)),
                      "failed_outputs": int(sum(bool(o.get("failed")) for r in recs for o in r["outputs"].values())),
                      "wall_s": int(time.time() - t0), "engine": rt.engine.stats(), "audit_ledger": ledger,
                      "bank_counters": bank, "guard": C.guard_report(), "host": socket.gethostname(),
                      "lock_sha256": lock_sha}
        except LAB.INTEGRITY_ERRORS as e:
            stop_integrity(rd, f"audit-shard {K.GROUP} {args.shard}", e)
            raise
        finally:
            if rt is not None:
                rt.engine.close()
        SV.create_once(rd / f"AUDIT_DONE.{K.GROUP}.s{k}of{n}.{tag}.json", SV.json_bytes(status))  # one per attempt
        SV.event("audit_shard_done", group=K.GROUP, shard=args.shard, units=len(recs), attempt=tag)
    out = {x: status[x] for x in ("group", "shard", "mode", "contexts", "units", "replayed", "executed", "reruns",
                                  "capped", "failed_units", "failed_outputs", "wall_s")}
    print(json.dumps(out, sort_keys=True))  # counts only: the unit records stay sealed
    return out


def count_units(cids: Sequence[str], seeds: Sequence[int]) -> Dict[str, int]:
    st = L.Store(gp_dir())
    return {g: int(sum(st.n_results(f"V_{g}", c, s) > 0 for c in cids for s in seeds)) for g in GROUPS}


def cmd_audit(args) -> int:
    lock, lock_sha, rep = enter("audit")
    rd = SV.run_dir()
    admit_audit(rd)
    man = _targets(rd)
    plan = {"schema": "jope.h335.audit-plan/v1", "lock_sha256": lock_sha,
            "shards": {"AGG": int(args.shards_agg), "BASE": int(args.shards_base)}}
    p = rd / "AUDIT_PLAN.json"
    if p.exists():
        if not args.resume:
            raise SV.RetryRefused("the audit was started before: --resume completes only units without a record")
        if json.loads(p.read_text()) != plan:
            raise SV.RetryRefused("a resume must keep the LOCKed run's shard layout (AUDIT_PLAN.json)")
        mode = "resume"
    else:
        SV.create_once(p, SV.json_bytes(plan))
        mode = "first"
    SV.event("audit_start", mode=mode, lock_sha256=lock_sha, shards=plan["shards"])
    cpus, w = SV.parse_cpus(args.cpus), int(args.workers_per_shard)
    jobs = [(g, k, plan["shards"][g]) for g in GROUPS for k in range(plan["shards"][g])]
    if len(jobs) * w > len(cpus):
        raise SystemExit("not enough CPUs for shards x workers")
    (rd / "shards").mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (g, k, n) in enumerate(jobs):
        mine = ",".join(str(c) for c in cpus[i * w:(i + 1) * w])
        env = dict(os.environ, H335_GROUP=g)
        env.pop("H317_LEARNERS_V1", None)
        cmd = ["taskset", "-c", mine, "nice", "-n", "19", sys.executable, "-u", str(Path(__file__).resolve()),
               "audit-shard", "--group", g, "--shard", f"{k}/{n}", "--cores", mine, "--jobs", str(args.jobs),
               "--torch-threads", str(args.torch_threads)] + (["--resume"] if mode == "resume" else [])
        log = open(rd / "shards" / f"V_{g}_s{k}of{n}.log", "a")
        procs.append((g, k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [f"{g}:{k}" for g, k, pr, _ in procs if pr.wait() != 0]
    for *_, log in procs:
        log.close()
    seeds = [int(s) for s in lock["design"]["seeds"]]
    out = {"audit": "failed" if bad else "done", "mode": mode, "shards": len(jobs), "failed_shards": bad,
           "integrity_stop": (rd / INTEGRITY_STOP).exists(),
           "units_recorded": count_units(sorted(man), seeds), "units_expected_per_group": len(man) * len(seeds)}
    SV.event("audit_done", **out)
    print(json.dumps(out, sort_keys=True))
    return 5 if bad else 0


# ----------------------------------------------------------------------------------------------------------------------
# Stats (xbabe2)
# ----------------------------------------------------------------------------------------------------------------------
def _unit_identity(r: dict, g: str, c: str, s: int, keys: set, gens: int, design: dict) -> List[str]:
    p = []
    if r.get("schema") != LAB.SCHEMA:
        p.append("schema")
    if r.get("cid") != c or int(r.get("seed", -1)) != s or r.get("group") != g or r.get("phase") != "V":
        p.append("identity")
    if set(r.get("outputs") or {}) != keys:
        p.append("output_keys")
    if not r.get("failed"):  # an h335_lab.failed_unit record carries no grammar / worker / gens
        if r.get("grammar_sha256") != design["grammar_sha256"][g]:
            p.append("grammar")
        if r.get("worker_sha256") != design["worker_sha256"]:
            p.append("worker")
        if r.get("gens") != gens:
            p.append("gens")
    return p


def inventory(cids: Sequence[str], seeds: Sequence[int], lock: dict) -> dict:
    """BEFORE any value is read: record identities (schema, cid, seed, group, phase, output keys, grammar / worker sha,
    gens; LLM: cid, entrant, binary sha, wrapper, seed), missing / duplicated records and the record-set sha."""
    d, llm_lock = lock["design"], lock["llm"]
    cps, stacks = [int(x) for x in d["checkpoints"]], tuple(d["stacks"])
    gens = {"AGG": int(d["gens_agg"]), "BASE": int(d["gens_agg"]) + K.OLD_OFFSET}
    st, rd = L.Store(gp_dir()), SV.run_dir()
    missing, dup, files, ident = [], [], [], []
    for g in GROUPS:
        keys = set(LAB.unit_keys(g, cps, stacks))
        for c in cids:
            for s in seeds:
                nr = st.n_results(f"V_{g}", c, s)
                if nr == 0:
                    missing.append((g, c, s))
                    continue
                if nr > 1:
                    dup.append((g, c[:12], s))
                files.append(st.path(f"V_{g}", c, s))
                bad = _unit_identity(st.result(f"V_{g}", c, s), g, c, s, keys, gens[g], d)
                if bad:
                    ident.append(f"{g}:{c[:12]}:{s}:{','.join(bad)}")
    recs, torn = SV.read_jsonl(rd / LLM_RECORDS)
    have: Dict[Tuple[str, str], int] = {}
    cset, eset = set(cids), set(llm_lock["entrants"])
    for r in recs:
        key = (r.get("cid"), r.get("entrant"))
        have[key] = have.get(key, 0) + 1
        if key[0] not in cset or key[1] not in eset or r.get("binary_sha256") != llm_lock["binary_sha256"] \
                or r.get("wrapper_version") != FA.WRAPPER_VERSION or r.get("seed") != FA.SEED:
            ident.append(f"LLM:{str(key[0])[:12]}:{key[1]}")
    llm_missing = [(c, e) for c in cids for e in llm_lock["entrants"] if (c, e) not in have]
    llm_dup = [(str(c)[:12], e) for (c, e), v in have.items() if v > 1]
    h = hashlib.sha256()
    for p in files + ([rd / LLM_RECORDS] if (rd / LLM_RECORDS).exists() else []):
        h.update(f"{p.relative_to(SV.STAGEV_ROOT)}\t{C.sha256_file(p)}\n".encode())
    return {"missing_units": missing, "dup_units": dup, "missing_llm": llm_missing, "dup_llm": llm_dup,
            "identity_errors": ident, "llm_torn_lines": torn, "records_sha256": h.hexdigest(),
            "unit_files": len(files)}


def final_incomplete_allowed(rd: Path) -> Tuple[bool, str]:
    if _utcnow() >= STAGE_V_COMPUTE_STOP_UTC:
        return True, f"past the Stage V compute stop {STAGE_V_COMPUTE_STOP}"
    if not (rd / LLM_DONE).exists():
        return False, f"no run/{LLM_DONE} and before the Stage V compute stop {STAGE_V_COMPUTE_STOP}"
    p = rd / "AUDIT_PLAN.json"
    if not p.exists():
        return False, f"no AUDIT_PLAN.json and before the Stage V compute stop {STAGE_V_COMPUTE_STOP}"
    shards = json.loads(p.read_text())["shards"]
    todo = [f"{g}:{k}/{n}" for g, n in sorted(shards.items()) for k in range(int(n))
            if not list(rd.glob(f"AUDIT_DONE.{g}.s{k}of{n}.*.json"))]
    if todo:
        return False, f"planned shard(s) without AUDIT_DONE {todo[:6]} and before {STAGE_V_COMPUTE_STOP}"
    return True, "every planned audit shard recorded AUDIT_DONE and LLM_DONE.json exists"


def load_records(cids: Sequence[str], seeds: Sequence[int]) -> Tuple[Dict[str, dict], Dict[tuple, dict]]:
    st, recs = L.Store(gp_dir()), {g: {} for g in GROUPS}
    for g in GROUPS:
        for c in cids:
            for s in seeds:
                r = st.result(f"V_{g}", c, s)
                if r is not None:
                    recs[g][(c, s)] = r
    llm = {(r["cid"], r["entrant"]): r for r in SV.read_jsonl(SV.run_dir() / LLM_RECORDS)[0]}
    return recs, llm


def entrant_values(recs, llm, keys: Dict[str, str], cids: Sequence[str], seeds: Sequence[int]):
    """({entrant: {cid: seed-mean}}, complete cids): contexts with every unit (both groups, both seeds) and every LLM
    record; the GP values are h335_lab.values_by_output seed means."""
    ok = [c for c in cids if all((c, s) in recs[g] for g in GROUPS for s in seeds)
          and all((c, e) in llm for e in FA.ENTRANTS)]
    keep = set(ok)
    by_out = LAB.values_by_output({g: {k: r for k, r in recs[g].items() if k[0] in keep} for g in GROUPS})
    ent = {}
    for name, key in keys.items():
        ent[name] = ({c: float(llm[(c, key[4:])]["utility"]) for c in ok} if key.startswith("LLM:")
                     else dict(by_out.get(key, {})))
    return ent, ok, by_out


def _counts(recs, llm, keys: Dict[str, str], cids: Sequence[str], seeds: Sequence[int]) -> dict:
    """Invalid / failed / live-eval / SEARCH-utility counts per entrant over the given contexts, and the live match."""
    out = {}
    for name, key in keys.items():
        if key.startswith("LLM:"):
            st = [llm[(c, key[4:])].get("status") for c in cids if (c, key[4:]) in llm]
            out[name] = {"invalid": int(sum(s != "completed" for s in st)), "outputs": len(st)}
            continue
        g = "AGG" if key.startswith("AGG.") else "BASE"
        os_ = [recs[g][(c, s)]["outputs"][key] for c in cids for s in seeds if (c, s) in recs[g]]
        lv = [int(o["live_evals"]) for o in os_ if o.get("live_evals") is not None]
        sv = [float(o["search_utility"]) for o in os_ if o.get("search_utility") is not None]
        out[name] = {"invalid": int(sum(bool(o.get("invalid")) for o in os_)), "outputs": len(os_),
                     "failed": int(sum(bool(o.get("failed")) for o in os_)),
                     "live_evals_mean": float(np.mean(lv)) if lv else None, "live_evals_min": min(lv) if lv else None,
                     "live_evals_max": max(lv) if lv else None, "live_evals_total": int(sum(lv)),
                     "search_utility_mean": float(np.mean(sv)) if sv else None}
    gaps, viol = [], 0
    a_key, o_key = keys["H335_AGG"], keys["OLD_MATCHED"]
    for c in cids:
        for s in seeds:
            a, b = recs["AGG"].get((c, s)), recs["BASE"].get((c, s))
            la = a["outputs"][a_key].get("live_evals") if a else None
            lb = b["outputs"][o_key].get("live_evals") if b else None
            if la is not None and lb is not None:  # failed units / segments carry no live count
                gap = abs(int(la) - int(lb))
                gaps.append(gap)
                viol += int(gap > K.LIVE_MATCH_TOL)
    return {"by_entrant": out, "live_match": {"max_abs_gap_agg_vs_old": max(gaps) if gaps else None,
                                              "tolerance": K.LIVE_MATCH_TOL, "violations": viol, "pairs": len(gaps)}}


def dedupe_flagged(dd: dict) -> List[str]:
    m = SV.DEDUPE_INTERPRETIVE
    return sorted(c for c, r in dd["per_context"].items()
                  if r[m]["max_jaccard"] > SV.DEDUPE_FLAG or r[m]["max_overlap"] > SV.DEDUPE_FLAG)


def compute_metrics(ent: Dict[str, Dict[str, float]], fam_of: Dict[str, str], flagged: Sequence[str]) -> dict:
    """The PREREG §5 readout on per-context values (pure; no I/O)."""
    agg = ent["H335_AGG"]
    p1 = H.primary_test(agg, ent[P1_CTRL], fam_of)
    p2 = H.primary_test(agg, ent[P2_CTRL], fam_of)
    contrasts = {}
    for name in (P1_CTRL, P2_CTRL) + DESCRIPTIVE_CONTROLS:
        contrasts[name] = {"primary": {P1_CTRL: p1, P2_CTRL: p2}.get(name) or H.primary_test(agg, ent[name], fam_of),
                           "headline": H.headline(agg, ent[name], fam_of)}
    for name in (P1_CTRL, P2_CTRL):
        contrasts[name]["adoption_gate"] = SV._slim(LAB.adoption_gate(agg, ent[name], fam_of))
    others = {n: H.headline(agg, v, fam_of) for n, v in ent.items() if n != "H335_AGG" and n not in contrasts}
    keep = [c for c in agg if c not in set(flagged)]
    sens: dict = {"excluded_contexts": len(agg) - len(keep), "flag_rule": f"{SV.DEDUPE_INTERPRETIVE} max Jaccard or "
                  f"overlap > {SV.DEDUPE_FLAG}"}
    if len({fam_of[c] for c in keep}) >= 2:
        sub = {n: {c: ent[n][c] for c in keep} for n in (P1_CTRL, P2_CTRL, "H335_AGG")}
        q1 = H.primary_test(sub["H335_AGG"], sub[P1_CTRL], fam_of)
        q2 = H.primary_test(sub["H335_AGG"], sub[P2_CTRL], fam_of)
        sens.update(P1=q1, P2=q2, verdict=H.verdict(q1, q2))
    return {"verdict": H.verdict(p1, p2), "primary": {"P1": p1, "P2": p2, "P2_counts_only_if_P1_passes": True},
            "headline": {**H.headline(agg, ent[P1_CTRL], fam_of), "vs": P1_CTRL},
            "contrasts": contrasts, "headline_other_entrants": others,
            "elo_audit": H.elo_panel(ent, fam_of, K.ELO_ANCHOR),
            "elo_pairwise_agg_vs_h257": H.elo_panel({"H335_AGG": agg, P1_CTRL: ent[P1_CTRL]}, fam_of, K.ELO_ANCHOR),
            "family_means": {n: H.family_mean(v, fam_of) for n, v in ent.items()},
            "dedupe_sensitivity": sens}


def _walls(rd: Path) -> dict:
    def get(n, k="wall_s"):
        return json.loads((rd / n).read_text()).get(k) if (rd / n).exists() else None

    shards = {p.name: json.loads(p.read_text()) for p in sorted(rd.glob("AUDIT_DONE.*.json"))}
    eng: Dict[str, float] = {}
    for s in shards.values():
        for k, v in (s.get("engine") or {}).items():
            if isinstance(v, (int, float)) and k in SV._ENGINE_KEYS:
                eng[k] = eng.get(k, 0) + v
    return {"prep": get(SV.PREP_DONE), "llm": get(LLM_DONE),
            "audit_shard_attempts": {n: s.get("wall_s") for n, s in shards.items()}, "engine_totals": eng}


def _attempt_counts(rd: Path) -> dict:
    n: Dict[tuple, int] = {}
    cul: Dict[tuple, int] = {}
    for p in sorted(rd.glob("unit_attempts.*.jsonl")):
        g = p.name.split(".")[1]
        for r in SV.read_jsonl(p)[0]:
            key = (g, r.get("cid"), r.get("seed"))
            d = cul if r.get("type") == "culprit" else n
            d[key] = d.get(key, 0) + 1
    return {"units_started": len(n), "units_rerun": int(sum(v > 1 for v in n.values())),
            "max_starts": max(n.values()) if n else 0, "units_with_culprit_lines": len(cul),
            "culprit_lines": int(sum(cul.values()))}


def run_stats(args) -> dict:
    lock, lock_sha, rep = enter("stats")
    rd = SV.run_dir()
    if (rd / INTEGRITY_STOP).exists():
        raise SV.RetryRefused("STOPPED_INTEGRITY is recorded (integrity violation): no readout")
    if (rd / SV.REG_FAIL).exists():
        reg = json.loads((rd / "native.json").read_text())["regression"]
        m = {"schema": SCHEMA_METRICS, "wave": K.WAVE, "final": True, "verdict": "REGRESSION_FAIL",
             "regression": {k: v for k, v in reg.items() if k != "per_context"}, "guard": C.guard_report()}
        SV.write_atomic(rd / METRICS, SV.json_bytes(m, compact=True))
        print(json.dumps({"verdict": m["verdict"]}))
        return m
    for n in (SV.PREP_DONE, "targets_manifest.json", "dedupe.json"):
        if not (rd / n).exists():
            raise StageVError(f"stats needs {n}")
    man = _targets(rd)
    cids = sorted(man)
    fam_of = {c: man[c]["family_id"] for c in cids}
    d, keys = lock["design"], lock["entrants"]
    seeds = [int(s) for s in d["seeds"]]
    inv = inventory(cids, seeds, lock)
    if inv["dup_units"] or inv["dup_llm"]:
        raise StageVError(f"duplicated records (a retry?): units {inv['dup_units'][:3]} llm {inv['dup_llm'][:3]}")
    if inv["identity_errors"]:
        raise StageVError(f"{len(inv['identity_errors'])} record identity check(s) failed before any value read: "
                          f"{inv['identity_errors'][:5]}")
    complete = not inv["missing_units"] and not inv["missing_llm"]
    why = None
    if not complete:
        if not args.final_incomplete:
            raise SV.RetryRefused(f"incomplete: {len(inv['missing_units'])} unit(s) and {len(inv['missing_llm'])} LLM "
                                  "record(s) missing; records stay sealed (no FIRST_OUTCOME_READ). --final-incomplete "
                                  "reads INCONCLUSIVE")
        ok, why = final_incomplete_allowed(rd)
        if not ok:
            raise SV.RetryRefused(f"--final-incomplete refused: {why}")
    marker = rd / SV.FIRST_OUTCOME
    resumed = marker.exists()
    if resumed:
        if not args.resume:
            raise SV.RetryRefused("FIRST_OUTCOME_READ exists: --resume re-aggregates the identical record set only")
        if json.loads(marker.read_text())["records_sha256"] != inv["records_sha256"]:
            raise SV.RetryRefused("records changed after FIRST_OUTCOME_READ: nothing recorded later may enter")
    else:
        SV.create_once(marker, SV.json_bytes({"utc": time.time(), "host": socket.gethostname(), "pid": os.getpid(),
                                              "lock_sha256": lock_sha, "records_sha256": inv["records_sha256"],
                                              "complete": complete, "missing_units": len(inv["missing_units"]),
                                              "missing_llm": len(inv["missing_llm"]), "final_incomplete_reason": why}))
    SV.event("stats_resume" if resumed else "first_outcome_read", complete=complete)
    # ---------------- first value read ----------------
    recs, llm = load_records(cids, seeds)
    ent, ok, by_out = entrant_values(recs, llm, keys, cids, seeds)
    dd = json.loads((rd / "dedupe.json").read_text())
    events = [e for f in sorted(rd.glob("events*.jsonl")) for e in SV.read_jsonl(f)[0]]
    m = {"schema": SCHEMA_METRICS, "wave": K.WAVE, "final": True, "complete": complete, "verdict": "INCONCLUSIVE",
         "lock_sha256": lock_sha, "locked": {"G_star": d["G_star"], "stack": d["stack"], "checkpoints": d["checkpoints"],
                                             "seeds": seeds, "entrants": keys, "anchor": K.ELO_ANCHOR},
         "contexts_used": len(ok), "families_used": len({fam_of[c] for c in ok})}
    if m["families_used"] >= 2:
        m.update(compute_metrics(ent, fam_of, dedupe_flagged(dd)))
    m["family_means_all_outputs"] = {k: H.family_mean(v, fam_of) for k, v in sorted(by_out.items())} if ok else {}
    if not complete:
        m["verdict"] = "INCONCLUSIVE"
        m["incomplete"] = {"missing_units": len(inv["missing_units"]), "missing_llm": len(inv["missing_llm"]),
                           "allowed_because": why,
                           "rule": "incomplete run => INCONCLUSIVE (PREREG §4.5); statistics are descriptive on the "
                                   "contexts with every record"}
    cnt = _counts(recs, llm, keys, ok, seeds)
    inv_n = {n: v["invalid"] for n, v in cnt["by_entrant"].items()}
    allr = [r for g in GROUPS for r in recs[g].values()]
    m.update({
        "counts": cnt["by_entrant"], "live_match": cnt["live_match"],
        "invalid_flag": {c: inv_n["H335_AGG"] <= inv_n[c] + 1 for c in (P1_CTRL, P2_CTRL)},
        "failed_units": {g: int(sum(bool(r.get("failed")) for r in recs[g].values())) for g in GROUPS},
        "failed_outputs": {g: int(sum(bool(o.get("failed")) for r in recs[g].values() for o in r["outputs"].values()))
                           for g in GROUPS},
        "segment_errors": {g: int(sum(len(r.get("segment_errors") or {}) for r in recs[g].values())) for g in GROUPS},
        "budget_shortfall_units": int(sum(bool((r.get("evo") or {}).get("budget_shortfall")) for r in allr)),
        "seed_F_evals_not_22": int(sum(1 for r in recs["AGG"].values()
                                       if not r.get("failed") and (r.get("seed_F") or {}).get("evals_used") != 22)),
        "llm_status": {e: {s: int(sum(llm[(c, e)].get("status") == s for c in ok if (c, e) in llm))
                           for s in ("completed", "failed", "timeout")} for e in FA.ENTRANTS},
        "unit_wall_s": {g: float(sum(float(r.get("wall_s", 0.0)) for r in recs[g].values())) for g in GROUPS},
        "wall_s": _walls(rd),
        "dedupe": {md: {k: v for k, v in s.items() if k != "flagged"} | {"flagged": s["flagged"][:10]}
                   for md, s in dd["summary"].items()},
        "disclosure": {"prep_resumes": sum(e.get("event") == "prep_start" and e.get("mode") == "resume" for e in events),
                       "audit_resumes": sum(e.get("event") in ("audit_start", "audit_shard_start")
                                            and e.get("mode") == "resume" for e in events),
                       "unit_attempts": _attempt_counts(rd), "stats_resumed": resumed,
                       "llm_torn_lines": inv["llm_torn_lines"], "unit_files": inv["unit_files"],
                       "records_sha256": inv["records_sha256"], "enter_api": rep.get("api")},
        "guard": C.guard_report(),
    })
    primary = m.get("primary")
    m = SV._r(m)
    if primary is not None:
        m["primary"] = primary  # full float64 precision for the inferential numbers (exact sign-flip p etc.)
    size = len(SV.json_bytes(m, compact=True))
    if size >= SV.METRICS_MAX_BYTES:
        raise StageVError(f"{METRICS} would be {size} bytes (>= {SV.METRICS_MAX_BYTES})")
    m["bytes_compact"] = size
    SV.write_atomic(rd / METRICS, SV.json_bytes(m, compact=True))
    panel = {c: {"family": fam_of[c],
                 "gp": {n: [recs["AGG" if k.startswith("AGG.") else "BASE"][(c, s)]["outputs"][k]["value"]
                            for s in seeds if (c, s) in recs["AGG" if k.startswith("AGG.") else "BASE"]]
                        for n, k in keys.items() if not k.startswith("LLM:")},
                 "llm": {e: llm[(c, e)]["utility"] for e in FA.ENTRANTS if (c, e) in llm}} for c in cids}
    SV.write_atomic(rd / "panel_stage_v.json", SV.json_bytes(panel, compact=True))
    SV.event("stats_done", verdict=m["verdict"], complete=complete)
    print(json.dumps({"verdict": m["verdict"], "complete": complete, "bytes": size, "out": str(rd / METRICS)}))
    return m


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-lock")
    w.add_argument("--decision", required=True, help="h335_lab.py decide --out JSON (Phase T), e.g. $DATA/decision_T.json")
    w.add_argument("--phase-t-run-dirs", required=True, help="comma-separated Phase T run dirs (their config.json)")
    w.add_argument("--frontier-binary", default=str(FRONTIER))
    vf = sub.add_parser("verify-lock-files")
    vf.add_argument("--phase", default="audit", choices=("prep", "audit", "stats"),
                    help="selected.json pins accepted for the TRAIN inputs check")
    vf.add_argument("--prep", action="store_true", help="also the prep-only encoder stack (xbabe2)")
    sub.add_parser("unlock")
    p = sub.add_parser("prep")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reference", default=str(STAGEV_ROOT / "inputs" / "best_dev_checkpoint.json"))
    p.add_argument("--costs", default=str(STAGEV_ROOT / "inputs" / "r882-secondary-costs.npz"))
    p.add_argument("--threads", type=int, default=2, help="encoder threads")
    p.add_argument("--llm-jobs", type=int, default=4, help="concurrent frontier_audit_h335 processes")
    for name in ("audit", "audit-shard"):
        a = sub.add_parser(name)
        a.add_argument("--resume", action="store_true")
        a.add_argument("--jobs", type=int, default=12, help="concurrent units per shard process")
        a.add_argument("--torch-threads", type=int, default=1)
        if name == "audit":
            a.add_argument("--cpus", required=True)
            a.add_argument("--shards-agg", type=int, required=True)
            a.add_argument("--shards-base", type=int, required=True)
            a.add_argument("--workers-per-shard", type=int, required=True)
        else:
            a.add_argument("--group", required=True, choices=GROUPS)
            a.add_argument("--shard", required=True)
            a.add_argument("--cores", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--final-incomplete", action="store_true", help="read an incomplete run as INCONCLUSIVE")
    s.add_argument("--resume", action="store_true", help="re-aggregate after FIRST_OUTCOME_READ (same records only)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "write-lock":
            cmd_write_lock(args)
            return 0
        if args.cmd == "verify-lock-files":
            return cmd_verify_lock_files(args)
        if args.cmd == "unlock":
            return cmd_unlock(args)
        if args.cmd == "prep":
            run_prep(args)
            return 0
        if args.cmd == "audit":
            return cmd_audit(args)
        if args.cmd == "audit-shard":
            run_audit_shard(args)
            return 0
        run_stats(args)
        return 0
    except SV.RegressionFail as e:
        print(f"REGRESSION_FAIL: {e}", file=sys.stderr)
        return 4
    except SV.RetryRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
