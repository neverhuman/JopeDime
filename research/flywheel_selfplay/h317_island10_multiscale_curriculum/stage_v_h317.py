#!/usr/bin/env python3
"""H317 Stage V driver: LOCK writer, pinned one-shot unlock, and the single DL-Val91 evaluation (PREREG §1, §2.3, §2.6,
§7 amendment A1). Built and tested on SYNTHETIC fixtures and TRAIN data only; it has never been run on DL-Val.

Subcommands (see stage_v_mounts.md for hosts, mounts and staging):
  write-lock  xbabe2, sandbox_h317.sh. Assemble $H317_DATA/stage_v/LOCK.json from the Gate H0 decisions (H0.1-H0.3
              `rapid_hypothesis_lab.py decide --out` JSONs, the GP decisions.json) and the `lock-fit` model dir:
              locked configs, seeds, pipelines, decision rules and `code_sha256` for every H317 source file, the
              worker binary, the frozen-DOPE/encoder files and every LOCK model file. Create-once, fsynced; prints
              its sha256. The parent writes the create-once receipt ~/jope_plan/ns/receipts/H317_LOCK_sha256.txt.
  unlock      xbabe2, `sandbox_h317_stagev.sh unlock` (= the TRAIN sandbox + the receipt bound read-only; no DL-Val
              byte visible). Refuses when the LOCK
              records no Gate H0 adoption (verdict NO_ADOPTION_STAGE_V_SKIPPED); otherwise
              `common_h317.unlock_stage_v()` (receipt == sha256(LOCK.json), every code hash rechecked, create-once
              STAGE_V_STARTED marker). DL-Val inputs are staged only after this marker exists.
  run --phase prep    xbabe2, `sandbox_h317_stagev.sh v` (V view: TRAIN + 91 DL-Val tables/pools at RES01).
              a. enter Stage V: common_h317.resume_stage_v() (the existing STAGE_V_STARTED must name this LOCK;
                 receipt + every LOCK file re-verified at its exact LOCK path; creates nothing)
              b. input verification (sha256 only, no decode): tables, pools, R882 reference, r882 costs
              c. dedupe (descriptive): per-row fingerprints of standardized and of raw column-sorted (interpretive)
                 feature rows, every DL-Val context vs all 1,219 TRAIN tables; recorded BEFORE any outcome read
              d. encoder host parity: 2 TRAIN SEARCH view sets + full view 0 re-encoded bit-exact vs stored ones
              e. DL-Val SEARCH-row view sets (canonical 32-view recipe, kind "search") + full-table view 0
              ---- FIRST_OUTCOME_READ (create-once, fsynced) ----
              f. R882 Native91 regression (H258 runner mapping: pick = argmin of the r882 total_cost vector of
                 r882-secondary-costs.npz[cid] (float32), utility = pool native utility at the pick; 91/91 bit-exact
                 vs best_dev_checkpoint.json utility.per_context (aligned by evaluation_contract.contexts[].cid),
                 mean within 1e-12 of 1.1377941898164525). Mismatch -> REGRESSION_FAIL: the stage stops here.
              g. TOP4_PICK (H258 definition, descriptive) and V-native (LOCK neural model per training seed: 0-reveal
                 top-1 of the full pool from full-table view 0; per-context value = training-seed mean).
  run --phase audit   xbabe0, `sandbox_h317_docker.sh --stage-v`. 7 V-AUDIT pipelines x 91 contexts x search seeds
              (317101, 317102) through the GP adapter (`gp_lab.run_pipeline`), single LOCK screener (both lock-fit
              seeds), SEARCH-row views, a fresh runtime per call; per-context memo only. Append-only JSONL per
              (context, seed, pipeline).
  run --phase stats   intersection-union primary, ADVANCE rule, three-way readout, W/T/L, LOFO, harm, drop-top-k,
              median, invalid counts, eval counts, wall, family-grouped MM Bradley-Terry Elo -> metrics.json (< 50 KB).

No-retry rule (PREREG §2.6). Before FIRST_OUTCOME_READ exists, a failed prep may be relaunched ONCE with
--relaunch-before-outcome (create-once RELAUNCH_USED; refused if any outcome record exists). After it exists, nothing
that was recorded is ever recomputed: --resume only completes units that have no record (a unit that was started but
not recorded is re-executed with byte-identical locked code and seeds and flagged `rerun_after_interruption`; every
resume is logged and disclosed in metrics). A recorded REGRESSION_FAIL stops every later phase.
After FIRST_OUTCOME_READ no unit raises: a pipeline exception or a non-finite value becomes a recorded unit with the
preregistered outcome-independent substitution (value 0.0 = SEARCH-mean predictor, invalid, error kept; SCREEN_TOPK_PICK
is recorded failed when H317_LOCKED failed, K being undefined), a budget mismatch becomes a recorded
`live_match_violation`; both are counted in the metrics disclosure. Only transient host failures (memory / disk /
descriptors / EAGAIN / thread exhaustion: no record, so --resume executes the unit, at most twice before it is recorded
failed) and integrity violations (guard / leak / ledger: the stage stops) propagate.
One process per audit shard (flock on AUDIT_RUNNING.s<k>of<n>).
Every Stage V host must present every LOCK file (source, worker, DOPE/encoder, LOCK models) at its exact LOCK path:
common_h317.resume_stage_v() rehashes all of them in every phase.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# Constants (PREREG §2.6 / §7 A1)
# ----------------------------------------------------------------------------------------------------------------------
SCHEMA_LOCK = "jope.h317.lock/v1"
SCHEMA_METRICS = "jope.h317.stage-v-metrics/v1"
# Everything DL-Val-bearing (V view, R882 inputs, DL-Val views, run records) lives here, outside $H317_DATA, so no
# Gate H0 sandbox ever has it bound. Names avoid the guard's dev tokens.
STAGEV_ROOT = Path(os.environ.get("H317_STAGEV_ROOT", "/home/ubuntu/jope-data/h317-stagev"))

PIPELINES = ("BANK_NN1", "H317_LOCKED", "EVOA_MATCHED", "SCREEN_TOPK_PICK", "H257_REPLICA", "BANK_NN8_PICK",
             "BANK_NN8_PORTFOLIO")  # execution order per (context, seed); SCREEN_TOPK_PICK needs H317_LOCKED first
PRIMARY = "H317_LOCKED"
CONTROLS = ("EVOA_MATCHED", "SCREEN_TOPK_PICK")
PORTFOLIO_REF = "BANK_NN8_PORTFOLIO"
ELO_ANCHOR_AUDIT = "EVOA_MATCHED"  # "reference = EVOA_MATCHED (dev91 AUDIT)"
NATIVE_ARMS = ("R882", "H317_NEURAL_TOP1", "TOP4_PICK")
ELO_ANCHOR_NATIVE = "R882"  # "reference = R882 slot0 (dev91)"
LIVE_MATCH_TOL = 2

# H258 manifest.json snapshot_inputs: inputs/r882/best_dev_checkpoint.json and inputs/r882-secondary-costs.npz
R882_REF_SHA256 = "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a"
R882_COSTS_SHA256 = "0e812aa7ea4ce46a85cbe17cb7eba3a58678cf13ec3b215d48a334dd5626f11f"
R882_NATIVE91_MEAN = 1.1377941898164525
R882_MEAN_TOL = 1e-12  # the H258 runner's criterion (abs diff > 1e-12 fails)
TOP4_ZGAP_THRESHOLD = 0.25
TOP4_H258_REFERENCE = 1.252000
# parent 2026-11-04: H258's 1.252000 = TOP4_PICK MEDIAN FAMILY UTILITY (median over its 3 seeds of the family-equal
# mean). TOP4_PICK reads no RNG, so every seed gives the same value: compare with the family-equal mean. Descriptive.
TOP4_H258_REFERENCE_KIND = "median over H258's 3 seeds of the family-equal mean (TOP4_PICK is seed-independent)"
WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
LOCK_BUDGET = {"steps": 3000, "slate": 128, "top_k": 48, "batch_contexts": 16, "n_seeds": 2}
LOCK_TRAIN_SEEDS = (317101, 317102)
GP_GENERATIONS = 10
GP_MAX_EXPANDED_WIDTH = 1000
GP_TIMEOUT_S = 60.0
CONTRACT_MARGIN = 0.001
SIGNFLIP_ALPHA = 0.025
UNDETECTED_BAND = 0.03
TIE_TOL = 1e-4
ELO_ITERS = 400
ELO_PRIOR = 0.01
DEDUPE_FLAG = 0.5
DEDUPE_STD_DECIMALS = 6
DEDUPE_MODES = ("std", "raw_sorted")
# parent 2026-11-04: both reported; raw_sorted is the interpretive one (per-table standardization is blind to row-subset
# duplication: 0 overlap even within TRAIN families sharing 60-75% of rows); dedupe is descriptive, never a gate.
DEDUPE_INTERPRETIVE = "raw_sorted"
METRICS_MAX_BYTES = 50_000
N_TARGETS = 91
N_TRAIN_TABLES = 1219
N_TARGET_FAMILIES = 19
GP_BASELINES = {"H0.4": "A", "H0.5": "evoA", "H0.6": "single"}
NEURAL_EXPERIMENTS = ("H0.1", "H0.2", "H0.3")

# run-dir markers / files
FIRST_OUTCOME = "FIRST_OUTCOME_READ"
RELAUNCH = "RELAUNCH_USED"
PREP_STARTED = "PREP_STARTED"
PREP_DONE = "PREP_DONE.json"
REG_FAIL = "STOPPED_REGRESSION_FAIL"
OUTCOME_FILES = ("native.json",)  # plus audit_units.*.jsonl and metrics*.json


class StageVError(RuntimeError):
    pass


class RetryRefused(StageVError):
    pass


class RegressionFail(StageVError):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# Paths, durable I/O, events
# ----------------------------------------------------------------------------------------------------------------------
def lock_path() -> Path:
    return C.DATA_ROOT / "stage_v" / "LOCK.json"  # fixed by common_h317.unlock_stage_v


def started_marker() -> Path:
    return lock_path().parent / "STAGE_V_STARTED"


def run_dir() -> Path:
    return STAGEV_ROOT / "run"


def views_search_dir() -> Path:
    return STAGEV_ROOT / "views-search"


def views_full0_dir() -> Path:
    return STAGEV_ROOT / "views-full0"


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def json_bytes(obj, compact: bool = False) -> bytes:
    if compact:
        return (json.dumps(obj, sort_keys=True, separators=(",", ":"), default=C._json_default) + "\n").encode()
    return (json.dumps(obj, sort_keys=True, indent=1, default=C._json_default) + "\n").encode()


def create_once(path: Path, data: bytes, mode: int = 0o444) -> str:
    """O_CREAT|O_EXCL write + fsync (file and directory). FileExistsError on a second attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return hashlib.sha256(data).hexdigest()


class JsonlLog:
    """Append-only, one fsynced line per record; a torn tail from a crash is closed with a newline before appending."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=C._json_default) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+b") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
                f.write(line.encode())
                f.flush()
                os.fsync(f.fileno())


def read_jsonl(path: Path) -> Tuple[List[dict], int]:
    """(records, n_unparsable_lines); unparsable lines are torn writes from an interrupted process."""
    path = Path(path)
    if not path.exists():
        return [], 0
    out, bad = [], 0
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    return out, bad


def _now() -> float:
    return time.time()


def event(kind: str, **kw) -> None:
    JsonlLog(run_dir() / "events.jsonl").append(
        {"event": kind, "utc": _now(), "host": socket.gethostname(), "pid": os.getpid(), **kw})


def _require_stage_v() -> None:
    if not C._STATE["stage_v"]:
        raise C.GuardViolation("Stage V input requested before the pinned unlock / enter_stage_v")


# ----------------------------------------------------------------------------------------------------------------------
# LOCK writer
# ----------------------------------------------------------------------------------------------------------------------
def source_files() -> List[Path]:
    """Every H317 source file: top-level *.py, *.sh, *.json and PREREG.md (DISCLOSURES/README/docs may still change)."""
    return sorted(p for p in HERE.iterdir()
                  if p.is_file() and (p.suffix in (".py", ".sh", ".json") or p.name == "PREREG.md"))


def dope_files() -> List[Path]:
    """Frozen DOPE + encoder stack used for the DL-Val views (MICRO_PINS of the view recipe + the landmark engine)."""
    import crop_shuffle_synthetic_views as V

    m = C.MICRO.resolve()
    return [m / rel for rel in sorted(V.MICRO_PINS)] + [m / "python" / "hpc" / "landmark_workspace.py",
                                                        m / "bin" / "libjope_hpc.so"]


def canonical_roots() -> Dict[str, str]:
    """Recorded for the record only: every Stage V host must present every LOCK file at its exact LOCK path."""
    return {"HERE": str(HERE), "MICRO": str(C.MICRO.resolve()), "DATA_ROOT": str(C.DATA_ROOT.resolve())}


def key_for(p: Path) -> str:
    """LOCK key: HERE-relative name for H317 source files (as common_h317.unlock_stage_v resolves them), else the
    absolute path on the LOCK host."""
    p = Path(p)
    if p.parent.resolve() == HERE:
        return p.name
    return str(p)


def resolve_key(key: str) -> Path:
    """LOCK key -> path, exactly as common_h317.unlock_stage_v / resume_stage_v resolve it (no remapping)."""
    return Path(key) if key.startswith("/") else HERE / key


def neural_decision(path: Path, experiment: str) -> dict:
    import rapid_hypothesis_lab as R

    d = json.loads(Path(path).read_text())
    plan = R.PLANNED[experiment]
    if d.get("experiment") != experiment or d.get("baseline") != plan["baseline"]:
        raise StageVError(f"{path}: not a {experiment} decision with baseline {plan['baseline']}")
    w = d.get("winner")
    if w != plan["baseline"] and w not in plan["challengers"]:
        raise StageVError(f"{path}: winner {w!r} is not a planned {experiment} arm")
    return {"file": str(path), "sha256": C.sha256_file(Path(path)), "winner": w, "baseline": plan["baseline"],
            "m": plan["m"], "adopted": w != plan["baseline"],
            "winner_family_mean": (d.get("challengers", {}).get(w, {}) or {}).get("family_mean")}


def locked_arm(w1: str, w2: str, w3: str) -> dict:
    import rapid_hypothesis_lab as R

    a1 = R.h01_arms()[w1]
    a2 = R.h02_arms(a1)[w2]
    a3 = R.h03_arms(a2)[w3]
    return asdict(a3)


def valid_stack_config(s: str) -> bool:
    if s in ("single", "CV1", "CV4", "CV5"):
        return True
    for pre in ("CV2@", "CV3@"):
        if s.startswith(pre) and s[len(pre):].isdigit():
            return True
    return False


def gp_decisions(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if "H0.4" not in d and isinstance(d.get("decisions"), dict):
        d = d["decisions"]
    out = {"file": str(path), "sha256": C.sha256_file(Path(path))}
    for name, base in GP_BASELINES.items():
        if name not in d or "used" not in d[name] or "adopted" not in d[name]:
            raise StageVError(f"{path}: missing pinned {name} decision (used/adopted)")
        out[name] = {"used": d[name]["used"], "adopted": d[name]["adopted"], "baseline": base,
                     "adoption": d[name]["adopted"] != base, "override": d[name]["used"] != d[name]["adopted"]}
    lc = {"scheme": out["H0.4"]["used"], "arm": out["H0.5"]["used"], "generations": GP_GENERATIONS,
          "stack": out["H0.6"]["used"]}
    if lc["scheme"] not in ("A", "B", "C", "D", "E", "F") or lc["arm"] not in ("evoA", "evoB") or not \
            valid_stack_config(lc["stack"]):
        raise StageVError(f"invalid locked GP config {lc}")
    out["locked_config"] = lc
    return out


def lock_model_manifest(lock_dir: Path, arm_expected: dict) -> dict:
    """lock_models.json + seed*/model.pt from `rapid_hypothesis_lab.py lock-fit`: arm == the composed H0.1-H0.3 winner,
    budget == PREREG §4, seeds 317101/317102, model sha256 as recorded, DOPE unchanged, training sets TRAIN-only."""
    import torch

    import rapid_hypothesis_lab as R

    lock_dir = Path(lock_dir).resolve()
    if not str(lock_dir).startswith(str(C.DATA_ROOT.resolve()) + os.sep):
        raise StageVError("the LOCK model dir must live under $H317_DATA (visible to the unlock sandbox; remapped on "
                          "the Stage V hosts)")
    lm = json.loads((lock_dir / "lock_models.json").read_text())
    if lm.get("arm") != arm_expected:
        raise StageVError(f"lock_models.json arm {lm.get('arm')} != composed Gate H0 winner {arm_expected}")
    if lm.get("budget") != LOCK_BUDGET:
        raise StageVError(f"lock budget {lm.get('budget')} != PREREG {LOCK_BUDGET}")
    if not lm.get("dope_unchanged", False):
        raise StageVError("lock-fit receipt does not certify an unchanged DOPE")
    seeds = sorted(int(s) for s in lm["model_sha256"])
    if tuple(seeds) != LOCK_TRAIN_SEEDS:
        raise StageVError(f"lock seeds {seeds} != {LOCK_TRAIN_SEEDS}")
    rows = C.train_rows()
    train, tfam = {r["context_sha256"] for r in rows}, {r["family_id"] for r in rows}
    sel = C.load_selected()
    dev_c = {r["context_sha256"] for r in sel["dev"]}
    dev_f = {r["family_id"] for r in sel["dev"]}
    subset = set(R.arm_a_subset(rows)) if arm_expected.get("subset") == "A525" else train
    per_seed = {}
    for s in seeds:
        p = lock_dir / f"seed{s}" / "model.pt"
        sha = C.sha256_file(p)
        if sha != lm["model_sha256"][str(s)]:
            raise StageVError(f"{p}: sha {sha} != lock_models.json")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tc, tf = set(ck["train_cids"]), set(ck["train_families"])
        if R._list_sha(ck["train_cids"]) != ck["train_cids_sha256"] or \
                R._list_sha(ck["train_families"]) != ck["train_families_sha256"]:
            raise StageVError(f"{p}: training manifest corrupt")
        if tc & dev_c or tf & dev_f:
            raise StageVError(f"{p}: trained on a DL-Val context/family")
        if tc != subset or not tf <= tfam:
            raise StageVError(f"{p}: training set ({len(tc)} contexts) is not the locked arm's training set "
                              f"(subset {arm_expected.get('subset')!r}: {len(subset)} contexts)")
        if ck.get("scale") != arm_expected["scale"] or ck.get("fold") != -1:
            raise StageVError(f"{p}: scale/fold {ck.get('scale')}/{ck.get('fold')} is not a LOCK fit")
        per_seed[str(s)] = {"sha256": sha, "train_cids_sha256": ck["train_cids_sha256"],
                            "train_families_sha256": ck["train_families_sha256"], "n_train_cids": len(tc)}
    files = sorted(p for p in lock_dir.rglob("*") if p.is_file())
    return {"lock_model_dir": str(lock_dir), "lock_models_json_sha256": C.sha256_file(lock_dir / "lock_models.json"),
            "lock_model_sha256": {s: v["sha256"] for s, v in per_seed.items()}, "per_seed": per_seed,
            "files": [str(p) for p in files], "dope_after": lm.get("dope_after")}


VIEWS_MERGED_SHA256 = "eb042276f1cdc1e409db3f1d9086ac83c9d86417f1875d6615257c60b449bb56"  # views_provenance.json
SELECTED_ALLOWED_BY_PHASE = {"prep": ("original",), "audit": ("redacted", "original"),
                             "stats": ("redacted", "original")}


def train_inputs_manifest_path() -> Path:
    return lock_path().parent / "train_inputs_manifest.json"


def merged_views_path() -> Path:
    """The merged TRAIN view file that BOTH readers use: the bank coder (passed explicitly by GPAdapter) and the
    encoder parity check (crop_shuffle_synthetic_views default). Hashed in the LOCK `inputs` group."""
    import crop_shuffle_synthetic_views as V

    return Path(V.NEURAL_ROOT) / V.MERGED_NAME


def write_train_inputs_manifest() -> Path:
    """sha256 of every TRAIN table the Stage V code may read (dedupe) as staged at the canonical RES01, each checked
    against its selected.json pin, plus the selected.json shas allowed per phase. Create-once (an identical re-write by
    a retried write-lock is accepted)."""
    tables = {}
    for r in C.train_rows():
        sha = C.sha256_file(C.table_path(r["context_sha256"]))
        if sha != r["train_only_jtt1_sha256"]:
            raise StageVError(f"TRAIN table {r['context_sha256'][:12]} sha != selected.json")
        tables[r["context_sha256"]] = sha
    man = {"schema": "jope.h317.stagev-train-inputs/v1", "res01": str(C.RES01), "tables": tables,
           "selected_sha256": {"redacted": C.SELECTED_REDACTED_SHA256, "original": C.SELECTED_SHA256},
           "selected_allowed_by_phase": SELECTED_ALLOWED_BY_PHASE}
    data, path = json_bytes(man), train_inputs_manifest_path()
    if path.exists():
        if path.read_bytes() != data:
            raise StageVError(f"{path} exists with different content")
    else:
        create_once(path, data)
    return path


def verify_train_inputs(lock: dict, phase: str) -> dict:
    """Every TRAIN table at THIS host's RES01 against the hashed manifest; selected.json against the phase's allowed
    pins (prep needs the original for the DL-Val row shas; the audit runs on the redacted copy). Only for LOCKs that
    carry the `inputs` group."""
    if "train_inputs" not in lock:
        return {}
    man = json.loads(train_inputs_manifest_path().read_text())  # its own sha was verified with code_sha256
    bad = [c[:12] for c, sha in man["tables"].items() if C.sha256_file(C.table_path(c)) != sha]
    if bad:
        raise C.GuardViolation(f"{phase}: {len(bad)} TRAIN table(s) at RES01 differ from the LOCK manifest: {bad[:5]}")
    sel = C.sha256_file(C.RES01 / "selected.json")
    allowed = {man["selected_sha256"][k] for k in man["selected_allowed_by_phase"][phase]}
    if sel not in allowed:
        raise C.GuardViolation(f"{phase}: RES01/selected.json sha {sel[:12]} not allowed in this phase")
    return {"train_tables_verified": len(man["tables"]), "selected_sha256": sel}


DECISION_RULES = {
    "contract_gate": "family-equal paired gain (arm - ref): percentile LCB95 (10,000 whole-family bootstrap draws, "
                     "RNG 317160) > +0.001; family median >= 0; <= 10% of families losing > 0.01; one-sided family "
                     "sign-flip p < 0.025 (10,000 draws, RNG 317160); leave-one-family-out minimum gain > 0",
    "primary_intersection_union": "H317_LOCKED passes the contract gate vs BOTH EVOA_MATCHED and SCREEN_TOPK_PICK "
                                  "(each at full level; IU needs no multiplicity adjustment)",
    "advance": "primary AND invalid(H317_LOCKED) <= invalid(c) + 1 for c in both controls AND family-equal point gain "
               "vs BANK_NN8_PORTFOLIO > 0",
    "readout_otherwise": "NO_ADVANCE-INFERIOR if UCB95 < 0 vs either control; else NO_ADVANCE-UNDETECTED if the "
                         "CI95 vs EVERY control lies inside (-0.03, +0.03); else INCONCLUSIVE; ties never advance",
    "invalid": "an output is INVALID iff its entire AUDIT prediction is the SEARCH-mean substitution "
               "(gp_lab.INVALID_RULE); invalid counts are summed over (context, search seed) outputs",
    "per_context_value": "mean over search seeds 317101, 317102 of the AUDIT utility (mean-predictor policy)",
    "v_native": "secondary: LOCK neural 0-reveal top-1 (training-seed mean of native utility) - R882, contract gate",
    "top4_pick": "descriptive, H258: gated by top_zgap(r882 cost) < 0.25: reveal argsort(rcost, stable)[:4], pick "
                 "argmin(fitness if gate else 2.0), ties in rcost order; else the R882 pick. Compared with H258's "
                 "1.252000 = median over its 3 seeds of the family-equal mean (TOP4_PICK is seed-independent)",
    "dedupe": "descriptive, never a gate: row fingerprints of DL-Val vs every TRAIN table, modes std (PREREG wording) "
              "and raw_sorted (interpretive); max Jaccard / overlap per context, contexts > 0.5 listed",
    "regression": "R882 Native91: pick = argmin(float32 r882 total_cost[cid]); utility = where(gate, -fitness, -2.0)"
                  "[pick]; 91/91 bit-exact vs best_dev_checkpoint.json; |mean - 1.1377941898164525| <= 1e-12",
    "elo": "family-grouped MM Bradley-Terry, 400 iterations, prior 0.01, tie 1e-4, anchors R882 (V-native panel) "
           "and EVOA_MATCHED (V-AUDIT panel) = 1500; family bootstrap CI95 (RNG 317160)",
    "stage_v_run_condition": "at least one Gate H0 adoption, else NO_ADOPTION_STAGE_V_SKIPPED (DL-Val unread)",
    "no_retry": "after FIRST_OUTCOME_READ nothing recorded is recomputed; one pre-outcome relaunch (RELAUNCH_USED)",
}


def build_lock(h01: Path, h02: Path, h03: Path, gp_dec: Path, lock_model_dir: Path,
               worker_bins: Sequence[Path] = (), extra_inputs: Sequence[Path] = ()) -> dict:
    neural = {e: neural_decision(p, e) for e, p in zip(NEURAL_EXPERIMENTS, (h01, h02, h03), strict=True)}
    arm = locked_arm(neural["H0.1"]["winner"], neural["H0.2"]["winner"], neural["H0.3"]["winner"])
    lm = lock_model_manifest(lock_model_dir, arm)
    gp = gp_decisions(gp_dec)
    adoptions = {e: neural[e]["adopted"] for e in NEURAL_EXPERIMENTS}
    adoptions.update({e: gp[e]["adoption"] for e in GP_BASELINES})
    runs = any(adoptions.values())
    workers = [Path(p).resolve() for p in (worker_bins or [C.WORKER_BIN])]
    if C.sha256_file(C.WORKER_BIN) != WORKER_SHA256:
        raise StageVError("pinned worker bootstrap.s1main sha mismatch")
    mv = merged_views_path()
    if C.sha256_file(mv) != VIEWS_MERGED_SHA256:
        raise StageVError(f"{mv}: merged TRAIN views sha != {VIEWS_MERGED_SHA256[:12]}…")
    tim = write_train_inputs_manifest()
    inputs = [C.RES01 / "champions.json", C.RES01 / "winners.json", mv, tim]
    groups: Dict[str, List[str]] = {"source": [key_for(p) for p in source_files()],
                                    "worker": [key_for(p) for p in workers],
                                    "dope": [key_for(p) for p in dope_files()],
                                    "lock_model": [key_for(Path(p)) for p in lm["files"]],
                                    "inputs": [str(p) for p in inputs]}
    if "stage_v_h317.py" not in groups["source"]:
        raise StageVError("stage_v_h317.py must be in the LOCK")
    code = {}
    for keys in groups.values():
        for k in keys:
            code[k] = C.sha256_file(resolve_key(k))
    return {
        "schema": SCHEMA_LOCK,
        "wave": C.WAVE,
        "slug": C.SLUG,
        "created_utc": _now(),
        "host": socket.gethostname(),
        "prereg_sha256": C.sha256_file(HERE / "PREREG.md"),
        "adoptions": adoptions,
        "stage_v_runs": runs,
        "verdict_if_skipped": None if runs else "NO_ADOPTION_STAGE_V_SKIPPED",
        "neural": {"decisions": neural, "locked_arm": arm, "budget": LOCK_BUDGET, "train_seeds": list(LOCK_TRAIN_SEEDS),
                   **{k: lm[k] for k in ("lock_model_dir", "lock_models_json_sha256", "lock_model_sha256", "per_seed",
                                         "dope_after")},
                   "dope_composite_sha256": C.DOPE_SHA256, "view_recipe": "h317-views-v1",
                   "v_native_context_input": "full-table view 0 (float32), as in the H0.1-H0.3 OOF predictions"},
        "gp": {"decisions": {k: gp[k] for k in GP_BASELINES}, "decisions_file": gp["file"],
               "decisions_sha256": gp["sha256"], "locked_config": gp["locked_config"],
               "max_expanded_width": GP_MAX_EXPANDED_WIDTH, "timeout_s": GP_TIMEOUT_S,
               "screener": "single LOCK screener: both lock-fit seed models averaged (rapid_hypothesis_lab.Screener)",
               "coder": "dope_view0 (D.NeuralContextCoder: target SEARCH view 0, bank full-table view 0)",
               "workers": {key_for(p): C.sha256_file(p) for p in workers}},
        "stage_v": {"targets": N_TARGETS, "families": N_TARGET_FAMILIES, "search_seeds": list(C.SEARCH_SEEDS),
                    "boot_seed": C.BOOT_SEED_STAGE_V, "n_boot": C.N_BOOT, "pipelines": list(PIPELINES),
                    "primary": PRIMARY, "controls": list(CONTROLS), "portfolio_ref": PORTFOLIO_REF,
                    "live_match_tol": LIVE_MATCH_TOL, "audit_fraction": C.AUDIT_FRACTION, "audit_seed": C.AUDIT_SEED,
                    "contract_margin": CONTRACT_MARGIN, "signflip_alpha": SIGNFLIP_ALPHA,
                    "undetected_band": UNDETECTED_BAND, "tie_tol": TIE_TOL,
                    "elo": {"iterations": ELO_ITERS, "prior": ELO_PRIOR, "tie": TIE_TOL,
                            "anchor_native": ELO_ANCHOR_NATIVE, "anchor_audit": ELO_ANCHOR_AUDIT},
                    "regression": {"reference_sha256": R882_REF_SHA256, "costs_sha256": R882_COSTS_SHA256,
                                   "native91_mean": R882_NATIVE91_MEAN, "mean_tol": R882_MEAN_TOL},
                    "top4": {"zgap_threshold": TOP4_ZGAP_THRESHOLD, "h258_reference": TOP4_H258_REFERENCE,
                             "h258_reference_kind": TOP4_H258_REFERENCE_KIND},
                    "dedupe": {"modes": list(DEDUPE_MODES), "std_decimals": DEDUPE_STD_DECIMALS,
                               "flag_threshold": DEDUPE_FLAG, "interpretive_mode": DEDUPE_INTERPRETIVE,
                               "gate": False}},
        "decision_rules": DECISION_RULES,
        "inputs_sha256": {str(p): C.sha256_file(Path(p)) for p in extra_inputs},
        "train_inputs": {"manifest": str(tim), "manifest_sha256": C.sha256_file(tim), "n_tables": N_TRAIN_TABLES,
                         "views_merged": str(mv), "views_merged_sha256": VIEWS_MERGED_SHA256,
                         "bank_files": [str(C.RES01 / "champions.json"), str(C.RES01 / "winners.json")],
                         "fold_manifest": "fold_manifest.json (source group)",
                         "selected_allowed_by_phase": SELECTED_ALLOWED_BY_PHASE,
                         "note": "inputs are hashed at the paths the code reads (canonical RES01 / $H317_DATA): on "
                                 "xbabe0 the staged copies mounted there are what resume_stage_v hashes"},
        "canonical_roots": canonical_roots(),
        "code_groups": groups,
        "code_sha256": code,
    }


def cmd_write_lock(args) -> None:
    lock = build_lock(Path(args.h01), Path(args.h02), Path(args.h03), Path(args.gp_decisions),
                      Path(args.lock_model_dir), [Path(p) for p in args.worker_bin],
                      [Path(p) for p in args.extra_input])
    out = Path(args.out) if args.out else lock_path()
    sha = create_once(out, json_bytes(lock))  # refuses to overwrite an existing LOCK
    print(json.dumps({"lock": str(out), "sha256": sha, "stage_v_runs": lock["stage_v_runs"],
                      "verdict_if_skipped": lock["verdict_if_skipped"], "adoptions": lock["adoptions"],
                      "gp_locked_config": lock["gp"]["locked_config"],
                      "neural_locked_arm": lock["neural"]["locked_arm"],
                      "code_files": len(lock["code_sha256"])}, indent=1))
    print(sha)


# ----------------------------------------------------------------------------------------------------------------------
# Unlock / enter
# ----------------------------------------------------------------------------------------------------------------------
def cmd_unlock(args) -> int:
    lock = json.loads(lock_path().read_text())
    if not lock.get("stage_v_runs"):
        print(json.dumps({"verdict": "NO_ADOPTION_STAGE_V_SKIPPED", "adoptions": lock.get("adoptions")}))
        return 3
    if started_marker().exists():
        raise RetryRefused("Stage V was already unlocked (one-shot STAGE_V_STARTED exists)")
    C.unlock_stage_v()
    rec = {"utc": _now(), "host": socket.gethostname(), "pid": os.getpid(), "lock_sha256": C.sha256_file(lock_path()),
           "marker": started_marker().read_text().strip(), "guard": C.guard_report()}
    create_once(lock_path().parent / "UNLOCK_RECEIPT.json", json_bytes(rec))
    print(json.dumps(rec))
    return 0


def verify_lock(phase: str) -> Tuple[dict, str, dict]:
    """Diagnostic pre-check mirroring common_h317.resume_stage_v (which stays authoritative): receipt == sha256(LOCK),
    schema, an adoption, EVERY LOCK file present at its exact LOCK path with its LOCK hash (all missing / changed files
    are named at once), and the existing STAGE_V_STARTED marker naming this LOCK."""
    lp = lock_path()
    posted = C.LOCK_RECEIPT.read_text().split()[0]
    got = C.sha256_file(lp)
    if got != posted:
        raise C.GuardViolation(f"LOCK.json sha {got} != posted receipt {posted}")
    lock = json.loads(lp.read_text())
    if lock.get("schema") != SCHEMA_LOCK:
        raise StageVError(f"LOCK schema {lock.get('schema')} != {SCHEMA_LOCK}")
    if not lock.get("stage_v_runs"):
        raise StageVError("NO_ADOPTION_STAGE_V_SKIPPED: the LOCK records no Gate H0 adoption")
    if set(lock["code_sha256"]) != {k for keys in lock["code_groups"].values() for k in keys}:
        raise StageVError("LOCK code_groups and code_sha256 disagree")
    missing, changed = [], []
    for key, sha in sorted(lock["code_sha256"].items()):
        p = resolve_key(key)
        if not p.exists():
            missing.append(key)
        elif C.sha256_file(p) != sha:
            changed.append(key)
    if changed:
        raise C.GuardViolation(f"{phase}: file(s) changed after LOCK: {changed[:8]}")
    if missing:
        raise StageVError(f"{phase}: {len(missing)} LOCK file(s) absent on this host (every LOCK file must be present "
                          f"at its exact LOCK path): {missing[:8]}")
    m = started_marker()
    if not m.exists():
        raise StageVError("Stage V is not unlocked: run `stage_v_h317.py unlock` first (one-shot)")
    parts = m.read_text().split()
    if len(parts) < 2 or parts[1] != got:
        raise C.GuardViolation("STAGE_V_STARTED does not bind this LOCK.json")
    return lock, got, {"phase": phase, "files_verified": len(lock["code_sha256"]),
                       **verify_train_inputs(lock, phase)}


def enter_stage_v(phase: str) -> Tuple[dict, str, dict]:
    """Enter Stage V in a process started AFTER the one-shot unlock (the DL-Val mounts appear only after it):
    the diagnostic pre-check, then common_h317.resume_stage_v() (same checks as the unlock; requires the existing
    marker naming this LOCK; creates nothing). The guard's Stage V state is set only by common_h317."""
    lock, sha, rep = verify_lock(phase)
    if not callable(getattr(C, "resume_stage_v", None)):
        raise StageVError("common_h317.resume_stage_v is required (common sha 1242bae2… or later)")
    C.resume_stage_v()
    if not C._STATE["stage_v"]:
        raise C.GuardViolation("resume_stage_v returned without entering Stage V")
    rep["api"] = "common_h317.resume_stage_v"
    return lock, sha, rep


# ----------------------------------------------------------------------------------------------------------------------
# Dedupe (PREREG §1): row fingerprints, before any outcome read
# ----------------------------------------------------------------------------------------------------------------------
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _mix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.uint64).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= _M1
        x ^= x >> np.uint64(27)
        x *= _M2
        x ^= x >> np.uint64(31)
    return x


def row_fingerprints(x: np.ndarray, mode: str) -> np.ndarray:
    """uint64 fingerprint per feature row. "std": columns z-scored within the table (affine-invariant), rounded to 6
    decimals; "raw_sorted": exact raw values sorted within the row (column-order invariant, robust to row subsets)."""
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros(0, np.uint64)
    n, w = x.shape
    if mode == "std":
        xf = np.where(np.isfinite(x), x, np.nan)
        with np.errstate(all="ignore"):
            mu = np.nanmean(xf, axis=0)
            sd = np.nanstd(xf, axis=0)
        mu = np.nan_to_num(mu)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        z = np.round((xf - mu) / sd, DEDUPE_STD_DECIMALS)
    elif mode == "raw_sorted":
        z = np.sort(x, axis=1)
    else:
        raise ValueError(mode)
    z = np.where(np.isnan(z), np.inf, z) + 0.0  # canonical NaN, -0.0 -> 0.0
    u = np.ascontiguousarray(z, np.float64).view(np.uint64).reshape(n, w)
    col = _mix64(np.arange(w, dtype=np.uint64) * _GOLD + np.uint64(w))
    h = _mix64(u ^ col[None, :]).sum(axis=1, dtype=np.uint64)
    return _mix64(h ^ np.uint64(w))


def dedupe_report(targets: Dict[str, np.ndarray], train: Iterable[Tuple[str, np.ndarray]],
                  fam_of: Dict[str, str], modes: Sequence[str] = DEDUPE_MODES) -> dict:
    """Per target context: max over TRAIN tables of the row-fingerprint Jaccard and of the overlap |A∩B|/|A|, plus the
    fraction of its rows found in ANY TRAIN table. Inputs are feature matrices (target column excluded)."""
    tcids: List[str] = []
    hs: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    ts: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    sizes: Dict[str, List[int]] = {m: [] for m in modes}
    for ti, (cid, x) in enumerate(train):
        tcids.append(cid)
        for m in modes:
            h = np.unique(row_fingerprints(x, m))
            hs[m].append(h)
            ts[m].append(np.full(len(h), ti, np.int64))
            sizes[m].append(len(h))
    nt = len(tcids)
    index = {}
    for m in modes:
        H = np.concatenate(hs[m]) if hs[m] else np.zeros(0, np.uint64)
        T = np.concatenate(ts[m]) if ts[m] else np.zeros(0, np.int64)
        o = np.argsort(H, kind="stable")
        index[m] = (H[o], T[o], np.asarray(sizes[m], np.float64))
    per: Dict[str, dict] = {}
    for cid in sorted(targets):
        x = targets[cid]
        rec = {"n_rows": int(len(x))}
        for m in modes:
            H, T, sz = index[m]
            A = np.unique(row_fingerprints(x, m))
            lo, hi = np.searchsorted(H, A, "left"), np.searchsorted(H, A, "right")
            hit = hi > lo
            counts = np.zeros(nt, np.float64)
            if hit.any():
                idx = np.concatenate([np.arange(a, b) for a, b in zip(lo[hit], hi[hit], strict=True)])
                counts = np.bincount(T[idx], minlength=nt).astype(np.float64)
            na = max(len(A), 1)
            jac = counts / np.maximum(len(A) + sz - counts, 1.0) if nt else np.zeros(0)
            ov = counts / na if nt else np.zeros(0)
            j = int(np.argmax(jac)) if nt else 0
            o = int(np.argmax(ov)) if nt else 0
            rec[m] = {"max_jaccard": float(jac[j]) if nt else 0.0,
                      "jaccard_train": tcids[j][:12] if nt else None,
                      "max_overlap": float(ov[o]) if nt else 0.0,
                      "overlap_train": tcids[o][:12] if nt else None,
                      "overlap_train_family": fam_of.get(tcids[o]) if nt else None,
                      "frac_rows_in_any_train": float(hit.mean()) if len(A) else 0.0,
                      "unique_rows": int(len(A))}
        per[cid] = rec
    summary = {}
    for m in modes:
        vals = [(c, r[m]) for c, r in per.items()]
        flagged = [{"cid12": c[:12], "jaccard": v["max_jaccard"], "overlap": v["max_overlap"],
                    "train12": v["overlap_train"], "frac_any": v["frac_rows_in_any_train"]}
                   for c, v in vals if v["max_jaccard"] > DEDUPE_FLAG or v["max_overlap"] > DEDUPE_FLAG]
        summary[m] = {"max_jaccard": max((v["max_jaccard"] for _, v in vals), default=0.0),
                      "max_overlap": max((v["max_overlap"] for _, v in vals), default=0.0),
                      "max_frac_rows_in_any_train": max((v["frac_rows_in_any_train"] for _, v in vals), default=0.0),
                      "n_jaccard_gt_0p5": int(sum(v["max_jaccard"] > DEDUPE_FLAG for _, v in vals)),
                      "n_overlap_gt_0p5": int(sum(v["max_overlap"] > DEDUPE_FLAG for _, v in vals)),
                      "flagged": flagged}
    return {"targets": len(per), "train_tables": nt, "modes": list(modes), "summary": summary, "per_context": per}


# ----------------------------------------------------------------------------------------------------------------------
# R882 regression, TOP4_PICK (H258 runner logic, origin/v1/next h258_top4_reveal_baseline/runner.py)
# ----------------------------------------------------------------------------------------------------------------------
def native_utility_h258(fitness: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """common_h252.load_pools: where(gate, -fitness(float64), -2.0) (== common_h317.native_utility)."""
    return np.where(np.asarray(gate, bool), -np.asarray(fitness, np.float64), -2.0).astype(np.float64)


def reference_values(ref_obj: dict) -> Tuple[List[str], Dict[str, float]]:
    """H258: cids = evaluation_contract.contexts[].cid; values = utility.per_context (same order)."""
    cids = [c["cid"] for c in ref_obj["evaluation_contract"]["contexts"]]
    vals = ref_obj["utility"]["per_context"]
    if len(cids) != len(vals) or len(set(cids)) != len(cids):
        raise RegressionFail("reference contexts / per_context length mismatch or duplicate cids")
    return cids, {c: float(v) for c, v in zip(cids, vals, strict=True)}


_SELECTION_KEYS = ("selected_choice", "selected_index")


def r882_regression(ref_order: Sequence[str], ref_vals: Dict[str, float], costs: Dict[str, np.ndarray],
                    pools: Dict[str, dict], target_order: Sequence[str], ref_contexts: Optional[Sequence[dict]] = None,
                    expected_mean: float = R882_NATIVE91_MEAN, n_expected: int = N_TARGETS) -> dict:
    """pick = int(argmin(float32 costs[cid])); u = native utility at the pick (float64); bit-exact per context; mean
    in target (selected.json dev) order as the H258 runner computed it. If a reference context carries an explicit
    selection index (one of _SELECTION_KEYS) it is cross-checked against the pick and REPORTED (`selection_mismatches`);
    the H258 runner never used it, so only the per-context utilities and the mean decide PASS."""
    per, n_match, sel_mismatch = {}, 0, 0
    ctx_by = {c.get("cid"): c for c in (ref_contexts or [])}
    for cid in ref_order:
        rc = np.asarray(costs[cid], np.float32)
        u_all = pools[cid]["utility"]
        if len(rc) != len(u_all):
            raise RegressionFail(f"{cid[:12]}: cost length {len(rc)} != pool length {len(u_all)}")
        pick = int(np.argmin(rc))
        u = float(u_all[pick])
        ok = u == ref_vals[cid]
        sel = next((ctx_by.get(cid, {}).get(k) for k in _SELECTION_KEYS if k in ctx_by.get(cid, {})), None)
        sel_ok = sel is None or int(sel) == pick
        sel_mismatch += int(not sel_ok)
        n_match += int(ok)
        per[cid] = {"pick": pick, "u": u, "ref": ref_vals[cid], "bit_exact": bool(ok), "selection_ok": bool(sel_ok)}
    same_set = set(ref_order) == set(target_order)
    mean_target = float(np.mean([per[c]["u"] for c in target_order])) if same_set else float("nan")
    mean_ref = float(np.mean([per[c]["u"] for c in ref_order]))
    ok = (same_set and len(ref_order) == n_expected and n_match == n_expected
          and abs(mean_target - expected_mean) <= R882_MEAN_TOL)
    return {"status": "PASS" if ok else "REGRESSION_FAIL", "pass": bool(ok), "contexts": len(ref_order),
            "bit_exact_matches": int(n_match), "selection_mismatches": int(sel_mismatch),
            "reference_set_equals_targets": bool(same_set), "mean_target_order": mean_target,
            "mean_reference_order": mean_ref, "expected_mean": expected_mean,
            "abs_diff": abs(mean_target - expected_mean) if same_set else None,
            "mean_bit_equal": bool(mean_target == expected_mean), "per_context": per}


def r882_top_zgap(rcost: np.ndarray) -> float:
    """stage1_hurdle_sos_critic.compute_r882_top_zgap (H252 snapshot fc0dd0e2…), on the float32 cost vector."""
    part = np.partition(rcost, 1)
    sd = float(np.std(rcost))
    if sd <= 1e-12:
        return 999.0
    return float((part[1] - part[0]) / sd)


def pick_best_revealed_gated_cost(order: Sequence[int], fitness: np.ndarray, gate: np.ndarray) -> int:
    idx = np.asarray(order, dtype=np.int64)
    y = np.where(gate[idx], fitness[idx], 2.0)
    return int(idx[int(np.argmin(y))])


def top4_pick(rcost: np.ndarray, fitness: np.ndarray, gate: np.ndarray, utility: np.ndarray) -> dict:
    """H258 TOP4_PICK (gated, equal budget) and TOP4_PICK_UNGATED (always 4 reveals; descriptive)."""
    rcost = np.asarray(rcost, np.float32)
    fitness = np.asarray(fitness, np.float64)
    gate = np.asarray(gate, bool)
    zg = r882_top_zgap(rcost)
    gated = bool(zg < TOP4_ZGAP_THRESHOLD)
    order = np.argsort(rcost, kind="stable")
    base = int(order[0])
    top4 = [int(i) for i in order[:4]]
    p_ung = pick_best_revealed_gated_cost(top4, fitness, gate)
    pick = p_ung if gated else base
    return {"top_zgap": zg, "gated": gated, "pick": pick, "u": float(utility[pick]), "pick_ungated": p_ung,
            "u_ungated": float(utility[p_ung]), "reveals": len(set(top4)) if gated else 0, "r882_pick": base}


# ----------------------------------------------------------------------------------------------------------------------
# LOCK neural model (screener + V-native)
# ----------------------------------------------------------------------------------------------------------------------
class LockModels:
    """The lock-fit models (one per training seed), sha-verified against the LOCK. `per_seed[s]` = single-model
    rapid_hypothesis_lab.Screener (V-native); `union()` = a fresh seed-averaged Screener (the GP screener)."""

    def __init__(self, lock: dict, threads: int = 1):
        import torch

        import multiscale_axial_bridge as MB
        import rapid_hypothesis_lab as R

        torch.set_num_threads(max(1, int(threads)))
        self.R = R
        self.device = torch.device("cpu")
        d = resolve_key(lock["neural"]["lock_model_dir"])
        self.models: Dict[int, object] = {}
        self.train_cids: set = set()
        self.train_families: set = set()
        self.per_seed: Dict[int, object] = {}
        for s, sha in sorted(lock["neural"]["lock_model_sha256"].items()):
            p = d / f"seed{s}" / "model.pt"
            if C.sha256_file(p) != sha:
                raise C.GuardViolation(f"LOCK model {p} changed after LOCK")
            ck = torch.load(p, map_location=self.device, weights_only=False)
            if R._list_sha(ck["train_cids"]) != ck["train_cids_sha256"] or \
                    R._list_sha(ck["train_families"]) != ck["train_families_sha256"]:
                raise R.ScreenerLeak(f"{p}: training manifest missing or corrupt")
            m = MB.build_model(ck["scale"]).to(self.device)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models[int(s)] = m
            self.train_cids |= set(ck["train_cids"])
            self.train_families |= set(ck["train_families"])
            self.per_seed[int(s)] = R.Screener([m], self.device, ck["train_cids"], ck["train_families"])

    def union(self):
        return self.R.Screener([self.models[s] for s in sorted(self.models)], self.device, self.train_cids,
                               self.train_families)

    def assert_target(self, cid: str, family: str) -> None:
        """Trivially true for DL-Val (families are disjoint from TRAIN), asserted anyway (raises ScreenerLeak)."""
        self.union().assert_target(cid, family)


def candidate_inputs(prog: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """feat412 = pool `program` (float32), proj16 = program @ W1 / sqrt(412) (bit-equal to the H58 TRAIN extract on
    1,208/1,208 champions), sem24 = the canonical Micro signature: exactly the neural training inputs."""
    import neural_data as ND

    prog = np.asarray(prog, np.float32)
    return prog, ND.micro_proj16(prog), ND.micro_sem24(prog)


def vnative_context(lm: LockModels, cid: str, family: str, view0: np.ndarray, prog: np.ndarray,
                    utility: np.ndarray) -> dict:
    """0-reveal top-1 of every candidate per training-seed model (curriculum_trainer.predict_contexts semantics:
    head 0, argsort(-score, stable)[0]); per-context value = mean native utility over the training seeds."""
    f412, p16, s24 = candidate_inputs(prog)
    v = np.asarray(view0, np.float32).reshape(1, -1)
    per = {}
    for s, scr in sorted(lm.per_seed.items()):
        sc = np.asarray(scr.score(v, f412, p16, s24, target_cid=cid, target_family=family))[0]
        top = int(np.argsort(-sc, kind="stable")[0])
        per[str(s)] = {"top1": top, "u": float(utility[top])}
    return {"per_seed": per, "u": float(np.mean([x["u"] for x in per.values()]))}


# ----------------------------------------------------------------------------------------------------------------------
# Prep phase (xbabe2 Stage V sandbox)
# ----------------------------------------------------------------------------------------------------------------------
def outcome_records_exist() -> List[str]:
    rd = run_dir()
    hits = [n for n in (FIRST_OUTCOME,) + OUTCOME_FILES if (rd / n).exists()]
    hits += sorted(p.name for p in rd.glob("audit_units.*.jsonl"))
    hits += sorted(p.name for p in rd.glob("metrics*.json"))
    hits += [REG_FAIL] if (rd / REG_FAIL).exists() else []
    return hits


def relaunch_once() -> None:
    hits = outcome_records_exist()
    if hits:
        raise RetryRefused(f"--relaunch-before-outcome refused: outcome records exist {hits}")
    try:
        create_once(run_dir() / RELAUNCH, f"{_now()} {socket.gethostname()} {os.getpid()}\n".encode())
    except FileExistsError:
        raise RetryRefused("the single pre-outcome relaunch was already used") from None


def admit_prep(resume: bool, relaunch: bool) -> str:
    """'first' | 'relaunch' | 'resume' per the no-retry rule; raises RetryRefused otherwise."""
    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / REG_FAIL).exists():
        raise RetryRefused("REGRESSION_FAIL is recorded: Stage V stopped")
    if (rd / PREP_DONE).exists():
        raise RetryRefused("prep is complete (PREP_DONE.json); nothing to resume")
    if not (rd / PREP_STARTED).exists():
        if relaunch or resume:
            raise RetryRefused("no previous prep attempt: run without --resume/--relaunch-before-outcome")
        create_once(rd / PREP_STARTED, f"{_now()} {socket.gethostname()} {os.getpid()}\n".encode())
        return "first"
    if outcome_records_exist():
        if not resume:
            raise RetryRefused("an outcome was read: no retry. --resume completes only never-recorded steps")
        return "resume"
    if not relaunch:
        raise RetryRefused("prep was started before (no outcome read yet): --relaunch-before-outcome (once)")
    relaunch_once()
    return "relaunch"


def verify_inputs(rows: Sequence[dict], reference: Path, costs: Path) -> dict:
    """sha256 of every DL-Val input against the pinned values (bytes only; nothing is decoded here)."""
    _require_stage_v()
    if C.sha256_file(C.RES01 / "selected.json") != C.SELECTED_SHA256:
        raise StageVError("prep needs the ORIGINAL selected.json (981960ad…) in the res-01 V view")
    bad = []
    for r in rows:
        cid = r["context_sha256"]
        if C.sha256_file(C.table_path(cid)) != r["train_only_jtt1_sha256"]:
            bad.append(("table", cid[:12]))
        if C.sha256_file(C.pool_path(cid)) != r["pool_sha256"]:
            bad.append(("pool", cid[:12]))
    for p, want in ((reference, R882_REF_SHA256), (costs, R882_COSTS_SHA256)):
        if C.sha256_file(p) != want:
            bad.append((p.name, "sha"))
    if bad:
        raise StageVError(f"Stage V input verification failed: {bad[:5]}")
    return {"tables": len(rows), "pools": len(rows), "reference_sha256": R882_REF_SHA256,
            "costs_sha256": R882_COSTS_SHA256, "selected_sha256": C.SELECTED_SHA256}


def run_dedupe(rows: Sequence[dict]) -> dict:
    _require_stage_v()
    train = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in train}
    targets = {r["context_sha256"]: C.read_jtt1(C.table_path(r["context_sha256"])).mat[:, :-1] for r in rows}

    def gen():
        for r in train:
            p = C.table_path(r["context_sha256"])
            yield r["context_sha256"], C.read_jtt1(p).mat[:, :-1]

    return dedupe_report(targets, gen(), fam_of)


def _view_arrays_equal(a: Dict[str, np.ndarray], path: Path) -> bool:
    with np.load(path, allow_pickle=False) as z:
        return set(a) == set(z.files) and all(np.array_equal(a[k], z[k]) for k in z.files)


def encoder_parity(enc, n: int = 2) -> dict:
    """Bit-identity of this host's encodings with the stored TRAIN encodings (xbabe1/xbabe3) on n pilot contexts:
    the SEARCH view set npz and the full-table view 0. Pre-outcome; a mismatch stops prep (relaunchable once)."""
    import crop_shuffle_synthetic_views as V

    rows = {r["context_sha256"]: r for r in C.train_rows()}
    stored = sorted(p for p in V.VIEW_SEARCH_DIR.glob("*.npz") if p.stem in rows)[:n]
    if len(stored) < n:
        raise StageVError(f"encoder parity needs {n} stored TRAIN SEARCH view sets in {V.VIEW_SEARCH_DIR}")
    merged = V.load_merged_views()
    pos = {str(c): i for i, c in enumerate(merged["cids"])}
    out = []
    for p in stored:
        cid = p.stem
        arr = V.encode_context(enc, cid, rows[cid], "search")
        env = C.read_jtt1(C.table_path(cid))
        f = enc.encode(env.mat[:, :-1], env.mat[:, -1])
        v0 = V.pack_view_feats(f["tokens_10x32"], f["dope629"], f["lm48_z"])
        out.append({"cid12": cid[:12], "search_npz_exact": _view_arrays_equal(arr, p),
                    "full_view0_exact": bool(np.array_equal(v0, merged["view0_f32"][pos[cid]]))})
    ok = all(o["search_npz_exact"] and o["full_view0_exact"] for o in out)
    if not ok:
        raise StageVError(f"encoder host parity FAILED: {out}")
    return {"contexts": out, "all_exact": ok, "host": socket.gethostname()}


def encode_targets(enc, rows: Sequence[dict]) -> Dict[str, dict]:
    """SEARCH-row view set (32 views, kind "search") + full-table view 0 per DL-Val context; resumable; sha manifest."""
    import crop_shuffle_synthetic_views as V

    _require_stage_v()
    man = {}
    for r in sorted(rows, key=lambda r: r["context_sha256"]):
        cid = r["context_sha256"]
        ps = views_search_dir() / f"{cid}.npz"
        if not V._view_file_ok(ps):
            V._save_npz(ps, V.encode_context(enc, cid, r, "search"))
        with np.load(ps, allow_pickle=False) as z:
            if str(z["kind"]) != "search":
                raise StageVError(f"{cid[:12]}: SEARCH view file kind {z['kind']}")
        pf = views_full0_dir() / f"{cid}.npz"
        env = C.read_jtt1(C.table_path(cid))
        if not pf.exists():
            f = enc.encode(env.mat[:, :-1], env.mat[:, -1])
            V._save_npz(pf, {"tokens_10x32_v0_f32": f["tokens_10x32"], "dope629_v0_f32": f["dope629"],
                             "lm48_z_v0_f32": f["lm48_z"], "recipe": np.array(V.RECIPE_VERSION),
                             "kind": np.array("full_view0"), "table_sha256": np.array(r["train_only_jtt1_sha256"])})
        man[cid] = {"family_id": r["family_id"], "dataset_id": r["dataset_id"],
                    "table_sha256": r["train_only_jtt1_sha256"], "nt": int(env.nt), "width": int(env.width),
                    "search_views_sha256": C.sha256_file(ps), "full_view0_sha256": C.sha256_file(pf)}
    return man


def full_view0(cid: str) -> np.ndarray:
    import crop_shuffle_synthetic_views as V

    with np.load(views_full0_dir() / f"{cid}.npz", allow_pickle=False) as z:
        if str(z["recipe"]) != V.RECIPE_VERSION or str(z["kind"]) != "full_view0":
            raise StageVError(f"{cid[:12]}: not a full_view0 file of {V.RECIPE_VERSION}")
        return V.pack_view_feats(z["tokens_10x32_v0_f32"], z["dope629_v0_f32"], z["lm48_z_v0_f32"])


def load_pool(cid: str) -> dict:
    _require_stage_v()
    with np.load(C.pool_path(cid), allow_pickle=False) as z:
        fitness = z["fitness"].astype(np.float64)
        gate = z["gate"].astype(bool)
        prog = z["program"].astype(np.float32)
    return {"fitness": fitness, "gate": gate, "utility": native_utility_h258(fitness, gate), "program": prog}


REGRESSION_DATA_ERRORS = (RegressionFail, KeyError, ValueError, TypeError, IndexError)  # malformed inputs = FAIL


def _integrity_error(e: BaseException) -> bool:
    """Leak / guard / ledger violations: never substituted, always stop the stage (resume after the cause is fixed)."""
    return isinstance(e, C.GuardViolation) or any(
        k.__name__ in ("ScreenerLeak", "BankLeakError", "AuditLeakError", "GuardViolation") for k in type(e).__mro__)


_INFRA_ERRNOS = {errno.ENOSPC, errno.ENOMEM, errno.EDQUOT, errno.EIO, errno.EMFILE, errno.ENFILE, errno.EROFS,
                 errno.EAGAIN}
MAX_UNIT_ATTEMPTS = 2  # a unit started twice without a record (host failures) is recorded failed on the 3rd attempt


def _infra_error(e: BaseException) -> bool:
    """Transient host failures (memory / disk / descriptors / EAGAIN / thread exhaustion): the unit gets NO record, so
    --resume executes it again (at most MAX_UNIT_ATTEMPTS starts; then it is a recorded failed unit)."""
    if isinstance(e, (MemoryError, BlockingIOError)):
        return True
    if isinstance(e, RuntimeError) and "can't start new thread" in str(e):
        return True
    return isinstance(e, OSError) and getattr(e, "errno", None) in _INFRA_ERRNOS


def _record_regression_fail(lock_sha: str, reg: dict, err: str) -> None:
    """native.json + the create-once STOPPED_REGRESSION_FAIL marker, written BEFORE RegressionFail propagates."""
    rd = run_dir()
    write_atomic(rd / "native.json", json_bytes({"stage": "regression_fail", "regression": reg, "error": err,
                                                 "lock_sha256": lock_sha}))
    if not (rd / REG_FAIL).exists():
        create_once(rd / REG_FAIL, json_bytes({**{k: v for k, v in reg.items() if k != "per_context"}, "error": err}))
    event("regression_fail", bit_exact=reg.get("bit_exact_matches"), mean=reg.get("mean_target_order"), error=err)


def _costs_and_pools(costs: Path, ref_order: Sequence[str], fam: Dict[str, str]) -> Tuple[dict, dict]:
    with np.load(costs, allow_pickle=False) as z:
        missing = [c for c in ref_order if c not in z.files]
        if missing:
            raise RegressionFail(f"{len(missing)} reference cids absent from the r882 cost file")
        rc = {c: z[c].astype(np.float32) for c in ref_order}
    if set(ref_order) != set(fam):
        raise RegressionFail("reference cids are not the Stage V targets")
    return rc, {c: load_pool(c) for c in ref_order}


def run_regression(lock_sha: str, rows: Sequence[dict], reference: Path, costs: Path) -> Tuple[dict, dict, dict]:
    """The R882 Native91 regression. Any verdict-relevant failure (a mismatch, or malformed / misaligned reference or
    cost inputs) is RECORDED (native.json + STOPPED_REGRESSION_FAIL) before RegressionFail propagates; on PASS
    native.json is written immediately (stage "regression")."""
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    reg: dict = {"status": "REGRESSION_FAIL", "pass": False}
    try:
        ref = json.loads(Path(reference).read_text())
        ref_order, ref_vals = reference_values(ref)
        target_order = [r["context_sha256"] for r in C.load_selected()["dev"]]  # selected.json order (H258 val_rows)
        rc, pools = _costs_and_pools(costs, ref_order, fam)
        reg = r882_regression(ref_order, ref_vals, rc, pools, target_order,
                              ref_contexts=ref["evaluation_contract"]["contexts"])
        if not reg["pass"]:
            raise RegressionFail(f"R882 Native91 regression: {reg['bit_exact_matches']}/{reg['contexts']} bit-exact, "
                                 f"mean {reg['mean_target_order']!r} vs {R882_NATIVE91_MEAN!r}")
    except REGRESSION_DATA_ERRORS as e:
        _record_regression_fail(lock_sha, reg, f"{type(e).__name__}: {e}"[:400])
        raise RegressionFail(str(e)) from e
    write_atomic(run_dir() / "native.json", json_bytes({"stage": "regression", "regression": reg,
                                                       "lock_sha256": lock_sha}))
    event("regression_pass", bit_exact=reg["bit_exact_matches"], mean=reg["mean_target_order"])
    return reg, rc, pools


def run_native(lock_sha: str, rows: Sequence[dict], reference: Path, costs: Path, lm: Optional[LockModels],
               prior: Optional[dict] = None) -> dict:
    """FIRST_OUTCOME_READ, then the regression (or its RECORDED PASS from a previous attempt: never recomputed); then
    TOP4_PICK + V-native; native.json is rewritten with stage "complete". A V-native failure on one context gets the
    native failure convention (u = -2.0, counted as failed; never retried); leak / guard violations stop the stage."""
    _require_stage_v()
    rd = run_dir()
    if not (rd / FIRST_OUTCOME).exists():
        create_once(rd / FIRST_OUTCOME, f"{_now()} {socket.gethostname()} {os.getpid()}\n".encode())
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    if prior is not None and (prior.get("regression") or {}).get("pass"):
        reg = prior["regression"]
        event("first_outcome_read", what="TOP4/V-native after a recorded regression PASS (resume)")
        rc, pools = _costs_and_pools(costs, list(reg["per_context"]), fam)
    else:
        event("first_outcome_read", what="R882 regression")
        reg, rc, pools = run_regression(lock_sha, rows, reference, costs)
    order = list(reg["per_context"])
    out = {"stage": "complete", "regression": reg, "lock_sha256": lock_sha}
    out["top4"] = {c: top4_pick(rc[c], pools[c]["fitness"], pools[c]["gate"], pools[c]["utility"]) for c in order}
    if lm is None:
        raise StageVError("V-native needs the LOCK models")
    vn = {}
    for c in order:
        lm.assert_target(c, fam[c])
        try:
            vn[c] = vnative_context(lm, c, fam[c], full_view0(c), pools[c]["program"], pools[c]["utility"])
        except Exception as e:  # noqa: BLE001 - outcome-independent substitution, recorded
            if _integrity_error(e) or _infra_error(e):
                raise
            vn[c] = {"u": C.GATE_FAIL_UTILITY, "failed": True, "error": f"{type(e).__name__}: {e}"[:300],
                     "per_seed": {}}
    out["vnative"] = vn
    write_atomic(rd / "native.json", json_bytes(out))
    return out


def run_prep(args, encoder_factory: Optional[Callable] = None, lock_models_factory: Optional[Callable] = None) -> dict:
    t0 = _now()
    lock, lock_sha, enter = enter_stage_v("prep")
    mode = admit_prep(args.resume, args.relaunch_before_outcome)
    event("prep_start", mode=mode, lock_sha256=lock_sha, enter=enter)
    rd = run_dir()
    rows = C.dev_rows_for_stage_v()
    sel_dev = {r["context_sha256"]: r for r in C.load_selected()["dev"]}
    rows = [sel_dev[r["context_sha256"]] for r in rows]  # full original rows (table / pool sha)
    reference, costs = Path(args.reference), Path(args.costs)
    if not (rd / "inputs_verified.json").exists():
        write_atomic(rd / "inputs_verified.json", json_bytes(verify_inputs(rows, reference, costs)))
    if not (rd / "dedupe.json").exists():
        rep = run_dedupe(rows)
        rep["lock_sha256"] = lock_sha
        write_atomic(rd / "dedupe.json", json_bytes(rep))
        event("dedupe_done", summary={m: {k: v for k, v in s.items() if k != "flagged"}
                                      for m, s in rep["summary"].items()})
    if not (rd / "targets_manifest.json").exists():
        import crop_shuffle_synthetic_views as V

        enc = encoder_factory() if encoder_factory else V.DopeViewEncoder(threads=args.threads)
        parity = encoder_parity(enc)
        man = encode_targets(enc, rows)
        dope_after = enc.dope_sha()
        if dope_after != C.DOPE_SHA256:
            raise StageVError("DOPE hash changed during encoding")
        write_atomic(rd / "encoder_parity.json", json_bytes(parity))
        write_atomic(rd / "targets_manifest.json", json_bytes({"targets": man, "dope_sha256_after": dope_after,
                                                               "recipe": V.RECIPE_VERSION}))
    lm = lock_models_factory(lock) if lock_models_factory else LockModels(lock, threads=args.threads)
    if lock_models_factory is None:  # warm every V-native import (Micro sem24/proj16) BEFORE the first outcome read
        candidate_inputs(np.zeros((1, 412), np.float32))
    native = json.loads((rd / "native.json").read_text()) if (rd / "native.json").exists() else None
    if native is not None and not (native.get("regression") or {}).get("pass"):
        # a recorded FAIL whose marker write was interrupted: complete the record, never re-run the regression
        _record_regression_fail(lock_sha, native.get("regression") or {}, native.get("error") or "recorded FAIL")
        raise RegressionFail("R882 Native91 regression failed (recorded): Stage V stops")
    if native is None or native.get("stage") != "complete":
        try:
            native = run_native(lock_sha, rows, reference, costs, lm, prior=native)
        except RegressionFail:
            rec = json.loads((rd / "native.json").read_text())
            print(json.dumps({"verdict": "REGRESSION_FAIL", "error": rec.get("error"),
                              **{k: v for k, v in rec["regression"].items() if k != "per_context"}}))
            raise
    done = {"utc": _now(), "wall_s": _now() - t0, "mode": mode, "lock_sha256": lock_sha, "regression_pass": True,
            "files": {n: C.sha256_file(rd / n) for n in ("inputs_verified.json", "dedupe.json", "encoder_parity.json",
                                                          "targets_manifest.json", "native.json")}}
    create_once(rd / PREP_DONE, json_bytes(done))
    event("prep_done", wall_s=done["wall_s"])
    return done


# ----------------------------------------------------------------------------------------------------------------------
# Audit phase (xbabe0): GP adapter + resumable unit loop
# ----------------------------------------------------------------------------------------------------------------------
def parse_cpus(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise ValueError("duplicate cpus")
    return out


def split_cores(cores: Sequence[int], jobs: int) -> List[List[int]]:
    if jobs < 1 or jobs > len(cores):
        raise ValueError(f"jobs {jobs} must be in 1..{len(cores)}")
    per = len(cores) // jobs
    return [list(cores[i * per:(i + 1) * per]) for i in range(jobs)]


class GPAdapter:
    """Thin adapter over the GP agent's stable API `gp_lab.run_pipeline` (one pipeline on one context). One adapter per
    job slot (its own worker cores). Per call: a fresh SEARCH-view provider, LOCK screener wrapper and context coder,
    so nothing but the immutable LOCK weights is shared across contexts; gp_lab builds a fresh engine/ledger/runtime
    per call (per-context envelope-keyed memo only)."""

    def __init__(self, lock: dict, cores: Sequence[int], lock_models: LockModels, gp_root: Path,
                 views_dir: Optional[Path] = None, meta_rows: Optional[Sequence[dict]] = None, runner=None,
                 guard: bool = True, screener_factory: Optional[Callable] = None,
                 coder_factory: Optional[Callable] = None):
        import gp_lab as L

        self.L = L
        self.max_expanded_width = int(lock["gp"]["max_expanded_width"])  # passed per call (gp_lab A3 argument)
        self.locked = L.LockedConfig(**lock["gp"]["locked_config"])
        self.timeout_s = float(lock["gp"]["timeout_s"])
        self.cores = list(cores)
        self.lm = lock_models
        self.gp_root = Path(gp_root)
        self.views_dir = Path(views_dir) if views_dir else views_search_dir()
        self.meta_rows = list(meta_rows) if meta_rows is not None else (
            list(C.train_rows()) + list(C.dev_rows_for_stage_v()))
        self.fam_of = {r["context_sha256"]: r["family_id"] for r in self.meta_rows}
        self.runner, self.guard = runner, guard
        self.screener_factory, self.coder_factory = screener_factory, coder_factory

    def _provider(self):
        import crop_shuffle_synthetic_views as V
        import screener_h317 as S

        vd = self.views_dir
        return S.SearchViewProvider(view_fn=lambda cid, kind: V.context_view_feats(cid, kind, view_dir=vd))

    def __call__(self, name: str, cid: str, seed: int, k_topk: Optional[int] = None) -> dict:
        import distributed_10gen_island_gp as D
        import screener_h317 as S

        if name == "SCREEN_TOPK_PICK" and k_topk is None:
            raise ValueError("SCREEN_TOPK_PICK needs K = live evals of H317_LOCKED")
        self.lm.assert_target(cid, self.fam_of[cid])  # target cid/family outside the LOCK training set
        prov = self._provider()
        scr = self.screener_factory() if self.screener_factory else S.NeuralScreenerAdapter(
            self.lm.union(), prov, name="lock_screener", family_of=self.fam_of)
        coder = self.coder_factory() if self.coder_factory else D.NeuralContextCoder(
            prov, merged_path=merged_views_path())
        return self.L.run_pipeline(name, cid, int(seed), scr, coder, locked=self.locked, cores=self.cores,
                                   gp_root=self.gp_root, k_topk=k_topk, meta_rows=self.meta_rows,
                                   timeout_s=self.timeout_s, guard=self.guard, runner=self.runner,
                                   max_expanded_width=self.max_expanded_width)


_ENGINE_KEYS = ("requests_search", "requests_audit", "memo_hits", "physical_runs", "failures", "timeouts",
                "wall_s_total", "pauses", "pause_s_total")


def compact_unit(rec: dict) -> dict:
    """The record kept per unit ("save the gene"): value, invalid, eval counts, ledger, engine counters, scalars."""
    v = rec.get("value")
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise StageVError(f"pipeline record without a finite value: {rec.get('pipeline')}")
    out = {"value": float(v), "value_m2": rec.get("value_m2"), "invalid": bool(rec.get("invalid")),
           "live_evals": int(rec.get("live_evals", -1))}
    for k in ("k", "seed_evals", "evo_live", "config", "audit_evals", "max_expanded_width"):
        if k in rec:
            out[k] = rec[k]
    su = rec.get("search_utility")  # requested GP record field (in-sample SEARCH S1 of the declared output)
    spec_search = (rec.get("spec") or {}).get("search") if isinstance(rec.get("spec"), dict) else None
    if su is None and isinstance(spec_search, dict):
        su = spec_search.get("utility")  # stacks already carry it in spec.search
    out["search_utility"] = su
    if rec.get("audit_half_utilities") is not None:
        out["audit_half_utilities"] = [float(x) for x in rec["audit_half_utilities"]]
    raw = rec.get("raw") or {}
    out["raw"] = {k: x for k, x in raw.items() if isinstance(x, (int, float, str, bool)) or x is None}
    out["audit_ledger"] = rec.get("audit_ledger")
    eng = rec.get("engine") or {}
    out["engine"] = {k: eng[k] for k in _ENGINE_KEYS if k in eng}
    out["bank_exclusion"] = rec.get("bank_exclusion")
    led = out["audit_ledger"] or {}
    if led and led.get("outputs") is not None and led.get("audit_requests") is None:
        raise StageVError("audit ledger incomplete")
    return out


def all_unit_records() -> Tuple[Dict[tuple, dict], Dict[tuple, int], int]:
    """(done units, started-unfinished counts, torn lines) over every audit_units.*.jsonl of the run dir."""
    done, started, torn = {}, {}, 0
    dups = []
    for p in sorted(run_dir().glob("audit_units.*.jsonl")):
        recs, bad = read_jsonl(p)
        torn += bad
        for r in recs:
            key = (r.get("cid"), int(r.get("seed", -1)), r.get("pipeline"))
            if r.get("type") == "unit":
                if key in done:
                    dups.append(key)
                done[key] = r
            elif r.get("type") == "unit_start":
                started[key] = started.get(key, 0) + 1
    if dups:
        raise StageVError(f"duplicated unit records (a retry?): {[(c[:12], s, p) for c, s, p in dups[:5]]}")
    return done, {k: v for k, v in started.items() if k not in done}, torn


def admit_audit(shard: Tuple[int, int], resume: bool) -> str:
    rd = run_dir()
    if (rd / REG_FAIL).exists():
        raise RetryRefused("REGRESSION_FAIL is recorded: Stage V stopped")
    if not (rd / PREP_DONE).exists():
        raise StageVError("audit needs PREP_DONE.json (prep + regression PASS)")
    if not json.loads((rd / PREP_DONE).read_text()).get("regression_pass"):
        raise RetryRefused("regression did not pass")
    k, n = shard
    m = rd / f"AUDIT_STARTED.s{k}of{n}"
    if m.exists() or (rd / f"audit_units.s{k}of{n}.jsonl").exists():
        if not resume:
            raise RetryRefused(f"audit shard {k}/{n} was started before: no retry; --resume completes only units "
                               "without a record")
        return "resume"
    create_once(m, f"{_now()} {socket.gethostname()} {os.getpid()}\n".encode())
    return "first"


FAILED_UNIT_U = 0.0  # PREREG §2.3 failure policy: a crashed output scores as the SEARCH-target-mean predictor (u = 0)


def _completed(r: Optional[dict]) -> bool:
    return bool(r) and r.get("status", "completed") == "completed" and r.get("live_evals") is not None


def matching_violations(name: str, recs: Dict[str, dict]) -> List[str]:
    """The budget-matching checks of the unit `name` (recorded on its unit record, never raised)."""
    a, out = recs.get(PRIMARY), []
    if name == "EVOA_MATCHED" and _completed(a) and _completed(recs.get(name)):
        gap = abs(a["live_evals"] - recs[name]["live_evals"])
        if gap > LIVE_MATCH_TOL:
            out.append(f"evoa_live_gap={gap}>{LIVE_MATCH_TOL}")
    if name == "SCREEN_TOPK_PICK" and _completed(a) and _completed(recs.get(name)):
        if recs[name]["live_evals"] != a["live_evals"]:
            out.append(f"screen_k_mismatch={recs[name]['live_evals']}!={a['live_evals']}")
    return out


def failed_unit(err: str) -> dict:
    """Outcome-independent substitution for a unit whose pipeline raised (or returned no finite value)."""
    return {"status": "failed", "value": FAILED_UNIT_U, "value_m2": C.GATE_FAIL_UTILITY, "invalid": True,
            "live_evals": None, "search_utility": None, "raw": {}, "audit_ledger": None, "engine": {},
            "bank_exclusion": None, "error": err[:400]}


def run_context_units(cid: str, seeds: Sequence[int], executor: Callable, log: JsonlLog, done: Dict[tuple, dict],
                      started: Dict[tuple, int], lock_sha: str, pipelines: Sequence[str] = PIPELINES) -> int:
    """All (seed, pipeline) units of one context, in order; recorded units are never recomputed. Returns # executed."""
    n = 0
    for seed in seeds:
        recs = {p: done[(cid, seed, p)] for p in pipelines if (cid, seed, p) in done}
        for name in pipelines:
            key = (cid, seed, name)
            if key in done:
                continue
            if name == "SCREEN_TOPK_PICK" and PRIMARY not in recs:
                raise StageVError("SCREEN_TOPK_PICK before H317_LOCKED")  # ordering bug: before any execution
            k = recs[PRIMARY].get("live_evals") if name == "SCREEN_TOPK_PICK" else None
            rerun = key in started
            log.append({"type": "unit_start", "cid": cid, "seed": seed, "pipeline": name, "utc": _now(),
                        "rerun_after_interruption": rerun})
            t0 = time.monotonic()
            if started.get(key, 0) >= MAX_UNIT_ATTEMPTS:  # rerun cap: a deterministic host failure cannot loop
                rec = failed_unit(f"host failure on {started[key]} prior attempts")
            elif name == "SCREEN_TOPK_PICK" and not _completed(recs[PRIMARY]):
                rec = failed_unit("K undefined: H317_LOCKED failed for this (context, seed)")
            else:
                try:
                    rec = compact_unit(executor(name, cid, seed, k))
                    rec["status"] = "completed"
                except Exception as e:  # noqa: BLE001 - recorded substitution; never raised per unit
                    if _integrity_error(e) or _infra_error(e):
                        raise  # no unit record: a leak/guard stop, or a host failure that --resume re-executes
                    rec = failed_unit(f"{type(e).__name__}: {e}")
            rec.update(type="unit", cid=cid, seed=int(seed), pipeline=name, wall_s=time.monotonic() - t0,
                       utc=_now(), host=socket.gethostname(), lock_sha256=lock_sha, rerun_after_interruption=rerun)
            if name == "SCREEN_TOPK_PICK":
                rec["k"] = k
            rec["live_match_violation"] = matching_violations(name, {**recs, name: rec})
            log.append(rec)
            done[key] = rec
            recs[name] = rec
            n += 1
    return n


def run_audit_loop(cids: Sequence[str], seeds: Sequence[int], executors: Sequence[Callable], shard: Tuple[int, int],
                   lock_sha: str, order_key: Optional[Callable[[str], tuple]] = None,
                   verify: Optional[Callable[[str], None]] = None) -> dict:
    """Contexts (LPT order) pulled from one queue by len(executors) job threads; each job owns its worker cores."""
    k, n = shard
    log = JsonlLog(run_dir() / f"audit_units.s{k}of{n}.jsonl")
    done, started, torn = all_unit_records()
    todo = sorted(cids, key=order_key) if order_key else list(cids)
    q: "queue.Queue[str]" = queue.Queue()
    for c in todo:
        q.put(c)
    stop = threading.Event()
    errors: List[Tuple[str, BaseException]] = []
    executed = [0]
    lk = threading.Lock()

    def job(ex):
        while not stop.is_set():
            try:
                cid = q.get_nowait()
            except queue.Empty:
                return
            try:
                if verify:
                    verify(cid)
                m = run_context_units(cid, seeds, ex, log, done, started, lock_sha)
                with lk:
                    executed[0] += m
            except BaseException as e:  # noqa: BLE001 - recorded, then re-raised after the join
                stop.set()
                with lk:
                    errors.append((cid, e))
                event("audit_error", cid12=cid[:12], error=f"{type(e).__name__}: {e}"[:400])
                return

    threads = [threading.Thread(target=job, args=(ex,), name=f"h317v-job{i}") for i, ex in enumerate(executors)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise StageVError(f"{len(errors)} context job(s) failed; first: {errors[0][0][:12]}: {errors[0][1]!r}")
    return {"executed_units": executed[0], "rerun_units": len([1 for k_ in started if k_[0] in set(cids)]),
            "torn_lines": torn, "contexts": len(cids)}


class ShardRunLock:
    """One process per audit shard: AUDIT_RUNNING.s<k>of<n> is created O_EXCL once and then held under an exclusive
    non-blocking flock for the process lifetime. A second live process is refused; a stale file (its holder died, so
    the kernel released the flock) is taken over and logged. The file is never deleted. flock, not a pid probe, decides
    liveness: in docker every container is a fresh pid namespace and hostname."""

    def __init__(self, shard: Tuple[int, int]):
        self.path = run_dir() / f"AUDIT_RUNNING.s{shard[0]}of{shard[1]}"
        self.fd: Optional[int] = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            stale = None
        except FileExistsError:
            fd = os.open(str(self.path), os.O_RDWR)
            stale = os.pread(fd, 4096, 0).decode(errors="replace").strip()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RetryRefused(f"audit shard already running ({self.path.name}: {stale})") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} {socket.gethostname()} {_now()}\n".encode(), 0)
        os.fsync(fd)
        self.fd = fd
        if stale is not None:
            event("audit_running_stale_taken_over", file=self.path.name, previous=stale[:200])
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)  # releases the flock; the file stays
            self.fd = None
        return False


def run_audit(args, executor_factory: Optional[Callable] = None) -> dict:
    lock, lock_sha, enter = enter_stage_v("audit")
    k, n = (int(x) for x in args.shard.split("/"))
    with ShardRunLock((k, n)):
        return _run_audit_locked(args, lock, lock_sha, enter, (k, n), executor_factory)


def _run_audit_locked(args, lock, lock_sha, enter, shard, executor_factory) -> dict:
    t0 = _now()
    k, n = shard
    mode = admit_audit((k, n), args.resume)
    rd = run_dir()
    man = json.loads((rd / "targets_manifest.json").read_text())["targets"]
    if len(man) != N_TARGETS:
        raise StageVError(f"targets manifest has {len(man)} contexts")
    cids = [c for i, c in enumerate(sorted(man)) if i % n == k]
    event("audit_start", shard=args.shard, mode=mode, lock_sha256=lock_sha, enter=enter, contexts=len(cids))

    def verify(cid):
        if C.sha256_file(C.table_path(cid)) != man[cid]["table_sha256"]:
            raise StageVError(f"{cid[:12]}: staged table sha != targets manifest")
        if C.sha256_file(views_search_dir() / f"{cid}.npz") != man[cid]["search_views_sha256"]:
            raise StageVError(f"{cid[:12]}: staged SEARCH views sha != targets manifest")

    if executor_factory is None:
        gp_root = STAGEV_ROOT / "gp"

        def executor_factory(cores):  # one LOCK model copy per job thread (no torch module shared across threads)
            return GPAdapter(lock, cores, LockModels(lock, threads=1), gp_root)

    groups = split_cores(parse_cpus(args.cpus), args.jobs)
    executors = [executor_factory(g) for g in groups]
    res = run_audit_loop(cids, C.SEARCH_SEEDS, executors, (k, n), lock_sha,
                         order_key=lambda c: (-man[c]["nt"] * man[c]["width"], c), verify=verify)
    res.update(wall_s=_now() - t0, mode=mode, shard=args.shard, jobs=args.jobs, cores_per_job=len(groups[0]))
    write_atomic(rd / f"AUDIT_DONE.s{k}of{n}.json", json_bytes(res))
    event("audit_done", **res)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------
def bt_ratings(wins: np.ndarray, anchor: int = 0, prior: float = ELO_PRIOR, iters: int = ELO_ITERS) -> np.ndarray:
    """MM Bradley-Terry Elo anchored at 1500 (copied from origin/v1/next research/flywheel_selfplay/
    h252_closed_loop_flywheel/common_h252.py::bt_ratings, sha abe974ed…, the H258 snapshot)."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    s = np.ones(wins.shape[:-1], dtype=np.float64)
    for _ in range(iters):
        denom = ((games + 2 * prior * (1 - np.eye(n))) / np.maximum(s[..., :, None] + s[..., None, :], 1e-12)).sum(-1)
        s = np.clip(totals / denom, 1e-6, 1e6)
        s /= np.maximum(s[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(s, 1e-9))


def family_wins_tensor(values_2d: np.ndarray, families: Sequence[str], tie: float = TIE_TOL) -> np.ndarray:
    """common_h252.family_wins_tensor: [N_families, A, A] equal-weighted family win shares (tie = 0.5)."""
    v = np.asarray(values_2d, np.float64)
    diff = v[:, :, None] - v[:, None, :]
    outcomes = np.where(diff > tie, 1.0, np.where(diff < -tie, 0.0, 0.5)) * (1.0 - np.eye(v.shape[1]))
    fams = np.asarray(families)
    return np.stack([outcomes[fams == f].mean(axis=0) for f in sorted(set(families))], axis=0)


def elo_panel(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str], anchor: str,
              seed: int = C.BOOT_SEED_STAGE_V, n_boot: int = C.N_BOOT) -> dict:
    arms = list(values)
    cids = sorted(set.intersection(*[set(v) for v in values.values()]))
    V = np.array([[values[a][c] for a in arms] for c in cids])
    W = family_wins_tensor(V, [fam_of[c] for c in cids])
    ai = arms.index(anchor)
    point = bt_ratings(W.sum(axis=0), anchor=ai)
    rng = np.random.default_rng(seed)
    F = W.shape[0]
    ix = rng.integers(0, F, size=(n_boot, F))
    boot = np.concatenate([bt_ratings(W[ix[i:i + 500]].sum(axis=1), anchor=ai) for i in range(0, n_boot, 500)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"anchor": anchor, "families": int(F), "contexts": len(cids),
            "elo": {a: {"elo": float(point[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, a in enumerate(arms)}}


def contract(arm: Dict[str, float], ref: Dict[str, float], fam_of: Dict[str, str],
             seed: int = C.BOOT_SEED_STAGE_V) -> dict:
    s = C.robustness_summary(arm, ref, fam_of, seed)
    s["ucb95"] = s["ci95"][1]
    s["gate"] = C.contract_advance_gate(s, "lcb95", margin=CONTRACT_MARGIN, signflip_alpha=SIGNFLIP_ALPHA)
    cids = sorted(set(arm) & set(ref))
    d = np.array([arm[c] - ref[c] for c in cids])
    s["context_wtl"] = [int((d > TIE_TOL).sum()), int((np.abs(d) <= TIE_TOL).sum()), int((d < -TIE_TOL).sum())]
    s["context_mean_gain"] = float(d.mean())
    return s


def readout(primary: Dict[str, dict], invalid: Dict[str, int], gain_vs_portfolio: float) -> dict:
    """PREREG §2.6: ADVANCE iff IU primary AND invalid rule (vs both controls) AND gain vs BANK_NN8_PORTFOLIO > 0;
    else INFERIOR (UCB95 < 0 vs either control) > UNDETECTED (every control's CI95 inside (-0.03, 0.03)) >
    INCONCLUSIVE."""
    iu = all(primary[c]["gate"]["pass"] for c in CONTROLS)
    inv_ok = {c: invalid[PRIMARY] <= invalid[c] + 1 for c in CONTROLS}
    port_ok = bool(gain_vs_portfolio > 0.0)
    failed = [n for n, ok in (("primary_intersection_union", iu), ("invalid_rule", all(inv_ok.values())),
                              ("gain_vs_BANK_NN8_PORTFOLIO", port_ok)) if not ok]
    if not failed:
        verdict = "ADVANCE"
    elif any(primary[c]["ci95"][1] < 0.0 for c in CONTROLS):
        verdict = "NO_ADVANCE-INFERIOR"
    elif all(-UNDETECTED_BAND < primary[c]["ci95"][0] and primary[c]["ci95"][1] < UNDETECTED_BAND for c in CONTROLS):
        verdict = "NO_ADVANCE-UNDETECTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "primary_iu_pass": iu, "invalid_rule": inv_ok, "gain_vs_portfolio": gain_vs_portfolio,
            "gain_vs_portfolio_pos": port_ok, "advance_failed_conditions": failed}


def _slim(s: dict) -> dict:
    keep = ("gain_family_mean", "gain_family_median", "ci95", "lcb95", "ucb95", "families", "contexts", "wtl",
            "context_wtl", "context_mean_gain", "signflip_p_one_sided", "lofo_min_gain", "harmed_share_gt_0p01",
            "drop_top_1", "drop_top_3", "drop_top_5", "drop_top_10")
    out = {k: s[k] for k in keep if k in s}
    if "gate" in s:
        out["gate_pass"] = s["gate"]["pass"]
        out["gate_checks"] = s["gate"]["checks"]
    return out


def _r(x, nd: int = 6):
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def unit_panel(done: Dict[tuple, dict], cids: Sequence[str], seeds: Sequence[int],
               pipelines: Sequence[str] = PIPELINES, partial: bool = False) -> Tuple[Dict[str, Dict[str, float]], dict]:
    missing = [(c, s, p) for c in cids for s in seeds for p in pipelines if (c, s, p) not in done]
    if missing and not partial:
        raise StageVError(f"{len(missing)} units missing, e.g. {[(c[:12], s, p) for c, s, p in missing[:3]]}")
    vals: Dict[str, Dict[str, float]] = {p: {} for p in pipelines}
    info: Dict[str, dict] = {}
    for p in pipelines:
        inv, live, su, wall, phys, memo, rerun, failed, viol, outputs = 0, [], [], 0.0, 0, 0, 0, 0, 0, 0
        for c in cids:
            per = [done[(c, s, p)] for s in seeds if (c, s, p) in done]
            if len(per) != len(seeds):
                continue
            vals[p][c] = float(np.mean([r["value"] for r in per]))
            for r in per:
                outputs += 1
                inv += int(r["invalid"])
                failed += int(r.get("status") == "failed")
                viol += int(bool(r.get("live_match_violation")))
                if r.get("live_evals") is not None:
                    live.append(int(r["live_evals"]))
                wall += float(r.get("wall_s", 0.0))
                phys += int((r.get("engine") or {}).get("physical_runs", 0))
                memo += int((r.get("engine") or {}).get("memo_hits", 0))
                rerun += int(bool(r.get("rerun_after_interruption")))
                if r.get("search_utility") is not None:
                    su.append(float(r["search_utility"]))
        info[p] = {"invalid": inv, "failed": failed, "live_match_violations": viol, "outputs": outputs,
                   "live_evals_mean": float(np.mean(live)) if live else None,
                   "live_evals_min": int(min(live)) if live else None,
                   "live_evals_max": int(max(live)) if live else None,
                   "live_evals_total": int(sum(live)), "physical_runs": phys, "memo_hits": memo, "wall_s": wall,
                   "rerun_units": rerun, "search_utility_mean": float(np.mean(su)) if su else None}
    return vals, {"pipelines": info, "missing_units": len(missing)}


def reliability(done: Dict[tuple, dict], cids: Sequence[str], seeds: Sequence[int]) -> dict:
    """PREREG §2.3: across-half correlation of paired (H317_LOCKED - control) AUDIT differences (needs the GP record
    field `audit_half_utilities` = [u_half0, u_half1])."""
    out = {}
    for ctrl in CONTROLS:
        a, b = [], []
        for c in cids:
            for s in seeds:
                x, y = done.get((c, s, PRIMARY)), done.get((c, s, ctrl))
                if x and y and x.get("audit_half_utilities") and y.get("audit_half_utilities"):
                    a.append(x["audit_half_utilities"][0] - y["audit_half_utilities"][0])
                    b.append(x["audit_half_utilities"][1] - y["audit_half_utilities"][1])
        out[ctrl] = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0 else None
    out["available"] = any(v is not None for k, v in out.items() if k in CONTROLS)
    return out


def compute_stats(done: Dict[tuple, dict], native: dict, fam_of: Dict[str, str], cids: Sequence[str],
                  seeds: Sequence[int] = C.SEARCH_SEEDS, dedupe_summary: Optional[dict] = None,
                  extra: Optional[dict] = None, partial: bool = False) -> dict:
    if partial:  # monitoring only (never the verdict): contexts with every unit recorded
        cids = [c for c in cids if all((c, s, p) in done for s in seeds for p in PIPELINES)]
        if len({fam_of[c] for c in cids}) < 2:
            raise StageVError("partial stats need complete contexts from >= 2 families")
    vals, info = unit_panel(done, cids, seeds, partial=partial)
    primary = {c: contract(vals[PRIMARY], vals[c], fam_of) for c in CONTROLS}
    invalid = {p: info["pipelines"][p]["invalid"] for p in PIPELINES}
    port = C.family_equal_mean(vals[PRIMARY], fam_of) - C.family_equal_mean(vals[PORTFOLIO_REF], fam_of)
    rd = readout(primary, invalid, port)
    descr = {p: _slim(contract(vals[PRIMARY], vals[p], fam_of)) for p in PIPELINES if p not in (PRIMARY,) + CONTROLS}
    vs_evoa = {p: _slim(contract(vals[p], vals["EVOA_MATCHED"], fam_of)) for p in PIPELINES if p != "EVOA_MATCHED"}
    live_gap = []
    for c in cids:
        for s in seeds:
            a, b = done.get((c, s, PRIMARY)), done.get((c, s, "EVOA_MATCHED"))
            if _completed(a) and _completed(b):
                live_gap.append(abs(a["live_evals"] - b["live_evals"]))
    m = {
        "schema": SCHEMA_METRICS, "wave": C.WAVE, "final": not partial,
        "verdict": rd["verdict"], "readout": rd,
        "primary": {c: _slim(primary[c]) for c in CONTROLS},
        "family_means": {p: C.family_equal_mean(vals[p], fam_of) for p in PIPELINES},
        "family_medians_of_gain_vs_primary": {p: descr[p]["gain_family_median"] for p in descr},
        "h317_locked_vs_other": descr, "vs_evoa_matched": vs_evoa,
        "invalid": invalid, "eval_counts": info["pipelines"], "missing_units": info["missing_units"],
        "live_match": {"max_abs_gap_locked_vs_evoa": int(max(live_gap)) if live_gap else None,
                       "tolerance": LIVE_MATCH_TOL,
                       "violations": [{"cid12": c[:12], "seed": s, "pipeline": p, "what": r["live_match_violation"]}
                                      for (c, s, p), r in sorted(done.items()) if r.get("live_match_violation")][:10],
                       "n_violations": int(sum(bool(r.get("live_match_violation")) for r in done.values()))},
        "failed_units": {"by_pipeline": {p: info["pipelines"][p]["failed"] for p in PIPELINES},
                         "total": int(sum(info["pipelines"][p]["failed"] for p in PIPELINES)),
                         "examples": [{"cid12": c[:12], "seed": s, "pipeline": p, "error": (r.get("error") or "")[:120]}
                                      for (c, s, p), r in sorted(done.items()) if r.get("status") == "failed"][:5],
                         "substitution": "value 0.0 (SEARCH-mean predictor), invalid (PREREG §2.3)"},
        "reliability": reliability(done, cids, seeds),
        "elo_audit": elo_panel(vals, fam_of, ELO_ANCHOR_AUDIT),
    }
    if native and native.get("vnative"):
        nat = {"R882": {c: native["regression"]["per_context"][c]["u"] for c in cids},
               "H317_NEURAL_TOP1": {c: native["vnative"][c]["u"] for c in cids},
               "TOP4_PICK": {c: native["top4"][c]["u"] for c in cids}}
        ung = {c: native["top4"][c]["u_ungated"] for c in cids}
        t4c = float(np.mean(list(nat["TOP4_PICK"].values())))
        t4f = C.family_equal_mean(nat["TOP4_PICK"], fam_of)
        m["v_native"] = {
            "neural_vs_r882": _slim(contract(nat["H317_NEURAL_TOP1"], nat["R882"], fam_of)),
            "neural_vs_top4": _slim(contract(nat["H317_NEURAL_TOP1"], nat["TOP4_PICK"], fam_of)),
            "top4_vs_r882": _slim(contract(nat["TOP4_PICK"], nat["R882"], fam_of)),
            "family_means": {a: C.family_equal_mean(v, fam_of) for a, v in nat.items()},
            "context_means": {a: float(np.mean(list(v.values()))) for a, v in nat.items()},
            "top4": {"family_mean": t4f, "context_mean": t4c, "h258_reference": TOP4_H258_REFERENCE,
                     "h258_reference_kind": TOP4_H258_REFERENCE_KIND, "descriptive": True,
                     "abs_diff_family_vs_h258": abs(t4f - TOP4_H258_REFERENCE),
                     "gated_contexts": int(sum(native["top4"][c]["gated"] for c in cids)),
                     "ungated_variant_context_mean": float(np.mean(list(ung.values())))},
            "neural_seed_agreement": float(np.mean([len({v["top1"] for v in native["vnative"][c]["per_seed"].values()})
                                                    == 1 for c in cids if not native["vnative"][c].get("failed")]
                                                   or [float("nan")])),
            "failed_contexts": int(sum(bool(native["vnative"][c].get("failed")) for c in cids)),
            "elo": elo_panel(nat, fam_of, ELO_ANCHOR_NATIVE),
        }
    if dedupe_summary:
        m["dedupe"] = {md: {k: v for k, v in s.items() if k != "flagged"} | {"flagged": s["flagged"][:10]}
                       for md, s in dedupe_summary.items()}
        m["dedupe"]["interpretive_mode"] = DEDUPE_INTERPRETIVE
        m["dedupe"]["gate"] = False
    if extra:
        m.update(extra)
    m["guard"] = C.guard_report()
    m = _r(m)
    if native and native.get("regression"):  # full float64 precision for the regression numbers
        reg = native["regression"]
        m["regression"] = {k: reg[k] for k in ("status", "bit_exact_matches", "contexts", "mean_target_order",
                                               "abs_diff", "mean_bit_equal", "selection_mismatches")}
    size = len(json_bytes(m, compact=True))
    if size >= METRICS_MAX_BYTES:
        raise StageVError(f"metrics.json would be {size} bytes (>= {METRICS_MAX_BYTES})")
    m["bytes_compact"] = size
    return m


def per_context_panel(done, native, fam_of, cids, seeds) -> dict:
    out = {}
    for c in cids:
        row = {"family": fam_of[c], "audit": {p: [done[(c, s, p)]["value"] for s in seeds if (c, s, p) in done]
                                              for p in PIPELINES}}
        if native and native.get("vnative"):
            row["native"] = {"R882": native["regression"]["per_context"][c]["u"], "neural": native["vnative"][c]["u"],
                             "top4": native["top4"][c]["u"]}
        out[c] = row
    return out


def run_stats(args) -> dict:
    lock, lock_sha, enter = enter_stage_v("stats")
    rd = run_dir()
    if (rd / REG_FAIL).exists():
        native = json.loads((rd / "native.json").read_text())
        m = {"schema": SCHEMA_METRICS, "wave": C.WAVE, "final": True, "verdict": "REGRESSION_FAIL",
             "regression": {k: v for k, v in native["regression"].items() if k != "per_context"},
             "guard": C.guard_report()}
        write_atomic(rd / "metrics.json", json_bytes(m, compact=True))
        return m
    man = json.loads((rd / "targets_manifest.json").read_text())["targets"]
    cids = sorted(man)
    fam_of = {c: man[c]["family_id"] for c in cids}
    if len(cids) != N_TARGETS or len(set(fam_of.values())) != N_TARGET_FAMILIES:
        raise StageVError("targets are not 91 contexts / 19 families")
    native = json.loads((rd / "native.json").read_text())
    done, started, torn = all_unit_records()
    events = [e for f in sorted(rd.glob("events*.jsonl")) for e in read_jsonl(f)[0]]  # incl. events.<host>.jsonl
    extra = {"lock_sha256": lock_sha, "seeds": list(C.SEARCH_SEEDS),
             "disclosure": {"relaunch_before_outcome_used": (rd / RELAUNCH).exists(),
                            "prep_resumes": sum(e.get("event") == "prep_start" and e.get("mode") == "resume"
                                                for e in events),
                            "audit_resumes": sum(e.get("event") == "audit_start" and e.get("mode") == "resume"
                                                 for e in events),
                            "rerun_units_after_interruption": sum(bool(r.get("rerun_after_interruption"))
                                                                  for r in done.values()),
                            "started_unfinished_units": len(started), "torn_lines": torn,
                            "failed_units": int(sum(r.get("status") == "failed" for r in done.values())),
                            "live_match_violations": int(sum(bool(r.get("live_match_violation"))
                                                             for r in done.values())),
                            "v_native_failed_contexts": int(sum(bool(v.get("failed")) for v in
                                                                (native.get("vnative") or {}).values())),
                            "audit_running_takeovers": sum(e.get("event") == "audit_running_stale_taken_over"
                                                           for e in events),
                            "enter_api": enter.get("api")},
             "wall_s": {"prep": json.loads((rd / PREP_DONE).read_text()).get("wall_s"),
                        "audit_shards": {p.name: json.loads(p.read_text()).get("wall_s")
                                         for p in sorted(rd.glob("AUDIT_DONE.*.json"))}}}
    dd = json.loads((rd / "dedupe.json").read_text())["summary"]
    m = compute_stats(done, native, fam_of, cids, dedupe_summary=dd, extra=extra, partial=args.partial)
    name = "metrics_partial.json" if args.partial else "metrics.json"
    write_atomic(rd / name, json_bytes(m, compact=True))
    if not args.partial:
        write_atomic(rd / "panel_stage_v.json", json_bytes(per_context_panel(done, native, fam_of, cids,
                                                                             C.SEARCH_SEEDS), compact=True))
    event("stats_done", verdict=m["verdict"], final=not args.partial)
    print(json.dumps({"verdict": m["verdict"], "bytes": m.get("bytes_compact"), "out": str(rd / name)}))
    return m


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-lock")
    for k in ("h01", "h02", "h03"):
        w.add_argument(f"--{k}", required=True, help=f"rapid_hypothesis_lab.py decide --out JSON of {k}")
    w.add_argument("--gp-decisions", required=True, help="gp_lab decisions.json (pinned H0.4/H0.5/H0.6)")
    w.add_argument("--lock-model-dir", required=True, help="rapid_hypothesis_lab.py lock-fit --out-dir")
    w.add_argument("--worker-bin", action="append", default=[], help="worker binaries (default: pinned s1main)")
    w.add_argument("--extra-input", action="append", default=[], help="other decision inputs to record by sha")
    w.add_argument("--out", default="", help="default $H317_DATA/stage_v/LOCK.json")
    sub.add_parser("unlock")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=("prep", "audit", "stats", "all"), default="all")
    r.add_argument("--resume", action="store_true", help="complete never-recorded units after an interruption")
    r.add_argument("--relaunch-before-outcome", action="store_true",
                   help="the single relaunch of a prep that failed before FIRST_OUTCOME_READ")
    r.add_argument("--reference", default=str(STAGEV_ROOT / "inputs" / "best_dev_checkpoint.json"))
    r.add_argument("--costs", default=str(STAGEV_ROOT / "inputs" / "r882-secondary-costs.npz"))
    r.add_argument("--threads", type=int, default=2, help="encoder / torch threads in prep")
    r.add_argument("--cpus", default="", help="audit worker cores, e.g. 16-63,80-119")
    r.add_argument("--jobs", type=int, default=14, help="concurrent context jobs (cores split evenly)")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--partial", action="store_true", help="stats on incomplete records (metrics_partial.json)")
    args = ap.parse_args(argv)
    if args.cmd == "write-lock":
        cmd_write_lock(args)
        return 0
    if args.cmd == "unlock":
        return cmd_unlock(args)
    try:
        if args.phase in ("prep", "all"):
            try:
                run_prep(args)
            except RegressionFail:
                if args.phase == "all":
                    run_stats(args)  # records verdict REGRESSION_FAIL
                raise
        if args.phase in ("audit", "all"):
            if not args.cpus:
                raise SystemExit("--cpus is required for the audit phase")
            run_audit(args)
        if args.phase in ("stats", "all"):
            run_stats(args)
    except RegressionFail as e:
        print(f"REGRESSION_FAIL: {e}", file=sys.stderr)
        return 4
    except RetryRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
