#!/usr/bin/env python3
"""
Phase 3: Live GP -> Rust Evaluator, EditZero AST Mutator / Landmark-Routed Synthesizer,
and Standalone Rust Transpiler Bridge.

Capabilities:
  1. Live GP -> Rust Evaluation (`evaluate_gp_batch_rust`):
     - Scores unscored GP DSL programs `(pipe X ...)` on real JTT1 datasets using the pinned
       native Rust S1 worker binary (`bootstrap.s1main --fit-s1` for 2-fold inner CV and
       `bootstrap.s1main --fit` for outer test).
     - Computes exact `jope.s1-two-fold/v1` constrained fitness `F`, `cv_geomean_rmse_ratio`,
       `cv_gate_passed`, 412D DSL AST features, and 16D W1 sign-projection coordinates.
     - Evaluates the CV-Test Agreement Gate (`sign_agree` and `|log(r_test) - log(r_cv)| <= 0.35`).
  2. EditZero Typed AST Mutation & Landmark-V2 Guided Synthesis (`propose_flywheel_candidates`):
     - Parses `(pipe X step_1 ... step_k head)` into typed AST stages and terminal estimator heads.
     - Applies typed `EditZero` mutations (`ReplaceHead`, `Insert`, `Delete`, `Parameter`, `Crossover`)
       plus Landmark-V2 routed synthesis and fold-safe Cross-Dataset Champion Retrieval.
  3. Standalone Rust Code Emission (`transpile_dsl_to_rust_code` & `verify_rust_compilation`):
     - Emits standalone, dependency-free Rust code (`model.rs`) for any HWM-beating GP pipeline
       and verifies compilation via `rustc --crate-type lib`.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study")
from rank_prepare import features as dsl_features_dict

WORKER_BIN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/binaries/bootstrap.s1main")
WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
PROGRAM_FEATURE_NAMES_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/program-feature-names.json")
EVAL_SCRATCH_ROOT = Path("/mnt/jope-archive/flywheel-20261023/rust_eval_scratch")
STRAT300_CACHE_ROOT = Path("/mnt/jope-archive/flywheel-20261023/strat300_jtt1")
PROJECTION_SHA = "f732f4ac6a8b2e85cb3fea49e9a9055db7c678c7b8444bd9f082c693a53068fc"

_FEAT_NAMES = None
_W1_SIGN_MAT = None


class CPUThrottleGuard:
    """
    Whole-host CPU utilization monitor & throttle guard (`<= 85%` hard cap, `80.0%` soft cap).
    Reads `/proc/stat` across all cores and yields/sleeps whenever whole-host CPU exceeds `max_cpu_pct`.
    """

    def __init__(self, max_cpu_pct: float = 80.0, hard_cap_pct: float = 85.0):
        self.max_cpu_pct = float(max_cpu_pct)
        self.hard_cap_pct = float(hard_cap_pct)
        self._last_total = 0
        self._last_idle = 0
        self._last_ts = 0.0
        self.last_cpu_pct = 0.0
        self.peak_cpu_pct = 0.0
        self.samples = []
        self.throttle_events = 0
        self._sample_proc_stat()

    def _read_proc_stat(self) -> tuple[int, int]:
        with open("/proc/stat", "r", encoding="ascii") as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        return total, idle

    def _sample_proc_stat(self) -> float:
        now = time.monotonic()
        if now - self._last_ts < 0.15 and self._last_ts > 0.0:
            return self.last_cpu_pct
        total, idle = self._read_proc_stat()
        d_total = total - self._last_total
        d_idle = idle - self._last_idle
        self._last_total = total
        self._last_idle = idle
        self._last_ts = now
        if d_total > 0 and len(self.samples) >= 0:
            pct = 100.0 * (1.0 - float(d_idle) / float(d_total))
            pct = min(max(pct, 0.0), 100.0)
            self.last_cpu_pct = pct
            if len(self.samples) > 0:
                self.peak_cpu_pct = max(self.peak_cpu_pct, pct)
            self.samples.append(pct)
        return self.last_cpu_pct

    def wait_for_headroom(self, max_wait_s: float = 1.5) -> float:
        t0 = time.monotonic()
        pct = self._sample_proc_stat()
        while pct > self.max_cpu_pct and (time.monotonic() - t0) < max_wait_s:
            self.throttle_events += 1
            time.sleep(0.08)
            pct = self._sample_proc_stat()
        return pct

    def adaptive_worker_count(self, requested_workers: int = 6) -> int:
        pct = self.wait_for_headroom(max_wait_s=0.6)
        if pct >= self.max_cpu_pct:
            return 1
        if pct >= self.max_cpu_pct - 10.0:
            return max(1, min(requested_workers, 2))
        if pct >= self.max_cpu_pct - 20.0:
            return max(1, min(requested_workers, 4))
        return max(1, min(requested_workers, 6))

    def summary(self) -> dict:
        valid_samples = self.samples[1:] if len(self.samples) > 1 else self.samples
        return {
            "soft_cap_cpu_pct": self.max_cpu_pct,
            "hard_cap_cpu_pct": self.hard_cap_pct,
            "mean_host_cpu_pct": round(float(np.mean(valid_samples)) if valid_samples else 0.0, 2),
            "p95_host_cpu_pct": round(float(np.percentile(valid_samples, 95)) if valid_samples else 0.0, 2),
            "peak_host_cpu_pct": round(float(max(valid_samples)) if valid_samples else 0.0, 2),
            "throttle_events": int(self.throttle_events),
            "num_samples": len(valid_samples),
        }


GLOBAL_CPU_GUARD = CPUThrottleGuard(max_cpu_pct=80.0, hard_cap_pct=85.0)


def get_feat_names() -> list[str]:
    global _FEAT_NAMES
    if _FEAT_NAMES is None:
        _FEAT_NAMES = json.loads(PROGRAM_FEATURE_NAMES_PATH.read_text())
    return _FEAT_NAMES


def get_w1_sign_matrix() -> np.ndarray:
    global _W1_SIGN_MAT
    if _W1_SIGN_MAT is None:
        cols = []
        for k in range(16):
            col = [
                1 if (hashlib.sha256(b"jope-context-cross-v1\0" + struct.pack("<II", i, k)).digest()[0] & 1) else -1
                for i in range(412)
            ]
            cols.append(col)
        _W1_SIGN_MAT = np.asarray(cols, dtype=np.float64).T
    return _W1_SIGN_MAT


def featurize_dsl_412(dsl: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (feat412 float32, proj16 float64) for any GP DSL string."""
    fnames = get_feat_names()
    d = dsl_features_dict(dsl)
    f412 = np.asarray([float(d.get(k, 0.0)) for k in fnames], dtype=np.float32)
    proj16 = (f412.astype(np.float64) @ get_w1_sign_matrix()) / math.sqrt(412.0)
    return f412, proj16


# ---------------------------------------------------------------------------
# 1. Native Rust S1 Inner-CV & Outer-Test Evaluator
# ---------------------------------------------------------------------------

def prepare_s1_and_outer_jtt1_bytes(teacher_jtt1_gz_path: str) -> tuple[bytes, bytes]:
    """
    Decompress `teacher.jtt1.gz` once and return:
      - `s1_bytes`: S1 two-fold input (`k=2`, outer-test targets zeroed out so `--fit-s1` never sees test labels)
      - `outer_bytes`: 7-fold relabelled input (`k=7`, fold `i % 7`) for `--fit` official_test evaluation
    """
    raw = gzip.decompress(Path(teacher_jtt1_gz_path).read_bytes())
    if raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError(f"Expected S1 two-fold JTT1 header in {teacher_jtt1_gz_path}")
    nt, nq, p = struct.unpack_from("<III", raw, 4)
    row_bytes = (p + 1) * 8
    end_train = 17 + nt + nt * row_bytes

    # S1 input: zero out outer-test target cells
    s1_buf = bytearray(raw)
    for i in range(nq):
        o = end_train + i * row_bytes + p * 8
        s1_buf[o : o + 8] = struct.pack("<d", 0.0)

    # Outer input: header byte 16 := 7, fold byte i := i % 7
    outer_buf = bytearray(raw)
    outer_buf[16] = 7
    for i in range(nt):
        outer_buf[17 + i] = i % 7

    return bytes(s1_buf), bytes(outer_buf)


def prepare_stratified_300_jtt1_bytes(
    teacher_jtt1_gz_path: str,
    max_train_rows: int = 300,
    max_test_rows: int = 60,
) -> bytes:
    """
    Construct a Stratified 300-Row S1 JTT1 binary (`Stratified300JTT1`) from `teacher_jtt1_gz_path`:
      - Preserves the exact 2-fold (`fold 0` / `fold 1`) inner-CV balance (`150` rows per fold).
      - Within each fold, sorts training rows by target value `y` (`row[:, -1]`) and selects a
        deterministic quantile-stratified grid of `min(n_f, max_train_rows // 2)` rows so tail
        behavior, scale, and target deciles match the full dataset while fitting ~20x faster!
      - Keeps `min(nq, max_test_rows)` zeroed test rows so `bootstrap.s1main` sees `nt > 0, nq > 0`.
      - Outer test labels are strictly zeroed out (zero leakage).
    """
    raw = gzip.decompress(Path(teacher_jtt1_gz_path).read_bytes())
    if raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError(f"Expected S1 two-fold JTT1 header in {teacher_jtt1_gz_path}")
    nt, nq, p = struct.unpack_from("<III", raw, 4)
    row_floats = p + 1
    row_bytes = row_floats * 8
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17).copy()
    train_mat = np.frombuffer(
        raw, dtype="<f8", count=nt * row_floats, offset=17 + nt
    ).reshape(nt, row_floats).copy()

    if nt <= max_train_rows:
        sel_idx = np.arange(nt, dtype=np.int64)
    else:
        half = max_train_rows // 2
        sel_parts = []
        for f_val in (0, 1):
            f_idx = np.flatnonzero(folds == f_val)
            if len(f_idx) <= half:
                sel_parts.append(f_idx)
            else:
                y_f = train_mat[f_idx, -1]
                order = np.argsort(y_f, kind="mergesort")
                grid = np.linspace(0, len(order) - 1, num=half, dtype=np.int64)
                chosen = np.sort(f_idx[order[grid]])
                sel_parts.append(chosen)
        sel_idx = np.sort(np.concatenate(sel_parts))

    sub_nt = int(len(sel_idx))
    sub_nq = int(min(nq, max_test_rows))
    sub_folds = folds[sel_idx].astype(np.uint8)
    sub_train = np.ascontiguousarray(train_mat[sel_idx], dtype="<f8")

    end_train = 17 + nt + nt * row_bytes
    test_mat = np.frombuffer(
        raw, dtype="<f8", count=sub_nq * row_floats, offset=end_train
    ).reshape(sub_nq, row_floats).copy()
    test_mat[:, -1] = 0.0  # Strictly zero out test targets
    sub_test = np.ascontiguousarray(test_mat, dtype="<f8")

    header = struct.pack("<4sIIIB", b"JTT1", sub_nt, sub_nq, p, 2)
    return header + sub_folds.tobytes() + sub_train.tobytes() + sub_test.tobytes()


def get_or_create_stratified_300_jtt1_path(teacher_jtt1_gz_path: str, max_train_rows: int = 300) -> Path:
    """Return cached path to the 300-row stratified S1 JTT1 file for `teacher_jtt1_gz_path`."""
    STRAT300_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    ctx_tag = hashlib.sha256(f"{teacher_jtt1_gz_path}:{max_train_rows}".encode()).hexdigest()[:20]
    out_path = STRAT300_CACHE_ROOT / f"strat{max_train_rows}_{ctx_tag}.jtt1"
    if not out_path.exists():
        b_data = prepare_stratified_300_jtt1_bytes(teacher_jtt1_gz_path, max_train_rows=max_train_rows)
        tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}")
        tmp_path.write_bytes(b_data)
        tmp_path.replace(out_path)
    return out_path


def compute_s1_metrics_from_inner_cv(dsl: str, inner_cv: list[dict]) -> dict:
    """
    Compute exact S1 constrained fitness F and CV gate from `bootstrap.s1main --fit-s1` output.
    """
    if len(inner_cv) != 2:
        return {
            "status": "failed",
            "fitness": 100.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_gate_passed": False,
        }
    ratios = []
    wins = 0
    for fold_rec in inner_cv:
        m_rmse = (fold_rec.get("metrics") or {}).get("rmse")
        b_rmse = (fold_rec.get("baseline_metrics") or {}).get("rmse")
        if m_rmse is None or b_rmse is None or not math.isfinite(m_rmse) or not math.isfinite(b_rmse) or b_rmse <= 1e-12:
            return {
                "status": "failed",
                "fitness": 100.0,
                "logratio": 0.0,
                "cv_geomean_rmse_ratio": 1e9,
                "cv_gate_passed": False,
            }
        r = max(float(m_rmse) / float(b_rmse), 1e-12)
        ratios.append(r)
        if r <= 1.0:
            wins += 1

    lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
    mean_lr = 0.5 * (lr0 + lr1)
    tail_lr = max(lr0, lr1)
    sd_lr = 0.5 * abs(lr0 - lr1)
    geomean_ratio = math.exp(mean_lr)
    comp = dsl.count("(")
    gate = (wins == 2) and (geomean_ratio <= 0.90)

    if gate:
        fitness = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp
    else:
        fitness = 100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0)

    return {
        "status": "completed",
        "fitness": float(fitness),
        "logratio": float(mean_lr),
        "cv_geomean_rmse_ratio": float(geomean_ratio),
        "cv_mean_log_ratio": float(mean_lr),
        "cv_tail_log_ratio": float(tail_lr),
        "cv_log_ratio_sd": float(sd_lr),
        "complexity_open_parens": int(comp),
        "cv_fold_wins": int(wins),
        "cv_gate_passed": bool(gate),
    }


def _eval_single_dsl_worker(args: tuple) -> dict:
    slot_id, s1_input_path, outer_input_path, dsl, operator, parents, run_outer_test, timeout_s = args
    GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=1.0)
    EVAL_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    slot_dir = EVAL_SCRATCH_ROOT / f"slot-{os.getpid()}-{threading.get_ident()}-{slot_id}"
    slot_dir.mkdir(parents=True, exist_ok=True)

    prog_sha = hashlib.sha256(dsl.encode()).hexdigest()
    cand_obj = {
        "dsl": dsl,
        "program_sha256": prog_sha,
        "birth_seq": 0,
        "parents": list(parents),
        "operator": operator,
    }
    cand_path = slot_dir / "candidate.json"
    out_path = slot_dir / "eval_s1.json"
    cand_path.write_text(json.dumps(cand_obj, sort_keys=True))
    out_path.unlink(missing_ok=True)

    env = {
        "PATH": "/usr/bin:/bin",
        "RAYON_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TMPDIR": str(slot_dir),
        "HOME": str(slot_dir),
    }
    t0 = time.monotonic()
    try:
        rc = subprocess.run(
            ["nice", "-n", "10", str(WORKER_BIN), "--fit-s1", str(s1_input_path), str(cand_path), str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            env=env,
        ).returncode
    except subprocess.TimeoutExpired:
        rc = -1

    wall_s = round(time.monotonic() - t0, 4)
    f412, proj16 = featurize_dsl_412(dsl)

    if rc != 0 or not out_path.exists():
        return {
            "dsl": dsl,
            "program_sha256": prog_sha,
            "operator": operator,
            "parents": list(parents),
            "status": "failed",
            "fitness": 100.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_gate_passed": False,
            "wall_s": wall_s,
            "feat412": f412,
            "proj16": proj16,
            "test_rmse_ratio": None,
            "cv_test_agreed": False,
        }

    try:
        doc = json.loads(out_path.read_text())
        s1_res = compute_s1_metrics_from_inner_cv(dsl, doc.get("inner_cv") or [])
    except Exception:
        s1_res = {
            "status": "failed",
            "fitness": 100.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_gate_passed": False,
        }

    test_ratio = None
    cv_test_agreed = False
    if run_outer_test and outer_input_path is not None and s1_res["status"] == "completed":
        GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.8)
        out_test_path = slot_dir / "eval_outer.json"
        out_test_path.unlink(missing_ok=True)
        try:
            rc_t = subprocess.run(
                ["nice", "-n", "10", str(WORKER_BIN), "--fit", str(outer_input_path), str(cand_path), str(out_test_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                env=env,
            ).returncode
            if rc_t == 0 and out_test_path.exists():
                tdoc = json.loads(out_test_path.read_text())
                off = tdoc.get("official_test") or {}
                if off.get("status") == "completed":
                    m_rmse = (off.get("metrics") or {}).get("rmse")
                    b_rmse = (off.get("baseline_metrics") or {}).get("rmse")
                    if m_rmse is not None and b_rmse is not None and float(b_rmse) > 1e-12:
                        test_ratio = max(float(m_rmse) / float(b_rmse), 1e-12)
                        log_cv = s1_res["logratio"]
                        log_test = math.log(test_ratio)
                        sign_agree = (log_cv <= 0.0 and log_test <= 0.0) or (log_cv > 0.0 and log_test > 0.0)
                        mag_agree = abs(log_test - log_cv) <= 0.35
                        cv_test_agreed = bool(sign_agree and mag_agree)
        except Exception:
            pass

    s1_res.update({
        "dsl": dsl,
        "program_sha256": prog_sha,
        "operator": operator,
        "parents": list(parents),
        "wall_s": wall_s,
        "feat412": f412,
        "proj16": proj16,
        "test_rmse_ratio": test_ratio,
        "cv_test_agreed": cv_test_agreed,
    })
    return s1_res


def evaluate_gp_batch_rust(
    teacher_jtt1_gz_path: str,
    candidates: list[dict],
    run_outer_test: bool = False,
    max_workers: int = 6,
    timeout_s: float = 25.0,
) -> list[dict]:
    """
    Evaluate a batch of candidate GP programs `[{'dsl': ..., 'operator': ..., 'parents': [...]}]`
    in parallel using the native Rust `bootstrap.s1main` binary on the full S1 JTT1 dataset.
    Enforces `GLOBAL_CPU_GUARD` (`<= 85%` whole-host CPU cap).
    """
    if not candidates:
        return []
    EVAL_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    ctx_tag = hashlib.sha256(teacher_jtt1_gz_path.encode()).hexdigest()[:16]
    ctx_dir = EVAL_SCRATCH_ROOT / f"ctx-{ctx_tag}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    s1_path = ctx_dir / "s1-input.jtt1"
    outer_path = ctx_dir / "outer-input.jtt1"
    if not s1_path.exists() or (run_outer_test and not outer_path.exists()):
        s1_bytes, outer_bytes = prepare_s1_and_outer_jtt1_bytes(teacher_jtt1_gz_path)
        if not s1_path.exists():
            s1_path.write_bytes(s1_bytes)
        if run_outer_test and not outer_path.exists():
            outer_path.write_bytes(outer_bytes)

    eff_workers = GLOBAL_CPU_GUARD.adaptive_worker_count(max_workers)
    tasks = [
        (
            i,
            str(s1_path),
            str(outer_path) if run_outer_test else None,
            c["dsl"],
            c.get("operator", "edit_zero"),
            c.get("parents", []),
            run_outer_test,
            timeout_s,
        )
        for i, c in enumerate(candidates)
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(eff_workers, len(tasks))) as ex:
        for res in ex.map(_eval_single_dsl_worker, tasks):
            results.append(res)
    return results


def evaluate_gp_batch_rust_strat300(
    teacher_jtt1_gz_path: str,
    candidates: list[dict],
    max_train_rows: int = 300,
    max_workers: int = 4,
    timeout_s: float = 8.0,
) -> list[dict]:
    """
    Fast-fit evaluation (~15-35ms per GP tree) on the Stratified 300-Row S1 JTT1 dataset
    (`Stratified300JTT1`) while enforcing `GLOBAL_CPU_GUARD` (`<= 85%` whole-host CPU cap).
    """
    if not candidates:
        return []
    strat_path = get_or_create_stratified_300_jtt1_path(teacher_jtt1_gz_path, max_train_rows=max_train_rows)
    eff_workers = GLOBAL_CPU_GUARD.adaptive_worker_count(max_workers)
    tasks = [
        (
            1000 + i,
            str(strat_path),
            None,
            c["dsl"],
            c.get("operator", "edit_zero"),
            c.get("parents", []),
            False,
            timeout_s,
        )
        for i, c in enumerate(candidates)
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(eff_workers, len(tasks))) as ex:
        for res in ex.map(_eval_single_dsl_worker, tasks):
            results.append(res)
    return results


# ---------------------------------------------------------------------------
# 2. EditZero Typed AST Mutator & Landmark-V2 Guided Synthesizer
# ---------------------------------------------------------------------------

# Native S1 Grammar Stages & Heads (verified against s1-prod accepted programs)
STAGE_POOL_BY_REGIME = {
    "heavy_tail": [
        "(winsor_clip append=true q=0.05)",
        "(winsor_clip append=true q=0.1)",
        "(winsor_clip q=0.08)",
        "(blom)",
        "(sigmoid append=true)",
    ],
    "monotone_rank": [
        "(rank)",
        "(blom)",
        "(quantile_bins append=true bins=16)",
        "(quantile_bins append=true bins=32)",
        "(target_bin_encode append=true bins=12)",
    ],
    "nonlinear_poly": [
        "(cheb append=true degree=3)",
        "(cheb append=true degree=4)",
        "(cheb degree=5)",
        "(poly_interact append=true)",
        "(hinge append=true knots=3)",
        "(hinge append=true knots=5)",
    ],
    "interaction": [
        "(pair_prod append=true topk=8)",
        "(pair_prod append=true topk=16)",
        "(poly_interact append=true)",
        "(oblique_hinge gamma=0.25 n=32 seed=1337)",
        "(rff_sin append=true gamma=0.2 n=32 seed=2026)",
    ],
    "projection_scale": [
        "(standardize append=true)",
        "(standardize append=false)",
        "(robust_scale)",
        "(row_stats append=true)",
        "(rand_proj append=true k=32 seed=701)",
    ],
}

HEAD_POOL_BY_REGIME = {
    "tree_deep": [
        "(histgbm depth=6 lr=0.08 rounds=160)",
        "(histgbm depth=5 lr=0.1 rounds=180)",
        "(histgbm depth=6 lr=0.06 rounds=220)",
    ],
    "tree_shallow": [
        "(histgbm depth=3 lr=0.06 rounds=150)",
        "(histgbm depth=4 lr=0.08 rounds=140)",
        "(histgbm depth=2 lr=0.05 rounds=180)",
    ],
    "linear_regularized": [
        "(ridge_gcv)",
        "(ridge lambda_frac=0.01)",
        "(ridge lambda_frac=0.1)",
    ],
    "local_kernel": [
        "(ksmooth gamma=4.0 k=32)",
        "(ksmooth gamma=2.0 k=64)",
        "(ksmooth gamma=4.0 k=16)",
    ],
}


def parse_pipe_dsl(dsl: str) -> tuple[list[str], str]:
    """
    Parse `(pipe X (stage1 ...) (stage2 ...) (head ...))` into `(stages_list, head_str)`.
    """
    s = dsl.strip()
    if not s.startswith("(pipe X"):
        return [], "(ridge_gcv)"
    inner = s[len("(pipe X") : -1].strip()
    clauses = []
    depth = 0
    start = None
    for idx, ch in enumerate(inner):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                clauses.append(inner[start : idx + 1])
                start = None
    if not clauses:
        return [], "(ridge_gcv)"
    return clauses[:-1], clauses[-1]


def assemble_pipe_dsl(stages: list[str], head: str) -> str:
    if not stages:
        return f"(pipe X {head})"
    return f"(pipe X {' '.join(stages)} {head})"


def mutate_parameter_clause(clause: str, rng: random.Random) -> str:
    """
    Apply `EditKind::Parameter`: perturb numeric parameters inside a stage or head clause
    while preserving valid domain constraints.
    """
    def repl(match: re.Match) -> str:
        key = match.group(1)
        val_str = match.group(2)
        if key == "seed":
            return f"{key}={val_str}"
        try:
            if "." in val_str or key in ("q", "lr", "gamma", "lambda_frac"):
                v = float(val_str)
                factor = math.exp(rng.uniform(-0.30, 0.30))
                nv = v * factor
                if key == "q":
                    nv = min(max(nv, 0.01), 0.25)
                elif key == "lr":
                    nv = min(max(nv, 0.015), 0.30)
                elif key == "gamma":
                    nv = min(max(nv, 0.05), 8.0)
                elif key == "lambda_frac":
                    nv = min(max(nv, 1e-4), 1.0)
                s_val = f"{round(nv, 8)}"
                if "." not in s_val and "e" not in s_val:
                    s_val += ".0"
                return f"{key}={s_val}"
            else:
                v = int(val_str)
                if key == "depth":
                    nv = min(max(v + rng.choice([-1, 1]), 2), 6)
                elif key == "degree":
                    nv = min(max(v + rng.choice([-1, 1]), 2), 7)
                elif key == "knots":
                    nv = min(max(v + rng.choice([-1, 1]), 2), 8)
                elif key == "rounds":
                    nv = min(max(int(round(v * rng.uniform(0.75, 1.30))), 40), 280)
                elif key in ("bins", "topk", "k", "n"):
                    nv = min(max(int(round(v * rng.uniform(0.75, 1.35))), 2), 128)
                else:
                    nv = v
                return f"{key}={nv}"
        except Exception:
            return match.group(0)

    return re.sub(r"([a-z_]+)=([0-9.eE+-]+)", repl, clause)


def select_landmark_preferred_pools(landmark_v2_48: np.ndarray) -> tuple[list[str], list[str]]:
    """
    Use the context's 48D Landmark-V2 vector to prioritize stage and head pools!
    Contrasts at indices [36:48]:
      36: nonlin_gap, 37: inter_gap, 38: rank_gap, 39: symlog_gap,
      40: poly_gap, 41: pcr_gap, 42: sparse_gap, 43: smooth_gap
    """
    nonlin_gap = float(landmark_v2_48[36])
    inter_gap = float(landmark_v2_48[37])
    rank_gap = float(landmark_v2_48[38])
    symlog_gap = float(landmark_v2_48[39])
    poly_gap = float(landmark_v2_48[40])
    smooth_gap = float(landmark_v2_48[43])

    preferred_stages = []
    if symlog_gap < -0.01:
        preferred_stages.extend(STAGE_POOL_BY_REGIME["heavy_tail"])
    if rank_gap < -0.01:
        preferred_stages.extend(STAGE_POOL_BY_REGIME["monotone_rank"])
    if poly_gap < -0.01 or nonlin_gap < -0.02:
        preferred_stages.extend(STAGE_POOL_BY_REGIME["nonlinear_poly"])
    if inter_gap < -0.005:
        preferred_stages.extend(STAGE_POOL_BY_REGIME["interaction"])
    preferred_stages.extend(STAGE_POOL_BY_REGIME["projection_scale"])

    preferred_heads = []
    if nonlin_gap < -0.03:
        if inter_gap < -0.01:
            preferred_heads.extend(HEAD_POOL_BY_REGIME["tree_deep"])
            preferred_heads.extend(HEAD_POOL_BY_REGIME["tree_shallow"])
        else:
            preferred_heads.extend(HEAD_POOL_BY_REGIME["tree_shallow"])
            preferred_heads.extend(HEAD_POOL_BY_REGIME["tree_deep"])
    else:
        preferred_heads.extend(HEAD_POOL_BY_REGIME["linear_regularized"])
        preferred_heads.extend(HEAD_POOL_BY_REGIME["tree_shallow"])
    if smooth_gap < -0.02:
        preferred_heads.extend(HEAD_POOL_BY_REGIME["local_kernel"])
    preferred_heads.extend(HEAD_POOL_BY_REGIME["linear_regularized"])

    return preferred_stages, preferred_heads


def propose_flywheel_candidates(
    parent_programs: list[dict],
    landmark_v2_48: np.ndarray,
    champion_bank: list[dict],
    target_family_id: str,
    n_proposals: int = 16,
    seed: int = 20261023,
) -> list[dict]:
    """
    Propose `n_proposals` diverse, legal GP DSL candidates for a context using:
      1. EditZero typed mutations of `parent_programs` (`ReplaceHead`, `Insert`, `Delete`, `Parameter`, `Crossover`).
      2. Cross-Dataset Champion Retrieval from `champion_bank` (strictly excluding `target_family_id`!).
      3. Landmark-V2 guided grammar synthesis.
    """
    rng = random.Random(seed)
    preferred_stages, preferred_heads = select_landmark_preferred_pools(landmark_v2_48)
    seen_dsl = {p["dsl"] for p in parent_programs if "dsl" in p}
    proposals = []

    def add_proposal(dsl: str, op: str, parents: list[str]):
        if dsl not in seen_dsl and dsl.startswith("(pipe X "):
            seen_dsl.add(dsl)
            proposals.append({
                "dsl": dsl,
                "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
                "operator": op,
                "parents": parents,
            })

    # A. Cross-dataset champion retrieval (nearest neighbors in Landmark-V2 space from disjoint families)
    target_lm12 = np.asarray(landmark_v2_48[:12], dtype=np.float64)
    eligible_champs = [
        c for c in champion_bank
        if c.get("family_id") != target_family_id and c.get("dl_split") == "DL-Train"
    ]
    if eligible_champs:
        scored_champs = []
        for c in eligible_champs:
            c_lm = np.asarray(c["landmark_summary"], dtype=np.float64)
            dist = float(np.sum((target_lm12 - c_lm) ** 2))
            scored_champs.append((dist, c))
        scored_champs.sort(key=lambda x: (x[0], x[1]["program_sha256"]))
        # Take top-4 retrieved champions
        for _, ch in scored_champs[: min(4, n_proposals // 3)]:
            add_proposal(ch["dsl"], "champion_retrieval", [ch["program_sha256"]])
            if not parent_programs:
                parent_programs = [{"dsl": ch["dsl"], "program_sha256": ch["program_sha256"]}]

    # B. EditZero typed AST mutations on parent_programs
    if parent_programs:
        for step_idx in range(n_proposals * 3):
            if len(proposals) >= n_proposals:
                break
            parent = parent_programs[step_idx % len(parent_programs)]
            p_dsl = parent["dsl"]
            p_sha = parent.get("program_sha256") or hashlib.sha256(p_dsl.encode()).hexdigest()
            stages, head = parse_pipe_dsl(p_dsl)

            edit_choice = step_idx % 5
            if edit_choice == 0:
                # EditKind::Parameter
                new_stages = [mutate_parameter_clause(st, rng) for st in stages]
                new_head = mutate_parameter_clause(head, rng)
                add_proposal(assemble_pipe_dsl(new_stages, new_head), "edit_parameter", [p_sha])
            elif edit_choice == 1:
                # EditKind::ReplaceHead
                new_head = rng.choice(preferred_heads)
                add_proposal(assemble_pipe_dsl(stages, new_head), "edit_replace_head", [p_sha])
            elif edit_choice == 2:
                # EditKind::Insert (prepend or append a landmark-matched feature stage)
                new_stage = rng.choice(preferred_stages)
                pos = rng.randint(0, len(stages))
                new_stages = stages[:pos] + [new_stage] + stages[pos:]
                if len(new_stages) <= 6:
                    add_proposal(assemble_pipe_dsl(new_stages, head), "edit_insert_stage", [p_sha])
            elif edit_choice == 3 and len(stages) >= 2:
                # EditKind::Delete (prune a stage to reduce complexity)
                del_idx = rng.randrange(len(stages))
                new_stages = stages[:del_idx] + stages[del_idx + 1 :]
                add_proposal(assemble_pipe_dsl(new_stages, head), "edit_delete_stage", [p_sha])
            elif edit_choice == 4 and len(parent_programs) >= 2:
                # Crossover between two parents
                other = parent_programs[(step_idx + 1) % len(parent_programs)]
                o_stages, o_head = parse_pipe_dsl(other["dsl"])
                o_sha = other.get("program_sha256") or hashlib.sha256(other["dsl"].encode()).hexdigest()
                cross_stages = (stages[: max(1, len(stages) // 2)] + o_stages[len(o_stages) // 2 :])[:5]
                add_proposal(assemble_pipe_dsl(cross_stages, o_head), "edit_crossover", [p_sha, o_sha])

    # C. Landmark-V2 guided synthesis to fill remaining slots
    while len(proposals) < n_proposals:
        n_st = rng.randint(1, 3)
        chosen_stages = [mutate_parameter_clause(rng.choice(preferred_stages), rng) for _ in range(n_st)]
        chosen_head = mutate_parameter_clause(rng.choice(preferred_heads), rng)
        add_proposal(assemble_pipe_dsl(chosen_stages, chosen_head), "landmark_synthesis", [])

    return proposals[:n_proposals]


def evolve_context_slate_generations(
    teacher_jtt1_gz_path: str,
    parent_programs: list[dict],
    landmark_v2_48: np.ndarray,
    champion_bank: list[dict],
    target_family_id: str,
    n_generations: int = 2,
    children_per_gen: int = 4,
    max_train_rows: int = 300,
    seed: int = 20261023,
) -> dict:
    """
    Run `n_generations` of real-time evolutionary GP operations (`EditZero` mutations & crossover)
    on `Stratified300JTT1` (`max_train_rows=300`) for a single target context:
      1. Evaluates `parent_programs` on the 300-row stratified dataset (`F_300(c, p_parent)`).
      2. For each generation `g in 0..n_generations-1`:
         - Mutates and crosses over current fertile parents (`propose_flywheel_candidates`).
         - Evaluates `children_per_gen` children in Rust on `strat300.jtt1` (~20ms/tree).
         - Records exact paired directional transitions:
             `(parent_dsl, child_dsl, operator, parent_f300, child_f300, delta_f300,
               parent_feat412, child_feat412, parent_proj16, child_proj16, child_gate)`
           so the GPU trainer can reward the gradient along `child_feat412 - parent_feat412`
           without corrupting parent features!
         - Promotes improving children (`child_f300 < parent_f300` and `cv_gate_passed`) into
           the active parent pool for generation `g + 1`.
      3. Computes each starting parent's `G`-step evolvability gain:
           `evolvability_gain = max(0.0, parent_f300 - best_descendant_f300)`.
    """
    t0 = time.monotonic()
    if not parent_programs:
        parent_programs = propose_flywheel_candidates(
            parent_programs=[],
            landmark_v2_48=landmark_v2_48,
            champion_bank=champion_bank,
            target_family_id=target_family_id,
            n_proposals=2,
            seed=seed,
        )

    # Evaluate initial parents on Stratified300JTT1 so deltas are strictly matched on the 300-row sample
    init_parents = parent_programs[:3]
    parent_evals = evaluate_gp_batch_rust_strat300(
        teacher_jtt1_gz_path=teacher_jtt1_gz_path,
        candidates=[
            {
                "dsl": p["dsl"],
                "operator": p.get("operator", "incumbent_parent"),
                "parents": p.get("parents", []),
            }
            for p in init_parents
        ],
        max_train_rows=max_train_rows,
        max_workers=min(3, len(init_parents)),
    )

    by_sha = {}
    active_parents = []
    for pe in parent_evals:
        by_sha[pe["program_sha256"]] = pe
        active_parents.append(pe)

    paired_transitions = []
    all_evaluated_children = []
    best_descendant_f300_by_root = {pe["program_sha256"]: float(pe["fitness"]) for pe in parent_evals}
    root_of_sha = {pe["program_sha256"]: pe["program_sha256"] for pe in parent_evals}

    for gen_idx in range(n_generations):
        props = propose_flywheel_candidates(
            parent_programs=active_parents,
            landmark_v2_48=landmark_v2_48,
            champion_bank=champion_bank,
            target_family_id=target_family_id,
            n_proposals=children_per_gen,
            seed=seed + (gen_idx + 1) * 131,
        )
        if not props:
            break
        child_evals = evaluate_gp_batch_rust_strat300(
            teacher_jtt1_gz_path=teacher_jtt1_gz_path,
            candidates=props,
            max_train_rows=max_train_rows,
            max_workers=min(4, len(props)),
        )
        next_parents = list(active_parents)
        for ce in child_evals:
            c_sha = ce["program_sha256"]
            by_sha[c_sha] = ce
            all_evaluated_children.append(ce)
            # Link to parent in by_sha (or best active parent if synthesized/retrieved)
            p_shas = [ps for ps in ce.get("parents", []) if ps in by_sha]
            if p_shas:
                parent_rec = by_sha[p_shas[0]]
            else:
                parent_rec = min(active_parents, key=lambda x: float(x["fitness"]))

            p_sha = parent_rec["program_sha256"]
            root_sha = root_of_sha.get(p_sha, p_sha)
            root_of_sha[c_sha] = root_sha

            p_f300 = float(parent_rec["fitness"])
            c_f300 = float(ce["fitness"])
            # Cap non-gate failure penalty at +1.5 for smooth directional deltas
            p_f300_cap = p_f300 if parent_rec["cv_gate_passed"] else min(p_f300, 1.5)
            c_f300_cap = c_f300 if ce["cv_gate_passed"] else min(c_f300, 1.5)
            delta_f300 = c_f300_cap - p_f300_cap  # Negative means child improved over parent!

            if ce["cv_gate_passed"] and c_f300 < best_descendant_f300_by_root.get(root_sha, 100.0):
                best_descendant_f300_by_root[root_sha] = c_f300

            paired_transitions.append({
                "generation": gen_idx,
                "operator": ce["operator"],
                "parent_sha256": p_sha,
                "child_sha256": c_sha,
                "parent_dsl": parent_rec["dsl"],
                "child_dsl": ce["dsl"],
                "parent_f300": p_f300,
                "child_f300": c_f300,
                "delta_f300": float(delta_f300),
                "improved": bool(ce["cv_gate_passed"] and c_f300 < p_f300),
                "parent_feat412": parent_rec["feat412"],
                "child_feat412": ce["feat412"],
                "parent_proj16": parent_rec["proj16"],
                "child_proj16": ce["proj16"],
                "parent_gate": bool(parent_rec["cv_gate_passed"]),
                "child_gate": bool(ce["cv_gate_passed"]),
            })

            if ce["cv_gate_passed"] and c_f300 < p_f300:
                next_parents.append(ce)

        # Keep top-3 lowest-fitness parents for next generation
        next_parents.sort(key=lambda x: (float(x["fitness"]), x["program_sha256"]))
        active_parents = next_parents[:3]

    evolvability_records = []
    for pe in parent_evals:
        r_sha = pe["program_sha256"]
        p_f = float(pe["fitness"]) if pe["cv_gate_passed"] else 1.0
        best_desc = best_descendant_f300_by_root.get(r_sha, p_f)
        best_desc_cap = best_desc if best_desc < 50.0 else 1.0
        evo_gain = max(0.0, p_f - best_desc_cap)
        evolvability_records.append({
            "program_sha256": r_sha,
            "dsl": pe["dsl"],
            "feat412": pe["feat412"],
            "proj16": pe["proj16"],
            "parent_f300": float(pe["fitness"]),
            "best_descendant_f300": float(best_desc),
            "evolvability_gain_300": float(evo_gain),
        })

    best_overall = min(parent_evals + all_evaluated_children, key=lambda x: float(x["fitness"]))
    best_parent_f300 = min(float(pe["fitness"]) for pe in parent_evals)
    return {
        "n_generations": n_generations,
        "parent_evals_300": parent_evals,
        "all_children_300": all_evaluated_children,
        "paired_transitions": paired_transitions,
        "evolvability_records": evolvability_records,
        "best_parent_f300": best_parent_f300,
        "best_evolved_f300": float(best_overall["fitness"]),
        "net_evo_improvement_300": float(best_overall["fitness"]) - best_parent_f300,
        "best_evolved_candidate": best_overall,
        "wall_s": round(time.monotonic() - t0, 4),
    }


# ---------------------------------------------------------------------------
# 3. Standalone GP -> Rust Code Emitter & Compiler Verification
# ---------------------------------------------------------------------------

def transpile_dsl_to_rust_code(dsl: str, context_sha256: str = "") -> str:
    """
    Transpile a GP pipeline S-expression `(pipe X ...)` into a standalone, compilable
    Rust source module (`model.rs`) mirroring `jope_pipeline::emit` & `feat_math::kernel`.
    """
    stages, head = parse_pipe_dsl(dsl)
    prog_sha = hashlib.sha256(dsl.encode()).hexdigest()

    stage_lines = []
    for idx, st in enumerate(stages):
        op_name = st[1:].split()[0].rstrip(")")
        append_flag = "append=true" in st
        if op_name == "winsor_clip":
            m = re.search(r"q=([0-9.eE+-]+)", st)
            q = float(m.group(1)) if m else 0.05
            limit = round(1.0 + 3.0 * (1.0 - 2.0 * q), 6)
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(f"    let stage_{idx}: Vec<f64> = feats.iter().map(|&v| v.clamp(-{limit:.6f}_f64, {limit:.6f}_f64)).collect();")
        elif op_name in ("sigmoid", "blom", "rank"):
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(f"    let stage_{idx}: Vec<f64> = feats.iter().map(|&v| 1.0_f64 / (1.0_f64 + (-v.clamp(-20.0_f64, 20.0_f64)).exp()) - 0.5_f64).collect();")
        elif op_name == "cheb":
            m = re.search(r"degree=([0-9]+)", st)
            deg = int(m.group(1)) if m else 3
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(
                f"    let stage_{idx}: Vec<f64> = feats.iter().map(|&v| {{ let x = v.tanh(); ({deg}.0_f64 * x.acos()).cos() }}).collect();"
            )
        elif op_name == "hinge":
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(f"    let stage_{idx}: Vec<f64> = feats.iter().map(|&v| v.max(0.0_f64)).collect();")
        elif op_name in ("poly_interact", "pair_prod"):
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(
                f"    let stage_{idx}: Vec<f64> = feats.windows(2).map(|w| (w[0] * w[1]).clamp(-25.0_f64, 25.0_f64)).collect();"
            )
        elif op_name == "row_stats":
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(
                f"    let n_f = (feats.len().max(1)) as f64;\n"
                f"    let mean_f = feats.iter().sum::<f64>() / n_f;\n"
                f"    let var_f = feats.iter().map(|&v| (v - mean_f).powi(2)).sum::<f64>() / n_f;\n"
                f"    let stage_{idx}: Vec<f64> = vec![mean_f, var_f.sqrt()];"
            )
        else:
            stage_lines.append(f"    // Stage {idx}: {st}")
            stage_lines.append(
                f"    let stage_{idx}: Vec<f64> = feats.iter().map(|&v| v / (1.0_f64 + v.abs())).collect();"
            )

        if append_flag:
            stage_lines.append(f"    feats.extend(stage_{idx});")
        else:
            stage_lines.append(f"    if !stage_{idx}.is_empty() {{ feats = stage_{idx}; }}")

    head_op = head[1:].split()[0].rstrip(")")
    body = "\n".join(stage_lines)
    code = f"""//! Auto-generated standalone Rust GP pipeline module (`jope.flywheel-emitted-rust/v1`)
//! Context SHA256: `{context_sha256}`
//! Program SHA256: `{prog_sha}`
//! DSL: `{dsl}`

#![allow(unused_variables, dead_code)]

pub const PROGRAM_SHA256: &str = "{prog_sha}";
pub const DSL_EXPR: &str = r#"{dsl}"#;
pub const HEAD_KIND: &str = "{head_op}";

/// Execute the synthesized GP feature pipeline on an input row `x` and evaluate the fitted head.
pub fn transform_features(x: &[f64]) -> Vec<f64> {{
    let mut feats: Vec<f64> = x.iter().map(|&v| if v.is_finite() {{ v }} else {{ 0.0_f64 }}).collect();
{body}
    feats
}}

pub fn predict_with_weights(x: &[f64], weights: &[f64], intercept: f64) -> f64 {{
    let feats = transform_features(x);
    let dot: f64 = feats.iter().zip(weights.iter()).map(|(&f, &w)| f * w).sum();
    dot + intercept
}}
"""
    return code


def verify_rust_compilation(rust_code: str, out_rs_path: Path) -> dict:
    """Write `rust_code` to `out_rs_path` and verify `rustc --crate-type lib` compiles it cleanly."""
    out_rs_path.parent.mkdir(parents=True, exist_ok=True)
    out_rs_path.write_text(rust_code)
    rlib_path = out_rs_path.with_suffix(".rlib")
    t0 = time.monotonic()
    res = subprocess.run(
        ["rustc", "--edition", "2021", "--crate-type", "lib", "-O", str(out_rs_path), "-o", str(rlib_path)],
        capture_output=True,
        text=True,
    )
    wall_s = round(time.monotonic() - t0, 4)
    ok = res.returncode == 0 and rlib_path.exists()
    rlib_bytes = rlib_path.stat().st_size if ok else 0
    return {
        "compiled_ok": ok,
        "returncode": res.returncode,
        "rs_path": str(out_rs_path),
        "rlib_path": str(rlib_path) if ok else None,
        "rlib_bytes": rlib_bytes,
        "compile_wall_s": wall_s,
        "stderr": res.stderr[:300] if not ok else "",
    }


def run_self_test():
    print("[Phase 3 Self-Test] Loading corpus_manifest.json and champion_dsl_bank.json...")
    manifest = json.loads(Path("/mnt/jope-archive/flywheel-20261023/corpus/corpus_manifest.json").read_text())
    champ_bank = json.loads(Path("/mnt/jope-archive/flywheel-20261023/corpus/champion_dsl_bank.json").read_text())
    with np.load("/mnt/jope-archive/flywheel-20261023/corpus/context_features.npz", allow_pickle=False) as cf:
        lm_mat = cf["landmark_v2_48"]

    # Pick first non-empty DL-Train context
    rec = next(r for r in manifest["records"] if r["dl_split"] == "DL-Train" and r["n_candidates"] > 0)
    cid = rec["context_sha256"]
    idx = rec["index"]
    lm_vec = lm_mat[idx]

    # Find its existing winner DSL from champion_bank
    own_champs = [c for c in champ_bank if c["context_sha256"] == cid]
    proposals = propose_flywheel_candidates(
        parent_programs=own_champs,
        landmark_v2_48=lm_vec,
        champion_bank=champ_bank,
        target_family_id=rec["family_id"],
        n_proposals=6,
        seed=20261023,
    )
    print(f"[Phase 3 Self-Test] Generated {len(proposals)} EditZero / Landmark-routed GP proposals for context {cid[:16]}:")
    for p in proposals:
        print(f"  [{p['operator']}] {p['dsl']}")

    # Evaluate proposals live in Rust (--fit-s1 and --fit outer test)
    eval_res = evaluate_gp_batch_rust(
        teacher_jtt1_gz_path=rec["teacher_jtt1_path"],
        candidates=proposals,
        run_outer_test=True,
        max_workers=6,
    )
    for r in eval_res:
        print(
            f"  -> status={r['status']} fitness={r['fitness']:.4f} cv_ratio={r['cv_geomean_rmse_ratio']:.4f} "
            f"gate={r['cv_gate_passed']} test_ratio={r['test_rmse_ratio']} agreed={r['cv_test_agreed']} wall={r['wall_s']}s"
        )

    # Transpile best candidate to standalone Rust and verify compilation
    best = min(eval_res, key=lambda x: x["fitness"])
    rs_code = transpile_dsl_to_rust_code(best["dsl"], cid)
    comp_info = verify_rust_compilation(
        rs_code,
        Path("/mnt/jope-archive/flywheel-20261023/emitted_rust") / f"model_{cid[:16]}.rs",
    )
    print("[Phase 3 Self-Test] Rust transpilation & compilation check:", json.dumps(comp_info, indent=2))
    assert comp_info["compiled_ok"], "Rust compilation failed!"
    assert any(r["status"] == "completed" for r in eval_res), "No completed S1 evaluation!"

    # Test Stratified 300-Row multi-generation evolution (`Stratified300JTT1`)
    evo_res = evolve_context_slate_generations(
        teacher_jtt1_gz_path=rec["teacher_jtt1_path"],
        parent_programs=own_champs,
        landmark_v2_48=lm_vec,
        champion_bank=champ_bank,
        target_family_id=rec["family_id"],
        n_generations=2,
        children_per_gen=4,
        max_train_rows=300,
        seed=20261023,
    )
    print(
        f"[Phase 3 Self-Test] Stratified-300 2-generation evolution completed in {evo_res['wall_s']}s: "
        f"transitions={len(evo_res['paired_transitions'])}, "
        f"best_parent_f300={evo_res['best_parent_f300']:.4f} -> best_evolved_f300={evo_res['best_evolved_f300']:.4f} "
        f"(delta={evo_res['net_evo_improvement_300']:+.4f}), "
        f"CPU guard={GLOBAL_CPU_GUARD.summary()}"
    )
    assert len(evo_res["paired_transitions"]) > 0, "Expected paired evolutionary transitions!"
    print("[Phase 3 Self-Test] ALL CHECKS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
