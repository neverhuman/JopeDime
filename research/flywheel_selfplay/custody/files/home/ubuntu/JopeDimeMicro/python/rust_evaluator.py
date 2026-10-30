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
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np

sys.dont_write_bytecode = True
_MICRO_ROOT = Path(__file__).resolve().parent.parent
_CORRECTED_STUDY_DIR = str(_MICRO_ROOT / "python")
if _CORRECTED_STUDY_DIR not in sys.path:
    sys.path.insert(0, _CORRECTED_STUDY_DIR)
try:
    from rank_prepare import features as dsl_features_dict
except (ModuleNotFoundError, PermissionError, OSError):
    dsl_features_dict = None

WORKER_BIN = _MICRO_ROOT / "bin" / "bootstrap.s1main"
WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
PROGRAM_FEATURE_NAMES_PATH = _MICRO_ROOT / "assets" / "projections" / "program-feature-names.json"
EVAL_SCRATCH_ROOT = _MICRO_ROOT / "results" / "rust_eval_scratch"
STRAT300_CACHE_ROOT = _MICRO_ROOT / "results" / "strat300_jtt1"
PROJECTION_SHA = "f732f4ac6a8b2e85cb3fea49e9a9055db7c678c7b8444bd9f082c693a53068fc"

_FEAT_NAMES = None
_W1_SIGN_MAT = None


def _read_proc_stat_identity(pid: int) -> tuple[int, int, int] | None:
    """
    Read `(ppid, pgrp, starttime_ticks)` from `/proc/<pid>/stat`.
    Returns `None` if `/proc/<pid>/stat` is absent or unreadable (fail-closed).
    """
    if pid is None or int(pid) <= 1:
        return None
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii")
        rparen = raw.rfind(")")
        if rparen < 0:
            return None
        after = raw[rparen + 2 :].split()
        # after[0] is state (field 3), after[1] is ppid (field 4), after[2] is pgrp (field 5),
        # after[19] is starttime in clock ticks (field 22)
        ppid = int(after[1])
        pgrp = int(after[2])
        start_ticks = int(after[19])
        return ppid, pgrp, start_ticks
    except Exception:
        return None


def _verify_owned_child_identity(
    pid: int,
    expected_owner_pid: int,
    expected_start_ticks: int | None,
) -> tuple[bool, int | None]:
    """
    Fail-closed ownership & identity verifier before signaling any child process:
      1. Refuses if `pid <= 1`, `expected_owner_pid != os.getpid()`, or `expected_start_ticks is None`.
      2. Reads live `(ppid, pgrp, start_ticks)` from `/proc/<pid>/stat`.
      3. Requires `live_ppid == expected_owner_pid` AND `live_start_ticks == expected_start_ticks`.
      4. Never signals an owner CLI or unrelated process; any missing identity or PID reuse returns `(False, None)`.
    """
    if (
        pid is None
        or int(pid) <= 1
        or expected_owner_pid != os.getpid()
        or expected_start_ticks is None
    ):
        return False, None
    ident = _read_proc_stat_identity(int(pid))
    if ident is None:
        return False, None
    live_ppid, live_pgrp, live_start_ticks = ident
    if live_ppid != expected_owner_pid or live_start_ticks != int(expected_start_ticks):
        return False, None
    return True, live_pgrp


_KNOWN_FIT_EXECUTION_MODES: dict[str, int] = {
    "s1_fold_rotations": 2,
    "s1_single_holdout": 2,
    "s1_two_fold": 2,
    "fit_s1": 2,
    "fit_seven_fold": 8,
    "confirm_s1_7fold": 8,
    "non_fit_dummy_child": 0,
}


class OwnedChildProcessRegistry:
    """
    Instance-scoped registry of `subprocess.Popen` children spawned by this invocation only.
    - Atomically checks `_closed_for_spawns` / `cancel_event`, enforces pre-`Popen` candidate and
      conservative native-fit reservations (`2` per `--fit-s1` subprocess / `6` for three rotations,
      `8` per seven-fold subprocess, fail-closed on unknown `execution_mode`, no refund on failure/timeout),
      and spawns+registers `Popen` under `self._lock`, eliminating the queued-work / child-spawn race.
    - Signals ONLY verified children/process groups created and owned by `os.getpid()` whose
      `/proc/<pid>/stat` `(ppid, starttime_ticks)` match the registered child; missing identity
      or PID reuse fails closed (`0` signals sent).
    - Reaps terminated children via `proc.wait()` so kernel `RUSAGE_CHILDREN` includes their CPU time
      before end-of-process budget finalization.
    """

    def __init__(self, cancel_event: threading.Event | None = None) -> None:
        self.owner_pid = os.getpid()
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._lock = threading.Lock()
        self._closed_for_spawns = False
        self._active_children: dict[int, dict[str, Any]] = {}
        self.launched_count = 0
        self.completed_normally_count = 0
        self.timed_out_count = 0
        self.cancelled_during_run_count = 0
        self.refused_after_close_count = 0
        self.reaped_count = 0
        self.failed_closed_identity_count = 0
        self.max_uncached_candidate_evaluations: int | None = None
        self.max_reserved_native_fits: int | None = None
        self.allow_non_fit_dummy_when_capped: bool = False
        self.reserved_candidate_evaluations: int = 0
        self.reserved_native_fits: int = 0
        self.reserved_subprocess_launches: int = 0
        self.refused_on_reservation_limit_count: int = 0
        self.failed_closed_unknown_mode_count: int = 0
        self._reserved_candidate_keys: set[str] = set()
        self._reserved_fits_by_candidate_key: dict[str, int] = {}

    def configure_reservation_limits(
        self,
        max_uncached_candidate_evaluations: int | None = None,
        max_reserved_native_fits: int | None = None,
        allow_non_fit_dummy_when_capped: bool = False,
    ) -> None:
        with self._lock:
            self.max_uncached_candidate_evaluations = (
                int(max_uncached_candidate_evaluations)
                if max_uncached_candidate_evaluations is not None
                else None
            )
            self.max_reserved_native_fits = (
                int(max_reserved_native_fits)
                if max_reserved_native_fits is not None
                else None
            )
            self.allow_non_fit_dummy_when_capped = bool(allow_non_fit_dummy_when_capped)

    def _reserve_before_popen_unlocked(
        self,
        execution_mode: str | None,
        candidate_key: str | None,
    ) -> dict[str, Any]:
        limits_active = (
            self.max_uncached_candidate_evaluations is not None
            or self.max_reserved_native_fits is not None
        )
        if execution_mode is None:
            if limits_active and not self.allow_non_fit_dummy_when_capped:
                self.failed_closed_unknown_mode_count += 1
                raise ValueError(
                    "FailClosedExecutionModeError: execution_mode is required before Popen when reservation caps are active"
                )
            eff_mode = "non_fit_dummy_child"
        else:
            eff_mode = str(execution_mode)
            if eff_mode not in _KNOWN_FIT_EXECUTION_MODES or (
                eff_mode == "non_fit_dummy_child"
                and limits_active
                and not self.allow_non_fit_dummy_when_capped
            ):
                self.failed_closed_unknown_mode_count += 1
                raise ValueError(
                    f"FailClosedExecutionModeError: unknown or disallowed execution_mode={execution_mode!r}"
                )

        fits_delta = int(_KNOWN_FIT_EXECUTION_MODES[eff_mode])
        if eff_mode == "non_fit_dummy_child":
            cand_delta = 0
            eff_cand_key = None
        else:
            eff_cand_key = (
                str(candidate_key)
                if candidate_key is not None
                else f"anon_cand_{self.reserved_subprocess_launches}_{time.monotonic_ns()}"
            )
            cand_delta = 0 if eff_cand_key in self._reserved_candidate_keys else 1

        cand_exceeded = (
            self.max_uncached_candidate_evaluations is not None
            and (self.reserved_candidate_evaluations + cand_delta) > self.max_uncached_candidate_evaluations
        )
        fits_exceeded = (
            self.max_reserved_native_fits is not None
            and (self.reserved_native_fits + fits_delta) > self.max_reserved_native_fits
        )
        if cand_exceeded or fits_exceeded:
            self.refused_on_reservation_limit_count += 1
            return {
                "admitted": False,
                "execution_mode": eff_mode,
                "candidate_delta": 0,
                "reserved_fits_delta": 0,
                "reserved_candidate_evaluations": int(self.reserved_candidate_evaluations),
                "reserved_native_fits": int(self.reserved_native_fits),
                "refused_reason": (
                    "max_uncached_candidate_evaluations_exceeded"
                    if cand_exceeded
                    else "max_reserved_native_fits_exceeded"
                ),
            }

        self.reserved_candidate_evaluations += cand_delta
        self.reserved_native_fits += fits_delta
        if fits_delta > 0:
            self.reserved_subprocess_launches += 1
        if eff_cand_key is not None:
            self._reserved_candidate_keys.add(eff_cand_key)
            self._reserved_fits_by_candidate_key[eff_cand_key] = (
                self._reserved_fits_by_candidate_key.get(eff_cand_key, 0) + fits_delta
            )
        return {
            "admitted": True,
            "execution_mode": eff_mode,
            "candidate_key": eff_cand_key,
            "candidate_delta": int(cand_delta),
            "reserved_fits_delta": int(fits_delta),
            "reserved_candidate_evaluations": int(self.reserved_candidate_evaluations),
            "reserved_native_fits": int(self.reserved_native_fits),
            "candidate_total_reserved_fits": (
                int(self._reserved_fits_by_candidate_key[eff_cand_key])
                if eff_cand_key is not None
                else 0
            ),
        }

    def reserve_before_popen(
        self,
        *,
        execution_mode: str | None,
        candidate_key: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        eff_cancel = cancel_event if cancel_event is not None else self.cancel_event
        with self._lock:
            if self._closed_for_spawns or self.cancel_event.is_set() or eff_cancel.is_set():
                self.refused_after_close_count += 1
                return {
                    "admitted": False,
                    "cancelled_before_launch": True,
                    "refused_on_reservation_limit": False,
                    "candidate_delta": 0,
                    "reserved_fits_delta": 0,
                }
            res = self._reserve_before_popen_unlocked(execution_mode, candidate_key)
            res["cancelled_before_launch"] = not res["admitted"]
            res["refused_on_reservation_limit"] = not res["admitted"]
            return res

    def close_for_spawns(self) -> None:
        with self._lock:
            self._closed_for_spawns = True
            self.cancel_event.set()

    def is_closed_for_spawns(self) -> bool:
        with self._lock:
            return bool(self._closed_for_spawns or self.cancel_event.is_set())

    def active_child_count(self) -> int:
        with self._lock:
            return len(self._active_children)

    def register_external_owned_popen(
        self,
        proc: subprocess.Popen,
        start_ticks_override: int | None = "__READ_PROC__",
    ) -> dict[str, Any]:
        """Register an owned `Popen` object (used by unit tests to test identity & PID-reuse fail-closed paths)."""
        with self._lock:
            ident = _read_proc_stat_identity(proc.pid)
            start_ticks = (
                (ident[2] if ident is not None else None)
                if start_ticks_override == "__READ_PROC__"
                else start_ticks_override
            )
            entry = {
                "proc": proc,
                "pid": int(proc.pid),
                "owner_pid": self.owner_pid,
                "start_ticks": start_ticks,
                "pgrp": ident[1] if ident is not None else None,
                "reaped": False,
                "terminated_by_registry": False,
                "identity_failed_closed": False,
            }
            self._active_children[int(proc.pid)] = entry
            return entry

    def _terminate_and_reap_entry(self, entry: dict[str, Any], grace_s: float = 0.15) -> dict[str, Any]:
        proc: subprocess.Popen = entry["proc"]
        pid = int(entry["pid"])
        if proc.poll() is not None:
            try:
                proc.wait(timeout=0.2)
            except Exception:
                pass
            if not entry["reaped"]:
                entry["reaped"] = True
                with self._lock:
                    self.reaped_count += 1
                    self._active_children.pop(pid, None)
            return {
                "pid": pid,
                "already_exited": True,
                "signaled": False,
                "identity_verified": True,
                "identity_failed_closed": False,
                "reaped": True,
                "returncode": proc.returncode,
            }

        ok_ident, live_pgrp = _verify_owned_child_identity(
            pid=pid,
            expected_owner_pid=int(entry["owner_pid"]),
            expected_start_ticks=entry["start_ticks"],
        )
        if not ok_ident:
            entry["identity_failed_closed"] = True
            with self._lock:
                self.failed_closed_identity_count += 1
                self._active_children.pop(pid, None)
            return {
                "pid": pid,
                "already_exited": False,
                "signaled": False,
                "identity_verified": False,
                "identity_failed_closed": True,
                "reaped": False,
                "returncode": proc.poll(),
            }

        entry["terminated_by_registry"] = True
        signaled = False
        try:
            if live_pgrp == pid and pid > 1:
                os.killpg(pid, signal.SIGTERM)
            else:
                proc.terminate()
            signaled = True
        except ProcessLookupError:
            pass
        except Exception:
            pass

        t_wait0 = time.monotonic()
        while proc.poll() is None and (time.monotonic() - t_wait0) < max(0.02, float(grace_s)):
            time.sleep(0.015)

        if proc.poll() is None:
            ok_ident_2, live_pgrp_2 = _verify_owned_child_identity(
                pid=pid,
                expected_owner_pid=int(entry["owner_pid"]),
                expected_start_ticks=entry["start_ticks"],
            )
            if ok_ident_2:
                try:
                    if live_pgrp_2 == pid and pid > 1:
                        os.killpg(pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    signaled = True
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            else:
                entry["identity_failed_closed"] = True
                with self._lock:
                    self.failed_closed_identity_count += 1
                    self._active_children.pop(pid, None)
                return {
                    "pid": pid,
                    "already_exited": False,
                    "signaled": signaled,
                    "identity_verified": False,
                    "identity_failed_closed": True,
                    "reaped": False,
                    "returncode": proc.poll(),
                }

        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass
        reaped_now = proc.poll() is not None
        if reaped_now and not entry["reaped"]:
            entry["reaped"] = True
            with self._lock:
                self.reaped_count += 1
                self._active_children.pop(pid, None)
        return {
            "pid": pid,
            "already_exited": False,
            "signaled": signaled,
            "identity_verified": True,
            "identity_failed_closed": False,
            "reaped": reaped_now,
            "returncode": proc.returncode,
        }

    def terminate_and_reap_owned_children(self, grace_s: float = 0.15) -> dict[str, Any]:
        t0 = time.monotonic()
        with self._lock:
            self._closed_for_spawns = True
            self.cancel_event.set()
            entries = list(self._active_children.values())
        per_child = [self._terminate_and_reap_entry(e, grace_s=grace_s) for e in entries]
        with self._lock:
            remaining_active = len(self._active_children)
        return {
            "inspected_children_count": len(entries),
            "signaled_children_count": sum(1 for r in per_child if r.get("signaled")),
            "reaped_children_count": sum(1 for r in per_child if r.get("reaped")),
            "failed_closed_identity_count": sum(1 for r in per_child if r.get("identity_failed_closed")),
            "remaining_active_children_count": remaining_active,
            "cleanup_wall_seconds": round(time.monotonic() - t0, 4),
            "per_child": per_child,
        }

    def run_owned_subprocess(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        timeout_s: float,
        cancel_event: threading.Event | None = None,
        pre_spawn_hook: Callable[[], None] | None = None,
        execution_mode: str | None = None,
        candidate_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Atomically checks cancellation/close state, enforces pre-`Popen` candidate and fit
        reservations, and spawns+registers `subprocess.Popen` under `self._lock`, closing
        the queued-work / child-spawn race.
        """
        eff_cancel = cancel_event if cancel_event is not None else self.cancel_event
        if pre_spawn_hook is not None:
            pre_spawn_hook()
        with self._lock:
            if self._closed_for_spawns or self.cancel_event.is_set() or eff_cancel.is_set():
                self.refused_after_close_count += 1
                return {
                    "launched": False,
                    "returncode": -125,
                    "timed_out": False,
                    "cancelled_before_launch": True,
                    "cancelled_during_run": False,
                    "refused_on_reservation_limit": False,
                    "reserved_fits_delta": 0,
                    "reaped": True,
                }
            rsv = self._reserve_before_popen_unlocked(execution_mode, candidate_key)
            if not rsv["admitted"]:
                return {
                    "launched": False,
                    "returncode": -126,
                    "timed_out": False,
                    "cancelled_before_launch": True,
                    "cancelled_during_run": False,
                    "refused_on_reservation_limit": True,
                    "reserved_fits_delta": 0,
                    "reservation": rsv,
                    "reaped": True,
                }
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            ident = _read_proc_stat_identity(proc.pid)
            entry = {
                "proc": proc,
                "pid": int(proc.pid),
                "owner_pid": self.owner_pid,
                "start_ticks": ident[2] if ident is not None else None,
                "pgrp": ident[1] if ident is not None else None,
                "reaped": False,
                "terminated_by_registry": False,
                "identity_failed_closed": False,
            }
            self._active_children[int(proc.pid)] = entry
            self.launched_count += 1

        t0 = time.monotonic()
        timed_out = False
        cancelled_during = False
        while True:
            rc = proc.poll()
            if rc is not None:
                try:
                    proc.wait(timeout=0.1)
                except Exception:
                    pass
                with self._lock:
                    if not entry["reaped"]:
                        entry["reaped"] = True
                        self.reaped_count += 1
                    self._active_children.pop(int(proc.pid), None)
                    if entry["terminated_by_registry"]:
                        self.cancelled_during_run_count += 1
                    else:
                        self.completed_normally_count += 1
                return {
                    "launched": True,
                    "returncode": int(rc),
                    "timed_out": False,
                    "cancelled_before_launch": False,
                    "cancelled_during_run": bool(entry["terminated_by_registry"]),
                    "refused_on_reservation_limit": False,
                    "reserved_fits_delta": int(rsv["reserved_fits_delta"]),
                    "reservation": rsv,
                    "reaped": True,
                }
            if self._closed_for_spawns or self.cancel_event.is_set() or eff_cancel.is_set():
                cancelled_during = True
                break
            if (time.monotonic() - t0) >= float(timeout_s):
                timed_out = True
                break
            eff_cancel.wait(timeout=0.02)

        reap_info = self._terminate_and_reap_entry(entry, grace_s=0.15)
        with self._lock:
            if timed_out:
                self.timed_out_count += 1
            elif cancelled_during:
                self.cancelled_during_run_count += 1
        return {
            "launched": True,
            "returncode": int(reap_info["returncode"] if reap_info["returncode"] is not None else -9),
            "timed_out": bool(timed_out),
            "cancelled_before_launch": False,
            "cancelled_during_run": bool(cancelled_during or entry["terminated_by_registry"]),
            "refused_on_reservation_limit": False,
            "reserved_fits_delta": int(rsv["reserved_fits_delta"]),
            "reservation": rsv,
            "reaped": bool(reap_info["reaped"]),
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "owner_pid": self.owner_pid,
                "closed_for_spawns": bool(self._closed_for_spawns),
                "launched_count": int(self.launched_count),
                "completed_normally_count": int(self.completed_normally_count),
                "timed_out_count": int(self.timed_out_count),
                "cancelled_during_run_count": int(self.cancelled_during_run_count),
                "refused_after_close_count": int(self.refused_after_close_count),
                "reaped_count": int(self.reaped_count),
                "failed_closed_identity_count": int(self.failed_closed_identity_count),
                "remaining_active_children_count": len(self._active_children),
                "max_uncached_candidate_evaluations": self.max_uncached_candidate_evaluations,
                "max_reserved_native_fits": self.max_reserved_native_fits,
                "reserved_candidate_evaluations": int(self.reserved_candidate_evaluations),
                "reserved_native_fits": int(self.reserved_native_fits),
                "reserved_subprocess_launches": int(self.reserved_subprocess_launches),
                "refused_on_reservation_limit_count": int(self.refused_on_reservation_limit_count),
                "failed_closed_unknown_mode_count": int(self.failed_closed_unknown_mode_count),
            }


class EvaluationCancellationScope:
    """
    Invocation-scoped cooperative cancellation & worker/child lifecycle tracker.
    Prevents any nested unit-test timeout or completed attempt from leaving a sticky
    global cancellation flag that could cancel outer or subsequent operations.
    """

    def __init__(
        self,
        scope_id: str = "default",
        remaining_wall_fn: Callable[[], float] | None = None,
    ) -> None:
        self.scope_id = str(scope_id)
        self.cancel_event = threading.Event()
        self.child_registry = OwnedChildProcessRegistry(cancel_event=self.cancel_event)
        self.remaining_wall_fn = remaining_wall_fn
        self._workers_lock = threading.Lock()
        self._active_worker_tokens: set[int] = set()
        self._next_token = 0
        self._all_workers_exited = threading.Event()
        self._all_workers_exited.set()
        self.teardown_receipts: list[dict[str, Any]] = []

    def is_cancelled(self) -> bool:
        return bool(self.cancel_event.is_set() or self.child_registry.is_closed_for_spawns())

    def cancel(self) -> None:
        self.cancel_event.set()
        self.child_registry.close_for_spawns()

    def cancel_and_terminate_owned_children(self, grace_s: float = 0.15) -> dict[str, Any]:
        self.cancel()
        return self.child_registry.terminate_and_reap_owned_children(grace_s=grace_s)

    def on_worker_enter(self) -> int:
        with self._workers_lock:
            self._next_token += 1
            tok = self._next_token
            self._active_worker_tokens.add(tok)
            self._all_workers_exited.clear()
            return tok

    def on_worker_exit(self, tok: int) -> None:
        with self._workers_lock:
            self._active_worker_tokens.discard(tok)
            if not self._active_worker_tokens:
                self._all_workers_exited.set()

    def active_worker_count(self) -> int:
        with self._workers_lock:
            return len(self._active_worker_tokens)

    def wait_for_all_workers_exited(self, timeout_s: float = 2.0) -> bool:
        return bool(self._all_workers_exited.wait(timeout=max(0.01, float(timeout_s))))


_ACTIVE_EVAL_SCOPE_LOCK = threading.Lock()
_ACTIVE_EVAL_SCOPE_STACK: list[EvaluationCancellationScope] = []


def push_eval_cancellation_scope(scope: EvaluationCancellationScope) -> EvaluationCancellationScope:
    with _ACTIVE_EVAL_SCOPE_LOCK:
        _ACTIVE_EVAL_SCOPE_STACK.append(scope)
        return scope


def pop_eval_cancellation_scope(scope: EvaluationCancellationScope | None = None) -> None:
    with _ACTIVE_EVAL_SCOPE_LOCK:
        if not _ACTIVE_EVAL_SCOPE_STACK:
            return
        if scope is None or _ACTIVE_EVAL_SCOPE_STACK[-1] is scope:
            _ACTIVE_EVAL_SCOPE_STACK.pop()
        elif scope in _ACTIVE_EVAL_SCOPE_STACK:
            _ACTIVE_EVAL_SCOPE_STACK.remove(scope)


def get_current_eval_cancellation_scope() -> EvaluationCancellationScope | None:
    with _ACTIVE_EVAL_SCOPE_LOCK:
        return _ACTIVE_EVAL_SCOPE_STACK[-1] if _ACTIVE_EVAL_SCOPE_STACK else None


class CPUThrottleGuard:
    """
    Whole-host CPU utilization monitor & throttle guard (`<= 85%` hard cap, `80.0%` soft cap).
    Reads `/proc/stat` across all cores and yields/sleeps whenever whole-host CPU exceeds `max_cpu_pct`.
    Cooperatively wakes immediately if the active `EvaluationCancellationScope` is cancelled.
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
        self.cancelled_wait_exits = 0
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

    def throttle_if_needed(
        self,
        max_wait_s: float = 1.5,
        sleep_interval_s: float = 0.10,
        cancel_event: threading.Event | None = None,
    ) -> float:
        """
        Self-throttle only this process's submission loop when whole-host `/proc/stat`
        CPU utilization exceeds `self.max_cpu_pct` (soft cap `80.0%`, hard cap `85.0%`).
        Never signals, renices, or modifies any other process on the host.
        Wakes immediately if `cancel_event` (or the active `EvaluationCancellationScope`) is set.
        """
        eff_cancel = cancel_event
        if eff_cancel is None:
            cur_scope = get_current_eval_cancellation_scope()
            if cur_scope is not None:
                eff_cancel = cur_scope.cancel_event
        t0 = time.monotonic()
        pct = self._sample_proc_stat()
        while pct > self.max_cpu_pct and (time.monotonic() - t0) < max_wait_s:
            if eff_cancel is not None and eff_cancel.is_set():
                self.cancelled_wait_exits += 1
                break
            self.throttle_events += 1
            if eff_cancel is not None:
                if eff_cancel.wait(timeout=sleep_interval_s):
                    self.cancelled_wait_exits += 1
                    break
            else:
                time.sleep(sleep_interval_s)
            pct = self._sample_proc_stat()
        return pct

    def wait_for_headroom(
        self,
        max_wait_s: float = 1.5,
        cancel_event: threading.Event | None = None,
    ) -> float:
        return self.throttle_if_needed(
            max_wait_s=max_wait_s,
            sleep_interval_s=0.10,
            cancel_event=cancel_event,
        )

    def adaptive_worker_count(
        self,
        requested_workers: int = 6,
        cancel_event: threading.Event | None = None,
    ) -> int:
        pct = self.throttle_if_needed(
            max_wait_s=0.6,
            sleep_interval_s=0.10,
            cancel_event=cancel_event,
        )
        if pct >= self.hard_cap_pct or pct >= self.max_cpu_pct:
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
            "cancelled_wait_exits": int(self.cancelled_wait_exits),
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
    if dsl_features_dict is None:
        raise RuntimeError(
            f"rank_prepare.features requires {_MICRO_ROOT / 'python' / 'rank_prepare.py'}"
        )
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


def compute_kfold_metrics_from_inner_cv(dsl: str, inner_cv: list[dict], expected_k: int) -> dict:
    """
    Compute K-fold constrained fitness F_K and multi-fold CV gate for K in {1, 2, 3, 5, 7}.
    Outer test labels are never used; `inner_cv` contains the K training-split fold records.
    """
    if len(inner_cv) != expected_k or expected_k < 1:
        return {
            "status": "failed",
            "k_folds": int(expected_k),
            "fitness": 100.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_mean_log_ratio": 0.0,
            "cv_tail_log_ratio": 0.0,
            "cv_log_ratio_sd": 0.0,
            "cv_fold_wins": 0,
            "cv_win_fraction": 0.0,
            "cv_gate_passed": False,
        }
    lrs = []
    wins = 0
    fold_train_rows = []
    fold_score_rows = []
    fold_wall_us = []
    for fold_rec in inner_cv:
        m_rmse = (fold_rec.get("metrics") or {}).get("rmse")
        b_rmse = (fold_rec.get("baseline_metrics") or {}).get("rmse")
        if (
            m_rmse is None
            or b_rmse is None
            or not math.isfinite(m_rmse)
            or not math.isfinite(b_rmse)
            or float(b_rmse) <= 1e-12
        ):
            return {
                "status": "failed",
                "k_folds": int(expected_k),
                "fitness": 100.0,
                "logratio": 0.0,
                "cv_geomean_rmse_ratio": 1e9,
                "cv_mean_log_ratio": 0.0,
                "cv_tail_log_ratio": 0.0,
                "cv_log_ratio_sd": 0.0,
                "cv_fold_wins": 0,
                "cv_win_fraction": 0.0,
                "cv_gate_passed": False,
                "fold_train_rows": [],
                "fold_score_rows": [],
                "used_fold_wall_ms": 0.0,
            }
        r = max(float(m_rmse) / float(b_rmse), 1e-12)
        if r <= 1.0:
            wins += 1
        lrs.append(math.log(r))
        fold_train_rows.append(int(fold_rec.get("train_rows", 0)))
        fold_score_rows.append(int(fold_rec.get("score_rows", 0)))
        fold_wall_us.append(int(fold_rec.get("wall_us", 0)))

    mean_lr = float(sum(lrs) / len(lrs))
    tail_lr = float(max(lrs))
    var_lr = float(sum((x - mean_lr) ** 2 for x in lrs) / len(lrs))
    sd_lr = float(math.sqrt(max(var_lr, 0.0)))
    geomean_ratio = float(math.exp(mean_lr))
    comp = int(dsl.count("("))

    min_wins = expected_k if expected_k <= 3 else int(math.ceil(0.80 * expected_k))
    tail_ok = True if expected_k <= 2 else (tail_lr <= 0.02)
    gate = (wins >= min_wins) and (geomean_ratio <= 0.90) and tail_ok

    if gate:
        fitness = 0.80 * mean_lr + 0.10 * tail_lr + 0.10 * sd_lr + 0.0005 * comp
    else:
        fitness = 100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0)

    return {
        "status": "completed",
        "k_folds": int(expected_k),
        "fitness": float(fitness),
        "logratio": float(mean_lr),
        "cv_geomean_rmse_ratio": float(geomean_ratio),
        "cv_mean_log_ratio": float(mean_lr),
        "cv_tail_log_ratio": float(tail_lr),
        "cv_log_ratio_sd": float(sd_lr),
        "complexity_open_parens": int(comp),
        "cv_fold_wins": int(wins),
        "cv_win_fraction": float(wins / expected_k),
        "cv_gate_passed": bool(gate),
        "fold_log_ratios": [round(float(x), 6) for x in lrs],
        "fold_train_rows": fold_train_rows,
        "fold_score_rows": fold_score_rows,
        "sum_fold_train_rows": int(sum(fold_train_rows)),
        "sum_fold_score_rows": int(sum(fold_score_rows)),
        "used_fold_wall_ms": round(sum(fold_wall_us) / 1000.0, 3),
    }


def prepare_stratified_kfold_jtt1_paths(
    teacher_jtt1_gz_path: str,
    k_folds: int = 5,
    max_train_rows: int | None = None,
    row_subset_mode: str = "all",
) -> tuple[str, list[Path]]:
    """
    Build cached JTT1 files for target-quantile-stratified K-fold inner CV (K in {2, 3, 5, 7})
    or 70/30 held-out training verification (`row_subset_mode == 'holdout30_verify'`).
    Outer test targets (`nq` rows) are strictly zeroed out (`0.0`) so zero test leakage can occur.

    Returns `(exec_mode, paths)` where:
      - `exec_mode == 's1_two_fold'`: 1 file with `k=2`, evaluated via `--fit-s1` using both `inner_cv[0..2]`
      - `exec_mode == 'fit_seven_fold'`: 1 file with `k=7`, evaluated via `--fit` using all `inner_cv[0..7]`
      - `exec_mode == 's1_fold_rotations'`: K files with `k=2`, where file `f` puts fold `f` as `0` (val)
        and complement as `1` (train); `--fit-s1` evaluates `inner_cv[0]` (`tr = folds != 0`, `va = folds == 0`)
      - `exec_mode == 's1_single_holdout'`: 1 file with `k=2`, where `folds == 0` is the 30% held-out
        verification subset of training rows and `folds == 1` is the 70% inner-selection subset;
        `--fit-s1` `inner_cv[0]` fits on the 70% subset and scores on the 30% subset.
    """
    STRAT300_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    max_tag = "all" if max_train_rows is None else str(int(max_train_rows))
    ctx_tag = hashlib.sha256(
        f"v2:{teacher_jtt1_gz_path}:k={k_folds}:max={max_tag}:mode={row_subset_mode}".encode()
    ).hexdigest()[:20]

    raw = None

    def _load_train_and_zeroed_test():
        nonlocal raw
        if raw is None:
            raw = gzip.decompress(Path(teacher_jtt1_gz_path).read_bytes())
        if raw[:4] != b"JTT1":
            raise ValueError(f"Invalid JTT1 magic in {teacher_jtt1_gz_path}")
        nt, nq, p = struct.unpack_from("<III", raw, 4)
        row_floats = p + 1
        row_bytes = row_floats * 8
        train_mat = np.frombuffer(
            raw, dtype="<f8", count=nt * row_floats, offset=17 + nt
        ).reshape(nt, row_floats).copy()
        end_train = 17 + nt + nt * row_bytes
        sub_nq = int(min(nq, 60))
        test_mat = np.frombuffer(
            raw, dtype="<f8", count=sub_nq * row_floats, offset=end_train
        ).reshape(sub_nq, row_floats).copy()
        test_mat[:, -1] = 0.0  # Strictly zero out outer test labels
        sub_test = np.ascontiguousarray(test_mat, dtype="<f8")
        return nt, sub_nq, p, train_mat, sub_test

    def _compute_holdout30_mask(train_mat: np.ndarray) -> np.ndarray:
        """
        Deterministically partition unique row byte patterns into 70% inner-selection (False)
        and 30% held-out verification (True) stratified by target quantile so identical rows
        never straddle the inner70 / holdout30 boundary (0 byte-level leakage).
        """
        nt_full = train_mat.shape[0]
        groups: dict[bytes, list[int]] = {}
        for idx in range(nt_full):
            rb = train_mat[idx].tobytes()
            if rb not in groups:
                groups[rb] = []
            groups[rb].append(idx)
        sorted_groups = sorted(groups.values(), key=lambda idxs: (float(train_mat[idxs[0], -1]), idxs[0]))
        is_holdout30 = np.zeros(nt_full, dtype=np.bool_)
        for rank_g, idxs in enumerate(sorted_groups):
            if (rank_g % 10) in (1, 4, 8):
                for idx in idxs:
                    is_holdout30[idx] = True
        return is_holdout30

    if row_subset_mode == "holdout30_verify":
        out_path = STRAT300_CACHE_ROOT / f"kfold_holdout30_{ctx_tag}.jtt1"
        if not out_path.exists():
            nt, sub_nq, p, train_mat, sub_test = _load_train_and_zeroed_test()
            is_h30 = _compute_holdout30_mask(train_mat)
            # Deterministic 70% train (folds=1) / 30% verification (folds=0) stratified by target quantile
            folds = np.where(is_h30, 0, 1).astype(np.uint8)
            sub_train = np.ascontiguousarray(train_mat, dtype="<f8")
            header = struct.pack("<4sIIIB", b"JTT1", int(nt), int(sub_nq), int(p), 2)
            tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_bytes(header + folds.tobytes() + sub_train.tobytes() + sub_test.tobytes())
            tmp.replace(out_path)
        return "s1_single_holdout", [out_path]

    def _select_active_train_rows(train_mat: np.ndarray) -> np.ndarray:
        nt_full = train_mat.shape[0]
        if row_subset_mode == "inner70":
            is_h30 = _compute_holdout30_mask(train_mat)
            active_idx = np.flatnonzero(~is_h30)
        else:
            active_idx = np.arange(nt_full, dtype=np.int64)
        sub = train_mat[active_idx]
        if max_train_rows is not None and sub.shape[0] > max_train_rows:
            ord_sub = np.argsort(sub[:, -1], kind="mergesort")
            grid = np.linspace(0, len(ord_sub) - 1, num=int(max_train_rows), dtype=np.int64)
            chosen = np.sort(ord_sub[grid])
            sub = sub[chosen]
        return np.ascontiguousarray(sub, dtype="<f8")

    if k_folds == 2:
        out_path = STRAT300_CACHE_ROOT / f"kfold2_{ctx_tag}.jtt1"
        if not out_path.exists():
            _, sub_nq, p, train_mat, sub_test = _load_train_and_zeroed_test()
            sub_train = _select_active_train_rows(train_mat)
            sub_nt = sub_train.shape[0]
            order = np.argsort(sub_train[:, -1], kind="mergesort")
            folds = np.zeros(sub_nt, dtype=np.uint8)
            folds[order] = (np.arange(sub_nt) % 2).astype(np.uint8)
            header = struct.pack("<4sIIIB", b"JTT1", int(sub_nt), int(sub_nq), int(p), 2)
            tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_bytes(header + folds.tobytes() + sub_train.tobytes() + sub_test.tobytes())
            tmp.replace(out_path)
        return "s1_two_fold", [out_path]

    if k_folds == 7:
        out_path = STRAT300_CACHE_ROOT / f"kfold7_{ctx_tag}.jtt1"
        if not out_path.exists():
            _, sub_nq, p, train_mat, sub_test = _load_train_and_zeroed_test()
            sub_train = _select_active_train_rows(train_mat)
            sub_nt = sub_train.shape[0]
            order = np.argsort(sub_train[:, -1], kind="mergesort")
            folds = np.zeros(sub_nt, dtype=np.uint8)
            folds[order] = (np.arange(sub_nt) % 7).astype(np.uint8)
            header = struct.pack("<4sIIIB", b"JTT1", int(sub_nt), int(sub_nq), int(p), 7)
            tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_bytes(header + folds.tobytes() + sub_train.tobytes() + sub_test.tobytes())
            tmp.replace(out_path)
        return "fit_seven_fold", [out_path]

    # General K-fold (e.g., K=3 or K=5): K rotation files where fold 0 is validation (1/K) and fold 1 is train ((K-1)/K)
    paths = [STRAT300_CACHE_ROOT / f"kfold{k_folds}_rot{f}_{ctx_tag}.jtt1" for f in range(k_folds)]
    if any(not p_f.exists() for p_f in paths):
        _, sub_nq, p, train_mat, sub_test = _load_train_and_zeroed_test()
        sub_train = _select_active_train_rows(train_mat)
        sub_nt = sub_train.shape[0]
        order = np.argsort(sub_train[:, -1], kind="mergesort")
        strat_ids = np.zeros(sub_nt, dtype=np.int64)
        strat_ids[order] = np.arange(sub_nt) % k_folds
        header = struct.pack("<4sIIIB", b"JTT1", int(sub_nt), int(sub_nq), int(p), 2)
        for f, p_f in enumerate(paths):
            folds_f = np.where(strat_ids == f, 0, 1).astype(np.uint8)
            tmp = p_f.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_bytes(header + folds_f.tobytes() + sub_train.tobytes() + sub_test.tobytes())
            tmp.replace(p_f)
    return "s1_fold_rotations", paths


def _eval_single_dsl_kfold_worker(args: tuple) -> dict:
    if len(args) == 8:
        slot_id, exec_mode, jtt1_paths, k_folds, dsl, operator, parents, timeout_s = args
        scope = get_current_eval_cancellation_scope()
        subprocess_runner = None
    elif len(args) == 9:
        slot_id, exec_mode, jtt1_paths, k_folds, dsl, operator, parents, timeout_s, scope = args
        subprocess_runner = None
    else:
        slot_id, exec_mode, jtt1_paths, k_folds, dsl, operator, parents, timeout_s, scope, subprocess_runner = args[:10]

    if scope is None:
        scope = get_current_eval_cancellation_scope()
    cancel_ev = scope.cancel_event if scope is not None else None
    worker_token = scope.on_worker_enter() if scope is not None else None

    try:
        if exec_mode not in _KNOWN_FIT_EXECUTION_MODES or exec_mode == "non_fit_dummy_child":
            if scope is not None:
                with scope.child_registry._lock:
                    scope.child_registry.failed_closed_unknown_mode_count += 1
            raise ValueError(
                f"FailClosedExecutionModeError: unknown or disallowed kfold execution_mode={exec_mode!r}"
            )
        if scope is not None and scope.is_cancelled():
            prog_sha = hashlib.sha256(dsl.encode()).hexdigest()
            f412, proj16 = featurize_dsl_412(dsl)
            expected_k = 1 if exec_mode == "s1_single_holdout" else int(k_folds)
            res = compute_kfold_metrics_from_inner_cv(dsl, [], expected_k)
            max_fits_per_call = int(_KNOWN_FIT_EXECUTION_MODES[exec_mode])
            res.update({
                "dsl": dsl,
                "program_sha256": prog_sha,
                "operator": operator,
                "parents": list(parents),
                "wall_s": 0.0,
                "exec_mode": exec_mode,
                "cancelled_before_launch": True,
                "refused_on_reservation_limit": False,
                "subprocess_calls": 0,
                "recorded_subprocess_calls_launched": 0,
                "completed_subprocesses_count": 0,
                "failed_or_timed_out_subprocesses_count": 0,
                "cancelled_before_launch_subprocesses_count": 1,
                "refused_on_reservation_limit_subprocesses_count": 0,
                "reserved_native_fits": 0,
                "used_cv_fold_fits": 0,
                "measured_used_cv_fold_fits": 0,
                "measured_discarded_completed_cv_fold_fits": 0,
                "total_native_driver_fits_executed": 0,
                "measured_completed_native_driver_fits": 0,
                "unobserved_failed_or_timed_out_internal_fits": 0,
                "actual_attempted_internal_fits_including_failed_subprocesses": 0,
                "max_driver_fits_per_subprocess_for_mode": max_fits_per_call,
                "inferred_max_possible_driver_fits_from_launched_subprocesses": 0,
                "inferred_upper_bound_provenance": (
                    "inferred_upper_bound_only_not_measured_attempted_fits "
                    "(recorded_subprocess_calls_launched * max_driver_fits_per_subprocess_for_mode)"
                ),
                "driver_wall_ms": 0.0,
                "feat412": f412,
                "proj16": proj16,
            })
            return res

        GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=1.0, cancel_event=cancel_ev)
        EVAL_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        slot_dir = EVAL_SCRATCH_ROOT / f"kslot-{os.getpid()}-{threading.get_ident()}-{slot_id}"
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
        cand_path.write_text(json.dumps(cand_obj, sort_keys=True))

        env = {
            "PATH": "/usr/bin:/bin",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "TMPDIR": str(slot_dir),
            "HOME": str(slot_dir),
        }
        t0 = time.monotonic()
        fold_records = []
        ok = True
        subprocess_calls = 0
        completed_subprocesses = 0
        failed_or_timed_out_subprocesses = 0
        cancelled_before_launch_subprocesses = 0
        refused_on_reservation_limit_subprocesses = 0
        reserved_native_fits_for_cand = 0
        total_native_driver_fits = 0
        driver_wall_us_total = 0
        cand_reservation_key = (
            f"{exec_mode}:{jtt1_paths[0].name if jtt1_paths else slot_id}:{prog_sha}:{slot_id}"
        )

        def _invoke_subproc(cmd: list[str], call_timeout_s: float) -> tuple[bool, int]:
            nonlocal subprocess_calls, cancelled_before_launch_subprocesses
            nonlocal refused_on_reservation_limit_subprocesses, reserved_native_fits_for_cand
            if scope is not None and scope.is_cancelled():
                cancelled_before_launch_subprocesses += 1
                return False, -125
            eff_timeout_s = float(call_timeout_s)
            if scope is not None and getattr(scope, "remaining_wall_fn", None) is not None:
                rem_w = float(scope.remaining_wall_fn())
                if rem_w <= 0.05:
                    scope.cancel()
                    cancelled_before_launch_subprocesses += 1
                    return False, -125
                eff_timeout_s = max(0.05, min(eff_timeout_s, rem_w))
            if subprocess_runner is not None:
                if scope is not None:
                    rsv = scope.child_registry.reserve_before_popen(
                        execution_mode=exec_mode,
                        candidate_key=cand_reservation_key,
                        cancel_event=cancel_ev,
                    )
                    if not rsv.get("admitted", False):
                        cancelled_before_launch_subprocesses += 1
                        if rsv.get("refused_on_reservation_limit", False):
                            refused_on_reservation_limit_subprocesses += 1
                            return False, -126
                        return False, -125
                    reserved_native_fits_for_cand += int(rsv.get("reserved_fits_delta", 0))
                r = subprocess_runner(cmd, env=env, timeout_s=eff_timeout_s, cancel_event=cancel_ev)
                if isinstance(r, dict):
                    if not r.get("launched", True):
                        cancelled_before_launch_subprocesses += 1
                        return False, int(r.get("returncode", -125))
                    subprocess_calls += 1
                    return True, int(r.get("returncode", 1))
                subprocess_calls += 1
                return True, int(getattr(r, "returncode", r))
            if scope is not None:
                r = scope.child_registry.run_owned_subprocess(
                    cmd,
                    env=env,
                    timeout_s=eff_timeout_s,
                    cancel_event=cancel_ev,
                    execution_mode=exec_mode,
                    candidate_key=cand_reservation_key,
                )
                if not r.get("launched", False):
                    cancelled_before_launch_subprocesses += 1
                    if r.get("refused_on_reservation_limit", False):
                        refused_on_reservation_limit_subprocesses += 1
                    return False, int(r.get("returncode", -125))
                reserved_native_fits_for_cand += int(r.get("reserved_fits_delta", 0))
                subprocess_calls += 1
                return True, int(r.get("returncode", -9))
            subprocess_calls += 1
            reserved_native_fits_for_cand += int(_KNOWN_FIT_EXECUTION_MODES[exec_mode])
            cp = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=eff_timeout_s,
                env=env,
            )
            return True, int(cp.returncode)

        try:
            if exec_mode == "s1_two_fold":
                out_p = slot_dir / "eval_k2.json"
                out_p.unlink(missing_ok=True)
                launched, rc = _invoke_subproc(
                    ["nice", "-n", "10", str(WORKER_BIN), "--fit-s1", str(jtt1_paths[0]), str(cand_path), str(out_p)],
                    float(timeout_s),
                )
                if launched and rc == 0 and out_p.exists():
                    completed_subprocesses += 1
                    doc = json.loads(out_p.read_text())
                    fold_records = list(doc.get("inner_cv") or [])
                    total_native_driver_fits += len(fold_records)
                    driver_wall_us_total += sum(int(r.get("wall_us", 0)) for r in fold_records)
                else:
                    if launched:
                        failed_or_timed_out_subprocesses += 1
                    ok = False
            elif exec_mode == "fit_seven_fold":
                out_p = slot_dir / "eval_k7.json"
                out_p.unlink(missing_ok=True)
                launched, rc = _invoke_subproc(
                    ["nice", "-n", "10", str(WORKER_BIN), "--fit", str(jtt1_paths[0]), str(cand_path), str(out_p)],
                    float(timeout_s),
                )
                if launched and rc == 0 and out_p.exists():
                    doc = json.loads(out_p.read_text())
                    fold_records = list(doc.get("inner_cv") or [])
                    total_native_driver_fits += len(fold_records) + (1 if doc.get("official_test") else 0)
                    driver_wall_us_total += int(doc.get("wall_us", sum(int(r.get("wall_us", 0)) for r in fold_records)))
                    if len(fold_records) == 7:
                        completed_subprocesses += 1
                    else:
                        failed_or_timed_out_subprocesses += 1
                        ok = False
                else:
                    if launched:
                        failed_or_timed_out_subprocesses += 1
                    ok = False
            elif exec_mode == "s1_single_holdout":
                out_p = slot_dir / "eval_holdout30.json"
                out_p.unlink(missing_ok=True)
                launched, rc = _invoke_subproc(
                    ["nice", "-n", "10", str(WORKER_BIN), "--fit-s1", str(jtt1_paths[0]), str(cand_path), str(out_p)],
                    float(timeout_s),
                )
                if launched and rc == 0 and out_p.exists():
                    doc = json.loads(out_p.read_text())
                    icv = list(doc.get("inner_cv") or [])
                    total_native_driver_fits += len(icv)
                    driver_wall_us_total += sum(int(r.get("wall_us", 0)) for r in icv)
                    if len(icv) >= 1:
                        completed_subprocesses += 1
                        fold_records = [icv[0]]
                    else:
                        failed_or_timed_out_subprocesses += 1
                        ok = False
                else:
                    if launched:
                        failed_or_timed_out_subprocesses += 1
                    ok = False
            elif exec_mode == "s1_fold_rotations":
                per_rot_timeout = max(3.0, float(timeout_s) / max(len(jtt1_paths), 1))
                for f_idx, p_f in enumerate(jtt1_paths):
                    if scope is not None and scope.is_cancelled():
                        cancelled_before_launch_subprocesses += 1
                        ok = False
                        break
                    GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.4, cancel_event=cancel_ev)
                    if scope is not None and scope.is_cancelled():
                        cancelled_before_launch_subprocesses += 1
                        ok = False
                        break
                    out_p = slot_dir / f"eval_rot_{f_idx}.json"
                    out_p.unlink(missing_ok=True)
                    launched, rc = _invoke_subproc(
                        ["nice", "-n", "10", str(WORKER_BIN), "--fit-s1", str(p_f), str(cand_path), str(out_p)],
                        per_rot_timeout,
                    )
                    if launched and rc == 0 and out_p.exists():
                        doc = json.loads(out_p.read_text())
                        icv = list(doc.get("inner_cv") or [])
                        total_native_driver_fits += len(icv)
                        driver_wall_us_total += sum(int(r.get("wall_us", 0)) for r in icv)
                        if len(icv) >= 1:
                            completed_subprocesses += 1
                            fold_records.append(icv[0])
                        else:
                            failed_or_timed_out_subprocesses += 1
                            ok = False
                            break
                    else:
                        if launched:
                            failed_or_timed_out_subprocesses += 1
                        ok = False
                        break
            else:
                ok = False
        except ValueError:
            shutil.rmtree(slot_dir, ignore_errors=True)
            raise
        except Exception:
            if subprocess_calls > completed_subprocesses + failed_or_timed_out_subprocesses:
                failed_or_timed_out_subprocesses = subprocess_calls - completed_subprocesses
            ok = False

        wall_s = round(time.monotonic() - t0, 4)
        shutil.rmtree(slot_dir, ignore_errors=True)
        f412, proj16 = featurize_dsl_412(dsl)

        expected_k = 1 if exec_mode == "s1_single_holdout" else int(k_folds)
        if not ok:
            res = compute_kfold_metrics_from_inner_cv(dsl, [], expected_k)
        else:
            res = compute_kfold_metrics_from_inner_cv(dsl, fold_records, expected_k)

        used_cv_fits = len(fold_records) if ok else 0
        discarded_cv_fits = 0 if ok else len(fold_records)
        max_fits_per_call = int(_KNOWN_FIT_EXECUTION_MODES[exec_mode])
        inferred_upper_bound = int(subprocess_calls * max_fits_per_call)

        res.update({
            "dsl": dsl,
            "program_sha256": prog_sha,
            "operator": operator,
            "parents": list(parents),
            "wall_s": wall_s,
            "exec_mode": exec_mode,
            "refused_on_reservation_limit": bool(refused_on_reservation_limit_subprocesses > 0),
            "subprocess_calls": int(subprocess_calls),
            "recorded_subprocess_calls_launched": int(subprocess_calls),
            "completed_subprocesses_count": int(completed_subprocesses),
            "failed_or_timed_out_subprocesses_count": int(failed_or_timed_out_subprocesses),
            "cancelled_before_launch_subprocesses_count": int(cancelled_before_launch_subprocesses),
            "refused_on_reservation_limit_subprocesses_count": int(refused_on_reservation_limit_subprocesses),
            "reserved_native_fits": int(reserved_native_fits_for_cand),
            "used_cv_fold_fits": int(used_cv_fits),
            "measured_used_cv_fold_fits": int(used_cv_fits),
            "measured_discarded_completed_cv_fold_fits": int(discarded_cv_fits),
            "total_native_driver_fits_executed": int(total_native_driver_fits),
            "measured_completed_native_driver_fits": int(total_native_driver_fits),
            "unobserved_failed_or_timed_out_internal_fits": (
                None if failed_or_timed_out_subprocesses > 0 else 0
            ),
            "actual_attempted_internal_fits_including_failed_subprocesses": (
                None if failed_or_timed_out_subprocesses > 0 else int(total_native_driver_fits)
            ),
            "max_driver_fits_per_subprocess_for_mode": int(max_fits_per_call),
            "inferred_max_possible_driver_fits_from_launched_subprocesses": int(inferred_upper_bound),
            "inferred_upper_bound_provenance": (
                "inferred_upper_bound_only_not_measured_attempted_fits "
                "(recorded_subprocess_calls_launched * max_driver_fits_per_subprocess_for_mode)"
            ),
            "driver_wall_ms": round(driver_wall_us_total / 1000.0, 3),
            "feat412": f412,
            "proj16": proj16,
        })
        return res
    finally:
        if scope is not None and worker_token is not None:
            scope.on_worker_exit(worker_token)


def evaluate_gp_batch_rust_kfold(
    teacher_jtt1_gz_path: str,
    candidates: list[dict],
    k_folds: int = 5,
    max_train_rows: int | None = None,
    row_subset_mode: str = "all",
    max_workers: int = 6,
    timeout_s: float = 20.0,
    eval_scope: EvaluationCancellationScope | None = None,
    subprocess_runner: Callable[..., Any] | None = None,
    prepared_paths_override: tuple[str, list[Path]] | None = None,
    worker_join_timeout_s: float = 5.0,
) -> list[dict]:
    """
    Evaluate a batch of GP candidates under target-quantile-stratified K-fold inner CV
    (`k_folds` in {2, 3, 5, 7}) or 70/30 held-out training verification (`row_subset_mode='holdout30_verify'`).
    Outer test targets are strictly zeroed out.
    Uses bounded cooperative cancellation and owned-child termination/reaping on exception.
    """
    if not candidates:
        return []
    scope = eval_scope if eval_scope is not None else get_current_eval_cancellation_scope()
    owns_local_scope = False
    if scope is None:
        scope = EvaluationCancellationScope(scope_id=f"batch_kfold_{os.getpid()}_{time.monotonic_ns()}")
        owns_local_scope = True

    if prepared_paths_override is not None:
        exec_mode, jtt1_paths = prepared_paths_override
    else:
        exec_mode, jtt1_paths = prepare_stratified_kfold_jtt1_paths(
            teacher_jtt1_gz_path=teacher_jtt1_gz_path,
            k_folds=k_folds,
            max_train_rows=max_train_rows,
            row_subset_mode=row_subset_mode,
        )
    eff_workers = GLOBAL_CPU_GUARD.adaptive_worker_count(max_workers, cancel_event=scope.cancel_event)
    tasks = [
        (
            2000 + i,
            exec_mode,
            jtt1_paths,
            k_folds,
            c["dsl"],
            c.get("operator", "edit_zero"),
            c.get("parents", []),
            timeout_s,
            scope,
            subprocess_runner,
        )
        for i, c in enumerate(candidates)
    ]
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=min(eff_workers, len(tasks)))
    futures: list[concurrent.futures.Future] = []
    try:
        for t_arg in tasks:
            if scope.is_cancelled():
                break
            futures.append(ex.submit(_eval_single_dsl_kfold_worker, t_arg))
        results = [fut.result() for fut in futures]
        ex.shutdown(wait=True, cancel_futures=False)
        return results
    except BaseException:
        t_clean0 = time.monotonic()
        scope.cancel()
        cancelled_unstarted = sum(1 for fut in futures if fut.cancel())
        child_cleanup = scope.child_registry.terminate_and_reap_owned_children(grace_s=0.15)
        done_futs, not_done_futs = concurrent.futures.wait(
            futures,
            timeout=max(0.1, float(worker_join_timeout_s)),
        )
        all_workers_exited = scope.wait_for_all_workers_exited(
            timeout_s=max(0.05, float(worker_join_timeout_s) - (time.monotonic() - t_clean0))
        )
        ex.shutdown(wait=all_workers_exited and len(not_done_futs) == 0, cancel_futures=True)
        scope.teardown_receipts.append({
            "layer": "evaluate_gp_batch_rust_kfold",
            "submitted_futures_count": len(futures),
            "cancelled_unstarted_futures_count": int(cancelled_unstarted),
            "done_futures_count": len(done_futs),
            "pending_futures_after_bounded_wait_count": len(not_done_futs),
            "all_workers_exited_cooperatively": bool(all_workers_exited and len(not_done_futs) == 0),
            "active_workers_remaining": int(scope.active_worker_count()),
            "child_cleanup": child_cleanup,
            "teardown_wall_seconds": round(time.monotonic() - t_clean0, 4),
        })
        raise
    finally:
        if owns_local_scope and scope.child_registry.active_child_count() > 0:
            scope.child_registry.terminate_and_reap_owned_children(grace_s=0.10)


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
    manifest = json.loads((_MICRO_ROOT / "assets" / "benchmark_data" / "corpus_manifest.json").read_text())
    champ_bank = json.loads((_MICRO_ROOT / "assets" / "benchmark_data" / "champion_dsl_bank.json").read_text())
    with np.load(str(_MICRO_ROOT / "assets" / "benchmark_data" / "context_features.npz"), allow_pickle=False) as cf:
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
        _MICRO_ROOT / "results" / "emitted_rust" / f"model_{cid[:16]}.rs",
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
