"""Rust fitness evaluator timings (TELEMETRY_SPEC rows E5-E7).

The flywheel evaluates candidates by running the Rust worker as a subprocess from a thread pool
(``rust_evaluator.py``: ``evaluate_gp_batch_rust*`` -> ``_eval_single_dsl_worker``). Each
worker first waits for host CPU headroom (``GLOBAL_CPU_GUARD.wait_for_headroom``, the 85% cap),
runs ``--fit-s1``, and optionally waits again and runs the outer ``--fit``.

``EvalBatchTelemetry`` splits each candidate's latency into executor queue wait, CPU-guard wait,
fit-s1 and outer fit. It keeps the subprocess return code, so timeouts (``rc == -1``) and
nonzero exits are counted separately; the evaluator's own result dicts fold both into
``status="failed"``. It also reports throughput and child CPU per candidate from
``getrusage(RUSAGE_CHILDREN)``.

Workers write into preallocated per-slot arrays, each slot written only by its own task, so no
lock is needed. ``batch_end`` builds one record per batch, off the evaluation path.
Use one instance per concurrently running batch stream. A ``batch_begin`` while a batch is still
active shares the per-slot arrays and the process-wide RUSAGE_CHILDREN counter, so that record is
marked ``overlapping_batches`` and its slot timings and CPU figures are unreliable.
"""

from __future__ import annotations

import resource
import time
from array import array
from collections.abc import Callable
from typing import Any

RECORD_KIND = "jope.telemetry.eval-batch/v1"
MAX_SLOTS = 1024
# Latency histogram: bucket k counts latencies in [2^k, 2^(k+1)) ms, k = 0..15; bucket 0 also
# takes anything under 1 ms, and the last bucket takes everything from 32.8 s up.
HIST_BUCKETS = 16

Emit = Callable[[str, dict[str, Any]], Any]


def _bucket(ms: float) -> int:
    if ms < 2.0:
        return 0
    return min(int(ms).bit_length() - 1, HIST_BUCKETS - 1)


def _quantile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    h = (len(sorted_vals) - 1) * p
    lo = int(h)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


class EvalBatchTelemetry:
    def __init__(self, emit: Emit | None = None) -> None:
        self.emit = emit
        self._submit_ns = array("q", [0]) * MAX_SLOTS
        self._start_ns = array("q", [0]) * MAX_SLOTS
        self._guard_ns = array("q", [0]) * MAX_SLOTS
        self._fit_ns = array("q", [0]) * MAX_SLOTS
        self._outer_ns = array("q", [0]) * MAX_SLOTS
        self._end_ns = array("q", [0]) * MAX_SLOTS
        self._rc = array("q", [0]) * MAX_SLOTS
        self._n = 0
        self._active = 0
        self._overlapped = False
        self._kind = ""
        self._workers = (0, 0)
        self._t0 = 0
        self._ru0 = 0.0
        self.batches = 0
        self.slots_dropped = 0

    def batch_begin(self, kind: str, n: int, workers_requested: int, workers_effective: int) -> None:
        if self._active:
            self._overlapped = True
        self._active += 1
        self._kind = kind
        self._n = min(int(n), MAX_SLOTS)
        self.slots_dropped += max(0, int(n) - MAX_SLOTS)
        self._workers = (int(workers_requested), int(workers_effective))
        now = time.perf_counter_ns()
        for i in range(self._n):
            self._submit_ns[i] = now
            self._start_ns[i] = self._guard_ns[i] = self._fit_ns[i] = 0
            self._outer_ns[i] = self._end_ns[i] = 0
            self._rc[i] = 0
        self._t0 = now
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._ru0 = ru.ru_utime + ru.ru_stime

    # Worker-side calls; `i` is the task's index in the batch.
    def worker_start(self, i: int) -> None:
        if i < self._n:
            self._start_ns[i] = time.perf_counter_ns()

    def guard_wait(self, i: int, ns: int) -> None:
        if i < self._n:
            self._guard_ns[i] += ns

    def fit_done(self, i: int, ns: int, rc: int) -> None:
        if i < self._n:
            self._fit_ns[i] = ns
            self._rc[i] = rc

    def outer_done(self, i: int, ns: int) -> None:
        if i < self._n:
            self._outer_ns[i] = ns

    def worker_end(self, i: int) -> None:
        if i < self._n:
            self._end_ns[i] = time.perf_counter_ns()

    def batch_end(self) -> dict[str, Any]:
        wall_ns = time.perf_counter_ns() - self._t0
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_s = ru.ru_utime + ru.ru_stime - self._ru0
        n = self._n
        done = [i for i in range(n) if self._end_ns[i]]
        total_ms = sorted((self._end_ns[i] - self._submit_ns[i]) / 1e6 for i in done)
        queue_ms = sorted((self._start_ns[i] - self._submit_ns[i]) / 1e6 for i in done if self._start_ns[i])
        guard_ms = sorted(self._guard_ns[i] / 1e6 for i in done)
        fit_ms = sorted(self._fit_ns[i] / 1e6 for i in done if self._fit_ns[i])
        outer_ms = sorted(self._outer_ns[i] / 1e6 for i in done if self._outer_ns[i])
        hist = [0] * HIST_BUCKETS
        for ms in total_ms:
            hist[_bucket(ms)] += 1
        rcs = [self._rc[i] for i in done]
        timeouts = sum(1 for rc in rcs if rc == -1)
        nonzero = sum(1 for rc in rcs if rc not in (0, -1))
        wall_s = wall_ns / 1e9
        workers = self._workers[1] or 1

        def q(v: list[float]) -> dict[str, float | None]:
            return {
                "p50": _quantile(v, 0.5),
                "p95": _quantile(v, 0.95),
                "p99": _quantile(v, 0.99),
                "max": v[-1] if v else None,
            }

        rec: dict[str, Any] = {
            "kind": self._kind,
            "n": n,
            "completed": len(done),
            "workers_requested": self._workers[0],
            "workers_effective": self._workers[1],
            "wall_s": wall_s,
            "throughput_candidates_per_s": len(done) / wall_s if wall_s > 0 else None,
            "latency_ms": q(total_ms),
            "latency_hist_log2_ms": hist,
            "executor_queue_wait_ms": q(queue_ms),
            "cpu_guard_wait_ms": q(guard_ms),
            "cpu_guard_wait_share": (sum(guard_ms) / sum(total_ms))
            if total_ms and sum(total_ms) > 0
            else None,
            "fit_s1_ms": q(fit_ms),
            "outer_fit_ms": q(outer_ms),
            "timeouts": timeouts,
            "nonzero_rc": nonzero,
            "timeout_rate": timeouts / len(done) if done else None,
            "cpu_s_per_candidate": cpu_s / len(done) if done else None,
            "worker_efficiency": cpu_s / (wall_s * workers) if wall_s > 0 else None,
            "overlapping_batches": self._overlapped,
            "slots_dropped_total": self.slots_dropped,
        }
        self._active = max(0, self._active - 1)
        if not self._active:
            self._overlapped = False
        self.batches += 1
        if self.emit is not None:
            self.emit(RECORD_KIND, rec)
        return rec

    def rc_of(self, i: int) -> int | None:
        """The return code of task ``i`` in the last batch, for the generation recorder."""
        return self._rc[i] if i < self._n and self._end_ns[i] else None
