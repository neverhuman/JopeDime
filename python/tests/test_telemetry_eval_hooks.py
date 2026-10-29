"""Evaluator batch telemetry: latency split, rc counts, throughput (rows E5-E7)."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import time

from jope.telemetry.eval_hooks import HIST_BUCKETS, MAX_SLOTS, EvalBatchTelemetry, _bucket


def worker(tel: EvalBatchTelemetry, i: int, guard_s: float, rc_kind: str) -> int:
    tel.worker_start(i)
    t = time.perf_counter_ns()
    time.sleep(guard_s)  # stands in for GLOBAL_CPU_GUARD.wait_for_headroom
    tel.guard_wait(i, time.perf_counter_ns() - t)
    t = time.perf_counter_ns()
    if rc_kind == "timeout":
        try:
            subprocess.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
            rc = 0
        except subprocess.TimeoutExpired:
            rc = -1
    else:
        rc = subprocess.run(
            [sys.executable, "-c", f"raise SystemExit({1 if rc_kind == 'fail' else 0})"]
        ).returncode
    tel.fit_done(i, time.perf_counter_ns() - t, rc)
    tel.worker_end(i)
    return rc


def test_batch_record_splits_latency_and_counts_rc() -> None:
    recs = []
    tel = EvalBatchTelemetry(emit=lambda k, r: recs.append(r))
    kinds = ["ok", "ok", "fail", "timeout"]
    tel.batch_begin("strat300", len(kinds), workers_requested=4, workers_effective=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        rcs = list(ex.map(lambda a: worker(tel, *a), [(i, 0.02, k) for i, k in enumerate(kinds)]))
    rec = tel.batch_end()
    assert rcs[2] == 1 and rcs[3] == -1
    assert (rec["n"], rec["completed"], rec["timeouts"], rec["nonzero_rc"]) == (4, 4, 1, 1)
    assert rec["timeout_rate"] == 0.25
    assert rec["cpu_guard_wait_ms"]["p50"] >= 15.0
    assert 0.0 < rec["cpu_guard_wait_share"] < 1.0
    # Two workers for four tasks: later tasks waited in the executor queue.
    assert rec["executor_queue_wait_ms"]["max"] > 10.0
    assert sum(rec["latency_hist_log2_ms"]) == 4 and len(rec["latency_hist_log2_ms"]) == HIST_BUCKETS
    assert rec["cpu_s_per_candidate"] is not None and rec["worker_efficiency"] is not None
    assert rec["overlapping_batches"] is False
    assert recs == [rec]
    assert (tel.rc_of(3), tel.rc_of(0), tel.rc_of(99)) == (-1, 0, None)


def test_overlap_and_slot_bounds_are_reported() -> None:
    tel = EvalBatchTelemetry()
    tel.batch_begin("full", MAX_SLOTS + 5, 1, 1)
    tel.batch_begin("full", 1, 1, 1)  # a second batch in the same process
    assert tel.batch_end()["overlapping_batches"] is True
    assert tel.slots_dropped == 5
    tel.worker_start(MAX_SLOTS + 1)  # out-of-range writes are ignored, not raised


def test_bucket_edges() -> None:
    assert [_bucket(ms) for ms in (0.1, 1.9, 2.0, 3.9, 4.0, 1023.0, 1e9)] == [
        0,
        0,
        1,
        1,
        2,
        9,
        HIST_BUCKETS - 1,
    ]
