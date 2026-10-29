"""Low-overhead IO instrumentation for the flywheel's resident-NumPy input path."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .train_hooks import NAN, StepBatch

IO_COLUMNS = (
    "io.batch_prep",
    "io.queue.depth",
    "io.queue.wait",
    "io.queue.fallback",
    "io.data_wait",
    "io.batches_per_s",
    "io.samples_per_s",
    "io.batch.bytes",
    "io.batch_bytes_per_s",
    "io.batch_gb_s",
)
H2D_COLUMNS = ("io.h2d.bytes", "io.h2d", "io.h2d_gb_s")
ARRAY_KEYS = (
    "ctx_b", "prog_b", "proj_b", "prior_b", "scale_b", "sos_b", "target_f_b",
    "gate_b", "rev_idx_b", "rev_f_b", "rev_gate_b",
)


class LocalReadObserver:
    """Aggregate logical file sizes and wall time around actual local reads."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.bytes = 0
        self.read_ms = 0.0
        self.read_operation_ms = 0.0
        self.reads = 0
        self._first_read_ns: int | None = None
        self._last_read_ns: int | None = None
        self._lock = threading.Lock()

    def _record_read(self, size: int, started: int, finished: int) -> None:
        with self._lock:
            self.bytes += size
            self.read_operation_ms += (finished - started) / 1e6
            self._first_read_ns = started if self._first_read_ns is None else min(self._first_read_ns, started)
            self._last_read_ns = finished if self._last_read_ns is None else max(self._last_read_ns, finished)
            self.read_ms = (self._last_read_ns - self._first_read_ns) / 1e6
            self.reads += 1

    def read(self, path: str | Path, operation: Any) -> Any:
        if not self.enabled:
            return operation()
        file_path = Path(path)
        size = file_path.stat().st_size
        started = time.perf_counter_ns()
        value = operation()
        self._record_read(size, started, time.perf_counter_ns())
        return value

    @contextmanager
    def open_npz(self, path: str | Path):
        if not self.enabled:
            with np.load(path, allow_pickle=False) as archive:
                yield archive
            return
        with self._lock:
            size = Path(path).stat().st_size
        started = time.perf_counter_ns()
        try:
            with np.load(path, allow_pickle=False) as archive:
                yield archive
        finally:
            self._record_read(size, started, time.perf_counter_ns())

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return self._snapshot_locked()

    def take_snapshot(self) -> dict[str, int | float]:
        with self._lock:
            result = self._snapshot_locked()
            self.bytes = 0
            self.read_ms = 0.0
            self.read_operation_ms = 0.0
            self.reads = 0
            self._first_read_ns = None
            self._last_read_ns = None
            return result

    def _snapshot_locked(self) -> dict[str, int | float]:
        return {
            "enabled": self.enabled,
            "logical_bytes": self.bytes,
            "read_ms": self.read_ms,
            "read_operation_ms": self.read_operation_ms,
            "reads": self.reads,
        }


class BatchIOStats:
    """Writes data-supply counters into the trainer's existing fixed-layout row."""

    def __init__(self, batch: StepBatch, columns: dict[str, int]) -> None:
        self.batch = batch
        self.columns = columns
        self._last_at = time.perf_counter()

    def record(self, step: int, batch: dict, prep_ms: float, queue: Any, data_wait_ms: float) -> int:
        arrays_bytes = 0
        for key in ARRAY_KEYS:
            arrays_bytes += batch[key].nbytes
        now = time.perf_counter()
        elapsed = max(now - self._last_at, 1e-9)
        self._last_at = now
        n = int(batch["ctx_b"].shape[0])
        queue_depth = getattr(queue, "last_queue_depth", 0) if queue is not None else 0
        queue_wait_ms = getattr(queue, "last_queue_wait_ms", 0.0) if queue is not None else 0.0
        fallback = getattr(queue, "last_pop_fallback", 0) if queue is not None else 0
        self.batch.put(self.columns["io.batch_prep"], prep_ms)
        self.batch.put(self.columns["io.queue.depth"], float(queue_depth))
        self.batch.put(self.columns["io.queue.wait"], float(queue_wait_ms))
        self.batch.put(self.columns["io.queue.fallback"], float(fallback))
        self.batch.put(self.columns["io.data_wait"], data_wait_ms)
        self.batch.put(self.columns["io.batches_per_s"], 1.0 / elapsed)
        self.batch.put(self.columns["io.samples_per_s"], n / elapsed)
        self.batch.put(self.columns["io.batch.bytes"], float(arrays_bytes))
        self.batch.put(self.columns["io.batch_bytes_per_s"], arrays_bytes / elapsed)
        self.batch.put(self.columns["io.batch_gb_s"], arrays_bytes / (elapsed * 1e9))
        return arrays_bytes


class H2DEventSampler:
    """Sample CUDA copy events without synchronizing; completed events are read lazily."""

    def __init__(self, torch_module: Any, device: str, batch: StepBatch, stride: int = 64, slots: int = 4) -> None:
        self.batch = batch
        self.stride = max(1, int(stride))
        self.events = []
        self.pending = [False] * slots
        self.steps = [0] * slots
        self.byte_counts = [0] * slots
        self.next_slot = 0
        self.dropped_samples = 0
        self.enabled = str(device).startswith("cuda") and torch_module.cuda.is_available()
        self.stream = None
        if self.enabled:
            try:
                with torch_module.cuda.device(device):
                    self.stream = torch_module.cuda.current_stream(device)
                    self.events = [
                        (torch_module.cuda.Event(enable_timing=True), torch_module.cuda.Event(enable_timing=True))
                        for _ in range(slots)
                    ]
            except Exception:
                self.enabled = False
                self.events = []

    def _poll(self) -> None:
        for slot, busy in enumerate(self.pending):
            if not busy:
                continue
            try:
                if not self.events[slot][1].query():
                    continue
                start, end = self.events[slot]
                elapsed_ms = start.elapsed_time(end)
                size = self.byte_counts[slot]
                self.batch.put(0, float(size))
                self.batch.put(1, elapsed_ms)
                self.batch.put(2, size / (elapsed_ms * 1e6) if elapsed_ms > 0 else NAN)
                self.batch.commit(self.steps[slot])
                self.pending[slot] = False
            except Exception:
                self.dropped_samples += 1
                self.pending[slot] = False
                self.enabled = False

    def begin(self, step: int, size_bytes: int) -> int:
        if not self.enabled:
            return -1
        self._poll()
        if not self.enabled:
            return -1
        if step % self.stride:
            return -1
        for offset in range(len(self.pending)):
            slot = (self.next_slot + offset) % len(self.pending)
            if not self.pending[slot]:
                self.next_slot = (slot + 1) % len(self.pending)
                self.steps[slot] = step
                self.byte_counts[slot] = size_bytes
                try:
                    self.events[slot][0].record(self.stream)
                except Exception:
                    self.dropped_samples += 1
                    self.enabled = False
                    return -1
                return slot
        self.dropped_samples += 1
        return -1

    def will_sample(self, step: int) -> bool:
        return self.enabled and step % self.stride == 0

    def end(self, slot: int) -> None:
        if slot >= 0:
            try:
                self.events[slot][1].record(self.stream)
                self.pending[slot] = True
            except Exception:
                self.dropped_samples += 1
                self.enabled = False
                self.pending[slot] = False

    def flush(self, synchronize: bool = True) -> bool:
        # At shutdown only, a single stream synchronization makes outstanding event rows readable.
        sync_attempted = False
        if self.enabled and synchronize:
            sync_attempted = True
            try:
                self.stream.synchronize()
            except Exception:
                self.dropped_samples += sum(self.pending)
                self.pending = [False] * len(self.pending)
                self.enabled = False
        if self.enabled:
            self._poll()
        self.batch.flush()
        return sync_attempted
