from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np

from jope.telemetry.io_hooks import (
    END_TO_END_COLUMN,
    H2D_COLUMNS,
    IO_COLUMNS,
    BatchIOStats,
    H2DEventSampler,
    LocalReadObserver,
)
from jope.telemetry.train_hooks import StepBatch


def test_local_read_observer_counts_logical_file_bytes_and_npz_reads(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, values=np.arange(8, dtype=np.float32))
    observer = LocalReadObserver()
    with observer.open_npz(path) as archive:
        assert archive["values"].shape == (8,)
    snapshot = observer.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["logical_bytes"] == path.stat().st_size
    assert snapshot["reads"] == 1
    assert snapshot["read_ms"] >= 0


def test_disabled_read_observer_does_not_stat_path():
    observer = LocalReadObserver(enabled=False)
    assert observer.read("/missing", lambda: "loaded") == "loaded"
    assert observer.snapshot() == {
        "enabled": False,
        "logical_bytes": 0,
        "read_ms": 0.0,
        "read_operation_ms": 0.0,
        "reads": 0,
    }


def test_batch_io_stats_write_batch_and_queue_values():
    submitted = []
    batch = StepBatch(
        IO_COLUMNS,
        lambda steps, values, rows: submitted.append(
            (list(steps[:rows]), list(values[: rows * len(IO_COLUMNS)]), rows)
        ),
        rows=1,
    )
    columns = {name: index for index, name in enumerate(IO_COLUMNS)}
    stats = BatchIOStats(batch, columns)
    arrays = {
        key: np.zeros((2, 4), dtype=np.float32)
        for key in (
            "ctx_b",
            "prog_b",
            "proj_b",
            "prior_b",
            "scale_b",
            "sos_b",
            "target_f_b",
            "gate_b",
            "rev_idx_b",
            "rev_f_b",
            "rev_gate_b",
        )
    }

    class Queue:
        last_queue_depth = 3
        last_queue_wait_ms = 0.25
        last_pop_fallback = 1

    size = stats.record(7, arrays, 1.5, Queue(), 2.5)
    batch.commit(7)
    assert size == sum(array.nbytes for array in arrays.values())
    steps, values, rows = submitted[0]
    assert (steps, rows) == ([7], 1)
    assert values[columns["io.batch_prep"]] == 1.5
    assert values[columns["io.queue.depth"]] == 3.0
    assert values[columns["io.queue.wait"]] == 0.25
    assert values[columns["io.queue.fallback"]] == 1.0
    assert values[columns["io.data_wait"]] == 2.5
    assert values[columns["io.batch.bytes"]] == float(size)


def test_batch_io_stats_leaves_queue_metrics_missing_when_queue_is_disabled():
    submitted = []
    batch = StepBatch(
        IO_COLUMNS,
        lambda steps, values, rows: submitted.append(list(values[: rows * len(IO_COLUMNS)])),
        rows=1,
    )
    columns = {name: index for index, name in enumerate(IO_COLUMNS)}
    stats = BatchIOStats(batch, columns)
    arrays = {
        key: np.zeros((1, 1), dtype=np.float32)
        for key in (
            "ctx_b",
            "prog_b",
            "proj_b",
            "prior_b",
            "scale_b",
            "sos_b",
            "target_f_b",
            "gate_b",
            "rev_idx_b",
            "rev_f_b",
            "rev_gate_b",
        )
    }

    stats.record(1, arrays, 0.5, None, 1.0)
    batch.commit(1)

    values = submitted[0]
    assert all(
        math.isnan(values[columns[name]])
        for name in (
            "io.queue.depth",
            "io.queue.wait",
            "io.queue.fallback",
        )
    )


def test_h2d_sampler_is_inert_without_cuda():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    batch = StepBatch(("io.h2d.bytes", "io.h2d", "io.h2d_gb_s"), lambda *_: None)
    sampler = H2DEventSampler(Torch, "cpu", batch)
    assert sampler.enabled is False
    assert sampler.will_sample(64) is False
    assert sampler.begin(64, 1024) == -1
    sampler.end(-1)
    sampler.flush()


def test_end_to_end_gb_s_uses_data_wait_plus_completed_h2d_span():
    submitted = []

    class Event:
        def record(self, _stream):
            pass

        def query(self):
            return True

        def elapsed_time(self, _end):
            return 2.0

    class Stream:
        def synchronize(self):
            pass

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device(_device):
            return nullcontext()

        @staticmethod
        def current_stream(_device):
            return Stream()

        @staticmethod
        def Event(enable_timing=True):
            assert enable_timing is True
            return Event()

    class Torch:
        cuda = Cuda()

    columns = (*H2D_COLUMNS, END_TO_END_COLUMN)
    batch = StepBatch(
        columns,
        lambda steps, values, rows: submitted.append(
            (list(steps[:rows]), list(values[: rows * len(columns)]), rows)
        ),
    )
    sampler = H2DEventSampler(Torch, "cuda:0", batch, stride=1, slots=1, end_to_end_col=3)
    slot = sampler.begin(7, 1_000_000, data_wait_ms=10.0)
    sampler.end(slot)
    sampler.flush(synchronize=False)

    steps, values, rows = submitted[0]
    assert (steps, rows) == ([7], 1)
    assert values[2] == 0.5
    assert values[3] == 1_000_000 / (12.0 * 1e6)
