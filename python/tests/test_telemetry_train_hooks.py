"""Host-scalar step batch: layout, NaN for missing values, block submission, failing sinks."""

from __future__ import annotations

import math
import struct
import tracemalloc

import pytest

from jope.telemetry.train_hooks import MAX_COLUMNS, MAX_ROWS, W_STEP_COLUMNS, StepBatch, StepTimer


class Sink:
    def __init__(self) -> None:
        self.blocks: list[tuple[list[int], list[float], int]] = []

    def __call__(self, steps: memoryview, values: memoryview, rows: int) -> None:
        self.blocks.append((list(steps[:rows]), list(values), rows))


def test_rows_submit_in_blocks_and_missing_values_stay_nan() -> None:
    sink = Sink()
    b = StepBatch(("train.loss", "optim.lr.group"), sink, rows=2)
    loss, lr = b.col("train.loss"), b.col("optim.lr.group")
    b.put(loss, 1.5)
    b.put(lr, 1e-3)
    b.commit(10)
    b.put(loss, 1.25)  # lr not reported on this step
    b.commit(11)
    assert len(sink.blocks) == 1
    steps, values, rows = sink.blocks[0]
    assert (steps, rows) == ([10, 11], 2)
    assert values[:3] == [1.5, 1e-3, 1.25] and math.isnan(values[3])
    b.put(loss, 1.0)
    b.commit(12)
    b.flush()
    steps, values, rows = sink.blocks[1]
    assert (steps, rows, values[0]) == ([12], 1, 1.0)
    assert math.isnan(values[2]), "the buffer was reset after the first block"
    assert b.submitted_rows == 3


def test_steps_above_float32_precision_are_exact() -> None:
    sink = Sink()
    b = StepBatch(("train.loss",), sink, rows=1)
    big = (1 << 24) + 1
    b.put(0, 0.5)
    b.commit(big)
    assert sink.blocks[0][0] == [big]


def test_missing_value_uses_the_rust_batch_sentinel_bits() -> None:
    sink = Sink()
    b = StepBatch(("train.loss",), sink, rows=1)
    b.commit(0)
    assert struct.pack(">d", sink.blocks[0][1][0]).hex() == "7ff84a4f50455445"


def test_failing_sink_is_counted_not_raised() -> None:
    def boom(*_):
        raise RuntimeError("sink down")

    b = StepBatch(("train.loss",), boom, rows=1)
    b.put(0, 1.0)
    b.commit(0)
    assert (b.submit_errors, b.submitted_rows) == (1, 0)


def _module_growth(steps: int) -> int:
    b = StepBatch(("train.loss", "step.wall"), lambda s, v, r: None, rows=MAX_ROWS)
    for step in range(256):  # warm up through several flushes
        b.put(0, 1.0)
        b.put(1, 2.0)
        b.commit(step)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for step in range(256, 256 + steps):
        b.put(0, 1.0)
        b.put(1, 2.0)
        b.commit(step)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return sum(
        s.size_diff
        for s in after.compare_to(before, "filename")
        if s.traceback[0].filename.endswith("jope/telemetry/train_hooks.py")
    )


def test_steady_state_put_commit_does_not_grow_with_steps() -> None:
    # At most a constant (the replaced `submitted_rows` int), never per-step growth.
    small, large = _module_growth(6_400), _module_growth(64_000)
    assert small <= 64 and large <= 64, (small, large)


def test_timer_spans_are_exclusive_and_unattributed_is_the_rest() -> None:
    sink = Sink()
    b = StepBatch(("step.span.fwd", "step.span.bwd", "step.wall", "step.unattributed"), sink, rows=1)
    t = StepTimer(b)
    t.begin()
    t.mark(0)
    t.mark(1)
    t.end(2, 3)
    b.commit(0)
    _, (fwd, bwd, wall, un), _ = sink.blocks[0]
    assert wall >= fwd + bwd and un == pytest.approx(wall - fwd - bwd, abs=1e-6)


def test_frozen_memory_bounds_are_enforced_at_construction() -> None:
    cols = [f"c{i}" for i in range(MAX_COLUMNS)]
    b = StepBatch(cols, lambda *a: None, rows=MAX_ROWS)
    assert len(b._values) == MAX_COLUMNS * MAX_ROWS and b._values.itemsize == 8
    assert b._values.itemsize * len(b._values) == 1 << 20, "1 MiB of float64"
    with pytest.raises(ValueError, match="columns"):
        StepBatch([*cols, "one-more"], lambda *a: None)
    with pytest.raises(ValueError, match="rows"):
        StepBatch(("a",), lambda *a: None, rows=MAX_ROWS + 1)
    with pytest.raises(ValueError, match="rows"):
        StepBatch(("a",), lambda *a: None, rows=0)


def test_layout_validation() -> None:
    with pytest.raises(ValueError):
        StepBatch((), lambda *a: None)
    with pytest.raises(ValueError):
        StepBatch(("a", "a"), lambda *a: None)


def test_w_shaped_loop_records_existing_host_values_only() -> None:
    """The calls the W instrumentation patch makes, on a tiny CPU model."""
    import torch

    torch.manual_seed(0)
    sink = Sink()
    tel = StepBatch(W_STEP_COLUMNS, sink, rows=4)
    col = {name: tel.col(name) for name in W_STEP_COLUMNS}
    timer = StepTimer(tel)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6, eta_min=1e-4)
    losses, dir_margin = [], []
    for step in range(1, 7):
        timer.begin()
        x = torch.randn(8, 3)
        timer.mark(col["step.span.data"])
        timer.mark(col["step.span.h2d"])
        loss = model(x).pow(2).mean()
        evo = step % 2 == 0
        if evo:
            dir_margin.append(0.1 * step)
        losses.append(float(loss.detach().cpu().item()))  # W already does this read-back
        timer.mark(col["step.span.fwd"])
        opt.zero_grad()
        loss.backward()
        timer.mark(col["step.span.bwd"])
        opt.step()
        tel.put(col["optim.lr.group"], opt.param_groups[0]["lr"])
        sched.step()
        timer.mark(col["step.span.opt"])
        tel.put(col["train.loss"], losses[-1])
        if evo:
            tel.put(col["train.loss.component/dir_margin"], dir_margin[-1])
        timer.end(col["step.wall"], col["step.unattributed"])
        tel.commit(step)
    tel.flush()
    rows = [
        (s, v[i * tel.width : (i + 1) * tel.width])
        for steps, v, n in sink.blocks
        for i, s in enumerate(steps[:n])
    ]
    assert [s for s, _ in rows] == [1, 2, 3, 4, 5, 6]
    assert [r[col["train.loss"]] for _, r in rows] == losses
    assert math.isnan(rows[0][1][col["train.loss.component/dir_margin"]])
    assert rows[1][1][col["train.loss.component/dir_margin"]] == pytest.approx(0.2)
    assert rows[0][1][col["optim.lr.group"]] == pytest.approx(1e-3), "LR in effect for step 1"
    assert all(math.isnan(r[col["train.loss.component/evo_val"]]) for _, r in rows), "never put, so NaN"
