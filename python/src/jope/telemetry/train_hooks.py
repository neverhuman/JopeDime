"""Host-scalar step telemetry for the R/P/W trainers (TELEMETRY_SPEC rows T1-T4, T8, T11-T12).

The frozen contract: these trainers already materialize some per-step values on the host (for
example W's ``float(loss.detach().cpu().item())``). Telemetry reuses exactly those values. It
adds no device-to-host copy and no synchronization, and it doesn't build a Python container per
step.

``StepBatch`` keeps one run-scoped, preallocated flat ``array('d')`` of ``rows x columns``
values and one ``array('Q')`` of step numbers. ``put(col, value)`` stores one float by flat
index, so no index tuple is built; ``commit(step)`` closes the row. Every ``rows`` steps the
full block goes to ``submit(steps_view, values_view, rows)`` in one call. The two memoryviews
are created once at construction and reused. A value that was never put in a row stays NaN, so a
missing value is never reported as zero.

``StepTimer`` gives host wall-clock spans from ``perf_counter_ns``. These are host-side spans
only; they are never GPU execution time.
"""

from __future__ import annotations

import struct
import time
from array import array
from collections.abc import Callable, Sequence

# Submit callback: (steps, values, rows). Normally the PyO3 handle's block submit; any callable
# taking two memoryviews and a row count works (tests, JSONL fallbacks).
Submit = Callable[[memoryview, memoryview, int], object]

MISSING_VALUE_BITS = 0x7FF84A4F50455445
NAN = struct.unpack(">d", MISSING_VALUE_BITS.to_bytes(8, "big"))[0]

# Frozen memory contract (TELEMETRY_SPEC, Final architecture): a batch is at most 4,096 float64
# columns x 32 rows (1 MiB) plus 32 u64 step numbers. Larger layouts fail at construction.
MAX_COLUMNS = 4096
MAX_ROWS = 32

# Column layout the flywheel trainer (W, `train_flywheel_model_for_round`) fills. A component
# column is named `<metric>/<group>`. Spans are host wall-clock on the trainer thread; the `fwd`
# span includes W's existing loss read-back.
W_STEP_COLUMNS = (
    "train.loss",
    "train.loss.component/dir_margin",
    "train.loss.component/evo_val",
    "optim.lr.group",
    "step.span.data",
    "step.span.h2d",
    "step.span.fwd",
    "step.span.bwd",
    "step.span.opt",
    "step.wall",
    "step.unattributed",
)


class StepBatch:
    """Fixed-layout batch of per-step host scalars.

    ``columns`` names the metrics in column order, for example
    ``("train.loss", "optim.lr.group", "step.wall")``. The layout is fixed for the run and is
    registered once with the recorder; ``col(name)`` returns a column index to use with ``put``.
    """

    def __init__(self, columns: Sequence[str], submit: Submit, rows: int = 32) -> None:
        if not columns:
            raise ValueError("StepBatch needs at least one column")
        if len(columns) > MAX_COLUMNS:
            raise ValueError(f"StepBatch has {len(columns)} columns; the frozen maximum is {MAX_COLUMNS}")
        if len(set(columns)) != len(columns):
            raise ValueError("StepBatch columns must be unique")
        self.columns = tuple(columns)
        self.width = len(self.columns)
        self.rows = int(rows)
        if not 1 <= self.rows <= MAX_ROWS:
            raise ValueError(f"rows must be in 1..={MAX_ROWS}, got {self.rows}")
        self._submit = submit
        self._values = array("d", [NAN]) * (self.rows * self.width)
        self._steps = array("Q", [0]) * self.rows
        self._values_view = memoryview(self._values)
        self._steps_view = memoryview(self._steps)
        self._row = 0
        self._base = 0
        self.submitted_rows = 0
        self.submit_errors = 0

    def col(self, name: str) -> int:
        return self.columns.index(name)

    def put(self, col: int, value: float) -> None:
        """Store one already-materialized host float in the current row."""
        self._values[self._base + col] = value

    def commit(self, step: int) -> None:
        """Close the current row as optimizer step ``step``; submit when the block is full."""
        self._steps[self._row] = step
        self._row += 1
        if self._row == self.rows:
            self._flush(self.rows)
        else:
            self._base += self.width

    def flush(self) -> None:
        """Submit a partial block (epoch end, shutdown). Never called per step."""
        if self._row:
            self._flush(self._row)

    def _flush(self, rows: int) -> None:
        try:
            self._submit(self._steps_view, self._values_view, rows)
            self.submitted_rows += rows
        except Exception:
            # The submit path counts its own drops; a failing sink must never stop training.
            self.submit_errors += 1
        for i in range(rows * self.width):
            self._values[i] = NAN
        self._row = 0
        self._base = 0


class StepTimer:
    """Host wall-clock spans for one step, in milliseconds, written into a ``StepBatch``.

    ``begin()`` at the start of a step, ``mark(col)`` at the end of each stage (the span runs
    from the previous mark), and ``end(wall_col, unattributed_col)`` at the end of the step.
    Stage spans are exclusive and on the host basis, so ``unattributed`` = wall - sum of spans.
    """

    def __init__(self, batch: StepBatch) -> None:
        self.batch = batch
        self._t0 = 0
        self._last = 0
        self._spans_ns = 0

    def begin(self) -> None:
        self._t0 = self._last = time.perf_counter_ns()
        self._spans_ns = 0

    def mark(self, col: int) -> float:
        now = time.perf_counter_ns()
        d = now - self._last
        self._last = now
        self._spans_ns += d
        self.batch.put(col, d / 1e6)
        return d / 1e6

    def end(self, wall_col: int, unattributed_col: int | None = None) -> None:
        wall = time.perf_counter_ns() - self._t0
        self.batch.put(wall_col, wall / 1e6)
        if unattributed_col is not None:
            self.batch.put(unattributed_col, (wall - self._spans_ns) / 1e6)
