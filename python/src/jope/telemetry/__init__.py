"""Python side of `crates/jope-telemetry`: guards and hooks that existing trainers call.

The hot path lives in Rust behind the optional PyO3 extension. These modules only collect
values the trainers already have, enforce the fail-closed guards, and hand records to an
``emit(kind, record)`` callable (the extension's record path, or a test sink).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from jope._telemetry import GpSession, metric_registry
    from jope._telemetry import Recorder as _NativeRecorder
except ImportError:  # source-only installs can still use the guards and test sinks
    _NativeRecorder = None
    GpSession = None
    metric_registry = None


class Recorder:
    """Thin Python adapter around the optional Rust recorder.

    Producer indices are fixed by ``producer_names`` at construction. ``submit`` accepts the
    reusable memoryviews from :class:`StepBatch`; it does not build a Python list per step.
    """

    def __init__(
        self,
        run_dir: str,
        producer_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if _NativeRecorder is None:
            raise RuntimeError("jope telemetry extension is unavailable; install the maturin package")
        self._native = _NativeRecorder(
            run_dir,
            list(producer_names),
            json.dumps(dict(metadata or {}), allow_nan=False, separators=(",", ":")),
        )

    def set_metric_layout(self, source: int, columns: Sequence[tuple[int, int]]) -> None:
        self._native.set_metric_layout(source, list(columns))

    def submit(self, source: int, steps: memoryview, values: memoryview, rows: int) -> int:
        return self._native.submit(source, steps, values, rows)

    def emit_scalar(
        self, source: int, metric_id: int, group_id: int, step: int, value: float, sampled: bool = True
    ) -> bool:
        return self._native.emit_scalar(source, metric_id, group_id, step, value, sampled)

    def emit_unavailable(
        self, source: int, metric_id: int, group_id: int, step: int, reason_id: int, sampled: bool = True
    ) -> bool:
        return self._native.emit_unavailable(source, metric_id, group_id, step, reason_id, sampled)

    def emit(self, kind: str, record: Mapping[str, Any]) -> bool:
        body = json.dumps(record, allow_nan=False, separators=(",", ":"))
        return self._native.record_json(kind, body)

    def close(self) -> dict[str, Any]:
        return json.loads(self._native.close())

    def interrupt(self) -> dict[str, Any]:
        """Finalize an incomplete run as interrupted after flushing its producers."""
        return json.loads(self._native.interrupt())


__all__ = ["GpSession", "Recorder", "metric_registry"]
