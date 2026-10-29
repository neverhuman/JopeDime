"""CUDA copy events for one H2D source of a Rust `Recorder` (torch shim, rows I4/I9).

Only what must touch torch lives here: creating events, recording them on the trainer's
current stream, querying them without synchronizing, and reading `elapsed_time`. The Rust
recorder owns the sampling stride, the slots, the dropped-sample counts and every value it
writes (`io.h2d.bytes`, `io.h2d`, `io.h2d_gb_s`, `io.batch_gb_s`).
"""

from __future__ import annotations

from typing import Any


class CudaH2D:
    def __init__(self, torch_module: Any, device: str, recorder: Any, source: int, slots: int = 4) -> None:
        self.rec, self.src = recorder, source
        self.events: list = []
        self.stream = None
        if recorder.h2d_enabled(source):
            try:
                with torch_module.cuda.device(device):
                    self.stream = torch_module.cuda.current_stream(device)
                    self.events = [
                        (
                            torch_module.cuda.Event(enable_timing=True),
                            torch_module.cuda.Event(enable_timing=True),
                        )
                        for _ in range(slots)
                    ]
            except Exception:
                recorder.h2d_unavailable(source)

    def will_sample(self, step: int) -> bool:
        return self.rec.h2d_will_sample(self.src, step)

    def _poll(self) -> None:
        for slot in self.rec.h2d_pending(self.src):
            try:
                start, end = self.events[slot]
                if end.query():
                    self.rec.h2d_complete(self.src, slot, start.elapsed_time(end))
            except Exception:
                self.rec.h2d_fail(self.src, slot)

    def begin(self, step: int, size_bytes: int, data_wait_ms: float | None = None) -> int:
        if not self.rec.h2d_enabled(self.src):
            return -1
        self._poll()
        slot = self.rec.h2d_begin(self.src, step, size_bytes, data_wait_ms)
        if slot >= 0:
            try:
                self.events[slot][0].record(self.stream)
            except Exception:
                self.rec.h2d_fail(self.src)
                return -1
        return slot

    def end(self, slot: int) -> None:
        if slot >= 0:
            try:
                self.events[slot][1].record(self.stream)
                self.rec.h2d_recorded(self.src, slot)
            except Exception:
                self.rec.h2d_fail(self.src, slot)

    def flush(self, synchronize: bool = True) -> bool:
        """At shutdown only: one stream synchronization makes outstanding samples readable."""
        sync_attempted = False
        if synchronize and self.rec.h2d_enabled(self.src):
            sync_attempted = True
            try:
                self.stream.synchronize()
            except Exception:
                self.rec.h2d_abandon(self.src)
        if self.rec.h2d_enabled(self.src):
            self._poll()
        return sync_attempted
