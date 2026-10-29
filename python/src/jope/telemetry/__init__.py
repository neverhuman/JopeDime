"""Re-exports of `jope._telemetry`, the Rust telemetry extension (`crates/jope-telemetry`).

There is no telemetry logic in Python. Two torch shims remain, because they must touch torch
objects: `cuda_events` records and queries CUDA copy events on the trainer's stream, and
`dope_guard` walks the fused model's DOPE parameters. Everything they observe goes to Rust.
"""

try:
    from jope._telemetry import (
        DOPE_NORMALIZER_BUFFERS,
        DOPE_SUBMODULE,
        DopeFreezeViolationError,
        DopeGuard,
        EvalBatch,
        GenerationRecorder,
        GpSession,
        LocalReads,
        Recorder,
        RecordSink,
        SealedPartitionError,
        ValidationRecorder,
        check_partition_label,
        metric_id,
        metric_registry,
        open_flywheel_step,
        open_python_trainer,
    )
except ImportError as error:  # telemetry was requested but the maturin extension is missing
    raise RuntimeError(
        "jope telemetry PyO3 extension is unavailable; build it with maturin (TELEMETRY_ENABLE.md)"
    ) from error

__all__ = [
    "DOPE_NORMALIZER_BUFFERS",
    "DOPE_SUBMODULE",
    "DopeFreezeViolationError",
    "DopeGuard",
    "EvalBatch",
    "GenerationRecorder",
    "GpSession",
    "LocalReads",
    "RecordSink",
    "Recorder",
    "SealedPartitionError",
    "ValidationRecorder",
    "check_partition_label",
    "metric_id",
    "metric_registry",
    "open_flywheel_step",
    "open_python_trainer",
]
