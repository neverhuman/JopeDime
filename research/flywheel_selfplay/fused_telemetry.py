"""Optional native telemetry for the cached-device fused research worker.

Importing this module never imports jope or opens a sink. Enabled callers must
have the owner-requested fixed native wheel. Rates describe actual host payload
through completed batch preparation (including device gathers), not PCIe bandwidth.
"""

from __future__ import annotations

import os
from pathlib import Path

_active = None


def open_round(round_idx, seed, metadata):
    global _active
    root = os.environ.get("JOPE_TELEMETRY_RUN_DIR")
    if not root:
        return None
    from jope.telemetry import Recorder, metric_registry

    if metric_registry is None:
        raise RuntimeError("jope telemetry native wheel is unavailable")
    registry = {name: mid for mid, name, *_ in metric_registry()}
    # Refuse the known broken wheel before opening a sink/thread.
    if "io.batch_gb_s" not in registry:
        raise RuntimeError("fixed jope._telemetry wheel missing io.batch_gb_s")
    if _active is not None:
        raise RuntimeError("previous round telemetry is still open")
    path = Path(root) / f"round-{round_idx:06d}-seed-{seed}"
    rec = Recorder(
        str(path),
        ["fused-flywheel", "host-probe"],
        {
            **metadata,
            "round": round_idx,
            "seed": seed,
            "io_rate_scope": "actual host payload / (slate assembly + completed device packing); includes GPU gather, not pure H2D bandwidth",
            "tokens_scope": "tabular input has no text-token path; GP DSL counts remain separate in round receipt",
            "overhead_qualified": False,
        },
    )
    _active = (rec, registry, path)
    return _active


def emit_step(
    session, *, step, loss, lr, grad_norm, samples, wall_s, host_bytes, delivery_s
):
    if session is None:
        return
    rec, reg, _ = session
    values = {
        "train.loss": loss,
        "optim.lr.group": lr,
        "optim.grad_norm.global": grad_norm,
        "step.wall": wall_s * 1000,
        "throughput.samples_per_s": samples / wall_s,
        "io.batch.bytes": host_bytes,
        "io.batch_bytes_per_s": host_bytes / delivery_s,
        "io.batch_gb_s": host_bytes / delivery_s / 1e9,
        "io.samples_per_s": samples / delivery_s,
    }
    for name, value in values.items():
        rec.emit_scalar(0, reg[name], 0, step, value)
    rec.emit_unavailable(0, reg["throughput.tokens_per_s"], 0, step, 10)
    rec.emit_unavailable(0, reg["io.h2d_gb_s"], 0, step, 12)


def finish_round(row):
    global _active
    if _active is None:
        return None
    rec, reg, path = _active
    # Caller has just executed the actual frozen-DOPE guard.
    rec.emit_scalar(0, reg["dope.freeze.equal"], 0, row["round"], 1.0)
    rec.emit_scalar(
        0, reg["params.count.trainable"], 0, row["round"], row["trainable_params"]
    )
    rec.emit(
        "jope.telemetry.fused-round/v1",
        {
            k: row[k]
            for k in (
                "round",
                "dope_sha256",
                "phase_wall_seconds",
                "promotions",
                "wall_seconds",
            )
        },
    )
    try:
        result = rec.close()
    finally:
        _active = None
    return {
        "run_dir": str(path),
        **{
            k: result[k]
            for k in ("status", "event_drops", "record_drops", "sink_errors")
            if k in result
        },
    }


def interrupt_active():
    global _active
    if _active is not None:
        rec, _, _ = _active
        _active = None
        rec.interrupt()
