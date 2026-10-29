"""Smoke test of the Rust telemetry extension through the one-line trainer call sites.

The logic and its unit tests live in `crates/jope-telemetry` (`cargo test -p jope-telemetry`).
This only checks that the built `jope._telemetry` module is importable and that the calls the
trainers make produce a complete run. It is skipped where the extension isn't installed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

tel = pytest.importorskip("jope._telemetry")


def _events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]


def _ids() -> dict[str, int]:
    return {name: mid for mid, name, *_ in tel.metric_registry()}


def test_registry_keeps_frozen_names() -> None:
    ids = _ids()
    assert len(ids) == 86
    assert ids["io.batch_gb_s"] == 0x041B
    assert tel.metric_id("throughput.tokens_per_s") == 0x0119


def test_python_trainer_step_rows(tmp_path: Path) -> None:
    rec = tel.open_python_trainer(tmp_path / "p", ["rank", "huber"], {"trainer": "smoke"})
    for step in range(1, 4):
        rec.begin()
        rec.mark("step.span.data")
        rec.put("train.loss", 1.0 / step)
        rec.put("train.loss.component/huber", 0.5)
        rec.put("optim.lr.group", 1e-3)
        rec.end()
        rec.commit(step)
    manifest = rec.close()
    assert manifest["status"] == "complete"
    assert manifest["event_drops"] == 0
    ids = _ids()
    losses = [
        e["event"]
        for e in _events(tmp_path / "p")
        if e["kind"] == "event" and e["event"]["metric_id"] == ids["train.loss"]
    ]
    assert [e["step"] for e in losses] == [1, 2, 3]
    with pytest.raises(ValueError):
        tel.open_python_trainer(tmp_path / "q", ["rank"]).put("no.such.column", 1.0)


class _NoCuda:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


def test_flywheel_round_on_cpu(tmp_path: Path) -> None:
    from jope.telemetry.cuda_events import CudaH2D

    reads = tel.LocalReads()
    npz = tmp_path / "a.npz"
    np.savez(npz, x=np.zeros(4))
    with reads.open(npz, lambda: np.load(npz, allow_pickle=False)) as archive:
        assert archive["x"].shape == (4,)
    rec = tel.open_flywheel_step(tmp_path, 1, 7, 8, "cpu", False, reads)
    main, evo = CudaH2D(_NoCuda, "cpu", rec, 1), CudaH2D(_NoCuda, "cpu", rec, 2)
    batch = {
        k: np.zeros((8, 3), dtype=np.float32)
        for k in (
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
    for step in range(1, 3):
        rec.begin()
        data_ms = rec.mark("step.span.data")
        assert rec.record_io(batch, 0.1, None, data_ms) == 11 * 8 * 3 * 4
        assert main.begin(step, 100, data_wait_ms=data_ms) == -1
        rec.put("train.loss", 0.5)
        rec.end()
        rec.commit(step)
    assert main.flush() is False and evo.flush() is False
    sink = rec.record_sink()
    worker = threading.Thread(
        target=lambda: sink.emit("jope.telemetry.eval-batch/v1", {"kind": "strat300", "n": 1})
    )
    worker.start()
    worker.join()
    stats = {"enabled": False, "optimizer_steps": 2}
    manifest = rec.finish_round(stats)
    assert manifest["status"] == "complete" and stats["telemetry"] is manifest
    assert stats["h2d_event_sample_drops"] == 0
    assert manifest["metadata"]["io_startup_reads"]["reads"] == 1
    run = next(tmp_path.glob("w-step-r1-s7-*"))
    rows = _events(run)
    ids = _ids()
    unavailable = {
        e["event"]["metric_id"]: e["event"].get("availability_reason_id")
        for e in rows
        if e["kind"] == "event" and e["event"].get("availability_reason_id")
    }
    assert unavailable[ids["io.batch_gb_s"]] == 7
    assert unavailable[ids["throughput.tokens_per_s"]] == 10
    kinds = [e["kind"] for e in rows if e["kind"] != "event"]
    assert "jope.telemetry.evo-queue/v1" in kinds and "jope.telemetry.eval-batch/v1" in kinds


def test_records_and_fail_closed_guards(tmp_path: Path) -> None:
    rec = tel.Recorder(str(tmp_path / "r"), ["p0"], {"smoke": True})
    sink = rec.record_sink()
    batch = tel.EvalBatch(sink)
    batch.batch_begin("strat300", 2, 2, 2)
    for i in range(2):
        batch.worker_start(i)
        batch.fit_done(i, 1_000_000, -1 if i else 0)
        batch.worker_end(i)
    out = batch.batch_end()
    assert out["completed"] == 2 and out["timeouts"] == 1 and batch.rc_of(1) == -1
    gen = tel.GenerationRecorder(tel.GpSession(seed=1), "ctx", 0, sink=sink, bank_shas=frozenset())
    gen.add({"dsl": "ttsln", "status": "failed", "fitness": 100.0}, rc=-1)
    assert gen.finish(0)["generation"] == 0
    val = tel.ValidationRecorder(sink)
    with pytest.raises(tel.SealedPartitionError):
        val.add(dataset="d", task="regression", partition="meta_test", loss=1.0)
    val.add(dataset="d", task="regression", partition=None, family="f", loss=1.0)
    assert val.finish(1)["sealed_label_missing"] == 1
    with pytest.raises(ValueError):
        rec.emit("jope.telemetry.test/v1", {"x": float("nan")})
    assert rec.close()["records_written"] == 3


def test_dope_guard_hashes_like_ns3_and_fails_closed() -> None:
    torch = pytest.importorskip("torch")
    import hashlib

    from jope.telemetry.dope_guard import DopeFreezeGuard, lock_dope

    class Fused(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dope_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
            self.head = torch.nn.Linear(2, 1)

    net = Fused()
    lock_dope(net)
    opt = torch.optim.SGD(net.head.parameters(), lr=0.1)
    guard = DopeFreezeGuard(net, opt)
    start = guard.run_start()
    assert start["violations"] == [] and start["hashes_equal"] is None
    assert start["num_param_tensors"] == 2
    expected = hashlib.sha256(
        "".join(
            f"dope_model.{n}={hashlib.sha256(f'<f4:{list(p.shape)}:'.encode() + p.detach().numpy().tobytes()).hexdigest()}\n"
            for n, p in sorted(net.dope_model.named_parameters())
        ).encode()
    ).hexdigest()
    assert start["params_sha256"] == expected
    net.train(True)
    assert guard.epoch_end(1)["hashes_equal"] is True
    with torch.no_grad():
        net.dope_model[0].weight.add_(1.0)
    with pytest.raises(tel.DopeFreezeViolationError, match="DOPE hash changed: dope_model.0.weight"):
        guard.epoch_end(2)
