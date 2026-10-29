import builtins
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fused_telemetry as telemetry

NAMES = [
    "io.batch_gb_s",
    "train.loss",
    "optim.lr.group",
    "optim.grad_norm.global",
    "step.wall",
    "throughput.samples_per_s",
    "io.batch.bytes",
    "io.batch_bytes_per_s",
    "io.samples_per_s",
    "throughput.tokens_per_s",
    "io.h2d_gb_s",
    "dope.freeze.equal",
    "params.count.trainable",
]


@pytest.fixture
def native(monkeypatch, tmp_path):
    class Recorder:
        def __init__(self, *args):
            self.args, self.scalars, self.missing = args, [], []
            self.closed = self.interrupted = False

        def emit_scalar(self, *args):
            self.scalars.append(args)

        def emit_unavailable(self, *args):
            self.missing.append(args)

        def emit(self, *args):
            self.record = args

        def close(self):
            self.closed = True
            return {
                "status": "complete",
                "event_drops": 0,
                "record_drops": 0,
                "sink_errors": 0,
            }

        def interrupt(self):
            self.interrupted = True

    monkeypatch.setattr(telemetry, "_active", None)
    monkeypatch.setenv("JOPE_TELEMETRY_RUN_DIR", str(tmp_path / "records"))
    module = types.ModuleType("jope.telemetry")
    module.Recorder = Recorder
    module.metric_registry = lambda: [(i, n) for i, n in enumerate(NAMES)]
    monkeypatch.setitem(sys.modules, "jope.telemetry", module)
    return module


def test_disabled_never_imports_native_or_opens_sink(monkeypatch):
    monkeypatch.delenv("JOPE_TELEMETRY_RUN_DIR", raising=False)
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert not name.startswith("jope")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    assert telemetry.open_round(1, 2, {}) is None
    telemetry.emit_step(
        None,
        step=0,
        loss=0,
        lr=0,
        grad_norm=0,
        samples=1,
        wall_s=1,
        host_bytes=0,
        delivery_s=1,
    )
    assert telemetry.finish_round({}) is None


def test_missing_metric_refuses_before_recorder_creation(native):
    native.metric_registry = list
    with pytest.raises(RuntimeError, match="io.batch_gb_s"):
        telemetry.open_round(1, 2, {})
    assert telemetry._active is None


def test_missing_extension_refuses(native):
    native.metric_registry = None
    with pytest.raises(RuntimeError, match="wheel is unavailable"):
        telemetry.open_round(1, 2, {})


def test_actual_payload_rates_and_unavailable_tokens_and_h2d(native):
    session = telemetry.open_round(7, 42, {"dope_sha256": "frozen"})
    rec, reg, path = session
    assert path.name == "round-000007-seed-42"
    assert rec.args[1] == ["fused-flywheel", "host-probe"]
    assert rec.args[2]["overhead_qualified"] is False
    telemetry.emit_step(
        session,
        step=10,
        loss=0.4,
        lr=0.001,
        grad_norm=0.2,
        samples=100,
        wall_s=2,
        host_bytes=1000,
        delivery_s=0.5,
    )
    by_metric = {m: v for _, m, _, _, v in rec.scalars}
    assert by_metric[reg["io.batch_gb_s"]] == pytest.approx(2e-6)
    assert by_metric[reg["io.batch_bytes_per_s"]] == 2000
    assert by_metric[reg["throughput.samples_per_s"]] == 50
    assert by_metric[reg["step.wall"]] == 2000
    assert reg["throughput.tokens_per_s"] not in by_metric
    assert (0, reg["throughput.tokens_per_s"], 0, 10, 10) in rec.missing
    assert (0, reg["io.h2d_gb_s"], 0, 10, 12) in rec.missing
    result = telemetry.finish_round(
        {
            "round": 7,
            "dope_sha256": "frozen",
            "trainable_params": 29211,
            "phase_wall_seconds": {},
            "promotions": 0,
            "wall_seconds": 3,
        }
    )
    assert result["status"] == "complete" and rec.closed
    assert telemetry._active is None


def test_interrupt_finalizes_active_run(native):
    rec, _, _ = telemetry.open_round(1, 42, {})
    telemetry.interrupt_active()
    assert rec.interrupted and telemetry._active is None
    telemetry.interrupt_active()
