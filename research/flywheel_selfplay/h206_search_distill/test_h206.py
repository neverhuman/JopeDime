"""CPU-only prereg/skeleton controls; no corpus, candidate forward or GPU probes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h206_prereg_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def report():
    return {
        "pid": 100,
        "environment": {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "torch_intraop": 1,
        "native_pools": [],
    }


def admitted_fixture():
    return {
        "prereg_pr_open": True,
        "queue_release_received": True,
        "executor_reviewed": True,
        "immutable_pins_bound": True,
        "host": "xbabe3",
        "gpu": "NVIDIA GeForce RTX 3090",
        "gpu_claimed": False,
        "nice": 19,
        "cpu_percent": 10,
        "disk_total_bytes": 1_000_000_000_000,
        "disk_used_bytes": 880_000_000_000,
        "memory_available_bytes": 32 * 2**30,
        "rss_bytes": 1_000_000_000,
        "os_threads": 64,
    }


def test_plan_authority_policy_and_unmeasured_state():
    plan = runner.load_plan()
    assert plan["authority"] in (HERE / "PREREG.md").read_text()
    assert plan["control_k"] == 1 and plan["terminal_updates"] == 1800
    assert len(plan["arm_slots"]) == 4 and plan["gpu_hours_cap"] == 18
    assert plan["dope_policy"]["status"] == "PLANNED_NOT_APPLIED"
    metrics = json.loads((HERE / "metrics.json").read_text())
    assert metrics["verdict"] is None and metrics["native_parity"] is None
    assert metrics["scientific_optimizer_updates"] == 0


def test_cpu_plan_exclusive_output_and_run_identity(tmp_path):
    output = tmp_path / "launch.json"
    assert runner.main(["--plan", "--run-id", "H206-unit", "--output", str(output)]) == 0
    plan = json.loads(output.read_text())
    assert plan["authority"] == runner.AUTHORITY
    assert plan["dope_policy"]["research_run_id"] == plan["research_run_id"] == "H206-unit"
    assert plan["runtime_admission"] == "NOT_PERFORMED" and not plan["queue_released"]
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(output)])
    with pytest.raises(SystemExit):
        runner.main(["--run-id", "../invalid"])


def test_even_admitted_fixture_cannot_execute_or_write_launch(tmp_path):
    assert runner.future_admission_reasons(admitted_fixture(), [report()]) == []
    output = tmp_path / "launch.json"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--execute", "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()
    assert runner.EXECUTOR_IMPLEMENTED is False


@pytest.mark.parametrize(
    "key", ["prereg_pr_open", "queue_release_received", "executor_reviewed", "immutable_pins_bound"]
)
def test_missing_queue_and_source_prerequisites_refuse(key):
    snapshot = admitted_fixture()
    snapshot[key] = False
    assert any(key in v for v in runner.future_admission_reasons(snapshot, [report()]))


@pytest.mark.parametrize("used", [880_000_000_001, 900_000_000_000, 938_000_000_000])
def test_disk_88_gate_is_stricter_than_running_guard(used):
    snapshot = admitted_fixture()
    snapshot["disk_used_bytes"] = used
    assert any("pre-reservation disk" in v for v in runner.future_admission_reasons(snapshot, [report()]))


def test_reserve_cache_and_fifty_gb_headroom_independently_required():
    snapshot = admitted_fixture()
    snapshot.update(disk_total_bytes=500_000_000_000, disk_used_bytes=440_000_000_000)
    assert any("post-cache" in v for v in runner.future_admission_reasons(snapshot, [report()]))
    snapshot.update(disk_total_bytes=400_000_000_000, disk_used_bytes=352_000_000_000)
    assert any("headroom" in v for v in runner.future_admission_reasons(snapshot, [report()]))


@pytest.mark.parametrize(
    "change",
    [
        {"host": "xbabe1"},
        {"gpu_claimed": True},
        {"nice": 0},
        {"cpu_percent": 85.1},
        {"cpu_percent": float("nan")},
        {"disk_total_bytes": None},
        {"memory_available_bytes": 0},
        {"rss_bytes": 16 * 2**30},
    ],
)
def test_host_ownership_and_other_resource_limits(change):
    snapshot = admitted_fixture()
    snapshot.update(change)
    assert runner.future_admission_reasons(snapshot, [report()])


def test_driver_overhead_allowed_but_numeric_excess_and_missing_reports_refuse():
    assert runner.future_admission_reasons(admitted_fixture(), [report()]) == []
    bad = report()
    bad["torch_intraop"] = 5
    assert any("numeric" in v for v in runner.future_admission_reasons(admitted_fixture(), [bad]))
    assert any("numeric" in v for v in runner.future_admission_reasons(admitted_fixture(), []))


def test_research_optimizer_cannot_include_frozen_anchor(monkeypatch):
    # Persist #1726's independent review regression control for this research path.
    import torch
    from jope.north_star.dope_freeze_guard import (
        DopeFreezeViolationError,
        assert_research_optimizer_policy,
        configure_research_dope,
    )

    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    net = torch.nn.Module()
    net.dope_model = torch.nn.Linear(2, 2)
    net.gamenet_anchor = torch.nn.Linear(2, 2)
    configure_research_dope(net, launch_metadata={}, unfreeze=True, research_run_id="H206-unit")
    opt = torch.optim.SGD(
        [
            {"params": net.dope_model.parameters()},
            {"params": net.gamenet_anchor.parameters()},
        ],
        lr=0.01,
    )
    with pytest.raises(DopeFreezeViolationError, match="includes the frozen anchor"):
        assert_research_optimizer_policy(opt, net, research_run_id="H206-unit")
    assert all(not p.requires_grad for p in net.gamenet_anchor.parameters())


def test_manifest_hashes_bind_all_prereg_sources():
    import hashlib

    manifest = json.loads((HERE / "manifest.json").read_text())
    assert manifest["status"] == "QUEUED_NOT_RUN" and manifest["evidence"] == []
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
