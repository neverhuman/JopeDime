import argparse
import importlib.util
import json
import tomllib
from pathlib import Path

import jope.research

MODULE_PATH = Path(jope.research.__file__).parent / "genome_boost_cluster_campaign.py"
SPEC = importlib.util.spec_from_file_location("genome_boost_cluster_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def write_gpu_spec(path: Path, job_id: str) -> None:
    path.write_text(
        campaign.render_spec(
            job_id,
            ["true"],
            path.parent,
            gpus_per_task=1,
            cpus_reserved=4,
            ram_gb=32,
            max_runtime_hours=8,
            max_restarts=1,
            preemptible=True,
            capability_path=Path("/tmp/synthetic-campaign-capability"),
        )
    )


def test_tabpfn_spec_has_exact_resources():
    text = campaign.render_spec(
        "tabpfn-test",
        ["true"],
        Path("/tmp"),
        gpus_per_task=1,
        cpus_reserved=4,
        ram_gb=32,
        max_runtime_hours=8,
        max_restarts=1,
        preemptible=True,
        capability_path=Path("/tmp/synthetic-campaign-capability"),
    )
    assert "gpus_per_task = 1" in text
    assert "cpus_reserved = 4" in text
    assert "ram_gb = 32" in text
    spec = tomllib.loads(text)
    assert spec["resource"]["placement"] == "single_node"
    assert "node" not in spec["resource"]
    assert spec["dependencies"]["smoke"] == [
        {
            "name": "campaign runtime capability available",
            "command": "/usr/bin/test",
            "args": ["-e", "/tmp/synthetic-campaign-capability"],
            "timeout_secs": 5,
        }
    ]
    assert "preemptible = true" in text
    assert "max_runtime_hours = 8" in text
    assert "max_restarts = 1" in text


def test_shards_are_deterministic_and_at_most_fifty():
    datasets = [f"{index:016x}" for index in range(759)]
    shards = campaign.shard_datasets(datasets)
    assert [len(shard) for shard in shards] == [50] * 15 + [9]
    assert [item for shard in shards for item in shard] == datasets


def test_submit_next_enqueues_only_one_gpu_job(tmp_path, monkeypatch):
    specs = [tmp_path / f"job-{index}.toml" for index in range(2)]
    for index, path in enumerate(specs):
        write_gpu_spec(path, f"job-{index}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"specs": [str(path) for path in specs]}))
    calls = []

    monkeypatch.setattr(campaign, "scq_jobs", lambda: {})
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    campaign.submit_next(argparse.Namespace(plan=plan, state=tmp_path / "state.json"))

    assert calls == [
        [campaign.SCQ_BINARY, "check", str(specs[0])],
        [campaign.SCQ_BINARY, "submit", str(specs[0])],
    ]
    assert json.loads((tmp_path / "state.json").read_text())["submitted"] == "job-0"


def test_submit_next_does_not_enqueue_while_lane_active(tmp_path, monkeypatch):
    spec = tmp_path / "job.toml"
    write_gpu_spec(spec, "job-0")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"specs": [str(spec)]}))
    monkeypatch.setattr(
        campaign,
        "scq_jobs",
        lambda: {"job-0": {"id": "job-0", "state": "running", "attempt": 0}},
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected submit")),
    )

    campaign.submit_next(argparse.Namespace(plan=plan, state=tmp_path / "state.json"))
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active"] == [{"attempt": 0, "id": "job-0", "state": "running"}]
