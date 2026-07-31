#!/usr/bin/env python3
"""Stage and run disjoint Golden46 registered-method shards on xbabe1/2/3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from golden46_execution import (
    CAMPAIGN_ROOT,
    COHORT_PATH,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_WORKER_ROOT,
    read_json,
    validate_worker_view,
)


NODES = ("xbabe1", "xbabe2", "xbabe3")
CONTROLLER = "xbabe2"
REMOTE_SCRIPTS = Path("/home/ubuntu/jope-dime/scripts")
NODE_RUNNER = REMOTE_SCRIPTS / "golden46_node_runner.py"
EXECUTION_SCRIPT = REMOTE_SCRIPTS / "golden46_execution.py"
RESERVES_GIB = {"xbabe1": 16.0, "xbabe2": 32.0, "xbabe3": 32.0}
PHASE_RESOURCES = {
    "cpu": {"cpu", "cpu_light"},
    "gpu": {"gpu", "gpu_heavy"},
}


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def on_node(node: str, command: list[str]) -> list[str]:
    return command if node == CONTROLLER else ["ssh", node, *command]


def stage_dataset(dataset_id: str) -> None:
    validate_worker_view(DEFAULT_WORKER_ROOT / dataset_id, dataset_id)
    for node in NODES:
        if node == CONTROLLER:
            continue
        directories = (
            REMOTE_SCRIPTS,
            DEFAULT_WORKER_ROOT / dataset_id,
            COHORT_PATH.parent,
            DEFAULT_ARTIFACT_ROOT / dataset_id,
        )
        run(["ssh", node, "mkdir", "-p", *(str(path) for path in directories)])
        for source in (EXECUTION_SCRIPT, NODE_RUNNER):
            run(["rsync", "-a", "--checksum", str(source), f"{node}:{source}"])
        run(["rsync", "-a", "--checksum", str(COHORT_PATH), f"{node}:{COHORT_PATH}"])
        run(
            [
                "rsync",
                "-a",
                "--checksum",
                f"{DEFAULT_WORKER_ROOT / dataset_id}/",
                f"{node}:{DEFAULT_WORKER_ROOT / dataset_id}/",
            ]
        )


def eligible_methods(dataset_id: str, resources: set[str]) -> list[str]:
    quant_root = Path("/home/ubuntu/remote_super/quant")
    if str(quant_root) not in sys.path:
        sys.path.insert(0, str(quant_root))
    from models.tools.model_zoo import DatasetInfo, specs_by_name

    cohort = read_json(COHORT_PATH)
    manifest = read_json(DEFAULT_WORKER_ROOT / dataset_id / "worker-manifest.json")
    info = DatasetInfo(
        dataset_id,
        int(manifest["train_rows"]),
        int(manifest["test_rows"]),
        int(manifest["n_features"]),
    )
    specs = specs_by_name("regression")
    frozen_ineligible = {
        item["method_id"]: item["reason"]
        for item in cohort["full_zts_terminal_ineligibilities"]
    }
    result = []
    for method_id in cohort["registered_method_ids"]:
        spec = specs[method_id]
        if spec.resource not in resources:
            continue
        eligible, reason = spec.check_eligibility(info)
        if not eligible:
            if dataset_id != "ZTS" or frozen_ineligible.get(method_id) != reason:
                raise ValueError(f"unexpected ineligibility for {method_id}: {reason}")
            continue
        result.append(method_id)
    return result


def node_command(
    node: str,
    dataset_id: str,
    resources: set[str],
    methods: list[str],
    *,
    force: bool,
) -> list[str]:
    command = [
        "/usr/bin/python3",
        str(NODE_RUNNER),
        "--worker-root",
        str(DEFAULT_WORKER_ROOT),
        "--artifact-root",
        str(DEFAULT_ARTIFACT_ROOT),
        "--node",
        node,
        "--dataset",
        dataset_id,
        "--resources",
        ",".join(sorted(resources)),
        "--methods",
        ",".join(methods),
        "--workers",
        "1" if resources & {"gpu", "gpu_heavy"} else "96",
        "--maximum-workers",
        "1" if resources & {"gpu", "gpu_heavy"} else "112",
        "--reserve-gib",
        str(RESERVES_GIB[node]),
    ]
    if force:
        command.append("--force")
    return on_node(node, command)


def phase_name(resources: set[str]) -> str:
    return "-".join(sorted(resources))


def pull_node(node: str, dataset_id: str, resources: set[str]) -> None:
    if node == CONTROLLER:
        return
    run(
        [
            "rsync",
            "-a",
            "--checksum",
            f"{node}:{DEFAULT_ARTIFACT_ROOT / dataset_id}/",
            f"{DEFAULT_ARTIFACT_ROOT / dataset_id}/",
        ]
    )
    remote_phase = CAMPAIGN_ROOT / "nodes" / node / dataset_id / phase_name(resources)
    local_phase = CAMPAIGN_ROOT / "nodes" / node / dataset_id / phase_name(resources)
    local_phase.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--checksum", f"{node}:{remote_phase}/", f"{local_phase}/"])


def read_summary(node: str, dataset_id: str, resources: set[str]) -> dict[str, Any]:
    path = CAMPAIGN_ROOT / "nodes" / node / dataset_id / phase_name(resources) / "summary.json"
    return read_json(path)


def run_phase(
    dataset_id: str,
    phase: str,
    nodes: tuple[str, ...] = NODES,
) -> dict[str, Any]:
    if len(nodes) < 2 or len(set(nodes)) != len(nodes) or any(node not in NODES for node in nodes):
        raise ValueError("a campaign phase requires two or three unique xbabe nodes")
    resources = PHASE_RESOURCES[phase]
    methods = eligible_methods(dataset_id, resources)
    shards = {node: methods[index:: len(nodes)] for index, node in enumerate(nodes)}
    processes: dict[str, subprocess.Popen[str]] = {}
    for node, shard in shards.items():
        if not shard:
            continue
        processes[node] = subprocess.Popen(
            node_command(node, dataset_id, resources, shard, force=False),
            text=True,
        )
    initial_return_codes = {node: process.wait() for node, process in processes.items()}
    for node in processes:
        pull_node(node, dataset_id, resources)

    initial_summaries = {
        node: read_summary(node, dataset_id, resources) for node in processes
    }
    failures = [
        item["method_id"]
        for summary in initial_summaries.values()
        for item in summary["failed"]
    ]
    retry_records = []
    for index, method_id in enumerate(failures):
        original = next(
            node
            for node, summary in initial_summaries.items()
            if any(item["method_id"] == method_id for item in summary["failed"])
        )
        retry_node = next(
            node
            for node in nodes[index % len(nodes) :] + nodes[: index % len(nodes)]
            if node != original
        )
        completed = run(
            node_command(retry_node, dataset_id, resources, [method_id], force=True),
            capture=True,
            check=False,
        )
        pull_node(retry_node, dataset_id, resources)
        summary = read_summary(retry_node, dataset_id, resources)
        retry_records.append(
            {
                "method_id": method_id,
                "original_node": original,
                "retry_node": retry_node,
                "return_code": completed.returncode,
                "retry_failed": bool(summary["failed"]),
            }
        )
    result = {
        "dataset_id": dataset_id,
        "phase": phase,
        "resources": sorted(resources),
        "method_count": len(methods),
        "shards": {node: shard for node, shard in shards.items()},
        "initial_return_codes": initial_return_codes,
        "initial_summaries": initial_summaries,
        "retries": retry_records,
    }
    output = CAMPAIGN_ROOT / "indexes" / f"{dataset_id}.{phase}.cluster-run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--dataset", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--dataset", required=True)
    execute.add_argument("--phase", required=True, choices=sorted(PHASE_RESOURCES))
    execute.add_argument("--nodes", default=",".join(NODES))
    args = parser.parse_args()
    if args.command == "stage":
        stage_dataset(args.dataset)
        result = {"dataset_id": args.dataset, "staged_nodes": list(NODES)}
    else:
        nodes = tuple(node for node in args.nodes.split(",") if node)
        result = run_phase(args.dataset, args.phase, nodes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
