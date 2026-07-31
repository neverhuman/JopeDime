#!/usr/bin/env python3
"""Aggregate auditable Golden46 baseline attempts, pressure, and timing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CAMPAIGN_ROOT = Path(
    "/home/ubuntu/remote_super/quant/campaigns/golden46-full42-20260723-v1"
)
DATASETS = ("ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296", "ZTS")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        values.append(value)
    return values


def report_scores(dataset_id: str) -> dict[str, float]:
    scores = {}
    artifact_root = CAMPAIGN_ROOT / "artifacts" / dataset_id
    for path in sorted(artifact_root.glob("*/report.json")):
        report = read_json(path)
        metrics = report.get("metrics", {})
        rmse = metrics.get("rmse") if isinstance(metrics, dict) else None
        if (
            report.get("status") == "completed"
            and report.get("benchmark_valid") is True
            and isinstance(rmse, (float, int))
            and math.isfinite(float(rmse))
        ):
            scores[path.parent.name] = float(rmse)
    return scores


def receipt_host(dataset_id: str, method_id: str) -> str | None:
    path = (
        CAMPAIGN_ROOT
        / "artifacts"
        / dataset_id
        / method_id
        / "worker-receipt.json"
    )
    if not path.is_file():
        return None
    environment = read_json(path).get("environment", {})
    return environment.get("host") if isinstance(environment, dict) else None


def aggregate(dataset_id: str) -> dict[str, Any]:
    node_root = CAMPAIGN_ROOT / "nodes"
    event_files = sorted(node_root.glob(f"*/{dataset_id}/*/events.jsonl"))
    telemetry_files = sorted(node_root.glob(f"*/{dataset_id}/*/telemetry.jsonl"))
    attempts = []
    starts = []
    source_events = []
    for path in event_files:
        node = path.parents[2].name
        phase = path.parent.name
        source_events.append({"path": str(path), "sha256": sha256(path)})
        for event in jsonl(path):
            if event.get("event") == "node_start":
                starts.append(float(event["ts_unix"]))
            if event.get("event") != "finished":
                continue
            attempts.append(
                {
                    "node": node,
                    "phase": phase,
                    "method_id": event["method_id"],
                    "return_code": int(event["return_code"]),
                    "seconds": float(event["seconds"]),
                    "ts_unix": float(event["ts_unix"]),
                }
            )
    attempts.sort(key=lambda item: (item["ts_unix"], item["node"], item["method_id"]))
    failures = [item for item in attempts if item["return_code"] != 0]
    recoveries = []
    for failure in failures:
        later = next(
            (
                item
                for item in attempts
                if item["method_id"] == failure["method_id"]
                and item["return_code"] == 0
                and item["ts_unix"] > failure["ts_unix"]
                and item["node"] != failure["node"]
            ),
            None,
        )
        final_host = receipt_host(dataset_id, failure["method_id"])
        recoveries.append(
            {
                "method_id": failure["method_id"],
                "failed_node": failure["node"],
                "failed_at": failure["ts_unix"],
                "return_code": failure["return_code"],
                "recovered_on_different_node": later is not None
                or (final_host is not None and final_host != failure["node"]),
                "recovery_node": later["node"] if later is not None else final_host,
                "recovery_at": later["ts_unix"] if later is not None else None,
            }
        )

    telemetry = []
    source_telemetry = []
    for path in telemetry_files:
        source_telemetry.append({"path": str(path), "sha256": sha256(path)})
        telemetry.extend(jsonl(path))
    cpu_samples = [
        float(row["cpu_percent"])
        for row in telemetry
        if isinstance(row.get("cpu_percent"), (float, int))
    ]
    children_rss = [
        int(row["children_rss_bytes"])
        for row in telemetry
        if isinstance(row.get("children_rss_bytes"), int)
    ]
    swap_used = [
        float(row["swap_used_gib"])
        for row in telemetry
        if isinstance(row.get("swap_used_gib"), (float, int))
    ]
    io_pressure = [
        float(row["io_pressure_avg10"])
        for row in telemetry
        if isinstance(row.get("io_pressure_avg10"), (float, int))
    ]
    available = [
        float(row["memory_available_gib"])
        for row in telemetry
        if isinstance(row.get("memory_available_gib"), (float, int))
    ]
    finish_times = [item["ts_unix"] for item in attempts]
    wall_seconds = (
        max(finish_times) - min(starts)
        if starts and finish_times and max(finish_times) >= min(starts)
        else None
    )
    successful_methods = {
        item["method_id"] for item in attempts if item["return_code"] == 0
    }
    special_indexes = []
    special_wall = 0.0
    for phase in ("cpu", "gpu"):
        path = CAMPAIGN_ROOT / "indexes" / f"{dataset_id}.{phase}.special-run.json"
        if not path.is_file():
            continue
        value = read_json(path)
        special_indexes.append({"path": str(path), "sha256": sha256(path)})
        special_wall += float(value.get("wall_time_seconds", 0.0))

    scores = report_scores(dataset_id)
    payload = {
        "schema": "golden46.campaign-metrics.v1",
        "campaign_id": "golden46-full42-20260723-v1",
        "dataset_id": dataset_id,
        "status": "complete" if len(scores) in {103, 123} else "in-progress",
        "validated_score_count": len(scores),
        "scores": dict(sorted(scores.items())),
        "best_score": min(scores.values()) if scores else None,
        "registered_attempts": {
            "finished_attempt_count": len(attempts),
            "successful_method_count": len(successful_methods),
            "failed_attempt_count": len(failures),
            "failed_unique_method_count": len(
                {item["method_id"] for item in failures}
            ),
            "failures": failures,
            "recoveries": recoveries,
            "aggregate_single_thread_worker_elapsed_seconds": sum(
                item["seconds"] for item in attempts
            ),
            "process_cpu_time": {
                "status": "not-recorded-by-frozen-node-runner",
                "note": (
                    "Workers were thread-limited to one; aggregate worker elapsed time "
                    "is reported separately and is not mislabeled as process CPU time."
                ),
            },
        },
        "registered_wall_time_seconds": wall_seconds,
        "registered_throughput_successes_per_hour": (
            len(successful_methods) * 3600.0 / wall_seconds
            if wall_seconds and wall_seconds > 0
            else None
        ),
        "pressure": {
            "telemetry_samples": len(telemetry),
            "mean_cpu_percent": (
                sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
            ),
            "peak_cpu_percent": max(cpu_samples) if cpu_samples else None,
            "peak_children_rss_bytes": max(children_rss) if children_rss else None,
            "minimum_available_memory_gib": min(available) if available else None,
            "peak_swap_used_gib": max(swap_used) if swap_used else None,
            "swap_activity_observed": any(
                row.get("swap_active") is True for row in telemetry
            ),
            "peak_io_pressure_avg10": max(io_pressure) if io_pressure else None,
        },
        "special_wall_time_seconds_sum": special_wall,
        "sources": {
            "events": source_events,
            "telemetry": source_telemetry,
            "special_indexes": special_indexes,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    for dataset_id in datasets:
        payload = aggregate(dataset_id)
        output = CAMPAIGN_ROOT / "indexes" / f"{dataset_id}.campaign-metrics.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "status": payload["status"],
                    "validated_score_count": payload["validated_score_count"],
                    "failed_attempt_count": payload["registered_attempts"][
                        "failed_attempt_count"
                    ],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
