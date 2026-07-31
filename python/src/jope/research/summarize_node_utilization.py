#!/usr/bin/env python3
"""Summarize campaign-aware node utilization captures."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--current-workers", required=True, type=int)
    parser.add_argument("--logical-cpus", required=True, type=int)
    args = parser.parse_args()

    nodes: list[dict[str, object]] = []
    estimated_workers: list[int] = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        interval = float(payload["interval_seconds"])
        active = [
            sample
            for sample in payload["samples"]
            if int(sample["campaign_worker_processes"]) > 0
        ]
        busy = [float(sample["cpu_busy_percent"]) for sample in active]
        peak = max(busy) if busy else None
        required = (
            min(
                args.logical_cpus,
                math.ceil(args.current_workers * 80.0 / peak),
            )
            if peak and peak > 0
            else args.logical_cpus
        )
        estimated_workers.append(required)
        nodes.append(
            {
                "node": payload["node"],
                "all_samples": len(payload["samples"]),
                "worker_samples": len(active),
                "worker_observed_seconds": len(active) * interval,
                "mean_busy_percent_during_worker_lifetime": (
                    sum(busy) / len(busy) if busy else None
                ),
                "p50_busy_percent_during_worker_lifetime": percentile(busy, 0.50),
                "p95_busy_percent_during_worker_lifetime": percentile(busy, 0.95),
                "maximum_busy_percent": peak,
                "fraction_worker_samples_at_least_80_percent": (
                    sum(value >= 80.0 for value in busy) / len(busy) if busy else None
                ),
                "minimum_mem_available_bytes": min(
                    int(sample["mem_available_kib"]) for sample in payload["samples"]
                )
                * 1024,
                "estimated_workers_for_80_percent_peak": required,
            }
        )

    required = max(estimated_workers)
    recommended = math.ceil(required / 4) * 4
    summary = {
        "schema": "jope-dime.cluster-utilization-summary.v1",
        "generated_at_unix_ms": time.time_ns() // 1_000_000,
        "current_workers_per_node": args.current_workers,
        "logical_cpus_per_node": args.logical_cpus,
        "current_theoretical_maximum_busy_percent": 100.0
        * args.current_workers
        / args.logical_cpus,
        "nodes": sorted(nodes, key=lambda item: str(item["node"])),
        "recommended_workers_per_node_for_peak_target": recommended,
        "recommended_theoretical_maximum_busy_percent": 100.0
        * recommended
        / args.logical_cpus,
        "interpretation": (
            "Candidate compute is bursty and generation orchestration dominates mean worker "
            "lifetime. Raising workers can meet the 80% peak target; increasing generation "
            "count alone cannot raise instantaneous utilization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
