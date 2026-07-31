#!/usr/bin/env python3
"""Per-node CPU slot multipliers and pressure-gated xbabe4 ramp."""

from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Iterable, Mapping

from test_exact_common import atomic_json

XBABE4 = "xbabe4"
DEFAULT_MULTIPLIER = 1.0
XBABE4_MULTIPLIER = 2.0
RAMP_MID_SLOTS = 30
RAMP_FINAL_SLOTS = 40
RAMP_MID_DURATION_MS = 10 * 60_000


def node_slot_target(effective_cores: int, multiplier: float = DEFAULT_MULTIPLIER) -> int:
    if effective_cores < 0 or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid effective cores or CPU-job slot multiplier")
    return max(0, math.floor(effective_cores * multiplier))


def node_capacity(
    *,
    node: str,
    physical_cores: int,
    effective_cores: int,
    running_slots: int,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> dict:
    if physical_cores < effective_cores or running_slots < 0:
        raise ValueError("invalid node capacity counters")
    return {
        "node": node,
        "physical_cores": physical_cores,
        "effective_cores": effective_cores,
        "cpu_job_slot_multiplier": multiplier,
        "slot_target": node_slot_target(effective_cores, multiplier),
        "running_slots": running_slots,
    }


def unsafe_sample(
    sample: Mapping[str, object],
    *,
    baseline_failure_rate: float,
    baseline_p95_runtime_seconds: float,
    baseline_throughput_per_slot: float,
) -> list[str]:
    def number(key: str, default: float = 0.0) -> float:
        value = sample.get(key)
        return default if value is None else float(value)

    reasons = []
    if number("cpu_psi") >= 0.25:
        reasons.append("cpu_psi")
    if number("memory_psi") >= 0.05:
        reasons.append("memory_psi")
    if number("io_psi") >= 0.10:
        reasons.append("io_psi")
    if number("available_memory_fraction", 1.0) < 0.15:
        reasons.append("available_memory")
    if number("failure_rate") - baseline_failure_rate > 0.0025:
        reasons.append("failure_rate_regression")
    runtime_regressed = (
        baseline_p95_runtime_seconds > 0
        and number("p95_runtime_seconds")
        > 1.5 * baseline_p95_runtime_seconds
    )
    throughput_regressed = (
        baseline_throughput_per_slot > 0
        and number("throughput_per_slot")
        <= 0.8 * baseline_throughput_per_slot
    )
    if runtime_regressed and throughput_regressed:
        reasons.append("runtime_and_throughput_regression")
    return reasons


def ramp_decision(
    state: Mapping[str, object],
    samples: Iterable[Mapping[str, object]],
    *,
    now_at_ms: int | None = None,
) -> dict:
    """Advance 20→30→40, or roll back after three unsafe minute samples."""

    timestamp = int(time.time() * 1000) if now_at_ms is None else now_at_ms
    values = [dict(sample) for sample in samples]
    baseline = state.get("baseline") or {}
    unsafe = [
        unsafe_sample(
            sample,
            baseline_failure_rate=float(baseline.get("failure_rate") or 0.0),
            baseline_p95_runtime_seconds=float(
                baseline.get("p95_runtime_seconds") or 0.0
            ),
            baseline_throughput_per_slot=float(
                baseline.get("throughput_per_slot") or 0.0
            ),
        )
        for sample in values[-3:]
    ]
    if len(unsafe) == 3 and all(reasons for reasons in unsafe):
        return {
            **state,
            "stage": "rolled_back",
            "multiplier": 1.0,
            "slot_target": int(state.get("effective_cores") or 20),
            "rollback_at_ms": timestamp,
            "rollback_reasons": sorted(
                {reason for reasons in unsafe for reason in reasons}
            ),
        }
    stage = str(state.get("stage") or "baseline")
    stage_started = int(state.get("stage_started_at_ms") or timestamp)
    if stage == "baseline":
        return {
            **state,
            "stage": "ramp_30",
            "multiplier": 1.5,
            "slot_target": RAMP_MID_SLOTS,
            "stage_started_at_ms": timestamp,
        }
    if stage == "ramp_30" and timestamp - stage_started >= RAMP_MID_DURATION_MS:
        return {
            **state,
            "stage": "target_40",
            "multiplier": XBABE4_MULTIPLIER,
            "slot_target": RAMP_FINAL_SLOTS,
            "stage_started_at_ms": timestamp,
        }
    return dict(state)


def xbabe2_burn_eligible(
    samples: Iterable[Mapping[str, object]],
    *,
    fresh_queue_depth: int,
) -> bool:
    values = list(samples)[-15:]
    if len(values) < 15 or fresh_queue_depth <= 0:
        return False
    if sum(float(row.get("cluster_cpu_fraction") or 0.0) for row in values) / 15 >= 0.75:
        return False
    return all(
        not unsafe_sample(
            row,
            baseline_failure_rate=float(row.get("baseline_failure_rate") or 0.0),
            baseline_p95_runtime_seconds=float(
                row.get("baseline_p95_runtime_seconds") or 0.0
            ),
            baseline_throughput_per_slot=float(
                row.get("baseline_throughput_per_slot") or 0.0
            ),
        )
        for row in values
    )


def load_node_policy(path: Path) -> dict:
    value = tomllib.loads(path.read_text())
    nodes = {}
    for row in value.get("node") or []:
        multiplier = float(row.get("cpu_job_slot_multiplier", DEFAULT_MULTIPLIER))
        nodes[str(row["name"])] = {
            **row,
            "cpu_job_slot_multiplier": multiplier,
        }
    return {
        "poll_ms": int(value.get("poll_ms", 10_000)),
        "telemetry_ms": int(value.get("telemetry_ms", 1_000)),
        "nodes": nodes,
    }


def validate_selfplay_policy(path: Path) -> dict:
    value = load_node_policy(path)
    if value["poll_ms"] != 2_000 or value["telemetry_ms"] != 1_000:
        raise ValueError("SCQ self-play policy must use 2s scheduling/1s telemetry")
    if value["nodes"].get(XBABE4, {}).get("cpu_job_slot_multiplier") != 2.0:
        raise ValueError("xbabe4 CPU-job multiplier must be 2.0")
    for node in ("xbabe1", "xbabe2", "xbabe3"):
        if value["nodes"].get(node, {}).get("cpu_job_slot_multiplier") != 1.0:
            raise ValueError(f"{node} multiplier must remain 1.0 at rollout start")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--status", type=Path)
    args = parser.parse_args()
    result = validate_selfplay_policy(args.policy)
    if args.status:
        atomic_json(args.status, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
