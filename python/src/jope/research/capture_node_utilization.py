#!/usr/bin/env python3
"""Capture short-interval CPU, memory, load, and campaign-process telemetry."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time


def cpu_counters() -> tuple[int, int]:
    with open("/proc/stat", encoding="utf-8") as handle:
        fields = handle.readline().split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def memory_available_kib() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def campaign_processes() -> int:
    count = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/cmdline", "rb") as handle:
                command = handle.read().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"jope-dime-regressor" in command and b"batch-worker" in command:
            count += 1
    return count


def write_payload(
    output: str,
    interval: float,
    samples: list[dict[str, object]],
    capture_complete: bool = False,
) -> None:
    payload = {
        "schema": "jope-dime.node-utilization.v1",
        "node": socket.gethostname().split(".")[0],
        "interval_seconds": interval,
        "capture_complete": capture_complete,
        "samples": samples,
    }
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--flush-seconds", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0 or args.flush_seconds <= 0:
        parser.error("seconds, interval, and flush-seconds must be positive")

    samples: list[dict[str, object]] = []
    previous_total, previous_idle = cpu_counters()
    deadline = time.monotonic() + args.seconds
    next_flush = time.monotonic() + args.flush_seconds
    while time.monotonic() < deadline:
        time.sleep(args.interval)
        total, idle = cpu_counters()
        delta_total = total - previous_total
        delta_idle = idle - previous_idle
        busy_percent = 0.0 if delta_total <= 0 else 100.0 * (delta_total - delta_idle) / delta_total
        previous_total, previous_idle = total, idle
        load_1m, load_5m, load_15m = os.getloadavg()
        samples.append(
            {
                "unix_ms": time.time_ns() // 1_000_000,
                "cpu_busy_percent": busy_percent,
                "mem_available_kib": memory_available_kib(),
                "load_1m": load_1m,
                "load_5m": load_5m,
                "load_15m": load_15m,
                "campaign_worker_processes": campaign_processes(),
            }
        )
        if time.monotonic() >= next_flush:
            write_payload(args.output, args.interval, samples)
            next_flush = time.monotonic() + args.flush_seconds

    write_payload(args.output, args.interval, samples, capture_complete=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
