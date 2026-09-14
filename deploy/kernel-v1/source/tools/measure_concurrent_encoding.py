#!/usr/bin/env python3
"""Observe useful eight-GPU encoding and GP work; this script creates no workload."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def sample(root):
    now = time.time()
    hardware = read(root / "runs/hardware.json")
    gp = read(root / "runs/gp-service.json")
    encoding = read(root / "runs/encoding-service.json")
    ages = {name: max(0, now - data.get("timestamp", 0))
            for name, data in [("hardware", hardware), ("gp", gp), ("encoding", encoding)]}
    active = (encoding.get("persistent_encoder") is True
              and encoding.get("exclusive_cpu") is False
              and encoding.get("phase") not in ("completed", "failed")
              and gp.get("active", 0) > 0)
    gpu_indices = {g.get("index") for g in hardware.get("gpus", [])}
    fresh = max(ages.values()) <= 5 and gpu_indices == set(range(8))
    return {
        "timestamp": now, "telemetry_age_seconds": ages, "fresh": fresh,
        "concurrent_work_active": active,
        "gpus": hardware.get("gpus", []), "cpu_total_percent": hardware.get("cpu_total"),
        "cpu_percent": hardware.get("cpu", []), "numa": hardware.get("numa", []),
        "available_gib": hardware.get("available_gb"), "disk_free_gib": hardware.get("disk_free_gb"),
        "disk_read_mib_s": hardware.get("disk_read_mib_s"),
        "disk_write_mib_s": hardware.get("disk_write_mib_s"),
        "gp": {k: gp.get(k) for k in ["active", "completed_views", "failed_views",
            "completed_evaluations", "running_evaluations", "evaluations_per_second", "executable_sha256"]},
        "encoding": {k: encoding.get(k) for k in ["phase", "completed", "total", "cpu_budget", "ranks"]},
    }


def atomically(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/jope-dime/parallel"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--wait-active", type=int, default=300)
    args = parser.parse_args()
    if args.seconds < 2 or args.warmup < 0 or args.wait_active < 0:
        parser.error("invalid measurement duration")
    if args.output.exists():
        parser.error("measurement output already exists")
    started = time.monotonic()
    while True:
        row = sample(args.root)
        if row["fresh"] and row["concurrent_work_active"]:
            break
        if time.monotonic() - started >= args.wait_active:
            atomically(args.output, {"schema": "parallel.concurrent_measurement.v1",
                "measurement_complete": False, "reason": "no fresh concurrent workload before deadline", "last": row})
            raise SystemExit(1)
        time.sleep(1)
    for _ in range(args.warmup):
        time.sleep(1)
    rows = []
    first = time.monotonic()
    for i in range(args.seconds + 1):
        rows.append(sample(args.root))
        if i < args.seconds:
            time.sleep(max(0, first + i + 1 - time.monotonic()))
    valid = [r for r in rows if r["fresh"] and r["concurrent_work_active"]]
    def values(key):
        return [r[key] for r in valid if isinstance(r.get(key), (int, float))]
    def mean(xs):
        return statistics.mean(xs) if xs else None
    gpu = []
    for index in range(8):
        samples = [g for r in valid for g in r["gpus"] if g["index"] == index]
        gpu.append({"index": index, "samples": len(samples),
            "mean_utilization_percent": mean([g["utilization"] for g in samples]),
            "minimum_utilization_percent": min([g["utilization"] for g in samples], default=None),
            "peak_memory_mib": max([g["memory_mb"] for g in samples], default=None),
            "mean_power_watts": mean([g["watts"] for g in samples])})
    def count(row, key):
        return row["gp"].get(key) or 0
    actual_span = rows[-1]["timestamp"] - rows[0]["timestamp"]
    gp_delta = sum(count(rows[-1], k) - count(rows[0], k)
                   for k in ["completed_evaluations", "running_evaluations"])
    encoder_delta = (rows[-1]["encoding"].get("completed") or 0) - (rows[0]["encoding"].get("completed") or 0)
    result = {"schema": "parallel.concurrent_measurement.v1", "measurement_complete": True,
        "requested_seconds": args.seconds, "actual_seconds": actual_span, "warmup_seconds": args.warmup,
        "samples": rows, "fresh_concurrent_samples": len(valid), "other_samples": len(rows)-len(valid),
        "gpus": gpu, "mean_cpu_percent": mean(values("cpu_total_percent")),
        "minimum_available_gib": min(values("available_gib"), default=None),
        "minimum_disk_free_gib": min(values("disk_free_gib"), default=None),
        "mean_gp_reported_evaluations_per_second": mean([r["gp"]["evaluations_per_second"] for r in valid]),
        "observed_gp_evaluation_delta": gp_delta, "observed_gp_evaluations_per_second": gp_delta/actual_span,
        "observed_encoder_cut_delta": encoder_delta, "observed_encoder_cuts_per_second": encoder_delta/actual_span,
        "new_failed_gp_views": count(rows[-1], "failed_views") - count(rows[0], "failed_views"),
        "workload_generated_by_measurement": False}
    atomically(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
