#!/usr/bin/env python3
"""H206 CPU-only preregistration skeleton. Execution is deliberately unavailable."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hwave_numeric_threads import THREAD_POLICY, admit_numeric_threads  # noqa: E402

AUTHORITY = "Jepson via Grok Bot, card answer 8:22 PM MT"
EXECUTOR_IMPLEMENTED = False
CACHE_BYTES = 40_000_000_000
REQUIRED_HEADROOM = 50_000_000_000


def load_plan() -> dict:
    plan = json.loads((HERE / "launch.json").read_text())
    expected = {
        "authority": AUTHORITY,
        "status": "QUEUED_NOT_RUN",
        "wave": "H206",
        "host": "xbabe3",
        "gpu": "NVIDIA GeForce RTX 3090",
        "search_k": [16, 64, 256],
        "expert_rounds": 3,
        "cpu_core_hours_cap": 400,
        "gpu_hours_cap": 18,
        "cache_bytes_cap": CACHE_BYTES,
        "thread_policy": THREAD_POLICY,
        "queue_released": False,
        "executor_implemented": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"preregistered plan mismatch: {key}")
    return plan


def future_admission_reasons(snapshot: dict, reports: list) -> list[str]:
    """Data-only checks for the FUTURE executor; not live resource qualification.

    The future supervisor must collect and bind fresh host/GPU/worker evidence.
    This function never probes GPUs, reads datasets, trains or releases the queue.
    Even a fixture with no reasons cannot enable this unimplemented executor.
    """
    reasons = []
    for key in ("prereg_pr_open", "queue_release_received", "executor_reviewed", "immutable_pins_bound"):
        if snapshot.get(key) is not True:
            reasons.append(key + " required")
    if snapshot.get("host") != "xbabe3" or snapshot.get("gpu") != "NVIDIA GeForce RTX 3090":
        reasons.append("xbabe3 RTX3090 required")
    if snapshot.get("gpu_claimed") is not False:
        reasons.append("GPU must be unclaimed")
    if snapshot.get("nice") != 19:
        reasons.append("nice19 required")
    cpu = snapshot.get("cpu_percent")
    if type(cpu) not in (int, float) or not math.isfinite(cpu) or not 0 <= cpu <= 85:
        reasons.append("host CPU<=85% required")
    total, used = snapshot.get("disk_total_bytes"), snapshot.get("disk_used_bytes")
    if type(total) is not int or type(used) is not int or not 0 <= used < total:
        reasons.append("valid user-visible disk bytes required")
    else:
        if used * 100 > total * 88:
            reasons.append("pre-reservation disk>88%")
        if total - used < REQUIRED_HEADROOM:
            reasons.append("pre-reservation headroom<50GB")
        if (used + CACHE_BYTES) * 1000 >= total * 938:
            reasons.append("post-cache-reservation disk>=93.8%")
    for key, minimum in (("memory_available_bytes", 16 * 2**30),):
        if type(snapshot.get(key)) is not int or snapshot[key] < minimum:
            reasons.append(key + " below minimum")
    rss = snapshot.get("rss_bytes")
    if type(rss) is not int or not 0 <= rss < 16 * 2**30:
        reasons.append("process-tree RSS<16GiB required")
    try:
        admit_numeric_threads(reports, os_threads=snapshot.get("os_threads"))
    except (TypeError, ValueError, AttributeError) as exc:
        reasons.append("numeric admission: " + str(exc))
    return reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print CPU-only queued plan")
    mode.add_argument("--execute", action="store_true", help="always refused by this skeleton")
    ap.add_argument("--run-id", default="H206-QUEUED-PREREG")
    ap.add_argument("--output", type=Path, help="exclusive-create a queued launch plan JSON")
    args = ap.parse_args(argv)
    if args.execute:
        ap.error("H206 QUEUED: executor not implemented; no GPU/scored inference authorized")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id):
        ap.error("invalid research run id")
    plan = load_plan()
    plan["research_run_id"] = args.run_id
    plan["dope_policy"]["research_run_id"] = args.run_id
    plan["preparation_only"] = True
    plan["runtime_admission"] = "NOT_PERFORMED"
    text = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
