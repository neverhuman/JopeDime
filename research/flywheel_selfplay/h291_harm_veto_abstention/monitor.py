"""Resource and host monitor for H291 (h291_harm_veto_abstention)."""
from __future__ import annotations

import os
import resource
import socket
import time
from pathlib import Path
from typing import Any, Dict

MEM_FLOOR_KB = 10 * 1024 * 1024  # 10 GiB
RSS_CAP_KB = 4 * 1024 * 1024     # 4 GiB


def read_mem_available_kb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def read_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def check_resources(stage: str) -> Dict[str, Any]:
    host = socket.gethostname()
    if host in ("xbabe0", "xbabe3"):
        raise RuntimeError(f"Forbidden host: {host}")
    mem_avail = read_mem_available_kb()
    rss_kb = read_rss_kb()
    if mem_avail < MEM_FLOOR_KB:
        raise RuntimeError(
            f"MemAvailable ({mem_avail} kB) below 10 GiB floor at stage={stage}"
        )
    if rss_kb > RSS_CAP_KB:
        raise RuntimeError(
            f"Peak RSS ({rss_kb} kB) exceeded 4 GiB cap at stage={stage}"
        )
    return {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": host,
        "pid": os.getpid(),
        "nice": os.nice(0),
        "mem_available_kb": mem_avail,
        "peak_rss_kb": rss_kb,
    }
