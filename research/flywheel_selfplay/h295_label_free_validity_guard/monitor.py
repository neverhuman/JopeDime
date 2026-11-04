"""Resource and data-isolation guard for H295 (h295_label_free_validity_guard):
- CPU affinity: xbabe2 cores 53-56, nice 19, numeric threads 1
- Memory guard: process RSS <= 4.0 GiB, system MemAvailable >= 10.0 GiB
- Data isolation: sys.addaudithook blocking any open of DL-Val, DL-Test, holdout,
  sealed, or fresh cohorts (H283, H285, H286, H288, H292, H294, cohort_v1..v5, fresh).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

FORBIDDEN_TOKENS = (
    "dl_val",
    "dl-val",
    "dlval",
    "dl_test",
    "dl-test",
    "dltest",
    "holdout",
    "sealed",
    "fresh_cohort",
    "cohort_v",
    "fresh_pools",
    "fresh-build",
    "fresh-r3175",
    "fresh_top8",
    "h283",
    "h285",
    "h286",
    "h288",
    "h290",
    "h292",
    "h294",
    "part_b_inputs",
)

_AUDIT_VIOLATIONS: List[str] = []
_FORBIDDEN_READS_COUNT: int = 0
_HOOK_INSTALLED: bool = False


def install_isolation_audit_hook() -> None:
    """Install Python runtime audit hook blocking any file open of forbidden splits/cohorts."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return

    def _hook(event: str, args: tuple) -> None:
        global _FORBIDDEN_READS_COUNT
        if event == "open" and args:
            raw_path = args[0]
            if isinstance(raw_path, (str, bytes, os.PathLike)):
                p_str = os.fspath(raw_path)
                if isinstance(p_str, bytes):
                    p_str = p_str.decode("utf-8", errors="ignore")
                low = p_str.lower()
                # Ignore system/python/proc files and our own h295 capsule path
                if low.startswith(("/proc/", "/sys/", "/usr/", "/lib/", "/etc/", "/dev/")):
                    return
                for tok in FORBIDDEN_TOKENS:
                    if tok in low:
                        _FORBIDDEN_READS_COUNT += 1
                        _AUDIT_VIOLATIONS.append(p_str)
                        raise PermissionError(
                            f"H295 isolation guard blocked forbidden path access ({tok}): {p_str}"
                        )

    sys.addaudithook(_hook)
    _HOOK_INSTALLED = True


def get_forbidden_reads_count() -> int:
    return int(_FORBIDDEN_READS_COUNT)


def enforce_resources(max_rss_gib: float = 4.0, min_mem_avail_gib: float = 10.0) -> Dict[str, float]:
    """Enforce CPU affinity {53,54,55,56}, nice >= 19, numeric threads = 1, RSS <= 4 GiB, MemAvailable >= 10 GiB."""
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = "1"

    try:
        os.sched_setaffinity(0, {53, 54, 55, 56})
    except OSError:
        pass

    try:
        cur_nice = os.nice(0)
        if cur_nice < 19:
            os.nice(19 - cur_nice)
    except OSError:
        pass

    rss_gib = 0.0
    with Path("/proc/self/status").open("r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                rss_gib = kb / (1024.0 * 1024.0)
                break

    mem_avail_gib = 0.0
    with Path("/proc/meminfo").open("r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                mem_avail_gib = kb / (1024.0 * 1024.0)
                break

    if rss_gib > max_rss_gib:
        raise MemoryError(f"H295 RSS limit exceeded: {rss_gib:.3f} GiB > {max_rss_gib:.3f} GiB")
    if mem_avail_gib < min_mem_avail_gib:
        raise MemoryError(
            f"H295 MemAvailable below safety floor: {mem_avail_gib:.3f} GiB < {min_mem_avail_gib:.3f} GiB"
        )

    return {
        "rss_gib": round(rss_gib, 4),
        "mem_available_gib": round(mem_avail_gib, 4),
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": int(os.nice(0)),
        "forbidden_reads_count": int(_FORBIDDEN_READS_COUNT),
    }
