#!/usr/bin/env python3
"""
H275 (`h275_rerank8_soft_risk`) Compute & Memory/Load/Swap-In-Rate Guard Wrapper.
Enforces:
  - CPU affinity pinned to xbabe2 cores 57-60 (`[57, 58, 59, 60]`)
  - `nice >= 19`, `threads <= 4`, `aggregate RSS <= 4 GiB`
  - Admission: `MemAvailable >= 12 GiB`, `load5 <= 100.0`, `swap_in_rate <= 500 pages/s`, `free disk >= 300 GB`
  - Cooperative pause (`.pause_requested`) when `MemAvailable < 12 GiB`, `load5 > 108.0`, or `swap_in_rate > 1000 pages/s`
    (resume when `MemAvailable >= 16 GiB`, `load5 <= 100.0`, and `swap_in_rate <= 200 pages/s`; NEVER checks `swap_free`)
  - Compute stop deadline `2026-11-03T23:00:00Z` (`1793746800.0`)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GIB = 1024**3
GB_DEC = 10**9
COMPUTE_STOP_UTC = 1793746800.0  # 2026-11-03T23:00:00Z
ALLOWED_CORES = [57, 58, 59, 60]


def read_mem_available_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def read_pswpin_pages() -> int:
    try:
        with open("/proc/vmstat", encoding="utf-8") as f:
            for line in f:
                if line.startswith("pswpin "):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def read_load() -> tuple[float, float]:
    l1, l5, _ = os.getloadavg()
    return float(l1), float(l5)


def read_proc_tree_rss_and_threads(root_pid: int) -> tuple[int, int, list[int]]:
    pids = {root_pid}
    try:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as sf:
                    st = sf.read()
                rpar = st.rfind(")")
                fields = st[rpar + 2 :].split()
                ppid = int(fields[1])
                if ppid in pids:
                    pids.add(pid)
            except OSError:
                continue
    except OSError:
        pass

    total_rss = 0
    total_threads = 0
    aff_union: set[int] = set()
    for pid in list(pids):
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as stf:
                for line in stf:
                    if line.startswith("VmRSS:"):
                        total_rss += int(line.split()[1]) * 1024
                    elif line.startswith("Threads:"):
                        total_threads += int(line.split()[1])
            aff_union.update(os.sched_getaffinity(pid))
        except OSError:
            continue
    return total_rss, total_threads, sorted(aff_union)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H275 compute & swap-in-rate/memory monitor"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("No command specified after --")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = out_dir / ".pause_requested"
    if pause_flag.exists():
        pause_flag.unlink()

    last_pswpin = read_pswpin_pages()
    last_pswpin_t = time.time()
    swap_in_rate_0 = 0.0

    while True:
        now0 = time.time()
        l1_0, l5_0 = read_load()
        avail0 = read_mem_available_bytes()
        free_disk0 = shutil.disk_usage(out_dir).free
        aff0 = sorted(os.sched_getaffinity(0))
        cur_pswpin = read_pswpin_pages()
        dt_s = max(now0 - last_pswpin_t, 1e-3)
        swap_in_rate_0 = max(0.0, float(cur_pswpin - last_pswpin) / dt_s)
        last_pswpin = cur_pswpin
        last_pswpin_t = now0
        if (
            now0 >= COMPUTE_STOP_UTC
            or free_disk0 < 300 * GB_DEC
            or aff0 != ALLOWED_CORES
        ):
            break
        if avail0 >= 12 * GIB and l5_0 <= 100.0 and swap_in_rate_0 <= 500.0:
            break
        time.sleep(2.0)

    now0 = time.time()
    l1_0, l5_0 = read_load()
    avail0 = read_mem_available_bytes()
    free_disk0 = shutil.disk_usage(out_dir).free
    aff0 = sorted(os.sched_getaffinity(0))

    admission: dict[str, Any] = {
        "utc": now0,
        "load1": l1_0,
        "load5": l5_0,
        "available": avail0,
        "swap_in_pages_per_sec": round(swap_in_rate_0, 3),
        "free": free_disk0,
        "rss": 0,
        "threads": 1,
        "affinity": aff0,
    }

    reasons: list[str] = []
    if now0 >= COMPUTE_STOP_UTC:
        reasons.append("compute_stop_2300Z_reached_at_admission")
    if avail0 < 12 * GIB:
        reasons.append(f"admission_MemAvailable<12GiB ({avail0})")
    if l5_0 > 100.0:
        reasons.append(f"admission_load5>100 ({l5_0})")
    if free_disk0 < 300 * GB_DEC:
        reasons.append(f"admission_free_disk<300GB ({free_disk0})")
    if aff0 != ALLOWED_CORES:
        reasons.append(f"affinity_mismatch ({aff0} != {ALLOWED_CORES})")

    if reasons:
        summary = {
            "status": "ABORTED_AT_ADMISSION",
            "exit_code": 1,
            "reasons": reasons,
            "admission": admission,
            "compute_stop": COMPUTE_STOP_UTC,
        }
        (out_dir / f"{args.run_id}.resource_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["H275_PAUSE_FILE"] = str(pause_flag)

    stdout_path = out_dir / f"{args.run_id}.stdout"
    t0 = time.time()
    peak_rss = 0
    peak_threads = 0
    peak_swap_in_rate = 0.0
    transitions: list[dict[str, Any]] = []
    paused = False
    pause_start = 0.0
    pause_total = 0.0
    last_sample: dict[str, Any] = {}
    prev_pswpin = read_pswpin_pages()
    prev_pswpin_t = t0

    with open(stdout_path, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        rss0, thr0, _ = read_proc_tree_rss_and_threads(proc.pid)
        admission["rss"] = rss0
        admission["threads"] = max(thr0, 1)
        peak_rss = rss0
        peak_threads = max(thr0, 1)

        while True:
            ret = proc.poll()
            now = time.time()
            l1, l5 = read_load()
            avail = read_mem_available_bytes()
            free_disk = shutil.disk_usage(out_dir).free
            rss, thr, aff = read_proc_tree_rss_and_threads(proc.pid)
            cur_pswpin = read_pswpin_pages()
            dt_s = max(now - prev_pswpin_t, 1e-3)
            swap_in_rate = max(0.0, float(cur_pswpin - prev_pswpin) / dt_s)
            prev_pswpin = cur_pswpin
            prev_pswpin_t = now

            peak_rss = max(peak_rss, rss)
            peak_threads = max(peak_threads, thr)
            peak_swap_in_rate = max(peak_swap_in_rate, swap_in_rate)

            last_sample = {
                "utc": now,
                "load1": l1,
                "load5": l5,
                "available": avail,
                "swap_in_pages_per_sec": round(swap_in_rate, 3),
                "free": free_disk,
                "rss": rss,
                "threads": thr,
                "affinity": aff or ALLOWED_CORES,
                "elapsed": now - t0,
                "pause_requested": paused,
            }

            if ret is not None:
                break

            if now >= COMPUTE_STOP_UTC:
                reasons.append("compute_stop_2300Z_exceeded")
            if rss > 4 * GIB:
                reasons.append(f"aggregate RSS>{4 * GIB} ({rss})")
            if thr > 4:
                reasons.append(f"threads>4 ({thr})")
            if free_disk < 300 * GB_DEC:
                reasons.append(f"free_disk<300GB ({free_disk})")

            if reasons:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                ret = proc.poll()
                break

            need_pause = (avail < 12 * GIB) or (l5 > 108.0) or (swap_in_rate > 1000.0)
            can_resume = (
                (avail >= 16 * GIB) and (l5 <= 100.0) and (swap_in_rate <= 200.0)
            )
            if need_pause and not paused:
                pause_flag.write_text("pause\n", encoding="utf-8")
                paused = True
                pause_start = now
                transitions.append(
                    {
                        "utc": now,
                        "action": "cooperative_pause",
                        "available": avail,
                        "swap_in_pages_per_sec": round(swap_in_rate, 3),
                        "load1": l1,
                        "load5": l5,
                    }
                )
            elif paused and can_resume:
                if pause_flag.exists():
                    pause_flag.unlink()
                paused = False
                pause_total += now - pause_start
                transitions.append(
                    {
                        "utc": now,
                        "action": "cooperative_resume",
                        "available": avail,
                        "swap_in_pages_per_sec": round(swap_in_rate, 3),
                        "load1": l1,
                        "load5": l5,
                    }
                )

            time.sleep(0.5)

        if paused:
            pause_total += time.time() - pause_start
            if pause_flag.exists():
                pause_flag.unlink()

    status = "PASS" if (ret == 0 and not reasons) else "ABORTED"
    summary = {
        "status": status,
        "exit_code": int(ret if ret is not None else -1),
        "reasons": reasons,
        "guard_type": "swap_in_rate_and_MemAvailable_ge_12GiB_no_swap_free",
        "admission": admission,
        "final_sample": last_sample,
        "elapsed_seconds": time.time() - t0,
        "peak": {
            "rss": peak_rss,
            "threads": peak_threads,
            "swap_in_pages_per_sec": round(peak_swap_in_rate, 3),
        },
        "transitions": transitions,
        "pause_requested_seconds": pause_total,
        "compute_stop": COMPUTE_STOP_UTC,
    }
    summary_path = out_dir / f"{args.run_id}.resource_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
