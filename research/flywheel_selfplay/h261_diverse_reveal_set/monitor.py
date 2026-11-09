#!/usr/bin/env python3
"""
H261 (`h261_diverse_reveal_set`) Compute & H239-A1 Memory/Load Guard Wrapper.
Enforces:
  - CPU affinity pinned to xbabe2 cores 57-60 (`[57, 58, 59, 60]`)
  - `nice >= 19`, `threads <= 4`, `aggregate RSS <= 4 GiB`
  - Admission: `MemAvailable >= 30 GiB`, `free disk >= 300 GB`
  - Cooperative pause (`.pause_requested`) when `MemAvailable < 20 GiB` or `load5 > 108.0`
  - Compute stop deadline `2026-11-03T17:15:00Z` (`1793726100.0`)
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
COMPUTE_STOP_UTC = 1793726100.0  # 2026-11-03T17:15:00Z
ALLOWED_CORES = [57, 58, 59, 60]


def read_mem_available_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
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
    max_threads = 0
    affinity: list[int] = []
    for pid in list(pids):
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as stf:
                for line in stf:
                    if line.startswith("VmRSS:"):
                        total_rss += int(line.split()[1]) * 1024
                    elif line.startswith("Threads:"):
                        max_threads = max(max_threads, int(line.split()[1]))
            if pid == root_pid:
                affinity = sorted(os.sched_getaffinity(pid))
        except OSError:
            continue
    return total_rss, max_threads, affinity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H261 resource & H239-A1 guard monitor"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command supplied to monitor.py")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = out_dir / ".pause_requested"
    if pause_flag.exists():
        pause_flag.unlink()

    os.sched_setaffinity(0, set(ALLOWED_CORES))
    try:
        os.nice(19)
    except OSError:
        pass

    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[k] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["H261_PAUSE_FILE"] = str(pause_flag)

    # Wait for admission (MemAvailable >= 30 GiB, disk free >= 300 GB)
    while True:
        now = time.time()
        if now >= COMPUTE_STOP_UTC:
            raise SystemExit("Compute stop deadline reached before admission")
        avail = read_mem_available_bytes()
        l1, l5 = read_load()
        free_disk = shutil.disk_usage(out_dir).free
        if avail >= 30 * GIB and free_disk >= 300 * GB_DEC:
            break
        time.sleep(2.0)

    t0 = time.time()
    stdout_path = out_dir / f"{args.run_id}.stdout"
    with open(stdout_path, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=subprocess.STDOUT)

        avail0 = read_mem_available_bytes()
        l1_0, l5_0 = read_load()
        free0 = shutil.disk_usage(out_dir).free
        rss0, thr0, aff0 = read_proc_tree_rss_and_threads(proc.pid)
        admission = {
            "utc": time.time(),
            "load1": l1_0,
            "load5": l5_0,
            "available": avail0,
            "free": free0,
            "rss": rss0,
            "threads": thr0,
            "affinity": aff0 or ALLOWED_CORES,
        }

        peak_rss = rss0
        peak_threads = max(1, thr0)
        transitions: list[dict[str, Any]] = []
        paused = False
        pause_start = 0.0
        pause_total = 0.0
        reasons: list[str] = []
        last_sample = dict(admission)

        while True:
            ret = proc.poll()
            now = time.time()
            avail = read_mem_available_bytes()
            l1, l5 = read_load()
            free_disk = shutil.disk_usage(out_dir).free
            rss, thr, aff = read_proc_tree_rss_and_threads(proc.pid)
            peak_rss = max(peak_rss, rss)
            peak_threads = max(peak_threads, thr)
            last_sample = {
                "utc": now,
                "load1": l1,
                "load5": l5,
                "available": avail,
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
                reasons.append("compute_stop_1715Z_exceeded")
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

            need_pause = (avail < 20 * GIB) or (l5 > 108.0)
            can_resume = (avail >= 24 * GIB) and (l5 <= 108.0)
            if need_pause and not paused:
                pause_flag.write_text("pause\n", encoding="utf-8")
                paused = True
                pause_start = now
                transitions.append(
                    {
                        "utc": now,
                        "action": "cooperative_pause",
                        "available": avail,
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
        "admission": admission,
        "final_sample": last_sample,
        "elapsed_seconds": time.time() - t0,
        "peak": {
            "rss": peak_rss,
            "threads": peak_threads,
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
