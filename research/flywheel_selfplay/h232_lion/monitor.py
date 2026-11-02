"""H232 terminal CPU guard (adapted from merged H170): supervisor plus worker tree, one exclusive scientific attempt.

hw_t2120 resources, xbabe1 only, nice 19, launched under `taskset` on 4 cores, one math thread:
- Host load PAUSES the worker, never stops it: SIGSTOP to the worker's own process group when load5 > 60, SIGCONT
  when load5 <= 55 (hysteresis). No other process is ever signalled.
- Hard stops: process-tree RSS >= 4 GiB, MemAvailable < 20 GiB (swap is full), disk free < 150 GB, more than 4 tree
  threads, and the absolute compute stop 2026-11-03T04:00:00Z (a paused worker is continued so it receives SIGTERM).
- Admission (before any claim): load5 <= 55 and every hard limit satisfied. Single attempt, no retries."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
HOST = "xbabe1"
LOAD5_PAUSE, LOAD5_RESUME = 60.0, 55.0
RSS_MAX = 4 * 2**30
MEM_MIN = 20 * 2**30
FREE_MIN = 150_000_000_000
THREADS_MAX = 4
COMPUTE_STOP = 1793678400.0  # 2026-11-03T04:00:00Z (hw_t2120)
CLAIMS = Path("/home/ubuntu/jope_plan/ns/receipts")


def violations(sample):
    """Hard stops only; host load is a pause (see paused)."""
    return [
        reason
        for bad, reason in [
            (sample["rss"] >= RSS_MAX, "aggregate RSS>=4GiB"),
            (sample["available"] < MEM_MIN, "MemAvailable<20GiB"),
            (sample["free"] < FREE_MIN, "disk free<150GB"),
            (sample["threads"] > THREADS_MAX, "aggregate threads>4"),
            (sample["utc"] >= COMPUTE_STOP, "compute stop 2026-11-03T04:00:00Z"),
        ]
        if bad
    ]


def paused(sample, now_paused):
    """Hysteresis: pause above load5 60; once paused, stay paused until load5 <= 55."""
    if now_paused:
        return sample["load5"] > LOAD5_RESUME
    return sample["load5"] > LOAD5_PAUSE


def admission_refusals(sample):
    return violations(sample) + (
        ["admission: load5>55"] if sample["load5"] > LOAD5_RESUME else []
    )


def snapshot(child=None):
    disk = psutil.disk_usage("/home/ubuntu")
    tree = [psutil.Process()]
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            tree += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for p in {p.pid: p for p in tree}.values():
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.NoSuchProcess:
            pass
    load1, load5, _ = os.getloadavg()
    return dict(
        utc=time.time(),
        load=load1,
        load5=load5,
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        rss=rss,
        threads=threads,
        affinity=sorted(os.sched_getaffinity(0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if (
        socket.gethostname().split(".")[0] != HOST
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or len(os.sched_getaffinity(0)) > 4
    ):
        raise RuntimeError("CPU xbabe1, nice 19, taskset <= 4 cores only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.exists():
        raise RuntimeError("exclusive fresh output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot()
    if admission_refusals(initial):
        raise RuntimeError(
            "prelaunch resource refusal: " + repr(admission_refusals(initial))
        )
    started = time.monotonic()
    if not args.smoke:
        with (CLAIMS / "H232_attempt.claim").open("x") as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    utc=time.time(),
                    compute_stop=COMPUTE_STOP,
                    admission=initial,
                    output=str(args.output.resolve()),
                    launch=str(args.launch.resolve()),
                ),
                f,
            )
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        }
    )
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        command.append("--smoke")
    reasons, pauses, is_paused, paused_seconds, paused_at = [], [], False, 0.0, None
    peak = dict(rss=0, threads=0)
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        resources.open("x") as samples,
    ):
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            sample = snapshot(child)
            sample["elapsed"] = time.monotonic() - started
            sample["paused"] = is_paused
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample)
            if not reasons and paused(sample, is_paused) != is_paused:
                is_paused = not is_paused
                os.killpg(child.pid, signal.SIGSTOP if is_paused else signal.SIGCONT)
                now = time.monotonic()
                if is_paused:
                    paused_at = now
                else:
                    paused_seconds += now - paused_at
                pauses.append(
                    dict(
                        utc=sample["utc"],
                        action="pause" if is_paused else "resume",
                        load5=sample["load5"],
                    )
                )
                print(json.dumps(pauses[-1]), flush=True)
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
                if is_paused:
                    os.killpg(
                        child.pid, signal.SIGCONT
                    )  # a stopped worker must run to receive SIGTERM
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    result = dict(
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        exit_code=code,
        reasons=reasons,
        elapsed_seconds=time.monotonic() - started,
        peak=peak,
        load_pauses=pauses,
        paused_seconds=paused_seconds
        + (time.monotonic() - paused_at if is_paused else 0.0),
        compute_stop=COMPUTE_STOP,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
