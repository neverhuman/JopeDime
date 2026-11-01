"""Terminal resource guard and sole launcher for H169's one CPU attempt."""

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


def violations(sample):
    limits = (
        (sample["load"] > 85, "load1>85"),
        (sample["cpu"] > 85, "host CPU>85%"),
        (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
        (sample["free"] < 40_000_000_000, "free disk<40GB"),
        (sample["disk_used_fraction"] >= 0.95, "disk used>=95%"),
        (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
        (sample["threads"] > 4, "aggregate threads>4"),
    )
    return [reason for bad, reason in limits if bad]


def snapshot(child=None):
    disk = psutil.disk_usage("/home/ubuntu")
    tree = [psutil.Process()]
    if child is not None:
        try:
            root = psutil.Process(child.pid)
            tree += [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for proc in tree:
        try:
            rss += proc.memory_info().rss
            threads += proc.num_threads()
        except psutil.NoSuchProcess:
            pass
    return {
        "load": os.getloadavg()[0],
        "cpu": psutil.cpu_percent(),
        "available": psutil.virtual_memory().available,
        "free": disk.free,
        "disk_used_fraction": disk.used / (disk.used + disk.free),
        "rss": rss,
        "threads": threads,
        "utc": time.time(),
    }


def stop(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if socket.gethostname().split(".")[0] != "xbabe2":
        raise RuntimeError("H169 is authorized on xbabe2 only")
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("nice 19 required")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output.with_suffix(".resources.jsonl")
    log_path = output.with_suffix(".stdout")
    summary_path = output.with_suffix(".resource_summary.json")
    started_path = output.with_suffix(".started")
    if any(path.exists() for path in (output, sample_path, log_path, summary_path, started_path)):
        raise RuntimeError("exclusive output required; H169 never resumes or retries")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot()
    if violations(initial):
        raise RuntimeError("prelaunch guard refused: " + repr(violations(initial)))
    started, deadline = time.monotonic(), time.time() + 12 * 3600
    with started_path.open("x") as marker:
        marker.write(json.dumps({"utc": time.time(), "launch": str(args.launch.resolve()), "smoke": args.smoke}) + "\n")
        marker.flush()
        os.fsync(marker.fileno())
    env = os.environ.copy()
    env.update({key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    )})
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, "-B", str(Path(__file__).with_name("runner.py")),
               "--output", str(output), "--launch", str(args.launch.resolve())]
    if args.smoke:
        command.append("--smoke")
    reason, code = [], None
    peak = {"rss": 0, "threads": 0, "load": 0.0, "cpu": 0.0, "disk_used_fraction": 0.0}
    with log_path.open("x") as log, sample_path.open("x") as samples:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                current = snapshot(child)
                current["elapsed"] = time.monotonic() - started
                samples.write(json.dumps(current) + "\n")
                samples.flush()
                for key in peak:
                    peak[key] = max(peak[key], current[key])
                reason = violations(current)
                if time.time() >= deadline:
                    reason.append("12h administrative cap")
                if reason:
                    stop(child)
                    break
                time.sleep(0.5)
            code = child.wait()
        except BaseException:
            stop(child)
            raise
    summary = {
        "status": "COMPLETE" if code == 0 and not reason else "INCOMPLETE",
        "exit_code": code,
        "reasons": reason,
        "peak": peak,
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
