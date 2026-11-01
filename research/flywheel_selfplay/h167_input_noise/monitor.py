"""Terminal host guard; the sole launcher for smoke and the one H167 attempt."""

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


def violations(sample):
    reasons = []
    for bad, reason in [
        (sample["load"] > 85, "raw load>85"),
        (sample["cpu"] > 85, "host CPU>85%"),
        (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
        (sample["free"] < 40_000_000_000, "free<40GB"),
        (sample["disk_used_fraction"] >= 0.95, "disk>=95% user-visible used"),
        (sample["rss"] >= 6_000_000_000, "RSS>=6GB"),
        (sample["threads"] > 4, "aggregate threads>4"),
    ]:
        if bad:
            reasons.append(reason)
    return reasons


def snapshot(process=None):
    disk = psutil.disk_usage("/home/ubuntu")
    tree = [psutil.Process()]
    if process is not None:
        try:
            child = psutil.Process(process.pid)
            tree += [child] + child.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss, threads = 0, 0
    for proc in tree:
        try:
            rss += proc.memory_info().rss
            threads += proc.num_threads()
        except psutil.NoSuchProcess:
            pass
    return dict(
        load=os.getloadavg()[0],
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        rss=rss,
        threads=threads,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if (
        socket.gethostname().split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("xbabe2 nice19 required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resource_path = args.output.with_suffix(".resources.jsonl")
    if resource_path.exists() or args.output.exists():
        raise RuntimeError("exclusive output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot()
    if violations(initial):
        raise RuntimeError("prelaunch resource refusal: " + repr(violations(initial)))
    if not args.smoke:
        marker = Path("/home/ubuntu/jope_plan/ns/receipts/H167_attempt.claim")
        with marker.open("x") as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    launch=str(args.launch.resolve()),
                    output=str(args.output.resolve()),
                    utc=time.time(),
                ),
                f,
            )
    env = os.environ.copy()
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        }
    )
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [
        sys.executable,
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        cmd.append("--smoke")
    started, reasons = time.monotonic(), []
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        resource_path.open("x") as samples,
    ):
        child = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while child.poll() is None:
            sample = snapshot(child)
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            reasons = violations(sample)
            if sample["elapsed"] > 12 * 3600:
                reasons.append("12h administrative cap")
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    summary = dict(
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        exit_code=code,
        reasons=reasons,
        elapsed_seconds=time.monotonic() - started,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
