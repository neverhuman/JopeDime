"""H195 terminal CPU guard (adapted from merged H192/H188/H184/H180/H176): supervisor plus worker tree, one exclusive
scientific attempt and at most one preregistered resume.

H195 differs from H192 in one guard: HistGBM needs OpenMP, whose runtimes start helper threads beyond the compute
threads (measured: 7-8 OS threads at OMP_NUM_THREADS=2-3). The worker is given OMP_NUM_THREADS=4 (every other pool 1),
and the guard caps the tree's measured CPU at 4.0 cores over a rolling 10 s window instead of the OS thread count,
which is recorded in every sample."""

import argparse
from collections import deque
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
HOST_CORES = os.sysconf("SC_NPROCESSORS_ONLN")
LOAD_LIMIT = 0.85 * HOST_CORES
CORE_LIMIT = 4.0
WINDOW = 10.0
DEADLINE = 4 * 3600
CLAIMS = (
    Path.home() / "jope-data/h195"
)  # attempt claims live on the run host; mirrored to ns/receipts by the owner


def violations(sample, host):
    return [
        reason
        for bad, reason in [
            (sample["load"] > LOAD_LIMIT, "load>85% of host online cores"),
            (sample["cpu"] > 85, "host CPU>85%"),
            (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
            (sample["free"] < 40_000_000_000, "disk free<40GB"),
            (
                host == "xbabe2" and sample["disk_used_fraction"] >= 0.938,
                "xbabe2 disk user-visible used>=93.8%",
            ),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (
                sample["tree_cores"] > CORE_LIMIT,
                "worker tree CPU>4 cores (10 s window)",
            ),
        ]
        if bad
    ]


def tree_of(child):
    tree = [psutil.Process()]
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            tree += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    return {p.pid: p for p in tree}.values()


def snapshot(child=None, history=None):
    disk = psutil.disk_usage("/home/ubuntu")
    rss = threads = 0
    cpu_seconds = 0.0
    for p in tree_of(child):
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
            t = p.cpu_times()
            cpu_seconds += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    now = time.monotonic()
    cores = 0.0
    if history is not None:
        history.append((now, cpu_seconds))
        while len(history) > 1 and now - history[0][0] > WINDOW:
            history.popleft()
        t0, c0 = history[0]
        cores = (cpu_seconds - c0) / (now - t0) if now - t0 >= 1.0 else 0.0
    return dict(
        load=os.getloadavg()[0],
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        rss=rss,
        threads=threads,
        tree_cpu_seconds=cpu_seconds,
        tree_cores=cores,
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", type=Path, required=True, help="log stem outside the run root"
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in ("xbabe1", "xbabe2") or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("CPU xbabe1/xbabe2 at nice 19 only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.with_suffix(".stdout").exists():
        raise RuntimeError("exclusive fresh output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot()
    if violations(initial, host):
        raise RuntimeError(
            "prelaunch resource refusal: " + repr(violations(initial, host))
        )
    started = time.monotonic()
    CLAIMS.mkdir(parents=True, exist_ok=True)
    claim = CLAIMS / ("H195_resume.claim" if args.resume else "H195_attempt.claim")
    with claim.open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                utc=time.time(),
                output=str(args.output.resolve()),
                root=str(args.root),
                launch=str(args.launch.resolve()),
                resume=args.resume,
            ),
            f,
        )
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in [
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        }
    )
    env.update(
        OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
    )
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--root",
        str(args.root),
        "--launch",
        str(args.launch),
    ]
    if args.resume:
        command.append("--resume")
    reasons = []
    history = deque()
    peak = dict(rss=0, threads=0, tree_cores=0.0)
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
            sample = snapshot(child, history)
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample, host)
            if sample["elapsed"] > DEADLINE:
                reasons.append("4h hard wall deadline")
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
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
        resume=args.resume,
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        core_limit=CORE_LIMIT,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
