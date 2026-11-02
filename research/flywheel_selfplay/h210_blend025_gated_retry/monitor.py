"""H210 terminal CPU guard (adapted from merged H192/H188/H184/H180/H176): supervisor plus worker tree, one exclusive
scientific attempt and at most one preregistered resume.

H210 differs from H192 in one guard: HistGBM needs OpenMP, whose runtimes start helper threads beyond the compute
threads (measured: 7-8 OS threads at OMP_NUM_THREADS=2-3). The worker is given OMP_NUM_THREADS=4 (every other pool 1),
and the guard caps the tree's measured CPU at 4.0 cores over a rolling 10 s window instead of the OS thread count,
which is recorded in every sample. H210 admits xbabe1 only (PREREG); xbabe2 is excluded.

H210 load gate (H203 lost both attempts to one-minute load spikes from other jobs on xbabe2): admission needs load1 and
load5 <= 50% of cores, host CPU <= 50% and disk used < 93.5%. While running, host overload ends the attempt only when
it is sustained: load5 > 85% of cores, or load1 / host CPU > 85% continuously for 300 s. Memory, disk, RSS and the
worker's own core cap stay instantaneous."""

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
ADMIT_LOAD = 0.50 * HOST_CORES
ADMIT_CPU = 50.0
ADMIT_USED = 0.935
MAX_USED = 0.938  # lanes_t1615 disk guard
SUSTAIN = 300.0
CORE_LIMIT = 4.0
ALLOWED_HOSTS = (
    "xbabe1",
)  # preregistered host boundary; checked before any output, claim or launch
WINDOW = 10.0
DEADLINE = 4 * 3600
CLAIMS = (
    Path.home() / "jope-data/h210"
)  # attempt claims live on the run host; mirrored to ns/receipts by the owner


def violations(sample):
    return [
        reason
        for bad, reason in [
            (sample["load5"] > LOAD_LIMIT, "load5>85% of host online cores"),
            (
                sample["load_over_s"] >= SUSTAIN,
                "load1>85% of host online cores for 300 s",
            ),
            (sample["cpu_over_s"] >= SUSTAIN, "host CPU>85% for 300 s"),
            (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
            (sample["free"] < 40_000_000_000, "disk free<40GB"),
            (sample["disk_used_fraction"] >= MAX_USED, "disk used>=93.8%"),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (
                sample["tree_cores"] > CORE_LIMIT,
                "worker tree CPU>4 cores (10 s window)",
            ),
        ]
        if bad
    ]


def admission_refusals(sample):
    """Stricter prelaunch gate: start only on a quiet xbabe1, with disk margin below the 93.8% guard."""
    return violations(sample) + [
        reason
        for bad, reason in [
            (sample["load"] > ADMIT_LOAD, "admission: load1>50% of cores"),
            (sample["load5"] > ADMIT_LOAD, "admission: load5>50% of cores"),
            (sample["cpu"] > ADMIT_CPU, "admission: host CPU>50%"),
            (sample["disk_used_fraction"] >= ADMIT_USED, "admission: disk used>=93.5%"),
        ]
        if bad
    ]


def track_overload(sample, since, now):
    """Seconds each host-load signal has been continuously over its 85% limit; `since` holds the start times."""
    for key, over in (
        ("load_over_s", sample["load"] > LOAD_LIMIT),
        ("cpu_over_s", sample["cpu"] > 85),
    ):
        since[key] = (since.get(key) or now) if over else None
        sample[key] = now - since[key] if over else 0.0
    return sample


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
        load5=os.getloadavg()[1],
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
    if host not in ALLOWED_HOSTS or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError(f"H210 runs on xbabe1 at nice 19 only (host {host!r})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.with_suffix(".stdout").exists():
        raise RuntimeError("exclusive fresh output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = track_overload(snapshot(), {}, time.monotonic())
    if admission_refusals(initial):
        raise RuntimeError(
            "prelaunch resource refusal: " + repr(admission_refusals(initial))
        )
    started = time.monotonic()
    CLAIMS.mkdir(parents=True, exist_ok=True)
    claim = CLAIMS / ("H210_resume.claim" if args.resume else "H210_attempt.claim")
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
    since = {}
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
            sample = track_overload(snapshot(child, history), since, time.monotonic())
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample)
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
