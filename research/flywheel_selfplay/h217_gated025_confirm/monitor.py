"""H217 terminal CPU guard (adapted from merged H210/H192/H188): supervisor plus worker tree, one exclusive scientific
attempt and at most one preregistered resume.

The worker gets OMP_NUM_THREADS=4 (every other pool 1). OpenMP helper threads briefly lift the tree above 4.0 cores
(H210 was stopped at 4.0136 over a 10 s window), so the tree cap is 4.5 cores over a 30 s window.

H217 runs on xbabe2 only (hw_t0330). Changes from H210, all operational:
- Every load and CPU limit is sized from the EFFECTIVE CPU affinity (os.sched_getaffinity), not the online count
  (H210 E1: 102 allowed of 128 online), and host CPU is the mean busy share of the allowed CPUs.
- One absolute deadline: the attempt claim records claim time + 4 h; the resume inherits it from that claim, and the
  worker receives it in H217_DEADLINE_EPOCH. No clock restarts.
- The source binding is checked (runner.py --check-launch) BEFORE the claim is written (H210 D1).
- xbabe2 swap is full: stop when MemAvailable < 20 GiB; admission needs 24 GiB.

Load gate (unchanged from H210 in shape): admission needs load1 and load5 <= 50% of effective cores, host CPU <= 50%,
MemAvailable >= 24 GiB and disk free >= 220 GB (amendment A1). While running, host overload ends the attempt only when it is sustained:
load5 > 85% of effective cores, or load1 / host CPU > 85% continuously for 300 s. Memory, disk, RSS and the worker's
own core cap stay instantaneous."""

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
AFFINITY = sorted(os.sched_getaffinity(0))
HOST_CORES = len(AFFINITY)  # effective allocation, not SC_NPROCESSORS_ONLN
LOAD_LIMIT = 0.85 * HOST_CORES
ADMIT_LOAD = 0.50 * HOST_CORES
ADMIT_CPU = 50.0
ADMIT_FREE = 220_000_000_000  # amendment A1: 20 GB margin above the stop
ADMIT_MEM = 24 * 2**30
MEM_MIN = 20 * 2**30  # hw_t0330: stop if available memory < 20 GB (xbabe2 swap full)
MIN_FREE = (
    200_000_000_000  # amendment A1 (hw_t0400): on xbabe2 the disk stop is free < 200 GB
)
SUSTAIN = 300.0
CORE_LIMIT = 4.5
ALLOWED_HOSTS = (
    "xbabe2",
)  # preregistered host boundary; checked before any output, claim or launch
WINDOW = 30.0
DEADLINE = 4 * 3600
DEADLINE_ENV = "H217_DEADLINE_EPOCH"
CLAIMS = Path.home() / "jope-data/h217"  # attempt claims live on the run host


def violations(sample):
    return [
        reason
        for bad, reason in [
            (sample["load5"] > LOAD_LIMIT, "load5>85% of effective cores"),
            (
                sample["load_over_s"] >= SUSTAIN,
                "load1>85% of effective cores for 300 s",
            ),
            (sample["cpu_over_s"] >= SUSTAIN, "host CPU>85% for 300 s"),
            (sample["available"] < MEM_MIN, "MemAvailable<20GiB"),
            (sample["free"] < MIN_FREE, "disk free<200GB (A1)"),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (
                sample["tree_cores"] > CORE_LIMIT,
                "worker tree CPU>4.5 cores (30 s window)",
            ),
        ]
        if bad
    ]


def admission_refusals(sample):
    """Stricter prelaunch gate: start only on a quiet host, with memory and disk margin below the stop rules."""
    return violations(sample) + [
        reason
        for bad, reason in [
            (sample["load"] > ADMIT_LOAD, "admission: load1>50% of cores"),
            (sample["load5"] > ADMIT_LOAD, "admission: load5>50% of cores"),
            (sample["cpu"] > ADMIT_CPU, "admission: host CPU>50%"),
            (sample["available"] < ADMIT_MEM, "admission: MemAvailable<24GiB"),
            (sample["free"] < ADMIT_FREE, "admission: disk free<220GB"),
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


def attempt_deadline(claims, resume, now):
    """Absolute deadline epoch. A fresh attempt starts the 4 h budget now; the resume inherits the attempt's deadline
    and is refused once it has passed."""
    if not resume:
        return now + DEADLINE
    attempt = claims / "H217_attempt.claim"
    if not attempt.exists():
        raise RuntimeError("resume requires the attempt claim")
    end = float(json.loads(attempt.read_text())["deadline_epoch"])
    if now >= end:
        raise RuntimeError("absolute deadline from the attempt claim has passed")
    return end


def check_source(launch):
    """Host and source binding through the runner itself, before any claim (H210 D1)."""
    done = subprocess.run(
        [
            sys.executable,
            str(HERE / "runner.py"),
            "--check-launch",
            "--launch",
            str(launch),
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or "SOURCE OK" not in done.stdout:
        raise RuntimeError("prelaunch source binding failed: " + done.stderr[-500:])


def tree_of(child):
    tree = [psutil.Process()]
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            tree += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    return {p.pid: p for p in tree}.values()


def host_cpu():
    """Mean busy percent of the allowed CPUs since the previous call."""
    per = psutil.cpu_percent(percpu=True)
    return sum(per[i] for i in AFFINITY) / HOST_CORES


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
    load1, load5, _ = os.getloadavg()
    return dict(
        load=load1,
        load5=load5,
        cpu=host_cpu(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        rss=rss,
        threads=threads,
        tree_cpu_seconds=cpu_seconds,
        tree_cores=cores,
        host_online_cores=os.cpu_count(),
        effective_cores=HOST_CORES,
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
        raise RuntimeError(f"H217 runs on xbabe2 at nice 19 only (host {host!r})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.with_suffix(".stdout").exists():
        raise RuntimeError("exclusive fresh output required")
    check_source(args.launch)
    host_cpu()
    time.sleep(1)
    initial = track_overload(snapshot(), {}, time.monotonic())
    if admission_refusals(initial):
        raise RuntimeError(
            "prelaunch resource refusal: " + repr(admission_refusals(initial))
        )
    end = attempt_deadline(CLAIMS, args.resume, time.time())
    started = time.monotonic()
    CLAIMS.mkdir(parents=True, exist_ok=True)
    claim = CLAIMS / ("H217_resume.claim" if args.resume else "H217_attempt.claim")
    with claim.open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                utc=time.time(),
                deadline_epoch=end,
                effective_cores=HOST_CORES,
                admission=initial,
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
        OMP_NUM_THREADS="4",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        **{DEADLINE_ENV: repr(end)},
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
            sample["utc"] = time.time()
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample)
            if sample["utc"] >= end:
                reasons.append("absolute 4h deadline from the attempt claim")
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
        deadline_epoch=end,
        peak=peak,
        resume=args.resume,
        effective_cores=HOST_CORES,
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
