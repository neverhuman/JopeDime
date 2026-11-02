"""H233 terminal CPU guard (adapted from merged H223/H217/H210): supervisor plus worker tree, one TRAIN smoke run and
one exclusive primary attempt (plus at most one preregistered resume if interrupted by host guards before 05:00Z).

H233 runs on xbabe2 CPU only at nice 19 (never xbabe0, xbabe1, xbabe3; no GPU, no qualification.lock).
Worker core cap (hw_t2205): taskset <= 4 cores (up to 8 cores when load5 < 50). The monitor enforces both OS
sched_setaffinity on the worker process tree (8 allowed cores when load5 < 50, 4 allowed cores when load5 >= 50)
and a 30 s window tree-CPU hard cap of 8.0 cores.
Hard aborts (hw_t2205): MemAvailable < 20 GiB (swap is full on xbabe2), disk free < 300 GB, process-tree RSS >= 4 GiB,
worker tree CPU > 8.0 cores over 30 s, or reaching the absolute compute stop 2026-11-03T05:00:00Z (epoch 1793682000.0).
Host load pause/resume (hw_t2205): SIGSTOP own worker process group when load5 > 60 (or sustained load1/host CPU > 85%
for 300 s), and SIGCONT when load5 <= 55 and load1/CPU <= 85%. Never kills or pauses other jobs.
"""

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
_shared_path = HERE.parent / "h217_gated025_confirm/monitor.py"
_shared_spec = importlib.util.spec_from_file_location("h217_monitor_runtime", _shared_path)
if _shared_spec is None or _shared_spec.loader is None:
    raise RuntimeError(f"cannot import shared monitor runtime {_shared_path}")
_shared_monitor = importlib.util.module_from_spec(_shared_spec)
sys.modules[_shared_spec.name] = _shared_monitor
_shared_spec.loader.exec_module(_shared_monitor)

AFFINITY = sorted(os.sched_getaffinity(0))
HOST_CORES = len(AFFINITY)  # effective allocation on xbabe2
LOAD_LIMIT = 0.85 * HOST_CORES
ADMIT_LOAD = 60.0
ADMIT_LOAD5 = 55.0
ADMIT_CPU = 50.0
ADMIT_FREE = 320_000_000_000  # 20 GB margin above 300 GB stop
ADMIT_MEM = 24 * 2**30  # 4 GiB margin above 20 GiB stop
MEM_MIN = 20 * 2**30  # hw_t2205: stop if MemAvailable < 20 GiB (xbabe2 swap full)
LOAD5_FAST_MAX = 50.0  # hw_t2205: up to 8 cores if load5 < 50, else <= 4 cores
LOAD5_MAX = 60.0  # hw_t2205: pause own worker if load5 > 60
LOAD5_RESUME = 55.0  # hw_t2205: resume own worker when load5 <= 55
RSS_MAX = 4 * 2**30  # hw_t2205: process-tree RSS < 4 GiB
MIN_FREE = 300_000_000_000  # hw_t2205: stop if free disk < 300 GB
SUSTAIN = 300.0
CORE_LIMIT = 8.0
FAST_CORES = 8
BASE_CORES = 4
THREADS = 7
BASE_THREADS = 3
ALLOWED_HOSTS = ("xbabe2",)
WINDOW = 30.0
DEADLINE = 7 * 3600
DEADLINE_ENV = "H233_DEADLINE_EPOCH"
RUN_END = 1793682000.0  # 2026-11-03T05:00:00Z compute stop (hw_t2205)
CONTINUATION_END = 1793682000.0  # same fixed 05:00:00Z compute stop
CLAIMS = Path.home() / "jope-data/h233"


def worker_affinity_for_load(load5, affinity=None):
    """hw_t2205: taskset <= 4 cores (up to 8 if load5 < 50)."""
    pool = AFFINITY if affinity is None else sorted(affinity)
    limit = FAST_CORES if load5 < LOAD5_FAST_MAX else BASE_CORES
    return pool[: min(limit, len(pool))]


def apply_worker_affinity(child, load5):
    """Apply OS CPU affinity (taskset) to all processes in our own worker tree."""
    target = set(worker_affinity_for_load(load5))
    if not target:
        return
    for proc in tree_of(child):
        try:
            os.sched_setaffinity(proc.pid, target)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def violations(sample):
    """Hard aborts (hw_t2205): memory, disk, RSS, and worker core cap. Host load is a pause."""
    return [
        reason
        for bad, reason in [
            (sample["available"] < MEM_MIN, "MemAvailable<20GiB"),
            (sample["free"] < MIN_FREE, "disk free<300GB"),
            (sample["rss"] >= RSS_MAX, "aggregate RSS>=4GiB"),
            (
                sample["tree_cores"] > CORE_LIMIT,
                "worker tree CPU>8.0 cores (30 s window)",
            ),
        ]
        if bad
    ]


def admission_refusals(sample):
    """Prelaunch gate: start only on a quiet host, with memory and disk margin above the stop rules."""
    return violations(sample) + [
        reason
        for bad, reason in [
            (sample["load"] > ADMIT_LOAD, "admission: load1>60"),
            (sample["load5"] > ADMIT_LOAD5, "admission: load5>55"),
            (sample["cpu"] > ADMIT_CPU, "admission: host CPU>50%"),
            (sample["available"] < ADMIT_MEM, "admission: MemAvailable<24GiB"),
            (sample["free"] < ADMIT_FREE, "admission: disk free<320GB"),
        ]
        if bad
    ]


def overloaded(sample, paused):
    """Host load pauses our own worker instead of ending the attempt: pause when load5 > 60 or sustained
    load1 / host CPU > 85% for 300 s; resume when load5 <= 55 and load1 / host CPU <= 85%."""
    if not paused:
        return (
            sample["load5"] > LOAD5_MAX
            or sample["load_over_s"] >= SUSTAIN
            or sample["cpu_over_s"] >= SUSTAIN
        )
    return not (
        sample["load5"] <= LOAD5_RESUME
        and sample["load"] <= LOAD_LIMIT
        and sample["cpu"] <= 85
    )


def track_overload(sample, since, now):
    """Seconds each host-load signal has been continuously over its 85% limit."""
    for key, over in (
        ("load_over_s", sample["load"] > LOAD_LIMIT),
        ("cpu_over_s", sample["cpu"] > 85),
    ):
        since[key] = (since.get(key) or now) if over else None
        sample[key] = now - since[key] if over else 0.0
    return sample


def attempt_deadline(claims, resume, now):
    """Absolute deadline epoch capped at 2026-11-03T05:00:00Z (RUN_END)."""
    if not resume:
        if now >= RUN_END:
            raise RuntimeError("compute stop 2026-11-03T05:00:00Z has passed")
        return min(now + DEADLINE, RUN_END)
    attempt = claims / "H233_attempt.claim"
    if not attempt.exists():
        raise RuntimeError("resume requires the attempt claim")
    if now >= CONTINUATION_END:
        raise RuntimeError("absolute continuation deadline has passed")
    return CONTINUATION_END


def check_source(launch):
    """Host and source binding through the runner itself, before any claim."""
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


tree_of = _shared_monitor.tree_of


def host_cpu():
    return _shared_monitor.host_cpu(AFFINITY)


def snapshot(child=None, history=None):
    return _shared_monitor.snapshot(
        child,
        history,
        affinity=AFFINITY,
        host_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        window=WINDOW,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", type=Path, required=True, help="log stem outside the run root"
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in ALLOWED_HOSTS or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError(f"H233 runs on xbabe2 at nice 19 only (host {host!r})")
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
    claim_name = (
        "H233_smoke.claim"
        if args.smoke
        else ("H233_resume.claim" if args.resume else "H233_attempt.claim")
    )
    claim = CLAIMS / claim_name
    with claim.open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                utc=time.time(),
                deadline_epoch=end,
                effective_cores=HOST_CORES,
                worker_cores_initial=len(worker_affinity_for_load(initial["load5"])),
                admission=initial,
                output=str(args.output.resolve()),
                root=str(args.root),
                launch=str(args.launch.resolve()),
                resume=args.resume,
                smoke=args.smoke,
            ),
            f,
        )
    init_threads = THREADS if initial["load5"] < LOAD5_FAST_MAX else BASE_THREADS
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
        OMP_NUM_THREADS=str(init_threads),
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
    if args.smoke:
        command.append("--smoke")
    reasons = []
    history = deque()
    since = {}
    peak = dict(rss=0, threads=0, tree_cores=0.0)
    paused, pauses, paused_seconds, paused_at = False, [], 0.0, None
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
        apply_worker_affinity(child, initial["load5"])
        while child.poll() is None:
            sample = track_overload(snapshot(child, history), since, time.monotonic())
            sample["elapsed"] = time.monotonic() - started
            sample["utc"] = time.time()
            apply_worker_affinity(child, sample["load5"])
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample)
            if sample["utc"] >= end:
                reasons.append("absolute deadline from the attempt claim")
            if not reasons and overloaded(sample, paused) != paused:
                paused = not paused
                os.killpg(child.pid, signal.SIGSTOP if paused else signal.SIGCONT)
                now = time.monotonic()
                if paused:
                    paused_at = now
                else:
                    paused_seconds += now - paused_at
                pauses.append(
                    dict(
                        utc=sample["utc"],
                        action="pause" if paused else "resume",
                        load=sample["load"],
                        load5=sample["load5"],
                        cpu=sample["cpu"],
                    )
                )
                print(json.dumps(pauses[-1]), flush=True)
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
                if paused:
                    os.killpg(child.pid, signal.SIGCONT)
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
        smoke=args.smoke,
        effective_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        core_limit=CORE_LIMIT,
        load_pauses=pauses,
        paused_seconds=paused_seconds
        + (time.monotonic() - paused_at if paused else 0.0),
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
