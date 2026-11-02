"""H223 terminal CPU guard (adapted from merged H217/H210/H192/H188): supervisor plus worker tree, one exclusive
scientific attempt and at most one preregistered resume.

The worker gets OMP_NUM_THREADS=7 (every other pool 1), and the tree cap is 8.0 cores over a 30 s window (hw_t1325:
worker tree <= 8 cores; one core of headroom for OpenMP helper bursts, as H217 kept 0.5 above 4 threads).

H223 runs on xbabe2 only. Carried over from H217: limits from the EFFECTIVE CPU affinity (os.sched_getaffinity), host
CPU as the mean busy share of the allowed CPUs, the source binding checked BEFORE the claim, and one absolute deadline
recorded in the attempt claim and inherited by the resume (H223_DEADLINE_EPOCH to the worker). The deadline is the
earlier of claim + 4 h and 2026-11-02T17:25:00Z, which leaves the closeout time before hw_t1325's 18:00Z.

Resources (hw_t1325): stop when MemAvailable < 20 GiB (swap is full), disk free < 300 GB, process-tree RSS >= 4 GiB,
or tree CPU > 8.0 cores. Amendment A1 (hw_t1400, pre-outcome): host load PAUSES the worker instead of stopping it.
The monitor sends SIGSTOP to the worker's own process group when load5 > 60 or load1 / host CPU has been > 85% for
300 s, and SIGCONT when load5 <= 55 and both are back under 85%; the deadline still ends a paused attempt.
Admission needs load1 <= 60 and load5 <= 55 (amendment A1), host CPU <= 50%, MemAvailable >= 24 GiB and free >= 320 GB."""

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
ADMIT_LOAD = 60.0  # amendment A1: admission load1 <= 60 ...
ADMIT_LOAD5 = 55.0  # ... and load5 <= 55, the resume threshold, so a launch does not pause at once
ADMIT_CPU = 50.0
ADMIT_FREE = 320_000_000_000  # 20 GB margin above the stop
ADMIT_MEM = 24 * 2**30
MEM_MIN = 20 * 2**30  # hw_t1325: stop if available memory < 20 GiB (xbabe2 swap full)
LOAD5_MAX = 60.0  # amendment A1 (hw_t1400): pause the worker if load5 > 60
LOAD5_RESUME = 55.0  # ... and resume it when load5 <= 55
RSS_MAX = 4 * 2**30  # hw_t1325: process-tree RSS < 4 GiB
MIN_FREE = 300_000_000_000  # hw_t1325: stop if free disk < 300 GB
SUSTAIN = 300.0
CORE_LIMIT = 8.0
THREADS = 7
ALLOWED_HOSTS = (
    "xbabe2",
)  # preregistered host boundary; checked before any output, claim or launch
WINDOW = 30.0
DEADLINE = 4 * 3600
DEADLINE_ENV = "H223_DEADLINE_EPOCH"
RUN_END = (
    1793640300.0  # 2026-11-02T17:25:00Z: closeout must finish by 18:00Z (hw_t1325)
)
CLAIMS = Path.home() / "jope-data/h223"  # attempt claims live on the run host


def violations(sample):
    """Hard aborts (amendment A1, hw_t1400): memory, disk, RSS and the worker core cap. Host load is a pause."""
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
    """Stricter prelaunch gate: start only on a quiet host, with memory and disk margin below the stop rules."""
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
    """Amendment A1 (hw_t1400): host load pauses the worker instead of ending the attempt. Pause when load5 > 60 or
    load1 / host CPU has been over 85% of the effective cores for 300 s; resume only when load5 <= 55 and load1 and
    host CPU are back under their 85% limits (hysteresis, so the worker does not flap)."""
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
        return min(now + DEADLINE, RUN_END)
    attempt = claims / "H223_attempt.claim"
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
        raise RuntimeError(f"H223 runs on xbabe2 at nice 19 only (host {host!r})")
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
    claim = CLAIMS / ("H223_resume.claim" if args.resume else "H223_attempt.claim")
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
        OMP_NUM_THREADS=str(THREADS),
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
                reasons.append("absolute deadline from the attempt claim")
            if not reasons and overloaded(sample, paused) != paused:
                # Amendment A1 (hw_t1400): stop or continue only this worker's own process group.
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
        deadline_epoch=end,
        peak=peak,
        resume=args.resume,
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
