"""H226 terminal CPU guard (adapted from merged H217/H223): supervisor plus worker tree, one exclusive scientific
attempt (plus an optional pre-full `--smoke` check on a separate `smoke-*` root) and at most one preregistered resume.

The worker is pinned to <= 4 allowed CPUs (`os.sched_setaffinity`, `OMP_NUM_THREADS=4`, every other pool 1), and the
tree cap is 4.5 cores over a 30 s window (hw_t1435: whole-tree affinity <= 4 cores).

H226 runs on xbabe2 only. Carried over from H217/H223: limits from the EFFECTIVE CPU affinity (`os.sched_getaffinity`),
host CPU as the mean busy share of the allowed CPUs, the source binding checked BEFORE the claim, and one absolute
deadline recorded in the attempt claim and inherited by the resume (`H226_DEADLINE_EPOCH` to the worker). The deadline
is the earlier of claim + 4 h and 2026-11-02T18:30:00Z (`1793644200.0`), leaving 30 minutes for hard closeout by 19:00Z.

Resources (hw_t1435): stop when MemAvailable < 20 GiB (swap is full), disk free < 300 GB, process-tree RSS >= 4 GiB,
worker affinity > 4 cores, or tree CPU > 4.5 cores. Host load PAUSES the worker instead of stopping it: the monitor
sends SIGSTOP to the worker's own process group when load5 > 60 (or load1 / host CPU > 85% for 300 s), rechecks every
60 s, and sends SIGCONT when load5 <= 55 (and load1 / host CPU <= 85%); the deadline still ends a paused attempt.
Admission needs load1 <= 60 and load5 <= 55, host CPU <= 50%, MemAvailable >= 24 GiB and free >= 320 GB."""

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
WORKER_CORES = 4
THREADS = 4
LOAD_LIMIT = 0.85 * HOST_CORES
ADMIT_LOAD = 60.0
ADMIT_LOAD5 = 55.0
ADMIT_CPU = 50.0
ADMIT_FREE = 320_000_000_000  # 20 GB margin above the 300 GB stop
ADMIT_MEM = 24 * 2**30
MEM_MIN = 20 * 2**30  # hw_t1435: stop if available memory < 20 GiB (xbabe2 swap full)
LOAD5_MAX = 60.0  # hw_t1435: pause the worker if load5 > 60
LOAD5_RESUME = 55.0  # ... recheck every 60 s and resume when load5 <= 55
PAUSE_RECHECK = 60.0
RSS_MAX = 4 * 2**30  # hw_t1435: process-tree RSS < 4 GiB
MIN_FREE = 300_000_000_000  # hw_t1435: stop if free disk < 300 GB
SUSTAIN = 300.0
CORE_LIMIT = 4.5
ALLOWED_HOSTS = (
    "xbabe2",
)  # preregistered host boundary; checked before any output, claim or launch
WINDOW = 30.0
DEADLINE = 4 * 3600
DEADLINE_ENV = "H226_DEADLINE_EPOCH"
RUN_END = (
    1793644200.0  # 2026-11-02T18:30:00Z: closeout must finish by 19:00Z (hw_t1435)
)
CLAIMS = Path.home() / "jope-data/h226"  # attempt claims live on the run host


def select_worker_cpus(allowed=AFFINITY, k=WORKER_CORES):
    """Select k least-busy CPUs from the allowed host affinity set for the worker process tree."""
    if len(allowed) <= k:
        return sorted(allowed)
    per = psutil.cpu_percent(percpu=True)
    return sorted(sorted(allowed, key=lambda cpu: (per[cpu], cpu))[:k])


def violations(sample):
    """Hard aborts (hw_t1435): memory, disk, RSS, worker affinity <= 4 cores, and the worker core cap. Host load pauses."""
    return [
        reason
        for bad, reason in [
            (sample["available"] < MEM_MIN, "MemAvailable<20GiB"),
            (sample["free"] < MIN_FREE, "disk free<300GB"),
            (sample["rss"] >= RSS_MAX, "aggregate RSS>=4GiB"),
            (
                sample.get("worker_affinity_cores", WORKER_CORES) > WORKER_CORES,
                "worker affinity>4 cores",
            ),
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
            (sample["load"] > ADMIT_LOAD, "admission: load1>60"),
            (sample["load5"] > ADMIT_LOAD5, "admission: load5>55"),
            (sample["cpu"] > ADMIT_CPU, "admission: host CPU>50%"),
            (sample["available"] < ADMIT_MEM, "admission: MemAvailable<24GiB"),
            (sample["free"] < ADMIT_FREE, "admission: disk free<320GB"),
        ]
        if bad
    ]


def overloaded(sample, paused):
    """Host load pauses the worker instead of ending the attempt. Pause when load5 > 60 or load1 / host CPU has been
    over 85% of the effective cores for 300 s; resume only when load5 <= 55 and load1 and host CPU are back under
    their 85% limits (hysteresis, so the worker does not flap)."""
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
    """Absolute deadline epoch. A fresh attempt starts the 4 h budget now (capped at RUN_END 18:30Z); the resume
    inherits the attempt's deadline and is refused once it has passed."""
    if not resume:
        return min(now + DEADLINE, RUN_END)
    attempt = claims / "H226_attempt.claim"
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
    worker_aff = WORKER_CORES
    if child is not None:
        try:
            worker_aff = len(os.sched_getaffinity(child.pid))
        except ProcessLookupError:
            worker_aff = WORKER_CORES
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
        worker_affinity_cores=worker_aff,
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in ALLOWED_HOSTS or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError(f"H226 runs on xbabe2 at nice 19 only (host {host!r})")
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
    worker_cpus = select_worker_cpus(AFFINITY, WORKER_CORES)
    claim_name = (
        "H226_smoke.claim"
        if args.smoke
        else ("H226_resume.claim" if args.resume else "H226_attempt.claim")
    )
    claim = CLAIMS / claim_name
    with claim.open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                utc=time.time(),
                deadline_epoch=end,
                effective_cores=HOST_CORES,
                worker_cpus=worker_cpus,
                admission=initial,
                output=str(args.output.resolve()),
                root=str(args.root),
                launch=str(args.launch.resolve()),
                resume=args.resume,
                smoke=args.smoke,
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
        OMP_THREAD_LIMIT=str(THREADS),
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
    paused, pauses, paused_seconds, paused_at, next_resume_check = (
        False,
        [],
        0.0,
        None,
        0.0,
    )
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
            preexec_fn=lambda: os.sched_setaffinity(0, worker_cpus),
        )
        while child.poll() is None:
            now_mono = time.monotonic()
            sample = track_overload(snapshot(child, history), since, now_mono)
            sample["elapsed"] = now_mono - started
            sample["utc"] = time.time()
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons = violations(sample)
            if sample["utc"] >= end:
                reasons.append("absolute deadline from the attempt claim")
            if not reasons:
                want_pause = overloaded(sample, paused)
                if not paused and want_pause:
                    paused = True
                    paused_at = now_mono
                    next_resume_check = now_mono + PAUSE_RECHECK
                    os.killpg(child.pid, signal.SIGSTOP)
                    pauses.append(
                        dict(
                            utc=sample["utc"],
                            action="pause",
                            load=sample["load"],
                            load5=sample["load5"],
                            cpu=sample["cpu"],
                        )
                    )
                    print(json.dumps(pauses[-1]), flush=True)
                elif paused and now_mono >= next_resume_check:
                    next_resume_check = now_mono + PAUSE_RECHECK
                    if not want_pause:
                        paused = False
                        paused_seconds += now_mono - paused_at
                        os.killpg(child.pid, signal.SIGCONT)
                        pauses.append(
                            dict(
                                utc=sample["utc"],
                                action="resume",
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
        smoke=args.smoke,
        effective_cores=HOST_CORES,
        worker_cpus=worker_cpus,
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
