"""H246 CPU monitor with cooperative H239 A1 memory/load pause guard and sustained-memory hard stop (own worker only).

H246 runs on xbabe2 CPU only at nice 19, taskset 52-55 (never xbabe0, xbabe1, xbabe3; cores 40-51 untouched; no GPU).
Admission: load5 <= 100, MemAvailable >= 30 GiB, disk free >= 300 GB, RSS < 4 GiB.
Cooperative pause (signaled via pause_file to runner at unit/member boundaries; no SIGSTOP):
  - Load: pause when load5 > 108, resume when load5 <= 100.
  - Memory: pause when MemAvailable < 20 GiB, resume when MemAvailable >= 25 GiB sustained continuously for 30 s.
Hard stops:
  - MemAvailable < 6 GiB instant
  - MemAvailable < 12 GiB sustained continuously for 30 s
  - Process-tree RSS >= 4 GiB
  - Disk free < 300 GB
  - Absolute compute stop 2026-11-03T10:00:00Z (epoch 1793700000.0)
"""

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
GIB = 2**30
ALLOWED_HOSTS = ("xbabe2",)
ALLOWED_AFFINITY = [52, 53, 54, 55]
OMP_THREADS = 4
RSS_MAX = 4 * GIB
MIN_FREE = 300_000_000_000
ADMIT_MEM = 30 * GIB
PAUSE_MEM = 20 * GIB
RESUME_MEM = 25 * GIB
STOP_MEM_SUSTAINED = 12 * GIB
STOP_MEM_INSTANT = 6 * GIB
MEM_SUSTAIN_SECONDS = 30.0
ADMIT_LOAD5 = 100.0
PAUSE_LOAD5 = 108.0
RESUME_LOAD5 = 100.0
COMPUTE_STOP = 1793700000.0  # 2026-11-03T10:00:00Z compute stop (hw_t0505)
RUN_END = COMPUTE_STOP
DEADLINE_ENV = "H246_DEADLINE_EPOCH"
PAUSE_ENV = "H246_PAUSE_FILE"
CLAIMS = Path.home() / "jope-data/h246"


def violations(sample):
    """Instant hard-stop violations (sustained <12 GiB is tracked by Guard.update)."""
    return [
        reason
        for bad, reason in [
            (sample["available"] < STOP_MEM_INSTANT, "MemAvailable<6GiB"),
            (sample["free"] < MIN_FREE, "disk free<300GB"),
            (sample["rss"] >= RSS_MAX, "aggregate RSS>=4GiB"),
            (sample.get("utc", 0.0) >= COMPUTE_STOP, "absolute compute stop 10:00Z"),
        ]
        if bad
    ]


def admission_refusals(sample):
    """Prelaunch admission gate: load5 <= 100, MemAvailable >= 30 GiB, disk free >= 300 GB, RSS < 4 GiB."""
    return violations(sample) + [
        reason
        for bad, reason in [
            (sample["load5"] > ADMIT_LOAD5, "admission: load5>100"),
            (sample["available"] < ADMIT_MEM, "admission: MemAvailable<30GiB"),
        ]
        if bad
    ]


class Guard:
    """Cooperative pause/resume state machine + 30 s sustained <12 GiB hard-stop tracker."""

    def __init__(self):
        self.low_12g_since = None
        self.recovered_25g_since = None
        self.memory_pause = False
        self.load_pause = False

    def update(self, sample):
        now = float(sample["utc"])
        available = int(sample["available"])
        load5 = float(sample["load5"])

        if available < STOP_MEM_SUSTAINED:
            if self.low_12g_since is None:
                self.low_12g_since = now
        else:
            self.low_12g_since = None

        reasons = violations(sample)
        if (
            self.low_12g_since is not None
            and now - self.low_12g_since >= MEM_SUSTAIN_SECONDS
        ):
            reasons.append("MemAvailable<12GiB sustained30s")

        if available < PAUSE_MEM:
            self.memory_pause = True
            self.recovered_25g_since = None
        elif self.memory_pause:
            if available >= RESUME_MEM:
                if self.recovered_25g_since is None:
                    self.recovered_25g_since = now
                if now - self.recovered_25g_since >= MEM_SUSTAIN_SECONDS:
                    self.memory_pause = False
                    self.recovered_25g_since = None
            else:
                self.recovered_25g_since = None

        if load5 > PAUSE_LOAD5:
            self.load_pause = True
        elif load5 <= RESUME_LOAD5:
            self.load_pause = False

        return reasons, bool(self.memory_pause or self.load_pause)


def tree_of(child=None):
    procs = [psutil.Process()]
    if child is not None:
        try:
            proc = psutil.Process(child.pid)
            procs += [proc] + proc.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    return list({p.pid: p for p in procs}.values())


def apply_worker_affinity(child):
    target = set(ALLOWED_AFFINITY)
    for proc in tree_of(child):
        try:
            os.sched_setaffinity(proc.pid, target)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def snapshot(child=None):
    rss = threads = 0
    for proc in tree_of(child):
        try:
            rss += proc.memory_info().rss
            threads += proc.num_threads()
        except psutil.NoSuchProcess:
            pass
    load1, load5, _ = os.getloadavg()
    usage = psutil.disk_usage("/home/ubuntu")
    return dict(
        utc=time.time(),
        load=load1,
        load5=load5,
        available=psutil.virtual_memory().available,
        free=usage.free,
        disk_used_fraction=round(usage.used / (usage.used + usage.free), 5),
        rss=rss,
        threads=threads,
        affinity=sorted(os.sched_getaffinity(0)),
    )


def check_source(launch):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", type=Path, required=True, help="log stem outside the run root"
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    host = socket.gethostname().split(".")[0]
    affinity = sorted(os.sched_getaffinity(0))
    if (
        host not in ALLOWED_HOSTS
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or affinity != ALLOWED_AFFINITY
    ):
        raise RuntimeError(
            f"H246 runs on xbabe2 CPU 52-55 at nice 19 only (host={host!r}, affinity={affinity})"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.with_suffix(".stdout").exists():
        raise RuntimeError("exclusive fresh output required")
    check_source(args.launch)
    initial = snapshot()
    if admission_refusals(initial):
        raise RuntimeError(
            "prelaunch resource refusal: " + repr(admission_refusals(initial))
        )
    end = COMPUTE_STOP
    started = time.monotonic()
    CLAIMS.mkdir(parents=True, exist_ok=True)
    claim_name = "H246_smoke.claim" if args.smoke else "H246_attempt.claim"
    claim = CLAIMS / claim_name
    with claim.open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                utc=time.time(),
                deadline_epoch=end,
                affinity=ALLOWED_AFFINITY,
                omp_threads=OMP_THREADS,
                admission=initial,
                output=str(args.output.resolve()),
                root=str(args.root),
                launch=str(args.launch.resolve()),
                smoke=args.smoke,
            ),
            f,
        )
    pause_file = args.output.with_suffix(".pause.json")
    if pause_file.exists():
        pause_file.unlink()
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
        OMP_NUM_THREADS=str(OMP_THREADS),
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        **{
            DEADLINE_ENV: repr(end),
            PAUSE_ENV: str(pause_file.resolve()),
        },
    )
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--root",
        str(args.root),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        command.append("--smoke")

    guard = Guard()
    reasons = []
    peak = dict(rss=0, threads=0)
    is_paused = False
    paused_at = None
    paused_seconds = 0.0
    transitions = []
    sample = initial
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
        apply_worker_affinity(child)
        while child.poll() is None:
            sample = snapshot(child)
            sample["elapsed"] = time.monotonic() - started
            sample["pause_requested"] = is_paused
            apply_worker_affinity(child)
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            for k in peak:
                peak[k] = max(peak[k], sample[k])
            reasons, next_paused = guard.update(sample)
            if next_paused != is_paused:
                if next_paused:
                    pause_file.write_text(json.dumps(sample) + "\n")
                    paused_at = time.monotonic()
                else:
                    if pause_file.exists():
                        pause_file.unlink()
                    paused_seconds += time.monotonic() - paused_at
                    paused_at = None
                is_paused = next_paused
                event = dict(
                    utc=sample["utc"],
                    action="cooperative_pause" if is_paused else "cooperative_resume",
                    available=sample["available"],
                    load5=sample["load5"],
                    memory_pause=guard.memory_pause,
                    load_pause=guard.load_pause,
                )
                transitions.append(event)
                print(json.dumps(event), flush=True)
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    if pause_file.exists():
        pause_file.unlink()
    result = dict(
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        exit_code=code,
        reasons=reasons,
        final_sample=sample,
        admission=initial,
        elapsed_seconds=time.monotonic() - started,
        deadline_epoch=end,
        peak=peak,
        smoke=args.smoke,
        affinity=ALLOWED_AFFINITY,
        omp_threads=OMP_THREADS,
        transitions=transitions,
        paused_seconds=paused_seconds
        + (time.monotonic() - paused_at if paused_at is not None else 0.0),
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
