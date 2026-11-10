"""H260 CPU guard: the merged H239 A1 supervisor (cooperative pauses, sustained-memory hard guard, own worker only),
retargeted per hw_t1530 to xbabe2 cores 44-47 and the 17:15Z compute stop.

Admission load5<=100 and MemAvailable>=30 GiB. Cooperative pause (pause file, no SIGSTOP) on load5>108 or MemAvailable<20 GiB;
resume at load5<=100 and MemAvailable>=25 GiB sustained 30 s. Hard stop on MemAvailable<12 GiB sustained 30 s or <6 GiB on
any sample, tree RSS>=4 GiB, disk free<300 GB, >4 tree threads or the compute stop. Signals only its own worker group."""

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
COMPUTE_STOP = 1793726100.0  # 2026-11-03 17:15 UTC (hw_t1530), never reset on resume.
GIB = 2**30
RSS_MAX = 4 * GIB
FREE_MIN = 300_000_000_000
THREADS_MAX = 4


def violations(sample):
    return [
        reason
        for bad, reason in [
            (sample["available"] < 6 * GIB, "MemAvailable<6GiB"),
            (sample["rss"] >= RSS_MAX, "aggregate RSS>=4GiB"),
            (sample["free"] < FREE_MIN, "disk free<300GB"),
            (sample["threads"] > THREADS_MAX, "aggregate threads>4"),
            (sample["utc"] >= COMPUTE_STOP, "absolute compute stop 17:15Z"),
        ]
        if bad
    ]


def admission_refusals(sample):
    return violations(sample) + [
        reason
        for bad, reason in [
            (sample["available"] < 30 * GIB, "admission RAM<30GiB"),
            (sample["load5"] > 100, "admission load5>100"),
        ]
        if bad
    ]


class Guard:
    def __init__(self):
        self.low_since = self.recovered_since = None
        self.memory_pause = self.load_pause = False

    def update(self, sample):
        now, available = sample["utc"], sample["available"]
        if available < 12 * GIB:
            if self.low_since is None:
                self.low_since = now
        else:
            self.low_since = None
        reasons = violations(sample)
        if self.low_since is not None and now - self.low_since >= 30:
            reasons.append("MemAvailable<12GiB sustained30s")
        if available < 20 * GIB:
            self.memory_pause = True
        if self.memory_pause:
            if available >= 25 * GIB:
                if self.recovered_since is None:
                    self.recovered_since = now
                if now - self.recovered_since >= 30:
                    self.memory_pause = False
                    self.recovered_since = None
            else:
                self.recovered_since = None
        if sample["load5"] > 108:
            self.load_pause = True
        elif sample["load5"] <= 100:
            self.load_pause = False
        return reasons, self.memory_pause or self.load_pause


def snapshot(child=None):
    tree = [psutil.Process()]
    if child is not None:
        try:
            process = psutil.Process(child.pid)
            tree += [process] + process.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for process in {p.pid: p for p in tree}.values():
        try:
            rss += process.memory_info().rss
            threads += process.num_threads()
        except psutil.NoSuchProcess:
            pass
    return dict(
        utc=time.time(),
        load5=os.getloadavg()[1],
        available=psutil.virtual_memory().available,
        free=psutil.disk_usage("/home/ubuntu").free,
        rss=rss,
        threads=threads,
        affinity=sorted(os.sched_getaffinity(0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if (
        socket.gethostname().split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or sorted(os.sched_getaffinity(0)) != [44, 45, 46, 47]
    ):
        raise RuntimeError("xbabe2 CPU44-47 nice19 required")
    initial = snapshot()
    if admission_refusals(initial):
        raise RuntimeError("admission refused: " + repr(admission_refusals(initial)))
    launch = json.loads(args.launch.read_text())
    resume_number = launch.get("resume_number", 0)
    if (
        type(resume_number) is not int
        or not 0 <= resume_number <= 2
        or bool(args.resume_from) != (resume_number > 0)
    ):
        raise RuntimeError("maximum two source-bound resumes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".attempt.claim").open("x") as f:
        json.dump(
            dict(
                pid=os.getpid(),
                admission=initial,
                launch=launch,
                compute_stop=COMPUTE_STOP,
            ),
            f,
        )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
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
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from)]
    if args.smoke:
        command.append("--smoke")
    pause_file = args.output.with_suffix(".pause.json")
    guard = Guard()
    transitions = []
    is_paused = False
    paused_at = None
    paused_seconds = 0.0
    peak = dict(rss=0, threads=0)
    started = time.monotonic()
    reasons = []
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        args.output.with_suffix(".resources.jsonl").open("x") as samples,
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
            sample["pause_requested"] = is_paused
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
                    pause_file.unlink()
                    paused_seconds += time.monotonic() - paused_at
                    paused_at = None
                is_paused = next_paused
                event = dict(
                    utc=sample["utc"],
                    action="cooperative_pause" if is_paused else "cooperative_resume",
                    available=sample["available"],
                    load5=sample["load5"],
                )
                transitions.append(event)
                print(json.dumps(event), flush=True)
            if reasons:
                # The child owns a new process group. No SIGSTOP/SIGCONT is used.
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
        final_sample=sample,
        admission=initial,
        elapsed_seconds=time.monotonic() - started,
        peak=peak,
        transitions=transitions,
        pause_requested_seconds=paused_seconds
        + (time.monotonic() - paused_at if paused_at is not None else 0),
        compute_stop=COMPUTE_STOP,
        resume_number=resume_number,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
