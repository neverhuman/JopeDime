"""Hard-stop resource supervisor for H147 CPU execution on xbabe2."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from runtime import (
    ABSOLUTE_DEADLINE_UNIX,
    DISK_FREE_MIN_BYTES,
    DISK_USED_MAX,
    H147_DATA,
    HERE,
    LOAD_MAX,
    MIN_NICE,
    RSS_LIMIT_BYTES,
    SCRATCH_LIMIT_BYTES,
    WALL_SECONDS_CAP,
    atomic_claim,
    save,
    tree_bytes,
)

MEM_AVAILABLE_MIN = 16 * 1024**3


def environment():
    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[k] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def sample(proc, custody):
    children = [proc] + proc.children(recursive=True) if proc else []
    rss = 0
    threads = 1  # Include this supervisor process.
    for p in children:
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.Error:
            pass
    usage = shutil.disk_usage(custody)
    used = 1.0 - usage.free / usage.total
    scratch = sum(tree_bytes(t) for t in {Path(custody).resolve(), H147_DATA.resolve()})
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=usage.free,
        disk_used=round(used, 5),
        scratch=scratch,
        load=os.getloadavg()[0],
        available=psutil.virtual_memory().available,
        now_unix=time.time(),
    )


def classify(s):
    """Hard-stop on any resource or deadline violation."""
    if s.get("now_unix", 0.0) > ABSOLUTE_DEADLINE_UNIX:
        return "abort", "DEADLINE"
    if s["threads"] > 4:
        return "abort", "TOTAL_THREADS"
    if s["rss"] >= RSS_LIMIT_BYTES:
        return "abort", "RSS"
    if s["free"] < DISK_FREE_MIN_BYTES or s.get("disk_used", 0.0) >= DISK_USED_MAX:
        return "abort", "DISK"
    if s["scratch"] >= SCRATCH_LIMIT_BYTES:
        return "abort", "SCRATCH"
    if s["load"] > LOAD_MAX:
        return "abort", "LOAD"
    if s["available"] < MEM_AVAILABLE_MIN:
        return "abort", "MEMORY"
    return "run", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custody", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.nice(max(0, MIN_NICE - os.getpriority(os.PRIO_PROCESS, 0)))
    custody = args.custody.resolve()
    custody.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if args.smoke else "run"
    started_marker = custody / f"{mode}.started"
    atomic_claim(
        started_marker,
        dict(
            mode=mode,
            started_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
        ),
    )

    pre = sample(None, custody)
    state, why = classify(pre)
    if state != "run":
        save(
            custody / f"{mode}-monitor.json",
            dict(status="ABORTED_PRELAUNCH", reason=why, sample=pre),
        )
        raise SystemExit(2)

    out = custody / mode
    cmd = [sys.executable, "-B", str(HERE / "runner.py"), "--output", str(out)]
    if args.smoke:
        cmd.append("--smoke")
    log = (custody / f"{mode}.log").open("w")
    p = subprocess.Popen(
        cmd,
        cwd=HERE,
        env=environment(),
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=lambda: os.setpriority(os.PRIO_PROCESS, 0, MIN_NICE),
    )
    proc = psutil.Process(p.pid)
    peak = dict(threads=1, rss=0, scratch=0, load=0.0, disk_used=0.0, samples=0)
    t0 = time.monotonic()
    aborted = None
    with (custody / f"{mode}-resources.jsonl").open("w") as stream:
        while p.poll() is None:
            s = sample(proc, custody)
            if time.monotonic() - t0 > WALL_SECONDS_CAP:
                aborted = "WALL_CAP"
                p.kill()
                break
            stream.write(__import__("json").dumps(s) + "\n")
            stream.flush()
            peak["samples"] += 1
            for k in ("threads", "rss", "scratch", "load", "disk_used"):
                peak[k] = max(peak[k], s[k])
            act, why = classify(s)
            if act == "abort":
                aborted = why
                p.kill()
                break
            time.sleep(0.25)

    code = p.wait()
    status = (
        "COMPLETE" if code == 0 and not aborted else f"INCOMPLETE_{aborted or code}"
    )
    save(
        custody / f"{mode}-monitor.json",
        dict(
            status=status,
            nice=os.getpriority(os.PRIO_PROCESS, 0),
            host=os.uname().nodename,
            limits=dict(
                threads=4,
                rss_bytes=RSS_LIMIT_BYTES,
                scratch_bytes=SCRATCH_LIMIT_BYTES,
                disk_free_bytes=DISK_FREE_MIN_BYTES,
                disk_used=DISK_USED_MAX,
                load1=LOAD_MAX,
                mem_available=MEM_AVAILABLE_MIN,
                wall_seconds_cap=WALL_SECONDS_CAP,
                deadline_unix=ABSOLUTE_DEADLINE_UNIX,
            ),
            prelaunch=pre,
            command=cmd,
            pid=p.pid,
            exit_status=code,
            aborted_reason=aborted,
            peak=peak,
            wall_seconds=time.monotonic() - t0,
        ),
    )
    if status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
