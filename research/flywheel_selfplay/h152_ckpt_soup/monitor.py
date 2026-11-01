"""External H152 supervisor (adapted from merged H146/H144): process-tree threads/RSS, disk, scratch and load are
hard stops for both the zero-eval smoke and the run; it never resumes or retries."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent


def environment():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        RAYON_NUM_THREADS="1",
        JE_ARROW_MALLOC_CONF="background_thread:false",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
    )


def classify(s):
    """Every limit is a hard stop (H152 brief: stop, do not pause)."""
    for bad, why in [
        (s["rss"] >= 6_000_000_000, "RSS"),
        (s["threads"] > 4, "TOTAL_THREADS"),
        (s["free"] < 40_000_000_000 or s["disk_used"] >= 0.95, "DISK"),
        (s["scratch"] >= 1_000_000_000, "SCRATCH"),
        (s["load"] > 85, "LOAD"),
        (s["available"] < 12 * 1024**3, "MEMORY"),
    ]:
        if bad:
            return "abort", why
    return "run", None


def sample(root, child=None):
    procs = [psutil.Process()]
    if child:
        try:
            p = psutil.Process(child.pid)
            procs += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for p in procs:
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.NoSuchProcess:
            pass
    fs = os.statvfs(root)
    used = 1 - (fs.f_bavail * fs.f_frsize) / (fs.f_blocks * fs.f_frsize)
    return dict(
        disk_used=round(used, 5),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=fs.f_bavail * fs.f_frsize,
        scratch=sum(
            p.stat().st_size
            for tree in {root, Path("/home/ubuntu/jope-data/h152")}
            for p in tree.rglob("*")
            if p.is_file()
        ),
        load=os.getloadavg()[0],
        available=psutil.virtual_memory().available,
    )


def stop(child):
    if child and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            os.killpg(child.pid, signal.SIGCONT)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--launch",
        type=Path,
        help="launch.json binding the merged source (required for --run)",
    )
    args = p.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    c = args.custody.resolve()
    c.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "run"
    root = Path("/home/ubuntu/jope-data/h152") / f"{label}-{c.name}"
    if root.exists() or (c / f"{label}-monitor.json").exists():
        raise RuntimeError("attempt already exists")
    if not args.smoke and args.launch is None:
        raise RuntimeError("--run requires --launch")
    # Exclusive started ledger: a failed attempt consumes it and cannot silently resume.
    with (c / f"{label}.started").open("x") as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    child = None
    record = dict(
        status="WAITING",
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        host=os.uname().nodename,
        limits=dict(
            threads=4,
            rss_bytes=6_000_000_000,
            scratch_bytes=1_000_000_000,
            disk_free_bytes=40_000_000_000,
            disk_used=0.95,
            load1=85,
            mem_available=12 * 1024**3,
        ),
    )
    started = time.monotonic()
    peak = dict(threads=0, rss=0, scratch=0, load=0.0, disk_used=0.0, samples=0)
    try:
        s = sample(c)
        action, why = classify(s)
        record["prelaunch"] = s
        if action == "abort":
            raise RuntimeError("prelaunch guard " + why)
        cmd = [
            sys.executable,
            "-B",
            str(HERE / "runner.py"),
            "--smoke" if args.smoke else "--run",
            "--root",
            str(root),
        ]
        if not args.smoke:
            cmd += ["--launch", str(args.launch.resolve())]
        with (c / f"{label}.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                env=environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(status="RUNNING", command=cmd, pid=child.pid)
            with (c / f"{label}-resources.jsonl").open("w") as resources:
                while child.poll() is None:
                    s = sample(c, child)
                    resources.write(json.dumps(s) + "\n")
                    resources.flush()
                    peak.update(
                        threads=max(peak["threads"], s["threads"]),
                        rss=max(peak["rss"], s["rss"]),
                        scratch=max(peak["scratch"], s["scratch"]),
                        load=max(peak["load"], s["load"]),
                        disk_used=max(peak["disk_used"], s["disk_used"]),
                        samples=peak["samples"] + 1,
                    )
                    action, why = classify(s)
                    if action == "abort":
                        raise RuntimeError("guard " + why)
                    time.sleep(0.25)
            code = child.wait()
            record["exit_status"] = code
            if code:
                raise RuntimeError(f"worker exit {code}")
        record["status"] = "COMPLETE"
    except BaseException as e:
        record.update(status="INCOMPLETE", error=str(e))
        raise
    finally:
        stop(child)
        record["peak"] = peak
        record["wall_seconds"] = time.monotonic() - started
        (c / f"{label}-monitor.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
