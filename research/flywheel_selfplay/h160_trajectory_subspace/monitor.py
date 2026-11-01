"""H160 bounded single-attempt supervisor adapted from merged H148: process-tree threads/RSS, disk, scratch, load and
memory are hard stops (no pause/resume); a failed attempt is consumed, never resumed."""

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
    for bad, why in [
        (s["rss"] >= 6_000_000_000, "RSS"),
        (s["threads"] > 4, "TOTAL_THREADS"),
        (s["free"] < 40_000_000_000, "DISK"),
        (s["disk_used"] >= 0.95, "DISK_PERCENT"),
        (s["scratch"] >= 1_000_000_000, "SCRATCH"),
        (s["load"] > 0.85 * s["cores"], "LOAD"),
        (s["available"] < 16 * 1024**3, "MEMORY"),
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
    used = fs.f_blocks - fs.f_bfree
    return dict(
        disk_used=used / (used + fs.f_bavail),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=fs.f_bavail * fs.f_frsize,
        scratch=sum(
            p.stat().st_size
            for tree in (root, Path("/home/ubuntu/jope-data/h160"))
            for p in tree.rglob("*")
            if p.is_file()
        ),
        load=os.getloadavg()[0],
        cores=len(os.sched_getaffinity(0)),
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
        "--label",
        default="run",
        help="output label for a full attempt (Amendment A1: run-a2)",
    )
    args = p.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    c = args.custody.resolve()
    c.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else args.label
    if not args.smoke and label not in ("run", "run-a2"):
        raise RuntimeError("registered attempt labels: run, run-a2")
    if (Path("/home/ubuntu/jope-data/h160") / label).exists():
        raise RuntimeError("attempt already exists")
    # Exclusive started ledger: failed attempt consumes it and cannot silently resume.
    with (c / f"{label}.started").open("x") as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    child = None
    record = dict(
        status="WAITING",
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        host=os.uname().nodename,
    )
    started = time.monotonic()
    limit = 900 if args.smoke else 36000
    try:
        s = sample(c)
        record["prelaunch"] = s
        action, why = classify(s)
        if action == "abort":
            raise RuntimeError("prelaunch guard " + why)
        cmd = [
            sys.executable,
            "-B",
            str(HERE / "run.py"),
            "--output",
            str(Path("/home/ubuntu/jope-data/h160") / label),
        ] + (["--smoke"] if args.smoke else ["--launch", str(c / "launch.json")])
        peak = dict(
            threads=0,
            rss=0,
            scratch=0,
            load=0.0,
            disk_used=0.0,
            min_available=None,
            samples=0,
        )
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
                last = 0.0
                while child.poll() is None:
                    s = sample(c, child)
                    if time.monotonic() - started > limit:
                        raise RuntimeError("wall cap")
                    peak.update(
                        threads=max(peak["threads"], s["threads"]),
                        rss=max(peak["rss"], s["rss"]),
                        scratch=max(peak["scratch"], s["scratch"]),
                        load=max(peak["load"], s["load"]),
                        disk_used=max(peak["disk_used"], s["disk_used"]),
                        min_available=s["available"]
                        if peak["min_available"] is None
                        else min(peak["min_available"], s["available"]),
                        samples=peak["samples"] + 1,
                    )
                    record["peak"] = peak
                    if time.monotonic() - last >= 5:
                        resources.write(json.dumps(s) + "\n")
                        resources.flush()
                        last = time.monotonic()
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
        record["wall_seconds"] = time.monotonic() - started
        (c / f"{label}-monitor.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
