"""External H146 supervisor (copied from merged H144): includes its own thread/RSS, never resumes cells."""

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
        (s["scratch"] >= 3_000_000_000, "SCRATCH"),
    ]:
        if bad:
            return "abort", why
    if s["load"] > 85 or s["available"] < 12 * 1024**3:
        return "pause", "HOST_PRESSURE"
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
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=fs.f_bavail * fs.f_frsize,
        scratch=sum(
            p.stat().st_size
            for tree in (root, Path("/home/ubuntu/jope-data/h146"))
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
    args = p.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    c = args.custody.resolve()
    c.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "run"
    if (c / label).exists():
        raise RuntimeError("attempt already exists")
    # Exclusive started ledger: failed attempt consumes it and cannot silently resume.
    with (c / f"{label}.started").open("x") as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    child = None
    record = dict(
        status="WAITING",
        stages=[],
        pauses=[],
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        host=os.uname().nodename,
    )
    started = time.monotonic()
    paused = False
    pause_start = None
    try:
        while True:
            s = sample(c)
            action, why = classify(s)
            if action == "abort":
                raise RuntimeError("prelaunch guard " + why)
            if action == "run":
                break
            print("WAIT host pressure", flush=True)
            time.sleep(15)
        cmd = [
            sys.executable,
            "-B",
            str(HERE / "runner.py"),
            "--output",
            str(c / label),
        ] + (["--smoke"] if args.smoke else [])
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
                    action, why = classify(s)
                    if action == "abort":
                        raise RuntimeError("guard " + why)
                    if action == "pause" and not paused:
                        os.killpg(child.pid, signal.SIGSTOP)
                        paused = True
                        pause_start = time.monotonic()
                        record["pauses"].append(dict(start=s["utc"], reason=why))
                    elif paused and s["load"] < 80 and s["available"] >= 12 * 1024**3:
                        os.killpg(child.pid, signal.SIGCONT)
                        paused = False
                        record["pauses"][-1].update(
                            end=s["utc"], seconds=time.monotonic() - pause_start
                        )
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
