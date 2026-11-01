"""Hard-stop CPU/resource supervisor for the sole H163 attempt."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ubuntu/jope-data/h163")


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")


def sample(child=None):
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
    fs = os.statvfs(DATA.parent)
    used = fs.f_blocks - fs.f_bfree
    return {"rss": rss, "threads": threads, "free": fs.f_bavail * fs.f_frsize,
            "disk_used": used / max(1, used + fs.f_bavail),
            "load": os.getloadavg()[0], "cores": len(os.sched_getaffinity(0)),
            "available": psutil.virtual_memory().available, "utc": time.time()}


def breach(s):
    if s["rss"] >= 6_000_000_000: return "RSS"
    if s["threads"] > 4: return "TOTAL_THREADS"
    if s["free"] < 40_000_000_000 or s["disk_used"] >= .95: return "DISK"
    if s["load"] > .85 * s["cores"]: return "LOAD"
    if s["available"] < 16 * 1024**3: return "MEMORY"
    return None


def stop(child):
    if child.poll() is None:
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: return
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    p.add_argument("--launch", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if os.uname().nodename != "xbabe2":
        raise RuntimeError("H163 is authorized on xbabe2 only")
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    custody = args.custody.resolve()
    custody.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "attempt1"
    pre = sample()
    reason = breach(pre)
    if reason:
        raise RuntimeError("prelaunch hard resource stop: " + reason)
    with (custody / f"{label}.started").open("x") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = DATA / label
    if out.exists():
        raise RuntimeError("output already exists; no resume or retry")
    cmd = [sys.executable, "-B", str(HERE / "run.py"), "--output", str(out)]
    if args.smoke:
        cmd.append("--smoke")
    else:
        if args.launch is None:
            raise RuntimeError("--launch required for scoring")
        cmd += ["--launch", str(args.launch.resolve())]
    child = subprocess.Popen(cmd, env=environment(), stdout=(custody / f"{label}.log").open("w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
    peak = {"rss": 0, "threads": 0, "load": 0.0, "disk_used": 0.0, "min_available": None}
    try:
        with (custody / f"{label}.resources.jsonl").open("w") as f:
            while child.poll() is None:
                s = sample(child)
                for k in ("rss", "threads", "load", "disk_used"):
                    peak[k] = max(peak[k], s[k])
                peak["min_available"] = s["available"] if peak["min_available"] is None else min(peak["min_available"], s["available"])
                f.write(json.dumps(s) + "\n"); f.flush()
                reason = breach(s)
                if reason:
                    raise RuntimeError("hard resource stop: " + reason)
                time.sleep(.5)
        code = child.wait()
        if code:
            raise RuntimeError(f"worker failed: {code}")
        result = "COMPLETE"
    except Exception as e:
        stop(child)
        result = "INCOMPLETE"
        (custody / f"{label}.failure.txt").write_text(str(e) + "\n")
        raise
    finally:
        (custody / f"{label}.supervisor.json").write_text(json.dumps({"result": result, "peak": peak}, indent=2) + "\n")


if __name__ == "__main__":
    main()
