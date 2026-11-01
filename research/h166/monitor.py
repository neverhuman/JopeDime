"""Bounded CPU supervisor for H166 constructor smoke and single wave execution."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ubuntu/jope-data/h166")
MAX_RSS = int(6 * 10**9)
MAX_THREADS = 4
MAX_LOAD = 85.0
MIN_MEM = 16 * 1024**3
MIN_DISK = 40 * 10**9
MAX_DISK_USED_PCT = 95.0
MAX_WALL = 7200
INTERVAL = 2.0


def disk_used_pct(used_bytes: int, free_bytes: int) -> float:
    denom = int(used_bytes) + int(free_bytes)
    return (float(used_bytes) / float(denom)) * 100.0 if denom > 0 else 100.0


def host_state():
    cpus = os.cpu_count() or 1
    load1, load5, load15 = os.getloadavg()
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/home/ubuntu")
    used_pct = disk_used_pct(disk.used, disk.free)
    return dict(
        cpus=cpus,
        load1=load1,
        load5=load5,
        load15=load15,
        load_limit=min(MAX_LOAD, 0.85 * cpus),
        available_bytes=int(mem.available),
        free_disk_bytes=int(disk.free),
        used_disk_pct=float(used_pct),
    )


def tree(proc):
    try:
        items = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return []
    out = []
    for p in items:
        try:
            if p.status() != psutil.STATUS_ZOMBIE:
                out.append(p)
        except psutil.Error:
            pass
    return out


def sample(proc):
    sup = psutil.Process(os.getpid())
    seen = set()
    procs = []
    for p in [sup] + tree(proc):
        if p.pid not in seen:
            seen.add(p.pid)
            procs.append(p)
    rss = 0
    threads = 0
    pids = []
    for p in procs:
        try:
            rss += int(p.memory_info().rss)
            threads += int(p.num_threads())
            pids.append(p.pid)
        except psutil.Error:
            pass
    return dict(pids=pids, rss_bytes=rss, threads=threads, **host_state())


def stop_owned(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in reversed(tree(proc)):
            try:
                os.kill(p.pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custody", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    custody = args.custody.resolve()
    custody.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if args.smoke else "run"
    receipt = custody / f"{mode}-monitor.json"
    log_path = custody / f"{mode}.log"
    out = DATA / mode
    if out.exists():
        raise RuntimeError(f"refusing existing output directory: {out}")
    pre = host_state()
    if (
        pre["load1"] > pre["load_limit"]
        or pre["available_bytes"] < MIN_MEM
        or pre["free_disk_bytes"] < MIN_DISK
        or pre["used_disk_pct"] >= MAX_DISK_USED_PCT
    ):
        receipt.write_text(
            json.dumps(dict(status="PRECHECK_REFUSED", precheck=pre), indent=2) + "\n"
        )
        raise SystemExit(2)

    env = dict(os.environ)
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[k] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JE_ARROW_MALLOC_CONF"] = "background_thread:false"
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-B",
        str(HERE / "run.py"),
        "--output",
        str(out),
    ]
    if args.smoke:
        cmd.append("--smoke")
    else:
        cmd.extend(["--launch", str(custody / "launch.json")])

    samples = []
    reason = None
    started = time.monotonic()
    with log_path.open("w") as log:
        child = subprocess.Popen(
            cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        proc = psutil.Process(child.pid)
        while child.poll() is None:
            snap = sample(proc)
            snap["elapsed_seconds"] = time.monotonic() - started
            samples.append(snap)
            if snap["rss_bytes"] >= MAX_RSS:
                reason = "RSS_CAP"
            elif snap["threads"] > MAX_THREADS:
                reason = "THREAD_CAP"
            elif snap["load1"] > snap["load_limit"]:
                reason = "LOAD_CAP"
            elif snap["available_bytes"] < MIN_MEM:
                reason = "MEM_FLOOR"
            elif (
                snap["free_disk_bytes"] < MIN_DISK
                or snap["used_disk_pct"] >= MAX_DISK_USED_PCT
            ):
                reason = "DISK_FLOOR"
            elif snap["elapsed_seconds"] > MAX_WALL:
                reason = "WALL_CAP"
            if reason:
                stop_owned(proc)
                break
            time.sleep(INTERVAL)
        rc = child.wait()

    summary = dict(
        status="PASS" if rc == 0 and not reason else (reason or "CHILD_FAILED"),
        exit_code=rc,
        abort_reason=reason,
        gpu_lock_held=False,
        output=str(out),
        peak_aggregate_rss_bytes=max((s["rss_bytes"] for s in samples), default=0),
        peak_total_threads=max((s["threads"] for s in samples), default=0),
        min_free_bytes=min(
            (s["free_disk_bytes"] for s in samples), default=pre["free_disk_bytes"]
        ),
        max_used_disk_pct=max(
            (s["used_disk_pct"] for s in samples), default=pre["used_disk_pct"]
        ),
        samples=samples,
    )
    receipt.write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
