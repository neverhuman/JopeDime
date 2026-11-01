"""Host-health and child-resource supervisor for the single bounded H172 CPU run."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ubuntu/jope-data/h172")
MAX_LOAD = 85.0
MIN_MEM_AVAILABLE = 16 * 1024**3
MIN_FREE_BYTES = 40 * 1024**3
MAX_USED_PCT = 95.0
MAX_RSS_BYTES = 6 * 1024**3
MAX_THREADS = 4


def mem_available_bytes():
    text = Path("/proc/meminfo").read_text()
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", text, re.M)
    return int(m.group(1)) * 1024


def disk_stats(path):
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used_pct = 100.0 * (1.0 - (st.f_bavail / st.f_blocks))
    return free, total, used_pct


def sample(proc):
    load1, load5, load15 = os.getloadavg()
    mem_avail = mem_available_bytes()
    free, _, used_pct = disk_stats("/home/ubuntu")
    rss = 0
    threads = 0
    try:
        sup = psutil.Process(os.getpid())
        procs = [sup, proc] + proc.children(recursive=True)
    except psutil.Error:
        procs = []
    seen = set()
    for p in procs:
        try:
            if p.pid in seen:
                continue
            seen.add(p.pid)
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.Error:
            pass
    return dict(
        ts=time.time(),
        load1=load1,
        load5=load5,
        load15=load15,
        mem_available_bytes=mem_avail,
        disk_free_bytes=free,
        disk_used_pct=used_pct,
        aggregate_rss_bytes=rss,
        total_threads=threads,
    )


def violation(s):
    if s["load1"] > MAX_LOAD:
        return f"load1 {s['load1']:.2f} > {MAX_LOAD}"
    if s["mem_available_bytes"] < MIN_MEM_AVAILABLE:
        return f"MemAvailable {s['mem_available_bytes']} < {MIN_MEM_AVAILABLE}"
    if s["disk_free_bytes"] < MIN_FREE_BYTES:
        return f"disk free {s['disk_free_bytes']} < {MIN_FREE_BYTES}"
    if s["disk_used_pct"] >= MAX_USED_PCT:
        return f"disk used {s['disk_used_pct']:.2f}% >= {MAX_USED_PCT}%"
    if s["aggregate_rss_bytes"] >= MAX_RSS_BYTES:
        return f"aggregate RSS {s['aggregate_rss_bytes']} >= {MAX_RSS_BYTES}"
    if s["total_threads"] > MAX_THREADS:
        return f"threads {s['total_threads']} > {MAX_THREADS}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custody", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.custody.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "run"
    output = DATA / tag
    log = args.custody / f"{tag}.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        TORCH_NUM_THREADS="1",
        TORCH_NUM_INTEROP_THREADS="1",
        PYTHONHASHSEED="0",
    )
    cmd = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-B",
        str(HERE / "run.py"),
        "--output",
        str(output),
    ]
    if args.smoke:
        cmd.append("--smoke")
    else:
        cmd.extend(["--launch", str(args.custody / "launch.json")])
    samples = []
    abort_reason = None
    with log.open("wb") as log_f:
        child = subprocess.Popen(
            cmd, cwd=HERE, env=env, stdout=log_f, stderr=subprocess.STDOUT
        )
        ps_child = psutil.Process(child.pid)
        while child.poll() is None:
            s = sample(ps_child)
            samples.append(s)
            bad = violation(s)
            if bad is not None:
                abort_reason = bad
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(2.0)
        rc = child.wait()
    summary = dict(
        status="PASS" if rc == 0 and abort_reason is None else "INCOMPLETE",
        exit_code=rc,
        abort_reason=abort_reason,
        gpu_lock_held=False,
        output=str(output),
        peak_aggregate_rss_bytes=max(
            (x["aggregate_rss_bytes"] for x in samples), default=0
        ),
        peak_total_threads=max((x["total_threads"] for x in samples), default=0),
        min_free_bytes=min((x["disk_free_bytes"] for x in samples), default=0),
        max_used_disk_pct=max((x["disk_used_pct"] for x in samples), default=0.0),
        samples=samples,
    )
    (args.custody / f"{tag}-monitor.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if rc != 0 or abort_reason is not None:
        raise SystemExit(abort_reason or f"child failed with {rc}")


if __name__ == "__main__":
    main()
