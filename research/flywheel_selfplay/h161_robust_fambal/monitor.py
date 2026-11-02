"""External supervisor for H161: includes its own thread/RSS and hard-stops on any guard breach."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h159_fail_logit_veto.monitor_common import run as _run_monitor
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
        (s["disk_used"] >= 0.95, "DISK_USED"),
        (s["scratch"] >= 3_000_000_000, "SCRATCH"),
    ]:
        if bad:
            return "abort", why
    if s["load"] > 85 or s["available"] < 16 * 1024**3:
        return "abort", "HOST_PRESSURE"
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
    if fs.f_blocks <= 0:
        raise RuntimeError("filesystem reports no blocks")
    used_blocks = fs.f_blocks - fs.f_bfree
    used_with_user_available = used_blocks + fs.f_bavail
    if used_blocks < 0 or used_with_user_available <= 0:
        raise RuntimeError("filesystem reports invalid block accounting")
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=fs.f_bavail * fs.f_frsize,
        disk_used=used_blocks / used_with_user_available,
        scratch=sum(
            p.stat().st_size
            for tree in (Path("/home/ubuntu/jope-data/h161"),)
            if tree.exists()
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
    _run_monitor(args, HERE, environment, sample, classify, stop)

if __name__ == "__main__":
    main()
