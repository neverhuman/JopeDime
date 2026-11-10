"""Bound only H307 child processes on xbabe2; no foreign process control."""

import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parent.parent
STOP = 1793781900  # 2026-11-04T08:45Z; never reset on resume.
assert os.uname().nodename == "xbabe2"
assert sorted(os.sched_getaffinity(0)) == [60, 61, 62, 63]
assert os.getpriority(os.PRIO_PROCESS, 0) == 19


def scratch_bytes():
    return sum(
        f.stat().st_size
        for f in Path("/home/ubuntu/jope-data/h307-astra").rglob("*")
        if f.is_file()
    )


name = sys.argv[1]
cmd = sys.argv[3:] if sys.argv[2] == "--exec" else [sys.executable, *sys.argv[2:]]
assert psutil.virtual_memory().available >= 20 * 2**30 and time.time() < STOP
assert psutil.cpu_percent(0.25) <= 85
source_hashes = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (ROOT / "source").glob("*.py")
}
started = time.time()
peak = {"rss": 0, "threads": 0, "host_cpu": 0}
reasons = []
with (
    (ROOT / f"{name}.stdout").open("x") as log,
    (ROOT / f"{name}.resources.jsonl").open("x") as samples,
):
    child = subprocess.Popen(
        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    while child.poll() is None:
        procs = [psutil.Process()]
        try:
            proc = psutil.Process(child.pid)
            procs += [proc, *proc.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
        rss = threads = 0
        for proc in procs:
            try:
                rss += proc.memory_info().rss
                threads += proc.num_threads()
            except psutil.NoSuchProcess:
                pass
        sample = dict(
            epoch=time.time(),
            rss=rss,
            threads=threads,
            host_cpu=psutil.cpu_percent(None),
            available=psutil.virtual_memory().available,
        )
        samples.write(json.dumps(sample) + "\n")
        samples.flush()
        for key in peak:
            peak[key] = max(peak[key], sample[key])
        reasons = [
            label
            for condition, label in [
                (scratch_bytes() > 2_000_000_000, "scratch>2GB"),
                (rss > 4 * 2**30, "RSS>4GiB"),
                (sample["host_cpu"] > 85, "hostCPU>85percent"),
                (sample["available"] < 20 * 2**30, "MemAvailable<20GiB"),
                (sample["epoch"] >= STOP, "absolute08:45Zstop"),
            ]
            if condition
        ]
        if reasons:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            break
        time.sleep(0.5)
    code = child.wait()
receipt = dict(
    source_sha256=source_hashes,
    status="PASS" if code == 0 and not reasons else "INCOMPLETE",
    exit_code=code,
    reasons=reasons,
    peak=peak,
    command=cmd,
    started_epoch=started,
    finished_epoch=time.time(),
    cpu_affinity=[60, 61, 62, 63],
    nice=19,
    numeric_threads=1,
    compute_stop=STOP,
)
(ROOT / f"{name}.resource_summary.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt))
sys.exit(0 if receipt["status"] == "PASS" else 1)
