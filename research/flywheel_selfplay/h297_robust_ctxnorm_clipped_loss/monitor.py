from pathlib import Path
import os
import sys
import time
import json
import subprocess
import signal
import psutil
from datetime import datetime, timezone

ROOT = Path("/home/ubuntu/jope-data/h297-astra")
STOP = datetime(2026, 11, 4, 5, 35, tzinfo=timezone.utc).timestamp()
name = sys.argv[1]
cmd = sys.argv[2:]
assert (
    sorted(os.sched_getaffinity(0)) == [100, 101, 102, 103]
    and os.getpriority(os.PRIO_PROCESS, 0) == 19
)
psutil.cpu_percent(None)
peak = {"rss": 0, "threads": 0, "copy_bytes": 0}
events = []
reasons = []
paused = False
last_size = 0
size = 0
start = time.time()
with (
    (ROOT / (name + ".stdout")).open("x") as log,
    (ROOT / (name + ".resources.jsonl")).open("x") as out,
):
    child = subprocess.Popen(
        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    while child.poll() is None:
        procs = [psutil.Process()]
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
        now = time.time()
        mem = psutil.virtual_memory().available
        cpu = psutil.cpu_percent(None)
        if now - last_size > 10:
            size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
            last_size = now
        sample = dict(
            epoch=now,
            rss=rss,
            threads=threads,
            mem_available=mem,
            host_cpu=cpu,
            copy_bytes=size,
            paused=paused,
        )
        out.write(json.dumps(sample) + "\n")
        out.flush()
        for k, v in [("rss", rss), ("threads", threads), ("copy_bytes", size)]:
            peak[k] = max(peak[k], v)
        reasons = [
            label
            for condition, label in [
                (rss > 4 * 2**30, "RSS>4GiB"),
                (threads > 4, "aggregate_threads>4"),
                (size > 3 * 2**30, "copy>3GiB"),
                (now >= STOP, "absolute05:35Zstop"),
            ]
            if condition
        ]
        if reasons:
            if paused:
                os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            break
        pause = (
            (mem < 12 * 2**30 or cpu > 85)
            if not paused
            else not (mem >= 13 * 2**30 and cpu <= 80)
        )
        if pause != paused:
            os.killpg(child.pid, signal.SIGSTOP if pause else signal.SIGCONT)
            paused = pause
            events.append(dict(epoch=now, paused=paused, mem=mem, cpu=cpu))
        time.sleep(0.5)
    code = child.wait()
receipt = dict(
    status="PASS" if code == 0 and not reasons else "INCOMPLETE",
    exit_code=code,
    reasons=reasons,
    peak=peak,
    pause_events=events,
    started_epoch=start,
    finished_epoch=time.time(),
    command=cmd,
    cpu_affinity=[100, 101, 102, 103],
    nice=19,
    absolute_stop=STOP,
)
(ROOT / (name + ".resource_summary.json")).write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt))
sys.exit(0 if receipt["status"] == "PASS" else 1)
