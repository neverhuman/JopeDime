"""H227 cooperative load pauses and hard resource/deadline admission."""

import datetime as dt
import math
import os
from pathlib import Path
import time

DEADLINE = dt.datetime.fromisoformat("2026-11-02T21:30:00+00:00").timestamp()


def admission(o, paused=False, now=None):
    now = time.time() if now is None else now
    for key in ["rss", "ram", "free", "output", "load5"]:
        if not math.isfinite(o[key]) or o[key] < 0:
            raise RuntimeError("invalid resource observation")
    if o["host"] != "xbabe1" or o["nice"] != 19 or not 0 < o["affinity"] <= 4:
        raise RuntimeError("xbabe1/nice19/at-most4 cores required")
    if (
        o["rss"] > 6 * 2**30
        or o["ram"] < 10 * 2**30
        or o["free"] < 50_000_000_000
        or o["output"] >= 2_000_000_000
        or now >= DEADLINE
    ):
        raise RuntimeError("hard RSS/MemAvailable/disk/output/deadline guard")
    return o["load5"] > 60 if paused else o["load5"] > 70


def observation(root, process=None):
    import psutil

    p = psutil.Process() if process is None else process
    processes = [p, *p.children(recursive=True)]
    rss = 0
    for child in processes:
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return dict(
        at=dt.datetime.now(dt.timezone.utc).isoformat(),
        host=os.uname().nodename.split(".")[0],
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        affinity=len(os.sched_getaffinity(0)),
        rss=rss,
        ram=psutil.virtual_memory().available,
        free=psutil.disk_usage("/").free,
        output=sum(x.stat().st_size for x in Path(root).rglob("*") if x.is_file()),
        load5=os.getloadavg()[1],
    )


def wait(root):
    paused = False
    while True:
        o = observation(root)
        paused = admission(o, paused)
        if not paused:
            return
        with (Path(root) / "pauses.jsonl").open("a") as f:
            import json

            f.write(json.dumps(o) + "\n")
        print("PAUSE H227 load5", o["load5"], flush=True)
        time.sleep(60)
