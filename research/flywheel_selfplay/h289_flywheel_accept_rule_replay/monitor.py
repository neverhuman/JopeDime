"""H289 resource guard: own process tree only; no remote process control."""

import os
import time
import psutil
from datetime import datetime, timezone

COMPUTE_STOP = datetime(2026, 11, 4, 2, 40, tzinfo=timezone.utc).timestamp()


def snapshot(child=None):
    processes = [psutil.Process()]
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            processes += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for p in {p.pid: p for p in processes}.values():
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.NoSuchProcess:
            pass
    return dict(
        utc=time.time(),
        available=psutil.virtual_memory().available,
        rss=rss,
        threads=threads,
        affinity=sorted(os.sched_getaffinity(0)),
        nice=os.getpriority(os.PRIO_PROCESS, 0),
    )


def admission_refusals(s):
    return [
        name
        for condition, name in [
            (s["available"] < 10 * 2**30, "MemAvailable<10GiB"),
            (s["rss"] > 4 * 2**30, "RSS>4GiB"),
            (s["threads"] > 4, "aggregate threads>4"),
            (s["utc"] >= COMPUTE_STOP, "absolute 02:40Z stop"),
            (s["affinity"] != [48, 50, 51, 52], "wrong affinity"),
            (s["nice"] != 19, "wrong nice"),
        ]
        if condition
    ]


class Guard:
    def update(self, s):
        return admission_refusals(s), False
