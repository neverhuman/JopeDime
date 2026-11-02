"""H229 resource admission: cooperative dispatch pauses, exact recorded hard stops."""

import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import time

DEADLINE = dt.datetime.fromisoformat("2026-11-02T22:30:00+00:00").timestamp()


def violations(o, paused=False, now=None):
    now = time.time() if now is None else now
    stop = []
    pause = []
    for field in ["rss", "ram", "free", "output", "load5"]:
        if not math.isfinite(o[field]) or o[field] < 0:
            raise RuntimeError(
                "invalid resource observation: " + field + "=" + repr(o[field])
            )
    if o["host"] != "xbabe1" or o["nice"] != 19 or not 0 < o["affinity"] <= 4:
        raise RuntimeError("xbabe1/nice19/at-most4 cores required")
    for field, operator, limit, bad in [
        ("rss", ">", 6 * 2**30, o["rss"] > 6 * 2**30),
        ("ram", "<", 6 * 2**30, o["ram"] < 6 * 2**30),
        ("free", "<", 80_000_000_000, o["free"] < 80_000_000_000),
        ("output", ">", 2_000_000_000, o["output"] > 2_000_000_000),
        ("epoch_utc", ">=", DEADLINE, now >= DEADLINE),
    ]:
        if bad:
            stop.append(
                dict(
                    field=field,
                    value=now if field == "epoch_utc" else o[field],
                    operator=operator,
                    threshold=limit,
                )
            )
    ram_limit = 16 * 2**30 if paused else 12 * 2**30
    load_limit = 60 if paused else 70
    if o["ram"] < ram_limit:
        pause.append(
            dict(field="ram", value=o["ram"], operator="<", threshold=ram_limit)
        )
    if o["load5"] > load_limit:
        pause.append(
            dict(field="load5", value=o["load5"], operator=">", threshold=load_limit)
        )
    return stop, pause


class HardStop(RuntimeError):
    def __init__(self, fields, observation):
        self.fields = fields
        self.observation = observation
        super().__init__(
            json.dumps(
                dict(event="HARD_STOP", violations=fields, observation=observation),
                allow_nan=False,
            )
        )


def admission(o, paused=False, now=None):
    stop, pause = violations(o, paused, now)
    if stop:
        raise HardStop(stop, o)
    return bool(pause)


def record(root, event, o, fields):
    row = dict(
        at=dt.datetime.now(dt.timezone.utc).isoformat(),
        event=event,
        violations=fields,
        observation=o,
        mode="cooperative stop-dispatch between context units",
    )
    with (Path(root) / "supervisor-guard.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return row


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


def shared_admission(root, o, now=None):
    """Carry supervisor pressure across a running unit to the worker dispatch gate."""
    root = Path(root)
    with (root / ".pause-latch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = root / ".pause-latch.json"
        paused = json.loads(state.read_text())["paused"] if state.exists() else False
        result = admission(o, paused, now)
        state.write_text(json.dumps(dict(paused=result)) + "\n")
        return result, violations(o, paused, now)[1]


def wait(root):
    paused = False
    while True:
        o = observation(root)
        try:
            next_paused, fields = shared_admission(root, o)
        except HardStop as error:
            record(root, "HARD_STOP", o, error.fields)
            raise
        if not next_paused:
            if paused:
                record(root, "RESUME_DISPATCH", o, [])
            return
        record(root, "PAUSE_DISPATCH" if not paused else "PAUSE_RECHECK", o, fields)
        paused = True
        print("PAUSE H229", json.dumps(fields), flush=True)
        time.sleep(60)
