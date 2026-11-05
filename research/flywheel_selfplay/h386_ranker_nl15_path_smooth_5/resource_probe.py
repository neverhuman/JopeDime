"""One fixed 5s union-pool physical/SMT admission; no second capacity sample."""

import json
import subprocess
import time
from pathlib import Path

POOLS = {"A": tuple(range(10)), "A2": (10, 11)}
ASSIGNED = POOLS["A"] + POOLS["A2"]


def selection(report):
    stats = report["sysstat"]["hosts"][0]["statistics"]
    assert len(stats) == 1
    values = {
        int(x["cpu"]): float(x["idle"])
        for x in stats[0]["cpu-load"]
        if x["cpu"] != "all"
    }
    assert all(0 <= v <= 100 for v in values.values())
    idle = {c: min(values[c], values[c + 64]) for c in ASSIGNED}
    passing = {
        name: [c for c in cores if idle[c] >= 85] for name, cores in POOLS.items()
    }
    chosen = "A_UNION_A2"
    return (
        [c for c in ASSIGNED if idle[c] >= 85][:8],
        idle,
        dict(
            chosen_pool=chosen,
            passing_cores_by_pool=passing,
            selection_rule="H386 union A+A2; first<=8 passing physical/SMT pairs over union",
        ),
    )


def eligible_from_report(report):
    eligible, idle, _ = selection(report)
    return eligible, idle


def admission_pass(eligible, host_cpu):
    return (
        len(set(eligible)) == len(eligible)
        and 4 <= len(eligible) <= 8
        and 0 <= host_cpu <= 70
        and set(eligible) <= set(ASSIGNED)
    )


def host_ticks():
    a = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    return sum(a[:8]), a[3] + a[4]


def probe():
    before = host_ticks()
    start = time.monotonic()
    raw = subprocess.check_output(
        [
            "mpstat",
            "-P",
            ",".join(map(str, (*ASSIGNED, *(c + 64 for c in ASSIGNED)))),
            "5",
            "1",
            "-o",
            "JSON",
        ],
        text=True,
    )
    after = host_ticks()
    dt = after[0] - before[0]
    assert dt > 0
    host = 100 * (1 - (after[1] - before[1]) / dt)
    assert 0 <= host <= 100
    eligible, idle, choice = selection(json.loads(raw))
    return (
        eligible,
        idle,
        raw,
        dict(
            cpu_percent=host,
            interval_seconds=time.monotonic() - start,
            aggregate_ticks_before=before,
            aggregate_ticks_after=after,
            method="Same single mpstat interval bracketed by aggregate /proc/stat; no additional per-core interval",
            pool_selection=choice,
        ),
    )
