"""PLAN §2/§3 resource guard for long CPU packets (P4a dictionary scan, P5 corpus build, the P7 run).

A packet pauses while the shared cpu-admission.json is stale (sampled more than 45 s ago, or more than 5 s in the future),
`admit` is not true, or xbabe2's rolling CPU is at least 75 %. It also pauses new writes while `/` has less than the DL-D7
host floor (300 GB) free. The guard only reads that file and statvfs; it never signals, throttles or restarts anything else.
Pauses are journaled (JSONL, fsynced) and summarized so each packet receipt can report them.

`require_slice_limits()` is the per-launch check of the lane slice aggregate (PLAN §3: CPUQuota 600 %, MemoryMax 12 GiB,
MemorySwapMax 0), set once with `systemctl --user set-property --runtime`.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from dltrain import common

ADMISSION = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/cpu-admission.json")
NODE = "xbabe2"
MAX_AGE_S = 45.0
MAX_FUTURE_S = 5.0
ROLLING_MAX = 75.0
HOST_FLOOR_BYTES = 300 * 10**9
SLICE = "jopedime-dltrain.slice"
SLICE_EXPECT = {"CPUQuotaPerSecUSec": "6s", "MemoryMax": str(12 * 1024**3), "MemorySwapMax": "0"}


class GuardTimeout(RuntimeError):
    pass


def check(now=None, path=ADMISSION, root="/", floor=HOST_FLOOR_BYTES):
    """Return (ok, reasons, observed). Any unreadable or ill-typed field counts against admission."""
    now = time.time() if now is None else now
    reasons, obs = [], {}
    try:
        d = json.loads(Path(path).read_text())
        sampled = d.get("sampled_unix_seconds")
        age = now - float(sampled) if isinstance(sampled, (int, float)) and not isinstance(sampled, bool) else None
        node = (d.get("nodes") or {}).get(NODE) or {}
        rolling = node.get("cpu_rolling_percent")
        obs = {"admit": d.get("admit"), "age_s": None if age is None else round(age, 3),
               "cpu_rolling_percent": rolling if isinstance(rolling, (int, float)) else None}
        if age is None or age > MAX_AGE_S or age < -MAX_FUTURE_S:
            reasons.append("admission_stale")
        if d.get("admit") is not True:
            reasons.append("admit_false")
        if not isinstance(rolling, (int, float)) or isinstance(rolling, bool) or not rolling < ROLLING_MAX:
            reasons.append("cpu_rolling_high")
    except (OSError, ValueError, AttributeError, TypeError):
        reasons.append("admission_unreadable")
    free = shutil.disk_usage(root).free
    obs["root_free_bytes"] = free
    if free < floor:
        reasons.append("host_floor")
    return not reasons, reasons, obs


class Guard:
    """Call tick() inside long loops (cheap: re-reads the admission file at most every check_every_s) and wait() before
    starting a unit of work. Both block while admission is refused."""

    def __init__(self, journal_path=None, poll_s=15.0, check_every_s=10.0, max_pause_s=None,
                 clock=time.time, sleep=time.sleep, checker=check):
        self.journal_path = Path(journal_path) if journal_path else None
        self.poll_s, self.check_every_s, self.max_pause_s = poll_s, check_every_s, max_pause_s
        self.clock, self.sleep, self.checker = clock, sleep, checker
        self.checks, self.pauses, self._last = 0, [], None

    def _journal(self, event, reasons, obs):
        if self.journal_path is None:
            return
        rec = {"utc_us": common.utc_us(), "event": event, "reasons": reasons, "observed": obs}
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _check(self):
        self.checks += 1
        self._last = self.clock()
        return self.checker()

    def wait(self):
        ok, reasons, obs = self._check()
        if ok:
            return 0.0
        start = self.clock()
        seen = set(reasons)
        self._journal("pause", reasons, obs)
        while True:
            if self.max_pause_s is not None and self.clock() - start >= self.max_pause_s:
                self._journal("timeout", reasons, obs)
                raise GuardTimeout(f"admission refused for {self.clock() - start:.0f}s: {sorted(seen)}")
            self.sleep(self.poll_s)
            ok, reasons, obs = self._check()
            if ok:
                break
            seen.update(reasons)
        paused = self.clock() - start
        self.pauses.append({"paused_s": round(paused, 3), "reasons": sorted(seen)})
        self._journal("resume", sorted(seen), obs)
        return paused

    def tick(self):
        if self._last is None or self.clock() - self._last >= self.check_every_s:
            return self.wait()
        return 0.0

    def summary(self):
        return {"checks": self.checks, "pauses": len(self.pauses), "paused_s": round(sum(p["paused_s"] for p in self.pauses), 3),
                "reasons": sorted({r for p in self.pauses for r in p["reasons"]}),
                "rule": f"pause while {ADMISSION.name} older than {MAX_AGE_S:g}s, admit!=true, {NODE} rolling CPU >= {ROLLING_MAX:g}%, "
                        f"or / free < {HOST_FLOOR_BYTES} B"}


def slice_limits(run=subprocess.run):
    r = run(["systemctl", "--user", "show", SLICE, *[a for k in SLICE_EXPECT for a in ("-p", k)]],
            capture_output=True, text=True, timeout=10)
    vals = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return vals, all(vals.get(k) == v for k, v in SLICE_EXPECT.items())


def require_slice_limits(run=subprocess.run):
    vals, ok = slice_limits(run)
    if not ok:
        raise RuntimeError(f"{SLICE} aggregate limits differ from PLAN §3 {SLICE_EXPECT}: {vals}; re-apply with "
                           f"systemctl --user set-property --runtime {SLICE} CPUQuota=600% MemoryMax=12G MemorySwapMax=0")
    return vals


def set_threads(n=4):
    """PLAN §3: torch.set_num_threads(4), pyarrow.set_cpu_count(4), pyarrow.set_io_thread_count(4) when those are present."""
    done = {}
    try:
        import torch
        torch.set_num_threads(n)
        done["torch"] = n
    except ImportError:
        pass
    try:
        import pyarrow
        pyarrow.set_cpu_count(n)
        pyarrow.set_io_thread_count(n)
        done["pyarrow"] = n
    except ImportError:
        pass
    return done


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="dltrain.cpuguard")
    ap.add_argument("cmd", choices=("check", "slice"))
    a = ap.parse_args(argv)
    if a.cmd == "check":
        ok, reasons, obs = check()
        print(json.dumps({"ok": ok, "reasons": reasons, "observed": obs}))
        return 0 if ok else 3
    vals, ok = slice_limits()
    print(json.dumps({"ok": ok, "slice": SLICE, "values": vals, "expect": SLICE_EXPECT}))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(_main())
