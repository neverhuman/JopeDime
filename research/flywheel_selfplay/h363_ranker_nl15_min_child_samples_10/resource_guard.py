"""Owner-corrected H363 guard: full 60s CPU windows and three 60s load samples."""

from pathlib import Path
import datetime, json, os, subprocess, time

D = Path("/home/ubuntu/jope-data/h363-jrb01")
E = Path("/home/ubuntu/jope_plan/evidence/H363/jrb01-20261105-t1125")
DEADLINE = datetime.datetime(2026, 11, 5, 15, 15, tzinfo=datetime.timezone.utc)


def ticks():
    x = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    return sum(x[:8]), x[3] + x[4]


class Window:
    def __init__(self, now, total, idle):
        self.window = (now, total, idle)
        self.last = (now, total, idle)
        self.cpu60 = None
        self.load_streak = 0

    def sample(self, now, total, idle, load, mem, scratch, utc=None):
        t0, a0, i0 = self.last
        diagnostic = 100 * (1 - (idle - i0) / (total - a0)) if total > a0 else None
        self.last = (now, total, idle)
        start, a, i = self.window
        elapsed = now - start
        full = elapsed >= 60.0
        if full:
            assert total > a and idle >= i
            self.cpu60 = 100 * (1 - (idle - i) / (total - a))
            self.window = (now, total, idle)
            self.load_streak = self.load_streak + 1 if load > 108 else 0
        reasons = []
        if mem < 20 * 2**30:
            reasons.append("MemAvailable <20GiB")
        if full and self.cpu60 > 85:
            reasons.append("host CPU >85% averaged over full60s window")
        if full and self.load_streak >= 3:
            reasons.append("load1 >108 on3 consecutive60s samples")
        if scratch > 2_000_000_000:
            reasons.append("own scratch >2GB")
        if utc is not None and utc >= DEADLINE:
            reasons.append("fixed compute deadline15:15Z")
        return dict(
            cpu_percent=diagnostic,
            cpu_5s_diagnostic_only=True,
            cpu_60s_percent=self.cpu60,
            cpu_60s_sampled=full,
            cpu_window_elapsed_seconds=elapsed if full else None,
            load1=load,
            load_60s_sampled=full,
            load_high_streak_60s=self.load_streak,
            mem_available_bytes=mem,
            scratch_bytes=scratch,
            reason=reasons,
        )


def snapshot(state, unit):
    total, idle = ticks()
    mem = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    size = 0
    for p in D.rglob("*"):
        try:
            if p.is_file():
                size += p.stat().st_size
        except FileNotFoundError:
            pass
    now = datetime.datetime.now(datetime.timezone.utc)
    row = state.sample(
        time.monotonic(), total, idle, os.getloadavg()[0], mem, size, now
    )
    row.update(
        utc=now.isoformat(),
        unit=unit,
        guard="CPUavg60>85 OR load1>108 on3consecutive60s OR MemAvail20GiB OR scratch2GB OR fixed15:15deadline",
    )
    return row


def log_row(row, log):
    with Path(log).open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def stop_owned(unit, row):
    assert unit.startswith("h363-") and unit.endswith(".service")
    path = D / "STOP.json" if unit == "h363-run.service" else E / (unit + ".STOP.json")
    with path.open("w") as f:
        json.dump(dict(status="INCOMPLETE_RESOURCE_GUARD", **row), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)


def monitor(unit="h363-run.service"):
    assert unit.startswith("h363-") and unit.endswith(".service")
    log = (
        D / "resource.jsonl"
        if unit == "h363-run.service"
        else E / "OPS-RESOURCES.jsonl"
    )
    total, idle = ticks()
    state = Window(time.monotonic(), total, idle)
    next_sample = time.monotonic()
    try:
        while True:
            row = snapshot(state, unit)
            log_row(row, log)
            if row["reason"]:
                stop_owned(unit, row)
                return
            next_sample += 5
            time.sleep(max(0, next_sample - time.monotonic()))
    except Exception as exc:
        row = dict(
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            unit=unit,
            reason=["resource monitor failure"],
            error=repr(exc),
        )
        stop_owned(unit, row)
