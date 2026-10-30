#!/usr/bin/env python3
"""Run the six preregistered H53 pilot cells serially with strict resource guards."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("H53_RUN_ROOT", str(Path.home() / "jope-data/h53/runs")))
STATUS = Path(os.environ.get("H53_STATUS_PATH", str(Path.home() / "jope_plan/ns/H53_STATUS.md")))
LOCK = Path.home() / "jope_plan/qualification.lock"
PILOT_STARTED_UTC = "2026-10-30T17:33:21+00:00"
PILOT_DEADLINE_UTC = "2026-10-30T18:03:21+00:00"
PILOT_SECONDS = 30 * 60
LOCK_FALLBACK_SECONDS = 20 * 60
RSS_LIMIT_KIB = 6 * 1024 * 1024
CELLS = [(seed, scale) for scale in (1.0, 0.1) for seed in (12, 13, 14)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(message: str) -> None:
    line = f"{now()} — {message}\n"
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    print(line.rstrip(), flush=True)


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable unavailable")


def cpu_percent() -> float:
    def sample():
        values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
        return sum(values), values[3] + values[4]
    first = sample()
    time.sleep(1)
    second = sample()
    total = second[0] - first[0]
    return 100.0 * (total - (second[1] - first[1])) / total


def gpu_free_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return int(text.splitlines()[0])


def rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def acquire_lock(deadline: float):
    handle = LOCK.open("r+")
    started = time.monotonic()
    while time.monotonic() < deadline:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle, False
        except BlockingIOError:
            if time.monotonic() - started >= LOCK_FALLBACK_SECONDS:
                handle.close()
                return None, True
            time.sleep(5)
    handle.close()
    return None, False


def run_cell(index: int, seed: int, scale: float, deadline: float) -> bool:
    preflight = {"mem_available_kib": mem_available_kib(), "cpu_utilization_percent": cpu_percent()}
    if preflight["mem_available_kib"] < 20 * 1024 * 1024:
        note(f"STOP before cell {index}: MemAvailable below 20 GiB; {json.dumps(preflight, sort_keys=True)}")
        return False
    if preflight["cpu_utilization_percent"] > 85:
        note(f"STOP before cell {index}: CPU above 85%; {json.dumps(preflight, sort_keys=True)}")
        return False
    handle, fallback = acquire_lock(deadline)
    if time.monotonic() >= deadline:
        note(f"STOP before cell {index}: 30-minute pilot deadline reached waiting for qualification.lock")
        return False
    device = "cpu" if fallback else "cuda:0"
    if device == "cuda:0":
        preflight["gpu_free_mib"] = gpu_free_mib()
        if preflight["gpu_free_mib"] < 20 * 1024:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
            note(f"STOP before cell {index}: GPU free below 20 GiB; {json.dumps(preflight, sort_keys=True)}")
            return False
    out = RUNS / f"scale-{scale:g}" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "research/flywheel_selfplay/h53/runner.py"),
               "--seed", str(seed), "--scale", str(scale), "--rounds", "3",
               "--device", device, "--out", str(out)]
    if handle:
        command.append("--lock-held")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"})
    log_path = out / "run.log"
    note(f"START cell {index}/6 seed={seed} scale={scale:g} device={device}; guard={json.dumps(preflight, sort_keys=True)}")
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=lambda: os.nice(15))
        rss_failure = False
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                proc.terminate()
                break
            current = rss_kib(proc.pid)
            if current is not None and current >= RSS_LIMIT_KIB:
                rss_failure = True
                proc.terminate()
                break
            time.sleep(3)
        code = proc.wait()
    if handle:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    if rss_failure:
        note(f"STOP after cell {index}: RSS reached/exceeded 6 GiB; cell incomplete, later cells not launched")
        return False
    if code != 0:
        note(f"STOP after cell {index}: runner exit={code}; log={log_path}; later cells not launched")
        return False
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["coordinator_preflight"] = preflight
    summary["coordinator_nice"] = 15
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")
    note(f"DONE cell {index}/6 seed={seed} scale={scale:g}; wall={summary['wall_seconds']:.1f}s; utility={summary['best_utility_in_pilot']:.6f}; mean-LCB={summary['family_bootstrap']['mean_lcb_one_sided_95']:.6f}; Elo={summary['elo_vs_r882_diagnostic']['ratings'].get('H53')}")
    return True


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    deadline_text = os.environ.get("H53_DEADLINE_UTC", PILOT_DEADLINE_UTC)
    utc_deadline = datetime.fromisoformat(deadline_text).replace(tzinfo=timezone.utc).timestamp()
    deadline = time.monotonic() + max(0.0, utc_deadline - time.time())
    note("H53 six-cell pilot begins; seeds=12/13/14 x scales=1.0/0.1; rounds=3; total deadline=30 minutes; no selection on DL-Val")
    completed = []
    for index, (seed, scale) in enumerate(CELLS, 1):
        if not run_cell(index, seed, scale, deadline):
            break
        completed.append({"seed": seed, "scale": scale})
    result = {"schema": "jope.h53.pilot-wave/v1", "completed": completed,
              "planned": [{"seed": s, "scale": x} for s, x in CELLS],
              "deadline_seconds": PILOT_SECONDS, "complete": len(completed) == len(CELLS)}
    (RUNS / "pilot-wave.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    note(f"PILOT WAVE STOPPED/COMPLETE: {len(completed)}/6 cells; receipt={RUNS / 'pilot-wave.json'}")


if __name__ == "__main__":
    main()
