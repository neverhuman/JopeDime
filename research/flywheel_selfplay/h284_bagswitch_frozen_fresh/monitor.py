"""Resource monitor wrapper for H284 (xbabe2 CPUs 57-60, nice 19, numeric threads=1, RSS <= 4 GiB)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

WAVE = "H284"
OWNER = "rd_argon"
CPUS = "57-60"
COMPUTE_STOP = 1793759400.0  # 2026-11-04T02:30:00Z
RSS_LIMIT = 4 * 1024**3


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def main():
    out = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".resources.jsonl")
    summary = out.with_suffix(".resource_summary.json")
    claim = out.with_suffix(".attempt.claim")
    pause = out.with_suffix(".pause.json")
    env = dict(os.environ, H265_PAUSE_FILE=str(pause))
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[k] = "1"
    cmd = ["taskset", "-c", CPUS, "nice", "-n", "19", sys.executable, *sys.argv[1:]]
    proc = subprocess.Popen(cmd, env=env)
    claim.write_text(
        json.dumps(
            dict(
                wave=WAVE,
                owner=OWNER,
                pid=proc.pid,
                cpus=CPUS,
                started_at=time.time(),
            ),
            indent=2,
        )
        + "\n"
    )
    peak, samples = 0, 0
    while proc.poll() is None:
        cur = rss(proc.pid)
        peak = max(peak, cur)
        samples += 1
        with log.open("a") as f:
            f.write(json.dumps(dict(t=time.time(), pid=proc.pid, rss=cur)) + "\n")
        if cur > RSS_LIMIT or time.time() > COMPUTE_STOP:
            proc.terminate()
            break
        time.sleep(2.0)
    rc = proc.wait()
    summary.write_text(
        json.dumps(
            dict(
                wave=WAVE,
                owner=OWNER,
                exit_code=rc,
                peak_rss_bytes=peak,
                samples=samples,
            ),
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
