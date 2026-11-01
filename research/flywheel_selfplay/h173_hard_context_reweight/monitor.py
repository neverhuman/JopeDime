"""H173 resource supervisor: nice 19, <=4 aggregate threads, RSS < 6 GB, load <= 85, MemAvailable >= 16 GiB, free disk >= 40 GB."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ubuntu/jope-data/h173")
LIMITS = dict(
    rss=6_000_000_000,
    threads=4,
    free=40_000_000_000,
    disk_used=0.95,
    scratch=3_000_000_000,
    load=85.0,
    available=16 * 1024**3,
    seconds=12 * 3600,
)


def env():
    e = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        e[k] = "1"
    e["CUDA_VISIBLE_DEVICES"] = ""
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def scratch_bytes(root=DATA):
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def sample(proc=None, sup_proc=None):
    rss = threads = 0
    seen = set()
    roots = []
    if sup_proc is not None:
        roots.append(sup_proc)
    else:
        try:
            roots.append(psutil.Process(os.getpid()))
        except psutil.Error:
            pass
    if proc is not None:
        roots.append(proc)
    for r in roots:
        try:
            items = [r] + r.children(recursive=True)
            for p in items:
                if p.pid in seen:
                    continue
                seen.add(p.pid)
                rss += p.memory_info().rss
                threads += p.num_threads()
        except psutil.Error:
            pass
    du = shutil.disk_usage("/home/ubuntu")
    st = os.statvfs("/home/ubuntu")
    user_total = (st.f_blocks - (st.f_bfree - st.f_bavail)) * st.f_frsize
    disk_used = 1.0 - (du.free / user_total) if user_total > 0 else 1.0
    vm = psutil.virtual_memory()
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        free=du.free,
        disk_used=disk_used,
        scratch=scratch_bytes(),
        load=os.getloadavg()[0],
        available=vm.available,
    )


def classify(s):
    if s["rss"] >= LIMITS["rss"]:
        return "abort", "RSS_LIMIT"
    if s["threads"] > LIMITS["threads"]:
        return "abort", "THREAD_LIMIT"
    if (
        s["free"] < LIMITS["free"]
        or s["disk_used"] >= LIMITS["disk_used"]
        or s["scratch"] >= LIMITS["scratch"]
    ):
        return "abort", "DISK_LIMIT"
    if s["load"] > LIMITS["load"]:
        return "abort", "HOST_LOAD"
    if s["available"] < LIMITS["available"]:
        return "abort", "MEMORY_PRESSURE"
    if s.get("elapsed", 0.0) > LIMITS["seconds"]:
        return "abort", "WALL_TIMEOUT"
    return "run", None


def terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    c = args.custody.resolve()
    c.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "run"
    out = c / label
    if out.exists():
        raise RuntimeError(f"attempt output already exists: {out}")
    with (c / f"{label}.started").open("x") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")
    pre = sample()
    action, reason = classify(pre)
    if action != "run":
        raise RuntimeError(f"preflight guard abort: {reason} {pre}")
    cmd = [sys.executable, "-B", str(HERE / "runner.py"), "--output", str(out)]
    if args.smoke:
        cmd.append("--smoke")
    started = time.monotonic()
    proc = None
    status, stop_reason = "RUNNING", None
    try:
        with (
            (c / f"{label}.log").open("w") as log,
            (c / f"{label}-resources.jsonl").open("w") as rf,
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=HERE,
                env=env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: os.setpriority(os.PRIO_PROCESS, 0, 19),
            )
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                s = sample(ps)
                s["elapsed"] = time.monotonic() - started
                rf.write(json.dumps(s) + "\n")
                rf.flush()
                act, why = classify(s)
                if act != "run":
                    status, stop_reason = "INCOMPLETE", why
                    terminate(proc)
                    break
                time.sleep(0.3)
            code = proc.wait()
            if status == "RUNNING":
                status = "COMPLETE" if code == 0 else "INCOMPLETE"
                stop_reason = None if code == 0 else f"EXIT_{code}"
    finally:
        terminate(proc)
        summary = dict(
            wave="H173",
            mode=label,
            status=status,
            stop_reason=stop_reason,
            pid=proc.pid if proc else None,
            nice=os.getpriority(os.PRIO_PROCESS, 0),
            exit_status=proc.returncode if proc else None,
            wall_seconds=time.monotonic() - started,
        )
        (c / f"{label}-monitor.json").write_text(json.dumps(summary, indent=2) + "\n")
        if status != "COMPLETE":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
