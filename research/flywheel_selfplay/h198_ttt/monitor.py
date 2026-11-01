"""H198 terminal CPU guard: supervisor plus worker tree, one exclusive scientific attempt.

Enforces prelaunch and running guards: xbabe2 CPU only, nice 19, <= 4 aggregate threads, RSS < 6 GB, 3h science wall cap,
user-visible disk <= 93.8%, MemAvailable >= 16 GiB, and output root < 300 MB.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
HOST_CORES = os.sysconf("SC_NPROCESSORS_ONLN")
LOAD_LIMIT = 0.85 * HOST_CORES
MAX_OUTPUT_BYTES = 300_000_000
MAX_WALL_SECONDS = 3 * 3600


def dir_bytes(root):
    if root is None:
        return 0
    total = 0
    p = Path(root)
    for item in [p, p.with_suffix(".stdout"), p.with_suffix(".resources.jsonl")]:
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
        elif item.is_dir():
            for sub in item.rglob("*"):
                if sub.is_file():
                    try:
                        total += sub.stat().st_size
                    except OSError:
                        pass
    return total


def immediate_violations(sample):
    return [
        reason
        for bad, reason in [
            (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
            (sample["free"] < 40_000_000_000, "disk free<40GB"),
            (sample["disk_used_fraction"] > 0.938, "disk user-visible used>93.8%"),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (sample["threads"] > 4, "aggregate threads>4"),
            (sample["output_bytes"] >= MAX_OUTPUT_BYTES, "output root>=300MB"),
        ]
        if bad
    ]


def violations(sample):
    out = immediate_violations(sample)
    if sample["load"] > LOAD_LIMIT:
        out.append("load>85% of host online cores")
    if sample["cpu"] > 85:
        out.append("host CPU>85%")
    return out


def snapshot(child=None, output_root=None):
    disk = psutil.disk_usage("/home/ubuntu")
    tree = [psutil.Process()]
    if child is not None:
        try:
            p = psutil.Process(child.pid)
            tree += [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss = threads = 0
    for p in {p.pid: p for p in tree}.values():
        try:
            rss += p.memory_info().rss
            threads += p.num_threads()
        except psutil.NoSuchProcess:
            pass
    return dict(
        load=os.getloadavg()[0],
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        free=disk.free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        rss=rss,
        threads=threads,
        output_bytes=dir_bytes(output_root),
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if (
        socket.gethostname().split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2/nice19 only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.exists():
        raise RuntimeError("exclusive fresh output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot(output_root=args.output)
    if violations(initial):
        raise RuntimeError("prelaunch resource refusal: " + repr(violations(initial)))
    started = time.monotonic()
    if not args.smoke:
        with Path("/home/ubuntu/jope_plan/ns/receipts/H198_attempt.claim").open(
            "x"
        ) as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    utc=time.time(),
                    output=str(args.output.resolve()),
                    launch=str(args.launch.resolve()),
                ),
                f,
            )
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        }
    )
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        command.append("--smoke")
    reasons = []
    overload_streak = 0
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        resources.open("x") as samples,
    ):
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            sample = snapshot(child, output_root=args.output)
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            imm = immediate_violations(sample)
            if sample["load"] > LOAD_LIMIT or sample["cpu"] > 85:
                overload_streak += 1
            else:
                overload_streak = 0
            if overload_streak >= 3:
                imm.append("sustained load/CPU>85%")
            if sample["elapsed"] > MAX_WALL_SECONDS:
                imm.append("3h hard wall deadline")
            if imm:
                reasons = imm
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    result = dict(
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        exit_code=code,
        reasons=reasons,
        elapsed_seconds=time.monotonic() - started,
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
        max_output_bytes=MAX_OUTPUT_BYTES,
        max_wall_seconds=MAX_WALL_SECONDS,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
