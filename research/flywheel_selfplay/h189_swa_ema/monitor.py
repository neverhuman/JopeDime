"""H189 terminal CPU guard with df-byte disk ratio and numbered attempts."""

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


def violations(sample):
    return [
        reason
        for bad, reason in [
            (sample["load"] > LOAD_LIMIT, "load>85% of host online cores"),
            (sample["cpu"] > 85, "host CPU>85%"),
            (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
            (sample["free"] < 40_000_000_000, "disk free<40GB"),
            (sample["disk_used_fraction"] >= 0.938, "disk user-visible used>=93.8%"),
            (sample["rss"] >= 6_000_000_000, "aggregate RSS>=6GB"),
            (sample["threads"] > 4, "aggregate threads>4"),
        ]
        if bad
    ]


def snapshot(child=None):
    disk = (
        subprocess.check_output(
            ["df", "-B1", "--output=size,used,avail", "/"], text=True
        )
        .splitlines()[1]
        .split()
    )
    disk_size, disk_used, disk_avail = map(int, disk)
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
        free=disk_avail,
        disk_used=disk_used,
        disk_used_fraction=disk_used / (disk_used + disk_avail),
        disk_size=disk_size,
        rss=rss,
        threads=threads,
        host_online_cores=HOST_CORES,
        load_limit=LOAD_LIMIT,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if (
        socket.gethostname().split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2/nice19 only")
    if not 1 <= args.attempt <= 2:
        raise RuntimeError("H189 permits at most two numbered attempts")
    if not args.smoke:
        launch = json.loads(args.launch.read_text())
        if launch.get("attempt") != args.attempt:
            raise RuntimeError("H189 launch attempt binding mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = args.output.with_suffix(".resources.jsonl")
    if resources.exists() or args.output.exists():
        raise RuntimeError("exclusive fresh output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot()
    if violations(initial):
        raise RuntimeError("prelaunch resource refusal: " + repr(violations(initial)))
    started = time.monotonic()
    if not args.smoke:
        with Path(
            f"/home/ubuntu/jope_plan/ns/receipts/H189_attempt{args.attempt}.claim"
        ).open("x") as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    utc=time.time(),
                    output=str(args.output.resolve()),
                    launch=str(args.launch.resolve()),
                    attempt=args.attempt,
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
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        resources.open("x") as samples,
    ):
        initial["elapsed"] = 0.0
        initial["stage"] = "prelaunch"
        samples.write(json.dumps(initial) + "\n")
        samples.flush()
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            sample = snapshot(child)
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            reasons = violations(sample)
            if sample["elapsed"] > 4 * 3600:
                reasons.append("4h wall deadline")
            if reasons:
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
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
