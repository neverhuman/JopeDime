"""Terminal host guard for H190 on xbabe1; the sole launcher for smoke and the one H190 attempt."""

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


def violations(sample):
    reasons = []
    for bad, reason in [
        (sample["load"] > 85, "raw load>85"),
        (sample["cpu"] > 85, "host CPU>85%"),
        (sample["available"] < 16 * 2**30, "MemAvailable<16GiB"),
        (sample["free"] < 80_000_000_000, "free<80GB"),
        (sample["rss"] >= 8_000_000_000, "RSS>=8GB"),
        (sample["threads"] > 8, "aggregate threads>8"),
        (sample.get("output_bytes", 0) >= 2_000_000_000, "output_dir>=2GB"),
    ]:
        if bad:
            reasons.append(reason)
    return reasons


def _dir_bytes(path):
    if path is None or not Path(path).exists():
        return 0
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def snapshot(process=None, output_dir=None):
    disk = psutil.disk_usage("/home/ubuntu")
    tree = [psutil.Process()]
    if process is not None:
        try:
            child = psutil.Process(process.pid)
            tree += [child] + child.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    rss, threads = 0, 0
    for proc in tree:
        try:
            rss += proc.memory_info().rss
            threads += proc.num_threads()
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
        output_bytes=_dir_bytes(output_dir),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    allowed_hosts = ("xbabe1", "xbabe2") if args.smoke else ("xbabe1",)
    if host not in allowed_hosts or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError(f"{allowed_hosts} nice19 required, got host={host}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resource_path = args.output.with_suffix(".resources.jsonl")
    if resource_path.exists() or args.output.exists():
        raise RuntimeError("exclusive output required")
    psutil.cpu_percent()
    time.sleep(1)
    initial = snapshot(output_dir=args.output)
    if violations(initial):
        raise RuntimeError("prelaunch resource refusal: " + repr(violations(initial)))
    if not args.smoke:
        marker = Path("/home/ubuntu/lane_work/h190-agy/H190_attempt.claim")
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as f:
            json.dump(
                dict(
                    pid=os.getpid(),
                    host=host,
                    launch=str(args.launch.resolve()),
                    output=str(args.output.resolve()),
                    utc=time.time(),
                ),
                f,
            )
    env = os.environ.copy()
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        }
    )
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [
        sys.executable,
        str(HERE / "runner.py"),
        "--output",
        str(args.output),
        "--launch",
        str(args.launch),
    ]
    if args.smoke:
        cmd.append("--smoke")
    started, reasons = time.monotonic(), []
    with (
        args.output.with_suffix(".stdout").open("x", encoding="utf-8") as log,
        resource_path.open("x", encoding="utf-8") as samples,
    ):
        child = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while child.poll() is None:
            sample = snapshot(child, output_dir=args.output)
            sample["elapsed"] = time.monotonic() - started
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            reasons = violations(sample)
            if sample["elapsed"] > 2.5 * 3600:
                reasons.append("2.5h administrative cap")
            if reasons:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    summary = dict(
        status="PASS" if code == 0 and not reasons else "INCOMPLETE",
        host=host,
        exit_code=code,
        reasons=reasons,
        elapsed_seconds=time.monotonic() - started,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
