"""H201 host admission and resource supervision, before scientific input reads."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def admit_host(host=None):
    host = socket.gethostname().split(".")[0] if host is None else host
    if host not in ("xbabe1", "xbabe2"):
        raise RuntimeError(
            "H201 excludes this host before scientific input reads: " + host
        )
    return host


def violations(sample, host):
    admit_host(host)
    return [
        name
        for name, bad in [
            ("CPU85", sample["cpu"] > 85),
            ("Mem12GiB", sample["available"] < 12 * 2**30),
            ("RSS4GB", sample["rss"] >= 4e9),
            ("output200MB", sample["output"] >= 200e6),
            ("wall90min", sample["elapsed"] > 5400),
            ("cores2", sample["cores"] > 2),
            ("free40GB", sample["free"] < 40e9),
            ("xbabe2_disk93.6", host == "xbabe2" and sample["used_fraction"] >= 0.936),
        ]
        if bad
    ]


def main():
    host = admit_host()
    if os.getpriority(os.PRIO_PROCESS, 0) != 19 or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError("nice19 and at most two pinned cores required")
    import psutil

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if (
        args.output != Path("/home/ubuntu/jope-data/h201/attempt-001")
        or not args.command
    ):
        raise RuntimeError("one prescribed H201 attempt only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    claim = args.output.parent / "attempt-001.claim"
    with claim.open("x") as f:
        f.write(
            json.dumps(dict(host=host, pid=os.getpid(), created=time.time())) + "\n"
        )
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "GOMAXPROCS",
            ]
        }
    )
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    child = None
    started = time.monotonic()
    psutil.cpu_percent()
    time.sleep(1)

    def sample():
        processes = [psutil.Process()]
        if child:
            try:
                p = psutil.Process(child.pid)
                processes += [p] + p.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        rss = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        disk = psutil.disk_usage("/home/ubuntu")
        return dict(
            cpu=psutil.cpu_percent(),
            available=psutil.virtual_memory().available,
            rss=rss,
            output=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
            elapsed=time.monotonic() - started,
            cores=len(os.sched_getaffinity(0)),
            free=disk.free,
            used_fraction=disk.used / (disk.used + disk.free),
        )

    first = sample()
    bad = violations(first, host)
    if bad:
        raise RuntimeError("prelaunch guard: " + str(bad))
    with (
        args.output.with_suffix(".stdout").open("x") as log,
        args.output.with_suffix(".resources.jsonl").open("x") as resources,
    ):
        child = subprocess.Popen(
            args.command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while child.poll() is None:
            row = sample()
            resources.write(json.dumps(row) + "\n")
            resources.flush()
            bad = violations(row, host)
            if bad:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    result = dict(
        status="PASS" if code == 0 and not bad else "INCOMPLETE",
        exit_code=code,
        reasons=bad,
        host=host,
        elapsed=time.monotonic() - started,
    )
    args.output.with_suffix(".resource_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
