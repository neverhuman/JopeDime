"""H202 host admission and resource supervision, before scientific input reads."""

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
            "H202 excludes this host before scientific input reads: " + host
        )
    return host


def violations(sample, host):
    admit_host(host)
    return [
        name
        for name, bad in [
            ("CPU85", sample["cpu"] > 85),
            ("Mem12GiB", sample["available"] < 12 * 2**30),
            ("RSS5GB", sample["rss"] >= 5e9),
            ("output300MB", sample["output"] >= 300e6),
            ("wall4h", sample["elapsed"] > 14400),
            ("threads3", sample["threads"] > 3),
            ("load108.8", sample["load"] > 108.8),
            ("free40GB", sample["free"] < 40e9),
            ("xbabe2_disk93.6", host == "xbabe2" and sample["used_fraction"] >= 0.936),
        ]
        if bad
    ]


def tree_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # The worker atomically renames its live counter file.
    return total


def stop_owned(child):
    if child.poll() is not None:
        return child.wait()
    try:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError("refusing to signal a process outside its owned group")
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return child.wait()
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def main():
    host = admit_host()
    if host != "xbabe1" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("actual H202 attempt requires xbabe1 at nice19")
    import psutil

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if (
        args.output != Path("/home/ubuntu/jope-data/h202/attempt-001")
        or not args.command
    ):
        raise RuntimeError("one prescribed H202 attempt only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        rss = threads = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
                threads += p.num_threads()
            except psutil.NoSuchProcess:
                pass
        disk = psutil.disk_usage("/home/ubuntu")
        return dict(
            cpu=psutil.cpu_percent(),
            available=psutil.virtual_memory().available,
            rss=rss,
            output=tree_bytes(args.output),
            elapsed=time.monotonic() - started,
            cores=len(os.sched_getaffinity(0)),
            threads=threads,
            load=os.getloadavg()[0],
            free=disk.free,
            used_fraction=disk.used / (disk.used + disk.free),
        )

    first = sample()
    bad = violations(first, host)
    if bad:
        raise RuntimeError("prelaunch guard: " + str(bad))
    claim = args.output.parent / "attempt-001.claim"
    with claim.open("x") as f:
        f.write(
            json.dumps(dict(host=host, pid=os.getpid(), created=time.time())) + "\n"
        )
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
        try:
            while child.poll() is None:
                row = sample()
                resources.write(json.dumps(row) + "\n")
                resources.flush()
                bad = violations(row, host)
                if bad:
                    stop_owned(child)
                    break
                time.sleep(0.5)
        except BaseException as exc:
            bad = ["monitor_exception: " + repr(exc)]
            stop_owned(child)
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
