"""One-shot external resource monitor; signals only its own new process group."""

from pathlib import Path
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

import psutil


THREAD_ENV = dict(
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    NUMEXPR_MAX_THREADS="1",
    TOKENIZERS_PARALLELISM="false",
)


def configure_thread_environment():
    # Called before importing numerical libraries, also inherited by child processes.
    os.environ.update(THREAD_ENV)


def total_threads():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise RuntimeError("Linux total-thread count unavailable")


def single_thread_tokens(builder, net, tables, device):
    # The pinned H46 builder otherwise explicitly requests four native threads.
    factory = builder.base.FusedLandmarkWorkspace

    def bounded_workspace(*args, **kwargs):
        kwargs["num_threads"] = 1
        return factory(*args, **kwargs)

    builder.base.FusedLandmarkWorkspace = bounded_workspace
    try:
        return builder.make_tokens(net, tables, device)
    finally:
        builder.base.FusedLandmarkWorkspace = factory


def token_thread_preflight(builder, net, tables, device):
    before = total_threads()
    if before > 4:
        raise RuntimeError(f"thread preflight before tokens: {before}>4")
    ids = sorted(tables)[:2]
    if not ids:
        raise RuntimeError("thread preflight requires fit contexts")
    tiny = {key: tables[key] for key in ids}
    single_thread_tokens(builder, net, tiny, device)
    after = total_threads()
    if after > 4:
        raise RuntimeError(f"thread preflight after tokens: {after}>4")
    return dict(
        status="PASS",
        contexts=ids,
        before=before,
        after=after,
        total_thread_limit=4,
        native_workspace_threads=1,
    )


def sample(proc=None):
    stat = os.statvfs("/")
    rss = cpu = threads = 0
    if proc:
        try:
            threads = proc.num_threads()
            for child in [proc] + proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                    times = child.cpu_times()
                    cpu += times.user + times.system
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        load1=os.getloadavg()[0],
        disk_used_percent=100 * (1 - stat.f_bavail / stat.f_blocks),
        mem_available=psutil.virtual_memory().available,
        rss=rss,
        tree_cpu_seconds=cpu,
        worker_threads=threads,
    )


def reason(value):
    for rejected, label in (
        (value["load1"] > 86.7, "load1>86.7"),
        (value["disk_used_percent"] >= 94, "disk>=94%"),
        (value["mem_available"] < 30 * 1024**3, "MemAvailable<30GiB"),
        (value["rss"] > 6 * 1024**3, "RSS>6GiB"),
        (value["worker_threads"] > 4, "worker_threads>4"),
    ):
        if rejected:
            return label
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody", required=True, type=Path)
    args = parser.parse_args()
    root = args.custody.resolve()
    if (
        (root / "monitor.json").exists()
        or (root / "run").exists()
        or not (root / "launch.json").exists()
    ):
        raise RuntimeError(
            "one launch only, after merged-source launch receipt; no retry"
        )
    first = sample()
    bad = reason(first)
    result = dict(
        preflight=first,
        limits=dict(
            nice=15,
            threads=4,
            RSS=6 * 1024**3,
            minimum_mem_available=30 * 1024**3,
            load1=86.7,
            disk_used_percent_strict=94,
        ),
    )
    if bad:
        result.update(status="NOT_LAUNCHED_RESOURCE_REFUSAL", reason=bad)
        (root / "monitor.json").write_text(json.dumps(result, indent=2) + "\n")
        return 2
    configure_thread_environment()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        H131_MONITORED="1",
    )
    started = time.monotonic()
    with (
        (root / "worker.log").open("x") as log,
        (root / "resource-samples.jsonl").open("x") as out,
    ):
        child = subprocess.Popen(
            [
                "nice",
                "-n",
                "15",
                sys.executable,
                "-u",
                str(Path(__file__).with_name("runner.py")),
                "--custody",
                str(root),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc = psutil.Process(child.pid)
        result["owned_pid"] = child.pid
        peak, minimum, count = 0, first["mem_available"], 0
        while child.poll() is None:
            value = sample(proc)
            peak, minimum = (
                max(peak, value["rss"]),
                min(minimum, value["mem_available"]),
            )
            count += 1
            out.write(json.dumps(value) + "\n")
            out.flush()
            bad = reason(value)
            if bad:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                result.update(
                    status="ABORTED_RESOURCE_LIMIT", reason=bad, stop_sample=value
                )
                break
            time.sleep(1)
        result.update(
            exit_status=child.wait(),
            peak_tree_rss=peak,
            minimum_mem_available=minimum,
            samples=count,
            wall_seconds=time.monotonic() - started,
            final=sample(),
        )
    result.setdefault("status", "WORKER_EXITED")
    (root / "monitor.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result["exit_status"] or (2 if bad else 0)


if __name__ == "__main__":
    raise SystemExit(main())
