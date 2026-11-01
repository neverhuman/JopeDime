"""H155 CPU supervisor; signals only its own worker, records the fixed attempt."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ubuntu/jope-data/h155")


def main():
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.started").open("x") as f:
        f.write(str(time.time()) + "\n")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        RAYON_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    worker = None
    paused = False
    report = {
        "status": "STARTED",
        "samples": 0,
        "max_cpu_percent": 0.0,
        "max_threads": 0,
        "max_rss": 0,
        "pauses": 0,
        "CPU_only": True,
        "GPU_jobs": 0,
    }
    try:
        with (
            (root / "run.log").open("x") as log,
            (root / "resources.jsonl").open("x") as resource_log,
        ):
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "runner.py"),
                    "--output",
                    str(root / "run"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            report["pid"] = worker.pid
            psutil.cpu_percent()
            while worker.poll() is None:
                cpu = psutil.cpu_percent(interval=1.0)
                procs = [psutil.Process()]
                try:
                    child = psutil.Process(worker.pid)
                    procs += [child] + child.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                threads = rss = 0
                for proc in procs:
                    try:
                        threads += proc.num_threads()
                        rss += proc.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                v = os.statvfs(root)
                free = v.f_bavail * v.f_frsize
                used = 100 * (1 - v.f_bfree / v.f_blocks)
                sample = dict(
                    utc_unix=time.time(),
                    host_cpu_percent=cpu,
                    process_threads=threads,
                    process_rss=rss,
                    free_bytes=free,
                    used_percent=used,
                    paused=paused,
                )
                resource_log.write(json.dumps(sample) + "\n")
                resource_log.flush()
                report["samples"] += 1
                report["max_cpu_percent"] = max(report["max_cpu_percent"], cpu)
                report["max_threads"] = max(report["max_threads"], threads)
                report["max_rss"] = max(report["max_rss"], rss)
                if (
                    free < 40_000_000_000
                    or used >= 95
                    or threads > 4
                    or rss >= 6_000_000_000
                ):
                    raise RuntimeError("H155 hard resource limit")
                pressure = cpu > 85 or psutil.virtual_memory().available < 16 * 1024**3
                if pressure and not paused and worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGSTOP)
                    paused = True
                    report["pauses"] += 1
                elif not pressure and paused and worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGCONT)
                    paused = False
            report["exit_code"] = worker.returncode
            report["status"] = (
                "COMPLETE" if worker.returncode == 0 else "FAILED_CONSUMED"
            )
    except BaseException as exc:
        report["status"] = "FAILED_CONSUMED"
        report["exception"] = repr(exc)
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            os.killpg(worker.pid, signal.SIGCONT)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        raise
    finally:
        (root / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
