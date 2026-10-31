"""One attempt; CPU resource monitor outside numerical workers."""

from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import psutil
from launcher import worker_environment

HERE = Path(__file__).resolve().parent
ARROW_ALLOCATOR_CONF = "background_thread:false"


def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def reason(s):
    for bad, label in [
        (s["rss"] >= 8_000_000_000, "RSS>=8GB"),
        (s["threads"] > 4, "threads>4"),
        (s["disk_percent"] >= 93.8, "disk>=93.8%"),
        (s["scratch"] >= 2_000_000_000, "scratch>=2GB"),
        (s["load1"] > 0.85 * s["cpus"], "load>85%"),
        (s["available"] < 16 * 1024**3, "MemAvailable<16GiB"),
    ]:
        if bad:
            return label
    return None


def sample(root, proc=None):
    st = os.statvfs("/")
    rss = threads = 0
    if proc:
        for p in [proc] + proc.children(recursive=True):
            try:
                rss += p.memory_info().rss
                threads += p.num_threads()
            except psutil.NoSuchProcess:
                pass
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rss=rss,
        threads=threads,
        disk_percent=100 * (1 - st.f_bavail / st.f_blocks),
        scratch=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        load1=os.getloadavg()[0],
        cpus=len(os.sched_getaffinity(0)),
        available=psutil.virtual_memory().available,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    args = p.parse_args()
    c = args.custody.resolve()
    launch = json.loads((c / "launch.json").read_text())
    if not launch["merged_source_verified"] or not launch["seed_freshness_verified"]:
        raise RuntimeError("merge and seed gates required")
    for name, digest in json.loads((HERE / "manifest.json").read_text())[
        "executor_sha256"
    ].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("executor changed")
    root = c / "run"
    root.mkdir()  # single attempt, never resume implicitly
    shutil.copyfile(c / "reference.json", root / "reference.json")
    env = worker_environment(
        dict(
            os.environ,
            GIT_DIR=launch["git_dir"],
            JE_ARROW_MALLOC_CONF=ARROW_ALLOCATOR_CONF,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPYCACHEPREFIX=str(root / "bytecode"),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            NUMEXPR_MAX_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
    )
    record = dict(
        source=launch,
        worker_environment={
            k: env[k]
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "JE_ARROW_MALLOC_CONF",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        },
        stages=[],
        status="RUNNING",
    )
    child = None
    try:
        for stage in ["baseline", "135", "136", "137", "summary"]:
            first = sample(root)
            bad = reason(first)
            if bad:
                raise RuntimeError("preflight: " + bad)
            cmd = [
                "nice",
                "-n",
                "19",
                sys.executable,
                str(
                    HERE
                    / ("launcher.py" if stage in ("135", "136", "137") else "runner.py")
                ),
                "--root",
                str(root),
                "--stage",
                stage,
            ]
            with (c / f"{stage}.log").open("w") as log:
                child = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                proc = psutil.Process(child.pid)
                peak = 0
                while child.poll() is None:
                    try:
                        s = sample(root, proc)
                    except psutil.NoSuchProcess:
                        break
                    peak = max(peak, s["rss"])
                    with (c / "resources.jsonl").open("a") as out:
                        out.write(json.dumps(dict(stage=stage, **s)) + "\n")
                    bad = reason(s)
                    if bad:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise RuntimeError("guard: " + bad)
                    time.sleep(0.25)
                code = child.wait()
            record["stages"].append(
                dict(stage=stage, exit_status=code, peak_rss=peak, command=cmd)
            )
            save(c / "monitor.json", record)
            if code:
                raise RuntimeError("stage failed: " + stage)
        record["status"] = "COMPLETE"
    except BaseException as error:
        record["status"] = "INCOMPLETE"
        record["error"] = str(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save(c / "monitor.json", record)


if __name__ == "__main__":
    main()
