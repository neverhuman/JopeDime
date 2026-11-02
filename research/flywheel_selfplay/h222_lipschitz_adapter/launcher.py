"""Independent H222 CPU supervisor; immutable Git source, no checkout mutation."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h222_lipschitz_adapter/"
DATA = Path("/home/ubuntu/jope-data/h222")
COMPUTE_END = "2026-11-02T17:00:00+00:00"
CLOSEOUT_END = "2026-11-02T17:30:00+00:00"


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def observe(child=None):
    processes = []
    if child and child.poll() is None:
        try:
            processes = [
                psutil.Process(child.pid),
                *psutil.Process(child.pid).children(recursive=True),
            ]
        except psutil.NoSuchProcess:
            pass
    rss = psutil.Process().memory_info().rss
    for p in processes:
        try:
            rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return dict(
        at=time.time(),
        load1=os.getloadavg()[0],
        load5=os.getloadavg()[1],
        free_disk_bytes=psutil.disk_usage("/").free,
        available_ram_bytes=psutil.virtual_memory().available,
        process_tree_rss_bytes=rss,
        host_cpu_percent=psutil.cpu_percent(interval=0.2),
        affinity=sorted(os.sched_getaffinity(0)),
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        child_pid=child.pid if child else None,
        local_output_bytes=sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
        if DATA.exists()
        else 0,
    )


def guard(observation, now, deadline):
    if now >= deadline:
        raise RuntimeError("absolute H222 compute deadline")
    if observation["load5"] > 60:
        raise RuntimeError("load5 > 60")
    if observation["free_disk_bytes"] < 300_000_000_000:
        raise RuntimeError("free disk < 300GB")
    if observation["process_tree_rss_bytes"] >= 4 * 2**30:
        raise RuntimeError("process tree RSS >= 4GiB")
    if observation.get("local_output_bytes", 0) >= 500_000_000:
        raise RuntimeError("H222 local output >=500MB")
    if observation["available_ram_bytes"] < 16 * 2**30:
        raise RuntimeError("MemAvailable < 16GiB")
    if observation["host_cpu_percent"] > 85:
        raise RuntimeError("host CPU >85%")
    if len(observation["affinity"]) > 4 or observation["nice"] != 19:
        raise RuntimeError("CPU affinity/nice boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--phase", choices=["smoke", "run"], required=True)
    args = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2":
        raise RuntimeError("xbabe2 only")
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    deadline = dt.datetime.fromisoformat(COMPUTE_END).timestamp()
    guard(observe(), time.time(), deadline)
    raw = subprocess.check_output(
        ["git", "show", args.source + ":" + PREFIX + "manifest.json"], cwd=REPO
    )
    manifest = json.loads(raw)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source, "origin/main"],
        cwd=REPO,
        check=True,
    )
    DATA.mkdir(exist_ok=True)
    phase = args.phase
    root = DATA / ("smoke-" + args.source[:12] if phase == "smoke" else "attempt-001")
    control = DATA / ("control-" + phase)
    if root.exists():
        raise RuntimeError("no repeated phase or scientific retry")
    launch = dict(
        source_commit=args.source,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        compute_deadline_utc=COMPUTE_END,
        closeout_deadline_utc=CLOSEOUT_END,
        limits=manifest["resources"],
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    if phase == "run":
        smoke = DATA / ("smoke-" + args.source[:12]) / "metrics.json"
        receipt = json.loads(smoke.read_text())
        if (
            receipt["status"] != "SMOKE_PASS"
            or receipt["source_commit"] != args.source
            or receipt["updates"] != 6
        ):
            raise RuntimeError("merged-source TRAIN smoke required")
        launch.update(
            smoke_metrics_path=str(smoke),
            smoke_metrics_sha256=hashlib.sha256(smoke.read_bytes()).hexdigest(),
        )
    control.mkdir(exist_ok=False)
    save(control / "launch.json", launch)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AWS_", "GH_", "GITHUB_", "JOPE_AWS"))
        and k != "CARGO_TARGET_DIR"
    }
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(DATA / "tmp"),
    )
    (DATA / "tmp").mkdir(exist_ok=True)
    bootstrap = "import subprocess,sys; raw=subprocess.check_output(['git','show',sys.argv[1]+':research/flywheel_selfplay/h222_lipschitz_adapter/runner.py'],cwd='/home/ubuntu/JopeDime'); del sys.argv[1]; exec(compile(raw,'H222-immutable-runner.py','exec'),{'__name__':'__main__','__file__':'/home/ubuntu/JopeDime/research/flywheel_selfplay/h222_lipschitz_adapter/runner.py'})"
    argv = [
        sys.executable,
        "-B",
        "-c",
        bootstrap,
        args.source,
        "--source",
        args.source,
        "--phase",
        phase,
        "--root",
        str(root),
        "--launch",
        str(control / "launch.json"),
    ]
    with (
        (control / "worker.stdout").open("x") as out,
        (control / "worker.stderr").open("x") as err,
    ):
        child = subprocess.Popen(
            argv,
            cwd=DATA,
            env=env,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        identity = psutil.Process(child.pid).create_time()
        save(
            control / "launch-receipt.json",
            dict(
                supervisor_pid=os.getpid(),
                worker_pid=child.pid,
                worker_create_time=identity,
                source_commit=args.source,
                phase=phase,
                root=str(root),
                launch=launch,
                argv=argv,
            ),
        )
        error = None
        samples = []
        try:
            while child.poll() is None:
                observation = observe(child)
                samples.append(observation)
                with (control / "resources.jsonl").open("a") as f:
                    f.write(json.dumps(observation) + "\n")
                guard(observation, time.time(), deadline)
                time.sleep(0.8)
        except BaseException as failure:
            error = str(failure)
            if (
                child.poll() is None
                and psutil.Process(child.pid).create_time() == identity
                and os.getpgid(child.pid) == child.pid
            ):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
        code = child.wait()
    save(
        control / "terminal.json",
        dict(
            source_commit=args.source,
            phase=phase,
            worker_exit_code=code,
            error=error,
            samples=len(samples),
            peak_process_tree_rss_bytes=max(
                (s["process_tree_rss_bytes"] for s in samples), default=None
            ),
            compute_deadline_utc=COMPUTE_END,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            raw_root=str(root),
        ),
    )
    return code if not error else 1


if __name__ == "__main__":
    raise SystemExit(main())
