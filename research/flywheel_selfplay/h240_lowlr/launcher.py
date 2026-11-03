"""Data-free H240 admission and source-bound memory resume; execs the monitor, no extra parent."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
STOP = 1793691000.0


def check_launch(launch, launch_path):
    output = Path(launch["output"]).resolve()
    path = Path(launch_path).resolve()
    n = launch.get("resume_number", 0)
    if type(n) is not int or not 0 <= n <= 2:
        raise RuntimeError("maximum two infrastructure resumes")
    if path != output.with_suffix(".launch.json"):
        raise RuntimeError("adjacent immutable launch receipt required")
    if output.exists() or output.with_suffix(".attempt.claim").exists():
        raise RuntimeError("exclusive unused attempt required")
    if not launch.get("merged_source_verified"):
        raise RuntimeError("merged source required")
    if n:
        prior = Path(launch["resume_from"]).resolve()
        if prior.parent != output.parent:
            raise RuntimeError("resume root boundary")
        previous = json.loads(prior.with_suffix(".launch.json").read_text())
        if previous.get("resume_number", 0) + 1 != n:
            raise RuntimeError("resume chain")
        for key in [
            "source_commit",
            "manifest_sha256",
            "fold_manifest_sha256",
            "prereg_sha256",
        ]:
            if previous[key] != launch[key]:
                raise RuntimeError("resume changed science")
        terminal = json.loads(prior.with_suffix(".resource_summary.json").read_text())
        if (
            terminal["status"] != "INCOMPLETE"
            or not terminal["reasons"]
            or not all(
                reason.startswith("MemAvailable<") for reason in terminal["reasons"]
            )
        ):
            raise RuntimeError("only assigned memory hard stops permit resume")
        if len(list(prior.glob("A[0-4]-24010[1-3].json"))) >= 15:
            raise RuntimeError("no unfinished cells to resume")
    elif launch.get("resume_from"):
        raise RuntimeError("initial worker cannot import prior cells")
    return output, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    launch = json.loads(args.launch.read_text())
    output, n = check_launch(launch, args.launch)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or sorted(os.sched_getaffinity(0)) != [40, 41, 42, 43]
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("xbabe2 CPU40-43 nice19 required")
    if (
        hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest()
        != launch["manifest_sha256"]
    ):
        raise RuntimeError("manifest binding")
    stable = 0
    with output.with_suffix(".admission.jsonl").open("x") as log:
        while True:
            sample = dict(
                utc=time.time(),
                load5=os.getloadavg()[1],
                available=psutil.virtual_memory().available,
                free=psutil.disk_usage("/home/ubuntu").free,
            )
            good = (
                sample["load5"] <= 100
                and sample["available"] >= 30 * 2**30
                and sample["free"] >= 300_000_000_000
            )
            stable = stable + 1 if good else 0
            log.write(
                json.dumps(
                    dict(
                        **sample,
                        admissible=good,
                        stable_samples=stable,
                        resume_number=n,
                    )
                )
                + "\n"
            )
            log.flush()
            print(
                json.dumps(
                    dict(
                        **sample,
                        admissible=good,
                        stable_samples=stable,
                        resume_number=n,
                    )
                ),
                flush=True,
            )
            if sample["utc"] >= STOP:
                raise RuntimeError("absolute compute deadline before admission")
            if stable >= 2:
                break
            time.sleep(30)
    os.environ["H240_SOURCE"] = launch["source_commit"]
    command = [
        sys.executable,
        "-B",
        str(HERE / "monitor.py"),
        "--output",
        str(output),
        "--launch",
        str(args.launch.resolve()),
    ]
    if n:
        command += ["--resume-from", launch["resume_from"]]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
