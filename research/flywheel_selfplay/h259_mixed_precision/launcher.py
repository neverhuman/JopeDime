"""Data-free H259 admission and source-bound memory resume; execs the monitor, no extra parent."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
STOP = 1793724300.0


def check_launch(launch, launch_path):
    output = Path(launch["output"]).resolve()
    path = Path(launch_path).resolve()
    n = launch.get("resume_number", 0)
    if n != 0 or launch.get("resume_from"):
        raise RuntimeError("H259 single run; no resumes")
    if path != output.with_suffix(".launch.json"):
        raise RuntimeError("adjacent immutable launch receipt required")
    if output.exists() or output.with_suffix(".attempt.claim").exists():
        raise RuntimeError("exclusive unused attempt required")
    if not (
        launch.get("published_prereg_verified") and launch.get("head_review_verified")
    ):
        raise RuntimeError("published reviewed v1/next prereg required")
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
    os.environ["H259_SOURCE"] = launch["source_commit"]
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
