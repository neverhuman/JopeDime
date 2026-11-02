"""Shared guarded runner lifecycle for H159 and H161 monitor scripts."""

import datetime
import json
import os
import subprocess
import sys
import time


def run(args, here, environment_fn, sample_fn, classify_fn, stop_fn):
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    c = args.custody.resolve()
    c.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "run"
    if (c / label).exists():
        raise RuntimeError("attempt already exists")
    with (c / f"{label}.started").open("x") as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    child = None
    record = dict(
        status="WAITING",
        stages=[],
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        host=os.uname().nodename,
    )
    started = time.monotonic()
    try:
        s = sample_fn(c)
        action, why = classify_fn(s)
        if action != "run":
            raise RuntimeError("prelaunch guard " + str(why))
        cmd = [
            sys.executable,
            "-B",
            str(here / "runner.py"),
            "--output",
            str(c / label),
        ] + (["--smoke"] if args.smoke else [])
        with (c / f"{label}.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                env=environment_fn(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(status="RUNNING", command=cmd, pid=child.pid)
            with (c / f"{label}-resources.jsonl").open("w") as resources:
                while child.poll() is None:
                    s = sample_fn(c, child)
                    resources.write(json.dumps(s) + "\n")
                    resources.flush()
                    action, why = classify_fn(s)
                    if action == "abort":
                        raise RuntimeError("guard " + why)
                    time.sleep(0.25)
            code = child.wait()
            record["exit_status"] = code
            if code:
                raise RuntimeError(f"worker exit {code}")
        record["status"] = "COMPLETE"
    except BaseException as e:
        record.update(status="INCOMPLETE", error=str(e))
        raise
    finally:
        stop_fn(child)
        record["wall_seconds"] = time.monotonic() - started
        (c / f"{label}-monitor.json").write_text(json.dumps(record, indent=2) + "\n")
