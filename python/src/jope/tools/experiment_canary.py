"""Bounded dummy-worker canary for isolation, lease ownership and GPU handoff."""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import experiment_runtime as runtime
from experiment_scheduler import admission_reason, spare_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=("normal", "rank-failure", "partial-launch", "controller-kill", "controller-term", "handoff"),
        default="normal",
    )
    args = parser.parse_args()
    output = runtime.output_root(args.root, args.output)
    deadline = time.monotonic() + args.seconds
    # Probe with an exclusive filename; no existing production artifact is touched.
    probe = args.root / "runs" / (".hardening-write-probe-" + str(os.getpid()))
    try:
        with probe.open("x"):
            pass
    except OSError as error:
        runtime.require(error.errno in (13, 30), f"unexpected write-probe error: {error}")
    else:
        probe.unlink()
        raise RuntimeError("experiment service permitted an escaped write")
    runtime.write(
        output / "isolation.json", {"escaped_writes": 0, "cgroup": Path("/proc/self/cgroup").read_text()}
    )
    while time.monotonic() < deadline:
        children = []
        try:
            with spare_lease(args.root, "hardening-canary", cpus=2, gpu=True), runtime.owned_ranks(children):
                children.append(
                    runtime.popen(
                        [sys.executable, "-c", "import time; time.sleep(7200)"], env=runtime.child_env()
                    )
                )
                runtime.write(
                    output / "workers.json",
                    {"controller": os.getpid(), "workers": [p.pid for p in children], "mode": args.mode},
                )
                if args.mode == "controller-kill":
                    os.kill(os.getpid(), signal.SIGKILL)
                if args.mode == "controller-term":
                    os.kill(os.getpid(), signal.SIGTERM)
                if args.mode == "partial-launch":
                    runtime.popen(["/nonexistent/hardening-canary"], env=runtime.child_env())
                if args.mode == "rank-failure":
                    children[0].kill()
                    children[0].wait()
                    raise RuntimeError("injected rank failure")
                reason = None
                while args.mode == "handoff" and time.monotonic() < deadline:
                    reason = admission_reason(args.root)
                    runtime.write(
                        output / "progress.json",
                        {
                            "phase": "waiting for mandatory handoff",
                            "timestamp": int(time.time()),
                            "reason": reason,
                        },
                    )
                    if reason:
                        break
                    time.sleep(0.2)
            runtime.write(
                output / "complete.json",
                {
                    "passed": args.mode != "handoff" or reason is not None,
                    "mandatory_handoff_observed": bool(reason),
                    "reason": reason,
                    "surviving_workers": sum(p.poll() is None for p in children),
                    "escaped_writes": 0,
                },
            )
            return
        except BlockingIOError as error:
            runtime.write(
                output / "progress.json",
                {"phase": "waiting", "reason": str(error), "timestamp": int(time.time())},
            )
            time.sleep(1)
        except (RuntimeError, FileNotFoundError) as error:
            if args.mode not in ("rank-failure", "partial-launch"):
                raise
            runtime.write(
                output / "complete.json",
                {
                    "passed": all(p.poll() is not None for p in children),
                    "injected_failure": str(error),
                    "surviving_workers": sum(p.poll() is None for p in children),
                    "escaped_writes": 0,
                },
            )
            return
    runtime.write(
        output / "complete.json", {"passed": False, "reason": "no scheduler-approved canary window"}
    )


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "canary")
