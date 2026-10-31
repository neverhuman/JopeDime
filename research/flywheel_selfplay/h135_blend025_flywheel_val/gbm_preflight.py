"""Synthetic full-recipe fit; external /proc sampling, no study inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

from launcher import one_thread, worker_environment


def worker(output, original):
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = str(Path(os.fsdecode(args[0])).resolve()).lower()
            if "/jope-data/" in path or any(
                x in path for x in ("/sealed", "/holdout", "meta_test", "meta-test")
            ):
                raise RuntimeError(
                    "study/protected input forbidden in synthetic preflight"
                )

    sys.addaudithook(audit)
    import numpy as np
    import threading
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.utils.parallel import Parallel
    import threadpoolctl
    from unittest.mock import patch
    from runner import imported

    _, helper = imported()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2048, 1089))
    y = x[:, 0] + 0.1 * x[:, 1]
    events = output / "events.jsonl"

    def event(phase, **fields):
        with events.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        phase=phase,
                        threads=len(list(Path("/proc/self/task").iterdir())),
                        python_threads=[t.name for t in threading.enumerate()],
                        native_pools=threadpoolctl.threadpool_info(),
                        **fields,
                    )
                )
                + "\n"
            )

    original_call = Parallel.__call__

    def traced_parallel(self, *args, **kwargs):
        event("binning_parallel", n_jobs=self.n_jobs)
        return original_call(self, *args, **kwargs)

    def fit():
        # Exercise the unchanged runner's nested requests as well as actual sklearn.
        with (
            threadpoolctl.threadpool_limits(limits=1, user_api="blas"),
            threadpoolctl.threadpool_limits(limits=4, user_api="openmp"),
            patch.object(Parallel, "__call__", traced_parallel),
        ):
            event("before_fit", parameters=dict(helper.MODEL, random_state=135))
            model = HistGradientBoostingRegressor(**helper.MODEL, random_state=135).fit(
                x, y
            )
            model.predict(x[:32])
            event(
                "after_fit",
                iterations=int(model.n_iter_),
                parameters=model.get_params(),
            )

    event("imports")
    fit() if original else one_thread(fit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original", action="store_true")
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(args.output, args.original)
        return
    from monitor import reason, sample
    import psutil

    output = args.output.resolve()
    output.mkdir()
    if reason(sample(output)):
        raise RuntimeError("resource preflight failed before synthetic worker")
    env = worker_environment(os.environ)
    env["PYTHONPYCACHEPREFIX"] = str(output / "unused-bytecode")
    command = [
        "nice",
        "-n",
        "19",
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(output),
        "--worker",
    ] + (["--original"] if args.original else [])
    samples, error = [], None
    started = time.monotonic()
    with (output / "worker.log").open("w") as log:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    row = sample(output, proc)
                    row["proc_task_count"] = len(
                        list(Path(f"/proc/{child.pid}/task").iterdir())
                    )
                except (psutil.NoSuchProcess, FileNotFoundError):
                    break
                samples.append(row)
                error = reason(row)
                if error or time.monotonic() - started > 180:
                    error = error or "synthetic timeout"
                    break
                time.sleep(0.005)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
    code = child.wait()
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in samples)
    )
    receipt = dict(
        status="PASS" if code == 0 and error is None else "FAIL",
        original=args.original,
        exit_status=code,
        error=error,
        samples=len(samples),
        maximum_threads=max((x["threads"] for x in samples), default=0),
        peak_rss=max((x["rss"] for x in samples), default=0),
        wall_seconds=time.monotonic() - started,
        command=command,
        launcher_sha256=hashlib.sha256(
            Path(__file__).with_name("launcher.py").read_bytes()
        ).hexdigest(),
        synthetic_rows=2048,
        features=1089,
        study_payload_reads=0,
        study_metadata_reads=0,
        caveat="External sampling is finite; this is a synthetic regression, not full-study qualification.",
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
