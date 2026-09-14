#!/usr/bin/env python3
"""Freeze all six rankings on eight GPUs in resumable campaign spare windows."""
import argparse
import experiment_runtime as runtime
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from experiment_scheduler import admission_reason, spare_lease, SPARE_CPUS
from seed_experiment import verify, inside, digest, write, seal_rankings


stop = runtime.stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=5400)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = inside(root, args.manifest)
    manifest = verify(root, manifest_path)
    output = inside(root, manifest["output"])
    binary = inside(root, args.binary)
    binding = {"manifest_sha256": digest(manifest_path), "executable_sha256": digest(binary), "world": 8}
    run_path = output / "proposal-run.json"
    if run_path.exists():
        if not (json.loads(run_path.read_text()) == binding):
            raise ValueError("proposal implementation changed on resume")
    else:
        write(run_path, binding)
    deadline = time.monotonic() + args.seconds
    stopping = False
    def interrupt(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    def publish(phase, reason):
        complete = len(list((output / "proposals").glob("*/cut-*/complete.json")))
        write(output / "progress.json", dict(schema="parallel.seed_experiment_progress.v1", timestamp=int(time.time()),
            phase=phase, reason=reason, completed=complete, total=len(manifest["views"]),
            saved_programs=complete*64, physical_fits=0, logical_evaluations=0,
            corpus_records=manifest["corpus_records"], selected_checkpoint=manifest["selected_checkpoint_source"],
            gpu_world=8, cpu_budget=16))
    attempt = 0
    while not stopping and time.monotonic() < deadline:
        children, logs = [], []
        try:
            with spare_lease(root, "ranked-proposals", gpu=True), runtime.owned_ranks(children):
                attempt += 1
                for rank in range(8):
                    log_path = output / "proposal-logs" / f"attempt-{attempt}-{time.time_ns()}-rank-{rank}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open("w")
                    logs.append(log)
                    command = ["taskset", "-c", ",".join(map(str,SPARE_CPUS[rank*2:rank*2+2])), str(binary),
                        "--root", str(root), "propose-ranked", "--manifest", str(manifest_path), "--rank", str(rank), "--world", "8"]
                    env = runtime.child_env( OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
                    children.append(runtime.popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
                while any(child.poll() is None for child in children):
                    reason = admission_reason(root)
                    if reason or stopping or time.monotonic() >= deadline:
                        stop(children)
                        write(output / "proposal-cancellations" / f"{time.time_ns()}.json",
                              dict(attempt=attempt, timestamp=int(time.time()), reason=reason or "bounded interruption", preserved=True))
                        raise BlockingIOError(reason or "bounded interruption")
                    if any(child.returncode not in (None, 0) for child in children):
                        raise RuntimeError("ranked proposal worker failed; inspect preserved logs")
                    publish("freezing pure rankings", "Eight GPUs selecting from all valid draws, exact champions, and grammar proposals")
                    time.sleep(1)
                if any(child.returncode != 0 for child in children):
                    raise RuntimeError("ranked proposal worker failed; inspect preserved logs")
                seal_rankings(root, manifest, manifest_path)
                publish("pure rankings frozen", "All six controls saved before retained-teacher assembly or evaluation")
                result = runtime.run([str(binary), "--root", str(root), "assemble-ranked", "--manifest", str(manifest_path)],
                                     root=root, env=runtime.child_env(output),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
                (output / "assembly.log").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError("GP population assembly failed; inspect assembly.log")
                write(output / "proposal-complete.json", dict(binding, complete=True, timestamp=int(time.time()),
                      rankings_freeze_sha256=digest(output / "rankings-frozen.json")))
                publish("completed", "Pure rankings and identical retained/exploration tails saved; GP evaluation pending")
                return
        except BlockingIOError as error:
            publish("waiting", str(error))
            time.sleep(1)
        except BaseException as error:
            publish("failed", str(error))
            raise
        finally:
            stop(children)
            for log in logs:
                log.close()
    publish("paused", "Bounded proposal window expired; completed rankings preserved for resume")


if __name__ == "__main__":
    runtime.entrypoint(main, "--manifest", "proposal")
