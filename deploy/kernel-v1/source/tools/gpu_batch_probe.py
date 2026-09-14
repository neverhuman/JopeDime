"""Compare declared GPU batches in spare windows; never publish training weights."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import experiment_runtime as runtime
from experiment_scheduler import admission_reason, spare_lease
from kernel_release import digest


def summarize(directory, microbatch, global_batch, steps, vocabulary_sha256, vocabulary_size):
    rows = [json.loads((directory / f"complete-{rank}.json").read_text()) for rank in range(8)]
    hashes = set()
    for rank, row in enumerate(rows):
        runtime.require(row.get("rank") == rank and row.get("world") == 8
                        and row.get("microbatch") == microbatch and row.get("global_batch") == global_batch
                        and row.get("steps") == steps and row.get("eligible_checkpoint") is False
                        and row.get("vocabulary_sha256") == vocabulary_sha256
                        and row.get("vocabulary_size") == vocabulary_size,
                        "probe rank or batch identity mismatch")
        times = row.get("step_seconds", [])
        runtime.require(len(times) == steps and all(isinstance(t, (float, int)) and not isinstance(t, bool)
                        and math.isfinite(t) and t > 0 for t in times), "invalid probe timing")
        hashes.add(row.get("parameter_sha256"))
    runtime.require(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes),
                    "probe rank parameters disagree")
    # Rank time maxima include the slowest worker; exclude declared warmup.
    times = [max(row["step_seconds"][i] for row in rows) for i in range(5, steps)]
    return {"microbatch": microbatch, "global_batch": global_batch, "warmup_steps": 5,
            "measured_steps": len(times), "measured_examples": len(times) * global_batch,
            "measured_seconds": sum(times), "examples_per_second": len(times) * global_batch / sum(times),
            "median_step_seconds": statistics.median(times), "eligible_checkpoint": False}


def sample_gpu():
    return runtime.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], timeout=3, env=runtime.child_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    output = runtime.output_root(args.root, args.output)
    binary = runtime.inside(args.root, args.binary)
    runtime.require(digest(binary) == args.binary_sha256, "probe executable changed")
    vocabulary = runtime.inside(args.root, args.vocabulary)
    vocabulary_sha256 = digest(vocabulary)
    vocabulary_size = len(json.loads(vocabulary.read_text())["tokens"])
    runtime.require(vocabulary_size > 3, "invalid probe vocabulary")
    request = {"schema": "parallel.gpu_batch_probe.v2",
               "vocabulary_sha256": vocabulary_sha256, "vocabulary_size": vocabulary_size, "binary_sha256": args.binary_sha256,
               "order": [256, 512, 512, 256], "world": 8, "measured_examples_per_block": 122880,
               "warmup_steps": 5, "seed": 20260907, "learning_protocol_changed": False,
               "scope": "full architecture numerical throughput; no corpus data, no eligible checkpoint"}
    manifest = output / "manifest.json"
    if manifest.exists():
        runtime.require(json.loads(manifest.read_text()) == request, "probe resume identity changed")
    else:
        runtime.write(manifest, request)
    deadline = time.monotonic() + args.seconds
    for block, microbatch in enumerate(request["order"]):
        receipt = output / f"block-{block}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            runtime.require(saved.get("status") in ("passed", "failed"), "invalid saved probe status")
            attempt = runtime.inside(output, Path(saved["attempt"]))
            if saved.get("status") == "passed":
                runtime.require(saved["metrics"] == summarize(attempt, microbatch, 8 * microbatch,
                    5 + request["measured_examples_per_block"] // (8 * microbatch), vocabulary_sha256, vocabulary_size), "saved probe changed")
            continue
        while time.monotonic() < deadline:
            attempt = None
            children = []
            started = time.monotonic()
            try:
                with spare_lease(args.root, "gpu-batch-probe", cpus=16, gpu=True), runtime.owned_ranks(children):
                    runtime.require(digest(binary) == args.binary_sha256, "probe executable changed")
                    runtime.require(digest(vocabulary) == vocabulary_sha256, "probe vocabulary changed")
                    attempt = output / f"block-{block}-attempt-{time.time_ns()}"
                    global_batch = 8 * microbatch
                    steps = 5 + request["measured_examples_per_block"] // global_batch
                    command = [str(binary), "training-probe", "--output", str(attempt), "--vocabulary", str(vocabulary),
                               "--microbatch", str(microbatch), "--global-batch", str(global_batch),
                               "--steps", str(steps), "--timeout", "180"]
                    with (output / f"{attempt.name}.log").open("x") as log:
                        child = runtime.popen(command, env=runtime.child_env(output), stdout=log, stderr=log)
                        children.append(child)
                        sampled = 0
                        samples = []
                        while child.poll() is None:
                            reason = admission_reason(args.root)
                            if reason: raise BlockingIOError(reason)
                            if time.monotonic() - started > 200: raise subprocess.TimeoutExpired(command, 200)
                            if time.monotonic() - sampled >= 2:
                                sample = sample_gpu()
                                samples.append({"timestamp": time.time(), "exit_code": sample.returncode,
                                                "csv": sample.stdout, "error": sample.stderr})
                                runtime.write(output / f"{attempt.name}-gpu.json", samples)
                                sampled = time.monotonic()
                            runtime.write(output / "progress.json", {"phase": "running", "block": block,
                                "microbatch": microbatch, "global_batch": global_batch, "gpu_world": 8, "timestamp": int(time.time())})
                            time.sleep(.2)
                        runtime.write(output / f"{attempt.name}-gpu.json", samples)
                    result = {"status": "passed" if child.returncode == 0 else "failed", "attempt": str(attempt),
                              "exit_code": child.returncode, "elapsed_seconds": time.monotonic() - started}
                    if child.returncode == 0:
                        result["metrics"] = summarize(attempt, microbatch, global_batch, steps, vocabulary_sha256, vocabulary_size)
                    runtime.write(receipt, result)
                break
            except BlockingIOError as error:
                if attempt is not None:
                    runtime.append(output / "interrupted.jsonl", {"attempt": str(attempt), "reason": str(error),
                        "elapsed_seconds": time.monotonic() - started, "wall_time_comparable": False})
                runtime.write(output / "progress.json", {"phase": "waiting", "reason": str(error), "timestamp": int(time.time())})
                time.sleep(2)
        if not receipt.exists():
            runtime.write(output / "progress.json", {"phase": "paused", "reason": "bounded admission wait expired", "timestamp": int(time.time())})
            return
    blocks = [json.loads((output / f"block-{i}.json").read_text()) for i in range(4)]
    runtime.write(output / "report.json", {"complete": True, "blocks": blocks,
        "all_passed": all(b["status"] == "passed" for b in blocks), "deployment_qualified": False,
        "next_gate": "frozen real-corpus, equal-example quality comparison before any training batch change"})
    runtime.write(output / "progress.json", {"phase": "completed", "complete": True, "timestamp": int(time.time())})


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "gpu-batch-probe")
