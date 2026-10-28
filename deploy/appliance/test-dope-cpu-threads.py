#!/usr/bin/env python3
"""Fresh sealed Rust CPU inference must honor its pool bounds without environment overrides."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("proof", type=Path)
    args = parser.parse_args()
    package, proof = args.package.resolve(), args.proof.resolve()
    if proof == Path("/tmp") or Path("/tmp") in proof.parents:
        parser.error("retain proof outside /tmp")
    proof.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    removed = []
    for key in list(env):
        if key in {"RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"} or key.startswith("CANDLE_"):
            removed.append(key)
            del env[key]
    env["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for task, method, classes in [
        ("regression", "ridge", 0),
        ("binary", "logistic", 2),
        ("multiclass", "multiclass_ridge", 3),
    ]:
        data = proof / f"{task}.csv"
        with data.open("w") as stream:
            stream.write(",".join([f"x{j}" for j in range(10)] + ["y"]) + "\n")
            for i in range(5000):
                x = [math.sin(i * (j + 1) * .013) for j in range(10)]
                y = str(x[0] + x[1] ** 2) if not classes else f"class{i % classes}"
                stream.write(",".join(map(str, x)) + "," + y + "\n")
        command = [
            str(package), "fit", "--train", str(data), "--target", "y", "--task", task,
            "--method", method, "--threads", "1", "--device", "cpu", "--landmark-mode", "fast",
            "--output", str(proof / f"{task}.jope"), "--landmark-receipt",
            str(proof / f"{task}-encoding.json"), "--json",
        ]
        counts, names, affinity = [], set(), None
        started = time.monotonic()
        with (proof / f"{task}-stdout.json").open("w") as out, (proof / f"{task}-stderr.txt").open("w") as err:
            process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
            while process.poll() is None:
                try:
                    affinity = sorted(os.sched_getaffinity(process.pid))
                    tasks = list(Path(f"/proc/{process.pid}/task").iterdir())
                    counts.append(len(tasks))
                    for thread in tasks:
                        try:
                            names.add((thread / "comm").read_text().strip())
                        except FileNotFoundError:
                            pass
                except (FileNotFoundError, ProcessLookupError):
                    pass
                time.sleep(.002)
            code = process.wait()
        results.append({
            "task": task, "command": command, "exit_code": code,
            "elapsed_seconds": time.monotonic() - started, "peak_threads": max(counts, default=0),
            "samples": len(counts), "thread_names": sorted(names), "cpu_affinity": affinity,
        })
        # Retain reproducing failures as well as successes.
        record = {
            "schema": "jope.dope-cpu-thread-proof/v1",
            "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
            "removed_thread_environment": sorted(removed), "cuda_visible_devices": "",
            "maximum_threads": 5, "results": results,
        }
        (proof / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        assert code == 0, f"{task} failed: inspect retained stderr/stdout"
        encoding = json.loads((proof / f"{task}-encoding.json").read_text())
        assert encoding["status"] == "none" and not any(encoding["mask"])
        assert encoding["version"] == 0 and "cpu" in encoding["device"].lower()
        assert len(encoding["embedding"]) == 512 and all(map(math.isfinite, encoding["embedding"]))
        assert counts and max(counts) <= 5, f"{task} escaped its owned pools: {max(counts, default=0)} threads"
    print(f"Dope CPU pool proof passed: {proof}")


if __name__ == "__main__":
    main()
