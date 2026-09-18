#!/usr/bin/env python3
"""Fresh real-cut encoding in a separate directory; compare every saved float."""
import argparse
import hashlib
import json
import math
import signal
import struct
import subprocess
import time
from pathlib import Path

import experiment_runtime as runtime
import experiment_stages as stage_artifacts
from binary_encode import owned_ranks, sha, write
from experiment_scheduler import SPARE_CPUS, admission_reason, spare_lease


def immutable(path, value):
    if path.exists():
        if not (json.loads(path.read_text()) == value):
            raise ValueError(f"Frozen reproducibility input changed: {path}")
    else:
        write(path, value)


def compare_floats(original, fresh):
    left, right = original.read_bytes(), fresh.read_bytes()
    if not (len(left) == len(right) and len(left) % 4 == 0):
        raise ValueError('binary_reproduce: artifact validation failed at original line 28')
    a, b = struct.unpack("<" + "f" * (len(left)//4), left), struct.unpack("<" + "f" * (len(right)//4), right)
    if not (all(math.isfinite(v) for v in a + b)):
        raise ValueError('binary_reproduce: artifact validation failed at original line 30')
    return dict(original_sha256=hashlib.sha256(left).hexdigest(), recomputed_sha256=hashlib.sha256(right).hexdigest(),
        exact=left == right, float_count=len(a), max_abs_error=max(abs(x-y) for x,y in zip(a,b)))


def verify(root, experiment, output):
    jobs = json.loads((experiment / "jobs-initial.json").read_text())["jobs"]
    original_index = json.loads((experiment / "contexts-initial/index.json").read_text())
    fresh_index = json.loads((output / "contexts/index.json").read_text())
    old = {(r["dataset"], r["cut"]): r for r in original_index["encodings"]}
    new = {(r["dataset"], r["cut"]): r for r in fresh_index["encodings"]}
    if not (old.keys() == new.keys() and len(old) == len(jobs)):
        raise ValueError('binary_reproduce: artifact validation failed at original line 41')
    rows = []
    for job in jobs:
        prepared_path = root / job["prepared"] / "prepared.json"
        if not (sha(prepared_path) == job["prepared_sha256"]):
            raise ValueError('binary_reproduce: artifact validation failed at original line 45')
        prepared = json.loads(prepared_path.read_text())
        cut = next(c for c in prepared["cuts"] if c["index"] == job["cut"])
        suffix = Path(prepared["dataset"]) / f"cut-{job['cut']}"
        comparisons = {}
        for component, filename in [("target", "target.f32"), ("foundation", "foundation_raw.f32"), ("hyperion", "hyperion.f32")]:
            original = experiment / "components" / component / suffix
            fresh = output / "components" / component / suffix
            for directory in [original, fresh]:
                receipt = json.loads((directory / f"{component}.json").read_text())
                if not (receipt["prepared_sha256"] == job["prepared_sha256"]):
                    raise ValueError('binary_reproduce: artifact validation failed at original line 55')
                if not (receipt["training_sha256"] == cut["training_sha256"]):
                    raise ValueError('binary_reproduce: artifact validation failed at original line 56')
                if not (receipt["preprocessing_sha256"] == cut["preprocessing_sha256"]):
                    raise ValueError('binary_reproduce: artifact validation failed at original line 57')
                if not (receipt["signature_sha256"] == sha(directory / filename)):
                    raise ValueError('binary_reproduce: artifact validation failed at original line 58')
                if receipt["final_test_targets_used"] is not False:
                    raise ValueError('binary_reproduce: artifact validation failed at original line 59')
            comparisons[component] = compare_floats(original / filename, fresh / filename)
        key = prepared["dataset"], job["cut"]
        comparisons["context"] = compare_floats(root / old[key]["path"], root / new[key]["path"])
        if not (comparisons["context"]["original_sha256"] == old[key]["sha256"]):
            raise ValueError('binary_reproduce: artifact validation failed at original line 63')
        if not (comparisons["context"]["recomputed_sha256"] == new[key]["sha256"]):
            raise ValueError('binary_reproduce: artifact validation failed at original line 64')
        rows.append(dict(dataset=key[0], cut=key[1], comparisons=comparisons))
    exact = all(c["exact"] for row in rows for c in row["comparisons"].values())
    return dict(schema="parallel.binary_independent_recomputation.v1", passed=exact,
        status="all_outputs_byte_identical" if exact else "numerical_differences_require_review",
        independently_recomputed_views=len(rows), independently_recomputed_raw_components=3*len(rows),
        all_outputs_byte_identical=exact, exact_equality_tolerance=0.0,
        max_abs_error={kind:max(row["comparisons"][kind]["max_abs_error"] for row in rows) for kind in ["target","foundation","hyperion","context"]},
        original_context_index_sha256=sha(experiment / "contexts-initial/index.json"),
        recomputed_context_index_sha256=sha(output / "contexts/index.json"),
        copied_original_components=False, copied_cut_tables=0, projection_refitted=False,
        original_final_tests_opened=False, rows=rows)


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("independent encoding interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, experiment = args.root.resolve(), args.experiment.resolve()
    python_env = stage_artifacts.python_environment(experiment)
    output = experiment / "reproducibility"
    output.mkdir(exist_ok=True)
    jobs = experiment / "jobs-initial.json"
    original = json.loads((experiment / "encoding-run.json").read_text())
    if not (sha(args.binary) == original["executable_sha256"]):
        raise ValueError('binary_reproduce: artifact validation failed at original line 94')
    hyperion_spec = root / "manifests/hyperion-reviewed4-safe-spec.json"
    spec = json.loads(hyperion_spec.read_text())
    if not (sha(hyperion_spec) == original["hyperion_spec_sha256"]):
        raise ValueError('binary_reproduce: artifact validation failed at original line 97')
    if not (sha(root / spec["persistent_binary"]) == spec["persistent_binary_sha256"]):
        raise ValueError('binary_reproduce: artifact validation failed at original line 98')
    inputs = [jobs, experiment / "aliases.json", experiment / "contexts-initial/index.json",
        experiment / "sampling/target-spec.json", experiment / "sampling/foundation-spec.json",
        hyperion_spec, root / "data/encodings/contexts-assembly-v2/assembly.json"]
    request = dict(schema="parallel.binary_recomputation_request.v1", views=20,
        executable_sha256=sha(args.binary), source_sha256=sha(Path(__file__)),
        inputs={str(path.relative_to(root)):sha(path) for path in inputs},
        original_outputs_reused=False, fresh_component_directory=str((output / "components").relative_to(root)),
        qualification=True, projection_refitted=False, cpus=list(SPARE_CPUS), gpu_world=8)
    immutable(output / "request.json", request)
    stages = []
    for component in ["target", "foundation", "hyperion"]:
        commands = []
        for rank in range(8):
            if component == "hyperion":
                command = [str(root / spec["persistent_binary"]), "--root", str(root)]
                component_spec = hyperion_spec
            else:
                command = [str(args.binary), "--root", str(root), f"{component}-rank"]
                component_spec = experiment / "sampling" / f"{component}-spec.json"
            command += ["--jobs", str(jobs), "--spec", str(component_spec), "--output", str(output / "components" / component),
                "--rank", str(rank), "--world", "8"]
            if component != "hyperion":
                command += ["--batch", "128" if component == "target" else "2"]
            commands.append(command)
        stages.append((component, commands))
    stages.append(("contexts", [[str(args.binary), "--root", str(root), "apply-binary-contexts", "--jobs", str(jobs),
        "--hyperion", str(output / "components/hyperion"), "--foundation", str(output / "components/foundation"),
        "--target", str(output / "components/target"), "--assembly", str(root / "data/encodings/contexts-assembly-v2"),
        "--output", str(output / "contexts"), "--aliases", str(experiment / "aliases.json")]]))
    immutable(output / "commands.json", {name:commands for name,commands in stages})
    if args.prepare_only:
        write(output / "progress.json", dict(phase="prepared", timestamp=int(time.time()),
            reason="Frozen fresh-encoding request; awaiting downstream comparison completion and scheduler admission"))
        return
    deadline = time.monotonic() + args.seconds
    for stage, commands in stages:
        complete = output / "stages" / f"{stage}.json"
        command_hash = hashlib.sha256(json.dumps(commands).encode()).hexdigest()
        if complete.exists():
            if not (json.loads(complete.read_text())["commands_sha256"] == command_hash):
                raise ValueError('binary_reproduce: artifact validation failed at original line 138')
            stage_artifacts.verify(root, commands, json.loads(complete.read_text()))
            continue
        while time.monotonic() < deadline:
            children, logs = [], []
            try:
                with spare_lease(root, "binary-reproducibility:" + stage, gpu=True), owned_ranks(children):
                    begun = time.monotonic()
                    for rank, command in enumerate(commands):
                        logpath = output / "logs" / f"{stage}-{time.time_ns()}-{rank}.log"
                        logpath.parent.mkdir(parents=True, exist_ok=True)
                        log = logpath.open("w")
                        logs.append(log)
                        cpus = SPARE_CPUS[rank*2:rank*2+2] if len(commands) == 8 else SPARE_CPUS[:2]
                        env = runtime.child_env(**python_env, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", JAIN_HYPERION_CUDA_MAX_LOAD_RATIO="1000")
                        children.append(runtime.popen(["taskset","-c",",".join(map(str,cpus))] + command,
                            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    while any(child.poll() is None for child in children):
                        reason = admission_reason(root)
                        if reason or time.monotonic() >= deadline:
                            write(output / "cancellations" / f"{stage}-{time.time_ns()}.json",
                                dict(stage=stage, reason=reason or "bounded window expired", seconds=time.monotonic()-begun,
                                    completed_components_preserved=True))
                            raise BlockingIOError(reason or "bounded window expired")
                        if not (not any(c.returncode not in [None,0] for c in children)):
                            raise ValueError(f"{stage} worker failed")
                        write(output / "progress.json", dict(phase=stage, timestamp=int(time.time()), reason="Fresh real-cut recomputation on eight GPUs", cpu_budget=16))
                        time.sleep(1)
                    if not (all(c.returncode == 0 for c in children)):
                        raise ValueError(f"{stage} failed")
                    immutable(complete, dict(stage=stage, complete=True, commands_sha256=command_hash,
                        seconds=time.monotonic()-begun, timestamp=int(time.time()), outputs=stage_artifacts.inventory(root, commands)))
                    break
            except BlockingIOError as error:
                write(output / "progress.json", dict(phase="waiting", reason=str(error), timestamp=int(time.time())))
                time.sleep(2)
            finally:
                for log in logs:
                    log.close()
        else:
            return
    receipt = verify(root, experiment, output)
    receipt["request_sha256"] = sha(output / "request.json")
    receipt["encoding_seconds"] = sum(json.loads(path.read_text())["seconds"] for path in (output / "stages").glob("*.json"))
    immutable(output / "verification.json", receipt)
    write(output / "progress.json", dict(phase="completed", passed=receipt["passed"], timestamp=int(time.time()),
        views=receipt["independently_recomputed_views"], max_abs_error=receipt["max_abs_error"]))
    print(json.dumps({k:v for k,v in receipt.items() if k != "rows"}))


if __name__ == "__main__":
    runtime.entrypoint(main, "--experiment", "binary")
