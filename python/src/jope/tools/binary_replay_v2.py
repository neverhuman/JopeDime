"""Recompute both sides of all 80 binary contexts, including every raw component."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import binary_verify as binary
from binary_encode import serial_job_waves
from binary_reproduce import compare_floats, immutable
import experiment_runtime as runtime
import experiment_stages as stages
from experiment_scheduler import SPARE_CPUS, admission_reason, spare_lease


def configuration(executable):
    from seed_experiment import executable_dependencies
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                            env=runtime.child_env(), capture_output=True, text=True, check=True)
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    runtime.require(len(gpus) == 8 and len(set(gpus)) == 8, "replay requires the frozen eight-GPU configuration")
    return {"runtime_dependencies": executable_dependencies(executable), "gpus": gpus,
            "precision": "frozen encoder specifications and executable defaults", "dispatch_threads": 2}


def commands_for_replay(root, original, output, executable, waves, specs):
    commands = {}
    hyperion = binary.read(specs["hyperion"])
    for side in ("a", "b"):
        for component in binary.COMPONENTS:
            for number, wave in enumerate(waves):
                ranks = []
                for rank in range(len(binary.read(wave)["jobs"])):
                    command = ([str(root / hyperion["persistent_binary"]), "--root", str(root)] if component == "hyperion"
                               else [str(executable), "--root", str(root), f"{component}-rank"])
                    command += ["--jobs", str(wave), "--spec", str(specs[component]), "--output", str(output / side / "components" / component), "--rank", str(rank), "--world", "8"]
                    if component != "hyperion":
                        command += ["--batch", "128" if component == "target" else "2"]
                    ranks.append(command)
                commands[f"{side}-{component}-{number:02}"] = ranks
        command = [str(executable), "--root", str(root), "apply-binary-contexts", "--jobs", str(original / "jobs.json"),
                   "--assembly", str(root / "data/encodings/contexts-assembly-v2"), "--aliases", str(original / "aliases.json"),
                   "--output", str(output / side / "contexts")]
        for component in binary.COMPONENTS:
            command += ["--" + component, str(output / side / "components" / component)]
        commands[side + "-contexts"] = [command]
    return commands


def audit(root, manifest_path):
    """Reproduce the full replay claim without launching or modifying anything."""
    output = manifest_path.parent
    manifest = binary.read(manifest_path)
    saved = binary.read(output / "verification.json")
    runtime.require(saved["manifest_sha256"] == binary.sha(manifest_path), "replay manifest binding changed")
    runtime.require(manifest.get("views") == manifest.get("pairs") == 80 and manifest.get("raw_components_per_side") == 240
                    and saved.get("independently_computed_contexts") == 160 and saved.get("independently_computed_raw_components") == 480,
                    "replay reported counts disagree with the required cohort")
    waves = sorted((output / "waves").glob("*.json"))
    jobs = [job for path in waves for job in binary.read(path)["jobs"]]
    runtime.require(len(waves) == 10 and len(jobs) == 80 and len({(j["prepared"], j["cut"]) for j in jobs}) == 80, "replay jobs incomplete or duplicated")
    # Earlier v2 manifests bind these exact preparation paths in their ten waves.
    original = root / manifest["source_experiment"] if "source_experiment" in manifest else (root / jobs[0]["prepared"]).parents[1]
    runtime.require(binary.sha(original / "jobs.json") == manifest["jobs_sha256"], "original replay jobs changed")
    runtime.require(binary.verify(root, original)["report_sha256"] == manifest["original_report_sha256"], "original replay artifacts changed")
    executable = root / manifest.get("executable", "bin/jope-parallel-neural-v56-experiment")
    runtime.require(binary.sha(executable) == manifest["encoder_sha256"], "frozen replay executable changed")
    if "configuration" in manifest:
        runtime.require(configuration(executable) == manifest["configuration"], "replay numerical environment changed")
    runtime.require([binary.read(p) for p in waves] == serial_job_waves(binary.read(original / "jobs.json")), "serial replay waves changed")
    specs = {name: original / "sampling" / f"{name}-spec.json" for name in ("target", "foundation")}
    specs["hyperion"] = root / "manifests/hyperion-reviewed4-safe-spec.json"
    runtime.require({k: binary.sha(v) for k, v in specs.items()} == manifest["specs"], "frozen replay specifications changed")
    expected_commands = commands_for_replay(root, original, output, executable, waves, specs)
    stage_files = list((output / "stages").glob("*.json"))
    runtime.require({p.stem for p in stage_files} == set(expected_commands) and len(stage_files) == 62, "missing full replay stage")
    for path in stage_files:
        stage = binary.read(path)
        commands = expected_commands[path.stem]
        runtime.require(stage["commands_sha256"] == binary.compact_sha(commands), "replay stage command changed")
        stages.verify(root, commands, stage)
    bound_vectors = {root / name for path in stage_files for name in binary.read(path)["outputs"] if name.endswith(".f32")}
    runtime.require(set((output / "a").rglob("*.f32")) | set((output / "b").rglob("*.f32")) == bound_vectors, "unexpected replay vector artifact")
    indexes = {side: binary.keyed(binary.read(output / side / "contexts/index.json")["encodings"], side) for side in ("a", "b")}
    expected = {(binary.read(root / j["prepared"] / "prepared.json")["dataset"], j["cut"]): j for j in jobs}
    runtime.require(indexes["a"].keys() == indexes["b"].keys() == expected.keys(), "replay context coverage mismatch")
    pairs = []
    for key, job in expected.items():
        prepared = binary.read(root / job["prepared"] / "prepared.json")
        cut = next(c for c in prepared["cuts"] if c["index"] == key[1])
        compared = {}
        for component, (filename, dimensions) in binary.COMPONENTS.items():
            suffix = Path("components") / component / key[0] / f"cut-{key[1]}"
            for side in ("a", "b"):
                path = output / side / suffix / filename
                metadata = binary.read(path.parent / f"{component}.json")
                binary.vector(path, dimensions)
                runtime.require(metadata["prepared_sha256"] == job["prepared_sha256"] and metadata["training_sha256"] == cut["training_sha256"] and metadata["preprocessing_sha256"] == cut["preprocessing_sha256"] and metadata["signature_sha256"] == binary.sha(path), "replay component binding changed")
            compared[component] = compare_floats(output / "a" / suffix / filename, output / "b" / suffix / filename)
        for side in ("a", "b"):
            binary.vector(root / indexes[side][key]["path"], 2048)
        compared["context"] = compare_floats(root / indexes["a"][key]["path"], root / indexes["b"][key]["path"])
        pairs.append({"dataset": key[0], "cut": key[1], "comparisons": compared})
    actual = {(p["dataset"], p["cut"]): p for p in pairs}
    claimed = {(p["dataset"], p["cut"]): p for p in saved["pairs"]}
    runtime.require(len(saved["pairs"]) == 80 and actual == claimed, "saved replay comparisons differ from current bytes")
    passed = all(c["exact"] for p in pairs for c in p["comparisons"].values())
    exact_pairs = sum(all(c["exact"] for c in p["comparisons"].values()) for p in pairs)
    runtime.require(saved["passed"] == passed and saved["verified_pairs"] == exact_pairs and saved.get("compared_pairs", 80) == 80, "saved replay coverage claim changed")
    return dict(saved, passed=passed, read_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True, help="Read-only original binary experiment")
    parser.add_argument("--output", type=Path, required=True, help="New dedicated experiment output")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    root, original = args.root.resolve(), args.experiment.resolve()
    output = runtime.output_root(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    verified = binary.verify(root, original)
    jobs = binary.read(original / "jobs.json")
    runtime.require(len(jobs["jobs"]) == 80 and verified["component_hashes"] == 240, "full replay requires all 80 admitted contexts")
    original_run = binary.read(original / "encoding-run.json")
    runtime.require(binary.sha(args.binary) == original_run["executable_sha256"], "replay executable differs from frozen encoder")
    python_env = stages.python_environment(original)
    environment = runtime.child_env(**python_env, OMP_NUM_THREADS="2", JAIN_HYPERION_CUDA_MAX_LOAD_RATIO="1000")
    hyperion_spec = root / "manifests/hyperion-reviewed4-safe-spec.json"
    specs = {name: original / "sampling" / f"{name}-spec.json" for name in ("target", "foundation")}
    specs["hyperion"] = hyperion_spec
    hyperion = binary.read(hyperion_spec)
    request = {"schema": "parallel.binary_full_replay.v2", "views": 80, "pairs": 80, "raw_components_per_side": 240,
        "original_verified_contexts": verified["context_hashes"],
        "source_experiment": str(original.relative_to(root)), "executable": str(args.binary.relative_to(root)),
        "original_report_sha256": verified["report_sha256"], "jobs_sha256": binary.sha(original / "jobs.json"),
        "encoder_sha256": binary.sha(args.binary), "specs": {k: binary.sha(v) for k, v in specs.items()},
        "projection_sha256": original_run["projection_sha256"], "child_environment": environment,
        "configuration": configuration(args.binary),
        "dispatch": "serial single cut per rank; global job index modulo eight; consecutive waves",
        "exact_tolerance": 0, "prior_outputs_preserved": True, "raw_components_reused": False,
        "driver_sha256": binary.sha(Path(__file__)), "runtime_sha256": binary.sha(Path(runtime.__file__))}
    immutable(output / "manifest.json", request)
    waves = []
    for number, wave in enumerate(serial_job_waves(jobs)):
        path = output / "waves" / f"{number:02}.json"
        immutable(path, wave)
        waves.append(path)
    deadline = time.monotonic() + args.seconds
    stopping = False
    def interrupt(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)

    def run_stage(name, commands):
        receipt_path = output / "stages" / (name + ".json")
        command_sha = binary.compact_sha(commands)
        if receipt_path.exists():
            receipt = binary.read(receipt_path)
            runtime.require(receipt["commands_sha256"] == command_sha, "replay stage command changed")
            stages.verify(root, commands, receipt)
            return True
        while not stopping and time.monotonic() < deadline:
            children, logs = [], []
            begun = None
            try:
                with spare_lease(root, "binary-full-replay:" + name, gpu=True), runtime.owned_ranks(children):
                    begun = time.monotonic()
                    for rank, command in enumerate(commands):
                        path = output / "logs" / f"{name}-{time.time_ns()}-{rank}.log"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        log = path.open("w")
                        logs.append(log)
                        env = dict(environment, CUDA_VISIBLE_DEVICES=str(rank))
                        affinity = ",".join(map(str, SPARE_CPUS[rank*2:rank*2+2]))
                        children.append(runtime.popen(["taskset", "-c", affinity] + command, cwd=root, env=env, stdout=log, stderr=log))
                    while any(c.poll() is None for c in children):
                        reason = admission_reason(root)
                        if reason or stopping or time.monotonic() >= deadline:
                            raise BlockingIOError(reason or "bounded interruption")
                        runtime.require(not any(c.returncode not in (None, 0) for c in children), f"replay stage failed: {name}")
                        runtime.write(output / "progress.json", {"phase": name, "timestamp": int(time.time()), "published_stages": len(list((output / "stages").glob("*.json"))), "required_stages": 62, "replay_pairs_verified": 0})
                        time.sleep(.5)
                    runtime.require(all(c.returncode == 0 for c in children), f"replay stage failed: {name}")
                    outputs = stages.inventory(root, commands)
                    immutable(receipt_path, {"complete": True, "commands_sha256": command_sha,
                        "seconds": time.monotonic() - begun, "outputs": outputs})
                    return True
            except BlockingIOError as error:
                if begun is not None:
                    runtime.write(output / "cancellations" / f"{time.time_ns()}.json", {"stage": name, "reason": str(error), "seconds": time.monotonic() - begun})
                runtime.write(output / "progress.json", {"phase": "waiting", "next_stage": name, "timestamp": int(time.time()), "reason": str(error)})
                time.sleep(1)
            finally:
                for log in logs:
                    log.close()
        return False

    for name, commands in commands_for_replay(root, original, output, args.binary, waves, specs).items():
        if not run_stage(name, commands):
            return
    indexes = {side: binary.keyed(binary.read(output / side / "contexts/index.json")["encodings"], side) for side in ("a", "b")}
    runtime.require(indexes["a"].keys() == indexes["b"].keys() and len(indexes["a"]) == 80, "replay pair inventory incomplete")
    pairs = []
    for key in indexes["a"]:
        compared = {}
        for component, (filename, _) in binary.COMPONENTS.items():
            suffix = Path("components") / component / key[0] / f"cut-{key[1]}" / filename
            compared[component] = compare_floats(output / "a" / suffix, output / "b" / suffix)
        compared["context"] = compare_floats(root / indexes["a"][key]["path"], root / indexes["b"][key]["path"])
        pairs.append({"dataset": key[0], "cut": key[1], "comparisons": compared})
    passed = all(c["exact"] for p in pairs for c in p["comparisons"].values())
    runtime.require(binary.verify(root, original) == verified, "original binary artifacts changed during replay")
    runtime.require(configuration(args.binary) == request["configuration"], "replay numerical environment changed during computation")
    exact_pairs = sum(all(c["exact"] for c in p["comparisons"].values()) for p in pairs)
    immutable(output / "verification.json", {"schema": "parallel.binary_full_replay_verification.v2", "passed": passed,
        "pairs": pairs, "compared_pairs": 80, "verified_pairs": exact_pairs, "independently_computed_contexts": 160, "independently_computed_raw_components": 480,
        "manifest_sha256": binary.sha(output / "manifest.json"), "originals_unchanged": True,
        "stage_seconds": sum(binary.read(p)["seconds"] for p in (output / "stages").glob("*.json")),
        "cancelled_seconds": sum(binary.read(p)["seconds"] for p in (output / "cancellations").glob("*.json"))})
    runtime.write(output / "progress.json", {"phase": "completed" if passed else "numerical differences", "passed": passed,
        "timestamp": int(time.time()), "replay_pairs_verified": exact_pairs, "exact_pairs": exact_pairs})
    runtime.require(passed, "full replay contains numerical differences; evidence preserved")


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "binary")
