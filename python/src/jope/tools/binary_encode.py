#!/usr/bin/env python3
"""Resume binary encoders in campaign-approved windows using one spare CPU lease."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import experiment_runtime as runtime
import experiment_stages as stage_artifacts
from experiment_scheduler import SPARE_CPUS, admission_reason, spare_lease


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


write = runtime.write


def serial_job_waves(manifest, world=8):
    """Keep each job's GPU assignment while allowing at most one cut per rank."""
    if not (world > 0):
        raise ValueError('binary_encode: artifact validation failed at original line 26')
    return [dict(manifest, jobs=manifest["jobs"][offset:offset+world])
            for offset in range(0, len(manifest["jobs"]), world)]


stop = runtime.stop
owned_ranks = runtime.owned_ranks


def main():
    def terminate(_signal, _frame):
        raise KeyboardInterrupt("encoding runner terminated; owned ranks will be stopped")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--initial-only", action="store_true", help="Save the first 20 contexts, then release resources for the primary comparison")
    parser.add_argument("--serial-hyperion", action="store_true", help="Freeze one-cut-per-GPU dispatch for a new encoding run; existing concurrent runs cannot change policy")
    args = parser.parse_args()
    root, experiment = args.root.resolve(), args.experiment.resolve()
    python_env = stage_artifacts.python_environment(experiment)
    if not (experiment.is_relative_to(root)):
        raise ValueError('binary_encode: artifact validation failed at original line 49')
    preparation = json.loads((experiment / "preparation-report.json").read_text())
    if not (preparation["admitted_contexts"] > 0):
        raise ValueError('binary_encode: artifact validation failed at original line 51')
    if not (sha(experiment / "aliases.json") == preparation["aliases_sha256"]):
        raise ValueError('binary_encode: artifact validation failed at original line 52')
    if not (sha(experiment / "jobs.json") == preparation["jobs_sha256"]):
        raise ValueError('binary_encode: artifact validation failed at original line 53')
    dependency_receipt = experiment / "python-deps.json"
    if dependency_receipt.exists():
        dependencies = json.loads(dependency_receipt.read_text())
        for path, expected in dependencies["files"].items():
            if not (sha(experiment / path) == expected):
                raise ValueError("isolated Python dependency changed")
        os.environ["PYTHONPATH"] = str(experiment / "python-deps") + os.pathsep + os.environ.get("PYTHONPATH", "")
    started = time.monotonic()
    deadline = started + args.seconds
    sampling = experiment / "sampling"
    component_dirs = {name: experiment / "components" / name for name in ["target", "foundation", "hyperion"]}
    extension = [args.python, str(Path(__file__).with_name("extend_component_sampling.py")), "--root", str(root),
        "--jobs", str(experiment / "jobs.json"), "--output", str(sampling),
        "--target-spec", "assets/sampling-reviewed4-v1/target-spec.json",
        "--foundation-spec", "assets/sampling-reviewed4-v1/foundation-spec.json"]
    hyperion_spec = root / "manifests/hyperion-reviewed4-safe-spec.json"
    spec = json.loads(hyperion_spec.read_text())
    if not (spec["concurrent_gp"] and sha(root / spec["persistent_binary"]) == spec["persistent_binary_sha256"]):
        raise ValueError('binary_encode: artifact validation failed at original line 70')
    # Bind the command inputs before any GPU work; binaries and original components stay immutable.
    binding = dict(schema="parallel.binary_encoding_run.v2", child_environment=runtime.child_env(**python_env), driver_sha256=sha(Path(__file__)), jobs_sha256=sha(experiment / "jobs.json"),
        aliases_sha256=sha(experiment / "aliases.json"), executable_sha256=sha(args.binary),
        hyperion_spec_sha256=sha(hyperion_spec), projection_sha256=sha(root / "data/encodings/contexts-assembly-v2/assembly.json"),
        projection_refitted=False, qualification=True, cpus=list(SPARE_CPUS), gpu_world=8)
    if args.serial_hyperion:
        policy = dict(schema="parallel.hyperion_serial_dispatch.v1", maximum_concurrent_cuts_per_rank=1,
            world=8, cpu_affinity=list(SPARE_CPUS), native_binary_sha256=spec["persistent_binary_sha256"],
            spec_sha256=sha(hyperion_spec), scheduling="consecutive waves; original job index modulo 8 remains GPU rank",
            cross_cut_coalescing=False, numerical_rounding_added=False, architecture_changed=False)
        binding["hyperion_dispatch"] = policy
    if (experiment / "encoding-run.json").exists():
        if not (json.loads((experiment / "encoding-run.json").read_text()) == binding):
            raise ValueError('binary_encode: artifact validation failed at original line 83')
    else:
        write(experiment / "encoding-run.json", binding)

    def coverage():
        return {component: len(list(directory.glob("*/cut-*/" + component + ".json"))) for component, directory in component_dirs.items()}

    def publish(phase, reason, **extra):
        completed = 0
        for path in [experiment / "contexts-initial/index.json", experiment / "contexts/index.json"]:
            if path.exists():
                completed = max(completed, len(json.loads(path.read_text())["encodings"]))
        write(experiment / "progress.json", dict(schema="parallel.binary_encoding_progress.v1", timestamp=int(time.time()),
            phase=phase, reason=reason, admitted_datasets=preparation["admitted_datasets"], total=preparation["admitted_contexts"],
            completed=completed, components=coverage(), cpu_budget=16, gpu_world=8,
            elapsed_seconds=time.monotonic()-started, qualification=True, **extra))

    stages = [("sampling-prepare", [extension + ["--prepare-only"]], False),
        ("sampling-qualify", [extension + ["--qualify-cuda", "--rank", str(rank), "--world-size", "8"] for rank in range(8)], True),
        ("sampling-publish", [extension + ["--merge-qualification", "--world-size", "8"]], False)]
    cohorts = [("initial", "jobs-initial.json")]
    if not args.initial_only:
        cohorts.append(("expanded", "jobs.json"))
    for cohort, jobfile in cohorts:
        jobs = experiment / jobfile
        for component in ["target", "foundation", "hyperion"]:
            if component == "hyperion" and args.serial_hyperion:
                for wave, manifest in enumerate(serial_job_waves(json.loads(jobs.read_text()))):
                    wavepath = experiment / "hyperion-waves" / f"{cohort}-{wave:02}.json"
                    if wavepath.exists():
                        if not (json.loads(wavepath.read_text()) == manifest):
                            raise ValueError('binary_encode: artifact validation failed at original line 113')
                    else:
                        write(wavepath, manifest)
                    commands = [[str(root / spec["persistent_binary"]), "--root", str(root), "--jobs", str(wavepath),
                        "--spec", str(hyperion_spec), "--output", str(component_dirs[component]),
                        "--rank", str(rank), "--world", "8"] for rank in range(len(manifest["jobs"]))]
                    stages.append((f"{cohort}-hyperion-wave-{wave:02}", commands, True))
                continue
            commands = []
            for rank in range(8):
                if component == "hyperion":
                    command = [str(root / spec["persistent_binary"]), "--root", str(root)]
                    component_spec = hyperion_spec
                else:
                    command = [str(args.binary), "--root", str(root), component + "-rank"]
                    component_spec = sampling / f"{component}-spec.json"
                command += ["--jobs", str(jobs), "--spec", str(component_spec), "--output", str(component_dirs[component]),
                    "--rank", str(rank), "--world", "8"]
                if component != "hyperion":
                    command += ["--batch", "128" if component == "target" else "2"]
                commands.append(command)
            stages.append((f"{cohort}-{component}", commands, True))
        commands = [[str(args.binary), "--root", str(root), "apply-binary-contexts", "--jobs", str(jobs),
            "--hyperion", str(component_dirs["hyperion"]), "--foundation", str(component_dirs["foundation"]),
            "--target", str(component_dirs["target"]), "--assembly", str(root / "data/encodings/contexts-assembly-v2"),
            "--output", str(experiment / ("contexts-initial" if cohort == "initial" else "contexts")),
            "--aliases", str(experiment / "aliases.json")]]
        stages.append((f"{cohort}-contexts", commands, True))

    for stage, commands, gpu in stages:
        receipt = experiment / "stages" / f"{stage}.json"
        command_sha = hashlib.sha256(json.dumps(commands).encode()).hexdigest()
        if receipt.exists():
            if not (json.loads(receipt.read_text())["commands_sha256"] == command_sha):
                raise ValueError('binary_encode: artifact validation failed at original line 146')
            stage_artifacts.verify(root, commands, json.loads(receipt.read_text()))
            continue
        attempt = 0
        while time.monotonic() < deadline:
            children, logs = [], []
            try:
                with spare_lease(root, "binary-qualification:" + stage, gpu=gpu), owned_ranks(children):
                    attempt += 1
                    begun = time.monotonic()
                    for rank, command in enumerate(commands):
                        logpath = experiment / "logs" / f"{stage}-{int(time.time())}-{rank}.log"
                        logpath.parent.mkdir(parents=True, exist_ok=True)
                        log = logpath.open("w")
                        logs.append(log)
                        cpus = SPARE_CPUS[rank*2:rank*2+2]
                        environment = runtime.child_env(**python_env, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", JAIN_HYPERION_CUDA_MAX_LOAD_RATIO="1000")
                        children.append(runtime.popen(["taskset", "-c", ",".join(map(str, cpus))] + command,
                            cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    while any(child.poll() is None for child in children):
                        reason = admission_reason(root)
                        if reason or time.monotonic() >= deadline:
                            stop(children)
                            cancellation = dict(schema="parallel.binary_encoding_cancellation.v1", stage=stage, attempt=attempt,
                                timestamp=int(time.time()), reason=reason or "bounded experiment deadline", seconds=time.monotonic()-begun,
                                completed_components_preserved=coverage(), commands_sha256=command_sha)
                            write(experiment / "cancellations" / f"{stage}-{time.time_ns()}.json", cancellation)
                            raise BlockingIOError(cancellation["reason"])
                        failed = [child.returncode for child in children if child.returncode not in [None, 0]]
                        if failed:
                            stop(children)
                            raise RuntimeError(f"{stage} child failed {failed}; inspect logs")
                        publish(stage, "Frozen encoders hold the shared spare CPU allocation" if gpu else "Preparing verified sampling recipes", attempt=attempt)
                        time.sleep(1)
                    if not (all(child.returncode == 0 for child in children)):
                        raise ValueError(f"{stage} failed; inspect logs")
                    write(receipt, dict(schema="parallel.binary_encoding_stage.v1", stage=stage, complete=True,
                        timestamp=int(time.time()), commands_sha256=command_sha, seconds=time.monotonic()-begun, components=coverage(), outputs=stage_artifacts.inventory(root, commands)))
                    break
            except BlockingIOError as error:
                publish("waiting", str(error), next_stage=stage)
                time.sleep(2)
            except KeyboardInterrupt as error:
                write(experiment / "cancellations" / f"{stage}-{time.time_ns()}.json",
                    dict(schema="parallel.binary_encoding_cancellation.v1", stage=stage,
                        timestamp=int(time.time()), reason=str(error), commands_sha256=command_sha,
                        completed_components_preserved=coverage()))
                publish("paused", "Encoding cancelled; completed components preserved", next_stage=stage)
                return
            except BaseException as error:
                stop(children)
                publish("failed", str(error), next_stage=stage)
                raise
            finally:
                for log in logs:
                    log.close()
        else:
            publish("paused", "Bounded encoding window expired; saved artifacts resume with the same command", next_stage=stage)
            return
    index = experiment / ("contexts-initial/index.json" if args.initial_only else "contexts/index.json")
    data = json.loads(index.read_text())
    for row in data["encodings"]:
        if not (sha(root / row["path"]) == row["sha256"]):
            raise ValueError('binary_encode: artifact validation failed at original line 206')
    saved_report = experiment / "report.json"
    if saved_report.exists():
        previous = json.loads(saved_report.read_text())
        if previous.get("status") == "completed":
            from binary_verify import verify
            receipt = verify(root, experiment)
            receipt["completed_stages_skipped_without_gpu_admission"] = True
            write(experiment / "restart-verification.json", receipt)
            return
    report = dict(preparation, schema="parallel.binary_qualification_report.v1", completed_contexts=len(data["encodings"]),
        status="initial_complete_expansion_pending" if args.initial_only else "completed",
        components=coverage(), context_index=str(index.relative_to(root)), context_index_sha256=sha(index),
        encoders_frozen=True, regression_projection_reused=True, binary_index_separate=True,
        cancellations=len(list((experiment / "cancellations").glob("*.json"))), elapsed_seconds=time.monotonic()-started)
    if args.serial_hyperion:
        report["hyperion_dispatch"] = binding["hyperion_dispatch"]
        report["encoding_run_sha256"] = sha(experiment / "encoding-run.json")
    write(experiment / "report.json", report)
    publish("initial completed" if args.initial_only else "completed",
        "Initial contexts verified; resources released for primary seed comparison" if args.initial_only else "All admitted binary contexts and original component hashes verified",
        complete=not args.initial_only)


if __name__ == "__main__":
    runtime.entrypoint(main, "--experiment", "binary")
