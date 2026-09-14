#!/usr/bin/env python3
"""Qualify serial Hyperion dispatch, then publish a separate deterministic index.

One cut per native process/rank removes timing-dependent cross-cut coalescing.
Weights, source rows, sweep seed, patch cap and projection remain frozen.
"""
import argparse
import experiment_runtime as runtime
import experiment_stages as stage_artifacts
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from binary_encode import owned_ranks, serial_job_waves, sha, write
from binary_reproduce import compare_floats, immutable
from experiment_scheduler import SPARE_CPUS, admission_reason, spare_lease


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("serial encoding interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, experiment = args.root.resolve(), args.experiment.resolve()
    python_env = stage_artifacts.python_environment(experiment)
    output = experiment / "deterministic-v2"
    output.mkdir(exist_ok=True)
    read = lambda path: json.loads(path.read_text())
    original_report = read(experiment / "report.json")
    original_run = read(experiment / "encoding-run.json")
    if not (original_report["completed_contexts"] == 80):
        raise ValueError('binary_serial: artifact validation failed at original line 40')
    if not (sha(args.binary) == original_run["executable_sha256"]):
        raise ValueError('binary_serial: artifact validation failed at original line 41')
    original_report_sha = sha(experiment / "report.json")
    from binary_verify import verify as verify_original
    original_audit = verify_original(root, experiment)
    immutable(output / "original-artifact-verification.json", original_audit)
    previous_replay = read(experiment / "reproducibility/verification.json")
    if not (previous_replay["independently_recomputed_views"] == 20):
        raise ValueError('binary_serial: artifact validation failed at original line 47')
    if not (previous_replay["max_abs_error"]["target"] == previous_replay["max_abs_error"]["foundation"] == 0.0):
        raise ValueError('binary_serial: artifact validation failed at original line 48')
    jobs = read(experiment / "jobs.json")
    initial = read(experiment / "jobs-initial.json")
    if not (jobs["jobs"][:20] == initial["jobs"] and len(initial["jobs"]) == 20):
        raise ValueError('binary_serial: artifact validation failed at original line 51')
    spec_path = root / "manifests/hyperion-reviewed4-safe-spec.json"
    spec = read(spec_path)
    if not (sha(spec_path) == original_run["hyperion_spec_sha256"]):
        raise ValueError('binary_serial: artifact validation failed at original line 54')
    if not (sha(root / spec["persistent_binary"]) == spec["persistent_binary_sha256"]):
        raise ValueError('binary_serial: artifact validation failed at original line 55')
    for job in jobs["jobs"]:
        if not (sha(root / job["prepared"] / "prepared.json") == job["prepared_sha256"]):
            raise ValueError('binary_serial: artifact validation failed at original line 57')
    policy = dict(schema="parallel.hyperion_serial_dispatch.v1", maximum_concurrent_cuts_per_rank=1,
        world=8, cpu_affinity=list(SPARE_CPUS), unchanged_native_binary=spec["persistent_binary"],
        unchanged_native_binary_sha256=spec["persistent_binary_sha256"], unchanged_spec_sha256=sha(spec_path),
        maximum_forward_cells=24000, frozen_sweep_seed=True,
        scheduling="consecutive waves of at most eight jobs; job global_index modulo 8 remains its GPU rank",
        cross_cut_coalescing=False, numerical_rounding_added=False, architecture_changed=False,
        target_and_foundation_reused_after_exact_fresh_recomputation=True)
    immutable(output / "launch-policy.json", policy)
    waves = []
    for number, wave in enumerate(serial_job_waves(jobs)):
        path = output / "waves" / f"all-{number:02}.json"
        immutable(path, wave)
        waves.append(path)
    initial_waves = []
    for number, wave in enumerate(serial_job_waves(initial)):
        path = output / "waves" / f"initial-{number:02}.json"
        immutable(path, wave)
        initial_waves.append(path)
    request = dict(schema="parallel.binary_deterministic_version.v1", source_sha256=sha(Path(__file__)),
        helper_sources={name:sha(Path(__file__).with_name(name)) for name in ["binary_encode.py", "binary_reproduce.py", "binary_verify.py", "experiment_scheduler.py"]},
        jobs_sha256=sha(experiment / "jobs.json"), initial_jobs_sha256=sha(experiment / "jobs-initial.json"),
        launch_policy_sha256=sha(output / "launch-policy.json"), original_report_sha256=original_report_sha,
        binary_sha256=sha(args.binary), projection_sha256=original_run["projection_sha256"],
        strict_pair_requirement="20 of 20 Hyperion vectors and projected contexts byte-identical; tolerance 0",
        original_independent_replay_sha256=sha(experiment / "reproducibility/verification.json"),
        initial_pair_views=20, maximum_contexts=80, qualification=True, original_outputs_preserved=True)
    immutable(output / "request.json", request)
    if args.prepare_only:
        write(output / "progress.json", dict(phase="prepared", reason="Awaiting secondary comparison and scheduler admission", timestamp=int(time.time())))
        return
    deadline = min(time.time()+args.seconds, args.deadline_unix or float("inf"))

    def run_stage(name, commands):
        complete = output / "stages" / f"{name}.json"
        command_sha = hashlib.sha256(json.dumps(commands).encode()).hexdigest()
        if complete.exists():
            if not (read(complete)["commands_sha256"] == command_sha):
                raise ValueError('binary_serial: artifact validation failed at original line 94')
            stage_artifacts.verify(root, [c for _, c in commands], read(complete))
            return True
        while time.time() < deadline:
            children, logs = [], []
            try:
                with spare_lease(root, "binary-serial:"+name, gpu=True), owned_ranks(children):
                    begun = time.monotonic()
                    for rank, command in commands:
                        log_path = output / "logs" / f"{name}-{time.time_ns()}-{rank}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("w")
                        logs.append(log)
                        cpus = SPARE_CPUS[rank*2:rank*2+2]
                        env = runtime.child_env(**python_env, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", JAIN_HYPERION_CUDA_MAX_LOAD_RATIO="1000")
                        children.append(runtime.popen(["taskset","-c",",".join(map(str,cpus))]+command,
                            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    while any(c.poll() is None for c in children):
                        reason = admission_reason(root)
                        if reason or time.time() >= deadline:
                            write(output / "cancellations" / f"{name}-{time.time_ns()}.json", dict(stage=name,
                                reason=reason or "deadline", seconds=time.monotonic()-begun, artifacts_preserved=True))
                            raise BlockingIOError(reason or "deadline")
                        if not (not any(c.returncode not in [None,0] for c in children)):
                            raise ValueError(f"{name} failed")
                        write(output / "progress.json", dict(phase=name, reason="One active cut per GPU rank; frozen kernels and inputs", timestamp=int(time.time())))
                        time.sleep(1)
                    if not (all(c.returncode == 0 for c in children)):
                        raise ValueError(f"{name} failed")
                    immutable(complete, dict(stage=name, complete=True, commands_sha256=command_sha,
                        seconds=time.monotonic()-begun, timestamp=int(time.time()), outputs=stage_artifacts.inventory(root, [c for _, c in commands])))
                    return True
            except BlockingIOError as error:
                write(output / "progress.json", dict(phase="waiting", reason=str(error), timestamp=int(time.time())))
                time.sleep(2)
            finally:
                for log in logs:
                    log.close()
        write(output / "progress.json", dict(phase="paused", reason="Bounded deadline; completed artifacts preserved", timestamp=int(time.time())))
        return False

    def hyperion_commands(wave, destination):
        return [(rank, [str(root/spec["persistent_binary"]), "--root", str(root), "--jobs", str(wave),
            "--spec", str(spec_path), "--output", str(destination), "--rank", str(rank), "--world", "8"])
            for rank in range(len(read(wave)["jobs"]))]

    def context_command(job_path, hyperion, destination):
        return [(0,[str(args.binary),"--root",str(root),"apply-binary-contexts","--jobs",str(job_path),
            "--hyperion",str(hyperion),"--foundation",str(experiment / "components/foundation"),
            "--target",str(experiment / "components/target"),"--assembly",str(root / "data/encodings/contexts-assembly-v2"),
            "--output",str(destination),"--aliases",str(experiment / "aliases.json")])]

    def verify_outputs(manifest, hyperion, index):
        rows = {(row["dataset"],row["cut"]):row for row in index["encodings"]}
        if not (len(rows) == len(manifest["jobs"])):
            raise ValueError('binary_serial: artifact validation failed at original line 146')
        for job in manifest["jobs"]:
            prepared = read(root / job["prepared"] / "prepared.json")
            if not (sha(root / job["prepared"] / "prepared.json") == job["prepared_sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 149')
            cut = next(c for c in prepared["cuts"] if c["index"] == job["cut"])
            directory = hyperion / prepared["dataset"] / f"cut-{job['cut']}"
            metadata = read(directory / "hyperion.json")
            if not (metadata["prepared_sha256"] == job["prepared_sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 153')
            if not (metadata["training_sha256"] == cut["training_sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 154')
            if not (metadata["preprocessing_sha256"] == cut["preprocessing_sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 155')
            if not (metadata["spec_sha256"] == sha(spec_path)):
                raise ValueError('binary_serial: artifact validation failed at original line 156')
            if not (metadata["signature_sha256"] == sha(directory / "hyperion.f32")):
                raise ValueError('binary_serial: artifact validation failed at original line 157')
            if not (metadata["final_test_targets_used"] is False):
                raise ValueError('binary_serial: artifact validation failed at original line 158')
            row = rows[prepared["dataset"],job["cut"]]
            if not (sha(root / row["path"]) == row["sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 160')
            if not (sha((root / row["path"]).with_suffix(".json")) == row["receipt_sha256"]):
                raise ValueError('binary_serial: artifact validation failed at original line 161')

    for label in ["a", "b"]:
        directory = output / f"pair-{label}"
        for wave, path in enumerate(initial_waves):
            if not run_stage(f"pair-{label}-wave-{wave}", hyperion_commands(path,directory / "hyperion")):
                return
        if not run_stage(f"pair-{label}-contexts",context_command(experiment / "jobs-initial.json",directory / "hyperion",directory / "contexts")):
            return
    pairs = []
    index_a, index_b = read(output / "pair-a/contexts/index.json"), read(output / "pair-b/contexts/index.json")
    verify_outputs(initial, output / "pair-a/hyperion", index_a)
    verify_outputs(initial, output / "pair-b/hyperion", index_b)
    a = {(r["dataset"],r["cut"]):r for r in index_a["encodings"]}
    b = {(r["dataset"],r["cut"]):r for r in index_b["encodings"]}
    if not (a.keys() == b.keys() and len(a) == 20):
        raise ValueError('binary_serial: artifact validation failed at original line 176')
    for key in a:
        suffix = Path(key[0]) / f"cut-{key[1]}/hyperion.f32"
        hyperion = compare_floats(output / "pair-a/hyperion" / suffix,output / "pair-b/hyperion" / suffix)
        context = compare_floats(root/a[key]["path"],root/b[key]["path"])
        pairs.append(dict(dataset=key[0],cut=key[1],hyperion=hyperion,context=context))
    passed = all(p["hyperion"]["exact"] and p["context"]["exact"] for p in pairs)
    qualification = dict(schema="parallel.binary_serial_pair_qualification.v1", passed=passed, views=20,
        exact_equality_tolerance=0.0, launch_policy_sha256=sha(output / "launch-policy.json"),
        max_abs_error={k:max(p[k]["max_abs_error"] for p in pairs) for k in ["hyperion","context"]}, pairs=pairs)
    immutable(output / "pair-qualification.json", qualification)
    if not passed:
        write(output / "progress.json", dict(phase="numerical differences",passed=False,reason="Serial dispatch did not establish exact reproducibility",timestamp=int(time.time())))
        return
    # Extend pair A by reference: its first20 immutable component files stay intact.
    canonical_hyperion = output / "pair-a/hyperion"
    for wave, path in enumerate(waves):
        if not run_stage(f"full-wave-{wave}",hyperion_commands(path,canonical_hyperion)):
            return
    if not run_stage("full-contexts",context_command(experiment / "jobs.json",canonical_hyperion,output / "contexts")):
        return
    index = read(output / "contexts/index.json")
    verify_outputs(jobs, canonical_hyperion, index)
    wrapper = dict(index, launch_policy=str((output / "launch-policy.json").relative_to(root)),
        launch_policy_sha256=sha(output / "launch-policy.json"),
        native_index_sha256=sha(output / "contexts/index.json"), serial_pair_qualification_sha256=sha(output / "pair-qualification.json"))
    immutable(output / "index.json",wrapper)
    if not (sha(experiment / "report.json") == original_report_sha):
        raise ValueError('binary_serial: artifact validation failed at original line 203')
    if not (verify_original(root, experiment) == original_audit):
        raise ValueError('binary_serial: artifact validation failed at original line 204')
    final = dict(schema="parallel.binary_deterministic_publication.v1", completed_contexts=len(index["encodings"]),
        admitted_contexts=80, pair_reproducibility_passed=True, pair_views=20, pair_max_abs_error=qualification["max_abs_error"],
        context_index=str((output / "index.json").relative_to(root)), context_index_sha256=sha(output / "index.json"),
        original_80_preserved=True, original_report_sha256=original_report_sha,
        launch_policy_sha256=sha(output / "launch-policy.json"), target_and_foundation_reused=True,
        hyperion_recomputed_with_serial_dispatch=True, projection_refitted=False, copied_cut_tables=0,
        seconds=sum(read(path)["seconds"] for path in (output / "stages").glob("*.json")))
    immutable(output / "report.json",final)
    write(output / "progress.json",dict(phase="completed",completed=80,total=80,passed=True,timestamp=int(time.time())))
    print(json.dumps(final))


if __name__ == "__main__":
    runtime.entrypoint(main, "--experiment", "binary")
