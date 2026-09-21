#!/usr/bin/env python3
"""Bounded evaluator parity and synthetic search-receipt smoke test."""

import argparse
import fcntl
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import experiment_runtime as runtime
from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import read, require, sha


def fixture(output):
    prepared = output / "synthetic-fixture"
    cut = prepared / "cut-0"
    cut.mkdir(parents=True)
    for name, indices in [("train", range(24)), ("validation", range(24, 32))]:
        (cut / (name + ".csv")).write_text("".join(f"{i},{i * i},{7 + 3 * i}\n" for i in indices))
    immutable(
        cut / "preprocessing.json",
        dict(
            schema="parallel.preprocessing.v1",
            columns=[dict(kind="numeric", impute=0.0, mean=0.0, scale=1.0, categories=[]) for _ in range(2)],
            labels=[],
            target_mean=0.0,
            target_scale=1.0,
        ),
    )
    immutable(
        prepared / "prepared.json",
        dict(
            schema="parallel.prepared.v1",
            dataset="synthetic-emission-fixture",
            family="synthetic-emission-fixture",
            task="regression",
            meta_partition="meta_train",
            seed=17,
            family_partition_frozen=False,
            clean_evaluation=True,
            clean_status="synthetic_unit_fixture_not_teaching",
            original_train_sha256="synthetic",
            original_test_sha256="never-opened",
            selected_train_indices=[],
            selected_test_indices=[],
            target_last=True,
            rows_total=32,
            cuts=[
                dict(
                    index=0,
                    train_indices=list(range(24)),
                    validation_indices=list(range(24, 32)),
                    training_sha256=sha(cut / "train.csv"),
                    validation_sha256=sha(cut / "validation.csv"),
                    preprocessing_sha256=sha(cut / "preprocessing.json"),
                )
            ],
        ),
    )
    return prepared


def validate_emission(output, binary, vocabulary):
    rows = read(output / "observations.json")
    receipt = read(output / "search.json")
    require(
        len(rows) == 1 and receipt["completed_evaluations"] == 1, "synthetic smoke evaluation count changed"
    )
    require(
        receipt["schema"] == "parallel.search.v1"
        and receipt["completed_at"] > 0
        and receipt["observations_sha256"] == sha(output / "observations.json")
        and receipt["final_tests_opened"] is False,
        "search ledger publication is incomplete",
    )
    row = rows[0]
    for key in ("evaluator_sha256", "evaluator_protocol", "prepared_sha256", "cut", "vocabulary_sha256"):
        require(receipt[key] == row[key], "search/observation binding conflict: " + key)
    require(
        row["evaluator_sha256"] == sha(binary)
        and row["vocabulary_sha256"] == sha(vocabulary)
        and row["evaluator_protocol"] == "featmath_training_fold_normalized_v1"
        and row["resource_protocol"] == "single_candidate_30s_8gib_v1"
        and row["metric_units"] == row["metrics"]["units"] == "cut_normalized_target"
        and row["failure"] is None
        and row["evaluation_seed"] == 0,
        "native outcome identity is incomplete",
    )
    keys = (
        "program_sha256",
        "family",
        "partition",
        "prepared_sha256",
        "cut",
        "training_sha256",
        "validation_sha256",
        "evaluator_sha256",
        "evaluator_protocol",
        "preprocessing_sha256",
        "target_scale",
        "metric_units",
        "vocabulary_sha256",
        "feature_view_sha256",
        "evaluation_seed",
        "resource_protocol",
    )
    identity = {key: row[key] for key in keys}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == row["evaluation_identity_sha256"], "evaluation identity digest changed")


def run(root, manifest, binary, build, output, budget_output):
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "request.json").exists(), "emission qualification already frozen")
    build_receipt = read(build / "target-build.json")
    require(
        build_receipt["status"] == "passed"
        and build_receipt["bindings"]["jope-parallel-native"] == sha(binary),
        "unqualified native build",
    )
    m = read(manifest)
    gp = read(root / "runs/gp-service.json")
    reference = Path(gp["executable"])
    require(
        gp["workers"] == 104 and gp["threads_each"] == 1 and sha(reference) == gp["executable_sha256"],
        "production evaluator/allocation changed",
    )
    rows = read(root / m["contexts"])["encodings"]
    selected = {}
    for row in sorted(rows, key=lambda row: (row["family"], row["cut"], row["prepared_sha256"])):
        if row["partition"] == "meta_train" and row["family"] not in selected:
            selected[row["family"]] = row
        if len(selected) == 2:
            break
    require(len(selected) == 2, "two training families required for evaluator parity")
    vocabulary = root / read(root / m["output"] / "state.json")["serving_corpus"] / "vocabulary.json"
    request = dict(
        schema="parallel.gp_emission_qualification.v1",
        executable=str(binary.relative_to(root)),
        executable_sha256=sha(binary),
        reference=str(reference.relative_to(root)),
        reference_sha256=sha(reference),
        build_sha256=sha(build / "target-build.json"),
        contexts=list(selected.values()),
        vocabulary_sha256=sha(vocabulary),
        programs=["(pipe X (ridge))", "(pipe X (rank) (ridge))"],
        ceiling_seconds=300,
        cpu_width=2,
        driver_sha256=sha(Path(__file__)),
        final_tests_opened=False,
        synthetic_fixture_not_admitted=True,
        startup_repetitions=3,
        maximum_median_startup_seconds=0.1,
    )
    immutable(output / "request.json", request)
    budget = Budget(budget_output)
    reserved = False
    start = None
    results = []
    result = dict(status="inconclusive")
    try:
        with (root / "runs/experiment-spare-cpus.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            budget.reserve(output.name, core_seconds=600)
            reserved = True
            start = time.monotonic()

            def invoke(command, path, timeout=30):
                require(time.monotonic() - start + timeout <= 300, "emission qualification ceiling exhausted")
                env = runtime.child_env(
                    LD_LIBRARY_PATH="/usr/lib/python3/dist-packages/torch/lib",
                    OMP_NUM_THREADS="1",
                    RAYON_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                )
                env.pop("JOPE_EXPERIMENT_DIR", None)

                def limits():
                    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

                children = []
                with path.open("xb") as log, runtime.owned_ranks(children):
                    child = runtime.popen(command, env=env, stdout=log, stderr=log, preexec_fn=limits)
                    children.append(child)
                    child.wait(timeout=timeout)
                    require(child.returncode == 0, "native parity/emission failed: " + path.name)

            for number, row in enumerate(selected.values()):
                prepared = root / row["prepared"]
                require(sha(prepared / "prepared.json") == row["prepared_sha256"], "parity cut changed")
                require(
                    read(prepared / "prepared.json")["meta_partition"] == "meta_train",
                    "sealed parity context",
                )
                for index, program in enumerate(request["programs"]):
                    genome = output / f"program-{index}.dsl"
                    if not genome.exists():
                        genome.write_text(program)
                    pair = []
                    for name, exe in [("reference", reference), ("candidate", binary)]:
                        require(
                            sha(exe)
                            == request[name + "_sha256" if name == "reference" else "executable_sha256"],
                            "evaluator changed",
                        )
                        destination = output / f"{number}-{index}-{name}.json"
                        invoke(
                            [
                                str(exe),
                                "--root",
                                str(root),
                                "evaluate-view",
                                "--prepared",
                                str(prepared),
                                "--cut",
                                str(row["cut"]),
                                "--genome",
                                str(genome),
                                "--output",
                                str(destination),
                            ],
                            destination.with_suffix(".log"),
                        )
                        pair.append(read(destination))
                    require(
                        pair[0]["rmse"] == pair[1]["rmse"]
                        and pair[0]["semantic_sha256"] == pair[1]["semantic_sha256"],
                        "evaluator semantics or RMSE changed",
                    )
                    results.append(
                        dict(family=row["family"], cut=row["cut"], program=program, rmse=pair[0]["rmse"])
                    )
            prepared = fixture(output)
            smoke = output / "synthetic-search"
            invoke(
                [
                    str(binary),
                    "--root",
                    str(root),
                    "search",
                    "--prepared",
                    str(prepared),
                    "--cut",
                    "0",
                    "--output",
                    str(smoke),
                    "--vocabulary",
                    str(vocabulary),
                    "--evaluations",
                    "1",
                    "--seconds",
                    "10",
                    "--qualification",
                ],
                output / "smoke.log",
            )
            validate_emission(smoke, binary, vocabulary)
            startup = {}
            for name, exe in [("reference", reference), ("candidate", binary)]:
                values = []
                for i in range(3):
                    begun = time.monotonic()
                    invoke([str(exe), "--help"], output / f"startup-{name}-{i}.log", timeout=5)
                    values.append(time.monotonic() - begun)
                startup[name] = values
            require(
                sorted(startup["candidate"])[1] < 0.1,
                "CPU evaluator still has heavyweight process initialization",
            )
            result.update(
                status="passed", synthetic_receipt_sha256=sha(smoke / "search.json"), startup_seconds=startup
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["reason"] = str(error)
    finally:
        result.update(
            request_sha256=sha(output / "request.json"),
            seconds=None if start is None else time.monotonic() - start,
            parity=results,
            executable_sha256=request["executable_sha256"],
            reference_sha256=request["reference_sha256"],
            final_tests_opened=False,
        )
        immutable(output / "complete.json", result)
        if reserved:
            budget.complete(output.name, output / "complete.json")
    require(result["status"] == "passed", result.get("reason", "emission qualification incomplete"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "binary", "build", "output", "budget-output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(
        a.root.resolve(),
        a.manifest.resolve(),
        a.binary.resolve(),
        a.build.resolve(),
        a.output.resolve(),
        a.budget_output.resolve(),
    )
