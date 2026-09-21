#!/usr/bin/env python3
"""Dispatch bounded Rust pointer-cut expansion; publish jobs only after validation."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "input", "output", "jobs", "gp-output", "report", "executable"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vocabulary", default="assets/grammar/featmath.json")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()

    def path(value):
        relative = pathlib.Path(value)
        assert not relative.is_absolute() and ".." not in relative.parts
        return root / relative

    source, destination = path(args.input), path(args.output)
    jobs_path, report_path = path(args.jobs), path(args.report)
    executable, vocabulary = path(args.executable), path(args.vocabulary)
    assert 1 <= args.workers <= 4 and 1 <= args.rounds <= 1000
    assert not jobs_path.exists() and not report_path.exists()
    inputs = sorted(source.glob("*/prepared.json"))
    assert inputs
    started = time.monotonic()

    def expand(manifest):
        parent = json.loads(manifest.read_text())
        output = destination / parent["dataset"]
        if not (output / "prepared.json").exists():
            command = [
                str(executable),
                "expand-cuts",
                "--prepared",
                str(manifest.parent),
                "--output",
                str(output),
                "--rounds",
                str(args.rounds),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "1",
                },
            )
            if result.returncode:
                raise RuntimeError(f"{parent['dataset']}: {result.stderr}")
        prepared = json.loads((output / "prepared.json").read_text())
        recipe = json.loads((output / "expansion.json").read_text())
        assert recipe["parent_prepared_sha256"] == sha(manifest)
        assert recipe["rounds"] == args.rounds
        for field in (
            "dataset",
            "family",
            "meta_partition",
            "storage",
            "entity_columns",
            "family_registry_binding",
            "original_train_sha256",
            "original_test_sha256",
            "selected_train_indices",
            "selected_test_indices",
        ):
            assert parent.get(field) == prepared.get(field), (parent["dataset"], field)
        assert not list(output.rglob("*.csv")), "CSV copying is forbidden"
        memberships = set()
        for cut in prepared["cuts"]:
            membership = tuple(cut["train_indices"])
            assert membership not in memberships
            memberships.add(membership)
            assert not set(membership).intersection(cut["validation_indices"])
            assert sha(output / f"cut-{cut['index']}" / "preprocessing.json") == cut["preprocessing_sha256"]
        assert prepared["cuts"]
        result = {
            "dataset": parent["dataset"],
            "prepared": str(output.relative_to(root)),
            "prepared_sha256": sha(output / "prepared.json"),
            "cuts": len(prepared["cuts"]),
            "duplicate_cuts_skipped": recipe["duplicate_cuts_skipped"],
            "family": prepared["family"],
            "partition": prepared["meta_partition"],
            "copied_dataset_bytes": 0,
        }
        jobs = [
            {
                "prepared": result["prepared"],
                "prepared_sha256": result["prepared_sha256"],
                "cut": cut["index"],
                "output": f"{args.gp_output}/{parent['dataset']}/cut-{cut['index']}",
            }
            for cut in prepared["cuts"]
        ]
        return result, jobs

    results, jobs = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(expand, manifest) for manifest in inputs]):
            result, expanded_jobs = future.result()
            results.append(result)
            jobs.extend(expanded_jobs)
            print(json.dumps({"completed_datasets": len(results), **result}), flush=True)
    results.sort(key=lambda row: row["dataset"])
    jobs.sort(key=lambda row: (row["prepared"], row["cut"]))
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_manifest = {
        "schema": "parallel.gp_jobs.v1",
        "qualification": True,
        "vocabulary": args.vocabulary,
        "jobs": jobs,
    }
    jobs_path.write_text(json.dumps(jobs_manifest, indent=2) + "\n")
    report = {
        "schema": "parallel.expanded_reviewed_cohort.v1",
        "passed": True,
        "datasets": results,
        "jobs": len(jobs),
        "jobs_sha256": sha(jobs_path),
        "rounds": args.rounds,
        "executable": args.executable,
        "executable_sha256": sha(executable),
        "vocabulary_sha256": sha(vocabulary),
        "worker_processes": args.workers,
        "seconds": time.monotonic() - started,
        "copied_dataset_bytes": 0,
        "original_final_test_csv_opened": False,
        "enqueued": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "datasets"}), flush=True)


if __name__ == "__main__":
    main()
