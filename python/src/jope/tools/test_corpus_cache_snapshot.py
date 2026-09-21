#!/usr/bin/env python3
"""Validate duplicate handling against a frozen, real multi-cut corpus sample."""

import argparse
import copy
import hashlib
import json
import os
import pathlib
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    sample = root / "runs/parity/corpus-cache-v1"
    labels = {
        row["program"]
        for row in map(json.loads, (sample / "baseline-cpu/labels.jsonl").read_text().splitlines())
    }
    ledger = next(iter(sorted((sample / "ledgers").iterdir())))
    original = next(
        row
        for row in json.loads((ledger / "observations.json").read_text())
        if row["genome"] in labels and row["failure"] is None
    )
    receipts = []
    for mode, expected in [
        ("different_valid_score", None),
        ("corrupt_hash", "program identity mismatch"),
        ("invalid_score", "invalid score"),
    ]:
        directory = sample / "duplicate-checks" / mode
        directory.mkdir(parents=True)
        for source in sorted((sample / "ledgers").iterdir()):
            dest = directory / source.name
            dest.mkdir()
            for name in ["observations.json", "search.json"]:
                (dest / name).symlink_to(source / name)
        duplicate = copy.deepcopy(original)
        if mode == "different_valid_score":
            duplicate["metrics"]["rmse"] = 9999999.0
            duplicate["runtime_seconds"] = 999.0
            duplicate["evaluation_bias"] = "deliberately_changed_test_fixture"
        elif mode == "corrupt_hash":
            duplicate["program_sha256"] = "0" * 64
        else:
            duplicate["metrics"]["rmse"] = -1.0
        extra = directory / "zzz-late-duplicate"
        extra.mkdir()
        (extra / "observations.json").write_text(json.dumps([duplicate]) + "\n")
        (extra / "search.json").symlink_to(ledger / "search.json")
        output = sample / f"duplicate-check-{mode}"
        command = [
            "taskset",
            "-c",
            "60",
            str(a.binary),
            "--root",
            str(root),
            "build-corpus",
            "--contexts",
            str(root / "data/encodings/contexts-union-v2.json"),
            "--vocabulary",
            str(root / "runs/regression-cycle-2/corpus-2/vocabulary.json"),
            "--observations",
            str(directory),
            "--output",
            str(output),
            "--qualification",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                RAYON_NUM_THREADS="1",
            ),
        )
        if expected is None:
            assert result.returncode == 0, result.stderr
            hashes = {name: sha(output / name) for name in ["records.bin", "labels.jsonl", "index.json"]}
            assert all(value == sha(sample / "baseline-cpu" / name) for name, value in hashes.items())
            receipts.append(dict(case=mode, first_observation_preserved=True, artifacts=hashes, passed=True))
        else:
            assert result.returncode != 0 and expected in result.stderr, result.stderr
            assert not output.exists(), "Rejected input produced a corpus"
            receipts.append(
                dict(case=mode, rejected_before_duplicate_skip=True, expected_error=expected, passed=True)
            )
    receipt = dict(
        schema="parallel.corpus_duplicate_validation.v1",
        passed=True,
        executable_sha256=sha(a.binary),
        sample_sha256=sha(sample / "sample.json"),
        cases=receipts,
        original_artifacts_modified=False,
        final_tests_opened=False,
    )
    (sample / "duplicate-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
