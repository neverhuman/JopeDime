#!/usr/bin/env python3
"""Recreate the three pointer-only expansions and verify whole-entity cuts."""
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = ROOT / "manifests/generated/reviewed4-expansion.json"
    report = json.loads(report_path.read_text())
    executable = ROOT / report["executable"]
    assert sha(executable) == report["executable_sha256"]
    registry = ROOT / "manifests/generated/families/qualification-v4.json"
    review = ROOT / "manifests/generated/provenance-review/review-v3.json"
    originals = json.loads((ROOT / "manifests/generated/review-v3-transfer.json").read_text())["files"]
    verified_files = 0
    for entry in originals:
        assert sha(ROOT / entry["destination"]) == entry["sha256"]
        verified_files += 1
    results = []
    temporary_root = ROOT / "runs/verification"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reviewed4-reproduction-", dir=temporary_root) as temporary:
        for dataset in report["datasets"]:
            identity = dataset["dataset"]
            expected = ROOT / dataset["prepared"]
            parent = ROOT / "data/prepared/reviewed-cohort-v4" / identity
            reproduced = Path(temporary) / identity
            subprocess.run([str(executable), "expand-cuts", "--prepared", str(parent), "--output", str(reproduced), "--rounds", "20"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"})
            expected_files = {str(p.relative_to(expected)): sha(p) for p in expected.rglob("*") if p.is_file()}
            actual_files = {str(p.relative_to(reproduced)): sha(p) for p in reproduced.rglob("*") if p.is_file()}
            assert expected_files == actual_files, identity
            assert not list(reproduced.rglob("*.csv"))
            prepared = json.loads((expected / "prepared.json").read_text())
            assert prepared["family_registry_binding"]["registry_sha256"] == sha(registry)
            assert prepared["family_registry_binding"]["entity_review_sha256"] == sha(review)
            assert prepared["meta_partition"] == "meta_validation"
            source_entry = next(f for f in originals if f["dataset"] == identity and Path(f["source"]).name == "train.csv")
            # Only original training rows are read; original final-test CSVs are
            # touched exclusively by the opaque byte hash verification above.
            with (ROOT / source_entry["destination"]).open() as source:
                training_rows = list(csv.reader(source))
            group_columns = prepared.get("entity_columns", [])
            positions = [prepared["storage"]["feature_columns"].index(name) for name in group_columns]
            group_keys = [tuple(row[i].strip() for i in positions) for row in training_rows] if positions else None
            memberships = set()
            for cut in prepared["cuts"]:
                training, heldout = cut["train_indices"], cut["validation_indices"]
                key = tuple(sorted(training))
                assert key not in memberships
                memberships.add(key)
                assert not set(training).intersection(heldout)
                assert set(training).union(heldout) == set(prepared["selected_train_indices"])
                if group_keys:
                    assert not {group_keys[i] for i in training}.intersection(group_keys[i] for i in heldout)
            assert len(memberships) == 100
            results.append(dict(dataset=identity, cuts=100, identical_reproduction_files=len(expected_files),
                                prepared_sha256=sha(expected / "prepared.json"), entity_columns=group_columns,
                                whole_entities_preserved=True, distinct_training_memberships=100))
    jobs_path = ROOT / "manifests/generated/gp-reviewed4-jobs.json"
    assert sha(jobs_path) == report["jobs_sha256"]
    receipt = dict(schema="parallel.reviewed_cut_reproduction.v1", passed=True, jobs=300,
                   jobs_sha256=sha(jobs_path), registry_sha256=sha(registry), review_sha256=sha(review),
                   executable_sha256=sha(executable), source_files_sha256_verified=verified_files,
                   datasets=results, original_final_test_targets_parsed=False,
                   original_final_test_files_only_opaque_hashed=True, copied_cut_csv_files=0,
                   temporary_reproduction_pointers_removed=True, enqueued=False)
    output = ROOT / "manifests/generated/reviewed4-reproducibility.json"
    with output.open("x") as target:
        json.dump(receipt, target, indent=2)
        target.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
