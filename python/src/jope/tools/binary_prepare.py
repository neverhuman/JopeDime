#!/usr/bin/env python3
"""Freeze and reconstruct the explicitly authorized first binary qualification cohort.

Only original backing tables are staged once. Cuts contain references, memberships,
recipes and hashes; original final-test payloads are never opened.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import experiment_runtime as runtime
from experiment_scheduler import spare_lease

COHORT = [
    (
        "Adult",
        "3b5b76f8c7ecdaa5",
        "adult",
        {"adult.csv", "adult-census.csv", "adult-test.csv", "openml:1590"},
    ),
    (
        "German credit",
        "83463302a33ba419",
        "credit-g",
        {
            "german-credit-data-creditability.csv",
            "german-credit-data-creditability-2.csv",
            "german-credit-data-creditability-preprocessed-dataset.csv",
            "germancreditdata.csv",
            "german-credit-risk-with-target.csv",
            "creditability-german-credit-data.csv",
            "credit-approval-germany.csv",
            "deprecated-german.csv",
            "openml:31",
        },
    ),
    ("Ionosphere", "17f1fa09f9aa23b1", "ionosphere", {"ionosphere.csv", "openml:59"}),
    ("Sonar", "b87fa58363acbb49", "sonar", {"sonar.csv", "openml:40"}),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not (json.loads(path.read_text()) == value):
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        runtime.write(path, value)


def prepared_directory(root, destination, command):
    def validate(path):
        prepared = json.loads((path / "prepared.json").read_text())
        runtime.require(bool(prepared["cuts"]), "preparation has no cuts")
        for cut in prepared["cuts"]:
            runtime.require(
                sha(path / f"cut-{cut['index']}/preprocessing.json") == cut["preprocessing_sha256"],
                "prepared recipe missing or changed",
            )
        from binary_verify import memberships

        memberships(root, prepared)

    if destination.exists():
        validate(destination)
        return
    staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = command + ["--output", str(staging)]
    while True:
        try:
            with spare_lease(root, "binary-preparation", cpus=2):
                result = runtime.run(
                    command,
                    root=root,
                    timeout=600,
                    env=runtime.child_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                result.check_returncode()
            break
        except BlockingIOError:
            # Preserve interrupted preparation and restart into a fresh staging tree.
            staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
            command[-1] = str(staging)
            time.sleep(1)
    validate(staging)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    staging.rename(destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stable_partition(family):
    value = hashlib.sha256(f"20260907:family:{family}".encode()).hexdigest()
    bucket = int(value[:16], 16) % 1000000
    return "meta_train" if bucket < 800000 else "meta_validation" if bucket < 900000 else "meta_test"


def freeze_aliases(audit, registry):
    families = []
    for name, dataset, family, keys in COHORT:
        aliases = [d for d in audit["datasets"] if d["source_key"] in keys or d.get("family") == family]
        registered = [
            f
            for f in registry["families"].values()
            if f["family"] == family or keys.intersection(f["source_keys"])
        ]
        assignments = {(f["family"], f["partition"]) for f in registered}
        assignments.update(
            (d["family"], d["meta_partition"]) for d in aliases if d.get("family") and d.get("meta_partition")
        )
        if not (len(assignments) <= 1):
            raise ValueError(f"Conflicting frozen aliases: {family} {assignments}")
        if assignments:
            frozen_family, partition = next(iter(assignments))
            if not (frozen_family == family):
                raise ValueError(
                    f"Existing family naming requires explicit reconciliation: {family} {frozen_family}"
                )
        else:
            partition = stable_partition(family)
        families.append(
            dict(
                name=name,
                dataset=dataset,
                family=family,
                partition=partition,
                assignment="preserved_existing" if assignments else "stable_sha256_bucket_v1",
                source_keys=sorted(keys),
                cross_task_aliases=sorted({(d["id"], d["task"], d["source_key"]) for d in aliases}),
                limitations=[
                    "Sonar acquisition/object groups are unresolved; duplicate feature groups are enforced, qualification only"
                ]
                if family == "sonar"
                else [],
            )
        )
    return dict(
        schema="parallel.binary_family_aliases.v1",
        qualification=True,
        frozen_before_partitioning=True,
        existing_assignments_preserved=True,
        families=families,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not (output.is_relative_to(root)):
        raise ValueError("binary_prepare: artifact validation failed at original line 77")
    audit = json.loads(args.audit.read_text())
    registry = json.loads(args.registry.read_text())
    acquisition = json.loads(args.acquisition.read_text())
    transfer = json.loads((args.audit.parent / "transfer.json").read_text())
    aliases = freeze_aliases(audit, registry)
    aliases.update(
        inventory_sha256=sha(args.audit),
        registry_sha256=sha(args.registry),
        acquisition_sha256=sha(args.acquisition),
    )
    save(output / "aliases.json", aliases)
    status, staged, jobs, initial = [], [], [], []
    for family in aliases["families"]:
        dataset = family["dataset"]
        records = [
            d
            for d in audit["datasets"]
            if d["id"] == dataset
            and Path(d["source_dir"]).parent.name == "binary"
            and Path(d["source_dir"]).parent.parent.name != "binary"
        ]
        if not (len(records) == 1):
            raise ValueError(dataset)
        original = records[0]
        directory = Path(original["source_dir"])
        metadata = json.loads((directory / "meta.json").read_text())
        if not (sha(directory / "meta.json") == original["metadata_sha256"]):
            raise ValueError("binary_prepare: artifact validation failed at original line 93")
        source = Path(metadata["source_path"])
        if not source.exists():
            source = Path("/home/ubuntu/remote_super/hyper/data") / source.relative_to(
                "/home/ubuntu/hyper/data"
            )
        matches = [
            row
            for row in acquisition["datasets"]
            if row["dataset_hash"] == dataset and row.get("status") == "prepared"
        ]
        if not (len(matches) == 1 and matches[0]["source_path"] == metadata["source_path"]):
            raise ValueError("binary_prepare: artifact validation failed at original line 98")
        if not (source.stat().st_size == metadata["source_size_bytes"]):
            raise ValueError("backing source acquisition size changed")
        source_sha = sha(source)
        destination = (output / "originals" / source.name).relative_to(root)
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not (sha(target) == source_sha):
                raise ValueError("binary_prepare: artifact validation failed at original line 105")
        else:
            shutil.copyfile(source, target)
            if not (sha(target) == source_sha):
                raise ValueError("binary_prepare: artifact validation failed at original line 108")
        staged.append(
            dict(
                source=str(source),
                destination=str(destination),
                bytes=source.stat().st_size,
                sha256=source_sha,
                dataset=dataset,
                role="approved_binary_original_once",
            )
        )
        local = output / "preparation-inputs" / dataset
        record = copy.deepcopy(original)
        record.update(
            admission="approved",
            reason="Explicitly authorized first-cohort binary qualification; exact reconstruction is mandatory",
            family=family["family"],
            meta_partition=family["partition"],
        )
        save(
            local / "audit.json",
            dict(
                schema=audit["schema"],
                seed=audit["seed"],
                roots=audit["roots"],
                datasets=[record],
                artifact_counts={},
                missing_roots=[],
                complete=False,
            ),
        )
        save(
            local / "transfer.json",
            dict(
                schema=transfer["schema"],
                files=[f for f in transfer["files"] if f["source"] == str(directory / "test.csv")],
            ),
        )
        save(
            local / "admission.json",
            dict(
                schema="parallel.binary_qualification_admission.v1",
                original_audit_admission=original["admission"],
                original_audit_reason=original["reason"],
                user_authorization="First cohort Adult/German credit/Ionosphere/Sonar, subject to exact reconstruction",
                acquisition_record=matches[0],
                aliases_sha256=sha(output / "aliases.json"),
                historical_metadata_sha256=original["metadata_sha256"],
                original_training_sha256=sha(directory / "train.csv"),
                backing_source_sha256=source_sha,
                qualification_only=True,
                final_test_targets_opened=False,
            ),
        )
    save(output / "originals-transfer.json", dict(schema="parallel.transfer.v1", files=staged))
    for family in aliases["families"]:
        dataset = family["dataset"]
        base = output / "prepared-initial" / dataset
        expanded = output / "prepared" / dataset
        try:
            prepared_directory(
                root,
                base,
                [
                    str(args.binary),
                    "--root",
                    str(root),
                    "prepare",
                    "--audit",
                    str(output / "preparation-inputs" / dataset / "audit.json"),
                    "--dataset",
                    dataset,
                    "--qualification",
                    "--backing-manifest",
                    str(output / "originals-transfer.json"),
                ],
            )
            prepared_directory(
                root,
                expanded,
                [
                    str(args.binary),
                    "--root",
                    str(root),
                    "expand-cuts",
                    "--prepared",
                    str(base),
                    "--rounds",
                    "4",
                ],
            )
            p = json.loads((expanded / "prepared.json").read_text())
            # Apply qualification and unresolved acquisition limitations before any encoder consumes the manifest.
            p["qualification_only"] = True
            p["binary_aliases_sha256"] = sha(output / "aliases.json")
            p["evaluation_limitations"] = sorted(
                set(
                    p.get("evaluation_limitations", [])
                    + family["limitations"]
                    + ["Frozen regression projection reused only for binary qualification"]
                )
            )
            p["clean_evaluation"] = False
            p["clean_status"] = "exact_original_training_reconstructed_binary_qualification"
            binding = output / "bindings" / dataset
            binding.mkdir(parents=True, exist_ok=True)
            save(binding / "prepared.json", p)
            for cut in p["cuts"]:
                save(
                    binding / f"cut-{cut['index']}" / "preprocessing.json",
                    json.loads((expanded / f"cut-{cut['index']}" / "preprocessing.json").read_text()),
                )
                # Rust pretty JSON has no trailing newline; retain exact recipe bytes required by the manifest.
                dest = binding / f"cut-{cut['index']}" / "preprocessing.json"
                original = expanded / f"cut-{cut['index']}" / "preprocessing.json"
                if sha(dest) != cut["preprocessing_sha256"]:
                    runtime.atomic_bytes(dest, original.read_bytes())
                job = dict(
                    prepared=str(binding.relative_to(root)),
                    prepared_sha256=sha(binding / "prepared.json"),
                    cut=cut["index"],
                    output=str(
                        (output / "unused-gp-output" / dataset / f"cut-{cut['index']}").relative_to(root)
                    ),
                )
                jobs.append(job)
                if cut["index"] < 5:
                    initial.append(job)
            membership = [
                hashlib.sha256(json.dumps(c["train_indices"], separators=(",", ":")).encode()).hexdigest()
                for c in p["cuts"]
            ]
            if not (len(membership) == len(set(membership)) and len(membership) <= 20):
                raise ValueError("binary_prepare: artifact validation failed at original line 149")
            if not (not list(binding.rglob("*.csv"))):
                raise ValueError("binary_prepare: artifact validation failed at original line 150")
            status.append(
                dict(
                    dataset=dataset,
                    name=family["name"],
                    family=family["family"],
                    partition=family["partition"],
                    status="admitted",
                    cuts=len(p["cuts"]),
                    memberships=membership,
                    prepared=job["prepared"],
                    prepared_sha256=job["prepared_sha256"],
                    limitations=family["limitations"],
                )
            )
        except subprocess.CalledProcessError as error:
            reason = error.stderr.strip()
            status.append(
                dict(
                    dataset=dataset,
                    name=family["name"],
                    status="insufficient_class_groups"
                    if "class" in reason or "group" in reason
                    else "reconstruction_withheld",
                    reason=reason,
                )
            )
    # Interleave source families so a bounded encoding window covers breadth first.
    jobs.sort(key=lambda j: (j["cut"], j["prepared"]))
    initial.sort(key=lambda j: (j["cut"], j["prepared"]))
    save(output / "jobs.json", dict(schema="parallel.gp_jobs.v1", qualification=True, jobs=jobs))
    save(output / "jobs-initial.json", dict(schema="parallel.gp_jobs.v1", qualification=True, jobs=initial))
    report = dict(
        schema="parallel.binary_preparation_report.v1",
        qualification=True,
        requested_datasets=4,
        admitted_datasets=sum(s["status"] == "admitted" for s in status),
        initial_contexts=len(initial),
        admitted_contexts=len(jobs),
        completed_contexts=0,
        maximum_contexts=80,
        copied_cut_tables=0,
        source_originals_staged_once=len(staged),
        aliases_sha256=sha(output / "aliases.json"),
        jobs_sha256=sha(output / "jobs.json"),
        original_final_tests_opened=False,
        datasets=status,
    )
    save(output / "preparation-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "binary")
