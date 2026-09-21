#!/usr/bin/env python3
"""Package reviewed validation originals and opaque test seals; never parse tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests/generated"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


def main():
    review_path = OUT / "provenance-review/review-v3.json"
    candidates_path = OUT / "provenance-review/pilot-candidates-v3.json"
    audit_path = OUT / "audit/audit.json"
    review = json.loads(review_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert candidates["review_sha256"] == sha(review_path)
    selected = {row["id"] for row in candidates["datasets"]}
    assert len(selected) == 3
    files, backing, withheld, datasets = {}, {}, [], []

    def entry(source, destination, dataset, role):
        assert source.is_file() and not source.is_symlink(), source
        value = dict(
            source=str(source),
            destination=str(destination),
            sha256=sha(source),
            bytes=source.stat().st_size,
            dataset=dataset,
            role=role,
        )
        previous = files.setdefault(str(destination), value)
        assert previous["sha256"] == value["sha256"], destination
        return value

    for row in review["datasets"]:
        if row["id"] not in selected:
            continue
        assert row["observational_origin"] == "verified_documented_measurements"
        assert row["pilot_candidate_status"] == "eligible_for_fresh_cohort_preparation"
        metadata = Path(row["metadata"]["path"])
        assert sha(metadata) == row["metadata"]["sha256"]
        assert sha(Path(row["documentation"]["path"])) == row["documentation"]["sha256"]
        assert sha(metadata.parent / "train.csv") == row["reconstruction"]["training_sha256"]
        destination = Path("data/originals/review-v3/regression") / row["id"]
        for source in sorted(metadata.parent.rglob("*")):
            if not source.is_file():
                continue
            assert not any(part in (".git", "__pycache__", ".venv", "target") for part in source.parts)
            if source.suffix == ".csvx":
                withheld.append(
                    dict(
                        source=str(source),
                        sha256=sha(source),
                        dataset=row["id"],
                        reason="Alternate table has no reviewed reconstruction recipe.",
                    )
                )
                continue
            assert source.suffix in (
                ".csv",
                ".json",
                ".jsonl",
                ".vs",
                ".txt",
                ".log",
                ".md",
                ".png",
                ".svg",
                ".pdf",
            ), source
            relative = source.relative_to(metadata.parent)
            role = (
                "real_dataset_original"
                if relative.name in ("meta.json", "train.csv", "test.csv")
                else "real_dataset_legacy_sidecar"
            )
            entry(source, destination / relative, row["id"], role)
        source = Path(row["reconstruction"]["source"])
        assert sha(source) == row["reconstruction"]["source_sha256"]
        b = entry(
            source,
            Path("data/originals/backing") / source.relative_to("/home/ubuntu"),
            row["id"],
            "real_backing_source",
        )
        backing[b["destination"]] = b
        matching = [
            d
            for d in audit["datasets"]
            if d["id"] == row["id"]
            and d["source_key"] == row["source_key"]
            and d["metadata_sha256"] == row["metadata"]["sha256"]
            and Path(d["source_dir"]) == metadata.parent
        ]
        assert len(matching) == 1 and matching[0]["split_feature_overlap"] == 0
        dataset = matching[0]
        # Only inherited dimensions/quality counts are needed. Avoid distributing
        # historical target values embedded in the broad read-only audit.
        if dataset.get("test"):
            dataset["test"]["target_values"] = []
        datasets.append(dataset)

    transfer_path = OUT / "review-v3-transfer.json"
    backing_path = OUT / "backing-review-v3-transfer.json"
    write_new(transfer_path, dict(schema="parallel.transfer.v1", files=list(files.values())))
    write_new(backing_path, dict(schema="parallel.transfer.v1", files=list(backing.values())))
    write_new(OUT / "review-v3-withheld-artifacts.json", withheld)
    scoped = OUT / "review-v3-audit"
    scoped_audit = dict(
        schema=audit["schema"],
        seed=audit["seed"],
        roots=[d["source_dir"] for d in datasets],
        datasets=datasets,
        artifact_counts={"unresolved": 3 * len(datasets)},
        missing_roots=[],
        complete=False,
    )
    originals = [f for f in files.values() if f["role"] == "real_dataset_original"]
    assert len(originals) == 3 * len(selected)
    write_new(scoped / "audit.json", scoped_audit)
    write_new(scoped / "transfer.json", dict(schema="parallel.transfer.v1", files=originals))
    receipt = dict(
        schema="parallel.scoped_audit_receipt.v1",
        source_audit_sha256=sha(audit_path),
        candidate_manifest_sha256=sha(candidates_path),
        review_sha256=sha(review_path),
        opaque_seal_transfer_sha256=sha(transfer_path),
        audit_sha256=sha(scoped / "audit.json"),
        transfer_sha256=sha(scoped / "transfer.json"),
        datasets=len(datasets),
        original_final_test_targets_parsed=False,
        test_files_read_only_for_opaque_byte_hash=True,
        historical_test_target_values_redacted=True,
        admission_requires_frozen_registry=True,
        copied_cut_csv_files=0,
        cohort_activated=False,
        legacy_sidecars="Archived provenance only; historical scores are not clean teaching labels.",
    )
    write_new(scoped / "scope.json", receipt)
    print(
        json.dumps(
            dict(
                datasets=len(datasets),
                files=len(files),
                bytes=sum(f["bytes"] for f in files.values()),
                withheld_artifacts=len(withheld),
                copied_cut_csv_files=0,
                original_final_test_targets_parsed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
