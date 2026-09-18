#!/usr/bin/env python3
"""Read-only exact inventory, membership and provenance verification for binary contexts."""
import argparse
import collections
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

from experiment_runtime import checked_path, inside, require, write

COMPONENTS = {"target": ("target.f32", 512), "foundation": ("foundation_raw.f32", 2048), "hyperion": ("hyperion.f32", 512)}


def sha(path):
    digest = hashlib.sha256()
    with checked_path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(checked_path(path).read_text())


def compact_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def vector(path, dimensions):
    data = checked_path(path).read_bytes()
    require(type(dimensions) is int and dimensions > 0 and len(data) == dimensions * 4, f"wrong vector dimension: {path}")
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", data)), f"nonfinite vector: {path}")


def keyed(rows, label):
    result = {}
    for row in rows:
        dataset, cut = row["dataset"], row["cut"]
        require(isinstance(dataset, str) and Path(dataset).name == dataset and dataset not in (".", ".."), f"invalid dataset path: {label}")
        require(type(cut) is int and cut >= 0, f"invalid cut: {label}")
        key = dataset, cut
        require(key not in result, f"duplicate {label} key: {key}")
        result[key] = row
    require(bool(result), f"empty {label} inventory")
    return result


def memberships(root, prepared):
    storage = prepared["storage"]
    source = inside(root, storage["source"])
    require(sha(source) == storage["source_sha256"], f"original source hash changed: {source}")
    mapping = storage["original_training_row_map"]
    require(all(type(i) is int and i >= 0 for i in mapping), "invalid physical row mapping")
    selected = prepared["selected_train_indices"]
    require(len(selected) == len(set(selected)) and all(type(i) is int and 0 <= i < len(mapping) for i in selected), "invalid training membership")
    wanted = {mapping[i] for i in selected}
    raw = {}
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = storage["feature_columns"] + [storage["target_column"]]
        require(len(columns) == len(set(columns)) and set(columns) <= set(reader.fieldnames), "invalid source columns")
        for index, row in enumerate(reader):
            if index in wanted:
                raw[index] = {k: row[k].strip() for k in columns}
    require(set(raw) == wanted, "missing reconstructed source rows")
    cuts = prepared["cuts"]
    require(len(cuts) == len({c["index"] for c in cuts}) and 0 < len(cuts) <= 20, "invalid cut inventory")
    seen_memberships, folds = set(), collections.defaultdict(list)
    for cut in cuts:
        train, heldout = cut["train_indices"], cut["validation_indices"]
        require(train and heldout and len(train) == len(set(train)) and len(heldout) == len(set(heldout)), "duplicate or empty membership")
        require(set(train).isdisjoint(heldout) and set(train) | set(heldout) == set(selected), "train/validation partition mismatch")
        member = tuple(sorted(train))
        require(member not in seen_memberships, "repeated training membership")
        seen_memberships.add(member)
        folds[cut["index"] // 5].extend(heldout)
        targets = []
        for indices in (train, heldout):
            rows = [raw[mapping[i]] for i in indices]
            labels = {r[storage["target_column"]] for r in rows}
            require(len(labels) == 2, "cut lacks both classes")
            targets.append(labels)
        require(targets[0] == targets[1], "class labels differ between memberships")
        for columns in (storage["feature_columns"], prepared.get("entity_columns", [])):
            if not columns:
                continue
            left = {tuple(raw[mapping[i]][c] for c in columns) for i in train}
            right = {tuple(raw[mapping[i]][c] for c in columns) for i in heldout}
            require(left.isdisjoint(right), "duplicate feature/entity group crosses memberships")
    for heldout in folds.values():
        require(len(heldout) == len(selected) and set(heldout) == set(selected), "within-repetition fold coverage mismatch")


def verify(root, experiment):
    root, experiment = checked_path(root), checked_path(experiment)
    report_path = experiment / "report.json"
    report_sha, report = sha(report_path), read(report_path)
    aliases = read(experiment / "aliases.json")
    require(sha(experiment / "aliases.json") == report["aliases_sha256"], "binary alias hash changed")
    jobs_path = experiment / "jobs.json"
    jobs = read(jobs_path)["jobs"]
    require(jobs and report.get("admitted_contexts") == report.get("completed_contexts") == len(jobs), "admitted jobs and reported context counts disagree")
    require(sha(jobs_path) == report["jobs_sha256"], "binary jobs hash changed")
    prepared, bound_jobs = {}, []
    for job in jobs:
        path = inside(root, job["prepared"])
        require(sha(path / "prepared.json") == job["prepared_sha256"], "prepared hash changed")
        p = prepared.setdefault(job["prepared"], read(path / "prepared.json"))
        require(p["task"] == "binary" and p["qualification_only"] is True, "not binary qualification preparation")
        require(p["binary_aliases_sha256"] == report["aliases_sha256"], "prepared aliases differ")
        family = [f for f in aliases["families"] if f["dataset"] == p["dataset"]]
        require(len(family) == 1 and family[0]["family"] == p["family"] and family[0]["partition"] == p["meta_partition"], "family partition mismatch")
        bound_jobs.append(dict(job, dataset=p["dataset"]))
    expected = keyed(bound_jobs, "admitted job")
    require(len(prepared) == report["admitted_datasets"], "admitted dataset count mismatch")
    for p in prepared.values():
        memberships(root, p)
        require({c["index"] for c in p["cuts"]} == {cut for dataset, cut in expected if dataset == p["dataset"]}, "prepared cuts differ from admitted jobs")
    run = read(experiment / "encoding-run.json")
    require(run["jobs_sha256"] == report["jobs_sha256"] and run["aliases_sha256"] == report["aliases_sha256"], "encoder run identity mismatch")
    specs = {name: experiment / "sampling" / f"{name}-spec.json" for name in ("target", "foundation")}
    specs["hyperion"] = root / "manifests/hyperion-reviewed4-safe-spec.json"
    require(sha(specs["hyperion"]) == run["hyperion_spec_sha256"], "Hyperion specification changed")
    for name, path in specs.items():
        spec = read(path)
        fields = ("module", "sampling", "parity") if name != "hyperion" else ("binary", "weights", "config", "source_manifest", "loader", "runtime_manifest", "persistent_binary")
        for field in fields:
            require(sha(inside(root, spec[field])) == spec[field + "_sha256"], f"frozen {name} {field} changed")
        if name == "hyperion":
            files = read(inside(root, spec["runtime_manifest"]))["files"]
            require(bool(files) and len({r["destination"] for r in files}) == len(files), "invalid Hyperion runtime inventory")
            for row in files:
                path = inside(root, row["destination"])
                require(sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], f"frozen Hyperion runtime changed: {path}")
    components = {}
    for name, (filename, dimensions) in COMPONENTS.items():
        directory = experiment / "components" / name
        expected_vectors, expected_receipts = set(), set()
        for key, job in expected.items():
            p = prepared[job["prepared"]]
            cut = next(c for c in p["cuts"] if c["index"] == job["cut"])
            recipe = inside(root, job["prepared"]) / f"cut-{key[1]}/preprocessing.json"
            require(sha(recipe) == cut["preprocessing_sha256"] and len(set(read(recipe)["labels"])) == 2, "preprocessing identity or labels changed")
            leaf = directory / key[0] / f"cut-{key[1]}"
            receipt, payload = leaf / f"{name}.json", leaf / filename
            metadata = read(receipt)
            identity = dict(dataset=p["dataset"], family=p["family"], cut=key[1], prepared_sha256=job["prepared_sha256"],
                            training_sha256=cut["training_sha256"], preprocessing_sha256=cut["preprocessing_sha256"],
                            final_test_targets_used=False, shape=[dimensions], spec_sha256=sha(specs[name]))
            require(all(metadata.get(k) == v for k, v in identity.items()), f"{name} component identity mismatch: {key}")
            vector(payload, dimensions)
            require(sha(payload) == metadata["signature_sha256"], "component bytes changed")
            components[key, name] = dict(receipt=str(receipt.relative_to(root)), receipt_sha256=sha(receipt),
                signature=str(payload.relative_to(root)), signature_sha256=sha(payload), spec_sha256=sha(specs[name]))
            expected_vectors.add(payload)
            expected_receipts.add(receipt)
        require(set(directory.rglob("*.f32")) == expected_vectors, f"{name} vector inventory contains missing or unexpected artifacts")
        require(set(directory.rglob(f"{name}.json")) == expected_receipts, f"{name} receipt inventory mismatch")
    index_path = inside(root, report["context_index"])
    require(sha(index_path) == report["context_index_sha256"], "context index changed")
    index = read(index_path)
    require(index["task"] == "binary" and index["qualification"] is True and index["dimensions"] == 2048, "invalid context index")
    actual = keyed(index["encodings"], "context")
    require(actual.keys() == expected.keys(), "admitted jobs and context inventory disagree")
    application_path = index_path.parent / "application.json"
    require(sha(application_path) == index["application_sha256"], "context application changed")
    application = read(application_path)
    require(application["jobs_sha256"] == report["jobs_sha256"] and application["projection_refitted"] is False, "context application provenance mismatch")
    assembly = inside(root, index["assembly"])
    require(sha(assembly) == index["assembly_sha256"] == run["projection_sha256"], "frozen projection changed")
    require(sha(inside(root, application["transforms"])) == application["transforms_sha256"] == read(assembly)["transforms_sha256"], "frozen projection transform changed")
    require(sha(index_path.parent / "reference-parity.json") == index["reference_parity_sha256"], "projection reference parity changed")
    require(read(index_path.parent / "reference-parity.json")["passed"] is True, "projection reference parity failed")
    expected_vectors = set()
    for key, row in actual.items():
        job, p = expected[key], prepared[expected[key]["prepared"]]
        cut = next(c for c in p["cuts"] if c["index"] == key[1])
        payload = inside(root, row["path"])
        require(payload.is_relative_to(index_path.parent), "context vector outside index directory")
        vector(payload, 2048)
        expected_vectors.add(payload)
        require(sha(payload) == row["sha256"] and sha(payload.with_suffix(".json")) == row["receipt_sha256"], "context bytes or receipt changed")
        metadata = read(payload.with_suffix(".json"))
        binding = {name: components[key, name] for name in COMPONENTS}
        require(metadata["components"] == binding and metadata["components_sha256"] == row["components_sha256"] == compact_sha(binding), "component-to-context provenance mismatch")
        identity = dict(task="binary", qualification=True, binary_aliases_sha256=report["aliases_sha256"], dataset=key[0], cut=key[1],
            family=p["family"], partition=p["meta_partition"], prepared_sha256=job["prepared_sha256"],
            training_sha256=cut["training_sha256"], preprocessing_sha256=cut["preprocessing_sha256"],
            encoder_spec_sha256=index["assembly_sha256"], application_sha256=index["application_sha256"],
            selected_training_rows_sha256=compact_sha(cut["train_indices"]), dataset_dimensions=1536, target_dimensions=512, output_sha256=row["sha256"])
        require(all(metadata.get(k) == v for k, v in identity.items()), "context identity mismatch")
        require(all(row[k] == job[k] for k in ("prepared", "prepared_sha256")) and row["family"] == p["family"] and row["partition"] == p["meta_partition"], "context index identity mismatch")
    require(set(index_path.parent.rglob("*.f32")) == expected_vectors, "unexpected context vector")
    initial = keyed(read(experiment / "contexts-initial/index.json")["encodings"], "initial context")
    initial_jobs = read(experiment / "jobs-initial.json")["jobs"]
    initial_expected = {(prepared[j["prepared"]]["dataset"], j["cut"]) for j in initial_jobs}
    require(len(initial_expected) == len(initial_jobs) == report["initial_contexts"] and initial.keys() == initial_expected and initial_expected <= expected.keys(), "initial overlap inventory mismatch")
    require(all(actual[key]["sha256"] == row["sha256"] == sha(inside(root, row["path"])) for key, row in initial.items()), "initial overlap changed")
    cut_csvs = sum(len(list((experiment / name).rglob("*.csv"))) for name in ("bindings", "prepared", "prepared-initial", "verification-rebuilt"))
    require(cut_csvs == 0 and sha(report_path) == report_sha, "copied cut tables or report mutation")
    return dict(schema="parallel.binary_artifact_verification.v2", passed=True, report_sha256=report_sha,
        prepared_datasets=len(prepared), original_source_hashes=len({p["storage"]["source_sha256"] for p in prepared.values()}),
        component_hashes=len(components), context_hashes=len(actual), initial_overlap_contexts=len(initial),
        initial_overlap_contexts_byte_identical=True, aliases_verified=True, copied_cut_tables=0, report_unchanged=True,
        unresolved_entity_groups=[p["family"] for p in prepared.values() if p["family"] == "sonar" and not p.get("entity_columns")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.experiment)
    if args.receipt:
        write(inside(args.experiment, args.receipt), result)
    print(json.dumps(result))
