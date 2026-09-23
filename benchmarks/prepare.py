#!/usr/bin/env python3
"""Prepare pinned ASM inputs outside the checkout. No fitting or outcome selection.

Requires PyYAML and scikit-learn 1.7.2. --freeze is a maintainer operation that
creates a new manifest once; ordinary preparation verifies the frozen identities.
The learning/evaluation worker receives development partitions only.
"""

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import sklearn
import yaml
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

PROTOCOL = "asm-grouped-development-v1/sklearn-1.7.2"
REPO = Path(__file__).resolve().parents[1]
MISSING = {"", "?", "nan", "NaN", "NA"}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keep(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refuse changed immutable input: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("input exceeds preparation download limit")
    return data


def components(rows, feature_indices, group_indices):
    """Union declared groups and identical feature vectors, irrespective of label."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    duplicate, declared = {}, {}
    for i, row in enumerate(rows):
        key = tuple(row[j] for j in feature_indices)
        if key in duplicate:
            union(i, duplicate[key])
        duplicate[key] = i
        if group_indices:
            key = tuple(row[j] for j in group_indices)
            if key in declared:
                union(i, declared[key])
            declared[key] = i
    return [root(i) for i in range(len(rows))]


def assignments(labels, groups, task, seed):
    """Seal one fifth first, then partition the remainder into five dev folds."""
    if sklearn.__version__ != "1.7.2":
        raise ValueError("split identity requires scikit-learn 1.7.2")
    classification = task != "regression"

    def splitter(offset):
        cls = StratifiedGroupKFold if classification else GroupKFold
        return cls(n_splits=5, shuffle=True, random_state=seed + offset)

    x = [[0]] * len(labels)
    dev, sealed = next(splitter(0).split(x, labels, groups))
    result = [None] * len(labels)
    dev_labels = [labels[i] for i in dev]
    dev_groups = [groups[i] for i in dev]
    for fold, (fit, hold) in enumerate(splitter(1).split([[0]] * len(dev), dev_labels, dev_groups)):
        if classification:
            expected = set(labels)
            for part in (fit, hold):
                if {dev_labels[i] for i in part} != expected:
                    raise ValueError("fixed split lacks a class; retain as preparation failure")
        for i in hold:
            result[int(dev[i])] = fold
    if classification and {labels[i] for i in sealed} != set(labels):
        raise ValueError("sealed partition lacks a class")
    if {groups[i] for i in dev} & {groups[i] for i in sealed}:
        raise ValueError("group crossed sealed boundary")
    membership = {}
    for group, fold in zip(groups, result):
        if group in membership and membership[group] != fold:
            raise ValueError("group crossed folds")
        membership[group] = fold
    return result


def prepare_one(spec, config, output, frozen=False):
    name, task = spec["id"], spec["task"]
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name
    ):
        raise ValueError("unsafe dataset identifier")
    base = f"https://raw.githubusercontent.com/{config['repository']}/{config['revision']}/datasets/{name}"
    metadata_url = base + "/metadata.yaml"
    if frozen:
        sha = spec["source"]["sha256"]
        blob_path = output / "downloads" / sha
        data = blob_path.read_bytes() if blob_path.exists() else fetch(spec["source"]["url"])
        metadata_path = output / "metadata" / (name + ".yaml")
        meta_bytes = metadata_path.read_bytes() if metadata_path.exists() else fetch(metadata_url)
        if digest(data) != sha or digest(meta_bytes) != spec["source"]["metadata_sha256"]:
            raise ValueError(f"{name}: source digest mismatch")
        url = spec["source"]["url"]
    else:
        pointer = fetch(base + f"/{name}.tsv.gz")
        url = f"https://media.githubusercontent.com/media/{config['repository']}/{config['revision']}/datasets/{name}/{name}.tsv.gz"
        if pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            fields = dict(line.split(" ", 1) for line in pointer.decode().strip().splitlines())
            data = fetch(url)
            if digest(data) != fields["oid"].removeprefix("sha256:") or len(data) != int(fields["size"]):
                raise ValueError(f"{name}: Git LFS identity mismatch")
        else:
            data, url = pointer, base + f"/{name}.tsv.gz"
        sha = digest(data)
        meta_bytes = fetch(metadata_url)
    keep(output / "downloads" / sha, data)
    keep(output / "metadata" / (name + ".yaml"), meta_bytes)
    metadata = yaml.safe_load(meta_bytes)
    if metadata["task"] != ("regression" if task == "regression" else "classification"):
        raise ValueError(f"{name}: original task mismatch")
    raw = gzip.decompress(data)
    parsed = list(csv.reader(io.StringIO(raw.decode()), delimiter="\t"))
    header, rows = parsed[0], parsed[1:]
    target = header.index("target")
    feature_indices = [i for i in range(len(header)) if i != target]
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError(f"{name}: empty or ragged input")
    labels = [row[target] for row in rows]
    if any(y in MISSING for y in labels):
        raise ValueError(f"{name}: missing target; no silent row removal")
    classes = sorted(set(labels)) if task != "regression" else []
    if (task == "binary" and len(classes) != 2) or (task == "multiclass" and len(classes) < 3):
        raise ValueError(f"{name}: original class count mismatch")
    kinds = {str(x["name"]): x["type"] for x in metadata["features"]}
    if any(kind not in {"continuous", "categorical", "binary", "ordinal"} for kind in kinds.values()):
        raise ValueError(f"{name}: unrecognized feature kind")
    features = [
        {"name": header[i], "kind": "numeric" if kinds[header[i]] == "continuous" else "categorical"}
        for i in feature_indices
    ]
    group_columns = spec["group_columns"]
    groups = components(rows, feature_indices, [header.index(g) for g in group_columns])
    folds = assignments(labels, groups, task, config["seed"])
    records = []
    for i, row in enumerate(rows):
        records.append(
            {
                "id": digest(encoded([sha, i, row])),
                "source_index": i,
                "group": digest(encoded([sha, groups[i]])),
                "fold": folds[i],
                "values": [None if row[j] in MISSING else row[j] for j in feature_indices],
                "target": labels[i],
            }
        )
    common = {
        "schema": "jope.prepared-table/v1",
        "dataset": name,
        "task": task,
        "source_sha256": sha,
        "features": features,
        "classes": classes,
    }
    development = encoded(
        dict(common, partition="development", rows=[r for r in records if r["fold"] is not None])
    )
    sealed = encoded(dict(common, partition="sealed_final", rows=[r for r in records if r["fold"] is None]))
    split = encoded(
        {
            "protocol": PROTOCOL,
            "seed": config["seed"],
            "source_sha256": sha,
            "rows": [{k: row[k] for k in ("id", "source_index", "group", "fold")} for row in records],
        }
    )
    # Both outputs have separate identities; workers receive development only.
    for folder, payload in [("development", development), ("sealed", sealed), ("splits", split)]:
        keep(output / folder / (name + ".json"), payload)
    result = {
        "id": name,
        "task": task,
        "metric": {"regression": "rmse", "binary": "auroc", "multiclass": "log_loss"}[task],
        "source": {
            "url": url,
            "sha256": sha,
            "bytes": len(data),
            "metadata_url": metadata_url,
            "metadata_sha256": digest(meta_bytes),
        },
        "rows": len(rows),
        "features": features,
        "classes": classes,
        "group_columns": group_columns,
        "split": {
            "protocol": PROTOCOL,
            "sha256": digest(split),
            "development_sha256": digest(development),
            "sealed_sha256": digest(sealed),
            "development_rows": sum(f is not None for f in folds),
            "sealed_rows": sum(f is None for f in folds),
            "fold_rows": [folds.count(i) for i in range(5)],
        },
        "license": {
            "original": str(metadata.get("license") or "unknown"),
            "evidence_url": metadata_url,
            "repository": "MIT",
            "note": "Repository license does not establish an unrecorded original dataset license; no dataset redistribution in inference bundles.",
        },
        "checkpoint_exposure": "unknown; no unseen-family or uncontaminated-checkpoint claim",
        "description": str(metadata.get("description") or "unknown"),
        "group_limitations": "Identical feature rows and declared columns stay together; missing subject/time identifiers cannot establish independent observations.",
    }
    if frozen and result != spec:
        raise ValueError(f"{name}: prepared identity differs from frozen manifest")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, help="create manifest from selection once; refuses overwrite")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("bulk data must be outside the working tree")
    config = json.loads((args.freeze or args.manifest).read_text())
    if len(config["datasets"]) != 30 or any(
        sum(d["task"] == task for d in config["datasets"]) != 10
        for task in ("regression", "binary", "multiclass")
    ):
        raise ValueError("panel requires ten datasets per task")
    if len({d["id"] for d in config["datasets"]}) != 30:
        raise ValueError("duplicate panel dataset identifier")
    if args.freeze and args.manifest.exists():
        raise ValueError("panel already frozen; never overwrite a measured panel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(
            pool.map(lambda spec: prepare_one(spec, config, output, not args.freeze), config["datasets"])
        )
    if args.freeze:
        manifest = {k: config[k] for k in ("panel_id", "repository", "revision", "seed", "selection")}
        manifest.update(schema="jope.benchmark-panel/v1", folds=5, datasets=entries)
        keep(args.manifest, encoded(manifest))
    print(
        json.dumps(
            {
                "schema": "jope.panel-preparation/v1",
                "status": "verified",
                "datasets": len(entries),
                "manifest_sha256": digest(args.manifest.read_bytes()),
                "output": str(output),
                "split_protocol": PROTOCOL,
                "sklearn": sklearn.__version__,
                "fits": 0,
                "campaign_started": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
