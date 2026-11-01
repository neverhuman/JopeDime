"""Catalog-bound JRB input admission helpers.

The binary panel builder reads only the admitted panel's train CSVs. It does not
open dataset test files or P3 sealed-roster contents. Its output is explicitly a
partial manifest until the complete role inventory is supplied and reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

SCHEMA = "jope.joint-regbin/v1"
PARTIAL_SCHEMA = "jope.joint-regbin.binary-panel-packet/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _family_tool():
    repo = Path(__file__).resolve().parents[2]
    tool_root = repo / "deploy/dl-trainonly/lane/src/dl_training_v1"
    if str(tool_root) not in sys.path:
        sys.path.insert(0, str(tool_root))
    from dltrain import family_keys

    return family_keys


def _srcfile_key(row: dict) -> str | None:
    candidate = row.get("family_key_derived")
    if isinstance(candidate, str) and candidate.startswith("srcfile:"):
        return candidate.strip()
    source_file = row.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        return None
    stem = Path(source_file).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"srcfile:{stem}" if stem else None


def _url_parent_key(url: str | None) -> str | None:
    """Use a dataset URL as parent evidence without collapsing broad URL dirs."""
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    path = parsed.path or "/"
    return f"urlparent:https://{host}{path}?{parsed.query}"


def review_binary_families(records: list[dict]) -> dict:
    """Promote catalog provisional keys and merge metadata aliases via P2 keys."""
    family_keys = _family_tool()
    binary = [row for row in records if row.get("task") == "binary"]
    by_id = {row["id"]: row for row in binary}
    parent = {dataset_id: dataset_id for dataset_id in by_id}

    def find(dataset_id: str) -> str:
        while parent[dataset_id] != dataset_id:
            parent[dataset_id] = parent[parent[dataset_id]]
            dataset_id = parent[dataset_id]
        return dataset_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[tuple[str, str], str] = {}
    evidence_by_id: dict[str, list[list[str]]] = {}
    review_by_id: dict[str, dict] = {}
    for dataset_id, row in sorted(by_id.items()):
        provenance = {
            key: value
            for key, value in {
                "platform": row.get("platform"),
                "url": row.get("source_url"),
                "dedupe_fingerprint": row.get("family_key"),
            }.items()
            if value is not None
        }
        keys = set(
            family_keys.derive(
                {
                    "canonical_sha256": row.get("canonical_sha256"),
                    "provenance": provenance,
                }
            )[0]
        )
        srcfile = _srcfile_key(row)
        if srcfile:
            keys.add(("srcfile", srcfile))
        url_parent = _url_parent_key(row.get("source_url"))
        if url_parent:
            keys.add(("url_parent", url_parent))
        if not keys:
            raise ValueError(
                f"binary dataset lacks a reviewed family key: {dataset_id}"
            )
        evidence = sorted([kind, value] for kind, value in keys)
        evidence_by_id[dataset_id] = evidence
        for kind, value in evidence:
            key = (kind, value)
            previous = owners.get(key)
            if previous is None:
                owners[key] = dataset_id
            else:
                union(dataset_id, previous)
        review_by_id[dataset_id] = {
            "dataset": dataset_id,
            "catalog_family_key": row.get("family_key"),
            "catalog_family_key_source": row.get("family_key_source"),
            "promoted_srcfile_key": srcfile,
            "source_url": row.get("source_url"),
            "url_parent_key": url_parent,
            "dedupe_tool_schema": family_keys.SCHEMA,
            "dedupe_tool_policy_sha256": family_keys.KEYS_POLICY_SHA256,
            "union_evidence": evidence,
            "review": "metadata keys reviewed; roster membership not inspected",
        }

    components: dict[str, list[str]] = defaultdict(list)
    for dataset_id in sorted(by_id):
        components[find(dataset_id)].append(dataset_id)

    family_by_id: dict[str, str] = {}
    component_rows = []
    for member_ids in sorted(components.values(), key=lambda members: members[0]):
        component_evidence = sorted(
            {
                tuple(item)
                for dataset_id in member_ids
                for item in evidence_by_id[dataset_id]
            }
        )
        srcfile_keys = sorted(
            value for kind, value in component_evidence if kind == "srcfile"
        )
        recorded_keys = sorted(
            value
            for dataset_id in member_ids
            if (value := by_id[dataset_id].get("family_key"))
        )
        canonical_keys = sorted(
            value for kind, value in component_evidence if kind == "canonical_sha256"
        )
        normalized_urls = sorted(
            value for kind, value in component_evidence if kind == "normalized_url"
        )
        if srcfile_keys:
            family = srcfile_keys[0]
        elif recorded_keys:
            family = f"dedupe:{recorded_keys[0]}"
        elif canonical_keys:
            family = f"canonical:{canonical_keys[0]}"
        elif normalized_urls:
            family = f"url:{normalized_urls[0]}"
        else:
            raise ValueError(f"binary family component has no stable key: {member_ids}")
        for dataset_id in member_ids:
            family_by_id[dataset_id] = family
        component_rows.append(
            {
                "family": family,
                "datasets": member_ids,
                "union_evidence": [list(item) for item in component_evidence],
            }
        )
    return {
        "schema": "jope.joint-regbin.binary-family-review/v1",
        "method": "catalog srcfile stem plus URL parent, merged by P2 family_keys union evidence",
        "dedupe_tool_schema": family_keys.SCHEMA,
        "dedupe_tool_policy_sha256": family_keys.KEYS_POLICY_SHA256,
        "dataset_count": len(binary),
        "family_count": len(component_rows),
        "unkeyed_count": 0,
        "families": component_rows,
        "records": [
            dict(review_by_id[dataset_id], family=family_by_id[dataset_id])
            for dataset_id in sorted(by_id)
        ],
        "family_by_dataset": family_by_id,
    }


def _load_csv(path: Path, expected_rows: int, expected_cols: int, target_index: int):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != expected_cols:
                raise ValueError(f"unexpected column count in train rows: {path}")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric train row in binary pool: {path}"
                ) from exc
    if len(rows) != expected_rows or not rows:
        raise ValueError(f"unexpected row count in train rows: {path}")
    table = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(table).all():
        raise ValueError(f"non-finite train values in binary pool: {path}")
    target_index %= expected_cols
    y = table[:, target_index]
    if len(np.unique(y)) != 2:
        raise ValueError(f"binary train target does not have two classes: {path}")
    x = np.delete(table, target_index, axis=1)
    return x, y


def build_binary_panel_packet(
    catalog_path: Path,
    panel_path: Path,
    panel_root: Path,
    output: Path,
) -> dict:
    """Create admitted panel-only train pools; never opens local test CSVs."""
    catalog_bytes = catalog_path.read_bytes()
    panel_bytes = panel_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    panel = json.loads(panel_bytes)
    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
    panel_sha = hashlib.sha256(panel_bytes).hexdigest()
    if catalog.get("inventory", {}).get("source_catalog", "").find("LATEST=v246") < 0:
        raise ValueError("the JRB input packet requires the pinned v246 catalog")
    catalog_rows = {row["id"]: row for row in catalog["datasets"]}
    family_review = review_binary_families(catalog["datasets"])
    family_by_id = family_review["family_by_dataset"]
    panel_rows = [row for row in panel["datasets"] if row.get("task") == "binary"]
    if len(panel_rows) != 240 or len({row["id"] for row in panel_rows}) != 240:
        raise ValueError("the admitted panel must contain 240 unique binary datasets")
    output = output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(Path("/tmp")) or output.is_relative_to(repo):
        raise ValueError(
            "binary panel packet must remain outside /tmp and the Git checkout"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing packet: {output}")
    output.mkdir(parents=True)
    pool_dir = output / "binary"
    pool_dir.mkdir()
    contexts = []
    for panel_row in sorted(panel_rows, key=lambda item: item["id"]):
        dataset_id = panel_row["id"]
        catalog_row = catalog_rows.get(dataset_id)
        if catalog_row is None or catalog_row.get("task") != "binary":
            raise ValueError(
                f"admitted binary panel id missing from v246 catalog: {dataset_id}"
            )
        local_dir = panel_root / dataset_id
        local_manifest_path = local_dir / "manifest.json"
        manifest = json.loads(local_manifest_path.read_text())
        train_path = local_dir / "train.csv"
        train_sha = sha256_file(train_path)
        train_meta = manifest["files"]["train"]
        if (
            train_sha != catalog_row["train_sha256"]
            or train_sha != train_meta["sha256"]
            or sha256_file(local_manifest_path) != panel_row["manifest_sha256"]
        ):
            raise ValueError(f"train payload hash mismatch: {dataset_id}")
        if manifest.get("id") != dataset_id or manifest.get("task") != "binary":
            raise ValueError(f"local train manifest identity mismatch: {dataset_id}")
        expected_rows = int(catalog_row["train_rows"])
        expected_cols = int(catalog_row["cols"])
        if (
            int(panel_row["train_rows"]) != expected_rows
            or int(train_meta["rows"]) != expected_rows
            or int(train_meta["columns"]) != expected_cols
        ):
            raise ValueError(f"train shape metadata mismatch: {dataset_id}")
        target_index = int(manifest["target"]["index"])
        x, y = _load_csv(train_path, expected_rows, expected_cols, target_index)
        family = family_by_id[dataset_id]
        relative = f"binary/{dataset_id}.npz"
        final_path = output / relative
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dataset_id}.", suffix=".npz", dir=pool_dir
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(
                    stream,
                    x=x,
                    y=y,
                    dataset=np.asarray(dataset_id),
                    task=np.asarray("binary"),
                    family=np.asarray(family),
                    train_csv_sha256=np.asarray(train_sha),
                    catalog_sha256=np.asarray(catalog_sha),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, final_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        payload_sha = sha256_file(final_path)
        context_id = canonical_json_sha(
            {
                "dataset": dataset_id,
                "family": family,
                "task": "binary",
                "role": "panel",
                "payload_sha256": payload_sha,
            }
        )
        contexts.append(
            {
                "id": context_id,
                "dataset": dataset_id,
                "family": family,
                "task": "binary",
                "role": "panel",
                "source_split": "train",
                "origin": "real",
                "has_meta_test": False,
                "payload": relative,
                "payload_sha256": payload_sha,
                "source_train_sha256": train_sha,
                "source_manifest_sha256": panel_row["manifest_sha256"],
            }
        )
    family_review.pop("family_by_dataset")
    family_review_sha = canonical_json_sha(family_review)
    (output / "binary-family-review.json").write_text(
        json.dumps(family_review, indent=2, sort_keys=True) + "\n"
    )
    family_counts: dict[str, int] = defaultdict(int)
    for context in contexts:
        family_counts[context["family"]] += 1
    manifest = {
        "schema": SCHEMA,
        "eligibility_verified": False,
        "admission_status": "PARTIAL_PANEL_ONLY",
        "catalog_sha256": catalog_sha,
        "panel_sha256": panel_sha,
        "family_review_sha256": family_review_sha,
        "policy": {
            "panel": "panel_v1 owner-admitted (t2015); binary train rows only",
            "sealed_roster": "P3 sealed_excluded families by owner reference; roster contents not read and overlap not computed",
            "eligibility": "not a complete joint train/validation/panel manifest",
        },
        "binary_panel_datasets": len(contexts),
        "binary_panel_families": len(family_counts),
        "binary_panel_family_shortfall_vs_240": max(0, 240 - len(family_counts)),
        "contexts": contexts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_binary_panel_packet(
        args.catalog, args.panel, args.panel_root, args.out
    )
    print(
        json.dumps(
            {
                "status": manifest["admission_status"],
                "datasets": manifest["binary_panel_datasets"],
                "families": manifest["binary_panel_families"],
                "family_shortfall": manifest["binary_panel_family_shortfall_vs_240"],
                "manifest_sha256": sha256_file(args.out / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
