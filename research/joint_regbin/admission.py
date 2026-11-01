"""Catalog-bound JRB input admission helpers.

The binary panel builder reads only the admitted panel's train CSVs. It does not
open dataset test files or P3 sealed-roster contents. Its output is explicitly a
partial manifest until the complete role inventory is supplied and reviewed.
"""

from __future__ import annotations

import argparse
import base64
import binascii
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

from . import protocol

SCHEMA = "jope.joint-regbin/v1"
PARTIAL_SCHEMA = "jope.joint-regbin.binary-panel-packet/v1"
CATALOG_POOL_SCHEMA = "jope.joint-regbin.catalog-pools/v1"
ROLE_CANDIDATE_SCHEMA = "jope.joint-regbin.task-role-candidates/v1"
CATALOG_V246_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
OWNER_ALIAS_RECEIPT_SHA256 = (
    "114c57bf328c3556315a4b69f2f230ee0987086bda5ee1043aa50016ad9147b4"
)
CATALOG_POOL_RECEIPT_SHA256 = (
    "9f82ecb4df818975072071b5d6464f3b1b6b9478ce54c6eed84b09da2a400953"
)
FLYWHEEL_SELECTED_SHA256 = (
    "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
)
FLYWHEEL_RUN_MANIFEST_SHA256 = (
    "b638084435e4a2363098586b0f1a5f9e014b552c45e2a6b7bb6f2cc829f4171f"
)
ELO_PANEL_V1_SHA256 = "e0ce676f62b4df77e586d42b6fa1ebfe48c148a0dd87d17fdd0cb98a51e67ab9"
PROTECTED_FAMILY_MANIFEST_SHA256 = (
    "0314599fe06515d3f3e976c0d435cbe4e9fbe26738014bd89470f525702f983b"
)
PROTECTED_FAMILY_MANIFEST_SCHEMA = "jope.h150.sealed-family-identity-manifest/v1"


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


def review_catalog_families(records: list[dict]) -> dict:
    """Close P2, source-file and URL aliases across regression and binary rows."""
    family_keys = _family_tool()
    selected = [row for row in records if row.get("task") in protocol.TASKS]
    by_id = {row.get("id"): row for row in selected}
    if not selected or None in by_id or len(by_id) != len(selected):
        raise ValueError("catalog family review has no rows or duplicate/missing IDs")
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
        attrs = family_keys.attrs_from_catalog_row(row)
        provenance = dict(attrs.get("provenance", {}))
        provenance.update(
            {
                key: value
                for key, value in {
                    "platform": row.get("platform"),
                    "url": row.get("source_url"),
                    "dedupe_fingerprint": row.get("family_key"),
                }.items()
                if value is not None
            }
        )
        attrs["provenance"] = provenance
        keys = set(family_keys.derive(attrs)[0])
        srcfile = _srcfile_key(row)
        if srcfile:
            keys.add(("srcfile", srcfile))
        url_parent = _url_parent_key(row.get("source_url"))
        if url_parent:
            keys.add(("url_parent", url_parent))
        if not keys:
            raise ValueError(
                f"catalog dataset lacks family alias evidence: {dataset_id}"
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
            "task": row["task"],
            "catalog_family_key": row.get("family_key"),
            "catalog_family_key_source": row.get("family_key_source"),
            "catalog_family_key_derived": row.get("family_key_derived"),
            "promoted_srcfile_key": srcfile,
            "url_parent_key": url_parent,
            "union_evidence": evidence,
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
            family = f"p2:{component_evidence[0][0]}:{component_evidence[0][1]}"
        member_tasks = sorted({by_id[dataset_id]["task"] for dataset_id in member_ids})
        for dataset_id in member_ids:
            family_by_id[dataset_id] = family
        component_rows.append(
            {
                "family": family,
                "datasets": member_ids,
                "tasks": member_tasks,
                "union_evidence": [list(item) for item in component_evidence],
            }
        )

    return {
        "schema": "jope.joint-regbin.catalog-family-alias-review/v1",
        "method": "catalog P2 family_keys plus srcfile stem and URL-parent union",
        "dedupe_tool_schema": family_keys.SCHEMA,
        "dedupe_tool_policy_sha256": family_keys.KEYS_POLICY_SHA256,
        "dataset_count": len(selected),
        "family_count": len(component_rows),
        "task_counts": {
            task: sum(row.get("task") == task for row in selected)
            for task in protocol.TASKS
        },
        "families": component_rows,
        "records": [
            dict(review_by_id[dataset_id], family=family_by_id[dataset_id])
            for dataset_id in sorted(by_id)
        ],
        "family_by_dataset": dict(sorted(family_by_id.items())),
    }


def build_catalog_pool_manifest(
    catalog: dict,
    catalog_sha256: str,
    owner_alias: dict,
    owner_alias_sha256: str,
) -> dict:
    """Build both task candidate pools without assigning roles or eligibility."""
    inventory = catalog.get("inventory", {})
    catalog_rows = catalog.get("datasets", [])
    if (
        not isinstance(catalog_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", catalog_sha256)
        or "LATEST=v246" not in inventory.get("source_catalog", "")
        or inventory.get("usable_records") != len(catalog_rows)
        or inventory.get("catalog_records", 0) - inventory.get("usable_records", 0)
        != 14
    ):
        raise ValueError("catalog pool input is not the curated v246 metadata")
    if (
        not isinstance(owner_alias_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", owner_alias_sha256)
        or owner_alias.get("schema") != "jope.jrb01.raw-catalog-alias-expansion/v1"
        or owner_alias.get("curated_catalog_sha256") != catalog_sha256
        or owner_alias.get("owner_identities") != 167
        or owner_alias.get("owner_identities_joined_raw_catalog") != 167
        or owner_alias.get("previously_missing_identities") != 14
        or owner_alias.get("previously_missing_now_joined") != 14
        or owner_alias.get("source_file_metadata_for_previous_missing_ids") != 0
        or owner_alias.get("binary_direct_owner_alias_intersections") != 0
        or owner_alias.get("binary_transitive_owner_alias_intersections") != 0
        or owner_alias.get("all_raw_catalog_transitive_unions_checked") is not True
        or owner_alias.get("no_p3_roster_or_plaintext_sealed_aliases_published")
        is not True
        or owner_alias.get("payloads_opened") != 0
    ):
        raise ValueError("owner alias evidence is incomplete or has an overlap")
    family_keys = _family_tool()
    pinned_sources = owner_alias.get("policy_source_pins")
    expected_sources = {
        "deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/common.py",
        "deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/family_keys.py",
    }
    repo = Path(__file__).resolve().parents[2]
    if (
        owner_alias.get("policy_schema") != family_keys.SCHEMA
        or owner_alias.get("policy_sha256") != family_keys.KEYS_POLICY_SHA256
        or not isinstance(pinned_sources, dict)
        or set(pinned_sources) != expected_sources
        or any(
            sha256_file(repo / path) != expected
            for path, expected in pinned_sources.items()
        )
    ):
        raise ValueError(
            "owner alias evidence does not pin the active family-key policy"
        )

    task_rows = [row for row in catalog_rows if row.get("task") in protocol.TASKS]
    expected_counts = {
        task: inventory.get("by_task", {}).get(task) for task in protocol.TASKS
    }
    actual_counts = {
        task: sum(row.get("task") == task for row in catalog_rows)
        for task in protocol.TASKS
    }
    if expected_counts != actual_counts:
        raise ValueError("catalog pool task counts differ from v246 inventory")
    for row in task_rows:
        if (
            row.get("s3_bucket") != "veox-jopedime-datasets-use2"
            or type(row.get("train_rows")) is not int
            or row["train_rows"] <= 0
            or not isinstance(row.get("train_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["train_sha256"])
        ):
            raise ValueError(f"catalog train metadata is invalid: {row.get('id')}")

    family_review = review_catalog_families(task_rows)
    review_by_id = {row["dataset"]: row for row in family_review["records"]}
    candidate_pools = {task: [] for task in protocol.TASKS}
    for row in sorted(task_rows, key=lambda item: (item["task"], item["id"])):
        reviewed = review_by_id[row["id"]]
        candidate_pools[row["task"]].append(
            {
                "dataset": row["id"],
                "task": row["task"],
                "family": reviewed["family"],
                "family_aliases": reviewed["union_evidence"],
                "family_key_source": row.get("family_key_source"),
                "train_rows": row["train_rows"],
                "train_sha256": row["train_sha256"],
            }
        )

    sourcefile_gaps = max(
        0,
        owner_alias["previously_missing_identities"]
        - owner_alias["source_file_metadata_for_previous_missing_ids"],
    )
    owner_regression_alias_intersections_reviewed = (
        owner_alias.get("regression_direct_owner_alias_intersections") == 0
        and owner_alias.get("regression_transitive_owner_alias_intersections") == 0
    )
    return {
        "schema": CATALOG_POOL_SCHEMA,
        "admission_status": "CATALOG_CANDIDATES_NOT_ADMITTED",
        "eligibility_verified": False,
        "candidate_pool_scope": "catalog_train_rows_only",
        "role_assignments_present": False,
        "qualification_status": "NOT_QUALIFIED",
        "catalog_sha256": catalog_sha256,
        "catalog_version": "v246",
        "owner_alias_expansion_sha256": owner_alias_sha256,
        "owner_identity_manifest_sha256": owner_alias["owner_identity_manifest_sha256"],
        "family_review_sha256": canonical_json_sha(family_review),
        "family_review": family_review,
        "candidate_counts": actual_counts,
        "candidate_pools": candidate_pools,
        "closure": {
            "catalog_aliases_closed": True,
            "catalog_rows_reviewed": len(task_rows),
            "cross_task_alias_families": sum(
                len(component["tasks"]) > 1 for component in family_review["families"]
            ),
            "owner_binary_recorded_alias_intersections": 0,
            "owner_binary_alias_reviewed": True,
            "owner_regression_alias_intersections_reviewed": (
                owner_regression_alias_intersections_reviewed
            ),
            "owner_identity_count": owner_alias["owner_identities"],
            "owner_identity_raw_catalog_joins": owner_alias[
                "owner_identities_joined_raw_catalog"
            ],
            "owner_aliases_fully_closed": (
                sourcefile_gaps == 0 and owner_regression_alias_intersections_reviewed
            ),
            "unresolved_owner_sourcefile_aliases": sourcefile_gaps,
            "owner_identity_manifest_sha256": owner_alias[
                "owner_identity_manifest_sha256"
            ],
            "protected_payloads_opened": 0,
        },
        "blockers": [
            "catalog rows are candidates and do not assign reviewed train/val/panel roles",
            "owner source-file alias closure is incomplete",
            "actual two-task worker and qualification receipt are not supplied by this manifest",
        ],
    }


def build_catalog_pool_manifest_from_files(
    catalog_path: Path, owner_alias_path: Path
) -> dict:
    """Load and pin approved metadata inputs, then build fail-closed pool candidates."""
    catalog_path = Path(catalog_path)
    owner_alias_path = Path(owner_alias_path)
    catalog_bytes = catalog_path.read_bytes()
    owner_alias_bytes = owner_alias_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    owner_alias = json.loads(owner_alias_bytes)
    owner_alias_sha = hashlib.sha256(owner_alias_bytes).hexdigest()
    if owner_alias_sha != OWNER_ALIAS_EXPANSION_SHA256:
        raise ValueError("owner alias expansion is not the pinned metadata receipt")
    return build_catalog_pool_manifest(
        catalog,
        hashlib.sha256(catalog_bytes).hexdigest(),
        owner_alias,
        owner_alias_sha,
    )


def _protected_catalog_member_ids(
    manifest: dict, manifest_sha256: str
) -> tuple[set[str], set[str]]:
    """Return protected catalog member IDs and the complete protected identity set."""
    if (
        manifest.get("schema") != PROTECTED_FAMILY_MANIFEST_SCHEMA
        or manifest_sha256 != PROTECTED_FAMILY_MANIFEST_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256 or "")
    ):
        raise ValueError("protected-family manifest is not the pinned identity receipt")

    dataset_ids = manifest.get("dataset_hashes")
    view_ids = manifest.get("view_ids")
    catalog_ids = manifest.get("catalog_sealed_ids")
    families = manifest.get("families")
    counts = manifest.get("counts")
    if (
        not isinstance(dataset_ids, list)
        or not isinstance(view_ids, list)
        or not isinstance(catalog_ids, list)
        or not isinstance(families, dict)
        or not isinstance(counts, dict)
        or any(not isinstance(value, str) or not value for value in dataset_ids)
        or any(not isinstance(value, str) or not value for value in view_ids)
        or any(not isinstance(value, str) or not value for value in catalog_ids)
        or len(set(dataset_ids)) != len(dataset_ids)
        or len(set(view_ids)) != len(view_ids)
        or len(set(catalog_ids)) != len(catalog_ids)
        or len(families) != len(set(families))
    ):
        raise ValueError("protected-family manifest identity inventory is malformed")
    view_member_ids = set()
    for view_id in view_ids:
        member_id, separator, suffix = view_id.partition("__")
        if not separator or not suffix:
            raise ValueError("protected-family manifest has a malformed view identity")
        view_member_ids.add(member_id)
    if view_member_ids != set(dataset_ids):
        raise ValueError("protected-family dataset and view identities disagree")

    if any(
        not isinstance(family_id, str)
        or not family_id
        or not isinstance(family_labels, list)
        or any(not isinstance(label, str) or not label for label in family_labels)
        for family_id, family_labels in families.items()
    ):
        raise ValueError("protected-family manifest family labels are malformed")
    labels = {label for family_labels in families.values() for label in family_labels}
    if not {"p3_holdout_sealed_excluded", "p3_holdout_heldout"} <= labels:
        raise ValueError("protected-family manifest lacks sealed or DL-Holdout labels")
    expected_counts = {
        "dataset_hashes": len(dataset_ids),
        "view_ids": len(view_ids),
        "catalog_sealed_ids": len(catalog_ids),
        "families_total": len(families),
    }
    if any(counts.get(name) != count for name, count in expected_counts.items()):
        raise ValueError("protected-family manifest counts do not match its identities")

    protected_dataset_ids = set(dataset_ids)
    protected_member_ids = protected_dataset_ids | set(catalog_ids)
    return protected_dataset_ids, protected_member_ids


def build_task_role_candidate_manifest(
    catalog: dict,
    catalog_sha256: str,
    owner_alias: dict,
    owner_alias_sha256: str,
    catalog_pool: dict,
    catalog_pool_sha256: str,
    selected: dict,
    selected_sha256: str,
    flywheel_run: dict,
    flywheel_run_sha256: str,
    panel: dict,
    panel_sha256: str,
    protected_manifest: dict,
    protected_manifest_sha256: str,
) -> dict:
    """Build role and five-fold candidates without asserting payload admission."""
    if (
        catalog.get("inventory", {}).get("usable_records")
        != len(catalog.get("datasets", []))
        or "LATEST=v246" not in catalog.get("inventory", {}).get("source_catalog", "")
        or not re.fullmatch(r"[0-9a-f]{64}", catalog_sha256 or "")
    ):
        raise ValueError("role candidates require curated v246 catalog metadata")
    if (
        owner_alias.get("schema") != "jope.jrb01.raw-catalog-alias-expansion/v1"
        or owner_alias.get("curated_catalog_sha256") != catalog_sha256
        or owner_alias.get("owner_identities") != 167
        or owner_alias.get("previously_missing_now_joined") != 14
        or owner_alias.get("source_file_metadata_for_previous_missing_ids") != 0
        or owner_alias.get("all_raw_catalog_transitive_unions_checked") is not True
        or owner_alias.get("no_p3_roster_or_plaintext_sealed_aliases_published")
        is not True
        or owner_alias.get("payloads_opened") != 0
        or not re.fullmatch(r"[0-9a-f]{64}", owner_alias_sha256 or "")
    ):
        raise ValueError("role candidate owner-alias receipt is incomplete")
    if (
        catalog_pool.get("schema") != CATALOG_POOL_SCHEMA
        or catalog_pool.get("catalog_sha256") != catalog_sha256
        or catalog_pool.get("owner_alias_expansion_sha256") != owner_alias_sha256
        or catalog_pool.get("eligibility_verified") is not False
        or catalog_pool.get("qualification_status") != "NOT_QUALIFIED"
        or not re.fullmatch(r"[0-9a-f]{64}", catalog_pool_sha256 or "")
    ):
        raise ValueError("role candidate catalog-pool receipt is not the pinned input")
    if (
        selected.get("protocol") != "full-DL-Train1219-DL-Val91/v1"
        or selected_sha256 != flywheel_run.get("input_hashes", {}).get("selected.json")
        or flywheel_run.get("task_counts")
        != {
            "train_binary": 0,
            "train_regression": len(selected.get("train", [])),
            "val_binary": 0,
            "val_regression": len(selected.get("dev", [])),
        }
        or flywheel_run.get("train_contexts")
        != [row.get("context_sha256") for row in selected.get("train", [])]
        or flywheel_run.get("dev_contexts")
        != [row.get("context_sha256") for row in selected.get("dev", [])]
        or flywheel_run.get("dope_sha256") != protocol.DOPE
        or not re.fullmatch(r"[0-9a-f]{64}", selected_sha256 or "")
        or not re.fullmatch(r"[0-9a-f]{64}", flywheel_run_sha256 or "")
    ):
        raise ValueError("flywheel train/dev metadata does not match its pinned run")
    if panel.get("schema") != "jope.elo-panel/v1" or not re.fullmatch(
        r"[0-9a-f]{64}", panel_sha256 or ""
    ):
        raise ValueError("task panel metadata schema or digest is invalid")
    protected_dataset_ids, protected_member_ids = _protected_catalog_member_ids(
        protected_manifest, protected_manifest_sha256
    )

    catalog_rows = catalog.get("datasets", [])
    catalog_by_id = {row.get("id"): row for row in catalog_rows}
    if None in catalog_by_id or len(catalog_by_id) != len(catalog_rows):
        raise ValueError("catalog has duplicate or missing dataset IDs")
    family_review = review_catalog_families(catalog_rows)
    family_by_id = family_review["family_by_dataset"]
    if not protected_dataset_ids <= catalog_by_id.keys():
        raise ValueError("pinned protected members do not join to curated catalog")
    protected_catalog_ids = protected_member_ids & catalog_by_id.keys()
    protected_families = {family_by_id[dataset] for dataset in protected_catalog_ids}
    family_evidence_by_id = {
        row["dataset"]: {tuple(key) for key in row["union_evidence"]}
        for row in family_review["records"]
    }
    protected_union_keys = set().union(
        *(family_evidence_by_id[dataset] for dataset in protected_dataset_ids)
    )
    protected_union_key_sha256 = canonical_json_sha(
        [list(key) for key in sorted(protected_union_keys)]
    )
    role_exclusion_counts = {
        "train": {task: 0 for task in protocol.TASKS},
        "val": {task: 0 for task in protocol.TASKS},
        "panel": {task: 0 for task in protocol.TASKS},
    }

    def is_protected(dataset: str) -> bool:
        return (
            dataset in protected_member_ids
            or family_by_id[dataset] in protected_families
        )

    pool_rows: dict[str, dict[str, dict]] = {}
    for task in protocol.TASKS:
        rows = catalog_pool.get("candidate_pools", {}).get(task)
        if not isinstance(rows, list):
            raise ValueError("catalog-pool task is missing from role inputs")
        pool_by_id = {row.get("dataset"): row for row in rows}
        expected_ids = {row["id"] for row in catalog_rows if row.get("task") == task}
        if len(pool_by_id) != len(rows) or set(pool_by_id) != expected_ids:
            raise ValueError(f"{task} candidate pool differs from curated catalog")
        pool_rows[task] = pool_by_id

    def selected_role_rows(role: str) -> list[dict]:
        rows = selected["train"] if role == "train" else selected["dev"]
        result = []
        seen_contexts = set()
        for row in rows:
            dataset = row.get("dataset_id")
            catalog_row = catalog_by_id.get(dataset)
            if (
                row.get("task") != "regression"
                or catalog_row is None
                or catalog_row.get("task") != "regression"
                or row.get("dl_split") != ("DL-Train" if role == "train" else "DL-Val")
                or row.get("source_split") not in ("train", "inner")
                or not isinstance(row.get("view_id"), str)
                or row["view_id"] in seen_contexts
            ):
                raise ValueError("flywheel selected row is malformed or not regression")
            seen_contexts.add(row["view_id"])
            for key in ("context_sha256", "pool_sha256", "train_only_jtt1_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row.get(key) or ""):
                    raise ValueError(
                        "flywheel selected row lacks pinned context hashes"
                    )
            if is_protected(dataset):
                role_exclusion_counts[role]["regression"] += 1
                continue
            result.append(
                {
                    "context": row["view_id"],
                    "dataset": dataset,
                    "task": "regression",
                    "role": role,
                    "source_split": row["source_split"],
                    "family": family_by_id[dataset],
                    "context_sha256": row["context_sha256"],
                    "pool_sha256": row["pool_sha256"],
                    "train_only_jtt1_sha256": row["train_only_jtt1_sha256"],
                }
            )
        return result

    regression_train = selected_role_rows("train")
    regression_val = selected_role_rows("val")
    binary_train = []
    for dataset, candidate in sorted(pool_rows["binary"].items()):
        catalog_row = catalog_by_id[dataset]
        if (
            type(candidate.get("train_rows")) is not int
            or candidate["train_rows"] != catalog_row.get("train_rows")
            or candidate.get("train_sha256") != catalog_row.get("train_sha256")
        ):
            raise ValueError("binary candidate row differs from pinned catalog")
        if is_protected(dataset):
            role_exclusion_counts["train"]["binary"] += 1
            continue
        binary_train.append(
            {
                "dataset": dataset,
                "task": "binary",
                "role": "train",
                "family": family_by_id[dataset],
                "train_rows": candidate["train_rows"],
                "train_sha256": candidate["train_sha256"],
                "family_aliases": candidate["family_aliases"],
            }
        )

    panel_entries = panel.get("datasets")
    if not isinstance(panel_entries, list):
        raise ValueError("panel metadata has no dataset list")
    panel_counts = {task: 0 for task in protocol.TASKS}
    panel_rows = {task: [] for task in protocol.TASKS}
    seen_panel_ids = set()
    panel_catalog_missing = {task: 0 for task in protocol.TASKS}
    for item in panel_entries:
        dataset, task = item.get("id"), item.get("task")
        if task not in protocol.TASKS or not isinstance(dataset, str):
            raise ValueError("panel row has unknown task or missing dataset identity")
        if dataset in seen_panel_ids:
            raise ValueError("panel metadata repeats a dataset identity")
        seen_panel_ids.add(dataset)
        panel_counts[task] += 1
        catalog_row = catalog_by_id.get(dataset)
        if catalog_row is None:
            panel_catalog_missing[task] += 1
            continue
        if catalog_row.get("task") != task or catalog_row.get("train_rows") != item.get(
            "train_rows"
        ):
            raise ValueError("panel row differs from curated catalog task or size")
        if is_protected(dataset):
            role_exclusion_counts["panel"][task] += 1
            continue
        panel_rows[task].append(
            {
                "dataset": dataset,
                "task": task,
                "role": "panel",
                "family": family_by_id[dataset],
                "train_rows": catalog_row["train_rows"],
                "train_sha256": catalog_row["train_sha256"],
            }
        )
    stats = panel.get("stats", {})
    for task in protocol.TASKS:
        if stats.get(f"{task}:selected") != panel_counts[task]:
            raise ValueError("panel task counts differ from the pinned v1 roster")

    cv_rows = [*regression_train, *regression_val, *binary_train]
    fold_by_family = protocol.family_folds(cv_rows, seed=136)
    for row in cv_rows:
        row["cv_fold"] = fold_by_family[row["family"]]
    fold_counts = {
        str(fold): {
            task: sum(row["task"] == task and row["cv_fold"] == fold for row in cv_rows)
            for task in protocol.TASKS
        }
        for fold in range(5)
    }

    training_rows = {"regression": regression_train, "binary": binary_train}
    panel_train_exclusions = {}
    for task in protocol.TASKS:
        panel_families = {row["family"] for row in panel_rows[task]}
        excluded_families = sorted(
            panel_families & {row["family"] for row in training_rows[task]}
        )
        panel_train_exclusions[task] = {
            "families_sha256": canonical_json_sha(excluded_families),
            "family_count": len(excluded_families),
            "train_context_or_dataset_count": sum(
                row["family"] in set(excluded_families) for row in training_rows[task]
            ),
        }

    assigned_rows = [
        *regression_train,
        *regression_val,
        *binary_train,
        *panel_rows["regression"],
        *panel_rows["binary"],
    ]
    if any(
        row["dataset"] in protected_member_ids or row["family"] in protected_families
        for row in assigned_rows
    ):
        raise ValueError("protected family remained in an assigned role")

    role_counts = {
        "train": {
            "regression": len(regression_train),
            "binary": len(binary_train),
        },
        "val": {"regression": len(regression_val), "binary": 0},
        "panel": {task: len(panel_rows[task]) for task in protocol.TASKS},
    }
    blockers = [
        "role records are metadata candidates; payload bytes were not opened or verified",
        "the task-specific two-head worker qualification receipt is not supplied here",
    ]
    missing_regression_panel = panel_catalog_missing["regression"]
    if missing_regression_panel:
        blockers.append(
            f"{missing_regression_panel} regression panel identities are absent from the usable curated catalog"
        )
    return {
        "schema": ROLE_CANDIDATE_SCHEMA,
        "admission_status": "ROLE_CANDIDATES_NOT_ADMITTED",
        "eligibility_verified": False,
        "qualification_status": "NOT_QUALIFIED",
        "catalog_sha256": catalog_sha256,
        "catalog_pool_manifest_sha256": catalog_pool_sha256,
        "owner_alias_expansion_sha256": owner_alias_sha256,
        "flywheel_selected_sha256": selected_sha256,
        "flywheel_run_manifest_sha256": flywheel_run_sha256,
        "panel_v1_sha256": panel_sha256,
        "flywheel_source_sha": flywheel_run.get("source_sha"),
        "dope_sha256": flywheel_run.get("dope_sha256"),
        "family_review_sha256": canonical_json_sha(family_by_id),
        "family_by_dataset": dict(sorted(family_by_id.items())),
        "role_counts": role_counts,
        "cv": {
            "protocol": "family-grouped-5-fold/v1",
            "seed": 136,
            "family_by_fold": dict(sorted(fold_by_family.items())),
            "validation_context_counts_by_fold": fold_counts,
        },
        "panel_train_exclusions": panel_train_exclusions,
        "protected_exclusion": {
            "manifest_sha256": protected_manifest_sha256,
            "manifest_schema": PROTECTED_FAMILY_MANIFEST_SCHEMA,
            "protected_member_count": len(protected_member_ids),
            "protected_catalog_dataset_count": len(protected_dataset_ids),
            "catalog_alias_family_count": len(protected_families),
            "catalog_alias_family_sha256": canonical_json_sha(
                sorted(protected_families)
            ),
            "family_key_schema": family_review["dedupe_tool_schema"],
            "family_key_policy_sha256": family_review["dedupe_tool_policy_sha256"],
            "protected_derived_union_key_count": len(protected_union_keys),
            "protected_derived_union_keys_sha256": protected_union_key_sha256,
            "excluded_role_counts": role_exclusion_counts,
            "assigned_role_dataset_overlap": 0,
            "assigned_role_family_overlap": 0,
        },
        "records": {
            "regression_train": regression_train,
            "regression_val": regression_val,
            "binary_train": binary_train,
            "regression_panel": panel_rows["regression"],
            "binary_panel": panel_rows["binary"],
        },
        "closure": {
            "catalog_p2_srcfile_url_aliases_closed": True,
            "unresolved_owner_aliases_collapsed_to_one_excluded_family": 14,
            "owner_alias_family_id": "owner-aliases:conservative-single-family",
            "owner_alias_family_source_sha256": owner_alias[
                "owner_identity_manifest_sha256"
            ],
            "owner_alias_family_assigned_a_role": False,
            "panel_catalog_missing_by_task": panel_catalog_missing,
            "panel_train_family_overlap_excluded_during_panel_evaluation": True,
            "p3_sealed_and_dl_holdout_roles_excluded_by_family_keys": True,
            "sealed_meta_test_and_dl_holdout_payloads_opened": 0,
        },
        "blockers": blockers,
    }


def build_task_role_candidate_manifest_from_files(
    catalog_path: Path,
    owner_alias_path: Path,
    catalog_pool_path: Path,
    selected_path: Path,
    flywheel_run_path: Path,
    panel_path: Path,
    protected_manifest_path: Path | None = None,
) -> dict:
    """Load exact approved role metadata and bind all source files by SHA-256."""
    if protected_manifest_path is None:
        protected_manifest_path = (
            Path(__file__).resolve().parents[2]
            / "research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json"
        )
    paths = {
        "catalog": Path(catalog_path),
        "owner_alias": Path(owner_alias_path),
        "catalog_pool": Path(catalog_pool_path),
        "selected": Path(selected_path),
        "flywheel_run": Path(flywheel_run_path),
        "panel": Path(panel_path),
        "protected_manifest": Path(protected_manifest_path),
    }
    expected = {
        "catalog": CATALOG_V246_SHA256,
        "owner_alias": OWNER_ALIAS_RECEIPT_SHA256,
        "catalog_pool": CATALOG_POOL_RECEIPT_SHA256,
        "selected": FLYWHEEL_SELECTED_SHA256,
        "flywheel_run": FLYWHEEL_RUN_MANIFEST_SHA256,
        "panel": ELO_PANEL_V1_SHA256,
        "protected_manifest": PROTECTED_FAMILY_MANIFEST_SHA256,
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    actual = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    if actual != expected:
        raise ValueError(
            "role-candidate input file hash differs from owner-pinned source"
        )
    objects = {name: json.loads(data) for name, data in raw.items()}
    return build_task_role_candidate_manifest(
        objects["catalog"],
        actual["catalog"],
        objects["owner_alias"],
        actual["owner_alias"],
        objects["catalog_pool"],
        actual["catalog_pool"],
        objects["selected"],
        actual["selected"],
        objects["flywheel_run"],
        actual["flywheel_run"],
        objects["panel"],
        actual["panel"],
        objects["protected_manifest"],
        actual["protected_manifest"],
    )


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


BINARY_TRAIN_ADMISSION_SCHEMA = "jope.joint-regbin.binary-train-admission/v1"
BINARY_TRAIN_CANDIDATE_MANIFEST_SHA256 = (
    "0fa17ee3ddf2a16a84dc90d6a6d19c6467bb1497597b68c86a279e0ac0055a06"
)
BINARY_PANEL_MANIFEST_SHA256 = (
    "5ebe5b429314a536a3aa9e9cb7783a9b97409835031e7d4fafc4e05a6cd02531"
)
OWNER_ALIAS_EXPANSION_SHA256 = (
    "114c57bf328c3556315a4b69f2f230ee0987086bda5ee1043aa50016ad9147b4"
)


def _packet_file(root: Path, relative: str) -> Path:
    """Resolve only relative train-packet files without protected path tokens."""
    rel = Path(relative)
    protected = ("holdout", "sealed", "meta_test", "outer_test", "test.csv")
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or not rel.parts
        or any(token in str(rel).lower().replace("-", "_") for token in protected)
    ):
        raise ValueError("binary admission payload path is not train-only")
    packet = Path(root).resolve()
    resolved = (packet / rel).resolve()
    if not resolved.is_relative_to(packet):
        raise ValueError("binary admission payload escapes candidate packet")
    if any(
        token in str(resolved.relative_to(packet)).lower().replace("-", "_")
        for token in protected
    ):
        raise ValueError("binary admission payload resolves to a protected path")
    return resolved


def build_binary_train_admission(
    candidate_packet: Path,
    catalog_path: Path,
    owner_alias_path: Path,
    panel_packet: Path,
) -> dict:
    """Admit the closed binary TRAIN subset while withholding srcfile-dependent rows.

    This is intentionally narrower than the full JRB manifest. The approved
    raw-catalog P2 expansion proves no owner-identity overlap through recorded
    keys; 14 owner identities have no source-file value in that catalog. We
    therefore select only binary components whose entire key closure contains
    no ``srcfile`` key, leaving all other candidate rows unadmitted.
    """
    candidate_packet = Path(candidate_packet).resolve()
    catalog_path = Path(catalog_path).resolve()
    owner_alias_path = Path(owner_alias_path).resolve()
    panel_packet = Path(panel_packet).resolve()
    candidate_path = candidate_packet / "manifest.json"
    family_review_path = candidate_packet / "binary-family-review.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    owner_alias = json.loads(owner_alias_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_packet.read_text(encoding="utf-8"))
    family_review = json.loads(family_review_path.read_text(encoding="utf-8"))

    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
    candidate_manifest_sha = sha256_file(candidate_path)
    panel_sha = sha256_file(panel_packet)
    owner_alias_sha = sha256_file(owner_alias_path)
    if (
        candidate.get("schema") != "jope.joint-regbin.binary-s3-train-pool/v1"
        or candidate.get("status") != "TRAIN_POOL_CANDIDATES_NOT_ADMITTED"
        or candidate.get("eligibility_verified") is not False
        or candidate.get("catalog_sha256") != catalog_sha
        or candidate_manifest_sha != BINARY_TRAIN_CANDIDATE_MANIFEST_SHA256
        or catalog.get("inventory", {}).get("source_catalog", "").find("LATEST=v246")
        < 0
    ):
        raise ValueError("binary candidate packet is not the pinned v246 TRAIN pool")
    family_review_sha = sha256_file(family_review_path)
    if candidate.get("family_review_sha256") != family_review_sha:
        raise ValueError("candidate family review hash mismatch")
    expected_review = review_binary_families(catalog["datasets"])
    expected_review.pop("family_by_dataset")
    if family_review != expected_review:
        raise ValueError("binary family review does not match pinned catalog policy")

    if (
        owner_alias.get("schema") != "jope.jrb01.raw-catalog-alias-expansion/v1"
        or owner_alias.get("curated_catalog_sha256") != catalog_sha
        or owner_alias_sha != OWNER_ALIAS_EXPANSION_SHA256
        or owner_alias.get("owner_identity_manifest_sha256")
        != candidate.get("exclusion_manifest_sha256")
        or owner_alias.get("owner_identities") != 167
        or owner_alias.get("owner_identities_joined_raw_catalog") != 167
        or owner_alias.get("previously_missing_now_joined") != 14
        or owner_alias.get("source_file_metadata_for_previous_missing_ids") != 0
        or owner_alias.get("binary_direct_owner_alias_intersections") != 0
        or owner_alias.get("binary_transitive_owner_alias_intersections") != 0
        or owner_alias.get("all_raw_catalog_transitive_unions_checked") is not True
        or owner_alias.get("no_p3_roster_or_plaintext_sealed_aliases_published")
        is not True
        or owner_alias.get("payloads_opened") != 0
    ):
        raise ValueError("owner P2 exclusion expansion is incomplete or has an overlap")
    panel_rows = panel.get("contexts", [])
    if (
        panel_sha != BINARY_PANEL_MANIFEST_SHA256
        or panel.get("schema") != "jope.joint-regbin/v1"
        or panel.get("catalog_sha256") != catalog_sha
        or panel.get("admission_status") != "PARTIAL_PANEL_ONLY"
        or panel.get("eligibility_verified") is not False
        or panel.get("binary_panel_datasets") != 240
        or panel.get("binary_panel_families") != 233
        or len(panel_rows) != 240
        or len({row.get("dataset") for row in panel_rows}) != 240
        or any(
            row.get("task") != "binary"
            or row.get("role") != "panel"
            or row.get("source_split") != "train"
            or row.get("origin") != "real"
            or row.get("has_meta_test") is not False
            for row in panel_rows
        )
    ):
        raise ValueError("binary panel packet is not the pinned train-only panel")

    candidate_rows = candidate.get("datasets", [])
    candidate_by_id = {row.get("dataset"): row for row in candidate_rows}
    if (
        len(candidate_by_id) != 843
        or len(candidate_rows) != 843
        or any(
            row.get("task") != "binary" or row.get("source_split") != "train"
            for row in candidate_rows
        )
    ):
        raise ValueError("binary candidate inventory must contain 843 TRAIN datasets")
    reviewed_by_id = {row["dataset"]: row for row in family_review["records"]}
    components = {row["family"]: row for row in family_review["families"]}
    if len(reviewed_by_id) != 843 or len(components) != 807:
        raise ValueError(
            "family review inventory differs from the binary candidate pool"
        )
    panel_ids = {row["dataset"] for row in panel_rows}
    safe_families = {
        family
        for family, component in components.items()
        if not any(evidence[0] == "srcfile" for evidence in component["union_evidence"])
    }
    safe_ids = {
        dataset_id
        for dataset_id, reviewed in reviewed_by_id.items()
        if reviewed["family"] in safe_families
    }
    train_ids = sorted(safe_ids - panel_ids)
    safe_panel_ids = safe_ids & panel_ids
    if len(safe_ids) != 7 or len(safe_panel_ids) != 2 or len(train_ids) != 5:
        raise ValueError(
            "closed binary subset differs from the reviewed five-family TRAIN scope"
        )
    train_families = {reviewed_by_id[dataset_id]["family"] for dataset_id in train_ids}
    if len(train_families) != 5:
        raise ValueError("admitted binary TRAIN subset needs five independent families")

    records = []
    for dataset_id in train_ids:
        source = candidate_by_id[dataset_id]
        reviewed = reviewed_by_id[dataset_id]
        component = components[reviewed["family"]]
        if (
            reviewed.get("family") != source.get("family")
            or source.get("study_role") is not None
            or source.get("has_meta_test") is not False
            or source.get("family_reviewed_from_catalog_metadata") is not True
            or source.get("target_classes") != 2
        ):
            raise ValueError(f"binary TRAIN candidate metadata mismatch: {dataset_id}")
        cache_path = _packet_file(candidate_packet, source["cache_file"])
        train_path = _packet_file(candidate_packet, source["train_file"])
        if (
            sha256_file(cache_path) != source["cache_sha256"]
            or sha256_file(train_path) != source["train_file_sha256"]
            or source["train_file_sha256"] != source["catalog_train_sha256"]
        ):
            raise ValueError(f"binary TRAIN payload hash mismatch: {dataset_id}")
        record = {
            "id": canonical_json_sha(
                {
                    "dataset": dataset_id,
                    "family": reviewed["family"],
                    "task": "binary",
                    "role": "train",
                    "payload_sha256": source["cache_sha256"],
                }
            ),
            "dataset": dataset_id,
            "family": reviewed["family"],
            "task": "binary",
            "role": "train",
            "source_split": "train",
            "origin": "real",
            "has_meta_test": False,
            "payload": source["cache_file"],
            "payload_sha256": source["cache_sha256"],
            "source_train_sha256": source["train_file_sha256"],
            "train_s3_key": source["train_s3_key"],
            "manifest_s3_key": source["manifest_s3_key"],
            "family_union_evidence": component["union_evidence"],
        }
        records.append(record)

    return {
        "schema": BINARY_TRAIN_ADMISSION_SCHEMA,
        "admission_status": "ADMITTED_BINARY_TRAIN_SUBSET",
        "binary_train_admitted": True,
        "full_binary_pool_admitted": False,
        "full_joint_eligibility_verified": False,
        "catalog_sha256": catalog_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "family_review_sha256": family_review_sha,
        "owner_alias_expansion_sha256": owner_alias_sha,
        "owner_identity_manifest_sha256": candidate["exclusion_manifest_sha256"],
        "panel_manifest_sha256": panel_sha,
        "counts": {
            "candidate_binary_records": len(candidate_rows),
            "candidate_family_components": len(components),
            "srcfile_dependent_candidate_records": len(candidate_rows) - len(safe_ids),
            "safe_srcfile_free_components": len(safe_families),
            "safe_panel_records_withheld_from_train": len(safe_panel_ids),
            "admitted_train_records": len(records),
            "admitted_train_families": len(train_families),
            "unproven_owner_srcfile_aliases": 14,
        },
        "closure": {
            "recorded_p2_owner_direct_intersections": 0,
            "recorded_p2_owner_transitive_intersections": 0,
            "all_raw_catalog_transitive_unions_checked": True,
            "owner_identity_count": 167,
            "owner_identity_raw_catalog_joins": 167,
            "selected_components_have_srcfile_alias": False,
            "full_candidate_srcfile_alias_closure": False,
            "excluded_unproven_srcfile_dependent_candidates": len(candidate_rows)
            - len(safe_ids),
            "test_or_sealed_payloads_opened": 0,
        },
        "records": records,
    }


def verify_binary_train_admission(
    admission: dict,
    candidate_packet: Path,
    catalog_path: Path,
    owner_alias_path: Path,
    panel_packet: Path,
) -> dict:
    """Rebuild the narrow admission proof and require byte-stable identities."""
    expected = build_binary_train_admission(
        candidate_packet, catalog_path, owner_alias_path, panel_packet
    )
    if admission != expected:
        raise ValueError("binary TRAIN admission differs from its audited inputs")
    return expected


ROLE_SCHEMA = "jope.joint-regbin.role-admission/v1"
REAL_S3_BUCKET = "veox-jopedime-datasets-use2"
DEFAULT_ROLE_CATALOG = Path.home() / "jope_plan/master/source/S3_DATASET_CATALOG.json"
R882_CHECKPOINT_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
R882_MESSAGE_SHA256 = "45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5"
R882_BUNDLE_DIGEST = "057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"

CONTEXT_FIELDS = {
    "id",
    "dataset",
    "family",
    "task",
    "role",
    "source_split",
    "origin",
    "has_meta_test",
    "payload",
    "payload_sha256",
}
ROLE_NAMES = {"train", "val", "panel"}
WORKER_QUALIFICATION_SCHEMA = "jope.joint-regbin.worker-qualification/v1"
REQUIRED_PROOFS = (
    "role_assignment_reviewed",
    "family_mapping_reviewed",
    "payload_hashes_verified",
    "sealed_exclusion_verified",
    "binary_family_alias_closure_verified",
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha(value: object, label: str) -> str:
    protocol.check_sha(value)
    return value


def _catalog(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    catalog = json.loads(raw)
    source = catalog.get("inventory", {}).get("source_catalog", "")
    if "LATEST=v246" not in source:
        raise ValueError("JRB role admission requires the pinned v246 S3 catalog")
    return catalog, hashlib.sha256(raw).hexdigest()


def _learner_evidence(evidence: dict) -> tuple[str, str, list[str]]:
    """Return worker/checkpoint pins and blockers; signature is receipt-bound."""
    blockers = []
    release = evidence.get("release") or {}
    worker = evidence.get("worker") or {}
    if (
        release.get("label") != "r882"
        or release.get("verify") != "PASS"
        or release.get("message_sha256") != R882_MESSAGE_SHA256
        or release.get("bundle_digest") != R882_BUNDLE_DIGEST
        or release.get("source_checkpoint_sha256") != R882_CHECKPOINT_SHA256
    ):
        blockers.append("signed r882 receipt/checkpoint binding is incomplete")
    worker_sha = worker.get("sha256")
    try:
        _sha(worker_sha, "worker sha256")
    except (TypeError, ValueError):
        worker_sha = "0" * 64
        blockers.append("worker source hash is missing or invalid")
    if worker.get("qualified") is not True:
        blockers.append("two-task worker qualification is not verified")
    worker_tasks = worker.get("tasks")
    if (
        not isinstance(worker_tasks, list)
        or len(worker_tasks) != len(protocol.TASKS)
        or any(not isinstance(task, str) for task in worker_tasks)
        or set(worker_tasks) != set(protocol.TASKS)
    ):
        blockers.append("worker does not prove regression and binary heads")
    updates = worker.get("actual_optimizer_updates")
    if type(updates) is not int or updates <= 0:
        blockers.append("worker qualification has no positive optimizer updates")
    if worker.get("parameters_changed") is not True:
        blockers.append("worker qualification has no changed-parameter proof")
    if worker.get("dope_sha256") != protocol.DOPE:
        blockers.append("worker qualification does not pin frozen DOPE")
    # Carry exact receipt bytes in the admission so preflight can re-hash them.
    receipt_bytes = None
    encoded_receipt = worker.get("qualification_receipt_bytes_b64")
    if isinstance(encoded_receipt, str):
        try:
            receipt_bytes = base64.b64decode(encoded_receipt, validate=True)
            if base64.b64encode(receipt_bytes).decode("ascii") != encoded_receipt:
                receipt_bytes = None
        except (binascii.Error, ValueError):
            receipt_bytes = None
    if receipt_bytes is None:
        blockers.append("worker qualification receipt bytes are missing or invalid")
    try:
        qualification_sha = _sha(
            worker.get("qualification_sha256"), "worker qualification sha256"
        )
    except (TypeError, ValueError):
        qualification_sha = None
        blockers.append("worker qualification receipt hash is missing or invalid")
    if receipt_bytes is not None:
        if hashlib.sha256(receipt_bytes).hexdigest() != qualification_sha:
            blockers.append(
                "worker qualification receipt hash does not match its bytes"
            )
        try:
            receipt = json.loads(receipt_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            receipt = None
        if not isinstance(receipt, dict):
            blockers.append("worker qualification receipt is not a JSON object")
        else:
            receipt_tasks = receipt.get("tasks")
            if (
                receipt.get("schema") != WORKER_QUALIFICATION_SCHEMA
                or receipt.get("worker_sha256") != worker_sha
                or receipt.get("qualified") is not True
                or not isinstance(receipt_tasks, list)
                or len(receipt_tasks) != len(protocol.TASKS)
                or any(not isinstance(task, str) for task in receipt_tasks)
                or set(receipt_tasks) != set(protocol.TASKS)
                or receipt.get("actual_optimizer_updates") != updates
                or type(receipt.get("actual_optimizer_updates")) is not int
                or receipt.get("parameters_changed") is not True
                or receipt.get("dope_sha256") != protocol.DOPE
            ):
                blockers.append(
                    "worker qualification receipt does not attest the supplied worker evidence"
                )
    return worker_sha, R882_CHECKPOINT_SHA256, blockers


def build_role_manifest(
    catalog_path: Path,
    contexts: list[dict],
    family_by_dataset: dict[str, str],
    evidence: dict,
) -> dict:
    """Build a role manifest, deriving eligibility only from complete evidence.

    `contexts` must be sourced from reviewed train-row-only packet metadata.
    Evidence booleans are receipt claims that a reviewer must bind to the listed
    hashes; they are not inferred from a candidate catalog or a partial packet.
    """
    catalog, catalog_sha = _catalog(catalog_path)
    records = {row["id"]: row for row in catalog.get("datasets", [])}
    blockers: list[str] = []
    seen_ids: set[str] = set()
    seen_dataset_identity: dict[str, tuple[str, str]] = {}
    role_families: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    family_roles: dict[str, set[str]] = {}
    kept = []

    if not contexts:
        blockers.append("no train/dev/panel role rows supplied")
    for raw in contexts:
        if set(raw) != CONTEXT_FIELDS:
            raise ValueError("context metadata fields differ from JRB schema")
        row = dict(raw)
        if row["id"] in seen_ids:
            raise ValueError("duplicate context identity")
        seen_ids.add(row["id"])
        if row["role"] not in ROLE_NAMES:
            raise ValueError("only train, val, and panel roles are accepted")
        if row["source_split"] not in ("train", "inner"):
            raise ValueError("role rows must originate from train-only source splits")
        if row["origin"] != "real" or row["has_meta_test"] is not False:
            raise ValueError("only unsealed real-S3 metadata is accepted")
        protocol.relative_payload(row["payload"])
        _sha(row["payload_sha256"], "payload sha256")
        catalog_row = records.get(row["dataset"])
        if catalog_row is None or catalog_row.get("s3_bucket") != REAL_S3_BUCKET:
            raise ValueError("context dataset is not from the approved real S3 bucket")
        if catalog_row.get("task") != row["task"] or row["task"] not in protocol.TASKS:
            raise ValueError(
                "catalog task differs or is outside JRB regression/binary scope"
            )
        family = family_by_dataset.get(row["dataset"])
        if not isinstance(family, str) or not family.strip() or row["family"] != family:
            raise ValueError("family mapping is missing or differs from reviewed input")
        prior = seen_dataset_identity.setdefault(row["dataset"], (family, row["task"]))
        if prior != (family, row["task"]):
            raise ValueError("dataset task/family identity changed")
        role_families[row["role"]].add(family)
        family_roles.setdefault(family, set()).add(row["role"])
        kept.append(row)

    if any("train" in roles and "val" in roles for roles in family_roles.values()):
        blockers.append("a family occurs in both train and dev/val roles")
    for role in ROLE_NAMES:
        for task in protocol.TASKS:
            if not any(r["role"] == role and r["task"] == task for r in kept):
                blockers.append(f"missing {role} {task} role")
    train_families = {
        task: {r["family"] for r in kept if r["role"] == "train" and r["task"] == task}
        for task in protocol.TASKS
    }
    if any(len(families) < 5 for families in train_families.values()):
        blockers.append("fewer than five independent train families for a task")

    proof = evidence.get("data") or {}
    for key in REQUIRED_PROOFS:
        if proof.get(key) is not True:
            blockers.append(f"data proof missing: {key}")
    for key in (
        "role_review_sha256",
        "family_review_sha256",
        "payload_audit_sha256",
        "sealed_exclusion_sha256",
    ):
        try:
            _sha(proof.get(key), key)
        except (TypeError, ValueError):
            blockers.append(f"data proof hash missing: {key}")
    role_identity = [
        {key: row[key] for key in sorted(CONTEXT_FIELDS)}
        for row in sorted(kept, key=lambda item: item["id"])
    ]
    if proof.get("role_review_sha256") != canonical_sha256(role_identity):
        blockers.append("role review hash does not bind the exact role rows")
    family_identity = dict(sorted(family_by_dataset.items()))
    if proof.get("family_review_sha256") != canonical_sha256(family_identity):
        blockers.append("family review hash does not bind the exact family mapping")
    payload_identity = [
        {
            "id": row["id"],
            "payload": row["payload"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in sorted(kept, key=lambda item: item["id"])
    ]
    if proof.get("payload_audit_sha256") != canonical_sha256(payload_identity):
        blockers.append("payload audit hash does not bind the exact payload inventory")

    worker_sha, checkpoint_sha, learner_blockers = _learner_evidence(evidence)
    blockers.extend(learner_blockers)
    manifest = {
        "schema": protocol.SCHEMA,
        "admission_schema": ROLE_SCHEMA,
        "admission_status": "ADMITTED"
        if not blockers
        else "BLOCKED_INCOMPLETE_EVIDENCE",
        "eligibility_verified": not blockers,
        "catalog_sha256": catalog_sha,
        "learner_sha256": worker_sha,
        "initial_checkpoint_sha256": checkpoint_sha,
        "contexts": sorted(
            kept, key=lambda row: (row["role"], row["task"], row["family"], row["id"])
        ),
        "family_by_dataset": dict(sorted(family_by_dataset.items())),
        "evidence": evidence,
        "blockers": sorted(set(blockers)),
    }
    return manifest


def rederive_role_manifest(
    manifest: dict, catalog_path: Path = DEFAULT_ROLE_CATALOG
) -> dict:
    """Rebuild every eligibility field from the pinned catalog and supplied proof."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("admission_schema") != ROLE_SCHEMA
    ):
        raise ValueError("worker-bound role admission receipt required")
    try:
        rebuilt = build_role_manifest(
            catalog_path,
            manifest["contexts"],
            manifest["family_by_dataset"],
            manifest["evidence"],
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError("could not rederive role admission from pinned proof") from exc
    if rebuilt != manifest:
        raise ValueError("manifest differs from rederived role admission proof")
    if rebuilt["eligibility_verified"] is not True:
        raise ValueError("incomplete JRB inputs cannot produce a runnable plan")
    return rebuilt


def make_worker_bound_plan(
    manifest: dict,
    config: dict | None = None,
    *,
    catalog_path: Path = DEFAULT_ROLE_CATALOG,
) -> dict:
    """Create the preregistered 3-arm plan only after rederiving all evidence."""
    admitted = rederive_role_manifest(manifest, catalog_path)
    worker = admitted["evidence"]["worker"]
    release = admitted["evidence"]["release"]
    if admitted["initial_checkpoint_sha256"] != R882_CHECKPOINT_SHA256:
        raise ValueError(
            "initial learner checkpoint must be signed r882 custody 0084479a"
        )
    if (
        release.get("verify") != "PASS"
        or release.get("message_sha256") != R882_MESSAGE_SHA256
    ):
        raise ValueError("signed r882 verification receipt is missing")
    if worker.get("sha256") != admitted["learner_sha256"]:
        raise ValueError("worker hash differs from the admitted manifest")
    plan = protocol.make_plan(admitted, config)
    plan["role_admission"] = admitted
    plan["learner_start"] = {
        "release": "r882",
        "bundle_digest": R882_BUNDLE_DIGEST,
        "signing_message_sha256": R882_MESSAGE_SHA256,
        "checkpoint_sha256": R882_CHECKPOINT_SHA256,
        "worker_sha256": admitted["learner_sha256"],
        "worker_qualification_sha256": worker["qualification_sha256"],
    }
    plan["sha256"] = protocol.digest(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    return plan


def verify_worker_binding(plan: dict) -> None:
    binding = plan.get("learner_start") or {}
    if (
        binding.get("release") != "r882"
        or binding.get("bundle_digest") != R882_BUNDLE_DIGEST
        or binding.get("signing_message_sha256") != R882_MESSAGE_SHA256
        or binding.get("checkpoint_sha256") != R882_CHECKPOINT_SHA256
        or binding.get("checkpoint_sha256") != plan["initial_checkpoint_sha256"]
        or binding.get("worker_sha256") != plan["learner_sha256"]
    ):
        raise ValueError("worker-bound r882 plan identity mismatch")


def verify_worker_bound_plan(
    plan: dict, *, catalog_path: Path = DEFAULT_ROLE_CATALOG
) -> None:
    protocol.verify_plan(plan)
    admission = plan.get("role_admission")
    verify_worker_binding(plan)
    expected = make_worker_bound_plan(
        admission, plan.get("config"), catalog_path=catalog_path
    )
    if expected != plan:
        raise ValueError("plan differs from rederived role admission proof")


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
