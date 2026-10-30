#!/usr/bin/env python3
"""HWV-06: CPU-only, source-pinned context inventory, never a numeric training grant.

Build with ``dataset_builder.py qualified-manifest --sources DIR --output DIR``.
Only JSON/JSONL metadata is read. References inside metadata are never followed.
In-source tests are collected explicitly with ``pytest input_manifest.py``; the
task's directory-level manifest/overlap selector remains a separate check.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SCHEMA = "jope.flywheel-context-inventory/v1"
ROOT = Path.home() / "jope-data/hwv-06"
SOURCES = (
    "S3_DATASET_CATALOG.json",
    "h15-admission-ledger.jsonl",
    "h15-independent-verification.json",
    "track2-eligible-manifest.json",
    "corpus-identities.json",
    "historical-role-identities.json",
)
REQUIRED_GUARDS = ("st_val_gate_32", "eval_phys_set", "DL-Holdout", "sealed_meta_test")
TRAIN_ROLES = {"train_fit", "train_nonpool"}


class ManifestError(ValueError):
    """Metadata cannot support the requested inventory or admission claim."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate metadata key: {key}")
        value[key] = item
    return value


def _json(data):
    def invalid(value):
        raise ManifestError(f"non-finite metadata literal: {value}")

    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid)


def load_sources(directory):
    """Verify every approved metadata byte before parsing; do not follow row locators."""
    directory = Path(directory)
    pins = _json((directory / "source-pins.json").read_bytes())
    if pins.get("schema") != "jope.hwv06.metadata-pins/v1" or set(
        pins["sources"]
    ) != set(SOURCES):
        raise ManifestError("complete metadata source pins required")
    values = {}
    for name in SOURCES:
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            raise ManifestError(f"expected bounded regular metadata file: {name}")
        data = path.read_bytes()
        if digest(data) != pins["sources"][name]:
            raise ManifestError(f"metadata hash mismatch: {name}")
        values[name] = (
            [_json(line) for line in data.splitlines() if line.strip()]
            if name.endswith("jsonl")
            else _json(data)
        )
    qa = values["h15-independent-verification.json"]
    if qa.get("state") != "PASS_INDEPENDENT_METADATA_BINDING_AND_AGGREGATES":
        raise ManifestError("H15 metadata verification missing")
    if qa["ledger_sha256"] != pins["sources"]["h15-admission-ledger.jsonl"]:
        raise ManifestError("H15 verification does not bind the ledger")
    if qa["frontier_rows"] != len(values["h15-admission-ledger.jsonl"]):
        raise ManifestError("H15 source denominator mismatch")
    return values, pins["sources"]


class _Aliases:
    def __init__(self):
        self.parent = {}

    def root(self, node):
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, left, right):
        a, b = sorted((self.root(left), self.root(right)))
        self.parent[b] = a


def _hash(value, length):
    return (
        isinstance(value, str)
        and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None
    )


def build_inventory(values, source_hashes):
    """Conservative Policy C metadata anti-join; retain withheld numeric admission."""
    catalog = values["S3_DATASET_CATALOG.json"]
    h15 = values["h15-admission-ledger.jsonl"]
    track = values["track2-eligible-manifest.json"]
    corpus = values["corpus-identities.json"]["records"]
    roles = values["historical-role-identities.json"]["roles"]
    cells = track["cells"]
    if track.get("schema") != "dope.mlc.v3.high.eligible_allowlist_manifest/v2r1":
        raise ManifestError("unknown Track 2 metadata schema")
    if track["eligible_cell_count"] != len(cells):
        raise ManifestError("Track 2 denominator mismatch")
    if (
        len(roles.get("st_val_gate_32", [])) != 32
        or len(set(roles["st_val_gate_32"])) != 32
    ):
        raise ManifestError("complete st_val_gate_32 identity roster required")
    by_id = {r["id"]: r for r in catalog["datasets"]}
    if len(by_id) != len(catalog["datasets"]):
        raise ManifestError("duplicate catalog dataset identity")
    by_context = {r["context_sha256"]: r for r in corpus}
    if len(by_context) != len(corpus):
        raise ManifestError("duplicate corpus context identity")
    aliases = _Aliases()
    physical = defaultdict(set)
    known_family = set()
    guards = defaultdict(set)
    protected_contexts = defaultdict(set)

    def dataset(identifier):
        if not _hash(identifier, 16):
            raise ManifestError(f"invalid dataset identity: {identifier!r}")
        node = "dataset:" + identifier
        aliases.root(node)
        return node

    def family(identifier, name, *, qualified=True):
        if isinstance(name, str) and name:
            aliases.join(dataset(identifier), "family:" + name)
            if qualified:
                known_family.add(identifier)

    def protect(category, identifier, context=None):
        guards[category].add(dataset(identifier))
        if context:
            protected_contexts[category].add(context)

    for identifier, row in by_id.items():
        dataset(identifier)
        if not _hash(row.get("canonical_sha256"), 64) or not row[
            "canonical_sha256"
        ].startswith(identifier):
            raise ManifestError("catalog canonical identity mismatch")
        # Derived/unreviewed source-file keys never become family authority.
        family(identifier, row.get("family_key"))
        origin = json.dumps(
            {
                key: row.get(key)
                for key in (
                    "name",
                    "collections",
                    "source_url",
                    "source_file",
                    "platform",
                )
            }
        ).lower()
        if "beyondarena" in origin or "/sealed/" in origin or "sealed_meta" in origin:
            protect("sealed_meta_test", identifier)
    for identifier in catalog["inventory"]["sealed_name_matches_excluded"]:
        protect("sealed_meta_test", identifier)
    for row in corpus:
        identifier = row["dataset_id"]
        family(identifier, row.get("family_id"))
        aliases.join(dataset(identifier), "context:" + row["context_sha256"])
        if row["dl_split"] in ("DL-Holdout", "DL-Val"):
            protect(row["dl_split"], identifier, row["context_sha256"])
    for source, category in (
        ("st_val_gate_32", "st_val_gate_32"),
        ("dl_holdout_91", "DL-Holdout"),
        ("dl_val_71", "DL-Val"),
    ):
        if not roles.get(source):
            raise ManifestError(f"protected identity roster missing: {source}")
        for context in roles[source]:
            if context not in by_context:
                raise ManifestError(f"unresolved protected context: {context}")
            protect(category, by_context[context]["dataset_id"], context)
    for row in h15:
        bindings = row.get("catalog_bindings", [])
        for binding in bindings:
            identifier = binding["dataset_hash"]
            # Unresolved historical aliases may conservatively widen exclusions,
            # but cannot certify a current dataset's family for positive selection.
            qualified = (
                row.get("namespace") == "canonical"
                and row.get("exact_dataset_binding") is True
                and row.get("canonical_identity_conflict") is False
                and identifier in by_id
                and binding.get("canonical_sha256")
                == by_id[identifier]["canonical_sha256"]
            )
            family(identifier, row.get("family_id"), qualified=qualified)
            aliases.join(dataset(identifier), "context:" + row["context_sha256"])
            for role in row.get("roles", []):
                category = {
                    "sealed_excluded": "sealed_meta_test",
                    "heldout": "DL-Holdout",
                    "dev": "DL-Val",
                }.get(role)
                if category:
                    protect(category, identifier, row["context_sha256"])
            if row.get("h12_reserved"):
                protect("H12_reservation", identifier, row["context_sha256"])
    for row in cells:
        identifier = row["dataset_id"]
        table = row.get("physical_table_id")
        precalc = row.get("precalc_cell_key")
        if not _hash(table, 64) or (precalc is not None and not _hash(precalc, 64)):
            raise ManifestError("Track 2 physical/precalculation identity invalid")
        if not isinstance(row.get("split"), str) or not row["split"]:
            raise ManifestError("Track 2 split identity missing")
        if precalc is None and row.get("precalc_plan_state") != "unsupported":
            raise ManifestError(
                "supported Track 2 cell lacks its precalculation identity"
            )
        physical[identifier].add(table)
        aliases.join(dataset(identifier), "physical:" + table)
        # All non-training roles, including exclusions, are quarantined by physical alias.
        if row["v3_role"] != "train_gradient":
            protect("eval_phys_set", identifier)
    if any(not guards.get(name) for name in REQUIRED_GUARDS):
        raise ManifestError("nonempty protected metadata rosters required")
    protected = {
        name: {aliases.root(node) for node in nodes} for name, nodes in guards.items()
    }
    component_members = defaultdict(list)
    for node in aliases.parent:
        component_members[aliases.root(node)].append(node)
    component_hashes = {
        root: digest(
            ("jope.hwv06.alias-component/v1\0" + "\0".join(sorted(nodes))).encode()
        )
        for root, nodes in component_members.items()
    }
    rejected = Counter()
    records = []
    identities = set()

    def admit(kind, identity, identifiers, details):
        if (kind, identity) in identities:
            raise ManifestError("duplicate source context/cell identity")
        identities.add((kind, identity))
        reason = None
        roots = {aliases.root(dataset(identifier)) for identifier in identifiers}
        if not identifiers or any(
            identifier not in by_id for identifier in identifiers
        ):
            reason = "missing_catalog_binding"
        elif any(
            by_id[identifier]["task"] != "regression" for identifier in identifiers
        ):
            reason = "not_regression"
        elif any(
            identifier not in known_family or not physical[identifier]
            for identifier in identifiers
        ):
            reason = "unresolved_family_or_physical_identity"
        elif len(roots) != 1:
            reason = "conflicting_alias_components"
        hits = [
            name
            for name, blocked in protected.items()
            if roots & blocked
            or (kind == "h15" and identity in protected_contexts[name])
        ]
        if hits:
            reason = "protected:" + ",".join(sorted(hits))
        if reason:
            rejected[kind + ":" + reason] += 1
            return
        root = next(iter(roots))
        records.append(
            {
                "source_kind": kind,
                "id": kind + ":" + identity,
                "source_identity": identity,
                "dataset_ids": sorted(identifiers),
                "component_sha256": component_hashes[root],
                "physical_table_ids": sorted(
                    {
                        table
                        for identifier in identifiers
                        for table in physical[identifier]
                    }
                ),
                "numeric_training_admission": False,
                **details,
            }
        )

    for row in h15:
        if (
            row.get("namespace") != "canonical"
            or row.get("canonical_identity_conflict")
            or not row.get("exact_dataset_binding")
        ):
            rejected["h15:unresolved_canonical_metadata"] += 1
            continue
        if not row.get("roles") or not set(row["roles"]) <= TRAIN_ROLES:
            rejected["h15:non_training_role"] += 1
            continue
        bindings = row.get("catalog_bindings", [])
        if any(
            binding["dataset_hash"] in by_id
            and binding["canonical_sha256"]
            != by_id[binding["dataset_hash"]]["canonical_sha256"]
            for binding in bindings
        ):
            rejected["h15:catalog_version_mismatch"] += 1
            continue
        admit(
            "h15",
            row["context_sha256"],
            {binding["dataset_hash"] for binding in bindings},
            {
                "source_roles": sorted(row["roles"]),
                "source_eligible_for_new_fits": row.get("eligible_for_new_fits", False),
            },
        )
    for row in cells:
        if row["v3_role"] != "train_gradient" or row["model_task"] != "regression":
            rejected["track2:non_regression_or_non_training_role"] += 1
            continue
        # A cell's inventory coordinate is dataset/split. An unsupported native-CV
        # precalculation has no key; never invent one or relabel its synthesized probe.
        identity = row["dataset_id"] + "/" + row["split"]
        admit(
            "track2",
            identity,
            {row["dataset_id"]},
            {
                "split": row["split"],
                "source_role": row["v3_role"],
                "source_precalc_cell_key": row.get("precalc_cell_key"),
                "precalc_plan_state": row["precalc_plan_state"],
                "landmark_report_origin": row["landmark_report_origin"],
                "legacy_wire_sha256": row["wire_dml3_sha256"],
                "legacy_encoding_sha256": row["fixed_npz_sha256"],
            },
        )
    records.sort(key=lambda row: row["id"])
    if not records:
        raise ManifestError("no eligible metadata contexts; no vacuous overlap pass")
    if len(records) + sum(rejected.values()) != len(h15) + len(cells):
        raise ManifestError("source denominator not fully accounted for")
    retained_components = {row["component_sha256"] for row in records}
    retained_h15 = {
        row["source_identity"] for row in records if row["source_kind"] == "h15"
    }
    allowed_roots = {
        root for root, value in component_hashes.items() if value in retained_components
    }
    overlaps = {
        name: len(allowed_roots & blocked) for name, blocked in protected.items()
    }
    if any(overlaps.values()):
        raise ManifestError("protected alias component admitted")
    manifest = {
        "schema": SCHEMA,
        "policy": "C: conservative physical/family quarantine",
        "task": "regression",
        "purpose": "metadata_inventory_only",
        "numeric_training_admission": False,
        "live_packet_compatible": False,
        "expected_frozen_dope_sha256_prefix": "c51a56d0",
        "dope_loaded_or_modified": False,
        "legacy_track2_checkpoint_sha256": track["checkpoint_sha256"],
        "source_proof_levels": {
            "h15": {
                key: values["h15-independent-verification.json"].get(key, "UNKNOWN")
                for key in (
                    "state",
                    "new_numeric_admissions",
                    "all_source_original_row_chain_unqualified",
                    "all_full_historical_exposure_unknown",
                )
            },
            "track2": track["scientific_and_parity_qualifications"],
        },
        "source_sha256": source_hashes,
        "records": records,
    }
    receipt = {
        "schema": "jope.hwv06.overlap-receipt/v1",
        "manifest_sha256": digest(encoded(manifest)),
        "source_rows": {"h15": len(h15), "track2": len(cells)},
        "retained_by_source": dict(Counter(r["source_kind"] for r in records)),
        "retained_contexts": len(records),
        "retained_alias_components": len(allowed_roots),
        "source_family_counts": {
            "retained_h15_canonical_families": len(
                {
                    row["family_id"]
                    for row in h15
                    if row["context_sha256"] in retained_h15
                }
            ),
            "retained_catalog_recorded_families": len(
                {
                    by_id[identifier]["family_key"]
                    for r in records
                    for identifier in r["dataset_ids"]
                    if by_id[identifier].get("family_key")
                }
            ),
            "family_physical_alias_components": len(allowed_roots),
        },
        "retained_datasets": len(
            {identifier for r in records for identifier in r["dataset_ids"]}
        ),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "protected_alias_components": {
            name: len(blocked) for name, blocked in protected.items()
        },
        "checked_metadata_component_overlaps": overlaps,
        "global_sealed_roster_available": False,
        "sealed_scope": "catalog exclusions, H15 registry roles, and denial of all marked BeyondArena origins; global authoritative roster not supplied",
        "numeric_training_admission": False,
        "fifty_round_live_qualification": "DEFERRED_CPU_ONLY_1225",
    }
    return manifest, receipt


def require_training_admission(manifest):
    if not manifest.get("numeric_training_admission"):
        raise ManifestError(
            "metadata-only inventory: numerical input/label admission remains withheld"
        )


def write_new(path, value, root):
    """Atomic create-only publication under the isolated output root; identical replay is safe."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ManifestError("output must stay in the isolated HWV-06 root")
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".hwv06-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ManifestError(
                    "existing output differs; preserve it and choose a new output directory"
                ) from None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-training-admission", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.sources.resolve().is_relative_to(
            ROOT.resolve()
        ) or not args.output.resolve().is_relative_to(ROOT.resolve()):
            raise ManifestError(
                "source copies and outputs must stay under ~/jope-data/hwv-06"
            )
        values, pins = load_sources(args.sources)
        manifest, receipt = build_inventory(values, pins)
        if args.require_training_admission:
            require_training_admission(manifest)
        for name, value in [
            ("expanded-manifest.json", manifest),
            ("overlap-receipt.json", receipt),
        ]:
            path = args.output / name
            if args.verify:
                if path.read_bytes() != encoded(value):
                    raise ManifestError(f"inventory replay differs: {name}")
            else:
                write_new(path, value, ROOT)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return 0
    except (ManifestError, KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        print(f"manifest refused: {error}", file=sys.stderr)
        return 2


def _manifest_test_inputs():
    """Metadata protocol fixtures contain identifiers/roles, never scientific samples."""
    a, b, c, d = [str(i) * 16 for i in range(1, 5)]

    def catalog_row(identifier, name):
        return {
            "id": identifier,
            "canonical_sha256": identifier + "0" * 48,
            "task": "regression",
            "family_key": name,
            "collections": [],
        }

    def h15_row(identifier, role):
        return {
            "context_sha256": identifier + "1" * 48,
            "family_id": "h15:" + identifier,
            "roles": [role],
            "namespace": "canonical",
            "exact_dataset_binding": True,
            "canonical_identity_conflict": False,
            "eligible_for_new_fits": False,
            "catalog_bindings": [
                {"dataset_hash": identifier, "canonical_sha256": identifier + "0" * 48}
            ],
        }

    def cell(identifier, split, role, table, key):
        return {
            "dataset_id": identifier,
            "split": split,
            "v3_role": role,
            "physical_table_id": table * 64,
            "precalc_cell_key": key * 64 if key else None,
            "precalc_plan_state": "supported" if key else "unsupported",
            "landmark_report_origin": "s3_precalc_native_cv"
            if key
            else "train_synthesized_v3_probes",
            "model_task": "regression",
            "wire_dml3_sha256": "a" * 64,
            "fixed_npz_sha256": "b" * 64,
        }

    contexts = [f"{i + 1:064x}" for i in range(32)]
    values = {
        "S3_DATASET_CATALOG.json": {
            "inventory": {"sealed_name_matches_excluded": [c]},
            "datasets": [
                catalog_row(a, "allowed"),
                catalog_row(b, "gate"),
                catalog_row(d, "eval"),
            ],
        },
        "h15-admission-ledger.jsonl": [
            h15_row(a, "train_fit"),
            h15_row(c, "sealed_excluded"),
        ],
        "h15-independent-verification.json": {"new_numeric_admissions": 0},
        "track2-eligible-manifest.json": {
            "schema": "dope.mlc.v3.high.eligible_allowlist_manifest/v2r1",
            "eligible_cell_count": 4,
            "checkpoint_sha256": "f" * 64,
            "scientific_and_parity_qualifications": {
                "training_admission_granted": False,
                "parity_qualified_strict_1e6": False,
            },
            "cells": [
                cell(a, "fit", "train_gradient", "a", "7"),
                cell(a, "full", "train_gradient", "a", None),
                cell(b, "fit", "train_gradient", "b", "8"),
                cell(d, "fit", "dev_val_ablation", "d", "9"),
            ],
        },
        "corpus-identities.json": {
            "records": [
                {
                    "context_sha256": context,
                    "dataset_id": b,
                    "family_id": "gate",
                    "dl_split": "DL-Holdout",
                }
                for context in contexts
            ]
        },
        "historical-role-identities.json": {
            "roles": {
                "st_val_gate_32": contexts,
                "dl_holdout_91": contexts[:1],
                "dl_val_71": contexts[1:2],
            }
        },
    }
    return values


def test_manifest_protects_every_guard_and_preserves_denominators():
    manifest, receipt = build_inventory(_manifest_test_inputs(), {})
    assert len(manifest["records"]) == 3
    assert (
        receipt["retained_contexts"] + sum(receipt["rejected_by_reason"].values()) == 6
    )
    assert all(
        receipt["protected_alias_components"][name] > 0 for name in REQUIRED_GUARDS
    )
    assert not any(receipt["checked_metadata_component_overlaps"].values())
    assert {
        identifier for row in manifest["records"] for identifier in row["dataset_ids"]
    } == {"1" * 16}
    assert receipt["global_sealed_roster_available"] is False


def test_manifest_overlap_through_family_and_physical_aliases_is_refused():
    import pytest

    for domain in ("family", "physical"):
        values = _manifest_test_inputs()
        if domain == "family":
            values["S3_DATASET_CATALOG.json"]["datasets"][0]["family_key"] = "gate"
        else:
            values["track2-eligible-manifest.json"]["cells"][-1][
                "physical_table_id"
            ] = "a" * 64
        with pytest.raises(ManifestError, match="no eligible metadata contexts"):
            build_inventory(values, {})


def test_manifest_missing_guard_and_unknown_family_are_not_zero_overlap_passes():
    import pytest

    values = _manifest_test_inputs()
    values["historical-role-identities.json"]["roles"]["st_val_gate_32"] = []
    with pytest.raises(ManifestError, match="complete st_val_gate_32"):
        build_inventory(values, {})
    values = _manifest_test_inputs()
    values["historical-role-identities.json"]["roles"]["st_val_gate_32"] = [
        "0" * 64
    ] * 32
    with pytest.raises(ManifestError, match="complete st_val_gate_32"):
        build_inventory(values, {})
    values = _manifest_test_inputs()
    values["S3_DATASET_CATALOG.json"]["datasets"][0]["family_key"] = None
    values["h15-admission-ledger.jsonl"][0]["family_id"] = None
    with pytest.raises(ManifestError, match="no eligible metadata contexts"):
        build_inventory(values, {})


def test_manifest_stale_or_unqualified_h15_cannot_certify_track2_family():
    import pytest

    for field, invalid in (
        ("namespace", "source_bytes"),
        ("exact_dataset_binding", False),
        ("canonical_identity_conflict", True),
        ("canonical_sha256", "1" * 64),
    ):
        values = _manifest_test_inputs()
        values["S3_DATASET_CATALOG.json"]["datasets"][0]["family_key"] = None
        row = values["h15-admission-ledger.jsonl"][0]
        if field == "canonical_sha256":
            row["catalog_bindings"][0][field] = invalid
        else:
            row[field] = invalid
        with pytest.raises(ManifestError, match="no eligible metadata contexts"):
            build_inventory(values, {})


def test_manifest_synthesized_cell_never_inherits_native_key_or_numeric_admission():
    import pytest

    manifest, _ = build_inventory(_manifest_test_inputs(), {})
    synthesized = next(
        row
        for row in manifest["records"]
        if row.get("precalc_plan_state") == "unsupported"
    )
    assert synthesized["source_precalc_cell_key"] is None
    assert synthesized["landmark_report_origin"] == "train_synthesized_v3_probes"
    assert all(
        row["numeric_training_admission"] is False for row in manifest["records"]
    )
    with pytest.raises(ManifestError, match="admission remains withheld"):
        require_training_admission(manifest)


def test_manifest_duplicate_and_nonfinite_metadata_are_refused():
    import pytest

    for value in [b'{"a": 1, "a": 2}', b'{"a": NaN}']:
        with pytest.raises(ManifestError):
            _json(value)
    values = _manifest_test_inputs()
    values["track2-eligible-manifest.json"]["cells"].append(
        values["track2-eligible-manifest.json"]["cells"][0]
    )
    values["track2-eligible-manifest.json"]["eligible_cell_count"] += 1
    with pytest.raises(ManifestError, match="duplicate source"):
        build_inventory(values, {})


def test_manifest_source_pin_mismatch_is_refused_before_parsing(tmp_path):
    import pytest

    (tmp_path / "source-pins.json").write_bytes(
        encoded(
            {
                "schema": "jope.hwv06.metadata-pins/v1",
                "sources": {name: "0" * 64 for name in SOURCES},
            }
        )
    )
    (tmp_path / SOURCES[0]).write_bytes(b"not JSON")
    with pytest.raises(ManifestError, match="metadata hash mismatch"):
        load_sources(tmp_path)


def test_manifest_publication_preserves_existing_outputs_and_root(tmp_path):
    import pytest

    target = tmp_path / "output.json"
    write_new(target, {"a": 1}, tmp_path)
    before = target.read_bytes()
    write_new(target, {"a": 1}, tmp_path)
    with pytest.raises(ManifestError, match="existing output differs"):
        write_new(target, {"a": 2}, tmp_path)
    assert target.read_bytes() == before
    with pytest.raises(ManifestError, match="isolated HWV-06 root"):
        write_new(tmp_path.parent / "escape.json", {}, tmp_path)


def test_manifest_real_s3_inventory_overlap_and_replay():
    import pytest

    source = os.environ.get("HWV06_TEST_SOURCES")
    if not source:
        pytest.skip("real metadata copies are supplied by the HWV-06 proof lane")
    values, pins = load_sources(source)
    manifest, receipt = build_inventory(values, pins)
    replay, receipt2 = build_inventory(values, pins)
    assert encoded(manifest) == encoded(replay) and receipt == receipt2
    assert receipt["source_rows"] == {"h15": 2785, "track2": 26679}
    assert (
        receipt["retained_by_source"]["h15"] > 0
        and receipt["retained_by_source"]["track2"] > 0
    )
    assert receipt["retained_alias_components"] > 0
    assert not any(receipt["checked_metadata_component_overlaps"].values())
    assert manifest["numeric_training_admission"] is False
    # Independently compare original source identities, rather than trusting the
    # builder's own overlap counters or its component hashes.
    catalog = values["S3_DATASET_CATALOG.json"]
    corpus = values["corpus-identities.json"]["records"]
    roles = values["historical-role-identities.json"]["roles"]
    protected_contexts = set().union(
        *(set(roles[name]) for name in ("st_val_gate_32", "dl_holdout_91", "dl_val_71"))
    )
    denied = set(catalog["inventory"]["sealed_name_matches_excluded"])
    denied.update(
        row["dataset_id"]
        for row in corpus
        if row["context_sha256"] in protected_contexts
        or row["dl_split"] in ("DL-Val", "DL-Holdout")
    )
    for row in values["h15-admission-ledger.jsonl"]:
        if set(row.get("roles", [])) - TRAIN_ROLES or row.get("h12_reserved"):
            denied.update(
                binding["dataset_hash"] for binding in row["catalog_bindings"]
            )
            protected_contexts.add(row["context_sha256"])
    cells = values["track2-eligible-manifest.json"]["cells"]
    eval_tables = {
        row["physical_table_id"] for row in cells if row["v3_role"] != "train_gradient"
    }
    denied.update(
        row["dataset_id"] for row in cells if row["physical_table_id"] in eval_tables
    )
    admitted = {
        identifier for row in manifest["records"] for identifier in row["dataset_ids"]
    }
    assert admitted.isdisjoint(denied)
    assert retained_h15_contexts(manifest).isdisjoint(protected_contexts)
    admitted_tables = {
        table for row in manifest["records"] for table in row["physical_table_ids"]
    }
    assert admitted_tables.isdisjoint(eval_tables)
    families = {row["id"]: row.get("family_key") for row in catalog["datasets"]}
    denied_families = {families.get(identifier) for identifier in denied} - {None}
    assert {families[identifier] for identifier in admitted}.isdisjoint(denied_families)


def retained_h15_contexts(manifest):
    return {
        row["source_identity"]
        for row in manifest["records"]
        if row["source_kind"] == "h15"
    }


def test_manifest_cli_bypasses_all_legacy_numerical_imports(tmp_path):
    import subprocess
    import pytest

    source = os.environ.get("HWV06_TEST_SOURCES")
    if not source:
        pytest.skip("real metadata copies are supplied by the HWV-06 proof lane")
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import builtins\noriginal = builtins.__import__\ndef guarded(name, *a, **kw):\n    if name.split('.')[0] in {'numpy', 'torch', 'boto3', 'botocore'}:\n        raise RuntimeError('forbidden scientific/network import: ' + name)\n    return original(name, *a, **kw)\nbuiltins.__import__ = guarded\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(tmp_path),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("dataset_builder.py")),
            "qualified-manifest",
            "--sources",
            source,
            "--output",
            str(tmp_path / "packet"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["retained_contexts"] > 0


if __name__ == "__main__":
    raise SystemExit(main())
