"""Validation for the owner-produced JRB-01 binary TRAIN panel packet.

The real packet remains outside the repository. Set
JOPE_JRB01_BINARY_PACKET to its directory when running the data-bound test.
All checks inspect the admitted manifest and train-only NPZ files; they never
open test, holdout, or sealed payloads.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from research.joint_regbin.admission import (
    build_catalog_pool_manifest,
    build_task_role_candidate_manifest,
    build_binary_train_admission,
    review_catalog_families,
    verify_binary_train_admission,
)
from research.joint_regbin import protocol
from research.joint_regbin import driver
from research.joint_regbin.admission import (
    R882_BUNDLE_DIGEST,
    R882_CHECKPOINT_SHA256,
    R882_MESSAGE_SHA256,
    build_role_manifest,
    canonical_sha256,
    make_worker_bound_plan,
    verify_worker_bound_plan,
)


DEFAULT_PACKET = Path.home() / "jope-data/jrb-01/panel-v1-binary-train"
PACKET = Path(os.environ.get("JOPE_JRB01_BINARY_PACKET", DEFAULT_PACKET))
CATALOG_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
PANEL_MANIFEST_SHA256 = (
    "5ebe5b429314a536a3aa9e9cb7783a9b97409835031e7d4fafc4e05a6cd02531"
)


class CatalogPoolManifestTests(unittest.TestCase):
    def _inputs(self):
        rows = [
            {
                "id": "reg-a",
                "task": "regression",
                "canonical_sha256": "a" * 64,
                "family_key": "f" * 64,
                "family_key_source": "provenance.dedupe_fingerprint",
                "family_key_derived": "srcfile:shared-data",
                "source_file": "regression/shared-data.csv",
                "source_url": "https://example.org/data/shared-data.csv",
                "platform": "pmlb",
                "s3_bucket": "veox-jopedime-datasets-use2",
                "train_rows": 12,
                "train_sha256": "1" * 64,
            },
            {
                "id": "bin-a",
                "task": "binary",
                "canonical_sha256": "b" * 64,
                "family_key": "f" * 64,
                "family_key_source": "provenance.dedupe_fingerprint",
                "family_key_derived": "srcfile:shared-data",
                "source_file": "binary/shared-data.csv",
                "source_url": "https://example.org/data/shared-data.csv",
                "platform": "pmlb",
                "s3_bucket": "veox-jopedime-datasets-use2",
                "train_rows": 9,
                "train_sha256": "2" * 64,
            },
            {
                "id": "reg-b",
                "task": "regression",
                "canonical_sha256": "c" * 64,
                "family_key": "f" * 64,
                "family_key_source": "provenance.dedupe_fingerprint",
                "family_key_derived": None,
                "source_file": "regression/other.csv",
                "source_url": "https://example.org/data/other.csv",
                "platform": "pmlb",
                "s3_bucket": "veox-jopedime-datasets-use2",
                "train_rows": 7,
                "train_sha256": "3" * 64,
            },
            {
                "id": "ml-a",
                "task": "multi_label",
                "canonical_sha256": "d" * 64,
                "s3_bucket": "veox-jopedime-datasets-use2",
                "train_rows": 11,
                "train_sha256": "4" * 64,
            },
        ]
        catalog = {
            "inventory": {
                "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
                "catalog_records": 18,
                "usable_records": 4,
                "by_task": {"regression": 2, "binary": 1, "multi_label": 1},
            },
            "datasets": rows,
        }
        catalog_sha = hashlib.sha256(b"catalog metadata").hexdigest()
        owner_alias = {
            "schema": "jope.jrb01.raw-catalog-alias-expansion/v1",
            "policy_schema": "jope.dl-family-keys/v1",
            "policy_sha256": "4cd876f7e215dc17941fde88bb23b95104b8512d75e82305c4e4967e9d609901",
            "policy_source_pins": {
                "deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/common.py": "c419b6d721a32879db830ab7f312e95fec8501c2b47f40581a5054b2317cfbbe",
                "deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/family_keys.py": "9b688e9b8a36a32485524243dc1adea7d30ee7404df4027d2b4d17bd6a900f7e",
            },
            "curated_catalog_sha256": catalog_sha,
            "owner_identity_manifest_sha256": "e" * 64,
            "owner_identities": 167,
            "owner_identities_joined_raw_catalog": 167,
            "previously_missing_identities": 14,
            "previously_missing_now_joined": 14,
            "source_file_metadata_for_previous_missing_ids": 0,
            "binary_direct_owner_alias_intersections": 0,
            "binary_transitive_owner_alias_intersections": 0,
            "all_raw_catalog_transitive_unions_checked": True,
            "no_p3_roster_or_plaintext_sealed_aliases_published": True,
            "payloads_opened": 0,
        }
        owner_alias_sha = hashlib.sha256(b"owner alias evidence").hexdigest()
        return catalog, catalog_sha, owner_alias, owner_alias_sha

    def test_catalog_alias_closure_merges_families_across_tasks(self) -> None:
        catalog, _, _, _ = self._inputs()

        review = review_catalog_families(catalog["datasets"])

        families = review["family_by_dataset"]
        self.assertEqual(families["reg-a"], families["bin-a"])
        self.assertEqual(families["bin-a"], families["reg-b"])
        component = next(
            row for row in review["families"] if "reg-a" in row["datasets"]
        )
        self.assertEqual(set(component["tasks"]), {"regression", "binary"})
        self.assertNotIn("ml-a", families)

    def test_candidate_pools_are_complete_metadata_but_never_admitted(self) -> None:
        catalog, catalog_sha, owner_alias, owner_alias_sha = self._inputs()

        manifest = build_catalog_pool_manifest(
            catalog, catalog_sha, owner_alias, owner_alias_sha
        )

        self.assertEqual(manifest["schema"], "jope.joint-regbin.catalog-pools/v1")
        self.assertEqual(
            manifest["admission_status"], "CATALOG_CANDIDATES_NOT_ADMITTED"
        )
        self.assertIs(manifest["eligibility_verified"], False)
        self.assertEqual(manifest["candidate_counts"], {"binary": 1, "regression": 2})
        self.assertEqual(manifest["candidate_pool_scope"], "catalog_train_rows_only")
        self.assertFalse(manifest["role_assignments_present"])
        self.assertTrue(manifest["closure"]["catalog_aliases_closed"])
        self.assertFalse(manifest["closure"]["owner_aliases_fully_closed"])
        self.assertFalse(
            manifest["closure"]["owner_regression_alias_intersections_reviewed"]
        )
        self.assertEqual(manifest["closure"]["unresolved_owner_sourcefile_aliases"], 14)
        self.assertTrue(
            all(
                row["task"] in {"regression", "binary"}
                for rows in manifest["candidate_pools"].values()
                for row in rows
            )
        )

    def test_candidate_pool_rejects_owner_alias_overlap_or_catalog_drift(self) -> None:
        catalog, catalog_sha, owner_alias, owner_alias_sha = self._inputs()
        owner_alias["binary_transitive_owner_alias_intersections"] = 1
        with self.assertRaisesRegex(ValueError, "owner alias evidence"):
            build_catalog_pool_manifest(
                catalog, catalog_sha, owner_alias, owner_alias_sha
            )

        catalog, catalog_sha, owner_alias, owner_alias_sha = self._inputs()
        catalog["datasets"][0]["train_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "train metadata"):
            build_catalog_pool_manifest(
                catalog, catalog_sha, owner_alias, owner_alias_sha
            )


class TaskRoleCandidateManifestTests(unittest.TestCase):
    def _inputs(self):
        rows = []
        for task, count, prefix in (
            ("regression", 6, "reg"),
            ("binary", 5, "bin"),
        ):
            for index in range(count):
                identity = len(rows) + 1
                rows.append(
                    {
                        "id": f"{prefix}-{index}",
                        "task": task,
                        "canonical_sha256": f"{identity:064x}",
                        "family_key": f"{identity + 100:064x}",
                        "family_key_source": "provenance.dedupe_fingerprint",
                        "family_key_derived": None,
                        "source_file": None,
                        "source_url": None,
                        "platform": "pmlb",
                        "s3_bucket": "veox-jopedime-datasets-use2",
                        "train_rows": 10 + identity,
                        "train_sha256": f"{identity + 200:064x}",
                    }
                )
        catalog_sha = "a" * 64
        catalog = {
            "inventory": {
                "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
                "usable_records": len(rows),
            },
            "datasets": rows,
        }
        owner_alias_sha = "b" * 64
        owner_alias = {
            "schema": "jope.jrb01.raw-catalog-alias-expansion/v1",
            "curated_catalog_sha256": catalog_sha,
            "owner_identity_manifest_sha256": "c" * 64,
            "owner_identities": 167,
            "previously_missing_now_joined": 14,
            "source_file_metadata_for_previous_missing_ids": 0,
            "all_raw_catalog_transitive_unions_checked": True,
            "no_p3_roster_or_plaintext_sealed_aliases_published": True,
            "payloads_opened": 0,
        }
        review = review_catalog_families(rows)
        candidate_pools = {task: [] for task in protocol.TASKS}
        for row in rows:
            family = review["family_by_dataset"][row["id"]]
            candidate_pools[row["task"]].append(
                {
                    "dataset": row["id"],
                    "task": row["task"],
                    "family": family,
                    "family_aliases": [["dedupe_fingerprint", row["family_key"]]],
                    "train_rows": row["train_rows"],
                    "train_sha256": row["train_sha256"],
                }
            )
        pool_sha = "d" * 64
        pool = {
            "schema": "jope.joint-regbin.catalog-pools/v1",
            "catalog_sha256": catalog_sha,
            "owner_alias_expansion_sha256": owner_alias_sha,
            "eligibility_verified": False,
            "qualification_status": "NOT_QUALIFIED",
            "candidate_pools": candidate_pools,
        }

        def roster_row(dataset, role, ordinal):
            return {
                "dataset_id": dataset,
                "task": "regression",
                "family_id": f"legacy-family-{dataset}",
                "view_id": f"view-{role}-{ordinal}",
                "dl_split": "DL-Train" if role == "train" else "DL-Val",
                "source_split": "train" if role == "train" else "inner",
                "context_sha256": f"{ordinal + 300:064x}",
                "pool_sha256": f"{ordinal + 400:064x}",
                "train_only_jtt1_sha256": f"{ordinal + 500:064x}",
            }

        selected = {
            "protocol": "full-DL-Train1219-DL-Val91/v1",
            "train": [roster_row(f"reg-{i}", "train", i) for i in range(5)],
            "dev": [roster_row("reg-5", "dev", 5)],
        }
        selected_sha = "e" * 64
        flywheel_run = {
            "input_hashes": {"selected.json": selected_sha},
            "task_counts": {
                "train_binary": 0,
                "train_regression": 5,
                "val_binary": 0,
                "val_regression": 1,
            },
            "train_contexts": [row["context_sha256"] for row in selected["train"]],
            "dev_contexts": [row["context_sha256"] for row in selected["dev"]],
            "dope_sha256": protocol.DOPE,
            "source_sha": "f" * 40,
        }
        panel = {
            "schema": "jope.elo-panel/v1",
            "stats": {"regression:selected": 1, "binary:selected": 1},
            "datasets": [
                {
                    "id": "reg-0",
                    "task": "regression",
                    "train_rows": rows[0]["train_rows"],
                },
                {
                    "id": "bin-0",
                    "task": "binary",
                    "train_rows": next(
                        r["train_rows"] for r in rows if r["id"] == "bin-0"
                    ),
                },
            ],
        }
        return (
            catalog,
            catalog_sha,
            owner_alias,
            owner_alias_sha,
            pool,
            pool_sha,
            selected,
            selected_sha,
            flywheel_run,
            "1" * 64,
            panel,
            "2" * 64,
        )

    def test_role_candidates_build_task_roles_folds_and_panel_exclusions(self):
        inputs = self._inputs()
        manifest = build_task_role_candidate_manifest(*inputs)

        self.assertEqual(
            manifest["schema"], "jope.joint-regbin.task-role-candidates/v1"
        )
        self.assertEqual(manifest["admission_status"], "ROLE_CANDIDATES_NOT_ADMITTED")
        self.assertFalse(manifest["eligibility_verified"])
        self.assertEqual(
            manifest["role_counts"],
            {
                "train": {"regression": 5, "binary": 5},
                "val": {"regression": 1, "binary": 0},
                "panel": {"regression": 1, "binary": 1},
            },
        )
        self.assertEqual(
            manifest["panel_train_exclusions"]["regression"]["family_count"], 1
        )
        self.assertEqual(
            manifest["panel_train_exclusions"]["binary"]["family_count"], 1
        )
        self.assertEqual(
            manifest["closure"][
                "unresolved_owner_aliases_collapsed_to_one_excluded_family"
            ],
            14,
        )
        self.assertFalse(manifest["closure"]["owner_alias_family_assigned_a_role"])
        fold_by_family = manifest["cv"]["family_by_fold"]
        for rows in manifest["records"].values():
            for row in rows:
                if row["role"] in {"train", "val"}:
                    self.assertEqual(row["cv_fold"], fold_by_family[row["family"]])

    def test_role_candidates_exclude_panel_rows_absent_from_catalog(self):
        inputs = list(self._inputs())
        panel = inputs[-2]
        panel["stats"]["regression:selected"] = 2
        panel["datasets"].append(
            {"id": "missing-reg-panel", "task": "regression", "train_rows": 25}
        )

        manifest = build_task_role_candidate_manifest(*inputs)

        self.assertEqual(
            manifest["closure"]["panel_catalog_missing_by_task"]["regression"], 1
        )
        self.assertEqual(len(manifest["records"]["regression_panel"]), 1)
        self.assertIn(
            "1 regression panel identities are absent",
            " ".join(manifest["blockers"]),
        )

    def test_role_candidates_refuse_flywheel_hash_or_task_drift(self):
        inputs = list(self._inputs())
        inputs[8]["input_hashes"]["selected.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "flywheel train/dev"):
            build_task_role_candidate_manifest(*inputs)

        inputs = self._inputs()
        inputs[8]["train_contexts"][0] = "9" * 64
        with self.assertRaisesRegex(ValueError, "flywheel train/dev"):
            build_task_role_candidate_manifest(*inputs)

        inputs = self._inputs()
        inputs[6]["train"][0]["task"] = "binary"
        with self.assertRaisesRegex(ValueError, "not regression"):
            build_task_role_candidate_manifest(*inputs)


# Catalog v246's explicit sealed-name-match exclusion IDs. This is the
# published catalog exclusion list, not a claim to possess the authoritative
# sealed-family roster.
SEALED_MATCH_EXCLUDED_IDS = frozenset(
    {
        "0a8428ef25a69175",
        "24713c7f7bfa7807",
        "36ef1e99b0f383cb",
        "48bcaee521b95d1b",
        "525fc7ef6e7d6d64",
        "5d21a759a92617cb",
        "6a3edcd68b228f1f",
        "a882d8b478db2ecb",
        "c4b1f6dfb2a8605f",
        "ca307077904135b9",
        "d2cf9a9eddfd7d35",
        "ee74ac8192a9f455",
        "f2fa5e94ae02ab7c",
        "8366db9ae9698880",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packet_manifest(packet: Path = PACKET) -> dict:
    return json.loads((packet / "manifest.json").read_text(encoding="utf-8"))


def rebuilt_npz_sha256(path: Path) -> str:
    """Repack NPY members with the builder's NumPy ZIP64 settings."""
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(path, "r") as source:
        with zipfile.ZipFile(
            rebuilt, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as target:
            for member in source.infolist():
                with source.open(member, "r") as source_member:
                    with target.open(
                        member.filename, "w", force_zip64=True
                    ) as target_member:
                        shutil.copyfileobj(source_member, target_member)
    return hashlib.sha256(rebuilt.getvalue()).hexdigest()


@unittest.skipUnless(PACKET.is_dir(), f"JRB binary packet unavailable: {PACKET}")
class BinaryDatasetPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = PACKET
        cls.manifest = packet_manifest(cls.packet)
        cls.contexts = cls.manifest["contexts"]
        cls.family_review = json.loads(
            (cls.packet / "binary-family-review.json").read_text(encoding="utf-8")
        )

    def test_packet_schema_and_binary_train_only_roles(self) -> None:
        self.assertEqual(self.manifest["schema"], "jope.joint-regbin/v1")
        self.assertEqual(self.manifest["admission_status"], "PARTIAL_PANEL_ONLY")
        self.assertIs(self.manifest["eligibility_verified"], False)
        self.assertEqual(len(self.contexts), self.manifest["binary_panel_datasets"])
        self.assertEqual(
            len({row["dataset"] for row in self.contexts}), len(self.contexts)
        )
        for row in self.contexts:
            with self.subTest(dataset=row["dataset"]):
                self.assertEqual(row["task"], "binary")
                self.assertEqual(row["role"], "panel")
                self.assertEqual(row["source_split"], "train")
                self.assertEqual(row["origin"], "real")
                self.assertIs(row["has_meta_test"], False)
                payload = PurePosixPath(row["payload"])
                self.assertFalse(payload.is_absolute())
                self.assertNotIn("..", payload.parts)
                self.assertEqual(payload.parts[0], "binary")
                self.assertEqual(
                    sha256_file(self.packet.joinpath(*payload.parts)),
                    row["payload_sha256"],
                )

    def test_binary_family_coverage_exceeds_three(self) -> None:
        families = {row["family"] for row in self.contexts}
        self.assertGreater(len(families), 3)
        self.assertEqual(len(families), self.manifest["binary_panel_families"])
        self.assertEqual(
            self.family_review["schema"], "jope.joint-regbin.binary-family-review/v1"
        )
        canonical_review = json.dumps(
            self.family_review, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_review).hexdigest(),
            self.manifest["family_review_sha256"],
        )
        reviewed = {
            row["dataset"]: row["family"] for row in self.family_review["records"]
        }
        packet_families = {reviewed[row["dataset"]] for row in self.contexts}
        self.assertEqual(packet_families, families)

    def test_no_catalog_sealed_match_id_overlap(self) -> None:
        self.assertEqual(self.manifest["catalog_sha256"], CATALOG_SHA256)
        dataset_ids = {row["dataset"] for row in self.contexts}
        overlap = dataset_ids & SEALED_MATCH_EXCLUDED_IDS
        self.assertEqual(overlap, set())

    def test_train_pool_npz_rebuild_is_byte_deterministic(self) -> None:
        for row in self.contexts:
            with self.subTest(dataset=row["dataset"]):
                payload = self.packet.joinpath(*PurePosixPath(row["payload"]).parts)
                self.assertEqual(rebuilt_npz_sha256(payload), row["payload_sha256"])


CANDIDATE_PACKET = Path(
    os.environ.get(
        "JOPE_JRB01_BINARY_CANDIDATE_PACKET",
        Path.home() / "jope-data/jrb-01/binary-s3-train-t2210",
    )
)
CANDIDATE_MANIFEST_SHA256 = (
    "0fa17ee3ddf2a16a84dc90d6a6d19c6467bb1497597b68c86a279e0ac0055a06"
)
SEALED_IDENTITY_MANIFEST_SHA256 = (
    "0314599fe06515d3f3e976c0d435cbe4e9fbe26738014bd89470f525702f983b"
)


@unittest.skipUnless(
    CANDIDATE_PACKET.is_dir(),
    f"JRB binary TRAIN candidate packet unavailable: {CANDIDATE_PACKET}",
)
class BinaryTrainCandidatePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = CANDIDATE_PACKET
        cls.manifest_path = cls.packet / "manifest.json"
        cls.manifest = packet_manifest(cls.packet)
        cls.records = cls.manifest["datasets"]
        cls.exclusion = cls.manifest["exclusion_proof"]
        cls.family_review = json.loads(
            (cls.packet / "binary-family-review.json").read_text(encoding="utf-8")
        )

    def test_candidate_manifest_keeps_train_only_unadmitted_scope(self) -> None:
        self.assertEqual(sha256_file(self.manifest_path), CANDIDATE_MANIFEST_SHA256)
        self.assertEqual(
            self.manifest["schema"], "jope.joint-regbin.binary-s3-train-pool/v1"
        )
        self.assertEqual(self.manifest["status"], "TRAIN_POOL_CANDIDATES_NOT_ADMITTED")
        self.assertIs(self.manifest["eligibility_verified"], False)
        self.assertEqual(self.manifest["catalog_sha256"], CATALOG_SHA256)
        self.assertEqual(
            self.manifest["exclusion_manifest_sha256"],
            SEALED_IDENTITY_MANIFEST_SHA256,
        )
        self.assertEqual(len(self.records), 843)
        self.assertEqual(len({row["dataset"] for row in self.records}), 843)
        self.assertEqual(len({row["family"] for row in self.records}), 807)
        self.assertEqual(self.manifest["counts"]["cached_train_only"], 843)
        self.assertEqual(self.manifest["counts"]["candidate_families"], 807)

        for row in self.records:
            with self.subTest(dataset=row["dataset"]):
                self.assertEqual(row["task"], "binary")
                self.assertEqual(row["source_split"], "train")
                self.assertEqual(row["origin"], "real")
                self.assertIs(row["has_meta_test"], False)
                self.assertIsNone(row["study_role"])
                self.assertIs(row["family_reviewed_from_catalog_metadata"], True)
                self.assertEqual(row["target_classes"], 2)
                self.assertEqual(
                    row["manifest_s3_key"], f"datasets/{row['dataset']}/manifest.json"
                )
                self.assertEqual(
                    row["train_s3_key"],
                    f"blobs/sha256/{row['catalog_train_sha256']}",
                )
                for field in ("cache_file", "train_file"):
                    relpath = PurePosixPath(row[field])
                    self.assertFalse(relpath.is_absolute())
                    self.assertNotIn("..", relpath.parts)

    def test_available_sealed_exclusion_proof_is_explicitly_incomplete(self) -> None:
        self.assertEqual(self.exclusion["actual_excluded_binary_ids"], [])
        self.assertEqual(
            self.exclusion["included_binary_ids_intersection_owner_deny_ids"], []
        )
        self.assertEqual(
            self.exclusion["included_binary_ids_intersection_catalog_sealed_ids"], []
        )
        self.assertIs(
            self.exclusion["identity_and_available_metadata_family_exclusion_verified"],
            True,
        )
        self.assertIs(
            self.exclusion["family_alias_complete_for_absent_catalog_ids"], False
        )
        self.assertIs(self.exclusion["denied_alias_expansion_complete"], False)
        absent_ids = set(
            self.exclusion["owner_identity_ids_absent_from_usable_catalog"]
        )
        self.assertEqual(len(absent_ids), 14)
        self.assertEqual(
            set(self.exclusion["catalog_sealed_name_exclusion_ids"]), absent_ids
        )
        candidate_ids = {row["dataset"] for row in self.records}
        self.assertTrue(candidate_ids.isdisjoint(absent_ids))
        # The packet is deliberately candidate-only until the missing alias
        # closure is resolved; this test does not promote it to study input.
        self.assertIs(self.manifest["eligibility_verified"], False)
        self.assertTrue(all(row["study_role"] is None for row in self.records))

    def test_all_train_and_cache_hashes_and_cache_rebuilds_match(self) -> None:
        self.assertEqual(
            self.family_review["schema"], "jope.joint-regbin.binary-family-review/v1"
        )
        self.assertEqual(
            sha256_file(self.packet / "binary-family-review.json"),
            self.manifest["family_review_sha256"],
        )
        for row in self.records:
            with self.subTest(dataset=row["dataset"]):
                train_path = self.packet.joinpath(
                    *PurePosixPath(row["train_file"]).parts
                )
                cache_path = self.packet.joinpath(
                    *PurePosixPath(row["cache_file"]).parts
                )
                self.assertEqual(sha256_file(train_path), row["train_file_sha256"])
                self.assertEqual(row["catalog_train_sha256"], row["train_file_sha256"])
                self.assertEqual(sha256_file(cache_path), row["cache_sha256"])
                self.assertEqual(rebuilt_npz_sha256(cache_path), row["cache_sha256"])

    def test_conservative_binary_train_admission(self) -> None:
        catalog = Path.home() / "jope_plan/master/source/S3_DATASET_CATALOG.json"
        owner_alias = (
            Path.home()
            / "jope_plan/evidence/JRB-01/obj02-t2240/RAW-CATALOG-ALIAS-EXPANSION.json"
        )
        panel = Path.home() / "jope-data/jrb-01/panel-v1-binary-train/manifest.json"
        if not all(path.is_file() for path in (catalog, owner_alias, panel)):
            self.skipTest("pinned JRB admission evidence is unavailable")

        admission = build_binary_train_admission(
            self.packet, catalog, owner_alias, panel
        )
        verified = verify_binary_train_admission(
            admission, self.packet, catalog, owner_alias, panel
        )

        self.assertEqual(admission, verified)
        self.assertEqual(
            admission["candidate_manifest_sha256"], CANDIDATE_MANIFEST_SHA256
        )
        self.assertEqual(admission["panel_manifest_sha256"], PANEL_MANIFEST_SHA256)
        self.assertEqual(admission["admission_status"], "ADMITTED_BINARY_TRAIN_SUBSET")
        self.assertTrue(admission["binary_train_admitted"])
        self.assertFalse(admission["full_binary_pool_admitted"])
        self.assertFalse(admission["full_joint_eligibility_verified"])
        self.assertEqual(admission["counts"]["candidate_binary_records"], 843)
        self.assertEqual(admission["counts"]["candidate_family_components"], 807)
        self.assertEqual(admission["counts"]["admitted_train_records"], 5)
        self.assertEqual(admission["counts"]["admitted_train_families"], 5)
        self.assertEqual(
            admission["counts"]["srcfile_dependent_candidate_records"], 836
        )
        self.assertEqual(
            admission["counts"]["safe_panel_records_withheld_from_train"], 2
        )
        self.assertEqual({row["role"] for row in admission["records"]}, {"train"})
        self.assertTrue(
            all(
                not any(key[0] == "srcfile" for key in row["family_union_evidence"])
                for row in admission["records"]
            )
        )

    def test_admission_rejects_candidate_manifest_drift(self) -> None:
        catalog = Path.home() / "jope_plan/master/source/S3_DATASET_CATALOG.json"
        owner_alias = (
            Path.home()
            / "jope_plan/evidence/JRB-01/obj02-t2240/RAW-CATALOG-ALIAS-EXPANSION.json"
        )
        panel = Path.home() / "jope-data/jrb-01/panel-v1-binary-train/manifest.json"
        if not all(path.is_file() for path in (catalog, owner_alias, panel)):
            self.skipTest("pinned JRB admission evidence is unavailable")

        with self._temporary_test_directory() as temp_dir:
            altered_packet = Path(temp_dir) / "candidate"
            altered_packet.mkdir()
            for filename in ("manifest.json", "binary-family-review.json"):
                shutil.copyfile(self.packet / filename, altered_packet / filename)
            altered_manifest_path = altered_packet / "manifest.json"
            altered_manifest = json.loads(altered_manifest_path.read_text())
            altered_manifest["eligibility_verified"] = True
            altered_manifest_path.write_text(json.dumps(altered_manifest))

            with self.assertRaisesRegex(ValueError, "pinned v246 TRAIN pool"):
                build_binary_train_admission(
                    altered_packet, catalog, owner_alias, panel
                )

    def test_admission_rejects_panel_catalog_and_role_drift(self) -> None:
        catalog = Path.home() / "jope_plan/master/source/S3_DATASET_CATALOG.json"
        owner_alias = (
            Path.home()
            / "jope_plan/evidence/JRB-01/obj02-t2240/RAW-CATALOG-ALIAS-EXPANSION.json"
        )
        panel = Path.home() / "jope-data/jrb-01/panel-v1-binary-train/manifest.json"
        if not all(path.is_file() for path in (catalog, owner_alias, panel)):
            self.skipTest("pinned JRB admission evidence is unavailable")

        original = json.loads(panel.read_text())
        for drift in ("catalog", "role"):
            with (
                self.subTest(drift=drift),
                self._temporary_test_directory() as temp_dir,
            ):
                altered = json.loads(json.dumps(original))
                if drift == "catalog":
                    altered["catalog_sha256"] = "0" * 64
                else:
                    altered["contexts"][0]["role"] = "train"
                altered_path = Path(temp_dir) / "panel.json"
                altered_path.write_text(json.dumps(altered))
                altered_sha = sha256_file(altered_path)
                with patch(
                    "research.joint_regbin.admission.BINARY_PANEL_MANIFEST_SHA256",
                    altered_sha,
                ):
                    with self.assertRaisesRegex(ValueError, "pinned train-only panel"):
                        build_binary_train_admission(
                            self.packet, catalog, owner_alias, altered_path
                        )

    def _temporary_test_directory(self) -> tempfile.TemporaryDirectory:
        test_root = Path(__file__).resolve().parents[2] / "target"
        test_root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=test_root)

    def test_admission_payload_path_rejects_protected_and_escaping_paths(self) -> None:
        from research.joint_regbin.admission import _packet_file

        for path in ("../outside.csv", "sealed/item.npz", "outer-test/item.npz"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                _packet_file(self.packet, path)


class RoleManifestAdmissionTests(unittest.TestCase):
    def _fixture(self):
        test_root = Path(__file__).resolve().parents[2] / "target"
        test_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        root = Path(temp_dir.name)
        self.addCleanup(temp_dir.cleanup)
        catalog = {
            "inventory": {"source_catalog": "catalog/v246.jsonl (LATEST=v246)"},
            "datasets": [],
        }
        contexts = []
        family_by_dataset = {}
        for task in protocol.TASKS:
            for role in ("train", "val", "panel"):
                for index in range(5 if role == "train" else 1):
                    dataset = f"{task}-{role}-{index}"
                    family = f"family-{task}-{role}-{index}"
                    catalog["datasets"].append(
                        {
                            "id": dataset,
                            "task": task,
                            "s3_bucket": "veox-jopedime-datasets-use2",
                            "family_key": family,
                        }
                    )
                    family_by_dataset[dataset] = family
                    contexts.append(
                        {
                            "id": hashlib.sha256(dataset.encode()).hexdigest(),
                            "dataset": dataset,
                            "family": family,
                            "task": task,
                            "role": role,
                            "source_split": "train",
                            "origin": "real",
                            "has_meta_test": False,
                            "payload": f"train/{dataset}.npz",
                            "payload_sha256": hashlib.sha256(
                                f"payload-{dataset}".encode()
                            ).hexdigest(),
                        }
                    )
        catalog_path = root / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        roles = [
            {key: row[key] for key in sorted(row)}
            for row in sorted(contexts, key=lambda item: item["id"])
        ]
        payloads = [
            {
                "id": row["id"],
                "payload": row["payload"],
                "payload_sha256": row["payload_sha256"],
            }
            for row in sorted(contexts, key=lambda item: item["id"])
        ]
        worker = {
            "sha256": hashlib.sha256(b"qualified-worker").hexdigest(),
            "qualified": True,
            "tasks": list(protocol.TASKS),
            "actual_optimizer_updates": 12,
            "parameters_changed": True,
            "dope_sha256": protocol.DOPE,
        }
        qualification_receipt = {
            "schema": "jope.joint-regbin.worker-qualification/v1",
            "worker_sha256": worker["sha256"],
            "qualified": worker["qualified"],
            "tasks": sorted(worker["tasks"]),
            "actual_optimizer_updates": worker["actual_optimizer_updates"],
            "parameters_changed": worker["parameters_changed"],
            "dope_sha256": worker["dope_sha256"],
        }
        receipt_bytes = json.dumps(
            qualification_receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        worker["qualification_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        worker["qualification_receipt_bytes_b64"] = base64.b64encode(
            receipt_bytes
        ).decode("ascii")
        evidence = {
            "data": {
                "role_assignment_reviewed": True,
                "family_mapping_reviewed": True,
                "payload_hashes_verified": True,
                "sealed_exclusion_verified": True,
                "binary_family_alias_closure_verified": True,
                "role_review_sha256": canonical_sha256(roles),
                "family_review_sha256": canonical_sha256(
                    dict(sorted(family_by_dataset.items()))
                ),
                "payload_audit_sha256": canonical_sha256(payloads),
                "sealed_exclusion_sha256": hashlib.sha256(
                    b"sealed-exclusion-proof"
                ).hexdigest(),
            },
            "release": {
                "label": "r882",
                "verify": "PASS",
                "message_sha256": R882_MESSAGE_SHA256,
                "bundle_digest": R882_BUNDLE_DIGEST,
                "source_checkpoint_sha256": R882_CHECKPOINT_SHA256,
            },
            "worker": worker,
        }
        return catalog_path, contexts, family_by_dataset, evidence

    def test_complete_reviewed_roles_make_equal_budget_r882_bound_plan(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        self.assertIs(manifest["eligibility_verified"], True)
        self.assertEqual(manifest["blockers"], [])
        plan = make_worker_bound_plan(manifest, catalog_path=catalog_path)
        verify_worker_bound_plan(plan, catalog_path=catalog_path)
        self.assertEqual(plan["initial_checkpoint_sha256"], R882_CHECKPOINT_SHA256)
        self.assertEqual(plan["learner_start"]["worker_sha256"], plan["learner_sha256"])
        self.assertEqual(
            plan["learner_start"]["worker_qualification_sha256"],
            evidence["worker"]["qualification_sha256"],
        )
        self.assertEqual(
            plan["role_admission"]["evidence"]["worker"][
                "qualification_receipt_bytes_b64"
            ],
            evidence["worker"]["qualification_receipt_bytes_b64"],
        )
        self.assertEqual(len(plan["jobs"]), 54)
        for seed in protocol.DEFAULTS["seeds"]:
            for fold in (0, 1, 2, 3, 4, "full"):
                jobs = [
                    j for j in plan["jobs"] if j["seed"] == seed and j["fold"] == fold
                ]
                self.assertEqual({j["arm"] for j in jobs}, set(protocol.ARMS))
                self.assertEqual(len({j["id"] for j in jobs}), 3)
                self.assertEqual(
                    {
                        plan["config"][key]
                        for key in ("rounds", "steps", "contexts", "tokens")
                    },
                    {50, 2, 32, 10},
                )

    def test_forged_qualified_worker_arbitrary_digest_is_rejected(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        evidence["worker"]["qualified"] = True
        evidence["worker"]["qualification_sha256"] = hashlib.sha256(
            b"arbitrary qualification digest"
        ).hexdigest()

        manifest = build_role_manifest(catalog_path, contexts, families, evidence)

        self.assertIs(manifest["eligibility_verified"], False)
        self.assertTrue(
            any(
                "receipt hash does not match its bytes" in blocker
                for blocker in manifest["blockers"]
            )
        )
        with self.assertRaisesRegex(ValueError, "incomplete JRB inputs"):
            make_worker_bound_plan(manifest, catalog_path=catalog_path)

    def test_worker_qualification_receipt_must_attest_worker_fields(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        receipt_bytes = base64.b64decode(
            evidence["worker"]["qualification_receipt_bytes_b64"], validate=True
        )
        receipt = json.loads(receipt_bytes)
        receipt["actual_optimizer_updates"] = 11
        receipt_bytes = json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        evidence["worker"]["qualification_sha256"] = hashlib.sha256(
            receipt_bytes
        ).hexdigest()
        evidence["worker"]["qualification_receipt_bytes_b64"] = base64.b64encode(
            receipt_bytes
        ).decode("ascii")

        manifest = build_role_manifest(catalog_path, contexts, families, evidence)

        self.assertIs(manifest["eligibility_verified"], False)
        self.assertTrue(
            any(
                "does not attest the supplied worker evidence" in blocker
                for blocker in manifest["blockers"]
            )
        )

    def test_worker_qualification_requires_receipt_bytes_even_with_valid_digest(
        self,
    ) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        evidence["worker"].pop("qualification_receipt_bytes_b64")

        manifest = build_role_manifest(catalog_path, contexts, families, evidence)

        self.assertIs(manifest["eligibility_verified"], False)
        self.assertIn(
            "worker qualification receipt bytes are missing or invalid",
            manifest["blockers"],
        )
        with self.assertRaisesRegex(ValueError, "incomplete JRB inputs"):
            make_worker_bound_plan(manifest, catalog_path=catalog_path)

    def test_not_qualified_worker_disposition_cannot_enable_complete_roles(
        self,
    ) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        evidence["worker"] = {
            "status": "NOT_QUALIFIED",
            "is_worker_qualification_receipt": False,
            "sha256": None,
            "qualified": False,
            "tasks": list(protocol.TASKS),
            "actual_optimizer_updates": None,
            "parameters_changed": None,
            "dope_sha256": protocol.DOPE,
            "qualification_sha256": None,
            "qualification_receipt_bytes_b64": None,
        }

        manifest = build_role_manifest(catalog_path, contexts, families, evidence)

        self.assertIs(manifest["eligibility_verified"], False)
        for blocker in (
            "worker source hash is missing or invalid",
            "two-task worker qualification is not verified",
            "worker qualification has no positive optimizer updates",
            "worker qualification has no changed-parameter proof",
            "worker qualification receipt bytes are missing or invalid",
            "worker qualification receipt hash is missing or invalid",
        ):
            with self.subTest(blocker=blocker):
                self.assertIn(blocker, manifest["blockers"])
        with self.assertRaisesRegex(ValueError, "incomplete JRB inputs"):
            make_worker_bound_plan(manifest, catalog_path=catalog_path)

    def test_role_manifest_rejects_noncanonical_catalog_bucket(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        catalog = json.loads(catalog_path.read_text())
        catalog["datasets"][0]["s3_bucket"] = "veox-jopedime-learning-use2"
        catalog_path.write_text(json.dumps(catalog))

        with self.assertRaisesRegex(ValueError, "approved real S3 bucket"):
            build_role_manifest(catalog_path, contexts, families, evidence)

    def test_worker_task_summary_requires_exact_supported_list(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        for tasks in [
            None,
            False,
            1,
            "regression",
            {"regression": {}, "binary": {}},
            [],
            ["regression"],
            ["regression", "binary", "binary"],
            ["regression", "regression"],
            ["regression", "multiclass"],
            ["regression", None],
            ["regression", 1],
            [{}, "binary"],
            ["regression", ["binary"]],
        ]:
            with self.subTest(tasks=tasks):
                evidence["worker"]["tasks"] = tasks
                manifest = build_role_manifest(
                    catalog_path, contexts, families, evidence
                )
                self.assertIs(manifest["eligibility_verified"], False)
                self.assertIn(
                    "worker does not prove regression and binary heads",
                    manifest["blockers"],
                )
                with self.assertRaisesRegex(ValueError, "incomplete JRB inputs"):
                    make_worker_bound_plan(manifest, catalog_path=catalog_path)

    def test_worker_task_summary_accepts_both_supported_orders(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        for tasks in [list(protocol.TASKS), list(reversed(protocol.TASKS))]:
            with self.subTest(tasks=tasks):
                evidence["worker"]["tasks"] = tasks
                manifest = build_role_manifest(
                    catalog_path, contexts, families, evidence
                )
                self.assertIs(manifest["eligibility_verified"], True)
                plan = make_worker_bound_plan(manifest, catalog_path=catalog_path)
                self.assertEqual(
                    plan["learner_start"]["worker_qualification_sha256"],
                    evidence["worker"]["qualification_sha256"],
                )

    def test_plan_rederives_proof_instead_of_trusting_cached_eligibility(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        evidence["worker"]["qualified"] = False
        manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        self.assertIs(manifest["eligibility_verified"], False)
        self.assertEqual(manifest["admission_status"], "BLOCKED_INCOMPLETE_EVIDENCE")
        manifest["eligibility_verified"] = True
        manifest["blockers"] = []

        with self.assertRaisesRegex(ValueError, "rederived role admission"):
            make_worker_bound_plan(manifest, catalog_path=catalog_path)

    def test_admission_code_drift_invalidates_frozen_plan(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        plan = make_worker_bound_plan(manifest, catalog_path=catalog_path)
        original_file_sha = protocol.file_sha

        def changed_admission_hash(path):
            if Path(path).name == "admission.py":
                return "0" * 64
            return original_file_sha(path)

        with patch(
            "research.joint_regbin.protocol.file_sha",
            side_effect=changed_admission_hash,
        ):
            with self.assertRaisesRegex(
                ValueError, "study implementation changed after preregistration"
            ):
                verify_worker_bound_plan(plan, catalog_path=catalog_path)

    def test_preflight_rederives_embedded_admission_before_opening_inputs(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        plan = make_worker_bound_plan(manifest, catalog_path=catalog_path)
        plan["role_admission"]["evidence"]["worker"]["qualified"] = False
        plan["sha256"] = protocol.digest(
            {key: value for key, value in plan.items() if key != "sha256"}
        )

        with self.assertRaisesRegex(ValueError, "rederived role admission"):
            driver.preflight(
                plan,
                Path("/missing/packet"),
                Path("/missing/worker"),
                Path("/missing/checkpoint"),
                catalog_path=catalog_path,
            )

    def test_plan_cli_rejects_eligible_flag_without_role_admission(self) -> None:
        catalog_path, contexts, families, evidence = self._fixture()
        manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        for key in (
            "admission_schema",
            "admission_status",
            "evidence",
            "family_by_dataset",
        ):
            manifest.pop(key)
        test_root = Path(__file__).resolve().parents[2] / "target"
        manifest_path = test_root / "jrb01-forged-eligible-manifest.json"
        out_path = test_root / "jrb01-forged-eligible-plan.json"
        manifest_path.write_text(json.dumps(manifest))
        try:
            with (
                patch(
                    "sys.argv",
                    [
                        "driver",
                        "plan",
                        str(manifest_path),
                        "--catalog",
                        str(catalog_path),
                        "--out",
                        str(out_path),
                    ],
                ),
                self.assertRaises(SystemExit) as stopped,
            ):
                driver.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertFalse(out_path.exists())
        finally:
            manifest_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)

    def test_missing_dev_or_worker_proof_never_enables_eligibility(self) -> None:
        catalog, contexts, families, evidence = self._fixture()
        contexts = [
            row
            for row in contexts
            if not (row["task"] == "binary" and row["role"] == "val")
        ]
        evidence["worker"]["qualified"] = False
        evidence["worker"]["actual_optimizer_updates"] = True
        manifest = build_role_manifest(catalog, contexts, families, evidence)
        self.assertIs(manifest["eligibility_verified"], False)
        self.assertTrue(
            any("missing val binary" in item for item in manifest["blockers"])
        )
        self.assertTrue(
            any("worker qualification" in item for item in manifest["blockers"])
        )
        self.assertTrue(
            any("positive optimizer updates" in item for item in manifest["blockers"])
        )
        with self.assertRaisesRegex(ValueError, "cannot produce a runnable plan"):
            make_worker_bound_plan(manifest, catalog_path=catalog)

    def test_family_overlap_and_plan_binding_drift_are_refused(self) -> None:
        catalog, contexts, families, evidence = self._fixture()
        train = next(
            row
            for row in contexts
            if row["task"] == "binary" and row["role"] == "train"
        )
        val = next(
            row for row in contexts if row["task"] == "binary" and row["role"] == "val"
        )
        val["family"] = train["family"]
        families[val["dataset"]] = train["family"]
        roles = [
            {key: row[key] for key in sorted(row)}
            for row in sorted(contexts, key=lambda item: item["id"])
        ]
        evidence["data"]["family_review_sha256"] = canonical_sha256(
            dict(sorted(families.items()))
        )
        evidence["data"]["role_review_sha256"] = canonical_sha256(roles)
        manifest = build_role_manifest(catalog, contexts, families, evidence)
        self.assertIs(manifest["eligibility_verified"], False)
        self.assertTrue(
            any("family occurs in both" in item for item in manifest["blockers"])
        )

        catalog_path, contexts, families, evidence = self._fixture()
        valid_manifest = build_role_manifest(catalog_path, contexts, families, evidence)
        plan = make_worker_bound_plan(valid_manifest, catalog_path=catalog_path)
        plan["learner_start"]["checkpoint_sha256"] = "f" * 64
        plan["sha256"] = protocol.digest(
            {key: value for key, value in plan.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            ValueError, "worker-bound r882 plan identity mismatch"
        ):
            verify_worker_bound_plan(plan, catalog_path=catalog_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
