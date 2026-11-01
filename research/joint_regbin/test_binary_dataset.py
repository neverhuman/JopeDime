"""Validation for the owner-produced JRB-01 binary TRAIN panel packet.

The real packet remains outside the repository. Set
JOPE_JRB01_BINARY_PACKET to its directory when running the data-bound test.
All checks inspect the admitted manifest and train-only NPZ files; they never
open test, holdout, or sealed payloads.
"""

from __future__ import annotations

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
    build_binary_train_admission,
    verify_binary_train_admission,
)


DEFAULT_PACKET = Path.home() / "jope-data/jrb-01/panel-v1-binary-train"
PACKET = Path(os.environ.get("JOPE_JRB01_BINARY_PACKET", DEFAULT_PACKET))
CATALOG_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
PANEL_MANIFEST_SHA256 = (
    "5ebe5b429314a536a3aa9e9cb7783a9b97409835031e7d4fafc4e05a6cd02531"
)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
