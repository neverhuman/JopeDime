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
import unittest
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_PACKET = Path.home() / "jope-data/jrb-01/panel-v1-binary-train"
PACKET = Path(os.environ.get("JOPE_JRB01_BINARY_PACKET", DEFAULT_PACKET))
CATALOG_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
