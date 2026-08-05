from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("migrate_foundation_inventory.py")
SPEC = importlib.util.spec_from_file_location("migrate_foundation_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class FoundationMigrationTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, dict[str, bytes]]:
        legacy = root / "legacy"
        legacy.mkdir()
        dataset_bytes = {
            "synth_0000000000000001": b"npy-one",
            "synth_0000000000000002": b"npy-one",
            "synth_0000000000000003": b"npy-three",
        }
        for dataset_id, payload in dataset_bytes.items():
            (legacy / f"{dataset_id}.npy").write_bytes(payload)
        projection = root / "foundation_projection.npz"
        projection.write_bytes(b"immutable-projection")
        standardization = root / "standardization.json"
        standardization.write_text(
            json.dumps(
                {
                    "dim": 2,
                    "mean": [0.0, 1.0],
                    "scale": [1.0, 2.0],
                    "foundation": {
                        "cache_dir": str(legacy),
                        "projection_file": str(projection),
                        "raw_dim": 3,
                        "proj_dim": 2,
                    },
                }
            )
        )
        return legacy, standardization, root / "published", dataset_bytes

    def test_migration_is_additive_content_addressed_and_zero_diff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            legacy, standardization, destination, expected = self.fixture(Path(raw))
            legacy_before = {
                path.name: (path.read_bytes(), path.stat().st_mode)
                for path in legacy.glob("*.npy")
            }

            receipt = migration.migrate(legacy, standardization, destination)
            replay = migration.migrate(legacy, standardization, destination)

            self.assertEqual(receipt, replay)
            self.assertEqual(receipt["dataset_count"], 3)
            self.assertEqual(receipt["object_count"], 2)
            self.assertTrue(receipt["source_destination_byte_identity"])
            for dataset_id, payload in expected.items():
                source = legacy / f"{dataset_id}.npy"
                projected = destination / "datasets" / f"{dataset_id}.npy"
                self.assertEqual(source.read_bytes(), payload)
                self.assertEqual(projected.read_bytes(), payload)
                self.assertEqual(
                    legacy_before[source.name], (source.read_bytes(), source.stat().st_mode)
                )
            inventory = json.loads(Path(receipt["inventory"]).read_text())
            rewritten = json.loads(Path(receipt["standardization"]).read_text())
            self.assertEqual(inventory["schema"], migration.INVENTORY_SCHEMA)
            self.assertEqual(
                rewritten["foundation"]["cache_dir"],
                str(destination / "datasets"),
            )
            self.assertEqual(
                rewritten["foundation"]["projection_sha256"],
                receipt["projection_sha256"],
            )

    def test_dataset_identity_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            legacy, standardization, destination, _ = self.fixture(Path(raw))
            migration.migrate(legacy, standardization, destination)
            changed = legacy / "synth_0000000000000001.npy"
            changed.write_bytes(b"different-identity")
            with self.assertRaisesRegex(
                migration.FoundationMigrationError, "collision"
            ):
                migration.migrate(legacy, standardization, destination)

    def test_matching_bytes_must_still_link_to_the_hashed_object(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            legacy, standardization, destination, _ = self.fixture(Path(raw))
            receipt = migration.migrate(legacy, standardization, destination)
            inventory = json.loads(Path(receipt["inventory"]).read_text())
            row = inventory["entries"][0]
            projected = destination / "datasets" / f"{row['dataset_id']}.npy"
            payload = projected.read_bytes()
            projected.unlink()
            projected.write_bytes(payload)
            with self.assertRaisesRegex(
                migration.FoundationMigrationError, "not linked"
            ):
                migration.migrate(legacy, standardization, destination)


if __name__ == "__main__":
    unittest.main()
