import copy
from pathlib import Path
import tempfile
import unittest

import q1_cut


def fixture():
    return {
        "schema": "jope.q1-cut/v1",
        "dataset_sha256": "a" * 64,
        "train_blob_sha256": "b" * 64,
        "train_rows": 4,
        "target": "y",
        "feature_columns": ["x1", "x2"],
        "folds": [
            {"repeat": 0, "train": [0, 1], "validation": [2, 3]},
            {"repeat": 0, "train": [2, 3], "validation": [0, 1]},
        ],
    }


class Q1CutTests(unittest.TestCase):
    def test_canonical_control_and_field_sensitivity(self):
        source = fixture()
        expected = "0c87e715ccb1a01fe05d4bc4e8b81f62f978d8aed0126248d97b866686acf249"
        digest = q1_cut.cut_sha256(source)
        reordered = dict(reversed(list(source.items())))
        self.assertEqual(q1_cut.cut_sha256(reordered), digest)
        self.assertEqual(digest, expected)
        for mutate in (
            lambda value: value.update(target="z"),
            lambda value: value["feature_columns"].reverse(),
            lambda value: value["folds"].reverse(),
            lambda value: value["folds"][0]["train"].reverse(),
        ):
            changed = copy.deepcopy(source)
            mutate(changed)
            self.assertNotEqual(q1_cut.cut_sha256(changed), digest)

    def test_rejects_invalid_fold_and_unknown_fields(self):
        value = fixture()
        value["folds"][0]["train"] = [0, 0]
        with self.assertRaises(ValueError):
            q1_cut.cut_sha256(value)
        value = fixture()
        value["cut_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            q1_cut.cut_sha256(value)

    def test_content_addressed_cut_binds_definition_and_roster_identity(self):
        value = fixture()
        digest = q1_cut.cut_sha256(value)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / f"{digest}.json"
            path.write_bytes(q1_cut.canonical_bytes(value))
            self.assertEqual(
                q1_cut.load_content_addressed(
                    path, digest, expected_dataset_sha256=value["dataset_sha256"]
                ),
                value,
            )
            with self.assertRaisesRegex(ValueError, "roster entry"):
                q1_cut.load_content_addressed(
                    path, digest, expected_dataset_sha256="c" * 64
                )

    def test_content_addressed_cut_rejects_tampering_and_symlinks(self):
        value = fixture()
        digest = q1_cut.cut_sha256(value)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / f"{digest}.json"
            path.write_bytes(q1_cut.canonical_bytes(value))
            changed = copy.deepcopy(value)
            changed["folds"][0]["validation"].reverse()
            path.write_bytes(q1_cut.canonical_bytes(changed))
            with self.assertRaisesRegex(ValueError, "content address"):
                q1_cut.load_content_addressed(path, digest)
            path.unlink()
            target = root / "target.json"
            target.write_bytes(q1_cut.canonical_bytes(value))
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                q1_cut.load_content_addressed(path, digest)

    def test_require_three_fold_fit_partition_enforces_disjoint_complement(self):
        valid = {
            "schema": q1_cut.SCHEMA,
            "dataset_sha256": "a" * 64,
            "train_blob_sha256": "b" * 64,
            "train_rows": 6,
            "target": "y",
            "feature_columns": ["x1", "x2"],
            "folds": [
                {"repeat": 0, "train": [2, 3, 4, 5], "validation": [0, 1]},
                {"repeat": 0, "train": [0, 1, 4, 5], "validation": [2, 3]},
                {"repeat": 0, "train": [0, 1, 2, 3], "validation": [4, 5]},
            ],
        }
        self.assertEqual(q1_cut.require_three_fold_fit_partition(valid), valid)
        duplicate_folds = copy.deepcopy(valid)
        duplicate_folds["folds"][1] = copy.deepcopy(valid["folds"][0])
        with self.assertRaisesRegex(ValueError, "disjoint three-way FIT partition"):
            q1_cut.require_three_fold_fit_partition(duplicate_folds)
        incomplete_complement = copy.deepcopy(valid)
        incomplete_complement["folds"][0]["train"] = [2, 3, 4]
        with self.assertRaisesRegex(ValueError, "ascending validation complement"):
            q1_cut.require_three_fold_fit_partition(incomplete_complement)


if __name__ == "__main__":
    unittest.main()
