import copy
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


if __name__ == "__main__":
    unittest.main()
