"""Data-free controls for family isolation, weighting, ties and bootstrap Elo."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import json

import numpy as np

if "h145_statistics" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "h145_statistics", Path(__file__).with_name("analysis.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
import h145_statistics as stat

if "h145_runtime" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "h145_runtime", Path(__file__).with_name("run.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
runtime = sys.modules["h145_runtime"]


class H145Controls(unittest.TestCase):
    def test_all_family_members_move_together_and_assessment_is_once(self):
        rows = [
            {"context_sha256": f"context-{i}", "family_id": f"family-{i // 3}"}
            for i in range(30)
        ]
        folds = stat.family_folds(rows)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertFalse(
                set(fold["fit_families"]) & set(fold["assessment_families"])
            )
            self.assertFalse(
                set(fold["fit_contexts"]) & set(fold["assessment_contexts"])
            )
        self.assertEqual(
            sorted(cid for fold in folds for cid in fold["assessment_contexts"]),
            sorted(row["context_sha256"] for row in rows),
        )
        reversed_folds = stat.family_folds(list(reversed(rows)))
        self.assertEqual(
            [f["assessment_families"] for f in folds],
            [f["assessment_families"] for f in reversed_folds],
        )

    def test_duplicate_context_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stat.family_folds(
                [{"context_sha256": "same", "family_id": str(i)} for i in range(6)]
            )

    def test_equal_family_weighting_not_equal_context_weighting(self):
        order, matrix = stat.family_matrix(
            np.array([[0.0], [0.0], [6.0]]), ["a", "a", "b"]
        )
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(float(matrix.mean()), 3.0)

    def test_nonfinite_panel_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            stat.family_matrix(np.array([[np.nan]]), ["a"])

    def test_tied_functions_have_exact_anchored_elo(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_equal(
            stat.bt_ratings(stat.family_wins(values, ["a", "b"]).sum(0), 0),
            [1500.0, 1500.0],
        )

    def test_dlval_subtree_is_skipped_without_json_decoding(self):
        source = json.dumps({"native_val": {"secret": "FORBIDDEN"}, "train": [1, 2]})
        loads = json.loads

        def admitted_loads(value, *args, **kwargs):
            if "FORBIDDEN" in value:
                raise AssertionError("forbidden subtree decoded")
            return loads(value, *args, **kwargs)

        with patch.object(runtime.json, "loads", side_effect=admitted_loads):
            self.assertEqual(json.loads(runtime.raw_field(source, "train")), [1, 2])

    def test_lexical_projection_handles_escaped_braces_and_missing_keys(self):
        source = json.dumps({"skip": {"quoted": '"}]'}, "keep": {"train": [3]}})
        self.assertEqual(json.loads(runtime.raw_field(source, "keep")), {"train": [3]})
        with self.assertRaises(KeyError):
            runtime.raw_field(source, "absent")

    def test_vectorized_mm_matches_canonical_with_repeated_family_multiplicity(self):
        if "h145_canonical_elo" not in sys.modules:
            path = (
                Path(__file__).resolve().parents[2]
                / "python/src/jope/north_star/elo_tracker.py"
            )
            spec = importlib.util.spec_from_file_location("h145_canonical_elo", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        elo = sys.modules["h145_canonical_elo"]
        values = np.array(
            [[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]
        )
        families = ["a", "a", "b", "c"]
        wins = stat.family_wins(values, families)
        for sample in ([0, 1, 2], [0, 0, 2]):
            records = []
            for occurrence, family_index in enumerate(sample):
                for i, family in enumerate(families):
                    if family == ["a", "b", "c"][family_index]:
                        records.append(
                            elo.DatasetEvalRecord(
                                dataset_id=f"{occurrence}-{i}",
                                family_id=f"draw-{occurrence}",
                                task_type="regression",
                                scores_by_contender={
                                    str(j): float(v) for j, v in enumerate(values[i])
                                },
                            )
                        )
            expected = elo.compute_family_grouped_bradley_terry_elo(
                records, anchor_contender="0", prior=0.01
            )["ratings"]
            actual = stat.bt_ratings(wins[sample].sum(0), 0)
            self.assertEqual(
                [round(float(v), 2) for v in actual],
                [expected[str(j)] for j in range(3)],
            )
        batch = stat.bt_ratings(np.stack([wins.sum(0), wins[[0, 0, 2]].sum(0)]), 0)
        np.testing.assert_allclose(
            batch[0], stat.bt_ratings(wins.sum(0), 0), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
