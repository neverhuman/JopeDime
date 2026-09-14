import json
import tempfile
import unittest
from pathlib import Path

import head_cost_census as h


class HeadCostCensusTests(unittest.TestCase):
    def test_program_head_is_last_operator(self):
        self.assertEqual(h.program_head("(pipe X (ridge))"), "ridge")
        self.assertEqual(h.program_head("(pipe X (quantile_bins bins=4) (histgbm2))"), "histgbm2")
        self.assertEqual(h.program_head(""), "unknown")

    def test_empty_roots_are_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = h.census([Path(folder) / "missing"], Path(folder) / "out.json")
            self.assertTrue(receipt["unverified"])
            self.assertEqual(receipt["new_fits"], 0)
            self.assertFalse(receipt["core_seconds_invented"])
            self.assertIsNone(receipt["shares"][0]["core_seconds"] if receipt["shares"] else None)

    def test_scan_counts_heads_and_timeouts_without_core_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "search" / "observations.json"
            path.parent.mkdir()
            path.write_text(json.dumps([
                {"genome": "(pipe X (histgbm2))", "program_sha256": "a" * 64,
                 "failure": None, "runtime_seconds": 1.0},
                {"genome": "(pipe X (histgbm2))", "program_sha256": "b" * 64,
                 "failure": "candidate_or_view_timeout", "runtime_seconds": 30.0},
                {"genome": "(pipe X (gpr))", "program_sha256": "c" * 64,
                 "failure": None, "metrics": {"seconds": 0.5}},
            ]))
            receipt = h.census([root], root / "census.json")
            self.assertFalse(receipt["unverified"])
            self.assertEqual(receipt["observations"], 3)
            by_head = {row["head"]: row for row in receipt["shares"]}
            self.assertEqual(by_head["histgbm2"]["timeouts"], 1)
            self.assertEqual(by_head["histgbm2"]["unique_programs"], 2)
            self.assertIsNone(by_head["histgbm2"]["core_seconds"])
            self.assertEqual(receipt["new_fits"], 0)

    def test_malformed_file_is_unread(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "observations.json").write_text("{bad")
            receipt = h.census([root], root / "census.json")
            self.assertEqual(receipt["unread"], [str(root / "observations.json")])
            self.assertEqual(receipt["observations"], 0)


if __name__ == "__main__":
    unittest.main()
