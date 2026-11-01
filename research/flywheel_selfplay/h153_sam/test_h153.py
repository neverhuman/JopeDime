"""Data-free H153 registration and dose checks."""

import unittest
import numpy as np
from science import dose_check, folds, batch_plan, SEEDS, ARMS


def panels(equal=True):
    c = np.zeros((3, 91))
    s = c.copy() if equal else c + 1
    return {"CTRL": c, "SAM": s, "ASAM": c.copy()}


def drifts(sam=1.0):
    return {"CTRL": [1.0] * 3, "SAM": [sam] * 3, "ASAM": [1.2] * 3}


class H153Tests(unittest.TestCase):
    def test_registered_inventory(self):
        self.assertEqual(SEEDS, (153101, 153102, 153103))
        self.assertEqual(ARMS, ("CTRL", "SAM", "ASAM"))

    def test_identical_panel_and_within_band_is_invalid(self):
        self.assertEqual(
            dose_check(drifts(), panels())["SAM"]["status"], "DOSE_INVALID"
        )

    def test_identical_panel_outside_band_is_valid(self):
        self.assertEqual(
            dose_check(drifts(0.94), panels())["SAM"]["status"], "DOSE_VALID"
        )

    def test_changed_panel_within_band_is_not_invalid(self):
        self.assertEqual(
            dose_check(drifts(), panels(False))["SAM"]["status"], "DOSE_VALID"
        )

    def test_incomplete_panel_refuses(self):
        p = panels()
        p["SAM"] = np.zeros((2, 91))
        with self.assertRaises(ValueError):
            dose_check(drifts(), p)

    def test_fold_assignment_uses_all_families_once(self):
        rows = [
            dict(
                context_sha256=f"c{i}",
                family_id=f"f{i}",
                dl_split="DL-Train",
                source_split="train",
            )
            for i in range(19)
        ]
        groups = folds(rows)
        self.assertEqual([len(group) for group in groups], [4, 4, 4, 4, 3])
        self.assertEqual(sorted(sum(groups, [])), sorted(f"f{i}" for i in range(19)))

    def test_batch_plan_has_40_updates(self):
        ids = [f"c{i}" for i in range(5)]
        orders = {c: np.arange(200) for c in ids}
        self.assertEqual(len(batch_plan(ids, orders, SEEDS[0])), 40)


if __name__ == "__main__":
    unittest.main()
