"""Data-free H153 registration and dose checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from science import dose_check, folds, batch_plan, SEEDS, ARMS
from monitor import classify, sample


def panels(equal=True):
    c = np.zeros((3, 91))
    s = c.copy() if equal else c + 1
    return {"CTRL": c, "SAM": s, "ASAM": c.copy()}


def drifts(sam=1.0):
    return {"CTRL": [1.0] * 3, "SAM": [sam] * 3, "ASAM": [1.2] * 3}


class H153Tests(unittest.TestCase):
    def test_resource_guard_exact_boundaries_are_allowed(self):
        action, reason = classify(
            dict(
                rss=5_999_999_999,
                threads=4,
                free=40_000_000_000,
                disk_used=0.949999,
                scratch=2_999_999_999,
                load=85,
                available=16 * 1024**3,
            )
        )
        self.assertEqual((action, reason), ("run", None))

    def test_resource_guard_aborts_for_disk_used_even_with_free_space(self):
        action, reason = classify(
            dict(
                rss=1_000_000_000,
                threads=2,
                free=100_000_000_000,
                disk_used=0.95,
                scratch=100_000_000,
                load=1,
                available=32 * 1024**3,
            )
        )
        self.assertEqual((action, reason), ("abort", "DISK_USED"))

    def test_resource_guard_aborts_on_host_pressure(self):
        base = dict(
            rss=1_000_000_000,
            threads=2,
            free=100_000_000_000,
            disk_used=0.5,
            scratch=100_000_000,
            load=1,
            available=32 * 1024**3,
        )
        for overrides in ({"load": 85.01}, {"available": 16 * 1024**3 - 1}):
            values = dict(base, **overrides)
            self.assertEqual(classify(values), ("abort", "HOST_PRESSURE"))

    def test_sample_measures_disk_used_fraction(self):
        statvfs = SimpleNamespace(f_blocks=100, f_bfree=5, f_bavail=4, f_frsize=1000)
        process = SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=123), num_threads=lambda: 2
        )
        with (
            patch("monitor.os.statvfs", return_value=statvfs),
            patch("monitor.psutil.Process", return_value=process),
            patch(
                "monitor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=32 * 1024**3),
            ),
            patch("monitor.os.getloadavg", return_value=(1.0, 1.0, 1.0)),
            patch("monitor.Path.rglob", return_value=iter(())),
        ):
            current = sample(".")
        self.assertEqual(current["free"], 4000)
        self.assertEqual(current["disk_used"], 0.95)

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
