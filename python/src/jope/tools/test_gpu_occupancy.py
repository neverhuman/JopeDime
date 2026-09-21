import unittest

from gpu_occupancy import update


class OccupancyTests(unittest.TestCase):
    def test_malformed_samples_cannot_inflate_diagnostic_occupancy(self):
        for ranks in (True, -1, 9, 8.5, float("nan")):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                update({}, 0, ranks, True)
        with self.assertRaises(ValueError):
            update({}, float("nan"), 8, True)
        with self.assertRaises(ValueError):
            update({}, 0, 8, True, 101)

    def test_idle_resident_workers_and_monitor_rank_are_counted_correctly(self):
        h = {}
        update(h, 0, 8, True, 50)
        update(h, 10, 0, True, 0, "packing")
        update(h, 20, 1, True, 10)
        result = update(h, 30, 8, True, 80)
        self.assertEqual(result["observed_gpu_task_seconds"], 90)
        self.assertAlmostEqual(result["task_occupancy"], 90 / 240)
        self.assertEqual(result["handoff_gap_p95_seconds"], 10)
        self.assertEqual(result["kernel_utilization_mean"], 20)
        self.assertEqual(result["acceptance"], "inconclusive")

    def test_restart_gap_is_unknown_and_cannot_inflate_occupancy(self):
        h = {}
        update(h, 0, 8, True, 100)
        result = update(h, 3600, 8, True, 100)
        self.assertEqual(result["unobserved_seconds"], 3600)
        self.assertIsNone(result["task_occupancy"])
        self.assertEqual(result["sample_window_acceptance"], "fail")
        self.assertEqual(result["acceptance"], "inconclusive")

    def test_full_hour_uses_time_weighted_occupancy(self):
        h = {}
        for t in range(0, 3601, 10):
            result = update(h, t, 0 if t == 10 else 8, True, 40)
        self.assertEqual(result["sample_window_acceptance"], "pass")
        self.assertEqual(result["acceptance"], "inconclusive")
        self.assertAlmostEqual(result["task_occupancy"], 3590 / 3600)
        self.assertAlmostEqual(result["full_hour_occupancy"], 3590 / 3600)
        self.assertEqual(result["handoff_gap_p95_seconds"], 10)

    def test_eight_percent_of_hour_cannot_pass(self):
        h = {}
        for t in range(0, 289, 10):
            result = update(h, t, 8, True, 100)
        for t in range(290, 3601, 10):
            result = update(h, t, 0, False, 0)
        self.assertLess(result["full_hour_occupancy"], 0.10)
        self.assertNotEqual(result["acceptance"], "pass")
