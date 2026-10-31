"""Data-free statistics, parameter-only averaging and native process controls."""

import unittest
import numpy as np
from science import ARMS, FOLD_SEEDS, ROUNDS, WEIGHTS, folds, holm, primary, secondary
from model_ops import average, equal, verify_dope


class Controls(unittest.TestCase):
    def fixture(self):
        rows = [
            dict(
                context_sha256=str(i),
                family_id=f"f{i % 205:03}",
                dl_split="DL-Train",
                task="regression",
                source_split="train",
            )
            for i in range(1219)
        ]
        return rows, {"R882": np.zeros(1219), **{a: np.full(1219, 0.02) for a in ARMS}}

    def test_partitions_cover_each_family_once(self):
        rows, _ = self.fixture()
        f = folds(rows)
        self.assertEqual(set(f), set(map(str, FOLD_SEEDS)))
        for m in f.values():
            self.assertEqual([list(m.values()).count(k) for k in range(5)], [41] * 5)
        self.assertNotEqual(f[str(FOLD_SEEDS[0])], f[str(FOLD_SEEDS[1])])
        rows[0]["dl_split"] = "DL-Val"
        with self.assertRaises(ValueError):
            folds(rows)

    def test_positive_fixed_arms_and_deterministic_single_selection(self):
        rows, v = self.fixture()
        r = primary(rows, v)
        self.assertEqual(r["selected_arm"], ARMS[0])
        for a in ARMS:
            self.assertTrue(r["arms"][a]["passed"])
            self.assertEqual(len(set(r["arms"][a]["gain_by_fold_seed"])), 1)
        self.assertIn("not independent", r["interpretation"])

    def test_ties_never_admit(self):
        rows, v = self.fixture()
        for a in ARMS:
            v[a][:] = 0
        r = primary(rows, v)
        self.assertIsNone(r["selected_arm"])
        for a in ARMS:
            self.assertEqual(r["arms"][a]["median_elo"], 1500)
            self.assertEqual(r["arms"][a]["holm"]["p"], 1)
            self.assertIsNone(r["arms"][a]["fold_seed_correlations"][0]["pearson"])

    def test_harm_blocks_large_outlier_gain(self):
        rows, v = self.fixture()
        for a in ARMS:
            v[a][:] = -0.01
            v[a][[r["family_id"] == "f000" for r in rows]] = 10
        r = primary(rows, v)
        self.assertGreater(r["arms"][ARMS[0]]["median_gain"], 0.001)
        self.assertFalse(r["arms"][ARMS[0]]["gates"]["harm_le_10pct"])
        self.assertIsNone(r["selected_arm"])

    def test_primary_is_family_balanced(self):
        rows, v = self.fixture()
        for a in ARMS:
            v[a][:] = 0
            v[a][[r["family_id"] == "f204" for r in rows]] = 1
        r = primary(rows, v)
        self.assertAlmostEqual(r["arms"][ARMS[0]]["median_gain"], 1 / 205)
        self.assertNotAlmostEqual(r["arms"][ARMS[0]]["median_gain"], v[ARMS[0]].mean())

    def test_holm_stepdown_and_monotone_adjustment(self):
        h = holm(dict(a=0.01, b=0.025, c=0.03))
        self.assertTrue(all(x["passed"] for x in h.values()))
        h = holm(dict(a=0.02, b=0.03, c=0.04))
        self.assertFalse(any(x["passed"] for x in h.values()))
        self.assertEqual([h[x]["adjusted_p"] for x in "abc"], [0.06, 0.06, 0.06])

    def test_secondary_is_one_fixed_native_context_mean(self):
        rows = [dict(family_id=f"f{i // 5}") for i in range(91)]
        ref = np.full(91, 1.1377942)
        v = ref + 0.01
        r = secondary(rows, ref, v, ARMS[0])
        self.assertEqual(r["verdict"], "BREAKTHROUGH")
        self.assertAlmostEqual(r["median_gain"], 0.01)
        v[-1] += 1
        r = secondary(rows, ref, v, ARMS[0])
        self.assertAlmostEqual(r["median_gain"], 0.01 + 1 / 91)
        with self.assertRaises(ValueError):
            secondary(rows, ref, v, "POSTHOC")

    def states(self):
        import torch

        return {
            r: {
                "adapter": torch.tensor(float(i)),
                "normalizer": torch.tensor([float(i + 10)]),
                "dope": torch.tensor([0.0]),
            }
            for i, r in enumerate(ROUNDS)
        }

    def test_only_named_parameters_averaged_and_buffers_from_r882(self):
        import torch

        s = self.states()
        for arm, weights in WEIGHTS.items():
            result = average(s, ("adapter",), arm)
            self.assertAlmostEqual(
                float(result["adapter"]),
                sum(i * w for i, w in enumerate(weights)),
                places=6,
            )
            self.assertTrue(equal(result["normalizer"], s[882]["normalizer"]))
            self.assertTrue(equal(result["dope"], s[882]["dope"]))
        self.assertTrue(
            equal(average(s, ("adapter",), "R882")["adapter"], s[882]["adapter"])
        )
        s[782]["adapter"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            average(s, ("adapter",), ARMS[0])

    def test_dope_requires_bitwise_not_numeric_equality(self):
        import torch

        s = self.states()
        verify_dope(s, ["dope"])
        s[782]["dope"] = torch.tensor([-0.0])
        self.assertTrue(torch.equal(s[782]["dope"], s[882]["dope"]))
        with self.assertRaisesRegex(RuntimeError, "bitwise"):
            verify_dope(s, ["dope"])

    def test_free_bytes_not_percentage_and_hard_boundaries(self):
        import monitor

        s = dict(
            rss=0,
            threads=4,
            disk_percent=99,
            free_bytes=40_000_000_000,
            scratch=0,
            load1=1,
            cpus=102,
            available=32 * 1024**3,
        )
        self.assertIsNone(monitor.reason(s))
        for k, v in [
            ("rss", 6_000_000_000),
            ("threads", 5),
            ("free_bytes", 39_999_999_999),
            ("scratch", 2_000_000_000),
        ]:
            self.assertIsNotNone(monitor.hard_reason(dict(s, **{k: v})))

    def test_a4_recovery_requires_continuous_load_and_memory(self):
        import monitor

        s = dict(
            rss=0,
            threads=1,
            disk_percent=90,
            free_bytes=100_000_000_000,
            scratch=0,
            load1=87,
            cpus=102,
            available=32 * 1024**3,
        )
        gate = monitor.RecoveryGate()
        self.assertEqual(gate.action(s, 0), "pause")
        self.assertIsNone(monitor.hard_reason(s))
        self.assertIsNone(gate.action(dict(s, load1=79), 1))
        self.assertIsNone(gate.action(dict(s, load1=79), 60.99))
        self.assertIsNone(gate.action(dict(s, load1=80), 61))
        self.assertIsNone(gate.action(dict(s, load1=79), 62))
        self.assertIsNone(gate.action(dict(s, load1=79, available=15 * 1024**3), 121))
        self.assertIsNone(gate.action(dict(s, load1=79), 122))
        self.assertEqual(gate.action(dict(s, load1=79), 182), "resume")
        self.assertEqual(
            gate.action(dict(s, load1=0, available=15 * 1024**3), 183), "pause"
        )
        self.assertEqual(monitor.hard_reason(dict(s, threads=5)), "threads>4")
        self.assertEqual(monitor.WALL_SECONDS, 7200)

    def test_a4_native_pause_resume_and_hard_abort(self):
        import json
        import subprocess
        import sys
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import monitor

        # Existing temporary-test policy: this holds receipts only, no source/build copies.
        with TemporaryDirectory() as folder:
            root = Path(folder)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                start_new_session=True,
            )
            record = dict(pause_intervals=[])
            original_sample = monitor.sample
            stopped_seen = []
            sample_count = [0]

            def synthetic_pressure(path, proc=None):
                row = original_sample(path, proc)
                sample_count[0] += 1
                row.update(
                    load1=100 if sample_count[0] <= 2 else 0,
                    cpus=102,
                    available=32 * 1024**3,
                    disk_percent=90,
                    free_bytes=100_000_000_000,
                    scratch=0,
                )
                state = (
                    Path(f"/proc/{child.pid}/status")
                    .read_text()
                    .split("State:")[1]
                    .splitlines()[0]
                )
                if "T" in state:
                    stopped_seen.append(sample_count[0])
                if (
                    record["pause_intervals"]
                    and record["pause_intervals"][0].get("disposition") == "resumed"
                ):
                    row["free_bytes"] = 39_000_000_000
                return row

            try:
                with (
                    patch.object(monitor, "sample", synthetic_pressure),
                    patch.object(monitor, "RECOVERY_SECONDS", 0.05),
                    patch.object(monitor, "POLL_SECONDS", 0.01),
                ):
                    with self.assertRaisesRegex(RuntimeError, "free<40GB"):
                        monitor.watch(
                            root, root, record, "fixture", time.monotonic() + 3, child
                        )
                self.assertTrue(stopped_seen)
                self.assertIsNotNone(child.poll())
                self.assertEqual(len(record["pause_intervals"]), 1)
                self.assertEqual(record["pause_intervals"][0]["disposition"], "resumed")
                self.assertGreaterEqual(
                    record["pause_intervals"][0]["duration_seconds"], 0.05
                )
                rows = [
                    json.loads(x)
                    for x in (root / "resources.jsonl").read_text().splitlines()
                ]
                self.assertTrue(any(x["paused"] for x in rows))
            finally:
                monitor.terminate(child)

    def test_a4_deadline_terminates_stopped_worker(self):
        import subprocess
        import sys
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import monitor

        with TemporaryDirectory() as folder:
            root = Path(folder)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                start_new_session=True,
            )
            record = dict(pause_intervals=[])
            original_sample = monitor.sample

            def pressure(path, proc=None):
                row = original_sample(path, proc)
                row.update(
                    load1=100,
                    cpus=102,
                    available=32 * 1024**3,
                    disk_percent=90,
                    free_bytes=100_000_000_000,
                    scratch=0,
                )
                return row

            try:
                with (
                    patch.object(monitor, "sample", pressure),
                    patch.object(monitor, "POLL_SECONDS", 0.01),
                ):
                    with self.assertRaisesRegex(RuntimeError, "wall budget"):
                        monitor.watch(
                            root, root, record, "fixture", time.monotonic() + 0.1, child
                        )
                self.assertIsNotNone(child.poll())
                self.assertEqual(
                    record["pause_intervals"][0]["disposition"], "terminal"
                )
                self.assertIsNotNone(record["pause_intervals"][0]["end_utc"])
            finally:
                monitor.terminate(child)


if __name__ == "__main__":
    unittest.main()
