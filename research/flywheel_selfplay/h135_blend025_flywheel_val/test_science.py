"""Data-free native-estimand and fail-closed controls."""

import unittest
import numpy as np
from science import native, sanity, summarize, ARMS
from monitor import reason


class Controls(unittest.TestCase):
    def test_native_does_not_clip_or_use_top_four(self):
        self.assertEqual(native([3, 2, 1], [-9, -8, -4], [True] * 3), (4.0, 2))
        self.assertEqual(
            native([0, 0, 1], [-9, -8, -4], [False, True, True]), (-2.0, 0)
        )

    def test_invalid_cost_refused(self):
        with self.assertRaises(ValueError):
            native([np.nan], [1], [True])

    def test_reference_gate_checks_every_context(self):
        ref = dict(best_utility=1.1377942, utility=dict(per_context=[1.1377942] * 91))
        self.assertEqual(sanity(np.full(91, 1.1377942), ref)["status"], "PASS")
        vals = np.full(91, 1.1377942)
        vals[:2] += [0.1, -0.1]
        with self.assertRaises(RuntimeError):
            sanity(vals, ref)

    def fixture(self):
        rows = [
            dict(context_sha256=f"{i:03}", family_id=f"f{i // 5}") for i in range(91)
        ]
        ref = np.zeros(91)
        values = np.stack([np.full(91, 0.01 + i * 0.001) for i in range(3)])
        return rows, ref, {a: values.copy() for a in ARMS}

    def test_primary_only_admits(self):
        rows, ref, arms = self.fixture()
        m = summarize(rows, ref, arms)
        self.assertEqual(m["verdict"], "CANDIDATE_REVIEW_ONLY")
        self.assertFalse(m["arms"]["BLEND_050"]["candidate"])
        self.assertAlmostEqual(m["arms"]["BLEND_025"]["median_gain"], 0.011)

    def test_identical_seeds_fail(self):
        rows, ref, arms = self.fixture()
        arms["BLEND_025"][:] = arms["BLEND_025"][0]
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertFalse(m["candidate"])
        self.assertFalse(m["gates"]["seed_independence"])

    def test_context_weighted_bootstrap(self):
        rows, ref, arms = self.fixture()
        # Last family has one context, so unequal family averaging is wrong.
        for a in ARMS:
            arms[a][:, -1] += 1
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertAlmostEqual(m["median_gain"], 0.011 + 1 / 91)
        self.assertLess(m["median_gain"], 0.011 + 1 / 19)

    def test_harm_and_trim_prevent_outlier_admission(self):
        rows, ref, arms = self.fixture()
        for a in ARMS:
            arms[a][:] = -0.02
            arms[a][:, :10] = 2
            arms[a] += np.arange(3)[:, None] * 0.0001
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertGreater(m["median_gain"], 0.001)
        self.assertFalse(m["gates"]["trim_gt_zero"])
        self.assertFalse(m["gates"]["harm_le_10pct"])

    def test_imported_fit_excludes_validation(self):
        from unittest.mock import patch
        import runner

        _, h = runner.imported()
        from pathlib import Path

        train = dict(context_sha256="train")
        val = dict(context_sha256="val")
        context = {
            k: dict(lm48_z=np.zeros(48), dope629=np.zeros(629))
            for k in ("train", "val")
        }
        pools = {k: dict(u=np.arange(200, dtype=float)) for k in ("train", "val")}
        captured = {}

        class Model:
            n_iter_ = 1

            def __init__(self, **params):
                captured["params"] = params

            def fit(self, x, y):
                captured["x"] = x.copy()
                captured["y"] = y.copy()
                return self

            def predict(self, x):
                captured["target"] = x.copy()
                return np.zeros(len(x))

            def get_params(self):
                return captured["params"]

        def program(pools, cid):
            return np.full((200, 412), 10 if cid == "train" else 99, dtype=np.float32)

        with (
            patch("sklearn.ensemble.HistGradientBoostingRegressor", Model),
            patch.object(h, "headroom"),
            patch.object(h, "check_rss"),
            patch.object(h, "release_memory"),
            patch.object(h, "program", program),
            patch.object(h, "sha", return_value="synthetic"),
            patch.object(h, "write"),
            patch.object(h.np, "savez_compressed"),
            patch.object(Path, "exists", return_value=False),
        ):
            h.fit_one(
                (135, (4,)),
                Path("/synthetic-no-io"),
                [train, val],
                context,
                pools,
                {"train": np.arange(4, 196)},
                {"train": 0, "val": 4},
            )
        self.assertEqual(captured["x"].shape, (192, 1089))
        self.assertTrue((captured["x"][:, :412] == 10).all())
        self.assertTrue((captured["target"][:, :412] == 99).all())
        self.assertEqual(captured["params"], dict(h.MODEL, random_state=135))
        np.testing.assert_array_equal(
            captured["y"], np.arange(4, 196) - np.arange(200).mean()
        )

    def test_native_histgb_total_thread_cap(self):
        import json
        import os
        import subprocess
        import sys
        from monitor import ARROW_ALLOCATOR_CONF

        code = """import json,psutil,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
before=psutil.Process().num_threads()
x=np.random.default_rng(0).normal(size=(500,10))
with threadpool_limits(limits=1,user_api='blas'),threadpool_limits(limits=4,user_api='openmp'):
    model=HistGradientBoostingRegressor(max_iter=3,random_state=135).fit(x,x[:,0])
    print(json.dumps(dict(before=before,after=psutil.Process().num_threads())))
"""
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            NUMEXPR_MAX_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            JE_ARROW_MALLOC_CONF=ARROW_ALLOCATOR_CONF,
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        counts = json.loads(result.stdout)
        self.assertLessEqual(counts["before"], 4)
        self.assertLessEqual(counts["after"], 4)

    def test_a3_environment_precedes_interpreter(self):
        from launcher import worker_environment

        env = worker_environment({"OMP_NUM_THREADS": "64", "RAYON_NUM_THREADS": "64"})
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "RAYON_NUM_THREADS",
        ):
            self.assertEqual(env[name], "1")
        self.assertEqual(env["JE_ARROW_MALLOC_CONF"], "background_thread:false")

    def test_a3_nested_limit_and_exception_restoration(self):
        import threadpoolctl
        from launcher import one_thread

        original = threadpoolctl.threadpool_limits

        def probe():
            with threadpoolctl.threadpool_limits(limits=4, user_api="openmp"):
                self.assertTrue(
                    all(p["num_threads"] == 1 for p in threadpoolctl.threadpool_info())
                )
            raise RuntimeError("synthetic exception")

        with self.assertRaisesRegex(RuntimeError, "synthetic exception"):
            one_thread(probe)
        self.assertIs(threadpoolctl.threadpool_limits, original)

    def test_baseline_thread_cap_restores_setter(self):
        from unittest.mock import patch, Mock
        import torch
        import runner

        calls = []

        def setter(n):
            calls.append(n)

        def imported_baseline(*args):
            torch.set_num_threads(4)
            return ("costs", "net")

        helper = Mock(baseline=imported_baseline)
        with (
            patch.object(torch, "set_num_threads", setter),
            patch.object(torch, "set_num_interop_threads") as inter,
        ):
            self.assertEqual(
                runner.baseline_one_thread(helper, "root"), ("costs", "net")
            )
            self.assertIs(torch.set_num_threads, setter)
            inter.assert_called_once_with(1)
        self.assertEqual(calls, [1, 1])

    def test_baseline_thread_cap_restores_after_error(self):
        from unittest.mock import patch, Mock
        import torch
        import runner

        setter = Mock()
        helper = Mock(baseline=Mock(side_effect=ValueError("fixture")))
        with (
            patch.object(torch, "set_num_threads", setter),
            patch.object(torch, "set_num_interop_threads"),
        ):
            with self.assertRaises(ValueError):
                runner.baseline_one_thread(helper, "root")
            self.assertIs(torch.set_num_threads, setter)

    def test_a4_recovery_requires_continuous_load_and_memory(self):
        import monitor

        s = dict(
            rss=0,
            threads=1,
            disk_percent=90,
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
        self.assertEqual(monitor.WALL_SECONDS, 14400)

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
                    row["disk_percent"] = 94
                return row

            try:
                with (
                    patch.object(monitor, "sample", synthetic_pressure),
                    patch.object(monitor, "RECOVERY_SECONDS", 0.05),
                    patch.object(monitor, "POLL_SECONDS", 0.01),
                ):
                    with self.assertRaisesRegex(RuntimeError, "disk>=93.8%"):
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

    def test_guard_boundaries(self):
        s = dict(
            rss=0,
            threads=4,
            disk_percent=90,
            scratch=0,
            load1=1,
            cpus=10,
            available=32 * 1024**3,
        )
        self.assertIsNone(reason(s))
        for key, v in [
            ("rss", 8_000_000_000),
            ("threads", 5),
            ("disk_percent", 93.8),
            ("scratch", 2_000_000_000),
        ]:
            self.assertIsNotNone(reason(dict(s, **{key: v})))


if __name__ == "__main__":
    unittest.main()
