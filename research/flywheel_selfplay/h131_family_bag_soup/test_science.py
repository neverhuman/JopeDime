"""Data-free controls for H131; no scientific payloads or optimizer execution."""

from collections import Counter
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from science import (
    ARMS,
    SEEDS,
    bootstrap_contexts,
    average_adapters,
    summarize,
    update_plan,
    assert_plan_equal,
)
from monitor import reason


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_bootstrap_preserves_whole_repeated_families(self):
        rows = [
            dict(family_id=str(f), context_sha256=f"{f}-{c}")
            for f in range(12)
            for c in range(f % 3 + 1)
        ]
        out = bootstrap_contexts(rows, 9701)
        counts = Counter(out["contexts"])
        self.assertEqual(len(out["family_draws"]), 12)
        self.assertLess(len(set(out["family_draws"])), 12)
        for row in rows:
            self.assertEqual(
                counts[row["context_sha256"]],
                out["family_multiplicity"].get(row["family_id"], 0),
            )
        self.assertEqual(out, bootstrap_contexts(rows, 9701))
        self.assertNotEqual(
            out["family_draws"], bootstrap_contexts(rows, 9702)["family_draws"]
        )

    def test_bootstrap_ignores_row_order(self):
        rows = [dict(family_id=str(i // 2), context_sha256=str(i)) for i in range(20)]
        self.assertEqual(
            bootstrap_contexts(rows, 9701), bootstrap_contexts(rows[::-1], 9701)
        )

    def test_duplicate_context_roster_refused(self):
        row = dict(family_id="a", context_sha256="b")
        with self.assertRaises(ValueError):
            bootstrap_contexts([row, row], 1)

    def test_soup_is_uniform_parameter_average(self):
        initial = {"a": torch.tensor([10.0, -2.0])}
        members = [{"a": torch.tensor([float(i), -float(i)])} for i in range(1, 9)]
        soup, half = average_adapters(initial, members)
        self.assertTrue(torch.equal(soup["a"], torch.tensor([4.5, -4.5])))
        self.assertTrue(torch.equal(half["a"], torch.tensor([7.25, -3.25])))
        self.assertTrue(torch.equal(initial["a"], torch.tensor([10.0, -2.0])))
        self.assertTrue(torch.equal(members[0]["a"], torch.tensor([1.0, -1.0])))

    def test_no_survivor_soup(self):
        with self.assertRaises(ValueError):
            average_adapters({"a": torch.zeros(1)}, [{"a": torch.zeros(1)}] * 7)

    def test_soup_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            average_adapters(
                {"a": torch.zeros(1)}, [{"a": torch.tensor([float("nan")])}] * 8
            )

    def test_soup_rejects_different_inventory(self):
        with self.assertRaises(ValueError):
            average_adapters({"a": torch.zeros(1)}, [{"a": torch.zeros(2)}] * 8)

    def cells(self, gain):
        ref = dict(
            mean=0.0,
            per_family={str(i): 0.0 for i in range(41)},
            per_context={str(i): 0.0 for i in range(41)},
        )
        rows = []
        for arm in ARMS:
            for seed in SEEDS:
                values = {str(i): gain(arm, seed, i) for i in range(41)}
                rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        assessment=dict(
                            mean=float(np.mean(list(values.values()))),
                            per_family=values,
                            per_context=values,
                        ),
                    )
                )
        return ref, rows

    def test_ties_not_admitted(self):
        ref, rows = self.cells(lambda a, s, f: 0.0)
        for value in summarize(ref, rows)["arms"].values():
            self.assertEqual(value["verdict"], "NO_ADVANCE")
            self.assertEqual(value["median_elo"], 1500.0)
            self.assertFalse(value["holm_pass"])

    def test_strict_margin_rejects_boundary(self):
        ref, rows = self.cells(lambda a, s, f: 0.001)
        self.assertTrue(
            all(
                x["verdict"] == "NO_ADVANCE"
                for x in summarize(ref, rows)["arms"].values()
            )
        )

    def test_positive_constant_admitted(self):
        ref, rows = self.cells(lambda a, s, f: 0.01)
        self.assertTrue(
            all(x["verdict"] == "ADMIT" for x in summarize(ref, rows)["arms"].values())
        )

    def test_every_seed_must_improve(self):
        ref, rows = self.cells(lambda a, s, f: -0.0001 if s == 97 else 0.01)
        self.assertTrue(
            all(
                x["verdict"] == "NO_ADVANCE"
                for x in summarize(ref, rows)["arms"].values()
            )
        )

    def test_family_harm_is_not_context_harm(self):
        ref, rows = self.cells(lambda a, s, f: -0.001 if f < 5 else 0.01)
        for value in summarize(ref, rows)["arms"].values():
            self.assertAlmostEqual(value["family_harm_fraction"], 5 / 41)
            self.assertEqual(value["verdict"], "NO_ADVANCE")

    def test_incomplete_panel_refused(self):
        ref, rows = self.cells(lambda a, s, f: 0.0)
        with self.assertRaises(ValueError):
            summarize(ref, rows[:-1])

    def test_nonfinite_refused(self):
        ref, rows = self.cells(lambda a, s, f: float("nan"))
        with self.assertRaises(ValueError):
            summarize(ref, rows)

    def test_resource_boundaries(self):
        good = dict(
            load1=86.7,
            disk_used_percent=93.999,
            mem_available=30 * 1024**3,
            rss=6 * 1024**3,
            worker_threads=4,
        )
        self.assertIsNone(reason(good))
        for field, value in [
            ("load1", 86.71),
            ("disk_used_percent", 94),
            ("mem_available", 30 * 1024**3 - 1),
            ("rss", 6 * 1024**3 + 1),
            ("worker_threads", 5),
        ]:
            self.assertIsNotNone(reason(dict(good, **{field: value})))

    def test_control_plan_matches_pinned_h125(self):
        root = Path(
            os.environ.get("H131_SOURCE_ROOT", str(Path(__file__).resolve().parents[3]))
        )
        path = root / "research/flywheel_selfplay/h97_reg_ema/runner.py"
        spec = importlib.util.spec_from_file_location("h131_test_h97", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        spec = importlib.util.spec_from_file_location(
            "h131_test_sampler", module.h55.H53B_PATH
        )
        sampler = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = sampler
        spec.loader.exec_module(sampler)
        pools = {
            str(i): dict(fitness=np.arange(256, dtype=float), pids=list(range(256)))
            for i in range(8)
        }
        for seed in SEEDS:
            actual = update_plan(
                pools, sampler.cached_training_indices, seed, module.h59.TrainingBatch
            )
            expected = module.make_update_plan(
                module.batch_plan(pools, sampler.cached_training_indices, seed),
                "CONTROL",
            )
            assert_plan_equal(actual, expected)
        expanded = ["0"] * 30 + ["1"]
        plan = update_plan(
            pools,
            sampler.cached_training_indices,
            9701,
            module.h59.TrainingBatch,
            expanded,
        )
        counts = Counter(c for b in plan for c in b.context_ids)
        self.assertGreater(counts["0"], 20 * counts["1"])
        self.assertEqual(set(counts), {"0", "1"})


class ThreadPreflightTests(unittest.TestCase):
    def fake_builder(self, fail=False):
        from types import SimpleNamespace

        self.calls = []

        def factory(*, num_threads):
            self.calls.append(num_threads)

        base = SimpleNamespace(FusedLandmarkWorkspace=factory)

        def make_tokens(net, tables, device):
            base.FusedLandmarkWorkspace(num_threads=4)
            self.seen = list(tables)
            if fail:
                raise ValueError("token failure")
            return "tokens", "masks"

        return SimpleNamespace(base=base, make_tokens=make_tokens), factory

    def test_tiny_tokens_bound_native_pool_and_restore_factory(self):
        import monitor
        from unittest.mock import patch

        builder, factory = self.fake_builder()
        with patch.object(monitor, "total_threads", side_effect=[2, 4]):
            receipt = monitor.token_thread_preflight(
                builder, None, {"c": 3, "a": 1, "b": 2}, None
            )
        self.assertEqual(self.calls, [1])
        self.assertEqual(self.seen, ["a", "b"])
        self.assertEqual(receipt["after"], 4)
        self.assertIs(builder.base.FusedLandmarkWorkspace, factory)

    def test_five_threads_after_tokens_refused(self):
        import monitor
        from unittest.mock import patch

        builder, factory = self.fake_builder()
        with patch.object(monitor, "total_threads", side_effect=[2, 5]):
            with self.assertRaisesRegex(RuntimeError, "after tokens: 5>4"):
                monitor.token_thread_preflight(builder, None, {"a": 1}, None)
        self.assertIs(builder.base.FusedLandmarkWorkspace, factory)

    def test_five_threads_before_tokens_refused_without_work(self):
        import monitor
        from unittest.mock import patch

        builder, _ = self.fake_builder()
        with patch.object(monitor, "total_threads", return_value=5):
            with self.assertRaisesRegex(RuntimeError, "before tokens: 5>4"):
                monitor.token_thread_preflight(builder, None, {"a": 1}, None)
        self.assertEqual(self.calls, [])

    def test_token_error_restores_original_factory(self):
        import monitor

        builder, factory = self.fake_builder(fail=True)
        with self.assertRaisesRegex(ValueError, "token failure"):
            monitor.single_thread_tokens(builder, None, {"a": 1}, None)
        self.assertIs(builder.base.FusedLandmarkWorkspace, factory)

    def test_proc_status_count_and_missing_field(self):
        import monitor
        from unittest.mock import patch

        with patch.object(
            Path, "read_text", return_value="Name: worker\nThreads:\t4\n"
        ):
            self.assertEqual(monitor.total_threads(), 4)
        with patch.object(Path, "read_text", return_value="Name: worker\n"):
            with self.assertRaises(RuntimeError):
                monitor.total_threads()

    def test_fresh_worker_configures_before_numerical_imports(self):
        import subprocess
        import json

        source = Path(__file__).resolve().parent
        script = "import runner, os, json; print(json.dumps(dict(env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','TOKENIZERS_PARALLELISM']}, intra=runner.torch.get_num_threads(), inter=runner.torch.get_num_interop_threads())))"
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="8",
            NUMEXPR_NUM_THREADS="8",
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=source,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(
            value["env"],
            dict(
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                TOKENIZERS_PARALLELISM="false",
            ),
        )
        self.assertEqual((value["intra"], value["inter"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
