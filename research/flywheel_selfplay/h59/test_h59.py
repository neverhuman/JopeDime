#!/usr/bin/env python3
"""CPU-only, synthetic tests for H59 loss, coordinator lock/disk guards, and red-team checks."""
from __future__ import annotations

from collections import namedtuple
import fcntl
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h59_runner_under_test", HERE / "runner.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import H59 runner")
h59 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h59
spec.loader.exec_module(h59)

wave_spec = importlib.util.spec_from_file_location("h59_run_wave_under_test", HERE / "run_wave.py")
if wave_spec is None or wave_spec.loader is None:
    raise RuntimeError("cannot import H59 run_wave")
run_wave = importlib.util.module_from_spec(wave_spec)
sys.modules[wave_spec.name] = run_wave
wave_spec.loader.exec_module(run_wave)

FakeStatvfs = namedtuple("FakeStatvfs", ["f_blocks", "f_bavail", "f_frsize"])


class H59LossTests(unittest.TestCase):
    def test_target_transform_matches_preregistered_utility_clip(self):
        fitness = torch.tensor([[-10.0, -0.5, 1.0, 10.0]])
        gates = torch.tensor([[True, True, True, False]])
        self.assertTrue(torch.equal(h59.preregistered_targets(fitness, gates), torch.tensor([[-4.0, -0.5, 1.0, 2.0]])))

    def test_clip_caps_pointwise_smooth_l1_before_reduction(self):
        pred = torch.tensor([[0.0, 10.0]], requires_grad=True)
        target = torch.zeros_like(pred)
        control = h59.loss_by_arm("CONTROL", pred, target)
        clipped = h59.loss_by_arm("CLIP", pred, target)
        self.assertLess(clipped, control)
        self.assertTrue(torch.isfinite(clipped))

    def test_scale_is_detached_and_computed_per_training_slate(self):
        targets = torch.tensor([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]], requires_grad=True)
        scale = h59.training_slate_scale(targets)
        self.assertFalse(scale.requires_grad)
        self.assertAlmostEqual(float(scale[1]), h59.SCALE_FLOOR, places=7)
        pred = torch.ones_like(targets, requires_grad=True)
        h59.loss_by_arm("NORM", pred, targets).backward()
        self.assertIsNotNone(pred.grad)

    def test_all_four_arms_produce_finite_scalar_losses(self):
        pred = torch.tensor([[0.1, 0.2, 1.0], [0.0, -0.2, 0.3]], requires_grad=True)
        target = torch.tensor([[0.0, 0.5, 2.0], [-0.3, 0.2, 0.1]])
        for arm in h59.ARMS:
            loss = h59.loss_by_arm(arm, pred, target)
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))


class H59ProtocolTests(unittest.TestCase):
    def test_all_arms_replay_the_same_batches_and_slate_indices(self):
        train_ids = ("c1", "c2", "c3")
        pools = {cid: {"size": 256} for cid in train_ids}
        orders = {cid: np.arange(256) for cid in train_ids}

        def cached(pool, order, size):
            return np.random.choice(order, size=size, replace=False)

        plan = h59.build_paired_batches(train_ids, pools, orders, cached, seed=12)
        plans = h59.paired_plans_for_arms(plan)
        self.assertEqual(len(plan), 40)
        self.assertTrue(all(plans[arm] is plan for arm in h59.ARMS))
        repeated = h59.build_paired_batches(train_ids, pools, orders, cached, seed=12)
        for first, second in zip(plan, repeated, strict=True):
            self.assertEqual(first.context_ids, second.context_ids)
            self.assertEqual(first.seed, second.seed)
            for first_indices, second_indices in zip(first.candidate_indices, second.candidate_indices, strict=True):
                np.testing.assert_array_equal(first_indices, second_indices)
        self.assertNotEqual(plan[0].seed, plan[1].seed)

    def test_dope_pin_and_adapter_only_inventory_are_fail_closed(self):
        class Parameter:
            requires_grad = True

            def numel(self):
                return h59.TRAINABLE_PARAMS

        names = h59.validate_trainable_inventory([("adapter_10tok.head.weight", Parameter())])
        self.assertEqual(names, ("adapter_10tok.head.weight",))
        with self.assertRaisesRegex(RuntimeError, "only non-anchor"):
            h59.validate_trainable_inventory([("dope.encoder.weight", Parameter())])
        h59.validate_dope_hash(h59.DOPE_SHA)
        with self.assertRaisesRegex(RuntimeError, "DOPE SHA mismatch"):
            h59.validate_dope_hash("0" * 64)

        class WrongParameter(Parameter):
            def numel(self):
                return 1

        with self.assertRaisesRegex(RuntimeError, "expected 29211"):
            h59.validate_trainable_inventory([("adapter_10tok.head.weight", WrongParameter())])

    def test_fold_b_terminal_read_requires_admission_and_is_single_use(self):
        with self.assertRaisesRegex(RuntimeError, "unless the arm is admitted"):
            h59.terminal_read_allowed("CLIP", 12, (), ())
        h59.terminal_read_allowed("CLIP", 12, ("CLIP",), ())
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            h59.terminal_read_allowed("CLIP", 12, ("CLIP",), (("CLIP", 12),))
        self.assertEqual(h59.plan_terminal_reads(()), ())
        self.assertEqual(len(h59.plan_terminal_reads(iter(("NORM",)))), 3)

    def test_strict_lcb_ties_fail_and_family_bootstrap_reports_cis(self):
        ties = {seed: {f"f{i}": 0.0 for i in range(5)} for seed in h59.SEEDS}
        result = h59.paired_family_admission(ties)
        self.assertEqual(result["mean_lcb95_one_sided"], 0.0)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["bootstrap_draws"], 10_000)
        self.assertEqual(set(result["per_seed"]), {"12", "13", "14"})
        self.assertTrue(all(row["family_mean_gain"] == 0.0 for row in result["per_seed"].values()))
        self.assertEqual(len(result["family_median_ci95"]), 2)
        self.assertTrue(all(len(row["family_median_ci95"]) == 2 for row in result["per_seed"].values()))
        self.assertFalse(result["admitted"] if result["mean_lcb95_one_sided"] == h59.ADMISSION_LCB else False)

    def test_admission_requires_strict_lcb_above_margin(self):
        gains = {seed: {f"f{i}": h59.ADMISSION_LCB for i in range(8)} for seed in h59.SEEDS}
        result = h59.paired_family_admission(gains)
        self.assertEqual(result["mean_lcb95_one_sided"], h59.ADMISSION_LCB)
        self.assertFalse(result["admitted"])

    def test_each_arm_gets_an_independent_pooled_admission_decision_and_3seed_elo(self):
        scores = {
            seed: {
                "arms": {
                    arm: {
                        "family_gains": {
                            f"f{i}": (-0.02 if arm == "CLIP" and i < 3 else 0.05)
                            for i in range(20)
                        },
                        "elo_vs_r882": {"ratings": {"r882": 1500.0, "H59": 1512.0 + seed}},
                    }
                    for arm in h59.ARMS
                }
            }
            for seed in h59.SEEDS
        }
        decisions = h59.admit_arms(scores)
        self.assertEqual(set(decisions), set(h59.ARMS))
        self.assertTrue(decisions["CONTROL"]["admitted"])
        self.assertFalse(decisions["CLIP"]["admitted"])
        self.assertAlmostEqual(decisions["CONTROL"]["elo_median_across_seeds"], 1525.0, places=5)
        self.assertEqual(len(decisions["CONTROL"]["elo_ci95_across_seeds"]), 2)

    def test_h61_null_margin_diagnostic_is_recorded_without_changing_preregistered_gate(self):
        gains = {seed: {f"f{i}": 0.0012 for i in range(12)} for seed in h59.SEEDS}
        res = h59.paired_family_admission(gains)
        self.assertTrue(res["admitted"])
        self.assertEqual(res["h61_null_margin_m"], 0.001546)
        self.assertFalse(res["clears_h61_null_margin_diagnostic"])

    def test_elo_is_family_grouped_and_anchored_to_r882(self):
        families = {"c1": "f1", "c2": "f1", "c3": "f2", "c4": "f2"}
        baseline = {"c1": 0.1, "c2": 0.3, "c3": 0.0, "c4": 0.2}
        candidate = {"c1": 0.2, "c2": 0.4, "c3": 0.1, "c4": 0.3}
        result = h59.elo_vs_r882(families, baseline, candidate)
        self.assertAlmostEqual(result["ratings"]["r882"], 1500.0)
        self.assertGreater(result["ratings"]["H59"], 1500.0)


class H59CoordinatorTests(unittest.TestCase):
    def test_launch_disk_guard_94_5_and_writer_disk_guard_95_0(self):
        stat_945 = lambda _: FakeStatvfs(f_blocks=1000, f_bavail=55, f_frsize=1024**3)
        stat_944 = lambda _: FakeStatvfs(f_blocks=1000, f_bavail=56, f_frsize=1024**3)
        stat_950 = lambda _: FakeStatvfs(f_blocks=1000, f_bavail=50, f_frsize=1024**3)

        with mock.patch.object(run_wave.os, "statvfs", side_effect=stat_945):
            self.assertIsNotNone(run_wave.launch_disk_guard_error())
            self.assertIsNone(run_wave.disk_guard_error())
        with mock.patch.object(run_wave.os, "statvfs", side_effect=stat_944):
            self.assertIsNone(run_wave.launch_disk_guard_error())
        with mock.patch.object(run_wave.os, "statvfs", side_effect=stat_950):
            self.assertIsNotNone(run_wave.disk_guard_error())
        with self.assertRaisesRegex(RuntimeError, "unsafe filesystem / before H59 write"):
            h59.check_writer_disk_budget(statvfs_fn=stat_950)

    def test_write_time_disk_guard_retries_same_cell_up_to_2_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "outcomes/cells/CONTROL/seed12"
            status_path = tmp_path / "H59_STATUS.log"
            log_path = tmp_path / "logs/CONTROL-seed12.log"
            attempts = []

            def fake_run_locked_cell(key, command, log, timeout, status, wait_total):
                attempts.append((key, log.name))
                out_dir.mkdir(parents=True, exist_ok=True)
                if len(attempts) <= 2:
                    (out_dir / "partial.tmp").write_text("partial")
                    raise run_wave.DiskGuardInfraVoid("unsafe filesystem / before H59 write", 1.5, 10.0)
                (out_dir / "summary.json").write_text('{"status":"complete"}')
                return {"key": key, "status": "complete"}, 2.0, 120.0

            with mock.patch.object(run_wave, "run_locked_cell", side_effect=fake_run_locked_cell):
                cell, total_wait, wall, void_wall = run_wave.run_cell_with_disk_guard_retries(
                    "CONTROL/seed12",
                    ["python3"],
                    log_path,
                    timeout=600.0,
                    status_path=status_path,
                    wait_total=0.0,
                    output_path=out_dir,
                )
            self.assertEqual(len(attempts), 3)
            self.assertEqual(cell["disk_guard_retry_count"], 2)
            self.assertAlmostEqual(void_wall, 20.0)
            self.assertAlmostEqual(total_wait, 5.0)
            self.assertAlmostEqual(wall, 120.0)

    def test_under_lock_preflight_failure_releases_lock_before_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_lock = Path(tmpdir) / "qualification.lock"
            status_path = Path(tmpdir) / "H59_STATUS.log"
            call_count = {"n": 0}

            def fake_safe_preflight():
                call_count["n"] += 1
                if call_count["n"] <= 2:
                    raise RuntimeError("MemAvailable below 16 GiB: 1000 KiB")
                return {"host_cpu_percent": 10.0, "mem_available_kib": 32 * 1024 * 1024}

            def verify_lock_released(_seconds):
                with tmp_lock.open("a+") as probe:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe, fcntl.LOCK_UN)

            with (
                mock.patch.object(run_wave, "LOCK", tmp_lock),
                mock.patch.object(run_wave, "launch_disk_guard_error", return_value=None),
                mock.patch.object(run_wave, "gpu_free_mib", return_value=24000),
                mock.patch.object(run_wave, "safe_preflight", side_effect=fake_safe_preflight),
                mock.patch.object(run_wave.time, "sleep", side_effect=verify_lock_released),
            ):
                handle, _waited, free_mib, _status = run_wave.acquire_lock(status_path, 0.0)
                try:
                    self.assertEqual(free_mib, 24000)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()

    def test_stage_r882_copy_refuses_live_loop_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            live_path = Path(tmpdir) / "flywheel-full-20261030" / "round-882.pt"
            live_path.parent.mkdir(parents=True)
            live_path.write_bytes(b"live")
            with self.assertRaisesRegex(RuntimeError, "live xbabe1"):
                h59.stage_r882_copy(live_path, Path(tmpdir) / "scratch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
