#!/usr/bin/env python3
"""CPU-only unit, red-team, identity, FIFO lock, and admission-boundary tests for H77 long ES."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h77_runner_test", HERE / "runner.py")
assert SPEC is not None and SPEC.loader is not None
h77 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h77
SPEC.loader.exec_module(h77)

METRICS_SPEC = importlib.util.spec_from_file_location("h77_metrics_test", HERE / "build_metrics.py")
assert METRICS_SPEC is not None and METRICS_SPEC.loader is not None
metrics = importlib.util.module_from_spec(METRICS_SPEC)
sys.modules[METRICS_SPEC.name] = metrics
METRICS_SPEC.loader.exec_module(metrics)

AUDIT_SPEC = importlib.util.spec_from_file_location("h77_audit_test", HERE / "audit.py")
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit
AUDIT_SPEC.loader.exec_module(audit)

WAVE_SPEC = importlib.util.spec_from_file_location("h77_wave_test", HERE / "run_wave.py")
assert WAVE_SPEC is not None and WAVE_SPEC.loader is not None
wave = importlib.util.module_from_spec(WAVE_SPEC)
sys.modules[WAVE_SPEC.name] = wave
WAVE_SPEC.loader.exec_module(wave)

fifo_mod = wave.fifo_lock


class H77LongESTests(unittest.TestCase):
    def test_preregistered_constants_match_t2335_contract(self):
        self.assertEqual(h77.SIGMA, 0.001)
        self.assertEqual(h77.SEEDS, (18, 19, 20))
        self.assertEqual(h77.DEFAULT_GENERATIONS, 24)
        self.assertEqual(h77.PAIRS, 32)
        self.assertEqual(h77.POPULATION, 64)
        self.assertEqual(h77.ES_LEARNING_RATE, 0.01)
        self.assertEqual(h77.ESH_LAMBDA, 1.0)
        self.assertEqual(h77.ARMS, ("ES", "ESH"))
        self.assertEqual(h77.NULL_ARM, "null_s001")
        self.assertEqual(h77.TRAINABLE_PARAMS, 29_211)
        self.assertEqual(wave.MAX_WAVE_GPU_WALL_SECONDS, 9000.0)
        self.assertEqual(metrics.DEFAULT_OUTPUT, HERE / "metrics.json")
        self.assertTrue((HERE / "PREREG.md").is_file())

    def test_centered_rank_order_and_centering_pop64(self):
        ranks = h77.centered_ranks(np.arange(64, dtype=np.float64))
        self.assertEqual(ranks.shape, (64,))
        self.assertAlmostEqual(float(ranks.sum()), 0.0, places=12)
        self.assertTrue(np.all(np.diff(ranks) > 0))

    def test_antithetic_population_is_32_pairs_pop64_and_shared_across_arms(self):
        center = torch.zeros(h77.TRAINABLE_PARAMS)
        seed_es = h77.stream_seed(18, 1, arm="ES", null_arm=False)
        seed_esh = h77.stream_seed(18, 1, arm="ESH", null_arm=False)
        seed_null = h77.stream_seed(18, 1, arm="ES", null_arm=True)
        self.assertEqual(seed_es, seed_esh)
        self.assertEqual(seed_es, seed_null)

        noises, pop = h77.antithetic_population(center, 0.001, seed_es)
        noises_esh, pop_esh = h77.antithetic_population(center, 0.001, seed_esh)
        self.assertEqual(len(noises), 32)
        self.assertEqual(len(pop), 64)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(noises, noises_esh, strict=True)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pop, pop_esh, strict=True)))
        for pos, neg in zip(pop[::2], pop[1::2], strict=True):
            self.assertTrue(torch.equal(pos, -neg))

        with self.assertRaisesRegex(ValueError, "sigma=0.001"):
            h77.antithetic_population(center, 0.003, seed_es)

    def test_es_vs_esh_harm_penalty_fitness_math(self):
        baseline = {"f1": 1.0, "f2": 1.0, "f3": 1.0, "f4": 1.0}
        # Case 1: No family harmed below -0.01 (gains are +0.05, +0.02, -0.005, -0.009)
        cand_no_harm = {"f1": 1.05, "f2": 1.02, "f3": 0.995, "f4": 0.991}
        res_es = h77.compute_fold_a_fitness(cand_no_harm, baseline, "ES")
        res_esh = h77.compute_fold_a_fitness(cand_no_harm, baseline, "ESH")
        self.assertAlmostEqual(res_es["fitness"], res_esh["fitness"], places=12)
        self.assertAlmostEqual(res_esh["harm_term_min0_gain_plus_001"], 0.0, places=12)
        self.assertAlmostEqual(res_esh["harm_penalty_subtracted"], 0.0, places=12)
        self.assertEqual(res_esh["fold_a_harm_fraction"], 0.0)

        # Case 2: Family f4 has gain = -0.11 (< -0.01), so min(0, gain_f + 0.01) = -0.10.
        # Across 4 equal-weight families, mean(min(0, gain_f + 0.01)) = -0.025.
        # ESH must penalize this harmed candidate by 0.025 (ESH = ES - 0.025 < ES).
        cand_harm = {"f1": 1.05, "f2": 1.02, "f3": 0.995, "f4": 0.89}
        res_es_harm = h77.compute_fold_a_fitness(cand_harm, baseline, "ES")
        res_esh_harm = h77.compute_fold_a_fitness(cand_harm, baseline, "ESH")
        self.assertAlmostEqual(res_esh_harm["harm_term_min0_gain_plus_001"], -0.025, places=12)
        self.assertAlmostEqual(res_esh_harm["harm_penalty_subtracted"], 0.025, places=12)
        self.assertAlmostEqual(res_esh_harm["fitness"], res_es_harm["fitness"] - 0.025, places=12)
        self.assertLess(res_esh_harm["fitness"], res_es_harm["fitness"])
        self.assertAlmostEqual(res_esh_harm["fold_a_harm_fraction"], 0.25, places=12)

    def test_centered_rank_es_gradient_is_finite_and_deterministic_pop64(self):
        noises = [torch.full((5,), float(i + 1)) for i in range(32)]
        ranks = h77.centered_ranks(np.arange(64, dtype=np.float64))
        first = h77.es_gradient(noises, ranks, 0.001)
        second = h77.es_gradient(noises, ranks, 0.001)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(float(torch.linalg.vector_norm(first)), 0.0)

    def test_only_preregistered_null_and_timing_cells_are_accepted(self):
        # Null is only allowed on ES seed 18 outcome
        bad_null = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ESH", seed=18, generations=24, null_arm=True, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "only the ES seed18 outcome"):
            h77.run_train(bad_null)

        bad_null_seed = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ES", seed=19, generations=24, null_arm=True, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "only the ES seed18 outcome"):
            h77.run_train(bad_null_seed)

        # Timing is only allowed on ES seed 18 with 1 generation
        bad_timing = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ESH", seed=18, generations=1, null_arm=False, mode="timing"
        )
        with self.assertRaisesRegex(RuntimeError, "only ES seed18 one-generation timing"):
            h77.run_train(bad_timing)

        # Outcome requires 24 generations
        bad_horizon = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ES", seed=18, generations=8, null_arm=False, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "fixed at 24 generations"):
            h77.run_train(bad_horizon)

    def test_protected_and_live_paths_and_trainable_inventory_guards(self):
        for bad in (
            "/tmp/sealed/data.npz",
            "/tmp/holdout/row.json",
            "/tmp/meta-test/x.npz",
            "/tmp/outer/fold.json",
            "/tmp/flywheel-full-20261030/round-882.pt",
        ):
            with self.assertRaisesRegex(RuntimeError, "rejected protected or live-loop path"):
                h77.safe_path(Path(bad))

        class FakeParam:
            def __init__(self, count: int, grad: bool = True):
                self.requires_grad = grad
                self._count = count

            def numel(self):
                return self._count

        names = h77.validate_trainable_inventory(
            [
                ("adapter_10tok.layer.weight", FakeParam(h77.TRAINABLE_PARAMS, True)),
                ("dope.backbone.weight", FakeParam(100_000, False)),
            ]
        )
        self.assertEqual(names, ("adapter_10tok.layer.weight",))
        with self.assertRaisesRegex(RuntimeError, "non-adapter or anchor"):
            h77.validate_trainable_inventory([("dope.backbone.weight", FakeParam(h77.TRAINABLE_PARAMS, True))])

    def test_timing_projection_checks_2_5_hour_cap(self):
        proj_ok = h77.project_wave_wall_from_timing(35.0, cap_seconds=9000.0)
        self.assertTrue(proj_ok["fits_within_cap"])
        self.assertAlmostEqual(proj_ok["projected_total_seconds"], 35.0 * 170.0, places=4)

        proj_over = h77.project_wave_wall_from_timing(60.0, cap_seconds=9000.0)
        self.assertFalse(proj_over["fits_within_cap"])
        self.assertGreater(proj_over["projected_total_seconds"], 9000.0)

    def test_pooled_admission_gates_and_elo_per_seed_and_median(self):
        # 1. Exact ties fail
        tie_rows = [
            {"arm": "ES", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.0 for i in range(20)}}
            for seed in (18, 19, 20)
        ]
        tie_res = metrics.pooled_admission(tie_rows, "ES")
        self.assertFalse(tie_res["admitted"])
        self.assertTrue(tie_res["is_tie"])
        self.assertEqual(tie_res["elo_vs_r882_median"], 1500.0)
        self.assertEqual(tie_res["elo_delta_vs_r882_median"], 0.0)

        # 2. Boundary LCB == 0.001 fails strict > 0.001 gate
        boundary_rows = [
            {"arm": "ES", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.001 for i in range(20)}}
            for seed in (18, 19, 20)
        ]
        boundary_res = metrics.pooled_admission(boundary_rows, "ES")
        self.assertEqual(boundary_res["mean_lcb95_one_sided"], 0.001)
        self.assertFalse(boundary_res["admitted"])

        # 3. Harm > 10% fails even with positive mean LCB95
        harm_gains = {f"f{i}": 0.05 for i in range(20)}
        harm_gains["f0"] = -0.02
        harm_gains["f1"] = -0.02
        harm_gains["f2"] = -0.02  # 3/20 = 15% > 10%
        harm_rows = [{"arm": "ES", "seed": seed, "family_gain_vs_r882": harm_gains} for seed in (18, 19, 20)]
        harm_res = metrics.pooled_admission(harm_rows, "ES")
        self.assertGreater(harm_res["mean_lcb95_one_sided"], 0.001)
        self.assertGreater(harm_res["harm_fraction_below_minus_0_01"], 0.10)
        self.assertFalse(harm_res["admitted"])

        # 4. Positive LCB95 > 0.001, median >= 0, harm <= 10% admits and reports per-seed + median Elo
        good_rows = [
            {"arm": "ESH", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.02 for i in range(20)}}
            for seed in (18, 19, 20)
        ]
        good_res = metrics.pooled_admission(good_rows, "ESH")
        self.assertTrue(good_res["admitted"])
        self.assertGreater(good_res["mean_lcb95_one_sided"], 0.001)
        self.assertEqual(good_res["harm_fraction_below_minus_0_01"], 0.0)
        self.assertEqual(set(good_res["elo_vs_r882_per_seed"].keys()), {"18", "19", "20"})
        self.assertGreater(good_res["elo_vs_r882_median"], 1500.0)
        self.assertGreater(good_res["elo_delta_vs_r882_median"], 0.0)

    def test_independent_admission_audit_catches_tampering(self):
        candidates = []
        for arm, gain in (("ES", 0.0), ("ESH", 0.02)):
            for seed in (18, 19, 20):
                candidates.append(
                    {"arm": arm, "seed": seed, "family_gain_vs_r882": {f"f{i}": gain for i in range(20)}}
                )
        computed = {arm: metrics.pooled_admission(candidates, arm) for arm in ("ES", "ESH")}
        fold_b = {"fold_assignment_sha256": "fold-hash-77", "candidates": candidates}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outcomes").mkdir()
            (root / "outcomes/admission.json").write_text(
                json.dumps(
                    {
                        "schema": "jope.h77.admission/v1",
                        "fold_seed": 4900,
                        "fold_assignment_sha256": "fold-hash-77",
                        "arms": computed,
                        "terminal_dl_val_reads_allowed": 3,
                    }
                )
            )
            audited = audit.audit_admission(root, fold_b)
            self.assertFalse(audited["ES"]["admitted"])
            self.assertTrue(audited["ESH"]["admitted"])

            tampered = json.loads((root / "outcomes/admission.json").read_text())
            tampered["arms"]["ES"]["admitted"] = True
            (root / "outcomes/admission.json").write_text(json.dumps(tampered))
            with self.assertRaisesRegex(SystemExit, "disagrees for ES"):
                audit.audit_admission(root, fold_b)

    def test_fifo_queue_places_h77_behind_h76_d2_w1_and_enforces_no_immediate_retake(self):
        """Verify H77 queues behind existing H76/D2/W1 tickets, polls at 30s, and backs off >=45s on legacy waiters."""
        self.assertEqual(wave.LOCK_POLL_SECONDS, 30.0)
        self.assertEqual(wave.LEGACY_BACKOFF_SECONDS, 45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lock_file = tmp / "qualification.lock"
            queue_dir = tmp / "qualification.queue"
            data_root = tmp / "jope-data"
            status_log = data_root / "h77/h77-wave/H77_STATUS.log"
            h76_log = data_root / "h76/h76-wave/H76_STATUS.log"
            h76_log.parent.mkdir(parents=True, exist_ok=True)

            # Enqueue H76 ticket at epoch_ns=100 ahead of H77 at epoch_ns=200
            t_h76 = fifo_mod.enqueue_ticket(queue_dir, pid=2001, wave="H76", epoch_ns=100)
            fixed_now = datetime(2026, 10, 31, 5, 35, 0, tzinfo=timezone.utc)
            h76_log.write_text(
                f"{(fixed_now - timedelta(seconds=15)).isoformat()} — WAIT qualification.lock busy; cumulative_wait=30.0s\n"
            )

            sleeps: list[float] = []

            def fake_sleep(sec: float) -> None:
                sleeps.append(sec)
                # After legacy backoff (45s) + 1st FIFO poll (30s), H76 finishes and removes its ticket
                if len(sleeps) >= 2 and t_h76.exists():
                    fifo_mod.remove_ticket(t_h76)

            with mock.patch.object(wave, "launch_disk_guard_error", return_value=None):
                handle, _, free_mib, _ = wave.acquire_lock(
                    status_log,
                    0.0,
                    after_release=True,
                    wave_name="H77",
                    pid=3001,
                    epoch_ns=200,
                    lock_path=lock_file,
                    queue_dir=queue_dir,
                    data_root=data_root,
                    flock_fn=lambda h, flags: None,
                    sleep_fn=fake_sleep,
                    gpu_free_fn=lambda: 24_000,
                    preflight_fn=lambda: {
                        "disk_used_percent": 90.0,
                        "host_cpu_percent": 20.0,
                        "mem_available_kib": 64 * 1024 * 1024,
                    },
                    is_alive_fn=lambda pid: pid in {2001, 3001},
                    now_utc_fn=lambda: fixed_now,
                )
                handle.close()

            self.assertEqual(free_mib, 24_000)
            self.assertEqual(sleeps, [45.0, 30.0])
            self.assertIn("FIFO queue ahead", status_log.read_text())
            self.assertEqual(list(queue_dir.iterdir()), [])

    def test_xbabe2_disk_abort_above_95_percent(self):
        stats_95 = SimpleNamespace(f_blocks=100 * 1024**3, f_bavail=5 * 1024**3, f_frsize=1)
        with self.assertRaisesRegex(RuntimeError, "95% abort threshold"):
            h77.check_disk_abort(statvfs_fn=lambda _: stats_95)
        with mock.patch.object(wave.os, "statvfs", return_value=stats_95):
            self.assertRegex(wave.disk_guard_error(), "95% full")

    def test_disk_guard_retries_reduce_timeout_and_charge_void_wall_against_cumulative_cap(self):
        """Regression test for qlt05 finding: retry timeout is reduced by void_wall and void_wall counts toward cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            observed_timeouts: list[float] = []

            def fake_run_locked_cell(key, command, log, timeout, status_path, wait_total, *, after_release=False):
                observed_timeouts.append(timeout)
                if len(observed_timeouts) == 1:
                    raise wave.DiskGuardInfraVoid("disk guard on attempt 1", wait_seconds=2.0, cell_wall_seconds=400.0)
                return {"key": key, "status": "complete"}, 3.0, 500.0

            with mock.patch.object(wave, "run_locked_cell", side_effect=fake_run_locked_cell):
                cell, waited, wall, void_wall = wave.run_cell_with_disk_guard_retries(
                    "ES/seed18",
                    ["runner"],
                    root / "ES-seed18.log",
                    1000.0,
                    root / "H77_STATUS.log",
                    0.0,
                    root / "out",
                )
            # Attempt 1 got 1000.0s; attempt 2 got 1000.0 - 400.0 = 600.0s
            self.assertEqual(observed_timeouts, [1000.0, 600.0])
            self.assertEqual(waited, 5.0)
            self.assertEqual(wall, 500.0)
            self.assertEqual(void_wall, 400.0)
            self.assertEqual(cell["total_charged_cell_wall_seconds"], 900.0)

            # If void_wall exhausts the remaining budget, retry is refused before launching another attempt
            def exhaust_on_first(key, command, log, timeout, status_path, wait_total, *, after_release=False):
                raise wave.DiskGuardInfraVoid("disk guard exhausted budget", wait_seconds=1.0, cell_wall_seconds=1000.0)

            with mock.patch.object(wave, "run_locked_cell", side_effect=exhaust_on_first) as mocked_run:
                with self.assertRaisesRegex(wave.InfraVoid, "cumulative GPU wall cap exhausted before attempt"):
                    wave.run_cell_with_disk_guard_retries(
                        "ES/seed18",
                        ["runner"],
                        root / "ES-seed18-b.log",
                        1000.0,
                        root / "H77_STATUS.log",
                        0.0,
                        root / "out2",
                    )
                self.assertEqual(mocked_run.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
