#!/usr/bin/env python3
"""CPU-only unit, red-team, identity, FIFO lock fairness, and admission-boundary tests for H80 long ES."""
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
H77_DIR = HERE.parent / "h77_long_es"

SPEC = importlib.util.spec_from_file_location("h80_runner_test", HERE / "runner.py")
assert SPEC is not None and SPEC.loader is not None
h80 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h80
SPEC.loader.exec_module(h80)

METRICS_SPEC = importlib.util.spec_from_file_location("h80_metrics_test", HERE / "build_metrics.py")
assert METRICS_SPEC is not None and METRICS_SPEC.loader is not None
metrics = importlib.util.module_from_spec(METRICS_SPEC)
sys.modules[METRICS_SPEC.name] = metrics
METRICS_SPEC.loader.exec_module(metrics)

AUDIT_SPEC = importlib.util.spec_from_file_location("h80_audit_test", HERE / "audit.py")
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit
AUDIT_SPEC.loader.exec_module(audit)

WAVE_SPEC = importlib.util.spec_from_file_location("h80_wave_test", HERE / "run_wave.py")
assert WAVE_SPEC is not None and WAVE_SPEC.loader is not None
wave = importlib.util.module_from_spec(WAVE_SPEC)
sys.modules[WAVE_SPEC.name] = wave
WAVE_SPEC.loader.exec_module(wave)

fifo_mod = wave.fifo_lock


class H80LongESTests(unittest.TestCase):
    def test_preregistered_constants_match_t0030_contract_and_h77_closeout(self):
        self.assertEqual(h80.SIGMA, 0.001)
        self.assertEqual(h80.SEEDS, (21, 22, 23))
        self.assertEqual(h80.NULL_SEED, 21)
        self.assertEqual(h80.DEFAULT_GENERATIONS, 16)
        self.assertEqual(h80.PAIRS, 16)
        self.assertEqual(h80.POPULATION, 32)
        self.assertEqual(h80.ES_LEARNING_RATE, 0.01)
        self.assertEqual(h80.ESH_LAMBDA, 1.0)
        self.assertEqual(h80.ARMS, ("ES", "ESH"))
        self.assertEqual(h80.NULL_ARM, "null_s001")
        self.assertEqual(h80.TRAINABLE_PARAMS, 29_211)
        self.assertEqual(wave.MAX_WAVE_GPU_WALL_SECONDS, 9000.0)
        self.assertEqual(metrics.DEFAULT_OUTPUT, HERE / "metrics.json")
        prereg_text = (HERE / "PREREG.md").read_text()
        self.assertIn(
            "r882 was trained on all 1,219 train contexts incl. fold B, so fold-B paired deltas are in-sample vs r882; only DL-Val (once, admitted arms only) is out-of-sample.",
            prereg_text,
        )
        self.assertIn("80_4900", prereg_text)
        self.assertIn("wave_name=\"H80\"", prereg_text)

        # Verify H77 closeout artifacts exist and point to H80
        h77_metrics = json.loads((H77_DIR / "metrics.json").read_text())
        self.assertEqual(h77_metrics["status"], "stopped_pre_outcome_infeasible_as_preregistered")
        self.assertFalse(h77_metrics["data_hygiene"]["fold_b_scored"])
        self.assertEqual(h77_metrics["data_hygiene"]["outcome_cells_started"], 0)
        self.assertEqual(h77_metrics["data_hygiene"]["dl_val_reads"], 0)
        self.assertEqual(h77_metrics["successor_wave"], "H80 (research/flywheel_selfplay/h80_long_es)")
        self.assertLess((H77_DIR / "metrics.json").stat().st_size, 50 * 1024)

    def test_centered_rank_order_and_centering_pop32(self):
        ranks = h80.centered_ranks(np.arange(32, dtype=np.float64))
        self.assertEqual(ranks.shape, (32,))
        self.assertAlmostEqual(float(ranks.sum()), 0.0, places=12)
        self.assertTrue(np.all(np.diff(ranks) > 0))

    def test_antithetic_population_is_16_pairs_pop32_and_shared_across_arms(self):
        center = torch.zeros(h80.TRAINABLE_PARAMS)
        seed_es = h80.stream_seed(21, 1, arm="ES", null_arm=False)
        seed_esh = h80.stream_seed(21, 1, arm="ESH", null_arm=False)
        seed_null = h80.stream_seed(21, 1, arm="ES", null_arm=True)
        self.assertEqual(seed_es, seed_esh)
        self.assertEqual(seed_es, seed_null)

        noises, pop = h80.antithetic_population(center, 0.001, seed_es)
        noises_esh, pop_esh = h80.antithetic_population(center, 0.001, seed_esh)
        self.assertEqual(len(noises), 16)
        self.assertEqual(len(pop), 32)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(noises, noises_esh, strict=True)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pop, pop_esh, strict=True)))
        for pos, neg in zip(pop[::2], pop[1::2], strict=True):
            self.assertTrue(torch.equal(pos, -neg))

        with self.assertRaisesRegex(ValueError, "sigma=0.001"):
            h80.antithetic_population(center, 0.003, seed_es)

    def test_es_vs_esh_harm_penalty_fitness_math(self):
        baseline = {"f1": 1.0, "f2": 1.0, "f3": 1.0, "f4": 1.0}
        cand_no_harm = {"f1": 1.05, "f2": 1.02, "f3": 0.995, "f4": 0.991}
        res_es = h80.compute_fold_a_fitness(cand_no_harm, baseline, "ES")
        res_esh = h80.compute_fold_a_fitness(cand_no_harm, baseline, "ESH")
        self.assertAlmostEqual(res_es["fitness"], res_esh["fitness"], places=12)
        self.assertAlmostEqual(res_esh["harm_term_min0_gain_plus_001"], 0.0, places=12)
        self.assertAlmostEqual(res_esh["harm_penalty_subtracted"], 0.0, places=12)
        self.assertEqual(res_esh["fold_a_harm_fraction"], 0.0)

        cand_harm = {"f1": 1.05, "f2": 1.02, "f3": 0.995, "f4": 0.89}
        res_es_harm = h80.compute_fold_a_fitness(cand_harm, baseline, "ES")
        res_esh_harm = h80.compute_fold_a_fitness(cand_harm, baseline, "ESH")
        self.assertAlmostEqual(res_esh_harm["harm_term_min0_gain_plus_001"], -0.025, places=12)
        self.assertAlmostEqual(res_esh_harm["harm_penalty_subtracted"], 0.025, places=12)
        self.assertAlmostEqual(res_esh_harm["fitness"], res_es_harm["fitness"] - 0.025, places=12)
        self.assertLess(res_esh_harm["fitness"], res_es_harm["fitness"])
        self.assertAlmostEqual(res_esh_harm["fold_a_harm_fraction"], 0.25, places=12)

    def test_centered_rank_es_gradient_is_finite_and_deterministic_pop32(self):
        noises = [torch.full((5,), float(i + 1)) for i in range(16)]
        ranks = h80.centered_ranks(np.arange(32, dtype=np.float64))
        first = h80.es_gradient(noises, ranks, 0.001)
        second = h80.es_gradient(noises, ranks, 0.001)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(float(torch.linalg.vector_norm(first)), 0.0)

    def test_only_preregistered_null_and_timing_cells_are_accepted(self):
        bad_null = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ESH", seed=21, generations=16, null_arm=True, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "only the ES seed21 outcome"):
            h80.run_train(bad_null)

        bad_null_seed = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ES", seed=22, generations=16, null_arm=True, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "only the ES seed21 outcome"):
            h80.run_train(bad_null_seed)

        bad_timing = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ESH", seed=21, generations=1, null_arm=False, mode="timing"
        )
        with self.assertRaisesRegex(RuntimeError, "only ES seed21 one-generation timing"):
            h80.run_train(bad_timing)

        bad_horizon = SimpleNamespace(
            device="cuda:0", lock_held=True, arm="ES", seed=21, generations=8, null_arm=False, mode="outcome"
        )
        with self.assertRaisesRegex(RuntimeError, "fixed at 16 generations"):
            h80.run_train(bad_horizon)

    def test_a_priori_h77_timing_projection_fits_under_9000s_cap(self):
        # H77 timing cell measured 120.445342s for 64 perturbations -> 60.222671s for 32 perturbations
        proj = h80.project_wave_wall_from_timing(120.44534218707122 / 2.0, cap_seconds=9000.0)
        self.assertTrue(proj["fits_within_cap"])
        self.assertEqual(proj["total_outcome_generations"], 112)
        self.assertLess(proj["projected_total_seconds"], 9000.0)

        proj_over = h80.project_wave_wall_from_timing(90.0, cap_seconds=9000.0)
        self.assertFalse(proj_over["fits_within_cap"])
        self.assertGreater(proj_over["projected_total_seconds"], 9000.0)

    def test_pooled_admission_gates_and_elo_per_seed_and_median(self):
        tie_rows = [
            {"arm": "ES", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.0 for i in range(20)}}
            for seed in (21, 22, 23)
        ]
        tie_res = metrics.pooled_admission(tie_rows, "ES")
        self.assertFalse(tie_res["admitted"])
        self.assertTrue(tie_res["is_tie"])
        self.assertEqual(tie_res["elo_vs_r882_median"], 1500.0)
        self.assertEqual(tie_res["elo_delta_vs_r882_median"], 0.0)

        boundary_rows = [
            {"arm": "ES", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.001 for i in range(20)}}
            for seed in (21, 22, 23)
        ]
        boundary_res = metrics.pooled_admission(boundary_rows, "ES")
        self.assertEqual(boundary_res["mean_lcb95_one_sided"], 0.001)
        self.assertFalse(boundary_res["admitted"])

        harm_gains = {f"f{i}": 0.05 for i in range(20)}
        harm_gains["f0"] = -0.02
        harm_gains["f1"] = -0.02
        harm_gains["f2"] = -0.02
        harm_rows = [{"arm": "ES", "seed": seed, "family_gain_vs_r882": harm_gains} for seed in (21, 22, 23)]
        harm_res = metrics.pooled_admission(harm_rows, "ES")
        self.assertGreater(harm_res["mean_lcb95_one_sided"], 0.001)
        self.assertGreater(harm_res["harm_fraction_below_minus_0_01"], 0.10)
        self.assertFalse(harm_res["admitted"])

        good_rows = [
            {"arm": "ESH", "seed": seed, "family_gain_vs_r882": {f"f{i}": 0.02 for i in range(20)}}
            for seed in (21, 22, 23)
        ]
        good_res = metrics.pooled_admission(good_rows, "ESH")
        self.assertTrue(good_res["admitted"])
        self.assertGreater(good_res["mean_lcb95_one_sided"], 0.001)
        self.assertEqual(good_res["harm_fraction_below_minus_0_01"], 0.0)
        self.assertEqual(set(good_res["elo_vs_r882_per_seed"].keys()), {"21", "22", "23"})
        self.assertGreater(good_res["elo_vs_r882_median"], 1500.0)
        self.assertGreater(good_res["elo_delta_vs_r882_median"], 0.0)

    def test_independent_admission_audit_catches_tampering(self):
        candidates = []
        for arm, gain in (("ES", 0.0), ("ESH", 0.02)):
            for seed in (21, 22, 23):
                candidates.append(
                    {"arm": arm, "seed": seed, "family_gain_vs_r882": {f"f{i}": gain for i in range(20)}}
                )
        computed = {arm: metrics.pooled_admission(candidates, arm) for arm in ("ES", "ESH")}
        fold_b = {"fold_assignment_sha256": "fold-hash-80", "candidates": candidates}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outcomes").mkdir()
            (root / "outcomes/admission.json").write_text(
                json.dumps(
                    {
                        "schema": "jope.h80.admission/v1",
                        "fold_seed": 4900,
                        "fold_assignment_sha256": "fold-hash-80",
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

    def test_fifo_queue_releases_between_cells_and_requeues_at_tail_behind_w1_or_h82(self):
        """Verify H80 queues at FIFO tail with wave_name='H80' behind W1/H82 tickets and backs off on legacy waiters."""
        self.assertEqual(wave.LOCK_POLL_SECONDS, 30.0)
        self.assertEqual(wave.LEGACY_BACKOFF_SECONDS, 45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lock_file = tmp / "qualification.lock"
            queue_dir = tmp / "qualification.queue"
            data_root = tmp / "jope-data"
            status_log = data_root / "h80/h80-wave/H80_STATUS.log"
            h82_log = data_root / "h82/h82-wave/H82_STATUS.log"
            h82_log.parent.mkdir(parents=True, exist_ok=True)

            # Enqueue W1 ticket at epoch_ns=100 ahead of H80 at epoch_ns=200
            t_w1 = fifo_mod.enqueue_ticket(queue_dir, pid=2001, wave="W1", epoch_ns=100)
            fixed_now = datetime(2026, 10, 31, 6, 30, 0, tzinfo=timezone.utc)
            h82_log.write_text(
                f"{(fixed_now - timedelta(seconds=15)).isoformat()} — WAIT qualification.lock busy; cumulative_wait=30.0s\n"
            )

            sleeps: list[float] = []

            def fake_sleep(sec: float) -> None:
                sleeps.append(sec)
                if len(sleeps) >= 2 and t_w1.exists():
                    fifo_mod.remove_ticket(t_w1)

            with mock.patch.object(wave, "launch_disk_guard_error", return_value=None):
                handle, _, free_mib, _ = wave.acquire_lock(
                    status_log,
                    0.0,
                    after_release=True,
                    wave_name="H80",
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

    def test_disk_guard_retries_reduce_timeout_and_charge_void_wall_against_cumulative_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            observed_timeouts: list[float] = []
            observed_after_release: list[bool] = []

            def fake_run_locked_cell(key, command, log, timeout, status_path, wait_total, *, after_release=True):
                observed_timeouts.append(timeout)
                observed_after_release.append(after_release)
                if len(observed_timeouts) == 1:
                    raise wave.DiskGuardInfraVoid("disk guard on attempt 1", wait_seconds=2.0, cell_wall_seconds=400.0)
                return {"key": key, "status": "complete"}, 3.0, 500.0

            with mock.patch.object(wave, "run_locked_cell", side_effect=fake_run_locked_cell):
                cell, waited, wall, void_wall = wave.run_cell_with_disk_guard_retries(
                    "ES/seed21",
                    ["runner"],
                    root / "ES-seed21.log",
                    1000.0,
                    root / "H80_STATUS.log",
                    0.0,
                    root / "out",
                )
            self.assertEqual(observed_timeouts, [1000.0, 600.0])
            self.assertEqual(observed_after_release, [True, True])
            self.assertEqual(waited, 5.0)
            self.assertEqual(wall, 500.0)
            self.assertEqual(void_wall, 400.0)
            self.assertEqual(cell["total_charged_cell_wall_seconds"], 900.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
