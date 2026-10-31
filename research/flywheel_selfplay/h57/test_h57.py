#!/usr/bin/env python3
"""Data-free H57 ES, identity, and admission-boundary tests."""
from __future__ import annotations

import importlib.util
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h57_runner_test", HERE / "runner.py")
assert SPEC is not None and SPEC.loader is not None
h57 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h57)
METRICS_SPEC = importlib.util.spec_from_file_location("h57_metrics_test", HERE / "build_metrics.py")
assert METRICS_SPEC is not None and METRICS_SPEC.loader is not None
metrics = importlib.util.module_from_spec(METRICS_SPEC)
METRICS_SPEC.loader.exec_module(metrics)
AUDIT_SPEC = importlib.util.spec_from_file_location("h57_audit_test", HERE / "audit.py")
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)
WAVE_SPEC = importlib.util.spec_from_file_location("h57_wave_test", HERE / "run_wave.py")
assert WAVE_SPEC is not None and WAVE_SPEC.loader is not None
wave = importlib.util.module_from_spec(WAVE_SPEC)
WAVE_SPEC.loader.exec_module(wave)


class H57Tests(unittest.TestCase):
    def test_terminal_metrics_refuses_unadmitted_receipt_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sigma001").mkdir()
            (root / "sigma001/seed15.json").write_text("{}")
            admissions = {arm: {"admitted": False} for arm in metrics.ARMS}
            with mock.patch.object(metrics, "load") as read:
                with self.assertRaisesRegex(SystemExit, "admitted arm/seed set"):
                    metrics.terminal_metrics(root, admissions)
                read.assert_not_called()

    def test_terminal_metrics_requires_exact_identity_and_single_read(self):
        for defect in ("missing", "extra", "misbound", "count"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "sigma001").mkdir()
                for seed in metrics.SEEDS:
                    (root / f"sigma001/seed{seed}.json").write_text(json.dumps({
                        "arm": "sigma001", "seed": seed,
                        "family_utility": {"synthetic": 1.0},
                        "family_delta_vs_r882_harness": {"synthetic": 0.1},
                        "terminal_dl_val_score_count": 1,
                    }))
                if defect == "missing":
                    (root / "sigma001/seed17.json").unlink()
                elif defect == "extra":
                    (root / "sigma001/seed999.json").write_text("{}")
                else:
                    path = root / "sigma001/seed15.json"
                    row = json.loads(path.read_text())
                    row["seed" if defect == "misbound" else "terminal_dl_val_score_count"] = 16 if defect == "misbound" else 0
                    path.write_text(json.dumps(row))
                admissions = {arm: {"admitted": arm == "sigma001"} for arm in metrics.ARMS}
                with self.assertRaisesRegex(SystemExit, "H57 terminal DL-Val"):
                    metrics.terminal_metrics(root, admissions)

    def test_terminal_metrics_accepts_exact_authorized_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sigma001").mkdir()
            for seed in metrics.SEEDS:
                (root / f"sigma001/seed{seed}.json").write_text(json.dumps({
                    "arm": "sigma001", "seed": seed,
                    "family_utility": {"synthetic": 1.0},
                    "family_delta_vs_r882_harness": {"synthetic": 0.1},
                    "terminal_dl_val_score_count": 1,
                }))
            admissions = {arm: {"admitted": arm == "sigma001"} for arm in metrics.ARMS}
            result = metrics.terminal_metrics(root, admissions)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["reads_allowed"], 3)
            self.assertEqual([row["seed"] for row in result["reads"]], list(metrics.SEEDS))

    def test_terminal_metrics_preserves_pending_and_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            admissions = {arm: {"admitted": False} for arm in metrics.ARMS}
            self.assertEqual(metrics.terminal_metrics(root, admissions), {
                "reads_allowed": 0, "status": "not run; no sigma arm admitted", "reads": [],
            })
            self.assertEqual(metrics.terminal_metrics(root / "absent", admissions)["reads"], [])
            admissions["sigma001"]["admitted"] = True
            pending = metrics.terminal_metrics(root / "absent", admissions)
            self.assertEqual(pending["status"], "pending admission-gated reads")
            self.assertEqual(pending["reads_allowed"], 3)
            self.assertEqual(pending["reads"], [])

    def test_metrics_default_output_stays_in_h57_research_directory(self):
        self.assertEqual(metrics.DEFAULT_OUTPUT, HERE / "metrics.json")

    def test_centered_rank_order_and_centering(self):
        ranks = h57.centered_ranks(np.arange(32, dtype=np.float64))
        self.assertAlmostEqual(float(ranks.sum()), 0.0)
        self.assertTrue(np.all(np.diff(ranks) > 0))

    def test_antithetic_population_is_exactly_paired_and_seeded(self):
        center = torch.zeros(h57.TRAINABLE_PARAMS)
        noise, population = h57.antithetic_population(
            center, 0.001, h57.stream_seed(15, "sigma001", 1, False)
        )
        noise_null, population_null = h57.antithetic_population(
            center, 0.001, h57.stream_seed(15, "sigma001", 1, True)
        )
        _, population_seeded = h57.antithetic_population(
            center, 0.001, h57.stream_seed(15, "sigma001", 1, False)
        )
        self.assertEqual(len(noise), 16)
        self.assertEqual(len(population), 32)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(noise, noise_null, strict=True)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(population_null, population_seeded, strict=True)))
        for positive, negative in zip(population[::2], population[1::2], strict=True):
            self.assertTrue(torch.equal(positive, -negative))

    def test_centered_rank_es_gradient_is_finite_and_deterministic(self):
        noises = [torch.full((3,), float(i + 1)) for i in range(16)]
        ranks = h57.centered_ranks(np.arange(32, dtype=np.float64))
        first = h57.es_gradient(noises, ranks, 0.001)
        second = h57.es_gradient(noises, ranks, 0.001)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(float(torch.linalg.vector_norm(first)), 0.0)

    def test_adapter_vector_round_trip_preserves_shapes_and_values(self):
        template = {"b": torch.arange(4, dtype=torch.float32).reshape(2, 2),
                    "a": torch.arange(3, dtype=torch.float32)}
        names = tuple(sorted(template))
        vector = h57.adapter_vector(template, names)
        restored = h57.vector_state(vector, template, names)
        self.assertEqual(tuple(restored), names)
        self.assertTrue(torch.equal(restored["a"], template["a"]))
        self.assertTrue(torch.equal(restored["b"], template["b"]))

    def test_only_preregistered_null_cell_is_accepted(self):
        args = SimpleNamespace(device="cuda:0", lock_held=True, arm="sigma003", seed=16,
                               generations=8, null_arm=True, mode="outcome")
        with self.assertRaisesRegex(RuntimeError, "only the sigma001 seed15"):
            h57.run_train(args)

    def test_timing_horizon_is_fixed_to_one_sigma001_seed15_generation(self):
        args = SimpleNamespace(device="cuda:0", lock_held=True, arm="sigma003", seed=15,
                               generations=1, null_arm=False, mode="timing")
        with self.assertRaisesRegex(RuntimeError, "only sigma001 seed15"):
            h57.run_train(args)

    def test_pooled_admission_rejects_ties_and_nonpositive_lcb(self):
        rows = [{"arm": "sigma001", "seed": seed,
                 "family_gain_vs_r882": {f"f{i}": 0.0 for i in range(8)}}
                for seed in (15, 16, 17)]
        result = metrics.pooled_admission(rows, "sigma001")
        self.assertEqual(result["mean_lcb95_one_sided"], 0.0)
        self.assertFalse(result["admitted"])

    def test_pooled_admission_requires_strict_margin(self):
        rows = [{"arm": "sigma001", "seed": seed,
                 "family_gain_vs_r882": {f"f{i}": 0.001 for i in range(8)}}
                for seed in (15, 16, 17)]
        result = metrics.pooled_admission(rows, "sigma001")
        self.assertEqual(result["mean_lcb95_one_sided"], 0.001)
        self.assertFalse(result["admitted"])

    def test_positive_family_cluster_margin_and_no_harm_admit(self):
        rows = [{"arm": "sigma003", "seed": seed,
                 "family_gain_vs_r882": {f"f{i}": 0.02 for i in range(8)}}
                for seed in (15, 16, 17)]
        result = metrics.pooled_admission(rows, "sigma003")
        self.assertGreater(result["mean_lcb95_one_sided"], 0.001)
        self.assertTrue(result["admitted"])

    def test_independent_admission_audit_rejects_tampered_decision(self):
        candidates = []
        for arm, gain in (("sigma001", 0.02), ("sigma003", 0.0)):
            for seed in (15, 16, 17):
                candidates.append({"arm": arm, "seed": seed,
                                   "family_gain_vs_r882": {f"f{i}": gain for i in range(8)}})
        computed = {arm: metrics.pooled_admission(candidates, arm)
                    for arm in ("sigma001", "sigma003")}
        fold_b = {"fold_assignment_sha256": "fold-hash", "candidates": candidates}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outcomes").mkdir()
            (root / "outcomes/admission.json").write_text(json.dumps({
                "schema": "jope.h57.admission/v1", "fold_seed": 4900,
                "fold_assignment_sha256": "fold-hash", "arms": computed,
                "terminal_dl_val_reads_allowed": 3,
            }))
            self.assertTrue(audit.audit_admission(root, fold_b)["sigma001"]["admitted"])
            record = json.loads((root / "outcomes/admission.json").read_text())
            record["arms"]["sigma001"]["admitted"] = False
            (root / "outcomes/admission.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(SystemExit, "disagrees for sigma001"):
                audit.audit_admission(root, fold_b)

    def test_lock_is_released_when_under_lock_preflight_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "qualification.lock"
            status_path = Path(temporary) / "status.log"
            checks = iter(({}, RuntimeError("host CPU pressure"), {}))
            sleep_observed_unlocked = []

            def preflight():
                value = next(checks)
                if isinstance(value, Exception):
                    raise value
                return value

            def sleep(_seconds):
                with lock_path.open("a+") as other:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    sleep_observed_unlocked.append(True)
                    fcntl.flock(other, fcntl.LOCK_UN)
                self.assertIn("WAIT host CPU pressure", status_path.read_text())

            with mock.patch.object(wave, "LOCK", lock_path), \
                 mock.patch.object(wave, "safe_preflight", side_effect=preflight), \
                 mock.patch.object(wave, "gpu_free_mib", return_value=8192), \
                 mock.patch.object(wave, "launch_disk_guard_error", return_value=None), \
                 mock.patch.object(wave.time, "sleep", side_effect=sleep):
                handle, _, free_mib, _ = wave.acquire_lock(status_path, 0.0)
                self.assertEqual(free_mib, 8192)
                self.assertEqual(sleep_observed_unlocked, [True])
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()

    def test_disk_guard_uses_statvfs_available_blocks(self):
        stats = SimpleNamespace(f_blocks=100 * 1024**3, f_bavail=5 * 1024**3, f_frsize=1)
        with mock.patch.object(wave.os, "statvfs", return_value=stats):
            self.assertRegex(wave.disk_guard_error(), "95% full")

    def test_cell_launch_uses_94_5_margin_without_changing_95_writer_guard(self):
        stats = SimpleNamespace(f_blocks=100 * 1024**3, f_bavail=5.4 * 1024**3, f_frsize=1)
        with mock.patch.object(wave.os, "statvfs", return_value=stats):
            self.assertIsNone(wave.disk_guard_error())
            self.assertRegex(wave.launch_disk_guard_error(), "below 94.5%")
        stats = SimpleNamespace(f_blocks=100 * 1024**3, f_bavail=5.6 * 1024**3, f_frsize=1)
        with mock.patch.object(wave.os, "statvfs", return_value=stats):
            self.assertIsNone(wave.launch_disk_guard_error())

    def test_disk_guard_infra_void_retries_same_cell_with_separate_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "sigma001-seed15.log"
            output = root / "out"
            success = ({"key": "sigma001/seed15", "status": "complete"}, 3.0, 100.0)
            with mock.patch.object(wave, "run_locked_cell", side_effect=[
                    wave.DiskGuardInfraVoid("post-training disk guard", 2.0, 40.0), success]) as run:
                cell, waited, wall, void_wall = wave.run_cell_with_disk_guard_retries(
                    "sigma001/seed15", ["runner"], log, 1000.0, root / "status.log", 0.0, output)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[2], log)
            self.assertEqual(run.call_args_list[1].args[2], root / "sigma001-seed15.retry1.log")
            self.assertEqual((waited, wall, void_wall), (5.0, 100.0, 40.0))
            self.assertEqual(cell["disk_guard_retry_count"], 1)
            self.assertEqual(cell["attempt_logs"], [str(log), str(root / "sigma001-seed15.retry1.log")])

    def test_disk_guard_retry_refuses_to_overwrite_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            output.mkdir()
            evidence = output / "partial-evidence.bin"
            evidence.write_bytes(b"preserve")
            with mock.patch.object(wave, "run_locked_cell", side_effect=
                                   wave.DiskGuardInfraVoid("post-training disk guard", 1.0, 10.0)) as run:
                with self.assertRaisesRegex(wave.InfraVoid, "retry refused because prior output files exist"):
                    wave.run_cell_with_disk_guard_retries(
                        "sigma001/seed15", ["runner"], root / "cell.log", 1000.0,
                        root / "status.log", 0.0, output)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(evidence.read_bytes(), b"preserve")

    def test_disk_guard_retry_limit_is_two_retries_and_tracks_void_wall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = wave.DiskGuardInfraVoid("post-training disk guard", 2.0, 40.0)
            with mock.patch.object(wave, "run_locked_cell", side_effect=[failure, failure, failure]) as run:
                with self.assertRaisesRegex(wave.DiskGuardInfraVoid, "exhausted 2 disk-guard retries") as raised:
                    wave.run_cell_with_disk_guard_retries(
                        "sigma001/seed15", ["runner"], root / "cell.log", 1000.0,
                        root / "status.log", 0.0, root / "out")
            self.assertEqual(run.call_count, 3)
            self.assertEqual(raised.exception.wait_seconds, 6.0)
            self.assertEqual(raised.exception.cell_wall_seconds, 120.0)

    def test_wait_reasons_include_lock_holder_gpu_and_disk_state(self):
        lock_lines = "1: FLOCK ADVISORY WRITE 4242 103:07:108299230 0 EOF\n"
        self.assertEqual(wave.lock_holder_pid_from_proc_locks(lock_lines, "103:07:108299230"), 4242)
        self.assertIsNone(wave.lock_holder_pid_from_proc_locks(lock_lines, "103:07:999"))
        self.assertEqual(wave.format_lock_wait_reason(4242), "qualification.lock busy; holder PID=4242")
        self.assertEqual(wave.gpu_wait_reason(4096), "GPU free 4096 MiB below 6144 MiB")
        stats = SimpleNamespace(f_blocks=100 * 1024**3, f_bavail=4 * 1024**3, f_frsize=1)
        with mock.patch.object(wave.os, "statvfs", return_value=stats):
            self.assertEqual(wave.preflight_wait_reason(RuntimeError("filesystem / is at or above 95% full")),
                             "disk 96.00% vs 95% guard")

    def test_preflight_snapshot_records_retryable_guard_instead_of_aborting(self):
        with mock.patch.object(wave, "safe_preflight", side_effect=RuntimeError("MemAvailable below 16 GiB: 1")):
            self.assertEqual(wave.preflight_snapshot(), {"blocked": "MemAvailable below 16 GiB: 1"})

    def test_resource_floor_matches_owner_packet(self):
        self.assertEqual(wave.MIN_MEM_AVAILABLE_KIB, 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
