#!/usr/bin/env python3
"""CPU-only fixture and red-team unit tests for the H61 selection-metric runner and FIFO coordinator."""
from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h61_runner_under_test", HERE / "runner.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import H61 runner")
h61 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h61
spec.loader.exec_module(h61)

wave_spec = importlib.util.spec_from_file_location("h61_wave_under_test", HERE / "run_wave.py")
if wave_spec is None or wave_spec.loader is None:
    raise RuntimeError("cannot import H61 run_wave")
h61_wave = importlib.util.module_from_spec(wave_spec)
sys.modules[wave_spec.name] = h61_wave
wave_spec.loader.exec_module(h61_wave)

fifo_mod = h61_wave.fifo_lock

FakeStatvfs = namedtuple("FakeStatvfs", ["f_blocks", "f_bavail", "f_frsize"])


def make_synthetic_pools(num_families: int = 8, contexts_per_family: int = 2, pool_size: int = 12) -> dict[str, dict]:
    """Build synthetic DL-Train pools where revealed anchors 0..3 have extreme leaked values."""
    pools: dict[str, dict] = {}
    for fam_idx in range(num_families):
        fam_id = f"fam_{fam_idx:02d}"
        for ctx_idx in range(contexts_per_family):
            cid = f"ctx_{fam_idx:02d}_{ctx_idx:02d}"
            # Revealed anchors 0..3 have artificially low fitness (-3.5 => utility +3.5) to detect 0..3 leakage
            fitness = np.full(pool_size, -0.5, dtype=np.float64)
            fitness[:4] = -3.5
            # Unrevealed candidates 4..P-1 vary cleanly
            for j in range(4, pool_size):
                fitness[j] = -0.2 - 0.1 * (j - 4)
            gate = np.ones(pool_size, dtype=bool)
            pools[cid] = {
                "rec": {
                    "context_sha256": cid,
                    "family_id": fam_id,
                    "dl_split": "DL-Train",
                    "source_split": "train",
                    "task": "regression",
                },
                "pids": [f"{cid}_prog_{j}" for j in range(pool_size)],
                "fitness": fitness,
                "gate": gate,
            }
    return pools


class H61RedTeamTests(unittest.TestCase):
    def test_redteam_a_all_arms_and_r882_scored_in_same_unrev_eval_mode(self):
        """(a) S0..S3 and r882 are scored for admission/Elo in leak-free unrev (4..P-1); S0 may select natively."""
        pools = make_synthetic_pools(num_families=6, contexts_per_family=2, pool_size=10)
        cids = sorted(pools)

        # Incumbent predicts index 0 (revealed anchor) as best overall, and index 4 (u=0.2) as best unrevealed.
        inc_preds = {}
        # Candidate predicts index 1 (revealed anchor, tied native u=3.5) and index 9 (u=0.7) as best unrevealed.
        cand_preds = {}
        for cid in cids:
            inc_row = np.linspace(1.0, 2.0, 10)
            inc_row[0] = -10.0  # revealed anchor 0
            inc_row[4] = 0.0   # unrevealed index 4 -> u = 0.2
            inc_preds[cid] = inc_row

            cand_row = np.linspace(1.0, 2.0, 10)
            cand_row[1] = -10.0  # revealed anchor 1 (native tie at +3.5)
            cand_row[9] = -1.0   # unrevealed index 9 -> u = 0.7
            cand_preds[cid] = cand_row

        # S0 selects with native rule (accepts native tie), but its reported utility MUST be unrev (0.7, never 3.5)
        s0_sel = h61.select_round_candidate("S0", cids, pools, inc_preds, cand_preds)
        self.assertTrue(s0_sel["accepted"])
        self.assertEqual(s0_sel["reported_eval_mode"], "unrev")
        self.assertAlmostEqual(s0_sel["reported_family_utility"], 0.7, places=6)
        self.assertTrue(all(idx >= 4 for idx in s0_sel["reported_chosen_indices"].values()))

        # Fold-B scoring for admission/Elo must score S0..S3 and r882 in leak-free unrev mode and reject native mode
        arm_preds = {arm: cand_preds for arm in h61.ARMS}
        with self.assertRaisesRegex(ValueError, "eval_mode='unrev'"):
            h61.score_fold_b_arms(pools, inc_preds, arm_preds, eval_mode="native")

        scored = h61.score_fold_b_arms(pools, inc_preds, arm_preds, eval_mode="unrev")
        self.assertEqual(scored["eval_mode"], "unrev")
        self.assertAlmostEqual(scored["baseline"]["mean"], 0.2, places=6)
        for arm in h61.ARMS:
            arm_res = scored["arms"][arm]
            self.assertEqual(arm_res["eval_mode"], "unrev")
            self.assertAlmostEqual(arm_res["family_utility"], 0.7, places=6)
            self.assertTrue(all(idx >= 4 for idx in arm_res["chosen_indices"].values()))
            self.assertGreater(arm_res["elo_vs_r882"]["ratings"]["H61"], 1500.0)

    def test_redteam_b_ties_fail_everywhere_and_identical_utilities_rejected_by_s1_s3(self):
        """(b) Ties fail everywhere (gain > m=0.001546 AND LCB95 > +0.001); identical utilities rejected by S1-S3."""
        pools = make_synthetic_pools(num_families=8, contexts_per_family=2, pool_size=10)
        cids = sorted(pools)
        preds = {cid: np.linspace(0.0, 1.0, 10) for cid in cids}

        # Identical candidate and incumbent utilities -> S0 accepts tie, S1/S2/S3 MUST reject
        s0_tie = h61.select_round_candidate("S0", cids, pools, preds, preds)
        self.assertTrue(s0_tie["accepted"])
        for arm in ("S1", "S2", "S3"):
            decision = h61.select_round_candidate(arm, cids, pools, preds, preds)
            self.assertFalse(decision["accepted"], f"{arm} must reject identical candidate/incumbent utilities")

        # Sub-margin gain (+0.0012 <= m=0.001546) must fail both round selection and wave admission
        sub_margin_gains = {seed: {f"f{i}": 0.0012 for i in range(12)} for seed in h61.SEEDS}
        sub_adm = h61.paired_family_admission(sub_margin_gains)
        self.assertFalse(sub_adm["admitted"])

        # Exact boundary m=0.001546 must also fail (strict > m required)
        boundary_gains = {seed: {f"f{i}": h61.NULL_MARGIN_M for i in range(12)} for seed in h61.SEEDS}
        self.assertFalse(h61.paired_family_admission(boundary_gains)["admitted"])

        # Exact zero ties fail wave admission
        zero_gains = {seed: {f"f{i}": 0.0 for i in range(12)} for seed in h61.SEEDS}
        zero_adm = h61.paired_family_admission(zero_gains)
        self.assertTrue(zero_adm["is_tie"])
        self.assertFalse(zero_adm["admitted"])

    def test_redteam_c_identical_training_updates_and_candidate_pool_hashes_across_arms(self):
        """(c) Assert identical RNG stream, context IDs, slate indices, and candidate pool hash per round across arms."""
        train_ids = ("c1", "c2", "c3", "c4")
        pools = {
            cid: {"size": 256, "pids": [f"{cid}_p{i}" for i in range(256)]}
            for cid in train_ids
        }
        orders = {cid: np.arange(256) for cid in train_ids}

        def cached_sampler(pool, order, size):
            return np.random.choice(order, size=size, replace=False)

        plan_a = h61.build_paired_batches(train_ids, pools, orders, cached_sampler, seed=12)
        plan_b = h61.build_paired_batches(train_ids, pools, orders, cached_sampler, seed=12)
        plans = h61.paired_plans_for_arms(plan_a)
        self.assertEqual(len(plan_a), h61.ROUNDS * h61.STEPS_PER_ROUND)
        for b1, b2 in zip(plan_a, plan_b, strict=True):
            self.assertEqual(b1.seed, b2.seed)
            self.assertEqual(b1.round_rng_seed, b2.round_rng_seed)
            self.assertEqual(b1.candidate_pool_sha256, b2.candidate_pool_sha256)

        # Tampering with one arm's candidate_pool_sha256 must raise RuntimeError
        tampered_batch = h61.TrainingBatch(
            round_index=plan_a[0].round_index,
            step_index=plan_a[0].step_index,
            seed=plan_a[0].seed,
            round_rng_seed=plan_a[0].round_rng_seed,
            context_ids=plan_a[0].context_ids,
            candidate_indices=plan_a[0].candidate_indices,
            candidate_pool_sha256="0" * 64,
        )
        bad_plans = dict(plans)
        bad_plans["S2"] = (tampered_batch,) + plan_a[1:]
        with self.assertRaisesRegex(RuntimeError, "stream/candidate-pool mismatch"):
            h61.assert_identical_round_streams(bad_plans)

    def test_redteam_d_tau_and_m_s2_chosen_only_by_dl_train_inner_cv(self):
        """(d) tau/m_S2 chosen only by DL-Train inner CV; assert no DL-Val or fold-B rows enter the CV."""
        pools_a = make_synthetic_pools(num_families=8, contexts_per_family=2, pool_size=10)
        all_fams = sorted({p["rec"]["family_id"] for p in pools_a.values()})
        inner_folds = [set(all_fams[i::4]) for i in range(4)]

        inc_by_fold = {}
        cand_by_fold = {}
        for fold_idx in range(4):
            inc_by_fold[fold_idx] = {cid: np.linspace(0.0, 1.0, 10) for cid in pools_a}
            cand_by_fold[fold_idx] = {cid: np.linspace(1.0, 0.0, 10) for cid in pools_a}

        cv_res = h61.select_s2_hyperparameters_inner_cv(
            inner_folds,
            pools_a,
            inc_by_fold,
            cand_by_fold,
            forbidden_context_ids=("fold_b_ctx_01", "dl_val_ctx_01"),
            forbidden_family_ids=("fold_b_fam_01", "dl_val_fam_01"),
        )
        self.assertIn(cv_res["tau"], h61.TAU_GRID)
        self.assertIn(cv_res["m_s2"], h61.M_S2_GRID)
        self.assertEqual(cv_res["dl_val_rows_used"], 0)
        self.assertEqual(cv_res["fold_b_rows_used"], 0)

        # Injecting a fold-B family or DL-Val split into pools_a must fail closed
        with self.assertRaisesRegex(RuntimeError, "fold-B or DL-Val"):
            h61.select_s2_hyperparameters_inner_cv(
                inner_folds,
                pools_a,
                inc_by_fold,
                cand_by_fold,
                forbidden_family_ids=(all_fams[0],),
            )

        leaked_pools = dict(pools_a)
        first_cid = sorted(leaked_pools)[0]
        leaked_pools[first_cid] = dict(leaked_pools[first_cid], rec=dict(leaked_pools[first_cid]["rec"], dl_split="DL-Val"))
        with self.assertRaisesRegex(RuntimeError, "non-DL-Train row"):
            h61.select_s2_hyperparameters_inner_cv(inner_folds, leaked_pools, inc_by_fold, cand_by_fold)

    def test_redteam_e_per_context_loss_and_utility_clipping_consistent_with_h59(self):
        """(e) Per-context loss/utility clipping consistent with H59 so -2.0 failure contexts cannot dominate bootstrap."""
        fitness = torch.tensor([[-10.0, -0.5, 1.0, 10.0]])
        gates = torch.tensor([[True, True, True, False]])
        targets = h61.preregistered_targets(fitness, gates)
        self.assertTrue(torch.equal(targets, torch.tensor([[-4.0, -0.5, 1.0, 2.0]])))

        pred = torch.tensor([[0.0, 10.0, -0.5, 2.0]], requires_grad=True)
        loss = h61.clipped_loss(pred, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(pred.grad)

        # Single -2.0 gate-failure swing from +3.5 to -2.0 (raw delta -5.5) is clipped to -ROBUST_CAP (-1.0)
        clipped_delta = h61.clip_context_gain(-2.0, 3.5)
        self.assertAlmostEqual(clipped_delta, -h61.ROBUST_CAP, places=7)

        pools = make_synthetic_pools(num_families=41, contexts_per_family=2, pool_size=10)
        cids = sorted(pools)
        inc_ctx = {cid: (4.0 if cid == "ctx_00_00" else 0.20) for cid in cids}
        cand_ctx = {cid: (-2.0 if cid == "ctx_00_00" else 0.25) for cid in cids}
        clipped_fam_gains = h61.compute_clipped_family_gains(cids, pools, cand_ctx, inc_ctx)
        self.assertAlmostEqual(clipped_fam_gains["fam_00"], (-1.0 + 0.05) / 2.0, places=6)

        gains_by_seed = {seed: clipped_fam_gains for seed in h61.SEEDS}
        adm = h61.paired_family_admission(gains_by_seed)
        self.assertGreater(adm["mean_lcb95_one_sided"], h61.ADMISSION_LCB)
        self.assertTrue(adm["admitted"])

    def test_redteam_f_disk_guard_94_5_percent_and_write_time_retryable_infra_void(self):
        """(f) Start cells only when statvfs < 94.5%; write-time disk guard is a retryable infra-void (same cell)."""
        def stat_945(_):
            return FakeStatvfs(f_blocks=1000, f_bavail=55, f_frsize=1024**3)

        with self.assertRaises(h61.RetryableDiskInfraVoid) as ctx_err:
            h61.check_disk_guard(stage="cell_start", statvfs_fn=stat_945)
        self.assertTrue(ctx_err.exception.retryable)
        self.assertTrue(ctx_err.exception.same_cell)
        self.assertFalse(ctx_err.exception.stop_wave)

        def stat_944(_):
            return FakeStatvfs(f_blocks=1000, f_bavail=56, f_frsize=1024**3)

        usage = h61.check_disk_guard(stage="cell_start", statvfs_fn=stat_944)
        self.assertLess(usage["used_percent"], 94.5)

        cells = [("S1", 12), ("S1", 13)]
        call_log: list[tuple[str, int, int]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def run_cell(arm: str, seed: int, attempt: int) -> dict:
                call_log.append((arm, seed, attempt))
                write_stat = stat_945 if (arm == "S1" and seed == 12 and attempt == 1) else stat_944
                out_file = tmp_path / f"{arm}_{seed}.json"
                h61.atomic_json(out_file, {"arm": arm, "seed": seed, "attempt": attempt}, statvfs_fn=write_stat)
                return {"saved": str(out_file)}

            wave_res = h61.run_cells_with_disk_retry(cells, run_cell, statvfs_fn=stat_944, max_attempts_per_cell=3)
            self.assertFalse(wave_res["wave_stopped"])
            self.assertEqual(len(wave_res["completed_cells"]), 2)
            self.assertEqual(len(wave_res["infra_void_retries"]), 1)
            self.assertEqual(wave_res["infra_void_retries"][0]["stage"], "write")
            self.assertEqual(call_log, [("S1", 12, 1), ("S1", 12, 2), ("S1", 13, 1)])

    def test_dope_guard_before_and_after_every_round_and_r882_copy_contract(self):
        """Verify DOPE guard runs before/after every round, adapter-only inventory, and r882 copy staging."""
        class FakeParam:
            def __init__(self, count: int, grad: bool = True):
                self.requires_grad = grad
                self._count = count

            def numel(self):
                return self._count

        names = h61.validate_trainable_inventory([("adapter_10tok.layer.weight", FakeParam(h61.TRAINABLE_PARAMS))])
        self.assertEqual(names, ("adapter_10tok.layer.weight",))
        with self.assertRaisesRegex(RuntimeError, "only non-anchor"):
            h61.validate_trainable_inventory([("dope.backbone.weight", FakeParam(h61.TRAINABLE_PARAMS))])

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_live = Path(tmpdir) / "flywheel-full-20261030" / "ckpt.pt"
            fake_live.parent.mkdir(parents=True)
            fake_live.write_bytes(b"test")
            with self.assertRaisesRegex(RuntimeError, "live xbabe1"):
                h61.stage_r882_copy(fake_live, Path(tmpdir) / "scratch")

    def test_h57_timing_projection_fits_within_60_min_cell_wall_cap(self):
        """Verify the pre-outcome GPU budget projection from H57 timing fits within <= 60 min (3600 s)."""
        proj = h61.project_gpu_budget_from_h57_timing()
        self.assertTrue(proj["fits_within_60_min_cap"])
        self.assertLessEqual(proj["projected_total_seconds"], 3600.0)
        self.assertAlmostEqual(proj["projected_total_seconds"], 2577.332, places=2)
        with self.assertRaisesRegex(RuntimeError, "exceeds cap"):
            h61.project_gpu_budget_from_h57_timing(cap_seconds=1000.0)

    def test_fifo_queue_two_waiters_and_requeuing_holder_served_in_arrival_order_and_dead_pid_pruned(self):
        """Two simulated waiters + one re-queuing holder -> served in arrival order; dead-PID ticket pruned; no starvation."""
        self.assertEqual(h61_wave.LOCK_POLL_SECONDS, 30.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lock_file = tmp_path / "qualification.lock"
            queue_dir = tmp_path / "qualification.queue"
            data_root = tmp_path / "jope-data"
            status_log = data_root / "h61/h61-wave/H61_STATUS.log"

            live_pids = {1001, 2002, 3003}

            def is_alive(pid: int) -> bool:
                return pid in live_pids

            def preflight_ok() -> dict:
                return {
                    "disk_used_percent": 90.0,
                    "host_cpu_percent": 15.0,
                    "mem_available_kib": 64 * 1024 * 1024,
                }

            with mock.patch.object(h61_wave, "launch_disk_guard_error", return_value=None):
                # 1. Enqueue a dead-PID ticket at epoch_ns=50 (older than everyone) -> must be pruned automatically
                dead_ticket = fifo_mod.enqueue_ticket(queue_dir, pid=999999, wave="DEAD", epoch_ns=50)
                self.assertTrue(dead_ticket.exists())

                # 2. Holder (PID=1001, wave=H61) arrives at epoch_ns=100 and acquires lock for cell 1
                service_order: list[str] = []
                holder_sleeps_1: list[float] = []
                h1, _, _, _ = h61_wave.acquire_lock(
                    status_log,
                    0.0,
                    after_release=False,
                    wave_name="H61",
                    pid=1001,
                    epoch_ns=100,
                    lock_path=lock_file,
                    queue_dir=queue_dir,
                    data_root=data_root,
                    flock_fn=lambda handle, flags: None,
                    sleep_fn=holder_sleeps_1.append,
                    gpu_free_fn=lambda: 24_000,
                    preflight_fn=preflight_ok,
                    is_alive_fn=is_alive,
                )
                h1.close()
                service_order.append("holder_cell1")
                self.assertFalse(dead_ticket.exists(), "dead-PID ticket must be pruned")
                self.assertEqual(holder_sleeps_1, [])

                # 3. While holder_cell1 was running, waiter_1 (PID=2002, epoch_ns=200) and waiter_2 (PID=3003, epoch_ns=300)
                #    enqueued tickets in qualification.queue.
                t_waiter1 = fifo_mod.enqueue_ticket(queue_dir, pid=2002, wave="H57", epoch_ns=200)
                t_waiter2 = fifo_mod.enqueue_ticket(queue_dir, pid=3003, wave="H59", epoch_ns=300)

                # 4. Holder releases cell 1 and re-queues for cell 2 at the tail (epoch_ns=400).
                #    While waiter_1 and waiter_2 are still in queue ahead of epoch_ns=400, holder MUST wait (30s poll)
                #    until waiter_1 and waiter_2 are served in arrival order!
                holder_sleeps_2: list[float] = []

                def simulate_queue_progression(sleep_sec: float) -> None:
                    holder_sleeps_2.append(sleep_sec)
                    if len(holder_sleeps_2) == 1:
                        # On poll 1, waiter_1 is oldest live ticket -> waiter_1 acquires lock and removes its ticket
                        ok1, oldest1 = fifo_mod.is_oldest_live_ticket(t_waiter1, queue_dir, is_alive_fn=is_alive)
                        self.assertTrue(ok1)
                        self.assertEqual(oldest1[1], 2002)
                        service_order.append("waiter_1")
                        fifo_mod.remove_ticket(t_waiter1)
                    elif len(holder_sleeps_2) == 2:
                        # On poll 2, waiter_2 is oldest live ticket -> waiter_2 acquires lock and removes its ticket
                        ok2, oldest2 = fifo_mod.is_oldest_live_ticket(t_waiter2, queue_dir, is_alive_fn=is_alive)
                        self.assertTrue(ok2)
                        self.assertEqual(oldest2[1], 3003)
                        service_order.append("waiter_2")
                        fifo_mod.remove_ticket(t_waiter2)

                h2, _, _, _ = h61_wave.acquire_lock(
                    status_log,
                    0.0,
                    after_release=True,
                    wave_name="H61",
                    pid=1001,
                    epoch_ns=400,
                    lock_path=lock_file,
                    queue_dir=queue_dir,
                    data_root=data_root,
                    flock_fn=lambda handle, flags: None,
                    sleep_fn=simulate_queue_progression,
                    gpu_free_fn=lambda: 24_000,
                    preflight_fn=preflight_ok,
                    is_alive_fn=is_alive,
                )
                h2.close()
                service_order.append("holder_cell2")

                self.assertEqual(
                    service_order,
                    ["holder_cell1", "waiter_1", "waiter_2", "holder_cell2"],
                )
                self.assertEqual(holder_sleeps_2, [30.0, 30.0])
                self.assertEqual(list(queue_dir.iterdir()), [])

    def test_legacy_waiter_status_log_backoff_at_least_45_seconds(self):
        """After releasing a cell, if another *_STATUS.log has WAIT qualification.lock in the last 90s, back off >=45s."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lock_file = tmp_path / "qualification.lock"
            queue_dir = tmp_path / "qualification.queue"
            data_root = tmp_path / "jope-data"
            h57_log = data_root / "h57/h57-wave/H57_STATUS.log"
            h61_log = data_root / "h61/h61-wave/H61_STATUS.log"
            h57_log.parent.mkdir(parents=True, exist_ok=True)
            h61_log.parent.mkdir(parents=True, exist_ok=True)

            fixed_now = datetime(2026, 10, 31, 4, 55, 0, tzinfo=timezone.utc)
            recent_wait_ts = (fixed_now - timedelta(seconds=20)).isoformat()
            h57_log.write_text(
                f"{recent_wait_ts} — WAIT qualification.lock busy; holder PID=3178624; cumulative_wait=490.0s\n"
            )

            sleeps: list[float] = []
            with mock.patch.object(h61_wave, "launch_disk_guard_error", return_value=None):
                handle, _, _, _ = h61_wave.acquire_lock(
                    h61_log,
                    0.0,
                    after_release=True,
                    wave_name="H61",
                    pid=3178624,
                    epoch_ns=500,
                    lock_path=lock_file,
                    queue_dir=queue_dir,
                    data_root=data_root,
                    flock_fn=lambda h, flags: None,
                    sleep_fn=sleeps.append,
                    gpu_free_fn=lambda: 24_000,
                    preflight_fn=lambda: {
                        "disk_used_percent": 90.0,
                        "host_cpu_percent": 15.0,
                        "mem_available_kib": 64 * 1024 * 1024,
                    },
                    is_alive_fn=lambda _: True,
                    now_utc_fn=lambda: fixed_now,
                )
                handle.close()

            self.assertEqual(len(sleeps), 1)
            self.assertGreaterEqual(sleeps[0], 45.0)

    def test_coordinator_disk_guard_same_cell_retry_limit_two(self):
        """Verify run_wave.py retries DiskGuardInfraVoid on the SAME cell up to MAX_DISK_GUARD_RETRIES=2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            status_log = tmp_path / "H61_STATUS.log"
            out_dir = tmp_path / "outcomes/cells/S1/seed12"
            log_path = tmp_path / "logs/S1-seed12.log"
            calls = {"count": 0}

            def fake_run_locked_cell(key, command, log, timeout, status_path, wait_total):
                calls["count"] += 1
                out_dir.mkdir(parents=True, exist_ok=True)
                if calls["count"] <= 2:
                    raise h61_wave.DiskGuardInfraVoid("writer disk guard tripped", wait_seconds=0.5, cell_wall_seconds=10.0)
                return {"key": key, "status": "complete"}, 1.0, 25.0

            with mock.patch.object(h61_wave, "run_locked_cell", side_effect=fake_run_locked_cell):
                cell, total_wait, wall, void_wall = h61_wave.run_cell_with_disk_guard_retries(
                    "S1/seed12",
                    ["python3"],
                    log_path,
                    timeout=300.0,
                    status_path=status_log,
                    wait_total=0.0,
                    output_path=out_dir,
                )
            self.assertEqual(calls["count"], 3)
            self.assertEqual(cell["disk_guard_retry_count"], 2)
            self.assertAlmostEqual(void_wall, 20.0, places=5)
            self.assertAlmostEqual(wall, 25.0, places=5)

    def test_bradley_terry_elo_3_seeds_family_grouped_median_and_ci95_no_auc(self):
        """Verify 3-seed family-grouped Bradley-Terry Elo vs r882 reports median + 95% CI, never AUC, and ties fail."""
        pools = make_synthetic_pools(num_families=8, contexts_per_family=2, pool_size=10)
        cids = sorted(pools)
        inc_preds = {cid: np.linspace(1.0, 2.0, 10) for cid in cids}
        cand_preds = {}
        for cid in cids:
            row = np.linspace(1.0, 2.0, 10)
            row[9] = -1.0  # best unrevealed candidate -> +0.5 gain
            cand_preds[cid] = row

        scores_by_seed = {}
        for seed in h61.SEEDS:
            arm_preds = {
                "S0": inc_preds,   # exact tie with r882 -> must fail admission
                "S1": cand_preds,  # positive gain -> admitted
                "S2": cand_preds,
                "S3": cand_preds,
            }
            scores_by_seed[seed] = h61.score_fold_b_arms(pools, inc_preds, arm_preds, eval_mode="unrev")

        decisions = h61.admit_arms(scores_by_seed)
        self.assertFalse(decisions["S0"]["admitted"])
        self.assertTrue(decisions["S0"]["is_tie"])
        self.assertEqual(decisions["S0"]["elo_median_across_seeds"], 1500.0)

        for arm in ("S1", "S2", "S3"):
            adm = decisions[arm]
            self.assertTrue(adm["admitted"])
            self.assertEqual(adm["primary_ranking_metric"], "bradley_terry_elo_family_grouped")
            self.assertFalse(adm["auc_reported"])
            self.assertGreater(adm["elo_median_across_seeds"], 1500.0)
            self.assertEqual(len(adm["elo_ci95_across_seeds"]), 2)
            self.assertGreater(adm["elo_family_bootstrap_median"], 1500.0)
            self.assertEqual(len(adm["elo_family_bootstrap_ci95"]), 2)

        planned = h61.plan_terminal_reads(arm for arm in h61.ARMS if decisions[arm]["admitted"])
        self.assertEqual(planned, tuple((arm, s) for arm in ("S1", "S2", "S3") for s in h61.SEEDS))
        with self.assertRaisesRegex(RuntimeError, "forbidden unless the arm is admitted"):
            h61.terminal_read_allowed("S0", 12, ("S1", "S2", "S3"), ())
        with self.assertRaisesRegex(RuntimeError, "duplicate H61 terminal DL-Val"):
            h61.terminal_read_allowed("S1", 12, ("S1", "S2", "S3"), (("S1", 12),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
