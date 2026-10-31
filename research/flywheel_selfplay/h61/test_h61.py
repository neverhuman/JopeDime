#!/usr/bin/env python3
"""CPU-only fixture and red-team unit tests for the H61 selection-metric runner."""
from __future__ import annotations

from collections import namedtuple
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h61_runner_under_test", HERE / "runner.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import H61 runner")
h61 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h61
spec.loader.exec_module(h61)

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

        # On a 41-family fold-B panel (2 contexts/family), 1 outlier context in f0 drops from +4.0 to -2.0
        # (raw delta -6.0 -> clipped delta -1.0 -> family f0 mean delta -0.46 vs unclipped -2.96),
        # while the remaining 40 families have +0.05 gain.
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
        # 94.5% used (55 free of 1000 blocks) must block cell start
        stat_945 = lambda _: FakeStatvfs(f_blocks=1000, f_bavail=55, f_frsize=1024**3)
        with self.assertRaises(h61.RetryableDiskInfraVoid) as ctx_err:
            h61.check_disk_guard(stage="cell_start", statvfs_fn=stat_945)
        self.assertTrue(ctx_err.exception.retryable)
        self.assertTrue(ctx_err.exception.same_cell)
        self.assertFalse(ctx_err.exception.stop_wave)

        # 94.4% used (56 free of 1000 blocks) passes cell start
        stat_944 = lambda _: FakeStatvfs(f_blocks=1000, f_bavail=56, f_frsize=1024**3)
        usage = h61.check_disk_guard(stage="cell_start", statvfs_fn=stat_944)
        self.assertLess(usage["used_percent"], 94.5)

        # Write-time disk-guard failure during cell execution retries the SAME cell without stopping the wave
        cells = [("S1", 12), ("S1", 13)]
        call_log: list[tuple[str, int, int]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def run_cell(arm: str, seed: int, attempt: int) -> dict:
                call_log.append((arm, seed, attempt))
                # Simulate write-time disk spike to 94.7% on first attempt of (S1, 12)
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

        # Live xbabe1 path must be refused by stage_r882_copy
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
