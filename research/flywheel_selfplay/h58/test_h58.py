#!/usr/bin/env python3
"""Unit tests for H58 multi-arm flywheel fixes (Arms A/B/C/D, t550)."""

from __future__ import annotations

import fcntl
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

# Support read-only sandboxes (e.g., codex exec --sandbox read-only) where
# torch.distributed.nn.jit.instantiator calls tempfile.TemporaryDirectory() on AdamW import.
try:
    tempfile.gettempdir()
except FileNotFoundError:
    class _InMemoryTempDir:
        def __init__(self, *args, **kwargs):
            self.name = "/dev/shm"
        def cleanup(self) -> None:
            pass
        def __enter__(self) -> str:
            return self.name
        def __exit__(self, exc_type, exc, tb) -> None:
            pass
    tempfile.TemporaryDirectory = _InMemoryTempDir  # type: ignore[misc,assignment]
    tempfile.tempdir = "/dev/shm"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay/h58"))
sys.path.insert(0, str(ROOT / "python/src"))

import prepare_cache as h58_prep  # noqa: E402
import runner as h58  # noqa: E402


def make_synthetic_pool(n: int = 256, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    prog = rng.normal(size=(n, 412)).astype(np.float32)
    proj16 = rng.normal(size=(n, 16)).astype(np.float32)
    fit = rng.uniform(-1.5, -0.1, size=n).astype(np.float64)
    # Make candidate 17 the true unrevealed oracle (-2.5)
    fit[17] = -2.50
    gate = np.ones(n, dtype=bool)
    # Make candidates 20..35 gate failures
    gate[20:36] = False
    bprior = rng.normal(loc=-0.5, scale=0.3, size=n).astype(np.float32)
    sprior = rng.uniform(-0.2, 0.4, size=(n, 2)).astype(np.float32)
    return {
        "rec": {"context_sha256": f"cid_{seed}", "family_id": f"fam_{seed % 5}"},
        "pids": [f"prog_{i}" for i in range(n)],
        "program": prog,
        "proj16": proj16,
        "fitness": fit,
        "logratio": fit.copy(),
        "gate": gate,
        "base_prior_cost": bprior,
        "sos_prior": sprior,
        "scale": max(float(np.std(fit[:4])), 0.10),
        "hard_negatives": set(),
    }


class TestH58Fixes(unittest.TestCase):
    def test_bug1_no_peek_anchor_and_slot_permutation(self) -> None:
        pool = make_synthetic_pool(256, seed=7)
        rng = np.random.default_rng(101)

        # In Arm A, slot 0 is deterministically the global best candidate (17), and top-48 are placed at 0..47
        slate_a = h58.sample_slate_arm(pool, 64, rng, arm="A")
        self.assertEqual(int(slate_a[0]), 17)
        self.assertEqual(len(slate_a), 64)
        order_all = np.argsort(pool["fitness"])
        self.assertEqual(list(slate_a[:48]), list(order_all[:48]))
        # Also verify slate_size=32 edge case in Arm A returns length 32
        slate_a_32 = h58.sample_slate_arm(pool, 32, rng, arm="A")
        self.assertEqual(len(slate_a_32), 32)
        self.assertEqual(list(slate_a_32), list(order_all[:32]))

        # In Arms B, C, D, slots 0..3 are ALWAYS [0, 1, 2, 3], and slots 4..63 are permuted
        first_unrev_slots = set()
        for draw in range(12):
            slate_b = h58.sample_slate_arm(pool, 64, np.random.default_rng(200 + draw), arm="B")
            self.assertEqual(list(slate_b[:4]), [0, 1, 2, 3])
            self.assertTrue(np.all(slate_b[4:] >= 4))
            self.assertIn(17, set(slate_b[4:].tolist()))
            first_unrev_slots.add(int(slate_b[4]))
        self.assertGreater(len(first_unrev_slots), 4, "unrevealed slots 4..M-1 must be permuted")

        # Verify loss gradient on revealed slots 0..3 is strictly zero in B/C/D and non-zero in A
        pred = torch.randn((4, 32), requires_grad=True)
        target = torch.randn((4, 32))
        gates = torch.ones((4, 32), dtype=torch.bool)
        gates[:, 10:14] = False

        for arm in ("B", "C", "D"):
            p = pred.detach().clone().requires_grad_(True)
            loss = h58.compute_loss_arm(arm, p, target, gates)
            loss.backward()
            self.assertIsNotNone(p.grad)
            self.assertAlmostEqual(float(p.grad[:, :4].abs().max()), 0.0, places=7)
            self.assertGreater(float(p.grad[:, 4:].abs().max()), 0.0)

        p_a = pred.detach().clone().requires_grad_(True)
        loss_a = h58.compute_loss_arm("A", p_a, target, gates)
        loss_a.backward()
        self.assertGreater(float(p_a.grad[:, :4].abs().max()), 0.0)

        # Verify evaluation argmin masks 0..3 in B/C/D even if row[0..3] has the lowest predicted cost
        pools = {"cid_7": pool}

        def fake_predict(cids, ixs):
            row = np.ones((len(cids), len(ixs[0])), dtype=np.float32)
            row[:, 0] = -999.0  # leaked anchor slot 0
            row[:, 17] = -5.0   # best unrevealed candidate 17
            return torch.as_tensor(row)

        eval_a = h58.evaluate_utility_arm(["cid_7"], pools, fake_predict, arm="A")
        eval_b = h58.evaluate_utility_arm(["cid_7"], pools, fake_predict, arm="B")
        # Arm B picks candidate 17 (the true oracle -> utility = 1.0), whereas Arm A picks leaked slot 0
        self.assertAlmostEqual(eval_b["mean"], 1.0, places=6)
        self.assertLess(eval_a["mean"], 0.90)

    def test_bug2_hard_negative_admission_and_sampling(self) -> None:
        pool_a = make_synthetic_pool(128, seed=11)
        pool_c = h58.clone_pools({"c": pool_a})["c"]

        non_hwm_rec = {
            "program_sha256": "novel_non_hwm_prog",
            "cv_gate_passed": True,
            "fitness": -0.20,  # worse than incumbent -2.50
            "logratio": -0.20,
            "feat412": [0.1] * 412,
            "proj16": [0.1] * 16,
            "base_prior_cost": -0.4,
            "sos_prior": [0.1, 0.25],
        }
        gate_fail_rec = {
            "program_sha256": "novel_gate_fail_prog",
            "cv_gate_passed": False,
            "fitness": 10.0,
            "logratio": 10.0,
            "feat412": [0.2] * 412,
            "proj16": [0.2] * 16,
            "base_prior_cost": -0.8,
            "sos_prior": [0.2, 0.25],
        }

        # Arm A/B rejects non-HWM and gate-failure candidates
        self.assertFalse(h58.admit_gp_candidate_to_pool(pool_a, non_hwm_rec, arm="B"))
        self.assertFalse(h58.admit_gp_candidate_to_pool(pool_a, gate_fail_rec, arm="B"))
        self.assertEqual(len(pool_a["pids"]), 128)

        # Execution failures (status != "completed") are NEVER admitted in ANY arm
        exec_fail_rec = dict(gate_fail_rec, program_sha256="exec_fail_prog", status="failed")
        for arm in ("A", "B", "C", "D"):
            self.assertFalse(h58.admit_gp_candidate_to_pool(pool_c, exec_fail_rec, arm=arm))
        self.assertEqual(len(pool_c["pids"]), 128)

        # Arm C/D admits completed non-HWM and completed CV gate failures and records their indices in pool["hard_negatives"]
        self.assertTrue(h58.admit_gp_candidate_to_pool(pool_c, non_hwm_rec, arm="C"))
        self.assertTrue(h58.admit_gp_candidate_to_pool(pool_c, gate_fail_rec, arm="C"))
        self.assertEqual(len(pool_c["pids"]), 130)
        self.assertIn(128, pool_c["hard_negatives"])
        self.assertIn(129, pool_c["hard_negatives"])

        # Seed initial hard negatives and verify C/D slate sampling includes them
        h58.seed_initial_hard_negatives_for_cd({"c": pool_c}, ["c"])
        slate_c = h58.sample_slate_arm(pool_c, 96, np.random.default_rng(55), arm="C")
        hn_in_slate = len(set(slate_c[4:].tolist()) & pool_c["hard_negatives"])
        self.assertGreaterEqual(hn_in_slate, 16)

        # Verify select_gp_proposal_indices preserves exploration within max_eval_per_ctx
        costs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        sel2 = h58.select_gp_proposal_indices(costs, seed=12345, max_eval_per_ctx=2, explore=True)
        self.assertEqual(len(sel2), 2)
        self.assertEqual(sel2[0], 0)
        self.assertIn(sel2[1], {2, 3, 4, 5}, "second pick under max_eval_per_ctx=2, explore=True must be an exploration pick from order[2:]")
        sel4 = h58.select_gp_proposal_indices(costs, seed=12345, max_eval_per_ctx=4, explore=True)
        self.assertEqual(len(sel4), 4)
        self.assertEqual(sel4[:2], [0, 1])
        self.assertTrue(all(idx in {2, 3, 4, 5} for idx in sel4[2:]))

    def test_rank3_shift_invariance_and_gradient_alignment(self) -> None:
        torch.manual_seed(99)
        pred = torch.randn((6, 48), dtype=torch.float32)
        target = torch.empty((6, 48), dtype=torch.float32).uniform_(-1.5, -0.1)
        gates = torch.ones((6, 48), dtype=torch.bool)
        gates[:, 36:] = False
        target[:, 36:] = 10.0

        # Add arbitrary per-context constant shifts delta [6, 1]
        delta = torch.tensor([[-12.5], [0.7], [4.2], [-3.1], [19.0], [-0.4]], dtype=torch.float32)
        loss_d_base = h58.compute_loss_arm("D", pred, target, gates)
        loss_d_shifted = h58.compute_loss_arm("D", pred + delta, target, gates)
        self.assertAlmostEqual(float(loss_d_base), float(loss_d_shifted), places=5)

        # Contrast with Arm B (smooth_l1 is NOT shift-invariant)
        loss_b_base = h58.compute_loss_arm("B", pred, target, gates)
        loss_b_shifted = h58.compute_loss_arm("B", pred + delta, target, gates)
        self.assertGreater(abs(float(loss_b_base) - float(loss_b_shifted)), 1.0)

        # Verify gradient direction on Arm D: oracle best passing slot gets positive dL/dpred
        # (so gradient descent decreases its predicted cost), and gate-failing slots get negative dL/dpred
        p_grad = torch.zeros((1, 24), dtype=torch.float32, requires_grad=True)
        t_grad = torch.linspace(-0.2, -1.0, 24).unsqueeze(0)
        # Put oracle best at slot 5 (index 5 >= 4) and gate failures at slots 20..23
        t_grad[0, 5] = -2.5
        g_grad = torch.ones((1, 24), dtype=torch.bool)
        g_grad[0, 20:] = False
        t_grad[0, 20:] = 10.0
        l_d = h58.compute_loss_arm("D", p_grad, t_grad, g_grad)
        l_d.backward()
        grad_row = p_grad.grad[0].detach().numpy()
        # Under gradient descent (pred <- pred - lr * grad), positive gradient decreases predicted cost
        # (improving rank), and negative gradient increases predicted cost (worsening rank).
        self.assertGreater(float(grad_row[5]), 0.0, "oracle best candidate must receive positive dL/dpred so its predicted cost decreases")
        self.assertLess(float(grad_row[20:].mean()), 0.0, "gate-failing candidates must receive negative dL/dpred so their predicted cost increases")

        # Verify equal-cost passing candidates in Arm D produce zero pairwise loss and zero opposing gradients
        p_tie = torch.zeros((1, 20), dtype=torch.float32, requires_grad=True)
        t_tie = torch.full((1, 20), -1.0, dtype=torch.float32)
        g_tie = torch.ones((1, 20), dtype=torch.bool)
        l_tie = h58.compute_loss_arm("D", p_tie, t_tie, g_tie)
        l_tie.backward()
        self.assertAlmostEqual(float(p_tie.grad.abs().max()), 0.0, places=6)

    def test_bug3_and_bug4_on_real_network(self) -> None:
        input_dir = Path.home() / "jope-data/h50/input"
        spec = importlib.util.spec_from_file_location("h58_test_fused", input_dir / "cpu_exact_fused.py")
        self.assertIsNotNone(spec)
        fused = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused)
        device = torch.device("cpu")
        net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")

        # Arm A: 29,211 trainable params (parent_delta_net trainable)
        tr_a = h58.reset_adapter_fold_clean(net, seed=12, freeze_parent_delta=False, scale=0.1)
        self.assertEqual(sum(p.numel() for p in tr_a), 29_211)
        self.assertEqual(h58.verify_dope_guard(net, "A"), h58.DOPE_SHA)
        active_snap_a = {
            name: p.detach().clone()
            for name, p in net.adapter_10tok.named_parameters()
            if not name.startswith("anchor.") and not name.startswith("parent_delta_net.")
        }

        # Arm B/C/D: 23,546 trainable params (parent_delta_net frozen), identical active init for same seed
        tr_b = h58.reset_adapter_fold_clean(net, seed=12, freeze_parent_delta=True, scale=0.1)
        self.assertEqual(sum(p.numel() for p in tr_b), 23_546)
        self.assertEqual(h58.verify_dope_guard(net, "B"), h58.DOPE_SHA)
        for name, p in net.adapter_10tok.named_parameters():
            if name.startswith("parent_delta_net."):
                self.assertFalse(p.requires_grad)
            elif not name.startswith("anchor."):
                self.assertTrue(p.requires_grad)
                self.assertTrue(torch.equal(p.detach(), active_snap_a[name]))

        # Verify Bug 3: Arm B predictor output differs from Arm A predictor output due to real ctx_feat/priors/scale
        pool = make_synthetic_pool(32, seed=42)
        pools = {"cid_42": pool}
        ctx_cache = {
            "cid_42": {
                "tokens_10x32": torch.randn((10, 32), dtype=torch.float32),
                "mask_10": torch.ones((10,), dtype=torch.bool),
                "ctx_feat_584": torch.randn((584,), dtype=torch.float32),
            }
        }
        net.eval()
        pred_a = h58.make_predictor_arm(net, pools, ctx_cache, device, "A")(["cid_42"], [np.arange(32)])
        pred_b = h58.make_predictor_arm(net, pools, ctx_cache, device, "B")(["cid_42"], [np.arange(32)])
        self.assertGreater(float((pred_a - pred_b).abs().max()), 1e-3)

        # Verify _ensure_jtt1_gz_for_pool fails closed when jtt1_gz_path or jtt1_gz_sha256 is missing/mismatched
        with self.assertRaises(RuntimeError):
            h58._ensure_jtt1_gz_for_pool("nonexistent_cid", make_synthetic_pool(32, seed=99))
        fixture_cid = "89033749285b8be2b0ded865cb81e244e65bd5da25e22e62dbc11ef1d71fb8c0"
        fixture_gz = Path.home() / "jope-data/h58/cache/jtt1_gz" / f"{fixture_cid}.jtt1.gz"
        fixture_gz_sha = h58.sha256(fixture_gz)
        bad_pool = make_synthetic_pool(32, seed=100)
        bad_pool["jtt1_gz_path"] = fixture_gz
        bad_pool["jtt1_gz_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            h58._ensure_jtt1_gz_for_pool("bad_sha_cid", bad_pool)

        # Verify end-to-end run_training_cell for all four arms (A, B, C, D) with live GP proposal + Rust evaluation
        base_pools = {}
        ctx_cache_multi = {}
        family_by_cid = {}
        for i in range(6):
            cid = f"cid_{i}"
            fam = f"fam_{i % 3}"
            p = make_synthetic_pool(64, seed=500 + i)
            p["rec"] = {"context_sha256": cid, "family_id": fam}
            p["jtt1_gz_path"] = fixture_gz
            p["jtt1_gz_sha256"] = fixture_gz_sha
            base_pools[cid] = p
            family_by_cid[cid] = fam
            ctx_cache_multi[cid] = {
                "tokens_10x32": torch.randn((10, 32), dtype=torch.float32),
                "mask_10": torch.ones((10,), dtype=torch.bool),
                "ctx_feat_584": torch.randn((584,), dtype=torch.float32),
            }
        journal_by_cid = {
            "cid_0": [
                {
                    "program_sha256": "j_non_hwm_0",
                    "cv_gate_passed": True,
                    "fitness": -0.30,
                    "logratio": -0.30,
                    "feat412": [0.1] * 412,
                    "proj16": [0.1] * 16,
                    "base_prior_cost": -0.5,
                    "sos_prior": [0.1, 0.25],
                }
            ]
        }
        fit_cids = ["cid_0", "cid_1", "cid_2", "cid_3"]
        eval_cids = ["cid_4", "cid_5"]
        for arm in ("A", "B", "C", "D"):
            res = h58.run_training_cell(
                arm=arm,
                seed=12,
                cell_id=1,
                net=net,
                base_pools=base_pools,
                fit_cids=fit_cids,
                eval_cids=eval_cids,
                family_by_cid=family_by_cid,
                ctx_cache=ctx_cache_multi,
                journal_by_cid=journal_by_cid,
                rounds=2,
                steps_per_round=1,
                batch_size=4,
                slate_size=32,
                device=device,
            )
            self.assertEqual(res["dope_sha256_before"], h58.DOPE_SHA)
            self.assertEqual(res["dope_sha256_after"], h58.DOPE_SHA)
            self.assertGreater(res["update_l2_from_init"], 0.0)
            self.assertEqual(res["trainable_params"], 29_211 if arm == "A" else 23_546)
            self.assertEqual(res["terminal_eval_sweeps"], 1)
            self.assertGreaterEqual(res["gp_proposer_calls"], 2)
            self.assertGreaterEqual(res["gp_evaluator_calls"], 2)
            self.assertGreaterEqual(res["gp_evaluated_completed"], 2)
            self.assertEqual(res["gp_execution_failed"], 0)
            if arm in ("A", "B"):
                self.assertEqual(res["journal_admitted"], 0)
                self.assertEqual(res["gp_admitted_gate_fails"] + res["gp_admitted_non_hwm"], 0)
            else:
                self.assertEqual(res["journal_admitted"], 1)
                self.assertGreater(res["gp_admitted"], 0)
                self.assertGreater(res["initial_hard_negatives"], 0)

        # Verify run_training_cell fails closed if native evaluations return only status="failed"
        real_eval_rust = h58.base.evaluator.evaluate_gp_batch_rust
        try:
            def _always_failed_eval(_gz_path, chosen, **_kwargs):
                return [
                    {
                        "program_sha256": c["program_sha256"],
                        "dsl": c["dsl"],
                        "status": "failed",
                        "cv_gate_passed": False,
                        "fitness": 10.0,
                        "logratio": 10.0,
                        "feat412": [0.0] * 412,
                        "proj16": [0.0] * 16,
                    }
                    for c in chosen
                ]
            h58.base.evaluator.evaluate_gp_batch_rust = _always_failed_eval
            with self.assertRaises(RuntimeError):
                h58.run_training_cell(
                    arm="C",
                    seed=12,
                    cell_id=1,
                    net=net,
                    base_pools=base_pools,
                    fit_cids=fit_cids,
                    eval_cids=eval_cids,
                    family_by_cid=family_by_cid,
                    ctx_cache=ctx_cache_multi,
                    journal_by_cid=journal_by_cid,
                    rounds=2,
                    steps_per_round=1,
                    batch_size=4,
                    slate_size=32,
                    device=device,
                )
        finally:
            h58.base.evaluator.evaluate_gp_batch_rust = real_eval_rust

        # Also test cold-cache feature extraction on `net`
        X = np.random.default_rng(7).normal(size=(64, 6)).astype(np.float64)
        y = np.random.default_rng(8).normal(size=64).astype(np.float64)
        ws = h58_prep.base.FusedLandmarkWorkspace()
        t10, m10, d629, lm48, ctx584 = h58_prep.extract_single_table_features(net, ws, X, y, device)
        self.assertEqual(t10.shape, (10, 32))
        self.assertEqual(m10.shape, (10,))
        self.assertEqual(d629.shape, (629,))
        self.assertEqual(lm48.shape, (48,))
        self.assertEqual(ctx584.shape, (584,))
        self.assertTrue(bool(m10[0]) and bool(m10[1]) and bool(m10[2]))

    def test_lock_check_and_single_pass_dual_eval(self) -> None:
        # Test verify_qualification_lock_held fails when unlocked, fails when held by an unrelated child
        # process, and passes ONLY when held by an open file descriptor of this process
        import subprocess
        lock_file = Path("/dev/shm/h58_test_qual.lock")
        lock_file.touch(exist_ok=True)
        try:
            with self.assertRaises(RuntimeError):
                h58.verify_qualification_lock_held(lock_file)

            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import fcntl, sys; f=open(sys.argv[1],'rb'); fcntl.flock(f, fcntl.LOCK_EX); print('locked', flush=True); sys.stdin.read(1)",
                    str(lock_file),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaises(RuntimeError):
                    h58.verify_qualification_lock_held(lock_file)
            finally:
                child.communicate("x", timeout=5)

            with lock_file.open("r+") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                h58.verify_qualification_lock_held(lock_file)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.unlink(missing_ok=True)

        # Test evaluate_utility_both_modes executes exactly 1 forward pass per batch
        pool = make_synthetic_pool(64, seed=77)
        pools = {"cid_77": pool}
        calls = []

        def counting_predict(cids, ixs):
            calls.append(len(cids))
            row = np.ones((len(cids), len(ixs[0])), dtype=np.float32)
            row[:, 0] = -999.0
            row[:, 17] = -5.0
            return torch.as_tensor(row)

        nat, unrev = h58.evaluate_utility_both_modes(["cid_77"], pools, counting_predict, arm="A")
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(unrev["mean"], 1.0, places=6)
        self.assertLess(nat["mean"], 0.90)

        # Test _spearman_rank_corr handles constant predictions (returns 0.0) and tied ranks
        self.assertEqual(h58._spearman_rank_corr(np.ones(8), np.arange(8, dtype=np.float64)), 0.0)
        self.assertEqual(h58._spearman_rank_corr(np.arange(8, dtype=np.float64), np.full(8, -2.5)), 0.0)
        tied_x = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64)
        tied_y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        self.assertAlmostEqual(
            h58._spearman_rank_corr(tied_x, tied_y),
            float(np.corrcoef([0.5, 0.5, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])[0, 1]),
            places=6,
        )

    def test_t537_bootstrap_and_elo(self) -> None:
        # Positive mean and non-negative median passes
        pos_gains = {f"fam_{i}": 0.025 for i in range(25)}
        res_pass = h58.bootstrap_t537(pos_gains, seed=123, draws=2000)
        self.assertTrue(res_pass["t537_gate_passed"])

        # Positive mean LCB95 but negative family median MUST FAIL t537
        skew_gains = {f"fam_{i}": (-0.005 if i < 14 else 0.15) for i in range(25)}
        res_skew = h58.bootstrap_t537(skew_gains, seed=123, draws=2000)
        self.assertGreater(res_skew["mean_lcb95"], 0.001)
        self.assertLess(res_skew["family_median"], 0.0)
        self.assertFalse(res_skew["t537_gate_passed"])

        # Elo panel check
        cids = [f"c_{i}" for i in range(10)]
        fams = {f"c_{i}": f"f_{i // 2}" for i in range(10)}
        scores = {
            "A": {c: 0.40 + (0.30 if i == 0 else 0.0) for i, c in enumerate(cids)},
            "B": {c: 0.50 + (0.25 if i == 1 else 0.0) for i, c in enumerate(cids)},
            "C": {c: 0.60 + (0.20 if i == 2 else 0.0) for i, c in enumerate(cids)},
            "D": {c: 0.70 for c in cids},
        }
        elo_out = h58.compute_elo_panel(cids, fams, scores, anchor="A", bootstrap_draws=100, seed=7)
        self.assertAlmostEqual(elo_out["ratings"]["A"], 1500.0, places=2)
        self.assertGreater(elo_out["ratings"]["D"], elo_out["ratings"]["C"])
        self.assertGreater(elo_out["ratings"]["C"], elo_out["ratings"]["B"])
        self.assertGreater(elo_out["ratings"]["B"], elo_out["ratings"]["A"])


if __name__ == "__main__":
    unittest.main()
