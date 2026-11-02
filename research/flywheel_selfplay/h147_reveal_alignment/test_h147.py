"""Data-free and synthetic controls for H147 matched-support reveal alignment, 5-fold CV, common support S, masked-anchor loss/gradients, intersection Holm m=2 gates, and hard-stop guards."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
from monitor import classify, environment
from runner import (
    baseline_gate,
    evaluate,
    score_context_from_costs,
    selected_and_catalog_check,
    verify,
)
from runtime import (
    BOUND_FILES,
    HERE,
    INPUT,
    MIXED_H58_CACHE,
    TRAIN_CACHE,
    VAL_CACHE,
    ABSOLUTE_DEADLINE_UNIX,
    Audit,
    assert_frozen_anchor,
    atomic_claim,
    bind_source,
    build,
    check_frozen,
    eval_batch_cache_key,
    guards,
    load_elo_tracker,
    loss,
    revealed_loss_and_grad_diagnostics,
    tensor_hash,
    validate_eval_cache_key,
)
from science import (
    ARMS,
    BOOT_SEED,
    CANDIDATE_ARMS,
    FOLD_SEED,
    HOLM_M,
    REVEAL_K,
    SEEDS,
    SLATE_SIZE,
    TOP_K_FITNESS,
    batch_plan,
    bradley_terry_elo_summary,
    build_common_support_slate,
    contrast,
    decide,
    folds,
    stable_fitness_order,
)


def rows(n=205):
    return [
        dict(
            context_sha256=f"c{i}",
            family_id=f"f{i}",
            dl_split="DL-Train",
            source_split="train",
        )
        for i in range(n)
    ]


class H147Tests(unittest.TestCase):
    def test_registered_arms_and_constants(self):
        self.assertEqual(SEEDS, (147101, 147102, 147103))
        self.assertEqual((FOLD_SEED, BOOT_SEED), (147140, 147141))
        self.assertEqual(ARMS, ("SUPPORT_CTRL", "STORAGE_REVEAL", "STORAGE_MASK"))
        self.assertEqual(CANDIDATE_ARMS, ("STORAGE_REVEAL", "STORAGE_MASK"))
        self.assertEqual(HOLM_M, 2)
        self.assertEqual(SLATE_SIZE, 192)
        self.assertEqual(REVEAL_K, 4)
        self.assertEqual(TOP_K_FITNESS, 48)

    def test_common_support_slate_and_permutations(self):
        rng = np.random.default_rng(147999)
        fitness = np.linspace(10.0, -5.0, 250, dtype=np.float64)
        # Put best fitness (lowest cost) at indices [50, 60, 70, 80]
        fitness[[50, 60, 70, 80]] = [-20.0, -19.0, -18.0, -17.0]
        order = stable_fitness_order(fitness)
        self.assertEqual(list(order[:4]), [50, 60, 70, 80])

        slate = build_common_support_slate(order, rng)
        ix_support = slate["ix_support_ctrl"]
        ix_storage = slate["ix_storage"]
        self.assertEqual(ix_support.shape, (192,))
        self.assertEqual(ix_storage.shape, (192,))
        # Identical 192-candidate set S
        self.assertEqual(set(ix_support.tolist()), set(ix_storage.tolist()))
        self.assertEqual(len(set(ix_support.tolist())), 192)
        # Both include A = {0, 1, 2, 3} and T_48 = set(order[:48])
        self.assertTrue({0, 1, 2, 3}.issubset(set(ix_support.tolist())))
        self.assertTrue(set(order[:48].tolist()).issubset(set(ix_support.tolist())))

        # SUPPORT_CTRL reveals B = 4 best-fitness members of S in slots 0..3
        self.assertEqual(list(ix_support[:4]), [50, 60, 70, 80])
        # Remaining 188 positions of SUPPORT_CTRL are in non-decreasing fitness order
        rem_support_f = fitness[ix_support[4:]]
        self.assertTrue(np.all(np.diff(rem_support_f) >= -1e-12))

        # STORAGE_REVEAL / STORAGE_MASK reveals A = [0, 1, 2, 3] in slots 0..3
        self.assertEqual(list(ix_storage[:4]), [0, 1, 2, 3])
        # Remaining 188 positions of STORAGE_REVEAL are in non-decreasing fitness order
        rem_storage_f = fitness[ix_storage[4:]]
        self.assertTrue(np.all(np.diff(rem_storage_f) >= -1e-12))

    def test_common_support_pool_size_refusal(self):
        rng = np.random.default_rng(147999)
        with self.assertRaises(ValueError):
            stable_fitness_order(np.zeros(191, dtype=np.float64))
        with self.assertRaises(ValueError):
            build_common_support_slate(np.arange(191, dtype=np.int64), rng)

    def test_paired_batch_plan_common_support(self):
        f_a = np.linspace(5.0, -5.0, 240, dtype=np.float64)
        f_b = np.linspace(3.0, -3.0, 240, dtype=np.float64)
        orders = {"a": stable_fitness_order(f_a), "b": stable_fitness_order(f_b)}
        plan1 = batch_plan(["a", "b"], orders, SEEDS[0], 0)
        plan2 = batch_plan(["b", "a"], orders, SEEDS[0], 0)
        self.assertEqual(len(plan1), 40)
        for p1, p2 in zip(plan1, plan2, strict=True):
            self.assertEqual(p1["round"], p2["round"])
            self.assertEqual(p1["round_step"], p2["round_step"])
            self.assertEqual(p1["cids"], p2["cids"])
            for s1, s2, r1, r2 in zip(
                p1["ix_support_ctrl"],
                p2["ix_support_ctrl"],
                p1["ix_storage"],
                p2["ix_storage"],
                strict=True,
            ):
                self.assertTrue(np.array_equal(s1, s2))
                self.assertTrue(np.array_equal(r1, r2))
                self.assertEqual(set(s1.tolist()), set(r1.tolist()))
                self.assertEqual(list(r1[:4]), [0, 1, 2, 3])

    def test_masked_anchor_zero_grad_and_target_renormalization(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(147888)
        pred_raw = torch.tensor(
            rng.standard_normal((4, 192)), dtype=torch.float32, requires_grad=True
        )
        fit = torch.tensor(rng.standard_normal((4, 192)), dtype=torch.float32)
        gate = torch.ones((4, 192), dtype=torch.bool)
        gate[:, 180:] = False

        # Unmasked loss (SUPPORT_CTRL / STORAGE_REVEAL) has nonzero gradient wrt pred[:, :4]
        val_unmasked = loss(pred_raw, fit, gate, mask_revealed=False)
        val_unmasked.backward()
        self.assertTrue(torch.isfinite(val_unmasked))
        self.assertGreater(float(pred_raw.grad[:, :4].abs().max().item()), 0.0)

        # Masked loss (STORAGE_MASK) has exact 0.0 gradient wrt pred[:, :4]
        pred_masked = pred_raw.detach().clone().requires_grad_(True)
        val_masked = loss(pred_masked, fit, gate, mask_revealed=True)
        val_masked.backward()
        self.assertTrue(torch.isfinite(val_masked))
        self.assertEqual(float(pred_masked.grad[:, :4].abs().max().item()), 0.0)
        self.assertGreater(float(pred_masked.grad[:, 4:].abs().max().item()), 0.0)

        # Mutating pred[:, :4] must not change masked loss at all
        pred_mut = pred_raw.detach().clone()
        pred_mut[:, :4] += 1000.0
        val_masked_mut = loss(pred_mut, fit, gate, mask_revealed=True)
        self.assertAlmostEqual(
            float(val_masked.item()), float(val_masked_mut.item()), places=7
        )

        # Check revealed_loss_and_grad_diagnostics helper and target softmax sum=1 over 188 unrevealed slots
        diag_mask = revealed_loss_and_grad_diagnostics(
            pred_raw, fit, gate, mask_revealed=True
        )
        self.assertEqual(diag_mask["revealed_loss_share"], 0.0)
        self.assertEqual(diag_mask["revealed_pred_grad_norm2_share"], 0.0)
        diag_unmask = revealed_loss_and_grad_diagnostics(
            pred_raw, fit, gate, mask_revealed=False
        )
        self.assertGreater(diag_unmask["revealed_loss_share"], 0.0)
        self.assertGreater(diag_unmask["revealed_pred_grad_norm2_share"], 0.0)

    def test_outcome_blind_choice_and_cost_utility_sign(self):
        # Raw fitness is cost (lower is better); native utility is -raw_fitness
        costs = np.full(192, 0.8, dtype=np.float64)
        costs[:6] = [0.1, 0.2, 0.3, 0.4, 0.9, 0.05]
        raw_fitness = np.full(192, 6.0, dtype=np.float64)
        raw_fitness[:6] = [1.0, 2.0, -3.0, 4.0, 5.0, -2.5]
        gate = np.ones(192, dtype=bool)
        gate[6] = False
        res = score_context_from_costs(costs, dict(fitness=raw_fitness, gate=gate))
        # Lowest predicted cost is index 5 (cost 0.05, raw_fitness -2.5 -> utility +2.5)
        self.assertEqual(res["choice"], 5)
        self.assertAlmostEqual(res["utility"], 2.5)
        self.assertFalse(res["invalid"])
        self.assertFalse(res["on_anchor"])
        self.assertEqual(res["unrev_top1_choice"], 5)
        self.assertAlmostEqual(res["unrev_top1_utility"], 2.5)
        self.assertAlmostEqual(res["u4_top4_unrev_utility"], 2.5)
        # Union of anchors [0,1,2,3] and top-4 unrevealed has best fitness at anchor 2 (-3.0 -> utility +3.0)
        self.assertAlmostEqual(res["u4_union_anchors_and_top4_unrev_utility"], 3.0)

        # Outcome-blind verification: mutating raw_fitness while holding predicted costs fixed
        # MUST NOT change the selected candidate index
        mut_fitness = raw_fitness.copy()
        mut_fitness[5] = 999.0
        mut_fitness[0] = -999.0
        res_mut = score_context_from_costs(costs, dict(fitness=mut_fitness, gate=gate))
        self.assertEqual(res_mut["choice"], res["choice"])
        self.assertEqual(res_mut["unrev_top1_choice"], res["unrev_top1_choice"])

    def test_ties_reject_and_intersection_holm_m2(self):
        r = rows(205)
        c_zero = contrast(r, np.zeros((3, 205)), np.zeros(205))
        self.assertEqual(c_zero["median_gain"], 0.0)
        self.assertEqual(c_zero["seeds_beating"], 0)
        d_zero = decide(
            {a: c_zero for a in ARMS},
            {a: c_zero for a in CANDIDATE_ARMS},
        )
        self.assertEqual(d_zero["verdict"], "NO_ADVANCE")
        self.assertIsNone(d_zero["best_advancing_arm"])

        # Positive spread on both vs_support_ctrl and vs_r882 advances
        c_pos = contrast(r, np.full((3, 205), 0.01), np.zeros(205))
        d_pos = decide(
            {
                "SUPPORT_CTRL": c_zero,
                "STORAGE_REVEAL": c_pos,
                "STORAGE_MASK": c_zero,
            },
            {"STORAGE_REVEAL": c_pos, "STORAGE_MASK": c_zero},
        )
        self.assertEqual(d_pos["verdict"], "ADVANCE")
        self.assertEqual(d_pos["best_advancing_arm"], "STORAGE_REVEAL")

        # Passing vs_r882 but failing vs_support_ctrl must NOT advance (intersection test)
        d_half = decide(
            {
                "SUPPORT_CTRL": c_zero,
                "STORAGE_REVEAL": c_pos,
                "STORAGE_MASK": c_zero,
            },
            {"STORAGE_REVEAL": c_zero, "STORAGE_MASK": c_zero},
        )
        self.assertEqual(d_half["verdict"], "NO_ADVANCE")
        self.assertIsNone(d_half["best_advancing_arm"])

        # Passing on only 1/3 seeds must NOT advance
        c_1seed = contrast(
            r,
            np.array([[0.01] * 205, [-0.001] * 205, [-0.001] * 205]),
            np.zeros(205),
        )
        d_1seed = decide(
            {
                "SUPPORT_CTRL": c_zero,
                "STORAGE_REVEAL": c_1seed,
                "STORAGE_MASK": c_zero,
            },
            {"STORAGE_REVEAL": c_1seed, "STORAGE_MASK": c_zero},
        )
        self.assertEqual(d_1seed["verdict"], "NO_ADVANCE")

    def test_bradley_terry_elo_summary(self):
        elo_mod = load_elo_tracker()
        r = rows(205)
        u_r882 = np.zeros(205, dtype=np.float64)
        u_arms = {
            "SUPPORT_CTRL": np.full((3, 205), 0.001, dtype=np.float64),
            "STORAGE_REVEAL": np.full((3, 205), 0.010, dtype=np.float64),
            "STORAGE_MASK": np.full((3, 205), 0.020, dtype=np.float64),
        }
        bt = bradley_terry_elo_summary(r, u_r882, u_arms, elo_mod)
        self.assertEqual(bt["r882"]["median_elo"], 1500.0)
        self.assertGreater(
            bt["STORAGE_MASK"]["median_elo"],
            bt["STORAGE_REVEAL"]["median_elo"],
        )
        self.assertGreater(
            bt["STORAGE_REVEAL"]["median_elo"],
            bt["SUPPORT_CTRL"]["median_elo"],
        )
        self.assertGreater(
            bt["SUPPORT_CTRL"]["median_elo"],
            1500.0,
        )

    def test_fold_partition_matches_fold_manifest(self):
        r = rows(205)
        f = folds(r)
        self.assertEqual([len(x) for x in f], [41] * 5)
        self.assertEqual(len(set(sum(f, []))), 205)

        # Verify against real train_only_selected_1219.json and fold_manifest.json
        train, val, catalog_receipt = selected_and_catalog_check()
        self.assertEqual(len(train), 1219)
        self.assertEqual(len(val), 91)
        self.assertEqual(catalog_receipt["known_catalog_sealed_id_overlap"], 0)
        real_folds = folds(train)
        fm = json.loads((HERE / "fold_manifest.json").read_text())
        for k, fam_list in enumerate(real_folds):
            self.assertEqual(fam_list, fm["folds"][k]["assessment_families"])
            assess_rows = [row for row in train if row["family_id"] in set(fam_list)]
            self.assertEqual(
                [row["context_sha256"] for row in assess_rows],
                fm["folds"][k]["assessment_context_ids"],
            )

    def test_missing_and_nonfinite_refusal(self):
        with self.assertRaises(ValueError):
            contrast(rows(), np.zeros((2, 205)), np.zeros(205))
        with self.assertRaises(ValueError):
            contrast(rows(), np.full((3, 205), np.nan), np.zeros(205))
        with self.assertRaises(ValueError):
            decide({}, {})

    def test_resource_boundaries_all_abort(self):
        s = dict(
            rss=1,
            threads=2,
            free=40_000_000_000,
            disk_used=0.94,
            scratch=0,
            load=85.0,
            available=16 * 1024**3,
        )
        self.assertEqual(classify(s), ("run", None))
        for k, v, expected_why in [
            ("rss", 6_000_000_000, "RSS"),
            ("threads", 5, "TOTAL_THREADS"),
            ("free", 39_999_999_999, "DISK"),
            ("disk_used", 0.95, "DISK"),
            ("scratch", 3_000_000_000, "SCRATCH"),
            ("load", 85.01, "LOAD"),
            ("available", 16 * 1024**3 - 1, "MEMORY"),
        ]:
            self.assertEqual(classify(dict(s, **{k: v})), ("abort", expected_why))

    def test_runtime_guards_and_atomic_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "claim.txt"
            atomic_claim(ledger, dict(study="H147"))
            self.assertTrue(ledger.exists())
            with self.assertRaises(FileExistsError):
                atomic_claim(ledger, dict(study="H147"))

            with (
                mock.patch("os.getpriority", return_value=19),
                mock.patch(
                    "shutil.disk_usage",
                    return_value=mock.Mock(free=50_000_000_000, total=100_000_000_000),
                ),
                mock.patch("os.getloadavg", return_value=(10.0, 10.0, 10.0)),
                mock.patch("runtime.tree_bytes", return_value=1000),
                mock.patch("runtime.time.time", return_value=ABSOLUTE_DEADLINE_UNIX - 1),
            ):
                res = guards(tmp)
                self.assertEqual(res["disk_free_bytes"], 50_000_000_000)

            with mock.patch("os.getpriority", return_value=10):
                with self.assertRaises(RuntimeError):
                    guards(tmp)

    def test_squash_compatible_bind_source_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            for idx, rel in enumerate(BOUND_FILES):
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"bound-content-{idx}\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "reviewed head"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            reviewed = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "squash commit"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            source = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
            prereg_sha = hashlib.sha256(
                (repo / BOUND_FILES[0]).read_bytes()
            ).hexdigest()
            receipt = bind_source(
                dict(reviewed_head=reviewed, source_commit=source),
                repo=repo,
                prereg_sha=prereg_sha,
            )
            self.assertEqual(receipt["reviewed_head"], reviewed)
            self.assertEqual(receipt["source_commit"], source)

            (repo / BOUND_FILES[2]).write_text("tampered\n")
            with self.assertRaises(RuntimeError):
                bind_source(
                    dict(reviewed_head=reviewed, source_commit=source),
                    repo=repo,
                    prereg_sha=prereg_sha,
                )

    def test_read_audit_and_mixed_cache_refusal(self):
        out_dir = Path("/home/ubuntu/jope_plan/evidence/H147/attempt_unit_test")
        a = Audit(
            [
                str(TRAIN_CACHE),
                str(VAL_CACHE),
            ],
            out_dir,
            forbidden_reads=(MIXED_H58_CACHE,),
        )
        a.check("open", (str(TRAIN_CACHE), "r", 0))
        a.check("open", (str(out_dir / "metrics.json"), "w", 0))
        a.check("open", ("/tmp/_remote_module_non_scriptable.py", "w", 0))

        for path, mode in [
            (str(MIXED_H58_CACHE), "r"),
            ("/home/ubuntu/jope-data/holdout/data", "r"),
            ("/home/ubuntu/jope-data/sealed/panel.json", "r"),
            ("/home/ubuntu/jope-data/meta-test/eval.json", "r"),
            ("/home/ubuntu/jope-data/meta_test/eval.json", "r"),
            ("/home/ubuntu/jope-data/outer-test/eval.json", "r"),
            ("/home/ubuntu/jope-data/outer_test/eval.json", "r"),
            ("/home/ubuntu/jope-data/qualification.lock", "r"),
            ("/home/ubuntu/JopeDime/data/flywheel_selfplay/live", "r"),
            ("/home/ubuntu/jope-data/h50/input/pools/unknown.npz", "r"),
            ("/home/ubuntu/JopeDime/README.md", "w"),
            ("/home/ubuntu/jope_plan/AGENT_CHAT.md", "w"),
            ("/home/ubuntu/jope_plan/evidence/H147/launch.json", "w"),
            (str(TRAIN_CACHE), "w"),
            ("/tmp/sealed/leak.json", "w"),
            (str(out_dir / "holdout" / "leak.json"), "w"),
        ]:
            with self.assertRaises(
                RuntimeError, msg=f"expected refusal for {path} ({mode})"
            ):
                a.check("open", (path, mode, 0))

    def test_anchor_frozen_and_cache_bitwise_parity(self):
        torch.set_num_threads(1)
        ckpt = Path("/home/ubuntu/jope-data/h147/inputs/round-882.pt")
        net, guard, names, frozen, fused = build(ckpt)
        anchor_sha = assert_frozen_anchor(net, names, net._anchor_sha256)
        self.assertEqual(anchor_sha, net._anchor_sha256)
        self.assertEqual(len(names), 47)

        train, val, _ = selected_and_catalog_check()
        sample_rows = val[:8]
        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in sample_rows:
            cid = r["context_sha256"]
            with np.load(INPUT / "pools" / f"{cid}.npz", allow_pickle=False) as z:
                program = z["program"][:256].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"][:256].astype(np.float64),
                    gate=z["gate"][:256].astype(bool),
                )
        with np.load(VAL_CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            tok = z["tokens_10x32"]
            mask = z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
                if c in pools
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool)
                for i, c in enumerate(ids)
                if c in pools
            }

        uncached = evaluate(
            net,
            pools,
            tokens,
            masks,
            sample_rows,
            split_name="DL-Val-Sample",
            use_cache=False,
            return_scores=True,
        )
        cached = evaluate(
            net,
            pools,
            tokens,
            masks,
            sample_rows,
            split_name="DL-Val-Sample",
            use_cache=True,
            return_scores=True,
        )
        self.assertEqual(uncached["values"], cached["values"])
        self.assertEqual(uncached["choices"], cached["choices"])
        for s_u, s_c in zip(uncached["scores"], cached["scores"], strict=True):
            self.assertTrue(np.array_equal(s_u, s_c))
        check_frozen(net, guard, names, frozen)

    def test_eval_cache_key_and_tensor_hash_and_verify(self):
        self.assertNotEqual(
            tensor_hash({"x": torch.tensor([0.0])}),
            tensor_hash({"x": torch.tensor([-0.0])}),
        )
        fake_pools = {
            "cid_a": dict(
                program=np.ones((8, 412), dtype=np.float32),
                fitness=np.linspace(-1.0, 1.0, 8, dtype=np.float64),
                gate=np.ones(8, dtype=bool),
            )
        }
        k1 = eval_batch_cache_key("DL-Val", ["cid_a"], [np.arange(8)], fake_pools)
        self.assertEqual(validate_eval_cache_key(k1), k1)
        with self.assertRaises(RuntimeError):
            validate_eval_cache_key(0)
        with self.assertRaises(RuntimeError):
            verify({"inputs": {str(TRAIN_CACHE): "0" * 64}})

    def test_baseline_exact_refusal(self):
        ref = {"utility": {"per_context": [1.0] * 91}, "best_utility": 1.0}
        self.assertEqual(baseline_gate([1.0] * 91, ref)["matches"], 91)
        with self.assertRaises(RuntimeError):
            baseline_gate([1.00000001] + [1.0] * 90, ref)

    def test_environment_one_thread(self):
        self.assertEqual(environment()["OMP_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
