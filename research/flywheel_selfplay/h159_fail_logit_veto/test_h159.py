"""Data-free unit suite for H159: TRAIN-calibrated fail-logit veto at selection time."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import monitor
import runtime as rt
import science as stat


class H159Tests(unittest.TestCase):
    def test_registered_constants(self):
        self.assertEqual(stat.SEEDS, (159101, 159102, 159103))
        self.assertEqual(stat.FOLD_SEED, 159140)
        self.assertEqual(stat.BOOT_SEED, 159141)
        self.assertEqual(stat.GRID_T, (0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
        self.assertEqual(stat.ARMS, ("R882", "VETO_FIX", "VETO_CV"))

    def test_select_with_veto_plain_active_and_fallback(self):
        cost = np.array([-1.5, -1.2, -0.8], dtype=np.float64)
        # Candidate 0 has high fail_logit (+2.0 -> prob ~0.88), candidate 1 has low (-2.0 -> prob ~0.12)
        logits = np.array([2.0, -2.0, 0.0], dtype=np.float64)

        plain = stat.select_with_veto(cost, logits, None)
        self.assertEqual(plain["choice"], 0)
        self.assertFalse(plain["changed"])
        self.assertFalse(plain["fallback"])

        vetoed = stat.select_with_veto(cost, logits, 0.7)
        self.assertEqual(vetoed["choice"], 1)
        self.assertEqual(vetoed["plain_choice"], 0)
        self.assertTrue(vetoed["changed"])
        self.assertFalse(vetoed["fallback"])
        self.assertEqual(vetoed["vetoed_count"], 1)

        # If all candidates have fail_prob >= t, fall back to plain argmin (choice 0)
        all_high = np.array([3.0, 2.5, 4.0], dtype=np.float64)
        fb = stat.select_with_veto(cost, all_high, 0.5)
        self.assertEqual(fb["choice"], 0)
        self.assertFalse(fb["changed"])
        self.assertTrue(fb["fallback"])
        self.assertEqual(fb["vetoed_count"], 3)

    def test_select_with_veto_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            stat.select_with_veto([], [], 0.5)
        with self.assertRaises(ValueError):
            stat.select_with_veto([1.0, np.nan], [0.0, 0.0], 0.5)
        with self.assertRaises(ValueError):
            stat.select_with_veto([1.0, 2.0], [0.0, np.inf], 0.5)
        with self.assertRaises(ValueError):
            stat.select_with_veto([1.0, 2.0], [0.0, 0.0], 0.0)

    def test_fit_threshold_tie_breaks_toward_largest_t(self):
        fam_utils = {t: {"f0": 1.0, "f1": 1.2} for t in stat.GRID_T}
        res = stat.fit_threshold_on_families(fam_utils, ["f0", "f1"])
        self.assertEqual(res["selected_t"], 0.95)

        # When t=0.70 strictly wins, select 0.70
        fam_utils[0.7]["f0"] = 1.5
        res2 = stat.fit_threshold_on_families(fam_utils, ["f0", "f1"])
        self.assertEqual(res2["selected_t"], 0.7)

    def test_folds_deterministic_and_disjoint(self):
        rows = [{"family_id": f"fam_{i:03d}"} for i in range(205)]
        g1 = stat.folds(rows, seed=159140)
        g2 = stat.folds(rows, seed=159140)
        self.assertEqual(g1, g2)
        self.assertEqual([len(g) for g in g1], [41, 41, 41, 41, 41])
        self.assertEqual(len({f for g in g1 for f in g}), 205)

    def test_analyze_h159_rule_neutral_when_all_fail_probs_below_0_5(self):
        train_records = []
        for i in range(25):
            train_records.append(
                dict(
                    context_sha256=f"tr_{i}",
                    family_id=f"tf_{i % 10}",
                    total_cost=np.array([-1.2, -0.9, -0.5]),
                    fail_logit=np.array([-3.0, -2.5, -2.0]),
                    fitness=np.array([-1.1, -0.8, -0.4]),
                    gate=np.array([True, True, True]),
                )
            )
        val_records = []
        for i in range(19):
            val_records.append(
                dict(
                    context_sha256=f"va_{i}",
                    family_id=f"vf_{i}",
                    total_cost=np.array([-1.3, -1.0, -0.4]),
                    fail_logit=np.array([-4.0, -3.0, -2.0]),
                    fitness=np.array([-1.15, -0.9, -0.3]),
                    gate=np.array([True, True, True]),
                )
            )
        out = stat.analyze_h159(train_records, val_records)
        self.assertEqual(out["verdict"], "RULE_NEUTRAL")
        self.assertEqual(out["r882_verdict"], "NO_ADVANCE")
        self.assertTrue(out["rule_neutral"])
        self.assertTrue(out["rule_neutral_across_entire_grid"])
        self.assertEqual(out["primary_veto_fix_dl_val"]["vs_r882"]["median_gain"], 0.0)

    def test_analyze_h159_advance_when_veto_rescues_cliff_contexts(self):
        # Construct records where candidate 0 is a -2 cliff with fail_logit=+2.0 (prob ~0.88)
        # and candidate 1 is valid (+1.2 utility) with fail_logit=-2.0 (prob ~0.12).
        train_records = []
        for i in range(25):
            train_records.append(
                dict(
                    context_sha256=f"tr_{i}",
                    family_id=f"tf_{i % 10}",
                    total_cost=np.array([-1.5, -1.1, -0.4]),
                    fail_logit=np.array([2.0, -2.0, -2.0]),
                    fitness=np.array([-1.4, -1.2, -0.5]),
                    gate=np.array([False, True, True]),
                )
            )
        val_records = []
        for i in range(19):
            val_records.append(
                dict(
                    context_sha256=f"va_{i}",
                    family_id=f"vf_{i}",
                    total_cost=np.array([-1.5, -1.1, -0.4]),
                    fail_logit=np.array([2.0, -2.0, -2.0]),
                    fitness=np.array([-1.4, -1.2, -0.5]),
                    gate=np.array([False, True, True]),
                )
            )
        out = stat.analyze_h159(train_records, val_records)
        self.assertEqual(out["verdict"], "ADVANCE")
        self.assertEqual(out["r882_verdict"], "ADVANCE")
        self.assertFalse(out["rule_neutral"])
        self.assertGreater(
            out["primary_veto_fix_dl_val"]["vs_r882"]["median_gain"], 3.0
        )
        self.assertEqual(out["oracle_info"]["dl_val"]["avoidable_on_grid_count"], 19)
        self.assertEqual(out["oracle_info"]["dl_val"]["avoidable_at_any_t_count"], 19)

    def test_monitor_classify_and_reserved_block_disk_used(self):
        ok = dict(
            rss=3_000_000_000,
            threads=3,
            free=100_000_000_000,
            disk_used=0.90,
            scratch=100_000_000,
            load=30.0,
            available=32 * 1024**3,
        )
        self.assertEqual(monitor.classify(ok), ("run", None))
        self.assertEqual(
            monitor.classify(dict(ok, rss=6_000_000_000)), ("abort", "RSS")
        )
        self.assertEqual(
            monitor.classify(dict(ok, threads=5)), ("abort", "TOTAL_THREADS")
        )
        self.assertEqual(
            monitor.classify(dict(ok, free=39_999_999_999)), ("abort", "DISK")
        )
        self.assertEqual(
            monitor.classify(dict(ok, disk_used=0.95)), ("abort", "DISK_USED")
        )
        self.assertEqual(
            monitor.classify(dict(ok, scratch=3_000_000_000)), ("abort", "SCRATCH")
        )
        self.assertEqual(
            monitor.classify(dict(ok, load=85.1)), ("abort", "HOST_PRESSURE")
        )
        self.assertEqual(
            monitor.classify(dict(ok, available=16 * 1024**3 - 1)),
            ("abort", "HOST_PRESSURE"),
        )

        # Reserved blocks test: f_blocks=1000, f_bfree=70, f_bavail=40 -> 930 / 970 >= 0.95
        statvfs = SimpleNamespace(
            f_blocks=1000,
            f_bfree=70,
            f_bavail=40,
            f_frsize=1_000_000_000,
        )
        proc = SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=123),
            num_threads=lambda: 2,
        )
        with (
            patch("monitor.os.statvfs", return_value=statvfs),
            patch("monitor.psutil.Process", return_value=proc),
            patch(
                "monitor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=32 * 1024**3),
            ),
            patch("monitor.os.getloadavg", return_value=(10.0, 10.0, 10.0)),
            patch("monitor.Path.rglob", return_value=iter(())),
        ):
            s = monitor.sample(".")
        self.assertEqual(s["free"], 40_000_000_000)
        self.assertAlmostEqual(s["disk_used"], 930 / 970)
        self.assertEqual(monitor.classify(s), ("abort", "DISK_USED"))

    def test_audit_allows_launch_and_blocks_protected_or_unallowed_paths(self):
        out_dir = Path("/home/ubuntu/jope-data/h159/test_custody/run")
        launch_path = Path("/home/ubuntu/jope-data/h159/test_custody/launch.json")
        allowed_input = Path("/home/ubuntu/jope-data/h159/inputs/round-882.pt")
        audit = rt.Audit({str(launch_path), str(allowed_input)}, out_dir)

        # Allowed reads
        audit.check("open", (str(launch_path), "r", 0))
        audit.check("open", (str(allowed_input), "rb", 0))
        # Allowed write inside output directory
        audit.check("open", (str(out_dir / "metrics.json"), "w", os.O_WRONLY))
        # Allowed PyTorch /tmp JIT stub write
        audit.check("open", ("/tmp/torch_jit_stub.py", "w", os.O_WRONLY))

        # Refused protected path
        with self.assertRaises(RuntimeError):
            audit.check("open", ("/home/ubuntu/jope-data/sealed/test.npz", "r", 0))
        # Refused unallowed jope-data read
        with self.assertRaises(RuntimeError):
            audit.check("open", ("/home/ubuntu/jope-data/h50/other.json", "r", 0))
        # Refused write to allowed input
        with self.assertRaises(RuntimeError):
            audit.check("open", (str(allowed_input), "w", os.O_WRONLY))


if __name__ == "__main__":
    unittest.main()
