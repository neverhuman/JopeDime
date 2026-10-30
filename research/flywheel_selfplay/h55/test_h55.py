#!/usr/bin/env python3
"""Focused, data-free tests for H55 admission and resource guardrails."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("h55_test_audit", HERE / "audit.py")
metrics = module("h55_test_metrics", HERE / "build_metrics.py")
wave = module("h55_test_wave", HERE / "run_wave.py")


class H55PolicyTests(unittest.TestCase):
    def test_tied_lcb_is_not_admitted(self):
        rows = []
        for arm in ("swa8", "ema25"):
            for seed in (12, 13, 14):
                rows.append({"arm": arm, "seed": seed,
                             "family_gain_vs_r882": {"f1": 0.0, "f2": 0.0}})
        result = metrics.family_cluster_admission(rows, "swa8")
        self.assertEqual(result["mean_lcb95"], 0.0)
        self.assertFalse(result["admitted"])

    def test_positive_margin_and_no_harm_can_admit(self):
        rows = [{"arm": "ema25", "seed": seed,
                 "family_gain_vs_r882": {f"f{i}": 0.02 for i in range(8)}}
                for seed in (12, 13, 14)]
        result = metrics.family_cluster_admission(rows, "ema25")
        self.assertGreater(result["mean_lcb95"], 0.001)
        self.assertTrue(result["admitted"])

    def test_cpu_rss_and_disk_limits_are_fail_closed(self):
        good = {"coordinator_preflight": {
            "child_max_rss_kib": audit.RSS_LIMIT_KIB - 1,
            "child_max_host_cpu_observed_percent": 85.0,
            "host_cpu_percent_before": 85.0,
            "disk_used_percent": 94.99,
            "disk_free_bytes": 2 * 1024**3,
        }}
        audit.audit_resources(good, "fixture")
        for key, value in (("child_max_rss_kib", audit.RSS_LIMIT_KIB),
                           ("child_max_host_cpu_observed_percent", 85.01),
                           ("disk_used_percent", 95.0),
                           ("disk_free_bytes", 1024**3 - 1)):
            bad = {"coordinator_preflight": dict(good["coordinator_preflight"], **{key: value})}
            with self.assertRaises(SystemExit, msg=key):
                audit.audit_resources(bad, "fixture")

    def test_terminal_read_limit_is_one_per_seed(self):
        chosen = wave.terminal_arm_by_preregistered_order({
            "arms": {"swa8": {"admitted": True}, "ema25": {"admitted": True}}
        })
        self.assertEqual(chosen, "swa8")
        self.assertIsNone(wave.terminal_arm_by_preregistered_order({"arms": {}}))
        reads = {(chosen, seed) for seed in (12, 13, 14)} if chosen else set()
        self.assertEqual(len(reads), 3)
        self.assertEqual({seed for _, seed in reads}, {12, 13, 14})


if __name__ == "__main__":
    unittest.main(verbosity=2)
