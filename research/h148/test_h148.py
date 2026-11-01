"""Data-free controls for selection leakage, family inference and strict fallback."""

import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "h148_statistics", Path(__file__).with_name("analysis.py")
)
stat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stat)


def rows(n=30):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i // 3:02}") for i in range(n)]


class H148Controls(unittest.TestCase):
    def test_nested_family_isolation_and_once_only_assessment(self):
        seen = []
        for outer in range(5):
            fit, select, assess = stat.partition(rows(), outer)
            groups = [
                {r["family_id"] for r in group} for group in (fit, select, assess)
            ]
            self.assertFalse(
                groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2]
            )
            self.assertEqual(set.union(*groups), {r["family_id"] for r in rows()})
            seen += [r["context_sha256"] for r in assess]
        self.assertEqual(sorted(seen), sorted(r["context_sha256"] for r in rows()))
        self.assertEqual(stat.folds(rows()), stat.folds(rows()[::-1]))

    def test_duplicate_context_rejected(self):
        r = rows()
        r[1]["context_sha256"] = r[0]["context_sha256"]
        with self.assertRaises(ValueError):
            stat.folds(r)

    def test_tie_and_margin_reject(self):
        for gain in [0.0, 0.0009, 0.001, -0.1]:
            values = np.zeros((30, 5))
            values[:, 1:] = gain
            self.assertEqual(stat.choose(rows(), values, 1)["selected_step"], 0)

    def test_negative_bootstrap_lower_bound_rejects_positive_mean(self):
        values = np.zeros((30, 5))
        values[:3, 1] = 1.0
        values[3:, 1] = -0.05
        result = stat.choose(rows(), values, 1)
        self.assertGreater(result["gain"], 0.001)
        self.assertLess(result["ci95"][0], 0)
        self.assertEqual(result["selected_step"], 0)

    def test_earliest_best_and_positive_lower_bound_accept(self):
        values = np.zeros((30, 5))
        values[:, 1:] = 0.02
        result = stat.choose(rows(), values, 1)
        self.assertEqual(result["selected_step"], 10)
        self.assertGreater(result["ci95"][0], 0)

    def test_weighting_is_equal_family(self):
        _, fm = stat.family_matrix(np.array([[0.0], [0.0], [6.0]]), ["a", "a", "b"])
        self.assertEqual(float(fm.mean()), 3.0)

    def test_incomplete_or_nonfinite_snapshot_refused(self):
        for v in [np.zeros((30, 4)), np.full((30, 5), np.nan)]:
            with self.assertRaises(ValueError):
                stat.choose(rows(), v, 1)

    def test_selection_does_not_consume_assessment(self):
        fit, select, assess = stat.partition(rows(), 0)
        v = np.zeros((len(select), 5))
        v[:, 2] = 0.01
        first = stat.choose(select, v, 1)
        for r in assess:
            r["utility"] = 1e9
        self.assertEqual(first, stat.choose(select, v, 1))
        self.assertNotIn(
            "assess", __import__("inspect").signature(stat.choose).parameters
        )

    def test_seed_and_plan_restrictions(self):
        orders = {f"c{i}": np.arange(200) for i in range(5)}
        plan = stat.batch_plan(set(orders), orders, stat.SEEDS[0])
        self.assertEqual(len(plan), 40)
        self.assertTrue(all(set(cids) <= set(orders) for _, cids, _ in plan))
        self.assertTrue(all(len(ix) == 192 for _, _, ixs in plan for ix in ixs))
        with self.assertRaises(ValueError):
            stat.batch_plan(set(orders), orders, 0)

    def test_neutral_anchored_elo_and_complete_seed_inference(self):
        panel = np.ones((3, 30, 3))
        result = stat.summarize(rows(), panel)
        self.assertEqual(result["median_elo"], [1500.0] * 3)
        self.assertEqual(result["contrasts"]["XFIT_minus_VALSEL"]["ci95"], [0.0, 0.0])
        decisions = [
            {"XFIT": {"selected_step": 0}, "VALSEL": {"selected_step": 0}}
        ] * 15
        self.assertEqual(stat.verdict(result, decisions), "RULE_NEUTRAL")
        with self.assertRaises(ValueError):
            stat.summarize(rows(), panel[:2])

    def test_canonical_bt_matches_scalar_reference(self):
        wins = np.array([[0.0, 3.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
        strengths = np.ones(3)
        games = wins + wins.T
        for _ in range(400):
            strength_new = (wins.sum(1) + 0.02) / (
                (games + 0.02 * (1 - np.eye(3)))
                / np.maximum(strengths[:, None] + strengths[None, :], 1e-12)
            ).sum(1)
            strengths = np.clip(strength_new, 1e-6, 1e6)
            strengths /= strengths[0]
        np.testing.assert_allclose(
            stat.bt_ratings(wins, 0), 1500 + 400 * np.log10(strengths), atol=1e-12
        )

    def test_resource_limits_and_pressure(self):
        spec = importlib.util.spec_from_file_location(
            "h148_monitor", Path(__file__).with_name("monitor.py")
        )
        monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor)
        ok = dict(
            rss=1_000_000_000,
            threads=2,
            free=100_000_000_000,
            scratch=10,
            load=10.0,
            cores=32,
            available=20 * 1024**3,
        )
        self.assertEqual(monitor.classify(ok)[0], "run")
        for field, value in [
            ("rss", 6_000_000_000),
            ("threads", 5),
            ("free", 39_999_999_999),
            ("scratch", 1_000_000_000),
        ]:
            self.assertEqual(monitor.classify(dict(ok, **{field: value}))[0], "abort")
        self.assertEqual(monitor.classify(dict(ok, load=28.0))[0], "pause")


if __name__ == "__main__":
    unittest.main(verbosity=2)
