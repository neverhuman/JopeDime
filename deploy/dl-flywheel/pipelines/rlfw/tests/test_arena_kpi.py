"""Hand-derived arena metrics and parent-cluster acceptance fixtures."""

from dataclasses import replace
import math
from pathlib import Path
import random
import tempfile
import unittest

from rlfw.arena.game_spec import ArmSpec, ContextSpec, GameSpec
from rlfw.arena.kpi import (
    SlotResult,
    arm_metrics,
    cluster_stat,
    context_units,
    outcome_score,
    paired_difference,
    ranking_metrics,
)

SHA = "a" * 64


def spec_at(root, contexts=5):
    return GameSpec(
        "toy",
        str(root),
        str(root / "runs/toy"),
        str(root / "state/arena.sqlite"),
        tuple(
            ContextSpec(f"c{i}", f"l{i}", "development", SHA, SHA)
            for i in range(contexts)
        ),
        (
            ArmSpec("hwm", None, SHA, SHA, SHA),
            ArmSpec("uniform_trie", None, SHA, SHA, SHA),
            ArmSpec("trie_sample:toy", SHA, SHA, SHA, SHA),
        ),
        (1,),
        SHA,
        SHA,
        SHA,
        SHA,
    )


def slots(context, lineage, arm, qs, censored=()):
    return [
        SlotResult(
            context,
            lineage,
            arm,
            1,
            i + 1,
            f"{i + 1:064x}",
            "censored"
            if i + 1 in censored
            else "program_failed"
            if q is None
            else "qualified_complete",
            None if i + 1 in censored or q is None else round(q * 1_000_000),
            q is not None and i + 1 not in censored,
            spec_sha256=SHA,
            receipt_sha256=SHA if q is not None and i + 1 not in censored else None,
        )
        for i, q in enumerate(qs)
    ]


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_roundtrip_and_drift(self):
        s = spec_at(self.root)
        s.freeze()
        s.verify_frozen()
        self.assertEqual(GameSpec.from_dict(s.to_dict()).sha256, s.sha256)
        with self.assertRaises(ValueError):
            replace(s, game_id="changed").freeze()
        p = self.root / "runs/toy/game_spec.json"
        p.write_text(p.read_text().replace('"tau":0.01', '"tau":0.02'))
        with self.assertRaises(ValueError):
            s.verify_frozen()

    def test_rejects_wrong_roles_paths_namespace_and_budget(self):
        with self.assertRaises(ValueError):
            ContextSpec("c", "l", "train", SHA, SHA)
        s = spec_at(self.root)
        for changes in (
            {"state_path": str(self.root / "outside.sqlite")},
            {"out_dir": str(self.root / "state/output")},
            {"budget": 9},
            {"publish_prefix": "service/results/"},
            {"benchmark_valid": True},
            {
                "state_path": str(
                    self.root / "flywheel_neural_continuous_20261009_v1/state/x.sqlite"
                )
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(s, **changes)
        outside = self.root / "elsewhere"
        outside.mkdir()
        (self.root / "state").symlink_to(outside)
        with self.assertRaises(ValueError):
            replace(s, state_path=str(self.root / "state/x.sqlite"))

    def test_pairing_binds_roster_order_cache_seeds_and_support(self):
        s = spec_at(self.root)
        s.require_pair(
            replace(
                s,
                game_id="challenger",
                out_dir=str(self.root / "runs/challenger"),
                arms=(
                    s.arms[0],
                    replace(
                        s.arms[2],
                        name="trie_sample:challenger",
                        checkpoint_sha256="b" * 64,
                    ),
                ),
            )
        )
        for changes in (
            {"contexts": tuple(reversed(s.contexts))},
            {"seeds": (2,)},
            {"cache_snapshot_sha256": "b" * 64},
        ):
            with self.assertRaises(ValueError):
                s.require_pair(replace(s, **changes))


class KPITests(unittest.TestCase):
    def test_margin_boundary_empty_and_lineage_macro(self):
        self.assertEqual(outcome_score(1.01, 1.0), ("TIE", 0.5))
        self.assertEqual(outcome_score(0.99, 1.0), ("TIE", 0.5))
        self.assertEqual(outcome_score(1.011, 1.0), ("WIN", 1.0))
        self.assertEqual(outcome_score(None, 1.0), ("LOSS", 0.0))
        data = (
            slots("a", "one", "X", [1.2] * 8)
            + slots("b", "one", "X", [None] * 8)
            + slots("c", "two", "X", [1.0] * 8)
        )
        result = arm_metrics(context_units(data, {"a": 1.0, "b": 1.0, "c": 1.0}))
        self.assertEqual(result["win_rate"]["mean"], 0.5)
        self.assertEqual(result["strict_win"]["mean"], 0.25)
        self.assertEqual(result["tie"]["mean"], 0.5)
        self.assertEqual(result["loss"]["mean"], 0.25)
        self.assertEqual(result["empty_v"]["mean"], 0.25)
        self.assertAlmostEqual(result["regret"]["mean"], -0.1)

    def test_unknowns_are_censored_and_never_point_losses(self):
        data = slots("a", "one", "X", [None] * 8, censored=(8,)) + slots(
            "b", "two", "X", [1.2] * 8
        )
        result = arm_metrics(context_units(data, {"a": 1.0, "b": 1.0}))
        self.assertEqual(result["win_rate"]["mean"], 1.0)
        self.assertEqual(result["censoring_bounds"], [0.5, 1.0])
        self.assertEqual(result["loss"]["mean"], 0.0)

    def test_best_curve_validity_and_no_hwm(self):
        u = context_units(
            slots("a", "one", "X", [None, 0.4, 0.6, None, None, None, None, 0.9]), {}
        )[0]
        self.assertEqual(u["best_q"], {1: None, 2: 0.4, 4: 0.6, 8: 0.9})
        self.assertIsNone(u["score"])
        m = arm_metrics([u])
        self.assertEqual(m["best_q@8"]["mean"], 0.9)
        self.assertEqual(m["valid@1"]["mean"], 0.0)
        self.assertEqual(m["no_comparable_hwm"], 1)

    def test_cluster_bootstrap_does_not_count_siblings_as_clusters(self):
        base = {"one": [0.0], "two": [1.0], "three": [0.5]}
        inflated = {**base, "one": [0.0] * 1000}
        x, y = cluster_stat(base), cluster_stat(inflated)
        self.assertEqual(x["mean"], y["mean"])
        self.assertEqual(x["ci95"], y["ci95"])
        self.assertEqual(y["lineages"], 3)

    def test_known_effect_bootstrap_coverage_500_panels(self):
        rng = random.Random(20261009)
        covered = 0
        for _ in range(500):
            panel = {str(i): [rng.gauss(0.1, 0.3)] for i in range(54)}
            interval = cluster_stat(panel)["ci95"]
            covered += interval[0] <= 0.1 <= interval[1]
        self.assertGreaterEqual(covered / 500, 0.92)
        self.assertLessEqual(covered / 500, 0.98)

    def test_paired_difference_keeps_common_parent_draws(self):
        a = context_units(
            slots("a", "one", "A", [1.2] * 8) + slots("b", "two", "A", [1.0] * 8),
            {"a": 1.0, "b": 1.0},
        )
        b = context_units(
            slots("a", "one", "B", [1.0] * 8) + slots("b", "two", "B", [0.8] * 8),
            {"a": 1.0, "b": 1.0},
        )
        d = paired_difference(a, b)
        self.assertEqual(d["score"]["mean"], 0.5)
        self.assertEqual(d["score"]["ci95"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            paired_difference(a, b[:1])

    def test_ranking_sources_are_separate_and_measured_only(self):
        data = [
            {
                "source": source,
                "role": "development",
                "context_id": "a",
                "lineage_id": "one",
                "program_sha256": f"{i:064x}",
                "outcome": "qualified_complete",
                "qualified": True,
                "receipt_sha256": SHA,
                "q_grid": round(q * 1_000_000),
                "predicted_q": pred,
            }
            for source in ("archive", "fresh")
            for i, q, pred in ((1, 0.0, 2.0), (2, 1.0, 1.0), (3, 2.0, 0.0))
        ]
        result = ranking_metrics(data)
        self.assertEqual(result["archive"]["pairwise_accuracy"]["mean"], 0.0)
        self.assertEqual(result["fresh"]["kendall_tau_b"]["mean"], -1.0)
        self.assertEqual(result["fresh"]["top1_hit"]["mean"], 0.0)
        with self.assertRaises(ValueError):
            ranking_metrics([{**data[0], "q_grid": math.nan}])
        with self.assertRaises(ValueError):
            ranking_metrics([{**data[0], "qualified": False}])

    def test_score_schema_refuses_unqualified_or_failure_numbers(self):
        with self.assertRaises(ValueError):
            replace(slots("a", "one", "X", [1.0] * 8)[0], qualified=False)
        with self.assertRaises(ValueError):
            replace(slots("a", "one", "X", [None] * 8)[0], q_grid=0.0)
        with self.assertRaises(TypeError):
            SlotResult(
                "a", "one", "X", 1, 1, SHA, "qualified_complete", predicted_q=0.9
            )


if __name__ == "__main__":
    unittest.main()
