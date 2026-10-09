"""Research-only gate, alpha budget, hashed receipts and failure custody."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from rlfw.arena.game_spec import digest
from rlfw.arena.kpi import LABEL
from rlfw.promotion_gate import (
    alpha_schedule,
    bootstrap_interval,
    evaluate_panel,
    GateConfig,
    run_gate,
    signed_receipt,
    verify_receipt,
    panel_from_runs,
)

SHA = "a" * 64


def panel(delta=0.1):
    roster = [
        {
            "context_id": f"c{i}",
            "lineage_id": f"parent{i}",
            "role": "development",
            "identity_sha256": SHA,
            "hwm_program_sha256": SHA,
        }
        for i in range(54)
    ]
    baseline = []
    for c in roster:
        baseline.append(
            {
                "context_id": c["context_id"],
                "lineage_id": c["lineage_id"],
                "seed": 7,
                "hwm": 0.0,
                "censored": False,
                "positive_gain": 1.0,
                "score": 1.0,
                "valid8": 1.0,
                "qualified_completion_rate": 1.0,
                "wall_seconds": 1.0,
                "unknown_cost_attempts": 0,
            }
        )
    challenger = deepcopy(baseline)
    for u in challenger:
        u["positive_gain"] += delta
    return {
        "schema": "jope.rlfw-gate-panel/v1",
        "label": LABEL + "; cpu inference",
        "panel_sha256": digest({"contexts": roster, "seeds": (7,), "role_map": SHA}),
        "pairing_sha256": SHA,
        "game_spec_sha256s": [SHA, "b" * 64],
        "checkpoint_sha256s": ["c" * 64, "d" * 64],
        "role_map_sha256": SHA,
        "roster": roster,
        "seeds": (7,),
        "arms": ["challenger", "champion"],
        "challenger": challenger,
        "champion": baseline,
        "slot_receipt_sha256s": [SHA],
        "benchmark_valid": False,
    }


class GateMathTests(unittest.TestCase):
    def test_spend_has_six_positive_increments_summing_to_point_zero_five(self):
        schedule = alpha_schedule()
        self.assertEqual(len(schedule), 6)
        self.assertTrue(all(s["look_alpha"] > 0 for s in schedule))
        self.assertAlmostEqual(sum(s["look_alpha"] for s in schedule), 0.05, places=14)
        self.assertLess(schedule[0]["look_alpha"], schedule[-1]["look_alpha"])
        # Finite bootstrap tails cannot certify the first two looks.
        for look in (1, 2):
            self.assertEqual(evaluate_panel(panel(), look)["verdict"], "INCONCLUSIVE")
        self.assertEqual(evaluate_panel(panel(), 3)["verdict"], "PROMOTE")

    def test_interval_is_parent_macro_and_reproducible(self):
        one = bootstrap_interval({"large": 0.0, "small": 1.0})
        two = bootstrap_interval({"small": 1.0, "large": 0.0})
        self.assertEqual(one, two)
        self.assertEqual(one["mean"], 0.5)
        self.assertEqual(one["bootstrap"], 2000)

    def test_null_simulation_1000_panels_entire_schedule_at_most_five_percent(self):
        import numpy as np
        from rlfw.arena.kpi import cluster_draws

        rng = np.random.default_rng(7)
        draws = np.asarray(cluster_draws(54, 2000, 7))
        promoted = 0
        # Different data at every scheduled look; shared draws retain correlation.
        # Positive J8 on both sides, exact measured differences have true mean0.
        # Win/valid/cost gates pass, so this isolates the potentially binding test.
        lower_by_look = {}
        for s in alpha_schedule():
            if s["look_alpha"] / 2 * 2000 < 1:
                continue
            null = rng.uniform(-0.25, 0.25, size=(1000, 54))
            lower = []
            for values in null:
                replicas = values[draws].mean(axis=1)
                lower.append(float(np.quantile(replicas, s["look_alpha"] / 2)))
            lower_by_look[s["look"]] = np.asarray(lower)
        successes = np.stack([x > 0 for x in lower_by_look.values()]).any(axis=0)
        promoted = int(successes.sum())
        self.assertLessEqual(promoted, 50, f"{promoted}/1000 null schedules promoted")
        # Bind vectorized simulation to the production evaluator on one real panel.
        example = panel()
        for u, delta in zip(
            sorted(example["challenger"], key=lambda u: u["lineage_id"]), null[0]
        ):
            u["positive_gain"] = 1.0 + float(delta)
        actual = evaluate_panel(example, 6)["stats"]["positive_gain"]["spent"][
            "interval"
        ][0]
        self.assertAlmostEqual(actual, lower_by_look[6][0], places=14)
        print(
            f"null gate simulation: {promoted}/1000 schedules promoted; seed7,54parents,6looks"
        )

    def test_incomplete_unknown_censored_and_unqualified_are_inconclusive(self):
        for mutation in (
            lambda p: p["challenger"][0].update(censored=True),
            lambda p: [
                u.update(hwm=None) for u in (p["challenger"][0], p["champion"][0])
            ],
            lambda p: p["challenger"][0].update(
                wall_seconds=None, unknown_cost_attempts=1
            ),
            lambda p: [p[k].pop() for k in ("challenger", "champion")],
        ):
            p = panel()
            mutation(p)
            self.assertEqual(evaluate_panel(p, 3)["verdict"], "INCONCLUSIVE")

    def test_nonpositive_lift_validity_and_cost_guards_reject(self):
        self.assertEqual(evaluate_panel(panel(0.0), 3)["verdict"], "REJECT")
        p = panel()
        for u in p["challenger"]:
            u["valid8"] = 0.97
        self.assertEqual(evaluate_panel(p, 3)["verdict"], "REJECT")
        p = panel(0.01)
        for u in p["challenger"]:
            u["wall_seconds"] = 1.06
        self.assertEqual(evaluate_panel(p, 3)["verdict"], "REJECT")
        for u in p["challenger"]:
            u["positive_gain"] = 2.0
        self.assertEqual(evaluate_panel(p, 3)["verdict"], "PROMOTE")

    def test_pairing_mismatch_and_duplicate_are_hard_errors(self):
        for mutation in (
            lambda p: p["challenger"][0].update(hwm=0.1),
            lambda p: p["challenger"].append(p["challenger"][0]),
            lambda p: p["champion"].pop(),
        ):
            p = panel()
            mutation(p)
            with self.assertRaises(ValueError):
                evaluate_panel(p, 3)


class GateStateTests(unittest.TestCase):
    def setUp(self):
        campaign_clock = mock.patch(
            "rlfw.promotion_gate.time.time", return_value=1791536400.0
        )
        campaign_clock.start()
        self.addCleanup(campaign_clock.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = GateConfig(
            str(self.root), time.time() + 60, SHA, apply_research_pointer=True
        )

    def test_receipt_roundtrip_hash_and_tamper_refusal(self):
        signed = signed_receipt({"verdict": "INCONCLUSIVE", "look": 1})
        self.assertEqual(
            verify_receipt(json.loads(json.dumps(signed))), signed["payload"]
        )
        signed["payload"]["verdict"] = "PROMOTE"
        with self.assertRaises(ValueError):
            verify_receipt(signed)

    def test_looks_consumed_then_champion_and_anchor_move_in_one_immutable_version(
        self,
    ):
        p = panel()
        for look in (1, 2):
            result = run_gate(self.cfg, p)
            self.assertEqual(
                verify_receipt(result["gate_receipt"])["decision"]["verdict"],
                "INCONCLUSIVE",
            )
            self.assertFalse((self.root / "state/champion.json").exists())
        result = run_gate(self.cfg, p)
        gate = verify_receipt(result["gate_receipt"])
        self.assertEqual(gate["decision"]["verdict"], "PROMOTE")
        self.assertTrue(
            verify_receipt(result["application_receipt"])["pointer_applied"]
        )
        pointer_bytes = (self.root / "state/champion.json").read_bytes()
        pointer = verify_receipt(json.loads(pointer_bytes))
        self.assertEqual(pointer["search_champion_sha256"], "c" * 64)
        self.assertEqual(pointer["anchor_sha256"], "c" * 64)
        self.assertEqual(
            len(list((self.root / "state/pointer_versions").glob("*.json"))), 1
        )
        # Old-baseline results become stale; they never move the current pointer.
        result = run_gate(self.cfg, p)
        self.assertEqual(
            verify_receipt(result["gate_receipt"])["decision"]["verdict"],
            "INCONCLUSIVE",
        )
        self.assertEqual(
            pointer_bytes, (self.root / "state/champion.json").read_bytes()
        )
        run_gate(self.cfg, p)
        run_gate(self.cfg, p)
        with self.assertRaisesRegex(ValueError, "six monthly"):
            run_gate(self.cfg, p)
        self.assertEqual(
            len(list((self.root / "state/gates").rglob("*.reservation.json"))), 6
        )

    def test_dry_gate_no_pointer_and_campaign_stop_stays_inconclusive(self):
        cfg = replace(self.cfg, apply_research_pointer=False)
        for _ in range(3):
            result = run_gate(cfg, panel())
        self.assertEqual(
            verify_receipt(result["gate_receipt"])["decision"]["verdict"], "PROMOTE"
        )
        self.assertIsNone(result["application_receipt"])
        self.assertFalse((self.root / "state/champion.json").exists())
        (self.root / "state/STOP.json").write_text("{}")
        result = run_gate(self.cfg, panel())
        self.assertEqual(
            verify_receipt(result["gate_receipt"])["decision"]["verdict"],
            "INCONCLUSIVE",
        )
        self.assertFalse((self.root / "state/champion.json").exists())

    def test_interrupted_evaluation_consumes_look_before_result(self):
        with mock.patch(
            "rlfw.promotion_gate.evaluate_panel",
            side_effect=RuntimeError("fixture interruption"),
        ):
            with self.assertRaises(RuntimeError):
                run_gate(self.cfg, panel())
        result = run_gate(self.cfg, panel())
        self.assertEqual(verify_receipt(result["gate_receipt"])["look"], 2)
        self.assertEqual(
            len(list((self.root / "state/gates").rglob("*.reservation.json"))), 2
        )

    def test_corrupt_reservation_and_state_links_refuse(self):
        run_gate(self.cfg, panel())
        path = next((self.root / "state/gates").rglob("*.reservation.json"))
        path.chmod(0o600)
        body = json.loads(path.read_text())
        body["payload"]["look"] = 4
        path.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, "integrity"):
            run_gate(self.cfg, panel())

    def test_pointer_symlink_cannot_replace_other_evidence(self):
        (self.root / "state").mkdir()
        other = self.root / "state/other.json"
        other.write_text("keep")
        (self.root / "state/champion.json").symlink_to(other)
        with self.assertRaisesRegex(ValueError, "links"):
            run_gate(self.cfg, panel())
        self.assertEqual(other.read_text(), "keep")

    def test_fixed_deadline_and_lineage_protocol_are_not_overridden(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, deadline_unix=1791550801)
        with self.assertRaises(ValueError):
            replace(self.cfg, expected_lineages=2)
        with mock.patch(
            "rlfw.promotion_gate.time.time", return_value=self.cfg.deadline_unix + 1
        ):
            with self.assertRaises(RuntimeError):
                run_gate(self.cfg, panel())

    def test_actual_run_loader_derives_measured_units_and_refuses_changed_output(self):
        from rlfw.arena.kpi import SlotResult
        from rlfw.arena.report import write_reports
        from rlfw.tests.test_arena_kpi import spec_at

        specs = []
        for name, delta in (("challenger", 200_000), ("champion", 100_000)):
            spec = replace(
                spec_at(self.root, 2),
                game_id=name,
                out_dir=str(self.root / "runs" / name),
                state_path=str(self.root / "state" / (name + ".sqlite")),
            )
            spec.freeze()
            specs.append(spec)
            rows = []
            for c in spec.contexts:
                rows.append(
                    SlotResult(
                        c.context_id,
                        c.lineage_id,
                        "hwm",
                        1,
                        1,
                        SHA,
                        "qualified_complete",
                        0,
                        True,
                        spec_sha256=spec.sha256,
                        receipt_sha256=digest([c.context_id, SHA, 0]),
                    )
                )
                for arm in spec.arms[1:]:
                    program_pin = (
                        "e" * 64
                        if arm.name == "uniform_trie"
                        else ("c" * 64 if name == "challenger" else "d" * 64)
                    )
                    measured_q = 0 if arm.name == "uniform_trie" else delta
                    rows.extend(
                        SlotResult(
                            c.context_id,
                            c.lineage_id,
                            arm.name,
                            1,
                            slot,
                            program_pin,
                            "qualified_complete",
                            measured_q,
                            True,
                            wall_seconds=1.0,
                            spec_sha256=spec.sha256,
                            receipt_sha256=digest(
                                [c.context_id, program_pin, measured_q]
                            ),
                        )
                        for slot in range(1, 9)
                    )
            write_reports(
                spec,
                rows,
                {c.context_id: 0.0 for c in spec.contexts},
                [c.context_id for c in spec.contexts],
            )
        result = panel_from_runs(
            specs[0].out_dir, specs[1].out_dir, "trie_sample:toy", "trie_sample:toy"
        )
        self.assertEqual(result["challenger"][0]["positive_gain"], 0.2)
        self.assertEqual(result["champion"][0]["positive_gain"], 0.1)
        self.assertEqual(evaluate_panel(result, 3)["verdict"], "INCONCLUSIVE")
        path = Path(specs[0].out_dir) / "slots.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            panel_from_runs(
                specs[0].out_dir, specs[1].out_dir, "trie_sample:toy", "trie_sample:toy"
            )


if __name__ == "__main__":
    unittest.main()
