"""Research-only gate, alpha budget, hashed receipts and failure custody."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from rlfw.arena.game_spec import canonical, digest
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
    load_custody,
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
        "device": "cpu",
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
        with mock.patch("rlfw.promotion_gate.time.time", return_value=1791570600.0):
            resumed = replace(self.cfg, deadline_unix=1791608400)
            resumed.guard()
            with self.assertRaises(ValueError):
                replace(self.cfg, deadline_unix=1791608401)
        with mock.patch("rlfw.promotion_gate.time.time", return_value=1791608400.0):
            with self.assertRaises(RuntimeError):
                resumed.guard()
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


class D5CustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rlfw-d5-gate-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i in range(263):
            kept = i != 0 and not 54 <= i < 64
            self.rows.append(
                {
                    "context_id": f"c{i}",
                    "lineage_id": f"parent{i if i < 54 else 53}",
                    "category": "clean" if kept else "parity_custody",
                    "kept": kept,
                    # Screen raw q and deployed micro-q are distinct values.
                    "h_c": 0.0000004 if kept else None,
                    "h_c_grid": 0 if kept else None,
                    "programs": [{"role": "hwm", "program_sha256": SHA}],
                }
            )
        self.write_json(
            "runs/parity_screen/custody.json",
            {
                "schema": "rlfw-D5-parity-custody/v1",
                "contexts": self.rows,
                "native_tolerance": 1e-9,
                "excludes_all_arms": True,
            },
        )
        self.write_json(
            "runs/parity_screen/summary.json",
            {
                "state": "completed",
                "exclusion_skew": {"p_value": 0.1},
            },
        )
        self.write_json(
            "logs/verification.json",
            {
                "passed": True,
                "custody_sha256": self.pin("runs/parity_screen/custody.json"),
                "summary_sha256": self.pin("runs/parity_screen/summary.json"),
                "screen_manifest_sha256": SHA,
            },
        )
        self.write_json("state/STOP.pre-D5.json", {"historical": True})
        self.write_json(
            "state/CUSTODY_D5.json",
            {
                "schema": "rlfw-D5-custody-pointer/v1",
                "deadline_unix": 1791608400,
                "excludes_all_arms": True,
                "reuse_clean_HWM_h_c": True,
                "blocked_parity_coverage": False,
                "custody_path": str(self.root / "runs/parity_screen/custody.json"),
                "custody_sha256": self.pin("runs/parity_screen/custody.json"),
                "screen_summary_path": str(
                    self.root / "runs/parity_screen/summary.json"
                ),
                "screen_verification_path": str(self.root / "logs/verification.json"),
                "screen_verification_sha256": self.pin("logs/verification.json"),
                "screen_manifest_sha256": SHA,
            },
        )
        self.panel = panel()
        for side in ("roster", "challenger", "champion"):
            self.panel[side] = [u for u in self.panel[side] if u["context_id"] != "c0"]
        self.rehash(self.panel)

    def write_json(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o600)  # This test's own generated state only.
        path.write_text(canonical(body))
        path.chmod(0o444)

    def pin(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def rehash(self, body):
        body["panel_sha256"] = digest(
            {
                "contexts": body["roster"],
                "seeds": body["seeds"],
                "role_map": body["role_map_sha256"],
            }
        )

    def overlay(self, ids):
        rows = [
            {
                "context_id": f"c{i}",
                "lineage_id": self.rows[i]["lineage_id"],
                "category": "parity_custody_arena",
                "excludes_all_arms": True,
                "native_tolerance": 1e-9,
                "q": None,
                "q_grid": None,
                "actual_attempts": 3,
            }
            for i in ids
        ]
        body = {
            "schema": "rlfw-D5-arena-custody-extension/v1",
            "screen_custody_sha256": self.pin("runs/parity_screen/custody.json"),
            "arena_exclusions": rows,
        }
        pin = digest(body)
        self.write_json(f"state/arena_custody/custody-{pin}.json", body)
        self.write_json("state/arena_custody/custody.json", body)

    def test_sealed_53_lineage_panel_passes_while_unbound_subset_stays_inconclusive(
        self,
    ):
        self.assertEqual(evaluate_panel(self.panel, 3)["verdict"], "INCONCLUSIVE")
        custody = load_custody(self.root)
        result = evaluate_panel(self.panel, 3, custody=custody)
        self.assertEqual(result["verdict"], "PROMOTE")
        self.assertEqual(result["coverage"]["contexts_kept"], 252)
        self.assertEqual(result["coverage"]["lineages_kept"], 53)
        self.assertEqual(len(result["per_lineage_deltas"]["positive_gain"]), 53)
        self.assertEqual(result["bootstrap"], 2000)
        self.assertEqual(result["bootstrap_seed"], 7)

    def test_arena_exclusion_is_paired_and_unmeasured_gain_cannot_promote(self):
        self.overlay([1])
        result = evaluate_panel(self.panel, 3, custody=load_custody(self.root))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["arena_excluded_in_panel"], ["c1"])
        self.assertEqual(result["coverage"]["contexts_kept"], 251)
        self.assertNotIn("parent1", result["per_lineage_deltas"]["positive_gain"])
        bounds = result["censoring_bounds"]["paired_delta_winrate"]
        self.assertLess(bounds["lower"], 0)
        self.assertGreater(bounds["upper"], 0)
        self.assertFalse(result["censoring_bounds"]["raw_gain"]["available"])
        self.assertEqual(len(self.panel["champion"]), 53)
        self.assertEqual(len(self.panel["challenger"]), 53)

    def test_screened_hwm_program_quality_and_lineage_are_bound(self):
        custody = load_custody(self.root)
        for change in (
            lambda p: p["roster"][0].update(hwm_program_sha256="b" * 64),
            lambda p: p["roster"][0].update(lineage_id="other"),
            lambda p: p["champion"][0].update(hwm=0.1),
        ):
            candidate = deepcopy(self.panel)
            change(candidate)
            self.rehash(candidate)
            with self.assertRaises(ValueError):
                evaluate_panel(candidate, 3, custody=custody)

    def test_unregistered_original_units_refuse_before_custody_projection(self):
        custody = load_custody(self.root)
        for cid, lineage in (("c64", "parent53"), ("c0", "parent0")):
            candidate = deepcopy(self.panel)
            for side in ("champion", "challenger"):
                candidate[side].append(
                    {**candidate[side][0], "context_id": cid, "lineage_id": lineage}
                )
            with self.assertRaisesRegex(ValueError, "unregistered original gate unit"):
                evaluate_panel(candidate, 3, custody=custody)
        candidate = deepcopy(self.panel)
        for side in ("champion", "challenger"):
            candidate[side][0]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "unregistered original gate unit"):
            evaluate_panel(candidate, 3, custody=custody)

    def test_missing_changed_or_linked_custody_refuses(self):
        path = self.root / "runs/parity_screen/custody.json"
        raw = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(raw + b" ")
        with self.assertRaisesRegex(ValueError, "byte drift"):
            load_custody(self.root)
        path.write_bytes(raw)
        path.chmod(0o444)
        pointer = self.root / "state/CUSTODY_D5.json"
        pointer.unlink()  # This test's own generated pointer only.
        with self.assertRaisesRegex(ValueError, "pointer missing"):
            load_custody(self.root)
        pointer.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "links"):
            load_custody(self.root)

    def test_cpu_inference_is_explicit_and_cuda_cli_refuses_before_artifacts(self):
        custody = load_custody(self.root)
        for device in (None, "cuda"):
            candidate = deepcopy(self.panel)
            if device is None:
                candidate.pop("device")
            else:
                candidate["device"] = device
            with self.assertRaisesRegex(ValueError, "explicit CPU inference"):
                evaluate_panel(candidate, 3, custody=custody)
        from rlfw.promotion_gate import main

        with mock.patch("rlfw.promotion_gate.panel_from_runs") as loader:
            with self.assertRaisesRegex(ValueError, "CPU inference"):
                main(
                    [
                        "--bench-root",
                        str(self.root),
                        "--challenger-run",
                        "unused",
                        "--champion-run",
                        "unused",
                        "--challenger-arm",
                        "challenger",
                        "--champion-arm",
                        "champion",
                        "--deadline-unix",
                        "1791608400",
                        "--source-sha256",
                        SHA,
                        "--device",
                        "cuda",
                    ]
                )
            loader.assert_not_called()

    def test_every_outside_roster_arena_exclusion_enters_separate_sensitivity(self):
        point_identity = digest(self.panel)
        self.overlay([64])
        result = evaluate_panel(self.panel, 3, custody=load_custody(self.root))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["arena_excluded_in_panel"], [])
        self.assertEqual(result["arena_excluded_outside_panel"], ["c64"])
        self.assertEqual(len(result["per_lineage_deltas"]["score"]), 53)
        self.assertEqual(digest(self.panel), point_identity)
        bounds = result["censoring_bounds"]["paired_delta_winrate"]
        self.assertAlmostEqual(bounds["lower"], -1 / (2 * 53))
        self.assertAlmostEqual(bounds["upper"], 1 / (2 * 53))
        inventory = result["censoring_bounds"]["arena_exclusion_inventory"]
        self.assertEqual(inventory["pooled_context_count"], 1)
        self.assertEqual(
            inventory["paired_per_arm_context_count"], {"challenger": 1, "champion": 1}
        )
        self.assertEqual(inventory["per_lineage_context_count"], {"parent53": 1})
        self.assertFalse(result["censoring_bounds"]["raw_gain"]["available"])
        self.overlay([64, 65])
        result = evaluate_panel(self.panel, 3, custody=load_custody(self.root))
        bounds = result["censoring_bounds"]["paired_delta_winrate"]
        self.assertAlmostEqual(bounds["lower"], -2 / (3 * 53))
        self.assertAlmostEqual(bounds["upper"], 2 / (3 * 53))
        self.assertEqual(result["arena_excluded_outside_panel"], ["c64", "c65"])

    def test_cumulative_52_context_and_44_lineage_boundaries_are_fixed(self):
        self.overlay(range(64, 105))
        self.assertEqual(load_custody(self.root)["coverage"]["contexts_excluded"], 52)
        self.overlay(range(64, 106))
        with self.assertRaisesRegex(ValueError, "BLOCKED_PARITY_COVERAGE"):
            load_custody(self.root)
        self.overlay(range(1, 10))
        self.assertEqual(load_custody(self.root)["coverage"]["lineages_kept"], 44)
        self.overlay(range(1, 11))
        with self.assertRaisesRegex(ValueError, "BLOCKED_PARITY_COVERAGE"):
            load_custody(self.root)

    def test_custody_drift_before_pointer_commit_retains_spend_without_pointer(self):
        cfg = GateConfig(str(self.root), 1791608400, SHA, apply_research_pointer=True)
        run_gate(cfg, self.panel)
        run_gate(cfg, self.panel)
        custody = load_custody(self.root)
        with mock.patch(
            "rlfw.promotion_gate.load_custody", side_effect=[custody, None]
        ):
            with self.assertRaisesRegex(RuntimeError, "custody changed"):
                run_gate(cfg, self.panel)
        self.assertFalse((self.root / "state/champion.json").exists())
        self.assertEqual(
            len(list((self.root / "state/gates").rglob("*.reservation.json"))), 3
        )


if __name__ == "__main__":
    unittest.main()
