"""Scratch observation/cost/uncertainty boundaries and actual Parquet readback."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlfw.analytics.build_scratch import (
    build,
    partial_receipt,
    remote_copy,
    remote_prefix,
    main,
    live_status_receipt,
    gpu_export,
)
from rlfw.analytics.extract import Copies, digest
from rlfw.analytics.report import read_daily, render, write_daily
from rlfw.analytics.scratch import EXPORT_SCHEMA, receipt_rows, verify_r22_metrics


def exported():
    return {
        "schema": EXPORT_SCHEMA,
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "started_unix": 1791654372,
        "anchor_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "points": [
            {
                "arm": arm,
                "epoch": epoch,
                "metric": "l6_validation_panel",
                "value": value,
                "unit": "loss",
                "n": 16,
                "updates": epoch * 142,
            }
            for arm, epoch, value in [
                ("control", 0, 0.87),
                ("control", 1, 0.86),
                ("actor", 1, 0.85),
            ]
        ],
        "costs": {
            "gpu_process_wall_seconds": 3600,
            "gpu_count": 1,
            "learner_cgroup_cpu_seconds": 7200,
        },
        "updates_per_arm": {"control": 142, "actor": 142},
    }


def rows(receipt, **kw):
    return receipt_rows(
        receipt,
        run="run1",
        lane="TEST",
        source="receipt.json:1",
        receipt_sha="c" * 64,
        consumer_commit="d" * 40,
        as_of_date="2026-10-10",
        **kw,
    )


class Scratch(unittest.TestCase):
    def test_r23_receipt_retains_steps_and_sticky_guard(self):
        receipt = {
            "schema": "jope.rlfw-g-abl-r23-pairs/v1",
            "non_qualifying": True,
            "promoted": False,
            "benchmark_valid": False,
            "live_state": False,
            "arms": {
                "ipo_tau_2": {
                    "pairs": 10,
                    "per_arm_steps": 568,
                    "epoch_loss": [0.2, 0.1],
                    "guard": {
                        "probe_contexts": 64,
                        "tripped": True,
                        "history": [
                            {"epoch": 0, "at_step": 0, "kl_mean": 0.13, "kl_max": 0.29},
                            {
                                "epoch": 2,
                                "at_step": 568,
                                "kl_mean": 0.01,
                                "kl_max": 0.02,
                            },
                        ],
                    },
                }
            },
        }
        result = rows(receipt)
        self.assertEqual(
            next(
                r["value"]
                for r in result
                if r["metric"] == "optimizer_updates_reported"
            ),
            568,
        )
        checks = [r for r in result if r["metric"] == "r23_exact_trie_kl_mean_to_v1"]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all("KL-DISQUALIFIED" in r["status"] for r in checks))
        self.assertIsNone(
            next(r["value"] for r in result if r["metric"] == "gpu_process_wall_hours")
        )

    def test_live_status_preserves_observed_updates_without_terminal_costs(self):
        document = {
            "unix": 1791746957,
            "utc": "2026-10-11T19:29:17Z",
            "state": "training",
            "updates": {"r23_control": 483},
            "evals": [
                {
                    "kind": "eval",
                    "arm": "r23_control",
                    "epoch": 1,
                    "updates": 142,
                    "l6_full_validation": {"loss": 0.8, "contexts": 263},
                }
            ],
        }
        result = rows(live_status_receipt(document))
        self.assertIn("PARTIAL", json.loads(result[0]["details"])["stop_reason"])
        self.assertIsNone(
            next(r["value"] for r in result if r["metric"] == "gpu_process_wall_hours")
        )
        document["evals"][0]["kind"] = "step"
        with self.assertRaisesRegex(ValueError, "evaluation inventory"):
            live_status_receipt(document)

    def test_host_gpu_sample_does_not_fill_whole_day_utilization(self):
        with tempfile.TemporaryDirectory() as root:
            copies = Copies(Path(root) / "inputs")
            with patch(
                "rlfw.analytics.build_scratch.subprocess.check_output",
                side_effect=[
                    b"GPU-one, 0, 2026/10/11 19:30:00.000, 92, 1000\n",
                    b"1791747000.0\n",
                ],
            ):
                document, sha, source = gpu_export("xbabe2", copies)
            self.assertEqual(document["samples"][0]["utilization_percent"], 92)
            self.assertEqual(digest(copies.root / sha), sha)
            result = rows(document)
            self.assertEqual(
                next(
                    r["value"]
                    for r in result
                    if r["metric"] == "gpu_utilization_sample_percent"
                ),
                92,
            )
            self.assertIsNone(
                next(
                    r["value"]
                    for r in result
                    if r["metric"] == "gpu_full_day_time_mean_percent"
                )
            )
            self.assertIn("xbabe2", source)
            with self.assertRaisesRegex(ValueError, "host required"):
                gpu_export("xbabe0", copies)

    def test_paired_evaluator_elapsed_cost_does_not_use_wall_cap(self):
        r = {
            "schema": "jope.r22-lpi-scratch-paired-eval/v1",
            "non_qualifying": True,
            "label": "R22 scratch-only",
            "started_unix": 1791660000,
            "finished_unix": 1791663600,
            "wall_seconds_cap": 7200,
            "device": "cuda:0",
        }
        self.assertEqual(
            next(
                x["value"] for x in rows(r) if x["metric"] == "gpu_process_wall_hours"
            ),
            1,
        )

    def test_growing_jsonl_snapshots_verify_prefix_and_exclude_partial_line(self):
        raw = b'{"epoch":1}\n{"epoch":'
        complete = raw[: raw.rfind(b"\n") + 1]
        with tempfile.TemporaryDirectory() as root:
            copies = Copies(Path(root))
            with patch(
                "rlfw.analytics.build_scratch.remote_read",
                side_effect=[str(len(raw)).encode(), raw, complete],
            ):
                p = remote_prefix("xbabe3", "/run/rows.jsonl", copies)
            self.assertEqual(p.read_bytes(), complete)
            self.assertIn(
                "prefix", copies.records["xbabe3:/run/rows.jsonl"]["snapshot_kind"]
            )
            with (
                patch(
                    "rlfw.analytics.build_scratch.remote_read",
                    side_effect=[str(len(raw)).encode(), raw, b'{"epoch":2}\n'],
                ),
                self.assertRaises(ValueError),
            ):
                remote_prefix("xbabe3", "/run/rows.jsonl", copies)

    def test_one_board_summary_per_ingest_no_per_arm_table_spam(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = root / "manifest.json"
            board = root / "board.md"
            board.write_text("all board bytes\n")
            manifest.write_text(
                json.dumps({"schema": "jope.cx-met-scratch-input/v1", "providers": []})
            )
            args = [
                "build_scratch",
                "--manifest",
                str(manifest),
                "--output-root",
                str(root / "out"),
                "--board",
                str(board),
            ]
            with (
                patch("sys.argv", args),
                patch("rlfw.analytics.build_scratch.scope", return_value={}),
                patch("rlfw.analytics.build_scratch.log_board") as log,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            self.assertEqual(log.call_count, 1)
            self.assertIn("measured n unmeasured", log.call_args.args[1])
            self.assertNotIn("TABLE", log.call_args.args[1])

    def test_shared_cost_once_and_learner_cpu_never_G(self):
        result = rows(exported())
        gpu = [r for r in result if r["metric"] == "gpu_process_wall_hours"]
        self.assertEqual(len(gpu), 1)
        self.assertEqual(gpu[0]["value"], 1)
        self.assertEqual(
            next(r["value"] for r in result if r["metric"] == "learner_cpu_core_hours"),
            2,
        )
        self.assertIsNone(
            next(
                r["value"] for r in result if r["metric"] == "G_per_evaluator_core_hour"
            )
        )

    def test_loss_delta_has_no_fold_q_significance(self):
        result = rows(exported())
        delta = next(r for r in result if r["kind"] == "scratch_delta")
        self.assertAlmostEqual(delta["value"], 0.01)
        self.assertIn("paired loss uncertainty unmeasured", delta["note"])
        self.assertEqual(delta["unit"], "loss")

    def test_changed_sample_count_cannot_join_control(self):
        r = exported()
        r["points"][-1]["n"] = 15
        self.assertIsNone(
            next(x["value"] for x in rows(r) if x["kind"] == "scratch_delta")
        )

    def test_unequal_or_unknown_optimizer_budget_cannot_join_control(self):
        for updates in (141, None):
            r = exported()
            r["points"][-1]["updates"] = updates
            self.assertIsNone(
                next(x["value"] for x in rows(r) if x["kind"] == "scratch_delta")
            )

    def test_partial_jsonl_keeps_epochs_but_has_no_terminal_cost(self):
        p = {
            "kind": "eval",
            "arm": "control",
            "epoch": 1,
            "updates": 142,
            "l6_validation_panel": {"loss": 0.87, "contexts": 16},
        }
        result = rows(partial_receipt([p]))
        self.assertEqual(json.loads(result[0]["details"])["epoch"], 1)
        self.assertIn("PARTIAL", json.loads(result[0]["details"])["stop_reason"])
        self.assertTrue(
            all(x["value"] is None for x in result if x["kind"] == "scratch_cost")
        )

    def test_remote_copy_rejects_mutation_and_preserves_read_only_cas(self):
        import hashlib

        raw = b'{"receipt": 1}'
        sha = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            copies = Copies(Path(root))
            replies = [str(len(raw)).encode(), sha.encode(), raw, sha.encode()]
            with patch("rlfw.analytics.build_scratch.remote_read", side_effect=replies):
                p = remote_copy("xbabe3", "/foreign/receipt.json", copies)
            self.assertEqual(digest(p), sha)
            self.assertEqual(p.stat().st_mode & 0o222, 0)
            replies[-1] = b"0" * 64
            with (
                patch("rlfw.analytics.build_scratch.remote_read", side_effect=replies),
                self.assertRaises(ValueError),
            ):
                remote_copy("xbabe3", "/foreign/changed.json", copies)

    def test_checkpoint_inference_has_no_invented_training_epoch(self):
        r = {
            "schema": "jope.cx-kv-checkpoint-eval/v1",
            "non_qualifying": True,
            "promotion_input": False,
            "program_execution": False,
            "optimizer_updates": 0,
            "checkpoint_name": "policy",
            "context_panel": [{}],
            "panel_kl_mean_nats": 0.03,
            "m2": {"multi_label_contexts": 66, "context_macro_spearman": 0.3},
        }
        result = rows(r)
        self.assertFalse(any(x["kind"] == "scratch_epoch" for x in result))
        self.assertEqual(
            next(
                x["n"] for x in result if x["metric"] == "M2_selected_support_spearman"
            ),
            66,
        )
        op = copy.deepcopy(r)
        op["schema"] = "jope.cx-op-r8-kv-eval/v1"
        op["arm"] = op.pop("checkpoint_name")
        op["context_panel_count"] = len(op.pop("context_panel"))
        self.assertEqual(
            next(
                x["n"]
                for x in rows(op)
                if x["metric"] == "development_panel_kl_to_anchor"
            ),
            1,
        )
        self.assertFalse(any(x["kind"] == "scratch_epoch" for x in rows(op)))
        op["program_execution"] = True
        with self.assertRaisesRegex(ValueError, "inference boundary"):
            rows(op)

    def test_gabl_training_loss_and_support_KL_are_distinct_from_validation(self):
        r = {
            "schema": "jope.rlfw-g-abl-real-pairs/v1",
            "non_qualifying": True,
            "promoted": False,
            "benchmark_valid": False,
            "live_state": False,
            "arms": {"ipo": {"pairs": 32, "epoch_loss": [0.4, 0.3]}},
            "eval": {"ipo": {"policy_kl_to_anchor_on_measured_support": 0.03}},
        }
        result = rows(r)
        self.assertEqual(
            [x["value"] for x in result if x["metric"] == "real_pair_training_loss"],
            [0.4, 0.3],
        )
        self.assertFalse(any("validation" in x["metric"] for x in result))
        self.assertIsNone(
            next(x["value"] for x in result if x["metric"] == "gpu_process_wall_hours")
        )

    def test_disqualified_arm_retained_and_nonfinite_refused(self):
        r = exported()
        r["points"][-1]["kl_disqualify"] = True
        self.assertIn(
            "KL-DISQUALIFIED",
            [x for x in rows(r) if x["kind"] == "scratch_epoch"][-1]["status"],
        )
        r["points"][-1]["value"] = float("nan")
        with self.assertRaises(ValueError):
            rows(r)

    def test_duplicate_epoch_metric_and_qualification_flags_refused(self):
        r = exported()
        r["points"].append(copy.deepcopy(r["points"][0]))
        with self.assertRaises(ValueError):
            rows(r)
        r = exported()
        r["promotion_input"] = True
        with self.assertRaises(ValueError):
            rows(r)

    def test_missing_costs_and_undated_run_remain_unknown(self):
        r = exported()
        r.pop("started_unix")
        r["costs"] = {}
        result = rows(r)
        self.assertTrue(
            all(
                x["value"] is None
                for x in result
                if x["kind"] == "scratch_cost" and "hours" in x["metric"]
            )
        )
        self.assertEqual(
            json.loads(result[0]["details"])["date_basis"], "snapshot_as_of_only"
        )

    def test_noise_requires_physical_pinned_pair_and_same_units_identity(self):
        r = exported()
        identity = {
            k: k + "-id"
            for k in [
                "context",
                "dataset_hash",
                "cut_id",
                "evaluator",
                "vocabulary",
                "protocol",
                "challenger_sha",
                "incumbent_sha",
            ]
        }
        p = {
            "arm": "actor",
            "delta_q": 0.2,
            "se_pair": 0.05,
            "unit": "log_rmse_ratio_q",
            "paired_evidence_identity": identity,
            "paired_evidence_sha256": "e" * 64,
        }
        r["q_comparisons"] = [p]
        self.assertIsNone(
            next(x["value"] for x in rows(r) if x["kind"] == "scratch_resolution")
        )
        proof = {"e" * 64: {"identity": identity, "delta_q": 0.2, "se_pair": 0.05}}
        resolved = next(
            x
            for x in rows(r, verified_pairings=proof)
            if x["kind"] == "scratch_resolution"
        )
        self.assertEqual(resolved["value"], 4)
        self.assertIn("EXCEEDS", resolved["status"])
        p["unit"] = "loss"
        self.assertIsNone(
            next(
                x["value"]
                for x in rows(r, verified_pairings=proof)
                if x["kind"] == "scratch_resolution"
            )
        )
        p["unit"] = "log_rmse_ratio_q"
        p["paired_evidence_identity"] = identity | {"evaluator": "different"}
        self.assertIsNone(
            next(
                x["value"]
                for x in rows(r, verified_pairings=proof)
                if x["kind"] == "scratch_resolution"
            )
        )

    def test_foreign_resource_closure_refused(self):
        r = exported()
        with self.assertRaises(ValueError):
            rows(r, closure={"run_receipt": {"sha256": "wrong"}})

    def test_receipt_jsonl_join_rejects_missing_eval_and_update_inventory(self):
        p = {
            "arm": "control",
            "epoch": 0,
            "updates": 0,
            "probe": {"kl_mean": 0},
            "l6_validation_panel": {"loss": 0.87},
        }
        r = {"initial_eval": [p], "updates_per_arm": {}}
        verify_r22_metrics(r, [p | {"kind": "eval"}])
        with self.assertRaises(ValueError):
            verify_r22_metrics(r, [])
        with self.assertRaises(ValueError):
            verify_r22_metrics(
                r, [p | {"kind": "eval"}, {"kind": "step", "arm": "control"}]
            )

    def test_failed_run_without_final_updates_preserves_unknown_count(self):
        r = {
            "schema": "jope.r22-lpi-scratch-train/v3",
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "error": "OOM",
            "device": "NVIDIA",
            "wall_seconds": 3600,
        }
        self.assertIn(
            "unmeasured", verify_r22_metrics(r, [{"kind": "step", "arm": "control"}])
        )
        result = rows(r)
        self.assertEqual(
            next(x["value"] for x in result if x["metric"] == "gpu_process_wall_hours"),
            1,
        )
        self.assertFalse(
            any(x["metric"] == "optimizer_updates_reported" for x in result)
        )
        self.assertEqual(json.loads(result[0]["details"])["run_error"], "OOM")

    def test_parquet_curves_escape_and_readback(self):
        r = exported()
        r["points"][-1]["arm"] = "<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as root:
            files = write_daily(rows(r), Path(root) / "daily")
            output = Path(root) / "dashboard.html"
            render(read_daily(files), output)
            text = output.read_text()
            self.assertIn("<svg", text)
            self.assertIn("NON-QUALIFYING", text)
            self.assertNotIn("<script>", text)
            self.assertIn("&lt;script&gt;", text)

    def test_end_to_end_copy_dedup_and_missing_provider_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "foreign" / "run1" / "receipt.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(exported()))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.cx-met-scratch-input/v1",
                        "providers": [
                            {"lane": "TEST", "patterns": [str(source), str(source)]},
                            {
                                "lane": "PENDING",
                                "patterns": [str(root / "missing" / "receipt.json")],
                            },
                        ],
                    }
                )
            )
            before = source.read_bytes()
            receipt = build(manifest, root / "owned", "d" * 40)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(
                sum(x.get("status") == "NON-QUALIFYING" for x in receipt["inventory"]),
                1,
            )
            self.assertTrue(any(x["lane"] == "PENDING" for x in receipt["inventory"]))
            self.assertTrue((root / "owned" / "dashboard.html").exists())
