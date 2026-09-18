import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay_planner as planner
from scientific_metadata import CONTEXT_FIELDS, read


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def obs(name, score, origin="neural_policy", **overrides):
    genome = f"(pipe X (ridge n={name}))"
    row = dict(family="family", dataset="dataset", cut=0, partition="meta_train",
               genome=genome, program_sha256=sha(genome), origin=origin, failure=None,
               independent_evaluation=False, evaluation_origin="execution",
               evaluation_identity_sha256=sha("evaluation:" + name),
               metrics=dict(rmse=score, units="original_target"))
    for key in CONTEXT_FIELDS:
        if key.endswith("_sha256"):
            row[key] = sha(key)
    row.update(overrides)
    return row


def write_obs(root, rows):
    path = Path(root) / "observations.json"
    path.write_text(json.dumps(rows))
    return path


class ReplayPlannerTests(unittest.TestCase):
    def test_public_plan_never_promotes_retrospective_scores_to_original_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_obs(root, [obs("teacher", 1., "retained_cut_teacher"),
                             obs("winner", .8, policy_logprob=-.1)])
            result = planner.plan(root, root / "out")
            self.assertEqual(result["diagnostic_teacher_beating_n"], 1)
            self.assertIsNone(result["teacher_beating_n"])
            self.assertIsNone(result["replay_argv"])
            self.assertFalse(result["qualification_ready"])
            self.assertFalse(result["train_launched"])
            self.assertIn("ranking weight", result["importance_sampling"])
            self.assertFalse(result["original_verification"][0]["verified"])
            self.assertEqual(read(root / "out/plan.json"), result)

    def test_public_cli_rejects_malformed_source_without_silent_omission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "observations.json").write_text('[{"failure":null,"failure":"timeout"}]')
            result = subprocess.run([sys.executable, "-B", planner.__file__, "--observations", str(root),
                                     "--output", str(root / "out")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_failed_unknown_reused_and_nontraining_observations_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_obs(root, [obs("teacher", 1., "retained_cut_teacher"),
                             obs("timeout", .2, failure="timeout"),
                             obs("reuse", .1, evaluation_origin="reuse"),
                             obs("sealed", .1, partition="meta_test"),
                             obs("bool", True), obs("missing", .1, evaluation_identity_sha256=None)])
            result = planner.plan(root)
            details = result["diagnostics"]
            self.assertEqual(result["diagnostic_teacher_beating_n"], 0)
            self.assertEqual(len(details["failed_observations"]), 2)
            self.assertEqual(len(details["reuse_or_unknown_origins"]), 1)
            self.assertEqual(len(details["unavailable"]), 2)
            self.assertEqual(result["observation_rows"], 6)

    def test_cross_runtime_identity_cannot_borrow_a_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_obs(root, [obs("teacher", 1., "retained_cut_teacher"),
                             obs("winner", .1, runtime_sha256=sha("different"))])
            result = planner.plan(root)
            self.assertEqual(result["diagnostics"]["inspected_groups"], 2)
            self.assertEqual(result["diagnostic_teacher_beating_n"], 0)

    def test_duplicate_winner_deduplicates_conflicting_score_quarantines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher, winner = obs("teacher", 1., "retained_cut_teacher"), obs("winner", .8)
            write_obs(root, [teacher] + [winner] * 20)
            self.assertEqual(planner.plan(root)["diagnostic_teacher_beating_n"], 1)
            changed = copy.deepcopy(winner)
            changed["metrics"]["rmse"] = .1
            write_obs(root, [teacher, winner, changed])
            result = planner.plan(root)
            self.assertEqual(result["diagnostic_teacher_beating_n"], 0)
            self.assertTrue(result["diagnostics"]["unavailable"])

    def test_self_hashed_original_json_cannot_clear_native_access_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            winner = obs("winner", .8)
            ctx = {key: winner[key] for key in CONTEXT_FIELDS}
            fake = dict(context=ctx, program_sha256=winner["program_sha256"],
                        evaluation_identity_sha256=winner["evaluation_identity_sha256"],
                        original_verified=True, independent_evaluation=True)
            original = root / "original.json"
            original.write_text(json.dumps(fake))
            winner["original_query"] = dict(path="original.json", sha256=planner.file_sha(original))
            write_obs(root, [obs("teacher", 1., "retained_cut_teacher"), winner])
            result = planner.plan(root)
            self.assertFalse(result["original_verification"][0]["verified"])
            self.assertIn("hashes are not execution", result["original_verification"][0]["unavailable_reason"])
            self.assertIsNone(result["teacher_beating_n"])

    def test_original_conditions_and_historical_receipts_are_not_rewritten(self):
        with self.assertRaisesRegex(ValueError, "reset ablation"):
            planner.argv("output", corpus="c", baseline="b", checkpoint="k")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_obs(root, [])
            out = root / "old"
            out.mkdir()
            (out / "plan.json").write_text('{"schema":"parallel.replay_plan.v1"}')
            before = (out / "plan.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "immutable"):
                planner.plan(root, out)
            self.assertEqual((out / "plan.json").read_bytes(), before)

    def test_original_computation_cannot_change_origin_or_query_reference(self):
        for field, value in (("origin", "retained_cut_teacher"),
                             ("original_query", dict(path="other.json", sha256=sha("other"))),
                             ("caller_label", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                winner = obs("winner", .8)
                winner["caller_label"] = 0
                changed = dict(winner, **{field: value})
                write_obs(root, [obs("teacher", 1., "retained_cut_teacher"), winner, changed])
                result = planner.plan(root)
                self.assertEqual(result["diagnostic_teacher_beating_n"], 0)
                self.assertTrue(any("computation identities" in row["reason"]
                                    for row in result["diagnostics"]["unavailable"]))

    def test_conflicted_teacher_program_cannot_remain_a_comparison_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = obs("teacher", 1., "retained_cut_teacher")
            changed = copy.deepcopy(teacher)
            changed["evaluation_identity_sha256"] = sha("second original execution")
            changed["metrics"]["rmse"] = 10.
            write_obs(root, [teacher, changed, obs("winner", .8)])
            result = planner.plan(root)
            self.assertEqual(result["diagnostics"]["teacher_groups"], 0)
            self.assertEqual(result["diagnostic_teacher_beating_n"], 0)
            self.assertTrue(any("repeated program" in row["reason"]
                                for row in result["diagnostics"]["unavailable"]))

    def test_public_plan_binds_processed_bytes_even_when_sources_change_after_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observations = write_obs(root, [obs("teacher", 1., "retained_cut_teacher"), obs("winner", .8)])
            selected = root / "selected.json"
            selected.write_text('{"selected_conditions":["original"]}')
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (observations, selected)}
            actual_read = planner.read_with_digest
            reads = []
            def change_after_read(path, *args):
                value = actual_read(path, *args)
                reads.append(Path(path))
                Path(path).write_text("[]")  # Concurrent writer after the bound read.
                return value
            with patch.object(planner, "read_with_digest", change_after_read):
                result = planner.plan(root, selected_experiment=selected)
            self.assertEqual(reads, [observations, selected])
            self.assertEqual(result["source_files"], [dict(path=str(observations), sha256=before[observations])])
            self.assertEqual(result["selected_experiment"]["sha256"], before[selected])
            self.assertEqual(result["diagnostic_teacher_beating_n"], 1)

    def test_original_reference_and_selected_manifest_refuse_payload_paths_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            payload = root / "data/targets.json"
            payload.write_text("[]")
            winner = obs("winner", .8, original_query=dict(path="data/targets.json", sha256=sha("[]")))
            write_obs(root, [obs("teacher", 1., "retained_cut_teacher"), winner])
            actual_open = Path.open
            def refuse_open(path, *args, **kwargs):
                if "data" in path.parts:
                    raise AssertionError("payload must be rejected before open")
                return actual_open(path, *args, **kwargs)
            with patch.object(Path, "open", refuse_open):
                result = planner.plan(root)
                self.assertIn("data/ payloads", result["original_verification"][0]["unavailable_reason"])
                with self.assertRaisesRegex(ValueError, "data/ payloads"):
                    planner.plan(root, selected_experiment=payload)


if __name__ == "__main__":
    unittest.main()
