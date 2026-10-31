"""Tiny synthetic development metadata controls, not H75 scientific evidence."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import feedback


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bind(admission, rows):
    admission["record_sha256s"] = {
        row["record_id"]: digest(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        for row in rows
    }


def fixture(cell="D", statuses=("completed", "completed")):
    context = digest("fit context")
    parent = digest("x0")
    entry = dict(
        role="DL-Train",
        family_id="fit-family",
        family_fold=1,
        alias_component="fit-alias",
        table_sha256=digest("table"),
        original_rows_sha256=digest("original rows"),
        cut_id="original-cut",
        evaluator_sha256=feedback.EVALUATOR,
        vocabulary_sha256=digest("vocabulary"),
        evaluation_protocol="fixture-full-train-cv",
        programs={parent: digest("parent features")},
        pre_cycle_best_fitness={"1": 0.4, "2": 0.3},
    )
    admission = dict(
        schema="jope.h75-closed-loop.feedback-admission/v1",
        run_id="fixture-only",
        source_git_sha="a" * 40,
        prereg_sha256=feedback.PREREG,
        execution_manifest_sha256=digest("fixture execution"),
        cell=cell,
        fold=0,
        seed=701,
        dope_sha256=feedback.DOPE,
        fold_manifest_sha256=feedback.FOLDS,
        selection_policy="fixture-top2-uniform2",
        forbidden_families=["scoring-family"],
        forbidden_alias_components=["scoring-alias"],
        fit_contexts={context: entry},
        source_objects=[],
        outcome_exports=[],
    )
    rows = []
    for i, status in enumerate(statuses):
        dsl = f"add(x0,{i + 1})"
        program = digest(dsl)
        entry["programs"][program] = digest("features " + program)
        source = dict(
            key=f"fixture-native/{i}.json",
            version_id="fixture-v1",
            sha256=digest(f"native {i}"),
        )
        admission["source_objects"].append(source)
        row = {key: admission[key] for key in feedback.BOUND}
        row.update({key: entry[key] for key in feedback.CONTEXT})
        row.update(
            schema="jope.h75-closed-loop.outcome/v1",
            cycle=1,
            phase="fit_acquisition",
            role="DL-Train",
            context_sha256=context,
            evaluation_scope="full_train_cv",
            nq=0,
            program_sha256=program,
            dsl=dsl,
            parent_program_sha256s=[parent],
            operator="fixture_mutation",
            proposal_seed=701 + i,
            selection_policy=admission["selection_policy"],
            selection_probability=0.5,
            status=status,
            gate=i == 0,
            fitness=0.2345678901234567 if i == 0 else 100.0,
            logratio=0.1234567890123456 if i == 0 else 0.0,
            source_object=source,
            record_id=f"native-execution-{i}",
        )
        rows.append(row)
    bind(admission, rows)
    return rows, admission


def emitted(result):
    return [
        row
        for partition in result["partitions"]
        for rows in partition["programs"].values()
        for row in rows
    ]


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        base = Path(
            os.environ.get(
                "JOPE_H75_TEST_TMP",
                str(Path.home() / "jope_plan/evidence/H75-feedback/obj02-t2312/tests"),
            )
        )
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_completed_positive_and_negative_targets_preserve_original_values(self):
        rows, admission = fixture()
        original = copy.deepcopy(rows)
        result = feedback.build_feedback(rows, admission)
        by_id = {row["outcome"]["record_id"]: row for row in emitted(result)}
        self.assertEqual(by_id["native-execution-0"]["target"], rows[0]["fitness"])
        self.assertEqual(by_id["native-execution-1"]["target"], 2.0)
        self.assertEqual(by_id["native-execution-1"]["outcome"]["fitness"], 100.0)
        self.assertEqual(result["counts"]["eligible_for_loss"], 2)
        self.assertEqual(result["counts"]["apparent_hwm"], 1)
        self.assertEqual(rows, original)

    def test_errors_timeouts_unknown_and_finite_sentinels_stay_audit_only(self):
        rows, admission = fixture(statuses=("failed", "timeout", "unknown"))
        for row in emitted(feedback.build_feedback(rows, admission)):
            self.assertIsNone(row["target"])
            self.assertFalse(row["eligible_for_loss"])
            self.assertFalse(row["apparent_hwm"])
            self.assertEqual(row["memory_use"], "audit_only")

    def test_only_d_gets_loss_feedback_and_only_b_d_get_search_memory(self):
        for cell in "ABCD":
            rows, admission = fixture(cell)
            output = emitted(feedback.build_feedback(rows, admission))
            with self.subTest(cell=cell):
                self.assertTrue(
                    all(row["eligible_for_loss"] == (cell == "D") for row in output)
                )
                self.assertTrue(
                    all(
                        row["memory_use"]
                        == ("fresh_search" if cell in "BD" else "shadow")
                        for row in output
                    )
                )

    def test_missing_feature_provenance_masks_but_preserves_completed_audit(self):
        rows, admission = fixture()
        admission["fit_contexts"][rows[0]["context_sha256"]]["programs"][
            rows[0]["program_sha256"]
        ] = None
        row = next(
            row
            for row in emitted(feedback.build_feedback(rows, admission))
            if row["outcome"]["record_id"] == rows[0]["record_id"]
        )
        self.assertEqual(row["outcome"]["status"], "completed")
        self.assertIsNone(row["target"])
        self.assertFalse(row["eligible_for_loss"])
        self.assertEqual(row["memory_use"], "audit_only")

    def test_empty_ledger_keeps_scope_and_never_invents_targets(self):
        rows, admission = fixture(statuses=())
        result = feedback.build_feedback(rows, admission)
        self.assertEqual(result["partitions"], [])
        self.assertEqual(result["admission_scope"]["fold"], 0)
        self.assertEqual(result["admission_scope"]["run_id"], "fixture-only")
        self.assertEqual(result["counts"]["eligible_for_loss"], 0)
        self.assertEqual(result["counts"]["unique_records"], 0)

    def test_contexts_are_separate_program_keyed_partitions(self):
        rows, admission = fixture(statuses=("completed",))
        second = copy.deepcopy(rows[0])
        second["context_sha256"] = digest("second fit context")
        second["record_id"] = "second-execution"
        admission["fit_contexts"][second["context_sha256"]] = copy.deepcopy(
            next(iter(admission["fit_contexts"].values()))
        )
        rows.append(second)
        bind(admission, rows)
        result = feedback.build_feedback(rows, admission)
        self.assertEqual(len(result["partitions"]), 2)
        self.assertEqual(
            {p["partition"]["context_sha256"] for p in result["partitions"]},
            {row["context_sha256"] for row in rows},
        )

    def test_exact_retry_deduplicates_and_conflicting_identity_refuses(self):
        rows, admission = fixture()
        result = feedback.build_feedback(rows + [copy.deepcopy(rows[0])], admission)
        self.assertEqual(result["counts"]["unique_records"], 2)
        self.assertEqual(result["counts"]["duplicate_retries"], 1)
        changed = dict(rows[0], fitness=0.1)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            feedback.build_feedback(rows + [changed], admission)

    def test_winner_only_or_mutated_execution_input_refuses(self):
        rows, admission = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete execution"):
            feedback.build_feedback(rows[:1], admission)
        rows[0]["fitness"] = 0.01
        with self.assertRaisesRegex(ValueError, "frozen execution"):
            feedback.build_feedback(rows, admission)

    def test_scoring_family_alias_and_fold_are_independently_refused(self):
        for change in (
            {"family_id": "scoring-family"},
            {"alias_component": "scoring-alias"},
            {"family_fold": 0},
        ):
            rows, admission = fixture()
            next(iter(admission["fit_contexts"].values())).update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                feedback.build_feedback(rows, admission)

    def test_foreign_role_phase_fold_run_seed_and_dope_refuse(self):
        for change in (
            {"role": "DL-Val"},
            {"phase": "terminal_confirmation"},
            {"fold": 1},
            {"run_id": "other"},
            {"seed": 702},
            {"dope_sha256": "f" * 64},
        ):
            rows, admission = fixture()
            rows[0].update(change)
            bind(
                admission, rows
            )  # Even a supplied ledger cannot waive these semantic boundaries.
            with self.subTest(change=change), self.assertRaises(ValueError):
                feedback.build_feedback(rows, admission)

    def test_table_row_cut_protocol_evaluator_and_vocabulary_join_exactly(self):
        for field in feedback.CONTEXT:
            rows, admission = fixture()
            rows[0][field] = "foreign"
            bind(admission, rows)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "context identity"),
            ):
                feedback.build_feedback(rows, admission)

    def test_dsl_parent_and_source_version_provenance_are_required(self):
        for kind in ("dsl", "parent", "source"):
            rows, admission = fixture()
            if kind == "dsl":
                rows[0]["dsl"] = "changed(x0)"
            elif kind == "parent":
                rows[0]["parent_program_sha256s"] = ["f" * 64]
            else:
                rows[0]["source_object"] = dict(
                    rows[0]["source_object"], version_id="different"
                )
            bind(admission, rows)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                feedback.build_feedback(rows, admission)

    def test_completed_missing_boolean_and_nonfinite_values_refuse(self):
        for change in (
            {"fitness": None},
            {"gate": None},
            {"gate": 1},
            {"fitness": True},
            {"fitness": float("nan")},
            {"logratio": float("inf")},
            {"selection_probability": 0},
        ):
            rows, admission = fixture()
            rows[0].update(change)
            if not any(
                type(v) is float and not math.isfinite(v) for v in change.values()
            ):
                bind(admission, rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                feedback.build_feedback(rows, admission)

    def test_outer_test_extra_field_and_unmeasured_status_refuse(self):
        for change in ({"test_rmse_ratio": 0.5}, {"status": "screened"}):
            rows, admission = fixture()
            rows[0].update(change)
            bind(admission, rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                feedback.build_feedback(rows, admission)

    def test_fit_family_and_alias_have_one_fold(self):
        rows, admission = fixture()
        entry = next(iter(admission["fit_contexts"].values()))
        extra = digest("second fit context")
        admission["fit_contexts"][extra] = copy.deepcopy(entry)
        feedback.validate_admission(
            admission
        )  # Multiple contexts in one family remain valid.
        for same_family in (True, False):
            changed = copy.deepcopy(admission)
            changed["fit_contexts"][extra]["family_fold"] = 2
            if not same_family:
                changed["fit_contexts"][extra]["family_id"] = "other-fit-family"
            with (
                self.subTest(same_family=same_family),
                self.assertRaisesRegex(ValueError, "fold"),
            ):
                feedback.validate_admission(changed)

    def test_unresolved_family_and_alias_identifiers_refuse(self):
        for field in ("family_id", "alias_component"):
            for value in ("unknown", "NONE", "null", "unresolved", " ", " fit-family "):
                rows, admission = fixture()
                entry = next(iter(admission["fit_contexts"].values()))
                entry[field] = value
                if field == "family_id":
                    for row in rows:
                        row[field] = value
                bind(admission, rows)
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, "identity"),
                ):
                    feedback.build_feedback(rows, admission)

    def test_json_duplicate_keys_and_nonfinite_constants_refuse(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                feedback.load_json(value)

    def test_atomic_output_never_replaces_existing_file_or_symlink(self):
        output = self.path / "feedback.json"
        feedback.atomic_create(output, b"complete\n")
        with self.assertRaises(FileExistsError):
            feedback.atomic_create(output, b"replacement\n")
        link = self.path / "link"
        link.symlink_to(output)
        with self.assertRaises(FileExistsError):
            feedback.atomic_create(link, b"replacement\n")
        self.assertEqual(output.read_bytes(), b"complete\n")
        self.assertEqual(list(self.path.glob(".h75-feedback-*")), [])

    def cli_inputs(self):
        rows, admission = fixture()
        source = self.path / "clean.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        admission["outcome_exports"] = [source_hash]
        manifest = self.path / "admission.json"
        manifest.write_text(json.dumps(admission))
        args = [
            sys.executable,
            "-B",
            str(Path(feedback.__file__)),
            "--admission",
            str(manifest),
            "--admission-sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "--records",
            str(source),
            "--records-sha256",
            source_hash,
            "--output",
            str(self.path / "output.json"),
        ]
        return args, source, manifest

    def test_actual_cli_writes_hashed_feedback_and_refuses_output_reuse(self):
        args, source, manifest = self.cli_inputs()
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = (self.path / "output.json").read_bytes()
        self.assertEqual(
            json.loads(result.stdout)["output_sha256"],
            hashlib.sha256(output).hexdigest(),
        )
        self.assertEqual(json.loads(output)["counts"]["eligible_for_loss"], 2)
        self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)

    def test_cli_rejects_unadmitted_export_before_opening_it(self):
        args, source, manifest = self.cli_inputs()
        args[args.index("--records-sha256") + 1] = "f" * 64
        source.unlink()
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not independently admitted", result.stderr)
        self.assertFalse((self.path / "output.json").exists())

    def test_cli_validates_admission_before_opening_outcome_bytes(self):
        args, source, manifest = self.cli_inputs()
        value = json.loads(manifest.read_text())
        value["dope_sha256"] = "f" * 64
        manifest.write_text(json.dumps(value))
        args[args.index("--admission-sha256") + 1] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        source.unlink()
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("frozen DOPE", result.stderr)

    def test_cli_rejects_overflowing_metadata_before_outcome_read(self):
        args, source, manifest = self.cli_inputs()
        # 1e400 is a JSON number that the stdlib decoder maps to infinity.
        data = manifest.read_text().rstrip()[:-1] + ',"extra_metadata":1e400}'
        manifest.write_text(data)
        args[args.index("--admission-sha256") + 1] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        source.unlink()
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Out of range float", result.stderr)
        self.assertFalse((self.path / "output.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
