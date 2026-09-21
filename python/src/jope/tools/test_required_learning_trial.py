import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import qualify_regression_gain as trial


class RequiredLearningTrialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "new-attempt"
        self.argv = [
            "--root",
            str(self.root),
            "--manifest",
            str(self.root / "original.json"),
            "--output",
            str(self.output),
            "--build",
            str(self.root / "build"),
        ]

    def assert_no_trial_effects(self, argv):
        # Missing originals must be detected before even dereferencing metadata,
        # not merely before rank dispatch or after publishing a new manifest.
        with contextlib.ExitStack() as stack:
            for name in (
                "read",
                "sha",
                "ref",
                "immutable",
                "atomic",
                "learning_inputs",
                "monitor_reference",
                "verify_trial_inputs",
                "gpu_handoff",
                "ranks",
            ):
                stack.enter_context(
                    patch.object(trial, name, side_effect=AssertionError("unexpected effect: " + name))
                )
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(trial.main(argv), 2)
        status = json.loads(output.getvalue())
        self.assertEqual(status["status"], "unavailable")
        for key in ("launch_allowed", "references_opened", "numerical_work_started", "state_published"):
            self.assertIs(status[key], False)
        self.assertFalse(self.output.exists())
        return status

    def test_missing_originals_block_before_references_or_output(self):
        status = self.assert_no_trial_effects(self.argv)
        self.assertEqual(
            status["required_constraints"],
            {
                "selected_seed_count": 3,
                "global_batch": 2048,
                "new_committed_steps": 1000,
                "training_ceiling_seconds": 300,
                "optimizer_reset_ablation_allowed": False,
                "retry_deadline_renewal_allowed": False,
            },
        )
        self.assertEqual(status["original_evidence_verifier"], "unavailable")

    def test_legacy_resume_cannot_duplicate_trial_or_renew_deadline(self):
        original = self.root / "historical-trial.json"
        retained = b'{"schema":"parallel.regression_gain_trial.v1","policies":{"broad_reset":{}},"retry_training_ceiling_seconds":300}\n'
        original.write_bytes(retained)
        self.assert_no_trial_effects(
            self.argv
            + [
                "--resume-trial",
                str(original),
                "--budget-output",
                str(self.root / "budget"),
            ]
        )
        self.assertEqual(original.read_bytes(), retained)
        self.assertFalse((self.root / "budget").exists())

    def test_caller_assertions_cannot_certify_original_selection(self):
        original = self.root / "original.json"
        retained = json.dumps(
            {
                "selected_seeds": [1, 2, 3],
                "conditions": ["broad", "fresh"],
                "global_batch": 2048,
                "new_committed_steps": 1000,
                "optimizer_compatible": True,
                "original_evidence_verified": True,
            }
        ).encode()
        original.write_bytes(retained)
        self.assert_no_trial_effects(self.argv)
        self.assertEqual(original.read_bytes(), retained)

    def test_direct_freeze_cannot_publish_invented_conditions(self):
        with patch.object(trial, "read", side_effect=AssertionError("metadata opened")):
            with patch.object(trial, "immutable", side_effect=AssertionError("policy published")):
                with self.assertRaisesRegex(trial.RequiredLearningTrialUnavailable, "selected-three-seed"):
                    trial.freeze_trial(
                        self.root, self.root / "original.json", self.output, self.root / "build"
                    )
        self.assertFalse(self.output.exists())

    def test_public_cli_returns_unavailable_without_creating_attempt(self):
        result = subprocess.run(
            [sys.executable, "-B", str(Path(trial.__file__).resolve()), *self.argv],
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "unavailable")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
