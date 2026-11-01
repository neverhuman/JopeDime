import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import production_backend as backend
import q1 as q1_harness


class ProductionBackendTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "schema": "jope.q1-request/v1",
            "mode": "qualification",
            "plan_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "pins": {
                name: value * 64
                for name, value in {
                    "bundle_sha256": "c",
                    "dope_sha256": "d",
                    "evaluator_sha256": "e",
                    "vocabulary_sha256": "f",
                    "protocol_sha256": "1",
                    "training_family_manifest_sha256": "2",
                }.items()
            },
            "metric": "rmse",
            "dataset": {
                "id": "0123456789abcdef",
                "task": "regression",
                "dataset_sha256": "0123456789abcdef" + "3" * 48,
                "family": "4" * 40,
                "family_authoritative": True,
                "rows": 100,
                "features": 8,
                "cut_sha256": "5" * 64,
                "holdout_group": "sealed_meta_test",
            },
            "guidance": "learned",
            "seed": 0,
            "budget": 3,
        }
        self.request["request_sha256"] = backend.digest(self.request)
        self.receipt = {
            "schema": "jope.q1-evaluation-receipt/v1",
            "request_sha256": self.request["request_sha256"],
            "identity": {
                "source_commit": self.request["source_commit"],
                "plan_sha256": self.request["plan_sha256"],
                "pins": self.request["pins"],
                "dataset_id": self.request["dataset"]["id"],
                "dataset_sha256": self.request["dataset"]["dataset_sha256"],
                "cut_sha256": self.request["dataset"]["cut_sha256"],
                "guidance": self.request["guidance"],
                "metric": self.request["metric"],
                "seed": self.request["seed"],
                "requested_budget": self.request["budget"],
                "dataset_binding_sha256": "6" * 64,
            },
            "charged": 3,
            "attempts": [
                {
                    "attempt": 0,
                    "outcome": "completed",
                    "fold_scores": [2.0, 1.0],
                    "loss": 1.5,
                    "program_sha256": "7" * 64,
                },
                {
                    "attempt": 1,
                    "outcome": "failed",
                    "fold_scores": [],
                    "loss": None,
                    "program_sha256": None,
                    "failure": {"code": "EVAL_ERROR", "reason": "fixture failure"},
                },
                {
                    "attempt": 2,
                    "outcome": "completed",
                    "fold_scores": [0.5, 1.5, 1.0],
                    "loss": 1.0,
                    "program_sha256": "8" * 64,
                },
            ],
            "champion_program": "8" * 64,
            "adapter": "production-engine/v1",
        }

    def test_projects_counts_and_scores_from_actual_attempt_receipts(self):
        response = backend.validate_receipt(self.receipt, self.request)
        self.assertEqual(q1_harness.response_valid(response, self.request), response)
        self.assertEqual((response["charged"], response["completed"]), (3, 2))
        self.assertEqual(response["fold_scores_completed"], 5)
        self.assertEqual(response["loss"], 1.0)
        self.assertEqual(response["champion_program"], "8" * 64)
        self.assertEqual(
            response["candidate_failures"],
            [{"attempt": 1, "reason": "EVAL_ERROR: fixture failure"}],
        )

    def test_refuses_wrong_request_identity(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["identity"]["dataset_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            backend.validate_receipt(receipt, self.request)

    def test_refuses_unrecognized_request_fields(self):
        request = copy.deepcopy(self.request)
        request["target"] = "inferred-target"
        request["request_sha256"] = backend.digest(
            {key: value for key, value in request.items() if key != "request_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "request fields"):
            backend.validate_receipt(self.receipt, request)

    def test_accepts_nonsealed_development_binding(self):
        request = copy.deepcopy(self.request)
        request["dataset"]["holdout_group"] = "development_fit"
        request["request_sha256"] = backend.digest(
            {key: value for key, value in request.items() if key != "request_sha256"}
        )
        receipt = copy.deepcopy(self.receipt)
        receipt["request_sha256"] = request["request_sha256"]
        response = backend.validate_receipt(receipt, request)
        self.assertEqual(response["status"], "complete")

    def test_refuses_missing_attempt_and_requested_budget_echo(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["attempts"].pop()
        with self.assertRaisesRegex(ValueError, "attempt inventory"):
            backend.validate_receipt(receipt, self.request)

    def test_refuses_unknown_and_nonterminal_attempt(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["attempts"][1]["outcome"] = "unknown"
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            backend.validate_receipt(receipt, self.request)

    def test_refuses_nonfinite_fold_score(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["attempts"][0]["fold_scores"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite JSON|completed attempt"):
            backend.validate_receipt(receipt, self.request)

    def test_call_engine_authenticates_with_matching_daemon_record(self):
        endpoint = "http://127.0.0.1:31234"
        record = {
            "url": endpoint,
            "token": "a" * 64,
            "pid": 1,
            "version": "1.0.0",
            "api_version": "1.0",
            "started_at": "2026-11-01T00:00:00Z",
        }
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.status = 200
        response.geturl.return_value = endpoint + "/v1/q1/evaluations"
        response.read.return_value = b"{}"
        opener = Mock()
        opener.open.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            daemon = Path(directory) / "daemon.json"
            daemon.write_text(json.dumps(record), encoding="utf-8")
            daemon.chmod(0o600)
            with (
                patch.dict("os.environ", {"JOPEDIME_HOME": directory}),
                patch.object(
                    backend, "build_opener", return_value=opener
                ) as build_opener,
            ):
                self.assertEqual(backend.call_engine(self.request, endpoint), {})

        sent_request = opener.open.call_args.args[0]
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer " + "a" * 64)
        self.assertEqual(sent_request.data, backend.canonical(self.request))
        self.assertIsInstance(build_opener.call_args.args[0], backend.ProxyHandler)
        self.assertEqual(build_opener.call_args.args[0].proxies, {})
        self.assertIsInstance(build_opener.call_args.args[1], backend.NoRedirectHandler)

    def test_call_engine_refuses_missing_or_mismatched_daemon_record(self):
        endpoint = "http://127.0.0.1:31234"
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"JOPEDIME_HOME": directory}):
                with self.assertRaisesRegex(ValueError, "daemon record is unavailable"):
                    backend.call_engine(self.request, endpoint)

                record = {
                    "url": "http://127.0.0.1:31235",
                    "token": "a" * 64,
                    "pid": 1,
                    "version": "1.0.0",
                    "api_version": "1.0",
                    "started_at": "2026-11-01T00:00:00Z",
                }
                daemon = Path(directory) / "daemon.json"
                daemon.write_text(json.dumps(record), encoding="utf-8")
                daemon.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    backend.call_engine(self.request, endpoint)

    def test_call_engine_refuses_broad_daemon_permissions(self):
        endpoint = "http://127.0.0.1:31234"
        record = {
            "url": endpoint,
            "token": "a" * 64,
            "pid": 1,
            "version": "1.0.0",
            "api_version": "1.0",
            "started_at": "2026-11-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            daemon = Path(directory) / "daemon.json"
            daemon.write_text(json.dumps(record), encoding="utf-8")
            daemon.chmod(0o644)
            with patch.dict("os.environ", {"JOPEDIME_HOME": directory}):
                with self.assertRaisesRegex(ValueError, "permissions are too broad"):
                    backend.call_engine(self.request, endpoint)

    def test_call_engine_refuses_symlinked_daemon_record(self):
        endpoint = "http://127.0.0.1:31234"
        record = {
            "url": endpoint,
            "token": "a" * 64,
            "pid": 1,
            "version": "1.0.0",
            "api_version": "1.0",
            "started_at": "2026-11-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "record.json"
            target.write_text(json.dumps(record), encoding="utf-8")
            target.chmod(0o600)
            (Path(directory) / "daemon.json").symlink_to(target)
            with patch.dict("os.environ", {"JOPEDIME_HOME": directory}):
                with self.assertRaisesRegex(ValueError, "regular file"):
                    backend.call_engine(self.request, endpoint)

    def test_redirect_handler_refuses_redirects(self):
        handler = backend.NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(
                Mock(), Mock(), 302, "Found", {}, "http://127.0.0.1:31235/"
            )
        )


if __name__ == "__main__":
    unittest.main()
