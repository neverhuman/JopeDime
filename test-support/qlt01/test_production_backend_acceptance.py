"""Engine-free acceptance controls for the QLT-01-F01 receipt adapter.

Run with QLT01_BACKEND_PATH pointing at the exact production_backend.py under test.
Synthetic receipts and loopback HTTP are contract controls only; they are not
engine execution or Q1 qualification evidence.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from unittest import mock
from urllib.error import URLError


backend_path = Path(
    os.environ.get(
        "QLT01_BACKEND_PATH",
        Path(__file__).resolve().parents[2]
        / "deploy/qualification/q1/harness/production_backend.py",
    )
).resolve()
spec = importlib.util.spec_from_file_location(
    "production_backend_under_test", backend_path
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def valid_request():
    request = {
        "schema": "jope.q1-request/v1",
        "mode": "qualification",
        "plan_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "pins": {
            "bundle_sha256": "c" * 64,
            "dope_sha256": "d" * 64,
            "evaluator_sha256": "e" * 64,
            "vocabulary_sha256": "f" * 64,
            "protocol_sha256": "1" * 64,
            "training_family_manifest_sha256": "2" * 64,
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
            "holdout_group": "development_fit",
        },
        "guidance": "learned",
        "seed": 0,
        "budget": 3,
    }
    request["request_sha256"] = backend.digest(request)
    return request


def valid_receipt(request):
    return {
        "schema": "jope.q1-evaluation-receipt/v1",
        "request_sha256": request["request_sha256"],
        "identity": {
            "source_commit": request["source_commit"],
            "plan_sha256": request["plan_sha256"],
            "pins": copy.deepcopy(request["pins"]),
            "dataset_id": request["dataset"]["id"],
            "dataset_sha256": request["dataset"]["dataset_sha256"],
            "cut_sha256": request["dataset"]["cut_sha256"],
            "guidance": request["guidance"],
            "metric": request["metric"],
            "seed": request["seed"],
            "requested_budget": request["budget"],
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


def refresh_request(request):
    unhashed = {key: value for key, value in request.items() if key != "request_sha256"}
    request["request_sha256"] = backend.digest(unhashed)


def bind_receipt(receipt, request):
    receipt["request_sha256"] = request["request_sha256"]
    identity = receipt["identity"]
    identity.update(
        {
            "source_commit": request["source_commit"],
            "plan_sha256": request["plan_sha256"],
            "pins": copy.deepcopy(request["pins"]),
            "dataset_id": request["dataset"]["id"],
            "dataset_sha256": request["dataset"]["dataset_sha256"],
            "cut_sha256": request["dataset"]["cut_sha256"],
            "guidance": request["guidance"],
            "metric": request["metric"],
            "seed": request["seed"],
            "requested_budget": request["budget"],
        }
    )


class ProductionBackendAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.request = valid_request()
        self.receipt = valid_receipt(self.request)
        self.endpoint = "http://127.0.0.1:8123"
        self.engine_url = self.endpoint + "/v1/q1/evaluations"
        self.daemon_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.daemon_dir.cleanup)
        self.daemon_path = Path(self.daemon_dir.name) / "daemon.json"
        self.daemon_token = "9" * 64
        self.write_daemon_record()
        environment = mock.patch.dict(
            os.environ, {"JOPEDIME_HOME": self.daemon_dir.name}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def write_daemon_record(self, **overrides):
        record = {
            "url": self.endpoint,
            "token": self.daemon_token,
            "pid": 123,
            "version": "1.0.0",
            "api_version": "1.0",
            "started_at": "2026-11-01T00:00:00Z",
        }
        record.update(overrides)
        self.daemon_path.write_text(json.dumps(record), encoding="utf-8")
        self.daemon_path.chmod(0o600)

    def assert_refused(self, receipt=None, request=None, pattern=None):
        with self.assertRaises(ValueError) as caught:
            backend.validate_receipt(
                self.receipt if receipt is None else receipt,
                self.request if request is None else request,
            )
        if pattern:
            self.assertRegex(str(caught.exception), pattern)

    def test_projects_only_recorded_attempts_and_selects_best_rmse(self):
        result = backend.validate_receipt(self.receipt, self.request)
        self.assertEqual(result["charged"], 3)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["fold_scores_completed"], 5)
        self.assertEqual(
            result["candidate_failures"],
            [{"attempt": 1, "reason": "EVAL_ERROR: fixture failure"}],
        )
        self.assertEqual(result["loss"], 1.0)
        self.assertEqual(result["champion_program"], "8" * 64)

    def test_all_failed_panel_is_failed_with_no_fabricated_score(self):
        receipt = copy.deepcopy(self.receipt)
        for index, attempt in enumerate(receipt["attempts"]):
            attempt.update(
                {
                    "attempt": index,
                    "outcome": "failed",
                    "fold_scores": [],
                    "loss": None,
                    "program_sha256": None,
                    "failure": {"code": "FIT_ERROR", "reason": f"failure {index}"},
                }
            )
        receipt["champion_program"] = None
        result = backend.validate_receipt(receipt, self.request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual((result["charged"], result["completed"]), (3, 0))
        self.assertIsNone(result["loss"])
        self.assertIsNone(result["champion_program"])
        self.assertEqual(len(result["candidate_failures"]), 3)

    def test_rejects_wrong_request_digest_and_unrecognized_request_fields(self):
        bad = copy.deepcopy(self.request)
        bad["dataset"]["cut_sha256"] = "9" * 64
        self.assert_refused(request=bad, pattern="request digest mismatch")
        bad = copy.deepcopy(self.request)
        bad["inferred_target"] = "target"
        refresh_request(bad)
        self.assert_refused(request=bad, pattern="request fields")

    def test_rejects_non_authoritative_family_identity(self):
        request = copy.deepcopy(self.request)
        request["dataset"]["family_authoritative"] = False
        refresh_request(request)
        receipt = copy.deepcopy(self.receipt)
        bind_receipt(receipt, request)
        self.assert_refused(receipt, request, pattern="dataset identity")

    def test_rejects_malformed_authoritative_family_identity(self):
        request = copy.deepcopy(self.request)
        request["dataset"]["family"] = "derived-family"
        request["dataset"]["family_authoritative"] = True
        refresh_request(request)
        receipt = copy.deepcopy(self.receipt)
        bind_receipt(receipt, request)
        self.assert_refused(receipt, request, pattern="dataset identity")

    def test_rejects_invalid_dataset_task_shape_or_holdout_identity(self):
        for field, value in (
            ("task", "binary"),
            ("rows", 0),
            ("features", -1),
            ("holdout_group", "unknown"),
        ):
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request["dataset"][field] = value
                refresh_request(request)
                receipt = copy.deepcopy(self.receipt)
                bind_receipt(receipt, request)
                self.assert_refused(receipt, request, pattern="dataset identity")

    def test_rejects_each_receipt_identity_mismatch(self):
        for path, value in (
            (("request_sha256",), "9" * 64),
            (("identity", "source_commit"), "9" * 40),
            (("identity", "plan_sha256"), "9" * 64),
            (("identity", "pins", "dope_sha256"), "9" * 64),
            (("identity", "dataset_id"), "9" * 16),
            (("identity", "dataset_sha256"), "9" * 64),
            (("identity", "cut_sha256"), "9" * 64),
            (("identity", "guidance"), "heuristic"),
            (("identity", "seed"), 1),
            (("identity", "requested_budget"), 4),
        ):
            with self.subTest(path=path):
                receipt = copy.deepcopy(self.receipt)
                target = receipt
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assert_refused(
                    receipt, pattern="request mismatch|identity mismatch"
                )

    def test_requires_sha256_dataset_binding_identity(self):
        for binding in (None, "bad", "g" * 64):
            with self.subTest(binding=binding):
                receipt = copy.deepcopy(self.receipt)
                receipt["identity"]["dataset_binding_sha256"] = binding
                self.assert_refused(receipt, pattern="identity mismatch")

    def test_refuses_nonmatching_charged_budget_and_noncontiguous_attempts(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["charged"] = 2
        self.assert_refused(receipt, pattern="charged budget mismatch")
        receipt = copy.deepcopy(self.receipt)
        receipt["attempts"][2]["attempt"] = 4
        self.assert_refused(receipt, pattern="contiguous")

    def test_refuses_unknown_or_nonterminal_attempts(self):
        for outcome in ("unknown", "running", "cancelled"):
            with self.subTest(outcome=outcome):
                receipt = copy.deepcopy(self.receipt)
                receipt["attempts"][1]["outcome"] = outcome
                self.assert_refused(receipt, pattern="nonterminal")

    def test_completed_attempt_requires_two_finite_nonnegative_scores_loss_and_program_hash(
        self,
    ):
        invalid_cases = (
            ("fold_scores", []),
            ("fold_scores", [1.0]),
            ("fold_scores", [1.0, -0.1]),
            ("fold_scores", [1.0, float("inf")]),
            ("loss", float("nan")),
            ("loss", -0.1),
            ("program_sha256", "bad"),
        )
        for field, value in invalid_cases:
            with self.subTest(field=field, value=value):
                receipt = copy.deepcopy(self.receipt)
                receipt["attempts"][0][field] = value
                self.assert_refused(receipt, pattern="completed attempt")

    def test_failed_attempt_requires_typed_nonempty_failure_and_no_score(self):
        mutations = (
            ("failure", None),
            ("failure", {"code": "", "reason": "bad"}),
            ("failure", {"code": "E", "reason": "  "}),
            ("fold_scores", [0.1]),
            ("loss", 0.1),
            ("program_sha256", "7" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                receipt = copy.deepcopy(self.receipt)
                receipt["attempts"][1][field] = value
                self.assert_refused(receipt, pattern="typed failure")

    def test_requires_receipt_champion_to_match_lowest_loss_attempt(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["champion_program"] = "7" * 64
        self.assert_refused(receipt, pattern="champion")
        receipt = copy.deepcopy(self.receipt)
        del receipt["champion_program"]
        self.assert_refused(receipt, pattern="champion")

    def test_requires_nonempty_adapter_identity(self):
        for adapter in (None, "", "   "):
            with self.subTest(adapter=adapter):
                receipt = copy.deepcopy(self.receipt)
                receipt["adapter"] = adapter
                self.assert_refused(receipt, pattern="adapter identity")

    def test_duplicate_keys_and_nonfinite_json_are_refused(self):
        with self.assertRaises(ValueError):
            backend.decode(b'{"schema":"a","schema":"b"}')
        with self.assertRaises(ValueError):
            backend.decode(b'{"score":NaN}')

    def test_call_engine_posts_original_request_once_to_exact_loopback_route(self):
        observed = {}
        receipt_bytes = json.dumps(self.receipt, separators=(",", ":")).encode()
        engine_url = self.engine_url

        class Response:
            status = 200

            def geturl(self):
                return engine_url

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return receipt_bytes

        def fake_open(http_request, timeout):
            observed["method"] = http_request.get_method()
            observed["url"] = http_request.full_url
            observed["content_type"] = http_request.get_header("Content-type")
            observed["authorization"] = http_request.get_header("Authorization")
            observed["body"] = http_request.data
            observed["timeout"] = timeout
            return Response()

        opener = mock.Mock()
        opener.open.side_effect = fake_open
        with mock.patch.object(backend, "build_opener", return_value=opener) as builder:
            result = backend.call_engine(self.request, self.endpoint, timeout=7)
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(observed["method"], "POST")
        self.assertEqual(observed["url"], self.engine_url)
        self.assertEqual(observed["content_type"], "application/json")
        self.assertEqual(observed["authorization"], "Bearer " + self.daemon_token)
        self.assertEqual(observed["body"], backend.canonical(self.request))
        self.assertEqual(observed["timeout"], 7)
        handlers = builder.call_args.args
        self.assertEqual(len(handlers), 2)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], backend.NoRedirectHandler)
        self.assertEqual(result, self.receipt)

    def test_call_engine_refuses_non_loopback_endpoint(self):
        for endpoint in (
            "http://localhost:8123",
            "http://example.com:8123",
            "https://127.0.0.1:8123",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "loopback HTTP"):
                    backend.call_engine(self.request, endpoint)

    def test_call_engine_refuses_non_success_status_oversize_and_bad_json(self):
        engine_url = self.engine_url

        class Response:
            def __init__(self, body, status=200, url=None):
                self.body = body
                self.status = status
                self.url = url or engine_url

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return self.url

            def read(self, _limit):
                return self.body

        opener = mock.Mock()
        opener.open.return_value = Response(b"{}", status=503)
        with mock.patch.object(backend, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "non-success status"):
                backend.call_engine(self.request, self.endpoint)
        opener = mock.Mock()
        opener.open.return_value = Response(
            b"{}", url="http://127.0.0.1:8124/v1/q1/evaluations"
        )
        with mock.patch.object(backend, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "URL changed"):
                backend.call_engine(self.request, self.endpoint)
        opener = mock.Mock()
        opener.open.return_value = Response(b"x" * (8 * 1024 * 1024 + 1))
        with mock.patch.object(backend, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "size limit"):
                backend.call_engine(self.request, self.endpoint)
        opener = mock.Mock()
        opener.open.return_value = Response(b"not-json")
        with mock.patch.object(backend, "build_opener", return_value=opener):
            with self.assertRaises(ValueError):
                backend.call_engine(self.request, self.endpoint)

    def test_call_engine_refuses_missing_mismatched_and_broad_daemon_records(self):
        self.daemon_path.unlink()
        with mock.patch.object(backend, "build_opener") as builder:
            with self.assertRaisesRegex(
                ValueError, r"record is (?:unavailable|not a regular file)"
            ):
                backend.call_engine(self.request, self.endpoint)
            builder.assert_not_called()
        self.write_daemon_record(url="http://127.0.0.1:8124")
        with self.assertRaisesRegex(ValueError, "does not match"):
            backend.call_engine(self.request, self.endpoint)
        self.write_daemon_record()
        self.daemon_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "permissions are too broad"):
            backend.call_engine(self.request, self.endpoint)

    def test_call_engine_refuses_invalid_token_and_symlink_daemon_record(self):
        self.write_daemon_record(token="not-a-sha256")
        with mock.patch.object(backend, "build_opener") as builder:
            with self.assertRaisesRegex(ValueError, "token is invalid"):
                backend.call_engine(self.request, self.endpoint)
            builder.assert_not_called()
        self.write_daemon_record()
        link = self.daemon_path.with_name("daemon-link.json")
        link.symlink_to(self.daemon_path)
        with mock.patch.object(backend, "daemon_file_path", return_value=link):
            with mock.patch.object(backend, "build_opener") as builder:
                with self.assertRaisesRegex(
                    ValueError, r"record is (?:unavailable|not a regular file)"
                ):
                    backend.call_engine(self.request, self.endpoint)
                builder.assert_not_called()

    def test_call_engine_refuses_transport_failure_without_fallback(self):
        opener = mock.Mock()
        opener.open.side_effect = URLError("offline")
        with mock.patch.object(backend, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "request failed"):
                backend.call_engine(self.request, self.endpoint)
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
