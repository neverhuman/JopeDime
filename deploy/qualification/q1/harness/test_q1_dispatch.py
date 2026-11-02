"""Synthetic dispatch controls; no evaluator, dataset, daemon or model inputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import q1_dispatch as dispatch
import test_production_backend


class DispatchTests(unittest.TestCase):
    def setUp(self):
        fixture = test_production_backend.ProductionBackendTests()
        fixture.setUp()
        self.request = fixture.request
        import production_backend

        self.response = production_backend.validate_receipt(
            fixture.receipt, self.request
        )

    def test_backend_http_diagnostic_is_redacted_and_exit_retained(self):
        raw = json.dumps(
            {
                "schema": "jope.q1-backend-error/v1",
                "phase": "engine_http",
                "http_status": 500,
                "engine_code": "IO_FAILURE",
                "detail": "Bearer PRIVATE_SENTINEL",
                "token": "PRIVATE_SENTINEL",
            }
        ).encode()
        with patch.object(
            dispatch.q1,
            "_invoke_backend",
            return_value=subprocess.CompletedProcess([], 2, b"", raw),
        ):
            with self.assertRaises(dispatch.BackendFatal) as caught:
                dispatch.invoke(self.request, ["unused"], {}, 1)
        evidence = caught.exception.diagnostic
        self.assertEqual(evidence["backend_returncode"], 2)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(evidence))
        self.assertEqual(
            json.loads(evidence["stderr_redacted"])["engine_code"], "IO_FAILURE"
        )

    def test_unstructured_oversized_or_hostile_values_are_omitted(self):
        for raw in [
            b"Bearer PRIVATE_SENTINEL",
            b"x" * 8192,
            b'{"schema":"jope.q1-backend-error/v1","phase":["PRIVATE_SENTINEL"]}',
        ]:
            evidence = dispatch.redacted_stderr(raw)
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(evidence))
            self.assertEqual(evidence["stderr_bytes"], len(raw))

    def test_first_fatal_is_durable_and_prevents_second_dispatch(self):
        error = subprocess.CompletedProcess([], -9, b"", b"Bearer PRIVATE_SENTINEL")
        with tempfile.TemporaryFile(mode="w+") as stream:
            with patch.object(
                dispatch.q1, "_invoke_backend", return_value=error
            ) as invoke:
                with self.assertRaises(dispatch.BackendFatal):
                    dispatch.dispatch_units(
                        [self.request, self.request], ["unused"], {}, 1, stream
                    )
                self.assertEqual(invoke.call_count, 1)
            stream.seek(0)
            rows = [json.loads(line) for line in stream]
        self.assertEqual([row["event"] for row in rows], ["started", "driver_failure"])
        self.assertNotIn("response", rows[1])
        self.assertEqual(rows[1]["diagnostic"]["backend_returncode"], -9)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(rows))

    def test_success_is_valid_and_driver_failure_still_fails_schema(self):
        result = subprocess.CompletedProcess(
            [], 0, dispatch.q1.canonical(self.response), b""
        )
        with tempfile.TemporaryFile(mode="w+") as stream:
            with patch.object(dispatch.q1, "_invoke_backend", return_value=result):
                dispatch.dispatch_units([self.request], ["unused"], {}, 1, stream)
            stream.seek(0)
            rows = [json.loads(line) for line in stream]
        self.assertEqual(rows[1]["event"], "finished")
        self.assertGreaterEqual(rows[1]["wall_seconds"], 0)
        self.assertEqual(
            dispatch.q1.response_valid(rows[1]["response"], self.request), self.response
        )
        with self.assertRaisesRegex(ValueError, "invalid response fields/schema"):
            dispatch.q1.response_valid(
                {"status": "failed", "reason": "driver"}, self.request
            )

    def test_timeout_stops_with_unknown_charge_and_no_retry(self):
        with patch.object(
            dispatch.q1,
            "_invoke_backend",
            side_effect=subprocess.TimeoutExpired(["unused"], 1, stderr=b"private"),
        ) as invoke:
            with self.assertRaises(dispatch.BackendFatal) as caught:
                dispatch.invoke(self.request, ["unused"], {}, 1)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(caught.exception.diagnostic["phase"], "timeout")
        self.assertIsNone(caught.exception.diagnostic["backend_returncode"])


if __name__ == "__main__":
    unittest.main()
