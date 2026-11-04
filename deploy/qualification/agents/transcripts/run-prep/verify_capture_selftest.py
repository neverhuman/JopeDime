"""Synthetic-only controls for the named-client verifier."""

import hashlib
import json
from pathlib import Path

def run(api) -> int:
    """Synthetic verifier controls only; never launch a client or engine."""
    CaptureError = api["CaptureError"]
    _CLIENTS = api["_CLIENTS"]
    decode_json = api["decode_json"]
    run_retry = api["run_retry"]
    verify = api["verify"]
    import tempfile
    import unittest

    run_id = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV"
    old_id = "run_01ARZ3NDEKTSV4RRFFQ69G5FAW"
    extra_id = "run_01ARZ3NDEKTSV4RRFFQ69G5FAX"

    def frames():
        wire = []
        args = {"idempotency_key": "new-run", "seed": "42"}

        def page(ids):
            return {"runs": [{"run_id": item} for item in ids], "next_after": None}

        calls = [
            ("run_status", {}, page([old_id])),
            ("run_start", args, {"run_id": run_id}),
            ("run_status", {}, page([old_id, run_id])),
            ("run_start", args, {"run_id": run_id}),
            ("run_status", {}, page([old_id, run_id])),
        ]
        for rpc_id, (name, arguments, data) in enumerate(calls):
            params = {"name": name, "arguments": dict(arguments)}
            request = dict(jsonrpc="2.0", id=rpc_id, method="tools/call", params=params)
            response = dict(
                jsonrpc="2.0",
                id=rpc_id,
                result=dict(isError=False, structuredContent=data),
            )
            wire.append(dict(direction="client_to_server", message=request))
            wire.append(dict(direction="server_to_client", message=response))
        return wire

    def encode(wire):
        return b"\n".join(json.dumps(row).encode() for row in wire)

    class Controls(unittest.TestCase):
        def test_success_retains_existing_run(self):
            self.assertEqual(
                run_retry(encode(frames())),
                dict(run_id=run_id, successful_attempts=2, inventory_counts=[1, 2, 2]),
            )

        def test_corrupted_retries_inventory_and_rpc_ids_refuse(self):
            args = ("params", "arguments")
            data = ("result", "structuredContent")

            def runs(ids):
                return [{"run_id": item} for item in ids]

            cases = [
                (6, (*args, "idempotency_key"), "other", "arguments"),
                (6, (*args, "seed"), "43", "arguments"),
                (2, (*args, "idempotency_key"), " ", "idempotency key"),
                (2, (*args, "dry_run"), 0, "Boolean"),
                (7, (*data, "run_id"), old_id, "different run ID"),
                (7, ("result", "isError"), True, "successful canonical"),
                (3, ("result", "structuredContent"), {}, "successful canonical"),
                (9, (*data, "next_after"), "more", "paginated"),
                (
                    9,
                    (*data, "runs"),
                    runs([old_id, run_id, run_id]),
                    "duplicate inventory",
                ),
                (
                    9,
                    (*data, "runs"),
                    runs([old_id, run_id, extra_id]),
                    "changed after retry",
                ),
                (9, (*data, "runs"), runs([run_id]), "changed after retry"),
                (1, (*data, "runs"), runs([old_id, run_id]), "first creation"),
                (9, ("result", "structuredContent"), {"runs": []}, "paginated"),
                (9, (*data, "runs"), [{"run_id": "invalid"}], "invalid inventory"),
                (0, ("id",), False, "RPC ID type"),
                (1, ("id",), "0", "no matching request"),
            ]
            for index, keys, value, error in cases:
                with self.subTest(index=index, keys=keys, value=value):
                    wire = frames()
                    target = wire[index]["message"]
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    with self.assertRaisesRegex(CaptureError, error):
                        run_retry(encode(wire))

        def test_order_missing_page_pending_and_duplicate_requests_refuse(self):
            for order in [
                [1, 0, *range(2, 10)],
                [*range(4), 6, 7, 4, 5, 8, 9],
                list(range(9)),
                [0, 0, *range(1, 10)],
                [*range(4), *range(6, 10)],
            ]:
                with self.subTest(order=order), self.assertRaises(CaptureError):
                    wire = frames()
                    run_retry(encode([wire[i] for i in order]))

        def test_id_reuse_after_completion_and_lifecycle_messages(self):
            wire = frames()
            for index in [6, 7]:
                wire[index]["message"]["id"] = 1
            prefix = [
                dict(
                    direction="client_to_server",
                    message=dict(jsonrpc="2.0", id="init", method="initialize"),
                ),
                dict(
                    direction="server_to_client",
                    message=dict(jsonrpc="2.0", id="init", result={}),
                ),
                dict(
                    direction="client_to_server",
                    message=dict(jsonrpc="2.0", method="notifications/initialized"),
                ),
            ]
            self.assertEqual(
                run_retry(encode(prefix + wire))["inventory_counts"], [1, 2, 2]
            )

        def test_json_types_duplicate_keys_and_nonfinite_numbers_refuse(self):
            for before, after in [(False, 0), (1, 1.0)]:
                wire = frames()
                wire[2]["message"]["params"]["arguments"]["seed"] = before
                wire[6]["message"]["params"]["arguments"]["seed"] = after
                with (
                    self.subTest(before=before, after=after),
                    self.assertRaises(CaptureError),
                ):
                    run_retry(encode(wire))
            for raw in [
                b'{"id":1,"id":2}',
                b'{"number":NaN}',
                b'{"number":1e999}',
                '{"number":1}'.encode("utf-16"),
                b"\xff",
            ]:
                with self.subTest(raw=raw), self.assertRaises(CaptureError):
                    decode_json(raw)

        def test_end_to_end_manifest_execution_pins_and_nonpromotion(self):
            with tempfile.TemporaryDirectory(prefix="mcp-retry-control-") as directory:
                root = Path(directory)
                files = {}

                def put(name, raw):
                    (root / name).write_bytes(raw)
                    files[name] = dict(
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
                    )

                for client, slug in _CLIENTS.items():
                    put(f"{slug}.mcp.jsonl", encode(frames()))
                    put(
                        f"{slug}.execution.json",
                        json.dumps(
                            dict(
                                client=client,
                                exit_code=0,
                                engine_exit=0,
                                timed_out=False,
                            )
                        ).encode(),
                    )
                manifest = dict(
                    schema="jopedime.named-client-transcripts/v1",
                    files=files,
                    clients={
                        client: dict(version="control-only") for client in _CLIENTS
                    },
                    a5=dict(status="QUALIFIED"),
                )
                path = root / "manifest.json"

                def check():
                    path.write_text(json.dumps(manifest))
                    return verify(
                        path,
                        root,
                        check_run_retry=True,
                        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )

                result = check()
                self.assertEqual(result["a5_status"], "UNVERIFIED:QUALIFIED")
                self.assertEqual(len(result["run_retry_contract"]), 3)
                with self.assertRaises(CaptureError):
                    verify(path, root, check_run_retry=True)
                with self.assertRaises(CaptureError):
                    verify(path, root, check_run_retry=True, manifest_sha256="0" * 64)
                manifest["clients"]["gemini_cli"]["version"] = " "
                with self.assertRaises(CaptureError):
                    check()
                manifest["clients"]["gemini_cli"]["version"] = "control-only"
                for field, value in [
                    ("timed_out", True),
                    ("exit_code", False),
                    ("engine_exit", 1),
                ]:
                    execution = dict(
                        client="claude_code",
                        exit_code=0,
                        engine_exit=0,
                        timed_out=False,
                    )
                    execution[field] = value
                    put("claude-code.execution.json", json.dumps(execution).encode())
                    with self.subTest(field=field), self.assertRaises(CaptureError):
                        check()
                put(
                    "claude-code.execution.json",
                    json.dumps(
                        dict(
                            client="claude_code",
                            exit_code=0,
                            engine_exit=0,
                            timed_out=False,
                        )
                    ).encode(),
                )
                (root / "claude-code.mcp.jsonl").write_bytes(b"changed")
                with self.assertRaises(CaptureError):
                    check()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


