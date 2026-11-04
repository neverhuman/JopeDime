#!/usr/bin/env python3
"""Bounded system/dataset MCP preparation; never run AG1 or qualify a named client."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

from capture_stdio import child_environment, digest
from verify_capture import CaptureError, decode_json, require, run_retry, tool_calls

from probe_prefix_protocol import (
    TOOLS,
    RUN_TOOLS,
    exited,
    stop_owned,
    Rpc,
    outcome,
    prefix,
    prefix_complete,
    run_replay,
    probe,
    capture_wire,
    restart_proof,
    restart_probe,
)


def self_test():
    import copy
    import io
    import tempfile
    from types import SimpleNamespace
    import unittest
    from unittest.mock import Mock, patch

    def reply(payload, error=False):
        return dict(result=dict(structuredContent=payload, isError=error))

    def wire(pairs):
        frames = []
        for identity, (name, arguments, data) in enumerate(pairs, 1):
            frames.extend(
                [
                    dict(
                        direction="client_to_server",
                        message=dict(
                            jsonrpc="2.0",
                            id=identity,
                            method="tools/call",
                            params=dict(name=name, arguments=arguments),
                        ),
                    ),
                    dict(
                        direction="server_to_client",
                        message=dict(
                            jsonrpc="2.0",
                            id=identity,
                            **reply(data),
                        ),
                    ),
                ]
            )
        return b"\n".join(json.dumps(frame).encode() for frame in frames)

    class Controls(unittest.TestCase):
        def test_restart_checks_persisted_identity_body_and_events(self):
            run = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV"
            other = "run_01ARZ3NDEKTSV4RRFFQ69G5FAW"
            body = dict(
                dry_run=False,
                idempotency_key="same-key",
                device="cpu",
                guidance="heuristic",
                budget=dict(evaluations="1", wall_seconds="5"),
            )
            page = dict(runs=[dict(run_id=run)], next_after=None)
            snapshot = dict(run_id=run, state="completed", event_watermark="1")
            events = dict(
                events=[dict(run_id=run, sequence="1", kind="run.finished")],
                next_after="1",
            )
            before = [
                ["run_status", {}, dict(runs=[], next_after=None)],
                ["run_start", body, dict(run_id=run)],
                ["run_status", {}, page],
                ["run_start", body, dict(run_id=run)],
                ["run_status", {}, page],
                ["run_status", dict(run_id=run), snapshot],
                ["run_events", dict(run_id=run, after="0"), events],
            ]
            after = [
                ["run_status", {}, page],
                ["run_start", body, dict(run_id=run)],
                ["run_status", {}, page],
                ["run_status", dict(run_id=run), snapshot],
                ["run_events", dict(run_id=run, after="0"), events],
            ]
            proof = restart_proof(wire(before), wire(after))
            self.assertEqual(proof["inventory_counts"], [0, 1, 1, 1, 1])
            self.assertEqual(proof["run_id"], run)
            changes = [
                (0, 2, dict(runs=[dict(run_id=other)], next_after=None)),
                (1, 1, dict(body, idempotency_key="changed")),
                (1, 1, dict(body, dry_run=0)),
                (1, 2, dict(run_id=other)),
                (1, 2, dict(run_id=run, state="changed")),
                (
                    2,
                    2,
                    dict(runs=[dict(run_id=run), dict(run_id=other)], next_after=None),
                ),
                (2, 2, dict(runs=[dict(run_id=run)], next_after="more")),
                (3, 2, dict(snapshot, state="paused")),
                (
                    4,
                    2,
                    dict(
                        events=[dict(run_id=run, sequence="1", kind="changed")],
                        next_after="1",
                    ),
                ),
                (4, 2, dict(events=[], next_after="0")),
            ]
            for index, field, replacement in changes:
                with self.subTest(index=index, field=field, replacement=replacement):
                    changed = copy.deepcopy(after)
                    changed[index][field] = replacement
                    with self.assertRaises(CaptureError):
                        restart_proof(wire(before), wire(changed))
            with self.assertRaises(CaptureError):
                restart_proof(wire(before), wire(after[1:]))

        def test_restart_capture_rejects_modified_sidecars_and_boolean_exit(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                files = {}
                for name in ["wire.jsonl", "client.raw", "server.raw", "server.stderr"]:
                    (root / name).write_bytes(b"payload")
                    files[name] = dict(bytes=7, sha256=digest(root / name))
                receipt = dict(
                    complete=True,
                    server_exit_code=0,
                    relay_exit_code=0,
                    executable_sha256="a" * 64,
                    files=files,
                )
                manifest = root / "capture.json"
                manifest.write_text(json.dumps(receipt))
                self.assertEqual(capture_wire(root, "a" * 64), b"payload")
                (root / "wire.jsonl").write_bytes(b"changed")
                with self.assertRaises(CaptureError):
                    capture_wire(root, "a" * 64)
                (root / "wire.jsonl").write_bytes(b"payload")
                receipt["server_exit_code"] = False
                manifest.write_text(json.dumps(receipt))
                with self.assertRaises(CaptureError):
                    capture_wire(root, "a" * 64)

        def test_restart_never_runs_after_first_stage_failure(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "run-replay.json").write_text(json.dumps(dict(error="refused")))
                args = SimpleNamespace(
                    out_dir=root,
                    seconds=1,
                    harness_source_sha="a" * 40,
                    binary_source_sha="b" * 40,
                    binary_sha256="c" * 64,
                    csv_sha256="d" * 64,
                )
                with (
                    patch(__name__ + ".probe", return_value=2),
                    patch(__name__ + ".subprocess.Popen") as spawn,
                    patch(__name__ + ".print"),
                ):
                    self.assertEqual(restart_probe(args), 2)
                spawn.assert_not_called()
                receipt = json.loads((root / "run-restart.json").read_text())
                self.assertFalse(receipt["restart_success"])
                self.assertFalse(receipt["run_persistence_proved"])
                self.assertIsNone(receipt["restart_contract"])
                self.assertEqual(receipt["a5_status"], "NOT_QUALIFIED")

        def test_restart_cli_requires_explicit_run_mode(self):
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__)), "--restart-retry"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --run-target", result.stderr)

        def test_forbidden_tools_refuse_before_writing(self):
            child = Mock()
            rpc = Rpc(child, time.monotonic() + 1)
            for name in [
                "run_start",
                "run_status",
                "model_export",
                "model_predict",
                "run_control",
            ]:
                with self.subTest(name=name), self.assertRaises(CaptureError):
                    rpc.tool(name, dict(dry_run=True))
            child.stdin.write.assert_not_called()
            selected = Rpc(child, time.monotonic() + 1, RUN_TOOLS)
            for name in ["run_control", "run_fork", "model_export", "model_predict"]:
                with self.subTest(selected=name), self.assertRaises(CaptureError):
                    selected.tool(name, {})
            child.stdin.write.assert_not_called()
            with self.assertRaises(CaptureError):
                rpc.send(dict(method="tools/call", params=dict(name="run_start")))
            child.stdin.write.assert_not_called()

        def test_allowed_names_are_exact(self):
            self.assertEqual(
                TOOLS, {"system_info", "dataset_register", "dataset_inspect"}
            )
            self.assertEqual(
                RUN_TOOLS - TOOLS, {"run_start", "run_status", "run_events"}
            )

        def test_rpc_and_tool_errors_are_not_successes(self):
            self.assertEqual(
                outcome(dict(error=dict(code=-32000))), (False, dict(code=-32000))
            )
            self.assertEqual(
                outcome(reply(dict(code="CAPABILITY_UNAVAILABLE"), True)),
                (False, dict(code="CAPABILITY_UNAVAILABLE")),
            )

        def test_structured_and_text_payloads(self):
            self.assertEqual(
                outcome(reply(dict(dataset_id="ds_fixture"))),
                (True, dict(dataset_id="ds_fixture")),
            )
            self.assertEqual(
                outcome(dict(result=dict(content=[dict(type="text", text='{"x":1}')]))),
                (True, dict(x=1)),
            )

        def test_registration_retry_uses_identical_body(self):
            rpc = Mock()
            rpc.tool.side_effect = [
                reply({}),
                reply({}),
                reply(dict(dataset_id="ds_fixture")),
                reply(dict(dataset_id="ds_fixture")),
                reply({}),
                reply({}),
            ]
            results, same = prefix(rpc, Path("approved-train.csv"), "same-key")
            self.assertEqual(len(results), 6)
            self.assertTrue(same)
            self.assertTrue(prefix_complete(results, same, True))
            self.assertEqual(rpc.tool.call_args_list[2], rpc.tool.call_args_list[3])
            self.assertEqual(
                rpc.tool.call_args_list[4].args[1], dict(dataset_id="ds_fixture")
            )
            self.assertTrue(
                all(call.args[0] in TOOLS for call in rpc.tool.call_args_list)
            )
            run = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV"
            page = dict(runs=[dict(run_id=run)], next_after=None)
            replies = [
                reply(
                    dict(
                        plan_digest="reviewed-plan",
                        placement=dict(evaluation="cpu", workers=1),
                    )
                ),
                reply(dict(runs=[], next_after=None)),
                reply(dict(run_id=run)),
                reply(page),
                reply(dict(run_id=run)),
                reply(page),
                reply(dict(run_id=run, state="completed", event_watermark="1")),
                reply(
                    dict(
                        events=[dict(run_id=run, sequence="1", kind="run.finished")],
                        next_after="1",
                    )
                ),
            ]
            rpc = Mock(deadline=time.monotonic() + 1)
            rpc.tool.side_effect = replies
            observed = []
            completed = run_replay(
                rpc, "ds_fixture", "target", "same-run-key", observed
            )
            self.assertEqual(
                completed, dict(run_id=run, state="completed", event_watermark="1")
            )
            calls = rpc.tool.call_args_list
            self.assertEqual(calls[2], calls[4])
            self.assertTrue(calls[0].args[1]["dry_run"])
            self.assertFalse(calls[2].args[1]["dry_run"])
            self.assertEqual(
                calls[2].args[1]["budget"], dict(evaluations="1", wall_seconds="5")
            )
            # Exercise the real wire verifier too; equal counts cannot hide a changed set.
            wire = []
            for i, (call, response) in enumerate(zip(calls, replies), 1):
                wire.extend(
                    [
                        dict(
                            direction="client_to_server",
                            message=dict(
                                jsonrpc="2.0",
                                id=i,
                                method="tools/call",
                                params=dict(name=call.args[0], arguments=call.args[1]),
                            ),
                        ),
                        dict(
                            direction="server_to_client",
                            message=dict(jsonrpc="2.0", id=i, **response),
                        ),
                    ]
                )
            self.assertEqual(
                run_retry(b"\n".join(json.dumps(x).encode() for x in wire))[
                    "inventory_counts"
                ],
                [0, 1, 1],
            )
            for index, replacement in [
                (
                    0,
                    reply(
                        dict(
                            plan_digest="plan",
                            placement=dict(evaluation="cuda", workers=1),
                        )
                    ),
                ),
                (0, reply(dict(code="CAPABILITY_UNAVAILABLE"), True)),
                (4, reply(dict(run_id="run_01ARZ3NDEKTSV4RRFFQ69G5FAW"))),
                (6, reply(dict(run_id=run, state="failed", event_watermark="1"))),
                (
                    7,
                    reply(
                        dict(
                            events=[
                                dict(
                                    run_id=run, sequence="1", kind="experiment.started"
                                )
                            ],
                            next_after="1",
                        )
                    ),
                ),
                (
                    7,
                    reply(
                        dict(events=[dict(run_id=run, sequence="2")], next_after="2")
                    ),
                ),
            ]:
                with self.subTest(replay_refusal=index):
                    altered = list(replies)
                    altered[index] = replacement
                    rpc = Mock(deadline=time.monotonic() + 1)
                    rpc.tool.side_effect = altered
                    with self.assertRaises(CaptureError):
                        run_replay(rpc, "ds_fixture", "target", "same-key", [])

        def test_registration_failure_never_invents_dataset(self):
            rpc = Mock()
            rpc.tool.side_effect = [
                reply({}),
                reply({}),
                reply(dict(code="UNAVAILABLE"), True),
                reply(dict(code="UNAVAILABLE"), True),
                reply({}),
            ]
            results, same = prefix(rpc, Path("approved-train.csv"), "same-key")
            self.assertEqual(results[4]["status"], "not_run")
            self.assertIsNone(same)
            self.assertEqual(rpc.tool.call_count, 5)

        def test_changed_retry_identity_is_visible(self):
            rpc = Mock()
            rpc.tool.side_effect = [
                reply({}),
                reply({}),
                reply(dict(dataset_id="ds_first")),
                reply(dict(dataset_id="ds_second")),
                reply({}),
                reply({}),
            ]
            _, same = prefix(rpc, Path("approved-train.csv"), "same-key")
            self.assertFalse(same)
            self.assertFalse(prefix_complete([dict(status="success")] * 6, same, True))

        def test_rpc_reply_ids_are_paired_without_boolean_coercion(self):
            for identity in [True, 2, 1]:
                with self.subTest(identity=identity):
                    read_fd, write_fd = os.pipe()
                    try:
                        os.write(
                            write_fd,
                            (
                                json.dumps(dict(jsonrpc="2.0", id=identity, result={}))
                                + "\n"
                            ).encode(),
                        )
                        with os.fdopen(read_fd, "rb", buffering=0) as stream:
                            child = Mock(stdin=io.BytesIO(), stdout=stream)
                            rpc = Rpc(child, time.monotonic() + 1)
                            if type(identity) is int and identity == 1:
                                self.assertEqual(
                                    rpc.request("initialize", {})["result"], {}
                                )
                            else:
                                with self.assertRaises(CaptureError):
                                    rpc.request("initialize", {})
                    finally:
                        os.close(write_fd)

        def test_partial_reply_times_out(self):
            read_fd, write_fd = os.pipe()
            try:
                os.write(write_fd, b'{"jsonrpc":"2.0"')
                with os.fdopen(read_fd, "rb", buffering=0) as stream:
                    rpc = Rpc(
                        Mock(stdin=io.BytesIO(), stdout=stream), time.monotonic() + 0.02
                    )
                    with self.assertRaises(CaptureError):
                        rpc.request("initialize", {})
            finally:
                os.close(write_fd)

        def test_owned_term_ignoring_group_is_stopped(self):
            code = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'
            child = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                self.assertTrue(select.select([child.stdout], [], [], 2)[0])
                self.assertEqual(child.stdout.readline(), b"ready\n")
                self.assertEqual(stop_owned(child, 0.05), -signal.SIGKILL)
            finally:
                if child.returncode is None:
                    stop_owned(child, 0.05)
                child.stdout.close()

    return (
        0
        if unittest.TextTestRunner(verbosity=2)
        .run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
        .wasSuccessful()
        else 1
    )


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument("--binary-source-sha")
    parser.add_argument("--harness-source-sha")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--csv-sha256")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument(
        "--run-target",
        help="opt in to one CPU regression run and retry for this target",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--restart-retry",
        action="store_true",
        help="after a completed CPU run, restart its private engine and retry the same request",
    )
    args = parser.parse_args()
    if args.restart_retry and not args.run_target:
        parser.error("--restart-retry requires --run-target")
    if args.self_test:
        return self_test()
    if not all(
        [
            args.binary,
            args.binary_sha256,
            args.binary_source_sha,
            args.harness_source_sha,
            args.csv,
            args.csv_sha256,
            args.out_dir,
        ]
    ):
        parser.error("binary/source/input pins and a fresh out-dir are required")
    args.binary = args.binary.resolve()
    args.csv = args.csv.resolve()
    args.out_dir = args.out_dir.absolute()
    try:
        return restart_probe(args) if args.restart_retry else probe(args)
    except (CaptureError, OSError) as exc:
        print(f"Prefix probe did not complete: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    sys.exit(main())
