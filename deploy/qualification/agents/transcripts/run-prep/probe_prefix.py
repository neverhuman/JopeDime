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


TOOLS = frozenset({"system_info", "dataset_register", "dataset_inspect"})
RUN_TOOLS = TOOLS | {"run_start", "run_status", "run_events"}


def exited(child):
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def stop_owned(child, grace):
    """Reserve the leader PID through the final group signal, including normal exit."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while exited(child) is None and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return child.wait()


class Rpc:
    def __init__(self, child, deadline, tools=TOOLS):
        self.child = child
        self.deadline = deadline
        self.tools = tools
        self.buffer = bytearray()
        self.next_id = 0

    def send(self, value):
        require(
            value.get("method")
            in {"initialize", "notifications/initialized", "tools/call"},
            "RPC method is outside prefix preparation",
        )
        if value["method"] == "tools/call":
            require(
                value.get("params", {}).get("name") in self.tools,
                "tool is outside the selected preparation",
            )
        raw = (json.dumps(value, ensure_ascii=False) + "\n").encode()
        require(len(raw) <= 1024 * 1024, "request exceeds frame limit")
        self.child.stdin.write(raw)
        self.child.stdin.flush()

    def request(self, method, params):
        self.next_id += 1
        identity = self.next_id
        self.send(dict(jsonrpc="2.0", id=identity, method=method, params=params))
        while True:
            if b"\n" in self.buffer:
                raw, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                value = decode_json(raw)
                require(
                    isinstance(value, dict) and value.get("jsonrpc") == "2.0",
                    "bad RPC frame",
                )
                if "id" not in value:
                    require("method" in value, "invalid notification")
                    continue
                require(
                    type(value["id"]) is int and value["id"] == identity,
                    "unpaired RPC reply",
                )
                require(("result" in value) != ("error" in value), "invalid RPC reply")
                return value
            remaining = self.deadline - time.monotonic()
            require(remaining > 0, "prefix RPC deadline exceeded")
            readable, _, _ = select.select([self.child.stdout], [], [], remaining)
            require(bool(readable), "prefix RPC deadline exceeded")
            chunk = os.read(self.child.stdout.fileno(), 65536)
            require(bool(chunk), "recorder closed before the reply")
            self.buffer.extend(chunk)
            require(len(self.buffer) <= 1024 * 1024, "reply exceeds frame limit")

    def tool(self, name, arguments):
        require(name in self.tools, "tool is outside the selected preparation")
        return self.request("tools/call", dict(name=name, arguments=arguments))


def outcome(reply):
    result = reply.get("result")
    if "error" in reply:
        return False, reply["error"]
    require(isinstance(result, dict), "tool result is not an object")
    require(
        result.get("isError") is None or type(result["isError"]) is bool,
        "invalid isError flag",
    )
    payload = result.get("structuredContent")
    if payload is None:
        content = result.get("content", [])
        require(
            isinstance(content, list) and all(isinstance(x, dict) for x in content),
            "invalid tool content",
        )
        texts = [x.get("text") for x in content if x.get("type") == "text"]
        require(
            len(texts) == 1 and isinstance(texts[0], str),
            "missing structured tool payload",
        )
        payload = decode_json(texts[0].encode())
    return result.get("isError") is not True, payload


def prefix(rpc, fixture, key, results=None):
    if results is None:
        results = []

    def call(step, name, arguments):
        ok, payload = outcome(rpc.tool(name, arguments))
        results.append(
            dict(
                step=step,
                tool=name,
                status="success" if ok else "error",
                payload=payload,
            )
        )
        return ok, payload

    call("system", "system_info", {})
    call("inventory_before", "dataset_inspect", {})
    arguments = dict(
        path=str(fixture), name="mcp05-prefix-training-only", idempotency_key=key
    )
    first_ok, first = call("register", "dataset_register", arguments)
    second_ok, second = call("register_retry", "dataset_register", arguments)
    dataset = first.get("dataset_id") if first_ok and isinstance(first, dict) else None
    if isinstance(dataset, str) and dataset:
        call("profile", "dataset_inspect", dict(dataset_id=dataset))
    else:
        results.append(
            dict(
                step="profile",
                tool="dataset_inspect",
                status="not_run",
                reason="registration returned no successful dataset ID",
            )
        )
    call("inventory_after", "dataset_inspect", {})
    repeated_id = (
        (
            isinstance(dataset, str)
            and bool(dataset)
            and second_ok
            and isinstance(second, dict)
            and second.get("dataset_id") == dataset
        )
        if first_ok and second_ok
        else None
    )
    return results, repeated_id


def prefix_complete(results, repeated_id, transport_complete):
    return (
        transport_complete
        and repeated_id is True
        and len(results) == 6
        and all(x["status"] == "success" for x in results)
    )


def run_replay(rpc, dataset, target, key, results):
    """One bounded real run; protocol evidence is never named-client qualification."""

    def call(step, name, arguments):
        ok, payload = outcome(rpc.tool(name, arguments))
        results.append(
            dict(
                step=step,
                tool=name,
                status="success" if ok else "error",
                payload=payload,
            )
        )
        require(ok and isinstance(payload, dict), "run replay refused at " + step)
        return payload

    spec = dict(
        dataset=dataset,
        target=target,
        task="regression",
        ignore_columns=[],
        test_dataset=None,
        split=dict(
            kind="random", holdout_fraction=0.2, group_column=None, time_column=None
        ),
        objective="rmse",
        effort="fast",
        budget=dict(evaluations="1", wall_seconds="5"),
        landmarks="none",
        guidance="heuristic",
        languages=["rust"],
        require_all_languages=True,
        device="cpu",
        seed="73",
        dry_run=True,
        plan_digest=None,
        idempotency_key=key + "-plan",
    )
    plan = call("run_plan", "run_start", spec)
    require(
        isinstance(plan.get("plan_digest"), str)
        and bool(plan["plan_digest"].strip())
        and isinstance(plan.get("placement"), dict)
        and plan["placement"].get("evaluation") == "cpu"
        and type(plan["placement"].get("workers")) is int
        and plan["placement"].get("workers") == 1,
        "plan must identify one CPU worker",
    )
    call("run_inventory_before", "run_status", {})
    start = dict(
        spec, dry_run=False, plan_digest=plan["plan_digest"], idempotency_key=key
    )
    first = call("run_create", "run_start", start)
    run = first.get("run_id")
    require(
        isinstance(run, str) and re.fullmatch(r"run_[0-7][0-9A-HJKMNP-TV-Z]{25}", run),
        "start returned no canonical run ID",
    )
    call("run_inventory_created", "run_status", {})
    retry = call("run_retry", "run_start", start)
    require(retry.get("run_id") == run, "retry changed run ID")
    call("run_inventory_retried", "run_status", {})
    cursor = 0
    last_kind = None
    while True:
        status = call(
            "run_poll",
            "run_status",
            dict(run_id=run, wait_seconds=0, after_event=str(cursor)),
        )
        require(status.get("run_id") == run, "status changed run ID")
        page = call(
            "run_events", "run_events", dict(run_id=run, after=str(cursor), limit=256)
        )
        events = page.get("events")
        require(isinstance(events, list) and len(events) <= 256, "invalid event page")
        for event in events:
            require(
                isinstance(event, dict)
                and event.get("run_id") == run
                and event.get("sequence") == str(cursor + 1),
                "event gap, duplicate or wrong run",
            )
            cursor += 1
            last_kind = event.get("kind")
            require(cursor <= 1024, "run replay event bound exceeded")
        require(page.get("next_after") == str(cursor), "event cursor differs")
        if status.get("state") in {"completed", "failed", "cancelled"}:
            require(status["state"] == "completed", "run did not complete successfully")
            if status.get("event_watermark") == str(cursor):
                require(
                    cursor > 0 and last_kind == "run.finished",
                    "completed run lacks its terminal event",
                )
                return dict(run_id=run, state="completed", event_watermark=str(cursor))
        require(time.monotonic() < rpc.deadline, "run replay deadline exceeded")
        time.sleep(0.25)


def probe(args, *, emit=True):
    require(hasattr(os, "WNOWAIT"), "POSIX waitid/WNOWAIT is required")
    require(
        args.run_target is None or bool(args.run_target.strip()),
        "run target must not be blank",
    )
    require(
        math.isfinite(args.seconds) and 0 < args.seconds <= 120,
        "seconds must be in (0,120]",
    )
    for value, length in [
        (args.binary_source_sha, 40),
        (args.harness_source_sha, 40),
        (args.binary_sha256, 64),
        (args.csv_sha256, 64),
    ]:
        require(
            re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            "invalid source/digest pin",
        )
    require(
        args.binary.is_file() and os.access(args.binary, os.X_OK), "missing executable"
    )
    require(digest(args.binary) == args.binary_sha256, "binary digest differs")
    require(
        args.csv.is_file() and args.csv.stat().st_size <= 1024 * 1024,
        "fixture exceeds 1 MiB",
    )
    require(digest(args.csv) == args.csv_sha256, "fixture digest differs")
    require(
        not args.out_dir.resolve().is_relative_to(Path("/tmp")),
        "never write under /tmp",
    )
    require(
        not any((p / ".git").exists() for p in args.out_dir.resolve().parents),
        "keep private captures outside Git working trees",
    )
    allowed = sorted(os.sched_getaffinity(0))
    slots = min(2, len(allowed))
    require(
        os.getloadavg()[0] + slots <= 0.85 * len(allowed),
        "host lacks CPU capacity below 85%",
    )
    os.sched_setaffinity(0, allowed[:slots])
    environment = child_environment(())
    os.umask(0o077)
    args.out_dir.mkdir(mode=0o700)  # Exclusive; preserve every earlier capture.
    for name in ["home", "engine-home", "tmp", "config", "data", "state", "cache"]:
        (args.out_dir / name).mkdir(mode=0o700)
    environment.update(
        PATH=os.defpath,
        HOME=str(args.out_dir / "home"),
        JOPEDIME_HOME=str(args.out_dir / "engine-home"),
        TMPDIR=str(args.out_dir / "tmp"),
        XDG_CONFIG_HOME=str(args.out_dir / "config"),
        XDG_DATA_HOME=str(args.out_dir / "data"),
        XDG_STATE_HOME=str(args.out_dir / "state"),
        XDG_CACHE_HOME=str(args.out_dir / "cache"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    fixture = args.out_dir / "train.csv"
    fixture.write_bytes(args.csv.read_bytes())
    require(digest(fixture) == args.csv_sha256, "fixture changed during copy")
    binary = str(args.binary.resolve())
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    deadline = clock + args.seconds
    engine = relay = None
    engine_exit = relay_exit = None
    results = []
    selected_tools = RUN_TOOLS if args.run_target else TOOLS
    replay = retry_contract = None
    repeated_id = None
    error = None
    engine_truncated = False
    relay_log = (args.out_dir / "relay.stderr").open("wb")
    try:
        engine = subprocess.Popen(
            [binary, "serve", "--idle-exit", "1m"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            require(exited(engine) is None, "owned engine exited before readiness")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "engine readiness deadline exceeded")
            status = subprocess.run(
                [binary, "serve", "--status"],
                env=environment,
                capture_output=True,
                timeout=min(2, remaining),
            )
            if status.returncode == 0 and status.stdout.startswith(b"running"):
                break
            time.sleep(0.05)
        recorder = Path(__file__).with_name("capture_stdio.py")
        relay = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(recorder),
                "--out-dir",
                str(args.out_dir / "capture"),
                "--seconds",
                str(max(0.1, deadline - time.monotonic())),
                "--max-bytes",
                "1048576",
                "--",
                binary,
                "mcp",
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=relay_log,
            start_new_session=True,
            bufsize=0,
        )
        rpc = Rpc(relay, deadline, selected_tools)
        hello = rpc.request(
            "initialize",
            dict(
                protocolVersion="2025-11-25",
                capabilities={},
                clientInfo=dict(
                    name="jope-run-replay" if args.run_target else "jope-prefix-probe",
                    version="1",
                ),
            ),
        )
        require("result" in hello, "initialize refused")
        rpc.send(dict(jsonrpc="2.0", method="notifications/initialized", params={}))
        results, repeated_id = prefix(
            rpc, fixture, "mcp05-prefix-" + args.out_dir.name, results
        )
        if args.run_target:
            require(
                repeated_id is True,
                "run replay requires successful dataset registration retry",
            )
            replay = run_replay(
                rpc,
                results[2]["payload"]["dataset_id"],
                args.run_target,
                "mcp05-run-" + args.out_dir.name,
                results,
            )
    except (CaptureError, OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finally:
        if relay:
            if relay.stdin and not relay.stdin.closed:
                try:
                    relay.stdin.close()
                except BrokenPipeError:
                    pass
            until = min(deadline, time.monotonic() + 3)
            while exited(relay) is None and time.monotonic() < until:
                time.sleep(0.02)
            relay_exit = stop_owned(relay, 7)
            relay.stdout.close()
        relay_log.close()
        if engine:
            try:
                subprocess.run(
                    [binary, "serve", "--stop"],
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            engine_exit = stop_owned(engine, 2)
            os.set_blocking(engine.stdout.fileno(), False)
            diagnostic = engine.stdout.read(65537) or b""
            engine_truncated = len(diagnostic) > 65536
            (args.out_dir / "engine.log").write_bytes(diagnostic[:65536])
            engine.stdout.close()
    capture_path = args.out_dir / "capture/capture.json"
    capture = {}
    try:
        if capture_path.is_file():
            capture = decode_json(capture_path.read_bytes())
        require(isinstance(capture, dict), "capture receipt is not an object")
        require(
            digest(args.binary) == args.binary_sha256, "binary changed during probe"
        )
        require(digest(fixture) == args.csv_sha256, "fixture changed during probe")
        if capture:
            require(
                capture.get("executable_sha256") == args.binary_sha256,
                "captured executable differs from the admitted binary",
            )
    except (CaptureError, OSError) as exc:
        error = error or str(exc)
        capture = {}
    if args.run_target and capture.get("complete") is True:
        try:
            retry_contract = run_retry(
                (args.out_dir / "capture/wire.jsonl").read_bytes()
            )
            if replay is not None:
                require(
                    retry_contract["run_id"] == replay["run_id"],
                    "wire replay identity differs",
                )
        except (CaptureError, OSError) as exc:
            error = error or str(exc)
    complete = relay_exit == 0 and capture.get("complete") is True
    prefix_success = prefix_complete(results[:6], repeated_id, complete)
    success = error is None and (
        complete and replay is not None and retry_contract is not None
        if args.run_target
        else prefix_success
    )
    receipt = dict(
        schema="jope.mcp-run-replay/v1"
        if args.run_target
        else "jope.mcp-prefix-probe/v1",
        started_utc=started,
        elapsed_seconds=time.monotonic() - clock,
        harness_source_sha=args.harness_source_sha,
        server_source_sha=args.binary_source_sha,
        binary_sha256=args.binary_sha256,
        fixture_sha256=args.csv_sha256,
        harness_sha256=digest(Path(__file__)),
        allowed_tools=sorted(selected_tools),
        results=results,
        dataset_retry_same_id=repeated_id,
        dataset_persistence_proved=False,
        transport_complete=complete,
        prefix_success=prefix_success and error is None,
        error=error,
        engine_exit=engine_exit,
        engine_pid=engine.pid if engine else None,
        engine_home=str((args.out_dir / "engine-home").resolve()),
        relay_exit=relay_exit,
        engine_log_truncated=engine_truncated,
        capture_receipt_sha256=digest(capture_path) if capture_path.is_file() else None,
        named_client_runs=0,
        actual_LLM_calls=0,
        a5_status="NOT_QUALIFIED",
        cpu_affinity=allowed[:slots],
        environment_names=sorted(environment),
        scope=(
            "Bounded CPU run protocol only; no model/control tool or named-client qualification"
            if args.run_target
            else "Protocol harness only; no run/model/control tool, fit, or named-client qualification"
        ),
    )
    if args.run_target:
        receipt.update(
            run_replay_success=success,
            run_replay=replay,
            run_retry_contract=retry_contract,
            run_persistence_proved=False,
            response_loss_injected=False,
        )
    name = "run-replay.json" if args.run_target else "prefix.json"
    (args.out_dir / name).write_text(json.dumps(receipt, indent=2) + "\n")
    if emit:
        print(json.dumps(receipt, indent=2))
    return 0 if success else 2


def capture_wire(root, binary_sha):
    manifest = root / "capture.json"
    require(
        manifest.is_file()
        and not manifest.is_symlink()
        and manifest.stat().st_size <= 65536,
        "capture receipt is unavailable or oversized",
    )
    receipt = decode_json(manifest.read_bytes())
    require(isinstance(receipt, dict), "capture receipt is not an object")
    require(
        receipt.get("complete") is True
        and type(receipt.get("server_exit_code")) is int
        and receipt.get("server_exit_code") == 0
        and type(receipt.get("relay_exit_code")) is int
        and receipt.get("relay_exit_code") == 0
        and receipt.get("executable_sha256") == binary_sha,
        "capture is incomplete or has a different executable",
    )
    files = receipt.get("files")
    require(
        isinstance(files, dict)
        and set(files) == {"wire.jsonl", "client.raw", "server.raw", "server.stderr"},
        "unexpected capture sidecars",
    )
    for name, description in files.items():
        path = root / name
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= 1048576,
            "capture sidecar is unavailable or oversized",
        )
        require(
            isinstance(description, dict)
            and type(description.get("bytes")) is int
            and path.stat().st_size == description["bytes"]
            and digest(path) == description.get("sha256"),
            "capture sidecar hash or length differs",
        )
    return (root / "wire.jsonl").read_bytes()


def restart_proof(before_raw, after_raw):
    first = run_retry(before_raw)
    require(first["inventory_counts"] == [0, 1, 1], "first store was not isolated")
    run = first["run_id"]
    before, after = tool_calls(before_raw), tool_calls(after_raw)
    originals = [
        call["arguments"]
        for call in before
        if call["name"] == "run_start" and call["arguments"].get("dry_run") is False
    ]
    require(len(originals) == 2, "first capture lacks explicit mutation bodies")
    original = originals[0]
    require(
        original.get("device") == "cpu"
        and original.get("guidance") == "heuristic"
        and original.get("budget") == {"evaluations": "1", "wall_seconds": "5"},
        "restart proof requires the bounded CPU run",
    )
    require(
        all(
            call["name"] in {"run_start", "run_status", "run_events"} for call in after
        ),
        "restart session called an unrelated tool",
    )
    starts = [call for call in after if call["name"] == "run_start"]
    require(len(starts) == 1, "need exactly one restart retry")
    retry = starts[0]
    require(
        json.dumps(original, sort_keys=True)
        == json.dumps(retry["arguments"], sort_keys=True),
        "restart retry changed the original body or key",
    )
    require(
        isinstance(retry["data"], dict) and retry["data"].get("run_id") == run,
        "restart retry failed or returned a different run",
    )
    pages = [
        call
        for call in after
        if call["name"] == "run_status" and call["arguments"] == {}
    ]
    require(
        len(pages) == 2
        and pages[0]["end"] < retry["start"]
        and retry["end"] < pages[1]["start"],
        "missing inventories bracketing the restart retry",
    )
    for page in pages:
        data = page["data"]
        require(
            isinstance(data, dict)
            and data.get("next_after", "missing") is None
            and isinstance(data.get("runs"), list)
            and len(data["runs"]) == 1
            and isinstance(data["runs"][0], dict)
            and data["runs"][0].get("run_id") == run,
            "persisted inventory differs, is failed or paginated",
        )

    def terminal(calls):
        snapshots = [
            call["data"]
            for call in calls
            if call["name"] == "run_status" and call["arguments"].get("run_id") == run
        ]
        require(bool(snapshots), "missing terminal snapshot")
        snapshot = snapshots[-1]
        require(
            isinstance(snapshot, dict)
            and snapshot.get("run_id") == run
            and snapshot.get("state") == "completed",
            "run is not completed after restart",
        )
        watermark = snapshot.get("event_watermark")
        require(
            isinstance(watermark, str)
            and re.fullmatch(r"[1-9][0-9]{0,3}", watermark) is not None
            and int(watermark) <= 1024,
            "invalid terminal watermark",
        )
        events = []
        for call in calls:
            if call["name"] != "run_events":
                continue
            data = call["data"]
            require(
                call["arguments"].get("run_id") == run
                and call["arguments"].get("after") == str(len(events))
                and isinstance(data, dict)
                and isinstance(data.get("events"), list)
                and len(data["events"]) <= 256,
                "invalid persisted event page",
            )
            for event in data["events"]:
                require(
                    isinstance(event, dict)
                    and event.get("run_id") == run
                    and event.get("sequence") == str(len(events) + 1),
                    "persisted event gap, duplicate or wrong run",
                )
                events.append(event)
                require(len(events) <= 1024, "persisted event bound exceeded")
            require(data.get("next_after") == str(len(events)), "event cursor differs")
        require(
            len(events) == int(watermark) and events[-1].get("kind") == "run.finished",
            "persisted terminal event is missing",
        )
        return watermark, json.dumps(events, sort_keys=True)

    require(terminal(before) == terminal(after), "terminal watermark or events changed")
    return dict(
        run_id=run,
        inventory_counts=[0, 1, 1, 1, 1],
        successful_attempts=3,
        event_watermark=terminal(after)[0],
        structural_contract="PASS",
    )


def restart_probe(args):
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    deadline = clock + args.seconds
    host_cpus = len(os.sched_getaffinity(0))
    first_exit = probe(args, emit=False)
    first = decode_json((args.out_dir / "run-replay.json").read_bytes())
    engine = relay = None
    engine_exit = relay_exit = None
    error = None
    proof = None
    log = None
    home = args.out_dir / "engine-home"
    try:
        require(first_exit == 0, "first-stage run replay failed")
        require(
            first.get("engine_home") == str(home.resolve())
            and type(first.get("engine_pid")) is int
            and first.get("engine_exit") is not None,
            "first engine has no stopped private-store identity",
        )
        before = capture_wire(args.out_dir / "capture", args.binary_sha256)
        require(
            digest(args.out_dir / "capture/capture.json")
            == first.get("capture_receipt_sha256"),
            "first capture receipt changed",
        )
        calls = tool_calls(before)
        original = next(
            call["arguments"]
            for call in calls
            if call["name"] == "run_start" and call["arguments"].get("dry_run") is False
        )
        run = first["run_replay"]["run_id"]
        require(time.monotonic() < deadline, "combined restart deadline exceeded")
        require(os.getloadavg()[0] + 2 <= 0.85 * host_cpus, "host lacks CPU headroom")
        require(
            digest(args.binary) == args.binary_sha256, "binary changed before restart"
        )
        environment = child_environment(())
        environment.update(
            PATH=os.defpath,
            HOME=str(args.out_dir / "home"),
            JOPEDIME_HOME=str(home),
            TMPDIR=str(args.out_dir / "tmp"),
            **{
                "XDG_" + kind + "_HOME": str(args.out_dir / directory)
                for kind, directory in [
                    ("CONFIG", "config"),
                    ("DATA", "data"),
                    ("STATE", "state"),
                    ("CACHE", "cache"),
                ]
            },
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            RAYON_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )
        engine = subprocess.Popen(
            [str(args.binary), "serve", "--idle-exit", "1m"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        require(engine.pid != first["engine_pid"], "engine identity did not change")
        while True:
            require(exited(engine) is None, "restarted engine exited before readiness")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "combined restart deadline exceeded")
            status = subprocess.run(
                [str(args.binary), "serve", "--status"],
                env=environment,
                capture_output=True,
                timeout=min(2, remaining),
            )
            if status.returncode == 0 and status.stdout.startswith(b"running"):
                break
            time.sleep(0.05)
        log = (args.out_dir / "restart-relay.stderr").open("wb")
        relay = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("capture_stdio.py")),
                "--out-dir",
                str(args.out_dir / "capture-restart"),
                "--seconds",
                str(max(0.1, deadline - time.monotonic())),
                "--max-bytes",
                "1048576",
                "--",
                str(args.binary),
                "mcp",
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
            bufsize=0,
        )
        rpc = Rpc(relay, deadline, RUN_TOOLS)
        hello = rpc.request(
            "initialize",
            dict(
                protocolVersion="2025-11-25",
                capabilities={},
                clientInfo=dict(name="jope-restart-probe", version="1"),
            ),
        )
        require("result" in hello, "restart initialize refused")
        rpc.send(dict(jsonrpc="2.0", method="notifications/initialized", params={}))

        def call(name, arguments):
            ok, data = outcome(rpc.tool(name, arguments))
            require(ok and isinstance(data, dict), "restart refused at " + name)
            return data

        call("run_status", {})
        call("run_start", original)
        call("run_status", {})
        snapshot = call("run_status", dict(run_id=run, wait_seconds=0, after_event="0"))
        cursor = 0
        while True:
            page = call("run_events", dict(run_id=run, after=str(cursor), limit=256))
            events = page.get("events")
            require(isinstance(events, list), "invalid restarted event page")
            cursor += len(events)
            require(0 < cursor <= 1024, "restarted event bound exceeded")
            if snapshot.get("event_watermark") == str(cursor):
                break
            require(bool(events), "restarted event page made no progress")
    except (
        CaptureError,
        OSError,
        subprocess.SubprocessError,
        StopIteration,
        KeyError,
    ) as exc:
        error = str(exc) or type(exc).__name__
    finally:
        if relay:
            try:
                relay.stdin.close()
            except BrokenPipeError:
                pass
            until = min(deadline, time.monotonic() + 3)
            while exited(relay) is None and time.monotonic() < until:
                time.sleep(0.02)
            relay_exit = stop_owned(relay, 7)
            relay.stdout.close()
        if log:
            log.close()
        if engine:
            engine_exit = stop_owned(engine, 2)
            os.set_blocking(engine.stdout.fileno(), False)
            diagnostic = engine.stdout.read(65537) or b""
            (args.out_dir / "restart-engine.log").write_bytes(diagnostic[:65536])
            engine.stdout.close()
    try:
        if error is None:
            require(relay_exit == 0, "restart relay failed")
            after = capture_wire(args.out_dir / "capture-restart", args.binary_sha256)
            require(
                digest(args.binary) == args.binary_sha256,
                "binary changed during restart",
            )
            proof = restart_proof(before, after)
    except (CaptureError, OSError) as exc:
        error = str(exc)
    receipt = dict(
        schema="jope.mcp-completed-restart/v1",
        started_utc=started,
        elapsed_seconds=time.monotonic() - clock,
        seconds_budget=args.seconds,
        harness_source_sha=args.harness_source_sha,
        server_source_sha=args.binary_source_sha,
        binary_sha256=args.binary_sha256,
        fixture_sha256=args.csv_sha256,
        engine_home=str(home.resolve()),
        first_engine_pid=first.get("engine_pid"),
        restarted_engine_pid=engine.pid if engine else None,
        first_engine_exit=first.get("engine_exit"),
        restarted_engine_exit=engine_exit,
        restarted_relay_exit=relay_exit,
        first_stage_exit=first_exit,
        prefix_success=first.get("prefix_success"),
        restart_success=error is None and proof is not None,
        run_persistence_proved=error is None and proof is not None,
        restart_contract=proof,
        error=error,
        named_client_runs=0,
        actual_LLM_calls=0,
        a5_status="NOT_QUALIFIED",
        response_loss_injected=False,
        scope="Completed-run restart/retry only; no SIGKILL cuts or named-client qualification",
    )
    (args.out_dir / "run-restart.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["restart_success"] else 2


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
