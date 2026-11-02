from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from capture_stdio import child_environment, digest
from verify_capture import CaptureError, decode_json, require, run_retry, tool_calls


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
    original_outcomes = [
        call["data"]
        for call in before
        if call["name"] == "run_start" and call["arguments"].get("dry_run") is False
    ]
    require(
        all(
            json.dumps(data, sort_keys=True)
            == json.dumps(retry["data"], sort_keys=True)
            for data in original_outcomes
        ),
        "restart retry changed the original outcome",
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

    before_terminal, after_terminal = terminal(before), terminal(after)
    require(before_terminal == after_terminal, "terminal watermark or events changed")
    return dict(
        run_id=run,
        inventory_counts=[0, 1, 1, 1, 1],
        successful_attempts=3,
        event_watermark=after_terminal[0],
        original_outcome_preserved=True,
        structural_contract="PASS",
    )


def restart_probe(args):
    from probe_prefix_protocol import (
        RUN_TOOLS,
        Rpc,
        capture_wire,
        exited,
        outcome,
        probe,
        restart_proof,
        stop_owned,
    )

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
