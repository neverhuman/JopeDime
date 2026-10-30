#!/usr/bin/env python3
"""Bounded raw-wire MCP spike. FAIL is a measured outcome, not a harness crash.

This is a targeted two-tool suite, not the upstream certification suite or all A4.
Python is only the test driver; there is no product telemetry or dataset access.
"""

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import platform
import queue
import subprocess
import threading
import time

VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05", "2026-07-28")
MODERN = VERSIONS[-1]
PREFIX = "io.modelcontextprotocol/"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Wire:
    def __init__(self, executable, version, record):
        self.version = version
        self.record = record
        self.pending = {}
        self.lines = queue.Queue()
        self.stderr = []
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.threads = [
            threading.Thread(target=self.read_stdout, daemon=True),
            threading.Thread(target=self.read_stderr, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def read_stdout(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def read_stderr(self):
        total = 0
        for line in self.process.stderr:
            if total < 65536:
                self.stderr.append(line)
            total += len(line)

    def meta(self, version=None):
        return {
            PREFIX + "protocolVersion": version or self.version,
            PREFIX + "clientInfo": {"name": "mcp-spike-wire", "version": "1"},
            PREFIX + "clientCapabilities": {},
        }

    def send(self, method, params=None, request_id=1, metadata=True):
        params = dict(params or {})
        if self.version == MODERN and metadata:
            params["_meta"] = self.meta()
        message = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        if request_id is not None:
            message["id"] = request_id
        self.raw(json.dumps(message, ensure_ascii=False))

    def raw(self, line):
        self.record["wire"].append({"direction": "in", "line": line})
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def receive(self, timeout=1.5):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            raise AssertionError(f"no response within {timeout}s") from error
        require(line is not None, "server closed stdout before responding")
        self.record["wire"].append({"direction": "out", "line": line.rstrip("\n")})
        message = json.loads(line)
        require(
            isinstance(message, dict) and message.get("jsonrpc") == "2.0",
            "stdout must contain JSON-RPC objects only",
        )
        require(
            ("result" in message) != ("error" in message), "invalid response envelope"
        )
        return message

    def response(self, request_id=1, timeout=1.5):
        if request_id in self.pending:
            return self.pending.pop(request_id)
        deadline = time.monotonic() + timeout
        while True:
            message = self.receive(max(0.001, deadline - time.monotonic()))
            if message.get("id") == request_id:
                return message
            other_id = message.get("id")
            require(other_id not in self.pending, "duplicate response ID")
            self.pending[other_id] = message

    def call(self, method, params=None, request_id=1, metadata=True):
        self.send(method, params, request_id, metadata)
        return self.response(request_id)

    def result(self, method, params=None, request_id=1):
        response = self.call(method, params, request_id)
        require("result" in response, f"{method}: {response}")
        return response["result"]

    def ready(self):
        if self.version != MODERN:
            info = self.result(
                "initialize",
                {
                    "protocolVersion": self.version,
                    "capabilities": {},
                    "clientInfo": {"name": "mcp-spike-wire", "version": "1"},
                },
                "init",
            )
            require(
                info.get("protocolVersion") == self.version,
                f"negotiated {info.get('protocolVersion')}, requested {self.version}",
            )
            require("tools" in info.get("capabilities", {}), "tools capability missing")
            require(info.get("serverInfo", {}).get("name"), "server identity missing")
            require(info.get("instructions"), "server instructions missing")
            self.send("notifications/initialized", request_id=None)

    def finish(self):
        self.process.stdin.close()
        try:
            self.record["exit_code"] = self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.record["forced_shutdown"] = True
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.record["exit_code"] = self.process.returncode
        for thread in self.threads:
            thread.join(timeout=1)
        self.record["stderr"] = "".join(self.stderr)
        self.record["unmatched_responses"] = self.pending
        remaining = []
        while not self.lines.empty():
            line = self.lines.get_nowait()
            if line is not None:
                remaining.append(line.rstrip("\n"))
        self.record["trailing_stdout"] = remaining


def lifecycle(wire):
    wire.ready()
    if wire.version == MODERN:
        info = wire.result("server/discover")
        require(
            MODERN in info.get("supportedVersions", []), "modern version not advertised"
        )
        require("tools" in info.get("capabilities", {}), "tools capability missing")
        require(info.get("instructions"), "discover instructions missing")
        require(info.get("resultType") == "complete", "discover resultType missing")
    wire.result("tools/list", request_id="ready")


def catalog(wire):
    wire.ready()
    result = wire.result("tools/list")
    tools = result.get("tools", [])
    require(
        [tool.get("name") for tool in tools] == ["echo", "slow_echo"], "wrong catalog"
    )
    for tool in tools:
        require(
            tool.get("inputSchema", {}).get("type") == "object", "input schema missing"
        )
        require(
            tool.get("outputSchema", {}).get("type") == "object",
            "output schema missing",
        )
    if wire.version == MODERN:
        require(result.get("resultType") == "complete", "tools/list resultType missing")
        require(
            isinstance(result.get("ttlMs"), int) and result["ttlMs"] >= 0,
            "modern cache lifetime missing",
        )
        require(
            result.get("cacheScope") in ("private", "public"),
            "modern cache scope missing",
        )


def echo(wire, name="echo", text="UTF-8 ☃ / escaped\nnewline"):
    wire.ready()
    result = wire.result(
        "tools/call", {"name": name, "arguments": {"text": text, "delay_ms": 1}}
    )
    require(result.get("isError") is not True, "echo returned tool failure")
    require(
        result.get("structuredContent") == {"text": text}, "structured content differs"
    )
    require(
        json.loads(result["content"][0]["text"]) == {"text": text},
        "text content differs",
    )
    if wire.version == MODERN:
        require(result.get("resultType") == "complete", "tools/call resultType missing")


def tool_failure(wire):
    wire.ready()
    result = wire.result(
        "tools/call",
        {"name": "slow_echo", "arguments": {"text": "x", "delay_ms": 2001}},
    )
    require(result.get("isError") is True, "invalid arguments must be a tool failure")
    require(result.get("content"), "tool failure needs readable content")
    if wire.version == MODERN:
        require(
            result.get("resultType") == "complete", "tool failure resultType missing"
        )


def unknown_tool(wire):
    wire.ready()
    response = wire.call("tools/call", {"name": "does_not_exist", "arguments": {}})
    require(
        response.get("error", {}).get("code") == -32602,
        "unknown tool must use the dispatcher's invalid-params error framing",
    )


def unknown_method(wire):
    wire.ready()
    response = wire.call("spike/unknown")
    require(
        response.get("error", {}).get("code") == -32601, "unknown method must be -32601"
    )


def string_id(wire):
    wire.ready()
    wire.result("tools/list", request_id="string-id-☃")


def notification(wire):
    wire.ready()
    wire.result("tools/list", request_id="bootstrap")
    wire.send("notifications/spike", request_id=None)
    wire.result("tools/list", request_id="barrier")
    require(not wire.pending, "notification produced a reply")


def version_validation(wire, requested="1900-01-01"):
    if wire.version != MODERN:
        response = wire.result(
            "initialize",
            {
                "protocolVersion": requested,
                "capabilities": {},
                "clientInfo": {"name": "wire", "version": "1"},
            },
        )
        require(
            response.get("protocolVersion") in VERSIONS[:-1], "invalid legacy fallback"
        )
        wire.send("notifications/initialized", request_id=None)
        wire.result("tools/list", request_id="fallback")
    else:
        params = {"_meta": wire.meta(requested)}
        response = wire.call("tools/list", params, metadata=False)
        error = response.get("error", {})
        require(
            error.get("code") == -32022, f"expected unsupported version: {response}"
        )
        require(
            error.get("data", {}).get("requested") == requested,
            "requested version missing",
        )
        require(
            MODERN in error.get("data", {}).get("supported", []),
            "supported versions missing",
        )
        wire.result("tools/list", request_id="retry-supported")


def metadata(wire):
    require(wire.version == MODERN, "modern-only case")
    # Each request must supply capabilities, even after a successful previous call.
    wire.result("tools/list", request_id="first")
    response = wire.call("tools/list", request_id="missing", metadata=False)
    require(
        response.get("error", {}).get("code") == -32602, "missing _meta was accepted"
    )
    wire.result("tools/list", request_id="recover")


def responsiveness(wire):
    wire.ready()
    latencies = []
    for index in range(20):
        slow_id, fast_id = f"slow-{index}", f"fast-{index}"
        wire.send(
            "tools/call",
            {"name": "slow_echo", "arguments": {"text": "slow", "delay_ms": 200}},
            slow_id,
        )
        # Give the slow request time to enter dispatch before sending the fast one.
        time.sleep(0.02)
        start = time.monotonic()
        response = wire.call(
            "tools/call", {"name": "echo", "arguments": {"text": "fast"}}, fast_id
        )
        latencies.append((time.monotonic() - start) * 1000)
        require(
            response.get("result", {}).get("structuredContent") == {"text": "fast"},
            "fast call failed",
        )
        slow = wire.response(slow_id)
        require(
            slow.get("result", {}).get("structuredContent") == {"text": "slow"},
            "slow call failed",
        )
    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    wire.record["latency"] = {
        "samples_ms": latencies,
        "p95_ms": p95,
        "threshold_ms": 100,
        "slow_ms": 200,
        "method": "echo during slow_echo; not product run_control",
    }
    require(p95 < 100, f"fixture p95 {p95:.3f}ms >= 100ms")


def malformed(wire):
    wire.ready()
    wire.result("tools/list", request_id="before-malformed")
    wire.raw("{")
    try:
        response = wire.receive(timeout=0.3)
    except AssertionError as error:
        wire.record["parse_error_observation"] = str(error)
        wire.result("tools/list", request_id="after-malformed")
        wire.record["recovered_after_malformed"] = True
        raise AssertionError(
            "no parse-error response; subsequent valid request succeeds"
        ) from error
    require(
        response.get("error", {}).get("code") == -32700, "parse error response missing"
    )
    require(response.get("id") is None, "parse error ID must be null")


def cancellation(wire):
    wire.ready()
    wire.send(
        "tools/call",
        {"name": "slow_echo", "arguments": {"text": "cancel", "delay_ms": 500}},
        "cancel",
    )
    time.sleep(0.02)
    wire.send(
        "notifications/cancelled",
        {"requestId": "cancel", "reason": "spike"},
        request_id=None,
    )
    wire.result("tools/list", request_id="barrier")
    time.sleep(0.55)
    require("cancel" not in wire.pending, "cancelled request emitted a late response")
    while not wire.lines.empty():
        message = wire.receive(timeout=0.1)
        require(
            message.get("id") != "cancel", "cancelled request emitted a late response"
        )


def initialization_order(wire):
    response = wire.call("tools/list")
    require("error" in response, "tools accepted before legacy initialize")
    wire.ready()
    wire.result("tools/list", request_id="after")


def run_case(executable, version, name, group, test):
    record = {"case": name, "group": group, "wire": []}
    start = time.monotonic()
    wire = Wire(executable, version, record)
    try:
        test(wire)
        record["status"] = "PASS"
    except (AssertionError, ValueError, BrokenPipeError, OSError) as error:
        record["status"] = "FAIL"
        record["reason"] = str(error)
    finally:
        wire.finish()
        if record["status"] == "PASS":
            if record.get("forced_shutdown") or record["exit_code"] != 0:
                record.update(
                    status="FAIL", reason="server did not shut down cleanly on EOF"
                )
            elif record["trailing_stdout"] or record["unmatched_responses"]:
                record.update(status="FAIL", reason="unexpected extra stdout responses")
        record["wall_seconds"] = time.monotonic() - start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    spike = pathlib.Path(__file__).resolve().parent
    source_files = {}
    for path in sorted(spike.rglob("*")):
        if (
            "evidence" not in path.relative_to(spike).parts
            and path.is_file()
            and (
                path.suffix in (".rs", ".toml")
                or path.name in ("Cargo.lock", "DESIGN.md", "conformance.py")
            )
        ):
            source_files[str(path.relative_to(spike))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=spike, text=True
    ).strip()
    result = {
        "schema": "jope.mcp-spike/v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "source_head": head,
        "source_files_sha256": source_files,
        "scope": "bounded raw-wire two-tool suite; no upstream certification or full A4 claim",
        "servers": {},
    }
    for name in ("dispatcher", "rmcp3"):
        executable = (args.bin_dir / f"mcp-spike-{name}").resolve()
        server = {
            "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "versions": {},
        }
        result["servers"][name] = server
        for version in VERSIONS:
            tests = [
                ("lifecycle", lifecycle),
                ("catalog", catalog),
                ("echo", echo),
                ("slow_echo", lambda wire: echo(wire, "slow_echo")),
                ("tool_failure", tool_failure),
                ("unknown_tool", unknown_tool),
                ("unknown_method", unknown_method),
                ("string_id", string_id),
                ("notification", notification),
                ("version_validation", version_validation),
                (
                    "excluded_2025_03_26",
                    lambda wire: version_validation(wire, "2025-03-26"),
                ),
            ]
            if version == MODERN:
                tests.append(("per_request_metadata", metadata))
            cases = [
                run_case(executable, version, case, "protocol", test)
                for case, test in tests
            ]
            cases.append(
                run_case(
                    executable, version, "responsiveness", "latency", responsiveness
                )
            )
            cases.append(
                run_case(executable, version, "malformed_json", "diagnostic", malformed)
            )
            group = "protocol" if version == MODERN else "cancellation"
            cases.append(
                run_case(executable, version, "cancellation", group, cancellation)
            )
            if version != MODERN:
                cases.append(
                    run_case(
                        executable,
                        version,
                        "pre_init_order",
                        "diagnostic",
                        initialization_order,
                    )
                )
            protocol = [case for case in cases if case["group"] == "protocol"]
            status = (
                "PASS" if all(case["status"] == "PASS" for case in protocol) else "FAIL"
            )
            server["versions"][version] = {"protocol_status": status, "cases": cases}
            print(
                f"{name:10} {version}: protocol {status} ({sum(c['status'] == 'PASS' for c in protocol)}/{len(protocol)})",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transcripts = args.output.with_suffix(".jsonl")
    with transcripts.open("w") as stream:
        for name, server in result["servers"].items():
            for version, observation in server["versions"].items():
                for case in observation["cases"]:
                    raw = {
                        "server": name,
                        "version": version,
                        "case": case["case"],
                        "wire": case.pop("wire"),
                        "stderr": case.pop("stderr"),
                    }
                    stream.write(
                        json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
    result["transcripts"] = {
        "file": transcripts.name,
        "sha256": hashlib.sha256(transcripts.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    # Both candidates may legitimately fail. Exit 0 means the experiment completed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
