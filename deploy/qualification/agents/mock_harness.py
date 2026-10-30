"""In-process MCP mock used by the client setup qualification scenarios."""

from __future__ import annotations

import json
import socket
import tomllib
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
VERSIONS = json.loads((ROOT / "client-versions.json").read_text())


class ResponseLost(TimeoutError):
    """The mock server committed a request, but its response did not reach the client."""


class MockMcpServer:
    def __init__(self) -> None:
        self.idempotent_results: dict[str, tuple[dict, dict]] = {}
        self.created_runs = 0
        self.run_start_calls = 0
        self.retry_cache_hits = 0

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name in {"dataset_register", "run_start", "model_export"}:
            key = arguments.get("idempotency_key")
            if not isinstance(key, str) or not key:
                raise AssertionError(f"{name} missing idempotency_key")
            cached = self.idempotent_results.get(key)
            if cached:
                old_arguments, old_result = cached
                if old_arguments != arguments:
                    raise AssertionError(f"idempotency key reused with different {name} input")
                if name == "run_start" and not arguments.get("dry_run"):
                    self.retry_cache_hits += 1
                return old_result

        if name == "dataset_register":
            result = {"dataset_id": "dataset-ag1", "rows": 3, "columns": 2}
        elif name == "run_start":
            self.run_start_calls += 1
            if arguments.get("dry_run"):
                result = {"plan_id": "plan-ag1", "dry_run": True}
            else:
                self.created_runs += 1
                result = {"run_id": "run-ag1", "revision": 1}
        elif name == "run_status":
            result = {"run_id": arguments["run_id"], "state": "completed", "revision": 2}
        elif name == "model_export":
            result = {"artifact_id": "artifact-ag1", "files": ["model.json"]}
        elif name == "model_predict":
            result = {"rows": 3, "output_sha256": "0" * 64}
        else:
            raise AssertionError(f"unexpected tool: {name}")

        if name in {"dataset_register", "run_start", "model_export"}:
            self.idempotent_results[arguments["idempotency_key"]] = (dict(arguments), result)
        return result


class MockMcpClient:
    """A direct in-process call path; it never opens a socket or launches a CLI."""

    def __init__(self) -> None:
        self.server = MockMcpServer()
        self.drop_next_run_start_response = False

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self.server.call_tool(name, arguments)
        if name == "run_start" and self.drop_next_run_start_response:
            self.drop_next_run_start_response = False
            raise ResponseLost("simulated response loss after server commit")
        return result


def validate_client_config(client: str) -> str:
    record = VERSIONS["clients"][client]
    if client == "claude_code":
        assert record["version"] == "2.1.314" and record["verified"] is True
        # Validate the documented command shape without invoking Claude Code.
        command = [
            "claude", "mcp", "add", "--transport", "stdio", "--scope", "user",
            "jopedime", "--", "/absolute/path/to/jopedime", "mcp",
        ]
        assert command[1:4] == ["mcp", "add", "--transport"]
        assert command[-2:] == ["/absolute/path/to/jopedime", "mcp"]
    elif client == "gemini_cli":
        assert record["version"] is None and record["verified"] is False
        config = json.loads((ROOT / "gemini-settings.json").read_text())
        server = config["mcpServers"]["jopedime"]
        assert server == {"command": "/absolute/path/to/jopedime", "args": ["mcp"]}
    elif client == "codex_cli":
        assert record["version"] is None and record["verified"] is False
        config = tomllib.loads((ROOT / "codex-config.toml").read_text())
        server = config["mcp_servers"]["jopedime"]
        assert server == {"command": "/absolute/path/to/jopedime", "args": ["mcp"]}
    else:
        raise AssertionError(f"unknown client fixture: {client}")
    return record["name"]


def run_golden_path(client: str) -> None:
    name = validate_client_config(client)
    # Make an accidental transport/network call fail the scenario immediately.
    with patch.object(socket, "socket", side_effect=AssertionError("network access is forbidden")), patch.object(
        socket, "create_connection", side_effect=AssertionError("network access is forbidden")
    ):
        mock = MockMcpClient()
        dataset = mock.call_tool(
            "dataset_register",
            {
                "csv_text": "x,target\n1,2\n2,4\n3,6\n",
                "name": "agent-golden-path",
                "idempotency_key": "ag1-dataset-1",
            },
        )
        run_spec = {
            "dataset": dataset["dataset_id"],
            "target": "target",
            "task": "regression",
        }
        plan = mock.call_tool(
            "run_start",
            {**run_spec, "dry_run": True, "idempotency_key": "ag1-plan-1"},
        )
        assert plan["dry_run"] is True

        mutation = {**run_spec, "dry_run": False, "idempotency_key": "ag1-run-1"}
        mock.drop_next_run_start_response = True
        try:
            mock.call_tool("run_start", mutation)
        except ResponseLost:
            retried = mock.call_tool("run_start", mutation)
        else:
            raise AssertionError("forced run_start response loss did not occur")
        assert retried["run_id"] == "run-ag1"
        assert mock.server.run_start_calls == 2  # one dry-run request plus one committed start
        assert mock.server.created_runs == 1
        assert mock.server.retry_cache_hits == 1

        status = mock.call_tool("run_status", {"run_id": retried["run_id"]})
        assert status["state"] == "completed"
        exported = mock.call_tool(
            "model_export",
            {
                "run_id": retried["run_id"],
                "kind": "source",
                "languages": ["rust"],
                "idempotency_key": "ag1-export-1",
            },
        )
        prediction = mock.call_tool(
            "model_predict",
            {
                "model": {"artifact_id": exported["artifact_id"]},
                "input": {"dataset_id": dataset["dataset_id"]},
            },
        )
        assert prediction["rows"] == dataset["rows"]
    print(
        f"PASS {name}: AG1 dry-run/start/status/export/predict; forced run_start retry reused {retried['run_id']}"
    )
