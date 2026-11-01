#!/usr/bin/env python3
"""Run a pinned candidate through three explicitly configured named MCP clients.

This records client executions and the capture_stdio wire sidecars. It does not
create semantic verification receipts or an A5 gate result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


CLIENTS = {
    "claude_code": ("claude-code", "claude"),
    "gemini_cli": (
        "gemini-cli",
        "Antigravity Gemini CLI (owner-approved mcp05_t1925_jrb02.md)",
    ),
    "codex_cli": ("codex-cli", "codex"),
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
PLACEHOLDERS = {
    "{candidate}",
    "{capture_dir}",
    "{capture_stdio}",
    "{dataset_id}",
    "{engine_home}",
    "{input_handoff_sha256}",
    "{source_sha}",
    "{train_path}",
}
ENV_NAMES = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
}


class CaptureError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_object(path: Path, label: str) -> tuple[dict, bytes]:
    require(not path.is_symlink() and path.is_file(), f"{label} must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"{label} is invalid JSON") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def check_source(repo: Path, source_sha: str) -> None:
    require(
        GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase hex",
    )
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CaptureError("cannot establish clean source identity") from error
    require(
        head == source_sha and not dirty,
        "checkout must be clean at the declared source SHA",
    )


def external_path(path: Path, repo: Path, label: str) -> Path:
    require(path.is_absolute(), f"{label} must be absolute")
    require(
        Path("/tmp") not in (path, *path.parents), f"{label} must not be under /tmp"
    )
    require(
        not any(part.is_symlink() for part in (path, *path.parents)),
        f"{label} contains a symlink",
    )
    resolved = path.resolve()
    require(
        not resolved.is_relative_to(repo),
        f"{label} must be outside the source checkout",
    )
    return resolved


def plan_clients(plan: dict, source_sha: str, candidate_sha: str) -> dict[str, dict]:
    require(
        plan.get("schema") == "jopedime.a5-three-client-command-plan/v1",
        "unsupported client plan schema",
    )
    require(plan.get("source_sha") == source_sha, "client plan source SHA mismatch")
    require(
        plan.get("candidate_artifact_sha256") == candidate_sha,
        "client plan candidate SHA mismatch",
    )
    clients = plan.get("clients")
    require(
        isinstance(clients, dict) and set(clients) == set(CLIENTS),
        "client plan must contain exactly the three named clients",
    )
    for key, (_slug, implementation) in CLIENTS.items():
        row = clients[key]
        require(isinstance(row, dict), f"client plan row must be an object: {key}")
        require(
            row.get("implementation") == implementation,
            f"client implementation mismatch: {key}",
        )
        require(
            isinstance(row.get("version"), str) and row["version"].strip(),
            f"client version is required: {key}",
        )
        executable = row.get("executable")
        executable_sha = row.get("executable_sha256")
        require(
            isinstance(executable, str) and Path(executable).is_absolute(),
            f"client executable path is required: {key}",
        )
        require(
            isinstance(executable_sha, str) and SHA.fullmatch(executable_sha),
            f"client executable SHA is invalid: {key}",
        )
        argv = row.get("argv")
        require(
            isinstance(argv, list)
            and argv
            and all(isinstance(arg, str) and "\0" not in arg for arg in argv),
            f"client argv must be a nonempty string array: {key}",
        )
        require(
            argv[0] == executable,
            f"client argv[0] must equal the pinned executable: {key}",
        )
        joined = "\n".join(argv)
        for marker in (
            "{capture_stdio}",
            "{capture_dir}",
            "{candidate}",
            "{engine_home}",
        ):
            require(
                marker in joined,
                f"client argv must configure the candidate capture server using {marker}: {key}",
            )
        names = row.get("environment_names", [])
        require(isinstance(names, list), f"environment_names must be an array: {key}")
        for name in names:
            require(
                isinstance(name, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None,
                f"invalid environment name: {key}",
            )
        timeout = row.get("timeout_seconds", 300)
        require(
            type(timeout) is int and 1 <= timeout <= 900,
            f"client timeout must be from 1 to 900 seconds: {key}",
        )
        row["timeout_seconds"] = timeout
    return clients


def selected_environment(extra_names: list[str], engine_home: Path) -> dict[str, str]:
    selected = {
        name: os.environ[name]
        for name in ENV_NAMES | set(extra_names)
        if name in os.environ
    }
    selected.setdefault("PATH", os.defpath)
    selected.update(
        JOPEDIME_HOME=str(engine_home),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    return selected


def render_argv(argv: list[str], values: dict[str, str]) -> list[str]:
    rendered = []
    for arg in argv:
        for marker, value in values.items():
            arg = arg.replace("{" + marker + "}", value)
        require(
            "{" not in arg and "}" not in arg,
            "argv contains an unsupported placeholder",
        )
        rendered.append(arg)
    return rendered


def terminate_group(process: subprocess.Popen, grace_seconds: float = 5) -> int:
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    def live_group_members() -> bool:
        proc = Path("/proc")
        if not proc.is_dir():
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False
        for entry in proc.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="ascii")
                fields = stat.rsplit(")", 1)[1].split()
                state, member_pgid = fields[0], int(fields[2])
            except (OSError, IndexError, ValueError):
                continue
            if member_pgid == pgid and state not in {"Z", "X"}:
                return True
        return False

    deadline = time.monotonic() + max(0, grace_seconds)
    while live_group_members() and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    if live_group_members():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 2
        while live_group_members() and time.monotonic() < kill_deadline:
            time.sleep(0.02)
        if live_group_members():
            raise CaptureError(f"process group {pgid} survived SIGKILL")
    return process.wait()


def render_client_commands(
    clients: dict[str, dict],
    *,
    candidate: Path,
    output: Path,
    engine_home: Path,
    capture_stdio: Path,
    dataset_id: str,
    handoff_sha: str,
    source_sha: str,
    train_path: Path,
) -> dict[str, list[str]]:
    commands = {}
    for key, row in clients.items():
        values = {
            "candidate": str(candidate),
            "capture_dir": str(output / "clients" / key / "transport"),
            "capture_stdio": str(capture_stdio),
            "dataset_id": dataset_id,
            "engine_home": str(engine_home / key),
            "input_handoff_sha256": handoff_sha,
            "source_sha": source_sha,
            "train_path": str(train_path),
        }
        command = render_argv(row["argv"], values)
        require(
            Path(command[0]).is_absolute(),
            f"client command must use an absolute executable: {key}",
        )
        commands[key] = command
    return commands


def stop_engine(
    candidate: Path,
    repo: Path,
    environment: dict[str, str],
    process: subprocess.Popen | None,
) -> tuple[int | None, int | None]:
    if process is None:
        return None, None
    stop_exit = None
    pending_error = None
    try:
        stop = subprocess.run(
            [str(candidate), "serve", "--stop"],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            preexec_fn=lambda: os.nice(19),
        )
        stop_exit = stop.returncode
    except BaseException as error:
        pending_error = error
    try:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except BaseException as error:
        if pending_error is None:
            pending_error = error
    try:
        engine_exit = terminate_group(process)
    except BaseException as error:
        if pending_error is None:
            pending_error = error
    if pending_error is not None:
        raise pending_error
    return stop_exit, engine_exit


def run_capture(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve(strict=True)
    check_source(repo, args.source_sha)
    candidate = external_path(args.candidate, repo, "candidate")
    require(
        candidate.is_file() and os.access(candidate, os.X_OK),
        "candidate must be an executable regular file",
    )
    candidate_sha = digest(candidate)
    require(candidate_sha == args.candidate_sha256, "candidate SHA-256 mismatch")
    handoff, handoff_raw = read_object(args.input_handoff, "TRAIN handoff")
    require(
        handoff.get("split") == "TRAIN", "input handoff must explicitly identify TRAIN"
    )
    dataset_id = handoff.get("dataset_id")
    require(
        isinstance(dataset_id, str) and dataset_id.strip(),
        "TRAIN handoff dataset_id is required",
    )
    handoff_sha = hashlib.sha256(handoff_raw).hexdigest()
    require(handoff_sha == args.input_handoff_sha256, "TRAIN handoff SHA-256 mismatch")
    train_path = external_path(args.train_path, repo, "TRAIN input")
    require(train_path.is_file(), "TRAIN input must be a regular file")
    plan, _ = read_object(args.client_plan, "client command plan")
    clients = plan_clients(plan, args.source_sha, candidate_sha)
    output = external_path(args.out_dir, repo, "capture output")
    engine_home = external_path(args.engine_home, repo, "engine home")
    require(not output.exists(), "capture output already exists")
    require(not engine_home.exists(), "candidate engine home already exists")
    capture_stdio = (
        repo / "deploy/qualification/agents/transcripts/run-prep/capture_stdio.py"
    )
    require(
        not capture_stdio.is_symlink() and capture_stdio.is_file(),
        "capture_stdio.py must be a regular source file",
    )

    for key, row in clients.items():
        executable = Path(row["executable"])
        require(
            executable.is_file() and os.access(executable, os.X_OK),
            f"client executable unavailable: {key}",
        )
        require(
            digest(executable) == row["executable_sha256"],
            f"client executable changed: {key}",
        )
    commands = render_client_commands(
        clients,
        candidate=candidate,
        output=output,
        engine_home=engine_home,
        capture_stdio=capture_stdio,
        dataset_id=dataset_id,
        handoff_sha=handoff_sha,
        source_sha=args.source_sha,
        train_path=train_path,
    )

    if args.preflight_only:
        return {
            "status": "PREFLIGHT_READY_NO_CLIENTS_LAUNCHED",
            "source_sha": args.source_sha,
            "candidate_artifact_sha256": candidate_sha,
            "input_handoff_sha256": handoff_sha,
            "clients": sorted(clients),
            "rendered_client_commands": commands,
            "a5_status": "NOT_QUALIFIED",
        }
    require(
        args.execute_named_clients,
        "launching named clients requires --execute-named-clients",
    )

    version = subprocess.run(
        [str(candidate), "--version", "--json"],
        env={
            **os.environ,
            "JOPEDIME_HOME": str(engine_home),
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
        preexec_fn=lambda: os.nice(19),
    )
    require(version.returncode == 0, "candidate version command failed")
    try:
        version_record = json.loads(version.stdout)
    except json.JSONDecodeError as error:
        raise CaptureError("candidate version output is invalid JSON") from error
    require(
        version_record.get("source_sha40") == args.source_sha,
        "candidate embedded source SHA mismatch",
    )
    require(digest(candidate) == candidate_sha, "candidate changed during preflight")

    output.mkdir(mode=0o700, parents=True)
    engine_home.mkdir(mode=0o700, parents=True)
    sidecars = output / "sidecars"
    sidecars.mkdir(mode=0o700)
    logs = output / "clients"
    logs.mkdir(mode=0o700)
    rows = {}
    engine_homes = {}
    started = datetime.now(timezone.utc).isoformat()

    for key, (slug, _client_name) in CLIENTS.items():
        row = clients[key]
        case = logs / key
        case.mkdir(mode=0o700)
        client_home = engine_home / key
        client_home.mkdir(mode=0o700)
        engine_homes[key] = str(client_home)
        environment = selected_environment([], client_home)
        engine = None
        active_client = None
        engine_stop_exit = None
        engine_exit = None
        client_exit = None
        timed_out = False
        capture_error = None
        capture = None
        capture_dir = case / "transport"
        command = commands[key]
        command_path = case / "command.json"
        command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        os.chmod(command_path, 0o600)

        try:
            with (
                (case / "engine.stdout").open("xb") as stdout,
                (case / "engine.stderr").open("xb") as stderr,
            ):
                engine = subprocess.Popen(
                    [str(candidate), "serve", "--idle-exit", "15m"],
                    cwd=repo,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    preexec_fn=lambda: os.nice(19),
                )

            ready = False
            for _ in range(200):
                if engine.poll() is not None:
                    break
                status = subprocess.run(
                    [str(candidate), "serve", "--status"],
                    cwd=repo,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3,
                    preexec_fn=lambda: os.nice(19),
                )
                if status.returncode == 0 and status.stdout.startswith("running"):
                    ready = True
                    break
                time.sleep(0.1)
            require(ready, f"{key} candidate engine did not become ready")

            with (
                (case / "stdout.log").open("xb") as stdout,
                (case / "stderr.log").open("xb") as stderr,
            ):
                try:
                    active_client = subprocess.Popen(
                        command,
                        cwd=case,
                        env=selected_environment(
                            row.get("environment_names", []), client_home
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                        preexec_fn=lambda: os.nice(19),
                    )
                except OSError as error:
                    capture_error = f"client launch failed: {error}"

                if active_client is not None:
                    try:
                        client_exit = active_client.wait(timeout=row["timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    finally:
                        client_exit = terminate_group(active_client)

            if active_client is not None:
                try:
                    require(
                        not capture_dir.is_symlink() and capture_dir.is_dir(),
                        f"{key} capture directory must be a regular directory",
                    )
                    capture, _ = read_object(
                        capture_dir / "capture.json", f"{key} transport receipt"
                    )
                    wire_path = capture_dir / "wire.jsonl"
                    require(
                        not wire_path.is_symlink()
                        and wire_path.is_file()
                        and wire_path.stat().st_size > 0,
                        f"{key} transport has no wire transcript",
                    )
                    require(
                        capture.get("complete") is True,
                        f"{key} transport capture is incomplete",
                    )
                    require(
                        capture.get("argv") == [str(candidate), "mcp"],
                        f"{key} transport launched a different candidate",
                    )
                    require(
                        capture.get("executable_sha256") == candidate_sha,
                        f"{key} transport candidate hash mismatch",
                    )
                    require(
                        capture.get("server_exit_code") == 0
                        and capture.get("relay_exit_code") == 0,
                        f"{key} transport server or relay failed",
                    )
                    limits = capture.get("limits")
                    require(
                        isinstance(limits, dict)
                        and limits.get("seconds", 0) <= 300
                        and limits.get("stream_bytes", 0) <= 16_777_216
                        and limits.get("frame_bytes") == 1_048_576,
                        f"{key} transport limits exceed the reviewed bounds",
                    )
                    require(
                        capture.get("extra_environment_names") == [],
                        f"{key} transport passed extra environment values",
                    )
                    wire_copy = sidecars / f"{slug}.mcp.jsonl"
                    wire_copy.write_bytes(wire_path.read_bytes())
                    os.chmod(wire_copy, 0o600)
                    capture_copy = sidecars / f"{slug}.capture.json"
                    capture_copy.write_bytes(
                        (capture_dir / "capture.json").read_bytes()
                    )
                    os.chmod(capture_copy, 0o600)
                except (OSError, CaptureError) as error:
                    capture_error = str(error)
        finally:
            try:
                if active_client is not None:
                    terminate_group(active_client)
            finally:
                engine_stop_exit, engine_exit = stop_engine(
                    candidate, repo, environment, engine
                )

        rows[key] = {
            "version": row["version"],
            "executable": row["executable"],
            "executable_sha256": row["executable_sha256"],
            "implementation": row["implementation"],
            "client_exit": client_exit,
            "timed_out": timed_out,
            "capture_error": capture_error,
            "capture_receipt_sha256": (
                digest(capture_dir / "capture.json")
                if (capture_dir / "capture.json").is_file()
                and not (capture_dir / "capture.json").is_symlink()
                else None
            ),
            "engine_home": str(client_home),
            "engine_exit": engine_exit,
            "engine_stop_exit": engine_stop_exit,
            "capture": capture,
        }

    complete = (
        len(rows) == len(CLIENTS)
        and all(row["engine_stop_exit"] == 0 for row in rows.values())
        and all(row["engine_exit"] == 0 for row in rows.values())
        and all(
            row["client_exit"] == 0
            and not row["timed_out"]
            and row["capture_error"] is None
            for row in rows.values()
        )
    )
    metadata = {
        key: {
            "version": row["version"],
            "executable": row["executable"],
            "executable_sha256": row["executable_sha256"],
            "implementation": row["implementation"],
        }
        for key, row in clients.items()
    }
    for key, (slug, _client_name) in CLIENTS.items():
        row = rows[key]
        execution = {
            "schema": "jopedime.named-client-execution/v1",
            "source": args.source_sha,
            "binary_sha256": candidate_sha,
            "client": key,
            "client_version": row["version"],
            "client_resolved_path": row["executable"],
            "client_launcher_sha256": row["executable_sha256"],
            "argv_sha256": hashlib.sha256(
                json.dumps(
                    json.loads(
                        (logs / key / "command.json").read_text(encoding="utf-8")
                    ),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "exit_code": row["client_exit"],
            "client_exit": row["client_exit"],
            "engine_exit": row["engine_exit"],
            "engine_stop_exit": row["engine_stop_exit"],
            "server_exit": (
                row["capture"].get("server_exit_code")
                if isinstance(row["capture"], dict)
                else None
            ),
            "relay_exit": (
                row["capture"].get("relay_exit_code")
                if isinstance(row["capture"], dict)
                else None
            ),
            "timed_out": row["timed_out"],
        }
        execution_path = sidecars / f"{slug}.execution.json"
        execution_path.write_text(
            json.dumps(execution, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(execution_path, 0o600)
        row.pop("capture", None)
    metadata_path = output / "client-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(metadata_path, 0o600)
    engine_exits = [row["engine_exit"] for row in rows.values()]
    stop_exits = [row["engine_stop_exit"] for row in rows.values()]
    result = {
        "schema": "jopedime.a5-three-client-capture/v1",
        "status": "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
        if complete
        else "INCOMPLETE",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": args.source_sha,
        "candidate_artifact_sha256": candidate_sha,
        "input_handoff_sha256": handoff_sha,
        "input_split": "TRAIN",
        "dataset_id": dataset_id,
        "engine_home": str(engine_home),
        "engine_homes": engine_homes,
        "engine_exit": 0
        if engine_exits and all(code == 0 for code in engine_exits)
        else None,
        "engine_stop_exit": 0
        if stop_exits and all(code == 0 for code in stop_exits)
        else None,
        "clients": rows,
        "sidecars": str(sidecars),
        "client_metadata": str(metadata_path),
        "semantic_receipts_created": False,
        "a5_status": "NOT_QUALIFIED",
    }
    result_path = output / "capture-run.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(result_path, 0o600)
    require(
        digest(candidate) == candidate_sha, "candidate changed during client capture"
    )
    return result


def self_test() -> None:
    """Data-free validation checks; does not execute a client or candidate."""
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="a5-three-client-selftest-", dir=Path.home()
    ) as directory:
        root = Path(directory)
        source = "a" * 40
        candidate_sha = "b" * 64
        plan = {
            "schema": "jopedime.a5-three-client-command-plan/v1",
            "source_sha": source,
            "candidate_artifact_sha256": candidate_sha,
            "clients": {},
        }
        for key, (_slug, implementation) in CLIENTS.items():
            executable = str(root / key)
            plan["clients"][key] = {
                "version": "fixture",
                "executable": executable,
                "executable_sha256": "c" * 64,
                "implementation": implementation,
                "argv": [
                    executable,
                    "{capture_stdio}",
                    "{capture_dir}",
                    "{candidate}",
                    "{engine_home}",
                ],
            }
        rows = plan_clients(plan, source, candidate_sha)
        rendered = render_argv(
            rows["claude_code"]["argv"],
            {
                "candidate": "/candidate",
                "capture_dir": "/capture",
                "capture_stdio": "/capture_stdio.py",
                "engine_home": "/engine",
            },
        )
        assert rendered[-2:] == ["/candidate", "/engine"]
        plan["clients"].pop("gemini_cli")
        try:
            plan_clients(plan, source, candidate_sha)
        except CaptureError:
            pass
        else:
            raise AssertionError("incomplete roster was accepted")
        try:
            render_argv(["client", "{unrecognized}"], {})
        except CaptureError:
            pass
        else:
            raise AssertionError("unknown command placeholder was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--input-handoff", type=Path)
    parser.add_argument("--input-handoff-sha256")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--client-plan", type=Path)
    parser.add_argument("--engine-home", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-named-clients", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("three-client capture plan controls: PASS (synthetic plan only)")
        return 0
    required = (
        args.repo,
        args.source_sha,
        args.candidate,
        args.candidate_sha256,
        args.input_handoff,
        args.input_handoff_sha256,
        args.train_path,
        args.client_plan,
        args.engine_home,
        args.out_dir,
    )
    if any(value is None for value in required):
        parser.error(
            "all candidate, TRAIN handoff, client plan and output bindings are required"
        )
    if not args.preflight_only and not args.execute_named_clients:
        parser.error(
            "pass --preflight-only or explicitly authorize --execute-named-clients"
        )
    try:
        result = run_capture(args)
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if result["status"] == "PREFLIGHT_READY_NO_CLIENTS_LAUNCHED"
            or result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
            else 2
        )
    except (CaptureError, OSError, subprocess.SubprocessError) as error:
        print(f"three-client capture refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
