"""Pinned client-plan validation and command rendering helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

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
# v14's reviewed recorder contract uses a 390-second ceiling. Keep the
# consumer aligned with the producer's declared limit; byte/frame caps stay fixed.
MAX_TRANSPORT_SECONDS = 390.0
MAX_TRANSPORT_STREAM_BYTES = 16_777_216
MAX_TRANSPORT_FRAME_BYTES = 1_048_576
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



def transport_limits_within_reviewed_bounds(limits: object) -> bool:
    if not isinstance(limits, dict):
        return False
    seconds = limits.get("seconds")
    stream_bytes = limits.get("stream_bytes")
    frame_bytes = limits.get("frame_bytes")
    return (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and seconds <= MAX_TRANSPORT_SECONDS
        and isinstance(stream_bytes, int)
        and not isinstance(stream_bytes, bool)
        and stream_bytes <= MAX_TRANSPORT_STREAM_BYTES
        and isinstance(frame_bytes, int)
        and not isinstance(frame_bytes, bool)
        and frame_bytes == MAX_TRANSPORT_FRAME_BYTES
    )



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



def client_selection(requested: list[str] | None, exclusion_reason: str | None):
    selected = list(CLIENTS) if requested is None else requested
    require(bool(selected), "at least one named client must be selected")
    require(
        all(isinstance(key, str) and key in CLIENTS for key in selected),
        "selected clients must be named in the pinned client plan",
    )
    require(len(selected) == len(set(selected)), "selected clients must be unique")
    selected_set = set(selected)
    selected = [key for key in CLIENTS if key in selected_set]
    excluded = [key for key in CLIENTS if key not in selected_set]
    if excluded:
        require(
            isinstance(exclusion_reason, str) and exclusion_reason.strip(),
            "--exclude-reason is required when clients are omitted",
        )
    else:
        require(
            exclusion_reason is None,
            "--exclude-reason is only valid when a client is omitted",
        )
    return selected, {key: exclusion_reason.strip() for key in excluded}



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

