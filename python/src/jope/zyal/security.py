"""Sandbox security: static AST rejection, then a networkless read-only container.

Modelled on nomad's `security.py`, with one addition it did not need: **split containment**.

Analysis mode lets an agent run real code against real labels, which is a genuine leakage surface.
The guard is not a policy check — it is the mount table. Only the SELECT shard is bind-mounted into
an analysis container. DEV and TEST are absent from the filesystem the code can see, so a program
that tries to read them fails with ENOENT rather than being trusted not to try.

Stage 1 (this module, free) rejects most hostile or careless code before Docker is ever started.
Stage 2 is the container. Neither alone is sufficient: the AST check cannot stop a runtime trick,
and the container alone would happily run code that quietly reads a path it should not.
"""

from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_IMPORTS = frozenset(
    {
        "collections",
        "functools",
        "itertools",
        "math",
        "numpy",
        "statistics",
        "heapq",
        "bisect",
    }
)

# Names that grant dynamic execution, filesystem reach, or introspection escape.
FORBIDDEN_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
        "__import__",
        "memoryview",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "builtins",
        "ctypes",
        "ftplib",
        "http",
        "httpx",
        "importlib",
        "io",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
    }
)

MAX_PLUGIN_BYTES = 20_480
IMAGE_LOCK = Path("ops/zyal/image-lock.json")


class SandboxRejection(RuntimeError):
    """Static rejection. The proposal never reaches a container."""


def static_check_plugin(source: str, entrypoint: str = "compute_family") -> None:
    """Reject a proposed plugin on structure alone. Raises SandboxRejection with a reason.

    Deliberately strict: an agent that needs `os` for a feature over genome text has misunderstood
    the task, and letting it through to burn a container slot teaches it nothing useful.
    """
    if len(source.encode()) > MAX_PLUGIN_BYTES:
        raise SandboxRejection(f"plugin source exceeds {MAX_PLUGIN_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxRejection(f"plugin does not parse: {exc}") from exc

    found_entrypoint = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise SandboxRejection(f"import {alias.name!r} outside allowlist")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise SandboxRejection(f"from-import {node.module!r} outside allowlist")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise SandboxRejection(f"forbidden call {node.func.id!r}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxRejection(f"forbidden name {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # blocks the __class__/__subclasses__/__globals__ sandbox-escape family
            raise SandboxRejection(f"dunder attribute access {node.attr!r}")
        elif isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            found_entrypoint = True

    if not found_entrypoint:
        raise SandboxRejection(f"plugin must define {entrypoint}(...)")


def sandbox_command(
    *,
    plugin: Path,
    harness: Path,
    data_shard: Path,
    output: Path,
    image_digest: str,
    mode: str = "feature",
    cpus: int = 4,
    memory: str = "8g",
    pids: int = 256,
) -> list[str]:
    """Build the docker invocation.

    `data_shard` MUST be the SELECT shard for analysis mode. DEV and TEST are never passed, so
    they are not merely forbidden inside the container — they do not exist there.
    """
    if mode not in {"feature", "analysis"}:
        raise ValueError(f"unknown sandbox mode {mode!r}")
    shard = data_shard.resolve()
    if mode == "analysis" and shard.name != "select":
        raise SandboxRejection(
            f"analysis mode may only mount the SELECT shard, got {shard.name!r} -- "
            "DEV and TEST must be unreachable, not merely disallowed"
        )
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65532:65532",
        "--pids-limit",
        str(pids),
        "--memory",
        memory,
        "--cpus",
        str(cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=512m",
        "--mount",
        f"type=bind,src={plugin.resolve()},dst=/work/plugin.py,readonly",
        "--mount",
        f"type=bind,src={harness.resolve()},dst=/opt/zyal/runner.py,readonly",
        "--mount",
        f"type=bind,src={shard},dst=/inputs/shard,readonly",
        "--mount",
        f"type=bind,src={output.resolve()},dst=/output",
        "--env",
        f"ZYAL_MODE={mode}",
        image_digest,
        "python",
        "/opt/zyal/runner.py",
        "--mode",
        mode,
    ]
