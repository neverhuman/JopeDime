"""Content-addressed, reproducible rustc cache for one emitted Rust source.

The historical scorer compiles inside a per-evaluation directory:

    rustc --edition=2021 -O <dest>/emitted/model.rs -o <dest>/emitted/model

That argv is `historical_argv`. A different `dest` embeds a different path, so
the same source becomes a different executable. This module compiles with the
path remapped to a fixed prefix and stores one executable per
(source, toolchain, flags) key. A hit re-hashes the file before it is returned
or executed. A miss never returns a stale binary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROTOCOL = "jope.repro-rust-executable/v1"
REMAP_PREFIX = "/src"
# rustc -O is opt-level 2. Kept explicit so the flag list is the cache identity.
REPRO_ARGS = (
    "--edition=2021",
    "-C",
    "opt-level=2",
    "-C",
    "codegen-units=1",
    "-C",
    "debuginfo=0",
    "-C",
    "strip=symbols",
    "-C",
    "link-arg=-Wl,--build-id=none",
)


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def historical_argv(rustc: str, source: Path, output: Path) -> list[str]:
    """The argv at hpc_verified_v5/native_evaluate.py compile step."""
    return [rustc, "--edition=2021", "-O", str(source), "-o", str(output)]


def toolchain_id(rustc: str) -> str:
    proc = subprocess.run([rustc, "-vV"], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def repro_argv(rustc: str, source: Path, output: Path, workdir: Path) -> list[str]:
    return [
        rustc,
        *REPRO_ARGS,
        f"--remap-path-prefix={workdir}={REMAP_PREFIX}",
        str(source),
        "-o",
        str(output),
    ]


def cache_key(source: bytes, toolchain: str, flags: tuple[str, ...] = REPRO_ARGS) -> str:
    payload = b"\0".join(
        [
            PROTOCOL.encode(),
            source,
            toolchain.encode(),
            "\0".join(flags).encode(),
            REMAP_PREFIX.encode(),
        ]
    )
    return sha256_bytes(payload)


@dataclass(frozen=True)
class CachedExecutable:
    key: str
    path: Path
    sha256: str
    hit: bool
    source_sha256: str


class ExecutableCache:
    """On-disk cache. `ensure` compiles at most once per key."""

    def __init__(self, root: Path, rustc: str = "rustc", toolchain: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rustc = rustc
        self.toolchain = toolchain if toolchain is not None else toolchain_id(rustc)
        self.compiles = 0
        self.hits = 0

    def _dir(self, key: str) -> Path:
        return self.root / key

    def _load(self, key: str, source_sha: str) -> CachedExecutable | None:
        directory = self._dir(key)
        binary = directory / "model"
        side = directory / "cache.json"
        if not binary.is_file() or not side.is_file():
            return None
        meta = json.loads(side.read_text())
        if meta.get("key") != key or meta.get("source_sha256") != source_sha:
            raise ValueError("cache sidecar does not match the requested source")
        body = binary.read_bytes()
        digest = sha256_bytes(body)
        if digest != meta.get("sha256"):
            raise ValueError("cached executable failed SHA-256 verification")
        self.hits += 1
        return CachedExecutable(key, binary, digest, True, source_sha)

    def ensure(self, source: bytes, timeout: float = 120.0) -> CachedExecutable:
        if not isinstance(source, bytes) or not source:
            raise ValueError("source must be non-empty bytes")
        source_sha = sha256_bytes(source)
        key = cache_key(source, self.toolchain)
        loaded = self._load(key, source_sha)
        if loaded is not None:
            return loaded
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        src_path = directory / "model.rs"
        binary = directory / "model"
        src_path.write_bytes(source)
        env = os.environ.copy()
        env["SOURCE_DATE_EPOCH"] = "0"
        proc = subprocess.run(
            repro_argv(self.rustc, src_path, binary, directory),
            cwd=directory,
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0 or not binary.is_file():
            detail = proc.stderr.decode("utf-8", "replace")[-400:]
            raise RuntimeError(f"rustc failed ({proc.returncode}): {detail}")
        body = binary.read_bytes()
        digest = sha256_bytes(body)
        side = {
            "schema": PROTOCOL,
            "key": key,
            "source_sha256": source_sha,
            "sha256": digest,
            "bytes": len(body),
            "toolchain_sha256": sha256_bytes(self.toolchain.encode()),
            "flags": list(REPRO_ARGS),
        }
        (directory / "cache.json").write_text(json.dumps(side, sort_keys=True))
        # Verify the bytes just written before anyone executes them.
        again = self._load(key, source_sha)
        if again is None or again.sha256 != digest:
            raise ValueError("cache write did not verify")
        self.compiles += 1
        self.hits -= 1  # _load counted the verifying read as a hit
        return CachedExecutable(key, binary, digest, False, source_sha)

    def execute(
        self,
        source: bytes,
        arguments: list[str],
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[bytes]:
        cached = self.ensure(source)
        # Re-read immediately before exec so a swapped file cannot run.
        if sha256_bytes(cached.path.read_bytes()) != cached.sha256:
            raise ValueError("executable changed after cache verification")
        return subprocess.run(
            [str(cached.path), *arguments],
            capture_output=True,
            timeout=timeout,
            check=False,
        )


def compile_once(rustc: str, argv: list[str], cwd: Path, timeout: float = 120.0) -> bytes:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "0"
    proc = subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"rustc failed ({proc.returncode}): {detail}")
    output = Path(argv[-1])
    if not output.is_file():
        raise RuntimeError("rustc reported success without an output file")
    return output.read_bytes()


def run_batch(
    cache: ExecutableCache,
    jobs: list[tuple[bytes, list[str]]],
    timeout: float = 60.0,
) -> list[subprocess.CompletedProcess[bytes]]:
    """Compile each distinct source once, then execute every job in order."""
    return [cache.execute(source, args, timeout=timeout) for source, args in jobs]
