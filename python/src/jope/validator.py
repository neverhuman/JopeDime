from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .prime_config import ssh_control_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_MANIFEST = PROJECT_ROOT / "rust" / "jope-validate" / "Cargo.toml"
VALIDATOR_RELEASE_BINARY = PROJECT_ROOT / "rust" / "jope-validate" / "target" / "release" / "jope-validate"


@dataclass
class RustValidator:
    manifest: Path = VALIDATOR_MANIFEST
    binary: Path | None = None
    engine: str = "featmath"
    binary_env: str = "JOPE_VALIDATE_BIN"
    release_binary: Path | None = VALIDATOR_RELEASE_BINARY
    # Default per-row wall-clock timeout for eval-regression. None = block forever
    # (legacy behaviour). Callers can pass timeout_for per row instead.
    default_timeout_s: float | None = None
    # Run the bridge on a remote host over ssh (paths refer to the REMOTE filesystem;
    # the cluster mirrors the configured project root). "local"/None = this machine.
    ssh_host: str | None = None

    def validate_many(self, genomes: Iterable[str | dict]) -> list[dict]:
        rows = []
        for item in genomes:
            rows.append(item if isinstance(item, dict) else {"genome": item})
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        cmd = self._command()
        proc = subprocess.run(
            cmd,
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"validator failed: {proc.stderr.strip()}")
        out = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        for row in out:
            row.setdefault("engine", self.engine)
        return out

    def validate_one(self, genome: str) -> dict:
        return self.validate_many([genome])[0]

    def emit_many(self, genomes: Iterable[str | dict]) -> list[dict]:
        """Transpile genomes to standalone Python+Rust source via the bridge `emit` cmd.

        Rows that the bridge can't emit come back with ``supported: False`` (+ an
        ``unsupported`` op list or ``errors``); the caller decides whether to fall back to
        a preview. If the bridge predates the `emit` subcommand it exits non-zero — surface
        that as a single ``supported: False`` row per input rather than raising, so callers
        degrade gracefully.
        """
        rows = [item if isinstance(item, dict) else {"genome": item} for item in genomes]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        proc = subprocess.run(
            self._command("emit"),
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            reason = proc.stderr.strip() or "emit unsupported by bridge"
            return [{"engine": self.engine, "supported": False, "errors": [reason]} for _ in rows]
        out = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        for row in out:
            row.setdefault("engine", self.engine)
        return out

    def emit_one(self, genome: str) -> dict:
        return self.emit_many([genome])[0]

    def grammar(self) -> dict:
        proc = subprocess.run(
            self._command("grammar"),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"grammar handshake failed: {proc.stderr.strip()}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def eval_regression_many(
        self,
        rows: Iterable[dict],
        *,
        timeout_for: Callable[[dict], float | None] | None = None,
    ) -> list[dict]:
        return list(self.iter_eval_regression_many(rows, timeout_for=timeout_for))

    def iter_eval_regression_many(
        self,
        rows: Iterable[dict],
        *,
        timeout_for: Callable[[dict], float | None] | None = None,
    ) -> Iterator[dict]:
        yield from self.iter_eval_many(rows, subcommand="eval-regression", timeout_for=timeout_for)

    def iter_eval_many(
        self,
        rows: Iterable[dict],
        *,
        subcommand: str = "eval-regression",
        timeout_for: Callable[[dict], float | None] | None = None,
    ) -> Iterator[dict]:
        """Lock-step streaming eval with optional per-row wall-clock timeouts.

        ``subcommand`` selects the bridge mode (``eval-regression`` | ``eval-binary``).
        On timeout the bridge process is killed, a synthetic ``timeout: true`` row is
        yielded (without a ``cache_key`` so it can never enter the ledger), the process
        is respawned WITH THE SAME SUBCOMMAND, and evaluation continues. A mid-stream
        crash (EOF) is handled the same way with ``bridge_crash: true``.
        """
        rows = list(rows)
        if not rows:
            return
        proc: _BridgeProcess | None = None
        try:
            proc = self._spawn(subcommand)
            for row in rows:
                timeout = timeout_for(row) if timeout_for is not None else self.default_timeout_s
                started = time.time()
                if proc is None or not proc.alive():
                    proc = self._spawn(subcommand)
                payload = json.dumps(row) + "\n"
                try:
                    proc.write(payload)
                except (BrokenPipeError, OSError):
                    # lane died between rows (e.g. ssh throttled/dropped): respawn once
                    proc.kill()
                    proc = self._spawn(subcommand)
                    try:
                        proc.write(payload)
                    except (BrokenPipeError, OSError):
                        proc.kill()
                        proc = None
                        yield self._synthetic_row(row, started, "bridge_crash", detail="write failed twice")
                        continue
                line = proc.read_line(timeout)
                if line is _TIMEOUT:
                    proc.kill()
                    proc = None
                    yield self._synthetic_row(row, started, "timeout")
                    continue
                if line is None:
                    err = proc.stderr_text()
                    proc.kill()
                    proc = None
                    yield self._synthetic_row(row, started, "bridge_crash", detail=err)
                    continue
                out = json.loads(line)
                out.setdefault("engine", self.engine)
                yield out
        finally:
            if proc is not None:
                proc.close()

    def _synthetic_row(self, row: dict, started: float, kind: str, detail: str = "") -> dict:
        elapsed = (time.time() - started) * 1000.0
        message = f"eval {kind} after {elapsed / 1000.0:.1f}s"
        if detail:
            message = f"{message}: {detail.strip()[:400]}"
        return {
            "engine": row.get("engine", self.engine),
            "dataset_hash": row.get("dataset_hash") or row.get("dataset_id") or "unknown",
            "canonical_genome": row.get("genome") or row.get("canonical_genome"),
            "valid": False,
            kind: True,
            "cache_hit": False,
            "cv": None,
            "metrics": None,
            "timings": {"total_ms": elapsed},
            "errors": [message],
        }

    def _spawn(self, subcommand: str = "eval-regression") -> _BridgeProcess:
        return _BridgeProcess(self._command(subcommand))

    def _command(self, command: str = "validate") -> list[str]:
        if self.binary is not None:
            cmd = [str(self.binary), command]
        else:
            env_binary = os.environ.get(self.binary_env)
            if env_binary:
                cmd = [env_binary, command]
            elif self.release_binary is not None and Path(self.release_binary).exists():
                cmd = [str(self.release_binary), command]
            else:
                cmd = ["cargo", "run", "-q", "--manifest-path", str(self.manifest), "--", command]
        if self.ssh_host and self.ssh_host != "local":
            # Persistent lock-step over one ssh channel per worker lane. ControlMaster
            # multiplexes all lanes over ONE TCP connection per host (sshd MaxStartups
            # refuses mass simultaneous dials). Killing the local ssh closes the remote
            # stdin -> the bridge exits cleanly.
            return [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=8",
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={ssh_control_path(self.ssh_host)}",
                "-o", "ControlPersist=300",
                self.ssh_host,
                *cmd,
            ]
        return cmd


_TIMEOUT = object()


class _BridgeProcess:
    """A bridge subprocess with a reader thread so reads can be bounded by a timeout."""

    def __init__(self, cmd: list[str]) -> None:
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_chunks: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for chunk in self.proc.stderr:
            self._stderr_chunks.append(chunk)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def write(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def read_line(self, timeout: float | None):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return _TIMEOUT
        return line

    def stderr_text(self) -> str:
        self._stderr_reader.join(timeout=0.5)
        return "".join(self._stderr_chunks)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        self.kill()


def validate_genome(genome: str) -> dict:
    return RustValidator().validate_one(genome)
