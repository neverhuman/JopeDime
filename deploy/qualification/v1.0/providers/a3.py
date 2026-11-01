#!/usr/bin/env python3
"""A3: exercise the pinned CPU release candidate's POSIX local-security boundary."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
CHECKS = (
    "daemon_owner_only",
    "authenticated_health",
    "missing_token",
    "wrong_token",
    "foreign_host",
    "cross_origin",
)


@dataclass(frozen=True)
class Binding:
    artifact: Path
    digest: str
    source: str
    result: Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding() -> Binding:
    if os.environ.get("JOPE_REL07_GATE_ID") != "A3":
        raise ValueError("this provider requires gate A3")
    sha = os.environ.get("JOPE_REL07_ARTIFACT_SHA256", "")
    source = os.environ.get("JOPE_REL07_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not re.fullmatch(
        r"[0-9a-f]{40}", source
    ):
        raise ValueError("full lowercase artifact and source hashes are required")
    artifact = Path(os.environ.get("JOPE_REL07_ARTIFACT", ""))
    result = Path(os.environ.get("JOPE_REL07_RESULT", ""))
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise ValueError("candidate must be an absolute regular non-symlink file")
    if (
        not result.is_absolute()
        or result.exists()
        or result.is_symlink()
        or not result.parent.is_dir()
    ):
        raise ValueError(
            "result must be a new absolute path in the gate evidence directory"
        )
    if digest(artifact) != sha:
        raise ValueError("candidate hash does not match")
    return Binding(artifact, sha, source, result)


def candidate(ctx: Binding) -> None:
    """Independent full-byte verifier precedes executing the candidate."""
    if digest(ctx.artifact) != ctx.digest:
        raise ValueError("candidate changed before verification")
    verified = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "-p",
            "jope-dist",
            "--",
            "verify",
            "--file",
            str(ctx.artifact),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    verification = json.loads(verified.stdout) if not verified.returncode else {}
    if (
        not isinstance(verification, dict)
        or verification.get("schema") != "jope.appliance-verify/v1"
        or verification.get("ok") is not True
    ):
        raise ValueError("independent full-byte package verification failed")
    completed = subprocess.run(
        [str(ctx.artifact), "--version", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode:
        raise ValueError("candidate version command failed")
    info = json.loads(completed.stdout)
    if not isinstance(info, dict):
        raise ValueError("candidate version is not an object")
    package, features = info.get("package"), info.get("features", {})
    if (
        not isinstance(package, dict)
        or not isinstance(features, dict)
        or info.get("source_sha40") != ctx.source
        or package.get("source_sha40") != ctx.source
        or Path(str(package.get("profile", ""))).stem not in {"full", "fused-cpu"}
        or package.get("scope") not in {"public", "internal", "internal-restricted"}
        or package.get("signer_role") != "release"
        or not package.get("methods")
        or features.get("embedded") is not True
        or features.get("dev_keys") is not False
        or info.get("trusts_development_key") is not False
        or features.get("cuda") is not False
        or features.get("dope_cuda") is not False
    ):
        raise ValueError(
            "A3 requires a source-matching, production-trust full CPU candidate"
        )
    if digest(ctx.artifact) != ctx.digest:
        raise ValueError("candidate changed during verification")


def probe(executable: Path, scratch: Path) -> list[dict]:
    """Probe only our newly started daemon; never retain its bearer token."""
    if os.name != "posix":
        raise ValueError("this provider requires native POSIX owner/permission checks")
    rows = []
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a3-", dir=scratch) as name:
        home = Path(name)
        home.chmod(0o700)
        env = os.environ.copy()
        env.update(
            JOPEDIME_HOME=str(home),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            RAYON_NUM_THREADS="1",
        )
        for key in (
            "JOPEDIME_ENDPOINT",
            "JOPE_STORE_FAILPOINT",
            "JOPE_STORE_FAILPOINT_OCCURRENCE",
            "JOPE_STORE_FAILPOINT_PARK",
            "JOPE_STORE_TRACE_BOUNDARIES",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        ):
            env.pop(key, None)
        # Inherit the coordinator's process group so its timeout also stops this child.
        child = subprocess.Popen(
            [str(executable), "serve", "--idle-exit", "60s"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            path = home / "daemon.json"
            deadline = time.monotonic() + 20
            while not path.exists():
                if child.poll() is not None:
                    raise ValueError("owned daemon exited before publishing its record")
                if time.monotonic() >= deadline:
                    raise ValueError("owned daemon record deadline exceeded")
                time.sleep(0.02)
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("owned daemon record is not a regular file")
            record = json.loads(path.read_text())
            url = urlsplit(record["url"])
            token = record["token"]
            if (
                record["pid"] != child.pid
                or url.scheme != "http"
                or url.hostname != "127.0.0.1"
                or not url.port
                or url.username
                or url.password
                or url.path not in {"", "/"}
                or not re.fullmatch(r"[0-9a-fA-F]{64}", token)
            ):
                raise ValueError(
                    "daemon record is not bound to the owned local process"
                )
            rows.append(
                dict(
                    check="daemon_owner_only",
                    passed=st.st_uid == os.getuid()
                    and stat.S_IMODE(st.st_mode) & 0o077 == 0,
                )
            )
            authority = f"127.0.0.1:{url.port}"
            cases = [
                (
                    "authenticated_health",
                    {"Authorization": "Bearer " + token},
                    200,
                    None,
                ),
                ("missing_token", {}, 401, "UNAUTHORIZED"),
                (
                    "wrong_token",
                    {
                        "Authorization": "Bearer "
                        + ("0" if token[0] != "0" else "1")
                        + token[1:]
                    },
                    401,
                    "UNAUTHORIZED",
                ),
                (
                    "foreign_host",
                    {"Authorization": "Bearer " + token, "Host": "attacker.invalid"},
                    400,
                    "INVALID_INPUT",
                ),
                (
                    "cross_origin",
                    {
                        "Authorization": "Bearer " + token,
                        "Origin": "https://attacker.invalid",
                    },
                    400,
                    "INVALID_INPUT",
                ),
            ]
            for name, headers, expected, code in cases:
                connection = http.client.HTTPConnection(
                    "127.0.0.1", url.port, timeout=2
                )
                try:
                    connection.request(
                        "GET", "/v1/system", headers={"Host": authority, **headers}
                    )
                    response = connection.getresponse()
                    body = response.read(1024 * 1024 + 1)
                    value = json.loads(body) if len(body) <= 1024 * 1024 else None
                    cors = any(
                        k.lower().startswith("access-control-")
                        for k, _ in response.getheaders()
                    )
                    passed = (
                        response.status == expected
                        and isinstance(value, dict)
                        and not cors
                    )
                    if code is not None:
                        passed = (
                            passed
                            and value.get("code") == code
                            and response.getheader("Content-Type", "").split(";", 1)[0]
                            == "application/problem+json"
                        )
                    rows.append(
                        dict(check=name, passed=passed, http_status=response.status)
                    )
                except (OSError, ValueError, http.client.HTTPException):
                    rows.append(
                        dict(
                            check=name,
                            passed=False,
                            diagnostic="request or response validation failed",
                        )
                    )
                finally:
                    connection.close()
        finally:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    return rows


def publish(ctx: Binding, rows: list[dict]) -> int:
    if [r["check"] for r in rows] != list(CHECKS) or any(
        type(r["passed"]) is not bool for r in rows
    ):
        raise ValueError("A3 requires all six measured checks")
    if digest(ctx.artifact) != ctx.digest:
        raise ValueError("candidate changed during A3 assertions")
    passed = sum(r["passed"] for r in rows)
    payload = dict(
        schema="jope.rel07.gate-execution/v1",
        id="A3",
        source_sha=ctx.source,
        artifact_sha256=ctx.digest,
        status="PASS" if passed == len(CHECKS) else "FAIL",
        counts=dict(
            collected=6,
            executed=6,
            passed=passed,
            failed=6 - passed,
            skipped=0,
            excluded=0,
        ),
        checks=rows,
    )
    fd = os.open(
        ctx.result,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload))
    return int(passed != len(CHECKS))


def main() -> int:
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        ctx = binding()
        candidate(ctx)
        return publish(ctx, probe(ctx.artifact, ctx.result.parent / "scratch"))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print(
            "A3 refused invalid candidate/input or failed owned-daemon setup; no PASS emitted",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
