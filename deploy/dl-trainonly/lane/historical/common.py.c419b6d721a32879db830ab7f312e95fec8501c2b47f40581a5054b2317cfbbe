"""Shared helpers for the dltrain research lane (PLAN §5 P0 step 3). Standard library only.

The patterns are copied here, not imported:
- canonical JSON;
- atomic and create-only (immutable) writes;
- streaming sha256;
- measured UTC microseconds;
- a receipt writer that refuses anything that looks like secret material.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"(?i)secret[_-]?access[_-]?key"),
    re.compile(r"(?i)session[_-]?token"),
    re.compile(r"(?i)security[_-]?token"),
    re.compile(r"(?i)x-amz-(?:signature|credential)"),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),   # access key ids
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{16,}"),                     # bearer tokens
    re.compile(r"(?:IQoJb3JpZ2luX2Vj|FwoGZXIvYXdzE|FQoGZXIvYXdzE)[A-Za-z0-9/+=]{16,}"),   # STS session-token bodies
    re.compile(r"(?i)--(?:[a-z0-9]+-)*(?:token|secret|password|passwd|credentials?)=(?!<redacted>)\S+"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),                                 # URL userinfo
)
SECRET_FLAG_RE = re.compile(r"(?i)--(?:[a-z0-9]+-)*(?:token|secret|password|passwd|credentials?|key)")
REDACT = "<redacted>"


class ImmutableConflict(RuntimeError):
    """A create-only path already holds different bytes."""


class SecretRefused(RuntimeError):
    """A receipt or log entry looked like it contained secret material."""


def canonical(obj) -> bytes:
    """Compact, sorted-key JSON as UTF-8 bytes; NaN and Infinity are refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_us() -> int:
    """Measured UTC time in microseconds; never an estimate."""
    return time.time_ns() // 1000


def utc_iso(us: int | None = None) -> str:
    us = utc_us() if us is None else us
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(us // 1_000_000)) + f".{us % 1_000_000:06d}Z"


def _fsync_dir(directory) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, data: bytes, mode: int = 0o664) -> dict:
    """Same-directory temp file, fsync, os.replace, directory fsync. Never leaves an empty target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)
    return {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)}


def immutable_write(path, data: bytes, mode: int = 0o444) -> dict:
    """Create-only write (O_EXCL). An existing file with equal bytes is success; different bytes raise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != data:
            raise ImmutableConflict(f"{path} exists with different bytes ({sha256_bytes(existing)} != {sha256_bytes(data)})")
        return {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "existed": True}
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(path, mode)
    _fsync_dir(path.parent)
    return {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "existed": False}


def assert_no_secrets(text: str) -> None:
    """Scan raw text (never JSON-escaped text, where an escaped newline would hide a key id)."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise SecretRefused(f"refused: text matches secret pattern {pattern.pattern!r}")


def assert_no_secrets_obj(obj) -> None:
    """Walk every key and string of a document; a secret-named flag in a list must be followed by the redaction marker."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_secrets(str(k))
            assert_no_secrets_obj(v)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, str) and SECRET_FLAG_RE.fullmatch(v) and (i + 1 >= len(obj) or obj[i + 1] != REDACT):
                raise SecretRefused(f"refused: flag {v} is not followed by {REDACT}")
            assert_no_secrets_obj(v)
    elif isinstance(obj, str):
        assert_no_secrets(obj)


def redact_argv(argv) -> list[str]:
    out, hide = [], False
    for a in argv:
        if hide:
            out.append(REDACT)
            hide = False
            continue
        name = a.split("=", 1)[0]
        if SECRET_FLAG_RE.fullmatch(name):
            if "=" in a:
                out.append(name + "=" + REDACT)
            else:
                out.append(a)
                hide = True
            continue
        out.append(a)
    return out


def receipt_bytes(doc: dict) -> bytes:
    assert_no_secrets_obj(doc)
    body = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    return body.encode("utf-8")


def write_receipt(path, doc: dict, immutable: bool = True) -> dict:
    """Receipts are create-only by default; they never carry the environment or key material."""
    data = receipt_bytes(doc)
    return immutable_write(path, data) if immutable else atomic_write(path, data)


def copy_verified(src, dst_dir, expected_sha256: str) -> dict:
    """Copy a binary to dst_dir/<sha256> only if its sha256 matches; re-verify the copy."""
    got = sha256_file(src)
    if got != expected_sha256:
        raise RuntimeError(f"sha256 mismatch for {src}: {got} != {expected_sha256}")
    dst = Path(dst_dir) / expected_sha256
    if dst.exists():
        if sha256_file(dst) != expected_sha256:
            raise ImmutableConflict(f"{dst} exists with a different sha256")
        return {"src": str(src), "dst": str(dst), "sha256": expected_sha256, "bytes": dst.stat().st_size, "existed": True}
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + expected_sha256[:12] + ".", dir=str(dst_dir))
    os.close(fd)
    shutil.copyfile(src, tmp)
    if sha256_file(tmp) != expected_sha256:
        os.unlink(tmp)
        raise RuntimeError("copy changed bytes")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    os.replace(tmp, dst)
    _fsync_dir(dst_dir)
    return {"src": str(src), "dst": str(dst), "sha256": expected_sha256, "bytes": dst.stat().st_size, "existed": False}


def runtime_identity(venv) -> dict:
    """sha256 of every file under the venv's site-packages, plus an aggregate digest (PLAN §2)."""
    venv = Path(venv)
    roots = sorted(venv.glob("lib/python*/site-packages"))
    files, agg = [], hashlib.sha256()
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts:
                digest = sha256_file(p)
                rel = str(p.relative_to(venv))
                files.append([rel, digest, p.stat().st_size])
                agg.update(f"{rel}\0{digest}\n".encode())
    return {"venv": str(venv), "python": sys.version, "site_packages": [str(r) for r in roots],
            "files": len(files), "bytes": sum(f[2] for f in files), "aggregate_sha256": agg.hexdigest(), "file_hashes": files}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.common")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rid = sub.add_parser("runtime-identity")
    rid.add_argument("--venv", required=True)
    rid.add_argument("--freeze", required=True, help="path to `uv pip freeze` output")
    rid.add_argument("--out", required=True)
    cp = sub.add_parser("copy-binary")
    cp.add_argument("--src", required=True)
    cp.add_argument("--dst-dir", required=True)
    cp.add_argument("--sha256", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "runtime-identity":
        doc = runtime_identity(args.venv)
        doc.update({"schema": "jope.dl-runtime-identity/v1", "utc_us": utc_us(),
                    "freeze": Path(args.freeze).read_text().splitlines(), "freeze_sha256": sha256_file(args.freeze)})
        res = write_receipt(args.out, doc)
        print(json.dumps({k: res[k] for k in ("path", "sha256", "bytes")} | {"files": doc["files"], "aggregate_sha256": doc["aggregate_sha256"]}))
    elif args.cmd == "copy-binary":
        print(json.dumps(copy_verified(args.src, args.dst_dir, args.sha256)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
