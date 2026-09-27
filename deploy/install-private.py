#!/usr/bin/env python3
"""Install a checksum-pinned private Linux CLI, or build only the existing CLI package.

This program never downloads a binary, obtains credentials, installs weights, fits
models, or changes a system service. Supply a reviewed source SHA and an independently
retained binary SHA256. Cold installation/build timing is separate from warm demos.
"""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

MAX_BINARY_BYTES = 512 * 1024 * 1024
FORBIDDEN_LIBRARIES = ("libtorch", "libc10", "libcuda", "libcudart", "libnvrtc", "libcudnn")
FORBIDDEN_PACKAGES = {"jope-parallel", "jope-engine", "tch", "torch-sys", "candle-kernels"}
CLI_FEATURES = ("asm",)
CLI_FEATURE_ARGUMENTS = ("--features", ",".join(CLI_FEATURES))


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pin(value, length, name):
    if not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise ValueError(name + " must be a full lowercase hexadecimal digest")


def private_directory(path):
    path = Path(path)
    if path.resolve().is_relative_to("/tmp"):
        raise ValueError("private artifacts require a durable directory outside /tmp")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("installation directory must be a private, owned physical directory: " + str(path))
    return path


def write_private(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def child_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("AWS_", "GH_", "GITHUB_")):
            env.pop(key)
    env["JOPE_OFFLINE"] = "1"
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    return env


def inspect_binary(path):
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("this private binary installer qualifies Linux x86_64 only")
    with Path(path).open("rb") as stream:
        header = stream.read(64)
    if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 62:
        raise ValueError("binary is not Linux x86_64 little-endian ELF64")
    if struct.unpack_from("<H", header, 16)[0] not in (2, 3):
        raise ValueError("ELF is not an executable or position-independent executable")
    dynamic = subprocess.check_output(["readelf", "--dynamic", str(path)], text=True, stderr=subprocess.STDOUT, timeout=10)
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic)
    if any(marker in name.lower() for name in needed for marker in FORBIDDEN_LIBRARIES):
        raise ValueError("CLI binary unexpectedly requires Torch/CUDA libraries")
    versions = subprocess.check_output(["readelf", "--version-info", str(path)], text=True, stderr=subprocess.STDOUT, timeout=10)
    requirements = {tuple(map(int, match.split("."))) for match in re.findall(r"GLIBC_([0-9]+\.[0-9]+(?:\.[0-9]+)?)", versions)}
    required = max(requirements) if requirements else None
    libc, current = platform.libc_ver()
    if required is not None and (libc != "glibc" or tuple(map(int, current.split("."))) < required):
        raise ValueError("host glibc is older than the binary's required symbols")
    return {"machine": "x86_64", "format": "ELF64", "needed_libraries": needed,
            "required_glibc": ".".join(map(str, required)) if required else None,
            "host_libc": {"name": libc, "version": current}, "torch_cuda_libraries": False}


def probe_binary(path):
    env = child_environment()
    version = subprocess.check_output([str(path), "--version"], text=True, env=env, timeout=10).strip()
    doctor = json.loads(subprocess.check_output([str(path), "doctor"], text=True, env=env, timeout=10))
    if doctor.get("schema") != "jope.doctor/v1":
        raise ValueError("binary did not return the existing doctor contract")
    capability = json.loads(subprocess.check_output(
        [str(path), "optimize", "capabilities"], text=True, env=env, timeout=10))
    if (capability.get("schema") != "jope.optimize-capabilities/v1"
            or capability.get("compiled") is not True
            or capability.get("fits") != 0 or capability.get("qualified") is not False):
        raise ValueError("private CLI lacks the required compiled ASM capability contract")
    commands = {}
    for command in ("benchmark", "demo", "optimize", "train"):
        help_text = subprocess.check_output([str(path), command, "--help"], text=True,
                                            env=env, timeout=10)
        commands[command] = hashlib.sha256(help_text.encode()).hexdigest()
    return {"version": version, "doctor": doctor, "asm_capabilities": capability,
            "required_feature_policy": {"schema": "jope.private-cli-features/v1",
                                        "cargo_features": list(CLI_FEATURES)},
            "command_help_sha256": commands}


def copy_pinned_binary(source, destination, expected_sha256, max_bytes=MAX_BINARY_BYTES):
    digest = hashlib.sha256()
    total = 0
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output, Path(source).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > max_bytes:
                    raise ValueError("binary exceeds 512 MiB limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("binary SHA256 does not match the separately supplied pin")
        os.chmod(destination, 0o700)
        return total
    except BaseException:
        # This invocation owns the newly created partial; the source is untouched.
        Path(destination).unlink(missing_ok=True)
        raise


def install_binary(source, expected_sha256, source_sha, prefix):
    started = time.monotonic()
    started_unix_ns = time.time_ns()
    validate_pin(expected_sha256, 64, "binary SHA256")
    validate_pin(source_sha, 40, "source SHA")
    source = Path(source).resolve(strict=True)
    if not source.is_file() or source.stat().st_size > MAX_BINARY_BYTES:
        raise ValueError("source is not a bounded regular binary")
    # Resolve the requested prefix once, then require private physical owned children.
    prefix = private_directory(Path(prefix).absolute())
    if prefix.resolve() != prefix:
        raise ValueError("installation prefix must be a physical absolute path")
    lock_fd = os.open(prefix / ".install.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(lock_fd, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        versions = private_directory(prefix / "versions")
        version_dir = private_directory(versions / expected_sha256)
        binaries = private_directory(prefix / "bin")
        receipts = private_directory(prefix / "receipts")
        destination = version_dir / "jope"
        metadata_path = version_dir / "artifact.json"
        current = binaries / "jope"
        prior = None
        if os.path.lexists(current):
            if not current.is_symlink():
                raise ValueError("refuse to replace an unmanaged executable")
            prior = os.readlink(current)
            target = current.resolve(strict=True)
            if target.parent.parent != versions or target.name != "jope":
                raise ValueError("refuse to replace an unmanaged executable link")
        already_present = os.path.lexists(destination)
        if already_present:
            if destination.is_symlink() or checksum(destination) != expected_sha256:
                raise ValueError("installed immutable binary changed")
            metadata = json.loads(metadata_path.read_text())
            if metadata["source_sha"] != source_sha or metadata["binary_sha256"] != expected_sha256:
                raise ValueError("installed binary provenance differs from supplied pins")
            # Revalidate the current feature contract even for a prior immutable version.
            # A checksum-preserving old CLI may predate the required ASM front door.
            probe = probe_binary(destination)
            if checksum(destination) != expected_sha256:
                raise ValueError("installed binary changed during verification")
        else:
            staging = version_dir / (".binary-" + uuid.uuid4().hex + ".partial")
            size = copy_pinned_binary(source, staging, expected_sha256)
            try:
                inspection = inspect_binary(staging)
                probe = probe_binary(staging)
                # Rehash after execution and before activation; immutable destination is new.
                if checksum(staging) != expected_sha256:
                    raise ValueError("staged binary changed during verification")
                os.rename(staging, destination)
                metadata = {"schema": "jope.private-cli-artifact/v1", "source_sha": source_sha,
                            "binary_sha256": expected_sha256, "bytes": size, "inspection": inspection,
                            "probe": probe, "source_authenticity": "caller-pinned; checksum is not a signature"}
                write_private(metadata_path, metadata)
            finally:
                staging.unlink(missing_ok=True)
        # Old immutable versions and receipts remain available when activation changes.
        link = binaries / (".activate-" + uuid.uuid4().hex)
        relative = "../versions/" + expected_sha256 + "/jope"
        os.symlink(relative, link)
        os.replace(link, current)
        directory_fd = os.open(binaries, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        receipt = {"schema": "jope.private-cli-install/v1", "ok": True,
                   "source_sha": source_sha, "binary_sha256": expected_sha256,
                   "binary": str(destination), "command": str(current), "prior_activation": prior,
                   "already_present": already_present, "started_unix_ns": started_unix_ns,
                   "completed_unix_ns": time.time_ns(), "cold_install_seconds": time.monotonic() - started,
                   "models_downloaded": 0, "fits": 0, "warm_demo_measured": False,
                   "artifact_metadata": str(metadata_path), "probe": probe}
        receipt_path = receipts / (uuid.uuid4().hex + ".json")
        write_private(receipt_path, receipt)
        receipt["receipt"] = str(receipt_path)
        return receipt


def validate_cli_graph(text):
    packages = {line.split()[0] for line in text.splitlines() if line.strip()}
    forbidden = {name for name in packages if name in FORBIDDEN_PACKAGES
                 or "cuda" in name.lower() or "cudnn" in name.lower()}
    if forbidden:
        raise ValueError("CLI-only build unexpectedly includes: " + ", ".join(sorted(forbidden)))
    if "jope-cli" not in packages:
        raise ValueError("dependency inventory did not contain jope-cli")
    return sorted(packages)


def build_source(checkout, source_sha, output, jobs=4):
    started = time.monotonic()
    started_unix_ns = time.time_ns()
    validate_pin(source_sha, 40, "source SHA")
    if not 1 <= jobs <= 20:
        raise ValueError("build jobs must be between 1 and 20")
    checkout = Path(checkout).resolve(strict=True)
    env = child_environment()
    def read(*args):
        return subprocess.check_output(args, cwd=checkout, env=env, text=True).strip()
    if Path(read("git", "rev-parse", "--show-toplevel")) != checkout:
        raise ValueError("source must be the existing repository root")
    if read("git", "rev-parse", "HEAD") != source_sha or read("git", "status", "--porcelain"):
        raise ValueError("source checkout must be clean at the separately supplied full SHA")
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("build receipt directory already exists")
    output = private_directory(output)
    graph_command = ["cargo", "tree", "--locked", "-p", "jope-cli", *CLI_FEATURE_ARGUMENTS,
                     "--edges", "normal,build", "--prefix", "none", "--format", "{p}"]
    graph = read(*graph_command)
    packages = validate_cli_graph(graph)
    inventory_path = output / "dependency-inventory.json"
    write_private(inventory_path, {"schema":"jope.private-cli-dependencies/v1",
                                  "command":graph_command,"cargo_features":list(CLI_FEATURES),
                                  "graph":graph,"packages":packages})
    metadata_path = output / "metadata.json"
    write_private(metadata_path, json.loads(read("cargo", "metadata", "--locked", "--no-deps", "--format-version", "1")))
    messages = output / "build.jsonl"
    command = ["cargo", "build", "--locked", "--release", "-p", "jope-cli", *CLI_FEATURE_ARGUMENTS, "--bin", "jope",
               "--jobs", str(jobs), "--message-format=json-render-diagnostics"]
    fd = os.open(messages, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    diagnostics = output / "build.stderr"
    error_fd = os.open(diagnostics, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream, os.fdopen(error_fd, "w") as stderr:
        completed = subprocess.run(command, cwd=checkout, env=env, stdout=stream, stderr=stderr, check=False)
        for captured in (stream, stderr):
            captured.flush()
            os.fsync(captured.fileno())
    if completed.returncode != 0:
        write_private(output / "build-failure.json", {"schema":"jope.private-cli-source-build/v1",
            "ok":False,"source_sha":source_sha,"command":command,"exit_code":completed.returncode,
            "cold_build_seconds":time.monotonic()-started,"fits":0,"warm_demo_measured":False,
            "cargo_messages":{"path":str(messages),"sha256":checksum(messages)},
            "diagnostics":{"path":str(diagnostics),"sha256":checksum(diagnostics)},
            "dependency_inventory":{"path":str(inventory_path),"sha256":checksum(inventory_path)}})
        raise ValueError("CLI source build failed; immutable diagnostics and failure receipt retained")
    executable = Path(read("cargo", "run", "-q", "--locked", "-p", "jope-ci", "--", "cargo-artifact", "select",
                           "--package", "jope-cli", "--bin", "jope", "--metadata", str(metadata_path), "--messages", str(messages)))
    if read("git", "rev-parse", "HEAD") != source_sha or read("git", "status", "--porcelain"):
        raise ValueError("source changed during the build")
    sha = checksum(executable)
    copied = output / "jope"
    size = copy_pinned_binary(executable, copied, sha)
    inspection = inspect_binary(copied)
    probe = probe_binary(copied)
    receipt = {"schema": "jope.private-cli-source-build/v1", "ok": True, "source_sha": source_sha,
               "binary": str(copied), "binary_sha256": sha, "bytes": size, "command": command,
               "rustc": read("rustc", "-vV"), "dependency_packages": packages, "inspection": inspection,
               "cargo_features": list(CLI_FEATURES), "dependency_command": graph_command,
               "dependency_inventory": {"path":str(inventory_path),"sha256":checksum(inventory_path)},
               "cargo_messages": {"path":str(messages),"sha256":checksum(messages)},
               "cargo_metadata": {"path":str(metadata_path),"sha256":checksum(metadata_path)},
               "diagnostics": {"path":str(diagnostics),"sha256":checksum(diagnostics)},
               "exit_code": completed.returncode,
               "dependency_graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
               "probe": probe, "jobs": jobs, "started_unix_ns": started_unix_ns,
               "completed_unix_ns": time.time_ns(), "cold_build_seconds": time.monotonic() - started,
               "build_environment": {key: env.get(key) for key in ("CARGO_INCREMENTAL", "CARGO_PROFILE_DEV_DEBUG", "CARGO_TARGET_DIR")},
               "torch_engine_dependency": False, "fits": 0, "warm_demo_measured": False,
               "cargo_target_selection": "Cargo compiler-artifact message; existing lane target reused"}
    write_private(output / "build-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    binary = commands.add_parser("binary", help="verify and activate a private local binary")
    binary.add_argument("--binary", type=Path, required=True)
    binary.add_argument("--sha256", required=True)
    binary.add_argument("--source-sha", required=True)
    binary.add_argument("--prefix", type=Path, required=True)
    source = commands.add_parser("source", help="build only jope-cli from an existing clean checkout")
    source.add_argument("--checkout", type=Path, required=True)
    source.add_argument("--source-sha", required=True)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        result = (install_binary(args.binary, args.sha256, args.source_sha, args.prefix)
                  if args.command == "binary" else build_source(args.checkout, args.source_sha, args.output, args.jobs))
    except Exception as error:
        print(json.dumps({"schema": "jope.private-cli-install-error/v1", "ok": False, "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
