#!/usr/bin/env python3
"""Verify and extract pinned Ubuntu browser libraries into task-local cache; never install."""
import argparse
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

LOCK = Path(__file__).with_name("browser-deps.lock.json")
# The archive mirror occasionally drops a connection; a job fetches about a hundred packages.
ATTEMPTS = 4
BACKOFF_SECONDS = 2.0


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name):
    name = name.removeprefix("./")
    if name in ("", "."):
        return ""
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        raise ValueError("unsafe package member path")
    return name.rstrip("/")


def inventory(archives, cache):
    expected = {}
    for archive in archives:
        # A seekable spool supports package hardlinks without loading an entire
        # decompressed filesystem archive into memory. The verified deb is custody.
        with tempfile.TemporaryFile(dir=cache) as spool:
            subprocess.run(["dpkg-deb", "--fsys-tarfile", str(archive)], stdout=spool, check=True)
            spool.seek(0)
            with tarfile.open(fileobj=spool) as tar:
                for item in tar.getmembers():
                    name = safe_name(item.name)
                    if not name or item.isdir():
                        continue
                    if item.issym():
                        expected[name] = {"kind": "symlink", "target": item.linkname}
                    elif item.isfile() or item.islnk():
                        with contextlib.closing(tar.extractfile(item)) as stream:
                            expected[name] = {"kind": "file", "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
                    else:
                        raise ValueError("unsupported package member type: " + name)
    return expected


def verify_runtime(root, expected):
    actual = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in list(directories):
            p = Path(parent) / name
            if p.is_symlink():
                actual[str(p.relative_to(root))] = {"kind": "symlink", "target": os.readlink(p)}
                directories.remove(name)
        for name in files:
            p = Path(parent) / name
            mode = p.lstat().st_mode
            if stat.S_ISLNK(mode):
                value = {"kind": "symlink", "target": os.readlink(p)}
            elif stat.S_ISREG(mode):
                value = {"kind": "file", "sha256": digest(p)}
            else:
                raise ValueError("unexpected extracted file type")
            actual[str(p.relative_to(root))] = value
    if actual != expected:
        raise ValueError("browser dependency cache differs from verified deb contents; refusing reuse")


def transient(error):
    """A dropped or refused transport, or a server-side HTTP error; never a 4xx or a content mismatch."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return isinstance(error, (urllib.error.URLError, http.client.HTTPException, OSError))


def download(package, path, archives):
    """Fetch one locked package, retrying only transient transport failures."""
    for attempt in range(1, ATTEMPTS + 1):
        # Exclusive temporary output is unpublished until size+hash verification.
        with tempfile.NamedTemporaryFile(dir=archives, delete=False) as output:
            temporary = Path(output.name)
            try:
                with urllib.request.urlopen(package["url"], timeout=60) as response:
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > package["size"]:
                            raise ValueError("package response exceeds locked size")
                        output.write(block)
                output.flush()
                if size != package["size"] or digest(temporary) != package["sha256"]:
                    raise ValueError("downloaded package differs from lock")
                temporary.replace(path)
                return
            except Exception as error:
                if attempt == ATTEMPTS or not transient(error):
                    raise
                print(f"browser-deps: attempt {attempt}/{ATTEMPTS} for {package['url']} failed: {error!r}; retrying", file=sys.stderr)
            finally:
                if temporary.exists():
                    temporary.unlink()
        time.sleep(BACKOFF_SECONDS * attempt)


def prepare(cache, receipt):
    if platform.machine() != "x86_64":
        raise ValueError("Ubuntu browser dependency lock is amd64 only")
    lock_bytes = LOCK.read_bytes()
    lock = json.loads(lock_bytes)
    if lock["schema"] != "jope.chromium-ubuntu24.04-dependencies/v1" or lock["architecture"] != "amd64":
        raise ValueError("unsupported browser dependency lock")
    identity = hashlib.sha256(lock_bytes).hexdigest()
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archives = cache / "archives"
    archives.mkdir(exist_ok=True)
    with (cache / "prepare.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        paths = []
        for package in lock["packages"]:
            url = urllib.parse.urlsplit(package["url"])
            if url.scheme != "https" or url.hostname not in ("us.archive.ubuntu.com", "security.ubuntu.com") or not url.path.startswith("/ubuntu/pool/"):
                raise ValueError("dependency is not from the pinned official Ubuntu archive")
            sha = package["sha256"]
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid package digest")
            path = archives / (sha + ".deb")
            if path.is_symlink():
                raise ValueError("package cache must not be a symlink")
            if not path.exists():
                download(package, path, archives)
            if path.stat().st_size != package["size"] or digest(path) != sha:
                raise ValueError("cached package differs from lock")
            paths.append(path)
        expected = inventory(paths, cache)
        runtime = cache / ("runtime-" + identity)
        if runtime.is_symlink():
            raise ValueError("runtime cache must not be a symlink")
        if not runtime.exists():
            runtime.mkdir()
            for path in paths:
                subprocess.run(["dpkg-deb", "--extract", str(path), str(runtime)], check=True)
        verify_runtime(runtime, expected)
        fontcache = cache / ("font-cache-" + identity)
        fontcache.mkdir(exist_ok=True)
        fontconfig = cache / ("fontconfig-" + identity + ".xml")
        config = ET.Element("fontconfig")
        ET.SubElement(config, "dir").text = str(runtime / "usr/share/fonts")
        ET.SubElement(config, "cachedir").text = str(fontcache)
        ET.ElementTree(config).write(fontconfig, encoding="utf-8", xml_declaration=True)
        proof = {"schema": "jope.browser-dependencies/v1", "lock_sha256": identity,
                 "packages": len(paths), "package_bytes": sum(p["size"] for p in lock["packages"]),
                 "inventory_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                 "inventory_files": len(expected), "runtime": str(runtime), "fontconfig": str(fontconfig),
                 "scope": "Task-local native browser dependencies. No apt install, maintainer scripts, host/image changes or runner provenance changes."}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x") as stream:
            stream.write(json.dumps(proof, indent=2) + "\n")
        print(str(runtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cache, args.receipt)
