#!/usr/bin/env python3
"""Verify and extract pinned Ubuntu task-local dependencies into task-local cache; never install."""
import argparse
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
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
# Socket timeout for each attempt, applied to the connect and to every read. When one of the mirror's addresses
# stops answering (91.189.91.83 on 2026-11-10), each fetch that resolves to it first waits this long before the
# next address is tried. About a hundred fetches share the 25-minute observatory-ui budget, so keep it small.
TIMEOUT_SECONDS = 10

# A connect to one address is bounded separately, and an address that fails to connect is tried last
# for the rest of the process, so a dead mirror address costs one short wait, not one per package.
CONNECT_SECONDS = 5
UNREACHABLE = set()


def connect_skipping_unreachable(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
    """socket.create_connection with a bounded connect that remembers unreachable addresses."""
    host, port = address
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    infos.sort(key=lambda info: (host, port, info[4][0]) in UNREACHABLE)
    reads = None if timeout is socket._GLOBAL_DEFAULT_TIMEOUT else timeout
    error = OSError(f"no address for {host}")
    for family, kind, proto, _, sockaddr in infos:
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(CONNECT_SECONDS if reads is None else min(CONNECT_SECONDS, reads))
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            sock.settimeout(reads)
            return sock
        except OSError as failure:
            sock.close()
            UNREACHABLE.add((host, port, sockaddr[0]))
            error = failure
    raise error


class SkippingHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = connect_skipping_unreachable


class SkippingHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(SkippingHTTPSConnection, req, context=self._context)


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


# Ubuntu pool paths are immutable and every package is checked against its locked size and sha256, so a second
# mirror changes availability only, never content. The locked URL is tried first. After DEMOTE_AFTER transient
# failures or slow fetches on one mirror in this process, the others are preferred for the remaining packages.
MIRRORS = ("https://us.archive.ubuntu.com/ubuntu/", "https://mirrors.edge.kernel.org/ubuntu/")
# A fetch slower than SLOW_SECONDS also counts against its mirror: a degraded mirror is usually slow, not down.
DEMOTE_AFTER = 3
SLOW_SECONDS = 5
MIRROR_FAILURES = {}


def sources(url):
    """(mirror, url) choices for one locked URL: its own mirror first unless demoted, then the others."""
    for prefix in MIRRORS:
        if url.startswith(prefix):
            ordered = sorted(MIRRORS, key=lambda mirror: (MIRROR_FAILURES.get(mirror, 0) >= DEMOTE_AFTER, mirror != prefix))
            return [(mirror, mirror + url[len(prefix):]) for mirror in ordered]
    return [(None, url)]


def download(package, path, archives):
    """Fetch one locked package, retrying only transient transport failures."""
    for attempt in range(1, ATTEMPTS + 1):
        choices = sources(package["url"])
        mirror, url = choices[(attempt - 1) % len(choices)]
        # Exclusive temporary output is unpublished until size+hash verification.
        with tempfile.NamedTemporaryFile(dir=archives, delete=False) as output:
            temporary = Path(output.name)
            try:
                started = time.monotonic()
                with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > package["size"]:
                            raise ValueError("package response exceeds locked size")
                        output.write(block)
                output.flush()
                if size != package["size"] or digest(temporary) != package["sha256"]:
                    raise ValueError("downloaded package differs from lock")
                if mirror is not None and time.monotonic() - started > SLOW_SECONDS:
                    MIRROR_FAILURES[mirror] = MIRROR_FAILURES.get(mirror, 0) + 1
                temporary.replace(path)
                return
            except (Exception,) as error:
                if mirror is not None and transient(error):
                    MIRROR_FAILURES[mirror] = MIRROR_FAILURES.get(mirror, 0) + 1
                if attempt == ATTEMPTS or not transient(error):
                    raise
                print(f"browser-deps: attempt {attempt}/{ATTEMPTS} for {url} failed: {error!r}; retrying", file=sys.stderr)
            finally:
                if temporary.exists():
                    temporary.unlink()
        time.sleep(BACKOFF_SECONDS * attempt)


def hermetic_fontconfig(runtime, fontcache):
    config = ET.Element("fontconfig")
    ET.SubElement(config, "dir").text = str(runtime / "usr/share/fonts")
    ET.SubElement(config, "cachedir").text = str(fontcache)
    base = runtime / "etc/fonts/fonts.conf"
    if not base.exists():
        return config  # Generic non-browser dependency locks need no font rules.
    dropins = runtime / "etc/fonts/conf.d"
    sources = [base] + (sorted(dropins.glob("*.conf")) if dropins.is_dir() else [])
    for path in sources:
        if path.name in ("50-user.conf", "51-local.conf"):
            continue
        if path.is_symlink() and Path(os.readlink(path)).is_absolute():
            target = Path(os.readlink(path))
            if not target.is_relative_to("/usr/share/fontconfig/conf.avail") or ".." in target.parts:
                raise ValueError("font configuration escapes the verified runtime")
            resolved = (runtime / target.relative_to("/")).resolve(strict=True)
        else:
            resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(runtime.resolve()):
            raise ValueError("font configuration escapes the verified runtime")
        for node in ET.parse(resolved).getroot():
            if node.tag in ("match", "alias", "selectfont"):
                if any(child.tag in ("include", "dir", "cachedir") for child in node.iter()):
                    raise ValueError("font rule imports ambient configuration")
                config.append(node)
    return config


def prepare(cache, receipt, lock_path=None):
    urllib.request.install_opener(urllib.request.build_opener(SkippingHTTPSHandler()))
    if platform.machine() != "x86_64":
        raise ValueError("Ubuntu browser dependency lock is amd64 only")
    lock_bytes = (LOCK if lock_path is None else lock_path).read_bytes()
    lock = json.loads(lock_bytes)
    if lock["schema"] not in ("jope.chromium-ubuntu24.04-dependencies/v1", "jope.ubuntu24.04-dependencies/v1") or lock["architecture"] != "amd64":
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
            official = url.netloc in ("us.archive.ubuntu.com", "security.ubuntu.com") and url.path.startswith("/ubuntu/pool/")
            if url.netloc == "snapshot.ubuntu.com":
                snapshot = re.fullmatch(r"/ubuntu/([0-9]{8}T[0-9]{6}Z)/pool/.+", url.path)
                official = False
                if snapshot:
                    try:
                        time.strptime(snapshot[1], "%Y%m%dT%H%M%SZ")
                        official = True
                    except ValueError:
                        pass
            if (url.scheme != "https" or url.query or url.fragment or not official
                    or ".." in PurePosixPath(urllib.parse.unquote(url.path)).parts):
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
        config = hermetic_fontconfig(runtime, fontcache)
        ET.ElementTree(config).write(fontconfig, encoding="utf-8", xml_declaration=True)
        proof = {"schema": "jope.browser-dependencies/v1", "lock_sha256": identity,
                 "packages": len(paths), "package_bytes": sum(p["size"] for p in lock["packages"]),
                 "inventory_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                 "inventory_files": len(expected), "runtime": str(runtime), "fontconfig": str(fontconfig),
                 "scope": "Task-local pinned Ubuntu dependencies. No apt install, maintainer scripts, host/image changes or runner provenance changes."}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x") as stream:
            stream.write(json.dumps(proof, indent=2) + "\n")
        print(str(runtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, help="explicit pinned Ubuntu dependency lock; defaults to Chromium")
    args = parser.parse_args()
    prepare(args.cache, args.receipt, args.lock)
