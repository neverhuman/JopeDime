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
ATTEMPTS = 6
BACKOFF_SECONDS = 2.0
# Cap so a long streak of transport retries on one package cannot stall the job by itself.
BACKOFF_CAP_SECONDS = 20.0
# Socket timeout for each attempt, applied to the connect and to every read. A TLS handshake on a degraded
# mirror exceeded 10s (observatory-ui job 135114099316, run 44392514985; the same class as typescript-parity
# "Network is unreachable"). 30s lets that handshake finish. Package count must not multiply the wait: the
# source that succeeds is cached and preferred for the rest of the process.
TIMEOUT_SECONDS = 30

# A connect to one address is bounded separately, and an address that fails to connect is tried last
# for the rest of the process, so a dead mirror address costs one short wait, not one per package.
CONNECT_SECONDS = 10
UNREACHABLE = set()


def backoff(attempt):
    """Exponential pause between transport retries. attempt is 1-based and already counted."""
    return min(BACKOFF_CAP_SECONDS, BACKOFF_SECONDS * (2 ** (attempt - 1)))


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
# A handshake that needed most of TIMEOUT_SECONDS and then succeeded is still a healthy fetch.
DEMOTE_AFTER = 3
SLOW_SECONDS = 15
MIRROR_FAILURES = {}
# Prefix from MIRRORS, or "snapshot" once the pinned snapshot is the source that actually served bytes.
PREFERRED_MIRROR = None


def sources(url):
    """(mirror, url) choices for one locked URL.

    The cached working mirror leads. A demoted mirror is tried after the others. Otherwise the locked
    mirror leads.
    """
    for prefix in MIRRORS:
        if url.startswith(prefix):
            ordered = sorted(
                MIRRORS,
                key=lambda mirror: (
                    mirror != PREFERRED_MIRROR,
                    MIRROR_FAILURES.get(mirror, 0) >= DEMOTE_AFTER,
                    mirror != prefix,
                    mirror,
                ),
            )
            return [(mirror, mirror + url[len(prefix):]) for mirror in ordered]
    return [(None, url)]


def load_preferred(cache):
    """Reuse the mirror that served a verified package for an earlier run in this tool cache."""
    global PREFERRED_MIRROR
    marker = cache / "preferred-mirror"
    if not marker.is_file() or marker.is_symlink():
        return
    text = marker.read_text().strip()
    if text in MIRRORS or text == "snapshot":
        PREFERRED_MIRROR = text


def remember(archives, mirror):
    """Record the source that just served verified bytes. Later packages in this process try it first."""
    global PREFERRED_MIRROR
    if mirror not in MIRRORS and mirror != "snapshot":
        raise ValueError("refusing to cache an unofficial mirror")
    PREFERRED_MIRROR = mirror
    marker = archives.parent / "preferred-mirror"
    temporary = marker.with_name("preferred-mirror.tmp")
    temporary.write_text(mirror + "\n")
    temporary.replace(marker)


def next_choice(choices, last_url):
    """The next source after last_url, or the first when last_url is absent from this list."""
    if not choices:
        raise ValueError("no download source")
    if last_url is None:
        return choices[0]
    urls = [url for _, url in choices]
    if last_url not in urls:
        return choices[0]
    return choices[(urls.index(last_url) + 1) % len(choices)]


# Ubuntu deletes superseded noble-updates/noble-security debs from the archive pool, so a pinned URL can start
# answering 404 on every mirror (libsoup3 3.4.4-5ubuntu0.10 on 2026-11-10). snapshot.ubuntu.com keeps each dated
# state of the archive. When the locked pool path is gone (HTTP 404/410) from every mirror, the same path is
# fetched from this one pinned snapshot, and the bytes must still match the locked size and sha256. Every URL in
# the three committed locks resolved at this stamp on 2026-11-11. A package pinned after the stamp is not in it;
# move the stamp forward when re-pinning, or that package keeps today's behaviour (the 404 is raised).
SNAPSHOT = "https://snapshot.ubuntu.com/ubuntu/"
SNAPSHOT_STAMP = "20261111T000000Z"
POOL_ROOTS = MIRRORS + ("https://security.ubuntu.com/ubuntu/",)
GONE = (404, 410)


def gone(error):
    """The mirror answered that this path does not exist (404) or was removed (410)."""
    return isinstance(error, urllib.error.HTTPError) and error.code in GONE


def snapshot_url(url):
    """The pinned snapshot copy of an official archive pool URL; None for any other URL."""
    for root in POOL_ROOTS:
        if url.startswith(root) and url[len(root):].startswith("pool/"):
            return SNAPSHOT + SNAPSHOT_STAMP + "/" + url[len(root):]
    return None


def download(package, path, archives):
    """Fetch one locked package. Transient failures are retried; a 404/410 moves on to the next source.

    The order is the cached working mirror, the locked mirror, the other mirror, then the pinned snapshot.
    The snapshot is used after every live mirror answered 404/410, and also once every live mirror has failed
    transport for this package (handshake timeout, unreachable, dropped connection). Snapshot bytes still have
    to match the locked size and sha256. A 404/410 is not a transport failure. It does not use an attempt,
    sleep, or count toward DEMOTE_AFTER. Any other 4xx and any content mismatch still fail at once.
    """
    global PREFERRED_MIRROR
    missing, last_gone, attempt = set(), None, 1
    last_url = None
    transport_failed = set()
    snapshot_transport_tried = False
    while True:
        live = [choice for choice in sources(package["url"]) if choice[1] not in missing]
        fallback = snapshot_url(package["url"])
        snapshot_choice = (None, fallback) if fallback and fallback not in missing else None
        mirrors_in_play = {mirror for mirror, _ in sources(package["url"]) if mirror}
        cached_snapshot = (
            PREFERRED_MIRROR == "snapshot"
            and snapshot_choice is not None
            and bool(mirrors_in_play)
            and all(MIRROR_FAILURES.get(mirror, 0) >= DEMOTE_AFTER for mirror in mirrors_in_play)
        )
        if cached_snapshot:
            choices = [snapshot_choice]
            snapshot = True
        elif (snapshot_choice is not None and not snapshot_transport_tried and mirrors_in_play
              and transport_failed >= mirrors_in_play):
            choices = [snapshot_choice]
            snapshot = True
            snapshot_transport_tried = True
        elif not live:
            if snapshot_choice is None:
                raise last_gone
            choices = [snapshot_choice]
            snapshot = True
        else:
            choices = live
            snapshot = False
        mirror, url = next_choice(choices, last_url)
        last_url = url
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
                name = package.get("filename") or PurePosixPath(urllib.parse.urlsplit(url).path).name
                if snapshot and transport_failed:
                    for failed in transport_failed:
                        MIRROR_FAILURES[failed] = max(MIRROR_FAILURES.get(failed, 0), DEMOTE_AFTER)
                    remember(archives, "snapshot")
                    print(f"browser-deps: pinned package {name} fetched from snapshot {SNAPSHOT_STAMP} "
                          f"after mirror transport failure, sha256 verified ({url})", file=sys.stderr)
                elif snapshot and cached_snapshot:
                    remember(archives, "snapshot")
                    print(f"browser-deps: pinned package {name} fetched from cached snapshot {SNAPSHOT_STAMP}, "
                          f"sha256 verified ({url})", file=sys.stderr)
                elif snapshot:
                    print(f"browser-deps: pinned package {name} gone from archive; served from snapshot "
                          f"{SNAPSHOT_STAMP}, sha256 verified ({url})", file=sys.stderr)
                elif mirror is not None and MIRROR_FAILURES.get(mirror, 0) < DEMOTE_AFTER:
                    remember(archives, mirror)
                temporary.replace(path)
                return
            except (Exception,) as error:
                if gone(error):
                    missing.add(url)
                    last_gone = error
                    print(f"browser-deps: {url} answered HTTP {error.code}; trying the next source", file=sys.stderr)
                    continue
                if mirror is not None and transient(error):
                    MIRROR_FAILURES[mirror] = MIRROR_FAILURES.get(mirror, 0) + 1
                    transport_failed.add(mirror)
                    if PREFERRED_MIRROR == mirror:
                        PREFERRED_MIRROR = None
                elif snapshot and transient(error) and PREFERRED_MIRROR == "snapshot":
                    # The cached snapshot missed. The next attempt may use a live mirror again.
                    PREFERRED_MIRROR = None
                if attempt == ATTEMPTS or not transient(error):
                    raise
                print(f"browser-deps: attempt {attempt}/{ATTEMPTS} for {url} failed: {error!r}; retrying", file=sys.stderr)
            finally:
                if temporary.exists():
                    temporary.unlink()
        time.sleep(backoff(attempt))
        attempt += 1


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
        load_preferred(cache)
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
