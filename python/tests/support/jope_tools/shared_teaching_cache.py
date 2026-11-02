#!/usr/bin/env python3
"""Verify immutable payloads once per server lifetime and lend sealed memfds."""

import argparse
import array
import fcntl
import hashlib
import json
import os
import resource
import socket
import stat
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from resume_expired_cycle import atomic, require

SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


def receive_request(connection, seconds=5):
    deadline = time.monotonic() + seconds
    raw = bytearray()
    while not raw.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        require(remaining > 0 and len(raw) < 65537, "cache handshake deadline or size exceeded")
        connection.settimeout(remaining)
        block = connection.recv(65537 - len(raw))
        require(bool(block), "cache client disconnected before request completion")
        raw.extend(block)
    require(len(raw) <= 65536 and raw.count(b"\n") == 1, "invalid cache request")
    require(time.monotonic() < deadline, "cache handshake deadline elapsed")
    request = json.loads(raw)
    require(isinstance(request, dict), "cache request must be an object")
    return request


class Cache:
    def __init__(self, root, limit):
        self.root, self.limit = root.resolve(), limit
        self.entries = OrderedDict()
        self.bytes = self.verifications = self.hits = 0
        self.epoch = uuid.uuid4().hex

    def get(self, request):
        require(isinstance(request, dict), "cache request must be an object")
        deadline = request.get("deadline_unix_ms")
        if deadline is not None:
            require(
                type(deadline) is int and deadline > time.time() * 1000, "cache absolute deadline elapsed"
            )
        path = Path(request["path"]).resolve(strict=True)
        digest, length = request["sha256"], request["bytes"]
        require(path.is_relative_to(self.root), "teaching payload outside workspace")
        require(stat.S_ISREG(path.stat().st_mode), "teaching payload must be a regular file")
        require(type(length) is int and 0 < length <= self.limit, "payload exceeds sealed cache limit")
        require(
            type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "invalid content digest",
        )
        key = (digest, length)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        # A server FD close cannot release a client's live mmap. Until a
        # qualified borrower/refcount protocol exists, retain every object for
        # this epoch and reject pressure before allocating another memfd.
        require(
            self.bytes + length <= self.limit,
            "sealed cache epoch capacity exhausted; live-reference eviction is unavailable",
        )
        fd = os.memfd_create("jope-teaching-" + digest, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            h, size = hashlib.sha256(), 0
            source_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(source_fd, "rb") as source, os.fdopen(os.dup(fd), "wb") as destination:
                require(
                    stat.S_ISREG(os.fstat(source.fileno()).st_mode), "teaching payload must be a regular file"
                )
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    if deadline is not None:
                        require(time.time() * 1000 < deadline, "cache absolute deadline elapsed")
                    size += len(block)
                    require(size <= length, "teaching payload grew during verification")
                    h.update(block)
                    destination.write(block)
            require(
                size == length and h.hexdigest() == digest, "teaching content changed during verification"
            )
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        except BaseException:
            os.close(fd)
            raise
        self.entries[key] = fd
        self.bytes += size
        self.verifications += 1
        return fd

    def close(self):
        for fd in self.entries.values():
            os.close(fd)
        self.entries.clear()
        self.bytes = 0


def report_cache(cache, pool, next_report, now):
    # These counters are reconstructible telemetry, never admission receipts.
    # fsync per hit serializes all ranks behind the filesystem journal.
    if now >= next_report:
        atomic(
            pool / "teaching-cache.json",
            dict(
                epoch=cache.epoch,
                pid=os.getpid(),
                entries=len(cache.entries),
                bytes=cache.bytes,
                verifications=cache.verifications,
                hits=cache.hits,
            ),
        )
        return now + 10
    return next_report


def serve(root, pool, limit):
    require(pool.resolve().is_relative_to(root.resolve() / "runs"), "cache namespace outside private runs")
    pool.mkdir(parents=True, exist_ok=True, mode=0o700)
    pool.chmod(0o700)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(65536, hard)), hard))
    lock = (pool / "teaching-cache.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = pool / "teaching.sock"
    path.unlink(missing_ok=True)
    cache = Cache(root, limit)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(32)
        listener.settimeout(1)
        next_report = 0
        try:
            while True:
                next_report = report_cache(cache, pool, next_report, time.monotonic())
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    try:
                        request = receive_request(connection)
                        deadline = request.get("deadline_unix_ms")
                        require(
                            type(deadline) is int and 0 < deadline - time.time() * 1000 <= 30_000,
                            "cache request requires a bounded absolute deadline",
                        )
                        fd = cache.get(request)
                        remaining = (deadline - time.time() * 1000) / 1000
                        require(remaining > 0, "cache absolute deadline elapsed")
                        connection.settimeout(remaining)
                        response = dict(sha256=request["sha256"], bytes=request["bytes"], epoch=cache.epoch)
                        connection.sendmsg(
                            [b"L"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                        )
                        connection.sendall(json.dumps(response).encode() + b"\n")
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        atomic(pool / "teaching-cache-error.json", dict(epoch=cache.epoch, error=str(error)))
        finally:
            cache.close()
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--limit-bytes", type=int, default=12 * 1024**3)
    a = p.parse_args()
    serve(a.root, a.pool, a.limit_bytes)
