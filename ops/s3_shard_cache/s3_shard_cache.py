"""Bounded S3 shard cache with LRU eviction and background prefetch.

Keeps training data off root disks: the cache root lives on each node's
data drive (default from $JOPE_S3_CACHE_DIR). Shards are fetched by N
parallel downloaders, K shards ahead of the consumer, from a manifest.
A shard that is not cached yet is streamed from S3 straight into memory
so training never blocks on the prefetcher.

Env:
  JOPE_S3_CACHE_DIR        cache root (required unless root= is passed)
  JOPE_S3_CACHE_CAP_BYTES  size cap, default 1.5 TB
  JOPE_S3_CACHE_WORKERS    parallel downloaders, default 16
  JOPE_S3_CACHE_PREFETCH   prefetch depth K, default 64
  AWS_PROFILE              e.g. veox-jepson
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

DEFAULT_CAP = int(1.5 * 1000**4)
ENV_DIR = "JOPE_S3_CACHE_DIR"


def _key_name(bucket: str, key: str) -> str:
    return hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()


class S3Fetcher:
    """Default fetcher backed by boto3. Tests inject a fake."""

    def __init__(self, workers: int):
        import boto3
        from botocore.config import Config

        self.s3 = boto3.Session().client("s3", config=Config(max_pool_connections=max(10, workers * 2)))

    def to_file(self, bucket: str, key: str, path: str) -> None:
        self.s3.download_file(bucket, key, path)

    def to_bytes(self, bucket: str, key: str) -> bytes:
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()


class ShardCache:
    def __init__(self, root: Optional[str] = None, cap_bytes: Optional[int] = None,
                 workers: Optional[int] = None, prefetch: Optional[int] = None,
                 fetcher=None):
        root = root or os.environ.get(ENV_DIR)
        if not root:
            raise ValueError(f"set {ENV_DIR} to a directory on the data drive")
        self.root = Path(root)
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        (self.root / "tmp").mkdir(parents=True, exist_ok=True)
        self.cap = int(cap_bytes or os.environ.get("JOPE_S3_CACHE_CAP_BYTES", DEFAULT_CAP))
        self.workers = int(workers or os.environ.get("JOPE_S3_CACHE_WORKERS", 16))
        self.prefetch = int(prefetch or os.environ.get("JOPE_S3_CACHE_PREFETCH", 64))
        self.fetcher = fetcher or S3Fetcher(self.workers)
        self.lock = threading.Lock()
        self.pinned: dict[str, int] = {}
        self.inflight: dict[str, threading.Event] = {}
        self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix="shardcache")
        self.stats = {"hits": 0, "misses_streamed": 0, "prefetched": 0, "evicted": 0}
        self.size = sum(p.stat().st_size for p in (self.root / "shards").iterdir())

    # ---- paths / LRU ----
    def path(self, bucket: str, key: str) -> Path:
        return self.root / "shards" / _key_name(bucket, key)

    def _touch(self, p: Path) -> None:
        now = time.time()
        os.utime(p, (now, now))

    def _evict(self, need: int) -> None:
        with self.lock:
            if self.size + need <= self.cap:
                return
            files = sorted((p.stat().st_atime, p) for p in (self.root / "shards").iterdir())
            for _, p in files:
                if self.size + need <= self.cap:
                    break
                if p.name in self.pinned:
                    continue
                try:
                    sz = p.stat().st_size
                    p.unlink()
                except FileNotFoundError:
                    continue
                self.size -= sz
                self.stats["evicted"] += 1

    # ---- fetch ----
    def _download(self, bucket: str, key: str, size_hint: int = 0) -> None:
        name = _key_name(bucket, key)
        dst = self.path(bucket, key)
        with self.lock:
            if dst.exists() or name in self.inflight:
                return
            ev = self.inflight[name] = threading.Event()
        try:
            self._evict(size_hint)
            tmp = self.root / "tmp" / f"{name}.{threading.get_ident()}"
            self.fetcher.to_file(bucket, key, str(tmp))
            sz = tmp.stat().st_size
            self._evict(sz)
            os.replace(tmp, dst)
            with self.lock:
                self.size += sz
                self.stats["prefetched"] += 1
        finally:
            with self.lock:
                self.inflight.pop(name, None)
            ev.set()

    def schedule(self, items: Iterable[tuple]) -> None:
        for it in items:
            bucket, key = it[0], it[1]
            hint = it[2] if len(it) > 2 else 0
            self.pool.submit(self._download, bucket, key, hint)

    def get(self, bucket: str, key: str) -> bytes:
        """Return shard bytes: cache hit, else stream from S3 into memory."""
        p = self.path(bucket, key)
        name = p.name
        with self.lock:
            self.pinned[name] = self.pinned.get(name, 0) + 1
        try:
            if p.exists():
                self._touch(p)
                self.stats["hits"] += 1
                return p.read_bytes()
            self.stats["misses_streamed"] += 1
            return self.fetcher.to_bytes(bucket, key)
        finally:
            with self.lock:
                self.pinned[name] -= 1
                if not self.pinned[name]:
                    del self.pinned[name]

    def iterate(self, manifest: list) -> Iterator[tuple]:
        """Yield (key, bytes) in manifest order, keeping K shards prefetched ahead."""
        ahead = 0
        for i, it in enumerate(manifest):
            while ahead < min(len(manifest), i + 1 + self.prefetch):
                self.schedule([manifest[ahead]])
                ahead += 1
            yield it[1], self.get(it[0], it[1])

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def load_manifest(path: str) -> list:
    """JSONL rows {bucket,key,bytes?} or lines s3://bucket/key."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("s3://"):
            b, _, k = line[5:].partition("/")
            out.append((b, k, 0))
        else:
            r = json.loads(line)
            out.append((r["bucket"], r["key"], int(r.get("bytes", 0))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm the shard cache from a manifest")
    ap.add_argument("manifest")
    ap.add_argument("--root")
    ap.add_argument("--cap-bytes", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    c = ShardCache(a.root, a.cap_bytes, a.workers)
    t = time.time()
    c.schedule(load_manifest(a.manifest))
    c.close()
    print(json.dumps({**c.stats, "bytes": c.size, "seconds": round(time.time() - t, 1)}))


if __name__ == "__main__":
    main()
