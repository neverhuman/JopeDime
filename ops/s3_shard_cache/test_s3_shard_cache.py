import os, sys, threading
sys.path.insert(0, os.path.dirname(__file__))
from s3_shard_cache import ShardCache, load_manifest


class Fake:
    def __init__(self, sizes):
        self.sizes = sizes; self.calls = []; self.lock = threading.Lock()
    def _body(self, k): return bytes([hash(k) % 256]) * self.sizes[k]
    def to_file(self, b, k, p):
        with self.lock: self.calls.append(("file", k))
        open(p, "wb").write(self._body(k))
    def to_bytes(self, b, k):
        with self.lock: self.calls.append(("mem", k))
        return self._body(k)


def test_prefetch_hit_and_stream(tmp_path):
    f = Fake({f"s{i}": 100 for i in range(5)})
    c = ShardCache(str(tmp_path), cap_bytes=10_000, workers=2, prefetch=2, fetcher=f)
    assert c.get("b", "s0") == f._body("s0")  # miss -> in-memory stream, nothing written
    assert c.stats["misses_streamed"] == 1 and c.size == 0
    c.schedule([("b", "s1"), ("b", "s1")]); c.close()
    assert [x for x in f.calls if x == ("file", "s1")] == [("file", "s1")]
    c.pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(1)
    assert c.get("b", "s1") == f._body("s1") and c.stats["hits"] == 1


def test_lru_cap(tmp_path):
    f = Fake({f"s{i}": 400 for i in range(6)})
    c = ShardCache(str(tmp_path), cap_bytes=1000, workers=1, prefetch=1, fetcher=f)
    keys = [k for k, _ in c.iterate([("b", f"s{i}", 400) for i in range(6)])]
    c.close()
    assert keys == [f"s{i}" for i in range(6)]
    assert c.size <= 1000 and c.stats["evicted"] >= 3
    on_disk = sum(p.stat().st_size for p in (tmp_path / "shards").iterdir())
    assert on_disk == c.size <= 1000


def test_env_root_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("JOPE_S3_CACHE_DIR", str(tmp_path / "c"))
    c = ShardCache(fetcher=Fake({})); c.close()
    assert (tmp_path / "c" / "shards").is_dir()
    m = tmp_path / "m.txt"; m.write_text('s3://bk/a/b\n{"bucket":"x","key":"y","bytes":5}\n')
    assert load_manifest(str(m)) == [("bk", "a/b", 0), ("x", "y", 5)]
