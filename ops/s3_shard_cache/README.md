# s3_shard_cache

Bounded on-disk S3 shard cache for training data. Root comes from
`JOPE_S3_CACHE_DIR` and must sit on the node's data drive, never `/`.

| node | `JOPE_S3_CACHE_DIR` |
| --- | --- |
| xbabe1 | `/mnt/amphora-backup/jope-s3cache` |
| xbabe2 | `/mnt/jope-archive/jope-s3cache` |
| xbabe3 | `/mnt/xbabe3-external/jope-s3cache` |
| xbabe4 (Docker) | named volume `jope_s3cache` mounted at `/cache` |

Defaults: cap 1.5 TB (`JOPE_S3_CACHE_CAP_BYTES`), 16 downloaders
(`JOPE_S3_CACHE_WORKERS`), prefetch depth 64 (`JOPE_S3_CACHE_PREFETCH`).
LRU eviction by atime; shards in use are pinned. Misses stream from S3
into memory so the trainer never waits on the prefetcher.

Warm a node: `python3 ops/s3_shard_cache/s3_shard_cache.py manifest.txt`
Test: `python3 -m pytest ops/s3_shard_cache`
