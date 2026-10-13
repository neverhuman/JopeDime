# S3 read cache for the xbabe hosts (design, S3-RET job c)

Status: design only. Nothing here is built, deployed or enabled. Owner: lane `jg_s3_retention`
(gap S3-RET, `S3_STORAGE_STRATEGY_20261013.md` §6 P2 / §8.8 P2). Date: 2026-10-13.

## 1. Problem

Internet egress from `us-east-2` to the xbabe hosts is the largest list-price item in the S3
bill: 819.9 GB in September and 1,944.7 GB from 1 to 12 October (Cost Explorer,
`USE2-DataTransfer-Out`), about 4.9 TB/month at the October pace, or about $438/month at the
$0.09/GB list price. Today a credit offsets it, so the billed amount is about $0. The exposure is
the list price after the credit ends (strategy §0).

Reads are not attributed yet. Cost Explorer reports egress per region and usage type. It has no
breakdown by bucket, prefix or caller. Each tool keeps its own local copies:

| Reader | Local copy today | Key | Shared across tools/hosts |
|---|---|---|---|
| `jope-datasets` (`crates/jope-datasets/src/blobs.rs`) | `$JOPE_HOME/datasets/blobs/<sha256>`, verified on fill and on reuse | sha256 | no / no |
| `jope-models` | `$JOPE_HOME/models/...` | sha256 | no / no |
| dltrain encodings (`deploy/dl-trainonly/.../dltrain/jtt1.py`) | per-run `cache_dir/<raw_sha256>.jtt1.gz` | raw sha256 | no / no |
| flywheel / GP Python runtimes (`deploy/dl-flywheel/pipelines/**/s3_library.py`, `program_inputs.py`) | mostly none; objects re-read per run | key + version | no / no |

As a result, the same immutable object can be downloaded once per tool, per run and per host.

## 2. Goals and non-goals

Goals:
1. Download each immutable S3 object version to the xbabe LAN at most once. Further reads are
   served from a host or LAN copy.
2. Never serve wrong bytes. Every served byte is verified against a content hash.
3. Never become a path for writes, deletes or sealed data.
4. Measure before and after: per-host receipts give hits, misses and the bytes avoided.

Non-goals:
- Changing what the writers write. Packing and tags are job (b).
- A shared POSIX filesystem, or FUSE.
- Caching listings of live prefixes.
- Changing any bucket, policy or lifecycle setting. Server access logs (§7) are listed as an
  option that needs owner approval.

## 3. What is safe to cache

| Read | Cacheable | Why |
|---|---|---|
| GET with a `version_id` | yes, indefinitely | Versioning is enabled on every JopeDime bucket. A version's bytes never change. |
| Content-addressed key (`blobs/sha256/<h>`, `objects/sha256/<h>`) | yes, indefinitely | The key names the content. The fill is verified against `<h>`. |
| A key from a manifest that pins `sha256` (rosters, receipts, `gp_hwm_ref`) | yes, keyed by that sha256 | The pin is the verifier. |
| A GET by key without a version | only after a HEAD | A HEAD costs a request and moves no payload. A hit requires the same `VersionId` and ETag. |
| Mutable pointers (`status/latest.json`, `HEAD.json`, `current.json`, `progress/*`) | no; always bypass | They change by design. A stale pointer is a correctness bug. |
| LIST | no in phase 1 | Live writers change prefixes. Sealed (closed) runs may be listed later from S3 Inventory instead. |
| `*-sealed` bucket, sealed catalogs, outer-test payloads | **never** | Sealed and outer-test data must stay outside the learning worker's input closure (AGENTS.md "Direction and evidence"). A shared cache readable by training jobs would break that. These are refused by a code deny list, not by configuration. |

## 4. Design

### 4.1 Layout (per host; never `/tmp`, never inside a git worktree)

```text
/data/jope-s3cache/                     # a large local volume; path set per host
  cas/sha256/<aa>/<sha256>              # immutable verified blobs, mode 0440
  cas/sha256/<aa>/<sha256>.partial      # in-flight fill (+ .partial.lock), as BlobDir does
  ver/<bucket>/<sha256(key)>/<version>  # 1-line pointer: <sha256> <bytes> <etag>
  index.sqlite                          # key, version, sha256, bytes, first_fill, last_access, hits, pins
  quarantine/<utc>/<sha256>             # entries that failed re-verification (moved, not removed)
  receipts/<yyyy-mm-dd>.json            # daily counters (§6)
```

The CAS layer reuses `jope-datasets`' `BlobDir` semantics: a `.partial` file plus rename, a
per-blob lock, and a hash on fill. When the cache is on the same filesystem as `$JOPE_HOME`,
existing dataset and model blobs can be hard-linked in rather than downloaded again.

### 4.2 Read path

```text
get(bucket, key, version?, sha256?) -> local path
  1. deny list: bucket or key matches the sealed/outer-test rules -> refuse (error, no fallback)
  2. resolve identity:
       sha256 given            -> cas/<sha256>
       version given           -> ver/<bucket>/<h(key)>/<version> -> sha256
       neither                 -> HEAD -> (version, etag) -> ver/... ; mutable-pointer keys bypass
  3. local hit: size check, plus a full re-hash if the entry was not verified within the
     scrub window -> serve
  4. LAN peer hit (phase 3): GET http://<peer>/cas/<sha256> -> stream-hash -> must equal sha256
  5. miss: GET from S3 (version-pinned) -> stream-hash into .partial -> compare with the pinned
     sha256 when one exists, else record the computed sha256 plus ETag/ChecksumSHA256 -> rename
  6. record hit/miss and bytes in index.sqlite; return the path (read-only file)
```

Integrity rules:
- A blob is published under `cas/<sha256>` only after its streamed sha256 matches.
- When a caller passes a `sha256` that disagrees with the `ver/` pointer, the call fails with an
  error. The cache never silently "fixes" it.
- Scrub: a daily idle-I/O job (`ionice -c3`) re-hashes entries, oldest-verified first, within a
  byte budget. A mismatch moves the entry to `quarantine/`, logs it, and the next read fetches it
  again. Scrub never deletes anything.
- S3 ETags are not content hashes for multipart objects. They are recorded for change detection
  only (the same rule as strategy §8.2 E1).

### 4.3 Capacity and eviction

Proposed budgets (to confirm against measurement): xbabe2 300 GB (1,007 GB free of 7.3 TB on
2026-10-13, 87% used), xbabe3 100 GB (187 GB free), xbabe1 none until the CI lanes' disk use is
known.

Eviction removes local copies. **Owner ruling 2026-10-13 04:48Z: no deletes without explicit
owner approval.** Until a ruling exempts cache entries, the cache runs in **no-evict mode**:
- at the high watermark it stops admitting new blobs and serves misses straight from S3 (a
  bypass, counted in the receipt);
- `pins` (active training manifests, rosters, frozen v1 encodings) are admitted first.

The proposal for the ruling: cache entries are verified copies of versioned S3 objects, and the
index records each one's bucket, key, version and sha256. Evicting one loses no information.
Allow LRU eviction (weighted by size) from a 95% high watermark down to 85%, skipping pinned
entries, and append every eviction to `receipts/`. If eviction must stay reversible, evict into
`~/JopeDime/.agent/trash` through `safe_rm`. That moves bytes without freeing space, so it only
helps for audit.

### 4.4 Interfaces

1. **Rust library (first).** A `CachedStore` wrapper next to `jope_datasets::store::Store`
   (`get_cached(bucket, key, version, sha256) -> PathBuf`). It uses the existing credential
   resolution (`credentials.rs`), `download_verified` and `BlobDir`. It goes into an existing
   crate so `Cargo.lock` and the evaluator identity stay unchanged
   ([ADR-0004](../../adr/ADR-0004-evaluator-identity-known-table.md)).
2. **CLI** for non-Rust callers: `jope-s3cache get s3://bucket/key --version V [--sha256 H]`
   prints the local path. `jope-s3cache stats` prints the counters. The torch/Python readers
   (dltrain encodings, flywheel runtimes) call the CLI or read `cas/` directly when they already
   hold the sha256. That keeps the Python change to a thin adapter, per ORDER 5.
3. **LAN peer endpoint (phase 3).** A read-only HTTP server on each host serves `cas/<sha256>`
   only, with no listing and no writes. Content addressing lets the client verify every byte.
   Bind it to the LAN interface, and serve no sealed or deny-listed content, because that is
   never admitted in the first place.
4. **Not proposed:** an S3-compatible local proxy (`AWS_ENDPOINT_URL_S3`). It would need zero
   client changes, but it would let every local process read S3 without credentials and would
   have to re-sign requests. Reconsider only after phases 1-3.

### 4.5 Concurrency

Fills are per-blob exclusive: the `.partial.lock` flock that `BlobDir` already uses. Readers of a
published blob need no lock, because blobs are immutable and never rewritten in place. The
SQLite index uses WAL mode and short transactions; the counters are append-only receipts.

## 5. Security and data hygiene

- The cache process uses the host's existing read path (`veox-jepson-prod` through
  `credential_process`). It never stores credentials. The index and receipts hold no secrets.
- The code deny list covers `*-sealed` buckets, sealed catalogs, and outer-test split payloads.
  The policy file cannot shorten it. This mirrors the retention brain's hard-T0 list
  (`crates/jope-learning/src/retention/classify.rs`).
- Cache files are mode 0440 under `ubuntu`. Every agent on the host runs as `ubuntu`, so this is
  hygiene, not a security boundary. The real control is the deny list on admission.
- No write path: the library exposes `get`/`stats` only. PUT, DELETE and lifecycle calls do not
  exist in it, and a test asserts that.

## 6. Measurement and receipts

Daily `receipts/<date>.json` per host: hits, misses, bypasses (deny list, mutable pointer, no
capacity), bytes served locally, bytes from LAN peers, bytes from S3, scrub results
(checked/ok/quarantined), and admissions refused at the watermark. The monthly S3 report (strategy
§5) adds `bytes_from_s3` per host next to the Cost Explorer egress line. That makes the saving
measured, not estimated.

## 7. Attribution first (phase 0)

Before building, find which prefixes drive the 4.9 TB/month:
1. **Free, no AWS change:** for one week, wrap the existing readers' GET calls with a per-host
   JSONL counter (bucket, key prefix depth 4, bytes, caller unit). `jope-datasets` already
   reports `downloaded` per blob.
2. **Needs owner approval (an AWS write):** S3 server access logging on
   `veox-jopedime-learning-use2` and `-datasets-use2` for 7 days, into the existing inventory
   bucket under a new prefix. It costs only the log storage. It gives exact per-key, per-requester
   egress.
3. Not recommended: CloudWatch request-metric filters ($0.30 per metric per month, prefix-level
   only), or Storage Lens advanced metrics.

Expected candidates, from the ledger and lane activity (hypotheses to test, not findings):
re-reads of `staging/lineage-a/encodings` (149.8 GB) and `lineage-b/accepted60000-restored`
(110.9 GB) by training runs; `gp-policy-dl/v2-trainonly/program-results` read through the GB-01
roster (26,462 referenced objects); BeyondArena `research/` payloads (264 GB); and
`gp-teacher-v6` workers reading `contexts/`.

## 8. Expected effect (estimate)

The saving is the fraction of egress that repeats reads of immutable versions:

`saving/month ≈ egress_GB × repeat_fraction × $0.09` (list price).

At 4.9 TB/month, a repeat fraction of 50% saves about $220/month and 80% saves about $350/month.
Adding LAN peer fetch lets each object cross the internet once for all three hosts rather than
once per host. These are list-price estimates; the billed saving is about $0 while the credit
lasts.

## 9. Rollout

| Phase | Change | Exit criterion |
|---|---|---|
| 0 | Attribution (§7.1; §7.2 if approved) | Top prefixes by egress, per host, with receipts |
| 1 | `CachedStore` + CLI on xbabe2, no-evict mode, deny list, scrub | Fixture tests: verified fill, corrupt-entry quarantine, deny-list refusal, no write API; a week of receipts |
| 2 | Python adapters for the top two readers from phase 0 (thin; torch paths only) | Measured drop in `bytes_from_s3` for those readers |
| 3 | LAN peer endpoint on xbabe2 and xbabe3 | Peer hits verified by sha256; zero sealed admissions |
| 4 | Eviction, only after an owner ruling (§4.3) | Eviction receipts reconcile with the index |

## 10. Failure modes

| Failure | Effect | Mitigation |
|---|---|---|
| Corrupt or partial local blob | Wrong bytes served | Hash on fill; atomic rename; scrub re-hash; quarantine |
| Stale mutable pointer served | Wrong run state | Mutable-pointer keys always bypass; HEAD-based identity for unversioned reads |
| Sealed or outer-test data cached | Leakage into the training closure | Code deny list on admission; test; receipts count refusals |
| Disk fills (xbabe2 at 87%) | Other lanes fail | Hard byte budget; stop admitting at the watermark; no-evict until ruled |
| Cache treated as source of truth | Data "kept" only locally | S3 stays canonical; the cache never serves an object S3 does not hold at that version; restore drills read S3 |
| LAN peer serves a wrong blob | Corruption spreads | The client verifies the sha256 of every peer byte; peers serve CAS only |
| Many hosts miss at once | Same object downloaded several times | Per-host fill lock; phase 3 peer-first lookup; accepted as a bounded first-fill cost |
| Cache dir in `/tmp` or a worktree | Lost at boot, or walked by `git status` | Path policy: refuse `/tmp`, refuse paths inside a git worktree |
