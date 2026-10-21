# Blob verification, memory and execution review

Review base: `18b671e90102df16de8b0cd95f93fd4378c911ef` (2026-10-21).
This change addresses the shared dataset/model verification path. It does not
change a fitted operator, evaluator mathematics, a model, or a historical evidence file.

## Changes in this PR

| Finding | Previous behavior | Change |
| --- | --- | --- |
| Binary blobs took the CSV path | `BlobDir` and model verification counted newlines, counted commas and materialized a lossy UTF-8 header while hashing arbitrary weights | `hash::digest_content` computes only SHA-256 and actual byte count in a single streaming pass |
| Async verification monopolized the caller | Cached blobs, completed destinations and resumable prefixes were read and hashed synchronously while polling their futures; model publication prehashed on the executor too | Those passes run through `digest_content_async` / `content_hasher_async` on admitted dedicated read threads |
| Unbounded offload would oversubscribe CPU and buffers | Moving every call to a blocking pool alone could admit hundreds of concurrent readers | One process-wide semaphore admits `min(available_parallelism at initialization, 4)` hashing jobs across callers and runtimes |
| Cancellation could defeat a worker limit | Releasing a permit when an awaiting future is dropped permits replacement work while its uninterruptible read still runs | The actual read worker owns the permit until work finishes, even if its receiving future or Tokio runtime is dropped |
| Header storage grew before truncation | A first read without a newline could append 1 MiB to a vector before truncating its length to 64 KiB; later chunks could grow that allocation again | Append only the remaining raw header prefix, while still counting every comma and hashing every byte |
| Repeated hashing code | `store.rs` maintained its own streaming SHA-256 reader | The completed-destination and resume paths share the content reader |

Each admitted hash uses a 1 MiB read buffer. The admission limit is a bound on
these hashing jobs and their buffers, not on process RSS, filesystem cache,
network downloads, or model publication snapshots. Synchronous verification
APIs remain synchronous. The asynchronous hashing API requires a Tokio runtime,
including a current-thread runtime. The worker limit is initialized once; it is
not a live CPU-affinity or cgroup admission controller.

Dedicated threads avoid a circular wait in the shared Tokio blocking pool: a
destination-lock waiter could occupy its last worker while the lock holder
awaited hashing queued to that same pool. Thread creation is fallible and
returns an I/O error. Cancelling a receiver or dropping its runtime leaves an
admitted read holding its process-wide permit until it actually stops; a
stalled read therefore retains capacity. This bounds active hash reads rather
than promising prompt cancellation or isolation from stalled storage.

SHA-256 remains mandatory on every cache hit. This is not a size/mtime shortcut.
The existing destination lock, ranged download, final hash verification,
immutable publication and failure accounting remain in place. Resume now checks
that the number of prefix bytes actually read matches the observed size before
issuing a ranged request. A cancelled read may finish in the background;
it cannot release its slot early or publish a download by itself.

## Repository review and next priorities

These are source findings, not measured end-to-end throughput claims. The existing
[architecture map](../architecture.md), [agent contract](../../AGENTS.md),
[layout exceptions](../../deploy/ci/repo-layout-allow.txt), and
[cleanup record](../../MASTER_CLEAN_UP.md) define the current boundaries.

| Priority | Exact area | Finding and required follow-up |
| --- | --- | --- |
| P1 | `crates/jope-datasets/src/store.rs::download_verified` | Stream chunks still use synchronous file writes and hashing on the polling thread. A dedicated bounded writer/hasher with backpressure needs cancellation, lock-ownership, disk-full and checksum-failure tests before replacing it. Measure executor latency and download throughput together. |
| P1 | `crates/jope-models/src/publish.rs`, `crates/jope-datasets/src/verified_object.rs` | Publication intentionally takes a complete verified source snapshot. Large tensors can therefore consume file-sized memory in addition to transport buffers. A streaming design must preserve source-mutation refusal, create-only publication, remote content verification and ambiguous-response reconciliation; merely streaming an unpinned path would weaken the boundary. |
| P1 | `crates/feat-math/src/stage_cache.rs` | Serialization during admission and cloning fitted transforms under a shard lock are candidate hot spots. A separately claimed review lane owns this area. Benchmark allocation/lock contention and preserve cache key, prediction and evaluator identity contracts. |
| P1 | `crates/jope-pipeline/src/parallel.rs`, `crates/jope-engine/src/gp.rs`, `crates/jope-engine/src/components.rs` | Pipeline search has explicit CPU admission and 32 MiB evaluation stacks; other entry points build global/fixed-size pools. Inventory actual nested Rayon/Torch/BLAS threads and admitted CPU quotas before choosing a common per-task policy. Do not reduce evaluation stacks without deep-fit tests. |
| P2 | `crates/jope-engine/src/eval_registry/service.rs` | Connection admission is already capped (`max_workers * 2 + 16`, with validated worker limits). Stress overload, drain and stalled clients; a lexical thread-spawn finding is not proof that this current path is uncapped. |
| P2 | `deploy/dl-flywheel/pipelines/rlfw/analytics/`, `docs/internal/`, `crates/jope-engine/assets/` | Generated HTML/parquet and historical evidence make navigation and checkout size expensive. Several individual HTML files are close to the existing 5 MiB cap. Move future derived dashboards toward a manifest/index plus external evidence custody; migrate existing evidence only with its owner, source pins, references and restore verification. No historical evidence is deleted here. |
| P2 | `docs/architecture.md`, `docs/parity/`, `docs/runbooks/` | The architecture page explicitly describes dated target design. Keep operational truth in source-bound receipts and capability records; add a current concise runtime map in a dedicated follow-up rather than treating old diagrams as qualification. |
| P2 | `.github/workflows/jankurai.yml`, `deploy/ci/install-jankurai.sh` | The integrated audit update pins 1.7.5 in both the installer and workflow and uses the same full-scope local entry point. The existing opt-in hosted runner condition remains; a skipped workflow is not a passing audit. |

The changes above remove one duplicate reader and keep runtime code in the
existing dataset/model crates. A wholesale directory rename would churn
source-bound identities and evidence locators without proving a performance gain.

## Verification

The new tests compare binary content hashes against both the independent
in-memory SHA-256 calculation and the CSV digest at empty, one-byte and chunk
boundary sizes. A multi-chunk first line proves the header bound does not truncate
column counts or hashing. A controlled blocking job on a current-thread runtime
proves executor progress and retained admission after cancellation. Existing blob,
resume, concurrency and immutable model-publication tests cover the receiving paths.

The companion [verification and audit record](blob-verification-review.json) records the completed checks and
Jankurai results, including execution limits. Full audit caps and unresolved
findings remain visible; this focused change does not claim a clean repository.

## Recorded result

All **93 tests** across the affected crates pass on Rust 1.97.1, with no ignored tests. Workspace formatting, whitespace, Python static inventory and the shipped text check pass. `just required` cannot run because `just` is absent. Clippy was blocked by exhausted filesystem capacity before checking; it is not a pass. These are dated results for the original source hashes in the JSON record. The integrated successor requires new exact-head hosted CI and substantive same-agent review under the current owner policy.

The full Jankurai 1.7.5 baseline is **51/100, 591 findings (502 hard, 89 soft), 21 caps**. The repeated six-file scope is **54/100 before and after**, with 46 findings and eight caps in each report. Both audit gates fail. The JSON record retains the fingerprint comparison, source hashes and a next action for every cap. A complete full-repository post-change rescan was not run. The unchanged blob-cache refusal is flagged for the word `stale`; its fingerprint is identical in the baseline. No exclusion or threshold was relaxed.

An optimized seven-pair microbenchmark of a warm 16 MiB binary fixture measured a median 41.46 ms for the original CSV-aware reader and 14.24 ms for content-only hashing, about **2.91x** faster. Full hashes and byte counts match. Runs were alternated on a contended host; this measures the component, not whole training or search throughput. Model publication still has the separate snapshot-memory limit described above.

The soft shape summary changes from 705 to 690 lines in `store.rs` and from 77 to 108 heuristic Rust advisories. Those signals are retained rather than treated as closed findings. The PR preserves the newer main's unrelated AutoML integration; the six affected source files match the tested source hashes. Hosted CI must qualify the combined head.
