# JopeDime HPC bottlenecks and timing evidence

Measured2026-10-09 UTC. The source receipt is `DL_FLYWHEEL_HPC_BOTTLENECK_PROFILE.json`, SHA256 `89762b2d867ee0acd67bd6527a8046a104bcafe1e832b7f34bb1804164742894`. These are observations from real completed work, not throughput promises.

## Ranked findings

1. **A queue rule stopped augmentation.** The original buffer held128 variants until training admission. Encoding and S3 completion did not release capacity. Independent generation/publication/encoding queues have replaced it; S3-published variants rose from138 to938 during the first eight minutes of the replacement. This is raw-input publication, not938 newly trained contexts.
2. **The model optimizer was only3.44% of the full-pass elapsed time.** All6,857 training contexts and263 validation contexts completed. Most time was input preparation, autoregressive proposal decoding, and other work. The model fits easily in GPU memory; more allocated memory alone does not remove those waits.
3. **GP workers spend93.62% of their active elapsed windows outside the five execution commands.** This is the strongest CPU-pipeline bottleneck signal. Existing logs do not yet separate all storage verification, source hashing, canonicalization and scheduling gaps. Those are hypotheses to measure individually, not an established93.62% S3 attribution.
4. **The feedback consumer was incomplete.** The full pass emitted858 durable proposals that had not yet been executed. The original-context adapter and inference-first path are being connected. Queued proposals never count as scored outcomes or rewards.
5. **The encoder qualification computes four encodings per new variant.** Repeatability, protected-input control and target-permutation tests explain the extra calls. They are not four model inputs. The first dedicated-GPU packet retains those checks and times each; the production version should compute once and use a qualified model/worker release plus sampled audits.

## Full training pass

| Stage | Wall seconds | Share |
|---|---:|---:|
| Input loading/preparation |309.41|36.65%|
| Model proposal decoding |252.64|29.92%|
| Optimizer stage |29.03|3.44%|
| Other work |253.22|29.99%|
| Total |844.30|100%|

There were429 optimizer updates. Optimizer-stage throughput was14.78 updates/s; end-to-end throughput was0.508 updates/s. These are stage-wall measurements, not a direct CUDA execution-time total. Other work includes validation, checkpoints, role checks and orchestration; no unmeasured subdivision is asserted. Maximum reported CUDA allocation was534,312,448 bytes, reserved679,477,248 bytes.

## Eighty measured GP jobs

The sum of each worker's first-command-to-last-command elapsed window was1,262.29 seconds. The commands used80.51 seconds in total. Worker windows overlap in real time, so their sum is a worker-time accounting denominator, not the wall duration of the entire parallel campaign.

| Command | Mean seconds | Median seconds | p95 seconds |
|---|---:|---:|---:|
| Native two-fold evaluation |0.1250|0.0544|0.5993|
| Rust code generation |0.2264|0.0837|1.2254|
| Rust compilation |0.5569|0.4409|0.7709|
| Emitted Rust fold0 execution |0.0508|0.0191|0.2409|
| Emitted Rust fold1 execution |0.0472|0.0169|0.2190|

Compilation caching is useful, but compilation alone is only44.55 seconds of those1,262.29 worker seconds. Removing it cannot explain a dramatic throughput gain by itself. The larger opportunity lies in the gaps around the commands.

## Immediate optimization tests

### A. Reuse verified immutable S3 reads

New source packet: `work/flywheel-20261008/proposals/hpc-cache-v1/verified_cache.py`.

Cache only after the first exactVersionId GET passes size and SHA256 checks. Key by bucket/key/VersionId/SHA256/length, use a bounded128MiB LRU, and verify retained bytes on hits. Do not cache missing objects, authentication failures or incomplete reads. Preserve create-only publication and the first real S3 readback. Existing code repeatedly verifies identical artifacts; immutable, already verified bytes can be reused within the worker.

Compare the same fixed contexts/programs and evaluator with cold and warm cache. Report GET/HEAD/PUT counts, transferred bytes, verified-cache hit rate, source-hash time, canonicalization time, CPU user/system time, end-to-end latency and identical scores/artifact identities. A faster mock test is not a measured cluster speedup.

### B. Memoize pure native canonicalization

Cache successfully validated canonical output by exact input bytes and pinned canonicalizer identity. Bound memory, preserve source checks and fail on changed content. Measure subprocess launches avoided and source-hash cost separately. This is a new runtime candidate; it does not mutate a currently running worker.

### C. Prepare training inputs ahead of the GPU

Profile parsing, shard reads, token/mask construction and transfers separately. Cache immutable prepared tensors and bounded shard data, prefetch the next batches, and use ragged/token-budget batches rather than materializing all dense grammar masks. Keep the existing frozen membership, parent-stratification, train-only normalization and exact checkpoint/batch journal.

### D. Batch model proposal decoding

Decode several contexts together and cache transformer state if the existing architecture supports it correctly. Measure actual CUDA-event forward intervals, total decoding wall time and native-valid output rate. Compare the same model/checkpoint and candidate restrictions. Do not silently replace model proposals with random fixed programs to improve throughput numbers.

### E. Separate durable journals from full checkpoints

The current run saved a full physical checkpoint each update. A future tested runtime can keep per-batch durable identities and optimizer progress while taking full model/optimizer/RNG snapshots at explicit intervals. Qualification must demonstrate deterministic recovery and state-loss bounds. No checkpoint policy is changed in the completed run or its paper evidence.

## Logging now and its limits

A dedicated read-only sampler runs on xbabe1, xbabe2 and xbabe3 every two seconds. Each host has a25%-of-one-core CPU quota,256MiB memory cap, no swap and low scheduling priority. It logs:

- UTC Unix nanoseconds and formatted UTC timestamps, monotonic nanoseconds, observed clock synchronization and clock resolution;
- host CPU ticks/load, available memory, CPU/memory/I/O pressure and network byte counters;
- each active Jope/Grok service's cumulative cgroup CPU user/system/total time, throttling, memory/peak/swap, page faults and I/O counters;
- sampled GPU utilization, memory usage, clock speed, temperature and power, with query duration;
- actual augmentation creation/publication timestamps and their elapsed difference from the existing SQLite journal;
- observer cost, hash-chained records and sealed per-minute log segments.

Host samples are **not** per-kernel GPU execution durations. New inference/encoder workers additionally record per-stage spans, CPU/resource deltas and CUDA-event intervals. Internal transfers that cannot be independently measured are marked unavailable, not relabeled wall time. UTC precision is not a guarantee of cross-host clock accuracy. Completed historical work retains its original timestamp precision; missing measurements are not fabricated.

Logs live on each host under `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/hpc_trace_20261009_v1/logs/`. Source: `work/flywheel-hpc-20261009/host_trace.py`, SHA256 `b181c4881521ca1630ca29ccbad69ce3a23470940424708dd56e5807375c31ed`. The current bounded collection window ends at the original growth-controller deadline. Sealed segments can be archived without copying an actively written file.

## Acceptance criteria

Promote a throughput change only when the exact evaluator/model/data split is preserved, real measured results match, failure/custody behavior remains correct, and wall throughput improves on a representative fixed packet. Report cold versus warm operation, p50/p95 latency, useful results per CPU-second, peak memory and network traffic. Count actual optimizer exposure and actual published measurements separately from queued work. Keep at least one baseline measurement for comparison.


## Verified caching experiment and live logging update

At 2026-10-09 04:44:31 UTC, a read-only matched test verified the same two actual native GP results with identical reconstructed measurements. Repeated uncached verification took 2.644 seconds and 18 S3 GETs; warm verified caching took 0.149 seconds and zero GETs (17.7 times faster for this verification stage). The cold cached pass took 3.185 seconds and 16 GETs, so this is not evidence that first-time work is faster. Each cached key includes bucket, key, immutable VersionId, byte count and SHA256; every hit rechecks bytes and deadlines. No absent/error result is cached. The canonicalizer memo is keyed by pinned binary and exact input, with source rehash on hits. Receipt: `DL_FLYWHEEL_HPC_CACHE_MATCHED_AB.json`.

Host telemetry now archives sealed segments every minute on xbabe1, xbabe2 and xbabe3. Each segment has a continuous event hash chain, exact S3 VersionId and readback. First verification observed 72 S3 objects across 36 closed segments. Logs use the `paper-evidence/20261009/hpc-trace-v1/` prefix under the existing training experiments namespace. Current open segments stay local until sealed. These logs contain cgroup CPU user/system/throttled counters, memory and I/O pressure, GPU samples, and publication latency, not raw source datasets or environment credentials.

The original full-run outbox consumer qualified two actual model proposals: 2 executions, 2 Rust parity passes, 16 create-only S3 PUTs and 2 verified result records, zero failures. Those consume the first two of the 858 queued proposals; they do not imply the entire queue is drained. Exact native build provenance is tied to source pin `6bf65b12e9f0e5ec2697720a6119f13ea27d4835` by the pre-existing S1 control receipt `714eb90e71a4253fb94f6605711709f0c2e1eb09146ba06e86da34d4c86d4014`.

The next scored-data dispatch uses eight separate processes (one CPU core and 1 GiB each), rather than sharing a mutable timing context among threads. Its fixed packet budget is 64 emissions per worker; statuses distinguish claimed, executed, published, cached, and failed attempts. A shared durable SQLite claim prevents two workers from owning one emission. Timing is per process and per emission, including CPU user/system and child user/system time.
