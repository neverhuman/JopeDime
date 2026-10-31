# CPU execution and S3 custody — t2101 / overnight15

The owner’s `rd_cpu_s3_t2101.md` and `overnight15_t2125.md` supersede the original four-worker limit in the append-only FINDINGS rules. They apply to W1 preparation, W2, W8 and later CPU fits. Scientific data closure and per-wave total budgets in [PROPOSALS](PROPOSALS.md) remain in force. H62 packages W2; H63 packages W8. OPUS coordinates execution; ASTRA coordinates result custody/publication. Claim each shard once before launching it.

## Host limits

| Host | Worker ceiling | Additional condition |
|---|---:|---|
| xbabe1 | 50 | nice 19; preserve at least 150 GB free; never touch the live flywheel or its processes |
| xbabe2 | 45 | nice 15; prefer remote CPU while loaded; no new large target directories |
| xbabe3 | 70 | nice 15; only small local scratch on the nearly full disk; promptly upload outputs |
| xbabe4 | 20 | Linux Docker only, Colima profile `jope-ci`; nice 15 inside the container; bound threads as well as workers |

These are ceilings, not reservations. Check current utilization, load, effective CPU allowance and free disk before every launch; back off above 85% host CPU. Account for other workloads and BLAS/OpenMP threads. Sample during execution and stop admitting new work when capacity shrinks; never pause or kill other owners’ jobs. Use all four eligible nodes through deterministic family-fold or seed shards, recording unavailable or saturated nodes and reassignment without duplicate execution. Never use xbabe0. No GPU is needed for these diagnostics.

On xbabe4, the observed Docker executable is `/usr/local/bin/docker`, and the context is `colima-jope-ci`; a noninteractive SSH PATH may omit it. Verify the daemon reports Linux before launching. Mac host capacity and the Linux VM allocation both constrain concurrency. Reuse an existing suitable image; do not claim an image has the required evaluator merely because it exists.

Freeze a shard ledger with run ID, wave, shard ID, family/fold or seed assignment, input hashes, host, worker/thread count, command and reserved CPU/fit budget. Count pilot, failed and resumed work once in the wave’s cumulative ledger. W8’s 2,400-fit/four-CPU-hour cap applies across hosts, not per host. No retries or missing-shard replacement may reset the budget. Keep real S3-derived inputs and original labels; DOPE stays frozen at `c51a56d0`.

## Immutable run custody

Use a fresh run ID under:

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/<wave>/<run_id>/`

Every run has a `manifest.json` naming source Git SHA, exact commands, input S3 keys/version IDs/SHA256s, local mirror identities where applicable, seeds, folds, hosts, limits and actual consumption, output object hashes, metrics and verdict. Report Elo, `val_reg_inner` and U4 by their declared tracks; use null plus an explanation for unmeasured fields. Preparation-only manifests explicitly say `preparation_only` and cannot support a scientific verdict. An output prefix is not permission to read new input datasets.

Upload each object using create-only semantics (`If-None-Match: *`). Do not overwrite existing run keys. Verify downloaded object bytes against the local SHA256; ETag or a self-declared metadata hash alone is insufficient. Record version IDs and verification receipts. Publish the manifest after its listed objects verify. Preserve local bulk outputs until verification succeeds, then remove only this run’s verified, no-longer-consumed bulk files. Keep small manifests/receipts locally. No broad cache or foreign-worktree cleanup belongs to this lane.

Potential HWM candidates also require an append-only pointer in `HWM_CANDIDATES.jsonl` under the campaign prefix. Do not truncate/recreate that shared index or silently overwrite another writer’s update. Coordinate a serialized, conditional append that preserves every earlier record; a separate run manifest alone does not claim the index was updated. No candidate pointer is warranted for preparation or an unmeasured hypothesis.

Never write sealed or holdout prefixes. Keep credentials out of commands, manifests and container mounts; use the host’s existing credential provider for custody. After verification, append the exact manifest key, SHA256, verdict and remaining limitations to [FINDINGS section 7](FINDINGS.md#7-log-append-only). A ready result PR contains small summaries and immutable evidence pointers, not bulk outputs or lock files.
