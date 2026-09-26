# Distributed training readiness — 25 September 2026

The storage publication prerequisite landed in PR99. Local synthetic CPU training
subsequently passed 97 tests, with direct regression, binary and multiclass CLI
updates and checkpoint reloads. This follow-up addresses Python distributed startup.

## Behavior changed

Previously, absent RANK/LOCAL_RANK defaulted to zero even with multiple participants,
and nonpositive WORLD_SIZE silently selected single-process training. Missing or
invalid rendezvous information could reach the collective initializer. The helper
now validates those values before joining; verifies rank/world/backend against an
existing process group; refuses unavailable local CUDA indexes and conflicting
explicit CUDA device selection; and gives newly created groups an explicit initialization/collective
120-second timeout (callers can supply a finite positive timeout_seconds).

Matching preinitialized groups retain their existing timeout.
This timeout is a process-group communication limit, not a resource grant, overall
training deadline, or multi-host cancellation controller. Process-local validation
cannot prove host identity, unique ranks across machines, consistent source/input
bytes, or eligibility to use a GPU.

## Acceptance

Run `python3 -m pytest -q python/tests/test_policy_training_contract.py` from the
repository with its Python dependencies installed. The maintained suite launches
real two- and three-process Gloo groups on one host. It checks initial parameter
broadcast, an actual optimizer update against a serial gradient reference, unused
parameter reduction, scalar aggregation, uneven disjoint shard coverage and clean
shutdown. The harness directly owns each rank process and supplies the env:// variables,
without a torchrun supervisor that detaches workers. Cleanup kills each owned
session on completion, failure or interruption, with bounded output draining.
Forced-timeout cases check cleanup with both living and exited session leaders.
Both process-count cases are required by `deploy/ci/training-proof.py`.

The initial refusal reproduction had 12 failures and one pass before the fix;
The initial 120-test local training run passed; after review, all 49 policy
training tests passed, including real process groups and both cleanup cases;
40 required training proofs and six proof-gate unit tests passed. Hosted results
belong in the PR's exact-head evidence. Custody is
`/home/ubuntu/jope-rescue-20260921/training-distributed-20260925/`.

## Host observation and next qualification

Read-only SSH and nvidia-smi probes succeeded for xbabe1 and xbabe3. Both reported
PyTorch 2.7.0 with CUDA 12.8. xbabe1 has RTX4090 and driver570.153.02;
xbabe3 has RTX3090 and driver580.126.09. Neither reported active GPU utilization
at the probe. Local xbabe2 was already verified for CPU training; its SSH alias
points elsewhere and must not be used as its identity. These are observations,
not reservations or NCCL qualification.

Remaining work, in order:

1. Bind a dry-run topology to each host's identity, source/runtime receipt, GPU UUID,
   admitted input/shard hashes and original grant/deadline. Refuse duplicate ranks,
   duplicate devices, missing shards, sealed inputs and expired grants globally.
2. Extend native admission and cancellation to every participant. Verify that a
   failed rank drains all participants without resetting the original deadline.
3. Qualify one GPU on xbabe2, then the two 3090 hosts, with synthetic inputs under
   those grants. Check NCCL connectivity and identical parameter hashes after a
   real optimizer update; retain failure and per-rank execution receipts.
4. Qualify heterogeneous three-rank sampling and reduction. The existing trainer
   truncates grouped-batch shards to equal lengths, dropping up to world_size-1
   batches per epoch. The helper's coverage test does not fix or qualify that
   trainer policy, weighted losses or distributed exact checkpoint recovery.
5. Provision scoped dataset-reader and append-only learning-writer roles through
   the credential owner. Keep xbabe2 as sole publisher initially. The existing
   host admin provider is not such a role; no production S3 write was used here.
   Conditional PUT is implemented and fixture-tested, but a live S3 race remains
   separate operational evidence.

The experimental campaign still explicitly selects WORLD_SIZE=1 and one RTX3090.
This patch does not turn that controller into a multi-host launcher or grant it
resources. It establishes validated startup and real local collective correctness
as prerequisites for that later integration.

Read-only AWS metadata refreshed during this follow-up: the selected profile resolves
to user jepson in group admin with the attached AdministratorAccess policy. Learning
bucket versioning is Enabled. That confirms access and versioning, not restricted
writer permissions, conditional-write enforcement, replication recovery or safe
production multi-writer operation. Metadata receipts are retained in custody.

## 26 September refresh: native bounded-run preflight

Main `e4f1b6c625299388478b8c876f54935f7b8b171b` includes the subsequent W7
bounded-run contracts, census validation, trainer assignment, authenticated
rendezvous, group drain/recovery fixtures and lossless partition v2 with checkpoint
state. These supersede the earlier source-gap list above, but remain fixture or
compile evidence; they do not establish live multi-host training.

The native controller now exposes a read-only operator check:

```sh
jope-native-loop preflight --run-dir /durable/run \
  --plan-sha256 FULL_PLAN_SHA256 --census /durable/host-census.json
```

It verifies the pinned frozen plan and journal, original grant/deadline window,
and typed host census. It refuses active, dispatched, cancelled or terminal runs,
unsafe or oversized census files, and the existing census identity, freshness,
ownership and resource mismatches. Success prints the native preflight receipt,
including the exact census digest and `allocation_held: false`. It creates no
files, contacts no host, and records no admission. The coordinator must obtain
fresh telemetry and journal its own preflight before dispatch. A concurrent start
can invalidate this observation immediately; this command is not a reservation.

Acceptance: `cargo test --locked -p jope-parallel --test bounded_preflight_cli
--test bounded_run --test bounded_run_recovery` (set `TMPDIR` to durable fixture
storage outside `/tmp`). The new CLI fixtures check positive three-host matching,
unchanged directory contents, no dispatch authority, pinned-plan and stale-host
refusal, cancellation, expiry, symlinks and bounded reads.

Read-only host probes on 26 September reached xbabe1 and xbabe3 by SSH and xbabe2
locally. All returned machine/boot identities and GPU UUIDs. The default PATH did
not resolve `jope-native-loop` or `jope-smartcluster-worker` on any host; that does
not establish absence from service-specific installations. xbabe2 has active native
observer and qualification services, and xbabe3 has a separate research runner.
These observations are not a complete native census: installed certificates,
executable/runtime/policy hashes and reservation ownership still need authoritative
collection. No grant was issued and no GPU workload was launched by this check.

Remaining integration: authoritative census collection; SmartCluster-backed group
supervisor and authenticated cross-host transport; delivery of the frozen assignment
and admitted communicator bootstrap; controller selection of partition v2 and
shared checkpoint/result transport. Keep the staged CPU, individual-GPU, two-host
and three-host qualification gates separate from these source changes.
