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
explicit CUDA device selection; and gives initialization/collectives an explicit
120-second timeout (callers can supply a finite positive timeout_seconds).

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
shutdown. Child process groups are killed on the test's 90-second timeout.
Both process-count cases are required by `deploy/ci/training-proof.py`.

The initial refusal reproduction had 12 failures and one pass before the fix;
120 local training tests passed after the fix, including both real process groups;
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
