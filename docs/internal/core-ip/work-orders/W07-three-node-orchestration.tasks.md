# W7 — Repository-managed execution on xbabe1, xbabe2 and xbabe3

Status: planned. Owner: native controller/SmartCluster and native distributed trainer.
Requires W1 closures, W5 bounded jobs and W6 update/checkpoint semantics. Follow the
[execution contract](EXECUTION-CONTRACT.md). Independent node reachability, same-host
collectives and a configured workflow are not multi-host numerical qualification.

## Architecture and source anchors

Use xbabe2 as coordinator and admitted CPU/GPU workers on all three hosts. All numerical
work flows through `jope-native-loop`, the native graph/provisioner, SmartCluster workers
and the native trainer. SSH is inspection/reviewed installation only, never the scheduler.

Inspect `crates/jope-engine/src/native_loop.rs`, `native_loop/run_approval.rs`,
`bin/jope-native-loop.rs`, `native_graph.rs`, `native_graph/transport.rs`,
`native_provisioner.rs`, `native_provisioner/service.rs`, `smartcluster_worker.rs`,
`training.rs`, `native_persistence.rs`, and `crates/jope-engine/native/runtime.cpp`.
At the baseline the controller CLI exposes Bind/BindParentDecision/Tick/Serve; proposed
Plan/Status/Verify/Cancel operations are work to implement, not existing commands.
Native runtime currently calls `cudaSetDevice(rank)`; separate local device selection
from communicator rank. Preserve the historical two-by-3000-update contract.

## Ordered tasks

### W7-T01 — Inventory hosts, runtime and reservations

- [ ] Refresh host/boot identities, address/certificate identity, GPU UUID/model/driver,
  available VRAM, CUDA/NCCL/compiler/runtime hashes, source/executable and installed policy.
  Planning observations were RTX4090 on xbabe1 and RTX3090 on xbabe2/3, approximately
  24 GiB each; verify actual inventory and availability at admission.
- [ ] Read active CI/user allocations and device ownership. Inventory repository-managed
  worker installation and credential boundaries; do not interrupt unrelated consumers.
- [ ] Acceptance: preflight refuses stale telemetry, mismatched boot/GPU/source bytes,
  conflicting ownership or unavailable capacity. Retain a current three-host census.

### W7-T02 — Add a versioned bounded-run contract and controller operations

Depends on T01 and W5/W6.

- [ ] Define a new schema alongside the old fixed run contract. Freeze participants,
  global/local ranks, device UUIDs, artifact closures, original deadline, CPU/RAM/GPU
  ceilings, work census, participant grants and result destinations.
- [ ] Implement reviewable Plan, read-only Status/Verify and idempotent Cancel operations
  in the existing controller. Bind execution to the planned artifact, not mutable flags.
- [ ] Acceptance: old manifests retain old semantics; invalid/underfunded plans fail
  before dispatch. Status distinguishes scheduled, admitted, running, committed, failed,
  drained and unknown; verification never starts numerical work.

### W7-T03 — Separate communicator identity from local CUDA device

Depends on T02.

- [ ] Pass global rank/world size separately from local rank/device index and GPU UUID.
  Validate local CUDA visibility against the frozen host/device assignment before NCCL init.
- [ ] Bind rank uniqueness, one owned device per worker and exact participant count.
  Do not assume global rank 2 implies CUDA index 2 on a one-GPU host.
- [ ] Acceptance: one GPU on each of three hosts maps correctly; invalid local index,
  duplicate rank/UUID and mismatched visibility fail before communicator creation.

### W7-T04 — Replace shared-filesystem rendezvous assumptions

Depends on T02/T03.

- [ ] Use authenticated participant communication and immutable verified artifact
  exchange through repository components. Bind rendezvous to run/attempt, participant
  set, grants and original deadline; reject replay from another attempt or boot.
- [ ] Admit the full synchronous group before any rank starts an optimizer attempt.
  Define message size limits, timeouts, delivery IDs and result authentication.
- [ ] Acceptance: missing participant, substituted source/input, forged peer, expired
  grant or stale rendezvous produces group refusal with zero committed updates.

### W7-T05 — Implement whole-group drain and coordinator recovery

Depends on T04.

- [ ] Persist dispatch/ownership state before acknowledging admission. On rank failure,
  lost control, grant expiry, cancellation or coordinator restart, reconcile all owned
  jobs and drain the whole synchronous group through its repository supervisor.
- [ ] Preserve original run/attempt deadlines and candidate/update counters on restart.
  Distinguish unacknowledged completion from never-started work; do not duplicate commits.
- [ ] Acceptance: kill one rank, sever coordinator control and restart the controller.
  Every participant has a terminal/reconciled disposition and no orphan worker remains.

### W7-T06 — Define the lossless three-rank numerical partition

Depends on T03 and W6 objective/consumption contract.

- [ ] Introduce a new partition version. Keep complete preference groups together,
  deterministically assign global groups and distribute remainders without dropping them.
  Specify empty-rank behavior and globally empty terms explicitly.
- [ ] Reduce each term using its global numerator and denominator, not an average of
  unequal local means. Specify gradient scaling/accumulation consistent with that reduction.
  Persist rank-local RNG/sampler state and the global committed update.
- [ ] Acceptance: uneven group sizes, nondivisible batches, masks and term-specific
  denominators match a declared single-process reference objective/update within a frozen
  numerical policy. Changing rank count requires an explicit transition and requalification.

### W7-T07 — Execute managed three-host CPU lifecycle proof

Depends on T02/T04/T05; phase admission required.

- [ ] Run one admitted CPU job on each host with verified inputs/results, then exercise
  cancellation and coordinator restart. Use the proposed 4 CPU/8 GiB per host, 15-minute
  phase only after admission; retain CI reservation compatibility.
- [ ] Acceptance: three distinct host/boot identities have actual execution receipts,
  successful valid results, refused invalid inputs and complete job/drain inventories.
  A same-host three-process test is a fixture, not this proof.

### W7-T08 — Qualify individual GPUs through managed jobs

Depends on T03/T05/T07 and W6; phase admission required.

- [ ] On each actual UUID, run admitted encoding/GP work and a small native training
  block. Prove nonzero optimizer updates, changed active parameters and complete checkpoint reload.
- [ ] Bound each host to one GPU, 8 CPUs/16 GiB and 15 minutes as proposed; retain device
  memory peaks, runtime hashes, updates and terminal failures. Verify all three profiles separately.
- [ ] Acceptance: each host executes both CPU and GPU work via repository components;
  no SSH-launched numerical process or CPU fallback is counted as GPU qualification.

### W7-T09 — Qualify two-host then heterogeneous three-host training

Depends on T06/T08; each phase separately admitted.

- [ ] First run xbabe2+xbabe3, then include xbabe1. Use the proposed 20-minute window
  per phase with the same per-host ceilings and original-deadline enforcement.
- [ ] Compare real updates with the declared reference, verify rank agreement, save
  and reload checkpoints, interrupt/resume on the same topology, and inject rank/control failure.
- [ ] Acceptance: no remainder loss, incorrect denominator weighting, silent topology
  transition, duplicated committed update or renewed deadline. Retain actual parameter
  deltas, global/rank counters, peer receipts and final no-orphan census.

## Verification, slices and handoff

```sh
cargo test --locked -p jope-parallel --lib native_loop -- --list
cargo test --locked -p jope-parallel --lib native_graph -- --list
cargo test --locked -p jope-parallel --lib native_provisioner -- --list
```

Proposed slices: (A) inventory/controller schema T01–T02; (B) device/transport T03–T04;
(C) durable drain/recovery T05; (D) partition/objective T06; (E) separately admitted
CPU, individual-GPU and distributed qualifications T07–T09. Freeze actual new CLI argv
and grant manifests before live dispatch. Do not prefill them with invented signatures.
Handoff: verified host-bound plan, controller journal, participant results, qualified
topology partition and restored checkpoint. W8/W10 consume these exact artifacts.

## Original finding routes

This package implements additional owner requirements; no original shard entry has it as its sole primary route.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
