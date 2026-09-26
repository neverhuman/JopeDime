# W7 trainer topology: implementation record

**Scope.** Trainer-side parts of W7-T03 (communicator identity versus local device) and W7-T06 (lossless multi-rank partition), in the lane `.agent/worktrees/bench`.

**Status.** Everything here is CPU source and fixtures, `implemented` / `fixture-proven` only.
- No GPU was used; every run had `CUDA_VISIBLE_DEVICES=""`.
- No NCCL communicator was created, and no multi-host or live run took place.
- `native/runtime.cpp` is compile-checked with the SDK at `/opt/jope-sdk`; it was not executed.

Evidence directories are under `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

---

## W7-T03 — separate communicator identity from the local CUDA device

### Defect (source trace)
- **Runtime.** `native/runtime.cpp` `jp_nccl_init(bytes, rank, world)` called `cudaSetDevice(rank)`.
- **Trainer.** It used `Device::Cuda(rank)` and `Cuda::synchronize(rank)`; the checkpoint barrier also used `Device::Cuda(rank)`.
- **Consequence.** On one GPU per host, global rank 2 selected CUDA index 2, which does not exist. On a multi-GPU host, rank order silently chose the device.
- **Not reproduced as a failing test.** The call is native and needs a device.

### Change
| Location | Change |
|---|---|
| `training/topology.rs` (new) | `Assignment` (`jope.trainer_assignment.v1`): `plan_sha256`, `world`, and per participant `global_rank`, `host_id`, `boot_id`, `local_rank`, `device_index`, `gpu_uuid`. `validate` requires the exact participant count, each global rank `0..world` once, each GPU UUID once (one owned device per worker), per-host local ranks exactly `0..n`, one boot per host, and well-formed UUIDs and plan digest. `device_index` is the ordinal as each worker sees devices, so two workers on one host restricted to their own GPUs may both use index 0; ownership is enforced by UUID. The first fixture run caught an earlier per-host index-uniqueness rule that contradicted this, and the rule was removed. `bind(assignment, global_rank, world, local)` also requires the same world, the assigned host and boot, the assigned index to be visible, the right UUID at that index, and no other participant's GPU visible. A `Binding` comes only from `bind` or the explicit, labelled `Binding::legacy_single_host(rank, world)` |
| `neural.rs` | `Nccl::init(&Binding, rendezvous)`: communicator rank = global rank, CUDA ordinal = the binding's device index. `local_devices()` reads host and boot identity plus every visible ordinal's UUID. The NCCL and training probes use the explicit legacy binding |
| `native/runtime.cpp` | `jp_nccl_init(bytes, rank, world, device, expected_uuid)` refuses an invisible index and a UUID mismatch before `cudaSetDevice(device)` and `ncclCommInitRank`. New `jp_device_count` and `jp_device_uuid` (nvidia-smi `GPU-8-4-4-4-12` form) |
| `training.rs` `rank_impl` | Optional frozen `assignment`: validated first, then bound against `local_devices()` before NCCL. Without one, the legacy binding is recorded. Every device use, sync and checkpoint barrier goes through the binding. `topology` is recorded in rank checkpoint receipts and `complete-{rank}.json` |

Every current caller passes no assignment, so behavior is unchanged and the receipt says `legacy_single_host`. The assignment arrives when the W7-T02 bounded-run contract is joined.

### Fields needed from the W7-T02 bounded-run contract
The trainer needs exactly these values per run. Grants, deadlines, ceilings and the work census stay in the plan.

| Field | Meaning |
|---|---|
| `plan_sha256` | digest of the frozen bounded-run plan |
| `world` | exact participant count |
| per participant `global_rank` | communicator rank, `0..world` |
| per participant `host_id` | stable host identity (`/etc/machine-id` on the worker) |
| per participant `boot_id` | boot identity (`/proc/sys/kernel/random/boot_id`) |
| per participant `local_rank` | rank among that host's participants, `0..n` |
| per participant `device_index` | CUDA ordinal as that worker sees devices, after its `CUDA_VISIBLE_DEVICES` |
| per participant `gpu_uuid` | nvidia-smi UUID of the owned device |

Delivery path to decide at integration: a `rank_impl` argument, or a frozen artifact bound by the plan. No mutable flag.

### Fixtures
File `crates/jope-engine/tests/trainer_topology.rs` (feature `neural`, CPU; no device is enumerated).

| Acceptance item | Test |
|---|---|
| One GPU on each of three hosts: ranks 0, 1, 2 bind CUDA index 0 with their own UUIDs; rank 2 is not index 2 | `one_gpu_on_each_of_three_hosts_maps_global_ranks_to_local_index_zero` |
| Two workers on one host bind by UUID, not rank order; a worker seeing both host GPUs is refused | `two_devices_on_one_host_bind_by_uuid_not_by_rank_order` |
| Refused before any communicator: duplicate or out-of-world rank, duplicate UUID, count/world mismatch, reused local rank, non-contiguous local ranks, one host under two boots, malformed UUID, missing host, unknown schema, malformed plan digest | `invalid_assignments_are_refused_before_any_communicator` |
| Mismatched visibility refused: no visible device, wrong device at the index, another rank's GPU visible, wrong host or boot, unassigned rank, world mismatch, invalid local index | `a_worker_whose_local_view_disagrees_is_refused` |
| The legacy mapping is explicit and labelled; the assignment round-trips and refuses unknown fields | `the_legacy_single_host_mapping_is_explicit_and_labelled` |

### Open for W7-T03
- **Delivery of the assignment** to `rank_impl` from the W7-T02 plan, and the controller spawning workers with the assigned `CUDA_VISIBLE_DEVICES`.
- **World sizes.** `rank_impl` still accepts only v1 world sizes (1, 2, 4, 8). A three-rank run needs the W7-T06 partition.
- **Other rank-as-device uses.** `proposals.rs` and `ranked_proposals.rs` still use `Device::Cuda(rank)`. They are not training and are recorded as follow-ups.
- **Not executed.** No native path ran; that is W7-T08.
