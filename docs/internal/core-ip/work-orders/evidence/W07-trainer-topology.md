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
- **Other rank-as-device uses.** `proposals.rs` and `ranked_proposals.rs` used `Device::Cuda(rank)`. Resolved in the follow-up section at the end of this record.
- **Not executed.** No native path ran; that is W7-T08.

---

## W7-T06 — lossless multi-rank partition (partition v2)

### Defect (reproduced first)
Test `partition_v1_cannot_form_three_ranks_and_averages_unequal_local_means`, run before any source change (`w6-pre-w7t06-20260926T101144Z`, 1/1 passed, which records the defect):
- **No three-rank partition.** `task_contract::training_partition(3, micro, 2048)` refuses every microbatch (64, 128, 256). v1 needs a world in {1, 2, 4, 8} and `global_batch % (world × microbatch) == 0`. The v1 sampler cursor `consumption::sampler_serial` also needs `global_batch % world == 0`.
- **Unequal shards change the objective.** v1 averages microbatch-local means over microbatches and ranks. For 7 comparisons on three ranks (3, 2, 2), that average differs from the single-process objective by more than 1e-3.

v1 cannot be repaired without changing what existing checkpoints mean, so v2 is a new capability and a new partition version. v1 is unchanged.

### Change
| Location | Change |
|---|---|
| `training/partition.rs` (new) | `Plan::new(world, global_batch, microbatch_rows)` with any positive world. `assign`: contiguous blocks of whole comparisons, `⌊G/W⌋ + [r < G mod W]` per rank, remainder to the lowest ranks, microbatches of at most `microbatch_rows / 8` comparisons, nothing dropped. `record()` is the `parallel.training_partition.v2` execution-partition record: shards, normalization `global_term_sums_v1`, empty-rank and empty-term rules, RNG and sampler identities, and the frozen numerical policy. `slots(rank, seed, local_step)` gives each microbatch's global sampler cursor and its dropout seed (`consumption::dropout_seed`, unchanged `mix_seed_local_step_rank_microbatch_v1`). `denominators(rows)` gives the global denominators from labels. `composite(…, &global)` is the v2 microbatch objective and its contribution to each global term. `empty_rank_objective` is an exact zero connected to every parameter. `Transport` and `local_scale` set the gradient scale. `CommittedUpdate`, `RankState` and `State` (`parallel.partition_state.v1`) are the persisted state, with `verify` and `resume`. `objective_identity()` is the v2 objective identity |
| `training/objective.rs` | `terms()` is extracted from `composite`, so v1 and v2 run the same term arithmetic. `composite` performs the same operations, and `identity()` and its digest are unchanged |
| `training/continuation.rs` | `check_parent_partition(mode, state, config, current, partition, policy)` takes an explicit v1 or v2 record, which must name the requested world and global batch. `check_parent` computes the v1 record and delegates, so its behavior is unchanged |
| `training.rs` | `pub mod partition;` |

### Reduction
`L = Σ_T c_T · N_T / D_T` over the whole global batch. A microbatch contributes `c_T · (N_T,mb / D_T,mb) · (D_T,mb / D_T)`: the v1 local mean, rescaled by the microbatch's share of the global denominator. Each rank sums its microbatches without dividing by their count, and rank gradients are summed.

| Term | Global denominator `D_T` | Globally empty when |
|---|---|---|
| `sequence_nll` | `Σ (0.05 + p⁴)` over rows (f32 weights, f64 sum) | never; every weight is at least 0.05 |
| `ranking_bt` | `Σ |Δv| · boost` over non-tied pairs within comparisons | every pair ties |
| `ranking_listwise` | comparisons (all validated comparisons are eligible) | never |
| `runtime_huber` | rows | never |
| `contextfree_bt` | the `ranking_bt` pair weight | every pair ties |

- **Local denominator above the global one.** This means the global denominators do not cover the microbatch. It is refused, and so is pair weight in a microbatch whose global pair weight is zero.
- **Globally empty term.** Value 0 and gradient 0, listed in `empty_terms`.
- **Empty rank** (more ranks than comparisons). No microbatch, no sampler cursor, and a defined zero gradient from `empty_rank_objective`. It joins every collective.
- **Gradient scale.** With the trainer's mean all-reduce (`Nccl::reduce` divides by the world size), each rank scales its objective and contributions by `W` (`local_scale(Transport::Mean)`). The reduction then delivers the sum.

### Frozen numerical policy
- **Reference.** The v1 `composite` over the whole global batch as one microbatch, in the same f32 arithmetic.
- **Loss and term tolerance.** `|L − L_ref| ≤ 1e-5 · max(1, |L_ref|)`, for the loss and for every global term value.
- **Gradient tolerance.** For each output tensor, the largest absolute gradient error must be at most `1e-5 · max|g_ref|`. An exactly zero reference gradient must therefore be reproduced exactly.
- **Dropout off.** Equality is claimed only with dropout off. Dropout seeds are rank-local, so a different rank count draws different masks.

**Measured** in `w6-post-w7t06-a-20260926T102534Z` (`--no-capture`, one line per configuration in `nextest.log`). The run covered 28 partitioned configurations, 4 all-ties configurations and the scaled mean transport.
- **Loss.** Largest error 1.49e-7.
- **Output gradients.** Largest error 2.24e-7 of the largest reference gradient.
- **Single microbatch, one rank.** Exact: error 0.
- **Margin.** About 45 times inside the policy.

### Persisted state (`parallel.partition_state.v1`)
- **Partition identity.** The partition record and its digest, the seed, the sampling origin and the committed step.
- **Per rank.** The RNG identity, the next local step, the sampler cursor of the next microbatch (`null` for an empty rank), and every next slot with its cursor and dropout seed. The RNG is counter based and reset before every microbatch, so these values are the complete RNG and sampler state.
- **The global committed update.** Optimizer and local step, comparisons, global denominators, empty terms, global term values and loss, gradient norm, clipping, the digest of the applied reduced gradient, and the parameter digest.
- **Verification.** `State::verify` re-derives every rank's state from the plan and checks the update's coverage, empty terms, loss arithmetic, clipping flag and step arithmetic. `resume` also requires the same seed.

### Rank-count changes (T06 continuation modes)
The objective does not depend on the rank count. The RNG streams and shard boundaries do, so a change in world or microbatch is a changed execution partition.

| Mode | Rank-count change |
|---|---|
| exact resume | refused by `check_parent_partition`, and by `State::resume` ("an explicit transition, never an exact resume") |
| new-corpus transition (optimizer carry) | allowed only when `declared_changes.execution_partition` equals the new v2 record; recorded `changed` |
| weight warm start | allowed; declares nothing |
| v1 → v2 | the reductions differ, so the objective identity differs: carry is refused and only a weight warm start can move a lineage |

### Fixtures
File `crates/jope-engine/tests/lossless_partition.rs` (feature `neural`, CPU). Simulated ranks each backpropagate their own microbatches into their own copy of the outputs, and the harness sums or averages the rank gradients.

| Acceptance item | Test |
|---|---|
| Reproduction: v1 cannot form three ranks, and averaging unequal local means changes the objective | `partition_v1_cannot_form_three_ranks_and_averages_unequal_local_means` |
| Lossless, deterministic, contiguous assignment of whole comparisons for 0–40 comparisons, worlds 1–9 and microbatches 1–5; the 2048-row, three-rank shards are 86/85/85 with microbatches 32/32/22 and 32/32/21; malformed plans refused | `assignment_is_lossless_deterministic_and_keeps_comparisons_whole` |
| Every partition matches the reference within the policy: 7 and 11 comparisons; worlds 1, 2, 3, 4, 5, 7, 8, 9, 12; one to all comparisons per microbatch; varying lengths, PAD and legal masks; ties, boosted pairs, and a rank whose first microbatches are all ties | `every_partition_matches_the_single_process_reference` |
| Mean transport with the world scale matches; without the scale it trains on `L / W` (negative control) | `a_mean_transport_needs_the_world_scale` |
| All ties: both BT terms are globally empty with value and gradient exactly 0; empty ranks have zero loss and exactly zero, defined gradients | `globally_empty_terms_are_zero_and_empty_ranks_contribute_zero_gradients` |
| Term-specific denominators (row weight share 1/7, pair weight share not 1/5); uncovered microbatch, pair weight against a globally tie-only batch, and incomplete comparisons refused | `term_denominators_are_global_and_must_cover_the_microbatch` |
| Worlds 1, 3, 4, 5 and 13 draw identical global record ids at every step; dropout seeds are the consumption-journal derivation and are distinct per rank and microbatch; the v1 cursor refuses an uneven shard | `every_rank_count_draws_the_same_global_batch_with_rank_local_rng` |
| Persisted state round-trips and resumes the next slots exactly; each of 15 altered fields is refused (cursor, seed, step, rank set, partition, digest, loss, term, clipping, empty terms, coverage, digest shape, schema, unknown field); a different world, microbatch or seed is not a resume; an empty rank persists an empty sampler state; a value for a globally empty term is refused | `persisted_rank_state_and_committed_update_resume_exactly` |
| Exact resume refuses 8→3 ranks, undeclared carry refused, a wrong declaration refused, a declared carry and a warm start accepted, the same three-rank partition resumes, v1→v2 carry refused as an objective change | `a_rank_count_change_is_an_explicit_transition` |
| The v2 objective identity differs from v1 in exactly six reduction fields; the v1 identity is unchanged | `the_v2_objective_differs_from_v1_only_in_its_reductions` |

### Runs
| Evidence directory | Command | Result |
|---|---|---|
| `w6-pre-w7t06-20260926T101144Z` | `--test lossless_partition`, reproduction only, before any source change | 1/1 passed (the defect is recorded) |
| `w6-post-w7t06-a-20260926T102534Z` | `--test lossless_partition --no-capture` | 10/10 passed, with measured errors |
| `w6-post-w7t06-b-20260926T102703Z` | neural suite: `--lib` training/neural/task_contract/controller/gpu_worker and 13 integration tests, including `objective_fixtures`, `continuation_modes`, `checkpoint_commit` and `trainer_topology` | 106/106 passed |
| `w6-post-w7t06-default-check-20260926T102756Z` | `cargo check --locked -p jope-parallel --all-targets` (default features) | exit 0 |

The v1 objective, identity digest and `check_parent` behavior are unchanged. `objective_fixtures` and `continuation_modes` pass unmodified.

### Open for W7-T06
- **The trainer still runs v1.** Wiring v2 into `rank_impl` touches the following, all outside this slice. None can run on this host without a GPU.
  - **Selection and sampling.** Select a `Plan` explicitly in the request, never by default. Sample each slot at its cursor. Compute the global denominators once per step from the global ids.
  - **Objective and reduction.** Call `partition::composite` scaled by `local_scale(Transport::Mean)`, with no division by accumulation. Backpropagate `empty_rank_objective` on an empty rank.
  - **Checkpoint.** Write `State` into the checkpoint manifest.
  - **Other v1 assumptions to generalize:**
    - `checkpoint_commit::verify_dir`, which recomputes the v1 record;
    - `training_exposure::checkpoint_baseline`, which assumes equal rows per rank;
    - the consumption-journal `Header`, which requires `accumulation × world × microbatch == global_batch`;
    - `validation_panel`, which allows only worlds 1, 2, 4, 8 and uses v1 validation averaging;
    - `checkpoint_recovery::verify_continuation`;
    - `rank_impl`'s own world and microbatch gate.
- **Launch dependencies.** A three-host run also needs the W7-T02 assignment delivery (W7-T03) and a W7-T04 rendezvous.
- **Load imbalance.** Uneven shards give rank 0 one extra comparison per step (86/85/85). Not measured.
- **Not executed.** No training step, device or communicator ran.

---

## W7-T03 follow-up — proposal workers select their device through a binding

### Defect (source trace)
- **`proposals::rank_impl`.** It used `Device::Cuda(rank)`.
- **`ranked_proposals::run`.** It used `Device::Cuda(rank)` unless `JOPE_RANKED_DEVICE=cpu`.
- **Consequence.** The same rank-as-ordinal assumption W7-T03 removed from training: global rank 2 on a one-GPU host selects a nonexistent CUDA index 2.
- **Not reproduced as a failing test.** Both paths create a CUDA device, so they cannot run here.

### Change
| Location | Change |
|---|---|
| `training/topology.rs` | `resolve(assignment, rank, world, local)`: `bind` against a frozen assignment, otherwise the labelled `legacy_single_host` mapping. The local device view is read only for an assignment, so the legacy path enumerates no device. `Binding::cuda_device()` is the owned device, and `Binding::owns(j)` is round-robin job sharding by communicator rank. `training::rank_impl` now uses `resolve` too, with the same behavior |
| `proposals.rs` | `rank_impl` takes an optional assignment; both callers pass none. It resolves the binding right after the topology check and before any input is read. The device is `binding.cuda_device()` and jobs go through `binding.owns(j)`. `complete-{rank}.json` records `topology` (the binding receipt, `legacy_single_host` today) |
| `ranked_proposals.rs` | `worker_device(binding, requested)`: CPU only when `JOPE_RANKED_DEVICE` is exactly `cpu`, otherwise the bound device. `run` shards views with `binding.owns(j)` and records `topology` in `parallel.ranked_proposals_complete.v1` |

Behavior without an assignment is unchanged: the same ordinal, the same shards, and the receipt now says so. The proposal completion that `native_graph` verifies checks only `rank`, `world` and equality with the numerical receipt, so the added field is compatible.

### Fixtures
File `crates/jope-engine/tests/trainer_topology.rs` (feature `neural`, CPU).

| Acceptance item | Test |
|---|---|
| Without an assignment, the legacy mapping is labelled and no device view is read; an out-of-world rank is refused. With an assignment, rank 2 on three one-GPU hosts gets CUDA 0; a failing or mismatched local view is refused, never replaced by the legacy mapping | `resolve_binds_an_assignment_or_labels_the_legacy_mapping` |
| Ten jobs on three ranks each have exactly one owner, by global rank. The ranked worker uses CUDA 0 on every one-GPU host, and CPU only for exactly `cpu` (not `CPU` or `cuda`). The legacy rank 5 of 8 keeps CUDA 5 and jobs 5 and 13 | `proposal_workers_shard_by_communicator_rank_and_use_the_bound_device` |

### Runs
| Evidence directory | Command | Result |
|---|---|---|
| `w6-post-w7t03fu-a-20260926T103159Z` | `trainer_topology`, `lossless_partition`, `gpu_entry_guards`, `ranked_proposals` (neural) | 18/19. The failure is `ranked_proposals` (below) |
| `w6-post-w7t03fu-default-check-20260926T103249Z` | `cargo check --locked -p jope-parallel --all-targets` | exit 0 |
| `w6-post-w7t03fu-ranked-default-20260926T103410Z` | `ranked_proposals`, `gpu_entry_guards` without `neural` | exit 4, 0 tests: both files are `neural`-only, so the coverage lanes (`rust-check`, `full-default`) run no test from them |
| `w6-post-w7t03fu-b-20260926T103523Z` | neural suite with the `--lib` filter widened to `proposals`, `ranked_proposals`, `smartcluster_worker` and `native_graph` | 140/140 |

**`ranked_proposals` failure is pre-existing and unrelated.** It fails with `propose-ranked: original native GPU grant verifier unavailable; numerical dispatch is held`. That error comes from `task_contract::require_qualified_gpu_experiment`, the first statement of `ranked_proposals::run`, before any changed line. Only a `neural` run can reach it, and it is outside this slice.

### Open
- **Assignment delivery.** Same join as W7-T03, owned by the controller lane.
- **Other fixed `Device::Cuda(0)` users** (`contexts.rs`, `monitoring.rs`, `components.rs`) and `encoding.rs` `Device::Cuda(gpu)` are single-device services, not rank-indexed, and were not changed.
