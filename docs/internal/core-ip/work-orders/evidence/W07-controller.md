# W7 — Repository-managed execution controller (source and fixture half)

Owner: Claude implementation subagent in `.agent/worktrees/honesty`, branch
`rr/w7-controller`, based on `723cc035a`. This record covers the source- and
fixture-provable parts of W7-T01, W7-T02, W7-T04 and W7-T05.

Every observation here is **fixture-proven** only. Identities are synthetic: fixture host
names, boot ids, GPU UUIDs and digests are generated. No host was contacted, nothing was
probed over a network, no GPU state was read, no SSH/AWS/S3 was used and no numerical work
ran. Nothing is `CI-verified` or `live-qualified`. The live phases W7-T07, W7-T08 and W7-T09
stay blocked, and so does the live three-host census that W7-T01 also requires.

Evidence runs live under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.
Each holds `head.txt`, `branch.txt`, `status.txt`, `diff.patch`, `diffstat.txt`,
`command.txt`, `invocation.txt`, `run.log` and `exit_code.txt`.

## What existed before (`723cc035a`)

- `jope-native-loop` exposed only `bind`, `bind-parent-decision`, `tick` and `serve` over
  the fixed `jope.native_loop.v1` contract (`native_loop::Config`: two 3000-update blocks,
  a single resident GPU job and native CPU jobs). Every argument was a required flag.
- There was no typed host census and no multi-host preflight. The nearest record,
  `native_loop::OriginalJobCensus`, is a SmartCluster job census for one daemon.
- Communicator rendezvous was the legacy same-allocation path `neural::Nccl::init`. Rank 0
  writes an NCCL unique id to a file such as `<launch>/nccl.id`. Other ranks poll that path
  for 60 seconds and read its first 128 bytes. The file is unauthenticated, is not bound to
  the run, attempt, boot, grant or deadline, and assumes a shared filesystem. Its
  staleness protection comes only from per-attempt launch directories.
- Dispatch and recovery were per single native job (`native_graph` task markers
  `submit-intent.json`, `submitted.json` and `failed.json`). There was no
  multi-participant group, whole-group drain or coordinator reconcile.

Pre-fix records:

- `w7c-t01-pre-20260926T095726Z` — source search showing no host census or preflight.
- `w7c-t02-pre-20260926T095725Z` — the baseline CLI refuses `plan`, `status`, `verify` and
  `cancel` as unrecognized subcommands.
- `w7c-t04-pre-20260926T101544Z` — the baseline shared-file `Nccl::init` receiving path,
  and no rendezvous or result message schema.
- `w7c-t05-pre-20260926T101545Z` — per-job markers only; no whole-group drain or restart
  reconcile.

These are **new capabilities**. There was no existing multi-host receiving path that
misbehaved and could be reproduced, so each pre-fix record documents the absence (and the
legacy shared-file path) instead of a failing test. The `fn preflight` and
`verify_admission` matches in those logs are unrelated single-host code in
`eval_registry/funding.rs`, `jope-campaigns` and `jope-bench`.

## Source layout

All of the new code is in `crates/jope-engine/src/native_loop/bounded/`, declared as
`native_loop::bounded`. The old loop code is unchanged apart from that one `pub mod`
line.

| File | Task | Contents |
|---|---|---|
| `bounded.rs` | — | Shared helpers: digest and name checks, domain-separated ed25519 signatures. |
| `bounded/census.rs` | T01 | `jope.native_host_census.v1`, `preflight`, `PreflightReceipt`, typed `RefusalReason`. |
| `bounded/contract.rs` | T02 | `jope.native_bounded_run.v1`, `PlanRefusal`, grants, work census, the successor rule. |
| `bounded/controller.rs` | T02, T05 | Plan, Status, Verify, Cancel, the hash-chained journal and its replay invariants, `Supervisor`, `Coordinator`. |
| `bounded/rendezvous.rs` | T04 | The challenge, hello, admission and result messages, bounded framing, artifact exchange. |
| `src/bin/jope-native-loop.rs` | T02 | The `plan`, `status`, `verify` and `cancel` verbs. |

## W7-T01 — Host census and preflight refusal (source half)

`HostCensus` (`jope.native_host_census.v1`) is a typed, versioned observation record. It is
bounded to 1 MiB, rejects unknown fields and is validated structurally. The controller
never gathers one. Each host observation carries:

- the frozen identity: host, boot id, address, certificate digest, source commit,
  executable digest, installed policy digest and the runtime closure (runtime, compiler,
  CUDA, NCCL, driver);
- the observation time;
- allocatable CPU and RAM;
- each GPU's UUID, model, local index, and total and available VRAM;
- every owner or reservation (CI, user or managed), with the run and attempt for managed
  owners.

`census::preflight(contract, plan_sha256, census, census_sha256, now)` compares the census
with the frozen plan. It returns **every** refusal reason, or a receipt that holds no
allocation. It refuses:

- stale telemetry (older than the frozen `max_age_ms`) and future telemetry (beyond
  `clock_skew_ms`);
- any difference in boot, address, certificate, source, executable, runtime or policy;
- a planned GPU UUID that is missing, or that is present under another local index;
- a device owned by anything other than this exact run attempt — a CI owner, a user, another
  run or another attempt of the same run;
- CPU, RAM or VRAM that is insufficient once other owners' reservations are subtracted;
- a missing host.

The `Coordinator` journals both admitted and refused preflights. Dispatch and rendezvous
require an admitted preflight whose receipt is still fresh.

Tests in `tests/bounded_run.rs`:

- `preflight_admits_a_fresh_matching_funded_census` (positive control)
- `preflight_refuses_stale_and_future_telemetry`
- `preflight_refuses_mismatched_boot_gpu_source_executable_runtime_policy` (9 reasons at
  once)
- `preflight_refuses_conflicting_device_ownership_but_not_its_own_attempt`
- `preflight_refuses_unavailable_capacity` (also covers a missing host)
- `census_record_is_typed_versioned_and_bounded`
- `execution_binds_to_the_planned_artifact` — a refused preflight blocks dispatch and
  rendezvous and launches nothing, and a stale receipt no longer permits dispatch.

Post-fix record: `w7c-t01-post-20260926T102532Z` (7 passed).

**Open.** The actual three-host census stays blocked: real boot ids, GPU UUIDs and models,
drivers, free VRAM, CI and user allocations, and certificate identities. No collector
exists. The planning observations (an RTX4090 on xbabe1 and RTX3090s on xbabe2 and xbabe3)
are not asserted anywhere in the code.

## W7-T02 — Versioned bounded-run contract and controller operations

`BoundedRunContract` (`jope.native_bounded_run.v1`) sits alongside `jope.native_loop.v1`.
It freezes:

- run, attempt and coordinator host, plus the coordinator and grant-authority public keys;
- `not_before` and the single `original_deadline`;
- the telemetry freshness policy, and the rendezvous policy (join timeout, message limit,
  heartbeat timeout);
- a sorted artifact closure of path, sha256 and byte count;
- per participant: global rank, local rank, the host identity, and the device UUID and
  local index; CPU, RAM, VRAM and wall ceilings; the participant's public key; an
  independently signed grant; and its result destination;
- the work census, with numerical, verification, artifact-movement, cleanup and
  failure-allowance items;
- the committed and planned update counters, and an optional predecessor attempt.

The trainer-side assignment type is **not** defined here. The contract's doc comment
records the join point: W7-T03 builds it from `global_rank`, `world_size`, `local_rank` and
the device `uuid`/`local_index` of the admitted plan.

### Refusal before dispatch

`PlanRefusal::{Legacy, Invalid, Underfunded}`. `controller::plan` validates the contract
before it creates any directory. `invalid_plans_fail_before_anything_is_created` covers 16
invalid cases, including:

- rank order, a duplicated device, a CPU participant with VRAM and a local-rank gap;
- a grant for another rank, and a broken grant signature;
- missing verification or cleanup work, and unequal synchronous updates;
- an unsorted closure, and a result destination that collides with the closure;
- an oversized message limit, and attempt 2 without a predecessor;
- a shared participant key, and a grant that outlives the deadline.

`underfunded_plans_fail_before_dispatch` covers:

- a grant below the CPU or VRAM ceiling;
- CPU core-milliseconds beyond `cpu × wall`, wall time beyond the ceiling, and peak memory
  beyond the ceiling;
- a grant window shorter than the wall ceiling;
- a remaining original window too short to fund the longest wall ceiling (and an elapsed
  deadline is invalid).

### Old semantics

`legacy_manifests_keep_legacy_semantics` shows three things:

- a `jope.native_loop.v1` `Config` still round-trips through its own loader;
- the same manifest is refused as `Legacy` rather than coerced, and creates nothing;
- a bounded contract fails the old loader's `deny_unknown_fields`.

On the CLI, the old verbs still require all nine loop arguments, and the new verbs refuse
them.

### Operations

- **Plan** writes the canonical contract bytes to `plan.json`, opens the journal with a
  `planned` record that holds the original deadline and counters, and prints a review with
  `dispatch_started: false`. Re-planning identical bytes is idempotent; different bytes
  are refused.
- **Execution binds to the plan artifact.** `Coordinator::open` takes the reviewed plan
  digest, not flags. Every journal record carries it, and replay refuses records bound to
  another plan. One writer is enforced with a controller lock.
- **Status** is read-only. It creates nothing, takes no lock (it reads `/proc/locks` to see
  whether a coordinator holds the run) and contacts nothing. It reports each rank as
  scheduled, admitted, running, committed, failed, drained or unknown:
  - dispatched but not launched is `unknown`;
  - launched with no live coordinator is `unknown`;
  - launched while awaiting admission is `scheduled`;
  - an admission that is persisted but not yet delivered is `admitted`.
- **Verify** is read-only. It checks canonical plan bytes and grant signatures, the whole
  hash chain and every replay invariant, the binding of any cancellation, and every
  acknowledged result against its signed envelope and admission. It takes no supervisor
  and reports `numerical_work_started: false`.
- **Cancel** is idempotent and bound to the plan digest. The first request is published
  without replacement, and later calls return it unchanged and write nothing.
  - With nothing dispatched, the attempt ends at once as `cancelled`: every rank is
    never-started and nothing is committed.
  - With a live coordinator, the request is delivered and the coordinator drains the whole
    group.
  - With dispatched work and no live coordinator, the drain waits for reconciliation.

Tests: `plan_freezes_canonical_contract_and_is_idempotent`,
`status_distinguishes_all_seven_states`, `status_and_verify_are_read_only_and_start_nothing`
(a byte, mode and mtime snapshot of the tree is unchanged, and the supervisor counters are
unchanged), `verify_detects_tampering_with_plan_journal_or_results`,
`cancel_is_idempotent_and_bound_to_the_plan`, `execution_binds_to_the_planned_artifact`,
`cli_plan_status_verify_cancel_round_trip` and
`cli_refuses_mixed_legacy_arguments_and_bad_contracts_before_creating_state`.

Post-fix record: `w7c-t02-post-20260926T102533Z`. It contains all 17 `bounded_run` tests,
the new `--help`, CLI refusals and the parity gate.

CLI argv, as frozen in source (`jope-native-loop`):

```text
jope-native-loop plan    --run-dir DIR --contract FILE [--predecessor-run-dir DIR]
jope-native-loop status  --run-dir DIR
jope-native-loop verify  --run-dir DIR [--plan-sha256 HEX]
jope-native-loop cancel  --run-dir DIR --plan-sha256 HEX --reason TEXT
```

There is deliberately no execute verb. Live dispatch through SmartCluster is W7-T07.

The capability matrix has four new `engine-subcommand` rows with status `new`. The counts
are recomputed: 94 rows, 88 moved and 6 new, for a total of 784.
`deploy/ci/parity-matrix.sh --no-build` reports ok.

## Commit structure

The journal vocabulary, and the Status mapping that T02's acceptance names, depend on the
admission and drain records. So the rendezvous (T04) and coordinator (T05) source landed
with the T01/T02 commit and are exercised there only by the Status state tests. Their own
acceptance fixtures land in the W7-T04 and W7-T05 commits.
