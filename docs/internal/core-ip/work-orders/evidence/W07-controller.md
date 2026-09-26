# W7 — Repository-managed execution controller (source and fixture half)

Owner: Claude implementation subagent in `.agent/worktrees/honesty`. T01/T02 landed on
branch `rr/w7-controller` (based on `723cc035a`) and were cherry-picked into
`feat/regression-remediation` as `8d66723d3`. T04, T05 and the plan → trainer assignment
join are on `rr/w7-controller-2`, based on `8d66723d3`, which also carries W7-T03
(`fdbde7c3f`). This record covers the source- and fixture-provable parts of W7-T01,
W7-T02, W7-T04 and W7-T05, and the W7-T02 × W7-T03 join.

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
- any difference in machine (added on `rr/w7-controller-2`), boot, address, certificate,
  source, executable, runtime or policy;
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
- `preflight_refuses_mismatched_boot_gpu_source_executable_runtime_policy` (10 reasons at
  once since the machine id was added; 9 at `8d66723d3`)
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
acceptance fixtures land in the W7-T04 and W7-T05 commits on `rr/w7-controller-2`.

## Host machine identity (contract change on `rr/w7-controller-2`)

The W7-T03 trainer assignment identifies a host by `/etc/machine-id`. The T02 plan had
only an operator-facing host name, so it could not supply that value without an implicit
naming convention. `HostIdentity` therefore gains `machine_id` (32 lowercase hex).

- Census preflight compares it (`RefusalReason::MachineMismatch`).
- The rendezvous hello reports it, and a difference is `WrongParticipant`.
- Plan validation refuses two host names that share a machine id or a boot id.

The contract is still unreleased and nothing live has used it; this is a schema change to
`jope.native_bounded_run.v1` and `jope.native_host_census.v1` before first use, not a new
version.

## W7-T04 — Authenticated rendezvous and immutable verified exchange

Source: `bounded/rendezvous.rs` and the admission/delivery steps of
`bounded/controller.rs`.

**Messages.** Every message is a length-prefixed frame. The frozen `max_message_bytes`
(bounded to 1 KiB–1 MiB by the contract) is enforced before the body is read. Each carries
a domain-separated ed25519 signature from a key frozen in the plan:

- `challenge` (coordinator): run, attempt, plan digest, rendezvous id, coordinator boot,
  opened time and join deadline. The rendezvous id digests the plan, attempt, coordinator
  boot, open time and a fresh kernel nonce, so a message captured before a coordinator
  restart cannot join a new rendezvous.
- `hello` (participant): run, attempt, plan, rendezvous id, rank, delivery id and send time;
  the participant's observed host, machine, boot, device, source commit, executable,
  runtime and installed policy; its own locally hashed input closure; its grant digest;
  and, from rank 0 only, the communicator bootstrap.
- `admission` (coordinator): every member with its hello digest, host, boot and device;
  the grants digest; the original deadline; and the bootstrap.
- `result` (participant): run, attempt, plan, admission digest, rank, delivery id, result
  digest and length, update count and completion time.

**Group refusal.** The first invalid message refuses the whole group, and the refusal is
terminal. It covers:

- a forged or unknown peer;
- another run, attempt, plan, rendezvous or boot;
- a wrong participant;
- substituted source or runtime, or a substituted input closure;
- an expired grant or deadline, a stale message or the join timeout;
- equivocation, where one delivery id carries different bytes (an identical redelivery is
  idempotent);
- an oversized or malformed frame.

`GroupRefusal` always records `started_ranks: 0` and `committed_updates: 0`.
`Coordinator::record_refusal` journals it, stops any launched rank process before it can
start work, and marks every rank `refused`.

**Admission before any rank starts.**

- `Coordinator::record_admission` requires the persisted launch record of every rank. It
  verifies that the admission is the planned coordinator's signed full-group admission and
  journals it *before* any participant learns of it.
- `deliver_admission` sends that journaled admission only, to one control stream per
  planned rank. It records `admission_delivered`, and a failed delivery drains the group.
- A participant starts work only after `verify_admission` accepts that admission against
  its own hello.
- A rank that exits before delivery is a failure, and no result is acknowledged before
  delivery.

**Result authentication.** `verify_result` checks, before acknowledgement:

- the planned participant key;
- the run, attempt, plan and admission;
- the byte length and digest;
- the destination bound;
- the synchronous update count.

**Artifact exchange.** `receive_artifact` accepts only bytes that match the planned closure
entry. It publishes them without replacement: an identical file is idempotent, and a
conflicting existing file is refused and kept.

Tests in `tests/bounded_run.rs` (T04 part):

- `full_group_is_admitted_and_acknowledged_before_any_rank_may_start` (positive control)
- `missing_participant_refuses_the_group_with_zero_commits`
- `substituted_source_or_input_refuses_the_group` (executable, runtime and a changed input
  byte)
- `forged_peer_refuses_the_group` (a foreign key, a field altered in transit, an unknown
  rank)
- `expired_grant_refuses_the_group` (at hello and at admission)
- `stale_or_replayed_rendezvous_is_refused` (an earlier rendezvous; another attempt, boot,
  plan or rendezvous; an old message; a forged or expired challenge; an admission from
  another rendezvous; equivocation)
- `delivery_ids_message_limits_and_timeouts_are_enforced`
- `results_are_authenticated_before_acknowledgement`
- `immutable_artifact_exchange_verifies_bytes_without_replacement`
- `same_host_socket_rendezvous_admits_only_the_authenticated_full_group` (real Unix-socket
  handshakes on threads; the forged case refuses and admits no rank)

Every refusal case asserts four things:

- the refusal is `started_ranks: 0` and `committed_updates: 0`;
- no launched rank process survives;
- a later admission or dispatch is refused;
- Status and Verify show `refused` with 60 000 committed updates, unchanged, and no
  `results/`.

Pre-fix record: `w7c-t04-pre-20260926T101544Z`. Post-fix record:
`w7c-t04-post-20260926T103831Z` (all of `bounded_run`: 27 passed).

**Open.** The cross-host transport, certificate or TLS identity, and actual network
timeouts are not implemented; the Unix-socket transport is a same-host fixture only. The
legacy `neural::Nccl::init` shared-file rendezvous is untouched (trainer partition). The
trainer still has to consume `AdmittedMember::communicator_bootstrap` instead of
`nccl.id`, which needs the live path.

## W7-T05 — Whole-group drain and coordinator recovery

Source: `Coordinator` in `bounded/controller.rs`.

**Persistence order.** Each step is journaled before the one that depends on it:

1. the dispatch intent;
2. the launch identity (ownership);
3. the full-group admission;
4. delivery of the admission;
5. result acknowledgement, whose bytes are published first;
6. the single group commit.

Journal replay enforces this order, one drain, at most one commit, and exactly one
disposition per rank before the terminal record.

**Drain causes.** A drain covers the whole group and is idempotent and resumable. The
causes are rank failure (a nonzero exit, a missing or invalid result, or an exit before
admission delivery), lost control, grant expiry, deadline, cancellation and coordinator
restart. Every rank ends with exactly one disposition:

- `failed` — the rank that caused the drain;
- `drained` — stopped by the drain;
- `never_started` — no supervised work exists for its dispatch id;
- `completed_unacknowledged` — an authenticated result was retained but never
  acknowledged;
- `completed_uncommitted` — the result was acknowledged, but the group drained;
- `refused`;
- `committed`.

**Restart.** A restarted coordinator refuses dispatch and polling until it reconciles.

- A commit already journaled before the crash is finished exactly once, at any time, and is
  never drained.
- Otherwise a new recovery commit passes the same gates as `poll`: no pending cancellation,
  and the group window still open. The window closes at the original deadline or the first
  participant grant expiry. If the whole group completed with authenticated results inside
  it, the coordinator acknowledges them and commits once. (This bullet was corrected after
  review of #111; see below.)
- Otherwise it drains. Work that was launched but whose launch was never journaled is found
  by dispatch id, stopped and labelled as such.
- Reopening a terminal run changes nothing.
- A successor attempt must keep the run, the original deadline, the topology and the
  committed counters, and name the predecessor's exact journal tip.

**Fixture.** `tests/bounded_run_recovery.rs`. Rank processes are re-executions of the test
binary. They join the authenticated rendezvous over a Unix socket, hold on heartbeats with
strictly increasing delivery ids, and stop themselves with exit 75 when control is lost.
This is a **same-host fixture only**: three processes under one boot on one host are not
the three-host proof (W7-T07), and no rank does numerical work.

- `positive_control_commits_the_whole_group_exactly_once`
- `killed_rank_drains_the_whole_group_with_zero_commits` (SIGKILL one rank)
- `severed_control_stops_every_rank_and_drains_the_group` (heartbeats stop; ranks exit
  themselves)
- `coordinator_restart_reconciles_and_drains_with_original_deadline_and_counters` (also
  refuses successor plans that renew the deadline, change topology or reset counters)
- `restart_commits_a_complete_unacknowledged_group_exactly_once`
- `restart_distinguishes_unacknowledged_completion_from_running_work`
- `restart_before_admission_distinguishes_never_started_from_unrecorded_launch`
- `cancellation_and_grant_expiry_drain_the_whole_group`

Each scenario ends by asserting that no rank process survives: none of the recorded pids
exists (reaped), and no `/proc/*/cmdline` names a dispatch id of the run.

One flake was found and fixed while repeating the run. Concurrent fixtures created in the
same millisecond had identical plans, so the orphan scan saw another test's ranks. Each
harness now uses a unique run name. After the fix, both targets passed four consecutive
repeats (36 of 36 each time).

Pre-fix record: `w7c-t05-pre-20260926T101545Z`. Post-fix record:
`w7c-t05-post-20260926T103906Z` (9 passed: the 8 scenarios plus the `rank_child` entry
point, which is a no-op outside a rank process).

**Open.** A SmartCluster-backed `Supervisor` and the cross-host drain are W7-T07. The
fixture supervisor keeps its records on disk like a daemon but reaps as the parent process.

### Review of #111 — grant expiry during coordinator recovery (P1, fixed)

**Finding.** The independent review of PR #111 (neverhumanbot, CHANGES_REQUESTED at
`0b303b293`) found a real defect in the W7-T05 recovery path. `poll()` drains the group once
any participant grant has expired. `reconcile()` checked only cancellation, a delivered
admission and the original deadline before it acknowledged and committed retained results.
So a coordinator restart between rank completion and acknowledgement bypassed the
grant-expiry gate. The reviewer's reproduction
(`independent-review-pr111/recovery-expiry.rs` and `.log`, under the custody root) showed
the difference at the same instant, the first grant expiry:

- poll gave `Drained` with 60 000 committed updates;
- reopen + reconcile gave `Committed` with 60 003.

**Audit of the other paths.** Before the fix, each authority-bearing step checked its own
subset of the window:

| Path | Before the fix | Gap |
|---|---|---|
| `poll` | cancellation, deadline, every grant | none |
| `reconcile` (new recovery commit) | cancellation, delivery, deadline | every grant |
| `acknowledge` (adopting a retained result) | none of its own; relied on callers | every gate, if reached by a new caller |
| `commit` (new group commit) | deadline | every grant |
| dispatch (`require_dispatchable`) | deadline and *this rank's* grant | another participant's grant |
| `record_admission` | deadline | every grant |
| `deliver_admission` (ranks may start) | none | deadline and every grant |
| journal replay / Verify | no time gate | a late acknowledgement or commit verified clean |

**Fix** (`bounded/controller.rs`). One gate, `group_window_closed(contract, now)`, closes at
the original deadline or the first participant grant expiry, whichever is earlier. It is
applied everywhere:

- `poll` and `reconcile` drain with the gate's cause (`grant_expired` or
  `deadline_elapsed`). `reconcile` applies it before any retained result is adopted, right
  after the cancellation check.
- Dispatch, launch, admission journaling, acknowledgement and any new group commit refuse
  after the window closes.
- `deliver_admission` drains the group instead of delivering.
- Journal replay (so Status, Verify and every reopen) refuses a dispatch intent, launch,
  admission, delivery, acknowledgement or commit whose `recorded_unix_ms` falls after the
  window.

Finishing an already-journaled commit stays distinct. It completes the dispositions and the
terminal record exactly once, at any time, and is never drained. Drain, disposition and
terminal records may still be journaled after the window, as a drain must be.

**Tests** — all pre-fix failures were observed first:

- `bounded_run_recovery.rs` (real rank processes, same host):
  - `restart_at_grant_expiry_drains_retained_results_instead_of_committing` — the
    reviewer's scenario. It now ends `Drained` with 60 000 updates, every rank
    `completed_unacknowledged`, no acknowledgement, commit or `results/`, and no orphans.
    Pre-fix it was `Committed` with 60 003.
  - Positive control `restart_just_before_grant_expiry_commits_exactly_once` — reconcile
    one millisecond before the first grant expiry commits once, and a later reopen changes
    nothing.
  - Positive control `an_already_journaled_commit_is_finished_not_drained_after_expiry` —
    the journal is truncated right after `group_committed` (the exact prefix a crash
    leaves). Reconcile after expiry finishes it as `Committed` with 60 003, one
    `group_committed`, no `drain_started`, and Verify passes.
- `bounded_run.rs` (in-process):
  - `dispatch_stops_when_any_participant_grant_expires` — rank 2's grant closes the window
    while rank 0's own grant and the deadline are open. Pre-fix, rank 0 was still
    dispatched.
  - `admission_and_delivery_stop_when_any_participant_grant_expires` — journaling the
    admission, and delivering it, at the window close. Pre-fix the admission was journaled
    at the close, and the test stopped there; delivery had no time gate at all. Now the
    admission is refused, delivery drains with `grant_expired`, and no peer receives a
    frame.
  - `verify_refuses_acknowledgements_or_commits_journaled_after_the_window_closed` — the
    journal is re-chained with one record re-stamped at the expiry. Acknowledgement,
    commit, dispatch intent, admission and delivery are refused; dispositions and the
    terminal record are accepted. Pre-fix, Verify accepted a late acknowledgement.

**Fixture flake found while repeating (and fixed).** Repeating the process fixture exposed
a race in the test-only `ProcessSupervisor::probe`, not in the controller. A rank could
become a zombie between `waitpid(WNOHANG) == 0` and the `/proc` stat read. The probe then
reported it `Gone` without reaping it, so `terminate` returned and `assert_no_orphans` found
the unreaped pid ("rank process N survives (or is unreaped)"). It appeared in 1 of 3 and 1
of 10 repeats under host load, in `cancellation_and_grant_expiry_drain_the_whole_group` and
`restart_distinguishes_unacknowledged_completion_from_running_work`. This race predates the
review fix. The probe now reaps a zombie of its own child immediately, so a rank is only
ever reported `Gone` once it has been reaped.

Records (all under the custody runs root):

- `w7c-review-expiry-pre-20260926T143503Z` — pre-fix. 42 run, 38 passed, 4 failed: the
  reviewer's scenario and the three other gaps. Both positive controls passed.
- `w7c-review-expiry-post-20260926T143709Z` — post-fix, `bounded_run` +
  `bounded_run_recovery`: 42 passed.
- `w7c-review-expiry-repeat-post-20260926T143714Z` — 3× repeat: 41/42, 42/42, 42/42. The
  one failure is the fixture zombie race above.
- `w7c-review-zombie-pre-20260926T144235Z` — the observed zombie-race failures (10×
  repeat, 1 failure), with the full failing log.
- `w7c-review-zombie-post-20260926T144235Z` — 20× repeat of `bounded_run_recovery` after
  the fixture fix: 20 of 20 runs passed all 12 tests.
- `w7c-review-expiry-neural-post-20260926T143919Z` — `neural` build, `CUDA_VISIBLE_DEVICES=""`:
  `bounded_assignment` 3 and `trainer_topology` 7, all passed. Run alone, never alongside
  the fence suites.

## Join — frozen plan → trainer assignment (W7-T02 × W7-T03)

Source: `bounded/assignment.rs`, compiled only with the `neural` feature because
`training` is neural-gated. The join is intentionally small:

- `trainer_assignment(contract)` is a pure function. It reads no clock, file, environment
  or device. It maps each participant to `topology::Participant`:
  - `host_id` is `machine_id`;
  - `boot_id` and `local_rank` are copied;
  - `device_index` is 0;
  - `gpu_uuid` is the planned UUID.
  The plan digest is SHA-256 of the canonical contract bytes, identical to the controller's
  `plan.json` digest. It requires `topology::validate` to accept the result, and it refuses
  a rank without a GPU or a UUID that is not in `nvidia-smi` form.
- `verify_trainer_assignment` refuses any assignment that differs from it.
- `worker_cuda_visible_devices(contract, rank)` is the planned UUID. This is the visibility
  contract: each worker sees exactly its own GPU, at CUDA ordinal 0. The plan's
  `local_index` is the census's host-wide index; CUDA's default enumeration need not match
  it, so it never becomes a CUDA ordinal.

Tests: `tests/bounded_assignment.rs`, neural-gated and CPU only with
`CUDA_VISIBLE_DEVICES=""`:

- `plan_maps_to_exactly_the_assignment_topology_accepts` — three one-GPU hosts, and two
  workers on one host. The derived assignment equals the expected one, `validate` accepts
  it, and the digest equals `controller::plan`'s. `topology::bind` succeeds for every rank
  under the restricted view.
- `plan_assignment_mismatch_is_refused` — covers another digest, swapped GPUs, the host
  index used as the CUDA ordinal, another boot or machine, a dropped rank, a renumbered
  local rank, the wrong schema, and an assignment derived from another plan.
- `plans_the_trainer_cannot_bind_are_refused` — covers a CPU rank, a non-`nvidia-smi` UUID,
  an unrestricted worker that can see another participant's GPU, and a rebooted host.

Pre-fix record: `w7c-join-pre-20260926T103432Z` (no derivation existed; the plan had no
machine id). Post-fix record: `w7c-join-post-20260926T103759Z`, a `neural` build against
`/opt/jope-sdk` with `CUDA_VISIBLE_DEVICES=""` set inside the script (its
`invocation.txt` records the outer environment before that). 8 passed: the 3 join tests
and the 5 W7-T03 `trainer_topology` tests. No device was enumerated or used. The run is
sequential, never alongside the fence suites.

**Open.** Two pieces need the live path (W7-T07/T08). One is how the Assignment reaches
`training::rank_impl`, which today uses the legacy single-host binding. The other is the
launcher that actually sets each worker's `CUDA_VISIBLE_DEVICES` to its planned UUID.

## W7 Batch B — live-path source integration (`rr2/w7-live`)

Owner: Claude implementation subagent in `.agent/worktrees/demo`, branch `rr2/w7-live` from
`main` `8a55c4fef`. It builds on Codex's read-only `jope-native-loop preflight` (PR113) and
does not change it. Every result below is **fixture-proven** on one host. No other host was
contacted, no GPU was used, `nvidia-smi` was never run by a test (injected command runner),
and nothing is live-qualified. Evidence runs use the `w7b-<item>` prefix under the custody
runs root.

### Item A — W7-T01 census collector and pinned merge

**New capability.** Before this change the CLI refused `census` as an unrecognized subcommand
(exit 2) and no source read a host's identity for the census (`w7b-A-pre-20260926T232754Z`).
There was no receiving path to reproduce a defect in.

Source: `bounded/collect.rs` and the `census` verbs of `jope-native-loop`.

```text
jope-native-loop census collect --out FILE --address IP --certificate FILE
    --source-commit HEX --executable FILE --runtime-library FILE... --compiler FILE
    [--cuda-library FILE...] [--nccl-library FILE...] --installed-policy FILE...
jope-native-loop census merge --out FILE --max-age-ms N --clock-skew-ms N
    --document PATH=SHA256...
```

`collect` reads the local host only and writes exactly one `jope.native_host_census.v1`
document, which it validates first. It never replaces an existing file and prints a
`jope.native_host_census_collection.v1` report with the document digest and
`allocation_held: false`. Nothing is opened, allocated or reserved.

| Field | Source |
|---|---|
| host, machine id, boot id | `/proc/sys/kernel/hostname`, `/etc/machine-id`, `/proc/sys/kernel/random/boot_id` |
| address | `--address`, refused unless assigned to a local interface (`getifaddrs`) |
| certificate | SHA-256 of the DER certificate (DER, or PEM with exactly one certificate); only the DER framing is checked here |
| executable, compiler | SHA-256 of the installed file (symlinks resolved; size and inode re-checked after hashing) |
| runtime, CUDA, NCCL, installed policy | `jope.native_file_closure.v1` digest of the sorted (canonical path, SHA-256, bytes) of the named files; CUDA/NCCL absent when none are named |
| source commit | `--source-commit`: the installer-declared commit; the executable digest binds the bytes |
| CPU, RAM | online CPUs × 1000; `MemAvailable` (available now, not total) |
| GPUs, driver | `nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free` |
| device owners | `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name`; each process is a non-managed (`user`) owner named `gpu-process-<pid>` |

**Nothing is invented.** `CUDA_VISIBLE_DEVICES=""` records no GPU and runs no probe.
An absent `nvidia-smi` records no GPU. A failing probe, an `[N/A]` value, a malformed
record, differing driver versions or a process on an unreported GPU refuses the whole
collection. So does a partial `CUDA_VISIBLE_DEVICES`, since the census is host-wide.

`merge` pins every document to the SHA-256 its collector printed, before interpreting it.
It then refuses:

- an altered document or a malformed pin, and the same document given twice;
- mixed or unsupported schema versions;
- a document that is not exactly one host;
- stale or future-dated telemetry, under the freshness bounds a plan's telemetry policy
  allows;
- two documents sharing a host, machine id, boot id, address or certificate.

The output is the merged census, hosts sorted by name and `captured_unix_ms` set to the
latest document's. It is written without replacement, with a
`jope.native_host_census_merge.v1` report listing every document's digest.

Tests in `tests/bounded_census.rs` (8) and the `collect` unit test (1):

- `collected_census_round_trips_through_preflight` — positive control: the collected
  document is the typed census the validator reads, and a plan built from its identity
  passes `preflight_dry_run`. A substituted executable is refused as `executable_mismatch`.
- `cpu_only_or_absent_nvidia_smi_records_no_gpu` — with `CUDA_VISIBLE_DEVICES=""` no probe
  runs (0 calls). With `nvidia-smi` absent, no GPU is recorded. A CPU plan is admitted, and
  a GPU plan on that census is refused as `gpu_missing`.
- `failing_malformed_or_unavailable_gpu_probes_refuse_the_collection` (7 cases)
- `device_processes_are_owners_that_preflight_refuses` — a compute process on the planned
  GPU refuses the preflight as `conflicting_ownership`; a CPU plan on the same host is
  admitted.
- `unverifiable_identity_inputs_refuse_the_collection` — a non-local or non-IP address, a
  bad source commit, missing, garbage or non-certificate DER, relative, missing or
  duplicated files, and no runtime library or policy.
- `merge_combines_pinned_documents_into_a_preflight_census` and
  `merge_refuses_tampered_stale_future_duplicate_and_mixed_documents`.
- `cli_collect_merge_and_preflight_round_trip_without_side_effects` — the real binary. The
  collector writes its one file and nothing else, refuses to replace it and refuses loop
  flags. A wrong pin refuses the merge and writes nothing. The merged census passes the
  operator `preflight` for a three-participant plan.

Post-fix record: `w7b-A-post-20260926T233650Z`, the full per-commit gate. It covers
`bounded_census` 8, `bounded_preflight_cli` 3, `bounded_run` 30, `bounded_run_recovery` 12,
the native lib tests 30, the fence suites (16, 8, 17 and `gp::fit_contract_tests` 39),
workspace check, fmt, diff check, coverage, test inventory and the parity matrix.

**Open (W7-T01).**

- No live census has been taken on xbabe1, xbabe2 or xbabe3; that is an operator runbook
  step (item G), executed only after separate admission.
- CI, runner and SmartCluster reservations are not read. Device processes are the only
  owners the collector observes. Reservations still have to be recorded by an admitted
  operator or a SmartCluster-aware collector.
- The source commit is declared, not observed. The certificate is verified by the
  transport (item D), not by the collector.
