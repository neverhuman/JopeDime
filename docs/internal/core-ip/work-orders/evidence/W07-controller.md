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

### Item B — W7-T03 assignment delivery and launcher

**Pre-fix record** (`w7b-B-pre-20260926T234107Z`, source trace). Every `rank_impl` caller
passed `None` for the assignment. Nothing applied the plan's device to a worker: the only
launchers that set `CUDA_VISIBLE_DEVICES` (`hyperion.rs`, `components.rs`) set it to the
rank number, which is the rank-as-device pattern W7-T03 removed. There was no launch record
and no worker-side visibility check. This is a delivery gap, not a reproducible failing call,
so the record is a source trace.

Source: `bounded/launch.rs` (default features) and `bounded/assignment.rs::bound_worker`
(`neural`).

- **Launch record** `jope.native_bounded_launch.v1`, one per rank. It holds the run, attempt,
  plan digest, rank, world, dispatch id, the planned host, machine and boot, and the exact
  visibility: the planned GPU UUID, or empty for a CPU rank. `launch_record` derives it from
  the frozen plan. `verify_launch_record` refuses any record that differs in any field.
  `load_launch` reads the plan bytes, which must hash to the admitted digest and be
  canonical, and the record (bounded, no symlink, no unknown field).
- **Launcher.** `worker_command` clears the environment and sets only
  `CUDA_VISIBLE_DEVICES` to the frozen value before exec. The plan never becomes argv.
- **Worker check.** `verify_worker_visibility` runs before any input, device or
  communicator work. A GPU rank must see exactly its planned UUID. It refuses an unset
  value, a local index such as `0` or `-1`, a duplicated or additional device, another
  rank's device and an empty value. A CPU rank accepts exactly the two hide-all values: the
  empty string (this launcher) and `-1`, which SmartCluster's executor sets for a job with no
  GPU.
- **Trainer side.** `bound_worker(contract, plan_sha256, record, CUDA_VISIBLE_DEVICES,
  local_devices)` checks the record and the visibility first. Only then does it read the
  device view, which must be exactly the one planned device at ordinal 0. It then derives
  the assignment from the plan and binds with `topology::bind`. It returns the assignment
  and binding for `rank_impl`. The call from a bounded rank into `rank_impl` lands with item
  C, because a bounded rank also needs the admitted communicator bootstrap.

Tests:

- `tests/bounded_launch.rs` (default features, 4):
  - `launch_records_are_exactly_what_the_plan_derives` — 11 single-field edits, a foreign
    plan, a wrong digest and an unplanned rank are all refused.
  - `load_launch_reads_only_the_frozen_plan_and_record` — non-canonical plan bytes,
    another digest, and a forged, extended, symlinked or oversized record are refused.
  - `the_launcher_environment_is_exactly_the_frozen_visibility` — `/usr/bin/env` exec'd
    through the launcher prints exactly `CUDA_VISIBLE_DEVICES=<planned UUID>`, or
    `CUDA_VISIBLE_DEVICES=` for the CPU rank, and nothing else.
  - `a_worker_refuses_any_other_visibility_before_starting` — 12 cases.
- `tests/bounded_assignment.rs` (`neural`, CPU): the new
  `a_worker_binds_only_from_its_frozen_launch_and_visibility`.
  - Every rank binds on a one-GPU-per-host layout and on a two-workers-per-host layout.
  - Six environment refusals happen without the device view being read.
  - The device view is refused when it shows another rank's GPU, a duplicated UUID, the
    planned UUID at ordinal 1 (a wrong local index), an extra or foreign device, or nothing.
  - Also refused: a rebooted host, a failing device view, edited or foreign launch records,
    and a genuine launch record on another host's view.

Post-fix records:

- `w7b-B-post-20260926T235111Z` — the full gate, including `bounded_launch` 4.
- `w7b-B-post-neural-20260926T235031Z` — `neural` build, `CUDA_VISIBLE_DEVICES=""`, run
  alone: `bounded_assignment` 4 and `trainer_topology` 7.

Two earlier records are marked `SUPERSEDED`. They predate the `-1` hide-all acceptance:
`w7b-B-post-neural-20260926T234636Z` passed, and `w7b-B-post-20260926T234709Z` was stopped
mid-run.

**Open (W7-T03) — a SmartCluster decision for the GPU phase.**

- SmartCluster's executor owns `CUDA_VISIBLE_DEVICES` for its jobs, and a job may not set
  it (`reserved_environment_name`).
- For a non-MIG GPU it exports the device's physical index, not its UUID (`gpu.rs`: the
  device `id` is `physical_index` unless MIG).
- `GpuRequest` cannot pin a device UUID.

So a GPU rank launched through SmartCluster sees `CUDA_VISIBLE_DEVICES=0`, and
`verify_worker_visibility` refuses it by design. The frozen assignment cannot be satisfied
through the scheduler as it stands. The options are for the owner:

1. SmartCluster accepts an exact UUID pin in `GpuRequest` and exports UUIDs.
2. The worker translates SmartCluster's index to a UUID through NVML under
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and verifies that against the plan.
3. Bounded GPU ranks bypass SmartCluster's GPU allocation.

CPU ranks (W7-T07) are unaffected. Native code is compile-checked only; no device was
enumerated.

### Item C — W7-T04 communicator bootstrap from admission

**Pre-fix record** (`w7b-C-pre-20260926T235601Z`, source trace). Every communicator creation
(`training.rs` and both neural probes) called `Nccl::init(binding, <dir>/nccl.id)`. Rank 0
wrote an unauthenticated unique id to a shared path, and the other ranks polled it. Nothing
consumed the verified admission's `communicator_bootstrap` apart from one rendezvous test
assertion. The call is native, so the record is a source trace.

Source: `bounded/launch.rs` (default features); `neural.rs` and `training.rs` (`neural`).

- `launch::nccl_bootstrap(contract, plan_sha256, launch_record, member)` returns an
  `AdmittedBootstrap`, which nothing else can construct. It requires:
  - a launch record the plan derives;
  - an admission for exactly this run, attempt and plan, re-verified against this plan
    (coordinator signature and full-group binding);
  - the member's admission digest;
  - this launch's rank and world;
  - a bootstrap of exactly 128 bytes, `NCCL_UNIQUE_ID_BYTES`.

  A stale or replayed admission never becomes an `AdmittedMember`, because
  `verify_admission` binds it to the rank's own hello and rendezvous.
- `neural::nccl_unique_id()` is rank 0's fresh id for its signed hello.
  `Nccl::init_admitted(binding, bootstrap)` checks `verify_admitted_bootstrap` (this
  binding's plan, rank and world) before any native call.
- **Old path kept for historical launches only.** `Nccl::init(binding, path)` still serves
  bindings without a frozen plan: the legacy single-host mapping and the probes. It refuses a
  binding derived from a frozen bounded plan before reading or writing anything.
- `training::communicator_source(assignment, admitted, rank, launch)` runs in `rank_impl`
  right after the assignment check, before any input or device work.
  - With no plan and no bootstrap, it returns the historical `launch/nccl.id`, unchanged.
  - A plan without a bootstrap is refused ("the shared nccl.id rendezvous is refused"), and
    so is a bootstrap without a plan.
  - A bootstrap for another plan, rank or world is refused.
  - A bounded launch directory that holds an `nccl.id` is refused.
  - `rank_impl` gains the `admitted` argument, and every existing caller passes `None`, so
    their behaviour is unchanged.

Tests:

- `tests/bounded_launch.rs` (default features, now 5):
  `the_nccl_bootstrap_comes_only_from_this_plans_verified_admission`.
  - Positive control: all three ranks take rank 0's id from the signed admission.
  - Forged: a bootstrap altered in transit, and an admission signed by a participant key,
    both give `ForgedCoordinator`.
  - Replayed: an earlier rendezvous's admission gives `StaleRendezvous`.
  - Wrong attempt: attempt 2's admission is refused as `ReplayedAttempt`, and attempt 2's
    membership never yields attempt 1's bootstrap.
  - Also refused: another plan's membership, another rank's membership, an edited launch
    record and a 64-byte bootstrap.
- `tests/bounded_assignment.rs` (`neural`, CPU, now 5):
  `bounded_plans_take_the_communicator_only_from_the_verified_admission`. It covers every
  `communicator_source` case. It shows that `Nccl::init` refuses a plan-bound binding and
  that `Nccl::init_admitted` refuses a mismatched bootstrap, both before any native call.

Post-fix records:

- `w7b-C-post-20260927T000223Z` — the full gate.
- `w7b-C-post-neural-20260927T000056Z` — `neural`, `CUDA_VISIBLE_DEVICES=""`, run alone:
  `bounded_assignment` 5, `checkpoint_commit` 5, `lossless_partition` 10,
  `partition_wiring` 7 and `trainer_topology` 7, plus the `training::`, `neural::` and
  `gpu_worker::` lib tests (7, 1 ignored).

**Open (W7-T04, trainer side).** A bounded training rank's entry into `rank_impl`, with the
assignment, the admitted bootstrap and the plan-selected partition, lands with item F. No
native call ran: `Nccl::init_admitted` with a matching bootstrap and a real NCCL
communicator are compile-checked only, and belong to W7-T08.

### Item D — W7-T04/T07 cross-host transport with certificate identity

**Pre-fix record** (`w7b-D-pre-20260927T000633Z`, source trace).

- The rendezvous transport was a same-host Unix socket only: `Connected` held a
  `UnixStream`, and the only entry points were `gather_unix` and `join_unix`.
- No TCP, TLS or certificate-bound channel existed anywhere in the bounded controller, and
  jope-parallel had no TLS dependency.
- The planned `certificate_sha256` was compared only against a census record, never against
  a peer.

This is a new capability.

**TLS stack.** The workspace already carries rustls 0.23 and tokio-rustls 0.26 with the
aws-lc-rs provider (jainnode's mTLS control channel). jope-parallel now uses the workspace
`rustls`. The provider is always named explicitly (`aws_lc_rs::default_provider()`), so the
workspace's single-provider guard still passes: `rustls feature "aws_lc_rs"` is the only
one. `Cargo.lock` changes by exactly one line, the dependency edge; no package is added.
Test certificates are built from DER in the test support code and signed with
ed25519-dalek, so no certificate crate is added.

**Identity model.** The census already pins each host's `certificate_sha256`, and so does
the plan. The transport makes that pin the channel identity:

- TLS 1.3 only, and mutual.
- Each side presents its host certificate. It accepts the peer only when the SHA-256 of the
  peer's end-entity certificate is a planned digest and the certificate is inside its
  validity window. The handshake signature is verified with the provider's algorithms.
- No chain is accepted, no CA, no host name (SNI is off).
- `planned_certificates` refuses two hosts that present one certificate.
  `coordinator_certificate` is the certificate of the planned host named `coordinator_host`,
  so a coordinator outside the plan has no identity and is refused.

**Rendezvous over the transport** (`rendezvous.rs`).

- `gather` and `join` now run over any `ControlStream` from an `Acceptor`. `gather_unix` and
  `join_unix` remain thin fixture wrappers.
- `offer_from(frame, PeerIdentity)` binds each hello to its channel: a certificate
  authenticated peer may speak only for ranks planned on the host that owns that
  certificate. Any other rank is refused as `WrongParticipant`, which refuses the whole group.
- `Connected.stream` is now a boxed `ControlStream`. The two fixtures in
  `tests/bounded_run.rs` that build one by hand now box their `UnixStream`; nothing else in
  them changes.

**Limits, deadlines and delivery ids** (`transport.rs`).

- Every frame is bounded before its body is read.
- `Timeouts::from_plan` sets I/O to the plan's heartbeat timeout, and connect and handshake
  to the join timeout, at most 10 s each.
- `Sequenced` carries strictly consecutive delivery ids. A repeated, regressed or skipped
  id, or a frame without an id, refuses the channel for good. The frame limit covers the
  8-byte id.

Tests in `tests/bounded_transport.rs` (9), loopback TCP with in-test self-signed Ed25519
certificates:

- `mutually_pinned_tls_carries_bounded_frames` — positive control. Frames flow both ways,
  and each peer is identified by its pinned certificate. Credentials load from PEM. A key
  readable by group or others, a symlinked key, a non-key and a non-certificate are refused.
- `a_wrong_certificate_is_refused_in_both_directions` — an unpinned client and an unpinned
  server are both refused, with a positive control.
- `expired_and_not_yet_valid_certificates_are_refused` — client and server, even though
  pinned.
- `oversize_frames_are_refused_before_their_body_is_read` — including the sequenced frame
  bound.
- `silent_peers_time_out` — a TCP peer that never starts the handshake, an authenticated
  peer that sends nothing, and a closed port.
- `replayed_bytes_and_delivery_ids_are_refused` — the recorded bytes of an authenticated
  session, replayed on a new connection, cannot complete a handshake. A duplicated delivery
  id refuses the channel for good, and so do a skipped id and a frame too short to carry one.
- `the_plan_pins_every_host_and_the_coordinator` — including a shared certificate and a
  coordinator outside the plan.
- `the_rendezvous_admits_only_hosts_presenting_their_planned_certificates` — the full
  coordinator (`FakeSupervisor`) over TLS. Three ranks on their own certificates are admitted
  and running. Rank 2 on host 0's valid, pinned certificate refuses the whole group
  (`wrong_participant`, rank 2): outcome refused, 60 000 committed updates, and no rank
  admitted.
- `a_hello_replayed_over_a_fresh_authenticated_connection_is_refused` — a captured hello
  replayed to a new rendezvous gives `StaleRendezvous`. A genuine rank-0 hello from host 1's
  certificate gives `WrongParticipant`.

Post-fix records:

- `w7b-D-post-20260927T001252Z` — the full gate. It adds two steps: exactly one rustls crypto
  provider, as in `deploy/ci/check.sh`, and no package added to `Cargo.lock`.
- `w7b-D-post-repeat-20260927T001125Z` — `bounded_transport`, `bounded_run` and
  `bounded_run_recovery` five times: 15 of 15 target runs passed.

**Decisions for the owner.** None blocks W7-T07; each is implemented in the fail-closed
direction.

1. **Pinned leaf certificates or a CA-issued PKI.** Pinning is what the frozen plan already
   specifies. A CA-issued leaf still works if it is presented alone, but a presented chain
   is refused.
2. **Certificate rotation.** The digest is frozen in the plan, so a rotated certificate
   needs a new plan attempt.
3. **The coordinator must be a planned host.** Otherwise it has no pinned identity.

**Open.** `gather` handshakes one connection at a time, so a slow peer can delay the others.
This record first said "though never past the join deadline"; that did not hold at
`3d12729a2`. The deadlines were per socket read, and review 5339604155 showed a peer that
trickles bytes inside every read timeout holding a handshake, and so the gathering, for as
long as it kept sending. It holds now, by construction: see
[Absolute transport deadlines](#absolute-transport-deadlines-review-5339604155) below.
Timeouts have been exercised only on loopback. Host certificate and key provisioning is an
operator step (runbook).

### Item E — W7-T05/T07 SmartCluster-backed group supervisor

**Pre-fix record** (`w7b-E-pre-20260927T001714Z`, source trace).

- The only `Supervisor` implementations were test fixtures: `FakeSupervisor` and the
  `bounded_run_recovery` process supervisor. The controller's own documentation named the
  SmartCluster-backed supervisor as missing.
- `jope-native-loop` had no rank worker and no coordinator process: `participant` and
  `coordinate` were unrecognized subcommands.

This is a new capability.

**Topology.** It is the same in the fixture and on three hosts, and SSH is never involved.

```text
coordinator host                               each rank's host
jope-native-loop coordinate --agents ──TLS──▶ jope-native-loop agent
  RemoteSupervisor (pinned certs)                SmartClusterSupervisor
                                                   │ authenticated exchange (pinned uid + exe)
                                                   ▼
                                                 scqd ── trusted-service job `jope-bounded-rank`
                                                           │
  rendezvous acceptor ◀────────────TLS─────────── jope-native-loop participant
```

On a single host, `coordinate --supervision` drives the host's own `SmartClusterSupervisor`
directly.

Source (all new): `bounded/supervisor.rs`, `bounded/agent.rs`, `bounded/participant.rs`,
`bounded/driver.rs`, plus the `participant`, `coordinate` and `agent` verbs. Small additions
elsewhere:

- `Sequenced::poll_receive`, an idle wait that consumes nothing;
- `launch::load_plan`, the plan-digest check `load_launch` already made, now shared;
- serde for `Liveness`.

- **`SmartClusterSupervisor`.** Every request goes through the repository's
  `native_scq::authenticated_exchange`, which pins the daemon's uid and executable digest. That
  file is only called, never changed.
  - **launch** refuses a rank planned on another host and a GPU rank (see the W7-T03 open
    item). It checks the host's plan artifact against the reviewed digest, then freezes the
    rank's launch record under `state/<dispatch id>/`.
  - It then submits a `TrustedService` job: the idempotency key is the dispatch id, automatic
    retry is off, the deadline is the plan's original deadline and the CPU/RAM ceilings are
    the plan's. The job carries the labels `jope.bounded.{dispatch_id,run,rank,plan_sha256}`.
    Its argv names only the frozen plan, the launch record, the outbox, the coordinator
    address, the rank's key and the host installation file.
  - **find** lists by the dispatch and plan labels, so a job submitted before a crash is found
    and never launched twice. **probe** maps the job's terminal status to `Liveness`.
    **terminate** cancels gracefully and waits at most 60 s. **retained_result** reads the
    rank's outbox with bounded, no-symlink reads.
  - **It acts only on its own plan's jobs.** A handle whose job has another tenant, project or
    plan digest is refused before anything is read or cancelled.
  - SmartCluster requires a job's tenant to be exactly the submitting peer's principal
    (`unix:<uid>`). The first fixture run was refused with "job tenant must exactly match the
    authenticated effective principal".
- **`participant`**, the rank worker:
  - it loads only the frozen plan and its launch record, and checks its own visibility first.
    A CPU rank accepts `""` or SmartCluster's `-1`; a GPU rank is refused;
  - it observes its own identity exactly as the census collector does, with `/proc/self/exe`
    as the executable, and hashes its input closure;
  - it joins the rendezvous over pinned TLS. Rank 0 carries a 32-byte group token, never an
    NCCL id;
  - it publishes `outbox/admitted`, then runs one SHA-256 update round per strictly increasing
    heartbeat, and signs its result (`jope.native_cpu_proof_result.v1`, `numerical: false`);
  - losing control ends it with exit 75 and no result. A refusal before admission is written
    to `outbox/refusal.txt`, a diagnostic that is never a result.
- **`coordinate`** (`driver::coordinate`) has two paths.
  - After a restart it only reconciles.
  - Otherwise it checks the heartbeat against the plan, then checks the coordinator key and
    the TLS certificate against the plan, and journals a preflight of the merged census. Only
    then does it open the rendezvous and dispatch every rank.
  - It then gathers over TLS. A refusal is journaled and dispatches nothing more. It journals
    the admission before delivering it, then heartbeats and polls until the group commits or
    drains.
- **`agent`** (`agent.rs`) serves one host's supervisor to the coordinator.
  - It holds its own copy of the frozen plan, by digest. It serves only its planned ranks, and
    only under the dispatch ids its plan derives.
  - Its listener presents this host's planned certificate and accepts only the plan's
    coordinator certificate. Requests travel on a `Sequenced` channel (16 MiB frames).
  - Between requests, a newly authenticated coordinator connection replaces the old one, so a
    restarted coordinator reconnects at once.
  - On the coordinator, `RemoteSupervisor` presents only the coordinator's planned
    certificate. It needs an agent for every rank and reaches each rank only through the
    planned certificate of that rank's host. Each reply has a bounded deadline: 30 s for a
    launch, 90 s for a terminate, and otherwise the plan's I/O deadline.

**Finding: an embedded `scqd` runs one job at a time.** Without delegated cgroups, the daemon
starts in `IsolationMode::SingleAllocation` (`cgroup.rs`, `initialize`), and its scheduler then
starts nothing while any job is active (`actor/scheduling.rs`).

The first fixture put all three ranks on one daemon. Rank 0 ran while ranks 1 and 2 stayed
queued, and the rendezvous refused the group with `join_timeout`, missing ranks 1 and 2. That
refusal is the whole-group rule working.

The fixture therefore gives each rank its own daemon and agent. That is the three-host
topology: one rank per host, each host's daemon behind its agent. On a real host with delegated
cgroups, one daemon runs several ranks at once.

Tests:

- `tests/bounded_smartcluster.rs` (4). Each rank has a real unprivileged embedded `scqd`,
  built from this workspace if absent, running the real participant binary as a trusted
  service. A real `jope-native-loop agent` process serves each daemon. The census is collected
  from the host installation.
  - `smartcluster_ranks_commit_the_whole_group_exactly_once` — the group commits once
    (60 005 updates) and all three results verify, with no orphan. A second `coordinate` on
    the committed run appends nothing. A supervisor bound to another plan artifact on the same
    daemon neither finds rank 0's job nor probes, cancels or reads it.
  - `a_killed_rank_drains_the_whole_group_through_smartcluster` — SIGKILL of rank 1's
    participant, which rank 1's daemon started: rank 1 is failed, ranks 0 and 2 are drained,
    nothing is committed, no results directory exists, and no orphan remains.
  - `cancellation_drains_the_whole_group_through_smartcluster` — `cancel` reaches the live
    coordinator, and every rank is drained.
  - `a_restarted_coordinator_reconciles_and_drains_with_no_orphan` — the `coordinate --agents`
    CLI is SIGKILLed after admission. The ranks lose control and stop. A restarted CLI
    reconciles every rank through the same agents: drained, the original deadline kept,
    `coordinator_restarted` journaled, the run verified, no orphan.
- `tests/bounded_agent.rs` (3), an in-memory supervisor over loopback TLS:
  - `an_agent_serves_only_its_own_plan_and_planned_ranks` — covers another digest, a host not
    planned for the ranks, another rank's dispatch and a forged dispatch id. The supervisor's
    own refusal is returned as a reply, and nothing is launched.
  - `remote_supervision_is_mutually_pinned_by_the_plan` — an agent presenting another host's
    certificate is refused. So is a planned non-coordinator client, which launches nothing. A
    remote supervisor presenting a non-coordinator certificate is refused, as is one missing a
    rank's agent or naming another plan. A rank routed to the wrong host's agent is refused
    by the pin. Forged handles and unplanned dispatch ids are refused before anything is sent.
    Terminate and retained results travel back, and a reconnecting coordinator replaces the
    old connection.
  - `an_agent_refuses_malformed_and_replayed_requests` — a malformed request gets a refusal
    reply. A replayed delivery id and an oversize frame end the connection, and nothing is
    launched.

Post-fix records:

- `w7b-E-post-20260927T003907Z` — the full gate. Affected targets: `bounded_agent` 3,
  `bounded_census` 8, `bounded_launch` 5, `bounded_preflight_cli` 3, `bounded_run` 30,
  `bounded_run_recovery` 12, `bounded_smartcluster` 4 and `bounded_transport` 9. It also ran
  the native lib tests (30), the fence suites (16, 8, 17 and `gp::fit_contract_tests` 39, 3
  ignored), workspace check, fmt, diff check, coverage, test inventory, the parity matrix,
  the single rustls provider and no lockfile package.
- `w7b-E-post-repeat-20260927T004337Z` — `bounded_smartcluster`, `bounded_agent` and
  `bounded_run_recovery`, five rounds each: 15 of 15 target runs passed.
- `w7b-E-post-neural-20260927T004437Z` — `neural`, `CUDA_VISIBLE_DEVICES=""`, run alone:
  `bounded_assignment` 5 and `bounded_launch` 5.

**Decisions for the owner.**

1. **Trusted-service job or the `smartcluster_worker.rs` worker path.** The brief names
   `smartcluster_worker.rs`. Its worker (`serve`) requires a root-owned policy at
   `/etc/jope-smartcluster/worker.json`, a delegated fenced cgroup, `SCQ_CONTROL_FD` and a
   fixed `Operation` set. None of that can run unprivileged here (no sudo), so no same-host
   fixture could exercise it.
   - What is implemented: each rank is a `TrustedService` job of the host's `scqd`. The
     supervisor uses the same pinned daemon exchange (`native_scq::authenticated_exchange`)
     that `smartcluster_worker.rs` uses.
   - What this forgoes: the participant does not verify, as the worker path does, that it
     runs inside its own fenced attempt.
   - Options:
     - (a) keep trusted services and add an `SCQ_JOB_ID`/label self-check to the participant;
     - (b) add a bounded-rank `Operation` to `smartcluster_worker.rs`. That needs the root
       policy and delegated cgroups on each host, and a privileged fixture.
2. **Daemon registration.** Each host's `scqd` configuration must map `jope-bounded-rank` to
   the installed `jope-native-loop`. The operator's uid must be an admin uid, because the rank
   jobs run as tenant `unix:<operator uid>`. The owner decides which account that is.
3. **One rank per host, or delegated cgroups.** Several ranks on one host need a daemon with
   delegated cgroups. The W7-T07 CPU proof has one rank per host.
4. **The `scqd` binary in CI.** `bounded_smartcluster` builds `scqd` with a nested
   `cargo build --locked --offline -p jain-smartcluster-daemon` when the target directory
   lacks it. The first run in a fresh CI target pays that build. The alternative is to build
   `scqd` explicitly before the test step.
5. **GPU ranks** stay refused by the supervisor until the W7-T03 visibility decision is made.

**Open.** Nothing ran on another host. The agent has no signal handler: SIGTERM ends it, and
it holds no state because the daemon owns the jobs. The CPU workload is a lifecycle proof,
not training.

### Item F — W7-T06 the frozen plan selects the training partition

**Stopped at an owner decision.** The plan now selects its partition. Writing a
plan-selected `run.json` needs a choice of launcher first; the options are below.

**Pre-fix record** (`w7b-F-pre-20260927T004609Z`).

- `controller.rs:649` writes the v1 partition record unconditionally. `train_controlled`
  refuses three ranks as "unsupported training topology", so no launch selects v2.
- A frozen plan could not select a partition. A plan carrying a `training` section is refused
  as ``unknown field `training` ``. This reproduction test fails.
- With the code from before the change, the canonical bytes of a fixed-time three-rank plan
  hash to `f73eaa7e1a0cbc3b01e6d50039fd194cc89ebd2011daed613931e9d0869cc976`. That digest was
  captured then and is now pinned in `a_plan_that_selects_no_training_keeps_its_exact_bytes`.

Source: `bounded/contract.rs` only (default features).

- `TrainingSelection { execution_partition, microbatch, global_batch }` is the optional
  `training` field of `BoundedRunContract`. It is never serialized when absent, so every
  earlier plan keeps its exact bytes and digest. An explicit `null` selects nothing, and Plan
  freezes the same bytes as a plan without the field.
- **Validation** (`validate_training`, part of `validate_structure`) mirrors the trainer's
  partition gate (`training::partition::execution`, outside batch qualification):
  - the microbatch must be 64, 128 or 256, and the global batch 2048, 4096, 8192 or 16384;
  - v1 needs 1, 2, 4 or 8 ranks and a whole division. Three ranks are refused with "select
    partition v2";
  - v2 needs one to eight ranks;
  - any other schema is refused.
- `execution_partition()` derives the record (`task_contract::training_partition_of`) for
  exactly the plan's world size. For three ranks at 256/2048 that is the v2 record with rows
  688, 680 and 680.
- A later attempt that changes, adds or removes the selection is a topology change
  (`validate_successor`), never an exact continuation.
- The Plan review gains a `training` key only when a partition is selected.

Tests:

- `tests/bounded_run.rs`, 3 new, 33 in all:
  - `a_plan_that_selects_no_training_keeps_its_exact_bytes` — pins the digest captured before
    the change;
  - `a_plan_can_select_the_training_partition` — covers v2 for three ranks, the frozen
    canonical bytes and `plan.json`, the review, and v1 for two ranks;
  - `a_training_selection_the_trainer_cannot_run_is_refused` — v1 for three ranks, an unknown
    schema, four bad batch shapes, v2 for nine ranks, an unknown selection field and an
    explicit null. A successor attempt that changes or drops the selection is refused.
- `tests/bounded_assignment.rs` (`neural`, CPU), 1 new, 6 in all:
  `a_plan_selects_exactly_the_partitions_the_trainer_runs`. For worlds 1 to 8, both schemas, 5
  microbatches and 6 global batches, the plan admits a selection exactly when the trainer's
  gate runs it. Every admitted record equals the gate's record, and `Execution::declared`
  accepts it with the same version and lossless rows.

Post-fix records:

- `w7b-F-post-20260927T005419Z` — the full gate, with `bounded_run` at 33 and every other
  E target unchanged.
- `w7b-F-post-neural-20260927T005022Z` — `neural`, `CUDA_VISIBLE_DEVICES=""`, run alone:
  `bounded_assignment` 6, `bounded_launch` 5, `lossless_partition` 10 and
  `partition_wiring` 7.

**Decision for the owner: which launcher writes a plan-selected `run.json` and runs its
ranks.**

The only writer of a training `run.json` is `controller::train_controlled`, the single-host
launcher. Its ranks (`train-rank`) bind devices by the legacy single-host mapping and meet
through the shared `nccl.id`. Giving it a bounded plan would ignore four things:

- the plan's hosts;
- its device UUIDs;
- its grants and admission;
- item C's rule that a bounded plan never uses `nccl.id`.

So no plan reaches `train_controlled`, and its v1 `run.json` is unchanged. The options:

1. **Bounded ranks train.** A training run spec is frozen from the plan:
   - the partition comes from `execution_partition()`;
   - the deadline is the plan's original deadline;
   - the step window comes from the counters.

   Each participant then enters `rank_impl` with the assignment from `bound_worker`, the
   admitted bootstrap from `nccl_bootstrap` and the plan's `partition::Version`. This is
   consistent with items B and C. It can only be exercised in the GPU phase (W7-T08/T09),
   after the W7-T03 SmartCluster GPU-visibility decision.
2. **`train_controlled` accepts a frozen single-host plan**, with every participant on this
   machine and boot. It would write a v2 `run.json`, set each rank's `CUDA_VISIBLE_DEVICES`
   through `launch::worker_command`, and run a local rendezvous for the bootstrap. That
   duplicates the coordinator.
3. **Not taken.** `train_controlled` takes only the partition from the plan and keeps its
   legacy binding and `nccl.id`. It would bypass plan admission.

### Item G — operator runbook for the three-host CPU proof

`docs/runbooks/w7-three-host-cpu-proof.md`. It covers:

- **§0** the owner decisions and admission;
- **§2** the build, installing `jope-native-loop` on each host and verifying it by digest,
  and the `scqd` pin and trusted-service registration;
- **§3** OpenSSL Ed25519 host certificates and keys;
- **§4** the installation file and input closure;
- **§5** census collection on each host and the pinned merge;
- **§6** a `jq` contract template for 4 CPU, 8 GiB and 15 minutes per rank, the grant
  signing procedure for the owner's key, and plan distribution by digest;
- **§7** preflight and admission;
- **§8** the supervision files and agents;
- **§9 to §12** four runs: commit, cancel, coordinator restart and a substituted input;
- **§13** no-orphan inventories;
- **§14** the receipts to retain.

Nothing in it has run on another host.

**Local dry run** (`w7b-G-dryrun-20260927T005826Z`; the script is kept with the record). It
ran §1 to §9 and §13 on this build host, which is xbabe2, as a fixture:

- one rank and loopback addresses;
- five updates instead of 540;
- a scratch embedded `scqd`, never the installed daemon;
- a throwaway grant-authority key and the lane's debug binary.

What it showed:

- OpenSSL-generated Ed25519 certificates and PKCS#8 keys load, and pin through the TLS
  transport.
- A grant signed with `openssl pkeyutl -rawin` over the documented message verifies at Plan.
- The census receipt reads `not_queried_cuda_visible_devices_empty` with 0 GPUs; `preflight`
  issues a receipt; the agent reports `listening`.
- `coordinate` ends `committed` with 5 updates, and `verify` counts 1 result.
- The participant's result is a `jope.native_cpu_proof_result.v1` document with
  `numerical: false`, and no process is left behind.

Reviewing the procedure against the code, not the dry run, found three further defects. All
are fixed in this commit:

- a census file name that could not be reused across collections;
- build receipts written into the checkout;
- private keys created before their mode was restricted. Keys are now generated under
  `umask 077`.

**Open for W7-T07 (live).**

- The live runs need the §0 admission.
- The daemon registration and admin uid are owner and operator changes on each host.
- Contract authoring and grant signing were assembled by hand at this point (`jq` and
  `openssl pkeyutl`). W7-D6 below replaced both with repository verbs: `contract author`,
  `grant sign` and `grant verify`. The owner still signs, with the grant-authority key, through
  `grant sign`.
- Reservations are checked by hand.

## Owner decisions 2026-09-27 (W7)

The owner delegated these decisions on 2026-09-27. Each takes the fail-closed option and each
can be reversed. They are implemented on `rr2/w7-live` as separate commits, and their custody
evidence uses the `w7c-<item>` prefix.

### W7-D1 — ranks stay trusted-service jobs, and each proves its job first (item E)

**Decision.** Ranks stay trusted-service jobs of each host's `scqd`. Before any work, a
participant proves it is running inside its planned SmartCluster job and attempt. If it cannot
prove that, it refuses.

**Pre-fix record** (`w7c-D1-pre-20260927T011116Z`). A participant started by hand, outside
SmartCluster, with valid frozen inputs ran past every check until its TCP connect failed
(`connect 127.0.0.1:…: Connection refused`). Nothing tied it to a job.

**Implementation** (`bounded/job_identity.rs`, called first in `participant::run`). The proof
uses only what `scqd` exposes to a job.

1. **Job environment.** `SCQ_JOB_ID`, `SCQ_ATTEMPT_NUMBER`, `SCQ_ATTEMPT_FENCE` and
   `SCQ_CONTROL_FD` must all be present and well formed.
2. **Control socket.** The peer of the inherited control socket must be this process's parent,
   running as the pinned daemon uid from the pinned daemon executable. The socket is inspected
   through a private duplicate. `scqd`'s exec helper replaces itself with the job program, so
   the daemon is the parent of the job leader.
3. **The daemon's record of the job.** The same daemon process, asked over its authenticated
   socket (`native_scq::authenticated_exchange`, pinned to that process identity), must report
   the job:
   - as live (`Starting` or `Running`);
   - with exactly this attempt number and fence as its active attempt;
   - with tenant `unix:<this process's euid>`;
   - labelled with this plan digest, run, rank and dispatch id;
   - running the host's trusted service with exactly this process's arguments.
4. **The host installation** (`jope.native_host_installation.v1`) now pins the daemon and the
   service (`smartcluster: {daemon: {socket, uid, executable_sha256}, service}`).
   `SmartClusterSupervisor::new` refuses a supervision whose daemon or service differs from
   what the installation pins.
5. **The verified job, attempt, fence and tenant** are recorded in the rank's signed result
   (`smartcluster_job`).

`scqd`'s `InspectAttempt` would also name the attempt's live process, but it refuses in
single-allocation mode ("live attempt cgroup absent"). The control-socket parent check is the
process binding that works in both isolation modes.

**Tests** (`bounded_smartcluster`, now 5).

`a_participant_proves_its_smartcluster_job_before_any_work` runs against a real daemon:

- **Genuine job (positive control).** A job submitted to rank 0's daemon with rank 0's frozen
  launch passes the proof. It is refused only later, at the connect, because no coordinator
  is listening.
- **Wrong job.** The same launch, labelled for rank 1's dispatch, is refused with "was
  submitted for another dispatch".
- **Different plan.** The same launch, labelled for another plan digest, is refused with "was
  submitted for another plan".
- **Outside SmartCluster.** Started by hand, the participant is refused with "SCQ_JOB_ID is
  unset".
- **Forged environment.** With the genuine job's environment but no control socket, it is
  refused with "is not an open descriptor".
- **Impostor daemon.** A control socket whose peer is the test process is refused with "is not
  the pinned SmartCluster executable".
- **Record checks.** The genuine job's real record, made live again, is accepted. It is
  refused for:
  - a replayed attempt (another fence, or another attempt number);
  - a job that is no longer live;
  - another tenant, another service or other arguments;
  - another job's answer.
- **Supervision mismatch.** A supervision whose daemon is not the installation's is refused.
- **No work started.** No refused job wrote `admitted`.

`smartcluster_ranks_commit_the_whole_group_exactly_once` now also checks that each result
names its own job and attempt.

Post-fix record: `w7c-D1-post-20260927T011514Z`, the full gate (`bounded_smartcluster` 5).

### W7-D3 — SmartCluster's GPU allocation stays authoritative (item B)

**Decision.** SmartCluster's allocation is not bypassed. The rank translates the device index
SmartCluster gives it into a GPU UUID and refuses unless that UUID is the planned one. It does
this before creating any communicator. CPU ranks are unaffected.

**Pre-fix record** (`w7c-D3-pre-20260927T011912Z`).

- `verify_worker_visibility` refused any index ("selects a device by local index, not by its
  planned UUID").
- `SmartClusterSupervisor` refused every GPU rank: "SmartCluster cannot pin the planned device
  UUID, so GPU ranks are not launched through it".

**Implementation.**

- **The supervisor** now asks SmartCluster for one exclusive GPU (`count: 1`,
  `compute_share_per_mille: 1000`, `Exclusive`) with the rank's planned VRAM ceiling.
  SmartCluster chooses the device. A GPU rank without a VRAM ceiling is refused.
- **`launch::resolve_visibility(record, CUDA_VISIBLE_DEVICES, runner)`** handles each case:
  - a CPU rank keeps the existing rule (`""` or `-1`);
  - a GPU rank whose value is exactly the planned UUID needs no translation;
  - a GPU rank given one SmartCluster index (1 to 4 digits) is translated with
    `nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits`, through an injectable
    command runner, and must be exactly the planned UUID.
- **`ResolvedVisibility::pinned()`** is the UUID the worker pins as its own
  `CUDA_VISIBLE_DEVICES` before any CUDA call. CUDA then selects the device by UUID, and
  `CUDA_DEVICE_ORDER` never matters. After pinning, `bound_worker`'s check of the device view
  (the planned UUID at ordinal 0) still runs at the device boundary.
- **The participant** calls `resolve_visibility` right after its job proof (W7-D1).

**Tests** (`bounded_launch`, now 7; CPU only, injected `nvidia-smi`, no GPU).

- `a_smartcluster_gpu_index_must_translate_to_the_planned_uuid`:
  - **Accepted:** SmartCluster's index 2, which is the planned GPU, gives the pinned UUID; the
    launcher's exact UUID needs no probe; a CPU rank never runs `nvidia-smi`.
  - **Refused:**
    - an index for another GPU, or one `nvidia-smi` does not report;
    - nine malformed values (several devices, `-1`, padded, signed, hex, empty, over-long);
    - an unset value, or another UUID;
    - an absent or failing `nvidia-smi`;
    - malformed lines;
    - a repeated index or UUID.
- `smartcluster_allocates_a_gpu_rank_exactly_one_exclusive_gpu` — the exact GPU request for a
  GPU rank, none for a CPU rank, and refusal of a GPU rank without a VRAM ceiling.

Post-fix record: `w7c-D3-post-20260927T012149Z`, the full gate (`bounded_launch` 7,
`bounded_smartcluster` 5).

### W7-D2 — bounded ranks train (item F)

**Decision.** Bounded ranks train. A bounded rank's entry into the trainer receives four
things:

- the admitted plan;
- its frozen launch record;
- the admitted NCCL bootstrap;
- the partition the plan selects (v2 when the plan selects it).

It writes the v2 `run.json` from the plan. `train_controlled` stays on the legacy v1 path,
untouched, and plans that select no partition keep their exact bytes.

**Pre-fix record** (`w7c-D2-pre-20260927T012445Z`, source trace).

- Every `rank_impl` caller passed no assignment and no admitted bootstrap.
- The only writer of a training `run.json` was `train_controlled`, which writes v1.
- The participant refused every GPU rank.
- A plan could not name the corpus or parent checkpoint it trains from.

This is a new capability.

**Implementation.**

- **`TrainingSelection` (contract)** gains three fields:
  - `corpus`, whose `index.json` must be in the input closure;
  - `checkpoint_steps`, in `1..=planned_updates`;
  - an optional `initialize`, a parent checkpoint whose `state.json` must be in the closure.

  A fresh start without a parent must be at 0 committed updates, and every rank of a training
  plan must have a GPU. Plans without a selection are byte-identical: the digest pinned by
  `a_plan_that_selects_no_training_keeps_its_exact_bytes` is unchanged.
- **`bounded/training_run.rs`** (default features, CPU):
  - `training_run_spec(plan)` is the `run.json` (`jope.bounded_training_run.v1`). It holds
    the plan digest, run, attempt and world, the selected execution-partition record, the
    corpus and parent digests, and the step window from the counters. It also holds the
    checkpoint cadence, the original deadline, each host's identity, the `frozen_plan`
    assignment and the `admitted_bootstrap` communicator.
  - `prepare(plan, record, bootstrap, input_root)` does the following, in order:
    1. checks the launch record against the plan;
    2. checks that the admitted bootstrap is this plan's, attempt's, rank's and world's;
    3. checks that this host's corpus index and parent state are the planned bytes, and that
       the parent is at the plan's committed start;
    4. publishes `run.json` without replacement, so every rank on a host writes the same
       bytes, and refuses another plan's;
    5. refuses a shared `nccl.id` in the run directory;
    6. creates a fresh `launch-<dispatch id>` directory. It refuses an existing one, and
       names the refusal when it holds a `train_controlled`-style `nccl.id`.
- **`training::bounded_rank`** (`neural`) enters `rank_impl` with:
  - the prepared inputs;
  - `Some(assignment)` and `Some(admitted)`, so `communicator_source` admits only the admitted
    bootstrap and `Nccl::init_admitted` creates the communicator;
  - the plan's `partition::Version`;
  - the control stop flag.

  It returns the completion receipt.
- **The participant**, for a plan that selects training:
  - it refuses at once in a build without the `neural` feature;
  - after its job proof (W7-D1) and visibility translation (W7-D3), it pins its planned UUID
    as its own `CUDA_VISIBLE_DEVICES`;
  - rank 0 bootstraps with a fresh NCCL unique id, and every rank's hello names its planned
    device;
  - after admission it runs `nccl_bootstrap`, then `prepare`, then `bound_worker` (the device
    view), then `bounded_rank`;
  - a watcher thread stops training on a stop, a lost heartbeat or end of stream, and then no
    result is published;
  - on success it signs a `jope.native_training_result.v1` result that names the `run.json`,
    partition, completion and parameter digests.

**Tests.**

- `tests/bounded_training.rs` (new, default features, 6):
  - `a_bounded_rank_writes_the_v2_run_json_from_its_plan` — for all three ranks: the exact
    `run.json` content, the v2 record with rows 688/680/680, the step window 60 000 → 60 003,
    fresh launch directories, and the v1 record for a two-rank v1 plan. A plan without a
    selection writes nothing.
  - `the_plan_and_the_admitted_bootstrap_must_agree` — another rank's bootstrap, another
    plan's bootstrap and a forged launch record are refused, and nothing is written.
  - `a_train_controlled_style_shared_nccl_id_is_refused` — an `nccl.id` in the launch or run
    directory, a reused launch directory and another plan's `run.json` are refused.
  - `a_rank_trains_only_from_the_planned_input_bytes` — an altered or missing corpus index,
    and a parent at another step, are refused.
  - `a_training_selection_names_its_corpus_parent_gpus_and_cadence` — the contract refusals,
    with a fresh start at 0 accepted.
  - `a_default_build_participant_refuses_a_training_plan_before_any_work` — the real binary.
- `tests/bounded_assignment.rs` (`neural`, CPU, now 7):
  `a_bounded_training_rank_takes_only_the_admitted_bootstrap_into_the_trainer`. For every rank,
  the prepared partition is the trainer's declared v2 and `communicator_source` gives
  `Admitted`. A missing bootstrap and an `nccl.id` in the launch directory are refused, and so
  is another rank's bootstrap.
- `bounded_run`'s F tests now select a corpus and parent.

**Device boundary.** The native path is compile-checked in the `neural` build and has not
run: the NCCL unique id, the device view, `rank_impl` and the communicator. No training step
ran, and no GPU was used.

Post-fix records:

- `w7c-D2-post-20260927T013044Z` — the full gate (`bounded_training` 6, `bounded_run` 34,
  `bounded_smartcluster` 5).
- `w7c-D2-post-neural-20260927T012946Z` — `neural`, `CUDA_VISIBLE_DEVICES=""`, run alone:
  `bounded_assignment` 7, `bounded_launch` 7, `bounded_training` 5 (the default-build-only
  refusal test is not compiled in `neural`), `checkpoint_commit` 5, `lossless_partition` 10,
  `partition_wiring` 7 and `trainer_topology` 7. It compiled the neural participant path and
  `training::bounded_rank`.
- `w7c-D2-post-neural-20260927T012830Z` is marked `SUPERSEDED`. It ran the default-build-only
  test under `neural`, where the binary goes on past the refusal that test expects.

### W7-D6 — repository verbs author contracts and sign and verify grants

**Decision.** Add repository verbs so that nothing is assembled by hand. They reuse the
existing grant format and verification code. The key is read only from a 0600 file path,
never from arguments or the environment, and is never logged.

**Pre-fix record** (`w7c-D6-pre-20260927T013453Z`).

- `jope-native-loop contract author`, `grant sign` and `grant verify` were unrecognized
  subcommands.
- No source outside the test fixtures called `sign_grant`.
- The previous runbook assembled the contract with `jq` and signed each grant with
  `openssl pkeyutl`.

**Implementation** (`bounded/authoring.rs` and three new verbs).

- **`contract author --request R --census C --input-root D --out F`.**
  - It reads a small declarative request (`jope.native_bounded_contract_request.v1`): the run,
    the keys, the window and grant lengths, the policies, the closure paths, and each rank's
    host, key, GPU UUID and ceilings. It also takes one per-rank work template, the counters,
    and optionally a training selection.
  - It builds the unsigned contract:
    - host identities and device indices come only from the merged census, and a host or GPU
      the census lacks is refused;
    - closure digests and sizes come only from the files under `D`; a path that is missing,
      repeated or not relative is refused;
    - every grant body is derived from its participant, with an empty signature;
    - the work census is expanded from the template;
    - grants must end within the window.
  - `validate_unsigned` then runs every contract check except the grant signatures. The output
    is a new file, never a replacement, and the verb prints the review: the digest of each
    grant body.
- **`grant sign --contract U --key K --out F`.**
  - The key comes from `participant::load_signing_key`: PKCS#8 Ed25519, owner-only mode, no
    symlink.
  - It refuses a key that is not the contract's `grant_authority_public_key`, and a contract
    that is already signed.
  - It refuses a grant whose window has already expired.
  - It signs every grant body with the existing `sign_grant`. The signed contract must pass
    `validate_at(now)`, which is Plan's full validation, before it is written as a new file.
- **`grant verify --contract F`** parses the contract, runs `validate_at(now)`, and prints the
  plan digest and each grant body digest.
- **`BoundedRunContract::validate_unsigned`** is `validate_structure` with the grant signatures
  skipped. It is used only for authored contracts. `parse`, `plan` and every other path keep
  verifying signatures.

**Tests** (`tests/bounded_grants.rs`, new, 5).

- `authored_contracts_are_signed_and_verified_by_the_grant_authority` — round trip.
  - Identities and devices come from the census, and the closure digests from the files.
  - The unsigned contract is not a plan.
  - Signed, it verifies and plans under the same digest.
  - The unsigned review names the same grant bodies.
- `a_wrong_key_a_tampered_grant_and_an_expired_window_are_refused` — covers:
  - a wrong key;
  - a grant altered after signing, and a signature moved to another rank;
  - verification after expiry, and signing after expiry;
  - re-signing an already signed contract.
- `the_authority_key_file_must_be_owner_only_and_not_a_symlink` — modes `0640`, `0644`, `0604`
  and `0660`, a symlink and a non-key are refused; `0600` loads.
- `authoring_refuses_what_the_census_and_inputs_do_not_show` — an unknown host, an unknown
  GPU, and a missing, duplicate or escaping closure path are refused. So are a grant longer
  than the window, another request schema, and a coordinator key that is the authority key.
- `the_cli_authors_signs_and_verifies_without_printing_the_key` — the real binary runs author,
  then sign, then verify, and neither the seed nor the key bytes ever appear in stdout or
  stderr. It also checks:
  - an output is never replaced;
  - a key with mode `0644` is refused, and nothing is written;
  - there is no argument that takes a key value.

Post-fix record: `w7c-D6-post-20260927T013703Z`, the full gate (`bounded_grants` 5).

### W7-D5 — SmartCluster setup in the runbook; no `scqd` build inside a test

**Decision.**

- The runbook's §0 carries a checklist, with exact verification commands, for four things:
  the daemon registration, the admin uid, the tenant, and one rank per host without delegated
  cgroups.
- CI must not build `scqd` inside the test on the first run in a lane that has a time budget.

**Pre-fix record** (`w7c-D5-pre-20260927T014011Z`, source trace).

- `bounded_smartcluster`'s `scqd_binary()` ran a nested
  `cargo build --locked --offline -p jain-smartcluster-daemon --bin scqd` when the target
  lacked `scqd`.
- The lane that runs the test, `full-default` (`timeout … 1500s`), had no `scqd` step.

**Implementation.**

- **The test never builds `scqd`.** `scqd_binary()` returns the target's own `scqd`, or fails at
  once with the exact build command.
- **The test sits in the lane that already builds `scqd`.**
  - `full-default` runs `cargo nextest` over the whole workspace.
  - `jain-smartcluster-daemon`'s integration tests make Cargo build its `scqd` binary as part of
    that same test build.
  - `deploy/ci/full-lane.sh` now checks, after `nextest list` (the build) and before
    `nextest run`, that `$target/debug/scqd` exists and is executable. It records its digest
    as lane evidence (`scqd.sha256`), and refuses to run the tests without it.
  - The hosted `rust-test` lane defers `jope-parallel`, so it never runs this test.
- **Runbook.** §0.1 lists the decided facts, and §0.3 has a seven-line host checklist:
  - the trusted service maps to exactly `$BIN`;
  - the operator is an admin uid;
  - there is no `worker_uid` override, so the job tenant is `unix:<operator uid>`;
  - the operator can reach the socket;
  - the daemon mode and cgroup delegation (one rank per host unless the daemon is `full` with
    delegated cgroups);
  - the digests of the binary and the daemon;
  - the ports are free.

**Timing and proof** (`w7c-D5-post-timing-20260927T014207Z`).

1. `bash -n` and `shellcheck -S warning` pass on `full-lane.sh`.
2. `cargo test --no-run -p jain-smartcluster-daemon --message-format=json` emits
   `target/debug/scqd` as a compiler artifact. The workspace test build produces it with no
   extra step.
3. A copy of the `bounded_smartcluster` binary with no `scqd` beside it refuses in 7 ms with
   the build instruction. No nested build process ran.
4. The whole bounded family ran with the lane's own runner (`cargo nextest run -P main
   --test-threads 16`): all 10 `bounded_*` targets, 92 tests, 92 passed, in 9.98 s (10.65 s
   of wall time).

   Against the lane's 1 500 s `timeout`, that is 0.7 %. The change adds no build step: `scqd`
   was already a product of the lane's workspace test build.
5. **Not timed locally: the full `full-default` lane.** It runs `nextest` over the whole
   workspace, which `deploy/ci/rust-test-scope.sh` measures at a 72 GB test link. This shared
   host was at 90 % disk (740 GB free) with other agents' builds running, so the lane's total
   was not re-measured here, and no earlier `full-default` receipt exists on this host.

   What is measured: the change adds 0 s of build and about 10 s of test time.

The same `nextest` run found a harness bug that `cargo test` had hidden. Under nextest, the
impostor-daemon case's socket already was descriptor 3, so `dup2(3, 3)` kept close-on-exec
and the child saw no descriptor 3. The harness now clears `FD_CLOEXEC` explicitly. The first
timing run (`w7c-D5-post-timing-20260927T014103Z`, 91 of 92) is marked `SUPERSEDED`.

Post-fix record: `w7c-D5-post-20260927T014236Z`, the full gate.

### W7-D4 — pinned leaf certificates (item D)

**Decision.** Accept pinned leaf certificates. Rotating a certificate requires a new plan, and
the coordinator must be a planned host. This is documented in the runbook, and the only code
change is a missing negative test.

**What was already enforced** (item D).

- The transport accepts only a peer whose end-entity certificate SHA-256 is planned, with no
  chain and no CA.
- `coordinator_certificate` refuses a coordinator host outside the plan.
- `the_plan_pins_every_host_and_the_coordinator` covers a shared certificate and an unplanned
  coordinator.

**Missing negative, now added** (`bounded_transport`, now 10):
`a_rotated_certificate_needs_a_new_plan`.

- A host whose certificate rotated is refused by the coordinator under the old plan.
- A rotated coordinator certificate is refused by a rank still holding the old plan.
- A new plan that pins the rotated certificate has a different digest, and admits it.
- A coordinator outside the plan has no pinned identity.

**Runbook.**

- §0.1 states the decision: pinned leaves, rotation needs a new census and plan, and the
  coordinator must be a planned host.
- §3 keeps the 7-day certificate and the rotation note.

**Follow-up to W7-D6.** The review that `contract author` prints for an unsigned contract
reported `plan_sha256` as the digest of the unsigned bytes. That digest is not what `plan`
freezes. `GrantReport.plan_sha256` is now `None` for an unsigned review, and only
`grant verify` and `grant sign` report the plan digest. `bounded_grants` asserts both cases.

**Runbook dry run after W7-D1 to W7-D6** (`w7c-runbook-dryrun-20260927T014529Z`, the script
is kept with the record). It ran the updated §1 to §9 and §13 on this build host, xbabe2, as a
fixture:

- one rank and loopback addresses;
- five updates;
- a scratch embedded `scqd`, with the §0.3 checklist run against its configuration;
- a throwaway grant-authority key.

The `contract author`, `grant sign` and `grant verify` verbs replaced the hand assembly.

- The checklist's tenant, socket and binary checks pass.
- The digest `grant verify` reported equals the digest `plan` froze, and the key never
  appeared in any output.
- The participant proved its job (W7-D1) under the scratch daemon. Its result carries
  `smartcluster_job` (attempt 1, fence 1, tenant `unix:1000`).
- `coordinate` ended `committed` with 5 updates, and `verify` counts 1 result.

Post-fix record: `w7c-D4-post-*`, the full gate (`bounded_transport` 10, `bounded_grants` 5).

## Landing on main (`rr2/w7-landing`)

`rr2/w7-live` (`961ad2333`, 16 commits on `8a55c4fef`) is replayed onto main `f46a9048f`, after
PRs #114, #115 and #116, as `rr2/w7-landing`. Each commit is a `cherry-pick -x` with its
original author, and each conflict resolution is described in the landed commit's message.
`rr2/w7-live` itself is unchanged. The pre-landing record is `w7-landing-pre-20260927T191733Z`:
the conflicts as git left them, each side's diff since `8a55c4fef`, and the scripts used.

**Conflicts and how they were resolved.**

- **`docs/parity/capability-matrix.md`** (the W7-T01, W7-T05 and W7-D6 commits). Only the
  `## Counts` total conflicted with main's `learning-subcommand` rows; the rows merged cleanly.
  Each time the table was regenerated from the rows with `deploy/ci/parity-census.py check`
  (a scripted replacement of the lines it recomputes), never merged by hand. #117 (Python
  removal P1) archives that tool; its Rust port is `jope-records parity-check`. The landed tree
  on `f46a9048f` had 810 rows: moved 735, new 47, pending 1, dropped 20, lost 7.
- **`deploy/ci/full-lane.sh`** (W7-D5). It conflicted with #116's test-routing block at the
  same place. Both are kept, before `touch nextest-started`: first #116's
  `check-test-routing.sh` calls, then W7-D5's `full-default` check that `scqd` exists, with its
  digest. `rr2/python-p1-landing` had not merged, so on `f46a9048f` the `scqd` path came from
  `python3 deploy/ci/cargo-artifact.py target-dir`. Stacked on #117 it comes from
  `cargo run -q --locked -p jope-ci -- cargo-artifact target-dir` (see "Stacked on #117" below).
- **`crates/jope-engine/src/neural.rs` and `training.rs`** (`961ad2333`). PR #114 (W2-T04) and
  W7 each added a "read the verified bytes once" API for the same `rank_impl` sites. PR #114's
  API (`manifest::read_bound`, `neural::load_bound_model`, `AdamW::load_bound`,
  `AdamW::from_verified`) is kept with every caller and test. W7's `AdamW::load_from_bytes`
  (PR #114's `from_verified` with the metadata parsed first) and its retained-buffer
  `CheckpointArtifacts` are dropped, and W7's callers go through PR #114's API. The bounded
  path keeps consuming the prepared index and parent-state bytes from
  `BoundedTraining::verified_input_bytes`, which PR #114's path-based API does not cover.
  `findings/w7-corrections-20260927.md` maps each rerouted use.
- **`deploy/ci/coverage.json`, `findings/test-inventory.json` and `Cargo.lock`** merged
  textually. The landed tree was checked by `coverage.py validate` (archived by #117; its Rust
  port is `jope-records coverage validate`) and `jope-records verify-test-inventory`: every W7
  target (`bounded_census`, `bounded_launch`,
  `bounded_transport`, `bounded_agent`, `bounded_smartcluster`, `bounded_training`,
  `bounded_grants`) has its coverage item and inventory entry.

**Secret-scan false positive, fixed at the source.** `deploy/ci/secret-scan.sh` (gitleaks,
`private-key` rule) flagged the W7-T05 commit. In `participant.rs`, `load_signing_key` had the
two PEM armour literals for `PRIVATE KEY` within 64 characters of each other. No key was
ever committed. The landed W7-T05 commit now joins the label at run time from `PKCS8_LABEL`,
so the parser accepts exactly the same PEM blocks. The scanner and its allowlist are
unchanged. The finding is recorded in `w7-landing-gitleaks-pre-*`; the landed range and every
commit reachable from the head now scan clean.

**Correction in this record.** Item G's "Open for W7-T07" list said there was no repository
tool for contract authoring or grant signing. W7-D6 added those verbs, so the list now says
so. The owner still signs, through `grant sign`.

**Test moved.** `checkpoint_consumption_uses_verified_tensor_bytes_after_path_replacement` is
now in `neural::bound_load_tests`, not `tests/training.rs`. PR #114's API is crate-private,
so the test moved to reach it. It proves the same properties: the restored step, parameters
and next update are identical, the replaced files are refused, and corrupted optimizer bytes
are refused. It also replaces every file right after its check.

**Verification** (`w7-landing-post-20260927T194951Z`, at `7ee50d897`; the commit that adds this
paragraph changes no code). All ran on xbabe2, CPU only, with `CUDA_VISIBLE_DEVICES=""` for
`neural`:

- These gates pass: `cargo fmt --check`, `git diff --check`, `cargo check --workspace
  --all-targets`, `cargo check -p jope-parallel --features neural --all-targets`,
  `coverage.py validate` (0 unassigned; archived by #117, its Rust port is `jope-records
  coverage validate`), `jope-records verify-test-inventory`, and
  `deploy/ci/check.sh`. `check.sh` includes gitleaks over every reachable commit and the
  parity matrix (810 rows).
- The `bounded_*` default targets pass 99 of 99: `bounded_run` 38, `bounded_run_recovery` 12,
  `bounded_transport` 10, `bounded_census` 8, `bounded_training` 8, `bounded_launch` 7,
  `bounded_grants` 5, `bounded_smartcluster` 5, `bounded_agent` 3 and `bounded_preflight_cli` 3.
- The full lanes pass through `deploy/ci/full-lane.sh`, including #116's routing checks and
  W7-D5's `scqd` digest:
  - `full-default`: 3 584 passed, 54 skipped, in 740 s against a 1 500 s budget;
  - `full-scratch`: 2 183 passed, 27 skipped, in 669 s against 900 s;
  - `full-neural`: 810 passed, 97 skipped, in 488 s against 900 s, with neural routing
    verified for 80 contracts.
- **An earlier pass under contention** (`w7-landing-post-20260927T193031Z`, before the
  secret-scan fix, run at idle I/O priority) had three failures. All three pass in isolation
  and in the full lanes above:
  - the fence test `projected_gp_candidates_are_exactly_representable_unique_and_durable`
    reached 25 of its 100 evaluations inside its 90 s search budget;
  - `a_killed_rank_drains_the_whole_group_through_smartcluster` saw no admission within 60 s;
  - under `neural`, `teaching_store::read_once_tests::a_search_receipt_is_validated_from_the_bytes_its_ledger_commits`
    hit the default nextest profile's 360 s limit. It takes 243–290 s there; the `main`
    profile allows 1 200 s.

## Stacked on #117 (`rr2/python-p1-landing`)

PR #117 (`rr2/python-p1-landing` at `e8640b7b0`, 29 commits on `f46a9048f`) archives
`deploy/ci/cargo-artifact.py` and ports it to `cargo run -q --locked -p jope-ci --
cargo-artifact`. `rr2/w7-landing` (`97e51859b`, 18 commits on `f46a9048f`) is rebased onto it with
`git rebase --onto e8640b7b0 f46a9048f rr2/w7-landing`. Authors and author dates are kept.
`git range-diff` shows 14 patches unchanged and the 4 changed ones listed here. The
records are `w7-stack-pre-20260928T032913Z` (the old head, a verified bundle of the 18 commits,
and both bases) and `w7-stack-post-20260928T033432Z` (the verification, the range-diff, the
commit map, the record comparison and the scripts used).

**Conflicts and how they were resolved.**

- **`docs/parity/capability-matrix.md`** (the W7-T01, W7-T05 and W7-D6 commits). As on
  `f46a9048f`, only the `## Counts` total conflicted, this time with #117's rows for the
  archived Python; the rows merged cleanly. Each time the table was regenerated from the rows
  with `jope-records parity-check`, the Rust port of `deploy/ci/parity-census.py` (a scripted
  replacement of the lines it recomputes), never merged by hand. Each commit message says so.
  That tree had 811 rows: moved 355, new 48, pending 1, dropped 400, lost 7. It was #117's 803
  rows plus W7's eight `jope-native-loop` rows, and it kept #117's `jope-ci` row (`new-bin` 15).
- **`deploy/ci/coverage.json`, `findings/test-inventory.json` and `Cargo.lock`** merged with no
  conflict. Each record item was compared with a three-way merge of `f46a9048f`, #117 and W7.
  `coverage.json` matches that merge in every item but one: #117's 201 changed items, including
  its `jope-ci` package and targets, and W7's seven. The one exception is the correction below.
  `test-inventory.json` matches it in every item: #117's 182 and W7's 7. `Cargo.lock` passes
  `cargo metadata --locked`.

**W7-D5 on #117.** In the W7-D5 commit, `deploy/ci/full-lane.sh` now reads the `scqd` path from
`cargo run -q --locked -p jope-ci -- cargo-artifact target-dir`. That is the same call
`fast-lane.sh` and `retain-nextest.sh` make on #117. #116's routing block and W7-D5's check are
still in the same order, both before `touch nextest-started`.

**Correction in this record.** The same commit fixes the `coverage.json` scope of
`bounded_smartcluster`. W7-T05 wrote it, and it still said `scqd` was "built from this workspace
if absent". It now says what W7-D5 does: the fixture never builds `scqd`, `full-default`'s
workspace test build produces it, and `full-lane.sh` checks the binary and records its digest
before the tests run.

**Retired Python in lines W7 adds.** Every line added in the 18 commits was searched for Python
paths and for the `deploy/ci` tools that #117's `docs/archive/python-archive-manifest.json`
archives. Only one of those lines ran retired Python: the `full-lane.sh` line above. Four
sentences in "Landing on main" above named an archived tool as if it still ran:
`parity-census.py check`, `coverage.py validate` (twice), and the `scqd` bullet's
`python3 deploy/ci/cargo-artifact.py target-dir`. Each now says what ran at the time, that the
tool is archived, and names its Rust port. The commit messages of the W7-T01 and W7-D6 landings
still name `parity-census.py check`, which is what ran on `f46a9048f`.

**Verification** (`w7-stack-post-20260928T033432Z`, at `99b6c0aec`; the commit that adds this
section changes no code). All ran on xbabe2, CPU only, with `CUDA_VISIBLE_DEVICES=""` for
`neural`:

- The `rust-check` and `rust-test` lanes of `.github/workflows/ci.yml` ran as the workflow runs
  them: `deploy/ci/receipt.py` wrapping `deploy/ci/fast-lane.sh`. Both passed, and both receipts
  record a clean tree before and after.
  - `rust-check`: the workspace check, `systemd_unit_paths` (4 of 4) and the MSRV check.
  - `rust-test`: 1 585 of 1 585 passed, 25 skipped, over the 17 product crates, including
    `jope-ci`.
- These gates pass:
  - `cargo fmt --check` and `git diff --check` over the stack;
  - `cargo check --workspace --all-targets --locked`;
  - `cargo check -p jope-parallel --features neural --all-targets --locked`;
  - `jope-records coverage validate` (0 unassigned) and `jope-records verify-test-inventory`;
  - `deploy/ci/parity-matrix.sh`, with and without `--no-build` (811 rows; `--help` lists 55
    subcommands);
  - `deploy/ci/check.sh`, whose gitleaks run scanned 534 commits and found no leaks.
- Under `cargo nextest run -P main`, the `bounded_*` default targets pass 99 of 99. The fence
  suites pass 85 of 85: `fit_boundary` 16, `view_contracts` 17, `gp_contracts` 8 and
  `gp::fit_contract_tests` 44.
- W7-D5's block was cut from `full-lane.sh` and run as `full-default` runs it. It recorded the
  digest of `target/debug/scqd`, at the path Cargo's metadata names.
- **Not re-run here:** the full lanes (`full-default`, `full-scratch` and `full-neural`) and the
  `neural` tests. The landing verification above ran them on `f46a9048f`.

**Next rebase (superseded).** This paragraph planned the replay onto a merged #117 at
`e8640b7b0`. #117 changed before it merged: it was rebased onto main `dd49b23bc` and gained
review rounds. The stack was moved again, as "Restacked on #117 at `bcdadfc77`" below records,
and then onto main once #117 merged ("On main after #117" below). The #117 SHAs cited here are
branch SHAs.

## Restacked on #117 at `bcdadfc77` (main `dd49b23bc`)

Before it merged, PR #117 was rebased onto main `dd49b23bc` (after #118, targeted GP search) and
gained review rounds three to seven. Its head became `bcdadfc77`, 48 commits on `dd49b23bc`.
The 29 #117 commits under the old stack (`f46a9048f..e8640b7b0`) are the first 29 of those, by
subject and order; 26 keep their patch-id. `e8640b7b0` was therefore no longer on #117.

`rr2/w7-landing` kept only its own 19 commits: the 16 `cherry-pick -x` commits from
`rr2/w7-live`, identified by their `(cherry picked from commit …)` trailers, and the three
landing-record commits, which touch only this file. None of them has a patch-id on #117. They
were moved with `git rebase --onto bcdadfc77 e8640b7b0 rr2/w7-landing`, keeping authors and author
dates, to `3836c85b9`. `git range-diff` shows 16 patches unchanged. The other three, W7-T01,
W7-T05 and W7-D6, differ only in the `## Counts` total and in one added paragraph of their commit
messages. The records are `w7-restack-pre-20260928T111547Z` (a verified bundle of the old head
`ede39dd56`, the classified commit list, both bases and the scripts used) and
`w7-restack-post-20260928T111620Z` (the verification, the range-diff and the record comparison).

**Conflicts and how they were resolved.**

- **`docs/parity/capability-matrix.md`** (the W7-T01, W7-T05 and W7-D6 commits). Only the
  `## Counts` total conflicted, this time with #118's three `learning-subcommand` and three
  `pipeline-subcommand` rows; the rows and the `engine-subcommand` count merged cleanly. Each time
  the total was regenerated from the rows with `jope-records parity-check` (a scripted
  replacement of the line it recomputes), never merged by hand. Each commit message says so. That
  tree had 817 rows: moved 355, new 54, pending 1, dropped 400, lost 7. That is #117's 809 rows
  plus W7's eight `jope-native-loop` rows.
- **`deploy/ci/coverage.json`, `findings/test-inventory.json` and `Cargo.lock`** merged with no
  conflict. Each record item was compared with a three-way merge of `e8640b7b0` (the old parent),
  `ede39dd56` (the old head) and `bcdadfc77`. No item conflicts, and both files match that merge
  in every item: `coverage.json` in all 440 (W7's eight, including the corrected
  `bounded_smartcluster` scope, and #118's four new targets), `test-inventory.json` in all 325
  (W7's seven and #118's four). `Cargo.lock` adds only W7's `rustls` edge to `jope-parallel` and
  passes `cargo metadata --locked`.
- **`deploy/ci/full-lane.sh`** did not conflict. W7-D5's block is unchanged: it reads the `scqd`
  path from `cargo run -q --locked -p jope-ci -- cargo-artifact target-dir` and still follows
  #116's routing block, before `touch nextest-started`.

**Verification** (`w7-restack-post-20260928T111620Z`, at `3836c85b9`). All ran on xbabe2, CPU
only, with `CUDA_VISIBLE_DEVICES=""` for `neural`:

- These gates pass: `cargo fmt --check`; `git diff --check` over the stack;
  `cargo check --workspace --all-targets --locked`;
  `cargo check -p jope-parallel --features neural --all-targets --locked`;
  `jope-records coverage validate` (0 unassigned) and `jope-records verify-test-inventory`;
  `deploy/ci/parity-matrix.sh --no-build` (817 rows); and `deploy/ci/check.sh`, whose gitleaks
  run scanned 586 commits and found no leaks.
- The full lanes pass through `deploy/ci/full-lane.sh`, including #116's routing checks and
  W7-D5's `scqd` digest:
  - `full-default`: 3 757 passed, 65 skipped, in 682 s against a 1 500 s budget;
  - `full-scratch`: 2 193 passed, 29 skipped, in 664 s against 900 s;
  - `full-neural`: 819 passed, 99 skipped, in 431 s against 900 s.
- `deploy/ci/check-test-routing.sh` verifies 80 `cpu` and 8 `host` contracts in the
  `full-default` collection, 80 `cpu` in `full-scratch`'s and 80 `neural` in `full-neural`'s.
- In `full-default`, the `bounded_*` targets pass 99 of 99 and the fence test binaries
  `fit_boundary` 16, `view_contracts` 17 and `gp_contracts` 8.

The #117 SHAs cited in this section are branch SHAs. Once #117 merged, the stack was replayed
onto main, as the next section records.

## On main after #117 (`7aec174e0`)

PR #117 merged as main `7aec174e0` by a GitHub rebase merge, 50 commits on `dd49b23bc`. The
first 48 are `bcdadfc77`'s 48 with new SHAs, and `git range-diff` shows all 48 unchanged. The
last two are review round eight: `02d5e49ef` (exact integer arithmetic, and numbers compared as
Python did, in `jope-records`, `jope-ci` and `crates/jope-engine/src/family_growth.rs`) and
`7aec174e0` (its record).

`rr2/w7-landing` now sits directly on main. Its 19 commits were moved with
`git rebase --onto 7aec174e0 bcdadfc77 rr2/w7-landing`, keeping authors and author dates, from
`3836c85b9` to `ae04755ce`. The records are `w7-post117-pre-20260928T125735Z` (the old head's
identity, a verified bundle of `3836c85b9` over `dd49b23bc`, #117's range-diff, the old commit
list and the script used) and `w7-post117-post-20260928T125840Z` (the verification).

- **No conflicts.** Of the files W7 changes, round eight touches only
  `docs/parity/capability-matrix.md`, where it recounts the `jope-records` row's tests (150 to
  154). That row is outside the `## Counts` table, and the rebase applied every W7 patch without
  a conflict. `git range-diff` shows all 19 W7 patches unchanged, and the difference between the
  old and the new W7 head is exactly round eight's difference, file by file (in the matrix, at a
  shifted line number).
- **Records.** `deploy/ci/coverage.json`, `findings/test-inventory.json` and `Cargo.lock` are
  identical to `3836c85b9`'s: round eight changes none of them, so nothing was regenerated. The
  matrix still has 817 rows and `jope-records parity-check` finds its `## Counts` table in
  agreement with them.
- **`full-default` run again.** Round eight changes `family_growth.rs`, in the `jope-parallel`
  package that also holds W7's `native_loop`, so the `full-default` family ran again at the new
  head. `full-scratch`, `full-neural` and the `neural` check were not re-run here; they passed
  at `3836c85b9` above, and round eight changes no W7 file.

**Verification** (`w7-post117-post-20260928T125840Z`, at `ae04755ce`; the commit that adds this
section changes no code). All ran on xbabe2, CPU only:

- These gates pass: `cargo fmt --check`; `git diff --check` over the 19 commits and the worktree;
  `cargo check --workspace --all-targets --locked`; `jope-records coverage validate`
  (0 unassigned) and `jope-records verify-test-inventory`; `deploy/ci/parity-matrix.sh
  --no-build` (817 rows); and `deploy/ci/check.sh`, whose gitleaks run scanned 588 commits and
  found no leaks.
- `full-default` passes through `deploy/ci/full-lane.sh`: 3 762 passed, 65 skipped, in 686 s
  against a 1 500 s budget. W7-D5's check found `scqd` and recorded its digest, the same digest
  as at `3836c85b9`. Compared with the `3836c85b9` run, the collection gains round eight's six
  new tests (in `jope-ci`, `jope-records` and `family_growth`) and loses the one
  `family_growth` test that round eight replaced.
- `deploy/ci/check-test-routing.sh` verifies 80 `cpu` and 8 `host` contracts in the
  `full-default` collection.
- In `full-default`, the `bounded_*` targets pass 99 of 99: `bounded_run` 38,
  `bounded_run_recovery` 12, `bounded_transport` 10, `bounded_census` 8, `bounded_training` 8,
  `bounded_launch` 7, `bounded_grants` 5, `bounded_smartcluster` 5, `bounded_agent` 3 and
  `bounded_preflight_cli` 3. The fence suites pass 85 of 85: `fit_boundary` 16,
  `view_contracts` 17, `gp_contracts` 8 and `gp::fit_contract_tests` 44. The five
  `family_growth::tests` pass.

**Next rebase.** W7 now sits on main itself. If main moves before W7 merges,
`git rebase origin/main` replays the stack. The `## Counts` total can conflict again if a PR that
reaches main first changes matrix rows; regenerate it with `jope-records parity-check`. The W7
SHAs cited here are branch SHAs; a rebase merge gives them new ones.

## Absolute transport deadlines (review 5339604155)

Independent review 5339604155 of `3d12729a2` requested changes for one finding, **[P1] enforce
the absolute deadline inside TLS handshake I/O**. Its loopback probe drove the real
`TlsAcceptor` with a 200 ms handshake timeout and no client credential: a handshake record
header announcing 16 KiB, then one byte every 75 ms. `poll_accept` stayed blocked for
1 502 ms, until the peer closed. `rendezvous::gather` and `Agent::serve` call it
synchronously, so their join-deadline and stop checks waited too. The client loop had the same
shape. The review also asked for the framed reads to be audited for the same assumption, and
for the Item D claim above to be corrected.

**Cause.** Every deadline was a socket timeout, which bounds one read or one write. A peer
that sends a byte inside every timeout never trips it. The handshake loops checked the
elapsed time only between `complete_io` calls, and one call can make several blocking reads.
A framed read (`read_exact`) makes as many reads as the peer cares to split its bytes into,
and one read of a TLS stream can make several socket reads before it has a whole record.

**Pre-fix record** (`w7-deadline-pre-20260928T140929Z`). Nine new tests, written against the
existing API and run on the unchanged sources at `3d12729a2`, all fail. Each operation ended
only when its trickling peer stopped, after 2.55 to 2.60 s, against a 300 ms deadline. The 13
existing tests of the two targets pass. (`w7-deadline-pre-20260928T140700Z` is the same run
with both join phases in one test, which stopped at the first.)

**The fix.** One mechanism, used everywhere:

- `transport::DeadlineSocket` wraps every socket of the bounded transport: TCP under TLS, and
  the same-host Unix fixture sockets. Before each read or write it sets the kernel timeout to
  the smaller of the per-operation timeout and the time left before an absolute deadline. At
  the deadline it refuses the operation with `TimedOut` and does not touch the socket. rustls
  performs all of its I/O through this socket, so the deadline covers every read and write
  inside one `complete_io` step, one record read and one flush.
- `ControlStream::set_deadline` sets or clears that deadline. It is a required method, and
  the `ControlStream` implementation for a bare `UnixStream` is gone, since a bare stream
  cannot hold a deadline. The fixtures box `DeadlineSocket::new(stream)` instead.
- A TLS handshake has one deadline over all of its I/O: the handshake allowance, cut to the
  caller's deadline when there is one (`TlsAcceptor::poll_accept_by`, which the rendezvous
  uses through `Acceptor::next_incoming(deadline)`).
- Callers put one deadline on each exchange, through `rendezvous::by_deadline`,
  `read_frame_by` and `write_frame_by`, and `Sequenced::send_by`, `receive_by` and
  `poll_receive(wait, within)`.
- `write_frame` hands length and body to the stream in one write. A TLS stream therefore
  queues a small frame, such as a heartbeat or an admission, whole: a write cut short by a
  deadline cannot leave a length queued without its body.
- Dropping a `TlsStream` sends `close_notify` under a bound of its own: never past the stream's
  deadline, and at most `transport::CLOSE_TIMEOUT` (1 s) from the drop, or the write timeout if
  that is shorter. As first committed the cap was one whole write timeout from the drop; see
  [the close cap](#close-cap-verifier-note-on-c9ca8b547) below.

No allowance is extended. Every deadline is an allowance the plan or the code already names:
the handshake, join or heartbeat timeout, the heartbeat interval, or a reply bound. It is now
measured once, from the start of the exchange, instead of per read. Where a site's old
per-write timeout was the join time left at accept, which is not an allowance for that write,
the new bound is shorter. Certificate pinning, grants and every refusal are unchanged.

### Every blocking read and write of the bounded transport and its callers

Line numbers are those of the commit that adds this section.

| Site | Operation | Before | Now |
|---|---|---|---|
| `transport.rs:641` `connect` | TCP connect | `connect_timeout`, one call | unchanged: already absolute |
| `transport.rs:653` `connect` | client handshake (`complete_io` loop) | per-read timeout; elapsed checked between steps | one deadline, the handshake allowance, over all of its I/O |
| `transport.rs:736` `accept_by` (`poll_accept`, `poll_accept_by`) | server handshake | as above | one deadline: the handshake allowance, cut to the caller's deadline |
| `transport.rs:546`, `554` `TlsStream` `Read`/`Write` | record reads, writes and flushes, several socket operations each | per-operation timeout only | every socket operation through `DeadlineSocket`, bounded by the caller's deadline |
| `transport.rs:575` `Drop for TlsStream` | `close_notify` and flush | per-write timeout, per write | capped at the stream's deadline and at `CLOSE_TIMEOUT` (1 s) from the drop, or the write timeout if shorter. First committed as one write timeout from the drop; see the close cap below |
| `transport.rs:785`, `795` `Sequenced::send`, `receive` | one frame | the caller's per-operation timeouts | unchanged; production callers use `send_by` and `receive_by` (`839`, `843`) |
| `transport.rs:850` `Sequenced::poll_receive` | idle wait, then one frame | per-read timeouts, so a started frame could trickle | `wait` for the first byte; the rest of the frame within `within` of it |
| `rendezvous.rs:370`, `382` `write_frame`, `read_frame` | one frame | primitives | primitives, bounded by the stream's deadline; `write_frame` writes once |
| `rendezvous.rs:1060` `gather` | the next handshake | handshake allowance per read | ends by the join deadline |
| `rendezvous.rs:1066`-`1071` `gather` | challenge write, hello read | join time remaining, per read | end by the join deadline itself; cleared once the hello is read |
| `rendezvous.rs:1154` `join` | challenge read, hello write, admission read | join timeout per read | one deadline for the whole exchange: the join timeout, cut to the original deadline |
| `rendezvous.rs:126`, `1125` `UnixAcceptor`, `join_unix` (fixtures) | as `gather` and `join` | bare `UnixStream` | `DeadlineSocket` over the fixture socket |
| `controller.rs:1450` `deliver_admission_with_clock` | admission write, per rank | the join time left at accept, per write | the plan's I/O (heartbeat) timeout, cut to the end of the group window. Each socket write is also still bounded by the per-operation timeout `gather` left on the stream (the join time left at accept, at least 1 ms); the smaller of the two applies |
| `driver.rs:57` `heartbeat_all` | heartbeat write, per rank | the join time left at accept, per write | one heartbeat interval per write, and each socket write also within the per-operation timeout `gather` left on the stream, whichever is smaller; a failed write is still observed through the rank's own exit |
| `participant.rs:274`, `292` | connect, join | as `connect`, `join` | as `connect`, `join` |
| `participant.rs:376` CPU phase | each heartbeat read | heartbeat timeout per read | the whole heartbeat within the heartbeat timeout, or control is lost |
| `participant.rs:505` `train` watcher (`neural`) | each heartbeat read | heartbeat timeout per read | as the CPU phase |
| `agent.rs:202` `Agent::serve` | accept | handshake allowance per read | the handshake allowance, absolute (at most 10 s) |
| `agent.rs:216` `Agent::serve` | next request | 500 ms per read, so a started request could trickle | `IDLE_POLL` for the first byte; the request within the plan's I/O timeout of it |
| `agent.rs:232` `Agent::serve` | reply write | 30 s per write | `REPLY_WRITE` (30 s) for the whole reply |
| `agent.rs:309` `RemoteSupervisor::call` | connect | as `connect` | as `connect` |
| `agent.rs:337`-`338` `RemoteSupervisor::call` | request and reply | reply bound per read | the reply bound (I/O timeout, 30 s launch, 90 s terminate) over the whole exchange |
| `supervisor.rs:146` `SmartClusterSupervisor::exchange`, `job_identity.rs:234` | SmartCluster control exchange | `native_scq::authenticated_exchange` (pinned, not edited): the whole write and read inside one 30 s `tokio::time::timeout` | unchanged: already absolute. The peer is the host's own daemon, pinned by uid and executable |
| `collect.rs:78` `SystemCommands::run` (census collection) | output of a local read-only inventory program | the child is killed after `COMMAND_TIMEOUT` (30 s), measured once | unchanged: no network peer. See the open note below |
| census merge, grants (`census.rs`, `authoring.rs`, `contract.rs`) | pinned per-host documents and signed grant files | no network I/O | not applicable: nothing is exchanged over a socket |

Consequences worth stating:

- **The join deadline now bounds the gathering.** Every handshake, challenge and hello ends by
  the join deadline itself, so a slow or trickling peer can delay the other ranks but never past
  it. After it, `admit` refuses the group (`JoinTimeout`), with nothing started or committed.
- **A joining rank's whole exchange is one allowance.** Before, the challenge read and the
  admission read each had the full join timeout. A rank now waits for its admission only until
  its own join timeout, measured from when it started to join. A rank starts to join only
  after it is dispatched, which is after the rendezvous opened, so this is no earlier than the
  coordinator's own join deadline. The coordinator journals the admission before delivering it,
  so a rank that joined the moment the rendezvous opened, in a group whose last rank arrived at
  the very end of the join window, has only the rank's launch latency to cover that journaling
  step.
- **The agent sees `stop` within bounded time** whatever its peers do: after its current step,
  and no peer can stretch a step. A handshake takes at most its allowance (at most 10 s), a
  started request at most the plan's I/O timeout, and a reply write at most 30 s. Dropping the
  connection after a failed reply adds at most `CLOSE_TIMEOUT` (1 s) for its courtesy close,
  so about 31 s in all from the start of the reply. As first committed, that close could take
  one more whole write timeout, `REPLY_WRITE`: about 60 s in all (see the close cap below). The
  supervisor operation being served, such as a terminate, takes as long as it takes, as
  before. One agent serves one peer at a time, so a stream of trickling connections can still
  occupy it for one allowance each. That is a fairness limit, and it is recorded below.
- **`gather`'s per-operation timeout stays on every admitted stream.** `gather` sets both
  per-operation timeouts of an accepted stream to the join time left at its accept (at least
  1 ms), and nothing resets them. The admission delivery and `heartbeat_all` set only
  deadlines, so each socket write there ends by the smaller of that per-operation timeout and
  the time left before its deadline. For a rank accepted late in the join window that timeout
  can be a few milliseconds. A socket write waits only when the socket buffers are full, which
  a small admission or heartbeat frame to a rank that is reading its control stream does not
  reach; for a rank that has stopped reading, the write only fails sooner.

**Plan sizing.** Two allowances now bound more than they did, and a plan must be sized for
them:

- **A rank's join timeout covers its whole join.** `rendezvous::join` puts one deadline, the
  plan's `join_timeout_ms` cut to the original deadline, over the challenge read, the hello
  write and the admission read together; before `c9ca8b547` each read had the whole join
  timeout. `join_timeout_ms` must therefore cover, from when a rank starts to join, its wait for
  the challenge, its hello, the coordinator's wait for the rest of the group, the journaling of
  the admission and its delivery.
- **A retained result must arrive whole within the heartbeat timeout.**
  `RemoteSupervisor::call` bounds every request except a launch (30 s) and a terminate (90 s)
  by the plan's I/O timeout, `heartbeat_timeout_ms`, over the whole exchange, request and
  reply. A `RetainedResult` reply carries the signed envelope and the result bytes, hex-encoded,
  in one frame of up to `AGENT_FRAME_LIMIT` (16 MiB), so `heartbeat_timeout_ms` must be long
  enough for the largest retained result the plan can produce to cross the link: 16 MiB takes
  about 1.3 s at 100 Mbit/s before TLS framing. A reply that misses its deadline fails that
  call, the channel is reopened for the next one, and a partial frame is never used.

**Tests.** Nine of the ten new tests use only the API that existed at `3d12729a2`, so they ran
unchanged as the pre-fix record above. The tenth needs `read_frame_by`. All use loopback TCP
or a Unix socket, 300 ms deadlines and peers that trickle for 2.5 s. They assert that the
operation ends no earlier than 50 ms before its deadline and no later than 500 ms after it;
the agent-stop test asserts only the upper bound. That slack is for loaded CI hosts; 800 ms
is still under the review's 1.5 s and far under the 2.5 s every test measured before the fix.

- `bounded_transport` (18, was 10):
  - `a_trickling_client_cannot_hold_the_server_handshake_past_its_deadline` is the review's
    probe. A silent peer ends at the same deadline, and a genuine client then completes its
    handshake on the same listener.
  - `a_trickling_server_cannot_hold_the_client_handshake_past_its_deadline`: likewise for the
    client, with a server that never answers and a positive control.
  - `a_trickled_hello_cannot_hold_the_rendezvous_past_its_join_deadline` uses an authenticated
    rank; `a_trickled_handshake_cannot_hold_the_rendezvous_past_its_join_deadline` an
    unauthenticated peer; and `a_trickled_hello_cannot_hold_the_fixture_rendezvous_past_its_join_deadline`
    the Unix fixture transport. Each gives `JoinTimeout` at the join deadline, with nothing
    started or committed.
  - `a_trickled_challenge_cannot_hold_a_joining_rank_past_its_join_timeout` and
    `a_trickled_admission_cannot_hold_a_joining_rank_past_its_join_timeout`.
  - `an_exchange_deadline_bounds_trickled_frames_and_stalled_writes`: a trickled frame on an
    established stream; a deadline already passed, which reads nothing, after which the whole
    frame is read with no deadline (positive control); and a 64 MiB write to a peer that
    stopped reading.
- `bounded_agent` (5, was 3): `a_trickling_agent_cannot_hold_a_remote_call_past_its_reply_deadline`,
  and `a_trickling_coordinator_cannot_hold_an_agent_past_its_stop`, which requires the agent
  to stop within the plan's I/O timeout plus 500 ms of the stop flag.
- Existing silent-peer, positive-control, certificate, rotation, replay and grant refusals pass
  unchanged. In `bounded_run`, two fixtures now box `DeadlineSocket::new(a)`, and the
  `DelayedWrite` fixture implements `set_deadline`: its writes never block, and its injected
  clock is what closes the window.

**Post-fix records.**

- `w7-deadline-post-20260928T141431Z`: `bounded_transport`, `bounded_agent`, `bounded_run`
  and `bounded_run_recovery` pass 73 of 73. The ten new tests take 0.4 to 0.9 s each,
  including their positive controls. This run used a 350 ms slack, which was then widened to
  500 ms for CI; the later runs use 500 ms.
- `w7-deadline-post-repeat-20260928T141638Z`: the same four targets five more times, 73 of 73
  each time.
- `w7-deadline-negative-20260928T141745Z` is a mutation control. `DeadlineSocket` was made to
  ignore its deadline, leaving per-operation timeouts only, which is the pre-fix behaviour.
  All ten new tests then fail after 2.56 to 2.67 s, and the 13 older tests of the two targets
  pass. The between-step elapsed check in the handshake loops was still in place, and it did
  not help. `transport.rs` was then restored, byte for byte (same SHA-256), and the sources
  were touched.

**Verification** (`w7-deadline-gate-20260928T141946Z`, at `c9ca8b547`, the fix commit; the
commit that adds this paragraph changes only this record). All ran on xbabe2, CPU only:

- These gates pass: `cargo fmt --check`; `git diff --check` over the fix commit and the
  worktree; `cargo check --workspace --all-targets --locked`; `cargo clippy -p jope-parallel
  --all-targets --locked`, with no finding in any file this change touches;
  `jope-records coverage validate` (0 unassigned) and `jope-records verify-test-inventory`
  (319 items); `deploy/ci/parity-matrix.sh --no-build` (817 rows); `deploy/ci/check.sh`,
  whose gitleaks run scanned 590 commits and found no leaks; and `cargo check -p
  jope-parallel --features neural --all-targets --locked`, which compiles the training
  watcher's heartbeat read. Neither `cargo check` reports a warning in a changed file.
  No test target was added, so `coverage.json` and the test inventories are unchanged.
- The bounded and fence selection passes 194 of 194. The `bounded_*` targets pass 109: the 99
  from before, plus 8 new in `bounded_transport` and 2 in `bounded_agent`. The fences pass 85:
  `fit_boundary` 16, `view_contracts` 17, `gp_contracts` 8 and `gp::fit_contract_tests` 44.
- `full-default` passes through `deploy/ci/full-lane.sh`: 3 772 passed and 65 skipped, in
  717 s against a 1 500 s budget. That is the 3 762 of `ae04755ce` plus the ten new tests.
  `deploy/ci/check-test-routing.sh` verifies 80 `cpu` and 8 `host` contracts. W7-D5's check
  found `scqd` and recorded its digest, the same as at `ae04755ce`. The `bounded_*` targets
  pass 109 of 109 and the fences 85 of 85. Under the lane's 16 test threads the ten new
  tests took 0.38 to 0.86 s each.

**Open.**

- A single-threaded agent, and `gather`, still serve one connection at a time. A continuous
  supply of trickling connections from an address that can reach the port costs each one
  allowance, so it can keep legitimate traffic waiting. For `gather` that ends in a
  fail-closed `JoinTimeout`. For the agent, the coordinator's calls to it can fail at their
  reply deadlines. Nothing is admitted or committed that should not be. Concurrent accept, or
  a per-source limit, is a separate change.
- In `collect.rs`, `SystemCommands::run` kills an inventory program at its absolute
  `COMMAND_TIMEOUT`, then joins the threads that read its output until end of file. A
  descendant that inherited the pipe could delay that end of file. The program is a local
  read-only probe run by the operator's own collector, not a network peer, so this change
  leaves it alone.
- Loopback only, as before. The deadlines have not been exercised across hosts.

## Rebase onto `4ff4e7b69` (PR #119)

`main` moved to `4ff4e7b697dcd5b7097dacbae9e99c07ffef8e28`: `7aec174e0` plus PR #119, which
removed the inactive `ops/ci` copies, pinned `actions/checkout` to v7.0.1, gave the `feat-math`
tests `tempfile` scratch directories, updated `README.md` and `CONTRIBUTING.md`, and added a
`feat-math -> tempfile` edge to `Cargo.lock`. `git rebase --onto 4ff4e7b69 7aec174e0` replayed
the 22 W7 commits with no conflict. `Cargo.lock` is the only file both sides change, in
different packages, and the rebased lock keeps both edges: `feat-math -> tempfile` and
`jope-parallel -> rustls`. `git range-diff 7aec174e0..829a0a696 4ff4e7b69..f71dfea7a` reports
all 22 patches unchanged (`=`). No W7 file names anything #119 removed. Record:
`w7-rebase119-pre-20260928T145644Z` (`range-diff.txt`, `rebase-heads.txt`).

## Close cap (verifier note on `c9ca8b547`)

**Finding.** `Drop for TlsStream` capped its courtesy `close_notify` and flush at the stream's
write timeout, measured from the drop. `Agent::serve` gives its stream a `REPLY_WRITE` (30 s)
write timeout. After a reply write to a coordinator that had stopped reading failed at its
30 s deadline, dropping that connection could therefore wait up to another 30 s, about 60 s in
all before the agent saw `stop`, where the consequences above said 30 s. The coordinator's
`RemoteSupervisor` dropped a failed channel under the same rule, with the plan's I/O timeout.

**Fix** (`transport.rs`). The close is capped at the new public `transport::CLOSE_TIMEOUT`
(1 s) from the drop, or at the stream's write timeout if that is shorter, and never past the
stream's deadline. Before, `CLOSE_TIMEOUT` applied only to a stream with no write timeout. The
close is cut short only when the peer has not read for that long, and then the peer reads end
of stream or a reset instead of `close_notify`. Every frame is length-prefixed and read whole,
so nothing takes a stream cut short as a complete one. The `Agent::serve` documentation names
the close bound, and the consequences above are corrected. The edit keeps every line number
the table cites.

**Test** (`bounded_transport`, 19, was 18):
`dropping_a_stream_whose_peer_stopped_reading_ends_within_the_close_timeout`. Over an
authenticated pair it fills the socket buffers with a 100 ms write timeout until one write
makes no progress (the peer never reads, so TLS records stay queued), then gives the stream a
5 s write timeout and no deadline, drops it, and asserts that the drop ends at `CLOSE_TIMEOUT`:
no earlier than 50 ms before it and no later than 500 ms after it. The lower bound shows the
close really waited, so the test tells a 1 s cap from a 5 s one.

**Pre-fix record** (`w7-rebase119-pre-20260928T150209Z`, on `f71dfea7a` with only
`CLOSE_TIMEOUT` made public and the new test added). The new test fails: the drop ended after
5.12 s against a 1 s cap, the whole 5 s write timeout. The 18 other `bounded_transport` tests
pass.

**Post-fix record** (`w7-rebase119-post-20260928T150504Z`). `bounded_transport` and
`bounded_agent` pass 24 of 24 three times running, the new test included; it takes 6.5 to
6.7 s, nearly all of it filling the buffers. That run's `cargo fmt --check` asked for one
line of the fix to be split; the committed fix reads the write timeout in its own statement
instead, which keeps the line count, and is otherwise the same.

**Verifier notes also settled here.** The `gather` per-operation timeout and the two
plan-sizing constraints are recorded above, under the consequences. The SmartCluster row's
`supervisor.rs:146` was rechecked: `SmartClusterSupervisor::exchange` calls
`authenticated_exchange` on line 146 at `c9ca8b547` and at every later W7 commit, and line 151
is inside `SmartClusterSupervisor::job`. The row now names the function.
