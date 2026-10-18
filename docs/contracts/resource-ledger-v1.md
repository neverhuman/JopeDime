# Physical resource ledger v1

`jope.resource-ledger/v1` is an additive reservation policy implemented in
[`resource_ledger.rs`](../../crates/jope-pipeline/src/resource_ledger.rs). It does
not change historical search budgets, RNG choices, receipts or readers.

Each policy pins an experiment, its original opening and absolute deadline,
terminal time headroom, physical caps and withheld terminal capacity. Retrying,
restoring or repeating a reservation cannot refresh that window. Work reserves
capacity before preparation, encoding, compilation or fitting. Checkpoint,
publication and custody stages can consume the terminal reserve, within the same
original deadline. Reserved completion capacity is not a new execution grant.

## Units and provenance

The integer dimensions are CPU core-microseconds, GPU device-microseconds,
per-attempt elapsed wall-microseconds, peak memory bytes, measured memory
byte-seconds, storage bytes and operations, evaluations, compilations, custody
operations and physical retries. Divide microseconds by 1,000,000 to report
seconds. Seconds ingestion rejects negative, nonfinite and overflowing values
and rounds positive fractional microseconds upward. All arithmetic is checked.
An absent cap authorizes no consumption of that dimension.

New reservation and attempt-cost IDs use I02's length-prefixed domain-separated
SHA-256 formula with canonical JSON and distinct `jope.resource-reservation/v1`
and `jope.resource-attempt-cost/v1` domains. The dependency-free local adapter
admits integer payloads only and rejects magnitudes above `2^53-1`, preserving
exact cross-language values. Historical worker and observation receipt digests
remain unchanged as evidence references.

One ledger's GPU component refers to one exact device/MIG allocation; separate
allocations use separate ledgers. Every measurement retains its collector, clock,
physical allocation, containment scope, interval, uncertainty and either a known
value or an explicit missing reason. A final inclusive cgroup memory peak is not
an exact process RSS maximum. Sampled RSS is never substituted for either.
Memory byte-seconds are unavailable unless actually collected; no integration
from a single memory peak is performed.

CPU and GPU have separate totals. Combining them requires a versioned conversion
and immutable preregistration digest dated no later than the original opening.
Credits retain the raw physical dimensions and are unavailable while either
component has unknown telemetry or reservations are unresolved. A conversion
does not authorize GPU work, certify the preregistration's external authority or
replace device allocation identity.

The containment scope is a path within an accounting hierarchy. Overlapping
intervals from the same or an ancestor/descendant scope cannot both contribute
known positive consumption of the same dimension. Disjoint child scopes may run
concurrently. A producer must use the same hierarchy for inclusive cgroup and
process counters; renaming a scope cannot justify double counting.

## State and reconciliation

The state machine is `requested -> reserved -> started -> terminal`. A proven
unopened worker gate can instead go from reserved to `proven_not_started`.
Completed work can reference a retained result. Raw failures, logs and partial
artifacts remain evidence even when no result survives. The ledger never drops
compile failures, killed children, cancelled or interrupted attempts.

Every transition requires the existing journal's persistence callback to succeed
before updating the in-memory state. The caller owns a single journal and must
serialize it together with the ledger, including across processes. A failed
reservation write cannot authorize work; a failed terminal write cannot release
resources. Recovery replays the ordered immutable event bodies and original
policy. Duplicated delivery is idempotent; conflicting identities, starts,
outcomes, costs before reservations/starts and orphaned retries are errors.

Each reservation binds the full original policy digest (including caps and
conversion), plus I05 experiment/draw/request/attempt/logical identities.
A real retry has a new physical attempt ID, names its original failure, preserves
the logical/request/draw joins and reserves a retry count. A logical completion
does not erase its predecessor's physical cost. These opaque SHA-256 joins are
the draft I05 adapter contract; I05 remains the event authority. The coordinator
must bind the accepted I05 types and validate that the supplied IDs refer to the
intended event closure during integration.

Reconciliation exposes requested vectors, active reservations, started and
terminal attempt counts, retained-result counts, computed known components,
unknown counts, conservative charges and released allowances. Cumulative known
consumption and retained physical storage bytes are distinct: a result digest
does not establish its retained byte count. Known retained-byte subtotals and
unknown retained-byte attempt counts remain separate. Cumulative known
charges include uncertainty; missing measurements retain the full reserved
allowance. Thus `computed_known=0` with a positive unknown count means no known
subtotal, not a measured zero. Final peak memory uses a maximum, while concurrent
memory reservations use a sum. Overruns remain recorded, flag reconciliation,
and refuse further admission. They are not clipped to the cap or discarded.

An unknown attempt keeps capacity charged permanently unless a future version
adds evidence-backed reconciliation; it is not silently converted to free work.
An unresolved durable reservation stays held after a controller error. Only
explicit `not_started: gate never released` evidence with no start/exit/signal
can release a worker's allocation at known zero. That local child zero says
nothing about upstream preparation or compilation cost.

Cold and warm cache states retain a cache lifetime ID; warm state references its
original preparation-cost digest. Warm task costs add to historical lifetime
costs and cannot rewrite preparation expenditure. External cache preparation
references require closure verification by the consuming coordinator. Amortized
reports are projections, never replacement physical totals.

## Existing execution and consumer census

| Surface | v1 integration and limit |
| --- | --- |
| `search::Budget::resource_cap_v1` | Explicit resource-vector adapter; legacy admission remains nonpreemptive. |
| `search::Observation::resource_cost_v1` | Retains measured active wall and evaluation count; CPU/memory remain unknown. Caller supplies accepted I05 joins. |
| `execution::WorkerOutcome::resource_cost_v1` | Converts existing final cgroup and wall counters; retains raw worker/log/partial evidence and actual exit/signal. No OS collector is added. |
| `execution::ExecutionPool::run_accounted_v1` | Opt-in bounded fit batch, durable reservations before input hashing/child release, existing journal lock and fixed memory/CPU/fit limits. Requires an already admitted immutable window. |
| Preparation/encoding/compile/cache/checkpoint/publication/custody | Typed stages available to I12/I64/I66; each caller must reserve and reconcile its own physical scope. No claim of automatic workspace-wide coverage. |
| Learning census/progress | Read-only; no new learning worker or alternate logical denominator. |

`run_accounted_v1` pins `resource-policy-v1.json` and records transitions in
`resource-events-v1.json` using the existing execution journal's atomic/fsynced
writer. Rebuild the ledger from that policy and ordered event vector before
resuming the same journal. The caller cannot supply a fresh ledger to erase
historical spend. A child already known to the ledger is not charged again.
Controller errors retain unresolved reservations for reconciliation against the
existing started/finished child journals. The after-completion start join refers
to that existing pre-release journal; reservation is durable before any child
can start. Batches reserve all fit memory together; callers submit bounded
batches rather than treating this method as a new scheduler.

The supervisor, exclusive host lock, child cgroup gate, descendant draining,
original execution/arm window, reporting headroom, <=30-second per-fit cap and
fixed controller/per-fit memory policy remain authoritative. Controller overhead
and upstream stages require separately reserved scopes. Timeouts, CPU affinity
and test fixtures do not qualify a systemd/cgroup host. The real-supervisor test
remains separately admitted and ignored in ordinary fixture verification.

The coordinator owns the JSON schema, final shared module/schema hooks, accepted
I05 binding, coverage rows, inventories, aggregate receipt hooks and integrated
required gate/review/CI. This worker adds only the minimal local module export
needed to compile its integration target. Fixture evidence is not live campaign,
custody, serving, GPU or scientific qualification.

Rollback disables the new opt-in policy while retaining its complete event and
cost history; old readers and original experiment windows remain intact.
