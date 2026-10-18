# Retention execution candidate (I52)

## Status

This is a disabled, mock-only implementation candidate. It does not connect to storage and it
does not remove data. The feature name `retention-executor-candidate` is deliberately not
registered in the package manifest by this worker task. The module is therefore not part of the
normal package build. Fixture tests include the source directly and mutate only an in-memory
namespace.

No live execution is authorized by this candidate. The receipt field `live_removal_count` remains
zero; `simulated_effect_count` counts changes to the fixture namespace only. Do not treat a passing
fixture test, an approval-shaped fixture, or a tombstone as an operational authorization.

## Candidate state machine

The planner accepts only complete authority snapshots with matching registered and observed
publisher sets, explicit physical version identifiers, unique targets, and nonempty root, pin,
and lifecycle generations. Its canonical plan binds the exact target set and all three
generations. Active pins and previously retired targets fail closed.

Reservation checks the same generations again, verifies that the approval names the exact plan,
and requires a distinct planner and reviewer identity plus an independent restore proof. The
fixture restore proof requires the original version to be absent in its isolated namespace,
replacement bytes to match, and readback to succeed. Production signature verification and
independent runtime recovery evidence must be supplied by separately reviewed integrations;
the booleans in these fixtures are not cryptographic verification.

The executor writes a tombstone intent before changing the mock namespace. Each request identity
binds one plan, reservation, and exact version. A publisher-generation or pin change aborts before
the simulated effect. Recovery observes whether the recorded exact request already took effect;
it never widens to a prefix and never issues a second request for an already observed effect. If a
version is absent without a recorded request, the candidate records that ambiguity and does not
retry.

## Required integration before any operational consideration

The owning integrator must register the opt-in feature and required-feature test target, connect
the candidate to the shared retention schemas and census/coverage records, and review the API and
failure semantics. Signature checking, durable tombstone storage, authoritative generation reads,
publisher fencing, independent restore evidence, request reconciliation, and protection against
concurrent reservations need real implementations with isolated integration proofs. Any future
live adapter needs a separate authorization and bounded acceptance plan. Until those reviews and
proofs are complete, keep the feature absent from the manifest and do not enable it.
