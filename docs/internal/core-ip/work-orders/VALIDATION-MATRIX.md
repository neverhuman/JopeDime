# Validation stages, evidence and release gates

This matrix complements task-specific acceptance. Read the
[execution contract](EXECUTION-CONTRACT.md) for resource ceilings and provenance.
No row is currently marked passed by this plan. Preserve original comparison
surfaces, thresholds and protocols; never loosen them after observing failures.

| Gate | Packages | Positive path | Adversarial/failure path | Required retained evidence |
|---|---|---|---|---|
| G0 trustworthy baseline | W0 dependency | Legitimate authority-backed regression fixtures execute | Original default/view/scratch/SDK failures reproduced; corpus absence explicit | Current collection, exact failure logs, feature/input identities, counts/exclusions |
| G1 release/input closure | W1 | Verified fitting-only release reaches worker; scorer sees its authorized labels | Forbidden-only perturbations, aliases, duplicate joins, replacement, denied reads/cache substitution | Source/release hashes, access traces, allowed-input controls, unchanged restricted outputs |
| G2 accepted semantics | W2 | Exact bundle and each typed transformation produce golden intermediates | Same-width swaps, vocabulary/weights/grammar edits, stale cache, check/load race | Release decision,136-tensor inventory, active/frozen/trainable roles, original `3e-4` comparison receipts |
| G3 admission/HWM | W3 | Qualified teacher passes measured policy and receives explicit HWM disposition | Missing/revoked/replayed provenance, wrong fitted subject/fold, ties/races/crashes | Before/after authority state, independent metric arithmetic, complete decisions/corpus linkage |
| G4 autoregression | W4 | Equivalent logical histories and fixed-logit replay agree | Future suffix/PAD effects, invalid numeric/support/EOS/length cases | Forward trace, parser differential census, token/RNG replay, scorer dependency/collision fixture |
| G5 search/fitted identity | W5 | Authorized candidate fits, selects/refits and preserves selected subject | Mutable parent, stale cache/score, incomplete folds, reservation races, blocked/OOM/late fitter | Independent selector, complete attempts, ancestry, fitted artifacts, no-orphan census |
| G6 training continuation | W6 | Actual supervised update and complete deterministic checkpoint resume | Missing optimizer/RNG, torn state, invalid comparisons/runtime labels, undeclared transition | Objective/gradient fixtures, actual sampled records, committed counters, next-record equivalence |
| G7 managed CPU lifecycle | W7 | CPU job executes on every named host | Missing participant, stale host/boot/source, coordinator restart/cancel | Three distinct execution receipts, verified results, grants, resource use, complete drain |
| G8 individual GPU | W7 | Each actual UUID executes GPU work/update/reload | Wrong local device, stale GPU ownership, runtime mismatch, expiry | Three hardware-specific runtime/update/checkpoint receipts; no CPU fallback |
| G9 distributed numerics | W7 | Two-host then three-host updates match frozen global objective | Uneven groups, remainder loss, missing rank, rank death, control loss, topology mismatch | Global numerator/denominator/reference comparison, per-rank RNG/state, real updates and group recovery |
| G10 offline durability | W8 | Committed graph rebuilds and outbox drains idempotently | Partial upload, response loss, conflict, missing/corrupt dependencies, full outbox | Dependency/commit order, failure census, logical deduplication, reconstructed state |
| G11 live S3 | W8 | Scoped publisher publishes/restores/resumes through repository code | Same-predecessor race gives one winner/one conflict; outage and stale pointers | Credential scope metadata, actual write/byte budget, full independent restore hashes, resumed computation |
| G12 fitted package | W9 | Same fitted subject serves qualified Python/Rust/C++17 outputs | Same-recipe state swap, missing stage, malformed schema, hostile string, replaced shipping bytes | Exact-bit native reference, no-fit instrumentation, bounded build, package/install verification |
| G13 real loop witness | W10 | Candidate→teacher→corpus→update→restore→search→export→S3 replay | No eligible teacher, no active parameter change, no-update control, failed joins | One joined real-data chain and every terminal outcome; benefit remains separate |
| G14 causal studies/transfer | W10 | Frozen complete comparisons;30 searches per arm/dataset/seed for transfer | Missing pairs, early score release, unknown exposure, post-hoc protocol change | Preregistration, whole-round commitment, independent per-unit-minimum scorer, tails/skill/failures |
| G15 performance | W11 | Qualified synchronized whole-request timing and matched comparison | Controlled delays, failed qualification, hidden retries, changed logical work | Complete warmup/measurement/process/GP journals, hardware identity, uncertainty and decision rule |
| G16 claims/replay | W12 | Independent package replay and validated scoped publication | Evidence swap, stale/revoked parent, synthetic/historical scope upgrade |195-entry reconciliation, trusted decision receipts, revocation closure, public synthetic conformance |

## How to execute a validation batch

1. Freeze task IDs, source and exact command; confirm required inputs and capabilities.
2. Collect tests and record expected cases before execution. An absent feature target,
   missing runner, zero collection or unexecuted required test is not coverage.
3. Run the valid path and each relevant mutation/failure through the same actual receiver.
4. Bind outputs to original identities and inspect authority-bearing side effects.
5. Reconcile collected, executed, failed, skipped/excluded and unavailable counts.
6. Restore/replay independently where the gate requires it; retain both receipts.
7. Review at the final source head, then merge through the exact-head lifecycle.

Cheap unit fixtures run before expensive live phases. Source, fixture and CI states
remain usable engineering evidence while the live gate stays open. A host-only proof
requires its explicit host lane; do not replace a real-corpus identity with a synthetic
fixture and reuse the real-corpus test name/claim.

## Non-regression deferrals and acceptance limits

Binary, multiclass, time-series and their specialized algorithms remain deferred.
Shared APIs must reject unsupported requests explicitly and preserve existing supported
callers according to the reviewed versioning policy. Existing classification failures
stay in the full diagnostic; regression-only success is reported as regression-only.

Python/Rust/C++17 fitted export is limited to declared supported regression compositions.
C++17 retains its Python typed-table host dependency. W4 retains order-blind reward
pooling as measured accepted behavior; an order-aware challenger belongs to W10.
Unknown historical family/checkpoint/encoder exposure cannot be manufactured or relabeled.

## Stop, recovery and completion

Stop admission on invalid authority, unverified inputs, resource conflict, exhausted
original window or an unavailable required capability. Drain owned jobs without
interrupting unrelated ones; preserve failed and incomplete work in the census.
Resume only compatible committed state within the original budget. An explicit new
run/transition requires its own identity and applicable admission; it cannot overwrite
the exhausted attempt or promote its partial outputs.

The final release decision requires G1–G13 and applicable G15/G16 controls plus honest
disposition of research/transfer G14. Complete transfer scoring remains blocked when
its required historical exposure is unknown; a narrower development result must use
a distinct identity. Positive scientific benefit is not required to complete an
honest study, and no study outcome automatically authorizes serving promotion.
