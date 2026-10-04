# Authority synthetic-startup fixture-only prerequisite

Status: local implementation; hosted CI and independent approval required before merge.
Base: bb756bcbd1224ffa1e5ca289611a2d377faadd1a. Sole lane: bt-implementation.
Branch: kernel2/bt07-authority-startup-fixture. ENG:JAILGUN selects this fix before B08.

## Behavior

Only the cfg(test) synthetic executor fixture changes. Prepare the child's authority handle and
cold fit/runtime identity before atomically publishing readiness; authorize after readiness.
The fixture preparation wait is bounded at 90 seconds and detects an early child exit.
The real start token is still 30 seconds; the consumed-handshake wait stays 30 seconds and
observed timeout wait stays 1 second. Real consume_start/custody.verify, atomic start-token
publication, RAII cleanup and existing failed-exit/signal/timeout safety assertions remain.
No product/runtime/custody/deadline changes, new dependencies, suppressions or regeneration.

## Tests and evidence

A new nonignored 31-second preparation-delay regression fails on the old ordering with
"synthetic executor failed before start" (one executed failure, nextest exit 100). After the
fix it verifies readiness before authorization, the exact unchanged token duration, and that
plausible result files cannot clear a failed exit. A second nonignored regression verifies
preparation failure leaves no published start token and keeps the claim in claimed state.
All 40 focused default authority cases pass; no old tests or assertions are replaced.

Acceptance: `cargo nextest run --locked -P pr -p jope-parallel --lib -E 'test(eval_registry::authority::tests::)'`.
Fresh baseline: PR 2099/29, main 4179/66, Python 231/2, golden 16/16, fitted ASan/UBSan
18/18 and C++20 18/18, strict original O3 plus extra g++17 O2 pass. Required after counts:
Product PR 2099/29 (+0: jope-parallel is a declared deferred package), main 4181/66 (+2),
and explicit PR-profile authority family 38→40 (+2); all old selections/skips/exclusions
remain unchanged, with zero-new workspace/owned lint ratchets. Actual just required 101 is the previously accepted unchanged
held floor, NOTPASS; full standing suites run separately. All frozen/identity/emitter bytes
remain untouched. The existing coverage item description records the added fixture coverage.

Durable commands, source/diff/log hashes, counts, audit, exact-head automatic/manual CI and
independent native review readbacks: `/home/ubuntu/jope-evidence/kernel2/bt/B07_AUTHORITY_STARTUP_FIX/`.
Original post-173 hosted FAILURE and 100/100 unchanged-main neural repetitions remain at
`B07_STABLE_EXPONENTIAL_EXECUTION/main-authority-classification`; likely startup timing,
not a retrospectively proved hosted root cause. B08 source stays held until this PR merges.
