# Go qualification for PR #416

The branch starts at main `5390576ffe98f1bea9b1378479e05fb1a141314a` and integrates
TypeScript main `d8b27f6fd89f65742a8c16aa68cbe4a658187941`, which landed during qualification.
The numerical implementation is the existing standalone Rust emitter compiled to Wasm;
Go provides the deployment API through a pinned pure Go runtime. This is build-time
Rust support, not a new independently implemented Go solver. See [the API/build contract](../go.md).

## Executed evidence

| Check | Observed result |
| --- | --- |
| Complete registry census | Five tests passed with default features and again without default features; 192 artifacts executed under both engines; every inherited refusal accounted for |
| Recipe rows | 53 executed, eight inherited Rust refusals |
| Bound rows | 24 executed, one inherited refusal |
| Legacy bound rows | 14 executed, 39 inherited refusals |
| Explicit v2 programs | All 95 scalar/row-vector cases executed |
| Codec/metadata additions | Five codec cases and one hostile metadata case executed |
| Native Python/Rust/Go golden target | All 84 tests passed, including the required-runtime gate; finite tolerance `1e-8*(1+abs(reference))`; both Go engines |
| Generated Go API | 100% raw statement counters covered on Go 1.25.12 and 1.26.10 |
| Race / lifecycle fuzz | Passed with actual emitted Wasm; 179,148 fuzz executions |
| CGO-disabled deployment | Example ran successfully; six consumer executables linked: Linux/macOS/Windows × amd64/arm64 |
| Source binding / pipeline | 21 and 18 tests passed, including every source member's tamper/deletion checks |
| CLI / fitted regressions | Two and two tests passed; saved schema/codecs/ordered-tail fixtures executed |
| Compatibility freeze | 13 passed; historical three passed; legacy three passed, one existing opt-in generator ignored |
| Static checks | All-target check, scoped Clippy, generated-target Clippy, fmt/diff, actionlint and coverage-map validation passed |
| Records and repository contracts | 259 tests passed; README counts and new target classification verified |
| Evidence verifier adversarial fixtures | 22 aggregate tests passed; absent/zero/partial/filtered/missing-engine/uncovered evidence rejected |

The original Go-enabled 83-test body passed at `d5c2436`. Review discovered that native
golden source is frozen, so `833091a` restores it byte-for-byte and derives a dedicated
feature-gated target with checked insertions. The derived body is equivalent apart from
comments and isolated output paths, plus one required-runtime assertion. The complete new
84-test binary then passed, unfiltered, at `6d0a0b6c900d3baeb76622887e2533ca08d74927`.
Both Go engines executed 73 golden-model qualifications each. The missing-flag negative probe correctly failed.
The combined evidence verifier accepted the complete lane logs and raw statement counters.

## Main refresh and final red-team fix

The TypeScript PR merged during the full golden run. The Go branch integrates that main,
retaining both public targets, required CI matrices, all original TypeScript source files and
its approved native-golden/freeze bytes. Updated joint binding, pipeline, CLI and fitted tests
passed (21, 18, two and two); records (259), Clippy, workflow lint and 22 aggregate tests passed.
The retained TypeScript evidence adversarial test also passed.

The final no-default-features census executed all 192 artifacts under both engines again.
All 192 Rust kernels and 384 module/checksum members are byte-identical to the pre-refresh
Go artifacts. Each of the 192 Go facades differs only by an explicit target-mean overflow guard
in Score. A regression demonstrated the previous erroneous acceptance even with perfect
predictions; the fix passes with 100% raw facade statement counters on Go 1.25/1.26 and the
race detector. Updated consumer executables link for all six platforms.

The 84-test full run above predates the main refresh. Structural comparison confirms all its
native assertions remain, with the same four Go hook pairs; upstream adds only 36 optional
TypeScript lines. The rebased generated target's representative/presence tests passed. The
complete integrated target still requires exact-head CI; source/behavior equivalence is
recorded separately from a new full-suite invocation. The identity join follows merged
TypeScript and also verifies the previously published Go branch's successor, preserving both
old transition records and every historical row. The final identity freeze has 13 passes.

## Current-main C17 integration (2026-10-23)

The branch now preserves main `f9543b85b356675d377e38d75c4e3ecea8b2d302`, including
C17 and the ctutils successor. All native golden and C17 helper bytes remain identical
to main. The Go adapter retains four checked insertion hooks and resolves the C17
module paths back to their original files. Its original anchor-drift panic and a
second fast-path anchor failure were retained before correction.

At `f028edbd839ec2ced4e6d8202732dfd62e40feaf`, the complete derived Go binary compiles
and lists **90 tests**; these tests have not yet been executed at this integration.
The evidence verifier now requires at least 90 golden and 19 pipeline tests and refuses
the earlier 84/18-test receipts. Earlier 84-test execution remains historical evidence.

The preceding combined source `1e62cf894c81ae023364a0b189994654bf199251` passed all
13 default-feature compatibility tests, all three historical tests, 21 math source-binding
tests, 19 pipeline source-binding tests, six selected pipeline emission tests, two selected
CLI source-binding tests, and all-target checks. Its later change affects only the owning
Go golden adapter and evidence checks, not numerical implementation or behavior fixtures.
The portable monitor correction also executed all 82 synthetic fixtures without skips.
Both the 1,051 enclosing and 48 portable source pins match actual current bytes.

The announced identity-only joins retain the main-line C17/security history and the
previously published Go branch. Every earlier transition, historical object, known
identity and numerical/emitted/model/search fixture is preserved. Full exact-head Go,
native PR and manual CI remain mandatory; the pending Tokio security successor in
PR #420 must be integrated before final merge qualification.

## Remaining qualification

The broader feat-math library run had 525 passes and two failures in unchanged C++ sanitizer
tests: LeakSanitizer cannot read `/proc/.../task` in this sandbox and reports the ptrace
restriction. Seven pre-existing tests were ignored, and the five Go census tests ran separately.
No sanitizer check was disabled. Hosted CI explicitly reruns the two native sanitizer tests.

Hosted Go and aggregate CI were still queued/pending at this report. Native Windows/macOS
execution therefore remains a CI requirement. Keep the PR draft
until the exact current head is qualified. No all-repository-tests-green claim is made.

Qualification/example debug builds use one codegen unit after observed zero-length Rust
archive objects in feat-math and jope-records on this host. Release settings and the standalone
Wasm compiler command are unchanged.

Go has no built-in branch coverage measurement. The 100% figure covers statements in emitted
`model.go`; it excludes inherited Rust numerical code and wazero. The census qualifies exercised
numerical behavior separately and preserves Rust's existing unsupported cases. Numeric-v2 fitted
images and model-object storage/source-export contracts are not expanded by this target.

[Machine-readable receipt hashes](go-qualification.json) record local logs, scope, source commits
and limitations. CI retains its own exact-head logs/coverage and refuses missing evidence.
