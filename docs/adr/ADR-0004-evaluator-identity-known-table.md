# ADR-0004 — Evaluator identity: accept known historical identities

**Status:** accepted 2026-09-18 · **Scope:** `crates/jope-campaigns/src/evaluation_v2.rs`

## Context
`JopeDimeEvaluatorIdentity.v1` hashes ~30 feat-math sources, the crate manifest and the workspace `Cargo.lock`
(`compute_implementation_sha256`). Its `validate()` compared a receipt's recorded identity with the *current
build* and rejected anything else, so every persisted `JopeDimeSemanticEvaluationReceipt.v2` became invalid the
moment feat-math was reformatted or the lockfile changed — which the single-workspace port does by construction
(one `Cargo.lock` for the whole repository).

## Decision
`validate()` still requires the DSL ABI to match, but accepts an implementation/lockfile pair that is either the
current build's **or** listed in the committed, append-only table
`crates/jope-campaigns/known-evaluator-identities.json`. Producing a receipt is unchanged: callers use
`EvaluatorIdentityV1::current()`. The table is excluded from the hashed inputs, so appending a row never changes
the current identity. A row is appended only together with the receipt path that proves the identity existed
(historical receipts in the learning store `runs/` prefix); the port's history scan seeds the first rows.

## Consequences
Old receipts remain verifiable and auditable under the identity that produced them; a new feat-math byte change
(WP2's `gpr.rs`, `registry.rs`) creates a new current identity without invalidating history. Reviewers see every
accepted historical identity in one file.

## Current identity transition: public stage names (2026-10-30)

PUB-02 adds the public-names module declaration to `crates/feat-math/src/lib.rs`, which is an ADR-0004
hashed input. After the merged API-01 lockfile identity transition, the implementation hash changes from
`44c1f73ba257e509a8350a72a6ea6790140d165eb6bd0ef9174c3e5dcb5a4d64` to
`bd23428e33c286f33b7c5161a00ecfc35f8d5aa831f7f011837c86639fe2d7ad`; the identity digest changes from
`75294fa33fd0855958db6ad6c6f75a85ab61d84a8e4d28dcea25b9305b7003d7` to
`ba253e90391fb943513dccc83e09c6fbed240c1acf5c47181134eefd441c6221`. The Cargo.lock hash remains
`89d5812adf6dcfd0b549435ee38863753a21550206623cefad553a970dc80420` across this PUB-02 transition (API-01
is the preceding lockfile change). The DSL ABI and evaluator calculations are unchanged. The identity fixture
records the new current identity; the historical JSON table gains no row here because this PR has no shipped
receipt to support one.
