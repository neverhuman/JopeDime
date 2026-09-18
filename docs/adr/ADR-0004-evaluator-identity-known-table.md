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
