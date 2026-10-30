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
hashed input. The implementation hash changes from
`22d2f11436495a5329abf1a1660c584d28ae2430923d9b5213d76036d1376110` to
`10a1514f0309607534d4c4593ae3908aceef7bbff1a41afbea1a248448c48cd1`; the identity digest changes
from `b1c89e347bd296ad7f2d8e4333e45e1580189d302a0bc2b77637d73dc824d12a` to
`2b9433410dc2a69cac595272f6033e9dd29e2dede5a48d23b32d0d4449aeec08`. The DSL ABI and Cargo.lock
hashes are unchanged. This transition exposes presentation metadata and does not alter evaluator
calculations. The identity fixture records the new current identity; the historical JSON table gains
no row here because this PR has no shipped receipt to support one.
