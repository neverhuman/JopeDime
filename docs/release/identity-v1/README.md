# v1.0.0 evaluator identity transition

The authorized product version bump changes the hashed `crates/jope-campaigns/Cargo.toml` and workspace `Cargo.lock`. These are version fields only; no evaluator source, DSL ABI, behavioral fixture, historical snapshot or original receipt changes.

| Identity field | Prior main 4fc1492e4 | Version-bumped source c195666e |
| --- | --- | --- |
| implementation_sha256 | bdc4b4680114f8d96aea207fc3d472cfcb9762809eb1e3bb4b26f2c6ad116b98 | 0baac311b86e9d76df9004953ee6d925fc3bb2b5e23520ef75c1127c856d3373 |
| cargo_lock_sha256 | 8ecdd1fc461f5700897e2e65dac13e3a05200bfcd2226e3b62d39bbe772c1fee | c9d7a9e5dbc6df8f9e888437bf2301c65ccff891d2064a3b896db9f1d257b3bd |
| identity_digest | db0af8bde9ccd66643a6986b29cafe68d349f6d94a27c559a71a7f1349d57fb8 | 00b1699ae383057ba63be50abaf24bd4de29359b85fc8674af28eef7a312e522 |

[ADR-0004](../../adr/ADR-0004-evaluator-identity-known-table.md) requires an actual receipt before accepting a historical identity. [prior-receipt.json](prior-receipt.json) was produced twice, byte-identically, by a fresh Cargo-resolved native binary at clean `adf24ba6f7d68973eaa6fdbbe00e5654863b4c03`. Every hashed input is byte-identical to reviewed main `4fc1492e4755b0777134a78705feedb0a2f73d30`. [prior-source-proof.json](prior-source-proof.json) retains exact commands, executable and receipt hashes, source, environment, elapsed time and exits. The command uses only the adjacent synthetic manifest and twelve-row training CSV, copied byte-for-byte from the existing proof fixture. It is local fixture evidence, not a learning-store or live training claim.

The known table appends only this receipt-backed prior pair and preserves both existing rows. The new pair is current, so needs no speculative historical row. The current evaluator fixture must be refreshed with the maintained identity-only mode after the source commit, with the old/new fixture digests bound in a named transition. The complete frozen suite and the unchanged NOREGRESS_V1 protocol must then pass at the final local candidate. This announcement and preparation do not authorize opening the release PR or tagging it before the remaining release gates hold.
