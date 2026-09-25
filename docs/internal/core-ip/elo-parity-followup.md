# Campaign Elo reference check

Observation date: 2026-09-25 UTC. This is a follow-up to the monograph’s scoring
chapter, not a new training result or a replacement for its historical evidence.

The full CPU diagnostic 36182992040 failed the campaign Elo parity test because
it imported `sotarena.elo` through a removed source layout. The failure reproduced
on merged main `d825daf802f042994cc5f9ab8f3726511aa0dd62`: one passing tie/centering
check and one `ModuleNotFoundError`.

The comparison now uses the maintained `jope-arena::elo` port of that reference.
The two implementations remain separate: the campaign scorer is unchanged, and
the arena implementation retains its own committed leaderboard-fixture tests in
`crates/jope-arena/tests/elo_regression.rs`. The renamed comparison keeps the same
three synthetic datasets, four methods, near-tie case and absolute `1e-9` rating
tolerance. This is Rust-to-Rust implementation parity; it does not claim a fresh
Python execution, new dataset evaluation or predictive improvement.

The arena dependency is test-only. The inputs CI lane runs the two focused
Bradley–Terry tests, retains a nested receipt with actual counts and rejects an
empty selection. Its path gate includes both campaign and arena source changes.
Other campaign tests retain their prior selection and qualification status.

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test -j4 --locked -p jope-dime-regressor --bin jope-dime-regressor \
  commands::train_only_redteam::tests::bradley_terry -- --test-threads=1
```

Source-bound local and CI receipts are retained under
`/home/ubuntu/jope-rescue-20260921/core-ip-elo-followup-20260925/`.
Remaining full-main failures involving unavailable frozen corpora or inactive fit
authority are not resolved by this comparison. Their original admission and
corpus identities remain intact.
