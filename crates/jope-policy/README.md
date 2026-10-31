# jope-policy

Candle port of the jope-dime step-60000 proposal policy, moved from jain-core `feat-core`'s
`jope_dime/` module (WP3 of `FABLE_GITHUB_PORT_PLAN.md`): grammar/automaton, policy forward,
grammar-masked decode, ranking and the 80-candidate seed-population writer, plus feat-core's
`jope_embed.rs` (the `jope.dataset_input.v1` → embedding contract). The provenance boundary
described in `src/lib.rs` is unchanged: the accepted step-60000 identity constants
(`ACCEPTED_STATE_SHA256`, `ACCEPTED_MODEL_SHA256`, parameter/config/vocabulary hashes) are the
feat-core bytes, and the private-field/`compile_fail` guarantees still hold (the doctests run).

## Features

| Here | feat-core | Meaning |
|---|---|---|
| (always on) | `jope-dime` | candle 0.11.0 CPU policy: `candle-core`, `candle-nn` are plain dependencies |
| `testing` | `jope-dime-testing` | untrained weights and the unbound helpers (`jope_policy::testing`). Never enable in a product build |
| `cuda` | `jope-dime-cuda` | `candle-core/cuda`, `candle-nn/cuda`; declared for the GPU lane, build-unverified exactly as in feat-core |

`default = []`.

## Test gates

`cargo test -p jope-policy` runs the unit tests and the 7 doctests; `--features testing` adds
`tests/jope_dime.rs` (25 runnable tests and 5 fixture-gated ignored tests with the committed
vocabularies and a tiny deterministic model;
`required-features = ["testing"]`, so cargo skips the target visibly without the feature).
Weight-bound tests are `#[ignore]`d and read their inputs from the environment
(`-- --include-ignored`):

| Test | Variables |
|---|---|
| `tests/jope_dime.rs` `exported_policy_*` | `JOPE_DIME_POLICY_SAFETENSORS` (exported `policy.safetensors`), `JOPE_DIME_EXPORT_RECEIPT` (default `<dir>/../step-60000.receipt.json`). This must be the accepted `tools/ot-export` receipt (`b50150ea…` on the qualification host). A `jope-parallel export-policy` re-export of the accepted checkpoint, such as the ASM-demo anchor, is refused as a derived re-export with no acceptance authority (OWNER-DECISIONS.md D13) |
| `tests/jope_dime.rs` `the_real_accepted_receipt_carries_only_pinned_fields` | `JOPE_DIME_EXPORT_RECEIPT` (the accepted receipt), optional `JOPE_DIME_REEXPORT_RECEIPT` (a re-export, verified only as non-authoritative derived evidence) |
| `tests/jope_dime.rs` generation replay | `JOPE_DIME_GENERATION_000011` (a jope-dime `generation-000011` results directory) |
| `tests/jope_dime.rs` end-to-end | the two above + `JOPE_DIME_CONTEXTS` (`contexts-assembly-v2` directory) |
| `src/jope_embed.rs::reviewed_preprocessing_bundle_loads_with_exact_v1_shapes` | `JAIN_TEST_MODEL_BUNDLE_ROOT` (directory with `preprocessing.safetensors`) |
| `src/jope_embed.rs::embedding_matches_python_reference` (ignored; `--include-ignored` fails if fixtures are missing) | `JOPE_PARITY_ART_DIR`, `JOPE_PARITY_DATASET_DIR`, `JOPE_PARITY_REF` |

## Provenance (source → destination)

Source: jain-core `3692748` (`jain-core-v10.0.0-split.4-57-g3692748`, 2026-09-18) `crates/feat-core/`.

| Source | Destination | State |
|---|---|---|
| `src/jope_dime/mod.rs` | `src/lib.rs` | edited (module list, feature names, one doc link) |
| `src/jope_dime/{config,decode,grammar,model,ranking,seeds}.rs` (6) | `src/` | feature-name renames only |
| `src/jope_embed.rs` | `src/jope_embed.rs` | four `#[cfg(feature = "jope")]` gates removed |
| `tests/jope_dime.rs` | `tests/jope_dime.rs` | edited (crate paths, feature name, fixture directory) |
| `src/hashing.rs` (`mix64` only) | `src/hashing.rs` | shim: the 6-line SplitMix64 finalizer `seeds::ProposalSeeds::derive` uses |

The vocabulary fixtures `artifacts/jope-dime/grammar/{featmath-jain-v1.json,featmath.json}` are
byte-identical to `crates/jope-engine/assets/grammar/`, so the test reads them from there instead
of carrying a second copy.

## Edits to copied code

- Feature names: `feature = "jope-dime-testing"` → `feature = "testing"` in every `cfg`/`cfg_attr`
  (`config.rs`, `decode.rs`, `model.rs`, `ranking.rs`, `seeds.rs`, `lib.rs`, `tests/jope_dime.rs`);
  prose mentions of `jope-dime-testing`/`jope-dime-cuda` follow.
- `src/lib.rs`: `mod hashing; pub mod jope_embed;` added; the `[`crate::jope`]` intra-doc link to
  feat-core's older policy is plain text; "Under the production `jope-dime` feature" reads
  "In a product build (no `testing` feature)"; doctest paths `feat_core::jope_dime::` → `jope_policy::`
  (also in `grammar.rs` docs) so the `compile_fail` error codes stay the ones asserted.
- `src/jope_embed.rs`: the `from_safetensors`/`load_safetensors` impls and the ignored bundle test
  lose their `#[cfg(feature = "jope")]` gate — candle is unconditional here.
- `tests/jope_dime.rs`: `feat_core::jope_dime::` → `jope_policy::`; grammar directory
  `artifacts/jope-dime/grammar` → `../jope-engine/assets/grammar`; header/command docs updated.
- `Cargo.toml`: dependency lines copied from feat-core (candle 0.11.0 `default-features = false`,
  `zip 9.0.5` deflate-only for the `.npz` projection, `feat-math` as a path dependency).
- `cargo fmt -p jope-policy` (no changes).

## Commands

```
cargo check -p jope-policy --all-targets --locked
cargo test  -p jope-policy --locked
cargo test  -p jope-policy --locked --features testing
```
