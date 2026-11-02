# P50 repo-rot pinned-source dispositions

Three P50 path findings are documented exceptions for actively executed source
whose filename is part of its compatibility identity. These are the only
repo-rot exceptions in this slice.

| Path | SHA-256 | Evidence and disposition | Owner | Expiry |
|---|---|---|---|---|
| `crates/feat-math/src/advanced_v3.rs` | `80d09c109f117ea16779714c11cecb5db9c9ab2c842cca4561c8add8c3c7aef2` | The append-only public schema-bound v3 API is actively exported from `feat-math`; `compat_freeze/evaluator_identity.v1.json` pins this exact path and byte hash, and the fitted portable emitters consume this module. The version identifies the compatibility boundary, not a parked copy. Renaming or editing it changes pinned implementation identity and potentially emitted bytes. Preserve path and bytes until a reviewed compatibility rebaseline updates the identity fixture and emitted-artifact proofs. | feat-math | 2027-04-30 |
| `crates/jope-campaigns/src/commands/exact_cv_v2.rs` | `b3b7fa89f6f4584edc66abab8fedc26989b1b021317950ebe53566a8dfd326c0` | Active one-fold proof evaluator, imported by `main.rs`. `evaluation_v2.rs` hashes both its label (`src/commands/exact_cv_v2.rs`) and compiled bytes with `include_bytes!`; renaming it changes the evaluator identity. Preserve path and bytes until an ADR-0004 identity rebaseline carries old identities forward. | jope-campaigns | 2027-04-30 |
| `crates/jope-campaigns/src/evaluation_v2.rs` | `865ad21acae020fd6b9966a98315420f16b814af6b0b9860ada76988f5c65965` | Active evaluator identity and receipt validator. The implementation's source hash includes its own label and bytes, plus the exact-CV module identity; compatibility tests pin this closure. Renaming changes identity for retained receipts. Preserve path and bytes until the ADR-0004 identity rebaseline carries old identities forward. | jope-campaigns | 2027-04-30 |

The P50 proof lane is `crates/feat-math/tests/compat_freeze.rs` plus
`jankurai audit-file` and the final full audit. The advanced-v3 row is the
only exception involving emitted artifact bytes; the two evaluator rows are
source-identity compatibility exceptions. These exceptions do not change
source, tests, emitted artifacts, or audit wording. Before expiry, the
evaluator-identity owner must either complete the documented identity
rebaseline and remove both exceptions or renew them with fresh exact-byte
evidence.
