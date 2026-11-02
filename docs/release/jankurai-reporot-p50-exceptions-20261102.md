# P50 repo-rot pinned-source disposition

One P50 path finding remains a documented false positive for actively executed source whose filename is part of its compatibility identity.

| Path | SHA-256 | Evidence and disposition | Owner | Expiry |
|---|---|---|---|---|
| `crates/feat-math/src/advanced_v3.rs` | `80d09c109f117ea16779714c11cecb5db9c9ab2c842cca4561c8add8c3c7aef2` | The append-only public schema-bound v3 API is actively exported from `feat-math`; `compat_freeze/evaluator_identity.v1.json` pins this exact path and byte hash, and the fitted portable emitters consume this module. The version identifies the compatibility boundary, not a parked copy. Renaming or editing it changes pinned implementation identity and potentially emitted bytes. Preserve path and bytes until a reviewed compatibility rebaseline updates the identity fixture and emitted-artifact proofs. | feat-math | 2027-04-30 |

The P50 proof lane is `crates/feat-math/tests/compat_freeze.rs` plus `jankurai audit-file` and the final full audit. This is the only exception involving emitted artifact bytes. The evaluator sources previously listed here were renamed with an ADR-0004 identity transition and have no exception.
