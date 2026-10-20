# Model cache lockfile identity transition

PR #387 adds the existing locked rustix dependency to jope-models for descriptor-safe cache access. Cargo.lock is an ADR-0004 evaluator identity input, so this changes the build identity even though the other hashed evaluator inputs and DSL ABI did not change. The separate mechanical match simplification in `crates/feat-math/src/combinators/residual.rs` is outside the 38 hashed evaluator inputs; historical numerical behavior is checked by the preserved compatibility receipts.

| Field | Previous fixture | New build (observed in CI) |
| --- | --- | --- |
| implementation_sha256 | 15402c880c81b39acfc4f7cf95a794a8f22e5e6df3e9a8683c9a8e863be8fb22 | b4b1ad10d3704086197c56cd4a9423e54a83686867a8b04d308882e87ac3c3a4 |
| cargo_lock_sha256 | 8def8544c114029ee1ae8d72fb71141c5a90fe55fec46818891a9abe7738bf5a | d96ade8a2a6c6d46869ac610321972f75fc8a9d60c663eefc33eb20b1abc24c9 |
| identity_digest | 3aaea86da467719a3a3cb6200798b3f2aa7e18f7a95125191e44c4a1634b6d65 | 0e95cc2f612846c2fe3b8f35bf88b9c258b36f229d47c00d88c9469f67c9e721 |

DSL ABI remains c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389. Use the documented identity-only mode in crates/feat-math/tests/compat_freeze.rs at the source-carrying commit. Preserve every behavioral fixture and historical manifest; retain the exact before/after identity fixtures in an appended transition ledger. No broad behavioral regeneration is authorized or needed. The existing known-evaluator table is unchanged because an identity-transition observation is not a historical runtime evaluator receipt. This does not confer training, replay or promotion authority.

Actual identity-only regeneration at source 49ae1ba7b06110828ecb05b47cf91be2f686034e passed 13/13. The [transition ledger](../../../crates/feat-math/tests/fixtures/compat_freeze/transitions/model_cache_workspace_lock_identity_rebaseline.v1.json) preserves both identity-fixture hashes; the [compact observation](../../internal/kernel2/receipts/model-cache-evaluator-identity-20261020/receipt.json) retains compressed and decoded log hashes. A clean compare and historical checks remain required after this source commit.
