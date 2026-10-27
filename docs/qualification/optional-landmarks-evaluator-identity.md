# Optional landmarks: announced evaluator identity transition

The published Dope dependency and reviewed Candle source reconciliation change the workspace
lockfile. ADR-0004 includes that lockfile in evaluator identity even when evaluator arithmetic
is unchanged. Source commit `758aed1b7ef28324d2d7dc0de31ebf5f52fd67b5` announces this transition.

| Field | Previous frozen identity | New identity |
| --- | --- | --- |
| `implementation_sha256` | `99f9159ec5a64f83487a81bcc9639ab1a70383d00eb221ad8b842afb11799be2` | `95d3e019b8ed2dfdb5b028661d1288c55450f8f0cb7f4fd4068ded90488baeb0` |
| `cargo_lock_sha256` | `8f1baae9469fb8ee76332aba2a968e9c41222c2be6a45a1e22bb7e4d095e1782` | `1e33b0f02db50282014933621f69920485f284c7774304f7463cbae508f168f3` |
| `identity_digest` | `c9f85a94ab86aa1f8313cb51b7f2626c7caafe32c8cf0f81eccb046fd8ea0aff` | `1aa42477fd648f0c1cf92a89c613d6a1c7314c9a3a8ed10e0caa62bc4480af57` |
| `dsl_abi_sha256` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` |

The identity-only mode regenerated `evaluator_identity.v1.json` and the source-commit header
in `lineage_sources.v1.json`. The known historical identity table is unchanged; no new historical
row or receipt was invented. All thirteen compatibility tests passed during regeneration and
again in comparison mode, including frozen emitted bytes, fitted predictions, serialization,
vocabulary, generator streams and seeded search outputs. No behavior fixture was rebaselined.

Acceptance:

```bash
JOPE_COMPAT_FREEZE_REGENERATE=identity \
JOPE_COMPAT_FREEZE_SHA=758aed1b7ef28324d2d7dc0de31ebf5f52fd67b5 \
  cargo test --locked -p feat-math --test compat_freeze
cargo test --locked -p feat-math --test compat_freeze
```

Retained logs: `target/landmarks/identity-only-rebaseline.log` and
`target/landmarks/identity-compare.log`. Aggregate required-gate failures and the new training,
consumer-quality and live-flywheel qualification remain separate; see
[the qualification record](optional-landmarks.json).
