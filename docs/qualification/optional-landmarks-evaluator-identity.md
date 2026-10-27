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


The separately reviewed required-gate repair at source commit
`e65022eee1ecfc397993f34d94aa6a093eea1b82` upgrades only the zstd packer dependency graph
and migrates two stable owned-file registrations to Tokio's sound API. The changed lockfile
announces a second identity transition; evaluator behavior and the DSL ABI remain frozen.

| Field | Before gate repair | After gate repair |
| --- | --- | --- |
| `implementation_sha256` | `95d3e019b8ed2dfdb5b028661d1288c55450f8f0cb7f4fd4068ded90488baeb0` | `8c6aad4f669de25306a617dfaa1ee65489453e8441c6ebfa768b0a3062373ef2` |
| `cargo_lock_sha256` | `1e33b0f02db50282014933621f69920485f284c7774304f7463cbae508f168f3` | `6d5517cf0c5526aaaac4f220da7bf22aa821948b42986d008611e9c6c066af1a` |
| `identity_digest` | `1aa42477fd648f0c1cf92a89c613d6a1c7314c9a3a8ed10e0caa62bc4480af57` | `ed9d5b8ca75f79d4826cae89c2268fd76c71b1aabd1f2b0760258a8943282413` |

The identity-only regeneration and a separate comparison each pass all thirteen tests.
Only evaluator identity and lineage source-commit headers changed; the known identity table
and behavior fixtures remain unchanged. Retained logs are
`gate-maintenance-identity-rebaseline.log` and `gate-maintenance-identity-compare.log`.


The final published Dope code pin25d1785c is announced by Jope source commit
`83e09f42ea6dd2378994a72721d1c44e575c9d10`. Only the Dope Git source line changed in
Cargo.lock for this final pin; unrelated Windows dependency choices were preserved. This is
the current fixture identity, superseding the intermediate identities above.

| Field | Final published-pin identity |
| --- | --- |
| `implementation_sha256` | `63e0cbc85d1ad226d386dc69ea48cfcd80fefdc8a9832dff21c4897c494c4971` |
| `cargo_lock_sha256` | `e6ac0054e9832e35bd67de6c571a337038d1f4c5e366397e384e7d4b410063b4` |
| `identity_digest` | `c56f45a5ddca4357951433c05382724453bba418b92eecc575687273c2106006` |

`final-pin-identity-rebaseline.log` and `final-pin-identity-compare.log` each record13 passing
compatibility tests. No behavior fixture or known historical identity row changed.
