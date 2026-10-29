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
Cargo.lock for this pin; unrelated Windows dependency choices were preserved. This records
the published-pin checkpoint before the product-scope transition below.

| Field | Final published-pin identity |
| --- | --- |
| `implementation_sha256` | `63e0cbc85d1ad226d386dc69ea48cfcd80fefdc8a9832dff21c4897c494c4971` |
| `cargo_lock_sha256` | `e6ac0054e9832e35bd67de6c571a337038d1f4c5e366397e384e7d4b410063b4` |
| `identity_digest` | `c56f45a5ddca4357951433c05382724453bba418b92eecc575687273c2106006` |

`final-pin-identity-rebaseline.log` and `final-pin-identity-compare.log` each record13 passing
compatibility tests. No behavior fixture or known historical identity row changed.

The product-scope dependency additions are recorded by source
`a9cb739732d94814228408f1bee3209456224080`. The lockfile adds Rust package dependency edges;
the evaluator arithmetic, DSL ABI, known identity rows and listed lineage sources are unchanged.
The product-scope checkpoint identity was:

| Field | Product-scope identity |
| --- | --- |
| `implementation_sha256` | `e80975f16e53370b3c1cba2dae3bff2e6853126f0afc44e96564420fecd6f7d4` |
| `cargo_lock_sha256` | `b6af27b4bf0b90a9c2746ba819fe5f882e5e442b68434c32bc6576f5891d6d6b` |
| `identity_digest` | `aec263812b16a3038cd52d123b76aecba1952895d7a2e60a607c92ab015912cc` |

[Draft PR431](https://github.com/neverhuman/JopeDime/pull/431) announces the old/new identity.
The owning identity-only regeneration and separate comparison each pass all 13 tests;
the historical test passes all 3 cases. Three append-only transitions retain the original
dependency, published-pin and product-scope fixture bytes in each existing history chain.
Retained logs are `target/landmarks/pr431-product-identity-regenerate.log`,
`target/landmarks/pr431-product-identity-compare.log` and
`target/landmarks/pr431-identity-historical.log`. No behavior fixture was regenerated.

The published P1/P2 pin is recorded by source
`64db44cc11f791c0e8442acaae4e9360d0628ad4`. Exactly one Cargo.lock Git source line changes
from Dope `25d1785c` to `2535a36c`; unrelated Windows dependency choices are preserved.
The current fixture identity is:

| Field | Published P1/P2 identity |
| --- | --- |
| `implementation_sha256` | `e0c41b1af055ff5b0af844a2077c2431d474769280c83a599483fed1b5bdeb3f` |
| `cargo_lock_sha256` | `e7e8edc191cc38636be3968f537782f609d82acc07d5bda33a140613e4920cb9` |
| `identity_digest` | `61185b72188ffeaad4abce1eb08773c00c0b6fa899aff866f3ab93219240a2c5` |

The owning identity-only regeneration and separate comparison each pass all 13 tests.
The fourth append-only PR431 transition extends every existing historical chain; the first
three transition records remain byte-identical. ABI, known identity rows, listed lineage
inputs, numerical/emission fixtures and seeded outputs remain unchanged. Retained logs are
`target/landmarks/p1-identity-regenerate.log`, `target/landmarks/p1-identity-compare.log` and
`target/landmarks/p1-identity-historical.log`. The PR body announces this transition.


## Merged Dope main re-pin, 28 October 2026

Source `91b7aabf124e4acf5ebe7fbee78ea8effc2ca132` pins published Dope main
`e2243e597c38f90ce600b306a3112b7b4d650516`, after Dope PRs #341 (exact preparation
speedups), #342 (lossless/preparation tests) and #343 (CHANGELOG). The existing
Dope629-prefix API is retained, independently of the forthcoming 512D API PR. Weights,
output dimensions and numerical recipes are unchanged. The FIT preparation-source marker
follows the new pin; components admitted for the old source cannot be silently relabeled.

| Field | Before re-pin | After re-pin |
| --- | --- | --- |
| `implementation_sha256` | `e0c41b1af055ff5b0af844a2077c2431d474769280c83a599483fed1b5bdeb3f` | `8780fd935368ae7a16cf4021c0949e665ea28421159d83d6289e407d864e38fa` |
| `cargo_lock_sha256` | `e7e8edc191cc38636be3968f537782f609d82acc07d5bda33a140613e4920cb9` | `c25699fa0c2e23a4897ac36f73bbbaa3c510376acea01eb3ec6d16348e6d066d` |
| `identity_digest` | `61185b72188ffeaad4abce1eb08773c00c0b6fa899aff866f3ab93219240a2c5` | `0a092f968c9d43d7bcf31b20847290ee4ee269caf47c8f440397a172bb9c40ae` |

Cargo generated six unrelated Windows dependency-edge changes while resolving this pin.
The owning lock reconciliation preserves those edges and validates the complete graph
with `cargo metadata --locked --offline`; exactly one Git source line changes.

The official identity-only regeneration passes all 13 compatibility tests. Only the
evaluator identity and lineage source-commit header change; the DSL ABI, known identity
rows, listed lineage inputs, numerical/emission/seeded fixtures and the four prior
transition records remain unchanged. A fifth append-only transition extends every
historical identity chain. Separate comparison, history, product and package results
are recorded in the re-pin PR when executed.

An initial generator invocation supplied an incorrect full source ID and was refused
before writing either identity fixture (11 passed, 2 source guards failed). That failed
log is retained separately. The successful invocation reads the full source ID from
Git and is retained in `target/dope-main-repin-20261028/identity-regenerate-a2.{log,json}`.
No behavior rebaseline or scientific/release qualification is implied.

## Tagged Dope engine 1.1.0 re-pin, 28 October 2026

Source `ea994457b48ace78035ec1e6ef7bad292efe4299` pins Dope
`15e12aa3dc33851655580f6aa7c9c66353143f23`, the commit named by the annotated tag
`dope-engine-v1.1.0` (tag object `7712722c2e08d65d873a3b8234e5cddcfa0bcfbc`), whose signed
engine release run 40680335064 passed. Since the previous pin `e2243e59`, Dope added the
additive 512D resident encoder API (#344) and documentation (#345, #346). The 629D outputs
JopeDime consumes are byte-identical (570/570 `mlc-v3 encode` records). Weights, output
dimensions and numerical recipes are unchanged. The FIT preparation-source marker follows the
new pin; components admitted for the old source cannot be silently relabeled.

| Field | Before re-pin | After re-pin |
| --- | --- | --- |
| `implementation_sha256` | `8780fd935368ae7a16cf4021c0949e665ea28421159d83d6289e407d864e38fa` | `ece0df30227e680e9e5eb9ced76f20279840d1b65ae41f7db65d3352729ff41f` |
| `cargo_lock_sha256` | `c25699fa0c2e23a4897ac36f73bbbaa3c510376acea01eb3ec6d16348e6d066d` | `a88a224d781ab7b23ff478134217bc17b4ef2ad9ce054bb6e1eadd973128926d` |
| `identity_digest` | `0a092f968c9d43d7bcf31b20847290ee4ee269caf47c8f440397a172bb9c40ae` | `6ce509f8671d7d0c50af6d8690d9aad96dbbb4c0b775f6edc91e112a1ab37729` |

Exactly one `Cargo.lock` Git source line changes and no other dependency edge moves;
`cargo metadata --locked --offline` accepts the graph. The official identity-only
regeneration passes all 13 compatibility tests. Only the evaluator identity and lineage
source-commit header change; the DSL ABI, known identity rows, listed lineage inputs,
numerical/emission/seeded fixtures and the five prior transition records remain unchanged.
A sixth append-only transition extends every historical identity chain. The regeneration log
is retained in `target/repin-v110-identity-regenerate.log`. No behavior rebaseline or
scientific/release qualification is implied.

## `feat-math` W1-W9 bit-exact HPC hotpaths port, 29 October 2026

Source `da2d02e8a0b54d615ad08e9509f20d2f2e1532dd` ports the bit-exact W1-W9 HPC hotpaths
(`ops.rs`, `ksmooth.rs`, `heads.rs`, `model.rs`, `advanced_v3.rs`, `advanced_v3/input.rs`,
`advanced_v3/pipeline.rs`, `advanced_v3/emit/mod.rs`, `artifact_manifest.rs`, `emit_rs.rs`,
`emit_rs/recipe_bundle.rs`, and `transpile.rs`) from `JopeDimeMicro` into `crates/feat-math`. All
numerical evaluations, serialized model payloads, emitted C++/Rust/JS/TS/Go/Python artifact bytes,
vocabulary/registry pins, and seeded generator/search streams remain bit-identical (`0 ULP`).
Because ADR-0004 hashes the `crates/feat-math/src/*` implementation files in
`evaluator_identity.v1.json`, this commit announces an identity-only transition chaining after
`search_narration_identity_rebaseline.v1.json`.

| Field | Before W1-W9 port | After W1-W9 port |
| --- | --- | --- |
| `implementation_sha256` | `6fe4aec834dc7ffae6fc856842111990f0b7ed39c2945e1d306fe1784aad7bd7` | `e50c8bbaf46bd559ec3f2a6bcf56a1a4732dbf29e68de7946fd962680058bdc8` |
| `cargo_lock_sha256` | `186e9ab64fbe50c507685c0f57af905045576fabc10e31988a595879bbe7a78d` | `186e9ab64fbe50c507685c0f57af905045576fabc10e31988a595879bbe7a78d` |
| `identity_digest` | `0da79c46a83bf6b639ef97a51a6fc210612cdac279b13a24c5d8298ce74d6d1c` | `0523027b6e356862d99ddce894ee81bc5c3728594dad6fe654e3bbb1c81a3f28` |
| `dsl_abi_sha256` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` |

`Cargo.lock` and `dsl_abi_sha256` are unchanged. The identity-only regeneration and subsequent
comparison mode both pass all 13 `compat_freeze` tests and all 3 `historical_v1` tests without
modifying any behavior fixture.
