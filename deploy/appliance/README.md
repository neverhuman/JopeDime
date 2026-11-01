# Single-file Rust deployment

This directory versions the engineering plan for shipping JopeDime as one signed,
offline Rust executable with its selected pretrained models and native AutoML
alternatives. The proposed installed command is `jopedime`; the initial target is
Linux x86-64 with glibc 2.36 or newer.

**Status: CPU appliance implemented (native-only and fixture profiles); neural, GPU and
release-host integration pending.** This directory holds the specification (#390, imported
unchanged), its [amendments](AMENDMENTS.md), and the deployment inputs the implementation reads.
The existing [`jope` release procedure](../../docs/release/README.md) is unchanged and still
describes the only published release; no `jopedime` package has been published.

What exists, with the evidence that proves it:

| Capability | Where | Proof |
| --- | --- | --- |
| Signed single-file package: footer, manifest, signature, bounded per-asset authentication | `crates/jope-assets` | `tests/envelope.rs`, `tests/decode_bounds.rs` |
| Native fit, save, reload and predict for all 26 task/recipe pairs; bounded `auto` selection | `crates/jope-runtime` | `tests/offline_fit_predict.rs`, `tests/method_roster.rs` |
| The `jopedime` executable: commands, exit statuses, receipts, self-verification | `crates/jope-appliance` | `tests/single_file.rs` |
| Model lock with rights blockers, deterministic pack, release stage and seal, audit, notices, SBOM, provenance | `crates/jope-dist` | `tests/closure.rs`, `tests/reproducible_pack.rs`, `tests/dependency_fence.rs`, `tests/schemas.rs` |
| Fixture appliance built, packed and run alone in an empty directory with an empty environment; without a network and under `strace` where the host provides them (the receipt records which) | `build.sh`, `test-clean-host.sh` | `e2e-smoke` lane receipts |

Not yet: the accepted step-60000 proposal path and the neural alternatives (P3–P4), CUDA (P1
second half, P5), the derived-asset cache and worker processes (P5), and `release.sh`
integration and publication (P6). The `full` profile is locked and *blocked* until then; see
[models.lock.json](models.lock.json).

## Internal fused CPU edition (ENG R2)

`fused-cpu` stages an internal release package with the complete source-pinned R882
DOPE/adapter/bank/config/golden closure and the same embedded Engine/MCP/client build as
`full`. It needs the verified real model cache and release signing. The complete compiled
R882 dependency set is checked by `jope-dist`; `full` separately retains its legacy
foundation closure and existing rights acknowledgement. No weights or learning criteria
change. Authentication and size do not qualify the scientific or release gates.

The previous synthetic development package is `fixture-fused-cpu`. It remains a fixture
and cannot satisfy D2. D2 uses the authenticated `fused-cpu` package, with its unchanged
40,000,000-byte ceiling and 5% baseline tolerance.

## Build and run

```bash
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
# CI fixture: build, pack and sign with the development key, verify, size and SBOM.
bash deploy/appliance/build.sh --profile fixture --cargo-profile dist-ci --out target/appliance/fixture
bash deploy/appliance/test-clean-host.sh target/appliance/fixture/jopedime target/appliance/clean-host
# Release scopes stop at a stage; the release host signs stage/signing-message.bin with cosign:
bash deploy/appliance/build.sh --profile native-only --out target/appliance/native-only
cargo run -p jope-dist -- seal --code target/appliance/native-only/code \
  --stage target/appliance/native-only/stage --signature manifest.sig --out jopedime
cargo run -p jope-dist -- verify --file jopedime     # trusts docs/release/signing.pub
```

```bash
jopedime --version --json
jopedime doctor
jopedime fit --train train.csv --target y --output model.jope          # --method auto (default)
jopedime predict --model model.jope --input rows.csv --output predictions.csv
jopedime fit-predict --train train.csv --target y --input rows.csv --output predictions.csv
jopedime verify-self
jopedime licenses
```

Exit statuses: 0 success, 2 invalid input, 3 integrity or compatibility, 4 backend unavailable,
5 time budget exhausted, 6 numerical failure, 7 output failure.

| File | Purpose |
| --- | --- |
| [SPECIFICATION.md](SPECIFICATION.md) | Complete 22-section engineering specification, architecture diagrams, exact file changes, formats, CI, release flow, and acceptance criteria. |
| [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) | Ordered implementation work packages and the evidence required to close each one. |
| [AMENDMENTS.md](AMENDMENTS.md) | Corrections and decisions made while implementing: C++ in `feat-math`, CUDA linkage, CI shape, package format v1, trust anchor, rights, selection, measurements. |
| [profiles/](profiles/) | `fixture` (CI), `native-only` (public), `full` (internal-restricted, blocked). |
| [models.lock.json](models.lock.json) | Each profile's exact asset closure, rights and blockers (`jope-dist lock --check`). |
| [dependencies-deny.toml](dependencies-deny.toml) | The shipped dependency fence (`jope-dist audit`). |
| [limits.toml](limits.toml), [size-budgets.toml](size-budgets.toml) | Package/input bounds and measured size budgets. |
| [fixtures/bundles/](fixtures/bundles/) | Manifests of the generated CI fixture tensors (no weights are committed). |
| [build.sh](build.sh), [test-clean-host.sh](test-clean-host.sh) | Build-pack-verify, and the empty-directory run. |

The specification was prepared on October 20, 2026, America/Denver, against commit
[`95dcb0c0e44d95184607740aacd387bf5c007e89`](https://github.com/neverhuman/JopeDime/commit/95dcb0c0e44d95184607740aacd387bf5c007e89)
and is imported unchanged. Its statements about work performed describe that review.
Design changes made while implementing it are recorded in [AMENDMENTS.md](AMENDMENTS.md)
rather than by editing the specification.

## Implementation contract

- Embed the selected safetensors, encoder configurations, preprocessing assets,
  vocabulary, notices, and runtime in the signed executable. Training and conversion
  may continue to use other languages.
- Keep application execution in Rust and offline. Reuse the existing native methods
  and model ports; external Python benchmark adapters are not Rust alternatives.
- Make the same GPU-capable Linux artifact start on a host without CUDA and select
  CPU when an eligible GPU path is unavailable. A driver-only NVIDIA host must pass
  the full GPU qualification using approved bundled user-space libraries.
- Preserve accepted model identities through lossless packaging. Treat precision
  changes and quantization as separately qualified model variants.
- Resolve every method's exact model dependency graph and distribution rights.
  An internal profile name does not grant redistribution rights.
- Extend the existing CI receipt/aggregate system and release authority. Bind
  qualification and publication to the exact signed candidate bytes.

The specification distinguishes Rust application code from host ABI dependencies,
GPU drivers, vendor libraries, and GPU kernels. Each supported platform requires
its own artifact and qualification.

## Review map

| Review area | Specification |
| --- | --- |
| Source audit and product scope | [Sections 2–3](SPECIFICATION.md#2-evidence-boundary-and-repository-audit) |
| Runtime ownership and model dependencies | [Sections 4–6](SPECIFICATION.md#4-architecture-and-ownership) |
| Exact new and modified files | [Section 7](SPECIFICATION.md#7-repository-layout) |
| Export, verified assets, and package signature | [Sections 8–10](SPECIFICATION.md#8-export-and-safetensors-contract) |
| GPU/CPU execution | [Section 11](SPECIFICATION.md#11-gpucpu-deployment-engineering) |
| CI and release orchestration | [Sections 15–16](SPECIFICATION.md#15-ci-design) |
| Qualification and staged PRs | [Sections 17–18](SPECIFICATION.md#17-acceptance-and-qualification) |
| Examples, unresolved evidence, and sources | [Sections 19–21](SPECIFICATION.md#19-machine-readable-contract-examples) |
| Product completion | [Section 22](SPECIFICATION.md#22-definition-of-done) |

Retire CUDA linkage, full model dependency resolution, distribution rights, and
artifact-size uncertainty early. Follow [P0–P1](IMPLEMENTATION_CHECKLIST.md#p0--inventory-and-contracts)
for the first executable PR. The examples in the specification are proposed
contracts, not production locks or release-ready configuration.

## Business-specific regression objectives

Automatic selection supports 38 regression objectives and portable custom costs through
`--fitness mae`, `--fitness pinball:quantile=0.9`, `--fitness @cost.json`, Rust, JSON-RPC and MCP.
Objective-aware output calibration (`--calibration`, on by default with an objective) is
cross-fitted during selection and stored in the model file; columns an objective reads are never
features. See [the objective guide](../../docs/regression-objectives.md) and
[41 executable recipes](../../examples/objectives/README.md). Order-dependent objectives require
chronological forward validation; defaults remain RMSE.

### Internal full-asset rehearsal

`build.sh --profile full --rehearsal` builds the same `embedded,engine,mcp,client`
CPU executable and resolves every full-profile asset, hash and rights record. It
writes `rehearsal/frames.bin`, `manifest.json` and `rehearsal.json` for internal
inspection while a promised runtime method is unavailable. The receipt records
the intended method roster and every unavailable method; no capability is enabled
or qualified by packaging its weights. Asset, rights and payload blockers still
refuse the operation. This mode requires the full internal-restricted profile and
its explicit owner acknowledgement.

This output is not a release candidate: it contains no footer, signing message or
sealable stage, and `jope-dist seal` refuses it. Normal `pack --stage` retains its
unavailable-method refusal. A full signed candidate and actual release gates still
require the proposal runtime implementation. Do not submit a rehearsal to signing
or report source controls as release-gate passes.
