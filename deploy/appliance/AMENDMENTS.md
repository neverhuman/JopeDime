# Appliance specification amendments

[SPECIFICATION.md](SPECIFICATION.md) is imported unchanged from #390. This file records where the
implementation in this change corrects it, decides something it left open, or deliberately
differs from it, with the evidence. Where an amendment and the specification disagree, the
amendment describes what the code does; the specification's remaining requirements still apply.

Measurements below were taken on 2026-10-21 in a 2-vCPU Linux x86-64 container with Rust 1.97.0
stable (the repository pins 1.97.1), from the head of this branch.

## A1 — `feat-math` embeds C++ today; the appliance graph excludes it

**Finding.** `crates/feat-math/build.rs` compiles `cpp/fitted_runtime.cpp` with `g++` into a
shared library, embeds it with `include_bytes!`, and loads it at run time through `memfd_create`
and `dlopen` (`advanced_v3/fitted_cpp_runtime.rs`, `fitted_runtime_image.rs`). Any binary that
depends on `feat-math` with default features therefore carries compiled C++, which contradicts the
"100% Rust" boundary of §3.2 and is invisible to `cargo tree`-only audits. The only caller is
`jope-campaigns` (`FittedCppRuntime`).

**Change.** A `fitted-cpp` feature, default-on so every existing build is unchanged, gates the
build step, the two modules and their re-exports; `tests/advanced_v3_fitted_cpp.rs` requires it.
`jope-runtime` and `jope-policy` take `feat-math` with `default-features = false`. Because the
C++ library is loaded through `memfd` and `dlopen`, neither a package-name audit nor `NEEDED`
would see it come back through feature unification, so the dependency fence
(`dependencies-deny.toml`) also denies the *feature* `feat-math/fitted-cpp` (and
`jope-assets/pack`) in every audited feature set; its negative control finds it enabled under
`jope-bench`. Together with the package deny lists and the `NEEDED` check in `build.sh`, this keeps
native code out of `jopedime`: its closure is 75 crates, none of which builds C or C++, and the
executable links only `libc.so.6`, `libm.so.6` and `libgcc_s.so.1`.

## A2 — The shared native path was extracted, not copied

§4.2 asks for the roster and the schema-bound fit/predict adapter to be shared. Implemented as a
*move* with re-exports, so the research panel and the appliance run one implementation:

| Moved to `jope-runtime` | From | Old path keeps working through |
| --- | --- | --- |
| `RustMethod`, `INPUT_POLICY`, `INPUT_STAGE`, the 17/5/4 recipe roster, `task_spec`, `validate_classes` | `jope-bench/src/panel/methods.rs` | `pub use` plus `Task → TaskKind` wrappers with the original messages |
| `Predictions`, `PROBABILITY_SUM_TOLERANCE`, `validate_probabilities`, `predict_bound` | `jope-pipeline/src/evaluation_score.rs` | `pub use` |
| `RAW_CELL_ENCODING`, `encode_cell`, `raw_table`, `FITTING_INPUT`, `fitting_input_sha256` | `jope-pipeline/src/prepared_data.rs` | `pub use` and two-line adapters over the pipeline's column type |

The panel's fit worker now fits through `jope_runtime::methods::native::fit_native`. The runtime's
task contract is `feat_math::advanced_v3::TaskKind`, so the deployment graph does not carry the
dataset catalog crate. No numerical code or serialized form changed: the jope-pipeline library
suite (292 tests) and the jope-bench panel suite (75 tests) pass unchanged on this branch.

## A3 — CUDA linkage is confirmed; this change ships the CPU build only

§11.1 is right: Candle 0.11.0 requests cudarc `dynamic-linking`, so a CUDA build would carry
`DT_NEEDED` entries for the CUDA libraries and fail to start on a CPU host. A second cudarc
(0.17.8, through `ug-cuda`) is also in `Cargo.lock`. Nothing in this change claims GPU execution:
`--device cuda:N` exits 4, `auto` selects the CPU and says why, `doctor` reports the driver probe
from `/proc` and `/dev` without loading any library, and the fence denies `candle-*`, `cudarc`,
`gemm*`, `ug*`, `tokenizers` and `onig*`. The pinned linkage patch, the same-artifact
CPU-host/driver-only-host test and the bundled-library inventory (P1 second half, P5) amend the
fence in the same reviewed change.

## A4 — CI: one existing lane and the product test scope, not seven new lanes

§15.2 proposes seven new lanes. On this repository a new workflow is stopped by runner admission
(`deploy/runner/job-policy.py` admits jobs only from `ci.yml`, `full-main.yml` and
`main-guard.yml`), GitHub artifact actions are banned (CI evidence stays on runner disk, #341),
and a tracked file over 5 MiB fails hygiene. So:

- `jope-assets`, `jope-runtime`, `jope-appliance` and `jope-dist` join the `rust-test` product
  scope (`deploy/ci/rust-test-scope.sh`); their 14 test targets are registered in
  `deploy/ci/coverage.json` and classified in the regression test inventory.
- The `e2e-smoke` lane builds `jopedime` with the PR profile, packs and signs the fixture profile,
  and runs `test-clean-host.sh` on the packed file. Its receipts (build, pack, verify, size, SBOM,
  clean-host, fit) are lane evidence; the packages stay in the lane's target and are removed.
- `ci-plan.py` and its Rust port gate both lanes on `crates/jope-{assets,runtime,appliance,dist}/`,
  `deploy/appliance/`, `models/bundles/` and `docs/release/signing.pub`.
- No fixture weight is committed: the fixture bundles' manifests are committed and
  `jope-dist fixture` regenerates the bytes deterministically and refuses on drift.

The trusted full-weight, GPU, performance and release lanes of §15.2 remain the target; they need
provisioned runners and the release-host integration (A10).

## A5 — Package format v1: the manifest is signed, assets are authenticated on read

The footer is §10.2's 128-byte layout, field for field, except bytes 96..128: they carry the
sha256 of the manifest instead of being reserved. The signature covers

```text
"jope.singlefile.manifest-signature.v1\0" || footer (128 bytes)
```

instead of `SHA256(prefix) || footer`. The footer binds the manifest; the manifest binds the
executable's length and sha256 and, for every asset, the sha256 of its stored frames and of its
decoded bytes. That is the "clear code-authentication story" §10.6 asks for:

- **Before running.** A download is authenticated with the signed `SHA256SUMS` or
  `jope-dist verify`, which checks the signature with independently held keys and then covers
  every byte: the executable hash, zero padding and every zero gap between frames, every frame,
  the manifest, and the whole-file sha256. A modified executable cannot vouch for itself, so
  §10.4's limit applies equally to both designs.
- **While running.** Every open reads and verifies the footer, the manifest and the signature,
  checks that the signer's role matches the package's scope (the development key signs only
  development packages; a release key never does), and hashes the executable region (a few MiB)
  against the manifest, so a signed manifest cannot be grafted onto another executable. It never
  hashes the 400+ MiB payload. An asset's stored frames are hashed and compared with the signed
  manifest *before* any byte reaches the decompressor; the decoded bytes are compared again. An
  asset a run does not need is never read. `jopedime verify-self` performs the full check on
  demand.

A whole-prefix signature would make every invocation of the full profile read and hash about
440 MB before printing `--version`; per-asset authentication gives the same guarantee for every
byte that is used, and the full check remains one command. `pack` and `seal` write the package
beside its destination, re-open it with the signing key alone, cover every byte, and only then
move it into place without overwriting anything. Further decisions inside the format:

- Fixed-width `r || s` signatures with a low `S`: `jope-dist seal` converts cosign's DER and
  normalizes `S`; readers refuse a high `S`, so a sealed file has exactly one valid encoding.
- The key id is the sha256 of the signer's SPKI DER, distinct from the PEM file's hash.
- Frames are 4096-aligned, at most 16 MiB decoded by default and 64 MiB by bound, decode in
  parallel, and must declare their content size, use no dictionary and carry no trailing bytes.
- Identical content is stored once; only whole frame lists may be shared, never single frames.
- Codecs: `shuffle+zstd` (safetensors byte planes regrouped by dtype width, then zstd 19),
  `zstd`, and `raw`; a compressed form that is not smaller is stored raw. All are lossless, the
  packaging-only identity class of §8.2: accepted model hashes stay valid.
- Bounds (`limits.toml`, mirrored by a test): the 16 MiB manifest cap is checked before the
  manifest is read, and bounds every later parse; then 4096 entries, 1024 methods, 8 GiB per
  asset, 64 MiB per frame and zstd window and 256-byte names are checked before any asset,
  frame or decode buffer is allocated.
- The manifest has one canonical byte string (serde field order, no whitespace, sorted entries
  and methods); any other encoding of the same values is refused.

Schemas use the repository's naming: `schemas/jope.appliance-manifest.v1.schema.json`,
`jope.appliance-model-lock.v1`, `jope.appliance-fitted.v1` and `jope.appliance-run.v1`.

## A6 — One trust anchor: the existing release key

`jope-assets/build.rs` compiles in `docs/release/signing.pub`, the public half of the root-held
cosign key that already signs every release's `SHA256SUMS`. No second release key exists to
provision, rotate or leak. Fixture packages are signed by a development key derived in code from a
fixed label; only builds with the `dev-keys` feature trust it, and `jopedime --version --json`
reports `trusts_development_key` so the release lane can refuse such a build.

## A7 — Rights block a profile; they never pass silently

`jope-dist lock` derives `models.lock.json` from the profiles and the committed bundle manifests.
A file whose redistribution class the scope does not admit, a `prohibited` file without its owner
acknowledgement, weights with no licence record (for example the shuffled TabICL regressor), or a
promised method this build cannot execute is recorded as a *blocker*; the lock still derives, CI
checks it without weights, and `pack` refuses a blocked profile. Each scope also has a fixed
ceiling that a profile cannot widen: `development` and `public` admit only `permitted` files
(development packages run on untrusted CI runners), `internal` adds `internal` and
`conditional`, and only `internal-restricted` may add `prohibited`, which also requires an
`owner_ack`. Today `fixture` and `native-only` are ready. T0115 D3 supplies a scoped
acknowledgement for Starlight in internal qualification/rehearsal candidates; public use remains
excluded. `full` remains blocked by the uncompiled proposal adapter. The full
profile's five accepted tensor files total exactly the 438,062,296 bytes of §5.1 (asserted).

## A8 — Commands implemented, and what is not yet

Implemented: `--version [--json]`, `version`, `doctor`, `methods`, `fit`, `predict`,
`fit-predict`, `verify-self` and `licenses`, with `--device`, `--threads`,
`--time-limit-seconds`, `--max-input-mib`, `--max-rows`, `--receipt` and `--json`, and the exit
statuses of §13.3 (2 input, 3 integrity, 4 backend, 5 budget, 6 numerical, 7 output).
`crates/jope-appliance/tests/single_file.rs` produces 2, 3, 4, 5 and 7; 6 is the category of
any error without one, including a contained panic, which `jope-runtime`'s unit tests cover.
`--receipt` and `--output` may not name an input (or each other), and a failed run's receipt
records the budget it consumed. A closed stdout keeps the run's exit status.

Not yet: `cache inspect|prune`, `--cache-dir` and `--no-cache` (this build keeps no cache: assets
are authenticated and decoded in memory when needed), `--memory-limit-mib` (no enforcement exists
to name), `--precision` (CPU f64 only), and `--seed` (selection is deterministic and seedless).
Same-binary worker processes (§4.3) are not implemented: the deadline is cooperative, checked
before every selection fold fit and before fitting a named method; once `auto` has chosen a
recipe, its final refit always runs and is charged, so consumed time can exceed the limit.
Every receipt states this. A panic inside a method is contained (`catch_unwind`) and becomes that
candidate's failure, or exit 6.

## A9 — A promised method that cannot run fails, it is not dropped

A package's manifest lists the methods its profile promises. `jopedime doctor` exits 3 and names
each promised method this build cannot execute; `jope-dist lock` records the same as a blocker.
Requesting an uncompiled neural method exits 4; nothing substitutes another method under its id.

## A10 — Selection, fitted state and receipts

- `--method auto` runs `jope.runtime-select/v1`, a new runtime behaviour with its own identity:
  the task's native roster (or `--candidates`), `stratified-round-robin/v1` folds (deterministic,
  `min(5, rows/2, smallest class)`), out-of-fold RMSE, `1 - AUROC` or log loss (the panel's
  primary metrics), lowest loss wins with ties to roster order, failed (or panicking) candidates
  recorded, then one refit on all rows. Preprocessing is refitted inside every fold. Measured: 400 rows × 3
  features, all 17 regression recipes, 2 threads: 214 s, of which `portfolio` takes 165 s;
  `--candidates` and `--time-limit-seconds` bound it.
- `model.jope` (`JOPEFIT1`): a canonical JSON header and the native bound model, length-prefixed,
  with a trailing sha256. Reload checks the header against this build's registry, the recipe hash,
  the bound model's own binding and the preprocessing policy; every native recipe reloads to
  bit-identical predictions (`tests/offline_fit_predict.rs`). The header states that kernel and
  smoother heads embed training rows.
- Prediction input matches feature columns by name in any order; an unknown column is refused
  except the training target's, which is ignored and reported. Outputs are written atomically.

## A11 — Release-host integration is the next step, and is not claimed here

`build.sh` stops release scopes at a stage directory whose `signing-message.bin` the release
host's root-held cosign step signs; `jope-dist seal` checks that signature against the release key
and re-verifies every byte. Wiring this into `deploy/release/release.sh` (its fixed root sign
script, `test-release.sh` counts, per-edition sums and publication scope) is P6; this change does
not modify the release procedure, and `native-only` is not published by it.

## A12 — A pipeline finding for any future installed binary

`jope-pipeline`'s `run::git_sha()` runs `git rev-parse HEAD` in the caller's working directory.
In an installed binary that stamps the *user's* repository into receipts. The appliance does not
depend on `jope-pipeline` (the fence denies it), so this change leaves it alone; it must be fixed
before pipeline code enters any shipped executable.

## A13 — Stripping fixed; size rebaselined for regression objectives

`[profile.dist]` says `strip = "symbols"`, but the executable kept its symbol tables: Cargo (1.95
and 1.97, reproduced with a one-file crate) passes `-C strip=debuginfo` to a package whose profile
override does not repeat `strip`, and `jope-appliance`, which links the executable, has one
(`opt-level = "s"`). The override now states `strip = "symbols"`. On main (1240120d)
that removes 591 KB from the `dist` executable (3,738,480 → 3,147,304 B) and 1.75 MB from the
`dist-ci` one (5,435,360 → 3,684,472 B).

Regression objectives (PR #411) then add 898,952 B to `dist` (+28.6% over stripped main) and
928,584 B to `dist-ci`, beyond the 5% tolerance, so `size-budgets.toml` was rebaselined to the
measured head with that attribution: the objective grammar (parse, validate, evaluate) is about
300 KB of machine code, the JSON-RPC/MCP adapters 80 KB, objective-aware selection and
calibration 45 KB, shared `serde_json`/`core` instantiations the rest of +688 KB of code, with
unwind tables, read-only data and relocations in proportion. No dependency was added. Net of
both, the native-only package is 4,065,423 B, 8.9% above its 2026-10-21 baseline of 3,733,648 B
(sealed with an independent P-256 key, as below).

## Measured sizes and build times

| Build | Executable | Package |
| --- | ---: | ---: |
| `--profile dist` (fat LTO, one codegen unit, stripped), `embedded` | 4,046,256 B | native-only 4,065,423 B |
| `--profile dist-ci` (thin LTO, 16 units), `embedded,dev-keys` | 4,613,056 B | fixture 4,730,485 B |
| 2026-10-21, before A13 (symbols not stripped): `dist` | 3,713,088 B | native-only 3,733,648 B |
| 2026-10-21, before A13 (symbols not stripped): `dist-ci` | 5,429,128 B | fixture 5,545,586 B |

`dist` builds in about 3 minutes and `dist-ci` in about 1.5 minutes from a warm dependency cache on
2 vCPUs. The native-only package was staged, signed with an independent `openssl` P-256 key
(base64 DER, as cosign writes it), sealed and verified with that key alone; the release build
refuses to run it (exit 3) because it trusts only `docs/release/signing.pub`, as intended. Budgets with a 5% tolerance are in `size-budgets.toml`; the full profile's entry is
provisional until its first internal-restricted build.


## A14 — Size attribution for immutable model objects and sessions

PR #413 adds the manifest/hashed-part store, create-only publication and import/export,
append-only session history, CLI/MCP interfaces and strict known-state JSON consumption.
Its pinned `serde_ignored` adapter is the sole new dependency. Neither the numerical
algorithms, optimization settings, signature verification nor 5% size tolerance changes.

A fresh baseline build of PR #411 (`e04cb3fdf3f6dade3e5d017419a126694e15e65d`)
on xbabe2 with Rust 1.97.1, `dist-ci`, `embedded,dev-keys` reproduced exactly 4,613,056 B
of code and a 4,730,485 B fixture package. The integrated source
`d04153fa6715c723522e8cf8d36bd8af8bf6f8ae`, built identically, measures 5,319,208 B
and 5,435,014 B: +706,152 B of executable (15.3%) and +704,529 B packaged (14.9%).
ELF comparison attributes 525,584 B to machine code, 32,576 B to
read-only data, 133,768 B to
exception/unwind tables and 7,728 B to dynamic relocations. These are
whole-feature section deltas; per-function attribution is not claimed.

The native-only `dist`, `embedded` build measures 4,542,168 B of executable and
4,561,056 B packaged (prior A13: 4,046,256 / 4,065,423 B). Its stage was sealed and
verified with an independent P-256 validation key. The actual model command refuses that
key before store access (exit 3); the release signing key was not used. Version diagnostics
can report an untrusted package with exit 0, so that command alone is not a refusal test.
The fixture budget and native-only budget now name these measured sizes. The blocked
full profile keeps its provisional 460,000,000 B final limit and follows the new code size;
no full/neural package or release-host integration is qualified by this measurement.

Raw build, signature, verification, old-budget failure, ELF-section and refusal receipts:
`/home/ubuntu/jope-evidence/pr-release-drive-20261022T2319Z/413-SIZE-ATTRIBUTION.json`.


## A15 — Fixture size attribution for the fitted math-pipeline decoder

The postmerge manual run of PR #427 (`f35bf5102445cec8d0ac71e9ab2808f9f25eb9a9`,
run `40033940797`, attempt 1) built, development-signed and fully verified its fixture
package, then correctly refused the old 5% size budget. It measured 5,843,568 B of code
and 5,959,302 B packaged, versus A14 baselines 5,319,208 / 5,435,014 B. The fixture
tensor identities are unchanged and `gpu_library_bytes` remains zero.

A same-host comparison on xbabe2 used immutable premerge-main `f9543b85b356675d377e38d75c4e3ecea8b2d302`
and merged-main sources, the same SDK image `093036bacc0efdddf4965722fdbd96244d127c28a7d23b0ecdc4b60c41a43f27`,
Rust 1.97.1, source path, `dist-ci` profile and `embedded,dev-keys` features. It reused
one target with 4 CPUs, 12 GiB memory and no swap or network, and completed in 111 seconds
without OOM. Executables measure 5,319,832 and 5,843,120 B: +523,288 B (+9.84%).
ELF section deltas are 392,272 B `.text`, 47,552 B `.rodata`, 45,132 B `.eh_frame`,
5,440 B `.eh_frame_hdr`, 23,124 B `.gcc_except_table`, 4,848 B `.rela.dyn`,
4,848 B `.data.rel.ro`, 104 B `.got` and 1,000 B relocation padding. These are
whole-feature section measurements; per-function attribution is not claimed.

An unpublished diagnostic tree `0dc890110e71a503fb939a9874791e940fe67e93`
(commit `d72709e8eb2b30053dfde8c3e3aa6fe527cf2898`) restores only the three premerge
decoder/export entrypoints: `jope-appliance/src/main.rs`, `jope-runtime/src/model_object.rs`
and `jope-runtime/src/lib.rs`. All other merged files, including Go/CUDA emitters,
security pins and release versions, remain. Built identically, it measures 5,321,296 B,
so the decoder and fitted portable export account for 521,824 B of the observed increase;
the remaining merged changes contribute 1,464 B. That diagnostic removes the feature only
to measure it and is not the product fix. The decoder, export behavior and all source
from the eight integrated PRs remain enabled in the product.

The fixture baseline now uses the larger actual CI measurements. Standalone and CI code
differ by 448 B, so they are not presented as byte-identical reproductions. No numerical
fixture, optimization setting, dependency, signature check or 5% tolerance is relaxed.
The unmeasured `native-only` and blocked `full` budgets stay unchanged; this measurement
does not qualify a production-signed or neural package.

Raw source manifests, build/section/cost receipts and the original failing CI report are
retained under `/home/ubuntu/jope-evidence/pr-all-clearance-20261023/qualification/size-comparison-20261023`,
`qualification/decoder-size-ablation-20261023` and `emergency-execution/ci8-public-appliance-readback.json`.

## A16 — Opt-in Dope GPU encoding and a separate package profile

The `dope-cuda` feature adds GPU Dope512 companion encodings to native fitting. The native
feature sets and their dependency fence remain unchanged. `dope-gpu` is a development-signed
profile containing the whole released MIT MLC-v3 checkpoint and its license, authenticated by
`jope-assets` and checked against the canonical checkpoint digest again by `DopeEncoder`.
There are no runtime model downloads. Native methods continue to fit on CPU.

The reviewed Candle manifest switches cudarc from `dynamic-linking` to `dynamic-loading`.
Its source pin is recorded in `vendor/security/PROVENANCE.json`. This enables startup before
CUDA is selected. Explicit CUDA initialization failures are backend errors; they do not select
CPU. This profile requires an installed NVIDIA driver and compatible CUDA libraries when GPU
inference is requested. This change does not bundle the CUDA toolkit libraries or claim
qualification on a driver-only host.

`dependencies-dope-gpu.toml` audits the separate GPU feature combinations, permits Candle's
mandatory reviewed tokenizer/Oniguruma build dependencies, and forbids network/training crates,
packer code, fitted C++, cuDNN, NCCL, `ug`, and eager CUDA linking. The package build retains its
ELF `DT_NEEDED` gate. `test-dope-gpu.sh` checks the actual sealed package, finite CUDA encoding,
companion receipt, device-hidden startup and an explicit unavailable-device refusal. These
checks must pass before package qualification is reported. Historical CPU-only claims above
remain observations of their original builds.

## A17 — Fixture size attribution for runtime Search commentary

PR #438's manual run `40764758400`, attempt 1, source
`aff0e9532f10bdd2eabd2883a9f22ec21638debb`, passed the pipeline smoke and built,
development-signed and verified the fixture package. Its size report correctly
refused the previous 5% budget: 6,579,256 B of code and 6,696,997 B packaged, versus
5,843,568 / 5,959,302 B. The increases are 735,688 / 737,695 B. The three generated
fixture tensor/config identities and their 103,220 raw bytes are unchanged;
`gpu_library_bytes` is zero. The retained package was independently verified on xbabe2.

A controlled xbabe2 comparison uses Rust 1.97.1, the same source path, reused lane
target, `dist-ci` profile, `embedded,dev-keys` features, source stamp and dependency
graph, with four build workers and one math thread. The full executable is
6,579,064 B. Restoring only the pre-feature runtime observer entrypoint files
(`jope-runtime/src/api.rs`, `math_pipeline.rs` and `lib.rs`) from main
`7c78fcff77da69660835b9fbb7acfadad3eea5d3` produces 5,878,648 B: the reachable
runtime observation code accounts for 700,416 B. Restoring the exact product files
and rebuilding reproduces the full binary byte for byte, SHA-256
`22cdb8356d8b79bbe44c6ccfd9daf0f4bac652bd083d250c4908b354b553d7a0`.
The diagnostic source change is preserved as a patch and is not the product fix.

The entrypoint-ablation ELF section deltas include 541,152 B `.text`, 27,200 B `.rodata`,
62,916 B `.eh_frame`, 9,304 B `.eh_frame_hdr`, 30,364 B `.gcc_except_table`,
17,304 B `.rela.dyn` and 11,664 B `.data.rel.ro`. These are section measurements,
not per-function attribution. The diagnostic retains the current dependency
graph and all other branch source; it is not a full pre-feature main checkout.
Its remaining 35,080 B over the historical code baseline is not separately
attributed and is within the old code budget's tolerance.

The fixture baseline uses the larger actual CI sizes. CI and the standalone full
build differ by 192 B and are not presented as byte-identical reproductions.
The 5% tolerance, signature checks, numerical fixtures, optimization settings and
all other profile budgets remain unchanged. This evidence qualifies the fixture
profile; production-signed, neural and GPU packages retain their separate gates.

Raw CI artifacts, their independent verification, the source snapshots and ablation
patch, compiler/dependency inventories, ELF sections and all three build receipts
are retained under `/home/ubuntu/jope_plan/search-narration-20261028/evidence/`:
`appliance-size-aff0e-ci/`, `appliance-size-ablation-source/`,
`appliance-size-pair-artifacts/`, `appliance-size-attribution-aff0e.json` and
`local-checks/size-{full,observer-off,full-restored}-aff0e-a1.{json,log}`.

## A18 — Fixture size attribution for feat-math W1-W9 HPC hotpaths (PR #440)

PR #440's PR CI run `40852221193` (job `123374364946`) and full `workflow_dispatch`
run `40852222525` (job `123374415516`) at source `ddbc1516a162b115cc5407295f305bf0afde2005`
passed the e2e pipeline smoke (`34 rows; programs agree: python rs`), dependency fence
(`3 feature set(s)`), ELF `DT_NEEDED` gate (`libc.so.6`, `libgcc_s.so.1`, `libm.so.6`),
and third-party notices generation, then refused the previous 5% fixture size budget
during `pack and sign (development key)`.
Building `deploy/appliance/build.sh --profile fixture --cargo-profile dist-ci` at
`ddbc1516a162b115cc5407295f305bf0afde2005` on xbabe2 development-signed and verified the
fixture package (`verify.json`: `ok = true`) and measured 7,073,600 B of code
(SHA-256 `7e4cde137fcc46897e6b1a27cdea7f0992a3e2c3cb3764ef49b51a3bd6db4eee`) and
7,188,517 B packaged (SHA-256 `49fc0a96b146b5ebd060da0bdd749f98c15535c3b76742d9df1580b9f1a55878`),
versus A17 baselines 6,579,256 / 6,696,997 B (+494,344 B code / +491,520 B packaged vs A17
CI baseline; +494,536 B vs A17's 6,579,064 B xbabe2 standalone code baseline). The three
generated fixture tensor/config identities (103,220 raw bytes) and two notice entries are
unchanged; `gpu_library_bytes` is zero. Clean-host network-isolated execution
(`test-clean-host.sh`) passed (`ok = true`).

A controlled xbabe2 three-phase comparison uses Rust 1.97.1, the same source path, reused
lane target, `dist-ci` profile, `embedded,dev-keys` features, source stamp and dependency
graph. The `full` executable at `ddbc1516a162b115cc5407295f305bf0afde2005` measures
7,073,600 B (`7e4cde137fcc46897e6b1a27cdea7f0992a3e2c3cb3764ef49b51a3bd6db4eee`). Restoring
only `crates/feat-math/src` from pre-H4 main `2f85b953f6776d566c2170216c4821e526ba7b60`
(`feat-math-off`) produces 6,579,064 B
(`0243e7583200d2af2c2418235d844e5db425826325365b090cb6c2b7a23e803b`), matching A17's
6,579,064 B xbabe2 executable with 0 B unattributed difference: the `crates/feat-math/src`
W1–W9 hotpath additions account for all 494,536 B of the same-host code increase. Restoring
the exact H4 `crates/feat-math/src` files and rebuilding (`full-restored`) reproduces the
7,073,600 B binary byte for byte, SHA-256
`7e4cde137fcc46897e6b1a27cdea7f0992a3e2c3cb3764ef49b51a3bd6db4eee`. The diagnostic source
change is preserved as a patch (`SHA-256: 3bc89eaf45516c8d964e9dcfbfdd5d43a51ffbfbc8b7fbf72986fe2c9cdbbe92`)
and is not the product fix.

The `crates/feat-math/src` ablation ELF section deltas (`full` minus `feat-math-off`) are
359,728 B `.text`, 66,572 B `.eh_frame`, 53,348 B `.gcc_except_table`, 7,040 B `.eh_frame_hdr`,
3,144 B `.rela.dyn`, 3,112 B `.data.rel.ro`, 1,536 B `.rodata`, 1,088 B `.relro_padding`,
56 B `.got`, and 8 B `.data`. These are whole-crate section measurements; per-function
attribution is not claimed.

The fixture baseline in `size-budgets.toml` now records the measured 7,073,600 B code and
7,188,517 B package sizes. The 5% tolerance, signature checks, numerical fixtures,
optimization settings and all other profile budgets (`native-only`, `full`, `dope-gpu`)
remain unchanged. This evidence qualifies the fixture profile; production-signed, neural
and GPU packages retain their separate gates.

Raw build/verification/size/clean-host receipts, the ablation patch, compiler/dependency
inventories, ELF sections and the three paired phase artifacts are retained under
`/home/ubuntu/jope_plan/evidence/h4-appliance-size-ddbc1516a-20261029/`:
`build-fixture-ddbc1516a/`, `clean-host-ddbc1516a/`, `ablation-source/`,
`pair-artifacts/{full,feat-math-off,full-restored}/`, and
`appliance-size-attribution-ddbc1516a.json`.
