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
`owner_ack`. Today `fixture` and `native-only` are ready and `full` is blocked by the
`prohibited` Starlight foundation regressor and the uncompiled proposal adapter. The full
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
