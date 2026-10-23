# CUDA C++ local qualification handoff

PR: https://github.com/neverhuman/JopeDime/pull/422

Branch: `feat/cuda-cpp-target-20261023`

Date: 2026-10-23

## Scope and user direction

Finish local hardware verification of the CUDA C++ output target. All CUDA-specific
hosted workflows, runner requests and hardware merge gates have been removed. Keep
hardware work local; do not add GPU CI or change runner policy. Standard Rust checks exclude CUDA contract/census targets and the CUDA golden
hook. The local qualifier explicitly selects the nondefault `cuda-parity-tests` feature.

The generated library provides every recipe accepted by the Rust emitter and every
fitted predictor accepted by bound Rust through a shared native Rust implementation.
Rust is required to build the bundle. Fitting and feature processing execute on the
CPU. Explicit GPU inference supports ridge, elastic-net and ridge-GCV heads with no
target mapping. Bound/string models, nonlinear heads, symbolic heads and target
mappings deliberately refuse device inference. Universal GPU model execution is an
open capability gap, separate from verification of the implemented linear heads.

## Verification completed here

Production emitter/native/kernel source was qualified at
`73c486c743a3dd9b5d693bcdc10d958468f3ffc4`. The final follow-up changes test fixtures,
local-only test selection and documentation. The later local-only CI commits and
identity successor are preserved; production numerical and emitter code remains
unchanged. Compilation inputs for generated CUDA libraries are
unchanged; the six generated files for the default fixture are hash-identical.
The final branch SHA and receipt are recorded in the PR review.

| Complete local compile-mode suite | Passed | Failed / ignored |
|---|---:|---|
| Target contracts | 11 | 0 / 0 |
| Registry/v2 census tests | 5 | 0 / 0 |
| Existing golden matrix | 89 | 0 / 0 |
| Selected ordered-bound tests | 2 | 0 / 0 |
| Artifact binding | 21 | 0 / 0 |
| Pipeline publication | 19 | 0 / 0 |
| CLI adapters | 2 | 0 / 0 |

The census executed 53 recipes / 8 matching Rust refusals, 24 advanced-bound
operations / 1 refusal, 14 legacy-bound operations / 39 matching Rust refusals,
and 101 scalar/row-vector cases. Compatibility (13) and historical fixtures (3)
also passed at the qualified source.

A combined cloud call stopped around 15 minutes during goldens. Its incomplete log
is unqualified. The whole golden suite was replayed (89/89 in 665.90 seconds), then
the remaining suites completed at the same clean source. Receipt creation and
verification both passed. The receipt reports zero device batches and GPU execution
as unknown.

Red-team failures reproduced CMake target collisions, partial/duplicated evidence
acceptance, the missing SSE rounding guard and a CMake dependency in ordinary CPU
CI. All have regression coverage. The final explicitly selected host suite passes 11 tests
with CMake deliberately unavailable; complete CMake checks still run through the
local qualifier. Default Cargo collection excludes these test targets. The final NVCC contract suite also passes 11 tests,
including compilation of the device-only test body. True subnormal-result and both
clipping-edge fixtures pass on the host; their GPU counterparts await execution.

Host AddressSanitizer/UndefinedBehaviorSanitizer passed with leak detection disabled
because this cloud sandbox denies LeakSanitizer process inspection. No leak-check
pass is claimed. Receipt controls (9), aggregate controls (21), affected-crate
all-target Clippy, formatting and repository/prose guards passed. The final review
records the replay of local contracts and identity/history checks after reconciliation.
Use `--features cuda-parity-tests` for direct Cargo contract commands; the local
qualifier adds it automatically. See [manual GPU commands](runbooks/manual-gpu-qualification.md)
for the retained former weekly regression/restore scripts.

There is no actual GPU prediction, GPU memory-checker, driver compatibility or
multi-architecture result from this workspace. The census measures named-operation
admission and fixture behavior, not exhaustive parameters, compositions or source
branch coverage. Do not describe these results as 100% GPU coverage.

## Export-format scope gaps

| Export family | CUDA support in this PR |
|---|---|
| Legacy Rust training recipes | Complete existing Rust admission through the native core |
| Advanced-v3 fitted/raw-string export | Same admission as bound Rust |
| Scalar-program-v2 and row-vector training recipes | Public CUDA export functions; 101 qualification cases |
| Separate `numeric_v2_emit::TargetV2` fitted format | Remains Python/Rust-only; no CUDA variant or adapter |
| Separate scalar/row-vector `emit_fitted_cpp17` artifacts | Remain C++17-specific; CUDA training recipes refit instead |
| Universal device execution | Three listed linear heads only; other model families and preprocessing execute on the host |

The unsupported fitted formats are capability gaps, rather than missing hardware
proof. Parity across every export format requires additional adapters and their own
state/semantics/closure tests. Preserve existing fitted-format identities when adding
them; do not substitute the legacy clamped predictor for the separate numeric-v2
semantics. The corresponding sources are `numeric_v2_emit.rs`,
`kernel/scalar_program/fitted.rs` and `kernel/row_vector.rs` under `crates/feat-math/src/`.

## Local prerequisites

Use Linux x86-64, the repository's pinned Rust toolchain, GCC/G++, Clang, Python 3
and pip. The full golden suite also compiles the existing C17 outputs, so Clang is
required even for CUDA qualification. CMake 3.24 or newer is required. Device mode
additionally requires an NVIDIA GPU, its compatible driver, nvcc and `nvidia-smi`.
The separate memory-check step needs NVIDIA Compute Sanitizer; the compiler wheels
do not install it.
Do not install or replace a driver without the local owner's authorization.

Optional hash-pinned compiler/build-tool setup from the repository root:

```bash
bash deploy/ci/cuda/compiler-tools.sh compile
source target/cuda-toolkit/env.sh
```

The installer provides official CUDA 13.0.88 compiler/runtime wheels and CMake; it
does not install a driver. Set `JOPE_CUDA_TOOLS_DIR` to choose a tool location, or
use the environment path printed by the installer when `CARGO_TARGET_DIR` is set.
`compiler-tools.sh host` installs only the build tools. An existing system toolkit
can instead be selected with `JOPE_NVCC=/path/to/nvcc`.

## Run the complete local matrix

Start from a clean committed checkout. The script refuses dirty tracked source so
receipts identify executable code precisely. Preserve local changes and commit
corrections before producing a new qualification receipt.

```bash
git rev-parse HEAD
git status --short
export JOPE_CUDA_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export JOPE_CUDA_EVIDENCE_ROOT="target/cuda-evidence/$JOPE_CUDA_RUN_ID"
bash deploy/ci/cuda/qualify.sh host
bash deploy/ci/cuda/qualify.sh compile
bash deploy/ci/cuda/qualify.sh device
```

The default build budget is two Cargo jobs and two test workers, no incremental
builds and line-table debug data. One target directory is reused. Host mode enables
AddressSanitizer/UndefinedBehaviorSanitizer with recovery disabled. Keep leak
detection enabled where supported. If an environment forces its removal, record
that limitation; runtime sanitizer settings are retained in `environment.txt`.

Each mode runs target contracts, complete registry/v2 censuses, all goldens, ordered
fitted tests, artifact binding, pipeline publication and CLI aliases. It discovers
each suite's selected tests and requires every selected test to pass. It then
writes `qualification.json` beside the logs and hashes every selection and log.
Compile mode compiles and links the device-only contract body but executes the host
entry point. Only device mode can record actual GPU batches.

Verify a retained result without rerunning the matrix:

```bash
python3 deploy/ci/cuda/evidence.py verify "$JOPE_CUDA_EVIDENCE_ROOT/host" host
python3 deploy/ci/cuda/evidence.py verify "$JOPE_CUDA_EVIDENCE_ROOT/compile" compile
python3 deploy/ci/cuda/evidence.py verify "$JOPE_CUDA_EVIDENCE_ROOT/device" device
```

## GPU-agent acceptance and remaining work

1. Run all three modes on the exact candidate SHA. Every discovered suite must
   complete, with zero failures or ignored tests and all four census denominators.
2. Confirm device predictions match host bits for every advertised linear-head
   fixture. `supports_cuda()` must match the manifest; errors must never fall back.
3. Run the device contract with default and nonblocking streams, empty and block-edge
   batches, 65,537 rows, concurrent predictions, NaN/infinity source-cell sanitization,
   separate x87/SSE rounding modes, denormal-flushing refusals, true subnormal
   outputs and both prediction-clipping edges.
4. Run NVIDIA Compute Sanitizer on generated device consumers, for example
   `compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./consumer`
   from a retained device consumer directory. Preserve its output separately;
   `qualify.sh device` does not itself run this checker. Include a nonblocking-stream
   consumer to verify transfer ordering and buffer lifetime on real hardware.
5. The default stress case stops at 65,537 rows. A larger, budgeted local batch
   exceeding 8,388,480 rows is still needed to execute the capped grid-stride branch.
   Allocation/launch failure injection and cross-device stream scenarios also remain
   unexecuted. Record those limits rather than asserting full source-branch coverage.
6. Record GPU name, driver, toolkit, architecture, exact SHA, run ID and all logs.
   A single device/toolchain run qualifies that environment only. Other architectures
   and toolchains remain unverified until separately exercised.
7. Fix every reproducible issue, retain its failing-before/passing-after proof and
   rerun affected checks at the new SHA. Update the PR with receipts and remaining
   limits. No deployment, serving promotion, merge or GPU CI change is requested.

The emitted API/build contract is documented in [cuda.md](cuda.md). The implementation
is under `crates/feat-math/src/cuda/`; local qualification and receipt tools are under
`deploy/ci/cuda/`; native and device regression consumers are under
`crates/feat-math/tests/support/cuda*`.
