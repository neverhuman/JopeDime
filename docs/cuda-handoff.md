# CUDA C++ local qualification handoff

PR: https://github.com/neverhuman/JopeDime/pull/422

Branch: `feat/cuda-cpp-target-20261023`

Date: 2026-10-23

## Scope and user direction

Finish local hardware verification of the CUDA C++ output target. All CUDA-specific
hosted workflows, runner requests and hardware merge gates have been removed. Keep
hardware work local; do not add GPU CI or change runner policy. The ordinary Rust
checks exclude the CUDA contract/census targets and CUDA golden hook. The local
qualifier explicitly selects the nondefault `cuda-parity-tests` feature.

The generated library provides every recipe accepted by the Rust emitter and every
fitted predictor accepted by bound Rust through a shared native Rust implementation.
Rust is required to build the bundle. Fitting and feature processing execute on the
CPU. Explicit GPU inference supports ridge, elastic-net and ridge-GCV heads with no
target mapping. Bound/string models, nonlinear heads, symbolic heads and target
mappings deliberately refuse device inference. Universal GPU model execution is an
open capability gap, separate from verification of the implemented linear heads.

## Verification already available

- The initial complete golden matrix passed 89 tests, with actual CUDA-compiled C++
  host callers and bitwise prediction comparisons.
- The complete published-head registry census passed: 53 accepted recipes / 8 Rust
  refusals; 24 advanced-bound operations / 1 refusal; 14 legacy-bound operations /
  39 Rust refusals; 95 scalar/row-vector cases at that checkpoint.
- Compatibility, historical fixtures, CLI aliases, pipeline emission and artifact
  binding passed at published source `4d7bf10bebf2920ae6fc57298a3027f0d43f6e45`.
- Red-team regressions reproduced CMake target-name collisions, acceptance of partial
  or duplicated evidence, and the missing SSE rounding guard. Corrected source and
  regression tests are included. The expanded v2 census passed all 101 cases,
  including composite scalar ASTs and row-stage compositions.
- The strengthened 11-test target contract suite passed with real nvcc. Host C++
  AddressSanitizer and UndefinedBehaviorSanitizer also passed 11 tests with leak
  detection disabled because this cloud sandbox denies LeakSanitizer process
  inspection. That result establishes no leak-check pass.
- Final source-bound run results and any later corrections are recorded in the PR
  review. Read its exact full SHA before reusing results from an earlier checkpoint.

This workspace has no NVIDIA device. There is no actual GPU prediction, GPU memory
checker, driver compatibility or multi-architecture qualification result. The census
measures named-operation admission and fixture behavior, not exhaustive parameter,
composition or source-branch coverage. Do not describe these results as 100% GPU
coverage.

## Local prerequisites

Use Linux x86-64, the repository's pinned Rust toolchain, GCC/G++, Clang, Python 3
and pip. The full golden suite also compiles the existing C17 outputs, so Clang is
required even for CUDA qualification. CMake 3.24 or newer is required. Device mode
additionally requires an NVIDIA GPU, its compatible driver, nvcc and `nvidia-smi`.
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
   separate x87/SSE rounding modes and denormal-flushing refusals.
4. Run NVIDIA Compute Sanitizer on generated device consumers, for example
   `compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./consumer`
   from a retained device consumer directory. Preserve its output separately;
   `qualify.sh device` does not itself run this checker. Include a nonblocking-stream
   consumer to verify transfer ordering and buffer lifetime on real hardware.
5. Record GPU name, driver, toolkit, architecture, exact SHA, run ID and all logs.
   A single device/toolchain run qualifies that environment only. Other architectures
   and toolchains remain unverified until separately exercised.
6. Fix every reproducible issue, retain its failing-before/passing-after proof and
   rerun affected checks at the new SHA. Update the PR with receipts and remaining
   limits. No deployment, serving promotion, merge or GPU CI change is requested.

The emitted API/build contract is documented in [cuda.md](cuda.md). The implementation
is under `crates/feat-math/src/cuda/`; local qualification and receipt tools are under
`deploy/ci/cuda/`; native and device regression consumers are under
`crates/feat-math/tests/support/cuda*`.
