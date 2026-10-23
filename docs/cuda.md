# CUDA C++ output

`jope emit --lang cuda` emits a CUDA C++17 library, a C++ model API and the complete
generated native Rust numerical implementation. Rust is required at build time. A
deployed application needs neither Rust nor Python installed.

The target has **complete functional coverage of the Rust training-recipe emitter**.
This is a hybrid implementation: fitting and feature processing run on the host.
`predict()` uses the shared native implementation. `predict_cuda()` explicitly runs
an admitted linear head on the NVIDIA device. It never silently falls back.

| Operation | Coverage and execution |
|---|---|
| `fit`, `predict`, `score` | Every recipe accepted by the Rust emitter, including target transforms, symbolic programs, nonlinear heads and train-fitted feature stages; host execution |
| `predict_cuda` | `ridge`, `elasticnet`, `ridge_gcv` with no target transform; host feature stages and standardization, device sequential FP64 head |
| Fitted raw-string prediction | Every model accepted by `advanced_v3::emit_bound`; host execution, embedded fitted state, exact schema and categorical handling |
| Scalar-v2 and row-vector recipes | Public `cuda::emit_scalar` and `cuda::emit_row_vector`; same Rust admission, host feature processing and explicit device ridge head |
| Unsupported device models | Error status 5; the caller can deliberately choose `predict()` |
| Device unavailable or CUDA runtime failure | Error status 6; no fallback |

The existing Rust emitter's refusals remain refusals. In particular, this target does
not implement registry operations that Rust itself cannot emit, create a new fitted
model format, or make all model families into GPU training implementations.

The separate numeric-v2 fitted format and the scalar/row-vector fitted C++17
artifact APIs have no CUDA adapter in this PR. See the [export-format scope gaps](cuda-handoff.md#export-format-scope-gaps)
before treating the recipe census as coverage of every export format.

## Emit and build

```bash
jope emit --genome '(pipe X (standardize) (ridge))' --lang cuda --output generated
# Aliases: cuda-cpp and cu. Use --lang v3:cuda for a pinned artifact envelope.
cmake -S generated/cuda -B generated/cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build generated/cuda/build --parallel 2
```

The bundle has six files: `model.hpp`, `model.cu`, `model_core.rs`,
`model_native.rs`, `CMakeLists.txt` and `manifest.json`, all inside `cuda/`.
The source manifest binds every compilation input. The v3 envelope also binds the
source manifest and the complete build contract; it records execution as unknown.

For pipeline publication, request `jope pipeline run … --emit cuda`. The artifact
receipt records every file, build command and source digest. `all` keeps its
historical target set; request CUDA explicitly.

The reference qualification environment is Linux x86-64, Rust 1.97.1, CUDA 13.0.88,
GCC 13 and CMake 3.24+. Select an NVIDIA architecture supported by the installed
toolkit with `-DCMAKE_CUDA_ARCHITECTURES=86`, for example, for an RTX 3090. CUDA
13 does not compile for every historical NVIDIA architecture. A CUDA build needs
the CUDA runtime on the deployment system and a compatible driver for device calls.

The complete host API can also be built without a CUDA toolkit or runtime:

```bash
cmake -S generated/cuda -B generated/cuda/build-host -DJOPE_CUDA_HOST_ONLY=ON
cmake --build generated/cuda/build-host --parallel 2
```

This is explicitly a host build; `cuda_available()` returns false and
`predict_cuda()` throws status 6 for an otherwise supported head.

## C++ API

```cpp
#include "model.hpp"

int main() {
    jope_cuda::Model model;
    model.fit({{0, 1}, {1, 2}, {2, 0}, {3, 2}}, {1, 2, 4, 7});
    const jope_cuda::Rows rows{{0.5, 1}, {2.5, 0}};
    const auto host = model.predict(rows);
    const double r2 = model.score({{0, 1}, {1, 2}}, {1, 2});
    (void)host;
    (void)r2;
    if (model.supports_cuda() && jope_cuda::Model::cuda_available()) {
        const auto device = model.predict_cuda(rows);
        (void)device;
    }
}
```

In a CMake consumer, add the emitted directory with `add_subdirectory(generated/cuda)`
and link the `model` target. Its transitive dependencies include the native Rust
archive, platform libraries and, for a CUDA build, `CUDA::cudart`.

Each model owns a non-reused native handle. Models are move-only; `close()` is
idempotent, and destruction releases ownership. Failed refits preserve the old model.
Input rows and returned predictions own their memory. Empty prediction batches are
valid after fitting. Inconsistent widths, empty training data and invalid handles
are refused. Prediction may run concurrently on an unchanged instance. Refitting,
moving or destroying that instance requires exclusive caller access.

Each bundle has source-derived C symbols and a unique C++ namespace recorded as
`cpp_namespace` in its manifest. To link different emitted models into one executable,
define `JOPE_CUDA_NO_DEFAULT_NAMESPACE` before including their headers and refer to
each recorded namespace. The default `jope_cuda` alias is convenient for one bundle.
The manifest's `cmake_target` names that bundle's library target. Add each bundle
with `add_subdirectory` and link its recorded target. Identical bundles reuse their
target; the first bundle also supplies the `model` convenience alias.

For an embedded fitted predictor, call `auto model = jope_cuda::Model::load()` and
`model.predict_strings(rows)`. `predict_csv(csv, expected_rows)` verifies the exact
header and row count. Fitting, numeric prediction and device prediction are refused
for that artifact. Fitted ordering/tail guards, unseen categories, missing values
and UTF-8 follow the bound Rust predictor.

## Arithmetic and boundaries

The original numerical Rust source is retained as a safe module. The native boundary
copies input before output commits, checks dimensions, lengths, alignment and integer
overflow, retains shared ownership throughout predictions, and contains unwinding
panics. As with any native ABI, callers of private C entry points must provide live
allocations of the declared lengths. Allocation exhaustion and process-aborting
builds cannot be recovered.

The GPU head uses one thread per row and sequential `__dmul_rn` / `__dadd_rn` operations,
with no FP32 conversion, reassociation, atomics or device transcendental functions.
The build fixes `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` and host
`-ffp-contract=off -fno-fast-math`. It requires host round-to-nearest mode and disabled
host denormal flushing. Nonfinite head inputs are refused. The original host feature
arithmetic and final prediction clamp remain in their original order.

[NVIDIA's floating-point documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html)
explains the separately rounded intrinsics. The compile flags and source review
establish the intended contract; actual device execution supplies its qualification.

`predict_cuda(rows, stream)` accepts a borrowed CUDA stream on the current device.
The caller owns its validity and lifetime. The method synchronizes that stream and
returns copied host results; it does not promise asynchronous transfers or persistent
device-resident model state. Allocation, launch, transfer and stream errors throw.
No speedup is claimed: small batches and expensive host preprocessing may make this
hybrid path slower. Fitting is always on the host.

## Local qualification and remaining hardware proof

The local scripts provide separate host, real-`nvcc` compilation and device modes.
Host and compile modes both run:

- The complete recipe registry, advanced bound registry and legacy-bound state census,
  matching every Rust acceptance or refusal and comparing prediction bits.
- Every scalar-v2 function in both policies and both carriers, plus row-vector stages.
- The existing golden recipe matrix, lifecycle/boundary contracts, model isolation,
  malformed-input checks and artifact/CLI/pipeline publication tests.
- C++ host callers under AddressSanitizer and UndefinedBehaviorSanitizer in host mode.
- Discovered test selections bound to their execution counts, and exact-head/run/attempt
  receipts that refuse partial, missing, ignored, zero-test, duplicated or altered evidence.

Compile mode builds actual CUDA device code, links it and executes the host API
without requiring a GPU. It **does not establish a GPU prediction pass**.
It compiles and links the device-only contract body as well as the generated kernel.

```bash
bash deploy/ci/cuda/qualify.sh host
JOPE_NVCC=/path/to/nvcc bash deploy/ci/cuda/qualify.sh compile
# On an NVIDIA device host, with nvcc and nvidia-smi available:
bash deploy/ci/cuda/qualify.sh device
```

CUDA qualification is local. This target adds no hosted CUDA job, GPU runner request,
artifact aggregation gate or hardware merge requirement. CUDA-specific contract/census targets and the CUDA golden hook require the nondefault
`cuda-parity-tests` feature. Standard Rust lanes do not select it; `qualify.sh` selects it
explicitly. The CUDA evidence validator also runs only through local qualification. Actual GPU qualification requires a fresh device-mode
receipt; missing hardware stays unqualified. Universal GPU execution for nonlinear/bound
models is outside this implementation's device coverage.

The [GPU-agent handoff](cuda-handoff.md) gives setup commands, verification status,
remaining work and receipt collection instructions. The tool installer writes a local
`env.sh`; it requires no GitHub Actions environment variables.

## Red-team coverage and blockers

The registry census covers every named operation in the Rust emitter's admission
denominator and compares predictions on its fixtures. It does not measure source
line/branch coverage or exhaust all compositions, parameter values, batch shapes,
architectures and toolchains. The device fixtures include default and nonblocking
streams, empty and block-edge batches, a larger batch, concurrent predictions,
nonfinite source cells and floating-environment refusals. Their execution remains
unqualified until a real device run passes.

The review exposed and repaired CMake target collisions between independent bundles,
partial-run and duplicated-census evidence acceptance, missing assertions for advertised
device admission, split x87/SSE rounding modes, and stream transfer ordering and
host-buffer lifetime. Reproducing failures and corrected regressions
were retained during implementation. Device transfers, kernel work and result copies
now share the borrowed stream, with synchronization during error unwinding.

The initial proposed hardware workflow reused CPU-only runner guards. That mismatch
was found during review. All CUDA-specific hosted workflows and merge gates were
removed under the user's local-CI direction; runner admission policies remain unchanged.
Device execution and hardware memory checking remain local follow-up work.

The CLI CUDA alias fixture and pipeline CUDA runtime fixture also require an explicit
`JOPE_CUDA_PARITY=host|compile|device`, set by `qualify.sh`. Default CI does not compile
or execute emitted CUDA libraries. The former scheduled GPU regression/restore
workflow is removed; [manual commands remain available](runbooks/manual-gpu-qualification.md).
