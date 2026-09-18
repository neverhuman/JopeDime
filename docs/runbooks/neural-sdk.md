# Neural SDK runbook — LibTorch for the `neural` feature on xbabe2

The `neural` feature of `crates/jope-engine` (package `jope-parallel`) links `tch = "=0.26.0"`
and compiles `crates/jope-engine/native/runtime.cpp`, which includes `<c10/cuda/CUDAStream.h>`,
`<cuda_runtime.h>` and `<nccl.h>` unconditionally. It only builds on the self-hosted `neural-cuda`
lane. This runbook says how to put the SDK on that runner's volume and how the receipt
`deploy/receipts/libtorch.json` is regenerated. `deploy/ci/neural-env.sh` exports the exact
environment the receipt was produced with; the lane must source it before any cargo command that
enables `neural`.

## What the receipt records

| Field | Value (receipt `jope.sdk-receipt/v1`) |
|---|---|
| LibTorch | `2.13.0+cu126`, `libtorch-shared-with-deps-2.13.0%2Bcu126.zip` from `download.pytorch.org`, `LIBTORCH_CXX11_ABI=1` |
| Why cu126 | `torch-sys 0.26.0` pins `TORCH_VERSION = "2.13.0"`; the host toolkit is CUDA 12.8 (`nvcc` 12.8.93, `libcudart.so.12.8.90`, `CUDART_VERSION 12080`). `download.pytorch.org` serves 2.13.0 only as `cu126` and `cu130` (**`cu128` is a 404**); `cu130` would mix a CUDA 13 LibTorch with the host's 12.8 headers/runtime and the CUDA-12 NCCL, so `cu126` is the flavour that matches the host. The driver (595.99.02, CUDA 13.2) accepts both. |
| CUDA | Host Ubuntu packaging: `CUDA_HOME=/usr/lib/cuda` (only `bin/`, `nvvm/`, a `cuda.h` symlink and `lib64/stubs`); `cuda_runtime.h` and `nccl.h` are in `/usr/include`; the runtime libraries are in `/usr/lib/x86_64-linux-gnu`. |
| Companion libraries | LibTorch 2.13 **no longer bundles** cuDNN, cuSPARSELt, cuFile, NCCL or NVSHMEM (`libtorch/lib` has 13 files; `libtorch_cuda.so` has `DT_NEEDED` on `libcudnn.so.9`, `libcusparseLt.so.0`, `libcufile.so.0`, `libnccl.so.2`, `libtorch_nvshmem.so` → `libnvshmem_host.so.3`). They come from the `nvidia-*-cu12` wheels pinned by the `torch==2.13.0+cu126` wheel metadata, unzipped (no pip, no Python). |
| NCCL | The wheel's **2.29.3** is linked and its header compiled against, not the host's `/usr/include/nccl.h` (2.26.2): `libtorch_cuda.so` imports 10 symbols the host library lacks (`ncclAlltoAll`, `ncclCommShrink`, `ncclCommWindow{Register,Deregister}`, `ncclDevCommDestroy`, `ncclGetLsaMultimemDevicePointer`, `ncclGetPeerDevicePointer`, `ncclPutSignal`, `ncclSignal`, `ncclWaitSignal`), so the plan's original `JOPE_NCCL_INCLUDE=/usr/include` cannot link. |

## Provisioning the runner volume

Default root is `/opt/jope-sdk` (override with `JOPE_NEURAL_SDK_ROOT`). Nothing here needs root
except creating the root directory; the volume is read-only for the runner user afterwards. Every
archive is verified against the sha256 in the receipt **before** it is unpacked.

```bash
SDK=/opt/jope-sdk
R=deploy/receipts/libtorch.json
mkdir -p "$SDK/dl" && cd "$SDK/dl"

# 1. LibTorch
curl -fsSL --retry 3 -o libtorch.zip "$(jq -r .libtorch.url "$R")"
echo "$(jq -r .libtorch.sha256 "$R")  libtorch.zip" | sha256sum -c -
unzip -q libtorch.zip -d "$SDK"            # creates $SDK/libtorch, build-version = 2.13.0+cu126

# 2. Companion libraries (wheels are zip files; only lib/ and include/ are extracted)
jq -r '.companions[] | "\(.url) \(.sha256) \(.filename)"' "$R" | while read -r url sha file; do
  curl -fsSL --retry 5 --retry-all-errors -C - -o "$file" "$url"
  echo "$sha  $file" | sha256sum -c -
  unzip -q -o "$file" 'nvidia/*/lib/*' 'nvidia/*/include/*' -d "$SDK"
done
rm -rf "$SDK/dl"

# 3. Check
JOPE_NEURAL_SDK_ROOT=$SDK deploy/ci/neural-env.sh
```

`pypi.nvidia.com` served the wheels reliably; `files.pythonhosted.org` reset large transfers on
xbabe2, hence `-C -` and `--retry-all-errors`. Disk: the zip + wheels are 2.1 GB, the unpacked tree
3.0 GB (5.6 GB while both are present).

Host prerequisites (already on xbabe2, from Ubuntu packages — never installed by CI): NVIDIA driver
≥ 560 (the receipt was made on 595.99.02), `nvidia-cuda-toolkit` 12.8 (`nvcc`, `/usr/include/cuda_runtime.h`,
`libcudart/cublas/cublasLt/cufft/curand/cusolver/cusparse/nvrtc/cupti .so` in `/usr/lib/x86_64-linux-gnu`),
`clang++` (build.rs default `CXX`; g++ 13 also works with `CXX=g++`).

## Using it

```bash
. deploy/ci/neural-env.sh
cargo check -p jope-parallel --features neural --locked
cargo build -p jope-parallel --features neural --locked
cargo test  -p jope-parallel --features neural --locked
```

`crates/jope-engine/build.rs` turns `JOPE_CUDA_LIBRARY_PATH` into `-L` and `-Wl,-rpath` entries and
links with `--disable-new-dtags`, so the produced binaries carry `DT_RPATH` for `libtorch/lib` and
every companion directory and run **without** `LD_LIBRARY_PATH` (SmartCluster clears the worker
environment). `JOPE_NCCL_SONAME=libnccl.so.2` / `JOPE_CUDART_SONAME=libcudart.so.12` make the
linker take the versioned files verbatim, because the wheels ship no unversioned `libnccl.so`.

## Regenerating the receipt

Do this whenever `tch`/`torch-sys` is bumped, the host CUDA toolkit or driver changes, or the
SDK volume is rebuilt. The receipt is evidence, not configuration: it is written from real command
exit codes on xbabe2, never by hand.

1. Read `TORCH_VERSION` in `~/.cargo/registry/src/*/torch-sys-<ver>/build.rs` and the flavours that
   exist for it under `https://download.pytorch.org/libtorch/<cuNNN>/`; pick the one matching the
   host toolkit major (`nvcc --version`, `CUDART_VERSION` in `/usr/include/cuda_runtime_api.h`).
2. Read the companion pins from the matching torch wheel's PEP 658 metadata
   (`https://download.pytorch.org/whl/<cuNNN>/torch-<ver>%2B<cuNNN>-cp312-cp312-manylinux_2_28_x86_64.whl.metadata`,
   `Requires-Dist: nvidia-*`) and the wheel URLs/sha256 from `https://pypi.org/pypi/<pkg>/<ver>/json`
   or the `https://download.pytorch.org/whl/<cuNNN>/<pkg>/` index (`#sha256=`). cuFile is not pinned
   by torch; take the newest `nvidia-cufile-cu12` 1.11.x for CUDA 12.
3. Provision a fresh volume as above, then with `. deploy/ci/neural-env.sh` sourced run, recording
   each exit code:
   `cargo check -p jope-parallel --features neural --locked`,
   `cargo build -p jope-parallel --features neural --locked`,
   `cargo test -p jope-parallel --features neural --locked -- --list | grep -c ': test$'`
   (the last one only lists — it does not touch the GPU; make sure `nvidia-smi` shows the GPU idle
   before running any actual `neural` test).
4. Fill `deploy/receipts/libtorch.json` (`schema: jope.sdk-receipt/v1`): `libtorch{version,url,sha256,bytes,cxx11_abi,cuda_flavour}`,
   `companions[]`, `cuda{home,nvcc_version,driver_version,gpu,compute_capability}`,
   `nccl{header,version}`, `host{hostname,kernel,gcc,clang,rustc}`, `commands[]` (command, env,
   exit code, log sha256), `result` (`check-passed` | `build-passed` | `failed at <step>: <first error>`),
   `produced_at` (`date -u +%FT%TZ`).
5. Update the values in `deploy/ci/neural-env.sh` and this runbook in the same commit; the
   `neural-cuda` job compares `libtorch/build-version` with the script's expectation and fails
   closed on a mismatch.

## Verification status

Receipt `deploy/receipts/libtorch.json` produced on xbabe2 at 2026-09-18T21:01:58Z from base commit `4235db672fa3`,
result **`build-passed`**: `cargo check` (exit 0, 3 pre-existing warnings), `cargo build` (exit 0;
`target/debug/jope-parallel` resolves every LibTorch, cuDNN, cuSPARSELt, cuFile, NVSHMEM, NCCL and
CUDA library through `DT_RPATH` with no `LD_LIBRARY_PATH`) and `cargo test -- --list` (exit 0;
549 tests in 40 binaries enumerate with `neural` on, of which 27 in 8 integration files are
`#![cfg(feature = "neural")]`). No test was executed and the GPU stayed idle; no source change was
needed — `build.rs` already accepts every override used. The `neural-cuda` lane may therefore
be wired as *verified for build*; running the 27 gated tests on the RTX 3090 is the lane's own
first green run, wrapped by `deploy/ci/gpu-qualification.sh`.
