#!/usr/bin/env bash
# Environment for building the `neural` feature (crates/jope-engine, package jope-parallel:
# tch 0.26.0 + native/runtime.cpp).
#
# The pinned expectations below mirror deploy/receipts/libtorch.json (deploy/ci/test-neural-env.sh
# asserts it); regenerate both together (docs/runbooks/neural-sdk.md). The SDK volume layout is:
#   $JOPE_NEURAL_SDK_ROOT/libtorch                    LibTorch 2.13.0+cu126 (build-version checked)
#   $JOPE_NEURAL_SDK_ROOT/nvidia/<pkg>/{lib,include}  cudnn, cusparselt, cufile, nccl, nvshmem from the
#                                                     pinned nvidia-*-cu12 wheels
#
# CUDA 12.8 is discovered from CUDA_HOME (default /usr/lib/cuda) in exactly one of two layouts:
#   toolkit       CUDA_HOME/include/cuda_runtime.h and CUDA_HOME/lib64/libcudart.so.12
#                 (runner image nvidia/cuda:12.8.1-devel-ubuntu24.04, CUDA_HOME=/usr/local/cuda)
#   distro-split  CUDA_HOME/include/cuda.h is a symlink into <prefix>/include, which holds
#                 cuda_runtime.h; libcudart.so.12 is in <prefix>/lib/x86_64-linux-gnu
#                 (Ubuntu nvidia-cuda-toolkit on the xbabe2 host: /usr/lib/cuda -> /usr/include)
#
# DISCOVERY ONLY. It qualifies files, versions and SONAMEs, and that $CXX resolves the discovered
# headers with build.rs's include order. It does not compile runtime.cpp, link, load CUDA or use a
# GPU; none of those is claimed. With JOPE_NEURAL_ENV_RECEIPT=<new file> it also writes (with jq) a
# jope.neural-discovery/v1 receipt, on failure as well as on success.
#
# Usage:
#   . deploy/ci/neural-env.sh                  # export the variables into the current shell
#   deploy/ci/neural-env.sh >> "$GITHUB_ENV"   # print KEY=VALUE lines for a workflow step

jope_neural_env_fail() {
  printf 'neural-env: %s\n' "$1" >&2
  return 1
}

# Record the failing check for the receipt, report it, and return 1.
jope_neural_env_require() { # <check> <detail>
  failed_check=$1
  failed_detail=$2
  jope_neural_env_fail "$2"
}

jope_neural_env_define() { # <header> <macro>: value of `#define <macro> <value>`
  awk -v macro="$2" '$1 == "#define" && $2 == macro { print $3; exit }' "$1"
}

jope_neural_env_library() { # <check> <path>: present, and its DT_SONAME equals its file name
  local want=${2##*/} got=""
  [[ -e $2 ]] || jope_neural_env_require "$1" "missing $2" || return 1
  got=$(readelf -d "$2" 2>/dev/null | awk -F'[][]' '/Library soname:/ { print $2; exit }') || got=""
  [[ $got == "$want" ]] \
    || jope_neural_env_require "$1" "$2 has SONAME '${got:-none}', expected '$want'" || return 1
  facts+=("libraries/$want=$(readlink -f "$2")")
}

jope_neural_env_discover() {
  local dir file header_link prefix out probe resolved_cuda resolved_nccl

  for dir in libtorch/lib libtorch/include nvidia/nccl/lib nvidia/nccl/include nvidia/cudnn/lib \
             nvidia/cusparselt/lib nvidia/cufile/lib nvidia/nvshmem/lib; do
    [[ -d $sdk/$dir ]] \
      || jope_neural_env_require sdk-layout "missing $sdk/$dir (provision per docs/runbooks/neural-sdk.md)" \
      || return 1
  done
  passed+=(sdk-layout)

  [[ -r $sdk/libtorch/build-version ]] \
    || jope_neural_env_require libtorch-build-version "missing $sdk/libtorch/build-version" || return 1
  libtorch_version=$(< "$sdk/libtorch/build-version")
  facts+=("libtorch/build_version=$libtorch_version")
  [[ $libtorch_version == "$expected_libtorch" ]] \
    || jope_neural_env_require libtorch-build-version \
         "LibTorch build-version is '$libtorch_version', receipt expects '$expected_libtorch'" || return 1
  passed+=(libtorch-build-version)

  # build.rs include/link inputs and the headers native/runtime.cpp includes.
  for file in lib/libtorch.so lib/libtorch_cpu.so lib/libtorch_cuda.so lib/libc10.so lib/libc10_cuda.so \
              include/c10/cuda/CUDAStream.h include/torch/csrc/api/include/torch/torch.h; do
    [[ -r $sdk/libtorch/$file ]] || jope_neural_env_require libtorch-files "missing $sdk/libtorch/$file" || return 1
  done
  passed+=(libtorch-files)

  command -v readelf > /dev/null \
    || jope_neural_env_require tools "readelf (binutils) is required to read SONAMEs" || return 1
  for file in nccl/lib/libnccl.so.2 cudnn/lib/libcudnn.so.9 cusparselt/lib/libcusparseLt.so.0 \
              cufile/lib/libcufile.so.0 nvshmem/lib/libnvshmem_host.so.3; do
    jope_neural_env_library companion-libraries "$sdk/nvidia/$file" || return 1
  done
  passed+=(companion-libraries)

  # The wheel's NCCL header, never the host's: libtorch_cuda imports symbols newer releases add.
  nccl_header=$sdk/nvidia/nccl/include/nccl.h
  [[ -r $nccl_header ]] || jope_neural_env_require nccl-header "missing $nccl_header" || return 1
  nccl_version="$(jope_neural_env_define "$nccl_header" NCCL_MAJOR).$(jope_neural_env_define "$nccl_header" NCCL_MINOR).$(jope_neural_env_define "$nccl_header" NCCL_PATCH)"
  facts+=("nccl/header=$nccl_header" "nccl/version=$nccl_version")
  [[ $nccl_version == "$expected_nccl" ]] \
    || jope_neural_env_require nccl-header "$nccl_header is NCCL $nccl_version, receipt expects $expected_nccl" \
    || return 1
  passed+=(nccl-header)

  [[ -d $cuda_home ]] || jope_neural_env_require cuda-home "CUDA_HOME=$cuda_home is not a directory" || return 1
  if [[ -r $cuda_home/include/cuda_runtime.h ]]; then
    layout=toolkit
    include_dir=$cuda_home/include
    lib_dir=$cuda_home/lib64
  elif [[ -L $cuda_home/include/cuda.h ]]; then
    header_link=$(readlink -f "$cuda_home/include/cuda.h")
    layout=distro-split
    include_dir=${header_link%/*}
    prefix=${include_dir%/*}
    lib_dir=$prefix/lib/x86_64-linux-gnu
  else
    jope_neural_env_require cuda-runtime-header \
      "no cuda_runtime.h in $cuda_home/include and no distro cuda.h symlink there (CUDA_HOME layout unknown)" \
      || return 1
  fi
  cuda_runtime_h=$include_dir/cuda_runtime.h
  facts+=("cuda/layout=$layout" "cuda/include_dir=$include_dir" "cuda/library_dir=$lib_dir")
  [[ -r $cuda_runtime_h && -r $include_dir/cuda_runtime_api.h ]] \
    || jope_neural_env_require cuda-runtime-header \
         "missing $cuda_runtime_h or cuda_runtime_api.h ($layout layout of CUDA_HOME=$cuda_home)" || return 1
  facts+=("cuda/cuda_runtime_h=$(readlink -f "$cuda_runtime_h")")
  passed+=(cuda-runtime-header)

  cudart_version=$(jope_neural_env_define "$include_dir/cuda_runtime_api.h" CUDART_VERSION)
  facts+=("cuda/cudart_version=$cudart_version")
  [[ $cudart_version == "$expected_cudart" ]] \
    || jope_neural_env_require cudart-version \
         "CUDART_VERSION is '$cudart_version' in $include_dir/cuda_runtime_api.h, receipt expects $expected_cudart" \
    || return 1
  passed+=(cudart-version)

  jope_neural_env_library cudart-library "$lib_dir/libcudart.so.12" || return 1
  passed+=(cudart-library)

  cxx_path=$(command -v "$cxx") || jope_neural_env_require compiler "CXX=$cxx is not an executable" || return 1
  out=$("$cxx" --version 2>&1) || jope_neural_env_require compiler "$cxx --version failed: ${out%%$'\n'*}" || return 1
  facts+=("compiler/command=$cxx" "compiler/path=$cxx_path" "compiler/version=${out%%$'\n'*}")
  passed+=(compiler)

  # Preprocess only, with build.rs's include order: the headers the compiler will actually pick.
  probe=$(printf '#include <cuda_runtime.h>\n#include <nccl.h>\n' \
            | "$cxx" -x c++ -std=c++20 -E -H -o /dev/null \
                -I"$sdk/libtorch/include" -I"$sdk/libtorch/include/torch/csrc/api/include" \
                -I"$cuda_home/include" -I"$sdk/nvidia/nccl/include" - 2>&1) \
    || jope_neural_env_require compiler-header-probe \
         "$cxx cannot preprocess <cuda_runtime.h>/<nccl.h> with build.rs include flags: ${probe%%$'\n'*}" \
    || return 1
  # `-H` prints one '. <path>' line per top-level include (paths may contain spaces).
  resolved_cuda=$(awk 'substr($0, 1, 2) == ". " && /\/cuda_runtime\.h$/ { print substr($0, 3); exit }' <<< "$probe")
  resolved_nccl=$(awk 'substr($0, 1, 2) == ". " && /\/nccl\.h$/ { print substr($0, 3); exit }' <<< "$probe")
  facts+=("compiler/resolved_cuda_runtime_h=$resolved_cuda" "compiler/resolved_nccl_h=$resolved_nccl")
  [[ -n $resolved_cuda && $(readlink -f "$resolved_cuda") == "$(readlink -f "$cuda_runtime_h")" ]] \
    || jope_neural_env_require compiler-header-probe \
         "$cxx resolves <cuda_runtime.h> to '${resolved_cuda:-nothing}', not $cuda_runtime_h" || return 1
  [[ -n $resolved_nccl && $(readlink -f "$resolved_nccl") == "$(readlink -f "$nccl_header")" ]] \
    || jope_neural_env_require compiler-header-probe \
         "$cxx resolves <nccl.h> to '${resolved_nccl:-nothing}', not $nccl_header" || return 1
  passed+=(compiler-header-probe)
}

jope_neural_env_receipt() { # <status>: write the discovery receipt when one was requested
  local path=${JOPE_NEURAL_ENV_RECEIPT:-} name json
  [[ -n $path ]] || return 0
  local -a exported=()
  if [[ $1 == discovered ]]; then
    for name in LIBTORCH LIBTORCH_CXX11_ABI CUDA_HOME JOPE_NCCL_INCLUDE JOPE_CUDA_LIBRARY_PATH \
                JOPE_NCCL_SONAME JOPE_CUDART_SONAME CXX; do
      exported+=("$name=${!name}")
    done
  fi
  # The jope.neural-discovery/v1 document. Positional arguments are $nf facts ("a/b/c=value",
  # nested into `observed` in insertion order), $np passed checks, then the exported "NAME=value"
  # lines; each item splits at its first '='. Empty failed_check, diagnostic and environment are null.
  # jq turns bytes that are not UTF-8 into U+FFFD; such input fails the write rather than being
  # recorded wrongly.
  # shellcheck disable=SC2016 # jq variables, not shell expansions
  local program='
def pair: split("=") | [.[0], (.[1:] | join("="))];
$ARGS.positional as $items
| if any($items[], $status, $failed, $detail, $sdk, $cuda_home, $source, $libtorch, $cudart, $nccl;
         contains("�"))
  then error("receipt input is not valid UTF-8") else . end
| {
    schema: "jope.neural-discovery/v1",
    status: $status,
    failed_check: (if $failed == "" then null else $failed end),
    diagnostic: (if $detail == "" then null else $detail end),
    scope: ("discovery only: SDK files, LibTorch build-version, header versions, library SONAMEs and "
            + "compiler header resolution; nothing was compiled, linked, loaded or run on a GPU"),
    qualification: {
      discovery: (if $status == "discovered" then "passed" else "failed" end),
      native_build: "not-attempted", native_link: "not-attempted",
      native_runtime: "not-attempted", gpu: "not-attempted"
    },
    expected: {libtorch_build_version: $libtorch, cudart_version: $cudart, nccl_version: $nccl},
    sdk_root: $sdk,
    cuda_home: {path: $cuda_home, source: $source},
    checks_passed: [$items[$nf:$nf + $np][] | select(. != "")],
    observed: (reduce ($items[:$nf][] | select(. != "") | pair) as [$key, $value]
                 ({}; setpath($key | split("/"); $value))),
    environment: (reduce ($items[$nf + $np:][] | select(. != "") | pair) as [$key, $value]
                    ({}; .[$key] = $value)
                  | if . == {} then null else . end)
  }'
  # Rendered first, then created exclusively: an existing path of any type is never written
  # (noclobber alone would still write into an existing FIFO or device). --ascii-output escapes
  # non-ASCII as \uXXXX, as the receipt always has.
  if ! json=$(jq -n --ascii-output --indent 2 --args \
                --arg status "$1" --arg failed "$failed_check" --arg detail "$failed_detail" \
                --arg sdk "$sdk" --arg cuda_home "$cuda_home" --arg source "$cuda_home_source" \
                --arg libtorch "$expected_libtorch" --arg cudart "$expected_cudart" \
                --arg nccl "$expected_nccl" --argjson nf "${#facts[@]}" --argjson np "${#passed[@]}" \
                "$program" -- "${facts[@]}" "${passed[@]}" "${exported[@]}") \
     || [[ -e $path || -L $path ]] \
     || ! (set -C && printf '%s\n' "$json" > "$path"); then
    jope_neural_env_fail "could not write the discovery receipt $path"
  fi
}

jope_neural_env_main() {
  local sdk="${JOPE_NEURAL_SDK_ROOT:-/opt/jope-sdk}"
  local expected_libtorch="2.13.0+cu126" expected_cudart=12080 expected_nccl=2.29.3
  local cuda_home="${CUDA_HOME:-/usr/lib/cuda}" cuda_home_source=environment
  [[ -n ${CUDA_HOME:-} ]] || cuda_home_source=default
  local cxx="${CXX:-clang++}" cxx_path=""
  local failed_check="" failed_detail="" layout="" include_dir="" lib_dir="" cuda_runtime_h=""
  local libtorch_version="" nccl_header="" nccl_version="" cudart_version=""
  local -a facts=() passed=()

  if ! jope_neural_env_discover; then
    jope_neural_env_receipt unavailable || true
    return 1
  fi

  # Read by crates/jope-engine/build.rs and torch-sys 0.26.0 build.rs.
  export LIBTORCH="$sdk/libtorch"
  export LIBTORCH_CXX11_ABI=1
  export CUDA_HOME="$cuda_home"
  export JOPE_NCCL_INCLUDE="$sdk/nvidia/nccl/include"
  # Order matters: wheel directories precede the CUDA runtime directory so DT_RPATH resolves
  # libnccl.so.2 to the wheel's 2.29.3 (host/image NCCL 2.26.2/2.25.1 lack symbols libtorch_cuda
  # imports). The last entry is where libcudart.so.12 was discovered, so -l:libcudart.so.12 links in
  # both layouts (the image keeps it in /usr/local/cuda/lib64, off the linker's default path).
  export JOPE_CUDA_LIBRARY_PATH="$sdk/nvidia/nccl/lib:$sdk/nvidia/cudnn/lib:$sdk/nvidia/cusparselt/lib:$sdk/nvidia/cufile/lib:$sdk/nvidia/nvshmem/lib:$lib_dir"
  # Verbatim SONAMEs: the wheels ship no unversioned libnccl.so dev symlink, so -lnccl would
  # otherwise resolve to the host library.
  export JOPE_NCCL_SONAME=libnccl.so.2
  export JOPE_CUDART_SONAME=libcudart.so.12
  export CXX="$cxx"
  jope_neural_env_receipt discovered
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  jope_neural_env_main
  for name in LIBTORCH LIBTORCH_CXX11_ABI CUDA_HOME JOPE_NCCL_INCLUDE JOPE_CUDA_LIBRARY_PATH \
              JOPE_NCCL_SONAME JOPE_CUDART_SONAME CXX; do
    printf '%s=%s\n' "$name" "${!name}"
  done
else
  jope_neural_env_main
fi
