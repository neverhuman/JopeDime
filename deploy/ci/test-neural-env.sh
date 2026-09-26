#!/usr/bin/env bash
# shellcheck disable=SC2016 # jq filters and the sourced-mode probe keep their $names for jq/bash
# Offline tests of deploy/ci/neural-env.sh against fixture SDK and CUDA trees.
#
# Discovery only: nothing here compiles runtime.cpp, links, loads CUDA or needs a GPU. The fixture
# libraries are text files whose SONAME a fake `readelf` (first on PATH) reports. The toolkit-layout
# cases preprocess the fixture headers with the real C++ compiler the runner image pins (clang++ 18;
# a case that needs it fails when no C++ compiler is on PATH); distro-layout and shadowing cases use
# a fake compiler that emulates `-E -H` with an explicit system include directory. The pinned
# versions come from deploy/receipts/libtorch.json (read with jq).
#
# Usage: bash deploy/ci/test-neural-env.sh
# Prints one PASS/FAIL line per case and a summary; exits nonzero when any case fails.
set -uo pipefail

ROOT=$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")
SCRIPT=$ROOT/deploy/ci/neural-env.sh
RECEIPT=$ROOT/deploy/receipts/libtorch.json
SCRATCH=$ROOT/target/tmp

die() {
  printf 'test-neural-env: %s\n' "$1" >&2
  exit 1
}

command -v jq > /dev/null || die 'jq is required'
EXPECTED_LIBTORCH=$(jq -er '.libtorch.version' "$RECEIPT") || die "no libtorch.version in $RECEIPT"
EXPECTED_CUDART=$(jq -er '.cuda.cudart_version_macro' "$RECEIPT") || die "no cuda.cudart_version_macro in $RECEIPT"
EXPECTED_NCCL=$(jq -er '.nccl.version' "$RECEIPT") || die "no nccl.version in $RECEIPT"
NAMES=(LIBTORCH LIBTORCH_CXX11_ABI CUDA_HOME JOPE_NCCL_INCLUDE JOPE_CUDA_LIBRARY_PATH
       JOPE_NCCL_SONAME JOPE_CUDART_SONAME CXX)
NAMES_JSON=$(jq -cn '$ARGS.positional' --args "${NAMES[@]}") || die 'jq cannot build the name list'
COMPANIONS=(nccl/lib/libnccl.so.2 cudnn/lib/libcudnn.so.9 cusparselt/lib/libcusparseLt.so.0
            cufile/lib/libcufile.so.0 nvshmem/lib/libnvshmem_host.so.3)
# full-lane.sh's pattern: sourced under errexit, a failure maps to exit 78.
SOURCE_PROBE='set -euo pipefail; source "$1" || exit 78; for n in '"${NAMES[*]}"'; do printf "%s=%s\n" "$n" "${!n}"; done'

# ---- fakes ----------------------------------------------------------------------------------------

write_fake_readelf() { # <path>: prints the SONAME a fixture library declares on its SONAME= lines
  cat > "$1" <<'EOF'
#!/usr/bin/env bash
while IFS= read -r line || [[ -n $line ]]; do
  if [[ $line == SONAME=* ]]; then
    printf ' 0x000000000000000e (SONAME)             Library soname: [%s]\n' "${line#SONAME=}"
  fi
done < "${@: -1}"
EOF
  chmod 755 "$1"
}

# Emulates `CXX -x c++ -std=c++20 -E -H -o /dev/null -I... -`: -I directories first, then
# FAKE_CXX_SYSTEM_INCLUDE, printing '. <path>' to stderr for each top-level include it resolves.
# FAKE_CXX_FORCE_<STEM> (e.g. FAKE_CXX_FORCE_NCCL for <nccl.h>) forces the reported path.
write_fake_cxx() { # <path>
  cat > "$1" <<'EOF'
#!/usr/bin/env bash
if [[ $# == 1 && $1 == --version ]]; then
  echo 'fake clang version 18.1.3'
  exit 0
fi
dirs=()
for arg in "$@"; do
  if [[ $arg == -I* ]]; then dirs+=("${arg#-I}"); fi
done
dirs+=("${FAKE_CXX_SYSTEM_INCLUDE-/nonexistent}")
while IFS= read -r line || [[ -n $line ]]; do
  if [[ $line != *'<'* ]]; then
    printf 'fake-cxx: no <header> in %s\n' "$line" >&2
    exit 1
  fi
  name=${line#*<}
  name=${name%%<*}
  name=${name%%>*}
  stem=${name%%.*}
  forced=FAKE_CXX_FORCE_${stem^^}
  found=${!forced:-}
  if [[ -z $found ]]; then
    for dir in "${dirs[@]}"; do
      if [[ -f $dir/$name ]]; then
        found=$dir/$name
        break
      fi
    done
  fi
  if [[ -z $found ]]; then
    printf '<stdin>:1:10: fatal error: %s: No such file or directory\n' "$name" >&2
    exit 1
  fi
  printf '. %s\n' "$found" >&2
done
EOF
  chmod 755 "$1"
}

# ---- fixture --------------------------------------------------------------------------------------

write() { # <path> [text]: create the parent directories and write text verbatim
  mkdir -p -- "$(dirname -- "$1")" && printf '%s' "${2-}" > "$1"
}

library() { # <path>: a fixture library whose SONAME is its file name
  write "$1" "SONAME=${1##*/}"$'\n'
}

nccl_header() { # <major.minor.patch>
  [[ $1 =~ ^([^.]*)\.([^.]*)\.([^.]*)$ ]] || fail "NCCL version '$1' is not major.minor.patch" || return 1
  write "$fx_sdk/nvidia/nccl/include/nccl.h" "#define NCCL_MAJOR ${BASH_REMATCH[1]}
#define NCCL_MINOR ${BASH_REMATCH[2]}
#define NCCL_PATCH ${BASH_REMATCH[3]}
"
}

runtime_headers() { # <include dir> [CUDART_VERSION]
  write "$1/cuda_runtime.h" $'#define JOPE_FIXTURE_CUDA_RUNTIME 1\n' || return 1
  write "$1/cuda_runtime_api.h" "#define CUDART_VERSION  ${2:-$EXPECTED_CUDART}"$'\n'
}

toolkit() { # [CUDART_VERSION]: toolkit layout at <root>/cuda-12.8; sets home
  home=$fx_root/cuda-12.8
  if ! { runtime_headers "$home/include" "${1:-}" \
           && library "$home/targets/x86_64-linux/lib/libcudart.so.12" \
           && ln -sT targets/x86_64-linux/lib "$home/lib64"; }; then
    fail 'cannot build the toolkit fixture'
  fi
}

distro() { # distro-split layout: <root>/usr/lib/cuda/include/cuda.h -> <root>/usr/include; sets home, prefix
  prefix=$fx_root/usr
  home=$prefix/lib/cuda
  if ! { write "$prefix/include/cuda.h" \
           && runtime_headers "$prefix/include" \
           && mkdir -p -- "$home/include" \
           && ln -sT ../../../include/cuda.h "$home/include/cuda.h" \
           && library "$prefix/lib/x86_64-linux-gnu/libcudart.so.12"; }; then
    fail 'cannot build the distro fixture'
  fi
}

fixture() { # a fresh fixture tree per case, under the repository's target/tmp scratch
  home="" prefix="" cxx="" rc=""
  fx_root=$(mktemp -d "$SCRATCH/neural-env XXXXXXXX") || fail 'mktemp failed' || return 1
  fx_sdk=$fx_root/sdk
  fx_bin=$fx_root/bin
  fx_receipt=$fx_root/evidence/neural-discovery.json
  fx_fake_cxx=$fx_bin/fake-cxx
  fx_out=$fx_root/run.stdout
  fx_err=$fx_root/run.stderr
  local name
  mkdir -p -- "$fx_root/evidence" "$fx_bin" || fail 'cannot create the fixture directories' || return 1
  write "$fx_sdk/libtorch/build-version" "$EXPECTED_LIBTORCH"$'\n' || return 1
  for name in libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libc10_cuda.so; do
    write "$fx_sdk/libtorch/lib/$name" || return 1
  done
  write "$fx_sdk/libtorch/include/c10/cuda/CUDAStream.h" || return 1
  write "$fx_sdk/libtorch/include/torch/csrc/api/include/torch/torch.h" || return 1
  for name in "${COMPANIONS[@]}"; do
    library "$fx_sdk/nvidia/$name" || return 1
  done
  nccl_header "$EXPECTED_NCCL" || return 1
  write_fake_readelf "$fx_bin/readelf" || return 1
  write_fake_cxx "$fx_fake_cxx"
}

cleanup() { # remove the current case's fixture tree
  if [[ -n ${fx_root:-} && $fx_root == "$SCRATCH/neural-env "* ]]; then
    rm -rf -- "$fx_root"
  fi
  fx_root=""
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

real_compiler() { # sets cxx; the runner image pins clang++ 18, so no compiler at all is a failure
  cxx=$(type -P clang++ || type -P g++) \
    || fail 'the runner image pins clang++ 18; no C++ compiler is on PATH'
}

# run_env <execute|source> <CUDA_HOME> <CXX> [NAME=VALUE ...]: run neural-env.sh from the repository
# root with JOPE_* and the exported names cleared, the fixture bin/ first on PATH and the fixture SDK,
# CUDA_HOME, CXX and receipt path set. `source` is full-lane.sh's sourced pattern. Sets rc; output
# goes to $fx_out and $fx_err.
run_env() {
  local mode=$1 cuda_home=$2 compiler=$3 name
  shift 3
  local -a clear=() argv=(bash "$SCRIPT")
  while IFS= read -r name; do
    if [[ $name == JOPE_* ]]; then clear+=(-u "$name"); fi
  done < <(compgen -e)
  for name in "${NAMES[@]}"; do clear+=(-u "$name"); done
  if [[ $mode == source ]]; then argv=(bash -c "$SOURCE_PROBE" full-lane.sh "$SCRIPT"); fi
  rc=0
  (cd "$ROOT" && exec env "${clear[@]}" PATH="$fx_bin:$PATH" JOPE_NEURAL_SDK_ROOT="$fx_sdk" \
      CUDA_HOME="$cuda_home" CXX="$compiler" JOPE_NEURAL_ENV_RECEIPT="$fx_receipt" "$@" "${argv[@]}") \
    < /dev/null > "$fx_out" 2> "$fx_err" || rc=$?
}

exported_json() { # the run's stdout KEY=VALUE lines as a JSON object (each line needs a '=')
  jq -cRn 'reduce (inputs | split("=")) as $kv ({};
    if ($kv | length) < 2 then error("not a KEY=VALUE line") else . + {($kv[0]): ($kv[1:] | join("="))} end)' \
    "$fx_out"
}

# ---- assertions -----------------------------------------------------------------------------------

why=""
fail() { # <reason>: record why the case failed
  why=$1
  return 1
}

expect_eq() { # <what> <expected> <actual>
  [[ $2 == "$3" ]] || fail "$1: expected '$2', got '$3'"
}

expect_status() { # <expected exit status>
  [[ $rc == "$1" ]] || fail "exit status: expected $1, got $rc"
}

expect_doc() { # <what> <jq filter> [jq options...]: the discovery receipt satisfies the filter
  local what=$1 filter=$2
  shift 2
  jq -e "$@" "$filter" "$fx_receipt" > /dev/null 2>&1 \
    || fail "receipt: $what (receipt: $(head -c 600 "$fx_receipt" 2> /dev/null | tr '\n' ' '))"
}

expect_unavailable() { # <check>: failed closed at <check>, nothing exported, an unavailable receipt
  expect_status 1 || return 1
  [[ ! -s $fx_out ]] || fail "stdout is not empty: $(head -c 300 "$fx_out")" || return 1
  grep -qF 'neural-env: ' "$fx_err" || fail "stderr lacks 'neural-env: '" || return 1
  expect_doc 'status is unavailable' '.status == "unavailable"' || return 1
  expect_doc "failed_check is $1" '.failed_check == $check' --arg check "$1" || return 1
  expect_doc 'qualification.discovery is failed' '.qualification.discovery == "failed"' || return 1
  expect_doc 'environment is null' 'has("environment") and .environment == null'
}

# ---- cases (same names and assertions as the former test-neural-env.py) ---------------------------

test_toolkit_layout_is_discovered_with_the_real_compiler() {
  local exported pkg claim library_path=""
  toolkit || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_status 0 || return 1
  exported=$(exported_json) || fail 'stdout is not KEY=VALUE lines' || return 1
  expect_eq 'exported names' "$NAMES_JSON" "$(jq -c 'keys_unsorted' <<< "$exported")" || return 1
  expect_eq CUDA_HOME "$home" "$(jq -r '.CUDA_HOME' <<< "$exported")" || return 1
  for pkg in nccl cudnn cusparselt cufile nvshmem; do
    library_path+=$fx_sdk/nvidia/$pkg/lib:
  done
  library_path+=$home/lib64
  expect_eq JOPE_CUDA_LIBRARY_PATH "$library_path" "$(jq -r '.JOPE_CUDA_LIBRARY_PATH' <<< "$exported")" || return 1
  expect_doc 'status is discovered' '.status == "discovered"' || return 1
  expect_doc 'observed.cuda.layout is toolkit' '.observed.cuda.layout == "toolkit"' || return 1
  expect_doc "observed.cuda.cudart_version is $EXPECTED_CUDART" '.observed.cuda.cudart_version == $v' \
    --arg v "$EXPECTED_CUDART" || return 1
  expect_doc 'observed.compiler.resolved_cuda_runtime_h is the resolved toolkit header' \
    '.observed.compiler.resolved_cuda_runtime_h == $p' --arg p "$(readlink -f "$home/include/cuda_runtime.h")" \
    || return 1
  expect_doc 'observed.compiler.resolved_nccl_h is the wheel header' '.observed.compiler.resolved_nccl_h == $p' \
    --arg p "$fx_sdk/nvidia/nccl/include/nccl.h" || return 1
  expect_doc 'environment equals the printed lines' '.environment == $e' --argjson e "$exported" || return 1
  for claim in native_build native_link native_runtime gpu; do
    expect_doc "qualification.$claim is not-attempted" '.qualification[$c] == "not-attempted"' --arg c "$claim" \
      || return 1
  done
}

test_sourced_mode_exports_the_same_environment() {
  local exported
  toolkit || return 1
  real_compiler || return 1
  run_env source "$home" "$cxx"
  expect_status 0 || return 1
  exported=$(exported_json) || fail 'stdout is not KEY=VALUE lines' || return 1
  expect_doc 'environment equals the sourced exports' '.environment == $e' --argjson e "$exported"
}

test_sourced_failure_writes_receipt_and_maps_to_unavailable() {
  toolkit || return 1
  rm -- "$home/include/cuda_runtime.h" || fail 'cannot remove cuda_runtime.h' || return 1
  real_compiler || return 1
  run_env source "$home" "$cxx"
  expect_status 78 || return 1
  expect_doc 'failed_check is cuda-runtime-header' '.failed_check == "cuda-runtime-header"'
}

test_distro_split_layout_resolves_through_the_cuda_h_symlink() {
  local exported library_path
  distro || return 1
  run_env execute "$home" "$fx_fake_cxx" FAKE_CXX_SYSTEM_INCLUDE="$prefix/include"
  expect_status 0 || return 1
  exported=$(exported_json) || fail 'stdout is not KEY=VALUE lines' || return 1
  library_path=$(jq -r '.JOPE_CUDA_LIBRARY_PATH' <<< "$exported")
  [[ $library_path == *":$prefix/lib/x86_64-linux-gnu" ]] \
    || fail "JOPE_CUDA_LIBRARY_PATH '$library_path' does not end with :$prefix/lib/x86_64-linux-gnu" || return 1
  expect_doc 'observed.cuda.layout is distro-split' '.observed.cuda.layout == "distro-split"' || return 1
  expect_doc 'observed.cuda.include_dir is <prefix>/include' '.observed.cuda.include_dir == $d' \
    --arg d "$prefix/include"
}

test_distro_layout_the_compiler_cannot_see_is_rejected() {
  distro || return 1
  run_env execute "$home" "$fx_fake_cxx" # no system include directory
  expect_unavailable compiler-header-probe
}

test_missing_runtime_header_fails_closed() {
  toolkit || return 1
  rm -- "$home/include/cuda_runtime.h" || fail 'cannot remove cuda_runtime.h' || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable cuda-runtime-header
}

test_cuda_runtime_version_must_match_the_receipt() {
  toolkit 12090 || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable cudart-version
}

test_cudart_must_exist_with_its_soname() {
  local library_path
  toolkit || return 1
  real_compiler || return 1
  library_path=$home/lib64/libcudart.so.12
  write "$library_path" $'SONAME=libcudart.so.13\n' || fail 'cannot rewrite libcudart.so.12' || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable cudart-library || return 1
  rm -- "$fx_receipt" "$library_path" || fail 'cannot remove the receipt and libcudart.so.12' || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable cudart-library
}

test_host_nccl_header_version_is_rejected() {
  nccl_header 2.26.2 || return 1
  toolkit || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable nccl-header
}

test_libtorch_build_version_must_match_the_receipt() {
  write "$fx_sdk/libtorch/build-version" $'2.13.0+cu130\n' || fail 'cannot rewrite build-version' || return 1
  toolkit || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable libtorch-build-version
}

companion_is_required() { # <companion>: one subtest; restores the library and removes the receipt
  local path=$fx_sdk/nvidia/$1
  rm -- "$path" || fail "cannot remove $path" || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  expect_unavailable companion-libraries || return 1
  library "$path" || fail "cannot restore $path" || return 1
  rm -- "$fx_receipt" || fail 'cannot remove the receipt'
}

test_every_companion_library_is_required() {
  local name
  local -a broken=()
  toolkit || return 1
  for name in "${COMPANIONS[@]}"; do
    companion_is_required "$name" || broken+=("[$name] $why")
  done
  ((${#broken[@]} == 0)) || fail "subtests failed: ${broken[*]}"
}

test_missing_compiler_fails_closed() {
  toolkit || return 1
  run_env execute "$home" "$fx_root/no-such-compiler"
  expect_unavailable compiler
}

test_compiler_resolving_a_shadowing_header_is_rejected() {
  local shadow
  toolkit || return 1
  shadow=$fx_root/usr/include/nccl.h
  write "$shadow" $'#define NCCL_MAJOR 2\n' || fail 'cannot write the shadowing header' || return 1
  run_env execute "$home" "$fx_fake_cxx" FAKE_CXX_FORCE_NCCL="$shadow"
  expect_unavailable compiler-header-probe
}

test_existing_receipt_is_never_overwritten() {
  write "$fx_receipt" $'retained\n' || fail 'cannot write the prior receipt' || return 1
  toolkit || return 1
  real_compiler || return 1
  run_env execute "$home" "$cxx"
  [[ $rc != 0 ]] || fail 'exit status: expected nonzero, got 0' || return 1
  printf 'retained\n' | cmp -s - "$fx_receipt" \
    || fail "the existing receipt was overwritten: $(head -c 300 "$fx_receipt")"
}

# ---- runner ---------------------------------------------------------------------------------------

CASES=(
  test_compiler_resolving_a_shadowing_header_is_rejected
  test_cuda_runtime_version_must_match_the_receipt
  test_cudart_must_exist_with_its_soname
  test_distro_layout_the_compiler_cannot_see_is_rejected
  test_distro_split_layout_resolves_through_the_cuda_h_symlink
  test_every_companion_library_is_required
  test_existing_receipt_is_never_overwritten
  test_host_nccl_header_version_is_rejected
  test_libtorch_build_version_must_match_the_receipt
  test_missing_compiler_fails_closed
  test_missing_runtime_header_fails_closed
  test_sourced_failure_writes_receipt_and_maps_to_unavailable
  test_sourced_mode_exports_the_same_environment
  test_toolkit_layout_is_discovered_with_the_real_compiler
)

mkdir -p -- "$SCRATCH" || die "cannot create $SCRATCH"
passed=0
failed=0
for case in "${CASES[@]}"; do
  why=""
  if fixture && "$case"; then
    printf 'PASS %s\n' "$case"
    passed=$((passed + 1))
  else
    printf 'FAIL %s: %s\n' "$case" "${why:-fixture setup failed}"
    if [[ -s ${fx_err:-} ]]; then
      head -n 20 "$fx_err" | sed 's/^/    neural-env stderr: /'
    fi
    failed=$((failed + 1))
  fi
  cleanup
  fx_err=""
done

# unittest's summary shape: deploy/ci/results.py counts the hygiene lane's tests from it.
printf 'Ran %d tests in %ds (%d passed, %d failed)\n\n' "${#CASES[@]}" "$SECONDS" "$passed" "$failed"
if ((failed > 0)); then
  printf 'FAILED (failures=%d)\n' "$failed"
  exit 1
fi
echo OK
