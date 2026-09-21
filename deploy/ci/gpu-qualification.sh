#!/usr/bin/env bash
# shellcheck disable=SC2317  # the check_* functions are invoked by name through run_step (indirect calls look unreachable to shellcheck)
# deploy/ci/gpu-qualification.sh — wrapper of the self-hosted `neural-cuda` lane (main.yml,
# nightly.yml). Runs the CUDA-bound builds and tests under the exact SDK environment of
# deploy/receipts/libtorch.json and writes a `jope.cuda-qualification/v1` receipt
# (schemas/jope.cuda-qualification.v1.schema.json) recording the GPU identity before and after,
# the SDK digests, every command with exit code and log sha256, and the number of skip markers
# ("SKIP"/"skipping") found in the logs. The receipt is validated against the schema before the
# script exits; the exit code is 0 only for result "passed" (or "dry-run").
#
# Usage: deploy/ci/gpu-qualification.sh [--dry-run] [--output <receipt.json>]
#   --dry-run   run no cargo command and touch no GPU: records the environment, marks every step
#               not-run, writes and validates the receipt (result "dry-run"), exits 0.
#
# Steps (fail closed; a step that needs a work package that has not landed is "blocked" with the
# exact reason and makes the result "blocked"):
#   sdk-env         . deploy/ci/neural-env.sh (LibTorch build-version must equal the receipt)
#   clippy-neural   cargo clippy -p jope-parallel --features neural --all-targets --locked
#   nextest-neural  cargo nextest run -P main -p jope-parallel --features neural --locked
#   battle-gpu      cargo test -p battle-gpu --locked -- --nocapture  (the log must show the
#                   "battle-gpu test device:" probe line; otherwise the GPU tests returned early)
#   hyperion-cuda   cargo test -p hyperion --features hyperion-cuda --locked  (blocked until the
#                   `hyperion` crate joins the workspace — WP3; crates/hyperion/host is excluded)
set -euo pipefail
cd "$(dirname "$0")/../.."

mode=run
receipt_out=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) mode=dry-run ;;
    --output) receipt_out=${2:?--output needs a path}; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "gpu-qualification: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

for tool in cargo jq sha256sum python3 git; do
  command -v "$tool" >/dev/null || { echo "gpu-qualification: missing tool $tool" >&2; exit 2; }
done

export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JOPE_PARITY_STRICT=1 CARGO_TERM_COLOR=never

sdk_receipt=deploy/receipts/libtorch.json
schema=schemas/jope.cuda-qualification.v1.schema.json
target=$(cargo metadata --no-deps --locked --format-version 1 | jq -r .target_directory)
outdir=${JOPE_QUALIFICATION_DIR:-$target/cuda-qualification}
receipt=${receipt_out:-$outdir/receipt.json}
rm -rf "$outdir"
mkdir -p "$outdir" "$(dirname "$receipt")"
echo "gpu-qualification: mode=$mode outdir=$outdir"

# ---- read-only GPU query --------------------------------------------------------------------
gpu_json() {
  local line
  if ! command -v nvidia-smi >/dev/null; then
    echo '{"available":false,"reason":"nvidia-smi not on PATH"}'; return
  fi
  if ! line=$(nvidia-smi --id=0 --query-gpu=name,uuid,driver_version,compute_cap,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null); then
    echo '{"available":false,"reason":"nvidia-smi query failed"}'; return
  fi
  jq -Rc 'split(", ") | {available:true, name:.[0], uuid:.[1], driver_version:.[2], compute_capability:.[3], memory_used_mib:(.[4]|tonumber), utilization_percent:(.[5]|tonumber)}' <<<"$line"
}
gpu_before=$(gpu_json)
echo "gpu before: $gpu_before"

# ---- SDK facts (never modifies anything) ----------------------------------------------------
sdk_root=${JOPE_NEURAL_SDK_ROOT:-/opt/jope-sdk}
sdk_available=false
sdk_reason=""
libtorch_expected=$(jq -r .libtorch.version "$sdk_receipt")
libtorch_zip_sha=$(jq -r .libtorch.sha256 "$sdk_receipt")
nccl_expected=$(jq -r '.companions[] | select(.package=="nvidia-nccl-cu12") | .version' "$sdk_receipt")
libtorch_actual=""
libtorch_cuda_sha=""
set +e
sdk_env_output=$(bash deploy/ci/neural-env.sh 2>&1)
sdk_env_code=$?
set -e
if [[ $sdk_env_code -eq 0 ]]; then
  sdk_available=true
  libtorch_actual=$(< "$sdk_root/libtorch/build-version")
  if [[ -f "$sdk_root/libtorch/lib/libtorch_cuda.so" ]]; then
    libtorch_cuda_sha=$(sha256sum "$sdk_root/libtorch/lib/libtorch_cuda.so" | cut -c1-64)
  fi
else
  sdk_reason=$(tail -1 <<<"$sdk_env_output")
fi
printf '%s\n' "$sdk_env_output" > "$outdir/sdk-env.log"

# ---- steps -----------------------------------------------------------------------------------
steps='[]'
add_step() { # <json object>
  steps=$(jq -c --argjson s "$1" '. + [$s]' <<<"$steps")
}
step_record() { # name command status exit_code seconds log log_sha256 skip_markers [extra-json]
  local extra=${9:-'{}'}
  jq -nc --arg n "$1" --arg c "$2" --arg st "$3" --argjson ec "$4" --argjson sec "$5" --arg log "$6" \
    --arg sha "$7" --argjson sk "$8" --argjson extra "$extra" \
    '{name:$n,command:$c,status:$st,exit_code:$ec,seconds:$sec,log:$log,log_sha256:$sha,skip_markers:$sk} + $extra'
}
run_step() { # name command [check-function]
  local name=$1 cmd=$2 check=${3:-}
  local log="$outdir/$name.log" start end code status markers sha extra='{}'
  if [[ $mode == dry-run ]]; then
    add_step "$(step_record "$name" "$cmd" not-run -1 0 "" "" 0 '{"reason":"dry-run"}')"
    return 0
  fi
  echo "== $name: $cmd"
  start=$(date +%s)
  set +e
  bash -o pipefail -c "$cmd" 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  set -e
  end=$(date +%s)
  markers=$(grep -c -E '\bSKIP\b|\bskipping\b' "$log" || true)
  sha=$(sha256sum "$log" | cut -c1-64)
  status=passed
  [[ $code -eq 0 ]] || status=failed
  if [[ $markers -gt 0 ]]; then status=failed; extra=$(jq -nc --arg r "$markers skip marker(s) in the log" '{reason:$r}'); fi
  if [[ $status == passed && -n $check ]]; then
    local why
    if ! why=$("$check" "$log"); then status=failed; extra=$(jq -nc --arg r "$why" '{reason:$r}'); fi
  fi
  if [[ -n $check && $status == passed ]]; then extra=$("$check" "$log" --extra 2>/dev/null || echo '{}'); fi
  add_step "$(step_record "$name" "$cmd" "$status" "$code" "$((end - start))" "${log#"$target"/}" "$sha" "$markers" "$extra")"
  echo "== $name: $status (exit $code, ${markers} skip markers, $((end - start)) s)"
}
block_step() { # name command reason
  add_step "$(step_record "$1" "$2" blocked -1 0 "" "" 0 "$(jq -nc --arg r "$3" '{reason:$r}')")"
  echo "== $1: BLOCKED — $3"
}

# Checks receive the log path; with --extra they print a JSON object to merge into the step.
check_clippy() {
  local log=$1
  if [[ ${2:-} == --extra ]]; then
    jq -nc --argjson w "$(grep -c -E '^warning: ' "$log" || true)" '{warnings:$w}'; return 0
  fi
  return 0
}
check_nextest() {
  local log=$1 junit="$target/nextest/main/junit-main.xml"
  if [[ ${2:-} == --extra ]]; then
    cp "$junit" "$outdir/junit-neural.xml"
    python3 - "$outdir/junit-neural.xml" <<'PY'
import json, sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
print(json.dumps({"junit": {"tests": int(root.get("tests", 0)), "failures": int(root.get("failures", 0)),
      "errors": int(root.get("errors", 0)), "path": sys.argv[1]}}))
PY
    return 0
  fi
  [[ -f $junit ]] || { echo "nextest wrote no junit at $junit"; return 1; }
  return 0
}
check_battle_gpu() {
  local log=$1
  [[ ${2:-} == --extra ]] && { echo '{}'; return 0; }
  grep -q 'battle-gpu test device:' "$log" \
    || { echo "no 'battle-gpu test device:' line: gpu_available() found no GPU and the GPU tests returned early"; return 1; }
  return 0
}

if [[ $mode == run ]]; then
  if [[ $sdk_available != true ]]; then
    add_step "$(step_record sdk-env "bash deploy/ci/neural-env.sh" failed "$sdk_env_code" 0 "${outdir#"$target"/}/sdk-env.log" "$(sha256sum "$outdir/sdk-env.log" | cut -c1-64)" 0 "$(jq -nc --arg r "$sdk_reason" '{reason:$r}')")"
  else
    add_step "$(step_record sdk-env "bash deploy/ci/neural-env.sh" passed 0 0 "${outdir#"$target"/}/sdk-env.log" "$(sha256sum "$outdir/sdk-env.log" | cut -c1-64)" 0)"
    # shellcheck source=deploy/ci/neural-env.sh
    # shellcheck disable=SC1091  # sourced at run time from the repo root; shellcheck -x follows it
    . deploy/ci/neural-env.sh
    run_step clippy-neural "cargo clippy -p jope-parallel --features neural --all-targets --locked" check_clippy
    run_step nextest-neural "cargo nextest run -P main -p jope-parallel --features neural --locked" check_nextest
    run_step battle-gpu "cargo test -p battle-gpu --locked -- --nocapture" check_battle_gpu
    if cargo metadata --no-deps --locked --format-version 1 | jq -e '.packages[] | select(.name=="hyperion")' >/dev/null; then
      run_step hyperion-cuda "cargo test -p hyperion --features hyperion-cuda --locked"
    else
      block_step hyperion-cuda "cargo test -p hyperion --features hyperion-cuda --locked" \
        "package 'hyperion' is not a workspace member (WP3 adds crates/hyperion); crates/hyperion/host is excluded from the workspace and points at ../../assets/source/hyperion-jain, which is not in the tree"
    fi
  fi
else
  add_step "$(step_record sdk-env "bash deploy/ci/neural-env.sh" not-run "$sdk_env_code" 0 "${outdir#"$target"/}/sdk-env.log" "$(sha256sum "$outdir/sdk-env.log" | cut -c1-64)" 0 "$(jq -nc --arg r "dry-run; SDK available=$sdk_available $sdk_reason" '{reason:$r}')")"
  run_step clippy-neural "cargo clippy -p jope-parallel --features neural --all-targets --locked"
  run_step nextest-neural "cargo nextest run -P main -p jope-parallel --features neural --locked"
  run_step battle-gpu "cargo test -p battle-gpu --locked -- --nocapture"
  run_step hyperion-cuda "cargo test -p hyperion --features hyperion-cuda --locked"
fi

gpu_after=$(gpu_json)
echo "gpu after: $gpu_after"
stable=$(jq -n --argjson a "$gpu_before" --argjson b "$gpu_after" \
  '($a.available == true) and ($b.available == true) and ($a.uuid == $b.uuid) and ($a.driver_version == $b.driver_version) and ($a.compute_capability == $b.compute_capability)')

# ---- result ----------------------------------------------------------------------------------
skip_total=$(jq '[.[].skip_markers] | add // 0' <<<"$steps")
blocked=$(jq -c '[.[] | select(.status=="blocked") | .name]' <<<"$steps")
if [[ $mode == dry-run ]]; then
  result=dry-run
elif jq -e 'any(.[]; .status=="failed")' <<<"$steps" >/dev/null; then
  result=failed
elif jq -e 'any(.[]; .status=="blocked")' <<<"$steps" >/dev/null; then
  result=blocked
elif [[ $stable != true ]]; then
  result=failed
else
  result=passed
fi

in_container=false
[[ -f /.dockerenv ]] && in_container=true
nvcc_version=$( (nvcc --version 2>/dev/null || "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version 2>/dev/null) | grep -o 'release [0-9.]*, V[0-9.]*' || echo "nvcc not found")
cxx_bin=${CXX:-clang++}
cxx_version=$("$cxx_bin" --version 2>/dev/null | head -1 || echo "$cxx_bin not found")
dirty=false
[[ -n $(git status --porcelain --untracked-files=no) ]] && dirty=true

jq -n \
  --arg produced_at "$(date -u +%FT%TZ)" \
  --arg mode "$mode" \
  --arg sha "$(git rev-parse HEAD)" --arg ref "$(git rev-parse --abbrev-ref HEAD)" --argjson dirty "$dirty" \
  --arg hostname "$(hostname)" --arg kernel "$(uname -r)" --arg rustc "$(rustc --version)" --arg cargo "$(cargo --version)" \
  --arg cxx "$cxx_version" --arg nvcc "$nvcc_version" --argjson in_container "$in_container" \
  --argjson before "$gpu_before" --argjson after "$gpu_after" --argjson stable "$stable" \
  --arg sdk_root "$sdk_root" --argjson sdk_available "$sdk_available" --arg sdk_receipt "$sdk_receipt" \
  --arg sdk_receipt_sha "$(sha256sum "$sdk_receipt" | cut -c1-64)" \
  --arg lt_expected "$libtorch_expected" --arg lt_actual "$libtorch_actual" --arg lt_zip "$libtorch_zip_sha" \
  --arg lt_cuda "$libtorch_cuda_sha" --arg cuda_home "${CUDA_HOME:-}" --arg nccl "$nccl_expected" --arg sdk_reason "$sdk_reason" \
  --argjson steps "$steps" --argjson skip "$skip_total" --argjson blocked "$blocked" --arg result "$result" \
  '{schema:"jope.cuda-qualification/v1", produced_at:$produced_at, mode:$mode,
    git:{sha:$sha, ref:$ref, dirty:$dirty},
    host:{hostname:$hostname, kernel:$kernel, rustc:$rustc, cargo:$cargo, cxx:$cxx, nvcc:$nvcc, in_container:$in_container},
    gpu:{before:$before, after:$after, stable:$stable},
    sdk:{root:$sdk_root, available:$sdk_available, receipt:$sdk_receipt, receipt_sha256:$sdk_receipt_sha,
         libtorch_version_expected:$lt_expected, libtorch_version_actual:$lt_actual,
         libtorch_zip_sha256_expected:$lt_zip, libtorch_cuda_so_sha256:$lt_cuda, cuda_home:$cuda_home,
         nccl_version_expected:$nccl, reason:$sdk_reason},
    steps:$steps, skip_markers:$skip, blocked_by:$blocked, result:$result}' > "$receipt"

python3 deploy/ci/validate-receipt.py "$schema" "$receipt"
echo "gpu-qualification: result=$result receipt=$receipt"
jq -r '.steps[] | "  \(.name): \(.status)\(if .reason then " — " + .reason else "" end)"' "$receipt"
case $result in passed|dry-run) exit 0 ;; *) exit 1 ;; esac
