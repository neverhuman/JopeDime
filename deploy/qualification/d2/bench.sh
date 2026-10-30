#!/usr/bin/env bash
# D2 benchmark harness. The executable hooks adapt the release candidate's CLI:
#   D2_READY_HOOK <run-id>                 checks shell readiness on a live engine
#   D2_AUTOSTART_HOOK <run-id>             starts the engine and checks readiness
#   D2_FIRST_MODEL_HOOK <dataset> fast <run-id>
#                                          creates and validates one fast model
# Each hook must exit 0 only when its operation succeeds. The auto-start hook
# owns any child-process cleanup; each run should use run-id to isolate outputs.
set -euo pipefail

readonly RUNNING_RUNS=20
readonly AUTOSTART_RUNS=20
readonly MODEL_RUNS=10

fail() {
  printf 'd2 benchmark: %s\n' "$*" >&2
  exit 2
}

require_hook() {
  local name=$1 value=$2
  [[ -n $value ]] || fail "set $name to an executable adapter"
  [[ -f $value && -x $value ]] || fail "$name must name an executable file: $value"
}

require_hook D2_READY_HOOK "${D2_READY_HOOK:-}"
require_hook D2_AUTOSTART_HOOK "${D2_AUTOSTART_HOOK:-}"
require_hook D2_FIRST_MODEL_HOOK "${D2_FIRST_MODEL_HOOK:-}"

dataset=${D2_REFERENCE_DATASET:-}
[[ -n $dataset && -f $dataset && -r $dataset ]] || fail "D2_REFERENCE_DATASET must name a readable QLT-05 reference dataset"
[[ ! $dataset =~ [[:cntrl:]] ]] || fail "dataset path cannot contain control characters"

report=${D2_REPORT:-d2-report.json}
[[ ! $report =~ [[:cntrl:]] ]] || fail "D2_REPORT path cannot contain control characters"
command -v date >/dev/null || fail "date is required"
command -v sort >/dev/null || fail "sort is required"
command -v awk >/dev/null || fail "awk is required"
command -v perl >/dev/null || fail "perl with Time::HiRes is required for monotonic timing"

timestamp() {
  perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%.0f", clock_gettime(CLOCK_MONOTONIC) * 1e9'
}

declare -a running_ms=()
declare -a autostart_ms=()
declare -a model_ms=()

measure() {
  local label=$1 hook=$2 run_id=$3 start end elapsed status
  shift 3
  start=$(timestamp)
  if "$hook" "$@" >/dev/null 2>&1; then
    status=0
  else
    status=$?
  fi
  end=$(timestamp)
  (( status == 0 )) || fail "$label run $run_id failed (hook exit $status)"
  elapsed=$(((end - start) / 1000000))
  printf '%s' "$elapsed"
}

# The readiness preflight is intentionally outside the timed 20 samples.
"$D2_READY_HOOK" preflight >/dev/null 2>&1 || fail "engine-running readiness preflight failed"

for ((i = 1; i <= RUNNING_RUNS; i++)); do
  running_ms+=("$(measure running "$D2_READY_HOOK" "$i" "$i")")
done
for ((i = 1; i <= AUTOSTART_RUNS; i++)); do
  autostart_ms+=("$(measure autostart "$D2_AUTOSTART_HOOK" "$i" "$i")")
done
for ((i = 1; i <= MODEL_RUNS; i++)); do
  model_ms+=("$(measure first-model-fast "$D2_FIRST_MODEL_HOOK" "$i" "$dataset" fast "$i")")
done

json_quote() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "$value"
}

json_numbers() {
  local first=1 value
  printf '['
  for value in "$@"; do
    (( first )) || printf ','
    printf '%s' "$value"
    first=0
  done
  printf ']'
}

max_ms() {
  printf '%s\n' "$@" | sort -n | tail -n 1
}

p95_ms() {
  printf '%s\n' "$@" | sort -n | awk -v n="$#" 'NR == int(0.95 * n + 0.999999) { print; found=1 } END { if (!found) exit 1 }'
}

running_p95=$(p95_ms "${running_ms[@]}")
autostart_p95=$(p95_ms "${autostart_ms[@]}")
model_p95=$(p95_ms "${model_ms[@]}")
running_max=$(max_ms "${running_ms[@]}")
autostart_max=$(max_ms "${autostart_ms[@]}")
model_max=$(max_ms "${model_ms[@]}")
if command -v sha256sum >/dev/null; then
  dataset_hash=$(sha256sum -- "$dataset" | awk '{print $1}')
elif command -v shasum >/dev/null; then
  dataset_hash=$(shasum -a 256 "$dataset" | awk '{print $1}')
else
  fail "sha256sum or shasum is required"
fi
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
running_pass=false
autostart_pass=false
model_pass=false
(( running_p95 < 500 )) && running_pass=true
(( autostart_p95 < 1500 )) && autostart_pass=true
(( model_max < 5000 )) && model_pass=true

report_dir=$(dirname -- "$report")
[[ -d $report_dir ]] || fail "report directory does not exist: $report_dir"
report_tmp=$(mktemp "$report.tmp.XXXXXX")
trap 'rm -f -- "$report_tmp"' EXIT
{
  printf '{\n  "schema": "jope.d2-benchmark/v1",\n'
  printf '  "completed_at_utc": "%s",\n' "$completed_at"
  printf '  "reference_dataset": {"path": '
  json_quote "$dataset"
  printf ', "sha256": "%s"},\n' "$dataset_hash"
  printf '  "thresholds_ms": {"shell_ready_running_p95_lt": 500, "shell_ready_auto_start_p95_lt": 1500, "first_validated_model_each_lt": 5000},\n'
  printf '  "shell_ready_running": {"runs": %d, "p95_ms": %s, "max_ms": %s, "pass": %s, "samples_ms": ' "$RUNNING_RUNS" "$running_p95" "$running_max" "$running_pass"
  json_numbers "${running_ms[@]}"
  printf '},\n'
  printf '  "shell_ready_auto_start": {"runs": %d, "p95_ms": %s, "max_ms": %s, "pass": %s, "samples_ms": ' "$AUTOSTART_RUNS" "$autostart_p95" "$autostart_max" "$autostart_pass"
  json_numbers "${autostart_ms[@]}"
  printf '},\n'
  printf '  "first_validated_model_fast": {"runs": %d, "successful_validations": %d, "p95_ms": %s, "max_ms": %s, "pass": %s, "samples_ms": ' "$MODEL_RUNS" "${#model_ms[@]}" "$model_p95" "$model_max" "$model_pass"
  json_numbers "${model_ms[@]}"
  printf '}\n}\n'
} >"$report_tmp"
mv -- "$report_tmp" "$report"
trap - EXIT
cat -- "$report"
if [[ $running_pass != true || $autostart_pass != true || $model_pass != true ]]; then
  exit 1
fi
