#!/usr/bin/env bash
# Linux-only REL-04 pre-stage. Full D1 still requires a signed candidate and macOS/Windows legs.
set -euo pipefail

usage() {
  echo "usage: $0 JOPEDIME WORKDIR [--csv TRAIN TARGET PREDICT] [--require-golden-path]" >&2
  exit 2
}
[[ $# -ge 2 ]] || usage

binary=$(realpath -- "$1")
work=$(realpath -m -- "$2")
shift 2
inside=false
require_golden=false
input_kind=synthetic
target=y
train_csv=
predict_csv=
options=()
while (($#)); do
  case $1 in
    --inside-netns) inside=true; shift ;;
    --require-golden-path) require_golden=true; options+=("$1"); shift ;;
    --csv)
      [[ $# -ge 4 && -n $3 ]] || usage
      train_csv=$(realpath -- "$2")
      target=$3
      predict_csv=$(realpath -- "$4")
      [[ -r $train_csv && -f $train_csv && -r $predict_csv && -f $predict_csv ]] || usage
      input_kind=supplied
      options+=(--csv "$train_csv" "$target" "$predict_csv")
      shift 4
      ;;
    *) usage ;;
  esac
done
script=$(realpath -- "${BASH_SOURCE[0]}")
script_dir=$(dirname -- "$script")
repo_root=$(cd -- "$script_dir/../.." && pwd)
[[ -f $binary && -x $binary ]] || { echo "clean-host pre-stage: executable missing: $binary" >&2; exit 2; }
case $work in /tmp | /tmp/*) echo "$0: never under /tmp (AGENTS.md)" >&2; exit 2 ;; esac
[[ ! -e $work ]] || { echo "$0: work directory exists: $work" >&2; exit 2; }

if [[ $inside != true ]]; then
  unshare_bin=$(command -v unshare || true)
  [[ -n $unshare_bin ]] || { echo "clean-host pre-stage: unshare is required for network isolation" >&2; exit 1; }
  exec "$unshare_bin" --user --map-root-user --net "$BASH" "$script" "$binary" "$work" "${options[@]}" --inside-netns
fi

ip_bin=$(command -v ip || true)
[[ -n $ip_bin ]] || { echo "clean-host pre-stage: ip is required to enable isolated loopback" >&2; exit 1; }
"$ip_bin" link set lo up
[[ $( "$ip_bin" -o link show | wc -l ) -eq 1 ]] || {
  echo "clean-host pre-stage: network namespace contains a non-loopback interface" >&2; exit 1;
}
[[ -z $( "$ip_bin" route show default ) ]] || {
  echo "clean-host pre-stage: network namespace has a default route" >&2; exit 1;
}

mkdir -p "$work/home" "$work/tmp" "$work/no-toolchain"
chmod 0700 "$work/home" "$work/tmp"
cp -- "$binary" "$work/jopedime"
chmod 0755 "$work/jopedime"
toolchain_path="$work/no-toolchain"
for tool in python python3 rustc node; do
  if PATH="$toolchain_path" command -v "$tool" >/dev/null 2>&1; then
    echo "clean-host pre-stage: unexpected toolchain executable in PATH: $tool" >&2
    exit 1
  fi
done

app() {
  env -i \
    HOME="$work/home" \
    JOPEDIME_HOME="$work/home" \
    TMPDIR="$work/tmp" \
    PATH="$toolchain_path" \
    "$work/jopedime" "$@"
}

engine_pid=
step=input
method_discovery=not-run
dataset_registration=not-run
run_discovery_status=not-run
discovery_reason=
export_mode=not-run
run_id=
dataset_id=
prediction_rows=null
link_audit=not-run
declare -A export_status=([python]=not-run [rust]=not-run [javascript]=not-run)
cleanup() {
  if [[ -n ${engine_pid:-} ]]; then
    app serve --stop >/dev/null 2>&1 || true
    if kill -0 "$engine_pid" 2>/dev/null; then
      kill "$engine_pid" 2>/dev/null || true
    fi
    wait "$engine_pid" 2>/dev/null || true
  fi
}
digest() {
  if [[ -f $1 ]]; then sha256sum < "$1" | cut -d' ' -f1; fi
}
write_receipt() {
  local status=$1
  jq -n \
    --argjson exit_status "$status" --arg failed_step "$step" \
    --arg source_sha "$(git -C "$repo_root" rev-parse HEAD)" \
    --arg harness_sha256 "$(digest "$script")" \
    --arg binary_sha256 "$(digest "$work/jopedime")" \
    --arg input_kind "$input_kind" --arg target "$target" \
    --arg train_sha256 "$(digest "$work/train.csv")" \
    --arg predict_sha256 "$(digest "$work/predict.csv")" \
    --arg method_discovery "$method_discovery" --arg dataset_registration "$dataset_registration" \
    --arg run_id "$run_id" --arg dataset_id "$dataset_id" \
    --arg run_discovery_status "$run_discovery_status" --arg discovery_reason "$discovery_reason" \
    --arg export_mode "$export_mode" --argjson prediction_rows "$prediction_rows" \
    --arg link_audit "$link_audit" \
    --arg python_status "${export_status[python]}" --arg python_sha256 "$(digest "$work/export-python.bin")" \
    --arg rust_status "${export_status[rust]}" --arg rust_sha256 "$(digest "$work/export-rust.bin")" \
    --arg javascript_status "${export_status[javascript]}" --arg javascript_sha256 "$(digest "$work/export-javascript.bin")" \
    '{schema:"jope.appliance-clean-host/v1", qualification:"pre-stage", platform:"linux-x86_64",
      status:(if $exit_status == 0 then "passed" else "failed" end), exit_status:$exit_status,
      failed_step:(if $exit_status == 0 then null else $failed_step end),
      prestage_checks_passed:($exit_status == 0), source_sha:$source_sha, harness_sha256:$harness_sha256,
      binary_sha256:$binary_sha256,
      isolation:"network namespace; loopback only; no default route",
      toolchain_path:"empty (python, python3, rustc and node unavailable to the app; not filesystem removal)",
      input:{kind:$input_kind,target:$target,train_sha256:$train_sha256,predict_sha256:$predict_sha256},
      method_discovery:$method_discovery,dataset_registration:$dataset_registration,
      run_discovery:{status:$run_discovery_status,reason:$discovery_reason,dataset_id:$dataset_id,run_id:$run_id},
      export_mode:$export_mode,
      export_status:{python:$python_status,rust:$rust_status,javascript:$javascript_status},
      source_exports:{python_sha256:$python_sha256,rust_sha256:$rust_sha256,javascript_sha256:$javascript_sha256},
      prediction_rows:$prediction_rows,link_audit:$link_audit,
      real_data_parity:{status:"ignored",reason:"this interface pre-stage does not establish model parity"},
      pending_platforms:["macos-arm64","windows-x86_64"],release_gate_d1:"not qualified"}' \
    > "$work/clean-host.json"
}
finish() {
  local status=$?
  trap - EXIT
  cleanup
  if [[ $status -ne 0 && $step == run_start && -f $work/discovery.err ]]; then
    discovery_reason=$(< "$work/discovery.err")
  fi
  write_receipt "$status" || { echo "clean-host pre-stage: receipt write failed" >&2; exit 1; }
  cat "$work/clean-host.json"
  exit "$status"
}
trap finish EXIT

if [[ $input_kind == supplied ]]; then
  cp -- "$train_csv" "$work/train.csv"
  cp -- "$predict_csv" "$work/predict.csv"
else
  # Deterministic interface fixture only; never a substitute for real-data qualification.
  awk 'BEGIN {
  print "x,y"
  for (i = 0; i < 48; i++) printf "%d,%.3f\n", i, (2 * i) + 1
}' > "$work/train.csv"
  printf 'x\n3\n17\n31\n' > "$work/predict.csv"
fi
# Fixtures use ordinary CSV with one physical line per record, including the header.
expected_rows=$(( $(wc -l < "$work/predict.csv") - 1 ))
[[ $expected_rows -gt 0 ]] || { echo "clean-host pre-stage: prediction input has no rows" >&2; exit 1; }

echo "== Linux pre-stage clean-host run (network namespace, loopback only)"
step=version
app --version --json > "$work/version.json"
# Discover methods before starting the engine so both execution branches have measured evidence.
step=method_discovery
app methods --task regression --json > "$work/methods.json"
jq -e '[.methods[] | select(.id | ascii_downcase | contains("ridge"))] | length > 0' \
  "$work/methods.json" >/dev/null
method_discovery=passed
# Measure this independently even if the later run lifecycle refuses discovery.
step=link_audit
"$repo_root/deploy/ci/link-audit.sh" "$work/jopedime" fused-cpu > "$work/link-audit.txt" 2>&1
cat "$work/link-audit.txt"
link_audit=passed
step=engine_start
app serve --idle-exit 10m > "$work/engine.log" 2>&1 &
engine_pid=$!

ready=false
for _ in $(seq 1 100); do
  if ! kill -0 "$engine_pid" 2>/dev/null; then
    cat "$work/engine.log" >&2
    echo "clean-host pre-stage: local engine exited before readiness" >&2
    exit 1
  fi
  if app serve --status 2>/dev/null | grep -q '^running'; then
    ready=true
    break
  fi
  sleep 0.1
done
[[ $ready == true ]] || { cat "$work/engine.log" >&2; echo "clean-host pre-stage: local engine did not start" >&2; exit 1; }

step=dataset_register
dataset_registration=failed
export_mode=run
if app dataset add "$work/train.csv" --name "rel04-$input_kind" --json \
  > "$work/dataset.json" 2> "$work/dataset.err"; then
  dataset_registration=passed
  dataset_id=$(jq -er '.dataset_id // .id' "$work/dataset.json")
  step=run_start
  run_discovery_status=failed
  app run \
    --dataset "$dataset_id" --target "$target" --task regression \
    --split '{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}' \
    --objective rmse --effort fast --budget '{"evaluations":"1","wall_seconds":"180"}' \
    --landmarks none --guidance heuristic --language python --language rust --language javascript \
    --require-all-languages --device cpu --seed 17 --wait 2 --json \
    > "$work/discovery.json" 2> "$work/discovery.err"
  run_id=$(jq -er 'select(.state == "completed") | .run_id' "$work/discovery.json")
  jq -e '.best != null' "$work/discovery.json" >/dev/null
  run_discovery_status=passed

  for language in python rust javascript; do
    step="export_$language"
    export_status[$language]=failed
    app export "$run_id" --kind source --lang "$language" --out "$work/export-$language.bin" --json \
      > "$work/export-$language.json"
    [[ -s "$work/export-$language.bin" ]] || {
      echo "clean-host pre-stage: $language run export is empty" >&2; exit 1;
    }
    export_status[$language]=passed
  done
  step=predict
  app predict --model "$run_id" --input "$work/predict.csv" --output "$work/predictions.csv"
else
  if [[ $require_golden == true ]]; then
    cat "$work/dataset.err" >&2
    echo "clean-host pre-stage: required golden path cannot substitute a local fit" >&2
    exit 1
  fi
  if ! grep -Fq 'Operation dataset_register is not implemented in this build' "$work/dataset.err"; then
    cat "$work/dataset.err" >&2
    echo "clean-host pre-stage: dataset registration failed for an unexpected reason" >&2
    exit 1
  fi
  run_discovery_status=ignored
  discovery_reason='dataset_register is an explicit unimplemented operation on this base; public run discovery cannot start'
  export_mode='local fitted object; run-backed exports pending dataset_register'
  cat "$work/dataset.err" > "$work/discovery-pending.txt"
  app serve --stop >/dev/null
  wait "$engine_pid" || true
  engine_pid=

  step=local_fit
  app fit --train "$work/train.csv" --target "$target" --method auto --candidates ridge \
    --folds 2 --model-root "$work/models" --device cpu --threads 1 --json \
    > "$work/fit.json"
  model=$(find "$work/models" -mindepth 1 -maxdepth 1 -type d -name 'sha256-*' -print -quit)
  [[ -n $model ]] || { echo "clean-host pre-stage: fitted model object is missing" >&2; exit 1; }
  for language in python rust javascript; do
    step="local_export_$language"
    export_status[$language]=failed
    app model export --model "$model" --format "$language" --output "$work/export-$language.bin" \
      > "$work/export-$language.json"
    [[ -s "$work/export-$language.bin" ]] || {
      echo "clean-host pre-stage: $language local export is empty" >&2; exit 1;
    }
    export_status[$language]=passed
  done
  step=local_predict
  app predict --model "$model" --input "$work/predict.csv" --output "$work/predictions.csv" \
    --device cpu --threads 1 --json
fi

step=prediction_rows
prediction_rows=$(( $(wc -l < "$work/predictions.csv") - 1 ))
[[ $prediction_rows -eq $expected_rows ]] || {
  echo "clean-host pre-stage: prediction row count mismatch" >&2; exit 1;
}
