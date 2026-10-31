#!/usr/bin/env bash
# Linux-only REL-04 pre-stage. Full D1 still requires a signed candidate and macOS/Windows legs.
set -euo pipefail

usage() {
  echo "usage: $0 JOPEDIME WORKDIR" >&2
  exit 2
}
[[ $# -eq 2 || ( $# -eq 3 && $3 == --inside-netns ) ]] || usage

binary=$(realpath -- "$1")
work=$(realpath -m -- "$2")
script=$(realpath -- "${BASH_SOURCE[0]}")
script_dir=$(dirname -- "$script")
repo_root=$(cd -- "$script_dir/../.." && pwd)
[[ -f $binary && -x $binary ]] || { echo "clean-host pre-stage: executable missing: $binary" >&2; exit 2; }
case $work in /tmp | /tmp/*) echo "$0: never under /tmp (AGENTS.md)" >&2; exit 2 ;; esac
[[ ! -e $work ]] || { echo "$0: work directory exists: $work" >&2; exit 2; }

if [[ ${3:-} != --inside-netns ]]; then
  unshare_bin=$(command -v unshare || true)
  [[ -n $unshare_bin ]] || { echo "clean-host pre-stage: unshare is required for network isolation" >&2; exit 1; }
  exec "$unshare_bin" --user --map-root-user --net "$BASH" "$script" "$binary" "$work" --inside-netns
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

# Synthetic, deterministic fixture only. This is an interface smoke test, not real-data parity.
awk 'BEGIN {
  print "x,y"
  for (i = 0; i < 48; i++) printf "%d,%.3f\n", i, (2 * i) + 1
}' > "$work/train.csv"
printf 'x\n3\n17\n31\n' > "$work/predict.csv"

echo "== Linux pre-stage clean-host run (network namespace, loopback only)"
app --version --json > "$work/version.json"
# Discover methods before starting the engine so both execution branches have measured evidence.
app methods --task regression --json > "$work/methods.json"
jq -e '[.methods[] | select(.id | ascii_downcase | contains("ridge"))] | length > 0' \
  "$work/methods.json" >/dev/null
app serve --idle-exit 10m > "$work/engine.log" 2>&1 &
engine_pid=$!
cleanup() {
  if [[ -n ${engine_pid:-} ]]; then
    app serve --stop >/dev/null 2>&1 || true
    if kill -0 "$engine_pid" 2>/dev/null; then
      kill "$engine_pid" 2>/dev/null || true
    fi
    wait "$engine_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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

run_discovery_status=passed
discovery_reason=
export_mode=run
run_id=
dataset_id=
if app dataset add "$work/train.csv" --name rel04-synthetic --json \
  > "$work/dataset.json" 2> "$work/dataset.err"; then
  dataset_id=$(jq -er '.dataset_id // .id' "$work/dataset.json")
  app run \
    --dataset "$dataset_id" --target y --task regression \
    --split '{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}' \
    --objective rmse --effort fast --budget '{"evaluations":"1","wall_seconds":"180"}' \
    --landmarks none --guidance heuristic --language python --language rust --language javascript \
    --require-all-languages --device cpu --seed 17 --wait 2 --json > "$work/discovery.json"
  run_id=$(jq -er 'select(.state == "completed") | .run_id' "$work/discovery.json")
  jq -e '.best != null' "$work/discovery.json" >/dev/null

  for language in python rust javascript; do
    app export "$run_id" --kind source --lang "$language" --out "$work/export-$language.bin" --json \
      > "$work/export-$language.json"
    [[ -s "$work/export-$language.bin" ]] || {
      echo "clean-host pre-stage: $language run export is empty" >&2; exit 1;
    }
  done
  app predict --model "$run_id" --input "$work/predict.csv" --output "$work/predictions.csv"
else
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

  app fit --train "$work/train.csv" --target y --method auto --candidates ridge \
    --folds 2 --model-root "$work/models" --device cpu --threads 1 --json \
    > "$work/fit.json"
  model=$(find "$work/models" -mindepth 1 -maxdepth 1 -type d -name 'sha256-*' -print -quit)
  [[ -n $model ]] || { echo "clean-host pre-stage: fitted model object is missing" >&2; exit 1; }
  for language in python rust javascript; do
    app model export --model "$model" --format "$language" --output "$work/export-$language.bin" \
      > "$work/export-$language.json"
    [[ -s "$work/export-$language.bin" ]] || {
      echo "clean-host pre-stage: $language local export is empty" >&2; exit 1;
    }
  done
  app predict --model "$model" --input "$work/predict.csv" --output "$work/predictions.csv" \
    --device cpu --threads 1 --json
fi

[[ $(wc -l < "$work/predictions.csv") -eq 4 ]] || {
  echo "clean-host pre-stage: prediction row count mismatch" >&2; exit 1;
}
"$repo_root/deploy/ci/link-audit.sh" "$work/jopedime" fused-cpu | tee "$work/link-audit.txt"

source_sha=$(git -C "$repo_root" rev-parse HEAD)
binary_sha=$(sha256sum < "$work/jopedime" | cut -d' ' -f1)
export_python_sha=$(sha256sum < "$work/export-python.bin" | cut -d' ' -f1)
export_rust_sha=$(sha256sum < "$work/export-rust.bin" | cut -d' ' -f1)
export_javascript_sha=$(sha256sum < "$work/export-javascript.bin" | cut -d' ' -f1)
jq -n \
  --arg source_sha "$source_sha" \
  --arg binary_sha256 "$binary_sha" \
  --arg run_id "$run_id" \
  --arg dataset_id "$dataset_id" \
  --arg run_discovery_status "$run_discovery_status" \
  --arg discovery_reason "$discovery_reason" \
  --arg export_mode "$export_mode" \
  --arg python_sha256 "$export_python_sha" \
  --arg rust_sha256 "$export_rust_sha" \
  --arg javascript_sha256 "$export_javascript_sha" \
  '{
    schema:"jope.appliance-clean-host/v1",
    qualification:"pre-stage",
    platform:"linux-x86_64",
    prestage_checks_passed:true,
    source_sha:$source_sha,
    binary_sha256:$binary_sha256,
    isolation:"user+network namespace; loopback only; no default route",
    toolchain_path:"empty (python, python3, rustc and node unavailable to the app)",
    method_discovery:"passed",
    run_discovery:{status:$run_discovery_status,reason:$discovery_reason,dataset_id:$dataset_id,run_id:$run_id},
    export_mode:$export_mode,
    source_exports:{python_sha256:$python_sha256,rust_sha256:$rust_sha256,javascript_sha256:$javascript_sha256},
    prediction_rows:3,
    link_audit:"passed",
    real_data_parity:{status:"ignored",reason:"awaiting approved real-data fixture owner"},
    pending_platforms:["macos-arm64","windows-x86_64"],
    release_gate_d1:"not qualified"
  }' > "$work/clean-host.json"
cat "$work/clean-host.json"
