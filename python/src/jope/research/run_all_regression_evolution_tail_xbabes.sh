#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target/release/jope-dime-regressor"
scq=/home/ubuntu/super_cluster/scq
source_index=/home/ubuntu/tabArenaLite/results/jope_dime_evolution/all-regression-evolution-20260723-v24/index.v1.json
campaign_id=all-regression-evolution-20260723-v25
golden_run=golden42-all-regression-20260723-v25-tail
tabarena_run=tabarena-lite-all-regression-20260723-v25
campaign_root="$workspace/runs/$campaign_id"
log_root="$workspace/logs/all-regression-evolution"
log_file="$log_root/$campaign_id.log"
indexer="$workspace/scripts/index_all_regression_evolution.py"

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the all-regression controller must start on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" || ! -x "$scq" || ! -x "$indexer" ]]; then
  echo "release binary, SCQ, or indexer is missing" >&2
  exit 1
fi
if [[ ! -f "$source_index" ]]; then
  echo "the v24 analysis index is missing" >&2
  exit 1
fi

selector="${JOPE_GOLDEN42_SELECTOR:-$(
  jq -r '[.datasets[] | select(.cohort == "golden42" and .status != "confirmed_winner") | .dataset] | join(",")' "$source_index"
)}"
if [[ -z "$selector" || ! "$selector" =~ ^[0-9a-f]+(,[0-9a-f]+)*$ ]]; then
  echo "the unresolved Golden42 selector is empty or invalid" >&2
  exit 1
fi

if [[ -e "$campaign_root/controller.pid" ]]; then
  recorded_pid="$(<"$campaign_root/controller.pid")"
  if kill -0 "$recorded_pid" 2>/dev/null; then
    echo "campaign controller $recorded_pid is already running" >&2
    exit 1
  fi
fi

mkdir -p "$campaign_root" "$log_root"
exec 9>"$campaign_root/controller.lock"
if ! flock -n 9; then
  echo "all-regression tail controller lease is already held" >&2
  exit 1
fi
printf '%s\n' "$$" >"$campaign_root/controller.pid"

"$scq" ping >/dev/null
active_jobs="$($scq --json ps | jq '[.Jobs.jobs[] | select(.state != "completed" and .state != "failed" and .state != "cancelled")] | length')"
if [[ "$active_jobs" != 0 ]]; then
  echo "SCQ has $active_jobs active jobs; refusing an overlapping launch" >&2
  exit 1
fi

"$indexer" \
  --repo "$workspace" \
  --output-root "$campaign_root" \
  --campaign-id "$campaign_id" \
  --golden-run "$golden_run" \
  --tabarena-run "$tabarena_run" \
  --watch-seconds 30 >>"$log_file" 2>&1 &
index_pid=$!
cleanup() {
  kill "$index_pid" 2>/dev/null || true
  wait "$index_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_with_resource_retries() {
  local label="$1"
  shift
  local attempt_log status
  while true; do
    attempt_log="$(mktemp -p "$campaign_root" ".${label}.attempt.XXXXXXXX")"
    if "$@" >"$attempt_log" 2>&1; then
      sed -n '1,$p' "$attempt_log" >>"$log_file"
      rm -f -- "$attempt_log"
      return 0
    else
      status=$?
    fi
    sed -n '1,$p' "$attempt_log" >>"$log_file"
    if grep -Eq \
      'competing Cargo/Rust/Jope/SCQ job|SCQ has competing active jobs|SCQ node .* became occupied|fewer than [0-9]+ online cores|less than [0-9]+ GiB available RAM|less than [0-9]+ GiB free disk|uses [0-9]+ bytes of swap' \
      "$attempt_log"; then
      echo "[$(date -u +%FT%TZ)] $label paused by resource/occupancy gate; retrying in 30 seconds" >>"$log_file"
      rm -f -- "$attempt_log"
      sleep 30
      continue
    fi
    rm -f -- "$attempt_log"
    return "$status"
  done
}

sleep 10
echo "[$(date -u +%FT%TZ)] reseeding Golden42 tail: $selector" >>"$log_file"
run_with_resource_retries golden42-tail "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  run-golden42-aggressive \
  --run-id "$golden_run" \
  --dataset "$selector" \
  --resume

echo "[$(date -u +%FT%TZ)] starting TabArena Lite 13-dataset campaign" >>"$log_file"
run_with_resource_retries tabarena "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  run-tabarena-lite-islands \
  --run-id "$tabarena_run" \
  --resume

"$indexer" \
  --repo "$workspace" \
  --output-root "$campaign_root" \
  --campaign-id "$campaign_id" \
  --golden-run "$golden_run" \
  --tabarena-run "$tabarena_run"
touch "$campaign_root/complete"
echo "[$(date -u +%FT%TZ)] tail and all TabArena datasets complete and validated" >>"$log_file"
