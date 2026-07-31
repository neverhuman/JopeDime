#!/usr/bin/env bash
set -euo pipefail

RUN_ID="golden42-v3-max-24713-20260722-v1"
DATASET="24713c7f7bfa7807"
NODE="xbabe2"
WORKERS=80
MIN_RAM_BYTES=$((64 * 1024 * 1024 * 1024))
MIN_DISK_BYTES=$((100 * 1024 * 1024 * 1024))
MAX_SWAP_BYTES=$((1024 * 1024 * 1024))
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/target/release/jope-dime-regressor"
ADMISSION="$REPO_ROOT/manifests/advanced-v3/fitted-cpp-v3-admission.v1.json"
RUN_ROOT="$REPO_ROOT/runs/$RUN_ID"
LOG_ROOT="$REPO_ROOT/logs/golden42-v3"
LOG_PATH="$LOG_ROOT/$RUN_ID.log"
PID_PATH="$LOG_ROOT/$RUN_ID.pid"
NODE_LEASE="/home/ubuntu/jope-golden42-controller-node-leases/xbabe2-v3.lock"

mem_value() {
  awk -v key="$1" '$1 == key { print $2 * 1024; exit }' /proc/meminfo
}

disk_available() {
  df -B1 --output=avail "$REPO_ROOT" | awk 'NR == 2 { print $1 }'
}

require_integer_ge() {
  local label="$1" actual="$2" expected="$3"
  if (( actual < expected )); then
    echo "$label is $actual bytes; need at least $expected" >&2
    exit 1
  fi
}

require_integer_lt() {
  local label="$1" actual="$2" limit="$3"
  if (( actual >= limit )); then
    echo "$label is $actual bytes; must be below $limit" >&2
    exit 1
  fi
}

if [[ "$(hostname -s)" != "$NODE" ]]; then
  echo "Golden42 V3 must be launched on $NODE" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "run root already exists: $RUN_ROOT" >&2
  exit 1
fi
if [[ -s "$PID_PATH" ]] && kill -0 "$(<"$PID_PATH")" 2>/dev/null; then
  echo "campaign PID $(<"$PID_PATH") is already alive" >&2
  exit 1
fi
mkdir -p "$(dirname "$NODE_LEASE")"
if ! command -v flock >/dev/null || ! flock -n "$NODE_LEASE" -c true; then
  echo "the xbabe2 Golden42 V3 node lease is unavailable" >&2
  exit 1
fi
if ! sudo -n true; then
  echo "passwordless sudo is required for guarded swap reclamation" >&2
  exit 1
fi
if (( $(nproc) < WORKERS )); then
  echo "only $(nproc) usable CPUs; need $WORKERS" >&2
  exit 1
fi

available_before="$(mem_value MemAvailable:)"
swap_total_before="$(mem_value SwapTotal:)"
swap_free_before="$(mem_value SwapFree:)"
swap_used_before=$((swap_total_before - swap_free_before))
if (( available_before - swap_used_before < MIN_RAM_BYTES )); then
  echo "reclaiming swap would leave less than 64 GiB available RAM" >&2
  exit 1
fi
require_integer_ge "free disk" "$(disk_available)" "$MIN_DISK_BYTES"

cd "$REPO_ROOT"
cargo build --workspace --release --all-features

sudo -n swapoff -a
if ! sudo -n swapon -a; then
  sudo -n swapon -a || true
  echo "failed to restore configured swap after reclamation" >&2
  exit 1
fi

available_after="$(mem_value MemAvailable:)"
swap_total_after="$(mem_value SwapTotal:)"
swap_free_after="$(mem_value SwapFree:)"
swap_used_after=$((swap_total_after - swap_free_after))
require_integer_ge "available RAM" "$available_after" "$MIN_RAM_BYTES"
require_integer_lt "used swap" "$swap_used_after" "$MAX_SWAP_BYTES"
require_integer_ge "free disk" "$(disk_available)" "$MIN_DISK_BYTES"
if (( $(nproc) < WORKERS )); then
  echo "usable CPU count fell below $WORKERS" >&2
  exit 1
fi

mkdir -p "$(dirname "$ADMISSION")" "$LOG_ROOT"
if [[ -e "$ADMISSION" ]]; then
  admission_hash="$(sha256sum "$ADMISSION" | awk '{print $1}')"
  mv "$ADMISSION" "${ADMISSION%.json}.superseded-$admission_hash.json"
fi
"$BIN" admit-fitted-cpp-v3 --output "$ADMISSION"

nohup setsid "$BIN" \
  --registry /home/ubuntu/remote_super/quant/regression \
  --kernel-version advanced-v3 \
  run-golden42-v3-max \
  --run-id "$RUN_ID" \
  --dataset "$DATASET" \
  --node "$NODE" \
  --workers "$WORKERS" \
  --population "$WORKERS" \
  >"$LOG_PATH" 2>&1 < /dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" >"$PID_PATH"

deadline=$((SECONDS + 300))
while (( SECONDS < deadline )); do
  if [[ -s "$RUN_ROOT/final-result.v1.json" ]]; then
    "$BIN" validate-golden42-v3-max --run-id "$RUN_ID"
    echo "Golden42 V3 completed and validated during launch."
    echo "PID file: $PID_PATH"
    echo "Log: $LOG_PATH"
    echo "Result: $RUN_ROOT/final-result.v1.json"
    exit 0
  fi
  if ! kill -0 "$controller_pid" 2>/dev/null; then
    echo "controller exited before launch verification; inspect $LOG_PATH" >&2
    exit 1
  fi
  if [[ -s "$RUN_ROOT/workers.v1.json" && -s "$RUN_ROOT/telemetry.v1.jsonl" ]]; then
    if python3 - "$RUN_ROOT/workers.v1.json" "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys

registry = json.loads(pathlib.Path(sys.argv[1]).read_text())
workers = registry.get("workers", [])
if len(workers) != 80:
    raise SystemExit(1)
for worker in workers:
    pid = int(worker["pid"])
    cpu = str(worker["cpu"])
    status = pathlib.Path(f"/proc/{pid}/status")
    if not status.exists():
        raise SystemExit(1)
    allowed = next(
        line.split(":", 1)[1].strip()
        for line in status.read_text().splitlines()
        if line.startswith("Cpus_allowed_list:")
    )
    if allowed != cpu:
        raise SystemExit(1)
receipt_root = pathlib.Path(sys.argv[2]) / "receipts"
if not any(receipt_root.rglob("slot-*.v1.json")):
    raise SystemExit(1)
PY
    then
      echo "Golden42 V3 is live with 80 pinned workers and candidate telemetry."
      echo "PID: $controller_pid"
      echo "PID file: $PID_PATH"
      echo "Log: $LOG_PATH"
      echo "Run root: $RUN_ROOT"
      exit 0
    fi
  fi
  sleep 1
done

echo "campaign did not reach the 80-worker launch gate within 300 seconds" >&2
exit 1
