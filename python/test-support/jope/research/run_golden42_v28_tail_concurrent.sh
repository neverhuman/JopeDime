#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v28-concurrent/release/jope-dime-regressor"
run_id=golden42-all-regression-20260723-v28-tail
selector=24713c7f7bfa7807,55f06d147bcc2a32,5f91a00bd3b66742,860e5558f732f638,b1f909fcfceeef4b,b78d323566930bd7,c32fa09b48fb2f27
log_root="$workspace/logs/all-regression-evolution"
log_file="$log_root/$run_id.concurrent.log"
attempt_root="$workspace/runs/$run_id/controller-attempts"

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the Golden42 controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V28 release binary is missing" >&2
  exit 1
fi

mkdir -p "$log_root" "$attempt_root"
expected_sha=8d95afd5b7a272688447e6f5f198764077ef9becd2d7fcd66d8f982fe0241f6c
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V28 release binary hash changed" >&2
  exit 1
fi

while true; do
  attempt_log="$(mktemp -p "$attempt_root" attempt.XXXXXXXX.log)"
  set +e
  "$binary" \
    --registry /home/ubuntu/remote_super/quant/regression \
    run-golden42-aggressive \
    --run-id "$run_id" \
    --dataset "$selector" \
    --resume >"$attempt_log" 2>&1
  status=$?
  set -e
  sed -n '1,$p' "$attempt_log" >>"$log_file"
  if [[ "$status" == 0 ]]; then
    rm -f -- "$attempt_log"
    exit 0
  fi
  if grep -Eq \
    'competing Cargo/Rust/Jope/SCQ job|SCQ has competing active jobs|SCQ node .* became occupied|fewer than [0-9]+ online cores|less than [0-9]+ GiB available RAM|less than [0-9]+ GiB free disk|uses [0-9]+ bytes of swap' \
    "$attempt_log"; then
    printf '[%s] launch/resume paused by resource or occupancy gate; retrying in 1 second\n' \
      "$(date -u +%FT%TZ)" >>"$log_file"
    rm -f -- "$attempt_log"
    sleep 1
    continue
  fi
  echo "unrecoverable Golden42 controller failure; retained $attempt_log" >>"$log_file"
  exit "$status"
done
