#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v34-result-recovery/release/jope-dime-regressor"
run_id=golden42-all-regression-20260723-v35-tail-overlap
selector=24713c7f7bfa7807,55f06d147bcc2a32,5f91a00bd3b66742,860e5558f732f638,b1f909fcfceeef4b,b78d323566930bd7
expected_sha=fef44f028030f4162d08e6d44899aea8775333b44d22b9d22f17c6cbb88ae0b0

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the Golden42 controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V35 release binary is missing" >&2
  exit 1
fi
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V35 release binary hash changed" >&2
  exit 1
fi

exec "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  run-golden42-aggressive \
  --run-id "$run_id" \
  --dataset "$selector" \
  --resume
