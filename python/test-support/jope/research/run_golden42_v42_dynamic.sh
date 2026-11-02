#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v43-dynamic-slots/release/jope-dime-regressor"
run_id=golden42-all-regression-20260723-v42-dynamic
selector=24713c7f7bfa7807,55f06d147bcc2a32,5f91a00bd3b66742,860e5558f732f638,b1f909fcfceeef4b,b78d323566930bd7
expected_sha=a2c12a5201b464e11deaa030bdeadad8c22ff98c9dfc8cc567ef5e5e2ff4d4b7

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the Golden42 controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V42 release binary is missing" >&2
  exit 1
fi
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V42 release binary hash changed" >&2
  exit 1
fi

exec "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  run-golden42-aggressive \
  --run-id "$run_id" \
  --dataset "$selector" \
  --resume
