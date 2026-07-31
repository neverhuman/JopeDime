#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v47-dynamic-timeout120/release/jope-dime-regressor"
selector=Another-Dataset-on-used-Fiat-500,Food_Delivery_Time,QSAR-TID-11,airfoil_self_noise,concrete_compressive_strength,diamonds,houses,miami_housing,physiochemical_protein,superconductivity,wine_quality
expected_sha=7d7f9da0cca95fc1f6eaf02420614c9f1575e12569db43539ced2d4222c77505
run_id="${1:?run ID is required}"

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the TabArena controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! "$run_id" =~ ^tabarena-lite-all-regression-20260723-v(48|49|50|51)-dynamic120$ ]]; then
  echo "the run ID is outside the frozen V48-V51 campaign set" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V48-V51 release binary is missing" >&2
  exit 1
fi
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V48-V51 release binary hash changed" >&2
  exit 1
fi

exec "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  --dataset "$selector" \
  run-tabarena-lite-islands \
  --run-id "$run_id" \
  --resume
