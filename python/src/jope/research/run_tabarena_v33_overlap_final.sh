#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v33-overlap-final/release/jope-dime-regressor"
run_id=tabarena-lite-all-regression-20260723-v33-overlap-final
selector=Another-Dataset-on-used-Fiat-500,Food_Delivery_Time,QSAR-TID-11,airfoil_self_noise,concrete_compressive_strength,diamonds,healthcare_insurance_expenses,houses,miami_housing,physiochemical_protein,superconductivity,wine_quality
expected_sha=c3e5600d4c46ab88c66d17a12b513473df7587e239b1a5a80b1ef55a7aad024f

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the TabArena controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V33 release binary is missing" >&2
  exit 1
fi
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V33 release binary hash changed" >&2
  exit 1
fi

exec "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  --dataset "$selector" \
  run-tabarena-lite-islands \
  --run-id "$run_id" \
  --resume
