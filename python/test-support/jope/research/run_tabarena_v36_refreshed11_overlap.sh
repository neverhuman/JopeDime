#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/jope-dime
binary="$workspace/target-v34-result-recovery/release/jope-dime-regressor"
run_id=tabarena-lite-all-regression-20260723-v36-refreshed11-overlap
selector=Another-Dataset-on-used-Fiat-500,Food_Delivery_Time,QSAR-TID-11,airfoil_self_noise,concrete_compressive_strength,diamonds,houses,miami_housing,physiochemical_protein,superconductivity,wine_quality
expected_sha=fef44f028030f4162d08e6d44899aea8775333b44d22b9d22f17c6cbb88ae0b0

if [[ "$(hostname -s)" != xbabe2 ]]; then
  echo "the TabArena controller must run on xbabe2" >&2
  exit 1
fi
if [[ ! -x "$binary" ]]; then
  echo "the frozen V36 release binary is missing" >&2
  exit 1
fi
actual_sha="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "the frozen V36 release binary hash changed" >&2
  exit 1
fi

exec "$binary" \
  --registry /home/ubuntu/remote_super/quant/regression \
  --dataset "$selector" \
  run-tabarena-lite-islands \
  --run-id "$run_id" \
  --resume
