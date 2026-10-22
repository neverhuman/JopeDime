#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked -p jope-appliance
binary="$root/target/debug/jopedime"
out="$root/target/fitness-example"
mkdir -p "$out"
"$binary" fitness list > "$out/catalog.json"
"$binary" fitness evaluate --input examples/fitness/evaluation_request.json > "$out/evaluation.json"
"$binary" fit --train examples/fitness/train.csv --target y --task regression \
  --method auto --candidates ridge,histgbm --folds 3 \
  --fitness @examples/fitness/grid_underforecast.json --output "$out/model.jope" --json > "$out/fit.json"
"$binary" predict --model "$out/model.jope" --input examples/fitness/predict.csv \
  --output "$out/predictions.csv" --json > "$out/predict.json"
"$binary" api < examples/fitness/api_request.jsonl > "$out/api.jsonl"
python3 - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
assert abs(json.loads((p/'evaluation.json').read_text())['value'] - 17/3) < 1e-12
fit = json.loads((p/'fit.json').read_text())
api = json.loads((p/'api.jsonl').read_text())['result']
assert fit['selection']['fitness_sha256'] == api['selection']['fitness_sha256']
assert (p/'predictions.csv').read_text() == api['predictions_csv']
print('fitness examples: CLI, persisted model, API and custom-score parity passed')
PY
