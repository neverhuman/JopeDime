#!/usr/bin/env bash
# Smoke-run the objective interfaces end to end: catalog, validation, scoring, a calibrated fit
# that is saved, reloaded and used to predict, the same fit over JSON-RPC, and an MCP handshake.
# Needs cargo and jq. Outputs go to target/fitness-example/.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked -p jope-appliance
binary="${CARGO_TARGET_DIR:-$root/target}/debug/jopedime"
out="$root/target/fitness-example"
mkdir -p "$out"
fail() { echo "objective examples: $*" >&2; exit 1; }

"$binary" fitness list > "$out/catalog.json"
[ "$(jq '.fitness | length' "$out/catalog.json")" -eq 38 ] || fail "catalog size changed"
"$binary" fitness validate pinball:quantile=0.9 > "$out/validate.json"
[ "$(jq -r .specification.quantile "$out/validate.json")" = 0.9 ] || fail "shorthand override"
"$binary" fitness evaluate --input examples/objectives/evaluation_request.json > "$out/evaluation.json"
jq -e '(.value - 17/3 | fabs) < 1e-12' "$out/evaluation.json" > /dev/null || fail "hand-checked cost"

for calibration in none auto; do
  "$binary" fit --train examples/objectives/train.csv --target y --task regression \
    --method auto --candidates ridge,histgbm --folds 3 --calibration "$calibration" \
    --fitness @examples/objectives/grid_underforecast.json \
    --output "$out/model-$calibration.jope" --json > "$out/fit-$calibration.json"
  "$binary" predict --model "$out/model-$calibration.jope" --input examples/objectives/predict.csv \
    --output "$out/predictions-$calibration.csv" --json > "$out/predict-$calibration.json"
done
jq -e '.selection.calibration == null' "$out/fit-none.json" > /dev/null || fail "none calibrated"
jq -e '.selection.calibration.family == "affine"' "$out/fit-auto.json" > /dev/null \
  || fail "auto did not calibrate"

"$binary" api < examples/objectives/api_request.jsonl > "$out/api.jsonl"
[ "$(jq -r .selection.fitness_sha256 "$out/fit-auto.json")" = \
  "$(jq -r .result.selection.fitness_sha256 "$out/api.jsonl")" ] || fail "CLI and API identities differ"
[ "$(cat "$out/predictions-auto.csv")" = "$(jq -r .result.predictions_csv "$out/api.jsonl")" ] \
  || fail "CLI and API predictions differ"

printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"run.sh","version":"1"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fitness_validate","arguments":{"fitness":"sortino"}}}' \
  | "$binary" mcp > "$out/mcp.jsonl"
jq -s -e '.[0].result.protocolVersion == "2025-06-18"
  and .[1].result.structuredContent.requires_chronological_validation == true' "$out/mcp.jsonl" \
  > /dev/null || fail "MCP handshake or validation"
echo "objective examples: catalog, validation, scoring, calibrated CLI model, API and MCP passed"
