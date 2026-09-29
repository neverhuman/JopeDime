#!/usr/bin/env bash
set -euo pipefail
umask 077
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SOURCE_DIR/env.sh"
load_probe_env "${1:-$SOURCE_DIR/.env}"
AWS=/home/ubuntu/.local/bin/aws
BOOTSTRAP="$SOURCE_DIR/rust/target/x86_64-unknown-linux-musl/release/bootstrap"
[[ -x "$BOOTSTRAP" ]] || { echo 'Build bootstrap first' >&2; exit 1; }
shopt -s nullglob
previous_intents=("${RUN_DIR%/*}"/rust-run*/invoke-*.intent.json)
count=${#previous_intents[@]}
initial_count=$count
[[ "$count" -lt 6 ]] || { echo 'Original six-call native budget exhausted' >&2; exit 1; }
[[ "$($AWS sts get-caller-identity --query Account --output text)" == "$AWS_ACCOUNT_ID" ]] || exit 1
# Fresh directory is the durable campaign/budget lock. A rerun fails before resource creation.
mkdir "$RUN_DIR"
cp "$BOOTSTRAP" "$RUN_DIR/bootstrap"
sha256sum "$RUN_DIR/bootstrap" > "$RUN_DIR/bootstrap.sha256"
zip -j -9 "$RUN_DIR/bootstrap.zip" "$RUN_DIR/bootstrap" > "$RUN_DIR/zip.log"
GROUP="/aws/lambda/$FUNCTION_NAME"
role_created=0; policy_created=0; logs_created=0; function_created=0
cleanup() {
  local failed=0
  set +e
  if (( function_created )); then "$AWS" lambda delete-function --function-name "$FUNCTION_NAME" > "$RUN_DIR/delete-function.json" 2>&1 || failed=1; fi
  if (( logs_created )); then "$AWS" logs delete-log-group --log-group-name "$GROUP" > "$RUN_DIR/delete-logs.json" 2>&1 || failed=1; fi
  if (( policy_created )); then "$AWS" iam delete-role-policy --role-name "$FUNCTION_NAME" --policy-name probe-only > "$RUN_DIR/delete-policy.json" 2>&1 || failed=1; fi
  if (( role_created )); then "$AWS" iam delete-role --role-name "$FUNCTION_NAME" > "$RUN_DIR/delete-role.json" 2>&1 || failed=1; fi
  printf '{"cleanup_failed":%s}\n' "$failed" > "$RUN_DIR/cleanup.json"
  (( failed == 0 )) || echo 'Cleanup incomplete; inspect receipts' >&2
}
trap cleanup EXIT
jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"lambda.amazonaws.com"},Action:"sts:AssumeRole"}]}' > "$RUN_DIR/trust.json"
"$AWS" iam create-role --role-name "$FUNCTION_NAME" --assume-role-policy-document "file://$RUN_DIR/trust.json" > "$RUN_DIR/role.json"
role_created=1
"$AWS" logs create-log-group --log-group-name "$GROUP"
logs_created=1
"$AWS" logs put-retention-policy --log-group-name "$GROUP" --retention-in-days 1
jq -n --arg log "arn:aws:logs:$AWS_REGION:$AWS_ACCOUNT_ID:log-group:$GROUP:*" '{Version:"2012-10-17",Statement:[
  {Effect:"Allow",Action:"s3:GetObjectVersion",Resource:"arn:aws:s3:::veox-jopedime-datasets-use2/blobs/sha256/d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931",Condition:{StringEquals:{"s3:VersionId":"ymyHYuYxnSHXPoK18mKgrPdHBQ4.q2Nq"}}},
  {Effect:"Allow",Action:["logs:CreateLogStream","logs:PutLogEvents"],Resource:$log}]}' > "$RUN_DIR/policy.json"
"$AWS" iam put-role-policy --role-name "$FUNCTION_NAME" --policy-name probe-only --policy-document "file://$RUN_DIR/policy.json"
policy_created=1
for attempt in $(seq 1 12); do
  if "$AWS" lambda create-function --function-name "$FUNCTION_NAME" --runtime provided.al2023 \
      --handler bootstrap --role "arn:aws:iam::$AWS_ACCOUNT_ID:role/$FUNCTION_NAME" \
      --architectures x86_64 --timeout 120 --memory-size 128 --zip-file "fileb://$RUN_DIR/bootstrap.zip" \
      --environment 'Variables={RAYON_NUM_THREADS=1}' > "$RUN_DIR/function.json" 2> "$RUN_DIR/create.err"; then
    function_created=1; break
  fi
  if ! grep -q 'cannot be assumed' "$RUN_DIR/create.err" || [[ "$attempt" == 12 ]]; then cat "$RUN_DIR/create.err" >&2; exit 1; fi
  sleep 5
done
"$AWS" lambda put-function-concurrency --function-name "$FUNCTION_NAME" --reserved-concurrent-executions 1 > "$RUN_DIR/concurrency.json"
"$AWS" lambda wait function-active-v2 --function-name "$FUNCTION_NAME"
for memory in 128 256 512; do
  if [[ "$memory" != 128 ]]; then
    "$AWS" lambda update-function-configuration --function-name "$FUNCTION_NAME" --memory-size "$memory" > "$RUN_DIR/config-$memory.json"
    "$AWS" lambda wait function-updated-v2 --function-name "$FUNCTION_NAME"
  fi
  for repeat in 0 1; do
    [[ "$count" -lt 6 ]] || break 2
    count=$((count+1)); [[ "$count" -le 6 ]] || exit 1
    prefix="$RUN_DIR/invoke-$memory-$repeat"
    # Record intent before the call; failures/timeouts count toward the six-call budget.
    printf '{"attempt":%d,"memory_mb":%d,"repeat":%d}\n' "$count" "$memory" "$repeat" > "$prefix.intent.json"
    AWS_MAX_ATTEMPTS=1 "$AWS" lambda invoke --function-name "$FUNCTION_NAME" \
      --invocation-type RequestResponse --log-type Tail --cli-read-timeout 150 \
      --cli-binary-format raw-in-base64-out --payload '{}' "$prefix.result.json" > "$prefix.meta.json"
    jq -r '.LogResult' "$prefix.meta.json" | base64 -d > "$prefix.log"
    jq -e '.StatusCode == 200 and (.FunctionError == null)' "$prefix.meta.json" >/dev/null
    jq -e '.algorithm == "feat-math.histgbm2" and .train_rows == 800 and .validation_rows == 200' "$prefix.result.json" >/dev/null
    jq -c '{memory_mb,cold,fit_s,handler_s,rmse,actual_trees,memory_after_fit}' "$prefix.result.json"
  done
done
printf '{"aggregate_native_attempts":%d,"successful_calls_this_attempt":%d,"budget_exhausted":true}\n' "$count" "$((count-initial_count))" > "$RUN_DIR/complete.json"
