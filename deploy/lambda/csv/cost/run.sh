#!/usr/bin/env bash
# Bounded, read-only S3 cost experiment. Runtime code is native Rust.
set -euo pipefail
umask 077
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SOURCE_DIR/env.sh"
load_probe_env "${1:-$SOURCE_DIR/.env}"
AWS=/home/ubuntu/.local/bin/aws
[[ "$($AWS sts get-caller-identity --query Account --output text)" == "$AWS_ACCOUNT_ID" ]] || exit 1
for arch in x86_64 aarch64; do
  [[ -x "$SOURCE_DIR/../rust/target/$arch-unknown-linux-musl/release/bootstrap-batch" ]] || exit 1
done
[[ -s "$RUN_DIR/local-baselines.json" ]] || exit 1
shopt -s nullglob
prior_intents=("$RUN_DIR"/aws-run*/intents/*.json)
count=${#prior_intents[@]}
[[ "$count" -lt 60 ]] || { echo 'Aggregate campaign budget exhausted' >&2; exit 1; }
STATE="$RUN_DIR/aws-run-attempt2"
mkdir "$STATE"
mkdir "$STATE/intents" "$STATE/receipts"
deadline="$(date -d "$(jq -r .absolute_deadline_utc "$RUN_DIR/campaign.json")" +%s)"
role_created=0; policy_created=0
functions_created=(); logs_created=()
ROLE="$FUNCTION_NAME-role"
cleanup() {
  local failed=0 fn group
  set +e
  for fn in "${functions_created[@]}"; do
    "$AWS" lambda delete-function --function-name "$fn" > "$STATE/delete-$fn.json" 2>&1 || failed=1
  done
  for group in "${logs_created[@]}"; do
    "$AWS" logs delete-log-group --log-group-name "$group" > /dev/null 2>&1 || failed=1
  done
  if (( policy_created )); then "$AWS" iam delete-role-policy --role-name "$ROLE" --policy-name read-five-blobs > "$STATE/delete-policy.json" 2>&1 || failed=1; fi
  if (( role_created )); then "$AWS" iam delete-role --role-name "$ROLE" > "$STATE/delete-role.json" 2>&1 || failed=1; fi
  printf '{"cleanup_failed":%d,"attempted_invocations":%d}\n' "$failed" "$count" > "$STATE/cleanup.json"
  (( failed == 0 )) || echo 'Cleanup failed; inspect receipts' >&2
}
trap cleanup EXIT
jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"lambda.amazonaws.com"},Action:"sts:AssumeRole"}]}' > "$STATE/trust.json"
"$AWS" iam create-role --role-name "$ROLE" --assume-role-policy-document "file://$STATE/trust.json" > "$STATE/role.json"
role_created=1
jq --arg region "$AWS_REGION" --arg account "$AWS_ACCOUNT_ID" --arg prefix "$FUNCTION_NAME" '{Version:"2012-10-17",Statement:([.[]|{Effect:"Allow",Action:"s3:GetObjectVersion",Resource:("arn:aws:s3:::veox-jopedime-datasets-use2/"+.key),Condition:{StringEquals:{"s3:VersionId":.version}}}]+[{Effect:"Allow",Action:["logs:CreateLogStream","logs:PutLogEvents"],Resource:("arn:aws:logs:"+$region+":"+$account+":log-group:/aws/lambda/"+$prefix+"-*:*")}])}' "$SOURCE_DIR/jobs.json" > "$STATE/policy.json"
"$AWS" iam put-role-policy --role-name "$ROLE" --policy-name read-five-blobs --policy-document "file://$STATE/policy.json"
policy_created=1
for architecture in x86_64 arm64; do
  target=x86_64; [[ "$architecture" == arm64 ]] && target=aarch64
  fn="$FUNCTION_NAME-$architecture"
  mkdir "$STATE/$architecture"
  cp "$SOURCE_DIR/../rust/target/$target-unknown-linux-musl/release/bootstrap-batch" "$STATE/$architecture/bootstrap"
  sha256sum "$STATE/$architecture/bootstrap" > "$STATE/$architecture/bootstrap.sha256"
  zip -j -9 "$STATE/$architecture/bootstrap.zip" "$STATE/$architecture/bootstrap" > "$STATE/$architecture/zip.log"
  "$AWS" logs create-log-group --log-group-name "/aws/lambda/$fn"
  logs_created+=("/aws/lambda/$fn")
  "$AWS" logs put-retention-policy --log-group-name "/aws/lambda/$fn" --retention-in-days 1
  for attempt in $(seq 1 12); do
    if "$AWS" lambda create-function --function-name "$fn" --runtime provided.al2023 --handler bootstrap \
        --role "arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE" --architectures "$architecture" --timeout 120 --memory-size 512 \
        --zip-file "fileb://$STATE/$architecture/bootstrap.zip" --environment 'Variables={RAYON_NUM_THREADS=1}' \
        > "$STATE/$architecture/function.json" 2> "$STATE/$architecture/create.err"; then
      functions_created+=("$fn"); break
    fi
    if ! grep -q 'cannot be assumed' "$STATE/$architecture/create.err" || [[ "$attempt" == 12 ]]; then cat "$STATE/$architecture/create.err" >&2; exit 1; fi
    sleep 5
  done
  "$AWS" lambda put-function-concurrency --function-name "$fn" --reserved-concurrent-executions 1 > "$STATE/$architecture/concurrency.json"
  "$AWS" lambda wait function-active-v2 --function-name "$fn"
done

configure() {
  local architecture="$1" memory="$2" label="$3"
  [[ "$memory" -le 3072 ]] || exit 1
  "$AWS" lambda update-function-configuration --function-name "$FUNCTION_NAME-$architecture" --memory-size "$memory" > "$STATE/config-$label.json"
  "$AWS" lambda wait function-updated-v2 --function-name "$FUNCTION_NAME-$architecture"
}

invoke() {
  local architecture="$1" memory="$2" phase="$3" repeat="$4" group_size="$5" ids="$6" prefix rate billed peak
  [[ "$(date +%s)" -lt "$deadline" && "$count" -lt 60 ]] || { echo 'Budget/deadline exhausted' >&2; exit 1; }
  count=$((count+1))
  prefix="$STATE/receipts/$(printf '%03d' "$count")"
  jq -n --arg phase "$phase" --arg architecture "$architecture" --argjson memory "$memory" --argjson repeat "$repeat" --argjson group_size "$group_size" --argjson ids "$ids" \
    '{phase:$phase,architecture:$architecture,memory_mb:$memory,repeat:$repeat,requested_batch_size:$group_size,job_ids:$ids}' > "$STATE/intents/$(printf '%03d' "$count").json"
  jq -n --argjson ids "$ids" '{job_ids:$ids}' > "$prefix.payload.json"
  AWS_MAX_ATTEMPTS=1 "$AWS" lambda invoke --function-name "$FUNCTION_NAME-$architecture" --invocation-type RequestResponse \
    --log-type Tail --cli-read-timeout 150 --cli-binary-format raw-in-base64-out --payload "file://$prefix.payload.json" \
    "$prefix.result.json" > "$prefix.meta.json"
  jq -r .LogResult "$prefix.meta.json" | base64 -d > "$prefix.log"
  jq -e '.StatusCode==200 and (.FunctionError==null)' "$prefix.meta.json" > /dev/null
  jq -e --argjson ids "$ids" '.batch_size == ($ids|length)' "$prefix.result.json" > /dev/null
  jq -e --slurpfile baseline "$RUN_DIR/local-baselines.json" '
    def absolute: if .<0 then -. else . end;
    all(.jobs[]; . as $got | ($baseline[0].jobs[]|select(.dataset_hash==$got.dataset_hash)) as $ref |
      .fold_sha256==$ref.fold_sha256 and .actual_trees==$ref.actual_trees and
      (.predictions|length)==($ref.predictions|length) and
      all(range(0;(.predictions|length)); . as $i |
        (($got.predictions[$i]-$ref.predictions[$i])|absolute) <= (1e-10*(1+($ref.predictions[$i]|absolute)))))' "$prefix.result.json" > /dev/null
  billed="$(sed -n 's/.*Billed Duration: \([0-9.]*\) ms.*/\1/p' "$prefix.log")"
  peak="$(sed -n 's/.*Max Memory Used: \([0-9]*\) MB.*/\1/p' "$prefix.log")"
  [[ -n "$billed" && -n "$peak" ]] || exit 1
  rate=0.0000166667; [[ "$architecture" == arm64 ]] && rate=0.0000133334
  jq --slurpfile intent "$STATE/intents/$(printf '%03d' "$count").json" --argjson billed "$billed" --argjson peak "$peak" --argjson rate "$rate" \
    '$intent[0]+{request_id:.request_id,cold:.cold,batch_size:.batch_size,handler_s:.handler_s,billed_ms:$billed,max_memory_mb:$peak,
       lambda_usd:($billed/1000*.memory_mb/1024*$rate+0.0000002),
       s3_gets:([.jobs[].s3_get_count]|add),
       total_usd:($billed/1000*.memory_mb/1024*$rate+0.0000002+([.jobs[].s3_get_count]|add)*0.0000004),
       usd_per_training:(($billed/1000*.memory_mb/1024*$rate+0.0000002+([.jobs[].s3_get_count]|add)*0.0000004)/.batch_size)}' \
    "$prefix.result.json" > "$prefix.summary.json"
  jq -c '{phase,architecture,memory_mb,cold,batch_size,billed_ms,usd_per_training}' "$prefix.summary.json"
}

# Same five jobs at every architecture/memory setting, including fresh S3 GETs.
for architecture in x86_64 arm64; do
  for memory in 512 1024 1769 2048; do
    configure "$architecture" "$memory" "screen-$architecture-$memory"
    for repeat in 0 1; do invoke "$architecture" "$memory" screen "$repeat" 5 '[0,1,2,3,4]'; done
  done
done
jq -s 'map(select(.phase=="screen" and .cold==false)) | min_by(.total_usd)' "$STATE"/receipts/*.summary.json > "$STATE/screen-winner.json"
best_arch="$(jq -r .architecture "$STATE/screen-winner.json")"
best_memory="$(jq -r .memory_mb "$STATE/screen-winner.json")"
configure "$best_arch" "$best_memory" batch-winner
invoke "$best_arch" "$best_memory" warmup 0 5 '[0,1,2,3,4]'
# Rotate order across repeats to reduce simple run-order confounding.
for repeat in 0 1 2; do
  case "$repeat" in 0) sizes=(1 2 3 5);; 1) sizes=(5 3 2 1);; 2) sizes=(2 5 1 3);; esac
  for size in "${sizes[@]}"; do
    for ((start=0;start<5;start+=size)); do
      ids="$(jq -cn --argjson start "$start" --argjson size "$size" '[range($start;([$start+$size,5]|min))]')"
      invoke "$best_arch" "$best_memory" batch "$repeat" "$size" "$ids"
    done
  done
done
# Additional cost experiment: warm reuse of the same immutable admitted inputs.
"$AWS" lambda update-function-configuration --function-name "$FUNCTION_NAME-$best_arch" \
  --environment 'Variables={RAYON_NUM_THREADS=1,CACHE_INPUTS=1}' > "$STATE/config-cache.json"
"$AWS" lambda wait function-updated-v2 --function-name "$FUNCTION_NAME-$best_arch"
invoke "$best_arch" "$best_memory" cache-prime 0 5 '[0,1,2,3,4]'
for repeat in 0 1 2; do invoke "$best_arch" "$best_memory" cache-batch "$repeat" 5 '[0,1,2,3,4]'; done
for id in 0 1 2 3 4; do invoke "$best_arch" "$best_memory" cache-single 0 1 "[$id]"; done
jq -s '.' "$STATE"/receipts/*.summary.json > "$STATE/summaries.json"
printf '{"attempted_invocations":%d,"complete":true}\n' "$count" > "$STATE/complete.json"
