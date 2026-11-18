#!/usr/bin/env bash
# Reclaim idle reserved Lambda concurrency from retired campaign function
# `jope-gp-hwm-hybrid-x86-20261001` after verifying 0 CloudWatch invocations over the lookback window.
#
# Usage:
#   AWS_PROFILE=veox-jepson-prod bash deploy/carbon/reclaim-idle-concurrency.sh \
#     --json-out /path/to/concurrency-reclaim-receipt.json

set -euo pipefail

FUNCTION_NAME="jope-gp-hwm-hybrid-x86-20261001"
REGIONS="us-east-2 us-east-1 us-west-2 eu-central-1 eu-west-1 ca-central-1"
LOOKBACK_DAYS=14
JSON_OUT=""
DRY_RUN=0

die() {
  printf '[carbon-concurrency] ERROR: %s\n' "$*" >&2
  exit 1
}

say() {
  printf '[carbon-concurrency] %s\n' "$*" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --function-name) FUNCTION_NAME="$2"; shift 2 ;;
    --regions) REGIONS="${2//,/ }"; shift 2 ;;
    --lookback-days) LOOKBACK_DAYS="$2"; shift 2 ;;
    --json-out) JSON_OUT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    *) die "unknown argument '$1'" ;;
  esac
done

START_ISO="$(date -u -d "${LOOKBACK_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)"
END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ENTRIES="[]"
for reg in $REGIONS; do
  say "Checking ${FUNCTION_NAME} in ${reg} (lookback: ${START_ISO} .. ${END_ISO})..."
  before_acct="$(aws lambda get-account-settings --region "$reg" --output json)"
  before_unreserved="$(jq -r '.AccountLimit.UnreservedConcurrentExecutions' <<< "$before_acct")"
  total_limit="$(jq -r '.AccountLimit.ConcurrentExecutions' <<< "$before_acct")"

  conc_json="$(aws lambda get-function-concurrency --region "$reg" --function-name "$FUNCTION_NAME" --output json 2>/dev/null || printf '{}')"
  reserved_before="$(jq -r '.ReservedConcurrentExecutions // 0' <<< "$conc_json")"

  cw_json="$(aws cloudwatch get-metric-statistics \
    --region "$reg" \
    --namespace AWS/Lambda \
    --metric-name Invocations \
    --dimensions "Name=FunctionName,Value=${FUNCTION_NAME}" \
    --start-time "$START_ISO" \
    --end-time "$END_ISO" \
    --period 86400 \
    --statistics Sum \
    --output json)"
  invocations="$(jq -r '[.Datapoints[].Sum] | add // 0' <<< "$cw_json")"
  if [[ "$invocations" != "0" && "$invocations" != "0.0" ]]; then
    die "refusing to release concurrency on ${FUNCTION_NAME} in ${reg}: observed ${invocations} invocations in last ${LOOKBACK_DAYS}d"
  fi

  action="noop"
  if [[ "$reserved_before" -gt 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      action="would_delete_concurrency"
    else
      aws lambda delete-function-concurrency --region "$reg" --function-name "$FUNCTION_NAME" >/dev/null
      action="deleted_concurrency"
    fi
  fi

  after_acct="$(aws lambda get-account-settings --region "$reg" --output json)"
  after_unreserved="$(jq -r '.AccountLimit.UnreservedConcurrentExecutions' <<< "$after_acct")"
  say "${reg}: action=${action} reserved_before=${reserved_before} unreserved=${before_unreserved}->${after_unreserved} (total=${total_limit})"

  ENTRIES="$(jq \
    --arg region "$reg" \
    --arg function_name "$FUNCTION_NAME" \
    --arg action "$action" \
    --argjson reserved_before "$reserved_before" \
    --argjson before_unreserved "$before_unreserved" \
    --argjson after_unreserved "$after_unreserved" \
    --argjson total_limit "$total_limit" \
    --argjson recent_invocations "$invocations" \
    '. + [{
      region: $region,
      function_name: $function_name,
      action: $action,
      reserved_before: $reserved_before,
      unreserved_before: $before_unreserved,
      unreserved_after: $after_unreserved,
      total_limit: $total_limit,
      recent_invocations: $recent_invocations
    }]' <<< "$ENTRIES")"
done

RECEIPT="$(jq -n \
  --arg captured_utc "$END_ISO" \
  --arg lookback_start_utc "$START_ISO" \
  --argjson lookback_days "$LOOKBACK_DAYS" \
  --argjson dry_run "$DRY_RUN" \
  --argjson regions "$ENTRIES" \
  '{
    schema: "jope.carbon.concurrency_reclaim_receipt.v1",
    captured_utc: $captured_utc,
    lookback_start_utc: $lookback_start_utc,
    lookback_days: $lookback_days,
    dry_run: ($dry_run == 1),
    regions: $regions
  }')"

if [[ -n "$JSON_OUT" ]]; then
  mkdir -p "$(dirname "$JSON_OUT")"
  printf '%s\n' "$RECEIPT" > "$JSON_OUT"
  say "Wrote concurrency receipt to ${JSON_OUT}"
else
  printf '%s\n' "$RECEIPT"
fi
