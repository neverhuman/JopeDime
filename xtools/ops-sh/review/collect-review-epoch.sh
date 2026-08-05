#!/usr/bin/env bash
set -Eeuo pipefail

# Collect an epoch's lane reports and verify the run really happened the way it
# claims.
#
# A lane that failed does not abort the collection. Nine good specifications are
# worth having, and an `exit 1` on lane 3 would strand them. Failures are
# counted, named, and reported in the summary instead.

bash -n "$0" || exit 2

die() { printf 'collect-review-epoch: %s\n' "$*" >&2; exit 2; }

EPOCH=""
LANES=10
REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch) EPOCH="${2:?}"; shift 2 ;;
        --lanes) LANES="${2:?}"; shift 2 ;;
        --repo)  REPO="${2:?}";  shift 2 ;;
        -h|--help) printf 'Usage: %s --epoch NN [--lanes N]\n' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "$EPOCH" ]] || die "--epoch is required"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$REPO" ]] || REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REVIEWS="$REPO/feedback"
EPOCH_DIR="$REVIEWS/epochs/epoch-$EPOCH"
JAILGUN="${JOPE_REVIEW_JAILGUN:-/home/ubuntu/.local/bin/jailgun}"

[[ -f "$EPOCH_DIR/run-id" ]] || die "no run-id for epoch $EPOCH — was it launched?"
RUN_ID="$(cat "$EPOCH_DIR/run-id")"

# Jailgun writes downloads and receipts under the config's artifacts_dir, which
# is not necessarily this checkout. Reading the config is the only way to look
# where the bytes actually went — resolving from $REPO silently finds an empty
# directory and reports ten failed lanes for a run that succeeded.
CONFIG="$REVIEWS/orchestration/jailgun-gpt56-pro.toml"
ARTIFACTS="$(sed -n 's/^artifacts_dir[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$CONFIG" | head -1)"
[[ -n "$ARTIFACTS" ]] || ARTIFACTS="$REVIEWS/epochs"
DOWNLOADS="$ARTIFACTS/downloads/$RUN_ID"
ARCHIVES="$EPOCH_DIR/archives"
REPORTS="$EPOCH_DIR/reports"
mkdir -p "$ARCHIVES" "$REPORTS"

COLLECTED=0
FAILED_LANES=()

for lane_number in $(seq 1 "$LANES"); do
    LANE="$(printf '%02d' "$lane_number")"
    TAB_DIR="$DOWNLOADS/tab-$LANE"
    mapfile -t CANDIDATES < <(find "$TAB_DIR" -type f -name '*.tar.gz' -print 2>/dev/null)
    if [[ "${#CANDIDATES[@]}" -ne 1 ]]; then
        FAILED_LANES+=("$LANE:expected-1-archive-found-${#CANDIDATES[@]}")
        continue
    fi
    COPIED="$ARCHIVES/epoch-$EPOCH-lane-$LANE.tar.gz"
    cp "${CANDIDATES[0]}" "$COPIED"
    if ! "$JAILGUN" tar-validate "$COPIED" >/dev/null 2>&1; then
        FAILED_LANES+=("$LANE:tar-validate-failed")
        continue
    fi
    EXTRACT="$EPOCH_DIR/extracted/lane-$LANE"
    mkdir -p "$EXTRACT"
    tar -xzf "$COPIED" -C "$EXTRACT" --no-same-owner --no-same-permissions
    mapfile -d '' -t MARKDOWN < <(find "$EXTRACT" -type f -name '*.md' -print0)
    if [[ "${#MARKDOWN[@]}" -lt 1 ]]; then
        FAILED_LANES+=("$LANE:no-markdown")
        continue
    fi
    if [[ ! -s "${MARKDOWN[0]}" ]]; then
        FAILED_LANES+=("$LANE:empty-markdown")
        continue
    fi
    cp "${MARKDOWN[0]}" "$REPORTS/epoch-$EPOCH-lane-$LANE.md"
    COLLECTED=$((COLLECTED + 1))
done

# Gates that prove the run used the model it claims, on every tab. Count per
# line rather than slurping: one malformed line makes a slurp fail, and a
# `|| echo 0` fallback would then report a healthy run as zero tabs, zero
# downloads and zero errors — which reads as "pass".
# `grep -c` PRINTS "0" and exits 1 on zero matches, so `|| echo 0` double-printed
# ("0\n0") and broke both the arithmetic gate and jq --argjson in epoch 06.
gate() { grep -c "$1" "$EPOCH_DIR/stdout.log" 2>/dev/null | head -1; }
PREFLIGHT_VERIFIED="$(jq '[.profiles[]? | select(.family == "GPT-5.6 Sol" and .effort == "Pro" and .verified == true)] | length' "$EPOCH_DIR/model-preflight.json" 2>/dev/null || echo 0)"
TAB_VERIFIED="$(gate '"message":"tab opened"')"
DOWNLOADS_COMPLETE="$(gate '"message":"download complete"')"
ERROR_EVENTS="$(gate '"severity":"error"')"
RECEIPT_COUNT="$(find "$ARTIFACTS/receipts/$RUN_ID" -type f -name 'tab-*-download.json' 2>/dev/null | wc -l)"

STATUS="pass"
[[ "$COLLECTED" -eq "$LANES" && "$ERROR_EVENTS" -eq 0 ]] || STATUS="partial"

jq -n \
  --arg epoch "$EPOCH" --arg run_id "$RUN_ID" --arg status "$STATUS" \
  --arg collected_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg payload_ref "$(cat "$EPOCH_DIR/payload/payload-ref.txt" 2>/dev/null || echo unknown)" \
  --argjson lanes "$LANES" --argjson collected "$COLLECTED" \
  --argjson preflight_verified "$PREFLIGHT_VERIFIED" --argjson tabs_model_verified "$TAB_VERIFIED" \
  --argjson downloads_complete "$DOWNLOADS_COMPLETE" --argjson download_receipts "$RECEIPT_COUNT" \
  --argjson error_events "$ERROR_EVENTS" \
  --argjson failed "$(printf '%s\n' "${FAILED_LANES[@]+"${FAILED_LANES[@]}"}" | jq -R . | jq -s 'map(select(. != ""))')" \
  '{schema_version:"jope-dime.review-collection/v1",epoch:$epoch,run_id:$run_id,
    payload_ref:$payload_ref,
    lanes:$lanes,reports_collected:$collected,failed_lanes:$failed,
    required_model_family:"GPT-5.6 Sol",required_effort:"Pro",
    preflight_profiles_verified:$preflight_verified,tabs_model_verified:$tabs_model_verified,
    downloads_complete:$downloads_complete,download_receipts:$download_receipts,
    error_events:$error_events,status:$status,collected_at:$collected_at}' \
  >"$EPOCH_DIR/collection-summary.json"

printf 'collect-review-epoch: epoch %s — %s/%s reports, status %s\n' "$EPOCH" "$COLLECTED" "$LANES" "$STATUS"
[[ "${#FAILED_LANES[@]}" -eq 0 ]] || printf '  failed lanes: %s\n' "${FAILED_LANES[*]}"
printf '  gates: preflight=%s tabs_verified=%s downloads=%s receipts=%s errors=%s\n' \
    "$PREFLIGHT_VERIFIED" "$TAB_VERIFIED" "$DOWNLOADS_COMPLETE" "$RECEIPT_COUNT" "$ERROR_EVENTS"
[[ "$COLLECTED" -gt 0 ]] || die "no reports collected"
